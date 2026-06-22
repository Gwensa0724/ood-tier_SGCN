import math
import random
import time
import copy
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph

from backbone import GCN
from data_utils import eval_acc, get_measures
from gnnsafe import GNNSafe

_SGCN_SEED_RATIO = 20
_SGCN_RANDOM_WALK_MAX_HOPS = 10
_SGCN_MIN_TRAIN_NODES = 32
_SGCN_VAL_SAMPLE_SIZE = 512


def _clone_data(data):
    cloned = data.clone()
    if hasattr(data, "node_idx"):
        cloned.node_idx = data.node_idx.clone()
    if hasattr(data, "splits"):
        cloned.splits = {k: v.clone() for k, v in data.splits.items()}
    if hasattr(data, "global_node_idx"):
        cloned.global_node_idx = data.global_node_idx.clone()
    return cloned


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_isin(elements, test_elements):
    """Compatibility helper for torch versions without torch.isin."""
    if hasattr(torch, "isin"):
        return torch.isin(elements, test_elements)
    if test_elements.numel() == 0:
        return torch.zeros_like(elements, dtype=torch.bool)
    return (elements.unsqueeze(-1) == test_elements.unsqueeze(0)).any(dim=-1)


class PyGNodeClassifier(nn.Module):
    """Remote-style classifier used for GraphSAGE and SGCN experiments."""

    def __init__(
        self,
        node_feats,
        n_classes,
        n_layers,
        n_hidden,
        dropout,
        mpnn,
        input_drop=0.0,
        edge_drop=0.0,
        jk=False,
        use_bn=True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.mpnn = mpnn
        self.jk = jk
        self.edge_drop = edge_drop
        self.use_bn = use_bn

        self.node_encoder = nn.Linear(node_feats, n_hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)

        for _ in range(n_layers):
            if mpnn == "sage":
                self.convs.append(SAGEConv(n_hidden, n_hidden))
            else:
                self.convs.append(GCNConv(n_hidden, n_hidden, add_self_loops=False))
            self.norms.append(nn.BatchNorm1d(n_hidden))

        self.pred_linear = nn.Linear(n_hidden, n_classes)

    def forward(self, x, edge_index, edge_attr=None):
        if self.training and self.edge_drop > 0 and edge_index.shape[1] > 0:
            keep_mask = torch.rand(edge_index.shape[1], device=edge_index.device) >= self.edge_drop
            edge_index = edge_index[:, keep_mask]

        h = F.relu(self.node_encoder(x))
        h = self.input_drop(h)
        h_last = None
        h_local = []

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            if h_last is not None:
                h = h + h_last[: h.shape[0], :]
            h_last = h
            if self.use_bn:
                h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            h_local.append(h)

        if self.jk and h_local:
            h = torch.sum(torch.stack(h_local), dim=0)

        return self.pred_linear(h)


def build_backbone(
    name,
    in_channels,
    out_channels,
    hidden_channels,
    num_layers,
    dropout,
    use_bn,
    input_drop=0.0,
    edge_drop=0.0,
    jk=False,
):
    if name == "gcn":
        return GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_bn=use_bn,
        )
    if name == "sage":
        return PyGNodeClassifier(
            node_feats=in_channels,
            n_classes=out_channels,
            n_layers=num_layers,
            n_hidden=hidden_channels,
            dropout=dropout,
            mpnn="sage",
            input_drop=input_drop,
            edge_drop=edge_drop,
            jk=jk,
            use_bn=use_bn,
        )
    if name == "sgcn":
        return PyGNodeClassifier(
            node_feats=in_channels,
            n_classes=out_channels,
            n_layers=num_layers,
            n_hidden=hidden_channels,
            dropout=dropout,
            mpnn="gcn",
            input_drop=input_drop,
            edge_drop=edge_drop,
            jk=jk,
            use_bn=use_bn,
        )
    raise ValueError(f"Unsupported backbone: {name}")


def forward_logits(model, data, device):
    return model(data.x.to(device), data.edge_index.to(device))


def classification_accuracy(logits, labels, idx):
    return eval_acc(labels[idx], logits[idx])


def msp_scores(logits, idx):
    probs = torch.softmax(logits[idx], dim=-1)
    return probs.max(dim=1).values.detach().cpu()


def _fit_gnnsafe_energy_scorer(data, labels, train_idx, val_idx, args, device):
    det_args = copy.deepcopy(args)
    det_args.backbone = "gcn"
    det_args.use_prop = True
    det_args.use_reg = False
    det_args.use_bn = True
    det_args.T = 1.0
    det_args.epochs = max(1, int(args.sgcn_energy_scorer_epochs))

    dataset = _clone_data(data)
    dataset.y = labels.clone()
    dataset.node_idx = torch.arange(dataset.num_nodes, dtype=torch.long)
    dataset.splits = {
        "train": train_idx.clone(),
        "valid": val_idx.clone(),
        "test": train_idx.clone(),
    }

    n_classes = int(labels.max().item() + 1)
    detector = GNNSafe(dataset.x.shape[1], n_classes, det_args).to(device)
    detector.reset_parameters()
    optimizer = torch.optim.Adam(detector.parameters(), lr=det_args.lr, weight_decay=det_args.weight_decay)
    criterion = nn.NLLLoss()

    dummy_ood = _clone_data(dataset)
    dummy_ood.node_idx = train_idx.clone()

    best_val = float("inf")
    best_state = None
    for _ in range(det_args.epochs):
        detector.train()
        optimizer.zero_grad()
        loss = detector.loss_compute(dataset, dummy_ood, criterion, device, det_args)
        loss.backward()
        optimizer.step()

        detector.eval()
        with torch.no_grad():
            logits = detector(dataset, device).cpu()
            valid_out = torch.log_softmax(logits[val_idx], dim=1)
            valid_loss = criterion(valid_out, labels[val_idx].squeeze(1))
            if valid_loss.item() < best_val:
                best_val = valid_loss.item()
                best_state = deepcopy(detector.state_dict())

    if best_state is not None:
        detector.load_state_dict(best_state)
    detector.eval()
    with torch.no_grad():
        neg_energy = detector.detect(dataset, dataset.node_idx, device, det_args).detach().cpu()
    return -neg_energy


def _aggregate_subgraph_energy(node_energy, node_idx, method, trim_ratio, bottomk_ratio):
    scores = node_energy[node_idx]
    if scores.numel() == 0:
        return 0.0
    if method == "mean":
        return float(scores.mean().item())
    if method == "median":
        return float(scores.median().item())
    if method == "trimmed_mean":
        trim_n = int(scores.numel() * max(0.0, min(trim_ratio, 0.49)))
        if trim_n == 0 or scores.numel() <= 2 * trim_n:
            return float(scores.mean().item())
        sorted_scores = torch.sort(scores).values
        return float(sorted_scores[trim_n:-trim_n].mean().item())
    if method == "bottomk_mean":
        k = max(1, int(math.ceil(scores.numel() * max(0.0, min(bottomk_ratio, 1.0)))))
        sorted_scores = torch.sort(scores).values
        return float(sorted_scores[:k].mean().item())
    raise ValueError(f"Unsupported subgraph energy aggregation: {method}")


def _subgraph_energy_to_weights(scores, mode, weight_min, weight_max, tau, hard_keep_ratio):
    if not scores:
        return np.array([], dtype=np.float64)
    arr = np.asarray(scores, dtype=np.float64)
    if mode == "none":
        return np.ones_like(arr, dtype=np.float64)
    if mode == "rank":
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(arr), dtype=np.float64)
        denom = max(len(arr) - 1, 1)
        scaled = 1.0 - ranks / denom
        return weight_min + (weight_max - weight_min) * scaled
    if mode == "sigmoid":
        mu = float(arr.mean())
        sigma = float(arr.std())
        if sigma < 1e-12:
            return np.ones_like(arr, dtype=np.float64)
        z = (arr - mu) / sigma
        scaled = 1.0 / (1.0 + np.exp(tau * z))
        return weight_min + (weight_max - weight_min) * scaled
    if mode == "hard":
        keep_n = max(1, int(math.ceil(len(arr) * max(0.0, min(hard_keep_ratio, 1.0)))))
        order = np.argsort(arr)
        weights = np.zeros_like(arr, dtype=np.float64)
        weights[order[:keep_n]] = 1.0
        return weights
    raise ValueError(f"Unsupported subgraph energy weighting mode: {mode}")


def evaluate_ood_metrics(model, id_data, id_idx, ood_datasets, device):
    logits_id = forward_logits(model, id_data, device)
    id_scores = msp_scores(logits_id, id_idx)
    results = []
    for dataset in ood_datasets:
        logits_ood = forward_logits(model, dataset, device)
        ood_scores = msp_scores(logits_ood, dataset.node_idx)
        auroc, aupr, fpr, _ = get_measures(id_scores, ood_scores)
        results.append(
            {
                "auroc": auroc,
                "aupr": aupr,
                "fpr95": fpr,
            }
        )
    return results


@torch.no_grad()
def evaluate_robustness(model, train_data, id_test_idx, ood_datasets, labels, device):
    logits_train_graph = forward_logits(model, train_data, device).cpu()
    id_acc = classification_accuracy(logits_train_graph, labels.cpu(), id_test_idx)

    ood_accs = []
    for dataset in ood_datasets:
        logits_ood_graph = forward_logits(model, dataset, device).cpu()
        ood_accs.append(classification_accuracy(logits_ood_graph, labels.cpu(), dataset.node_idx))

    mean_ood_acc = float(np.mean(ood_accs)) if ood_accs else 0.0
    degradation = id_acc - mean_ood_acc
    return {
        "id_acc": float(id_acc),
        "ood_accs": [float(v) for v in ood_accs],
        "mean_ood_acc": mean_ood_acc,
        "degradation": float(degradation),
    }


def train_fullbatch(
    model,
    data,
    train_idx,
    val_idx,
    test_idx,
    labels,
    device,
    epochs,
    lr,
    weight_decay,
    sample_weights=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    y = labels.squeeze(1).to(device)
    train_idx_dev = train_idx.to(device)
    best_val = float("-inf")
    best_state = None
    weight_dev = sample_weights.to(device) if sample_weights is not None else None

    for _ in range(epochs):
        model.train()
        logits = forward_logits(model, data, device)
        if weight_dev is None:
            loss = criterion(logits[train_idx_dev], y[train_idx_dev])
        else:
            loss_per_node = F.cross_entropy(
                logits[train_idx_dev],
                y[train_idx_dev],
                reduction="none",
            )
            node_weights = weight_dev[train_idx_dev]
            loss = (loss_per_node * node_weights).sum() / node_weights.sum().clamp_min(1e-12)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = forward_logits(model, data, device).cpu()
            val_acc = classification_accuracy(logits, labels.cpu(), val_idx)
            if val_acc > best_val:
                best_val = val_acc
                best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = forward_logits(model, data, device).cpu()
    return {
        "model": model,
        "train_acc": float(classification_accuracy(logits, labels.cpu(), train_idx)),
        "val_acc": float(classification_accuracy(logits, labels.cpu(), val_idx)),
        "test_acc": float(classification_accuracy(logits, labels.cpu(), test_idx)),
    }


def _cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample_subgraph_nodes(
    edge_index,
    n_nodes,
    train_idx,
    method,
    n_sample,
    subgraph_max_nodes=None,
    unsampled_nodes=None,
):
    device = edge_index.device

    if subgraph_max_nodes is not None and subgraph_max_nodes > 0:
        n_sample = subgraph_max_nodes
    n_sample = min(n_sample, n_nodes)
    has_priority = unsampled_nodes is not None and len(unsampled_nodes) > 0

    if method == "random_node":
        if has_priority:
            n_priority = len(unsampled_nodes)
            if n_priority >= n_sample:
                perm = torch.randperm(n_priority, device=device)[:n_sample]
                return unsampled_nodes[perm].sort().values

            remaining = n_sample - n_priority
            sampled_mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
            sampled_mask[unsampled_nodes] = False
            sampled_pool = sampled_mask.nonzero(as_tuple=False).squeeze(1)
            perm = torch.randperm(len(sampled_pool), device=device)[:remaining]
            return torch.cat([unsampled_nodes, sampled_pool[perm]]).sort().values

        return torch.randperm(n_nodes, device=device)[:n_sample].sort().values

    if method == "random_edge":
        n_edges = edge_index.shape[1]
        if has_priority:
            priority_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
            priority_mask[unsampled_nodes] = True
            edge_has_priority = priority_mask[edge_index[0]] | priority_mask[edge_index[1]]
            priority_edges = edge_has_priority.nonzero(as_tuple=False).squeeze(1)
            other_edges = (~edge_has_priority).nonzero(as_tuple=False).squeeze(1)

            n_priority_sample = min(n_sample * 2, len(priority_edges))
            perm_priority = torch.randperm(len(priority_edges), device=device)[:n_priority_sample]
            chosen = edge_index[:, priority_edges[perm_priority]].flatten().unique()

            if len(chosen) < n_sample and len(other_edges) > 0:
                extra_edges = other_edges[
                    torch.randperm(len(other_edges), device=device)[: min(n_sample * 2, len(other_edges))]
                ]
                chosen = torch.cat([chosen, edge_index[:, extra_edges].flatten().unique()]).unique()
        else:
            edge_perm = torch.randperm(n_edges, device=device)[: min(n_sample * 2, n_edges)]
            chosen = edge_index[:, edge_perm].flatten().unique()

        if len(chosen) < n_sample:
            extra = torch.randperm(n_nodes, device=device)[: n_sample - len(chosen)]
            chosen = torch.cat([chosen, extra]).unique()
        return chosen[:n_sample].sort().values

    if method in ("random_walk", "snowball"):
        if has_priority:
            priority_train = unsampled_nodes[torch.isin(unsampled_nodes, train_idx)]
            seed_pool = priority_train if len(priority_train) > 0 else unsampled_nodes
        else:
            seed_pool = train_idx

        n_seeds = min(max(n_sample // _SGCN_SEED_RATIO, 1), len(seed_pool))
        seeds = seed_pool[torch.randperm(len(seed_pool), device=device)[:n_seeds]]
        visited = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        visited[seeds] = True
        row, col = edge_index
        max_hops = 2 if method == "snowball" else _SGCN_RANDOM_WALK_MAX_HOPS

        for _ in range(max_hops):
            if int(visited.sum()) >= n_sample:
                break
            visited[col[visited[row]]] = True

        node_idx = visited.nonzero(as_tuple=False).squeeze(1)
        if len(node_idx) > n_sample:
            node_idx = node_idx[torch.randperm(len(node_idx), device=device)[:n_sample]]
        elif len(node_idx) < n_sample:
            remaining = (~visited).nonzero(as_tuple=False).squeeze(1)
            extra = remaining[torch.randperm(len(remaining), device=device)[: n_sample - len(node_idx)]]
            node_idx = torch.cat([node_idx, extra])
        return node_idx.sort().values

    raise ValueError(f"Unsupported sampling method: {method}")


def _clone_state_dict(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _prepare_sampled_subgraphs(
    edge_index,
    n_nodes,
    train_idx_dev,
    n_subgraphs,
    n_sample,
    subgraph_max_nodes,
    sampling_method,
    min_subgraph_nodes,
    min_train_nodes_in_subgraph,
    max_subgraph_edges,
    node_energy,
    energy_aggregation,
    energy_trim_ratio,
    energy_bottomk_ratio,
    device,
):
    sampled = []
    epoch_sampled_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)

    for _ in range(n_subgraphs):
        unsampled_nodes = epoch_sampled_mask.logical_not().nonzero(as_tuple=False).squeeze(1)
        if len(unsampled_nodes) == 0:
            unsampled_nodes = None

        node_idx = _sample_subgraph_nodes(
            edge_index=edge_index,
            n_nodes=n_nodes,
            train_idx=train_idx_dev,
            method=sampling_method,
            n_sample=n_sample,
            subgraph_max_nodes=subgraph_max_nodes,
            unsampled_nodes=unsampled_nodes,
        )

        if min_subgraph_nodes > 0 and len(node_idx) < min_subgraph_nodes:
            target_nodes = min(min_subgraph_nodes, n_nodes)
            n_extra = target_nodes - len(node_idx)
            all_nodes = torch.arange(n_nodes, device=device)
            candidate_mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
            candidate_mask[node_idx] = False
            candidates = all_nodes[candidate_mask]
            if len(candidates) > 0:
                perm = torch.randperm(len(candidates), device=device)[:n_extra]
                node_idx = torch.cat([node_idx, candidates[perm]]).unique().sort().values

        train_in_subgraph = tensor_isin(node_idx, train_idx_dev).sum().item()
        if train_in_subgraph < min_train_nodes_in_subgraph:
            n_need = min_train_nodes_in_subgraph - train_in_subgraph
            extra_train = train_idx_dev[
                torch.randperm(len(train_idx_dev), device=device)[: min(n_need, len(train_idx_dev))]
            ]
            node_idx = torch.cat([node_idx, extra_train]).unique().sort().values

        epoch_sampled_mask[node_idx] = True
        train_mask = tensor_isin(node_idx, train_idx_dev)
        if not train_mask.any():
            continue

        in_subgraph = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        in_subgraph[node_idx] = True
        keep_edges = in_subgraph[edge_index[0]] & in_subgraph[edge_index[1]]
        sub_edges_global = edge_index[:, keep_edges]
        if max_subgraph_edges > 0 and sub_edges_global.size(1) > max_subgraph_edges:
            perm = torch.randperm(sub_edges_global.size(1), device=device)[:max_subgraph_edges]
            sub_edges_global = sub_edges_global[:, perm]

        global_to_local = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
        global_to_local[node_idx] = torch.arange(node_idx.size(0), device=device)
        sub_edge_index = global_to_local[sub_edges_global]

        subgraph_score = 0.0
        if node_energy is not None:
            subgraph_score = _aggregate_subgraph_energy(
                node_energy=node_energy,
                node_idx=node_idx,
                method=energy_aggregation,
                trim_ratio=energy_trim_ratio,
                bottomk_ratio=energy_bottomk_ratio,
            )

        sampled.append(
            {
                "node_idx": node_idx,
                "train_mask": train_mask,
                "sub_edge_index": sub_edge_index,
                "subgraph_score": subgraph_score,
            }
        )

    return sampled


def train_sgcn(
    model,
    data,
    train_idx,
    val_idx,
    test_idx,
    labels,
    device,
    epochs,
    lr,
    weight_decay,
    n_subgraphs,
    local_epochs,
    subgraph_max_nodes,
    truncation_ratio,
    aggregation_method,
    sampling_method,
    max_subgraph_edges,
    sample_weights=None,
    subgraph_ratio=0.5,
    input_drop=0.0,
    edge_drop=0.0,
    jk=False,
    min_subgraph_nodes=0,
    min_train_nodes_in_subgraph=_SGCN_MIN_TRAIN_NODES,
    energy_weighting_mode="none",
    energy_aggregation="median",
    energy_trim_ratio=0.1,
    energy_bottomk_ratio=0.2,
    energy_weight_min=0.5,
    energy_weight_max=1.5,
    energy_sigmoid_tau=1.0,
    energy_hard_keep_ratio=0.5,
    energy_scorer_args=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    x_full = data.x.to(device)
    y_full = labels.squeeze(1).to(device)
    edge_index = data.edge_index.to(device)
    train_idx_dev = train_idx.to(device)
    val_idx_dev = val_idx.to(device)
    weight_dev = sample_weights.to(device) if sample_weights is not None else None
    node_energy = None
    use_energy_weighting = energy_weighting_mode != "none"
    if use_energy_weighting:
        if energy_scorer_args is None:
            raise ValueError("energy_scorer_args must be provided when SGCN energy weighting is enabled.")
        scorer_args = energy_scorer_args
        node_energy = _fit_gnnsafe_energy_scorer(
            data=data,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            args=scorer_args,
            device=device,
        ).to(device)

    n_nodes = data.num_nodes
    if subgraph_max_nodes is not None and subgraph_max_nodes > 0:
        n_sample = subgraph_max_nodes
    else:
        n_sample = max(1, int(n_nodes * subgraph_ratio))

    if n_subgraphs <= 0:
        n_subgraphs = max(1, math.ceil(n_nodes / max(n_sample, 1)))

    best_val = float("-inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        epoch_init_state = _clone_state_dict(model)
        local_states = []
        val_scores = []
        sampled_subgraphs = _prepare_sampled_subgraphs(
            edge_index=edge_index,
            n_nodes=n_nodes,
            train_idx_dev=train_idx_dev,
            n_subgraphs=n_subgraphs,
            n_sample=n_sample,
            subgraph_max_nodes=subgraph_max_nodes,
            sampling_method=sampling_method,
            min_subgraph_nodes=min_subgraph_nodes,
            min_train_nodes_in_subgraph=min_train_nodes_in_subgraph,
            max_subgraph_edges=max_subgraph_edges,
            node_energy=node_energy if use_energy_weighting else None,
            energy_aggregation=energy_aggregation,
            energy_trim_ratio=energy_trim_ratio,
            energy_bottomk_ratio=energy_bottomk_ratio,
            device=device,
        )
        subgraph_scores = [item["subgraph_score"] for item in sampled_subgraphs]
        subgraph_weights = _subgraph_energy_to_weights(
            scores=subgraph_scores,
            mode=energy_weighting_mode,
            weight_min=energy_weight_min,
            weight_max=energy_weight_max,
            tau=energy_sigmoid_tau,
            hard_keep_ratio=energy_hard_keep_ratio,
        )
        if use_energy_weighting and len(subgraph_weights) > 0:
            positive_mask = subgraph_weights > 0
            if positive_mask.any():
                subgraph_weights = subgraph_weights.copy()
                subgraph_weights[positive_mask] = subgraph_weights[positive_mask] / max(
                    float(subgraph_weights[positive_mask].mean()),
                    1e-12,
                )

        for sg_idx, sampled in enumerate(sampled_subgraphs):
            _cuda_sync(device)
            t_sample_start = time.time()
            node_idx = sampled["node_idx"]
            train_mask = sampled["train_mask"]
            sub_edge_index = sampled["sub_edge_index"]
            x_sub = x_full[node_idx]
            y_sub = y_full[node_idx]
            subgraph_loss_weight = 1.0 if sg_idx >= len(subgraph_weights) else float(subgraph_weights[sg_idx])
            if energy_weighting_mode == "hard" and subgraph_loss_weight <= 0.0:
                continue

            model.load_state_dict(epoch_init_state)
            optimizer.state.clear()
            model.train()
            last_loss = 0.0
            for _ in range(local_epochs):
                pred = model(x_sub, sub_edge_index)
                if weight_dev is None:
                    loss = criterion(pred[train_mask], y_sub[train_mask])
                else:
                    loss_per_node = F.cross_entropy(
                        pred[train_mask],
                        y_sub[train_mask],
                        reduction="none",
                    )
                    train_node_weights = weight_dev[node_idx][train_mask]
                    loss = (
                        loss_per_node * train_node_weights
                    ).sum() / train_node_weights.sum().clamp_min(1e-12)
                if use_energy_weighting:
                    loss = loss * subgraph_loss_weight
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

            with torch.no_grad():
                if len(val_idx_dev) == 0:
                    val_score = -last_loss
                else:
                    val_sample_size = min(_SGCN_VAL_SAMPLE_SIZE, len(val_idx_dev))
                    val_sample = val_idx_dev[torch.randperm(len(val_idx_dev), device=device)[:val_sample_size]]
                    eval_node_idx = torch.cat([node_idx, val_sample]).unique().sort().values
                    in_eval = torch.zeros(n_nodes, dtype=torch.bool, device=device)
                    in_eval[eval_node_idx] = True
                    keep_eval = in_eval[edge_index[0]] & in_eval[edge_index[1]]
                    eval_edges_global = edge_index[:, keep_eval]
                    if max_subgraph_edges > 0 and eval_edges_global.size(1) > max_subgraph_edges:
                        perm = torch.randperm(eval_edges_global.size(1), device=device)[:max_subgraph_edges]
                        eval_edges_global = eval_edges_global[:, perm]
                    global_to_local_eval = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
                    global_to_local_eval[eval_node_idx] = torch.arange(eval_node_idx.size(0), device=device)
                    eval_edge_index = global_to_local_eval[eval_edges_global]
                    pred_eval = model(x_full[eval_node_idx], eval_edge_index)
                    val_mask_local = tensor_isin(eval_node_idx, val_sample)
                    val_loss = criterion(pred_eval[val_mask_local], y_full[eval_node_idx][val_mask_local])
                    val_score = -val_loss.item()

            local_states.append(_clone_state_dict(model))
            val_scores.append(val_score)

        if local_states:
            n_keep = max(1, int(len(local_states) * (1.0 - truncation_ratio)))
            kept_idx = sorted(range(len(val_scores)), key=lambda i: val_scores[i], reverse=True)[:n_keep]
            if energy_weighting_mode == "hard":
                kept_idx = [i for i in kept_idx if i < len(subgraph_weights) and subgraph_weights[i] > 0]
                if not kept_idx:
                    fallback = 0 if not local_states else min(len(local_states) - 1, int(np.argmin(np.asarray(subgraph_scores, dtype=np.float64))))
                    kept_idx = [fallback]
            kept_scores = torch.tensor([val_scores[i] for i in kept_idx], dtype=torch.float, device=device)

            if aggregation_method == "avg":
                weights = torch.ones(len(kept_idx), dtype=torch.float, device=device) / len(kept_idx)
            elif aggregation_method == "weighted":
                shifted = kept_scores - kept_scores.min() + 1e-8
                weights = shifted / shifted.sum()
            else:
                weights = torch.softmax(kept_scores, dim=0)

            if use_energy_weighting:
                energy_weights = torch.tensor([subgraph_weights[i] for i in kept_idx], dtype=torch.float, device=device)
                energy_weights = energy_weights / energy_weights.mean().clamp_min(1e-12)
                weights = weights * energy_weights
                weights = weights / weights.sum().clamp_min(1e-12)

            agg_state = {}
            for key in epoch_init_state:
                stacked = torch.stack([local_states[i][key].float() for i in kept_idx], dim=0)
                view_shape = [-1] + [1] * (stacked.dim() - 1)
                agg_state[key] = (stacked * weights.view(view_shape)).sum(dim=0).to(epoch_init_state[key].dtype)
            model.load_state_dict(agg_state)
            optimizer.state.clear()
        else:
            model.load_state_dict(epoch_init_state)

        model.eval()
        with torch.no_grad():
            logits = forward_logits(model, data, device).cpu()
            val_acc = classification_accuracy(logits, labels.cpu(), val_idx)
            if val_acc > best_val:
                best_val = val_acc
                best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = forward_logits(model, data, device).cpu()
    return {
        "model": model,
        "train_acc": float(classification_accuracy(logits, labels.cpu(), train_idx)),
        "val_acc": float(classification_accuracy(logits, labels.cpu(), val_idx)),
        "test_acc": float(classification_accuracy(logits, labels.cpu(), test_idx)),
    }


def build_induced_subgraph(full_data, global_node_idx):
    """Build a compact induced subgraph and preserve global-node bookkeeping."""
    global_node_idx = torch.unique(global_node_idx).sort().values
    edge_index, _ = subgraph(
        global_node_idx,
        full_data.edge_index,
        relabel_nodes=True,
        num_nodes=full_data.num_nodes,
    )
    graph = Data(
        x=full_data.x[global_node_idx].clone(),
        edge_index=edge_index,
        y=full_data.y[global_node_idx].clone(),
    )
    graph.node_idx = torch.arange(global_node_idx.numel(), dtype=torch.long)
    graph.global_node_idx = global_node_idx.clone()
    return graph


def build_mixed_test_graph(full_data, id_test_global_idx, ood_global_idx):
    """
    Construct an induced mixed-test graph centered on ID-test and OOD-test nodes.
    The returned graph stores both global indices and per-node ID/OOD bookkeeping.
    """
    id_test_global_idx = torch.unique(id_test_global_idx).sort().values
    ood_global_idx = torch.unique(ood_global_idx).sort().values
    mixed_global_idx = torch.unique(torch.cat([id_test_global_idx, ood_global_idx], dim=0)).sort().values
    graph = build_induced_subgraph(full_data, mixed_global_idx)

    id_test_mask = tensor_isin(graph.global_node_idx, id_test_global_idx)
    is_ood = tensor_isin(graph.global_node_idx, ood_global_idx)
    graph.id_test_mask = id_test_mask.clone()
    graph.is_ood = is_ood.clone()
    graph.id_test_local_idx = graph.node_idx[id_test_mask]
    graph.ood_local_idx = graph.node_idx[is_ood]
    return graph


def hard_remove_nodes_from_graph(graph, remove_local_idx):
    """
    Remove nodes and their incident edges from a compact graph without edge rewiring.
    Metadata for global ids and ID/OOD masks is preserved for the remaining nodes.
    """
    remove_local_idx = torch.unique(remove_local_idx).sort().values
    keep_mask = ~tensor_isin(graph.node_idx, remove_local_idx)
    keep_local_idx = graph.node_idx[keep_mask]
    edge_index, _ = subgraph(
        keep_local_idx,
        graph.edge_index,
        relabel_nodes=True,
        num_nodes=graph.num_nodes,
    )

    filtered = Data(
        x=graph.x[keep_mask].clone(),
        edge_index=edge_index,
        y=graph.y[keep_mask].clone(),
    )
    filtered.node_idx = torch.arange(int(keep_mask.sum().item()), dtype=torch.long)
    filtered.global_node_idx = graph.global_node_idx[keep_mask].clone()
    filtered.id_test_mask = graph.id_test_mask[keep_mask].clone()
    filtered.is_ood = graph.is_ood[keep_mask].clone()
    filtered.id_test_local_idx = filtered.node_idx[filtered.id_test_mask]
    filtered.ood_local_idx = filtered.node_idx[filtered.is_ood]
    return filtered


@torch.no_grad()
def evaluate_id_accuracy_on_graph(model, graph, device):
    """Evaluate accuracy only on the original ID-test nodes retained in the graph."""
    if not hasattr(graph, "id_test_local_idx"):
        raise ValueError("Graph is missing `id_test_local_idx` metadata.")
    if graph.id_test_local_idx.numel() == 0:
        raise ValueError("No retained ID-test nodes remain for evaluation.")
    logits = forward_logits(model, graph, device).cpu()
    return float(classification_accuracy(logits, graph.y.cpu(), graph.id_test_local_idx))


def compute_mixed_detection_metrics(scores, is_ood_mask):
    """
    Compute AUROC / AUPR / FPR95 on a mixed pool.
    We follow the repo's existing metric convention: ID scores are passed as positives.
    """
    is_ood_mask = is_ood_mask.to(torch.bool)
    id_scores = scores[~is_ood_mask].detach().cpu()
    ood_scores = scores[is_ood_mask].detach().cpu()
    auroc, aupr, fpr95, threshold = get_measures(id_scores, ood_scores)
    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "fpr95": float(fpr95),
        "threshold": float(threshold),
    }
