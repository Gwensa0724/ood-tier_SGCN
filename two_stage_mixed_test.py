import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_utils import rand_splits
from gnnsafe import GNNSafe
from parse import parser_add_main_args
from two_stage_utils import (
    build_backbone,
    build_induced_subgraph,
    build_mixed_test_graph,
    compute_mixed_detection_metrics,
    evaluate_id_accuracy_on_graph,
    hard_remove_nodes_from_graph,
    set_seed,
    train_fullbatch,
    train_sgcn,
)


def parse_comma_list(value):
    if value is None or value == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value):
    if value is None or value == "":
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Amazon mixed-test OOD purification with backbone scan")
    parser_add_main_args(parser)
    parser.add_argument("--results_dir", type=str, default="results/mixed_test_amazon_budget_scan")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--contam_strategies",
        type=str,
        nargs="+",
        default=["random_attach"],
        choices=["natural", "random_attach", "targeted_attach"],
    )
    parser.add_argument("--contam_ratios", type=float, nargs="+", default=[1.0])
    parser.add_argument("--backbones", type=str, default="gcn,sage,sgcn")
    parser.add_argument("--attach_budgets", type=str, default="8,12,16,24,32")
    parser.add_argument("--attach_budget", type=int, default=None)
    parser.add_argument("--sgcn_subgraphs", type=int, default=8)
    parser.add_argument("--sgcn_local_epochs", type=int, default=3)
    parser.add_argument("--sgcn_subgraph_max_nodes", type=int, default=4096)
    parser.add_argument("--sgcn_truncation_ratio", type=float, default=0.2)
    parser.add_argument("--sgcn_aggregation", type=str, default="sgcn", choices=["sgcn", "avg", "weighted"])
    parser.add_argument("--sgcn_sampling", type=str, default="random_node", choices=["random_node", "random_edge", "random_walk", "snowball"])
    parser.add_argument("--sgcn_max_subgraph_edges", type=int, default=200000)
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clone_data(data):
    cloned = data.clone()
    if hasattr(data, "node_idx"):
        cloned.node_idx = data.node_idx.clone()
    if hasattr(data, "splits"):
        cloned.splits = {k: v.clone() for k, v in data.splits.items()}
    if hasattr(data, "global_node_idx"):
        cloned.global_node_idx = data.global_node_idx.clone()
    if hasattr(data, "id_test_mask"):
        cloned.id_test_mask = data.id_test_mask.clone()
    if hasattr(data, "is_ood"):
        cloned.is_ood = data.is_ood.clone()
    if hasattr(data, "id_test_local_idx"):
        cloned.id_test_local_idx = data.id_test_local_idx.clone()
    if hasattr(data, "ood_local_idx"):
        cloned.ood_local_idx = data.ood_local_idx.clone()
    return cloned


def validate_args(args):
    if args.dataset != "amazon-photo":
        raise NotImplementedError("two_stage_mixed_test.py currently only supports --dataset amazon-photo.")
    if args.ood_type != "label":
        raise NotImplementedError("two_stage_mixed_test.py currently only supports --ood-type label.")
    if not args.contam_ratios:
        raise ValueError("Please provide at least one contamination ratio.")
    if any(r <= 0 or r > 1 for r in args.contam_ratios):
        raise ValueError("All contamination ratios must be in the interval (0, 1].")

    backbones = parse_comma_list(args.backbones)
    if not backbones:
        raise ValueError("Please provide at least one backbone.")
    for backbone in backbones:
        if backbone not in {"gcn", "sage", "sgcn"}:
            raise ValueError(f"Unsupported backbone: {backbone}")

    budgets = resolve_attach_budgets(args)
    if not budgets:
        raise ValueError("Please provide at least one attach budget.")
    if any(b < 0 for b in budgets):
        raise ValueError("attach budgets must be non-negative.")


def resolve_attach_budgets(args):
    if args.attach_budget is not None:
        budgets = [int(args.attach_budget)]
    elif args.attach_budgets is not None:
        budgets = parse_int_list(args.attach_budgets)
    else:
        budgets = [8, 12, 16, 24, 32]
    return sorted(dict.fromkeys(budgets))


def load_amazon_mixed_protocol(data_dir, train_prop, valid_prop):
    dataset_ind_raw, _dataset_ood_tr, dataset_ood_te = load_graph_dataset(data_dir, "amazon-photo", "label")

    full_data = clone_data(dataset_ind_raw)
    if full_data.y.dim() == 1:
        full_data.y = full_data.y.unsqueeze(1)
    full_data.node_idx = torch.arange(full_data.num_nodes, dtype=torch.long)

    trusted_global_idx = dataset_ind_raw.node_idx.clone()
    trusted_graph = build_induced_subgraph(full_data, trusted_global_idx)
    trusted_graph.splits = rand_splits(trusted_graph.node_idx, train_prop=train_prop, valid_prop=valid_prop)

    id_test_global_idx = trusted_graph.global_node_idx[trusted_graph.splits["test"]].clone()
    ood_test_global_idx = dataset_ood_te.node_idx.clone()
    if torch.numel(id_test_global_idx) == 0 or torch.numel(ood_test_global_idx) == 0:
        raise ValueError("Mixed-test protocol requires non-empty ID test nodes and OOD test nodes.")

    return {
        "full_data": full_data,
        "trusted_graph": trusted_graph,
        "id_test_global_idx": id_test_global_idx,
        "ood_test_global_idx": ood_test_global_idx,
    }


def sample_ood_subset(ood_global_idx, contam_ratio, perm=None):
    if perm is None:
        perm = torch.randperm(ood_global_idx.numel())
    if contam_ratio >= 1.0:
        return ood_global_idx[perm].clone()
    sample_size = max(1, int(round(ood_global_idx.numel() * contam_ratio)))
    return ood_global_idx[perm[:sample_size]].clone()


def build_cross_edge_targets(graph, strategy, max_budget):
    if max_budget <= 0 or graph.ood_local_idx.numel() == 0 or graph.id_test_local_idx.numel() == 0:
        return torch.empty((graph.ood_local_idx.numel(), 0), dtype=torch.long)

    if strategy == "natural":
        return torch.empty((graph.ood_local_idx.numel(), 0), dtype=torch.long)

    if strategy == "random_attach":
        num_id = graph.id_test_local_idx.numel()
        rows = []
        for _ in range(graph.ood_local_idx.numel()):
            perm = graph.id_test_local_idx[torch.randperm(num_id)]
            if max_budget <= num_id:
                row = perm[:max_budget]
            else:
                repeat_factor = (max_budget + num_id - 1) // max(num_id, 1)
                row = perm.repeat(repeat_factor)[:max_budget]
            rows.append(row)
        return torch.stack(rows, dim=0)

    if strategy == "targeted_attach":
        degrees = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)
        id_degrees = degrees[graph.id_test_local_idx]
        ranked_id = graph.id_test_local_idx[torch.argsort(id_degrees, descending=True)]
        top_k = ranked_id[: min(max_budget, ranked_id.numel())]
        if top_k.numel() == 0:
            return torch.empty((graph.ood_local_idx.numel(), 0), dtype=torch.long)
        if top_k.numel() < max_budget:
            repeat_factor = (max_budget + top_k.numel() - 1) // max(top_k.numel(), 1)
            top_k = top_k.repeat(repeat_factor)[:max_budget]
        return top_k.unsqueeze(0).repeat(graph.ood_local_idx.numel(), 1)

    raise ValueError(f"Unsupported contamination strategy: {strategy}")


def build_contaminated_graph_from_targets(base_graph, target_matrix, attach_budget):
    contaminated = clone_data(base_graph)
    if attach_budget <= 0 or target_matrix.numel() == 0:
        return contaminated

    target_idx = target_matrix[:, :attach_budget]
    source_idx = base_graph.ood_local_idx.unsqueeze(1).repeat(1, attach_budget)
    edge_pairs = torch.stack([target_idx.reshape(-1), source_idx.reshape(-1)], dim=0)
    extra = torch.stack(
        [
            torch.cat([edge_pairs[0], edge_pairs[1]], dim=0),
            torch.cat([edge_pairs[1], edge_pairs[0]], dim=0),
        ],
        dim=0,
    )
    edge_index = torch.cat([contaminated.edge_index, extra], dim=1)
    edge_index = torch.unique(edge_index.t(), dim=0).t().contiguous()
    contaminated.edge_index = edge_index
    return contaminated


def compute_contamination_diagnostics(graph):
    row, col = graph.edge_index
    id_mask = graph.id_test_mask
    ood_mask = graph.is_ood

    id_to_ood = id_mask[row] & ood_mask[col]
    cross_edge_count = int(id_to_ood.sum().item())

    ood_neighbors_per_id = torch.bincount(row[id_to_ood], minlength=graph.num_nodes)[graph.id_test_local_idx]
    id_with_ood_neighbor = int((ood_neighbors_per_id > 0).sum().item())
    id_total = max(int(graph.id_test_local_idx.numel()), 1)

    adjacency = [[] for _ in range(graph.num_nodes)]
    for src, dst in graph.edge_index.t().tolist():
        adjacency[src].append(dst)

    two_hop_ratios = []
    id_reach_ood_two_hop = 0
    for node in graph.id_test_local_idx.tolist():
        neighbors = adjacency[node]
        if not neighbors:
            two_hop_ratios.append(0.0)
            continue
        reach = set()
        for nbr in neighbors:
            reach.update(adjacency[nbr])
        reach.discard(node)
        if not reach:
            two_hop_ratios.append(0.0)
            continue
        ood_hits = sum(1 for nbr in reach if bool(graph.is_ood[nbr]))
        ratio = ood_hits / len(reach)
        two_hop_ratios.append(ratio)
        if ood_hits > 0:
            id_reach_ood_two_hop += 1

    return {
        "cross_edge_count": cross_edge_count,
        "id_nodes_with_ood_1hop": id_with_ood_neighbor,
        "id_nodes_with_ood_1hop_ratio": id_with_ood_neighbor / id_total,
        "avg_ood_neighbors_per_id": float(ood_neighbors_per_id.float().mean().item()) if ood_neighbors_per_id.numel() else 0.0,
        "avg_ood_two_hop_ratio": float(sum(two_hop_ratios) / len(two_hop_ratios)) if two_hop_ratios else 0.0,
        "id_nodes_with_ood_2hop_ratio": id_reach_ood_two_hop / id_total,
    }


def train_backbone(backbone_name, data, labels, train_idx, val_idx, test_idx, args, device):
    n_classes = int(labels.max().item() + 1)
    model = build_backbone(
        name=backbone_name,
        in_channels=data.x.shape[1],
        out_channels=n_classes,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_bn=args.use_bn,
        input_drop=args.sgcn_input_drop,
        edge_drop=args.sgcn_edge_drop,
        jk=args.sgcn_jk,
    ).to(device)

    if backbone_name == "sgcn":
        return train_sgcn(
            model=model,
            data=data,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            labels=labels,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_subgraphs=args.sgcn_subgraphs,
            local_epochs=args.sgcn_local_epochs,
            subgraph_max_nodes=args.sgcn_subgraph_max_nodes,
            truncation_ratio=args.sgcn_truncation_ratio,
            aggregation_method=args.sgcn_aggregation,
            sampling_method=args.sgcn_sampling,
            max_subgraph_edges=args.sgcn_max_subgraph_edges,
            subgraph_ratio=0.5,
            input_drop=args.sgcn_input_drop,
            edge_drop=args.sgcn_edge_drop,
            jk=args.sgcn_jk,
            min_subgraph_nodes=args.sgcn_min_subgraph_nodes,
            min_train_nodes_in_subgraph=args.sgcn_min_train_nodes,
            energy_weighting_mode=args.sgcn_energy_weighting,
            energy_aggregation=args.sgcn_energy_aggregation,
            energy_trim_ratio=args.sgcn_energy_trim_ratio,
            energy_bottomk_ratio=args.sgcn_energy_bottomk_ratio,
            energy_weight_min=args.sgcn_energy_weight_min,
            energy_weight_max=args.sgcn_energy_weight_max,
            energy_sigmoid_tau=args.sgcn_energy_sigmoid_tau,
            energy_hard_keep_ratio=args.sgcn_energy_hard_keep_ratio,
            energy_scorer_args=args,
        )

    return train_fullbatch(
        model=model,
        data=data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        labels=labels,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def train_detector(data, labels, train_idx, val_idx, args, device):
    n_classes = int(labels.max().item() + 1)
    detector = GNNSafe(data.x.shape[1], n_classes, args).to(device)
    detector.reset_parameters()
    optimizer = torch.optim.Adam(detector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss()

    dataset = clone_data(data)
    dataset.y = labels.clone()
    dataset.splits = {
        "train": train_idx.clone(),
        "valid": val_idx.clone(),
        "test": train_idx.clone(),
    }
    if not hasattr(dataset, "node_idx"):
        dataset.node_idx = torch.arange(dataset.num_nodes, dtype=torch.long)

    dummy_ood = clone_data(dataset)
    dummy_ood.node_idx = train_idx.clone()

    best_val = float("inf")
    best_state = None
    for _ in range(args.epochs):
        detector.train()
        optimizer.zero_grad()
        loss = detector.loss_compute(dataset, dummy_ood, criterion, device, args)
        loss.backward()
        optimizer.step()

        detector.eval()
        with torch.no_grad():
            logits = detector(dataset, device).cpu()
            valid_out = torch.log_softmax(logits[val_idx], dim=1)
            valid_loss = criterion(valid_out, labels[val_idx].squeeze(1))
            if valid_loss.item() < best_val:
                best_val = valid_loss.item()
                best_state = {k: v.detach().clone() for k, v in detector.state_dict().items()}

    if best_state is not None:
        detector.load_state_dict(best_state)
    detector.eval()
    return detector


def safe_id_accuracy(model, graph, device):
    if not hasattr(graph, "id_test_local_idx") or graph.id_test_local_idx.numel() == 0:
        return float("nan")
    return float(evaluate_id_accuracy_on_graph(model, graph, device))


def filter_graph_with_detector(graph, scores, removal_k):
    if removal_k <= 0 or graph.num_nodes == 0:
        return clone_data(graph)
    removal_k = min(int(removal_k), int(graph.num_nodes))
    remove_local_idx = torch.argsort(scores.detach().cpu(), descending=False)[:removal_k]
    return hard_remove_nodes_from_graph(graph, remove_local_idx)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows, group_keys, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary = {group_keys[i]: key[i] for i in range(len(group_keys))}
        for metric in metric_keys:
            values = np.array([float(row[metric]) for row in group_rows], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values)) if values.size else float("nan")
            summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
        summary["n_seeds"] = len(group_rows)
        summaries.append(summary)
    return summaries


def rank_values(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def pearson_corr(x, y):
    if len(x) < 2:
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    if len(x) < 2:
        return float("nan")
    return pearson_corr(rank_values(np.asarray(x, dtype=float)), rank_values(np.asarray(y, dtype=float)))


def summarize_correlations(mean_rows, metric_keys):
    grouped = {}
    for row in mean_rows:
        key = (row["strategy"], row["contam_ratio"], row["backbone"])
        grouped.setdefault(key, []).append(row)

    rows = []
    for (strategy, contam_ratio, backbone), group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        budgets = [float(row["attach_budget"]) for row in group_rows]
        for metric in metric_keys:
            values = [float(row[f"{metric}_mean"]) for row in group_rows]
            rows.append(
                {
                    "strategy": strategy,
                    "contam_ratio": contam_ratio,
                    "backbone": backbone,
                    "metric": metric,
                    "pearson": pearson_corr(budgets, values),
                    "spearman": spearman_corr(budgets, values),
                }
            )
    return rows


def ratio_tag(value):
    return f"{value:.2f}".replace(".", "p")


def plot_metric_grid(mean_rows, strategy, contam_ratio, metric_keys, title, out_path, ylabel_map=None):
    budgets = sorted({int(row["attach_budget"]) for row in mean_rows if row["strategy"] == strategy and row["contam_ratio"] == contam_ratio})
    backbones = sorted({row["backbone"] for row in mean_rows if row["strategy"] == strategy and row["contam_ratio"] == contam_ratio})
    if not budgets or not backbones:
        return

    n_metrics = len(metric_keys)
    ncols = 2 if n_metrics > 1 else 1
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metric_keys):
        ax = axes_flat[idx]
        for backbone in backbones:
            ys = []
            for budget in budgets:
                matched = [
                    row
                    for row in mean_rows
                    if row["strategy"] == strategy
                    and row["contam_ratio"] == contam_ratio
                    and row["backbone"] == backbone
                    and int(row["attach_budget"]) == budget
                ]
                ys.append(float(matched[0][f"{metric}_mean"]) if matched else float("nan"))
            ax.plot(budgets, ys, marker="o", linewidth=2, label=backbone)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("attach budget")
        ax.set_ylabel(ylabel_map.get(metric, metric) if ylabel_map else metric)
        ax.grid(True, alpha=0.25)
    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_budget_curves(mean_rows, out_dir):
    ylabel_map = {
        "cross_edge_count": "cross edges",
        "id_nodes_with_ood_1hop_ratio": "ratio",
        "avg_ood_two_hop_ratio": "ratio",
        "clean_id_acc": "accuracy",
        "contaminated_id_acc": "accuracy",
        "filtered_id_acc": "accuracy",
        "auroc": "AUROC",
        "aupr": "AUPR",
        "fpr95": "FPR95",
    }

    groups = sorted({(row["strategy"], row["contam_ratio"]) for row in mean_rows})
    for strategy, contam_ratio in groups:
        label = f"{strategy}_r{ratio_tag(contam_ratio)}"
        plot_metric_grid(
            mean_rows,
            strategy,
            contam_ratio,
            ["cross_edge_count", "id_nodes_with_ood_1hop_ratio", "avg_ood_two_hop_ratio"],
            f"Structure response curves - {strategy} / ratio={contam_ratio:.2f}",
            os.path.join(out_dir, f"budget_scan_{label}_structure.png"),
            ylabel_map=ylabel_map,
        )
        plot_metric_grid(
            mean_rows,
            strategy,
            contam_ratio,
            ["clean_id_acc", "contaminated_id_acc", "filtered_id_acc"],
            f"Performance response curves - {strategy} / ratio={contam_ratio:.2f}",
            os.path.join(out_dir, f"budget_scan_{label}_performance.png"),
            ylabel_map=ylabel_map,
        )
        plot_metric_grid(
            mean_rows,
            strategy,
            contam_ratio,
            ["auroc", "aupr", "fpr95"],
            f"Detection response curves - {strategy} / ratio={contam_ratio:.2f}",
            os.path.join(out_dir, f"budget_scan_{label}_detection.png"),
            ylabel_map=ylabel_map,
        )


def write_summary_md(path, detection_rows, backbone_rows, mean_rows, attach_budgets):
    include_budget = len(attach_budgets) > 1
    lines = [
        "# Amazon Mixed-Test OOD Purification + Robustness",
        "",
        f"- Attach budgets: {', '.join(str(b) for b in attach_budgets)}",
        "- Mixed pool is constructed from `dataset_ind.splits[\"test\"]` plus the sampled OOD nodes.",
        "- Filtering uses hard node removal with `k = sampled OOD node count` for each setting.",
        "",
    ]

    detection_groups = sorted(
        {
            (
                row["strategy"],
                row["contam_ratio"],
                row["attach_budget"] if include_budget else None,
            )
            for row in detection_rows
        }
    )
    for strategy, contam_ratio, attach_budget in detection_groups:
        section = f"## Stage 1 Detection - {strategy} / ratio={contam_ratio:.2f}"
        if include_budget:
            section += f" / budget={int(attach_budget)}"
        lines.extend(
            [
                section,
                "",
                "| Seed | AUROC | AUPR | FPR95 | OOD Nodes | Cross Edges | ID w/ OOD 1-hop | Avg OOD 2-hop Ratio |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        group_rows = [
            row
            for row in detection_rows
            if row["strategy"] == strategy
            and row["contam_ratio"] == contam_ratio
            and ((not include_budget) or row["attach_budget"] == attach_budget)
        ]
        for row in sorted(group_rows, key=lambda item: item["seed"]):
            lines.append(
                f"| {row['seed']} | {row['auroc']:.4f} | {row['aupr']:.4f} | {row['fpr95']:.4f} | "
                f"{int(row['ood_nodes'])} | {int(row['cross_edge_count'])} | {row['id_nodes_with_ood_1hop_ratio']:.4f} | "
                f"{row['avg_ood_two_hop_ratio']:.4f} |"
            )
        lines.append("")

    backbone_groups = sorted(
        {
            (
                row["strategy"],
                row["contam_ratio"],
                row["attach_budget"] if include_budget else None,
            )
            for row in backbone_rows
        }
    )
    for strategy, contam_ratio, attach_budget in backbone_groups:
        section = f"## Stage 2 Robustness - {strategy} / ratio={contam_ratio:.2f}"
        if include_budget:
            section += f" / budget={int(attach_budget)}"
        lines.extend(
            [
                section,
                "",
                "| Seed | Backbone | Clean ID Acc | Contaminated ID Acc | Filtered ID Acc | Acc Drop | Acc Recovery | Remaining Gap |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        group_rows = [
            row
            for row in backbone_rows
            if row["strategy"] == strategy
            and row["contam_ratio"] == contam_ratio
            and ((not include_budget) or row["attach_budget"] == attach_budget)
        ]
        for row in sorted(group_rows, key=lambda item: (item["seed"], item["backbone"])):
            lines.append(
                f"| {row['seed']} | {row['backbone']} | {row['clean_id_acc']:.4f} | {row['contaminated_id_acc']:.4f} | "
                f"{row['filtered_id_acc']:.4f} | {row['acc_drop_contam']:.4f} | {row['acc_recovery']:.4f} | {row['remaining_gap']:.4f} |"
            )
        lines.append("")

    lines.extend(["## Mean over Seeds", ""])
    if include_budget:
        lines.extend(
            [
                "| Strategy | Ratio | Budget | Backbone | Clean ID Acc | Contaminated ID Acc | Filtered ID Acc | Acc Drop | Acc Recovery | Remaining Gap |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Strategy | Ratio | Backbone | Clean ID Acc | Contaminated ID Acc | Filtered ID Acc | Acc Drop | Acc Recovery | Remaining Gap |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
    for row in sorted(
        mean_rows,
        key=lambda item: (
            item["strategy"],
            item["contam_ratio"],
            int(item["attach_budget"]),
            item["backbone"],
        ),
    ):
        if include_budget:
            lines.append(
                f"| {row['strategy']} | {float(row['contam_ratio']):.2f} | {int(row['attach_budget'])} | {row['backbone']} | "
                f"{row['clean_id_acc_mean']:.4f} +- {row['clean_id_acc_std']:.4f} | "
                f"{row['contaminated_id_acc_mean']:.4f} +- {row['contaminated_id_acc_std']:.4f} | "
                f"{row['filtered_id_acc_mean']:.4f} +- {row['filtered_id_acc_std']:.4f} | "
                f"{row['acc_drop_contam_mean']:.4f} +- {row['acc_drop_contam_std']:.4f} | "
                f"{row['acc_recovery_mean']:.4f} +- {row['acc_recovery_std']:.4f} | "
                f"{row['remaining_gap_mean']:.4f} +- {row['remaining_gap_std']:.4f} |"
            )
        else:
            lines.append(
                f"| {row['strategy']} | {float(row['contam_ratio']):.2f} | {row['backbone']} | "
                f"{row['clean_id_acc_mean']:.4f} +- {row['clean_id_acc_std']:.4f} | "
                f"{row['contaminated_id_acc_mean']:.4f} +- {row['contaminated_id_acc_std']:.4f} | "
                f"{row['filtered_id_acc_mean']:.4f} +- {row['filtered_id_acc_std']:.4f} | "
                f"{row['acc_drop_contam_mean']:.4f} +- {row['acc_drop_contam_std']:.4f} | "
                f"{row['acc_recovery_mean']:.4f} +- {row['acc_recovery_std']:.4f} | "
                f"{row['remaining_gap_mean']:.4f} +- {row['remaining_gap_std']:.4f} |"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_graph_dataset(data_dir, dataset_name, ood_type):
    from dataset import load_graph_dataset as _load_graph_dataset

    return _load_graph_dataset(data_dir, dataset_name, ood_type)


def main():
    args = parse_args()
    validate_args(args)
    ensure_dir(args.results_dir)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    backbones = parse_comma_list(args.backbones)
    attach_budgets = resolve_attach_budgets(args)
    run_seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.runs))
    include_budget = len(attach_budgets) > 1

    detection_rows = []
    backbone_rows = []

    for run_seed in run_seeds:
        set_seed(run_seed)
        protocol = load_amazon_mixed_protocol(args.data_dir, args.train_prop, args.valid_prop)
        trusted_graph = protocol["trusted_graph"]
        train_labels = trusted_graph.y.clone()
        backbone_models = {}
        backbone_base_rows = {}

        for backbone in backbones:
            set_seed(run_seed)
            trained = train_backbone(
                backbone_name=backbone,
                data=trusted_graph,
                labels=train_labels,
                train_idx=trusted_graph.splits["train"],
                val_idx=trusted_graph.splits["valid"],
                test_idx=trusted_graph.splits["test"],
                args=args,
                device=device,
            )
            backbone_models[backbone] = trained["model"]
            backbone_base_rows[backbone] = {
                "train_acc": trained["train_acc"],
                "val_acc": trained["val_acc"],
                "test_acc": trained["test_acc"],
                "clean_id_acc": trained["test_acc"],
            }

        set_seed(run_seed)
        detector = train_detector(
            data=trusted_graph,
            labels=train_labels,
            train_idx=trusted_graph.splits["train"],
            val_idx=trusted_graph.splits["valid"],
            args=args,
            device=device,
        )

        ood_perm_cache = {}
        for strategy in args.contam_strategies:
            for contam_ratio in args.contam_ratios:
                cache_key = (strategy, contam_ratio)
                if cache_key not in ood_perm_cache:
                    ood_perm_cache[cache_key] = torch.randperm(protocol["ood_test_global_idx"].numel())
                selected_ood_global_idx = sample_ood_subset(protocol["ood_test_global_idx"], contam_ratio, perm=ood_perm_cache[cache_key])
                base_mixed_graph = build_mixed_test_graph(
                    protocol["full_data"],
                    protocol["id_test_global_idx"],
                    selected_ood_global_idx,
                )
                target_matrix = build_cross_edge_targets(base_mixed_graph, strategy, max(attach_budgets))

                for attach_budget in attach_budgets:
                    contaminated_graph = build_contaminated_graph_from_targets(base_mixed_graph, target_matrix, attach_budget)
                    with torch.no_grad():
                        mixed_scores = detector.detect(
                            contaminated_graph,
                            contaminated_graph.node_idx,
                            device,
                            args,
                        ).detach().cpu()
                    detection = compute_mixed_detection_metrics(mixed_scores, contaminated_graph.is_ood)
                    diagnostics = compute_contamination_diagnostics(contaminated_graph)
                    removal_k = int(contaminated_graph.ood_local_idx.numel())
                    filtered_graph = filter_graph_with_detector(contaminated_graph, mixed_scores, removal_k)

                    detection_rows.append(
                        {
                            "seed": run_seed,
                            "strategy": strategy,
                            "contam_ratio": contam_ratio,
                            "attach_budget": attach_budget,
                            "auroc": detection["auroc"],
                            "aupr": detection["aupr"],
                            "fpr95": detection["fpr95"],
                            "mixed_nodes": int(contaminated_graph.num_nodes),
                            "id_nodes": int(contaminated_graph.id_test_local_idx.numel()),
                            "ood_nodes": int(contaminated_graph.ood_local_idx.numel()),
                            "removal_k": removal_k,
                            **diagnostics,
                        }
                    )

                    for backbone in backbones:
                        model = backbone_models[backbone]
                        clean_id_acc = float(backbone_base_rows[backbone]["clean_id_acc"])
                        contaminated_id_acc = safe_id_accuracy(model, contaminated_graph, device)
                        filtered_id_acc = safe_id_accuracy(model, filtered_graph, device)
                        backbone_rows.append(
                            {
                                "seed": run_seed,
                                "strategy": strategy,
                                "contam_ratio": contam_ratio,
                                "attach_budget": attach_budget,
                                "backbone": backbone,
                                "clean_id_acc": clean_id_acc,
                                "contaminated_id_acc": contaminated_id_acc,
                                "filtered_id_acc": filtered_id_acc,
                                "acc_drop_contam": clean_id_acc - contaminated_id_acc,
                                "acc_recovery": filtered_id_acc - contaminated_id_acc,
                                "remaining_gap": clean_id_acc - filtered_id_acc,
                                "auroc": detection["auroc"],
                                "aupr": detection["aupr"],
                                "fpr95": detection["fpr95"],
                                "cross_edge_count": diagnostics["cross_edge_count"],
                                "id_nodes_with_ood_1hop": diagnostics["id_nodes_with_ood_1hop"],
                                "id_nodes_with_ood_1hop_ratio": diagnostics["id_nodes_with_ood_1hop_ratio"],
                                "avg_ood_neighbors_per_id": diagnostics["avg_ood_neighbors_per_id"],
                                "avg_ood_two_hop_ratio": diagnostics["avg_ood_two_hop_ratio"],
                                "id_nodes_with_ood_2hop_ratio": diagnostics["id_nodes_with_ood_2hop_ratio"],
                                "train_acc": backbone_base_rows[backbone]["train_acc"],
                                "val_acc": backbone_base_rows[backbone]["val_acc"],
                                "test_acc": backbone_base_rows[backbone]["test_acc"],
                            }
                        )

    metric_keys = [
        "clean_id_acc",
        "contaminated_id_acc",
        "filtered_id_acc",
        "acc_drop_contam",
        "acc_recovery",
        "remaining_gap",
        "auroc",
        "aupr",
        "fpr95",
        "cross_edge_count",
        "id_nodes_with_ood_1hop_ratio",
        "avg_ood_two_hop_ratio",
    ]
    mean_rows = aggregate_rows(backbone_rows, ["strategy", "contam_ratio", "attach_budget", "backbone"], metric_keys)
    correlation_rows = summarize_correlations(
        mean_rows,
        [
            "cross_edge_count",
            "id_nodes_with_ood_1hop_ratio",
            "avg_ood_two_hop_ratio",
            "contaminated_id_acc",
            "filtered_id_acc",
            "acc_recovery",
            "auroc",
            "aupr",
            "fpr95",
        ],
    )

    legacy_detection_fields = ["seed", "strategy", "contam_ratio"]
    legacy_backbone_fields = ["seed", "strategy", "contam_ratio"]
    legacy_mean_fields = ["strategy", "contam_ratio"]
    if include_budget:
        legacy_detection_fields.append("attach_budget")
        legacy_backbone_fields.append("attach_budget")
        legacy_mean_fields.append("attach_budget")

    legacy_detection_fields.extend(
        [
            "auroc",
            "aupr",
            "fpr95",
            "mixed_nodes",
            "id_nodes",
            "ood_nodes",
            "removal_k",
            "cross_edge_count",
            "id_nodes_with_ood_1hop",
            "id_nodes_with_ood_1hop_ratio",
            "avg_ood_neighbors_per_id",
            "avg_ood_two_hop_ratio",
            "id_nodes_with_ood_2hop_ratio",
        ]
    )
    legacy_backbone_fields.extend(
        [
            "backbone",
            "clean_id_acc",
            "contaminated_id_acc",
            "filtered_id_acc",
            "acc_drop_contam",
            "acc_recovery",
            "remaining_gap",
        ]
    )
    legacy_mean_fields.extend(
        [
            "backbone",
            "clean_id_acc_mean",
            "clean_id_acc_std",
            "contaminated_id_acc_mean",
            "contaminated_id_acc_std",
            "filtered_id_acc_mean",
            "filtered_id_acc_std",
            "acc_drop_contam_mean",
            "acc_drop_contam_std",
            "acc_recovery_mean",
            "acc_recovery_std",
            "remaining_gap_mean",
            "remaining_gap_std",
        ]
    )

    write_csv(
        os.path.join(args.results_dir, "mixed_detection_summary.csv"),
        legacy_detection_fields,
        detection_rows,
    )
    write_csv(
        os.path.join(args.results_dir, "mixed_test_backbone_summary.csv"),
        legacy_backbone_fields,
        backbone_rows,
    )
    write_csv(
        os.path.join(args.results_dir, "mixed_test_backbone_mean.csv"),
        legacy_mean_fields,
        mean_rows,
    )
    write_csv(
        os.path.join(args.results_dir, "mixed_budget_scan_detection.csv"),
        [
            "seed",
            "strategy",
            "contam_ratio",
            "attach_budget",
            "auroc",
            "aupr",
            "fpr95",
            "mixed_nodes",
            "id_nodes",
            "ood_nodes",
            "removal_k",
            "cross_edge_count",
            "id_nodes_with_ood_1hop",
            "id_nodes_with_ood_1hop_ratio",
            "avg_ood_neighbors_per_id",
            "avg_ood_two_hop_ratio",
            "id_nodes_with_ood_2hop_ratio",
        ],
        detection_rows,
    )
    write_csv(
        os.path.join(args.results_dir, "mixed_budget_scan_summary.csv"),
        [
            "seed",
            "strategy",
            "contam_ratio",
            "attach_budget",
            "backbone",
            "clean_id_acc",
            "contaminated_id_acc",
            "filtered_id_acc",
            "acc_drop_contam",
            "acc_recovery",
            "remaining_gap",
            "auroc",
            "aupr",
            "fpr95",
            "cross_edge_count",
            "id_nodes_with_ood_1hop",
            "id_nodes_with_ood_1hop_ratio",
            "avg_ood_neighbors_per_id",
            "avg_ood_two_hop_ratio",
            "id_nodes_with_ood_2hop_ratio",
            "train_acc",
            "val_acc",
            "test_acc",
        ],
        backbone_rows,
    )
    write_csv(
        os.path.join(args.results_dir, "mixed_budget_scan_mean.csv"),
        [
            "strategy",
            "contam_ratio",
            "attach_budget",
            "backbone",
            "clean_id_acc_mean",
            "clean_id_acc_std",
            "contaminated_id_acc_mean",
            "contaminated_id_acc_std",
            "filtered_id_acc_mean",
            "filtered_id_acc_std",
            "acc_drop_contam_mean",
            "acc_drop_contam_std",
            "acc_recovery_mean",
            "acc_recovery_std",
            "remaining_gap_mean",
            "remaining_gap_std",
            "auroc_mean",
            "auroc_std",
            "aupr_mean",
            "aupr_std",
            "fpr95_mean",
            "fpr95_std",
            "cross_edge_count_mean",
            "cross_edge_count_std",
            "id_nodes_with_ood_1hop_ratio_mean",
            "id_nodes_with_ood_1hop_ratio_std",
            "avg_ood_two_hop_ratio_mean",
            "avg_ood_two_hop_ratio_std",
            "n_seeds",
        ],
        mean_rows,
    )
    write_csv(
        os.path.join(args.results_dir, "mixed_budget_scan_correlation.csv"),
        ["strategy", "contam_ratio", "backbone", "metric", "pearson", "spearman"],
        correlation_rows,
    )
    write_summary_md(
        os.path.join(args.results_dir, "summary.md"),
        detection_rows,
        backbone_rows,
        mean_rows,
        attach_budgets,
    )
    plot_budget_curves(mean_rows, args.results_dir)
    print(f"Saved experiment results to {args.results_dir}")


if __name__ == "__main__":
    main()
