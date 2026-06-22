import argparse
import csv
import os

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from data_utils import rand_splits
from parse import parser_add_main_args
from ogb_compat import patch_master_csv_read
from two_stage_utils import (
    build_backbone,
    evaluate_ood_metrics,
    evaluate_robustness,
    set_seed,
    train_fullbatch,
    train_sgcn,
)


patch_master_csv_read()
from ogb.nodeproppred import NodePropPredDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Arxiv backbone evaluation with optional SGCN energy weighting")
    parser_add_main_args(parser)
    parser.add_argument("--unknown_year", type=int, default=2016)
    parser.add_argument("--backbones", type=str, default="gcn,sage,sgcn")
    parser.add_argument("--results_dir", type=str, default="results/two_stage_arxiv")
    parser.add_argument("--sgcn_subgraphs", type=int, default=8)
    parser.add_argument("--sgcn_local_epochs", type=int, default=3)
    parser.add_argument("--sgcn_subgraph_max_nodes", type=int, default=4096)
    parser.add_argument("--sgcn_truncation_ratio", type=float, default=0.2)
    parser.add_argument("--sgcn_aggregation", type=str, default="sgcn", choices=["sgcn", "avg", "weighted"])
    parser.add_argument("--sgcn_sampling", type=str, default="random_node", choices=["random_node", "random_edge", "random_walk", "snowball"])
    parser.add_argument("--sgcn_max_subgraph_edges", type=int, default=200000)
    return parser.parse_args()


def parse_csv_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_full_arxiv(data_dir):
    ogb_dataset = NodePropPredDataset(name="ogbn-arxiv", root=f"{data_dir}/ogb")
    edge_index = torch.as_tensor(ogb_dataset.graph["edge_index"])
    node_feat = torch.as_tensor(ogb_dataset.graph["node_feat"])
    labels = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    years = torch.as_tensor(ogb_dataset.graph["node_year"]).reshape(-1, 1)
    full_data = Data(x=node_feat, edge_index=edge_index, y=labels)
    full_data.node_idx = torch.arange(labels.size(0))
    full_data.node_year = years
    return full_data


def _build_induced_dataset(full_data, center_mask, all_mask):
    edge_index, _ = subgraph(all_mask, full_data.edge_index)
    dataset = Data(x=full_data.x, edge_index=edge_index, y=full_data.y)
    dataset.node_idx = torch.arange(full_data.y.size(0))[center_mask]
    dataset.node_year = full_data.node_year
    return dataset


def load_protocol(data_dir, unknown_year):
    full_data = load_full_arxiv(data_dir)
    years = full_data.node_year.squeeze(1)

    candidate_mask = years <= unknown_year
    training_data = _build_induced_dataset(full_data, candidate_mask, candidate_mask)

    test_year_bound = [2017, 2018, 2019, 2020]
    dataset_ood_te = []
    for i in range(len(test_year_bound) - 1):
        center_mask = (years > test_year_bound[i]) & (years <= test_year_bound[i + 1])
        all_mask = years <= test_year_bound[i + 1]
        dataset_ood_te.append(_build_induced_dataset(full_data, center_mask, all_mask))

    return training_data, dataset_ood_te


def attach_random_splits(data, train_prop, valid_prop):
    data.splits = rand_splits(data.node_idx, train_prop=train_prop, valid_prop=valid_prop)
    return data


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


def summarize_detection(results):
    if not results:
        return {"mean_auroc": 0.0, "mean_aupr": 0.0, "mean_fpr95": 0.0}
    return {
        "mean_auroc": float(np.mean([r["auroc"] for r in results])),
        "mean_aupr": float(np.mean([r["aupr"] for r in results])),
        "mean_fpr95": float(np.mean([r["fpr95"] for r in results])),
    }


def export_summary(path, rows):
    header = [
        "seed",
        "unknown_year",
        "backbone",
        "sgcn_energy_weighting",
        "id_acc",
        "mean_ood_acc",
        "degradation",
        "mean_auroc",
        "mean_aupr",
        "mean_fpr95",
        "train_acc",
        "val_acc",
        "test_acc",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    args.dataset = "arxiv"
    args.method = "gnnsafe"
    args.backbone = "gcn"

    ensure_dir(args.results_dir)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    backbones = parse_csv_list(args.backbones)
    all_rows = []

    for run_seed in range(args.seed, args.seed + args.runs):
        set_seed(run_seed)
        training_data, dataset_ood_te = load_protocol(args.data_dir, args.unknown_year)
        training_data = attach_random_splits(training_data, args.train_prop, args.valid_prop)
        labels = training_data.y.clone()

        for backbone in backbones:
            set_seed(run_seed)
            trained = train_backbone(
                backbone_name=backbone,
                data=training_data,
                labels=labels,
                train_idx=training_data.splits["train"],
                val_idx=training_data.splits["valid"],
                test_idx=training_data.splits["test"],
                args=args,
                device=device,
            )
            model = trained["model"]
            robustness = evaluate_robustness(
                model=model,
                train_data=training_data,
                id_test_idx=training_data.splits["test"],
                ood_datasets=dataset_ood_te,
                labels=labels,
                device=device,
            )
            detection = summarize_detection(
                evaluate_ood_metrics(
                    model=model,
                    id_data=training_data,
                    id_idx=training_data.splits["test"],
                    ood_datasets=dataset_ood_te,
                    device=device,
                )
            )
            all_rows.append(
                {
                    "seed": run_seed,
                    "unknown_year": args.unknown_year,
                    "backbone": backbone,
                    "sgcn_energy_weighting": args.sgcn_energy_weighting if backbone == "sgcn" else "n/a",
                    "id_acc": robustness["id_acc"],
                    "mean_ood_acc": robustness["mean_ood_acc"],
                    "degradation": robustness["degradation"],
                    "mean_auroc": detection["mean_auroc"],
                    "mean_aupr": detection["mean_aupr"],
                    "mean_fpr95": detection["mean_fpr95"],
                    "train_acc": trained["train_acc"],
                    "val_acc": trained["val_acc"],
                    "test_acc": trained["test_acc"],
                }
            )

    export_summary(os.path.join(args.results_dir, "backbone_summary.csv"), all_rows)
    print(f"Saved experiment results to {args.results_dir}")


if __name__ == "__main__":
    main()
