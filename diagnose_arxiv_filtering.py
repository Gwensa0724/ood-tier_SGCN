import argparse
import csv
import os
from copy import deepcopy

import torch

from parse import parser_add_main_args
from two_stage_arxiv import (
    attach_random_splits,
    ensure_dir,
    load_protocol,
    summarize_detection,
    train_backbone,
)
from two_stage_utils import evaluate_ood_metrics, evaluate_robustness, set_seed


def parse_csv_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnostic arxiv SGCN energy-weighting experiments")
    parser_add_main_args(parser)
    parser.add_argument("--unknown_year", type=int, default=2016)
    parser.add_argument("--backbones", type=str, default="gcn,sage,sgcn")
    parser.add_argument("--energy_weightings", type=str, default="none,rank,sigmoid,hard")
    parser.add_argument("--results_dir", type=str, default="results/arxiv_filter_diagnosis")
    parser.add_argument("--sgcn_subgraphs", type=int, default=8)
    parser.add_argument("--sgcn_local_epochs", type=int, default=3)
    parser.add_argument("--sgcn_subgraph_max_nodes", type=int, default=4096)
    parser.add_argument("--sgcn_truncation_ratio", type=float, default=0.2)
    parser.add_argument("--sgcn_aggregation", type=str, default="sgcn", choices=["sgcn", "avg", "weighted"])
    parser.add_argument("--sgcn_sampling", type=str, default="random_node", choices=["random_node", "random_edge", "random_walk", "snowball"])
    parser.add_argument("--sgcn_max_subgraph_edges", type=int, default=200000)
    return parser.parse_args()


def export_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    args.method = "gnnsafe"
    args.backbone = "gcn"
    args.dataset = "arxiv"
    ensure_dir(args.results_dir)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    backbones = parse_csv_list(args.backbones)
    energy_weightings = parse_csv_list(args.energy_weightings)
    if not backbones:
        raise ValueError("Please provide at least one backbone.")
    if "sgcn" in backbones and not energy_weightings:
        raise ValueError("Please provide at least one SGCN energy weighting mode.")

    rows = []
    for run_seed in range(args.seed, args.seed + args.runs):
        set_seed(run_seed)
        training_data, dataset_ood_te = load_protocol(args.data_dir, args.unknown_year)
        training_data = attach_random_splits(training_data, args.train_prop, args.valid_prop)
        labels = training_data.y.clone()

        for backbone in backbones:
            weighting_modes = energy_weightings if backbone == "sgcn" else ["n/a"]
            for weighting_mode in weighting_modes:
                run_args = deepcopy(args)
                if backbone == "sgcn":
                    run_args.sgcn_energy_weighting = weighting_mode

                set_seed(run_seed)
                trained = train_backbone(
                    backbone_name=backbone,
                    data=training_data,
                    labels=labels,
                    train_idx=training_data.splits["train"],
                    val_idx=training_data.splits["valid"],
                    test_idx=training_data.splits["test"],
                    args=run_args,
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
                rows.append(
                    {
                        "seed": run_seed,
                        "unknown_year": args.unknown_year,
                        "backbone": backbone,
                        "sgcn_energy_weighting": weighting_mode,
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

    export_csv(
        os.path.join(args.results_dir, "diagnostic_summary.csv"),
        [
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
        ],
        rows,
    )
    print(f"Saved diagnostic experiment results to {args.results_dir}")


if __name__ == "__main__":
    main()
