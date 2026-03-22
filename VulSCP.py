import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

from model import load_data, VulSCPTrainer


def load_split_dataframe(pathname, filename):
    pathname = pathname if pathname.endswith(("/", "\\")) else pathname + "/"
    return load_data(pathname + filename)


def get_run_dataframes(pathname):
    train_df = load_split_dataframe(pathname, "train.pkl")
    valid_df = load_split_dataframe(pathname, "valid.pkl")
    test_df = load_split_dataframe(pathname, "test.pkl")
    return train_df, valid_df, test_df


def parse_options():
    parser = argparse.ArgumentParser(description="Train VulSCP on a fixed 7:2:1 split.")
    parser.add_argument(
        "-i",
        "--data-path",
        default="./data/pkl/",
        help="Directory containing train.pkl, valid.pkl, and test.pkl",
    )
    parser.add_argument("--result-dir", default=None, help="Output directory for result files")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Embedding hidden size")
    parser.add_argument("--runs", type=int, default=5, help="Number of repeated runs")
    parser.add_argument("--seed-base", type=int, default=42, help="Base random seed for repeated runs")
    parser.add_argument(
        "--summary-json-name",
        default="experiment_summary.json",
        help="Summary JSON filename written in the result directory",
    )
    return parser.parse_args()


def default_result_dir(data_path):
    normalized = data_path.replace("\\", "/")
    if normalized.endswith("/"):
        normalized = normalized[:-1]
    if normalized.endswith("/pkl"):
        return normalized[:-4] + "/results"
    return normalized + "/results"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_positive_confusion_matrix(mcm):
    arr = np.array(mcm)
    if arr.ndim == 3 and arr.shape[0] >= 2:
        arr = arr[1]
    if arr.ndim != 2 or arr.shape != (2, 2):
        return {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    tn = int(arr[0, 0])
    fp = int(arr[0, 1])
    fn = int(arr[1, 0])
    tp = int(arr[1, 1])
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def calculate_binary_metrics(cm):
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    p_denom = tp + fp
    r_denom = tp + fn
    acc_denom = tp + fp + tn + fn
    fpr_denom = fp + tn

    precision = (tp / p_denom * 100.0) if p_denom else 0.0
    recall = (tp / r_denom * 100.0) if r_denom else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = ((tp + tn) / acc_denom * 100.0) if acc_denom else 0.0

    # Keep the project's current FPR/FNR naming convention.
    fpr = (fn / r_denom * 100.0) if r_denom else 0.0
    fnr = (fp / fpr_denom * 100.0) if fpr_denom else 0.0

    return {
        "ACC": round(acc, 4),
        "P": round(precision, 4),
        "R": round(recall, 4),
        "F1": round(f1, 4),
        "FPR": round(fpr, 4),
        "FNR": round(fnr, 4),
    }


def read_best_run_result(result_path):
    history = load_data(result_path)
    best_epoch = None
    best_acc = -1.0
    best_f1 = -1.0
    best_val_score = None

    for epoch, payload in history.items():
        val_score = payload.get("val_score", {})
        acc = safe_float(val_score.get("ACC", 0.0))
        f1 = safe_float(val_score.get("W_f1", 0.0))
        if (acc > best_acc) or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_epoch = int(epoch)
            best_val_score = val_score

    if best_val_score is None:
        return {
            "best_epoch": None,
            "confusion_matrix": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
            "metrics": {"ACC": 0.0, "P": 0.0, "R": 0.0, "F1": 0.0, "FPR": 0.0, "FNR": 0.0},
            "raw_val_score": {},
        }

    cm = extract_positive_confusion_matrix(best_val_score.get("MCM", []))
    metrics = calculate_binary_metrics(cm)
    raw_score_export = {
        "ACC": safe_float(best_val_score.get("ACC", 0.0)),
        "M_fpr": safe_float(best_val_score.get("M_fpr", 0.0)),
        "M_fnr": safe_float(best_val_score.get("M_fnr", 0.0)),
        "M_f1": safe_float(best_val_score.get("M_f1", 0.0)),
        "W_fpr": safe_float(best_val_score.get("W_fpr", 0.0)),
        "W_fnr": safe_float(best_val_score.get("W_fnr", 0.0)),
        "W_f1": safe_float(best_val_score.get("W_f1", 0.0)),
    }
    return {
        "best_epoch": best_epoch + 1,
        "confusion_matrix": cm,
        "metrics": metrics,
        "raw_val_score": raw_score_export,
    }


def aggregate_run_summaries(run_results):
    if not run_results:
        zero_metrics = {"ACC": 0.0, "P": 0.0, "R": 0.0, "F1": 0.0, "FPR": 0.0, "FNR": 0.0}
        return {
            "completed_runs": 0,
            "authoritative_metric_source": "test_metrics",
            "avg_metrics": zero_metrics,
            "std_metrics": zero_metrics,
        }

    metric_keys = ["ACC", "P", "R", "F1", "FPR", "FNR"]
    metric_matrix = np.array([[safe_float(run["test_metrics"][k]) for k in metric_keys] for run in run_results], dtype=float)
    avg_metrics = {
        k: round(float(metric_matrix[:, idx].mean()), 4)
        for idx, k in enumerate(metric_keys)
    }
    std_metrics = {
        k: round(float(metric_matrix[:, idx].std(ddof=0)), 4)
        for idx, k in enumerate(metric_keys)
    }
    return {
        "completed_runs": len(run_results),
        "authoritative_metric_source": "test_metrics",
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def train_project(args):
    hidden_size = args.hidden_size
    result_dir = args.result_dir or default_result_dir(args.data_path)
    os.makedirs(result_dir, exist_ok=True)
    summary_json_path = os.path.join(result_dir, args.summary_json_name)

    run_summary = {
        "project": "VulSCP",
        "data_path": args.data_path,
        "result_dir": result_dir,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "runs": args.runs,
            "seed_base": args.seed_base,
            "summary_json_name": args.summary_json_name,
        },
        "evaluation_protocol": "fixed split repeated runs",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_results": [],
        "aggregate": {},
        "status": "running",
    }
    write_json(summary_json_path, run_summary)

    print(f"[Train] data_path={args.data_path}")
    print(f"[Train] fixed split=7:2:1, repeated_runs={args.runs}, epochs={args.epochs}, batch={args.batch_size}")
    print(f"[Train] result_dir={result_dir}")
    print(f"[Train] summary_json={summary_json_path}")

    train_df, valid_df, test_df = get_run_dataframes(args.data_path)

    for run_index in range(args.runs):
        run_seed = args.seed_base + run_index
        set_global_seed(run_seed)

        trainer = VulSCPTrainer(
            result_save_path=result_dir,
            item_num=run_index,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=hidden_size,
            best_sc=0,
        )
        trainer.preparation(
            X_train=train_df["data"],
            y_train=train_df["label"],
            X_valid=valid_df["data"],
            y_valid=valid_df["label"],
            X_test=test_df["data"],
            y_test=test_df["label"],
        )
        trainer.train()

        run_result = read_best_run_result(trainer.result_save_path)
        test_result = trainer.test_best_model()
        if test_result is not None:
            run_result["test_confusion_matrix"] = test_result["confusion_matrix"]
            run_result["test_metrics"] = test_result["metrics"]
            run_result["test_raw_score"] = test_result.get("raw_score", {})
        run_result["run_index"] = run_index
        run_result["seed"] = int(run_seed)
        run_result["result_file"] = trainer.result_save_path
        run_summary["run_results"].append(run_result)
        run_summary["aggregate"] = aggregate_run_summaries(run_summary["run_results"])
        run_summary["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(summary_json_path, run_summary)
        print(f"[Train] Run {run_index} finished. Summary updated.")

    run_summary["status"] = "completed"
    run_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(summary_json_path, run_summary)
    print(f"[Train] Experiment summary written: {summary_json_path}")
    return run_summary


if __name__ == "__main__":
    cli_args = parse_options()
    train_project(cli_args)
