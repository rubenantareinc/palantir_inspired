from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.io import ensure_dir, read_csv, save_csv, save_json

TEXT_COLUMNS = ["doc", "sent_mean", "cue_sum", "n_articles"]
NON_FEATURE_COLUMNS = ["City", "Country", "bucket"]


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline model on structured + text features.")
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Task type to train.",
    )
    parser.add_argument(
        "--features",
        choices=["structured_only", "text_only", "combined"],
        default="combined",
        help="Feature set to train metrics/importance for (ablation covers all).",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file.")
    return parser.parse_args()


def has_usable_time_column(df: pd.DataFrame) -> bool:
    if "bucket" not in df.columns:
        return False
    return df["bucket"].notna().all()


def time_split(df: pd.DataFrame, test_size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("bucket")
    unique_buckets = sorted(pd.to_datetime(df["bucket"]).unique())
    cut = int(len(unique_buckets) * (1 - test_size))
    train_buckets = set(unique_buckets[:cut])
    test_buckets = set(unique_buckets[cut:])
    train_df = df[pd.to_datetime(df["bucket"]).isin(train_buckets)].copy()
    test_df = df[pd.to_datetime(df["bucket"]).isin(test_buckets)].copy()
    return train_df, test_df


def pick_structured_columns(df: pd.DataFrame, label: str) -> List[str]:
    drop_cols = set(TEXT_COLUMNS + NON_FEATURE_COLUMNS + [label])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in drop_cols]


def build_preprocessor(
    df: pd.DataFrame,
    label: str,
    feature_set: str,
    tfidf_max_features: int,
    ngram_range: Tuple[int, int],
) -> Tuple[ColumnTransformer, List[str], bool]:
    numeric_features: List[str] = []
    include_text = feature_set in {"text_only", "combined"}

    if feature_set == "structured_only":
        numeric_features = pick_structured_columns(df, label)
    elif feature_set == "text_only":
        numeric_features = [col for col in TEXT_COLUMNS if col != "doc" and col in df.columns]
    elif feature_set == "combined":
        numeric_features = pick_structured_columns(df, label) + [
            col for col in TEXT_COLUMNS if col != "doc" and col in df.columns
        ]
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    transformers = []
    if include_text:
        transformers.append(
            (
                "tfidf",
                TfidfVectorizer(max_features=tfidf_max_features, ngram_range=ngram_range),
                "doc",
            )
        )
    if numeric_features:
        transformers.append(
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_features)
        )

    if not transformers:
        raise ValueError(f"No features available for feature set: {feature_set}")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    return preprocessor, numeric_features, include_text


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols: List[str], include_text: bool) -> List[str]:
    names: List[str] = []
    if include_text:
        tfidf = preprocessor.named_transformers_["tfidf"]
        tfidf_names = list(tfidf.get_feature_names_out())
        names.extend([f"tfidf:{n}" for n in tfidf_names])
    if numeric_cols:
        names.extend([f"num:{n}" for n in numeric_cols])
    return names


def build_target(df: pd.DataFrame, task: str, label: str, train_idx: pd.Index) -> pd.Series:
    if task == "regression":
        return df[label].astype(float)

    train_vals = df.loc[train_idx, label].astype(float)
    threshold = float(np.percentile(train_vals, 75))
    return (df[label].astype(float) >= threshold).astype(int)


def evaluate_regression(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, preds))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2 = float(r2_score(y_true, preds))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_classification(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")
    f1 = float(f1_score(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan")
    report = classification_report(y_true, pred, output_dict=True)
    return {"auroc": auc, "f1": f1, "classification_report": report}


def train_once(
    df: pd.DataFrame,
    task: str,
    feature_set: str,
    cfg: dict,
) -> Tuple[Dict[str, object], pd.DataFrame | None]:
    label = "Conflict_Events_Next"
    if label not in df.columns:
        raise ValueError("Conflict_Events_Next missing from model_table.csv")

    df = df.copy()
    if "doc" in df.columns:
        df["doc"] = df["doc"].fillna("").astype(str)

    tfidf_max_features = int(cfg["features"]["tfidf_max_features"])
    ngram_range = tuple(cfg["features"]["tfidf_ngram_range"])
    seed = int(cfg.get("seed", 42))

    if has_usable_time_column(df):
        train_df, test_df = time_split(df, test_size=0.25)
    else:
        stratify = None
        if task == "classification":
            stratify = (df[label] >= df[label].quantile(0.75)).astype(int)
        train_df, test_df = train_test_split(
            df,
            test_size=0.25,
            random_state=seed,
            stratify=stratify,
        )

    y_full = build_target(df, task, label, train_df.index)
    y_train = y_full.loc[train_df.index]
    y_test = y_full.loc[test_df.index]

    preprocessor, numeric_cols, include_text = build_preprocessor(
        df,
        label,
        feature_set,
        tfidf_max_features,
        ngram_range,
    )
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0)
        train_df[numeric_cols] = train_df[numeric_cols].fillna(0)
        test_df[numeric_cols] = test_df[numeric_cols].fillna(0)

    if task == "regression":
        model = Ridge(alpha=1.0, random_state=seed)
    else:
        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )

    pipe = Pipeline([("pre", preprocessor), ("model", model)])

    X_train = train_df
    X_test = test_df

    pipe.fit(X_train, y_train)

    if task == "regression":
        preds = pipe.predict(X_test)
        metrics = evaluate_regression(y_test.to_numpy(), preds)
    else:
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = evaluate_classification(y_test.to_numpy(), proba, pred)

    metrics.update({
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_set": feature_set,
        "task": task,
    })

    fi_df = None
    if hasattr(model, "coef_"):
        feature_names = get_feature_names(pipe.named_steps["pre"], numeric_cols, include_text)
        coefs = pipe.named_steps["model"].coef_.ravel()
        fi_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        fi_df["abs_coef"] = fi_df["coef"].abs()
        fi_df = fi_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    return metrics, fi_df


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = Path("data/processed/model_table.csv")
    if not data_path.exists():
        raise FileNotFoundError("Run merge step first: python -m src.features.merge_structured_text")

    df = read_csv(data_path)
    outputs = ensure_dir("outputs")

    ablation_rows = []
    for feature_set in ["structured_only", "text_only", "combined"]:
        metrics, fi_df = train_once(df, args.task, feature_set, cfg)
        row = {"feature_set": feature_set}
        if args.task == "regression":
            row.update({"mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"]})
        else:
            row.update({"auroc": metrics["auroc"], "f1": metrics["f1"]})
        ablation_rows.append(row)

        if feature_set == args.features:
            metrics_path = Path(outputs) / f"metrics_{args.task}_{feature_set}.json"
            save_json(metrics, metrics_path)
            if fi_df is not None:
                fi_path = Path(outputs) / f"feature_importance_{args.task}_{feature_set}.csv"
                save_csv(fi_df, fi_path)

    ablation_df = pd.DataFrame(ablation_rows)
    save_csv(ablation_df, Path(outputs) / "ablation_results.csv")

    print(f"Saved ablation results -> {Path(outputs) / 'ablation_results.csv'}")

    if args.task == "regression":
        print(f"Saved metrics -> {Path(outputs) / f'metrics_{args.task}_{args.features}.json'}")
    else:
        print(f"Saved metrics -> {Path(outputs) / f'metrics_{args.task}_{args.features}.json'}")


if __name__ == "__main__":
    main()
