from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from src.utils.io import read_csv, save_csv, ensure_dir


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_bucket(docs: pd.DataFrame, cfg: dict) -> pd.Timestamp:
    strategy = cfg.get("merge_bucket_strategy", "latest")
    if strategy == "config":
        value = cfg.get("merge_bucket_value")
        if value in (None, "", "null"):
            raise ValueError("merge_bucket_value must be set when merge_bucket_strategy is 'config'")
        return pd.to_datetime(value)
    if strategy != "latest":
        raise ValueError(f"Unsupported merge_bucket_strategy: {strategy}")
    return pd.to_datetime(docs["bucket"]).max()


def main(cfg_path: str = "configs/config.yaml") -> None:
    cfg = load_config(cfg_path)

    structured_path = Path("data/processed/structured.csv")
    text_path = Path("data/processed/city_bucket_docs.csv")

    if not structured_path.exists():
        raise FileNotFoundError("Run structured ingest first: python -m src.ingest.load_structured")
    if not text_path.exists():
        raise FileNotFoundError("Run text feature build first: python -m src.features.build_city_month_docs")

    structured = read_csv(structured_path)
    docs = read_csv(text_path)
    docs["bucket"] = pd.to_datetime(docs["bucket"])

    bucket = select_bucket(docs, cfg)
    docs_bucket = docs[docs["bucket"] == bucket].copy()

    merged = structured.merge(docs_bucket, how="left", left_on="City", right_on="city")
    merged = merged.drop(columns=["city"], errors="ignore")

    merged["has_text"] = merged["doc"].notna().astype(int)
    merged["n_articles_missing"] = merged["n_articles"].isna().astype(int)

    out_dir = ensure_dir("data/processed")
    out_path = Path(out_dir) / "model_table.csv"
    save_csv(merged, out_path)

    print(f"Selected bucket: {bucket}")
    print(f"Saved model table -> {out_path}")


if __name__ == "__main__":
    main()
