from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from src.utils.io import ensure_dir, read_csv, save_csv

REQUIRED_COLUMNS = ["City", "Country", "Conflict_Events_Next"]


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_name(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def main(cfg_path: str = "configs/config.yaml") -> None:
    cfg = load_config(cfg_path)
    structured_path = Path(cfg.get("structured_path", "data/raw/MiddleEast.csv"))

    if not structured_path.exists():
        raise FileNotFoundError(f"Structured data not found: {structured_path}")

    df = read_csv(structured_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Structured data missing required columns: {missing}")

    df = df.copy()
    df["City"] = df["City"].map(normalize_name)
    df["Country"] = df["Country"].map(normalize_name)

    out_dir = ensure_dir("data/processed")
    structured_out = Path(out_dir) / "structured.csv"
    save_csv(df, structured_out)

    cities = (
        df["City"]
        .dropna()
        .astype(str)
        .map(normalize_name)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    cities_path = Path(out_dir) / "cities.txt"
    with cities_path.open("w", encoding="utf-8") as f:
        for city in cities:
            if city:
                f.write(f"{city}\n")

    print(f"Saved structured working copy -> {structured_out}")
    print(f"Saved {len(cities):,} cities -> {cities_path}")


if __name__ == "__main__":
    main()
