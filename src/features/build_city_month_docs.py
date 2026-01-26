from __future__ import annotations
import re
import pandas as pd
from pathlib import Path
import yaml
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.io import ensure_dir, save_csv, read_csv

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def cue_count(text: str, cues: list[str]) -> int:
    if not text:
        return 0
    count = 0
    for c in cues:
        count += len(re.findall(rf"\b{re.escape(c.lower())}\b", text))
    return int(count)

def main(cfg_path: str = "configs/config.yaml") -> None:
    cfg = load_config(cfg_path)

    raw_path = Path("data/raw/gdelt_articles.csv")
    if not raw_path.exists():
        raise FileNotFoundError("Run ingest first: python -m src.ingest.fetch_gdelt")

    df = read_csv(raw_path)
    if df.empty:
        raise ValueError("No articles fetched. Expand date range or loosen query_spine in config.")

    text_fields = cfg["text_fields"]
    freq = cfg["bucketing"]["freq"]
    cues = cfg["conflict_cues"]
    min_doc_len = int(cfg["features"]["min_doc_length"])

    df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
    df = df.dropna(subset=["seendate"])
    df["bucket"] = df["seendate"].dt.to_period(freq).dt.to_timestamp()

    for col in text_fields:
        df[col] = df[col].fillna("").astype(str).map(normalize_text)
    df["text"] = df[text_fields].agg(" ".join, axis=1).map(normalize_text)

    analyzer = SentimentIntensityAnalyzer()
    df["sent_compound"] = df["text"].map(lambda t: analyzer.polarity_scores(t)["compound"])
    df["cue_count"] = df["text"].map(lambda t: cue_count(t, cues))

    grouped = []
    for (city, bucket), g in df.groupby(["city", "bucket"]):
        joined = " ".join(g["text"].tolist()).strip()
        if len(joined) < min_doc_len:
            continue
        grouped.append({
            "city": city,
            "bucket": bucket,
            "doc": joined,
            "n_articles": int(len(g)),
            "sent_mean": float(g["sent_compound"].mean()),
            "cue_sum": int(g["cue_count"].sum()),
        })

    docs = pd.DataFrame(grouped).sort_values(["city", "bucket"]).reset_index(drop=True)
    if docs.empty:
        raise ValueError("All docs filtered out. Lower min_doc_length or fetch more articles.")

    out_dir = ensure_dir("data/processed")
    save_csv(docs, Path(out_dir) / "city_bucket_docs.csv")
    print(f"Saved city-bucket docs -> {Path(out_dir) / 'city_bucket_docs.csv'}")

if __name__ == "__main__":
    main()
