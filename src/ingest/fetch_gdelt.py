from __future__ import annotations
import time
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml

from src.utils.io import ensure_dir, save_csv

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_city_articles(city: str, query_spine: str, start: str, end: str, maxrecords: int, sort: str) -> list[dict]:
    # Query: mention the city explicitly + a "conflict-ish" spine
    q = f'"{city}" AND {query_spine}'

    params = {
        "query": q,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start.replace("-", "") + "000000",
        "enddatetime": end.replace("-", "") + "235959",
        "maxrecords": maxrecords,
        "sort": sort,
    }

    r = requests.get(GDELT_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    articles = data.get("articles", [])
    out = []
    for a in articles:
        out.append({
            "city": city,
            "seendate": a.get("seendate"),
            "url": a.get("url"),
            "title": a.get("title"),
            "description": a.get("description"),
            "language": a.get("language"),
            "domain": a.get("domain"),
            "sourceCountry": a.get("sourceCountry"),
        })
    return out

def main(cfg_path: str = "configs/config.yaml") -> None:
    cfg = load_config(cfg_path)

    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]
    query_spine = cfg["gdelt"]["query_spine"]
    maxrecords = int(cfg["gdelt"]["maxrecords_per_city"])
    sort = cfg["gdelt"]["sort"]
    cities_path = Path("data/processed/cities.txt")
    if cities_path.exists():
        with cities_path.open("r", encoding="utf-8") as f:
            cities = [line.strip() for line in f if line.strip()]
        if not cities:
            raise ValueError("cities.txt is empty. Re-run load_structured to populate it.")
        print(f"Using {len(cities):,} cities from {cities_path}")
    else:
        cities = cfg["cities"]
        print("Using cities from config.yaml")

    raw_dir = ensure_dir("data/raw")
    all_rows = []

    for city in tqdm(cities, desc="Fetching GDELT"):
        try:
            rows = fetch_city_articles(city, query_spine, start, end, maxrecords, sort)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] Failed city={city}: {e}")
        time.sleep(0.8)  # be polite to GDELT

    df = pd.DataFrame(all_rows)
    out_path = Path(raw_dir) / "gdelt_articles.csv"
    save_csv(df, out_path)
    print(f"Saved {len(df):,} articles -> {out_path}")

if __name__ == "__main__":
    main()
