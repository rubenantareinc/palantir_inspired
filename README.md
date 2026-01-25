# City-Level Conflict Risk from News Text Signals

*Inspiration: A "Palantir-inspired" analyst workflow focused on fusing structured indicators with media-derived signals.*

## Abstract
This project reframes a city-level conflict prediction prototype into an NLP-first pipeline for the Middle East and North Africa (MENA). The core idea is to transform news articles into time-bucketed, city-level text features (sentiment, topics, event cues) and evaluate how much predictive signal they add beyond standard socioeconomic indicators. The target label is conflict risk in the *next* time window, making the task forward-looking rather than cross-sectional. The emphasis is on reproducibility, careful aggregation, and clear limits on what the models can claim. Results are expected to show that text features provide modest but reliable lift over numeric baselines, especially for cities with sparse structured data.

## Problem
Can we forecast *near-term* conflict risk at the city level using news text signals combined with standard socioeconomic indicators?

## Why it matters
City-level early warning is a practical granularity for NGOs, journalists, and analysts. Structured indicators alone are slow to update; text streams update daily but are noisy. A transparent NLP pipeline helps quantify when the text signal is helpful and when it is not.

## Data
**Planned sources (not yet downloaded):**
- **Conflict events:** ACLED (city/date/event type).
- **Socioeconomic indicators:** World Bank, UNHCR displacement statistics (annual or monthly where possible).
- **Text sources:** GDELT, NewsAPI, or open corpora of MENA news with geotags.

**Core target label (time-aware):**
- `Conflict_Events_Next`: number of conflict events in the *next* time bucket for a city.
- Optional classification label: `High_Risk` if next-window events exceed a threshold (e.g., top quartile).

## Method
A full method write-up lives in `docs/methods.md`. In short, the pipeline:
1. Ingests news articles with timestamps and locations.
2. Normalizes and filters text.
3. Aggregates features into city–time buckets.
4. Trains baseline and transformer-based models to predict next-window conflict risk.

### Text Signal Engineering
- **Preprocessing:** language ID, deduplication, URL/entity cleanup, headline-body concatenation.
- **Shallow signals:** TF–IDF n-grams, sentiment polarity, and topic proportions.
- **Event cues:** keyword patterns for protests, clashes, or security incidents; optional event extraction models.

### From Articles to City-Level Risk Features
- **Geocoding:** use article metadata if present; otherwise extract place names and resolve to city (geopy + gazetteer).
- **Time bucketing:** aggregate daily/weekly/monthly windows (default: monthly).
- **Aggregation:** counts, normalized rates per article, and trend deltas.
- **Missingness:** explicit indicators when a city has few/no articles in a window.

### Multilingual Extension (English + Arabic)
Arabic is strategically important for MENA monitoring because many local events are first reported in Arabic outlets, especially for smaller cities. The plan is to add Arabic news streams, use AraBERT for text representations, and normalize dialect-heavy content with light preprocessing rather than aggressive filtering.

## Experiments
1. **Baseline 1:** TF–IDF + regression/classification.
2. **Baseline 2:** sentiment/topic aggregation + regression/classification.
3. **Strong model:** transformer embeddings (BERT/AraBERT) aggregated to city–time features.

## Results
*Placeholder until experiments are run.* Expected: text + numeric features outperform numeric-only baselines; improvements are modest but consistent.

## Error Analysis
- Review top false positives/negatives by city.
- Inspect whether misclassifications correlate with news volume spikes or geocoding failures.

## Ethics
- Avoid overstating causal claims.
- Be clear about reporting bias and media coverage gaps.
- No use for tactical or targeting decisions.

## How to run
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python -m src.ingest.fetch_news --source gdelt --start 2021-01-01 --end 2021-12-31`
4. `python -m src.features.build_text_features --bucket monthly`
5. `python -m src.train.run --model tfidf`

## Roadmap
- Add Arabic text pipeline and AraBERT features.
- Evaluate time + location split strategies.
- Add event extraction for richer signals.
- Improve geocoding and gazetteer coverage.

---

## Repo Blueprint
```
repo/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/               # raw downloads (not committed)
│  ├─ interim/           # cleaned/normalized artifacts
│  └─ processed/         # model-ready tables
├─ docs/
│  ├─ methods.md         # detailed method section
│  └─ paper_outline.md   # mini-paper outline
├─ notebooks/
│  └─ 01_eda.ipynb       # exploratory analysis
├─ src/
│  ├─ ingest/
│  │  ├─ fetch_news.py
│  │  └─ load_structured.py
│  ├─ features/
│  │  ├─ text_features.py
│  │  └─ aggregation.py
│  ├─ models/
│  │  ├─ baseline_tfidf.py
│  │  ├─ baseline_topics.py
│  │  └─ transformer_pooling.py
│  ├─ train/
│  │  └─ run.py
│  └─ utils/
│     ├─ geo.py
│     └─ time.py
└─ demo/
   └─ cli.py              # minimal CLI demo
```

## Requirements (plan)
`requirements.txt` should include:
- pandas, numpy, scikit-learn
- nltk, spacy, sentence-transformers, transformers, torch
- langdetect, geopy, rapidfuzz
- matplotlib, seaborn
- tqdm
- streamlit (optional demo)

## Demo plan
- A small CLI that takes a city name + month and prints predicted risk and top contributing text features.
