# Mini-Paper Outline

## Introduction
- Problem framing: city-level conflict risk forecasting.
- Motivation: structured indicators are slow; text signals are timely but noisy.
- Contribution: reproducible NLP pipeline with explicit aggregation and evaluation.

## Related Work (categories)
- Conflict early warning systems.
- Event extraction and political violence forecasting.
- News-based risk indicators.
- Multilingual NLP for MENA.

## Data
- ACLED events for labels.
- World Bank/UNHCR for structured controls.
- News sources (GDELT/NewsAPI) for text.
- Label definition: next-month city-level conflict events.

## Methods
- Text preprocessing and deduplication.
- Feature extraction (TF–IDF, sentiment, topics, transformer embeddings).
- Aggregation from article to city–time features.
- Modeling baselines + strong model.

## Experiments
- Time-based train/val/test split.
- Location-based stress test (holdout countries or cities).
- Ablations: numeric-only vs text-only vs combined.

## Results
- Placeholder for metrics (MAE/RMSE or F1/AUROC).
- Expected trends; no claims until run.

## Limitations & Ethics
- Media bias, coverage gaps, and geocoding errors.
- Correlation vs causation.
- Responsible use constraints.

## Conclusion & Future Work
- Summarize text signal value.
- Future work ideas:
  - Event extraction for richer signals.
  - Arabic dialect coverage and source weighting.
  - Causal inference cautions and counterfactual evaluation.
  - Alternative labels (e.g., displacement spikes).
