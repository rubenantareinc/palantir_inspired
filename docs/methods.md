# Method

This project builds a city-level conflict risk model by converting news articles into time-bucketed text features and combining them with structured indicators. The goal is not to claim causality but to test whether text adds incremental predictive signal over a simple numeric baseline.

## Data Assembly
1. **Conflict labels:** count conflict events per city by month (ACLED). The label is `Conflict_Events_Next`, defined as the next-month count for the same city.
2. **Structured controls:** GDP growth, inflation, unemployment, refugee volume, election/coup flags (where available).
3. **Text corpus:** articles from GDELT/NewsAPI with timestamps, source metadata, and (when possible) location tags.

## Text Signal Engineering
- **Filtering and normalization:**
  - Language ID and basic cleanup (URLs, HTML artifacts, boilerplate).
  - Deduplication using fuzzy matching on headlines + lead paragraphs.
  - Merge headline and body to preserve event cues.
- **Shallow lexical features:**
  - TF–IDF over unigrams/bigrams.
  - Polarity sentiment scores and subjectivity.
- **Topical structure:**
  - Topic modeling (e.g., NMF or LDA) to capture themes like protests, governance, or economic stress.
- **Event cues:**
  - Simple pattern lexicons for conflict verbs ("clash", "attack", "protest") and actor mentions.
  - Optional event extraction models as a stretch goal.

## From Articles to City-Level Risk Features
- **Location resolution:**
  - Prefer direct metadata; otherwise extract place names and resolve to cities using a gazetteer.
  - Store match confidence for robustness testing.
- **Aggregation windows:**
  - Monthly by default; weekly as a sensitivity analysis.
- **Feature aggregation:**
  - Counts (articles per city/month), normalized rates, and rolling deltas.
  - Topic proportions and sentiment averages.
  - Missingness indicators for low-coverage city-months.

## Modeling
- **Baseline 1:** TF–IDF + linear regression or logistic regression.
- **Baseline 2:** Aggregated sentiment/topic features + regression/classification.
- **Strong model:** Transformer embeddings (BERT for English, AraBERT for Arabic) pooled to city–time features.

## Multilingual Extension (English + Arabic)
Arabic is strategically important for MENA monitoring because local outlets often report earlier and with finer geographic detail than English coverage. The plan is to add Arabic sources, detect language, and encode Arabic text with AraBERT. Dialect-heavy content will be lightly normalized rather than aggressively filtered to avoid dropping relevant local reporting.

## Limitations
- Text volume varies widely by city, leading to sampling bias.
- Location extraction can be noisy for smaller towns.
- The model remains correlational; it estimates risk, not cause.
