.PHONY: setup structured fetch features merge train_regression train_classification ablate_regression

setup:
	pip install -r requirements.txt

structured:
	python -m src.ingest.load_structured

fetch:
	python -m src.ingest.fetch_gdelt

features:
	python -m src.features.build_city_month_docs

merge:
	python -m src.features.merge_structured_text

train_regression:
	python -m src.models.train_baseline --task regression --features combined

train_classification:
	python -m src.models.train_baseline --task classification --features combined

ablate_regression:
	python -m src.models.train_baseline --task regression --features structured_only
	python -m src.models.train_baseline --task regression --features text_only
	python -m src.models.train_baseline --task regression --features combined

