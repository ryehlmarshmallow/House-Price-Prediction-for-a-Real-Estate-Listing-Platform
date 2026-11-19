.PHONY: setup data clean features train evaluate monitoring dictionary pipeline api app test lint

setup:
	uv sync

data:
	uv run python scripts/01_generate_context.py

clean:
	uv run python scripts/02_clean_data.py

features:
	uv run python scripts/03_build_features.py

train:
	uv run python scripts/04_train_models.py

evaluate:
	uv run python scripts/05_run_evaluation.py

monitoring:
	uv run python scripts/06_run_monitoring.py

dictionary:
	uv run python scripts/07_export_data_dictionary.py

pipeline: data clean features train evaluate monitoring dictionary

api:
	uv run uvicorn revaluation.api.main:app --host 0.0.0.0 --port 8000

app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest -q

lint:
	uv run ruff check src app scripts tests
