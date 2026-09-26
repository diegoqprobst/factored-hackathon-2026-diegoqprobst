.PHONY: setup download pipeline test

setup:
	uv sync

download:
	uv run --env-file .env python -m src.pipeline.download --exclude digital_events

pipeline:
	uv run python -m src.pipeline.run all

test:
	uv run pytest -q
