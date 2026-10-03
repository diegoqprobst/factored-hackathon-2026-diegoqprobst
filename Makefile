.PHONY: setup download pipeline test router-data router

setup:
	uv sync

download:
	uv run --env-file .env python -m src.pipeline.download --exclude digital_events

pipeline:
	uv run python -m src.pipeline.run all

test:
	uv run pytest -q

router-data:
	uv run --env-file .env python -m src.router.paraphrase

router:
	uv run --group embeddings python -m src.router.evaluate
