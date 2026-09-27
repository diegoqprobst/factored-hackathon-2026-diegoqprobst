.PHONY: setup download pipeline test router-data router serve serve-demo serve-baseline eval-cases eval serve-demo-baseline

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

serve:
	uv run --group embeddings --env-file .env uvicorn --factory src.agent.api:create_app --port 8000

serve-demo:
	DEMO_MODE=1 uv run --group embeddings --env-file .env uvicorn --factory src.agent.api:create_app --port 8000

serve-baseline:
	AGENT_MODE=baseline uv run --env-file .env uvicorn --factory src.agent.api:create_app --port 8000

eval-cases:
	uv run python -m src.eval.run build

eval:
	uv run --group embeddings --env-file .env python -m src.eval.run run --systems baseline,hybrid --repeats 3

serve-demo-baseline:
	DEMO_MODE=1 AGENT_MODE=baseline uv run --env-file .env uvicorn --factory src.agent.api:create_app --port 8000
