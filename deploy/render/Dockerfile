# Render free tier (512 MB): torch-free image. Hybrid agent with the TF-IDF router + OpenRouter LLM extraction.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SANDBOX_PATH=/app/data/sandbox.db \
    AGENT_MODE=hybrid DEMO_MODE=1 ROUTER_PATH=models/router_tfidf_v1
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o req.txt && pip install -r req.txt
COPY src ./src
COPY policy ./policy
COPY models/router_tfidf_v1 ./models/router_tfidf_v1
COPY deploy/render/entrypoint.sh ./entrypoint.sh
CMD ["sh", "./entrypoint.sh"]
