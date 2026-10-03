FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860 HOME=/home/user HF_HOME=/home/user/.hf \
    SANDBOX_PATH=/home/user/data/sandbox.db AGENT_MODE=hybrid DEMO_MODE=1
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN pip install uv
COPY --chown=user pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --group embeddings --no-hashes --no-emit-project -o req.txt \
 && TORCH=$(sed -nE 's/^torch==([^ ;]+).*/\1/p' req.txt | head -1) \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH}" \
 && grep -vE '^(torch==|nvidia-|triton)' req.txt > req-nogpu.txt \
 && pip install -r req-nogpu.txt
COPY --chown=user src ./src
COPY --chown=user policy ./policy
COPY --chown=user models ./models
COPY --chown=user entrypoint.sh ./entrypoint.sh
USER user
RUN python -c "from sentence_transformers import SentenceTransformer; from src.router.classifier import E5_REVISION; SentenceTransformer('intfloat/multilingual-e5-small', revision=E5_REVISION)"
EXPOSE 7860
CMD ["sh", "./entrypoint.sh"]
