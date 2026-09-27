from pathlib import Path

D = Path("deploy/hf_space")


def test_space_files_are_consistent():
    readme = (D / "README.md").read_text()
    assert "sdk: docker" in readme and "app_port: 7860" in readme
    docker = (D / "Dockerfile").read_text()
    assert "download.pytorch.org/whl/cpu" in docker and "useradd -m -u 1000" in docker and "EXPOSE 7860" in docker
    assert "COPY data" not in docker and "COPY --chown=user data" not in docker  # data never baked into the image
    entry = (D / "entrypoint.sh").read_text()
    assert entry.startswith("#!/bin/sh") and "set -e" in entry and "HF_DATA_TOKEN" in entry and "52428800" in entry


def test_render_files_are_consistent():
    docker = Path("deploy/render/Dockerfile").read_text()
    assert "download.pytorch.org" not in docker and "torch==" not in docker and "--group embeddings" not in docker and "router_tfidf_v1" in docker
    assert "COPY data" not in docker
    blueprint = Path("render.yaml").read_text()
    for key in ("BANK_SESSION_SECRET", "OPENROUTER_API_KEY", "HF_DATA_TOKEN"):
        assert f"key: {key}\n        sync: false" in blueprint  # secrets are never in the file
    assert "healthCheckPath: /health" in blueprint and "plan: free" in blueprint
    assert "${PORT:-10000}" in Path("deploy/render/entrypoint.sh").read_text()
    assert ".env" in Path(".dockerignore").read_text().split()
