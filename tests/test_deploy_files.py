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
