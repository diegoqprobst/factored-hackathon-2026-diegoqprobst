#!/bin/sh
# Download the sandbox database from the PRIVATE dataset repo, refuse to start without it, then serve.
set -e
mkdir -p "$(dirname "$SANDBOX_PATH")"
if [ ! -s "$SANDBOX_PATH" ]; then
  python -c "
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ['HF_DATASET'], 'sandbox.db', repo_type='dataset', token=os.environ['HF_DATA_TOKEN'])
shutil.copy(p, os.environ['SANDBOX_PATH'])"
fi
SIZE=$(wc -c < "$SANDBOX_PATH")
if [ "$SIZE" -lt 52428800 ]; then echo "sandbox.db too small ($SIZE bytes): refusing to start" >&2; exit 1; fi
exec uvicorn --factory src.agent.api:create_app --host 0.0.0.0 --port "${PORT:-10000}"
