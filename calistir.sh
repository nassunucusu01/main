#!/usr/bin/env bash
# KuantLab'ı yerelde başlatır: http://localhost:8080
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m app.setup_llama
exec .venv/bin/python -m uvicorn app.server:app --host 0.0.0.0 --port "${PORT:-8080}"
