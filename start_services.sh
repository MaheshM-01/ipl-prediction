#!/bin/sh
set -eu

uvicorn src.api.main_api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

streamlit run src/frontend/ui.py --server.port 8501 --server.address 0.0.0.0
