#!/usr/bin/env bash
# Start the prediction API in the background, then the dashboard in front.
# Used as the Docker container's command.
set -euo pipefail

# The API only listens inside the container; the dashboard is its one client
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &

# Wait until the API answers, so the first visitor never sees a connection error
for _ in $(seq 1 30); do
  if python -c "import httpx; httpx.get('http://127.0.0.1:8000/health').raise_for_status()" 2>/dev/null; then
    break
  fi
  sleep 1
done

# exec hands the process over to Streamlit, so it receives stop signals directly
exec streamlit run src/dashboard/app.py \
  --server.address=0.0.0.0 \
  --server.port=7860 \
  --server.headless=true
