#!/bin/bash
# ============================================================
#  Starts the Restaurant Voice Agent on Linux (any server).
#  Prereqs (one-time): see DEPLOYMENT_GUIDE.txt section 3.
#    1. Python 3.10-3.12 + venv created + packages installed
#    2. Qdrant running on localhost:6333
#    3. Collection restored (python restore_qdrant.py)
# ============================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/voice_inference_service"

# Load API keys + provider config
set -a
. ./keys.env
set +a

# Point BM25 at the restaurant_data folder that ships in this package
export VIS_BM25_DATA_DIR="$HERE/restaurant_data"

# Use the venv if it exists next to this script, else system python
if [ -x "$HERE/venv/bin/python" ]; then
    exec "$HERE/venv/bin/python" app.py
else
    exec python3 app.py
fi
