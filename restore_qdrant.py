# Restores the BGE-M3 Qdrant collection from qdrant_export_bgem3.json on
# any machine — no embedding model, no API keys, no re-embedding needed.
# Prereq: Qdrant running on localhost:6333 (see DEPLOYMENT_GUIDE.txt).
# Usage:  python restore_qdrant.py
import json
import os
import sys

import qdrant_client
from qdrant_client.http import models as qmodels

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_FILE = os.path.join(HERE, "qdrant_export_bgem3.json")

if not os.path.exists(EXPORT_FILE):
    sys.exit(f"Export file not found: {EXPORT_FILE}")

with open(EXPORT_FILE, encoding="utf-8") as f:
    data = json.load(f)

qc = qdrant_client.QdrantClient(host="localhost", port=6333)
try:
    qc.delete_collection(data["collection"])
except Exception:  # noqa: BLE001 — collection may not exist yet
    pass
qc.create_collection(
    collection_name=data["collection"],
    vectors_config=qmodels.VectorParams(
        size=data["vector_size"],
        distance=getattr(qmodels.Distance, data["distance"].upper()),
    ),
)
points = [
    qmodels.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
    for p in data["points"]
]
# upsert in batches of 100
for i in range(0, len(points), 100):
    qc.upsert(collection_name=data["collection"], points=points[i:i + 100])

info = qc.get_collection(data["collection"])
print(f"Restored collection '{data['collection']}': {info.points_count} points")
