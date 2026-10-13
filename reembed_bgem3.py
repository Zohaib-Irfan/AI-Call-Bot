"""One-shot: build a BGE-M3-embedded Qdrant collection
(restaurant_knowledge_bgem3, 1024 dims) from the same paragraph chunks
the OpenAI collection uses. The old restaurant_knowledge collection is
left untouched — rollback is a one-line env flip. BGE-M3 is a strong
multilingual embedder (Urdu included), unlike text-embedding-3-small
whose Urdu representation is comparatively weak.

Run with the service venv python from /workspace/voice_inference_service.
"""
import glob
import os
import re
import time
import uuid

import qdrant_client
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

DATA_DIR = os.environ.get("VIS_BM25_DATA_DIR", "/workspace/restaurant_data")
COLLECTION = "restaurant_knowledge_bgem3"
VECTOR_SIZE = 1024  # BGE-M3 dense output


def chunk_paragraphs(data_dir):
    chunks = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        for p in re.split(r"\n\s*\n", content):
            p = p.strip()
            if p:
                chunks.append({"text": p, "source": filename})
    return chunks


def main():
    chunks = chunk_paragraphs(DATA_DIR)
    print(f"Chunked {len(chunks)} paragraphs from {DATA_DIR}")

    print("Loading BAAI/bge-m3 (downloads ~2.3GB on first run)...")
    t0 = time.perf_counter()
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    vectors = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False,
    )
    print(f"Embedded {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s")
    assert vectors.shape[1] == VECTOR_SIZE, f"dim {vectors.shape[1]} != {VECTOR_SIZE}"

    qc = qdrant_client.QdrantClient(host="localhost", port=6333)
    try:
        qc.delete_collection(COLLECTION)
    except Exception as exc:  # noqa: BLE001
        print(f"(delete skipped: {exc!r})")
    qc.create_collection(
        collection_name=COLLECTION,
        vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
    )

    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["source"] + ":" + str(i))),
            vector=vectors[i].tolist(),
            payload={"text": c["text"], "source": c["source"]},
        )
        for i, c in enumerate(chunks)
    ]
    qc.upsert(collection_name=COLLECTION, points=points)
    info = qc.get_collection(COLLECTION)
    print(f"Collection '{COLLECTION}' point count: {info.points_count}")

    # quick sanity: Urdu query should surface the right chunk
    for q, expect in [
        ("بیف بریانی کی قیمت", "menu_biryani"),
        ("ڈیلیوری کتنی دیر میں", "dilevery"),
        ("کھلنے کا وقت", "restaurant_info"),
    ]:
        qv = model.encode(q, normalize_embeddings=True).tolist()
        hits = qc.query_points(collection_name=COLLECTION, query=qv, limit=1).points
        top = hits[0]
        print(f"  '{q}' -> {top.payload['source']} (score={top.score:.3f}) "
              f"{'OK' if expect in top.payload['source'] else 'CHECK'}")


main()
