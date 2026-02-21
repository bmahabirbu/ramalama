"""Qdrant vector database storage for the RAG pipeline using llama.cpp embeddings."""

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ramalama.utils.logger import logger

COLLECTION_NAME = "rag"
EMBEDDING_BATCH_SIZE = 32


class LlamaCppEmbedder:
    """Generate embeddings via a llama.cpp server's ``/v1/embeddings`` endpoint."""

    def __init__(self, api_url: str):
        self.embeddings_url = f"{api_url.rstrip('/')}/v1/embeddings"
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("Embedding dimension unknown; call embed() first")
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches and return their vectors."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"input": texts}).encode("utf-8")
        req = Request(
            self.embeddings_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server embedding request returned {e.code}: {body}") from None

        data = sorted(result["data"], key=lambda d: d["index"])
        vectors = [d["embedding"] for d in data]

        if vectors and self._dim is None:
            self._dim = len(vectors[0])
            logger.debug(f"Detected embedding dimension: {self._dim}")

        return vectors


def store_in_qdrant(chunks: list[str], ids: list[int], output_dir: str | Path, embedder: LlamaCppEmbedder) -> None:
    """Embed *chunks* via llama.cpp and persist them in a Qdrant on-disk collection."""
    try:
        import qdrant_client
        from qdrant_client import models
    except ImportError:
        raise ImportError("qdrant-client is required for RAG. Install with: pip install qdrant-client") from None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Embedding {len(chunks)} chunks via llama.cpp")
    vectors = embedder.embed(chunks)
    dim = embedder.dimension

    logger.debug(f"Storing {len(chunks)} chunks (dim={dim}) in Qdrant at {output_dir}")

    qclient = qdrant_client.QdrantClient(path=str(output_dir))

    qclient.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE, on_disk=True),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
    )

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        points = [
            models.PointStruct(
                id=ids[i],
                payload={"document": chunks[i]},
                vector=vectors[i],
            )
            for i in range(start, end)
        ]
        qclient.upsert(collection_name=COLLECTION_NAME, points=points)

    logger.debug(f"Upserted {len(chunks)} points into Qdrant collection '{COLLECTION_NAME}'")
