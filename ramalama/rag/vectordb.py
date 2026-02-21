"""Qdrant vector database storage for the RAG pipeline."""

import os
from pathlib import Path

from ramalama.utils.logger import logger

EMBED_MODEL = os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v2-small-en")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "prithivida/Splade_PP_en_v1")
COLLECTION_NAME = "rag"


def store_in_qdrant(chunks: list[str], ids: list[int], output_dir: str | Path) -> None:
    """Embed *chunks* and persist them in a Qdrant on-disk collection."""
    try:
        import qdrant_client
        from qdrant_client import models
    except ImportError:
        raise ImportError(
            "qdrant-client[fastembed] is required for RAG. "
            "Install with: pip install 'qdrant-client[fastembed]'"
        ) from None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Storing {len(chunks)} chunks in Qdrant at {output_dir}")

    qclient = qdrant_client.QdrantClient(path=str(output_dir))
    qclient.set_model(EMBED_MODEL)
    qclient.set_sparse_model(SPARSE_MODEL)

    qclient.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qclient.get_fastembed_vector_params(on_disk=True),
        sparse_vectors_config=qclient.get_fastembed_sparse_vector_params(on_disk=True),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
    )

    dense_vector_name = qclient.get_vector_field_name()
    sparse_vector_name = qclient.get_sparse_vector_field_name()

    dense_embeddings = list(qclient._embed_documents(chunks, EMBED_MODEL, embed_type="passage"))
    sparse_embeddings = list(qclient._sparse_embed_documents(chunks, SPARSE_MODEL))

    points = []
    for idx, (doc, dense_vec), sparse_vec in zip(ids, dense_embeddings, sparse_embeddings):
        vectors: dict[str, models.Vector] = {dense_vector_name: dense_vec}
        if sparse_vector_name is not None:
            vectors[sparse_vector_name] = sparse_vec
        points.append(models.PointStruct(id=idx, payload={"document": doc}, vector=vectors))

    qclient.upsert(collection_name=COLLECTION_NAME, points=points)
