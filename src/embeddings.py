"""Embedding service using sentence-transformers with GPU acceleration."""

from typing import Optional

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Embedding service using local sentence-transformers model with GPU."""

    def __init__(self, model_path: str, dimension: int = 1024, device: str = "cuda"):
        """Initialize with local model path.

        Args:
            model_path: Path to the local model
            dimension: Embedding dimension
            device: Device to use ("cuda" for GPU, "cpu" for CPU)
        """
        self.model_path = model_path
        self.dimension = dimension
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy load the model on GPU if available."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_path, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not text:
            # Return zero vector for empty text
            return [0.0] * self.dimension

        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
