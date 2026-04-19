"""Document ingestion pipeline."""

from pathlib import Path

from .chunker import MultiStrategyChunker
from .config import Config
from .embeddings import EmbeddingService
from .vectorstore import VectorStore


class IngestPipeline:
    """Pipeline for ingesting documents into the vector store."""

    def __init__(
        self,
        config: Config,
        embedding_service: EmbeddingService,
        vectorstore: VectorStore,
    ):
        """Initialize the pipeline.

        Args:
            config: Configuration object
            embedding_service: Embedding service for vectorization
            vectorstore: Vector store for persistence
        """
        self.config = config
        self.embedding_service = embedding_service
        self.vectorstore = vectorstore
        self.chunker = MultiStrategyChunker(
            strategies=[
                {
                    "strategy": "paragraph",
                    "chunk_size": config.chunking.chunk_size,
                    "chunk_overlap": config.chunking.chunk_overlap,
                },
                {
                    "strategy": "small_window",
                    "chunk_size": 200,
                    "chunk_overlap": 30,
                },
                {
                    "strategy": "sliding",
                    "chunk_size": 400,
                    "chunk_overlap": 200,
                },
            ],
        )

    def run(self, input_dir: Path = None) -> int:
        """Run the ingestion pipeline.

        Args:
            input_dir: Directory to ingest from (defaults to config's input_dir)

        Returns:
            Number of documents processed
        """
        input_dir = input_dir or Path(self.config.documents_input_dir)
        supported_exts = [".txt", ".md", ".json"]

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all supported files
        files = []
        for ext in supported_exts:
            files.extend(input_dir.rglob(f"*{ext}"))

        print(f"Found {len(files)} documents to ingest")

        total_chunks = 0
        for file_path in files:
            chunks = self._process_file(file_path)
            total_chunks += len(chunks)

        print(f"Ingestion complete: {len(files)} documents, {total_chunks} chunks")
        return len(files)

    def _process_file(self, file_path: Path) -> list:
        """Process a single file.

        Args:
            file_path: Path to the file

        Returns:
            List of chunks created
        """
        print(f"Processing: {file_path.name}")

        # Read file content
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Chunk the content
        chunk_objects = self.chunker.chunk(content, source=str(file_path))

        if not chunk_objects:
            return []

        # Prepare data for vector store
        chunks = [c.content for c in chunk_objects]
        metadatas = [c.metadata for c in chunk_objects]
        ids = [f"{file_path.stem}_{c.metadata.get('chunking_strategy', 'unk')}_{c.metadata.get('global_index', i)}"
               for i, c in enumerate(chunk_objects)]

        # Add to vector store
        self.vectorstore.add_chunks(chunks, metadatas, ids)

        print(f"  -> {len(chunks)} chunks created")
        return chunk_objects
