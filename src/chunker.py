"""Recursive text chunking with JSON awareness."""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: dict = field(default_factory=dict)


class RecursiveChunker:
    """Recursive chunker that splits text by paragraphs and sentences.

    Features:
    - JSON-aware splitting with path tracking
    - Paragraph splitting by double newlines
    - Sentence-level splitting for oversized chunks
    - Overlap between chunks for context preservation
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        json_sensitive: bool = True,
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            json_sensitive: Whether to detect and parse JSON structures
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.json_sensitive = json_sensitive

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split text into chunks recursively.

        Args:
            text: Input text to chunk
            source: Source identifier for metadata

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        # Try JSON parsing if enabled
        if self.json_sensitive:
            json_result = self._try_parse_json(text)
            if json_result is not None:
                return json_result

        # Split by paragraphs
        paragraphs = self._split_paragraphs(text)
        chunks = []

        for idx, para in enumerate(paragraphs):
            if not para.strip():
                continue

            para_chunks = self._chunk_paragraph(para, source, idx)
            chunks.extend(para_chunks)

        # Re-index chunks after processing
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx

        return chunks

    def _chunk_paragraph(self, para: str, source: str, para_idx: int) -> list[Chunk]:
        """Chunk a single paragraph, applying overlap within the paragraph only."""
        if len(para) <= self.chunk_size:
            return [Chunk(
                content=para,
                metadata={
                    "source": source,
                    "chunk_index": 0,
                    "paragraph_index": para_idx,
                }
            )]

        # Split long paragraph by sentences
        sentence_chunks = self._split_by_sentences(para)
        chunks = []
        for s_idx, sent_chunk in enumerate(sentence_chunks):
            chunks.append(Chunk(
                content=sent_chunk,
                metadata={
                    "source": source,
                    "chunk_index": 0,  # Will be re-indexed later
                    "paragraph_index": para_idx,
                    "sentence_index": s_idx,
                }
            ))

        # Apply overlap only within same paragraph
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _try_parse_json(self, text: str) -> Optional[list[Chunk]]:
        """Attempt to parse text as JSON and chunk accordingly.

        Returns:
            List of Chunks if JSON parsing succeeded, None otherwise
        """
        text = text.strip()

        # Quick heuristic: must start with { or [
        if not (text.startswith("{") or text.startswith("[")):
            return None

        try:
            data = json.loads(text)
            return self._chunk_json(data, text)
        except json.JSONDecodeError:
            return None

    def _chunk_json(self, data, raw_text: str, json_path: str = "") -> list[Chunk]:
        """Chunk JSON data by keys or values."""
        chunks = []

        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{json_path}.{key}" if json_path else key
                if isinstance(value, (dict, list)):
                    chunks.extend(self._chunk_json(value, raw_text, path))
                else:
                    chunk_content = f'"{key}": {json.dumps(value)}'
                    chunks.append(Chunk(
                        content=chunk_content,
                        metadata={
                            "json_path": path,
                            "json_key": key,
                        }
                    ))
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                path = f"{json_path}[{idx}]"
                if isinstance(item, (dict, list)):
                    chunks.extend(self._chunk_json(item, raw_text, path))
                else:
                    chunk_content = json.dumps(item)
                    chunks.append(Chunk(
                        content=chunk_content,
                        metadata={
                            "json_path": path,
                            "json_index": idx,
                        }
                    ))
        else:
            chunks.append(Chunk(
                content=json.dumps(data),
                metadata={"json_path": json_path or "root"}
            ))

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs by double newlines."""
        paragraphs = re.split(r"\n\n+", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text into chunks by sentences while respecting chunk_size."""
        # Sentence-ending punctuation pattern
        sentence_pattern = r"(?<=[.!?])\s+"

        sentences = re.split(sentence_pattern, text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If single sentence exceeds chunk_size, split it further
            if len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split long sentence by words
                words = sentence.split()
                current_subchunk = ""
                for word in words:
                    test_chunk = (current_subchunk + " " + word).strip()
                    if len(test_chunk) > self.chunk_size:
                        if current_subchunk:
                            chunks.append(current_subchunk.strip())
                        # Keep building even if it exceeds (for long words)
                        current_subchunk = word
                    else:
                        current_subchunk = test_chunk
                if current_subchunk:
                    chunks.append(current_subchunk.strip())
            elif len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Start new chunk with current sentence
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Add overlap between adjacent chunks within same paragraph."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = result[-1]
            curr_chunk = chunks[i]

            # Only apply overlap if same paragraph
            if prev_chunk.metadata.get("paragraph_index") != curr_chunk.metadata.get("paragraph_index"):
                result.append(curr_chunk)
                continue

            # Get last N characters from previous chunk as overlap
            overlap_size = min(self.chunk_overlap, len(prev_chunk.content))
            overlap_text = prev_chunk.content[-overlap_size:] if overlap_size > 0 else ""

            # Create new chunk with overlap prefix
            new_content = overlap_text + curr_chunk.content
            new_metadata = curr_chunk.metadata.copy()
            new_metadata["has_overlap"] = True

            result.append(Chunk(content=new_content, metadata=new_metadata))

        return result


class MultiStrategyChunker:
    """Applies multiple chunking strategies for better retrieval coverage.

    Strategies:
    1. Paragraph chunks (normal) - balanced context
    2. Small window chunks - precise matches for specific terms
    3. Sliding window chunks - same content with different boundaries

    All strategies share the same source metadata for deduplication.
    """

    def __init__(
        self,
        strategies: list[dict] = None,
        json_sensitive: bool = True,
        use_paragraph: bool = True,
        use_small_window: bool = True,
        use_sliding: bool = True,
    ):
        """Initialize with chunking strategies.

        Args:
            strategies: List of strategy configs. Each config is a dict with:
                - chunk_size: int
                - chunk_overlap: int
                - strategy: "paragraph" | "small_window" | "sliding"
            json_sensitive: Whether to detect JSON structures
            use_paragraph: Enable paragraph chunking strategy
            use_small_window: Enable small window chunking strategy
            use_sliding: Enable sliding window chunking strategy
        """
        default_strategies = [
            {"strategy": "paragraph", "chunk_size": 500, "chunk_overlap": 50},
            {"strategy": "small_window", "chunk_size": 200, "chunk_overlap": 30},
            {"strategy": "sliding", "chunk_size": 400, "chunk_overlap": 200},
        ]
        if strategies is not None:
            self.strategies = strategies
        else:
            enabled = []
            if use_paragraph:
                enabled.append(default_strategies[0])
            if use_small_window:
                enabled.append(default_strategies[1])
            if use_sliding:
                enabled.append(default_strategies[2])
            # Fall back to paragraph if all disabled
            if not enabled:
                enabled = [default_strategies[0]]
            self.strategies = enabled
        self.json_sensitive = json_sensitive

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split text using multiple strategies.

        Args:
            text: Input text to chunk
            source: Source identifier for metadata

        Returns:
            Combined list of chunks from all strategies
        """
        if not text or not text.strip():
            return []

        all_chunks = []
        base_chunker = RecursiveChunker(
            chunk_size=500,
            chunk_overlap=50,
            json_sensitive=self.json_sensitive,
        )

        for strategy in self.strategies:
            strategy_name = strategy.get("strategy", "paragraph")
            chunk_size = strategy.get("chunk_size", 500)
            chunk_overlap = strategy.get("chunk_overlap", 50)

            if strategy_name == "paragraph":
                # Standard paragraph chunking
                chunker = RecursiveChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    json_sensitive=self.json_sensitive,
                )
                chunks = chunker.chunk(text, source)
                for c in chunks:
                    c.metadata["chunking_strategy"] = "paragraph"
                all_chunks.extend(chunks)

            elif strategy_name == "small_window":
                # Small window for precise term matching
                chunker = RecursiveChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    json_sensitive=self.json_sensitive,
                )
                chunks = chunker.chunk(text, source)
                for c in chunks:
                    c.metadata["chunking_strategy"] = "small_window"
                all_chunks.extend(chunks)

            elif strategy_name == "sliding":
                # Sliding window with large overlap - captures same content
                # from different starting positions
                chunks = self._sliding_window_chunk(
                    text, source, chunk_size, chunk_overlap
                )
                for c in chunks:
                    c.metadata["chunking_strategy"] = "sliding"
                all_chunks.extend(chunks)

        # Re-index all chunks and add strategy-level index
        strategy_counts = {}
        for chunk in all_chunks:
            strategy = chunk.metadata.get("chunking_strategy", "unknown")
            if strategy not in strategy_counts:
                strategy_counts[strategy] = 0
            chunk.metadata["global_index"] = strategy_counts[strategy]
            strategy_counts[strategy] += 1

        return all_chunks

    def _sliding_window_chunk(
        self, text: str, source: str, window_size: int, step: int
    ) -> list[Chunk]:
        """Create sliding window chunks over text.

        Unlike paragraph-based chunking, this creates overlapping windows
        at fixed byte offsets, ensuring key terms appear in multiple chunks
        at different positions.
        """
        if len(text) <= window_size:
            return [Chunk(
                content=text,
                metadata={"source": source, "chunking_strategy": "sliding"}
            )]

        chunks = []
        start = 0
        window_id = 0

        while start < len(text):
            end = start + window_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary for cleaner chunks
            if end < len(text):
                # Look for sentence end punctuation followed by space
                boundary = self._find_sentence_boundary(chunk_text)
                if boundary > window_size // 2:
                    chunk_text = chunk_text[:boundary]
                    start = start + boundary
                else:
                    start = start + step
            else:
                start = end

            if chunk_text.strip():
                chunks.append(Chunk(
                    content=chunk_text.strip(),
                    metadata={
                        "source": source,
                        "chunking_strategy": "sliding",
                        "window_id": window_id,
                        "window_start": start - len(chunk_text.strip()),
                    }
                ))
                window_id += 1

        return chunks

    def _find_sentence_boundary(self, text: str) -> int:
        """Find the last sentence boundary in text.

        Returns the byte offset of the last [.!?] followed by space/newline,
        or 0 if no boundary found.
        """
        import re
        # Find all sentence endings
        pattern = r"[.!?][\s\n]"
        matches = list(re.finditer(pattern, text))
        if matches:
            # Return position after the last sentence ending punctuation
            last_match = matches[-1]
            return last_match.end()
        return 0
