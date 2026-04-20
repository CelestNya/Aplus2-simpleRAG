"""Chroma vector store wrapper."""

from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings

from .config import SearchConfig
from .embeddings import EmbeddingService


@dataclass
class SearchResult:
    """A search result from the vector store."""
    content: str
    metadata: dict
    distance: float


class VectorStore:
    """Chroma-based vector store with embedding support."""

    def __init__(
        self,
        persist_dir: str,
        embedding_service: EmbeddingService,
        search_config: SearchConfig | None = None,
    ):
        """Initialize the vector store.

        Args:
            persist_dir: Directory for Chroma persistence
            embedding_service: Service for generating embeddings
            search_config: Search threshold parameters (uses defaults if None)
        """
        self.persist_dir = persist_dir
        self.embedding_service = embedding_service
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._search_config = search_config or SearchConfig()

    def _get_client(self) -> chromadb.PersistentClient:
        """Lazy initialize the Chroma client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        """Get or create the default collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(name="documents")
        return self._collection

    def reset(self) -> None:
        """Delete all documents from the collection."""
        client = self._get_client()
        try:
            client.delete_collection(name="documents")
        except Exception:
            pass
        self._collection = None

    def add_chunks(
        self,
        chunks: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            return

        collection = self._get_collection()

        # Generate embeddings for all chunks
        embeddings = self.embedding_service.embed(chunks)

        # Chroma rejects empty metadata dicts; coerce them to None
        safe_metadatas = [m if m else None for m in metadatas]

        collection.add(
            documents=chunks,
            metadatas=safe_metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    def search(
        self, query_text: str, top_k: int = 5, post_process: bool | None = None
    ) -> list[SearchResult]:
        """Search with query expansion for exact keyword matches.

        For queries containing specific terms (like ChatGPT, GPT-4, etc.),
        performs an exact-match fallback search to supplement semantic similarity.
        This ensures precise matches aren't buried under semantically similar
        but contextually different content.

        Args:
            query_text: The search query
            top_k: Number of results to return
            post_process: Override config post_process setting. None uses config default.
        """
        collection = self._get_collection()
        enable_post_process = (
            post_process if post_process is not None else self._search_config.post_process
        )
        query_keywords = self._extract_keywords(query_text)

        # Primary: semantic vector search
        query_embedding = self.embedding_service.embed_single(query_text)

        # Fetch more candidates to ensure keyword matches surface
        actual_count = collection.count()
        n_results = min(top_k * 5, actual_count)
        if n_results == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        # Convert to SearchResult objects
        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                metadata = metadata or {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                search_results.append(
                    SearchResult(
                        content=doc,
                        metadata=metadata,
                        distance=distance,
                    )
                )

        # Post-processing: exact match, keyword boost, deduplicate, filter
        if enable_post_process:
            # Exact match fallback: if query has keywords, also search by exact match
            exact_matches = []
            if query_keywords:
                exact_matches = self._exact_match_search(collection, query_keywords)

            # Merge: exact matches get a significant distance bonus
            if exact_matches:
                search_results = self._merge_exact_matches(
                    search_results, exact_matches, query_keywords
                )

            # Final keyword boost for fine-tuning within merged results
            if query_keywords:
                search_results = self._boost_keyword_matches(search_results, query_keywords)

            # Deduplicate by content: keep the one with lowest distance
            search_results = self._deduplicate_results(search_results)

            # Filter by relevance threshold
            search_results = self._filter_irrelevant(search_results, query_keywords)

        # Hard distance cap from config: results above this threshold are discarded.
        if (
            search_results
            and self._get_collection().count() > 1
            and search_results[0].distance > self._search_config.hard_cap
        ):
            return []

        search_results.sort(key=lambda r: r.distance)
        return search_results[:top_k]

    def _exact_match_search(self, collection, keywords: list[str]) -> list[dict]:
        """Find documents where keywords appear verbatim (case-insensitive)."""
        exact_matches = []
        all_data = collection.get(include=["documents", "metadatas"])

        if not all_data.get("documents"):
            return exact_matches

        for i, doc in enumerate(all_data["documents"]):
            doc_lower = doc.lower()
            matched = sum(1 for kw in keywords if kw.lower() in doc_lower)
            if matched > 0:
                metadata = all_data["metadatas"][i] or {}
                exact_matches.append({
                    "content": doc,
                    "metadata": metadata,
                    "matched_count": matched,
                })

        # Sort by matched keyword count (more matches = better)
        exact_matches.sort(key=lambda x: -x["matched_count"])
        return exact_matches[:20]  # Limit to top 20 exact matches

    def _merge_exact_matches(
        self,
        semantic_results: list[SearchResult],
        exact_matches: list[dict],
        keywords: list[str],
    ) -> list[SearchResult]:
        """Merge exact matches into semantic results with distance bonus.

        Exact matches get their distance reduced significantly so they rank
        higher than semantically similar but contextually different content.
        """
        # Create a set of already-seen contents for deduplication
        seen_contents = {r.content for r in semantic_results}

        merged = list(semantic_results)

        for em in exact_matches:
            content = em["content"]
            if content in seen_contents:
                # Already have this content, update distance if exact match is better
                for r in merged:
                    if r.content == content:
                        if em["matched_count"] >= len(keywords):
                            # Full keyword match - reduce distance significantly
                            r.distance = r.distance * 0.15
                        else:
                            # Partial match - modest boost
                            r.distance = r.distance * 0.5
                        break
            else:
                # New exact match - add with heavily reduced distance
                # Use a very low base distance since we know it's an exact match
                # More keyword matches = bigger discount
                match_ratio = em["matched_count"] / len(keywords)
                # Base distance for new exact matches: 0.2 (very close)
                # But give preference to more keyword matches
                base_distance = 0.25 - (match_ratio * 0.15)
                new_result = SearchResult(
                    content=content,
                    metadata=em["metadata"],
                    distance=base_distance,
                )
                merged.append(new_result)
                seen_contents.add(content)

        return merged

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract important keywords from query text.

        Extracts alphanumeric sequences (like ChatGPT, GPT-4) that should be
        boosted when found verbatim in documents.
        """
        import re
        # Extract words longer than 2 chars, including hyphens and camelCase
        keywords = re.findall(r"[A-Za-z0-9]{3,}(?:[-_][A-Za-z0-9]+)*", text)
        return [k.lower() for k in keywords]

    def _boost_keyword_matches(
        self, results: list[SearchResult], keywords: list[str]
    ) -> list[SearchResult]:
        """Boost results containing exact query keywords.

        For each result, if it contains query keywords, reduce its distance
        (make it rank higher). The boost is proportional to keyword coverage.
        """
        for r in results:
            content_lower = r.content.lower()
            matched_keywords = sum(1 for kw in keywords if kw in content_lower)
            if matched_keywords > 0:
                # Strong boost: each matched keyword reduces distance by 40%
                # Multiple keywords stack multiplicatively
                keyword_boost = 1 - (0.4 ** matched_keywords)
                r.distance = r.distance * (1 - keyword_boost)

        return results

    def _deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate content, keeping the result with lowest distance.

        Multi-strategy chunking produces multiple chunks with identical content
        but different boundaries. Deduplication keeps only the best-ranked one.
        """
        seen: dict[str, SearchResult] = {}
        for r in results:
            # Use content as key for deduplication
            if r.content not in seen:
                seen[r.content] = r
            else:
                # Keep the one with lower distance
                if r.distance < seen[r.content].distance:
                    seen[r.content] = r
        return list(seen.values())

    def _filter_irrelevant(
        self, results: list[SearchResult], keywords: list[str]
    ) -> list[SearchResult]:
        """Filter out results with distance above relevance threshold.

        Uses adaptive thresholding:
        - For queries with keywords: keep keyword matches (exact) regardless of
          distance; non-keyword results filtered to min_dist + 0.3
        - For pure semantic queries: filter to min_dist + 0.3
        """
        if not results:
            return results

        min_dist = min(r.distance for r in results)

        if keywords:
            has_keyword = lambda r: any(
                kw in r.content.lower() for kw in keywords
            )
            keyword_results = [r for r in results if has_keyword(r)]
            non_keyword_results = [r for r in results if not has_keyword(r)]

            # Non-keyword results: only keep if within filter_delta of best
            threshold = min_dist + self._search_config.filter_delta
            filtered_non = [r for r in non_keyword_results if r.distance <= threshold]

            return keyword_results + filtered_non

        # No keywords: adaptive threshold relative to best result
        threshold = min_dist + self._search_config.filter_delta
        return [r for r in results if r.distance <= threshold]
