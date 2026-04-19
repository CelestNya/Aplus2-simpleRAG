"""Configuration management for SimpleRAG."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LLMConfig:
    """LLM API configuration."""
    base_url: str
    model: str
    api_key: str


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_path: str
    dimension: int = 1024


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    persist_dir: str


@dataclass
class ChunkingConfig:
    """Chunking configuration for document processing."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    use_paragraph: bool = True
    use_small_window: bool = True
    use_sliding: bool = True


@dataclass
class SearchConfig:
    """Search configuration for query relevance thresholds."""
    default_top_k: int = 5
    hard_cap: float = 50.0       # Absolute distance cap; results above this are discarded
    filter_delta: float = 0.3    # Relative threshold: keep results within min_dist + delta
    post_process: bool = True   # Enable keyword matching, boosting, and filtering


@dataclass
class Config:
    """Main configuration container."""
    llm: LLMConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    documents_input_dir: str = "./documents"


class ConfigManager:
    """Loads and manages configuration from YAML file."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)

    def load(self) -> Config:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return Config(
            llm=LLMConfig(**data["llm"]),
            embedding=EmbeddingConfig(**data["embedding"]),
            vectorstore=VectorStoreConfig(**data["vectorstore"]),
            chunking=ChunkingConfig(**data.get("chunking", {})),
            search=SearchConfig(**data.get("search", {})),
            documents_input_dir=data.get("documents", {}).get("input_dir", "./documents"),
        )


def load_config(config_path: str = "config.yaml") -> Config:
    """Convenience function to load configuration."""
    return ConfigManager(config_path).load()
