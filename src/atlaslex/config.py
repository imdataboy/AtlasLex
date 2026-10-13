"""Configuration shared by the CLI and pipeline. No models load at import time."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, set_key

EMBED_MODEL = "antoinelouis/french-me5-base"
RERANK_MODEL = "antoinelouis/crossencoder-me5-base-mmarcoFR"
EMBED_REVISION = "a04bf812045058a507e47b4a097e16cb0ce825dc"
RERANK_REVISION = "4ad0d002cd56ebb8f8b63a86b75073736c559c7c"
PIPELINE_VERSION = "1"


def project_home() -> Path:
    if value := os.getenv("ATLASLEX_HOME"):
        return Path(value).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout
    return Path.home() / ".atlaslex"


@dataclass
class Settings:
    home: Path = field(default_factory=project_home)
    api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3-flash-preview"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection: str = "atlaslex"
    use_query_rewrite: bool = True
    dense_top_k: int = 10
    sparse_top_k: int = 10
    final_top_k: int = 3
    min_rerank_score: float = 0.1
    max_chunk_chars: int = 1800
    overlap_chars: int = 200

    @property
    def raw_dir(self) -> Path:
        return self.home / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.home / "data" / "processed"

    @property
    def env_path(self) -> Path:
        return self.home / ".env"

    def save_api_key(self, key: str) -> None:
        key = key.strip()
        if not key or any(c.isspace() for c in key) or "\x00" in key:
            raise ValueError("Enter a non-empty API key without whitespace.")
        self.home.mkdir(parents=True, exist_ok=True)
        # Create with restricted permissions before writing the credential.
        fd = os.open(self.env_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(fd)
        self.env_path.chmod(0o600)
        set_key(str(self.env_path), "GEMINI_API_KEY", key)
        self.env_path.chmod(0o600)
        self.api_key = key

    @classmethod
    def load(cls, home: Path | None = None) -> "Settings":
        settings = cls(home=(home or project_home()).resolve())
        values = {**dotenv_values(settings.env_path), **os.environ}
        mapping = {
            "api_key": ("GEMINI_API_KEY", str),
            "gemini_model": ("GEMINI_MODEL", str),
            "qdrant_host": ("QDRANT_HOST", str),
            "qdrant_port": ("QDRANT_PORT", int),
            "collection": ("QDRANT_COLLECTION", str),
            "dense_top_k": ("DENSE_TOP_K", int),
            "sparse_top_k": ("SPARSE_TOP_K", int),
            "final_top_k": ("FINAL_TOP_K", int),
            "min_rerank_score": ("MIN_RERANK_SCORE", float),
        }
        for attr, (name, convert) in mapping.items():
            if values.get(name):
                try:
                    setattr(settings, attr, convert(values[name]))
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid {name} configuration.") from None
        flag = str(values.get("USE_QUERY_REWRITE", "true")).lower()
        if flag not in {"true", "false", "1", "0"}:
            raise ValueError("USE_QUERY_REWRITE must be true or false.")
        settings.use_query_rewrite = flag in {"true", "1"}
        if not 1 <= settings.qdrant_port <= 65535:
            raise ValueError("QDRANT_PORT must be between 1 and 65535.")
        if min(settings.dense_top_k, settings.sparse_top_k, settings.final_top_k) < 1:
            raise ValueError("Retrieval limits must be positive.")
        if not 0 <= settings.min_rerank_score <= 1:
            raise ValueError("MIN_RERANK_SCORE must be between 0 and 1.")
        if not settings.collection.replace("_", "").replace("-", "").isalnum():
            raise ValueError("QDRANT_COLLECTION must contain letters, digits, - or _.")
        return settings
