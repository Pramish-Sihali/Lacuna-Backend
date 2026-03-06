"""
Configuration settings for Lacuna backend.
Loads environment variables and provides application-wide settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database Configuration
    DATABASE_URL: str = "postgresql+asyncpg://lacuna_user:lacuna_password@localhost:5432/lacuna_db"

    # AWS Bedrock Configuration
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    AWS_BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
    AWS_BEDROCK_LLM_MODEL: str = "amazon.nova-lite-v1:0"
    AWS_BEDROCK_TIMEOUT: int = 120  # seconds for LLM requests

    # Vector Configuration
    VECTOR_DIMENSION: int = 1024  # Titan Embeddings V2 outputs 1024-dimensional vectors

    # Application Settings
    UPLOAD_DIR: str = "./uploads"
    CHUNK_SIZE: int = 500  # tokens
    CHUNK_OVERLAP: int = 50  # tokens
    SMALL_DOC_CHAR_THRESHOLD: int = 8000  # Docs <= this skip chunking & chunk embedding
    SMALL_DOC_MAX_CONCEPTS: int = 8       # Max concepts from whole-doc extraction
    SMALL_DOC_MIN_CONCEPTS: int = 5       # Min concepts from whole-doc extraction
    DEFAULT_PROJECT_ID: int = 1

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:3001"

    # OCR Configuration
    TESSERACT_CMD: str = "/usr/bin/tesseract"

    # Processing Configuration
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    SUPPORTED_FILE_TYPES: List[str] = [".pdf", ".docx", ".doc"]

    # Clustering Configuration
    HDBSCAN_MIN_CLUSTER_SIZE: int = 3
    HDBSCAN_MIN_SAMPLES: int = 2
    HDBSCAN_METRIC: str = "euclidean"

    # Gap Detection Configuration
    GAP_SIMILARITY_THRESHOLD: float = 0.7
    MIN_COVERAGE_THRESHOLD: float = 0.3

    # Concept Normalisation Configuration
    # Cosine-similarity threshold above which two concepts are considered the same.
    # Lower = more aggressive merging (0.72 merges "neural networks" ↔ "deep learning").
    CONCEPT_SIMILARITY_THRESHOLD: float = 0.72
    # Relaxed threshold used when the project has fewer than
    # CONCEPT_SMALL_PROJECT_THRESHOLD concepts
    CONCEPT_SIMILARITY_THRESHOLD_SMALL: float = 0.68
    # Projects with fewer concepts than this use the small-project threshold.
    # Set to 50 so the relaxed threshold fires for typical 10-paper collections.
    CONCEPT_SMALL_PROJECT_THRESHOLD: int = 50

    # Minimum LLM extraction confidence to accept a concept (0.0-1.0)
    MIN_CONCEPT_CONFIDENCE: float = 0.75

    # /api/concepts/map hard caps
    MAP_MAX_CONCEPT_NODES: int = 20  # cap on non-gap nodes returned to frontend
    MAP_MAX_GAP_NODES: int = 3       # cap on gap nodes returned to frontend

    # Clustering Configuration (advanced)
    # Cosine similarity above which two concepts in different clusters are linked
    CLUSTER_BRIDGE_THRESHOLD: float = 0.60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def get_allowed_origins(self) -> List[str]:
        """Parse ALLOWED_ORIGINS string into a list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]


# Global settings instance
settings = Settings()
