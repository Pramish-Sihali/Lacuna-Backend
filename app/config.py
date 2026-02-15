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

    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_LLM_MODEL: str = "qwen2.5:3b"
    OLLAMA_TIMEOUT: int = 300  # 5 minutes for LLM requests

    # Vector Configuration
    VECTOR_DIMENSION: int = 768  # nomic-embed-text outputs 768-dimensional vectors

    # Application Settings
    UPLOAD_DIR: str = "./uploads"
    CHUNK_SIZE: int = 500  # tokens
    CHUNK_OVERLAP: int = 50  # tokens
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
