"""
Configuration System for DocVault

This module handles all project configuration using Pydantic.
Pydantic gives us:
- Automatic type validation
- Clear default values
- IDE autocomplete
- Easy loading from .env or environment variables
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Main DocVault project configuration.

    Values are loaded in this order (descending priority):
    1. System environment variables
    2. .env file in project root
    3. Default values defined here
    """

    # ==========================================
    # General Configuration
    # ==========================================
    project_name: str = Field(
        default="docvault",
        description="Project name"
    )

    environment: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Runtime environment"
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )

    # ==========================================
    # Project Paths
    # ==========================================
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root (calculated automatically)"
    )

    data_dir: Path = Field(
        default=Path("data"),
        description="Data directory"
    )

    documents_dir: Path = Field(
        default=Path("data/documents"),
        description="Documents directory for ingestion"
    )

    # ==========================================
    # Qdrant Vector Database
    # ==========================================
    qdrant_collection_name: str = Field(
        default="docvault_documents",
        description="Qdrant collection name"
    )

    qdrant_storage_path: Path = Field(
        default=Path("data/qdrant_storage"),
        description="Qdrant persistent storage path"
    )

    qdrant_in_memory: bool = Field(
        default=False,
        description="Use in-memory storage (no persistence, for testing)"
    )

    # ==========================================
    # LLM Configuration
    # ==========================================
    llm_provider: str = Field(
        default="ollama_local",
        description="LLM provider type (ollama_local, ollama_server, openai, anthropic)"
    )

    llm_model: Optional[str] = Field(
        default=None,
        description="LLM model name (provider-specific, uses provider default if None)"
    )

    llm_server_url: Optional[str] = Field(
        default=None,
        description="LLM server URL (for ollama_server provider)"
    )

    llm_temperature: float = Field(
        default=0.7,
        description="Generation temperature (0=deterministic, 1=creative)"
    )

    llm_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens in LLM response"
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )

    # ==========================================
    # Pydantic Settings Configuration
    # ==========================================
    model_config = SettingsConfigDict(
        env_file=".env",                    # Read from .env file
        env_file_encoding="utf-8",
        case_sensitive=False,                # Case insensitive
        extra="ignore"                       # Ignore extra variables
    )

    # ==========================================
    # Utility Methods
    # ==========================================
    def get_full_path(self, relative_path: Path) -> Path:
        """
        Convert a relative path to absolute based on project_root.

        Args:
            relative_path: Path relative to project

        Returns:
            Absolute path
        """
        if relative_path.is_absolute():
            return relative_path
        return self.project_root / relative_path

    def ensure_directories(self) -> None:
        """
        Create necessary directories if they don't exist.
        Useful for project initialization.
        """
        directories = [
            self.get_full_path(self.data_dir),
            self.get_full_path(self.documents_dir),
            self.get_full_path(self.qdrant_storage_path),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory verified: {directory}")

    def display_config(self) -> None:
        """
        Display current configuration in a readable format.
        Useful for debugging.
        """
        print("\n" + "=" * 50)
        print(f"  {self.project_name.upper()} - Configuration")
        print("=" * 50)
        print(f"Environment:      {self.environment}")
        print(f"Log Level:        {self.log_level}")
        print(f"Project Root:     {self.project_root}")
        print(f"Data Dir:         {self.get_full_path(self.data_dir)}")
        print(f"Documents Dir:    {self.get_full_path(self.documents_dir)}")
        print(f"Qdrant Collection:{self.qdrant_collection_name}")
        print(f"Qdrant Storage:   {self.get_full_path(self.qdrant_storage_path)}")
        print(f"Qdrant In-Memory: {self.qdrant_in_memory}")
        print("=" * 50 + "\n")


# ==========================================
# Global Configuration Instance
# ==========================================
# This is the instance we'll use throughout the project
settings = Settings()


# ==========================================
# Helper Function for Testing
# ==========================================
def get_settings() -> Settings:
    """
    Get configuration instance.
    Useful for dependency injection in testing.

    Returns:
        Settings instance
    """
    return settings


# ==========================================
# To use from other modules:
# ==========================================
# from config.settings import settings
#
# print(settings.project_name)
# settings.ensure_directories()
# ==========================================
