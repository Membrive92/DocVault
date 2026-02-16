"""
Manages ingestion state to track which files have been indexed.

Stores state in a JSON file that records the file path, modification time,
chunk count, and indexing timestamp. Used by the pipeline to skip files
that have already been indexed and haven't changed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import INDEX_STATE_FILE

logger = logging.getLogger(__name__)


class IngestionStateManager:
    """
    Tracks which files have been indexed and their metadata.

    Uses a JSON file to persist state across pipeline runs. Detects
    file modifications via mtime comparison to support incremental
    re-indexing.
    """

    def __init__(self, state_file: Optional[str | Path] = None) -> None:
        """
        Initialize the state manager.

        Args:
            state_file: Path to JSON state file. Defaults to INDEX_STATE_FILE
                       resolved relative to project root.
        """
        if state_file is None:
            from config.settings import settings

            self.state_file = settings.get_full_path(Path(INDEX_STATE_FILE))
        else:
            self.state_file = Path(state_file)

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, dict] = self._load_state()

        logger.info("State manager initialized: %s", self.state_file)

    def _load_state(self) -> dict[str, dict]:
        """Load state from JSON file. Returns empty dict if file doesn't exist."""
        if not self.state_file.exists():
            return {}

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load state file, starting fresh: %s", e)
            return {}

    def _save_state(self) -> None:
        """Persist current state to JSON file."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def is_indexed(self, file_path: str | Path) -> bool:
        """
        Check if a file has been indexed and hasn't changed since.

        Compares the file's current mtime against the stored mtime
        to detect modifications.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if file is indexed AND unchanged since last indexing.
        """
        path = Path(file_path)
        key = str(path.resolve())

        if key not in self.state:
            return False

        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return False

        stored_mtime = self.state[key].get("mtime")
        return stored_mtime == current_mtime

    def mark_indexed(
        self,
        file_path: str | Path,
        chunk_count: int,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Record a file as successfully indexed.

        Args:
            file_path: Path to the indexed file.
            chunk_count: Number of chunks created from this file.
            metadata: Optional additional metadata to store.
        """
        path = Path(file_path)
        key = str(path.resolve())

        self.state[key] = {
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "mtime": path.stat().st_mtime,
            "chunk_count": chunk_count,
            "metadata": metadata or {},
        }

        self._save_state()
        logger.debug("Marked as indexed: %s (%d chunks)", path.name, chunk_count)

    def remove_indexed(self, file_path: str | Path) -> None:
        """
        Remove a file from the indexed state.

        Args:
            file_path: Path to the file to remove.
        """
        key = str(Path(file_path).resolve())
        if key in self.state:
            del self.state[key]
            self._save_state()
            logger.debug("Removed from index: %s", file_path)

    def get_stats(self) -> dict:
        """
        Get indexing statistics.

        Returns:
            Dictionary with total_files and total_chunks counts.
        """
        return {
            "total_files": len(self.state),
            "total_chunks": sum(
                entry.get("chunk_count", 0) for entry in self.state.values()
            ),
        }
