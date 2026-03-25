from __future__ import annotations

from pathlib import Path

from aria.config import AriaConfig


class RepoReader:
    """
    Read-only filesystem access scoped to the configured repo root.
    """

    def __init__(self, config: AriaConfig) -> None:
        self._root = Path(config.data_sources.repo.root).resolve()

    def read_text(self, relative_path: str, encoding: str = "utf-8") -> str:
        """
        Read a text file under the repo root using a relative path.
        """
        target = (self._root / relative_path).resolve()
        if not target.is_relative_to(self._root):
            raise ValueError("Attempt to access path outside repo root.")
        return target.read_text(encoding=encoding)

