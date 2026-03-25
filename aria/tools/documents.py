from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from aria.config import AriaConfig


@dataclass
class IngestedDocument:
    path: str
    text: str


class DocumentIngestor:
    """
    Simple document ingestion for V1.

    For now, this reads raw text from supported formats and returns it to
    the agent. In V2, this will be wired into a vector store (e.g., Chroma).
    """

    SUPPORTED_EXT = {".txt", ".md"}

    def __init__(self, config: AriaConfig) -> None:
        self._config = config
        self._root = Path(self._config.data_sources.documents.ingest_path)

    def iter_documents(self) -> Iterable[IngestedDocument]:
        if not self._root.exists():
            return []
        docs: List[IngestedDocument] = []
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.SUPPORTED_EXT:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(IngestedDocument(path=str(path), text=text))
        return docs

