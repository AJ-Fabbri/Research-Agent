"""
Tool layer for ARIA.

Provides thin, purpose-built wrappers around external capabilities:
- Web search
- Financial data
- Document ingestion (stub for V1, vector store in V2)
- Repo / filesystem access
"""

from .web_search import WebSearchClient
from .financial import FinancialDataClient
from .documents import DocumentIngestor
from .repo import RepoReader

__all__ = [
    "WebSearchClient",
    "FinancialDataClient",
    "DocumentIngestor",
    "RepoReader",
]

