from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from ddgs import DDGS

from aria.config import AriaConfig

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    url: str
    title: str
    content: str
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Domain quality tiers (higher = better)
# ---------------------------------------------------------------------------

# Tier 4 — authoritative primary sources
_TIER4 = {
    # Central banks / government
    "federalreserve.gov", "newyorkfed.org", "bls.gov", "census.gov",
    "ecb.europa.eu", "bankofengland.co.uk", "bis.org", "imf.org",
    "worldbank.org", "treasury.gov", "sec.gov", "cftc.gov",
    # Academic
    "ssrn.com", "nber.org", "jstor.org",
    # Tier-1 financial journalism
    "ft.com", "wsj.com", "bloomberg.com", "reuters.com", "economist.com",
    # Commodity / market data authorities
    "gold.org",  # World Gold Council
    "cmegroup.com", "ice.com", "lbma.org.uk",
}

# Tier 3 — established financial/business press
_TIER3 = {
    "cnbc.com", "marketwatch.com", "barrons.com", "forbes.com",
    "businessinsider.com", "fortune.com", "theatlantic.com",
    "apnews.com", "nytimes.com", "washingtonpost.com", "theguardian.com",
    "axios.com", "politico.com",
    "kitco.com", "mining.com", "mining-technology.com",
    "spglobal.com", "moodys.com", "fitchratings.com",
    "gs.com", "jpmorgan.com", "morganstanley.com", "blackrock.com",
    "vanguard.com", "ubs.com", "credit-suisse.com", "hsbc.com",
    "investopedia.com", "seekingalpha.com",
    "fxstreet.com", "fxempire.com",
    "lseg.com", "refinitiv.com",
    "clevelandfed.org", "stlouisfed.org", "dallasfed.org",
    "abrardeen.com", "aberdeen.com", "aberdeeninvestments.com",
    "morningstar.com"
}

# Tier 2 — acceptable secondary sources
_TIER2 = {
    "goldsilver.com", "augustapreciousmetals.com", "jmbullion.com",
    "moneyweek.com", "thestreet.com", "benzinga.com",
    "finance.yahoo.com", "livewiremarkets.com",
    "analyticsinsight.net", "entrepreneur.com",
    "cbsnews.com", "nbcnews.com", "abcnews.go.com",
    "cnn.com", "bbc.com", "bbc.co.uk",
    "zacks.com", "motleyfool.com",
}

# Domains to exclude entirely
_DENYLIST = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com",
    "pinterest.com", "linkedin.com",
    # Known low-quality gold-bug / AI content farms
    "goldco.com", "birchgold.com", "wewantrealnews.com",
    "discoveryalert.com.au", "goldzeus.com", "debuglies.com",
    "cryptoquorum.com", "yuantrends.com", "suneyez.com",
    "knowinsiders.com", "finscann.com", "intellectia.ai",
    "tmastreet.com",
}


def _domain_score(url: str) -> int:
    """Return a quality tier (4=best, 0=unknown, -1=blocked) for a URL."""
    m = re.search(r"https?://(?:www\.)?([^/\s]+)", url)
    if not m:
        return 0
    host = m.group(1).lower()
    parts = host.split(".")
    # Root domain = last two labels (handles subdomains)
    root = f"{parts[-2]}.{parts[-1]}" if len(parts) >= 2 else host

    if root in _DENYLIST or host in _DENYLIST:
        return -1
    if root in _TIER4 or host in _TIER4:
        return 4
    if root in _TIER3 or host in _TIER3:
        return 3
    if root in _TIER2 or host in _TIER2:
        return 2
    # .gov / .edu / .org get a small bump over completely unknown domains
    tld = parts[-1]
    if tld in ("gov", "edu"):
        return 3
    if tld == "org":
        return 2
    return 1  # unknown but not blocked


class WebSearchClient:
    """
    Web search using DuckDuckGo, with domain-quality reranking.

    Fetches 3× max_results from DuckDuckGo, drops blocked domains, sorts
    the remainder by quality tier (highest first), then returns the top
    max_results. The model only ever sees the highest-quality sources
    available for each query.
    """

    def __init__(self, config: AriaConfig) -> None:
        self._config = config

    def search(self, query: str, max_results: int = 6) -> List[WebSearchResult]:
        if not self._config.data_sources.web_search:
            return []

        raw: List[WebSearchResult] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=max_results * 3):
                    url = item.get("href") or item.get("link") or ""
                    title = item.get("title") or ""
                    content = item.get("body") or item.get("snippet") or item.get("content") or ""
                    score = _domain_score(url)
                    if score == -1:
                        continue  # drop blocked domains entirely
                    raw.append(WebSearchResult(url=url, title=title, content=content, score=score))
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)

        # Sort by tier descending, preserve DuckDuckGo relevance order within each tier.
        raw.sort(key=lambda r: -(r.score or 0))
        return raw[:max_results]
