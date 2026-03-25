"""
Tests for WebSearchClient domain quality ranking.

Runs offline — no DuckDuckGo calls. Tests the _domain_score function
which is the core intelligence of the search client (reranking by source quality).
"""
from __future__ import annotations

import pytest

from aria.tools.web_search import _domain_score


class TestDomainScore:
    # ------------------------------------------------------------------
    # Tier 4 — authoritative primary sources
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("url", [
        "https://www.federalreserve.gov/monetary-policy",
        "https://newyorkfed.org/research",
        "https://bls.gov/news.release/cpi.htm",
        "https://sec.gov/cgi-bin/browse-edgar",
        "https://ft.com/content/some-article",
        "https://bloomberg.com/news/articles/market-update",
        "https://reuters.com/markets/rates-bonds/",
        "https://wsj.com/finance/stocks/nvda",
        "https://nber.org/papers/w12345",
        "https://gold.org/goldhub/data",
        "https://imf.org/en/Publications/WEO",
    ])
    def test_tier4_domains_score_4(self, url: str) -> None:
        assert _domain_score(url) == 4, f"Expected tier 4 for {url}"

    # ------------------------------------------------------------------
    # Tier 3 — established financial/business press
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("url", [
        "https://cnbc.com/2026/03/market-update",
        "https://marketwatch.com/story/gold-outlook",
        "https://investopedia.com/gold-price-analysis",
        "https://seekingalpha.com/article/xyz",
        "https://stlouisfed.org/on-the-economy/2026",
        "https://clevelandfed.org/research",
        "https://spglobal.com/ratings/en/research",
        "https://university.edu/research/paper",  # .edu → tier 3
    ])
    def test_tier3_domains_score_3(self, url: str) -> None:
        assert _domain_score(url) == 3, f"Expected tier 3 for {url}"

    # ------------------------------------------------------------------
    # Tier 2 — acceptable secondary sources
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("url", [
        "https://zacks.com/stock-analysis/nvda",
        "https://motleyfool.com/investing/gold",
        "https://finance.yahoo.com/quote/GC=F",
        "https://thestreet.com/markets/commodities",
        "https://opendomain.org/report",  # .org → tier 2
    ])
    def test_tier2_domains_score_2(self, url: str) -> None:
        assert _domain_score(url) == 2, f"Expected tier 2 for {url}"

    # ------------------------------------------------------------------
    # Tier 1 — unknown but not blocked
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("url", [
        "https://someunknownblog.com/gold-thesis",
        "https://randomsite.net/market-analysis",
        "https://niche-finance-site.co/report",
    ])
    def test_unknown_domains_score_1(self, url: str) -> None:
        assert _domain_score(url) == 1, f"Expected tier 1 for {url}"

    # ------------------------------------------------------------------
    # Denylist — returns -1
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("url", [
        "https://facebook.com/market-news",
        "https://twitter.com/financetips",
        "https://x.com/stockpicks",
        "https://instagram.com/tradingtips",
        "https://goldco.com/buy-gold",
        "https://birchgold.com/ira",
        "https://intellectia.ai/analysis",
    ])
    def test_denylisted_domains_return_minus_one(self, url: str) -> None:
        assert _domain_score(url) == -1, f"Expected -1 (blocked) for {url}"

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_invalid_url_returns_zero(self) -> None:
        assert _domain_score("not-a-url") == 0
        assert _domain_score("") == 0

    def test_subdomain_resolves_to_root_domain(self) -> None:
        # research.stlouisfed.org → root = stlouisfed.org → tier 3
        assert _domain_score("https://research.stlouisfed.org/fred/") == 3
        # www prefix stripped
        assert _domain_score("https://www.ft.com/markets") == 4

    def test_www_prefix_does_not_affect_score(self) -> None:
        assert _domain_score("https://bloomberg.com/news") == _domain_score("https://www.bloomberg.com/news")

    def test_path_does_not_affect_score(self) -> None:
        base = _domain_score("https://reuters.com")
        with_path = _domain_score("https://reuters.com/markets/us/fed-rates-2026-03-20/")
        assert base == with_path == 4
