"""
Parallel AI web search — context enrichment for 8-K filings.

Docs: https://docs.parallel.ai/api-reference/search-beta/search
Requires PARALLEL_API_KEY in environment.
"""

import time

import requests

PARALLEL_SEARCH_URL = "https://api.parallel.ai/v1beta/search"
_DELAY = 0.12  # ~8 req/s; well under the 600/min rate limit


def search_filing_context(
    objective: str,
    api_key: str,
    max_results: int = 3,
) -> list[str]:
    """
    Search for recent web context relevant to a filing event.
    Uses "fast" mode — optimised for low latency over deep research.

    Returns a flat list of excerpt strings (empty list on any failure).
    """
    try:
        resp = requests.post(
            PARALLEL_SEARCH_URL,
            headers={
                "x-api-key": api_key,
                "parallel-beta": "search-extract-2025-10-10",
                "Content-Type": "application/json",
            },
            json={
                "objective": objective,
                "mode": "fast",
                "max_results": max_results,
            },
            timeout=30,
        )
        time.sleep(_DELAY)
        if not resp.ok:
            return []
        excerpts: list[str] = []
        for result in resp.json().get("results", []):
            for ex in (result.get("excerpts") or []):
                if ex:
                    excerpts.append(str(ex))
        return excerpts
    except Exception:
        return []
