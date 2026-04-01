"""
SEC EDGAR API client for pulling 8-K filings.

SEC EDGAR endpoints used:
  - https://www.sec.gov/files/company_tickers.json       (ticker -> CIK map)
  - https://www.sec.gov/cgi-bin/browse-edgar             (live Atom feed)
  - https://data.sec.gov/submissions/CIK{cik}.json       (primary doc lookup)
  - https://www.sec.gov/Archives/edgar/data/...           (filing documents)

Rate limit: SEC allows ~10 requests/second. A 0.11s delay is applied between
requests to stay well within limits and avoid being blocked.
"""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import requests

BASE_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
EDGAR_CURRENT_URL    = "https://www.sec.gov/cgi-bin/browse-edgar"
TICKERS_URL          = "https://www.sec.gov/files/company_tickers.json"
ARCHIVES_URL         = "https://www.sec.gov/Archives/edgar/data"

_ATOM_NS = "http://www.w3.org/2005/Atom"

_RATE_LIMIT_DELAY = 0.11  # seconds between requests


@dataclass
class Filing:
    accession_number: str        # e.g. "0000320193-24-000123"
    cik: str                     # numeric string without leading zeros
    company_name: str
    filed_date: str              # "YYYY-MM-DD"
    report_date: str             # "YYYY-MM-DD"
    form_type: str               # "8-K" or "8-K/A"
    description: str
    primary_doc: str = ""        # primary document filename, e.g. "aapl-20241231.htm"
    filed_time: str = ""         # "HH:MM:SS" from Atom feed (Eastern time)
    documents: list[dict] = field(default_factory=list)

    @property
    def index_url(self) -> str:
        acc_no_dashes = self.accession_number.replace("-", "")
        return f"{ARCHIVES_URL}/{self.cik}/{acc_no_dashes}/{self.accession_number}-index.htm"

    @property
    def primary_document_url(self) -> Optional[str]:
        """URL of the main 8-K HTML/text document."""
        acc_no_dashes = self.accession_number.replace("-", "")
        base = f"{ARCHIVES_URL}/{self.cik}/{acc_no_dashes}"

        if self.primary_doc:
            return f"{base}/{self.primary_doc}"

        for doc in self.documents:
            if doc.get("type") in ("8-K", "8-K/A"):
                return f"{base}/{doc['name']}"
        if self.documents:
            return f"{base}/{self.documents[0]['name']}"
        return None


class EdgarClient:
    """
    Client for the SEC EDGAR public API.

    Per SEC policy, the User-Agent must identify the application and include
    contact information: https://www.sec.gov/developer
    """

    def __init__(self, user_agent: str = "sec-pull/1.0 (contact@example.com)"):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._ticker_map: Optional[dict[str, str]] = None

    # ------------------------------------------------------------------
    # Low-level request helpers
    # ------------------------------------------------------------------

    def _get_json(self, url: str, params: Optional[dict] = None) -> dict:
        resp = self.session.get(url, params=params, timeout=20)
        resp.raise_for_status()
        time.sleep(_RATE_LIMIT_DELAY)
        return resp.json()

    def _get_bytes(self, url: str) -> bytes:
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        time.sleep(_RATE_LIMIT_DELAY)
        return resp.content

    # ------------------------------------------------------------------
    # Ticker map
    # ------------------------------------------------------------------

    def _load_ticker_map(self) -> dict[str, str]:
        """Fetch the SEC's full ticker->CIK mapping (cached after first call)."""
        data = self._get_json(TICKERS_URL)
        # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
        return {
            entry["ticker"].upper(): str(entry["cik_str"])
            for entry in data.values()
        }

    # ------------------------------------------------------------------
    # Global recent filings feed (Atom)
    # ------------------------------------------------------------------

    def get_recent_filings(
        self,
        max_results: int = 20,
        form_type: str = "8-K",
        large_cap_ciks: Optional[frozenset[str]] = None,
        since_date: Optional[str] = None,
    ) -> list[Filing]:
        """
        Return the most recently filed 8-Ks across ALL companies, newest first.

        Uses the EDGAR live Atom feed (/cgi-bin/browse-edgar?action=getcurrent).
        The feed returns at most 40 entries per page; when large_cap_ciks or
        since_date is supplied we paginate until we have max_results matching
        entries (up to 20 pages / ~800 candidates).

        Args:
            max_results:     How many filings to return.
            form_type:       "8-K" (also matches 8-K/A from the feed).
            large_cap_ciks:  Frozenset of zero-padded 10-digit CIK strings to
                             filter on.
            since_date:      "YYYY-MM-DD". Stop paginating once filings older
                             than this date are encountered.
        """
        results: list[Filing] = []
        seen: set[str] = set()
        max_pages = 20 if (large_cap_ciks or since_date) else 1
        per_page = 40
        done = False

        for page in range(max_pages):
            params = {
                "action": "getcurrent",
                "type": form_type,
                "dateb": "",
                "owner": "include",
                "count": per_page,
                "start": page * per_page,
                "output": "atom",
            }
            resp = self.session.get(EDGAR_CURRENT_URL, params=params, timeout=20)
            resp.raise_for_status()
            time.sleep(_RATE_LIMIT_DELAY)

            root = ET.fromstring(resp.content)
            entries = root.findall(f"{{{_ATOM_NS}}}entry")
            if not entries:
                break

            new_this_page = 0
            for entry in entries:
                filing = self._parse_atom_entry(entry)
                if filing is None:
                    continue
                if filing.accession_number in seen:
                    continue
                seen.add(filing.accession_number)
                new_this_page += 1
                if since_date and filing.filed_date and filing.filed_date < since_date:
                    done = True
                    break
                if large_cap_ciks and filing.cik.zfill(10) not in large_cap_ciks:
                    continue
                results.append(filing)
                if len(results) >= max_results:
                    break

            if done or len(results) >= max_results:
                break
            if new_this_page == 0:
                break

        return results[:max_results]

    def _parse_atom_entry(self, entry: ET.Element) -> Optional["Filing"]:
        """Parse a single Atom <entry> element into a Filing.

        Actual feed format (confirmed):
          <title>  8-K - COMPANY NAME (0000123456) (Filer)
          <link>   href=".../edgar/data/{cik}/{acc_no_dashes}/{acc_no}-index.htm"
          <updated>2026-03-02T16:58:52-05:00
          <summary> <b>Filed:</b> 2026-03-02 <b>AccNo:</b> ... Item N.NN: ...
          <id>     urn:tag:sec.gov,2008:accession-number=0000943374-26-000113
        """
        entry_id = entry.findtext(f"{{{_ATOM_NS}}}id", "")
        acc_match = re.search(r"accession-number=(\d{10}-\d{2}-\d{6})", entry_id)
        if not acc_match:
            return None
        accession = acc_match.group(1)

        link_el = entry.find(f"{{{_ATOM_NS}}}link")
        href = link_el.get("href", "") if link_el is not None else ""
        cik_match = re.search(r"/edgar/data/(\d+)/", href)
        cik = (cik_match.group(1).lstrip("0") or "0") if cik_match else "0"

        title = entry.findtext(f"{{{_ATOM_NS}}}title", "")
        t_match = re.match(r"^([^\s]+)\s+-\s+(.+?)\s+\(\d+\)\s+\(Filer\)", title)
        if t_match:
            form_type    = t_match.group(1)
            company_name = t_match.group(2).strip()
        else:
            form_type    = "8-K"
            company_name = title

        updated    = entry.findtext(f"{{{_ATOM_NS}}}updated", "")
        filed_date = updated[:10] if updated else ""
        filed_time = updated[11:19] if len(updated) >= 19 else ""

        summary_html = entry.findtext(f"{{{_ATOM_NS}}}summary", "")
        items = re.findall(r"(Item \d+\.\d+: [^\n<]+)", summary_html)
        description = "; ".join(i.strip() for i in items)

        return Filing(
            accession_number=accession,
            cik=cik,
            company_name=company_name,
            filed_date=filed_date,
            filed_time=filed_time,
            report_date="",
            form_type=form_type,
            description=description,
        )

    # ------------------------------------------------------------------
    # Primary document resolution
    # ------------------------------------------------------------------

    def resolve_primary_doc(self, filing: Filing) -> Optional[str]:
        """
        Ensure filing.primary_doc is populated. For Atom feed filings it won't
        be set, so look it up via the company's submissions JSON.
        Returns the filename, or None if unavailable.
        """
        if filing.primary_doc:
            return filing.primary_doc

        try:
            data = self._get_json(
                f"{BASE_SUBMISSIONS_URL}/CIK{filing.cik.zfill(10)}.json"
            )
            recent = data.get("filings", {}).get("recent", {})
            accessions   = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])
            for i, acc in enumerate(accessions):
                if acc == filing.accession_number:
                    doc = primary_docs[i] if i < len(primary_docs) else ""
                    filing.primary_doc = doc
                    return doc or None
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Filing document index
    # ------------------------------------------------------------------

    def get_documents(self, filing: Filing) -> list[dict]:
        """
        Fetch the document list for a filing and populate `filing.documents`.

        Each document dict has keys: name, type, description, size.
        """
        acc_no_dashes = filing.accession_number.replace("-", "")
        index_url = (
            f"{ARCHIVES_URL}/{filing.cik}/{acc_no_dashes}"
            f"/{filing.accession_number}-index.json"
        )
        try:
            data = self._get_json(index_url)
            docs = [
                {
                    "name": item.get("name", ""),
                    "type": item.get("type", ""),
                    "description": item.get("description", ""),
                    "size": item.get("size", ""),
                }
                for item in data.get("directory", {}).get("item", [])
            ]
            filing.documents = docs
            return docs
        except requests.HTTPError:
            return []
