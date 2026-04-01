#!/usr/bin/env python3
"""
24-hour 8-K report — two-stage pipeline.

Stage 1: Per-filing extraction (Cerebras) — pulls every material number,
         event, and fact from each filing in parallel.

Stage 2: Single editorial pass (Cerebras) — takes all Stage 1 extractions,
         drops noise, removes duplicates, ranks by importance, writes the
         final display content.

The result is a high-density briefing optimized for a high-time-value reader:
numbers that matter are present, text is concise, empty filings are gone.

Usage:
    python report.py

Requires in .env or environment:
    CEREBRAS_API_KEY
    POLYGON_API_KEY
    SES_FROM_EMAIL
    SES_TO_EMAIL
    AWS_DEFAULT_REGION   (optional, defaults to us-east-1)
"""

import html as _html
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

from edgar import EdgarClient, ARCHIVES_URL
from extractor import extract_filing_text, extract_exhibit_text
from summarizer import extract_filing_data, run_editorial_pass, enrich_drug_from_search
from search import search_filing_context
import polygon as poly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filed_ts(filing) -> str:
    if filing.filed_time:
        return f"{filing.filed_date} {filing.filed_time} ET"
    return filing.filed_date


def _ticker_for(filing, client: EdgarClient) -> str:
    return getattr(client, "_cik_to_ticker", {}).get(filing.cik.zfill(10), "")


def _mkt_cap_float(tkr: str, market_data: dict) -> float:
    return (market_data.get(tkr) or {}).get("market_cap") or 0.0


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def _load_ticker_maps(client: EdgarClient) -> None:
    """Ensure ticker map + reverse CIK→ticker map are loaded."""
    if client._ticker_map is None:
        print("Loading ticker map...", file=sys.stderr)
        client._ticker_map = client._load_ticker_map()
    if not hasattr(client, "_cik_to_ticker"):
        client._cik_to_ticker = {  # type: ignore[attr-defined]
            cik.zfill(10): tkr for tkr, cik in client._ticker_map.items()
        }


def _fetch_market_data(pairs: list[tuple], client: EdgarClient, polygon_key: str) -> dict:
    needed = {
        _ticker_for(f, client)
        for f, _ in pairs
        if _ticker_for(f, client)
    }
    if not needed:
        return {}
    print(f"Fetching market data for {len(needed)} ticker(s)...", file=sys.stderr)
    out: dict = {}
    for tkr in needed:
        data = poly.get_market_data(tkr, polygon_key)
        if data:
            out[tkr] = data
    return out


# ---------------------------------------------------------------------------
# Pre-filter (code-level, before Stage 2)
# ---------------------------------------------------------------------------

def _dedup_by_cik(pairs: list[tuple]) -> list[tuple]:
    """Keep one filing per CIK — prefer earnings over non-earnings, then richer extraction."""
    seen: dict[str, int] = {}
    result: list[tuple] = []
    for filing, ext in pairs:
        cik = filing.cik
        if cik not in seen:
            seen[cik] = len(result)
            result.append((filing, ext))
        else:
            idx = seen[cik]
            _, ex0 = result[idx]
            if ext.get("is_earnings") and not ex0.get("is_earnings"):
                result[idx] = (filing, ext)
            elif ext.get("is_earnings") == ex0.get("is_earnings"):
                score_new = len(str(ext.get("key_facts", []))) + len(str(ext.get("amounts", [])))
                score_old = len(str(ex0.get("key_facts", []))) + len(str(ex0.get("amounts", [])))
                if score_new > score_old:
                    result[idx] = (filing, ext)
    return result


def _prefilter(pairs: list[tuple], client: EdgarClient, market_data: dict) -> list[tuple]:
    """
    Drop non-public entities (no SEC ticker).
    Earnings from >$1B companies: mark has_material_info=True so Stage 2 keeps them.
    """
    kept = []
    for filing, ext in pairs:
        tkr = _ticker_for(filing, client)
        if not tkr:
            continue   # non-public entity

        # Ensure earnings from large caps survive Stage 2 even if extractor was sparse
        cap = _mkt_cap_float(tkr, market_data)
        if ext.get("is_earnings") and cap >= 1_000_000_000:
            ext = dict(ext)
            ext["has_material_info"] = True

        kept.append((filing, ext))
    return kept


# ---------------------------------------------------------------------------
# Exhibit fetching — three-strategy cascade
# ---------------------------------------------------------------------------

def _exhibits_from_sgml(raw: bytes) -> list[str]:
    """
    Strategy 1: extract EX-99.x blocks embedded in an SGML filing bundle.
    Many 8-K submissions include exhibits inline in the same .txt file.
    No additional HTTP request needed — we already have the bytes.
    """
    try:
        text = raw.decode("utf-8", errors="replace")
        if "<TYPE>EX-99" not in text.upper():
            return []
        blocks = re.findall(
            r"<TYPE>EX-99[^\n]*\n.*?<TEXT>\n?(.*?)</TEXT>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        result = []
        for block in blocks[:3]:
            txt = extract_exhibit_text(block.encode("utf-8"))
            if txt:
                result.append(txt)
        return result
    except Exception:
        return []


def _exhibit_urls_from_html(raw: bytes, doc_url: str) -> list[str]:
    """
    Strategy 2: parse EX-99.x href links from the primary 8-K HTML cover page.
    Most modern 8-K filings have an exhibit table with clickable links.
    No additional HTTP request needed — we already have the bytes.
    """
    try:
        text = raw.decode("utf-8", errors="replace")
        links = re.findall(r'href=["\']([^"\'#?\s]+)["\']', text, re.IGNORECASE)
        base = doc_url.rsplit("/", 1)[0] + "/"
        seen: set[str] = set()
        result: list[str] = []
        for link in links:
            if not re.search(r"ex[\-_]?99|exhibit[\-_]?99", link, re.IGNORECASE):
                continue
            full_url = urljoin(base, link)
            if "sec.gov" in full_url and full_url not in seen:
                seen.add(full_url)
                result.append(full_url)
        return result[:3]
    except Exception:
        return []


def _exhibit_urls_from_index(filing, client: EdgarClient) -> list[str]:
    """
    Strategy 3: look up EX-99.x filenames from the EDGAR filing index JSON.
    Fallback when the primary doc contains no exhibit links.
    """
    try:
        docs = client.get_documents(filing)
        if not docs:
            return []
        acc  = filing.accession_number.replace("-", "")
        base = f"{ARCHIVES_URL}/{filing.cik}/{acc}"
        result = []
        for doc in docs:
            if re.match(r"EX-99", doc.get("type", "").strip().upper()):
                name = doc.get("name", "")
                if name:
                    result.append(f"{base}/{name}")
        return result[:3]
    except Exception:
        return []


def _fetch_exhibit_parts(raw: bytes, doc_url: str, filing, client: EdgarClient) -> list[str]:
    """
    Run the three strategies in order and return a list of clean exhibit texts.
    """
    # Strategy 1: embedded SGML
    parts = _exhibits_from_sgml(raw)
    if parts:
        return parts

    # Strategy 2: HTML links in cover page
    ex_urls = _exhibit_urls_from_html(raw, doc_url)

    # Strategy 3: index JSON fallback
    if not ex_urls:
        ex_urls = _exhibit_urls_from_index(filing, client)

    for url in ex_urls:
        try:
            eraw = client._get_bytes(url)
            txt  = extract_exhibit_text(eraw)
            if txt:
                parts.append(txt)
        except Exception:
            continue
        if len(parts) >= 3:
            break

    return parts


# ---------------------------------------------------------------------------
# Post-Stage-2 safeguard: ensure large-cap earnings are always notable
# ---------------------------------------------------------------------------

def _enforce_large_cap_earnings(
    editorial: dict,
    filtered: list[tuple],
    client: EdgarClient,
    market_data: dict,
) -> dict:
    """
    If any earnings filing from a company with market cap >= $1B is absent
    from notable, synthesise a notable entry from Stage 1 data and prepend it.
    This is a safety net for overly aggressive Stage 2 pruning.
    """
    notable_idxs = {item["idx"] for item in editorial.get("notable", [])}
    secondary_idxs = {item["idx"] for item in editorial.get("secondary", [])}

    injected = []
    for idx, (filing, ext) in enumerate(filtered):
        if not ext.get("is_earnings"):
            continue
        tkr = _ticker_for(filing, client)
        cap = _mkt_cap_float(tkr, market_data)
        if cap < 1_000_000_000:
            continue
        if idx in notable_idxs:
            continue  # already there

        # Only inject if Stage 1 actually extracted financial figures.
        # If the exhibit fetch failed and we have no data, don't show an empty shell.
        has_numbers = any(
            ext.get(k) and ext.get(k) not in ("null", None)
            for k in ("revenue", "eps_gaap", "eps_adj", "net_income", "ebitda",
                      "operating_income", "fcf", "vs_estimates",
                      "guidance_next_q", "guidance_fy")
        ) or bool([f for f in (ext.get("key_facts") or []) if f])
        if not has_numbers:
            continue

        # Build a paragraph from Stage 1 data
        sentence = ext.get("one_sentence") or ""
        facts     = [f for f in (ext.get("key_facts") or []) if f]
        paragraph = sentence
        if facts:
            paragraph = (paragraph + " " if paragraph else "") + " ".join(facts[:3])
        if not paragraph:
            paragraph = f"Earnings reported — {poly.fmt_mktcap(cap)} market cap."

        injected.append({
            "idx": idx,
            "paragraph": paragraph,
        })

        # Remove from secondary if it was there
        editorial["secondary"] = [
            s for s in editorial.get("secondary", []) if s["idx"] != idx
        ]

    if injected:
        editorial = dict(editorial)
        editorial["notable"] = injected + list(editorial.get("notable", []))

    return editorial


# ---------------------------------------------------------------------------
# Build Stage 2 input
# ---------------------------------------------------------------------------

def _compact_entry(idx: int, filing, ext: dict, tkr: str, mkt: dict) -> dict:
    """
    Compact representation of one filing for the Stage 2 editorial prompt.
    Omits null / empty fields to save tokens.
    """
    e: dict = {
        "idx":      idx,
        "company":  filing.company_name,
        "form":     filing.form_type,
        "filed":    _filed_ts(filing),
        "items":    filing.description or "",
        "has_material_info": ext.get("has_material_info", True),
        "is_earnings": ext.get("is_earnings", False),
    }
    if tkr:
        e["ticker"] = tkr
    cap = mkt.get("market_cap")
    if cap:
        e["mkt_cap"] = poly.fmt_mktcap(cap)
    price = mkt.get("price")
    if price is not None:
        e["price"] = poly.fmt_price(price, mkt.get("change_pct"))
    desc = mkt.get("description")
    if desc:
        # Truncate to first two sentences to save tokens
        sentences = re.split(r'(?<=[.!?])\s+', desc.strip())
        e["company_description"] = " ".join(sentences[:2])

    if ext.get("is_earnings"):
        # Build a compact financials string
        fin = []
        for label, key in [
            ("Rev",      "revenue"),
            ("Rev YoY",  "revenue_yoy"),
            ("GM",       "gross_margin"),
            ("OpInc",    "operating_income"),
            ("OpMgn",    "operating_margin"),
            ("NI",       "net_income"),
            ("EBITDA",   "ebitda"),
            ("FCF",      "fcf"),
            ("EPS GAAP", "eps_gaap"),
            ("EPS adj",  "eps_adj"),
            ("EPS YoY",  "eps_yoy"),
        ]:
            v = ext.get(key)
            if v and v != "null":
                fin.append(f"{label}: {v}")
        if fin:
            e["financials"] = " | ".join(fin)
        for key in ("period", "vs_estimates", "guidance_next_q", "guidance_fy",
                    "key_operational_metrics"):
            v = ext.get(key)
            if v and v != "null" and v != []:
                e[key] = v
        # Pass drug pipeline so Stage 2 can reference specific assets in biotech paragraphs.
        # Compact format: omit empty fields to save tokens.
        pipeline = ext.get("drug_pipeline") or []
        if isinstance(pipeline, list) and pipeline:
            compact_pipeline = []
            for d in pipeline:
                if not isinstance(d, dict) or not d.get("name"):
                    continue
                entry = {"name": d["name"]}
                for f in ("indication", "mechanism", "stage", "market_or_revenue"):
                    if d.get(f):
                        entry[f] = d[f]
                compact_pipeline.append(entry)
            if compact_pipeline:
                e["drug_pipeline"] = compact_pipeline
    else:
        for key in ("event_type", "parties", "amounts"):
            v = ext.get(key)
            if v and v != "null" and v != []:
                e[key] = v

    # Always include key_facts and one_sentence if present
    for key in ("key_facts", "one_sentence"):
        v = ext.get(key)
        if v and v != "null" and v != []:
            e[key] = v

    web = ext.get("web_context")
    if web and isinstance(web, list):
        e["web_context"] = [str(x) for x in web[:3]]

    co_descs = ext.get("company_descriptions")
    if co_descs and isinstance(co_descs, dict):
        e["company_descriptions"] = co_descs

    return e


# ---------------------------------------------------------------------------
# Earnings table helpers
# ---------------------------------------------------------------------------

_EARNINGS_FIELDS = [
    ("Revenue",       "revenue"),
    ("Revenue YoY",   "revenue_yoy"),
    ("Gross margin",  "gross_margin"),
    ("Op income",     "operating_income"),
    ("Op margin",     "operating_margin"),
    ("Net income",    "net_income"),
    ("EBITDA",        "ebitda"),
    ("FCF",           "fcf"),
    ("EPS adj",       "eps_adj"),
    ("EPS GAAP",      "eps_gaap"),
    ("EPS YoY",       "eps_yoy"),
    ("vs estimates",  "vs_estimates"),
    ("Guide Q",       "guidance_next_q"),
    ("Guide FY",      "guidance_fy"),
]


def _earnings_rows(ext: dict) -> list[tuple[str, str]]:
    """Return (label, value) pairs for structured financial fields only.
    key_operational_metrics are prose context — they belong in the paragraph, not the table.
    """
    rows = []
    for label, key in _EARNINGS_FIELDS:
        v = ext.get(key)
        if v and v not in ("null", None):
            rows.append((label, str(v)))
    # Short operational metrics (e.g. "MAU: 10M") are table-appropriate; long sentences are not.
    ops = ext.get("key_operational_metrics")
    if ops and ops not in ("null", None, []):
        items = ops if isinstance(ops, list) else [str(ops)]
        for item in items:
            if item and len(str(item)) <= 60:
                rows.append(("", str(item)))
    return rows


# ---------------------------------------------------------------------------
# Drug pipeline web enrichment
# ---------------------------------------------------------------------------

def _enrich_drug_pipeline(
    filtered: list[tuple],
    parallel_key: str,
    api_key: str,
) -> None:
    """
    For each drug in an earnings filing's pipeline that is missing mechanism/stage/market,
    do a web search and extract structured data via Cerebras. Modifies dicts in-place.
    """
    for filing, ext in filtered:
        if not ext.get("is_earnings"):
            continue
        pipeline = ext.get("drug_pipeline") or []
        if not isinstance(pipeline, list):
            continue
        for drug in pipeline:
            if not isinstance(drug, dict) or not drug.get("name"):
                continue
            # Skip if all key fields are already populated
            if drug.get("mechanism") and drug.get("stage") and drug.get("market_or_revenue"):
                continue
            query = (
                f"{drug['name']} {filing.company_name} "
                f"mechanism of action clinical stage indication market size"
            )
            excerpts = search_filing_context(query, parallel_key, max_results=2)
            if not excerpts:
                continue
            enriched = enrich_drug_from_search(drug["name"], filing.company_name, excerpts, api_key)
            for field in ("mechanism", "stage", "market_or_revenue", "indication"):
                if not drug.get(field) and enriched.get(field):
                    drug[field] = enriched[field]
            print(
                f"  +DRUG [{drug['name']}] {filing.company_name[:35]}",
                file=__import__("sys").stderr,
            )


# ---------------------------------------------------------------------------
# Notable ticker list + drug pipeline helpers
# ---------------------------------------------------------------------------

def _notable_tickers(notable: list[dict], filtered: list[tuple], client: EdgarClient) -> list[str]:
    seen: set[str] = set()
    tickers: list[str] = []
    for item in notable:
        filing, _ = filtered[item["idx"]]
        tkr = _ticker_for(filing, client)
        if tkr and tkr not in seen:
            seen.add(tkr)
            tickers.append(tkr)
    return tickers


def _drug_table_rows(notable: list[dict], filtered: list[tuple], client: EdgarClient) -> list[dict]:
    """Collect drug pipeline rows from notable earnings filings, merging multiple indications per drug."""
    # key: (company_str, drug_name) -> merged row dict
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []

    for item in notable:
        filing, ext = filtered[item["idx"]]
        if not ext.get("is_earnings"):
            continue
        pipeline = ext.get("drug_pipeline") or []
        if not isinstance(pipeline, list):
            continue
        tkr = _ticker_for(filing, client)
        co = filing.company_name + (f" ({tkr})" if tkr else "")
        for drug in pipeline:
            if not isinstance(drug, dict) or not drug.get("name"):
                continue
            key = (co, drug.get("name") or "")
            indication = drug.get("indication") or ""
            if key not in merged:
                order.append(key)
                merged[key] = {
                    "company":           co,
                    "name":              drug.get("name") or "",
                    "indications":       [indication] if indication else [],
                    "mechanism":         drug.get("mechanism") or "",
                    "stage":             drug.get("stage") or "",
                    "market_or_revenue": drug.get("market_or_revenue") or "",
                }
            else:
                row = merged[key]
                if indication and indication not in row["indications"]:
                    row["indications"].append(indication)
                # Fill in missing fields from later entries
                for f in ("mechanism", "stage", "market_or_revenue"):
                    if not row[f] and drug.get(f):
                        row[f] = drug[f]

    rows = []
    for key in order:
        row = merged[key]
        rows.append({
            "company":           row["company"],
            "name":              row["name"],
            "indication":        "; ".join(row["indications"]),
            "mechanism":         row["mechanism"],
            "stage":             row["stage"],
            "market_or_revenue": row["market_or_revenue"],
        })
    return rows


def _print_drug_table(rows: list[dict]) -> None:
    if not rows:
        return
    import textwrap
    print()
    print("▌ BIOTECH PIPELINE  (earnings)")
    print()
    last_co = None
    for row in rows:
        if row["company"] != last_co:
            print(f"  {row['company']}")
            last_co = row["company"]
        stage = f"[{row['stage']}]" if row["stage"] else ""
        ind_mech = row["indication"]
        if row["mechanism"]:
            ind_mech += f"  ({row['mechanism']})" if ind_mech else row["mechanism"]
        line = f"    {row['name']:<16} {stage:<14} {ind_mech}"
        print(line[:_W])
        if row["market_or_revenue"]:
            print(f"    {'':16} {row['market_or_revenue']}"[:_W])
    print()


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

_W = 72


def _print_report(
    filtered: list[tuple],
    editorial: dict,
    client: EdgarClient,
    market_data: dict,
    today: str,
) -> None:
    import textwrap

    notable   = editorial.get("notable", [])
    secondary = editorial.get("secondary", [])
    n_total   = len(filtered)

    print("=" * _W)
    print(f"  24HR Report - {today}  |  {len(notable)} notable  |  {n_total} shown")
    print("=" * _W)

    tickers = _notable_tickers(notable, filtered, client)
    if tickers:
        print()
        print(f"  NOTABLE: {', '.join(tickers)}")

    drug_rows = _drug_table_rows(notable, filtered, client)
    _print_drug_table(drug_rows)

    if notable:
        print()
        print("▌ NOTABLE")
        print()
        for item in notable:
            print("-" * _W)
            filing, ext = filtered[item["idx"]]
            tkr = _ticker_for(filing, client)
            mkt = market_data.get(tkr, {})
            cap = poly.fmt_mktcap(mkt.get("market_cap"))

            # Line 1: name (ticker)  #earnings
            name_line = filing.company_name + (f" ({tkr})" if tkr else "")
            if ext.get("is_earnings"):
                name_line += "  #earnings"
            print(f"  {name_line}")

            # Line 2: market cap · datetime
            meta = "  ·  ".join(filter(None, [cap, _filed_ts(filing)]))
            if meta:
                print(f"  {meta}")
            print()

            # Paragraph
            para = item.get("paragraph", "")
            if para:
                for line in textwrap.wrap(para, width=_W - 4):
                    print(f"  {line}")
                print()


    if secondary:
        print()
        print("▌ SECONDARY")
        print()
        for item in secondary:
            filing, ext = filtered[item["idx"]]
            tkr = _ticker_for(filing, client)
            mkt = market_data.get(tkr, {})
            cap = poly.fmt_mktcap(mkt.get("market_cap"))
            name_line = filing.company_name + (f" ({tkr})" if tkr else "")
            if ext.get("is_earnings"):
                name_line += "  #earnings"
            meta = "  ·  ".join(filter(None, [cap, _filed_ts(filing)]))
            print(f"  {name_line}")
            if meta:
                print(f"  {meta}")
            print(f"  {item.get('one_liner', '')}")
            print()

    print("=" * _W)
    print(f"  {len(notable)} notable  ·  {len(secondary)} secondary  ·  {n_total} total shown")


# ---------------------------------------------------------------------------
# HTML email
# ---------------------------------------------------------------------------

def _e(s) -> str:
    return _html.escape(str(s)) if s else ""


def _earnings_table_html(ext: dict) -> str:
    rows = _earnings_rows(ext)
    if not rows:
        return ""
    cells = ""
    for i in range(0, len(rows), 2):
        l_label, l_val = rows[i]
        r_label, r_val = rows[i + 1] if i + 1 < len(rows) else ("", "")
        cells += (
            f'<tr>'
            f'<td style="color:#6b7280;padding:3px 10px 3px 0;white-space:nowrap;">{_e(l_label)}</td>'
            f'<td style="font-weight:500;padding:3px 20px 3px 0;">{_e(l_val)}</td>'
            f'<td style="color:#6b7280;padding:3px 10px 3px 0;white-space:nowrap;">{_e(r_label)}</td>'
            f'<td style="font-weight:500;padding:3px 0;">{_e(r_val)}</td>'
            f'</tr>'
        )
    return (
        '<table style="border-collapse:collapse;font-size:12px;color:#374151;'
        'margin:8px 0 6px;width:100%;">'
        + cells + "</table>"
    )


def _drug_pipeline_html(rows: list[dict]) -> str:
    if not rows:
        return ""
    header_cells = "".join(
        f'<th style="text-align:left;padding:5px 10px 5px 0;color:#6b7280;'
        f'font-weight:600;font-size:11px;white-space:nowrap;">{h}</th>'
        for h in ["Company", "Drug / Candidate", "Indication", "Mechanism", "Stage", "Market / Revenue"]
    )
    body_rows = ""
    for row in rows:
        cols = [row["company"], row["name"], row["indication"],
                row["mechanism"], row["stage"], row["market_or_revenue"]]
        cells = "".join(
            f'<td style="padding:5px 10px 5px 0;font-size:12px;color:#374151;'
            f'vertical-align:top;">{_e(c)}</td>'
            for c in cols
        )
        body_rows += f"<tr>{cells}</tr>"
    return (
        '<div style="font-size:13px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;'
        'color:#111827;border-left:3px solid #111827;padding-left:8px;'
        'margin-top:24px;margin-bottom:10px;">Biotech Pipeline</div>'
        '<div style="overflow-x:auto;">'
        '<table style="border-collapse:collapse;width:100%;font-size:12px;">'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{body_rows}</tbody>'
        '</table></div>'
    )


def _build_html_email(
    filtered: list[tuple],
    editorial: dict,
    since: str,
    client: EdgarClient,
    market_data: dict,
) -> str:
    notable   = editorial.get("notable", [])
    secondary = editorial.get("secondary", [])
    today     = datetime.now(timezone.utc).strftime("%Y/%m/%d")

    # ── Notable blocks ────────────────────────────────────────────────────
    notable_html = ""
    if not notable:
        notable_html = '<p style="color:#6b7280;font-size:14px;margin:0;">No notable filings today.</p>'

    for item in notable:
        filing, ext = filtered[item["idx"]]
        tkr  = _ticker_for(filing, client)
        mkt  = market_data.get(tkr, {})
        cap  = poly.fmt_mktcap(mkt.get("market_cap"))
        ts   = _e(_filed_ts(filing))
        name = _e(filing.company_name)

        earn_tag = (
            ' <span style="font-size:10px;font-weight:700;color:#2563eb;'
            'background:#eff6ff;padding:1px 5px;border-radius:3px;vertical-align:middle;">'
            '#earnings</span>'
        ) if ext.get("is_earnings") else ""

        ticker_span = (
            f' <span style="font-weight:400;color:#6b7280;">({_e(tkr)})</span>'
        ) if tkr else ""

        meta_parts = "  &nbsp;·&nbsp;  ".join(filter(None, [_e(cap), ts]))

        para      = _e(item.get("paragraph", ""))
        earn_tbl  = _earnings_table_html(ext)

        notable_html += f"""
<div style="margin-bottom:24px;padding-bottom:20px;border-bottom:1px solid #e5e7eb;">
  <div style="font-size:15px;font-weight:700;margin-bottom:2px;">{name}{ticker_span}{earn_tag}</div>
  <div style="font-size:12px;color:#9ca3af;margin-bottom:10px;">{meta_parts}</div>
  <div style="font-size:13px;color:#374151;line-height:1.6;margin-bottom:6px;">{para}</div>
  {earn_tbl}
  <a href="{filing.index_url}" style="font-size:11px;color:#2563eb;text-decoration:none;">SEC filing →</a>
</div>"""

    # ── Secondary rows ────────────────────────────────────────────────────
    secondary_html = ""
    for item in secondary:
        filing, ext = filtered[item["idx"]]
        tkr  = _ticker_for(filing, client)
        mkt  = market_data.get(tkr, {})
        cap  = poly.fmt_mktcap(mkt.get("market_cap"))
        name = _e(filing.company_name)
        ts   = _e(_filed_ts(filing))
        liner = _e(item.get("one_liner", ""))

        earn_tag = (
            ' <span style="font-size:10px;font-weight:700;color:#2563eb;'
            'background:#eff6ff;padding:1px 4px;border-radius:3px;vertical-align:middle;">'
            '#earnings</span>'
        ) if ext.get("is_earnings") else ""

        ticker_span = (
            f' <span style="font-weight:400;color:#6b7280;">({_e(tkr)})</span>'
        ) if tkr else ""

        meta_parts = "  &nbsp;·&nbsp;  ".join(filter(None, [_e(cap), ts]))

        secondary_html += f"""
<div style="padding:9px 0;border-bottom:1px solid #f3f4f6;">
  <div style="font-size:13px;font-weight:600;color:#111827;">{name}{ticker_span}{earn_tag}</div>
  <div style="font-size:11px;color:#9ca3af;margin:2px 0 4px;">{meta_parts}</div>
  <div style="font-size:13px;color:#374151;">{liner}</div>
</div>"""

    sec_section = ""
    if secondary_html:
        sec_section = f"""
  <div style="font-size:13px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
              color:#111827;border-left:3px solid #111827;padding-left:8px;
              margin-top:32px;margin-bottom:12px;">Secondary</div>
  {secondary_html}"""

    n_total = len(filtered)
    ticker_list = _notable_tickers(notable, filtered, client)
    ticker_html = ""
    if ticker_list:
        ticker_html = (
            '<div style="font-size:12px;color:#374151;margin-top:6px;">'
            f'<strong>Notable:</strong> {", ".join(_e(t) for t in ticker_list)}</div>'
        )
    drug_rows = _drug_table_rows(notable, filtered, client)
    drug_html = _drug_pipeline_html(drug_rows)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
             background:#fff;margin:0;padding:8px 4px;color:#111827;">
<div style="max-width:640px;margin:0 auto;padding:0 4px;">

  <div style="border-bottom:2px solid #111827;padding:12px 0 10px;margin-bottom:20px;">
    <span style="font-size:17px;font-weight:700;">24HR Report &ndash; {today}</span>
    <div style="font-size:12px;color:#6b7280;margin-top:3px;">
      <strong style="color:#111827;">{len(notable)} notable</strong> &nbsp;·&nbsp;
      {len(secondary)} secondary &nbsp;·&nbsp; {n_total} total &nbsp;·&nbsp; last 24 h since {since}
    </div>
    {ticker_html}
  </div>

  {drug_html}

  <div style="font-size:13px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
              color:#111827;border-left:3px solid #111827;padding-left:8px;
              margin-bottom:16px;{' margin-top:24px;' if drug_html else ''}">Notable</div>
  {notable_html}
  {sec_section}

  <div style="font-size:11px;color:#d1d5db;text-align:center;margin-top:20px;
              padding-top:12px;border-top:1px solid #f3f4f6;">
    sec-pull · SEC EDGAR · {today}
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# SES sender
# ---------------------------------------------------------------------------

def _send_ses_email(subject: str, html_body: str, text_body: str) -> None:
    import boto3
    from_addr = os.environ.get("SES_FROM_EMAIL", "")
    to_addr   = os.environ.get("SES_TO_EMAIL", "")
    region    = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    if not from_addr or not to_addr:
        print("Warning: SES_FROM_EMAIL or SES_TO_EMAIL not set — skipping email.", file=sys.stderr)
        return
    ses = boto3.client("ses", region_name=region)
    ses.send_email(
        Source=from_addr,
        Destination={"ToAddresses": [to_addr]},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Html": {"Data": html_body, "Charset": "UTF-8"},
                "Text": {"Data": text_body, "Charset": "UTF-8"},
            },
        },
    )
    print(f"Email sent → {to_addr}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500,
                        help="Max filings to fetch (default: 500).")
    args = parser.parse_args()

    api_key     = os.environ.get("CEREBRAS_API_KEY")
    polygon_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Error: CEREBRAS_API_KEY not set", file=sys.stderr)
        return 1
    if not polygon_key:
        print("Warning: POLYGON_API_KEY not set — market data skipped.", file=sys.stderr)

    since  = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d")
    client = EdgarClient(user_agent="sec-pull/1.0 (contact@example.com)")

    # ── Fetch filings ──────────────────────────────────────────────────────
    print(f"Fetching 8-Ks since {since} ...", file=sys.stderr)
    filings = client.get_recent_filings(since_date=since, max_results=args.limit)
    print(f"Found {len(filings)}. Running Stage 1 extraction...", file=sys.stderr)

    # ── Stage 1: per-filing extraction ────────────────────────────────────
    raw_pairs: list[tuple] = []

    for i, filing in enumerate(filings, 1):
        if not filing.primary_doc:
            client.resolve_primary_doc(filing)
        url = filing.primary_document_url
        if not url:
            print(f"[{i:3}/{len(filings)}] SKIP  {filing.company_name}", file=sys.stderr)
            continue

        try:
            raw  = client._get_bytes(url)
            text = extract_filing_text(raw)

            # Fetch EX-99.x exhibits — the primary 8-K is often a cover page;
            # the real numbers are in EX-99.1. Three-strategy cascade:
            # SGML-embedded → HTML link parse → index JSON lookup.
            ex_parts     = _fetch_exhibit_parts(raw, url, filing, client)
            exhibit_text = "\n\n--- EXHIBIT ---\n\n".join(ex_parts)
            if exhibit_text:
                text = text + "\n\n--- EXHIBIT 99 ---\n\n" + exhibit_text

            items = [s.strip() for s in filing.description.split(";") if s.strip()]
            ext = extract_filing_data(
                text=text,
                company=filing.company_name,
                form_type=filing.form_type,
                filed_date=filing.filed_date,
                items=items,
                description=filing.description,
                api_key=api_key,
            )
        except requests.HTTPError as exc:
            print(f"[{i:3}/{len(filings)}] HTTP ERR  {filing.company_name}: {exc}", file=sys.stderr)
            continue
        except Exception as exc:
            print(f"[{i:3}/{len(filings)}] ERR  {filing.company_name}: {exc}", file=sys.stderr)
            continue

        material = ext.get("has_material_info", True)
        earn_tag = " [EARNINGS]" if ext.get("is_earnings") else ""
        ex_tag   = " +EX" if exhibit_text else ""
        flag     = "✓" if material else "–"
        print(f"[{i:3}/{len(filings)}] {flag} {filing.company_name[:42]:<42}{earn_tag}{ex_tag}", file=sys.stderr)
        raw_pairs.append((filing, ext))

    # ── Market data ────────────────────────────────────────────────────────
    _load_ticker_maps(client)
    market_data: dict = {}
    if polygon_key:
        market_data = _fetch_market_data(raw_pairs, client, polygon_key)

    # ── Pre-filter (non-public entities) ──────────────────────────────────
    filtered = _prefilter(raw_pairs, client, market_data)
    pre_count = len(filtered)
    filtered = _dedup_by_cik(filtered)
    print(f"\nStage 1 complete: {len(filtered)} filings after pre-filter and dedup "
          f"({len(raw_pairs) - pre_count} non-public, {pre_count - len(filtered)} dupes dropped).", file=sys.stderr)

    if not filtered:
        print("Nothing to report.", file=sys.stderr)
        return 0

    # ── Web search enrichment ──────────────────────────────────────────────
    # Only search for two classes of filing:
    #   1. Earnings — look up consensus EPS/revenue estimates and analyst reactions.
    #   2. High-signal non-earnings — M&A, divestitures, major legal/regulatory events.
    # Skip everything else to avoid wasting quota on noise.
    _HIGH_SIGNAL_EVENTS = {"acquisition", "divestiture", "legal", "regulatory"}

    parallel_key = os.environ.get("PARALLEL_API_KEY")
    if parallel_key:
        search_targets = []
        for filing, ext in filtered:
            if not ext.get("has_material_info", True):
                continue
            tkr = _ticker_for(filing, client)
            cap = _mkt_cap_float(tkr, market_data)
            if ext.get("is_earnings") and cap >= 500_000_000:
                # Earnings only for mid-cap+ — small-cap consensus data is sparse anyway
                search_targets.append((filing, ext, "earnings"))
            elif ext.get("event_type") in _HIGH_SIGNAL_EVENTS and cap >= 1_000_000_000:
                # High-signal events only at large-cap companies
                search_targets.append((filing, ext, "event"))

        print(f"Web search: {len(search_targets)} target(s) "
              f"({sum(1 for *_, t in search_targets if t == 'earnings')} earnings, "
              f"{sum(1 for *_, t in search_targets if t == 'event')} events)...", file=sys.stderr)

        for filing, ext, kind in search_targets:
            tkr = _ticker_for(filing, client)
            web: list[str] = []

            if kind == "earnings":
                period = ext.get("period") or filing.filed_date[:7]
                objective = (
                    f"{filing.company_name} ({tkr}) {period} earnings results: "
                    f"consensus EPS estimate vs actual EPS, consensus revenue estimate vs actual, "
                    f"analyst reactions, beat or miss, guidance vs consensus."
                )
                excerpts = search_filing_context(objective, parallel_key, max_results=3)
                web.extend(excerpts)

            else:
                # Main event search
                objective = (
                    f"{filing.company_name} ({tkr}) {ext.get('event_type', 'corporate event')}: "
                    f"{ext.get('one_sentence') or filing.description}. "
                    f"Analyst reactions and market implications."
                )
                excerpts = search_filing_context(objective, parallel_key, max_results=2)
                web.extend(excerpts)

                # Description search only for named counterparties (filer is covered by Polygon).
                co_descs: dict[str, str] = {}
                for party in (ext.get("parties") or [])[:3]:
                    name = party.split("(")[0].strip().rstrip(",")
                    if name and name.lower() != filing.company_name.lower():
                        desc = search_filing_context(
                            f"What does {name} do? One sentence business description.",
                            parallel_key,
                            max_results=1,
                        )
                        if desc:
                            co_descs[name] = desc[0]
                        break  # one counterparty is enough
                    # keep looping if this party was the filer itself

                if co_descs:
                    ext["company_descriptions"] = co_descs

            if web:
                ext["web_context"] = web
            if ext.get("web_context") or ext.get("company_descriptions"):
                print(f"  +WEB [{kind}] {filing.company_name[:40]}", file=sys.stderr)
    else:
        print("PARALLEL_API_KEY not set — skipping web enrichment.", file=sys.stderr)

    # ── Drug pipeline enrichment ───────────────────────────────────────────
    if parallel_key:
        _enrich_drug_pipeline(filtered, parallel_key, api_key)

    # ── Stage 2: editorial pass ───────────────────────────────────────────
    # Build compact entries for every filtered filing (idx → filtered[idx])
    entries = [
        _compact_entry(
            idx, filing, ext,
            _ticker_for(filing, client),
            market_data.get(_ticker_for(filing, client), {}),
        )
        for idx, (filing, ext) in enumerate(filtered)
    ]

    # Only send material entries to Stage 2 to stay within context limits.
    # Non-material entries are dropped (Stage 2 rule #1 anyway).
    s2_entries = [e for e in entries if e.get("has_material_info", True)]
    print(f"Running Stage 2 editorial pass over {len(s2_entries)} material filings...", file=sys.stderr)

    _BATCH = 60   # stay well within ~8K-token Cerebras input limit per call
    try:
        if len(s2_entries) <= _BATCH:
            editorial = run_editorial_pass(s2_entries, api_key)
        else:
            # Batch Stage 2: run multiple calls, then merge results
            editorial = {"notable": [], "secondary": []}
            for start in range(0, len(s2_entries), _BATCH):
                batch = s2_entries[start:start + _BATCH]
                result = run_editorial_pass(batch, api_key)
                editorial["notable"].extend(result.get("notable", []))
                editorial["secondary"].extend(result.get("secondary", []))
            print(f"  (batched {-(-len(s2_entries) // _BATCH)} calls)", file=sys.stderr)
    except Exception as exc:
        print(f"Stage 2 error: {exc} — using fallback output.", file=sys.stderr)
        # Fallback: show all material filings as secondary
        editorial = {
            "notable": [],
            "secondary": [
                {"idx": i, "one_liner": ext.get("one_sentence", "")}
                for i, (_, ext) in enumerate(filtered)
                if ext.get("has_material_info")
            ],
        }

    # Safeguard: large-cap earnings must always be in notable
    editorial = _enforce_large_cap_earnings(editorial, filtered, client, market_data)

    n_notable   = len(editorial.get("notable", []))
    n_secondary = len(editorial.get("secondary", []))
    print(f"Stage 2 complete: {n_notable} notable, {n_secondary} secondary.", file=sys.stderr)

    # ── Terminal output ────────────────────────────────────────────────────
    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    _print_report(filtered, editorial, client, market_data, today)

    # ── Build & send email ─────────────────────────────────────────────────
    subject   = f"24HR Report - {today} — {n_notable} notable"
    html_body = _build_html_email(filtered, editorial, since, client, market_data)

    # Plain-text fallback
    lines = [f"24HR Report - {today} | {n_notable} notable | {n_secondary} secondary\n"]
    for item in editorial.get("notable", []):
        filing, ext = filtered[item["idx"]]
        tkr = _ticker_for(filing, client)
        mkt = market_data.get(tkr, {})
        cap = poly.fmt_mktcap(mkt.get("market_cap"))
        lines.append(f"\n{'─'*60}")
        name_line = filing.company_name + (f" ({tkr})" if tkr else "")
        if ext.get("is_earnings"):
            name_line += "  #earnings"
        lines.append(name_line)
        meta = "  ·  ".join(filter(None, [cap, _filed_ts(filing)]))
        if meta:
            lines.append(meta)
        lines.append(item.get("paragraph", ""))
        lines.append(filing.index_url)

    if editorial.get("secondary"):
        lines.append(f"\n{'─'*60}\nSECONDARY\n{'─'*60}")
        for item in editorial["secondary"]:
            filing, ext = filtered[item["idx"]]
            tkr = _ticker_for(filing, client)
            mkt = market_data.get(tkr, {})
            cap = poly.fmt_mktcap(mkt.get("market_cap"))
            name_line = filing.company_name + (f" ({tkr})" if tkr else "")
            if ext.get("is_earnings"):
                name_line += "  #earnings"
            meta = "  ·  ".join(filter(None, [cap, _filed_ts(filing)]))
            lines.append(f"\n{name_line}")
            if meta:
                lines.append(meta)
            lines.append(f"  {item.get('one_liner', '')}")

    text_body = "\n".join(lines)

    try:
        _send_ses_email(subject, html_body, text_body)
    except Exception as exc:
        print(f"Email error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
