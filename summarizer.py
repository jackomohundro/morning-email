"""
Two-stage Cerebras pipeline for 8-K analysis.

Stage 1 — extract_filing_data():
    Per-filing extraction. Pulls every material number, event, and fact.
    No opinion, no ranking — just completeness.

Stage 2 — run_editorial_pass():
    Single call over all Stage 1 extractions. Acts as an editor:
    drops noise, removes duplicates, ranks by importance, writes the
    final display content (headlines, detail lines, one-liners).
"""

import json

import requests

CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
DEFAULT_MODEL    = "qwen-3-235b-a22b-instruct-2507"


# ============================================================
# Stage 1 — per-filing extraction
# ============================================================

_S1_SYSTEM = """\
You are a financial data extraction engine. Extract every material number, \
fact, and named party from SEC filings. Be exhaustive with figures. \
No opinions. Respond with valid JSON only."""

_S1_STANDARD = """\
Extract all material information from this {form_type} filing.
Company: {company}  |  Filed: {filed_date}  |  Items: {items}

Respond ONLY with JSON:
{{
  "has_material_info": true,
  "event_type": "acquisition" | "divestiture" | "agreement" | "leadership_change" | \
"debt" | "dividend" | "buyback" | "legal" | "regulatory" | "guidance" | "other",
  "parties": ["other named companies, executives, regulators involved — with roles"],
  "amounts": ["every dollar figure, %, share count with full context"],
  "key_facts": ["one concrete fact per item — numbers required where present"],
  "one_sentence": "one tight sentence capturing the whole filing with key numbers"
}}

Set has_material_info=false if the filing contains only exhibits, signatures, \
routine administrative notices, or repetitive boilerplate with no new information.

Filing text:
{text}"""

_S1_EARNINGS = """\
Extract all financial results from this {form_type} earnings filing.
Company: {company}  |  Filed: {filed_date}  |  Items: {items}

FORMAT RULES — apply to every field:
- Dollar values: use human-readable shorthand — "$377.9M", "$1.2B", "$4.5T". Never raw integers.
- Percentages: "12.3%", "-8.1%"
- EPS: "$1.23", "-$3.29"
- YoY changes: "+12%" or "-8%"
- Ranges: "$127-130B"
- key_operational_metrics: SHORT items only (e.g. "MAU: 10M", "Units sold: 45K"). \
  Do NOT put full sentences here. Max 50 chars per item.
- key_facts: 2-4 brief sentences covering the most important non-financial context \
  (guidance rationale, key product milestones, strategic events). NOT a dump of all facts.
- drug_pipeline: pharma/biotech companies only — list each drug or biologic candidate \
  mentioned (including approved products). Include name, disease/condition treated (indication), \
  mechanism of action, clinical stage (Phase 1/2/3/Approved/NDA filed), and any revenue or \
  addressable market size data mentioned. Leave as [] for non-pharma/biotech companies.
- one_sentence: one tight sentence with the 2-3 most important numbers.

Respond ONLY with JSON — use null for any field absent from the filing:
{{
  "has_material_info": true,
  "event_type": "earnings",
  "period": null,
  "revenue": null,
  "revenue_yoy": null,
  "gross_margin": null,
  "operating_income": null,
  "operating_margin": null,
  "net_income": null,
  "eps_gaap": null,
  "eps_adj": null,
  "eps_yoy": null,
  "ebitda": null,
  "fcf": null,
  "vs_estimates": null,
  "guidance_next_q": null,
  "guidance_fy": null,
  "key_operational_metrics": [],
  "key_facts": [],
  "drug_pipeline": [],
  "one_sentence": null
}}

Filing text:
{text}"""


def _is_earnings(items: list[str], description: str) -> bool:
    return "2.02" in (" ".join(items) + " " + description)


def extract_filing_data(
    text: str,
    company: str,
    form_type: str,
    filed_date: str,
    items: list[str],
    description: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Stage 1: extract all material numbers and facts from a single filing.
    Returns a dict with has_material_info, event_type, and filing-specific fields.
    On error returns {"has_material_info": False, "error": str}.
    """
    earnings  = _is_earnings(items, description)
    template  = _S1_EARNINGS if earnings else _S1_STANDARD
    max_tok   = 2500 if earnings else 600

    prompt = template.format(
        form_type=form_type,
        company=company,
        filed_date=filed_date,
        items="; ".join(items) if items else "unspecified",
        text=text,
    )

    try:
        resp = requests.post(
            CEREBRAS_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _S1_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": max_tok,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        result = json.loads(raw)
        result["is_earnings"] = earnings
        return result
    except json.JSONDecodeError as exc:
        return {"has_material_info": False, "is_earnings": earnings,
                "error": f"JSON parse failed: {exc}"}
    except Exception as exc:
        return {"has_material_info": False, "is_earnings": earnings, "error": str(exc)}


# ============================================================
# Stage 2 — editorial pass over all extractions
# ============================================================

_S2_SYSTEM = """\
You are a senior analyst writing a morning briefing for hedge fund PMs and traders. \
You have deep knowledge of equity markets, corporate finance, and sector dynamics. \
Your job is not to recite facts — it is to provide judgment: what is genuinely surprising, \
what the numbers imply about the business, what investors should think about next. \
\
You will receive structured data extracted from SEC 8-K filings. \
Some entries include a 'web_context' field with web search excerpts — \
use these for consensus estimates, analyst reactions, beat/miss context, and market implications. \
Some entries include 'company_description' and/or 'company_descriptions' (for counterparties) — \
use these to identify what each company does on first mention. \
\
Write with authority and specificity. Avoid filler. Never pad a paragraph with numbers \
that belong in the table. Use the numbers to make an argument, not to fill space. \
Do not fabricate facts not present in the filing data, web_context, or company_descriptions. \
Respond with valid JSON only."""

_S2_PROMPT = """\
You have received structured extractions from today's SEC 8-K filings. \
Write the final briefing.

CURATION RULES:
1. DROP if has_material_info=false, or filing has no concrete numbers or named events.
2. DROP duplicates — if parent + subsidiary filed identical content, keep the most informative.
3. DROP entirely: delisting notices, NYSE/Nasdaq compliance deficiency letters, going-concern \
warnings, reverse stock splits done solely to meet listing requirements, routine debt \
refinancings or covenant amendments with no strategic change, boilerplate shelf registration \
updates, and any filing whose sole purpose is administrative compliance.
4. NOTABLE (must be genuinely market-moving): earnings with real financial figures, \
transformative M&A or divestitures, major commercial agreements with significant dollar figures, \
CEO/CFO changes at mkt_cap >= $1B, large buyback or dividend initiations/cuts, material \
litigation outcomes or regulatory actions that alter the business, strategic pivots, \
major clinical trial results or FDA actions.
5. SECONDARY: anything with at least one concrete number or named fact worth knowing but \
not notable. Routine refinancings that meaningfully lower cost of capital may appear here.
6. Order notable: earnings first by mkt_cap (largest first), then non-earnings by transaction \
size or estimated market impact. Order secondary by importance.

WRITING RULES FOR NOTABLE PARAGRAPHS:
- 3-5 sentences. The goal is not to summarize — it is to give the reader a view.
- Open by briefly identifying what the company does on first mention (use company_description \
if available): "Acme Corp, a specialty chemicals maker, ..."
- Lead with the hard numbers and what is SURPRISING. For earnings: state the actual vs. \
consensus directly — "EPS of $X vs. consensus $Y" or "missed consensus by $Z." \
For deals: state the price and what it implies. Facts first, always.
- ALWAYS include management guidance and forward-looking statements when present — \
these are among the most important things a reader wants. Guidance numbers (revenue range, \
EPS outlook, margin targets), management commentary on demand trends, and any strategic \
outlook must appear. Just attribute them: "management guided to X," "the company expects Y," \
"guidance of $Z implies..." — not stated as your own conclusion.
- STRICT ATTRIBUTION RULE: Never state a claim as fact unless it comes from the numbers \
themselves. If the company said something ("called out macro headwinds," "guided to X"), \
attribute it to the company. If an analyst said something, attribute it to analysts or \
web_context. Do not launder management spin into the briefing as your own conclusion. \
Bad: "macro conditions weighed on results." Good: "management cited macro headwinds." \
Bad: "the company is outperforming peers." Good: "EPS came in 40% above consensus."
- NO EDITORIAL ADJECTIVES unless directly supported by a number. Do not write "strong," \
"impressive," "robust," "solid," "leading," "well-positioned," or any similar modifier \
unless you follow it immediately with the specific number that justifies it. If you cannot \
point to a number, delete the adjective.
- NO UNSUBSTANTIATED COMPARISONS. Do not say a company is outperforming, gaining share, \
or doing better than peers unless the filing or web_context explicitly states this with data.
- Use web_context to provide beat/miss vs consensus, analyst reactions, and forward implications. \
If consensus was $1.20 EPS and actual was $1.05, say so plainly.
- Connect to the bigger picture when it's obvious and factual: known sector headwinds, \
macro backdrop, peer data from web_context. If you don't have data to support it, skip it.
- End with what to watch: the specific upcoming catalyst, risk, or open question the result \
raises — stated as a question or risk, not a prediction.
- Do NOT repeat numbers already in the financial table. Use one or two numbers to make a \
point; the table carries the rest.
- If you see a pattern across multiple filings (e.g. three retailers all missing on margins), \
note it in the most important entry.

WRITING RULES FOR SECONDARY:
- One tight sentence. Lead with the number or event, not the company name.

HARD RULE: Earnings from mkt_cap >= $1B with extracted financial figures are always NOTABLE.

Filings:
{entries_json}

Respond ONLY with JSON:
{{
  "notable": [
    {{
      "idx": <integer>,
      "paragraph": "3-5 sentence analyst-quality narrative with judgment, context, and forward view"
    }}
  ],
  "secondary": [
    {{
      "idx": <integer>,
      "one_liner": "one tight sentence leading with the key number or event"
    }}
  ]
}}"""


def enrich_drug_from_search(
    drug_name: str,
    company: str,
    excerpts: list[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Given web search excerpts about a drug, extract mechanism, stage, market data.
    Returns a dict with keys: mechanism, stage, market_or_revenue, indication.
    Values are None for anything not found in the excerpts.
    """
    if not excerpts:
        return {}
    text = "\n---\n".join(excerpts[:3])
    prompt = (
        f"Search results about {drug_name} ({company}):\n\n{text}\n\n"
        f"Extract ONLY information present in the above text:\n"
        f'{{"mechanism":"mechanism of action (e.g. \'thromboxane A2 receptor antagonist\')","stage":"development stage (Phase 1/2/3/Approved/NDA filed/Preclinical)","market_or_revenue":"market size or revenue data if mentioned","indication":"primary disease or condition"}}\n'
        f"Use null for any field not found in the text."
    )
    try:
        resp = requests.post(
            CEREBRAS_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": (
                        "Extract drug information from provided search results. "
                        "Respond with valid JSON only. "
                        "Do not fabricate anything not present in the provided text."
                    )},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception:
        return {}


def run_editorial_pass(
    entries: list[dict],
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Stage 2: single editorial LLM call over all Stage 1 extractions.

    entries: list of compact dicts, each with an "idx" field pointing back
             to the caller's filtered list.

    Returns {"notable": [...], "secondary": [...]} or raises on hard failure.
    Falls back to a trivial sort-by-has_material_info if JSON parse fails.
    """
    prompt = _S2_PROMPT.format(
        entries_json=json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    )

    resp = requests.post(
        CEREBRAS_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _S2_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        timeout=120,
    )
    resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Hard fallback: treat everything with material info as secondary
        return {
            "notable": [],
            "secondary": [
                {"idx": e["idx"], "one_liner": e.get("one_sentence", "")}
                for e in entries
                if e.get("has_material_info")
            ],
        }
