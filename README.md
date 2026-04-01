# SEC 8-K Morning Briefing Pipeline

A two-stage LLM pipeline that monitors the SEC EDGAR live feed, extracts structured data from 8-K filings, enriches it with real-time market data and web search, and produces an analyst-quality morning briefing delivered by email.

Runs daily on a Raspberry Pi. No human in the loop.

---

## What it does

Every morning, the pipeline:

1. **Ingests** all 8-K filings from the past 24 hours via the SEC EDGAR Atom feed (~200–400 filings on a typical day)
2. **Filters** to publicly traded companies only (requires a known SEC ticker)
3. **Fetches and cleans** the primary filing document, with a three-strategy cascade for finding EX-99.x press release exhibits
4. **Stage 1 (extraction):** runs each filing through Cerebras inference — pulls every material number, named party, and fact into structured JSON. Earnings filings get a separate prompt and schema optimised for financial results
5. **Enriches** with Polygon.io market data (price, daily change %, market cap) and targeted Parallel AI web search (consensus estimates, analyst reactions, counterparty descriptions)
6. **Stage 2 (editorial):** single LLM call over all Stage 1 outputs — curates, deduplicates, ranks by market impact, and writes analyst-quality paragraphs
7. **Delivers** an HTML email with a two-tier structure: notable filings (3–5 sentence narratives) and secondary filings (one-liners)

---

## Pipeline architecture

```
SEC EDGAR Atom feed
        │
        ▼
  Fetch & paginate                          edgar.py
  (up to 800 candidates, 24h window)
        │
        ▼
  Resolve primary doc + exhibits            report.py
  (3-strategy cascade: SGML inline →
   HTML link parse → index JSON fallback)
        │
        ▼
  Clean text extraction                     extractor.py
  (HTML strip, boilerplate removal,
   Item section isolation)
        │
        ▼
  Stage 1: per-filing extraction            summarizer.py
  Cerebras LLM, two prompt schemas:
  ┌─────────────────┬──────────────────┐
  │ earnings (2.02) │ non-earnings     │
  │ revenue, EPS,   │ event_type,      │
  │ margins, FCF,   │ parties,         │
  │ guidance,       │ amounts,         │
  │ drug pipeline   │ key_facts        │
  └─────────────────┴──────────────────┘
        │
        ▼
  Market data enrichment                    polygon.py
  (price, change%, market cap via Polygon)
        │
        ▼
  Web search enrichment                     search.py
  (Parallel AI: consensus vs actual,
   analyst reactions, counterparty descs)
  — earnings ≥ $500M market cap only
  — M&A/legal/regulatory ≥ $1B only
        │
        ▼
  Pre-filter + dedup                        report.py
  (drop non-public, dedup by CIK,
   enforce large-cap earnings survive)
        │
        ▼
  Stage 2: editorial pass                   summarizer.py
  Single Cerebras call over all
  Stage 1 outputs — curates, ranks,
  writes final content
        │
        ▼
  HTML email → AWS SES
```

---

## Design decisions worth noting

### Two-stage separation of concerns
Stage 1 is explicitly instructed to be exhaustive and opinion-free — its only job is to extract every number. Stage 2 is explicitly instructed to act as an editor — its only job is judgment. This separation matters: a single combined prompt conflates extraction quality with editorial quality and degrades both. The two-stage design also means Stage 1 runs in parallel across all filings, and Stage 2 sees the full day's picture before ranking anything.

### Earnings vs. non-earnings prompt branching
Earnings filings (those containing Item 2.02) get a completely different Stage 1 schema: revenue, EPS (GAAP and adjusted), margins, FCF, guidance, and a drug pipeline field for pharma/biotech. Non-earnings get a more general event schema. Using one prompt for both produces mediocre extraction for both.

### Exhibit hunting cascade
Most of the signal in an 8-K earnings filing lives in EX-99.1 — the press release — not the 8-K cover page itself. The pipeline tries three strategies in order without extra HTTP round-trips where possible: (1) extract exhibit blocks embedded inline in the SGML bundle, (2) parse EX-99.x links from the HTML cover page, (3) fall back to the EDGAR filing index JSON. This dramatically improves extraction quality for earnings filings compared to only reading the cover page.

### Selective web search
Web search is gated on both filing type and market cap. Consensus data is only useful for mid-cap+ earnings (small-cap consensus coverage is sparse). Analyst reactions are only useful for large-cap events. Searching everything wastes quota and injects noise into Stage 2.

### Hard safeguards around LLM decisions
Stage 2 can be overly aggressive in pruning. A post-Stage-2 enforcement step ensures any earnings filing from a company with market cap ≥ $1B that actually has extracted financial figures always appears in notable, regardless of what Stage 2 decided. LLM outputs are treated as a strong prior, not ground truth.

### Strict prompt attribution rules
The Stage 2 prompt has explicit rules against editorial adjectives without supporting numbers, against laundering management spin as analyst conclusion, and against unsubstantiated peer comparisons. The goal is a briefing that reads like a good analyst note, not a press release summary.

### Batching for large filing days
Stage 2 has a token budget. On heavy filing days (earnings season), the Stage 1 outputs are batched into groups of 60 and Stage 2 is called multiple times, with results merged. The notable/secondary structure makes merging straightforward.

---

## File structure

| File | Role |
|------|------|
| `report.py` | Main orchestration: ingestion, exhibit fetching, enrichment, batching, email delivery |
| `edgar.py` | SEC EDGAR API client — Atom feed pagination, CIK/ticker resolution, document index |
| `extractor.py` | Text cleaning — HTML stripping, boilerplate removal, Item section isolation |
| `summarizer.py` | LLM pipeline — Stage 1 extraction prompts, Stage 2 editorial prompt, Cerebras API calls |
| `polygon.py` | Polygon.io market data — price, daily change, market cap; formatting helpers |
| `search.py` | Parallel AI web search — consensus estimates, analyst reactions, counterparty descriptions |

---

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```
CEREBRAS_API_KEY=...
POLYGON_API_KEY=...
PARALLEL_API_KEY=...
SES_FROM_EMAIL=you@yourdomain.com
SES_TO_EMAIL=you@gmail.com
AWS_DEFAULT_REGION=us-east-1
```

AWS SES credentials should be in `~/.aws/credentials`. The IAM user needs only `ses:SendEmail`.

Run:
```bash
python report.py
```

---

## Dependencies

- [Cerebras](https://cerebras.ai/) — LLM inference (Qwen-3 235B)
- [Polygon.io](https://polygon.io/) — market data
- [Parallel AI](https://parallel.ai/) — web search
- [AWS SES](https://aws.amazon.com/ses/) — email delivery
- [SEC EDGAR](https://www.sec.gov/developer) — public filing data (no API key required)
- `requests` — the only pip dependency
