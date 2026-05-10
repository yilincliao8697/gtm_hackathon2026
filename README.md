# RateCase GTM Stakeholder Intelligence

Youtube Demo Video Link: https://youtu.be/jssdf0qK9F4

Prototype for turning certificate-of-service filings into a stakeholder graph
and GTM benchmark output.

## Tech Stack

**Backend / API**
- FastAPI + Uvicorn — web service & dashboard
- pdfplumber — PDF text extraction (with optional poppler/tesseract OCR fallback)
- SQLite — run history, feedback labels, outreach drafts (`runs.db`)

**AI / Intelligence**
- **OpenAI** (`gpt-4o-mini` via the `openai` Python SDK) — drafts per-contact outreach emails
- **Tavily Search API** (`https://api.tavily.com/search`) — web research dossier on each contact / organization, fed into the OpenAI prompt
- Custom rule-based GTM relevance scoring (stakeholder fit, org confidence, contact quality, network density, case proximity, GTM actionability)
- Cross-run feedback learning — per-category and per-domain biases derived from 👍/👎 labels

**Frontend**
- Self-contained HTML graph visualization (rendered server-side, no build step)
- Vanilla JS for feedback buttons and node interactions

**Deployment**
- Render (Web Service via `render.yaml` blueprint)

**Development**
- **OpenAI Codex** — used for prototyping and iterating on the extraction, scoring, and dashboard code

## Requirements

- Python 3.8+ (a virtualenv is recommended). The repository includes `research/.venv` in development environments.
- `pdfplumber` is required for PDF inputs; the script accepts pre-extracted `.txt` as an alternative.
- Optional for OCR fallbacks: `poppler` (provides `pdftotext`) and `tesseract-ocr`; Python packages `pdf2image` and `pytesseract` enable OCR.

Install the Python dependency inside your virtualenv if needed:

```bash
source research/.venv/bin/activate
pip install pdfplumber pdf2image pytesseract
```

Install system packages (optional, recommended for scanned PDFs):

- macOS (Homebrew): `brew install poppler tesseract`
- Ubuntu/Debian: `sudo apt-get install poppler-utils tesseract-ocr`

## Usage (script is the source of truth)

The script exposes a small CLI. Key options and their defaults (as of the code):

- `--input` (required): Certificate PDF or extracted `.txt` file.
- `--output-json` (required): Path where the graph JSON will be written.
- `--output-html` (required): Path for the self-contained HTML graph demo.
- `--output-csv` (optional): Path to write CSV of top contacts (columns: email, name, organization, domain, category, score, recommended_action, score_explanation).
- `--max-contacts` (default: `180`): Maximum contacts to render in the HTML.
- `--max-orgs` (default: `35`): Maximum organizations to render in the HTML.

Example (run from repo root):

```bash
research/.venv/bin/python research/gtm/certificate_service_graph.py \
  --input research/data/raw/CertificateOfService/A2106022_application_certificateOfService.pdf \
  --output-json research/gtm/results/A2106022_graph.json \
  --output-html research/gtm/results/A2106022_graph.html \
  --output-csv research/gtm/results/A2106022_contacts.csv
```

## Extraction fallbacks and OCR

The extractor will attempt several methods (in order) to get text from a PDF:

1. `pdfplumber` native extraction (fast, preferred).
2. `pdftotext` (poppler) via a subprocess, if available.
3. OCR via `pdf2image` + `pytesseract` (requires Tesseract/poppler and the Python packages) for scanned or image-based PDFs.

If no emails are found at any stage, the script will still write the JSON/HTML outputs but the contacts list will be empty. For the best results on scanned PDFs, install the system OCR/tools listed above.

## Case node derivation

- The graph's case node is now derived from the detected docket strings found in the document. If a docket is present, the first detected docket is used as the case id/label (e.g., `case:A.17-01-012`). If no docket is found, the script falls back to using the source filename as the case id/label.

## Output cleanup policy

- The script writes the files you request via the CLI. To avoid a proliferation of versions in `ratecase/research/gtm/results/`, we keep one canonical output per run (the files you explicitly specify).
- There was a temporary run artifact cleanup performed in-tree to remove older variant outputs; going forward prefer to pass explicit output paths with clear names (for example `A2106022_graph.json` and `A2106022_contacts.csv`) and the script will overwrite those paths on subsequent runs.

## What the script does

- Extracts emails from the input (PDF via `pdfplumber` and fallbacks, or `.txt`).
- Infers organizations and stakeholder categories from email domains using a built-in mapping and heuristics.
- Scores contacts and organizations with a rule-based GTM relevance model (scoring components include stakeholder fit, organization confidence, contact quality, network density, case proximity, and GTM actionability).
- Emits a JSON graph, a self-contained HTML demo that visualizes the case, dockets, organizations, and people, and optionally a CSV of top contacts if `--output-csv` is supplied.

## Notes & recommended next steps

- For the most robust extraction on scanned PDFs, install `poppler` and `tesseract-ocr` on your machine and re-run the script.
- Use the CSV export to feed contacts into an enrichment pipeline or CRM.
- If you expect many PDFs, consider adding a small worker that runs the extraction and enrichment and persists outputs to a canonical datastore rather than keeping many files on disk.



## Dashboard

A lightweight FastAPI dashboard is included at `ratecase/research/gtm/dashboard.py` that lets you upload a Certificate of Service (PDF or .txt) and returns the interactive graph and CSV/JSON outputs.

Run instructions:

```bash
source research/.venv/bin/activate
pip install fastapi uvicorn python-multipart
uvicorn ratecase.research.gtm.dashboard:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/` in your browser. The upload form will run the extractor in the repo venv and display the generated HTML graph.

Runs and feedback persist to a SQLite store at `runs.db` (tables: `runs`, `feedback`, `outreach`). Identical PDFs are deduped by SHA-256 hash — re-uploading the same file returns the prior run.

## Feedback and cross-run learning

The dashboard supports two layers of feedback that turn one-off scoring into a model that improves with use.

### Per-run adjustment (visible immediately)
- Each contact row has 👍 / 👎 / ✕ buttons; clicking a person node on the canvas opens a popover with the same controls.
- Labels POST to `/run/{run_id}/feedback`, persist to `feedback`, and the run is re-rendered with adjusted scores via `adjust_graph_with_feedback` (`certificate_service_graph.py`).
- Magnitudes: `PERSON_FEEDBACK_DELTA = 150`, `ORG_FEEDBACK_PER_LABEL = 50` (capped at `±200`).
- The toolbar shows `● Adjusted by N labels` with a toggle to view the unadjusted baseline.

### Cross-run learning (carries to new uploads)
Labels are aggregated across **all** runs into per-category and per-domain biases, applied at the time a new PDF is uploaded.

- Aggregation: `store.feedback_aggregates(conn)` produces `{categories, domains, totals}` using the latest non-`clear` label per `(run_id, contact_email)`.
- Conversion to deltas: `compute_prior_deltas` in `certificate_service_graph.py` with bounded multipliers:
  - Category: `±6` per net label, capped at `±60`
  - Domain: `±12` per net label, capped at `±120`
- Application: `apply_learned_priors(graph, aggregates)` adds the bias to person and organization baseline scores and re-sorts `top_contacts` / `top_organizations` before the graph is stored. The applied priors are recorded under `graph.benchmarks.learned_priors`.

Biases stack — a `pge.com` contact (category `utility`) absorbs both the domain prior and the category prior. The domain signal is specific (this firm); the category signal is the safety net for firms you have never labeled.

The home page shows a `● Learning from N labels` banner with the top biased categories/domains; each run page's toolbar shows the same summary. Score scale is `SCORE_CAP = 1000`, so a single label moves things by ~1% — the system needs many labels before any bucket reaches its cap.

### Outreach drafts (Claw 2)

Per-contact outreach emails are generated by a two-step AI pipeline (`outreach.py`):

1. **Tavily** (`https://api.tavily.com/search`) — fetches a short web-research dossier on the contact and their organization.
2. **OpenAI** (`gpt-4o-mini` via the official `openai` Python SDK) — drafts a tailored sales email using the Tavily dossier plus the stakeholder graph context (case, docket, role, score).

Trigger via `POST /run/{run_id}/generate-outreach?limit=N` (calls `outreach.generate_for_contacts`). Drafts are persisted to the `outreach` table and exported in the CSV.

- `OPENAI_API_KEY` — **required**; without it the endpoint returns 400.
- `TAVILY_API_KEY` — optional; without it the OpenAI prompt falls back to graph context only (no web research).

Configure both in `gtm/.env` or `research/.env`. On a deployed instance (e.g., Render), set them as environment variables instead.
