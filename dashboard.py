"""GTM dashboard: upload Certificate of Service PDFs, view stakeholder graphs, persist runs in SQLite."""

from __future__ import annotations

import csv
import html
import io
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from certificate_service_graph import (
    adjust_graph_with_feedback,
    apply_learned_priors,
    build_graph_from_path,
    compute_prior_deltas,
    graph_to_csv_rows,
    render_html,
)
from outreach import generate_for_contacts, keys_status
from store import (
    add_feedback,
    connect,
    feedback_aggregates,
    feedback_for_run,
    get_run,
    get_run_by_hash,
    hash_bytes,
    insert_run,
    list_runs,
    outreach_for_run,
    upsert_outreach,
)

app = FastAPI(title="RateCase GTM Dashboard")


def _index_learning_banner(aggregates: dict) -> str:
    totals = (aggregates or {}).get("totals") or {}
    n = totals.get("labeled_contacts") or 0
    if n == 0:
        return (
            "<div style='font-size:12px;color:#5b6770;margin:4px 0 12px;'>"
            "Model has no feedback yet. Label a few contacts inside a run; new uploads will inherit those biases automatically.</div>"
        )
    deltas = compute_prior_deltas(aggregates)
    cats = deltas.get("categories") or {}
    doms = deltas.get("domains") or {}
    merged = {**{f"category {k}": v for k, v in cats.items()}, **{f"domain {k}": v for k, v in doms.items()}}
    items = sorted(merged.items(), key=lambda kv: -abs(kv[1]))[:5]
    chips = " ".join(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:{'#dcfce7' if v > 0 else '#fee2e2'};color:#172026;font-size:11px;margin-right:6px;'>"
        f"{html.escape(name)} {v:+d}</span>"
        for name, v in items
    )
    return (
        f"<div style='font-size:12px;color:#5b6770;margin:4px 0 12px;'>"
        f"<strong style='color:#16a34a;'>● Learning</strong> from {n} label{'s' if n != 1 else ''} "
        f"({totals.get('positive', 0)} 👍 / {totals.get('negative', 0)} 👎). New uploads inherit these biases automatically:<br>"
        f"<div style='margin-top:6px;'>{chips or '—'}</div>"
        "</div>"
    )


def _index_html(runs, aggregates: dict | None = None) -> str:
    rows = "".join(
        f"<tr>"
        f"<td><a href='/run/{html.escape(r['id'])}'>{html.escape(r['case_label'] or r['pdf_filename'])}</a></td>"
        f"<td>{html.escape(r['pdf_filename'])}</td>"
        f"<td>{r['unique_emails']}</td>"
        f"<td>{r['unique_domains']}</td>"
        f"<td>{r['docket_count']}</td>"
        f"<td>{html.escape(r['created_at'][:19].replace('T', ' '))}</td>"
        f"</tr>"
        for r in runs
    )
    if not rows:
        rows = "<tr><td colspan='6' style='color:#888;text-align:center;padding:18px;'>No runs yet — upload a certificate above.</td></tr>"
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>GTM Dashboard</title>
<style>
body {{ font-family: Inter, system-ui, sans-serif; max-width: 980px; margin: 32px auto; padding: 0 18px; color: #172026; }}
h1 {{ margin: 0 0 6px; font-size: 22px; }}
p {{ color: #5b6770; }}
.upload {{ border: 1px solid #d7dee4; border-radius: 8px; padding: 16px; background: #f7f9fb; margin: 18px 0; }}
table {{ width: 100%; border-collapse: collapse; border: 1px solid #d7dee4; border-radius: 8px; overflow: hidden; font-size: 13px; }}
th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #eef1f3; }}
th {{ background: #fbfcfd; color: #5b6770; font-weight: 600; }}
tr:last-child td {{ border-bottom: 0; }}
a {{ color: #2364aa; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
button {{ background: #172026; color: #fff; border: 0; border-radius: 6px; padding: 7px 14px; cursor: pointer; }}
</style></head>
<body>
<h1>GTM Stakeholder Dashboard</h1>
<p>Upload a Certificate of Service PDF or .txt. Identical files are deduped — re-uploading the same PDF returns the prior run.</p>
{_index_learning_banner(aggregates or {})}
<div class='upload'>
  <form action='/upload' method='post' enctype='multipart/form-data'>
    <input type='file' name='file' accept='.pdf,.txt' required>
    <button type='submit'>Upload &amp; analyze</button>
  </form>
</div>
<h2 style='font-size:15px;'>Past runs</h2>
<table>
<thead><tr><th>Case</th><th>File</th><th>Emails</th><th>Orgs</th><th>Dockets</th><th>Created (UTC)</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body></html>
"""


def _learned_priors_block(priors: dict | None) -> str:
    if not priors:
        return ""
    totals = priors.get("totals") or {}
    n = totals.get("labeled_contacts") or 0
    if n == 0:
        return ""
    cats = priors.get("categories") or {}
    doms = priors.get("domains") or {}
    applied = priors.get("applied")

    def _top(d: dict, sign: int, k: int = 3) -> list[tuple[str, int]]:
        items = [(name, delta) for name, delta in d.items() if (delta > 0 if sign > 0 else delta < 0)]
        items.sort(key=lambda kv: -abs(kv[1]))
        return items[:k]

    def _fmt(items: list[tuple[str, int]]) -> str:
        return ", ".join(f"{html.escape(name)} {delta:+d}" for name, delta in items) or "—"

    boosted = _top({**cats, **doms}, +1)
    penalized = _top({**cats, **doms}, -1)
    headline = (
        f"<span style='color:#16a34a;'>● Learning</span> from "
        f"<strong>{n}</strong> label{'s' if n != 1 else ''} across all runs"
    )
    if not applied:
        headline += " <span style='color:#5b6770;'>(no signal applied to this run)</span>"
    return (
        f"<div style='margin-top:6px;padding-top:6px;border-top:1px solid #eef1f3;font-size:12px;color:#5b6770;'>"
        f"{headline}<br>"
        f"<span style='color:#0f7a3a;'>↑</span> {_fmt(boosted)}<br>"
        f"<span style='color:#b91c1c;'>↓</span> {_fmt(penalized)}"
        "</div>"
    )


def _toolbar(run_id: str, adjusted: bool, label_count: int, priors: dict | None = None) -> str:
    if label_count == 0:
        status = "<span style='color:#5b6770;'>No labels yet — click 👍/👎 on a person node.</span>"
    elif adjusted:
        toggle = f"<a href='/run/{run_id}?adjusted=0'>view baseline</a>"
        status = f"<span style='color:#16a34a;'>● Adjusted by {label_count} label{'s' if label_count != 1 else ''}</span> &nbsp;|&nbsp; {toggle}"
    else:
        toggle = f"<a href='/run/{run_id}'>view adjusted</a>"
        status = f"<span style='color:#5b6770;'>● Showing baseline ({label_count} label{'s' if label_count != 1 else ''} not applied)</span> &nbsp;|&nbsp; {toggle}"
    return (
        "<div style='position:fixed;right:12px;top:12px;z-index:9999;background:#fff;"
        "padding:8px 12px;border:1px solid #d7dee4;border-radius:8px;font:13px system-ui;"
        "box-shadow:0 6px 18px rgba(23,32,38,.08);max-width:520px;'>"
        f"{status}<br>"
        f"<a href='/run/{run_id}/json' target='_blank'>JSON</a> &nbsp;|&nbsp; "
        f"<a href='/run/{run_id}/csv' target='_blank'>CSV</a> &nbsp;|&nbsp; "
        "<a href='/'>Home</a>"
        f"{_learned_priors_block(priors)}"
        "</div>"
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with connect() as conn:
        runs = list_runs(conn)
        aggregates = feedback_aggregates(conn)
    return HTMLResponse(_index_html(runs, aggregates))


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> Response:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    pdf_hash = hash_bytes(data)

    with connect() as conn:
        existing = get_run_by_hash(conn, pdf_hash)
        if existing:
            return RedirectResponse(f"/run/{existing['id']}", status_code=303)

    suffix = Path(file.filename or "upload.pdf").suffix.lower() or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        graph = build_graph_from_path(Path(tmp.name))

    graph["source"] = file.filename or "upload"

    with connect() as conn:
        aggregates = feedback_aggregates(conn)
    apply_learned_priors(graph, aggregates)
    graph_html = render_html(graph)

    with connect() as conn:
        run_id = insert_run(conn, pdf_hash, file.filename or "upload", graph, graph_html)
    return RedirectResponse(f"/run/{run_id}", status_code=303)


@app.get("/run/{run_id}", response_class=HTMLResponse)
def view_run(run_id: str, adjusted: int = 1) -> HTMLResponse:
    with connect() as conn:
        row = get_run(conn, run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        labels = feedback_for_run(conn, run_id)
    graph = json.loads(row["graph_json"])
    apply_adjusted = bool(adjusted) and bool(labels)
    if apply_adjusted:
        graph = adjust_graph_with_feedback(graph, labels)
    priors = (graph.get("benchmarks") or {}).get("learned_priors")
    graph_html = render_html(graph)
    context_script = (
        "<script>"
        f"window.GTM_RUN_ID={json.dumps(run_id)};"
        f"window.GTM_FEEDBACK={json.dumps(labels)};"
        "</script>"
    )
    graph_html = graph_html.replace("<body>", "<body>" + context_script, 1)
    return HTMLResponse(_toolbar(run_id, apply_adjusted, len(labels), priors) + graph_html)


@app.get("/run/{run_id}/json")
def download_json(run_id: str) -> Response:
    with connect() as conn:
        row = get_run(conn, run_id)
    if not row:
        raise HTTPException(status_code=404)
    return Response(
        content=row["graph_json"],
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={run_id}_graph.json"},
    )


@app.get("/run/{run_id}/csv")
def download_csv(run_id: str, limit: str = "25") -> Response:
    with connect() as conn:
        row = get_run(conn, run_id)
        if not row:
            raise HTTPException(status_code=404)
        outreach = outreach_for_run(conn, run_id)
    graph = json.loads(row["graph_json"])
    if limit == "all":
        rows = [
            {
                "email": n.get("email"),
                "name": n.get("label"),
                "organization": n.get("organization", ""),
                "domain": n.get("domain", ""),
                "category": n.get("category", ""),
                "score": n.get("score", 0),
                "recommended_action": n.get("recommended_action", ""),
                "score_explanation": n.get("score_explanation", ""),
            }
            for n in graph.get("nodes", [])
            if n.get("kind") == "person" and n.get("email")
        ]
        rows.sort(key=lambda r: (-r["score"], r["email"]))
        suffix = "all"
    else:
        try:
            n = max(1, int(limit))
        except ValueError:
            n = 25
        rows = graph_to_csv_rows(graph)[:n]
        suffix = f"top{n}"
    for r in rows:
        match = outreach.get((r.get("email") or "").lower(), {})
        r["outreach_subject"] = match.get("subject", "")
        r["outreach_body"] = match.get("body", "")
        r["research_summary"] = match.get("research_summary", "")
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "email", "name", "organization", "domain", "category", "score",
            "recommended_action", "score_explanation",
            "outreach_subject", "outreach_body", "research_summary",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={run_id}_contacts_{suffix}.csv"},
    )


@app.post("/run/{run_id}/generate-outreach")
async def generate_outreach(run_id: str, limit: int = 25) -> Response:
    keys = keys_status()
    if not keys["openai"]:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set in research/.env or gtm/.env")
    with connect() as conn:
        row = get_run(conn, run_id)
        if not row:
            raise HTTPException(status_code=404)
    graph = json.loads(row["graph_json"])
    contacts = graph.get("benchmarks", {}).get("top_contacts", [])[: max(1, limit)]
    case_label = row["case_label"] or "this proceeding"

    results = await generate_for_contacts(contacts, case_label)

    with connect() as conn:
        for r in results:
            upsert_outreach(
                conn,
                run_id,
                r["contact_email"],
                r["subject"],
                r["body"],
                r["research_summary"],
                r.get("error", ""),
            )

    errors = [r for r in results if r.get("error")]
    return Response(
        content=json.dumps({
            "generated": len(results) - len(errors),
            "errors": len(errors),
            "tavily_enabled": keys["tavily"],
            "first_error": errors[0]["error"] if errors else "",
        }),
        media_type="application/json",
    )


@app.post("/run/{run_id}/feedback")
def post_feedback(run_id: str, contact_email: str = Form(...), label: str = Form(...), note: str | None = Form(None)) -> Response:
    if label not in {"positive", "negative", "clear"}:
        raise HTTPException(status_code=400, detail="label must be 'positive', 'negative', or 'clear'")
    email = contact_email.lower()
    with connect() as conn:
        row = get_run(conn, run_id)
        if not row:
            raise HTTPException(status_code=404)
        category, domain = "", ""
        try:
            graph = json.loads(row["graph_json"])
            for node in graph.get("nodes", []):
                if node.get("kind") == "person" and (node.get("email") or "").lower() == email:
                    category = node.get("category") or ""
                    domain = (node.get("domain") or "").lower()
                    break
        except (json.JSONDecodeError, TypeError):
            pass
        add_feedback(conn, run_id, email, label, note, category=category or None, domain=domain or None)
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
