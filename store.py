"""SQLite-backed store for GTM dashboard runs and feedback labels."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DB_PATH = Path(__file__).resolve().parent / "runs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    pdf_hash TEXT UNIQUE NOT NULL,
    pdf_filename TEXT NOT NULL,
    case_label TEXT,
    unique_emails INTEGER,
    unique_domains INTEGER,
    docket_count INTEGER,
    graph_json TEXT NOT NULL,
    graph_html TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    contact_email TEXT NOT NULL,
    label TEXT NOT NULL,
    note TEXT,
    category TEXT,
    domain TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_run ON feedback(run_id);
CREATE INDEX IF NOT EXISTS idx_feedback_email ON feedback(contact_email);

CREATE TABLE IF NOT EXISTS outreach (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    contact_email TEXT NOT NULL,
    subject TEXT,
    body TEXT,
    research_summary TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, contact_email),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_outreach_run ON outreach(run_id);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(feedback)").fetchall()}
    if "category" not in cols:
        conn.execute("ALTER TABLE feedback ADD COLUMN category TEXT")
    if "domain" not in cols:
        conn.execute("ALTER TABLE feedback ADD COLUMN domain TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_category ON feedback(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_domain ON feedback(domain)")
    conn.commit()
    return conn


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short_id(pdf_hash: str) -> str:
    return pdf_hash[:12]


def get_run_by_hash(conn: sqlite3.Connection, pdf_hash: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE pdf_hash = ?", (pdf_hash,)).fetchone()


def get_run(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def list_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, pdf_filename, case_label, unique_emails, unique_domains, docket_count, created_at "
        "FROM runs ORDER BY created_at DESC"
    ).fetchall()


def insert_run(
    conn: sqlite3.Connection,
    pdf_hash: str,
    pdf_filename: str,
    graph: dict[str, Any],
    graph_html: str,
) -> str:
    summary = graph.get("summary", {})
    case_node = next((n for n in graph.get("nodes", []) if n.get("kind") == "case"), None)
    case_label = case_node["label"] if case_node else pdf_filename
    run_id = short_id(pdf_hash)
    conn.execute(
        "INSERT INTO runs (id, pdf_hash, pdf_filename, case_label, unique_emails, unique_domains, "
        "docket_count, graph_json, graph_html, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            pdf_hash,
            pdf_filename,
            case_label,
            summary.get("unique_emails", 0),
            summary.get("unique_domains", 0),
            len(summary.get("dockets", [])),
            json.dumps(graph, ensure_ascii=False),
            graph_html,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return run_id


def add_feedback(
    conn: sqlite3.Connection,
    run_id: str,
    contact_email: str,
    label: str,
    note: str | None = None,
    category: str | None = None,
    domain: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO feedback (run_id, contact_email, label, note, category, domain, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, contact_email, label, note, category, domain, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def feedback_for_run(conn: sqlite3.Connection, run_id: str) -> dict[str, str]:
    """Latest non-clear label per contact email for this run."""
    rows = conn.execute(
        "SELECT contact_email, label FROM feedback WHERE run_id = ? "
        "ORDER BY created_at DESC",
        (run_id,),
    ).fetchall()
    seen: set[str] = set()
    latest: dict[str, str] = {}
    for row in rows:
        email = row["contact_email"]
        if email in seen:
            continue
        seen.add(email)
        if row["label"] != "clear":
            latest[email] = row["label"]
    return latest


def upsert_outreach(
    conn: sqlite3.Connection,
    run_id: str,
    contact_email: str,
    subject: str,
    body: str,
    research_summary: str,
    error: str = "",
) -> None:
    conn.execute(
        "INSERT INTO outreach (run_id, contact_email, subject, body, research_summary, error, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(run_id, contact_email) DO UPDATE SET "
        "subject=excluded.subject, body=excluded.body, research_summary=excluded.research_summary, "
        "error=excluded.error, created_at=excluded.created_at",
        (
            run_id,
            contact_email,
            subject,
            body,
            research_summary,
            error,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def outreach_for_run(conn: sqlite3.Connection, run_id: str) -> dict[str, dict[str, str]]:
    rows = conn.execute(
        "SELECT contact_email, subject, body, research_summary, error FROM outreach WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    return {
        row["contact_email"]: {
            "subject": row["subject"] or "",
            "body": row["body"] or "",
            "research_summary": row["research_summary"] or "",
            "error": row["error"] or "",
        }
        for row in rows
    }


def feedback_aggregates(conn: sqlite3.Connection) -> dict[str, dict[str, dict[str, int]]]:
    """Aggregate latest-per-(run,email) labels into category and domain counts across all runs.

    Returns:
        {
            "categories": {"regulator": {"positive": 2, "negative": 5}, ...},
            "domains":    {"dwt.com":   {"positive": 3, "negative": 0}, ...},
            "totals":     {"positive": int, "negative": int, "labeled_contacts": int},
        }
    """
    rows = conn.execute(
        """
        SELECT f.contact_email, f.run_id, f.label, f.category, f.domain, f.created_at
        FROM feedback f
        JOIN (
            SELECT run_id, contact_email, MAX(created_at) AS max_ts
            FROM feedback
            GROUP BY run_id, contact_email
        ) latest
          ON f.run_id = latest.run_id
         AND f.contact_email = latest.contact_email
         AND f.created_at = latest.max_ts
        WHERE f.label IN ('positive', 'negative')
        """
    ).fetchall()

    categories: dict[str, dict[str, int]] = {}
    domains: dict[str, dict[str, int]] = {}
    pos = neg = 0
    for row in rows:
        label = row["label"]
        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        cat = (row["category"] or "").strip()
        dom = (row["domain"] or "").strip().lower()
        if cat:
            bucket = categories.setdefault(cat, {"positive": 0, "negative": 0})
            bucket[label] = bucket.get(label, 0) + 1
        if dom:
            bucket = domains.setdefault(dom, {"positive": 0, "negative": 0})
            bucket[label] = bucket.get(label, 0) + 1

    return {
        "categories": categories,
        "domains": domains,
        "totals": {"positive": pos, "negative": neg, "labeled_contacts": pos + neg},
    }


def feedback_history(conn: sqlite3.Connection, emails: Iterable[str]) -> dict[str, list[str]]:
    """Return all labels ever applied to each email across all runs (for adaptive scoring later)."""
    emails = list(emails)
    if not emails:
        return {}
    placeholders = ",".join("?" * len(emails))
    rows = conn.execute(
        f"SELECT contact_email, label FROM feedback WHERE contact_email IN ({placeholders})",
        emails,
    ).fetchall()
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row["contact_email"], []).append(row["label"])
    return out
