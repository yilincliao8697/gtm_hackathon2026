"""
Build a stakeholder graph from a certificate-of-service PDF or extracted text.

This is a hackathon-oriented prototype: it extracts emails, infers
organizations from email domains, classifies likely stakeholder types, and
emits both machine-readable JSON and a self-contained HTML graph demo.

Usage:
    python src/certificate_service_graph.py \
      --input data/raw/CertificateOfService/A2106022_application_certificateOfService.pdf \
      --output-json results/certificate_service_graph.json \
      --output-html results/certificate_service_graph.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
DOCKET_RE = re.compile(r"\b[ARCI]\.?\d{2}-\d{2}-\d{3}\b", re.IGNORECASE)
COMPACT_APPLICATION_RE = re.compile(r"\bA(\d{2})(\d{2})(\d{3})\b")

KNOWN_DOMAINS = {
    "cpuc.ca.gov": ("California Public Utilities Commission", "regulator"),
    "pge.com": ("Pacific Gas and Electric Company", "utility"),
    "sce.com": ("Southern California Edison", "utility"),
    "sdge.com": ("San Diego Gas & Electric", "utility"),
    "socalgas.com": ("Southern California Gas Company", "utility"),
    "semprautilities.com": ("Sempra Utilities", "utility"),
    "pacificorp.com": ("PacifiCorp", "utility"),
    "libertyutilities.com": ("Liberty Utilities", "utility"),
    "bves.com": ("Bear Valley Electric Service", "utility"),
    "turn.org": ("The Utility Reform Network", "advocacy"),
    "ucan.org": ("Utility Consumers' Action Network", "advocacy"),
    "earthjustice.org": ("Earthjustice", "advocacy"),
    "sierraclub.org": ("Sierra Club", "advocacy"),
    "cforat.org": ("Californians for Renewable Energy", "advocacy"),
    "clean-coalition.org": ("Clean Coalition", "advocacy"),
    "storagealliance.org": ("California Energy Storage Alliance", "market_participant"),
    "caiso.com": ("California ISO", "market_operator"),
    "energy.ca.gov": ("California Energy Commission", "regulator"),
    "dwt.com": ("Davis Wright Tremaine LLP", "law_firm"),
    "buchalter.com": ("Buchalter", "law_firm"),
    "goodinmacbride.com": ("Goodin, MacBride, Squeri & Day", "law_firm"),
    "keyesfox.com": ("Keyes & Fox LLP", "law_firm"),
    "bbklaw.com": ("Best Best & Krieger LLP", "law_firm"),
    "adamsbroadwell.com": ("Adams Broadwell Joseph & Cardozo", "law_firm"),
    "nossaman.com": ("Nossaman LLP", "law_firm"),
    "dentons.com": ("Dentons", "law_firm"),
    "morganlewis.com": ("Morgan Lewis", "law_firm"),
    "perkinscoie.com": ("Perkins Coie", "law_firm"),
}

SCORE_CAP = 1000

CATEGORY_WEIGHTS = {
    "regulator": 300,
    "utility": 280,
    "law_firm": 240,
    "advocacy": 220,
    "market_operator": 200,
    "public_agency": 180,
    "market_participant": 140,
    "unknown": 60,
}

ACTIONABILITY_WEIGHTS = {
    "utility": 220,
    "law_firm": 200,
    "market_participant": 180,
    "market_operator": 140,
    "advocacy": 120,
    "public_agency": 80,
    "regulator": 50,
    "unknown": 40,
}

SCORING_MODEL = [
    {
        "component": "stakeholder_fit",
        "max_points": 300,
        "description": "How important this stakeholder category is in a utility rate-case ecosystem.",
    },
    {
        "component": "organization_confidence",
        "max_points": 150,
        "description": "Known domains get more confidence than inferred domains.",
    },
    {
        "component": "contact_quality",
        "max_points": 100,
        "description": "Named professional emails are easier to research than generic inboxes.",
    },
    {
        "component": "network_density",
        "max_points": 200,
        "description": "More contacts from the same organization means stronger evidence that the org matters.",
    },
    {
        "component": "case_proximity",
        "max_points": 150,
        "description": "Contacts mentioned in the certificate narrative are closer to the filing event.",
    },
    {
        "component": "gtm_actionability",
        "max_points": 220,
        "description": "Prioritizes utility-side, law-firm, consultant, and market participants over regulator outreach.",
    },
]

BENCHMARK_METHOD = {
    "name": "Certificate-of-Service GTM Relevance Benchmark",
    "summary": (
        "A rule-based benchmark that ranks stakeholders by how useful they are for GTM "
        "research in a rate-case ecosystem. It is created from public service-list signals, "
        "not from web search or an AI scoring model."
    ),
    "person_formula": (
        "person_score = stakeholder_fit + organization_confidence + contact_quality + "
        "network_density + case_proximity + gtm_actionability, capped at 1000"
    ),
    "organization_formula": (
        "organization_score = stakeholder_fit + organization_confidence + network_density + "
        "docket_context + gtm_actionability, capped at 1000"
    ),
    "created_from": [
        "email domain",
        "recognized organization/domain list",
        "inferred stakeholder category",
        "number of contacts from the same organization",
        "whether the contact appears in the certificate narrative",
        "number of dockets referenced by the certificate",
    ],
}


@dataclass(frozen=True)
class Contact:
    email: str
    name: str
    domain: str
    organization: str
    category: str
    score: int
    score_components: dict[str, int]
    score_explanation: str
    recommended_action: str


def extract_text(path: Path) -> str:
    """
    Extract text from a file. For PDFs, try a sequence of fallbacks:
    1. `pdfplumber` native extraction
    2. `pdftotext` (poppler) if available
    3. OCR via `pdf2image` + `pytesseract` if available
    For `.txt` files, read directly.
    """
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")

    try:
        import pdfplumber
    except ImportError as exc:
        raise SystemExit(
            "pdfplumber is required for PDF input. Run this with research/.venv/bin/python "
            "or pass a pre-extracted .txt file."
        ) from exc

    # First attempt: pdfplumber extraction
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        text = ""

    # If we found emails, return early
    if EMAIL_RE.search(text or ""):
        return text

    # Attempt pdftotext (poppler) as a fast fallback
    try:
        import subprocess
        proc = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, timeout=20)
        if proc.returncode == 0 and proc.stdout and EMAIL_RE.search(proc.stdout):
            return proc.stdout
    except Exception:
        # pdftotext not available or failed; continue to OCR
        pass

    # Attempt OCR fallback using pdf2image + pytesseract
    try:
        from pdf2image import convert_from_path
        import pytesseract
        from PIL import Image
        # Convert PDF to images (may require poppler); set simple dpi
        pages = convert_from_path(str(path), dpi=200)
        ocr_text_parts = []
        for page_im in pages:
            try:
                page_text = pytesseract.image_to_string(page_im)
                ocr_text_parts.append(page_text)
            except Exception:
                continue
        ocr_text = "\n\n".join(ocr_text_parts)
        if EMAIL_RE.search(ocr_text or ""):
            return ocr_text
    except Exception:
        # OCR not available or failed
        pass

    # Fallback: return whatever pdfplumber produced (maybe empty)
    return text


def normalize_dockets(text: str) -> list[str]:
    dockets = {match.group(0).upper().replace(".", "") for match in DOCKET_RE.finditer(text)}
    dockets.update(
        f"A{match.group(1)}-{match.group(2)}-{match.group(3)}"
        for match in COMPACT_APPLICATION_RE.finditer(text)
    )

    normalized = []
    for docket in dockets:
        if "-" in docket:
            normalized.append(docket if docket[1] == "." else f"{docket[0]}.{docket[1:]}")
        elif len(docket) == 8 and docket[0].isalpha():
            normalized.append(f"{docket[0]}.{docket[1:3]}-{docket[3:5]}-{docket[5:]}")
    return sorted(set(normalized))


def domain_to_org(domain: str) -> tuple[str, str]:
    if domain in KNOWN_DOMAINS:
        return KNOWN_DOMAINS[domain]

    if domain.endswith(".gov"):
        return (title_from_domain(domain), "public_agency")
    if any(token in domain for token in ("law", "legal", "llp", "apc")):
        return (title_from_domain(domain), "law_firm")
    if any(token in domain for token in ("energy", "solar", "power", "grid", "microgrid")):
        return (title_from_domain(domain), "market_participant")
    return (title_from_domain(domain), "unknown")


def title_from_domain(domain: str) -> str:
    root = domain.split(".")[0]
    root = re.sub(r"[-_]+", " ", root)
    replacements = {
        "pge": "PG&E",
        "sdge": "SDG&E",
        "sce": "SCE",
        "cpuc": "CPUC",
    }
    return replacements.get(root.lower(), root.title())


def name_from_email(email: str) -> str:
    local = email.split("@", 1)[0]
    local = re.sub(r"\+.*$", "", local)
    parts = [part for part in re.split(r"[._-]+", local) if part and not part.isdigit()]
    if len(parts) >= 2 and all(len(part) > 1 for part in parts[:2]):
        return " ".join(part.capitalize() for part in parts[:3])
    if len(parts) == 1 and len(parts[0]) > 3:
        return parts[0].capitalize()
    return email


def contact_recommended_action(category: str) -> str:
    if category == "regulator":
        return "Map as decision ecosystem; avoid direct sales outreach."
    if category == "utility":
        return "Prioritize for account research and utility-side discovery."
    if category == "law_firm":
        return "Research as channel, referral, or expert-network target."
    if category == "advocacy":
        return "Track for objections, policy themes, and stakeholder pressure."
    if category in {"market_participant", "market_operator"}:
        return "Evaluate as partner, competitor, or market-signal source."
    if category == "public_agency":
        return "Map for jurisdiction context; use care with outreach."
    return "Research manually before outreach."


VOWELS = set("AEIOU")


def is_cryptic_alias(local: str) -> bool:
    """True if the email local-part looks like a cryptic distribution alias (T3M3, WAG9, jwwd) rather than a real person."""
    local = local.split("+", 1)[0]
    if any(sep in local for sep in (".", "_", "-")):
        return False
    if len(local) > 6 or len(local) < 2:
        return False
    if any(c.isdigit() for c in local):
        return True
    upper = local.upper()
    if local == upper and len(local) >= 3:
        return True
    if not any(c in VOWELS for c in upper):
        return True
    return False


def contact_quality_score(local: str) -> tuple[int, str]:
    """Return (points, label) for contact_quality based on the email local-part shape."""
    local = local.split("+", 1)[0]
    if local.lower().startswith(("info", "service", "regulatory", "noreply", "no-reply", "donotreply")):
        return 50, "role inbox"
    if is_cryptic_alias(local):
        return 25, "alias / distribution"
    if "." in local or "_" in local:
        return 100, "named professional"
    return 60, "single-word"


def score_contact(email: str, category: str, domain: str, domain_count: int, narrative: str) -> tuple[int, dict[str, int], str, str]:
    local = email.split("@", 1)[0]
    category_points = min(CATEGORY_WEIGHTS.get(category, 60), 300)
    known_domain_points = 150 if domain in KNOWN_DOMAINS else 60
    contact_quality_points, quality_label = contact_quality_score(local)
    density_points = min(domain_count * 20, 200)
    proximity_points = 150 if email in narrative else 50
    actionability_points = ACTIONABILITY_WEIGHTS.get(category, 40)

    components = {
        "stakeholder_fit": category_points,
        "organization_confidence": known_domain_points,
        "contact_quality": contact_quality_points,
        "network_density": density_points,
        "case_proximity": proximity_points,
        "gtm_actionability": actionability_points,
    }
    score = min(sum(components.values()), SCORE_CAP)
    explanation = (
        f"{category.replace('_', ' ')} fit {category_points}, "
        f"org confidence {known_domain_points}, contact quality {contact_quality_points} ({quality_label}), "
        f"network density {density_points}, case proximity {proximity_points}, "
        f"GTM actionability {actionability_points}."
    )
    return score, components, explanation, contact_recommended_action(category)


def score_organization(domain: str, category: str, contact_count: int, docket_count: int) -> tuple[int, dict[str, int], str, str]:
    category_points = min(CATEGORY_WEIGHTS.get(category, 60), 300)
    known_domain_points = 200 if domain in KNOWN_DOMAINS else 80
    density_points = min(contact_count * 30, 300)
    docket_context_points = min(docket_count * 20, 100)
    actionability_points = ACTIONABILITY_WEIGHTS.get(category, 40)

    components = {
        "stakeholder_fit": category_points,
        "organization_confidence": known_domain_points,
        "network_density": density_points,
        "docket_context": docket_context_points,
        "gtm_actionability": actionability_points,
    }
    score = min(sum(components.values()), SCORE_CAP)
    explanation = (
        f"{category.replace('_', ' ')} fit {category_points}, "
        f"org confidence {known_domain_points}, contact density {density_points}, "
        f"docket context {docket_context_points}, GTM actionability {actionability_points}."
    )
    return score, components, explanation, contact_recommended_action(category)


def extract_contacts(text: str) -> list[Contact]:
    emails = sorted({match.group(0).lower() for match in EMAIL_RE.finditer(text)})
    domain_counts = Counter(email.split("@", 1)[1] for email in emails)
    contacts = []

    narrative = text[:3000].lower()
    for email in emails:
        domain = email.split("@", 1)[1]
        organization, category = domain_to_org(domain)
        score, components, explanation, action = score_contact(
            email=email,
            category=category,
            domain=domain,
            domain_count=domain_counts[domain],
            narrative=narrative,
        )
        contacts.append(
            Contact(
                email=email,
                name=name_from_email(email),
                domain=domain,
                organization=organization,
                category=category,
                score=score,
                score_components=components,
                score_explanation=explanation,
                recommended_action=action,
            )
        )

    return sorted(contacts, key=lambda item: (-item.score, item.email))


def select_contacts(
    contacts: list[Contact],
    org_scores: dict[str, dict[str, Any]],
    max_contacts: int,
    max_orgs: int,
) -> list[Contact]:
    contacts_by_domain = defaultdict(list)
    for contact in contacts:
        contacts_by_domain[contact.domain].append(contact)

    top_domains = [
        item["domain"]
        for item in sorted(org_scores.values(), key=lambda value: (-value["score"], value["domain"]))[
            :max_orgs
        ]
    ]

    selected: list[Contact] = []
    selected_emails = set()
    round_index = 0
    while len(selected) < max_contacts:
        added_this_round = False
        for domain in top_domains:
            domain_contacts = contacts_by_domain[domain]
            if round_index >= len(domain_contacts):
                continue
            contact = domain_contacts[round_index]
            if contact.email in selected_emails:
                continue
            selected.append(contact)
            selected_emails.add(contact.email)
            added_this_round = True
            if len(selected) >= max_contacts:
                break
        if not added_this_round:
            break
        round_index += 1
    return selected


def org_rank_key(org: dict[str, Any]) -> tuple[int, int, str]:
    return (-org["score"], -org["contact_count"], org["domain"])


def build_recommendations(org_scores: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    orgs = sorted(org_scores.values(), key=org_rank_key)

    def first_org(*categories: str) -> dict[str, Any] | None:
        return next((org for org in orgs if org["category"] in categories), None)

    recommendations = []
    utility = first_org("utility")
    law_firm = first_org("law_firm")
    regulator = first_org("regulator")
    advocacy = first_org("advocacy")
    market = first_org("market_participant", "market_operator")

    if utility:
        recommendations.append(
            {
                "label": "Prioritize utility account",
                "target": utility["organization"],
                "target_type": "Utility-side organization",
                "reason": "Best utility-side account to research first. Use it for ICP research, account planning, and understanding the buyer side of this proceeding.",
            }
        )
    if law_firm:
        recommendations.append(
            {
                "label": "Research advisor ecosystem",
                "target": law_firm["organization"],
                "target_type": "Law firm or advisor",
                "reason": "Worth investigating because law firms and advisors can reveal repeat rate-case experts, referral paths, and implementation partners.",
            }
        )
    if regulator:
        recommendations.append(
            {
                "label": "Map regulator context",
                "target": regulator["organization"],
                "target_type": "Regulator or commission",
                "reason": "Important for understanding the proceeding ecosystem, but these contacts should be mapped as context rather than treated as sales leads.",
            }
        )
    if advocacy:
        recommendations.append(
            {
                "label": "Watch intervenor pressure",
                "target": advocacy["organization"],
                "target_type": "Advocacy or intervenor organization",
                "reason": "Useful for spotting objections, policy pressure, contested issues, and arguments that may shape the case.",
            }
        )
    if market:
        recommendations.append(
            {
                "label": "Evaluate market signal",
                "target": market["organization"],
                "target_type": "Market participant or operator",
                "reason": "Could be a partner, competitor, customer signal, or market-structure clue. Research before deciding whether it is GTM-relevant.",
            }
        )
    return recommendations


def build_graph(text: str, source_path: Path, max_contacts: int, max_orgs: int) -> dict[str, Any]:
    contacts = extract_contacts(text)
    dockets = normalize_dockets(text)
    domain_counts = Counter(contact.domain for contact in contacts)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    source_id = f"doc:{source_path.name}"
    # Choose a primary case docket using heuristics:
    # 1) If the filename contains a compact docket (A2106022), prefer that match.
    # 2) Otherwise prefer an application docket starting with 'A.' if present.
    # 3) Fallback to the first detected docket or the filename.
    primary = None
    if dockets:
        # try to match compact docket in filename
        m = re.search(r'A(\d{2})(\d{2})(\d{3})', source_path.name, re.IGNORECASE)
        if m:
            normalized = f"A.{m.group(1)}-{m.group(2)}-{m.group(3)}"
            if normalized in dockets:
                primary = normalized
        # prefer application dockets
        if primary is None:
            for d in dockets:
                if d.upper().startswith("A."):
                    primary = d
                    break
        if primary is None:
            primary = dockets[0]
        case_id = f"case:{primary}"
        case_label = primary
    else:
        base = source_path.stem
        case_id = f"case:{base}"
        case_label = base

    nodes.append(
        {
            "id": source_id,
            "label": source_path.name,
            "kind": "document",
            "category": "source",
            "score": 0,
        }
    )
    nodes.append(
        {
            "id": case_id,
            "label": case_label,
            "kind": "case",
            "category": "case",
            "score": SCORE_CAP,
        }
    )
    edges.append({"source": source_id, "target": case_id, "kind": "contains_certificate"})
    for docket in dockets:
        docket_id = f"docket:{docket}"
        nodes.append(
            {
                "id": docket_id,
                "label": docket,
                "kind": "docket",
                "category": "docket",
                "score": int(SCORE_CAP * 0.8),
            }
        )
        edges.append({"source": case_id, "target": docket_id, "kind": "served_on_service_list"})

    org_scores = {}
    by_domain = defaultdict(list)
    for contact in contacts:
        by_domain[contact.domain].append(contact)

    for domain, members in by_domain.items():
        organization, category = domain_to_org(domain)
        score, components, explanation, action = score_organization(
            domain=domain,
            category=category,
            contact_count=len(members),
            docket_count=len(dockets),
        )
        org_scores[domain] = {
            "domain": domain,
            "organization": organization,
            "category": category,
            "contact_count": len(members),
            "score": score,
            "score_components": components,
            "score_explanation": explanation,
            "recommended_action": action,
        }

    selected_contacts = select_contacts(contacts, org_scores, max_contacts, max_orgs)
    selected_domains = {contact.domain for contact in selected_contacts}

    for domain in sorted(selected_domains, key=lambda value: org_rank_key(org_scores[value])):
        org = org_scores[domain]
        org_id = f"org:{domain}"
        nodes.append(
            {
                "id": org_id,
                "label": org["organization"],
                "kind": "organization",
                "category": org["category"],
                "domain": domain,
                "contact_count": org["contact_count"],
                "score": org["score"],
                "score_components": org["score_components"],
                "score_explanation": org["score_explanation"],
                "recommended_action": org["recommended_action"],
            }
        )
        edges.append({"source": case_id, "target": org_id, "kind": "organization_served"})

    for contact in selected_contacts:
        contact_id = f"person:{contact.email}"
        org_id = f"org:{contact.domain}"
        nodes.append(
            {
                "id": contact_id,
                "label": contact.name,
                "kind": "person",
                "category": contact.category,
                "email": contact.email,
                "domain": contact.domain,
                "organization": contact.organization,
                "score": contact.score,
                "score_components": contact.score_components,
                "score_explanation": contact.score_explanation,
                "recommended_action": contact.recommended_action,
            }
        )
        edges.append({"source": org_id, "target": contact_id, "kind": "has_contact"})
        edges.append({"source": contact_id, "target": case_id, "kind": "served_on_case"})

    return {
        "source": str(source_path),
        "summary": {
            "unique_emails": len(contacts),
            "unique_domains": len(domain_counts),
            "dockets": dockets,
            "rendered_contacts": len(selected_contacts),
            "rendered_organizations": len(selected_domains),
        },
        "benchmarks": {
            "method": BENCHMARK_METHOD,
            "scoring_model": SCORING_MODEL,
            "recommendations": build_recommendations(org_scores),
            "top_contacts": [contact.__dict__ for contact in contacts[:50]],
            "top_organizations": sorted(
                org_scores.values(), key=org_rank_key
            )[:25],
            "category_counts": Counter(contact.category for contact in contacts),
        },
        "nodes": nodes,
        "edges": edges,
    }


def add_layout(graph: dict[str, Any]) -> None:
    width = 1100
    height = 720
    center_x = width / 2
    center_y = height / 2

    nodes = {node["id"]: node for node in graph["nodes"]}
    orgs = [node for node in graph["nodes"] if node["kind"] == "organization"]
    people = [node for node in graph["nodes"] if node["kind"] == "person"]
    dockets = [node for node in graph["nodes"] if node["kind"] == "docket"]

    # Position the case node at the center if present.
    case_nodes = [n for n in nodes.values() if n.get("kind") == "case"]
    if case_nodes:
        cnode = case_nodes[0]
        cnode["x"] = center_x
        cnode["y"] = center_y
    for index, node in enumerate(dockets):
        angle = 2 * math.pi * index / max(len(dockets), 1) - math.pi / 2
        node["x"] = center_x + math.cos(angle) * 115
        node["y"] = center_y + math.sin(angle) * 115
    for index, node in enumerate(orgs):
        angle = 2 * math.pi * index / max(len(orgs), 1) - math.pi / 2
        radius = 270 + (index % 3) * 35
        node["x"] = center_x + math.cos(angle) * radius
        node["y"] = center_y + math.sin(angle) * radius

    org_index = defaultdict(int)
    for person in people:
        org = nodes.get(f"org:{person['domain']}")
        if not org:
            continue
        slot = org_index[person["domain"]]
        org_index[person["domain"]] += 1
        angle = 2 * math.pi * slot / max(min(org.get("contact_count", 1), 12), 1)
        radius = 34 + 9 * (slot // 12)
        person["x"] = org["x"] + math.cos(angle) * radius
        person["y"] = org["y"] + math.sin(angle) * radius

    doc_node = next((node for node in graph["nodes"] if node["kind"] == "document"), None)
    if doc_node:
        doc_node["x"] = 120
        doc_node["y"] = 80


def render_html(graph: dict[str, Any]) -> str:
    data = json.dumps(graph, ensure_ascii=False)
    summary = graph["summary"]
    top_orgs = graph["benchmarks"]["top_organizations"][:8]
    top_contacts = graph["benchmarks"]["top_contacts"][:50]
    recommendations = graph["benchmarks"]["recommendations"]
    scoring_model = graph["benchmarks"]["scoring_model"]
    benchmark_method = graph["benchmarks"]["method"]

    org_rows = "\n".join(
        f"<tr><td>{html.escape(org['organization'])}</td><td>{org['contact_count']}</td>"
        f"<td>{org['category']}</td><td>{org['score']}</td>"
        f"<td>{html.escape(org['score_explanation'])}</td></tr>"
        for org in top_orgs
    )
    contact_rows = "\n".join(
        f"<tr data-email=\"{html.escape(contact['email'])}\">"
        f"<td>{html.escape(contact['name'])}</td><td>{html.escape(contact['organization'])}</td>"
        f"<td>{contact['category']}</td><td>{contact['score']}</td>"
        f"<td class=\"why-cell\">{html.escape(contact.get('score_explanation') or '')}</td>"
        f"<td class=\"fb-cell\">"
        f"<button class=\"fb-btn-row\" data-email=\"{html.escape(contact['email'])}\" data-label=\"positive\" title=\"Mark useful\">👍</button>"
        f"<button class=\"fb-btn-row\" data-email=\"{html.escape(contact['email'])}\" data-label=\"negative\" title=\"Skip\">👎</button>"
        f"<button class=\"fb-btn-row\" data-email=\"{html.escape(contact['email'])}\" data-label=\"clear\" title=\"Clear label\">✕</button>"
        f"</td></tr>"
        for contact in top_contacts
    )
    recommendation_cards = "\n".join(
        f"<div class=\"rec\"><strong>{html.escape(item['label'])}: {html.escape(item['target'])}</strong>"
        f"<small>{html.escape(item['target_type'])}</small>"
        f"<span>{html.escape(item['reason'])}</span></div>"
        for item in recommendations
    )
    scoring_rows = "\n".join(
        f"<tr><td>{html.escape(item['component'])}</td><td>{item['max_points']}</td>"
        f"<td>{html.escape(item['description'])}</td></tr>"
        for item in scoring_model
    )
    source_signal_items = "\n".join(
        f"<li>{html.escape(item)}</li>" for item in benchmark_method["created_from"]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ratecase Certificate Service Graph</title>
<style>
:root {{
  color-scheme: light;
  --ink: #172026;
  --muted: #5b6770;
  --line: #d7dee4;
  --panel: #f7f9fb;
  --regulator: #0b6e69;
  --utility: #a7421b;
  --law: #5252a3;
  --advocacy: #63710f;
  --market: #2364aa;
  --unknown: #68717a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: #ffffff;
}}
main {{ min-height: 100vh; }}
.shell {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
.stage {{ display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 18px; align-items: start; }}
.summary-panel {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 16px; }}
.lower-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; margin-top: 18px; }}
.wide-panel {{ grid-column: 1 / -1; }}
.panel {{ border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 16px; }}
.method-panel {{ background: #172026; color: #fff; }}
.method-panel p, .method-panel li {{ color: #d9e1e7; }}
.method-panel code {{ display: block; color: #fff; background: rgba(255,255,255,.10); border-radius: 6px; padding: 8px; margin: 8px 0; white-space: normal; }}
.method-panel h3 {{ margin: 18px 0 8px; font-size: 13px; color: #fff; letter-spacing: 0; }}
.method-panel table {{ background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.10); }}
.method-panel th {{ background: rgba(255,255,255,.06); color: #d9e1e7; }}
.method-panel td {{ color: #e8edf1; border-bottom-color: rgba(255,255,255,.08); }}
.method-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }}
.method-panel ul {{ margin: 8px 0 0; padding-left: 18px; }}
h1 {{ font-size: 24px; line-height: 1.15; margin: 0 0 8px; letter-spacing: 0; }}
h2 {{ font-size: 15px; margin: 22px 0 10px; letter-spacing: 0; }}
.panel h2, .summary-panel h2 {{ margin-top: 0; }}
p {{ color: var(--muted); margin: 0 0 14px; }}
.metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }}
.metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: #fff; }}
.metric strong {{ display: block; font-size: 21px; }}
.metric span {{ color: var(--muted); font-size: 12px; }}
canvas {{ width: 100%; max-width: 1100px; aspect-ratio: 1100 / 720; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; font-size: 12px; }}
th, td {{ padding: 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 600; background: #fbfcfd; }}
tr:last-child td {{ border-bottom: 0; }}
.rec-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }}
.rec {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: #fbfcfd; min-height: 92px; }}
.rec strong {{ display: block; font-size: 13px; margin-bottom: 4px; }}
.rec small {{ display: block; color: var(--muted); font-size: 11px; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; }}
.rec span {{ display: block; color: var(--muted); font-size: 12px; line-height: 1.35; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px; }}
.chip {{ display: inline-flex; gap: 6px; align-items: center; color: var(--muted); font-size: 12px; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--unknown); }}
.tooltip {{
  position: fixed;
  display: none;
  pointer-events: none;
  max-width: 280px;
  padding: 9px 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 10px 28px rgba(23, 32, 38, 0.12);
  font-size: 12px;
}}
.popover {{
  position: fixed;
  display: none;
  z-index: 99;
  min-width: 220px;
  max-width: 320px;
  padding: 11px 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 14px 36px rgba(23, 32, 38, 0.18);
  font-size: 12px;
}}
.popover strong {{ display: block; font-size: 13px; margin-bottom: 2px; }}
.popover small {{ display: block; color: var(--muted); margin-bottom: 8px; }}
.popover .fb-row {{ display: flex; gap: 6px; }}
.popover button, .fb-btn-row {{
  cursor: pointer;
  border: 1px solid var(--line);
  background: #fbfcfd;
  border-radius: 6px;
  padding: 4px 9px;
  font-size: 12px;
}}
.popover button.active-pos, .fb-btn-row.active-pos {{ background: #dcfce7; border-color: #86efac; }}
.popover button.active-neg, .fb-btn-row.active-neg {{ background: #fee2e2; border-color: #fca5a5; }}
.fb-cell {{ white-space: nowrap; }}
.fb-cell .fb-btn-row {{ padding: 2px 6px; margin-right: 2px; font-size: 11px; }}
.why-cell {{ font-size: 11px; color: var(--muted); line-height: 1.4; max-width: 360px; }}
.dl-row {{ display: flex; gap: 8px; }}
.dl-btn {{ display: inline-block; background: #172026; color: #fff; border-radius: 6px; padding: 6px 12px; font-size: 12px; text-decoration: none; cursor: pointer; }}
.dl-btn:hover {{ background: #303d46; }}
.dl-btn-ghost {{ background: #fff; color: #172026; border: 1px solid var(--line); }}
.dl-btn-ghost:hover {{ background: #f7f9fb; }}
.popover .fb-status {{ color: var(--muted); font-size: 11px; margin-top: 6px; min-height: 14px; }}
.popover .fb-close {{ position: absolute; top: 6px; right: 8px; border: 0; background: transparent; color: var(--muted); cursor: pointer; font-size: 14px; padding: 0; }}
tr.fb-positive td {{ background: #f0fdf4; }}
tr.fb-negative td {{ background: #fef2f2; }}
@media (max-width: 1180px) {{ .stage {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: repeat(4, 1fr); }} .rec-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
@media (max-width: 760px) {{ .shell {{ padding: 16px; }} .lower-grid, .method-grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: repeat(2, 1fr); }} .rec-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
  <div class="shell">
    <section class="stage">
      <div>
        <h1>Certificate of Service Stakeholder Graph</h1>
        <p>Extracted from {html.escape(Path(graph['source']).name)}. The graph links the source document, application, served dockets, organizations, and the highest-scoring contacts.</p>
        <div class="legend">
          <span class="chip"><span class="dot" style="background:var(--regulator)"></span>regulator</span>
          <span class="chip"><span class="dot" style="background:var(--utility)"></span>utility</span>
          <span class="chip"><span class="dot" style="background:var(--law)"></span>law firm</span>
          <span class="chip"><span class="dot" style="background:var(--advocacy)"></span>advocacy</span>
          <span class="chip"><span class="dot" style="background:var(--market)"></span>market</span>
        </div>
        <canvas id="graph" width="1100" height="720"></canvas>
      </div>
      <aside class="summary-panel">
        <h2>GTM Benchmark Output</h2>
        <p>Scores are built from named components, then capped at 100.</p>
        <div class="metrics">
          <div class="metric"><strong>{summary['unique_emails']}</strong><span>unique emails</span></div>
          <div class="metric"><strong>{summary['unique_domains']}</strong><span>organizations/domains</span></div>
          <div class="metric"><strong>{len(summary['dockets'])}</strong><span>served dockets</span></div>
          <div class="metric"><strong>{summary['rendered_contacts']}</strong><span>contacts rendered</span></div>
        </div>
      </aside>
    </section>
    <section class="lower-grid">
      <div class="panel wide-panel method-panel">
        <h2>{html.escape(benchmark_method['name'])}</h2>
        <p>{html.escape(benchmark_method['summary'])}</p>
        <div class="method-grid">
          <div>
            <strong>People score formula</strong>
            <code>{html.escape(benchmark_method['person_formula'])}</code>
            <strong>Organization score formula</strong>
            <code>{html.escape(benchmark_method['organization_formula'])}</code>
          </div>
          <div>
            <strong>Created from these filing signals</strong>
            <ul>{source_signal_items}</ul>
          </div>
        </div>
        <h3>Score components</h3>
        <table><thead><tr><th>Component</th><th>Max</th><th>Meaning</th></tr></thead><tbody>{scoring_rows}</tbody></table>
      </div>
      <div class="panel wide-panel">
        <h2>Benchmark Recommendations</h2>
        <div class="rec-grid">{recommendation_cards}</div>
      </div>
      <div class="panel wide-panel">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
          <h2 style="margin:0;">Top Contacts <small style="color:var(--muted);font-weight:400;font-size:11px;">(click 👍/👎 to teach the system)</small></h2>
          <div class="dl-row">
            <button class="dl-btn" id="generate-outreach" type="button">Generate outreach (top 25)</button>
            <a class="dl-btn dl-btn-ghost" id="dl-top25">Download top 25 (CSV)</a>
            <a class="dl-btn dl-btn-ghost" id="dl-all">Download all</a>
          </div>
        </div>
        <div id="outreach-status" style="margin-top:8px;font-size:12px;color:var(--muted);min-height:16px;"></div>
        <table><thead><tr><th>Contact</th><th>Organization</th><th>Type</th><th>Score</th><th>Why</th><th>Label</th></tr></thead><tbody>{contact_rows}</tbody></table>
      </div>
      <div class="panel wide-panel">
        <h2>Top Organizations</h2>
        <table><thead><tr><th>Organization</th><th>Contacts</th><th>Type</th><th>Score</th><th>Why</th></tr></thead><tbody>{org_rows}</tbody></table>
      </div>
    </section>
    <div id="tooltip" class="tooltip"></div>
    <div id="feedback-popover" class="popover">
      <button class="fb-close" type="button" aria-label="Close">×</button>
      <strong id="fb-name"></strong>
      <small id="fb-meta"></small>
      <div class="fb-row">
        <button data-label="positive" type="button">👍 Useful</button>
        <button data-label="negative" type="button">👎 Skip</button>
        <button data-label="clear" type="button">Clear</button>
      </div>
      <div class="fb-status" id="fb-status"></div>
    </div>
  </div>
</main>
<script>
const graph = {data};
const RUN_ID = window.GTM_RUN_ID || null;
const feedback = window.GTM_FEEDBACK || {{}};
const canvas = document.getElementById('graph');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const colors = {{
  regulator: '#0b6e69',
  utility: '#a7421b',
  law_firm: '#5252a3',
  advocacy: '#63710f',
  market_operator: '#2364aa',
  market_participant: '#2364aa',
  public_agency: '#48646f',
  docket: '#111827',
  case: '#111827',
  source: '#68717a',
  unknown: '#68717a'
}};
const nodes = new Map(graph.nodes.map(node => [node.id, node]));

function radius(node) {{
  if (node.kind === 'case') return 30;
  if (node.kind === 'docket') return 17;
  if (node.kind === 'organization') return Math.max(12, Math.min(28, 10 + Math.sqrt(node.contact_count || 1) * 3));
  if (node.kind === 'document') return 13;
  return Math.max(4, Math.min(8, 3 + node.score / 350));
}}

function nodeColor(node) {{
  return colors[node.category] || colors.unknown;
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 1;
  for (const edge of graph.edges) {{
    const a = nodes.get(edge.source);
    const b = nodes.get(edge.target);
    if (!a || !b || a.x == null || b.x == null) continue;
    ctx.strokeStyle = edge.kind === 'has_contact' ? 'rgba(91,103,112,.14)' : 'rgba(91,103,112,.28)';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }}
  for (const node of graph.nodes) {{
    if (node.x == null) continue;
    const r = radius(node);
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
    ctx.fillStyle = nodeColor(node);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = node.kind === 'person' ? 1 : 2;
    ctx.stroke();
    const fbLabel = node.email && feedback[node.email];
    if (fbLabel) {{
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 3, 0, Math.PI * 2);
      ctx.strokeStyle = fbLabel === 'positive' ? '#16a34a' : '#dc2626';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }}
    if (node.kind !== 'person') {{
      ctx.fillStyle = '#172026';
      ctx.font = node.kind === 'case' ? '600 13px system-ui' : '600 11px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      wrapLabel(node.label, node.x, node.y + r + 5, node.kind === 'case' ? 150 : 110);
    }}
  }}
}}

function wrapLabel(text, x, y, maxWidth) {{
  const words = String(text).split(' ');
  let line = '';
  let lineNo = 0;
  for (const word of words) {{
    const test = line ? line + ' ' + word : word;
    if (ctx.measureText(test).width > maxWidth && line) {{
      ctx.fillText(line, x, y + lineNo * 13);
      line = word;
      lineNo += 1;
    }} else {{
      line = test;
    }}
    if (lineNo > 1) break;
  }}
  if (line) ctx.fillText(line, x, y + lineNo * 13);
}}

function findNode(evt) {{
  const rect = canvas.getBoundingClientRect();
  const x = (evt.clientX - rect.left) * canvas.width / rect.width;
  const y = (evt.clientY - rect.top) * canvas.height / rect.height;
  for (let i = graph.nodes.length - 1; i >= 0; i--) {{
    const node = graph.nodes[i];
    if (node.x == null) continue;
    const dx = x - node.x;
    const dy = y - node.y;
    if (Math.sqrt(dx * dx + dy * dy) <= radius(node) + 4) return node;
  }}
  return null;
}}

canvas.addEventListener('mousemove', evt => {{
  const node = findNode(evt);
  if (!node) {{
    tooltip.style.display = 'none';
    return;
  }}
  const detail = node.email ? `<br>${{node.email}}` : node.domain ? `<br>${{node.domain}}` : '';
  const why = node.score_explanation ? `<br>${{node.score_explanation}}` : '';
  const action = node.recommended_action ? `<br><em>${{node.recommended_action}}</em>` : '';
  tooltip.innerHTML = `<strong>${{node.label}}</strong><br>${{node.kind}} · ${{node.category || 'n/a'}}${{detail}}<br>score: ${{node.score || 0}}${{why}}${{action}}`;
  tooltip.style.left = `${{evt.clientX + 14}}px`;
  tooltip.style.top = `${{evt.clientY + 14}}px`;
  tooltip.style.display = 'block';
}});
canvas.addEventListener('mouseleave', () => tooltip.style.display = 'none');

const popover = document.getElementById('feedback-popover');
const fbName = document.getElementById('fb-name');
const fbMeta = document.getElementById('fb-meta');
const fbStatus = document.getElementById('fb-status');
let activeContact = null;

function syncRow(email) {{
  const tr = document.querySelector(`tr[data-email="${{email}}"]`);
  if (!tr) return;
  tr.classList.remove('fb-positive', 'fb-negative');
  const label = feedback[email];
  if (label === 'positive') tr.classList.add('fb-positive');
  if (label === 'negative') tr.classList.add('fb-negative');
  tr.querySelectorAll('.fb-btn-row').forEach(b => {{
    b.classList.remove('active-pos', 'active-neg');
    if (label === 'positive' && b.dataset.label === 'positive') b.classList.add('active-pos');
    if (label === 'negative' && b.dataset.label === 'negative') b.classList.add('active-neg');
  }});
}}

function refreshAllRows() {{
  document.querySelectorAll('tr[data-email]').forEach(tr => syncRow(tr.dataset.email));
}}

function syncPopoverButtons(email) {{
  const label = feedback[email];
  popover.querySelectorAll('.fb-row button').forEach(b => {{
    b.classList.remove('active-pos', 'active-neg');
    if (label === 'positive' && b.dataset.label === 'positive') b.classList.add('active-pos');
    if (label === 'negative' && b.dataset.label === 'negative') b.classList.add('active-neg');
  }});
  fbStatus.textContent = label ? `Currently: ${{label}}` : 'No label yet';
}}

function openPopoverForEmail(email, anchorEvt) {{
  const node = graph.nodes.find(n => n.email === email);
  if (!node) return;
  activeContact = node;
  fbName.textContent = node.label || node.email;
  fbMeta.textContent = `${{node.email}} · ${{node.organization || ''}} · ${{node.category || ''}}`;
  syncPopoverButtons(email);
  let x, y;
  if (anchorEvt) {{
    x = anchorEvt.clientX + 14;
    y = anchorEvt.clientY + 14;
  }} else {{
    const rect = canvas.getBoundingClientRect();
    x = node.x * rect.width / canvas.width + rect.left + 14;
    y = node.y * rect.height / canvas.height + rect.top + 14;
  }}
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  popover.style.display = 'block';
  const pw = popover.offsetWidth;
  const ph = popover.offsetHeight;
  popover.style.left = `${{Math.min(x, vw - pw - 10)}}px`;
  popover.style.top = `${{Math.min(y, vh - ph - 10)}}px`;
}}

canvas.addEventListener('click', evt => {{
  const node = findNode(evt);
  if (!node || node.kind !== 'person' || !node.email) {{
    popover.style.display = 'none';
    activeContact = null;
    return;
  }}
  openPopoverForEmail(node.email, evt);
}});

popover.querySelector('.fb-close').addEventListener('click', () => {{
  popover.style.display = 'none';
  activeContact = null;
}});

document.addEventListener('keydown', evt => {{
  if (evt.key === 'Escape') {{ popover.style.display = 'none'; activeContact = null; }}
}});

async function submitLabel(email, label) {{
  if (!RUN_ID) {{
    fbStatus.textContent = 'Cannot save: not served via dashboard.';
    return false;
  }}
  const fd = new FormData();
  fd.append('contact_email', email);
  fd.append('label', label);
  const res = await fetch(`/run/${{RUN_ID}}/feedback`, {{method: 'POST', body: fd}});
  if (!res.ok) {{
    fbStatus.textContent = `Save failed (${{res.status}})`;
    return false;
  }}
  if (label === 'clear') delete feedback[email];
  else feedback[email] = label;
  syncRow(email);
  draw();
  return true;
}}

popover.querySelectorAll('.fb-row button').forEach(btn => {{
  btn.addEventListener('click', async () => {{
    if (!activeContact) return;
    const label = btn.dataset.label;
    const ok = await submitLabel(activeContact.email, label);
    if (ok) {{
      syncPopoverButtons(activeContact.email);
      fbStatus.textContent = label === 'clear' ? 'Cleared' : `Saved: ${{label}}`;
    }}
  }});
}});

document.querySelectorAll('.fb-btn-row').forEach(btn => {{
  btn.addEventListener('click', async (evt) => {{
    evt.stopPropagation();
    const email = btn.dataset.email;
    const label = btn.dataset.label;
    await submitLabel(email, label);
  }});
}});

refreshAllRows();

const dlTop = document.getElementById('dl-top25');
const dlAll = document.getElementById('dl-all');
const genBtn = document.getElementById('generate-outreach');
const genStatus = document.getElementById('outreach-status');
if (RUN_ID) {{
  if (dlTop) dlTop.href = `/run/${{RUN_ID}}/csv?limit=25`;
  if (dlAll) dlAll.href = `/run/${{RUN_ID}}/csv?limit=all`;
}} else {{
  if (dlTop) dlTop.style.display = 'none';
  if (dlAll) dlAll.style.display = 'none';
  if (genBtn) genBtn.style.display = 'none';
}}

if (genBtn && RUN_ID) {{
  genBtn.addEventListener('click', async () => {{
    genBtn.disabled = true;
    const startedAt = Date.now();
    genStatus.textContent = 'Generating outreach for top 25 (Tavily research + email drafts; ~30s)…';
    try {{
      const res = await fetch(`/run/${{RUN_ID}}/generate-outreach?limit=25`, {{ method: 'POST' }});
      const elapsed = Math.round((Date.now() - startedAt) / 1000);
      if (!res.ok) {{
        const detail = await res.text();
        genStatus.style.color = '#dc2626';
        genStatus.textContent = `Generation failed (${{res.status}}): ${{detail.slice(0, 200)}}`;
        return;
      }}
      const data = await res.json();
      genStatus.style.color = data.errors ? '#a16207' : '#16a34a';
      const tavilyNote = data.tavily_enabled ? '' : ' (Tavily key missing — drafts use no web context)';
      genStatus.textContent = `✓ Drafts generated for ${{data.generated}} contacts in ${{elapsed}}s${{tavilyNote}}${{data.errors ? `, ${{data.errors}} failed (first: ${{data.first_error}})` : ''}}. Re-download the CSV to include outreach_subject, outreach_body, research_summary columns.`;
    }} catch (err) {{
      genStatus.style.color = '#dc2626';
      genStatus.textContent = `Network error: ${{err.message}}`;
    }} finally {{
      genBtn.disabled = false;
    }}
  }});
}}

draw();
</script>
</body>
</html>
"""


def write_csv(graph: dict[str, Any], output_csv: Path) -> None:
    """Write top contacts to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    top = graph.get("benchmarks", {}).get("top_contacts", [])
    fieldnames = ["email","name","organization","domain","category","score","recommended_action","score_explanation"]
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for c in top:
            writer.writerow({
                "email": c.get("email"),
                "name": c.get("name"),
                "organization": c.get("organization"),
                "domain": c.get("domain"),
                "category": c.get("category"),
                "score": c.get("score"),
                "recommended_action": c.get("recommended_action"),
                "score_explanation": c.get("score_explanation"),
            })


def write_outputs(graph: dict[str, Any], output_json: Path, output_html: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    output_html.write_text(render_html(graph), encoding="utf-8")


def build_graph_from_path(
    source_path: Path,
    max_contacts: int = 180,
    max_orgs: int = 35,
) -> dict[str, Any]:
    """In-process entrypoint: extract, build, and lay out the graph for one file."""
    text = extract_text(source_path)
    graph = build_graph(text, source_path, max_contacts=max_contacts, max_orgs=max_orgs)
    add_layout(graph)
    return graph


PERSON_FEEDBACK_DELTA = 150
ORG_FEEDBACK_PER_LABEL = 50
ORG_FEEDBACK_CAP = 200

CATEGORY_PRIOR_PER_LABEL = 6
CATEGORY_PRIOR_CAP = 60
DOMAIN_PRIOR_PER_LABEL = 12
DOMAIN_PRIOR_CAP = 120


def _signed_delta(positive: int, negative: int, per_label: int, cap: int) -> int:
    raw = (positive - negative) * per_label
    return max(-cap, min(cap, raw))


def compute_prior_deltas(aggregates: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Convert raw category/domain feedback counts into bounded score deltas."""
    cat_deltas: dict[str, int] = {}
    for cat, counts in (aggregates.get("categories") or {}).items():
        delta = _signed_delta(counts.get("positive", 0), counts.get("negative", 0), CATEGORY_PRIOR_PER_LABEL, CATEGORY_PRIOR_CAP)
        if delta:
            cat_deltas[cat] = delta
    dom_deltas: dict[str, int] = {}
    for dom, counts in (aggregates.get("domains") or {}).items():
        delta = _signed_delta(counts.get("positive", 0), counts.get("negative", 0), DOMAIN_PRIOR_PER_LABEL, DOMAIN_PRIOR_CAP)
        if delta:
            dom_deltas[dom] = delta
    return {"categories": cat_deltas, "domains": dom_deltas}


def apply_learned_priors(graph: dict[str, Any], aggregates: dict[str, Any]) -> dict[str, Any]:
    """Bias baseline person/org scores by cross-run feedback aggregates.

    aggregates shape matches store.feedback_aggregates(): {categories, domains, totals}.
    Mutates graph in-place (also stores deltas under benchmarks.learned_priors) and returns it.
    """
    if not aggregates or not aggregates.get("totals", {}).get("labeled_contacts"):
        return graph

    deltas = compute_prior_deltas(aggregates)
    cat_d = deltas["categories"]
    dom_d = deltas["domains"]
    if not cat_d and not dom_d:
        graph.setdefault("benchmarks", {})["learned_priors"] = {
            "totals": aggregates.get("totals", {}),
            "categories": {},
            "domains": {},
            "applied": False,
        }
        return graph

    def _bias(cat: str, dom: str) -> int:
        return cat_d.get(cat, 0) + dom_d.get((dom or "").lower(), 0)

    for node in graph.get("nodes", []):
        if node.get("kind") == "person":
            bias = _bias(node.get("category", ""), node.get("domain", ""))
            if bias:
                node["score"] = max(0, min(SCORE_CAP, node.get("score", 0) + bias))
                comps = dict(node.get("score_components") or {})
                comps["learned_prior"] = bias
                node["score_components"] = comps
                node["score_explanation"] = (node.get("score_explanation") or "").rstrip(".") + f". Learned prior: {bias:+d}."
        elif node.get("kind") == "organization":
            bias = _bias(node.get("category", ""), node.get("domain", ""))
            if bias:
                node["score"] = max(0, min(SCORE_CAP, node.get("score", 0) + bias))
                comps = dict(node.get("score_components") or {})
                comps["learned_prior"] = bias
                node["score_components"] = comps
                node["score_explanation"] = (node.get("score_explanation") or "").rstrip(".") + f". Learned prior: {bias:+d}."

    bench = graph.setdefault("benchmarks", {})
    for contact in bench.get("top_contacts", []):
        bias = _bias(contact.get("category", ""), contact.get("domain", ""))
        if bias:
            contact["score"] = max(0, min(SCORE_CAP, contact.get("score", 0) + bias))
            comps = dict(contact.get("score_components") or {})
            comps["learned_prior"] = bias
            contact["score_components"] = comps
            contact["score_explanation"] = (contact.get("score_explanation") or "").rstrip(".") + f". Learned prior: {bias:+d}."
    bench["top_contacts"] = sorted(
        bench.get("top_contacts", []),
        key=lambda c: (-c.get("score", 0), c.get("email", "")),
    )

    for org in bench.get("top_organizations", []):
        bias = _bias(org.get("category", ""), org.get("domain", ""))
        if bias:
            org["score"] = max(0, min(SCORE_CAP, org.get("score", 0) + bias))
            comps = dict(org.get("score_components") or {})
            comps["learned_prior"] = bias
            org["score_components"] = comps
            org["score_explanation"] = (org.get("score_explanation") or "").rstrip(".") + f". Learned prior: {bias:+d}."
    bench["top_organizations"] = sorted(
        bench.get("top_organizations", []),
        key=lambda o: (-o.get("score", 0), -o.get("contact_count", 0), o.get("domain", "")),
    )

    bench["learned_priors"] = {
        "totals": aggregates.get("totals", {}),
        "categories": cat_d,
        "domains": dom_d,
        "applied": True,
    }
    return graph


def adjust_graph_with_feedback(graph: dict[str, Any], feedback: dict[str, str]) -> dict[str, Any]:
    """Apply per-contact feedback labels to person and organization scores in-place.

    Person nodes: +15 for positive, -15 for negative (capped 0..100).
    Org nodes: +/-5 per labeled contact, clamped to ±20 total, then capped 0..100.
    Re-sorts top_contacts and top_organizations so the table reflects new ranks.
    """
    if not feedback:
        return graph

    org_label_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})

    def person_delta(label: str) -> int:
        return PERSON_FEEDBACK_DELTA if label == "positive" else -PERSON_FEEDBACK_DELTA

    def org_delta(counts: dict[str, int]) -> int:
        raw = (counts["positive"] - counts["negative"]) * ORG_FEEDBACK_PER_LABEL
        return max(-ORG_FEEDBACK_CAP, min(ORG_FEEDBACK_CAP, raw))

    for node in graph.get("nodes", []):
        if node.get("kind") == "person" and node.get("email") in feedback:
            label = feedback[node["email"]]
            delta = person_delta(label)
            node["score"] = max(0, min(SCORE_CAP, node.get("score", 0) + delta))
            components = dict(node.get("score_components") or {})
            components["feedback"] = delta
            node["score_components"] = components
            node["score_explanation"] = (node.get("score_explanation") or "").rstrip(".") + f". Feedback {label}: {delta:+d}."
            domain = node.get("domain")
            if domain:
                org_label_counts[domain][label] += 1

    for node in graph.get("nodes", []):
        if node.get("kind") == "organization":
            domain = node.get("domain")
            if domain in org_label_counts:
                counts = org_label_counts[domain]
                delta = org_delta(counts)
                node["score"] = max(0, min(SCORE_CAP, node.get("score", 0) + delta))
                components = dict(node.get("score_components") or {})
                components["feedback"] = delta
                node["score_components"] = components
                node["score_explanation"] = (node.get("score_explanation") or "").rstrip(".") + (
                    f". Feedback signal {delta:+d} ({counts['positive']} positive, {counts['negative']} negative)."
                )

    bench = graph.setdefault("benchmarks", {})

    existing_emails = {c.get("email") for c in bench.get("top_contacts", [])}
    for node in graph.get("nodes", []):
        if (
            node.get("kind") == "person"
            and node.get("email") in feedback
            and node.get("email") not in existing_emails
        ):
            bench.setdefault("top_contacts", []).append({
                "email": node.get("email"),
                "name": node.get("label", node.get("email")),
                "organization": node.get("organization", ""),
                "domain": node.get("domain", ""),
                "category": node.get("category", "unknown"),
                "score": node.get("score", 0),
                "score_components": dict(node.get("score_components") or {}),
                "score_explanation": node.get("score_explanation", ""),
                "recommended_action": node.get("recommended_action", ""),
            })
            existing_emails.add(node.get("email"))

    for contact in bench.get("top_contacts", []):
        if contact.get("email") in feedback:
            label = feedback[contact["email"]]
            delta = person_delta(label)
            contact["score"] = max(0, min(SCORE_CAP, contact.get("score", 0) + delta))
            components = dict(contact.get("score_components") or {})
            components["feedback"] = delta
            contact["score_components"] = components
            contact["score_explanation"] = (contact.get("score_explanation") or "").rstrip(".") + f". Feedback {label}: {delta:+d}."
    def _fb_priority(item: dict[str, Any]) -> int:
        fb = (item.get("score_components") or {}).get("feedback", 0)
        if fb > 0:
            return -1
        if fb < 0:
            return 1
        return 0

    def _raw_score(item: dict[str, Any]) -> int:
        components = item.get("score_components") or {}
        return sum(components.values())

    bench["top_contacts"] = sorted(
        bench.get("top_contacts", []),
        key=lambda c: (_fb_priority(c), -_raw_score(c), -c.get("score", 0), c.get("email", "")),
    )

    for org in bench.get("top_organizations", []):
        domain = org.get("domain")
        if domain in org_label_counts:
            counts = org_label_counts[domain]
            delta = org_delta(counts)
            org["score"] = max(0, min(100, org.get("score", 0) + delta))
            components = dict(org.get("score_components") or {})
            components["feedback"] = delta
            org["score_components"] = components
            org["score_explanation"] = (org.get("score_explanation") or "").rstrip(".") + (
                f". Feedback signal {delta:+d} ({counts['positive']} positive, {counts['negative']} negative)."
            )
    bench["top_organizations"] = sorted(
        bench.get("top_organizations", []),
        key=lambda o: (_fb_priority(o), -_raw_score(o), -o.get("score", 0), -o.get("contact_count", 0), o.get("domain", "")),
    )

    bench["feedback_summary"] = {
        "positive": sum(1 for v in feedback.values() if v == "positive"),
        "negative": sum(1 for v in feedback.values() if v == "negative"),
        "total_labeled": len(feedback),
        "person_delta": PERSON_FEEDBACK_DELTA,
        "org_delta_per_label": ORG_FEEDBACK_PER_LABEL,
        "org_delta_cap": ORG_FEEDBACK_CAP,
    }
    return graph


def graph_to_csv_rows(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the same contact rows that write_csv emits, without touching disk."""
    fieldnames = ["email", "name", "organization", "domain", "category", "score", "recommended_action", "score_explanation"]
    return [
        {field: contact.get(field) for field in fieldnames}
        for contact in graph.get("benchmarks", {}).get("top_contacts", [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Certificate PDF or extracted .txt")
    parser.add_argument("--output-json", required=True, help="Path for graph JSON")
    parser.add_argument("--output-html", required=True, help="Path for self-contained HTML graph")
    parser.add_argument("--max-contacts", type=int, default=180, help="Contacts to render in HTML")
    parser.add_argument("--max-orgs", type=int, default=35, help="Organizations to render in HTML")
    parser.add_argument('--output-csv', help='Optional CSV path for top contacts')
    args = parser.parse_args()

    source_path = Path(args.input)
    text = extract_text(source_path)
    graph = build_graph(text, source_path, max_contacts=args.max_contacts, max_orgs=args.max_orgs)
    add_layout(graph)
    write_outputs(graph, Path(args.output_json), Path(args.output_html))
    if getattr(args, 'output_csv', None):
        write_csv(graph, Path(args.output_csv))

    summary = graph["summary"]
    print(
        f"Extracted {summary['unique_emails']} emails across {summary['unique_domains']} domains; "
        f"rendered {summary['rendered_contacts']} contacts and {summary['rendered_organizations']} orgs."
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
