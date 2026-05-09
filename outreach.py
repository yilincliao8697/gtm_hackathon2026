"""Tavily research + OpenAI email drafting for GTM outreach."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
load_dotenv(BASE.parent / ".env", override=False)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = "gpt-4o-mini"
TAVILY_URL = "https://api.tavily.com/search"
PARALLEL = 5


def keys_status() -> dict[str, bool]:
    return {"tavily": bool(TAVILY_API_KEY), "openai": bool(OPENAI_API_KEY)}


async def tavily_search(client: httpx.AsyncClient, query: str, max_results: int = 3) -> list[dict[str, Any]]:
    if not TAVILY_API_KEY:
        return []
    try:
        r = await client.post(
            TAVILY_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=20.0,
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except httpx.HTTPError:
        return []


def _summarize_results(results: list[dict[str, Any]]) -> str:
    parts = []
    for r in results[:3]:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()[:280]
        if title or content:
            parts.append(f"- {title}: {content}")
    return "\n".join(parts)


async def research_organization(
    client: httpx.AsyncClient, organization: str, case_label: str
) -> dict[str, Any]:
    if not organization:
        return {"summary": "", "sources": []}
    query = f"{organization} CPUC California utility regulation {case_label}".strip()
    results = await tavily_search(client, query, max_results=3)
    return {
        "summary": _summarize_results(results),
        "sources": [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results[:3]],
    }


def _email_prompt(contact: dict[str, Any], case_label: str, research_summary: str) -> str:
    name = contact.get("name") or contact.get("email") or "the contact"
    organization = contact.get("organization") or "their organization"
    category = (contact.get("category") or "stakeholder").replace("_", " ")
    why = contact.get("score_explanation") or ""
    research_block = f"Recent context from web research:\n{research_summary}" if research_summary else "No additional public research was available."
    return f"""You are drafting a sales outreach email for a product that helps parties in CPUC rate-case proceedings justify their rates faster and more efficiently — so utilities get the rates they need while end customers are treated fairly.

Contact: {name} ({contact.get("email")})
Organization: {organization}
Stakeholder type: {category}
Why this contact matters: {why}
Case docket: {case_label}

{research_block}

Write a single email that:
- Opens with ONE sentence personalizing the problem to this specific contact and their organization — what's painful or at stake for *them* in this proceeding (lean on the research and stakeholder type; never invent facts). Cite the docket here.
- Follows with ONE sentence on how our product helps: faster, more efficient rate justification — securing the rates they need while keeping outcomes fair to end customers.
- Asks a specific, short question — invite a 15-minute conversation OR a pointer to the right person.
- Stays under 130 words. No hype, no exclamation marks, plain text only.

Return JSON with two fields: subject (under 60 characters) and body (the email body, no greeting/signature placeholders beyond a generic "Hi <First Name>," and "Best,\\n[Your name]")."""


async def draft_email(
    contact: dict[str, Any], case_label: str, research_summary: str
) -> dict[str, str]:
    if not OPENAI_API_KEY:
        return {"subject": "", "body": "", "error": "OPENAI_API_KEY not set"}
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return {"subject": "", "body": "", "error": "openai package not installed"}

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You draft concise, professional outreach emails. Output strictly valid JSON."},
                {"role": "user", "content": _email_prompt(contact, case_label, research_summary)},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            timeout=25.0,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        return {
            "subject": (data.get("subject") or "").strip(),
            "body": (data.get("body") or "").strip(),
            "error": "",
        }
    except Exception as exc:
        return {"subject": "", "body": "", "error": f"{type(exc).__name__}: {exc}"}


async def _generate_one(
    http_client: httpx.AsyncClient,
    contact: dict[str, Any],
    case_label: str,
    org_research_cache: dict[str, dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    organization = contact.get("organization") or ""
    async with semaphore:
        if organization and organization not in org_research_cache:
            org_research_cache[organization] = await research_organization(http_client, organization, case_label)
        research = org_research_cache.get(organization, {"summary": "", "sources": []})
        email = await draft_email(contact, case_label, research["summary"])
    return {
        "contact_email": (contact.get("email") or "").lower(),
        "subject": email["subject"],
        "body": email["body"],
        "research_summary": research["summary"],
        "error": email.get("error", ""),
    }


async def generate_for_contacts(
    contacts: list[dict[str, Any]], case_label: str
) -> list[dict[str, Any]]:
    if not contacts:
        return []
    org_cache: dict[str, dict[str, Any]] = {}
    semaphore = asyncio.Semaphore(PARALLEL)
    async with httpx.AsyncClient() as http_client:
        tasks = [_generate_one(http_client, c, case_label, org_cache, semaphore) for c in contacts]
        return await asyncio.gather(*tasks)
