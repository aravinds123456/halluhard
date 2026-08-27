"""HalluHard webscraper evidence for cascade seed claims.

This is the structured-analysis step reused on cascade turn-0 answers:
extract checkable particulars, Serper-search, fetch top pages/PDFs, filter
relevant passages, then judge each claim. The LLM does not get to call a
true textbook mechanism a hallucination just because it lacks a citation.

Hallucinating only if a particular is contradicted or fabricated against
that evidence. Thin snippets, a failed fetch, or a true textbook mechanism
stay Not Hallucinating.

Pass --no-web or CASCADE_WEB=0 for the LLM-only fallback (not the paper
path). Pass CASCADE_WEB_FETCH=0 to keep Serper snippets without page fetch.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Callable

from cascade import DEFAULT_JUDGE_REASONING_EFFORT, env_str
from prompts_pack import fill_prompt

# forecasting/web_verify.py -> repo root, so `import libs.*` works.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FALSE_VERDICTS = ("contradicted", "fabricated")
MAX_CLAIMS = 3
SEARCH_CHARS = 300
MAX_URLS_TO_FETCH = 2
MAX_PDFS_TO_FETCH = 1
MAX_FILTER_WORDS = 1500
MAX_EVIDENCE_CHARS = 8000
MIN_PAGE_CHARS = 100

GptFn = Callable[..., Any]
SearchFn = Callable[[str], tuple[str, str, str | None]]
# claim, snippets, urls -> (filtered_text, page records, error)
FetchFn = Callable[[str, str, list[str]], tuple[str, list[dict], str | None]]


def invoke_gpt(gpt_fn: GptFn, prompt: str, *, role: str = "judge", as_json: bool = True):
    """Call gpt_fn with HalluHard effort split; tests may omit the extra kwargs."""
    try:
        return gpt_fn(prompt, as_json=as_json, role=role)
    except TypeError:
        try:
            return gpt_fn(prompt, as_json=as_json)
        except TypeError:
            return gpt_fn(prompt)


def web_flag_disabled() -> bool:
    if env_str("CASCADE_WEB", "1").lower() in {"0", "false", "no", "off"}:
        return True
    return "--no-web" in sys.argv or "--no_web" in sys.argv


def fetch_flag_disabled() -> bool:
    return env_str("CASCADE_WEB_FETCH", "1").lower() in {"0", "false", "no", "off"}


def serper_configured() -> bool:
    return bool(os.environ.get("SERPER_API_KEY", "").strip())


def web_requested() -> bool:
    return not web_flag_disabled()


def require_serper_unless_disabled(*, dry_run: bool = False) -> None:
    if dry_run or web_flag_disabled():
        return
    if not serper_configured():
        raise SystemExit(
            "Cascade seed judging uses HalluHard webscraper evidence (Serper "
            "search + page/PDF fetch). Set SERPER_API_KEY, or pass --no-web / "
            "CASCADE_WEB=0 for the LLM-only fallback. That fallback will "
            "over-label true mechanisms as Hallucinating and is not the paper path."
        )


def format_serper_payload(results: dict) -> str:
    """Reuse the HalluHard Serper formatter without spinning a planner LLM."""
    from libs.serper.client import SerperSearchClient

    client = SerperSearchClient.__new__(SerperSearchClient)
    return SerperSearchClient._format_single_result(client, results or {})


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("web_verify cannot nest inside a running event loop")


def _organic_hits(raw: dict) -> list[dict]:
    rows = []
    for item in raw.get("organic") or []:
        url = str(item.get("link") or "").strip()
        if not url:
            continue
        rows.append(
            {
                "title": str(item.get("title") or ""),
                "url": url,
                "snippet": str(item.get("snippet") or ""),
            }
        )
    return rows


def _truncate_words(text: str, max_words: int = MAX_FILTER_WORDS) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).strip()


def format_evidence(pages_text: str, snippets: str) -> str:
    snippets = (snippets or "").strip() or "No search results found."
    pages_text = (pages_text or "").strip()
    if pages_text:
        return (
            f"Fetched page/PDF passages:\n{pages_text}\n\n"
            f"Serper snippets:\n{snippets}"
        )[:MAX_EVIDENCE_CHARS]
    return (
        "Serper snippets (page fetch failed or returned nothing):\n" + snippets
    )[:MAX_EVIDENCE_CHARS]


async def _serper_search_async(claim: str, num_results: int = 5) -> dict:
    from libs.serper import SerperSearchClient

    query = (claim or "").strip()[:SEARCH_CHARS]
    if not query:
        return {
            "snippets": "",
            "query": "",
            "error": "empty claim",
            "hits": [],
            "urls": [],
        }

    async with SerperSearchClient() as client:
        raw, _n = await client.search(query, num_results=num_results)
        hits = _organic_hits(raw or {})
        return {
            "snippets": client._format_single_result(raw),
            "query": query,
            "error": None,
            "hits": hits,
            "urls": [hit["url"] for hit in hits],
        }


def serper_search(claim: str, num_results: int = 5) -> tuple[str, str, str | None]:
    try:
        payload = _run_async(_serper_search_async(claim, num_results=num_results))
    except Exception as error:
        return "", (claim or "").strip()[:SEARCH_CHARS], f"{type(error).__name__}: {error}"
    return payload["snippets"], payload["query"], payload["error"]


async def _fetch_one_url(url: str, title: str = "", snippet: str = "") -> dict | None:
    """Fetch one HTML page or PDF the same way HalluHard webscraper does."""
    from libs.browser_fetcher import BrowserFetcher, extract_pdf_links
    from libs.html_cleaner import HtmlCleaner
    from libs.information_extraction import check_if_url_is_pdf, extract_pdf_as_markdown

    try:
        if await check_if_url_is_pdf(url):
            markdown = await extract_pdf_as_markdown(url)
            if not markdown or len(markdown.strip()) < MIN_PAGE_CHARS:
                return None
            return {
                "title": title or "PDF",
                "url": url,
                "snippet": snippet,
                "content": markdown,
                "kind": "pdf",
            }

        fetcher = BrowserFetcher()
        html, error = await fetcher.fetch_html(url, force_selenium=False)
        if error or not html:
            return None
        pdf_links = extract_pdf_links(html, base_url=url)
        cleaned = HtmlCleaner().clean(html, source_url=url)
        if not cleaned or len(cleaned.strip()) < MIN_PAGE_CHARS:
            return None
        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "content": cleaned,
            "kind": "html",
            "pdf_links": pdf_links or [],
        }
    except Exception:
        return None


async def _filter_pages(claim: str, pages: list[dict]) -> str:
    if not pages:
        return ""
    try:
        from libs.information_extraction import extract_relevant_sentences

        filtered, _calls, _blocks = await extract_relevant_sentences(
            websearch_results=pages,
            claim=claim,
            max_output_words=MAX_FILTER_WORDS,
            embedding_semaphore=asyncio.Semaphore(8),
            block_size=3000,
            overlap=200,
        )
        if (filtered or "").strip():
            return filtered.strip()
    except Exception:
        pass
    joined = "\n\n".join(
        f"{page.get('title') or page.get('url') or 'source'}:\n{page.get('content') or ''}"
        for page in pages
        if page.get("content")
    )
    return _truncate_words(joined)


async def _fetch_and_filter_async(
    claim: str,
    snippets: str,
    urls: list[str],
    hits: list[dict] | None = None,
) -> tuple[str, list[dict], str | None]:
    from libs.information_extraction import is_pdf_url

    info = {hit["url"]: hit for hit in (hits or []) if hit.get("url")}
    ordered = []
    seen = set()
    for url in urls or []:
        if url and url not in seen:
            ordered.append(url)
            seen.add(url)
    html_urls = [url for url in ordered if not is_pdf_url(url)][:MAX_URLS_TO_FETCH]
    pdf_urls = [url for url in ordered if is_pdf_url(url)][:MAX_PDFS_TO_FETCH]

    pages: list[dict] = []
    extra_pdfs: list[str] = []
    for url in html_urls:
        hit = info.get(url, {})
        page = await _fetch_one_url(url, title=hit.get("title", ""), snippet=hit.get("snippet", ""))
        if not page:
            continue
        pages.append(page)
        for pdf_url in page.get("pdf_links") or []:
            if pdf_url not in seen:
                extra_pdfs.append(pdf_url)
                seen.add(pdf_url)

    for url in (pdf_urls + extra_pdfs)[:MAX_PDFS_TO_FETCH]:
        hit = info.get(url, {})
        page = await _fetch_one_url(url, title=hit.get("title", "") or "PDF", snippet=hit.get("snippet", ""))
        if page:
            pages.append(page)

    if not pages:
        return "", [], "page fetch failed or returned nothing"

    filtered = await _filter_pages(claim, pages)
    records = [
        {
            "title": page.get("title") or "",
            "url": page.get("url") or "",
            "kind": page.get("kind") or "html",
        }
        for page in pages
    ]
    return filtered, records, None


def fetch_and_filter(
    claim: str,
    snippets: str,
    urls: list[str],
    hits: list[dict] | None = None,
) -> tuple[str, list[dict], str | None]:
    try:
        return _run_async(_fetch_and_filter_async(claim, snippets, urls, hits=hits))
    except Exception as error:
        return "", [], f"{type(error).__name__}: {error}"


def _as_dict(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    text = str(payload or "").strip()
    if not text:
        return {}
    if "```" in text:
        import re

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def extract_candidates(question: str, answer: str, gpt_fn: GptFn, max_claims: int = MAX_CLAIMS) -> list[dict]:
    payload = _as_dict(
        invoke_gpt(
            gpt_fn,
            fill_prompt(
                "claim_candidates",
                question=question[:1500],
                answer=answer[:4000],
                max_claims=str(max_claims),
            ),
            role="aux",
        )
    )
    raw_claims = payload.get("claims")
    if raw_claims is None and payload.get("claim"):
        raw_claims = [payload]
    rows = []
    for item in raw_claims or []:
        if isinstance(item, str):
            text = item.strip()
            entities: list[str] = []
        elif isinstance(item, dict):
            text = str(item.get("claim") or "").strip()
            entities = [str(e) for e in (item.get("entities") or []) if str(e).strip()][:4]
        else:
            continue
        if text:
            rows.append({"claim": text[:800], "entities": entities})
        if len(rows) >= max_claims:
            break
    return rows


def judge_claim_against_evidence(claim: str, evidence: str, gpt_fn: GptFn) -> dict:
    payload = _as_dict(
        invoke_gpt(
            gpt_fn,
            fill_prompt(
                "web_claim_judge",
                claim=claim[:800],
                evidence=(evidence or "No search results found.")[:MAX_EVIDENCE_CHARS],
                snippets=(evidence or "No search results found.")[:MAX_EVIDENCE_CHARS],
            ),
            role="judge",
        )
    )
    verdict = str(payload.get("verdict") or "insufficient").strip().lower()
    if verdict not in {"supported", "contradicted", "fabricated", "insufficient"}:
        verdict = "insufficient"
    return {
        "verdict": verdict,
        "reason": str(payload.get("reason") or "").strip(),
    }


def judge_claim_against_snippets(claim: str, snippets: str, gpt_fn: GptFn) -> dict:
    """Back-compat wrapper; the judge now reads fetched pages when present."""
    return judge_claim_against_evidence(claim, snippets, gpt_fn)


def pick_false_claim(claim_rows: list[dict]) -> dict | None:
    contradicted = [row for row in claim_rows if row.get("verdict") == "contradicted"]
    fabricated = [row for row in claim_rows if row.get("verdict") == "fabricated"]
    if contradicted:
        return contradicted[0]
    if fabricated:
        return fabricated[0]
    return None


async def _live_evidence_async(claim: str) -> dict:
    payload = await _serper_search_async(claim)
    snippets = payload["snippets"]
    query = payload["query"]
    error = payload["error"]
    urls = payload["urls"]
    hits = payload["hits"]
    pages_text = ""
    pages: list[dict] = []
    evidence_kind = "snippets"
    if not fetch_flag_disabled() and urls and not error:
        pages_text, pages, fetch_error = await _fetch_and_filter_async(
            claim, snippets, urls, hits=hits
        )
        if pages_text:
            evidence_kind = "pages"
        elif fetch_error and not error:
            error = fetch_error
    return {
        "snippets": snippets,
        "query": query,
        "error": error,
        "pages_text": pages_text,
        "pages": pages,
        "evidence_kind": evidence_kind,
        "urls": urls,
    }


def _live_evidence(claim: str) -> dict:
    try:
        return _run_async(_live_evidence_async(claim))
    except Exception as error:
        return {
            "snippets": "",
            "query": (claim or "").strip()[:SEARCH_CHARS],
            "error": f"{type(error).__name__}: {error}",
            "pages_text": "",
            "pages": [],
            "evidence_kind": "snippets",
            "urls": [],
        }


def verify_seed_answer(
    question: str,
    answer: str,
    gpt_fn: GptFn,
    *,
    search_fn: SearchFn | None = None,
    fetch_fn: FetchFn | None = None,
    max_claims: int = MAX_CLAIMS,
) -> dict:
    """Extract checkable particulars, retrieve webscraper evidence, return a seed verdict.

    Hallucinating only if a particular is contradicted or fabricated against
    fetched pages (or Serper snippets if fetch fails). Supported textbook
    mechanisms stay Not Hallucinating.
    """
    candidates = extract_candidates(question, answer, gpt_fn, max_claims=max_claims)
    claim_rows = []
    for candidate in candidates:
        claim_text = candidate["claim"]
        if search_fn is not None:
            snippets, query, error = search_fn(claim_text)
            pages_text = ""
            pages: list[dict] = []
            evidence_kind = "snippets"
            urls: list[str] = []
            if fetch_fn is not None:
                pages_text, pages, fetch_error = fetch_fn(claim_text, snippets, urls)
                if pages_text:
                    evidence_kind = "pages"
                elif fetch_error and not error:
                    error = fetch_error
        else:
            live = _live_evidence(claim_text)
            snippets = live["snippets"]
            query = live["query"]
            error = live["error"]
            pages_text = live["pages_text"]
            pages = live["pages"]
            evidence_kind = live["evidence_kind"]
            urls = live.get("urls") or []
        evidence = format_evidence(pages_text, snippets)
        if error and not snippets and not pages_text:
            judged = {"verdict": "insufficient", "reason": error}
        else:
            judged = judge_claim_against_evidence(claim_text, evidence, gpt_fn)
        claim_rows.append(
            {
                "claim": claim_text,
                "entities": candidate.get("entities") or [],
                "query": query,
                "snippets": (snippets or "")[:4000],
                "pages_text": (pages_text or "")[:4000],
                "pages": pages,
                "urls": urls[: MAX_URLS_TO_FETCH + MAX_PDFS_TO_FETCH],
                "evidence_kind": evidence_kind,
                "search_error": error,
                **judged,
            }
        )
    chosen = pick_false_claim(claim_rows)
    hallucinating = chosen is not None
    if hallucinating:
        reason = f"{chosen['verdict']}: {chosen.get('reason') or chosen['claim']}"
        false_claim = chosen["claim"]
        entities = chosen.get("entities") or []
    elif not claim_rows:
        reason = "no checkable particular to verify"
        false_claim = ""
        entities = []
    else:
        reason = "web evidence did not contradict or show fabrication for extracted particulars"
        false_claim = ""
        entities = []
    used_pages = any(row.get("evidence_kind") == "pages" for row in claim_rows)
    return {
        "method": "webscraper",
        "evidence_kind": "pages" if used_pages else "snippets",
        "judge": f"gpt-5-mini-{DEFAULT_JUDGE_REASONING_EFFORT}",
        "hallucinating": hallucinating,
        "false_claim": false_claim,
        "entities": entities,
        "reason": reason,
        "claims": claim_rows,
    }
