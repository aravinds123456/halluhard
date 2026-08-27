"""Serper-backed claim verification for cascade seeds.

This is the HalluHard structured-analysis step reused on cascade turn-0
answers: extract checkable particulars, search the web, judge each claim
against snippets. The LLM does not get to call a true textbook mechanism a
hallucination just because it lacks a citation.

Pass --no-web or CASCADE_WEB=0 to use the LLM-only fallback. That fallback
is not the paper path.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Callable

from cascade import DEFAULT_JUDGE_REASONING_EFFORT, env_str
from prompts_pack import fill_prompt

# forecasting/web_verify.py -> repo root, so `import libs.serper` works.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FALSE_VERDICTS = ("contradicted", "fabricated")
MAX_CLAIMS = 3
SEARCH_CHARS = 300

GptFn = Callable[..., Any]
SearchFn = Callable[[str], tuple[str, str, str | None]]


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


def serper_configured() -> bool:
    return bool(os.environ.get("SERPER_API_KEY", "").strip())


def web_requested() -> bool:
    return not web_flag_disabled()


def require_serper_unless_disabled(*, dry_run: bool = False) -> None:
    if dry_run or web_flag_disabled():
        return
    if not serper_configured():
        raise SystemExit(
            "Cascade seed judging uses Serper web evidence (HalluHard structured "
            "analysis). Set SERPER_API_KEY, or pass --no-web / CASCADE_WEB=0 for "
            "the LLM-only fallback. That fallback will over-label true mechanisms "
            "as Hallucinating and is not the paper path."
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
    raise RuntimeError("web_verify.serper_search cannot nest inside a running event loop")


def serper_search(claim: str, num_results: int = 5) -> tuple[str, str, str | None]:
    from libs.serper import SerperSearchClient

    query = (claim or "").strip()[:SEARCH_CHARS]
    if not query:
        return "", "", "empty claim"

    async def _search() -> str:
        async with SerperSearchClient() as client:
            raw, _n = await client.search(query, num_results=num_results)
            return client._format_single_result(raw)

    try:
        return _run_async(_search()), query, None
    except Exception as error:
        return "", query, f"{type(error).__name__}: {error}"


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


def judge_claim_against_snippets(claim: str, snippets: str, gpt_fn: GptFn) -> dict:
    payload = _as_dict(
        invoke_gpt(
            gpt_fn,
            fill_prompt(
                "web_claim_judge",
                claim=claim[:800],
                snippets=(snippets or "No search results found.")[:6000],
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


def pick_false_claim(claim_rows: list[dict]) -> dict | None:
    contradicted = [row for row in claim_rows if row.get("verdict") == "contradicted"]
    fabricated = [row for row in claim_rows if row.get("verdict") == "fabricated"]
    if contradicted:
        return contradicted[0]
    if fabricated:
        return fabricated[0]
    return None


def verify_seed_answer(
    question: str,
    answer: str,
    gpt_fn: GptFn,
    *,
    search_fn: SearchFn | None = None,
    max_claims: int = MAX_CLAIMS,
) -> dict:
    """Extract checkable particulars, Serper-search each, return a seed verdict.

    Hallucinating only if a particular is contradicted or fabricated against
    snippets. Supported textbook mechanisms stay Not Hallucinating.
    """
    search = search_fn or serper_search
    candidates = extract_candidates(question, answer, gpt_fn, max_claims=max_claims)
    claim_rows = []
    for candidate in candidates:
        snippets, query, error = search(candidate["claim"])
        if error and not snippets:
            judged = {"verdict": "insufficient", "reason": error}
        else:
            judged = judge_claim_against_snippets(candidate["claim"], snippets, gpt_fn)
        claim_rows.append(
            {
                "claim": candidate["claim"],
                "entities": candidate.get("entities") or [],
                "query": query,
                "snippets": (snippets or "")[:4000],
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
        reason = "serper did not contradict or show fabrication for extracted particulars"
        false_claim = ""
        entities = []
    return {
        "method": "serper",
        "judge": f"gpt-5-mini-{DEFAULT_JUDGE_REASONING_EFFORT}",
        "hallucinating": hallucinating,
        "false_claim": false_claim,
        "entities": entities,
        "reason": reason,
        "claims": claim_rows,
    }
