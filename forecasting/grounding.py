"""Grounding backend for cascade seeds.

Architecture: DatasetAdapter -> GroundingBackend -> VerifiedSeed -> TreeEngine.

The tree may grow only on VERIFIED_FALSE seeds. HalluHard webscraper evidence
(search + page/PDF fetch + structured verdict + confirmation) is the default
GroundingBackend. A later FactBench backend can implement the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

VERIFIED_FALSE = "VERIFIED_FALSE"
SUPPORTED = "SUPPORTED"
INSUFFICIENT = "INSUFFICIENT"
NOT_VERIFIED = "NOT_VERIFIED"
SEED_STATUSES = (VERIFIED_FALSE, SUPPORTED, INSUFFICIENT, NOT_VERIFIED)

GptFn = Callable[..., Any]


@dataclass
class VerifiedSeed:
    status: str
    claim: str = ""
    entities: list[str] = field(default_factory=list)
    verdict: str = ""
    reason: str = ""
    evidence: dict = field(default_factory=dict)

    def tree_eligible(self) -> bool:
        return self.status == VERIFIED_FALSE and bool(self.claim)


class GroundingBackend(Protocol):
    def verify(self, question: str, answer: str) -> VerifiedSeed: ...


class WebscraperGroundingBackend:
    """HalluHard structured-analysis path used as a reusable grounding backend."""

    def __init__(self, gpt_fn: GptFn, *, search_fn=None, fetch_fn=None):
        self.gpt_fn = gpt_fn
        self.search_fn = search_fn
        self.fetch_fn = fetch_fn

    def verify(self, question: str, answer: str) -> VerifiedSeed:
        import web_verify

        result = web_verify.verify_seed_answer(
            question,
            answer,
            self.gpt_fn,
            search_fn=self.search_fn,
            fetch_fn=self.fetch_fn,
        )
        return verified_seed_from_web(result)


def status_from_web(result: dict | None) -> str:
    if not result:
        return NOT_VERIFIED
    stored = result.get("seed_status")
    if stored in SEED_STATUSES:
        return stored
    if result.get("hallucinating") and (result.get("false_claim") or "").strip():
        return VERIFIED_FALSE
    claims = result.get("claims") or []
    verdicts = {str(row.get("verdict") or "").lower() for row in claims}
    if "supported" in verdicts and not (verdicts & {"contradicted", "fabricated"}):
        return SUPPORTED
    return INSUFFICIENT


def verified_seed_from_web(result: dict | None) -> VerifiedSeed:
    result = result or {}
    status = status_from_web(result)
    return VerifiedSeed(
        status=status,
        claim=str(result.get("false_claim") or "").strip(),
        entities=list(result.get("entities") or []),
        verdict=str((result.get("confirmed_verdict") or result.get("verdict") or "")).strip(),
        reason=str(result.get("reason") or "").strip(),
        evidence=result,
    )


def verified_seed_from_record(record: dict) -> VerifiedSeed:
    stored_status = record.get("seed_status")
    web = record.get("web_verification") if isinstance(record.get("web_verification"), dict) else {}
    if stored_status in SEED_STATUSES:
        return VerifiedSeed(
            status=stored_status,
            claim=str(record.get("web_false_claim") or web.get("false_claim") or "").strip(),
            entities=list(record.get("entities") or web.get("entities") or []),
            verdict=str(web.get("confirmed_verdict") or web.get("verdict") or "").strip(),
            reason=str(record.get("judge_reason") or web.get("reason") or "").strip(),
            evidence=web,
        )
    return verified_seed_from_web(web if web else None)


def tree_eligible_record(record: dict, *, dry_run: bool = False) -> bool:
    """TreeEngine gate: only VERIFIED_FALSE seeds, unless this is a dry-run stub."""
    if dry_run:
        return True
    return verified_seed_from_record(record).tree_eligible()
