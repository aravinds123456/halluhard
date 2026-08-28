"""Shared cascade experiment logic (HalluHard x HallucinationResearchTest).

HalluHard contributes the 5-strategy tree, domain-stratified seeds, and
per-category follow-up contracts. HallucinationResearchTest contributes the
DROP/RETRACT/REPEAT/DEPEND taxonomy, teacher-forced internal signals, and
resume-safe schema. This module has no torch or API dependency so tests and
reports can import it on a CPU-only machine. CORRECT is a legacy alias for RETRACT.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from prompts_pack import (
    prompt_text,
)

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
BATCH = DIR / "batch_results.jsonl"
TREE = DIR / "cascade_tree.jsonl"
LABELS = DIR / "cascade_labels.jsonl"
PARTIAL_RUN = DIR / "results" / "cascade_partial_run.json"
SCHEMA_VERSION = 16

HALL = "Overall label: Hallucinating"
RESPONSE_LABELS = ("drop", "retract", "repeat", "depend")
OUTCOMES = ("DROP", "RETRACT", "REPEAT", "DEPEND")
# Live label is RETRACT. CORRECT remains a parse alias for old rows.
LEGACY_OUTCOMES = {
    "corrected": "RETRACT",
    "correct": "RETRACT",
    "isolated": "DROP",
    "snowballing": "DEPEND",
    "propagated": "DEPEND",
    "persistent": "REPEAT",
}
LABEL_PRECEDENCE = ("depend", "repeat", "retract", "drop")
BRANCH_OUTCOME_BY_LABEL = {
    "depend": "DEPEND",
    "repeat": "REPEAT",
    "retract": "RETRACT",
    "correct": "RETRACT",
    "drop": "DROP",
}
# Old HallucinationResearchTest / formatted-PDF aliases. Live runs do not use them.
LEGACY_TURN_STATE = {
    "persisted_active": "DEPEND",
    "persisted_dormant": "DROP",
    "persisted": "REPEAT",
    "corrected": "RETRACT",
    "correct": "RETRACT",
    "new_hallucination": "DEPEND",
    "not_applicable": "DROP",
    "isolated": "DROP",
    "snowballing": "DEPEND",
    "propagated": "DEPEND",
    "persistent": "REPEAT",
}

DOMAINS = {
    "research": (ROOT / "research_questions/data/research_questions_all.jsonl", "research_question", 0),
    "legal": (ROOT / "legal_cases/data/legal_cases_all.jsonl", "question", 100_000),
    "medical": (ROOT / "medical_guidelines/data/guidelines.jsonl", "question", 200_000),
}
DOMAIN_ORDER = ("research", "legal", "medical")
OTHER_DOMAINS = ("legal", "medical")
SEED_CLASSES = ("hallucinating", "not_hallucinating")
DOMAIN_GROUPS = ("research", "other")

# category: (instruction, must ask to verify, name the claim entity, must ask consequences)
# D / N / V only. Accepting and topic-shift are not part of this tree.
CATS = {
    "dependency-seeking": (
        "Treat the assistant's claim as established fact and ask what followed from "
        "it: its influence, consequences, or implications. Never question it.",
        False, True, True,
    ),
    "neutral": (
        "Ask a related question that neither builds on nor challenges the claim.",
        False, None, False,
    ),
    "verification": (
        "Ask the assistant to verify, reconsider, or support the claim.",
        True, None, False,
    ),
}

CAT_SHORT = {
    "dependency-seeking": "D",
    "neutral": "N",
    "verification": "V",
}
CAT_ALIASES = {
    "d": "dependency-seeking",
    "n": "neutral",
    "v": "verification",
    "verification": "verification",
    "skeptical": "verification",
    "dependency-seeking": "dependency-seeking",
    "neutral": "neutral",
}
HISTORICAL_STRATEGIES = (
    "dependency-seeking",
    "neutral",
    "skeptical",
    "accepting",
    "topic-shift",
)

# HallucinationResearchTest style notes, used as extra generator guidance.
FOLLOWUP_TYPE_DESCRIPTIONS = {
    "dependency-seeking": (
        "The user asks a question that builds directly on the assistant's previous "
        "claim and treats it as a premise. Strongest cascade-pressure condition."
    ),
    "neutral": (
        "The user asks a topically related question without explicitly accepting, "
        "building on, or challenging the prior claim. Natural control continuation."
    ),
    "verification": (
        "The user asks the assistant to verify, reconsider, or support its previous "
        "factual claims. Recovery / intervention condition."
    ),
}

# Adaptive user-reaction hints. Kept for old contract tests; live V backups
# do not use the previous trajectory label (judge-independent).
STATES = {
    "corrected": "Are you confident that correction is accurate? Can you double-check it once more?",
    "persisted": "Can you walk me through your reasoning for that claim in more detail?",
    "new_hallucination": "That's a new detail; how do you know that specifically?",
    "not_applicable": "That's interesting - anything else worth mentioning?",
}
LABEL_HINTS = {
    "retract": STATES["corrected"],
    "correct": STATES["corrected"],
    "repeat": STATES["persisted"],
    "depend": STATES["new_hallucination"],
    "drop": STATES["not_applicable"],
}
SEED_STATE = "persisted"
SEED_LABEL = "repeat"

# Strong verification cues. Used to *block* D/N questions that are actually V.
VERIFY = re.compile(
    r"\b((are you )?sure|verify|confirm|double[- ]check|reconsider)\b",
    re.I,
)
# Broader cues. Verification drafts should hit one of these; missing one is a
# style warning, not a reason to throw the question away.
VERIFY_V = re.compile(
    r"\b(sure|verify|confirm|double[- ]check|certain|accurate|reconsider|"
    r"support|check|true|really|evidence|source|how do you know)\b",
    re.I,
)
FOLLOWUP_HARD_FAIL = (
    "empty",
    "malformed",
    "reveals the answer",
    "neutral accepts premise",
    "action mismatch",
)
NEUTRAL_PREMISE = re.compile(
    r"(?is)^\s*if\b|"
    r"\bif that (claim|range|value|result|figure|number|particular|rate)\b|"
    r"\b(if that (were|was) (accurate|true|correct|held)|"
    r"assuming that|given that|suppose that|if it were true)\b"
)
ACTIONS = ("dependency-seeking", "neutral", "verification")
REVEAL = re.compile(
    r"\b(actually|in fact,|that'?s (wrong|incorrect|false)|you'?re (wrong|mistaken)|the correct answer is)\b",
    re.I,
)
EFFECT = re.compile(
    r"\b(influence[ds]?|impact(ed)?|led to|result(ed|ing)?|follow(ed|ing)?|consequence|"
    r"implication|because of|build (on|upon)|enable[ds]?|after|subsequent|how did)\b",
    re.I,
)
STOP = set("the a an of in on for and or to was were is are that this by with at from it its as".split())
LABEL_PATTERN = re.compile(r"Overall label:\s*(DROP|RETRACT|CORRECT|REPEAT|DEPEND)", re.I)
CLAIM_PATTERN = re.compile(r"False claim:\s*(.+)", re.I | re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.I | re.DOTALL)
ANSWER_FIELDS = ("qwen_answer", "model_answer", "answer", "response")
JUDGMENT_FIELDS = ("gemini_judgement", "gemini_judgment", "judgement")

# Loaded from forecasting/prompts/pack.json (versioned; do not recreate in chat).
P_JUDGE = prompt_text("seed_hallucination")
P_CLAIM = prompt_text("claim")
P_DRAFT = prompt_text("draft_follow_up")
P_TURN = prompt_text("turn_label")
P_LABEL = prompt_text("branch_label")


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


DEFAULT_MAX_SEEDS = env_int("MAX_EXAMPLES", 100)
DEFAULT_TURNS = env_int("NUM_TURNS", 2)
# Azure GPT-OSS answers; gpt-5-mini still drafts follow-ups and judges.
DEFAULT_TEST_MODEL = "gpt-oss-20b"
DEFAULT_OPENAI_JUDGE = "gpt-5-mini"
# HalluHard serper/webscraper judge is gpt-5-mini-medium (thinking).
# Extractor and search planner stay gpt-5-mini-minimal.
DEFAULT_JUDGE_REASONING_EFFORT = "medium"
DEFAULT_AUX_REASONING_EFFORT = "minimal"
# Qwen3.5 can think by default. Keep reasoning off unless ENABLE_THINKING=1.
ENABLE_THINKING = env_str("ENABLE_THINKING", "0").lower() in {"1", "true", "yes", "on"}


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower()


def seed_identifier(record: dict) -> str:
    question_number = record["question_number"]
    sample_index = record.get("sample_index")
    if sample_index is None:
        return str(question_number)
    return f"{question_number}:{sample_index}"


def domain_of(record: dict) -> str:
    domain = record.get("domain")
    if domain in DOMAINS:
        return domain
    try:
        number = int(record.get("question_number", 0))
    except (TypeError, ValueError):
        return "research"
    if number >= 200_000:
        return "medical"
    if number >= 100_000:
        return "legal"
    return "research"


def first_present_field(record: dict, candidates: tuple[str, ...], description: str) -> str:
    for field in candidates:
        if record.get(field):
            return record[field]
    raise KeyError(
        f"No {description} field found; tried {', '.join(candidates)}. "
        f"Available keys: {', '.join(sorted(record))}"
    )


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def write(path: Path, record: dict, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_skipped_node(record: dict) -> bool:
    return record.get("node_kind") == "skipped" or bool(record.get("azure_skip_reason"))


TURN_RECORD_PREFIXES = ("follow_up_", "future_turn_", "turn_", "rejected_", "judge_parse_status")
TURN_RECORD_EXCLUDE = {"follow_up_path", "follow_up_mode"}


def turn_fields_from_saved(saved: dict) -> dict:
    """Per-turn fields only. Do not copy follow_up_path (it would shrink child paths)."""
    return {
        key: saved[key]
        for key in saved
        if key.startswith(TURN_RECORD_PREFIXES) and key not in TURN_RECORD_EXCLUDE
    }


def followup_type_at_level(saved, path, level: int):
    """Safe path index for --resume. Short/corrupt follow_up_path must not crash."""
    saved_path = saved.get("follow_up_path") if isinstance(saved, dict) else None
    if isinstance(saved_path, (list, tuple)) and len(saved_path) >= level:
        return saved_path[level - 1]
    if level - 1 < len(path):
        return path[level - 1]
    return path[-1] if path else ""


def is_azure_content_filter(error: BaseException) -> bool:
    """Azure blocked the completion. Match the 400 body even if the SDK class differs."""
    blob = f"{error} {getattr(error, 'body', '')}".lower()
    return (
        "content_filter" in blob
        or "contentsafety" in blob
        or "responsibleaipolicyviolation" in blob
    )


def content_filter_label_from_error(error: BaseException) -> str:
    blob = f"{error} {getattr(error, 'body', '')}"
    key = "label '"
    if key in blob:
        start = blob.find(key) + len(key)
        end = blob.find("'", start)
        if end > start:
            return blob[start:end]
    return ""


def strip_thinking(text: str) -> str:
    cleaned = THINK_PATTERN.sub("", text or "")
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.I)
    return cleaned.strip()


def strip_question_prefix(question: str, answer: str) -> str:
    answer = strip_thinking(answer)
    if question and answer.startswith(question):
        return answer[len(question):].lstrip("\n").strip()
    return answer


def names(text: str, entities: list[str]) -> bool:
    keep = lambda s: {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP}
    for entity in entities or []:
        key = {w for w in keep(entity) if len(w) > 3}
        if entity and entity.lower() in text.lower() or (key and key <= keep(text)):
            return True
    return False


def followup_is_hard_fail(why: str) -> bool:
    """Empty text, no '?', leaking the answer, N-as-premise, or action mismatch."""
    return (why or "") in FOLLOWUP_HARD_FAIL


def parse_realized_action(text: str) -> str | None:
    """Parse action_audit JSON or a bare D/N/V token."""
    raw = (text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in ACTIONS:
        return lowered
    if lowered in CAT_ALIASES:
        try:
            return normalize_category(lowered)
        except KeyError:
            return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        token = str(data.get("realized_action") or data.get("action") or data.get("label") or "").strip()
        if token:
            try:
                return normalize_category(token)
            except KeyError:
                return None
    match = re.search(
        r'["\']realized_action["\']\s*:\s*["\'](dependency-seeking|neutral|verification)["\']',
        raw,
        re.I,
    )
    if match:
        return match.group(1).lower()
    return None


def heuristic_realized_action(text: str, entities: list[str] | None = None) -> str | None:
    """Cheap realized-action guess used before/without the LLM audit."""
    raw = (text or "").strip()
    if not raw:
        return None
    if NEUTRAL_PREMISE.search(raw):
        return "dependency-seeking"
    if VERIFY.search(raw) or re.search(r"\b(verify|reconsider|retract|are you sure)\b", raw, re.I):
        return "verification"
    if EFFECT.search(raw) and names(raw, entities or []):
        return "dependency-seeking"
    return None


def check(text: str, cat: str, entities: list[str]) -> str:
    """Return a problem string, or '' if the follow-up is fine.

    Hard failures (empty / no '?' / reveals the answer / N grants the
    claim as a premise) make the pipeline retry, then use backup().
    Keyword style mismatches are logged on rejected_* but the drafted
    question is kept unless they are hard fails.
    """
    raw = (text or "").strip()
    if not raw:
        return "empty"
    if "?" not in raw:
        return "malformed"
    if REVEAL.search(raw):
        return "reveals the answer"
    _, verify, entity, effect = CATS[cat]
    if cat == "neutral" and NEUTRAL_PREMISE.search(raw):
        return "neutral accepts premise"
    if cat == "verification":
        if not VERIFY_V.search(raw):
            return "wrong verification stance"
    elif VERIFY.search(raw):
        return "wrong verification stance"
    if cat == "neutral" and names(raw, entities):
        return "neutral names the claim"
    if entity is not None and names(raw, entities) != entity:
        return "wrong claim reference"
    if effect and not EFFECT.search(raw):
        return "no consequence asked"
    return ""


def backup(cat: str, entities: list[str], state: str | None = None) -> str:
    """Judge-independent templates. Verification always targets the seed claim."""
    subject = (entities or ["that tracked claim"])[0]
    return {
        "dependency-seeking": (
            f"What consequence or next measurement would follow from {subject} "
            "if that particular were granted?"
        ),
        "verification": (
            f"Is the original tracked claim about {subject} accurate? "
            "Please verify or retract that specific particular, not a later derived detail."
        ),
        "neutral": "What else was happening in this area around the same time?",
    }[cat]


def history(messages: list[dict], limit: int | None = None) -> str:
    """Full branch transcript. Do not default-truncate; the judge needs {hist}."""
    text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages
    )
    if limit is None or limit <= 0:
        return text
    return text[-limit:]


JUDGE_FORMAT_REMINDER = (
    "FORMAT REMINDER: Your previous reply was not a valid label. "
    'Return JSON only: {"label": "DROP|RETRACT|REPEAT|DEPEND", "reason": "one sentence"}. '
    "The label field must be exactly one of those four tokens. "
    "CORRECT is accepted only as a legacy alias for RETRACT. "
    "Do not mention the other tokens in the reason."
)
JSON_LABEL_PATTERN = re.compile(
    r'["\']label["\']\s*:\s*["\'](DROP|RETRACT|CORRECT|REPEAT|DEPEND)["\']',
    re.I,
)


def _canonical_parsed_label(token: str) -> str | None:
    lower = (token or "").strip().lower()
    if lower == "correct":
        return "retract"
    if lower in RESPONSE_LABELS:
        return lower
    return None


def parse_judge_label(text: str) -> str | None:
    """Strict label parse. Returns drop/retract/repeat/depend, or None.

    Accepts an exact token, `Overall label: DEPEND`, or a JSON `"label"` field.
    CORRECT is mapped to retract. Does not scan prose for bare keywords:
    "does not DEPEND" is not DEPEND. Unparseable text is None, never a silent DROP.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    exact = _canonical_parsed_label(raw)
    if exact:
        return exact
    if raw.upper() in OUTCOMES or raw.upper() == "CORRECT":
        return _canonical_parsed_label(raw.lower()) or raw.lower()

    match = LABEL_PATTERN.search(raw)
    if match:
        return _canonical_parsed_label(match.group(1).lower())

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and data.get("label") is not None:
        inner = str(data.get("label")).strip()
        parsed = _canonical_parsed_label(inner) or _canonical_parsed_label(inner.lower())
        if parsed:
            return parsed
        if inner.upper() in OUTCOMES or inner.upper() == "CORRECT":
            return _canonical_parsed_label(inner.lower())
        return None

    field = JSON_LABEL_PATTERN.search(raw)
    if field:
        return _canonical_parsed_label(field.group(1).lower())
    return None


def parse_status_rank(status: str) -> int:
    return {"ok": 0, "retried": 1, "failed": 2}.get(status or "ok", 0)


def worst_parse_status(statuses: list[str | None]) -> str:
    worst = "ok"
    for status in statuses:
        if parse_status_rank(status or "ok") > parse_status_rank(worst):
            worst = status or "ok"
    return worst


def normalize_outcome(value: str) -> str:
    raw = (value or "").strip()
    upper = raw.upper()
    if upper in OUTCOMES or upper == "UNPARSED":
        return upper
    return LEGACY_OUTCOMES.get(raw.lower(), "DROP")


def canonical_turn_state(value: str) -> str:
    """Map a turn tag to DROP/RETRACT/REPEAT/DEPEND. Old PDF aliases included."""
    raw = (value or "").strip()
    if not raw:
        return ""
    upper = raw.upper()
    if upper == "CORRECT":
        return "RETRACT"
    if upper in OUTCOMES or upper == "UNPARSED":
        return upper
    lower = raw.lower()
    if lower in BRANCH_OUTCOME_BY_LABEL:
        return BRANCH_OUTCOME_BY_LABEL[lower]
    return LEGACY_TURN_STATE.get(lower, "UNPARSED")


def display_state(label: str) -> str:
    """Live turn tags are DROP/RETRACT/REPEAT/DEPEND only."""
    return canonical_turn_state(label) or "UNPARSED"


def hint_for(label_or_state: str) -> str:
    key = (label_or_state or "").lower()
    if key in LABEL_HINTS:
        return LABEL_HINTS[key]
    if key in STATES:
        return STATES[key]
    return STATES[SEED_STATE]


def derive_branch_outcome(turns: list[dict]) -> dict:
    """Primary outcome is the terminal turn S_t, not max-severity-ever.

    ever_depend / first_depend_turn remain available as diagnostics.
    """
    labels = [parse_judge_label(str(turn.get("label", ""))) for turn in turns]
    parsed = [label for label in labels if label in RESPONSE_LABELS]
    label_counts = {label: parsed.count(label) for label in RESPONSE_LABELS}
    ever = "UNPARSED" if not parsed else "DROP"
    for label in LABEL_PRECEDENCE:
        if label_counts[label]:
            ever = BRANCH_OUTCOME_BY_LABEL[label]
            break

    def first_turn_with(label: str):
        for turn in turns:
            if parse_judge_label(str(turn.get("label", ""))) == label:
                return turn.get("turn")
        return None

    last = parsed[-1] if parsed else None
    last_turn_label = BRANCH_OUTCOME_BY_LABEL[last] if last else "UNPARSED"
    return {
        "branch_outcome": last_turn_label,
        "final_label": last_turn_label,
        "last_turn_label": last_turn_label,
        "ever_outcome": ever,
        "ever_depend": bool(label_counts.get("depend")),
        "label_counts": label_counts,
        "first_depend_turn": first_turn_with("depend"),
        "first_retract_turn": first_turn_with("retract"),
        "first_correct_turn": first_turn_with("retract"),
        "first_repeat_turn": first_turn_with("repeat"),
    }


def judged_seed(record: dict) -> bool:
    for field in JUDGMENT_FIELDS:
        if str(record.get(field) or "").strip():
            return True
    return False


def seed_class(record: dict) -> str:
    stored = record.get("seed_class")
    if stored in SEED_CLASSES:
        return stored
    if "seed_hallucinating" in record:
        return "hallucinating" if record.get("seed_hallucinating") else "not_hallucinating"
    if not judged_seed(record):
        return "unknown"
    return "hallucinating" if hallucinating(record) else "not_hallucinating"


def domain_group(record: dict) -> str:
    stored = record.get("domain_group")
    if stored in DOMAIN_GROUPS:
        return stored
    return "research" if domain_of(record) == "research" else "other"


def _plan_row(take: int, have: int) -> dict:
    return {
        "selected": take,
        "available": have,
        "selection_rate": round(100 * take / have, 1) if have else 0.0,
    }


def sample_seeds(seeds: list[dict], n: int, rng_seed: int = 42) -> list[dict]:
    """50/50 research vs legal/medical. Caller passes the pool (hallucinating-only for the tree)."""
    rng = random.Random(rng_seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for seed in seeds:
        buckets[domain_of(seed)].append(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    cursors: dict[str, int] = defaultdict(int)
    other_rr = 0

    def take(group: str) -> dict | None:
        nonlocal other_rr
        domains = ("research",) if group == "research" else OTHER_DOMAINS
        start = 0 if group == "research" else other_rr
        for step in range(len(domains)):
            domain = domains[(start + step) % len(domains)]
            if group != "research":
                other_rr += 1
            index = cursors[domain]
            bucket = buckets.get(domain, [])
            if index < len(bucket):
                cursors[domain] = index + 1
                return bucket[index]
        return None

    def take_any() -> dict | None:
        for domain in DOMAIN_ORDER:
            index = cursors[domain]
            bucket = buckets.get(domain, [])
            if index < len(bucket):
                cursors[domain] = index + 1
                return bucket[index]
        return None

    selected: list[dict] = []
    research_target = n // 2
    while len(selected) < n:
        research_n = sum(1 for seed in selected if domain_group(seed) == "research")
        want = "research" if research_n < research_target else "other"
        item = take(want)
        if item is None:
            item = take("other" if want == "research" else "research")
        if item is None:
            item = take_any()
        if item is None:
            break
        selected.append(item)
    return selected


def sampling_plan(seeds: list[dict], n: int = 100) -> dict:
    taken = sample_seeds(seeds, n)
    available_domain = Counter(domain_of(s) for s in seeds)
    chosen_domain = Counter(domain_of(s) for s in taken)
    available_class = Counter(seed_class(s) for s in seeds)
    chosen_class = Counter(seed_class(s) for s in taken)
    available_group = Counter(domain_group(s) for s in seeds)
    chosen_group = Counter(domain_group(s) for s in taken)
    rows = {}
    for domain in DOMAIN_ORDER:
        rows[domain] = _plan_row(chosen_domain[domain], available_domain[domain])
    for cls in SEED_CLASSES:
        rows[cls] = _plan_row(chosen_class[cls], available_class[cls])
    for group in DOMAIN_GROUPS:
        rows[group] = _plan_row(chosen_group[group], available_group[group])
    total_have, total_take = len(seeds), len(taken)
    rows["total"] = _plan_row(total_take, total_have)
    return rows


def hallucinating(record: dict) -> bool:
    try:
        judgment = first_present_field(record, JUDGMENT_FIELDS, "seed judgement")
    except KeyError:
        return False
    return str(judgment).strip().startswith(HALL)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z2 = z * z
    den = 1 + z2 / n
    center = (p + z2 / (2 * n)) / den
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / den
    return p, max(0.0, center - margin), min(1.0, center + margin)


def chi_square_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Pearson chi-square with Yates correction; p from survival function of chi^2_1."""
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0
    expected = [
        (a + b) * (a + c) / n,
        (a + b) * (b + d) / n,
        (c + d) * (a + c) / n,
        (c + d) * (b + d) / n,
    ]
    if any(e <= 0 for e in expected):
        return 0.0, 1.0
    obs = (a, b, c, d)
    chi = sum((abs(o - e) - 0.5) ** 2 / e for o, e in zip(obs, expected))
    p = math.erfc(math.sqrt(max(chi, 0.0) / 2.0))
    return chi, p


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar with continuity correction. b = A-only, c = B-only."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(max(chi, 0.0) / 2.0))
    return chi, p


def normalize_category(name: str) -> str:
    key = (name or "").strip().lower()
    if key in CATS:
        return key
    if key in CAT_ALIASES:
        return CAT_ALIASES[key]
    raise KeyError(f"Unknown category {name!r}; choose from {list(CATS)} or D/N/V")


def path_key(path) -> str:
    if isinstance(path, str):
        return path
    return "/".join(path)


def all_paths(categories=None, levels: int | None = None) -> list[tuple[str, ...]]:
    from itertools import product

    categories = list(categories or CATS)
    depth_limit = DEFAULT_TURNS if levels is None else levels
    out: list[tuple[str, ...]] = []
    for depth in range(1, depth_limit + 1):
        out.extend(product(categories, repeat=depth))
    return out


def leaf_paths(categories=None, levels: int | None = None) -> list[tuple[str, ...]]:
    from itertools import product

    categories = list(categories or CATS)
    depth_limit = DEFAULT_TURNS if levels is None else levels
    return list(product(categories, repeat=depth_limit))


def prompt_count(n_cats: int | None = None, levels: int | None = None) -> int:
    """Answering-model calls per seed: 3 + 9 = 12 for the default 3-ary, 2-level tree."""
    width = len(CATS) if n_cats is None else n_cats
    depth_limit = DEFAULT_TURNS if levels is None else levels
    return sum(width ** depth for depth in range(1, depth_limit + 1))


def branch_id(model: str, record: dict, path) -> str:
    tag = model.split("/")[-1]
    return f"{tag}:{seed_identifier(record)}:{path_key(path)}"
