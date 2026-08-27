"""Generate and judge seed hallucinations with the model under test.

HallucinationResearchTest's seed sampler, adapted to HalluHard's three
domains. Scoring another model's confidence on Qwen's text is a different
quantity from that model's own generation confidence, so each model needs
to produce its own seeds before the branching stage runs.

    TEST_MODEL="meta-llama/Llama-3.1-8B-Instruct" python forecasting/generate_seeds.py
    TEST_MODEL="meta-llama/Llama-3.1-8B-Instruct" \
        python forecasting/pipeline.py tree --seeds forecasting/seeds_meta-llama-llama-3.1-8b-instruct.jsonl --resume
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

from cascade import (
    DEFAULT_TEST_MODEL,
    DEFAULT_TURNS,
    DIR,
    DOMAINS,
    ENABLE_THINKING,
    env_float,
    env_int,
    env_str,
    hallucinating,
    judged_seed,
    model_slug,
    seed_identifier,
    strip_question_prefix,
    strip_thinking,
    write,
)
from prompts_pack import (
    DEFAULT_PILOT_QUESTIONS,
    fill_prompt,
    log_experiment,
    prompt_ids,
    prompt_pack_version,
    prompt_text,
    require_pilot,
    write_pilot_stage,
)
import web_verify

MODEL_NAME = env_str("TEST_MODEL", env_str("QWEN_MODEL", DEFAULT_TEST_MODEL))
SEED_SCHEMA_VERSION = 3
# GPT-OSS hidden reasoning counts against this. Same default as runtime MAX_TOKENS.
SEED_MAX_NEW_TOKENS = env_int("SEED_MAX_NEW_TOKENS", env_int("MAX_TOKENS", 32768))
MAX_QUESTIONS = env_int("MAX_QUESTIONS", 0)
SAMPLES_PER_QUESTION = env_int("SAMPLES_PER_QUESTION", 1)
TEMPERATURE = env_float("TEMPERATURE", 0.7)
TOP_P = env_float("TOP_P", 0.95)
TOP_K = env_int("TOP_K", 0)
BASE_SEED = env_int("BASE_SEED", 1234)
DOMAIN_FILTER = env_str("SEED_DOMAIN", "all")

SEEDS_PATH = Path(env_str("SEEDS_PATH", str(DIR / f"seeds_{model_slug(MODEL_NAME)}.jsonl")))
REJUDGE_POOL_PATH = DIR / "results" / "rejudge_pool.jsonl"
DEFAULT_TREE_OUT = DIR / "cascade_tree_dnv.jsonl"

SEED_JUDGE_TEMPLATE = prompt_text("seed_judge")

SEED_LABEL_PATTERN = re.compile(
    r"Overall label:\s*(Not\s+Hallucinating|Hallucinating)",
    re.IGNORECASE,
)
SEED_REASON_PATTERN = re.compile(r"Reason:\s*(.+)", re.IGNORECASE | re.DOTALL)


def argv_has(*names: str) -> bool:
    return any(name in sys.argv for name in names)


def argv_option_int(name: str, default: int | None = None) -> int | None:
    """Parse `--limit 400` or `--limit=400`. Returns default if the flag is absent."""
    eq = name + "="
    for index, token in enumerate(sys.argv[1:], start=1):
        if token == name:
            if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("-"):
                return int(sys.argv[index + 1])
            raise SystemExit(f"{name} needs an integer, e.g. {name} 400")
        if token.startswith(eq):
            return int(token.split("=", 1)[1])
    return default


def argv_option_str(name: str, default: str | None = None) -> str | None:
    eq = name + "="
    for index, token in enumerate(sys.argv[1:], start=1):
        if token == name:
            if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("-"):
                return sys.argv[index + 1]
            raise SystemExit(f"{name} needs a value")
        if token.startswith(eq):
            return token.split("=", 1)[1]
    return default


def already_webscraper(record: dict) -> bool:
    return (
        record.get("judge_method") == "webscraper"
        and record.get("prompt_pack_version") == prompt_pack_version()
        and bool(record.get("web_verification"))
    )


def parse_seed_judgement(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    blob = raw
    if "```" in blob:
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", blob, re.DOTALL)
        if fenced:
            blob = fenced.group(1).strip()
    if blob.startswith("{"):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            reason = str(parsed.get("reason") or "").strip()
            label_raw = str(parsed.get("label") or parsed.get("overall_label") or "").strip().lower()
            if parsed.get("hallucinating") is True or label_raw in {"hallucinating", "true"}:
                return "Hallucinating", reason
            if parsed.get("hallucinating") is False or label_raw in {
                "not hallucinating", "hedged", "grounded", "false",
            }:
                return "Not Hallucinating", reason
    match = SEED_LABEL_PATTERN.search(text or "")
    if match:
        normalized = re.sub(r"\s+", " ", match.group(1)).strip().lower()
        label = "Not Hallucinating" if normalized.startswith("not") else "Hallucinating"
    elif re.search(r"\bnot\s+hallucinat", text or "", re.I):
        label = "Not Hallucinating"
    elif re.search(r"\bhallucinat", text or "", re.I):
        label = "Hallucinating"
    else:
        print(f"Warning: unparseable seed judgement, treating as clean. Raw: {(text or '')[:160]}")
        label = "Not Hallucinating"
    reason_match = SEED_REASON_PATTERN.search(text or "")
    reason = reason_match.group(1).strip().split("\n")[0] if reason_match else ""
    return label, reason


def load_questions() -> dict[int, tuple[str, str]]:
    """question_number -> (question, domain), using HalluHard offsets."""
    questions = {}
    wanted = DOMAINS if DOMAIN_FILTER == "all" else {DOMAIN_FILTER: DOMAINS[DOMAIN_FILTER]}
    for domain, (path, key, offset) in wanted.items():
        if not path.exists():
            continue
        for index, line in enumerate(path.open(encoding="utf-8")):
            record = json.loads(line)
            question = (record.get(key) or record.get("question") or "").strip()
            if question:
                questions[offset + index] = (question, domain)
    if not questions:
        raise FileNotFoundError("No HalluHard domain question files found.")
    return dict(sorted(questions.items()))


def load_existing_seeds():
    if not SEEDS_PATH.exists():
        return set(), {}
    processed = set()
    answers_by_question = {}
    for record in (json.loads(line) for line in SEEDS_PATH.open() if line.strip()):
        if record.get("seed_schema_version", 0) != SEED_SCHEMA_VERSION:
            continue
        if record.get("model_name") != MODEL_NAME:
            continue
        processed.add((record["question_number"], record.get("sample_index", 0)))
        answers_by_question.setdefault(record["question_number"], set()).add(
            record.get("model_answer") or record.get("qwen_answer", "")
        )
    return processed, answers_by_question


def load_all_seed_records() -> list[dict]:
    if not SEEDS_PATH.exists():
        return []
    with SEEDS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rewrite_seeds(records: list[dict]) -> None:
    if not records:
        SEEDS_PATH.write_text("", encoding="utf-8")
        return
    write(SEEDS_PATH, records[0], False)
    for record in records[1:]:
        write(SEEDS_PATH, record, True)


def apply_seed_judgement(
    record: dict,
    label: str,
    reason: str,
    judge_raw: str,
    judge_model: str,
    web_verification: dict | None = None,
) -> dict:
    """Stamp a new seed-judge label onto an existing generation. Does not change the answer."""
    updated = dict(record)
    updated["gemini_judgement"] = f"Overall label: {label}"
    updated["judge_reason"] = reason
    updated["judge_raw"] = judge_raw
    updated["judge_model_name"] = judge_model
    updated["prompt_pack_version"] = prompt_pack_version()
    updated["prompt_ids"] = prompt_ids()
    if web_verification is not None:
        updated["web_verification"] = web_verification
        updated["web_false_claim"] = web_verification.get("false_claim") or ""
        updated["entities"] = web_verification.get("entities") or updated.get("entities") or []
        updated["judge_method"] = web_verification.get("method") or "webscraper"
        updated["seed_status"] = web_verification.get("seed_status") or (
            "VERIFIED_FALSE" if web_verification.get("hallucinating") else "INSUFFICIENT"
        )
    else:
        updated["judge_method"] = updated.get("judge_method") or "llm"
        if "seed_status" not in updated:
            updated["seed_status"] = "NOT_VERIFIED"
    return updated


def rejudge_target_indexes(
    records: list[dict],
    question_numbers: set[int] | None,
    *,
    limit: int | None = None,
) -> list[int]:
    """Indexes of saved seed rows to relabel.

    `--pilot` passes a question-id slice. `--limit 400` without `--pilot`
    takes the first 400 matching seed *rows*, not the first 400 HalluHard
    question-bank ids.
    """
    indexes = []
    for index, record in enumerate(records):
        if record.get("seed_schema_version", 0) != SEED_SCHEMA_VERSION:
            continue
        if record.get("model_name") != MODEL_NAME:
            continue
        if question_numbers is not None and record.get("question_number") not in question_numbers:
            continue
        if record.get("duplicate_answer"):
            continue
        indexes.append(index)
        if limit is not None and len(indexes) >= limit:
            break
    return indexes


def write_rejudge_pool(records: list[dict], path: Path | None = None, *, dry_run: bool = False) -> list[dict]:
    """Write rejudged rows. Return tree-eligible halls (VERIFIED_FALSE, or Hallucinating in dry-run)."""
    path = Path(path or REJUDGE_POOL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    from grounding import tree_eligible_record

    halls = []
    for record in records:
        if record.get("duplicate_answer"):
            continue
        if dry_run:
            if judged_seed(record) and hallucinating(record):
                halls.append(record)
        elif tree_eligible_record(record):
            halls.append(record)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return halls


def start_tree_from_pool(
    *,
    seeds: Path,
    out: Path,
    max_seeds: int,
    levels: int,
    fresh: bool,
    resume: bool,
    skip_pilot: bool,
    dry_run: bool,
    no_web: bool,
    model: str | None = None,
) -> None:
    from argparse import Namespace

    from pipeline import cmd_tree

    cmd_tree(
        Namespace(
            seeds=str(seeds),
            out=str(out),
            max_seeds=max_seeds,
            levels=levels,
            categories="all",
            dry_run=dry_run,
            pilot=False,
            skip_pilot=skip_pilot,
            fresh=fresh,
            no_web=no_web,
            resume=resume,
            model=model or MODEL_NAME,
        )
    )


def sample_seed_value(question_number: int, sample_index: int) -> int:
    return (BASE_SEED * 1_000_003 + int(question_number) * 1_009 + sample_index) % (2**31 - 1)


def raw_step_logits(outputs):
    logits = getattr(outputs, "logits", None)
    if logits:
        return logits
    print("Warning: transformers did not return raw logits; falling back to processed scores.")
    return outputs.scores


def generate_seed_answer(question: str, question_number: int, sample_index: int):
    import runtime

    rng_seed = sample_seed_value(question_number, sample_index)
    if runtime.uses_azure_answer(MODEL_NAME):
        answer = runtime.answer_chat(
            [{"role": "user", "content": question}],
            max_new_tokens=SEED_MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            model_name=MODEL_NAME,
        )
        return strip_question_prefix(question, answer), {}, rng_seed

    import torch
    from features import generation_features
    from runtime import init_model

    init_model(MODEL_NAME)
    tokenizer = runtime.tokenizer
    messages = [{"role": "user", "content": question}]
    model_inputs = runtime.build_model_inputs(messages)
    model_inputs = {key: value.to(runtime.device) for key, value in model_inputs.items() if hasattr(value, "to")}
    input_length = model_inputs["input_ids"].shape[1]
    torch.manual_seed(rng_seed)
    sampling_kwargs = {"do_sample": True, "temperature": TEMPERATURE, "top_p": TOP_P}
    if TOP_K:
        sampling_kwargs["top_k"] = TOP_K
    with torch.no_grad():
        outputs = runtime.model.generate(
            **model_inputs,
            max_new_tokens=SEED_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            **sampling_kwargs,
        )
    generated_token_ids = outputs.sequences[0, input_length:]
    answer = strip_thinking(tokenizer.decode(generated_token_ids, skip_special_tokens=True)).strip()
    special_ids = set(tokenizer.all_special_ids or [])
    features = generation_features(raw_step_logits(outputs), generated_token_ids, special_ids)
    return strip_question_prefix(question, answer), features, rng_seed


def judge_seed(question: str, answer: str, *, use_web: bool | None = None):
    from runtime import call_gemini, gpt, judge_backend

    if use_web is None:
        use_web = web_verify.web_requested()
    if use_web:
        web_verify.require_serper_unless_disabled()
        from grounding import WebscraperGroundingBackend
        verified = WebscraperGroundingBackend(gpt).verify(question, answer)
        result = verified.evidence
        label = "Hallucinating" if verified.tree_eligible() else "Not Hallucinating"
        return label, verified.reason or "", json.dumps(result, ensure_ascii=False), result
    prompt = fill_prompt("seed_judge", question=question, answer=answer)
    if judge_backend() == "gemini":
        raw_text = call_gemini(prompt)
    else:
        raw_text = str(gpt(prompt, as_json=False, role="judge")).strip()
    label, reason = parse_seed_judgement(raw_text)
    return label, reason, raw_text, None


def rejudge_existing(
    question_items,
    pilot: bool,
    *,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict]:
    """Relabel saved answers with the current seed_judge prompt. Does not call GPT-OSS."""
    from runtime import active_judge_model, judge_backend, setup_gemini

    records = load_all_seed_records()
    if not records:
        raise SystemExit(f"No seed file to rejudge at {SEEDS_PATH}")
    question_numbers = {number for number, _ in question_items} if question_items is not None else None
    targets = rejudge_target_indexes(records, question_numbers, limit=limit)
    if not targets:
        raise SystemExit(
            f"No matching judged rows in {SEEDS_PATH.name} to rejudge. "
            "Generate seeds first, then rejudge."
        )
    todo = [index for index in targets if not (resume and already_webscraper(records[index]))]
    print(
        f"Rejudging {len(todo)} saved answers"
        f"{f' ({len(targets) - len(todo)} already current webscraper, skipped)' if len(todo) != len(targets) else ''}"
        f" with {prompt_ids()['seed_judge']} "
        f"(pack v{prompt_pack_version()}). GPT-OSS is not called."
    )
    if web_verify.web_requested() and not web_verify.fetch_flag_disabled():
        print(f"Fetch backend: {web_verify.describe_fetch_backend()}")
    if not todo:
        print("Nothing left to rejudge.")
        return [records[index] for index in targets]
    if judge_backend() == "gemini":
        setup_gemini()
    judge_name = active_judge_model()
    label_counts = Counter()
    verifications = []
    for index, record_index in enumerate(todo, start=1):
        record = records[record_index]
        answer = (record.get("model_answer") or record.get("qwen_answer") or "").strip()
        question = record.get("question") or ""
        qid = record.get("question_number")
        progress = f"[{index}/{len(todo)}] q{qid}#{record.get('sample_index', 0)}"
        if not answer:
            print(f"{progress}: empty answer, skipping")
            continue
        label, reason, judge_raw, web_verification = judge_seed(question, answer)
        if web_verification is not None:
            verifications.append(web_verification)
        label_counts[label] += 1
        records[record_index] = apply_seed_judgement(
            record, label, reason, judge_raw, judge_name, web_verification=web_verification
        )
        rewrite_seeds(records)
        print(f"{progress}: {label}" + (f" - {reason[:80]}" if reason else ""))
    total = sum(label_counts.values())
    hallucinating_n = label_counts["Hallucinating"]
    print(f"\nRejudged {total} answers: {hallucinating_n} hallucinating, {total - hallucinating_n} clean")
    if total:
        print(f"Hallucination rate: {hallucinating_n / total:.1%}")
    if verifications:
        print(web_verify.evidence_summary(verifications))
        print(f"Fetch backend: {web_verify.describe_fetch_backend()}")
    print(f"Wrote {SEEDS_PATH}")
    if pilot:
        write_pilot_stage("seeds", n=len(question_items or []), judged=total, model=MODEL_NAME)
        print("Recorded 10-example seed-prompt debug in forecasting/results/pilot.json")
    return [records[index] for index in targets]


def run_rejudge_then_tree(
    *,
    limit: int = 400,
    tree_out: Path | None = None,
    max_seeds: int = 0,
    levels: int = DEFAULT_TURNS,
    fresh: bool = True,
    resume: bool = False,
    skip_pilot: bool = False,
    dry_run: bool = False,
    no_web: bool = False,
    pilot: bool = False,
) -> None:
    """Rejudge `limit` saved seed rows, then grow the D/N/V tree on those halls."""
    if fresh and resume:
        raise SystemExit("Use --fresh or --resume, not both.")
    if no_web:
        os.environ["CASCADE_WEB"] = "0"
    require_pilot(stage="seeds", n=10 if pilot else limit, dry_run=dry_run, skip_pilot=skip_pilot)
    question_items = None
    if pilot:
        questions = load_questions()
        question_items = list(questions.items())[:DEFAULT_PILOT_QUESTIONS]
        row_limit = None
    else:
        row_limit = limit
    print(
        f"Scale: rejudge {('pilot 10 questions' if pilot else f'{limit} saved seed rows')}"
        f", then branch the VERIFIED_FALSE ones"
        f"{' (DRY_RUN: current labels, no judge API)' if dry_run else ''}."
    )
    if dry_run:
        records = load_all_seed_records()
        targets = rejudge_target_indexes(
            records,
            {number for number, _ in question_items} if question_items is not None else None,
            limit=row_limit,
        )
        pool = [records[index] for index in targets]
        print(f"DRY_RUN=1: would rejudge {len(pool)} rows in {SEEDS_PATH.name}; using current labels for the tree stub.")
    else:
        pool = rejudge_existing(question_items, pilot, limit=row_limit, resume=resume)
    halls = write_rejudge_pool(pool, dry_run=dry_run)
    print(f"Tree pool: {len(halls)} VERIFIED_FALSE / {len(pool)} rejudged -> {REJUDGE_POOL_PATH}")
    if not halls and not dry_run:
        raise SystemExit(
            "No VERIFIED_FALSE seeds in the rejudge pool, so the tree was not started. "
            "A first-pass Hallucinating label is not enough. Rejudge with pack v7 confirmation, "
            "or check fetched_pages if the webscraper failed."
        )
    tree_max = max_seeds if max_seeds and max_seeds > 0 else len(halls)
    out = Path(tree_out or DEFAULT_TREE_OUT)
    require_pilot(stage="tree", n=tree_max, dry_run=dry_run, skip_pilot=skip_pilot)
    start_tree_from_pool(
        seeds=REJUDGE_POOL_PATH,
        out=out,
        max_seeds=tree_max,
        levels=levels,
        fresh=fresh and not resume,
        resume=resume,
        skip_pilot=skip_pilot,
        dry_run=dry_run,
        no_web=no_web,
    )


def main():
    questions = load_questions()
    processed_samples, answers_by_question = load_existing_seeds()
    question_items = list(questions.items())
    pilot = argv_has("--pilot")
    skip_pilot = argv_has("--skip-pilot")
    rejudge = argv_has("--rejudge")
    then_tree = argv_has("--tree")
    resume = argv_has("--resume")
    dry_run = env_str("DRY_RUN", "") == "1" or argv_has("--dry-run")
    cli_limit = argv_option_int("--limit")
    limit = DEFAULT_PILOT_QUESTIONS if pilot else (cli_limit if cli_limit is not None else MAX_QUESTIONS)
    if then_tree and not rejudge:
        raise SystemExit("Pass --rejudge --tree together, or use: python forecasting/pipeline.py scale --limit 400")
    if then_tree and not pilot and not limit:
        limit = 400
    require_pilot(stage="seeds", n=limit or 10**9, dry_run=dry_run, skip_pilot=skip_pilot)
    if then_tree:
        run_rejudge_then_tree(
            limit=limit or 400,
            tree_out=Path(argv_option_str("--tree-out") or DEFAULT_TREE_OUT),
            max_seeds=argv_option_int("--max-seeds", 0) or 0,
            levels=argv_option_int("--levels", DEFAULT_TURNS) or DEFAULT_TURNS,
            fresh=not resume,
            resume=resume,
            skip_pilot=skip_pilot,
            dry_run=dry_run,
            no_web=argv_has("--no-web", "--no_web"),
            pilot=pilot,
        )
        return
    if limit:
        question_items = question_items[:limit]
    pending = [
        (number, question, domain, sample_index)
        for number, (question, domain) in question_items
        for sample_index in range(SAMPLES_PER_QUESTION)
        if (number, sample_index) not in processed_samples
    ]
    from runtime import (
        active_judge_model,
        azure_deployment,
        azure_reasoning_effort,
        azure_send_temperature,
        judge_backend,
        setup_gemini,
        uses_azure_answer,
    )

    print(f"Test model: {MODEL_NAME}")
    if uses_azure_answer(MODEL_NAME):
        print(f"Azure deployment: {azure_deployment(MODEL_NAME)} (override with AZURE_OPENAI_DEPLOYMENT)")
        print(f"Azure reasoning_effort: {azure_reasoning_effort() or 'off'}")
        if azure_send_temperature():
            print(f"Azure temperature: {TEMPERATURE}")
        else:
            print("Azure temperature: omitted (GPT-OSS often rejects it; set AZURE_SEND_TEMPERATURE=1 to sample)")
    print(f"Judge model: {active_judge_model()}")
    if web_verify.web_flag_disabled():
        print("Seed evidence: LLM-only (--no-web / CASCADE_WEB=0). Not the HalluHard paper path.")
    elif web_verify.fetch_flag_disabled():
        print("Seed evidence: gpt-5-mini-medium thinking + Serper snippets (CASCADE_WEB_FETCH=0)")
    else:
        print(
            "Seed evidence: gpt-5-mini-medium thinking + Serper search + page/PDF fetch "
            "(HalluHard --type webscraper)"
        )
        print(f"Fetch backend: {web_verify.describe_fetch_backend()}")
    print(f"Thinking: {'on' if ENABLE_THINKING else 'off'}")
    print(f"Prompt pack: v{prompt_pack_version()} ids={prompt_ids()}")
    print("Workflow: debug prompts on ~10 examples (--pilot), then scale.")
    print(f"Questions: {len(question_items)} HalluHard items, {len(pending)} pending generations")
    if processed_samples and not rejudge:
        print(
            f"Already judged {len(processed_samples)} samples in {SEEDS_PATH.name}; "
            "pass --rejudge to relabel those answers with the current seed_judge prompt."
        )
    print(
        f"Seed max_tokens: {SEED_MAX_NEW_TOKENS} "
        "(hidden GPT-OSS reasoning counts against this; empty content usually means the cap was too low)"
    )
    print(f"Output: {SEEDS_PATH.name}")
    log_experiment(
        "seed_pilot" if pilot else "seed_full",
        n=len(question_items),
        pending=len(pending),
        dry_run=dry_run,
        model=MODEL_NAME,
    )
    if dry_run:
        print("DRY_RUN=1 set; validation passed, exiting before model/API calls.")
        return
    web_verify.require_serper_unless_disabled(dry_run=dry_run)
    if rejudge:
        pool = rejudge_existing(
            question_items if pilot else None,
            pilot,
            limit=None if pilot else (limit or None),
            resume=resume,
        )
        halls = write_rejudge_pool(pool)
        print(f"Tree pool: {len(halls)} VERIFIED_FALSE / {len(pool)} rejudged -> {REJUDGE_POOL_PATH}")
        print(
            f"\nNext: python forecasting/pipeline.py scale --limit {limit or 400} "
            f"--skip-pilot --resume"
            f"\n  (rejudge already done; scale --resume skips current webscraper rows and grows the tree)"
            f"\nOr tree only: python forecasting/pipeline.py tree --fresh --skip-pilot "
            f"--seeds {REJUDGE_POOL_PATH} --out {DEFAULT_TREE_OUT} --max-seeds {len(halls) or 1} --levels 2"
        )
        return
    if not pending:
        print("Nothing to do.")
        return

    if judge_backend() == "gemini":
        setup_gemini()

    label_counts = Counter()
    duplicate_count = 0
    for index, (question_number, question, domain, sample_index) in enumerate(pending, start=1):
        answer, features, rng_seed = generate_seed_answer(question, question_number, sample_index)
        progress = f"[{index}/{len(pending)}] q{question_number}#{sample_index}"
        if not answer:
            print(
                f"{progress}: empty generation, skipping "
                f"(no visible assistant text after retry; GPT-OSS spent the token budget on "
                f"hidden reasoning or returned an empty message. "
                f"SEED_MAX_NEW_TOKENS={SEED_MAX_NEW_TOKENS})"
            )
            continue
        seen_answers = answers_by_question.setdefault(question_number, set())
        is_duplicate = answer in seen_answers
        seen_answers.add(answer)
        if is_duplicate:
            duplicate_count += 1
            record = {
                "seed_schema_version": SEED_SCHEMA_VERSION,
                "question_number": question_number,
                "sample_index": sample_index,
                "domain": domain,
                "question": question,
                "model_answer": answer,
                "qwen_answer": answer,
                "model_name": MODEL_NAME,
                "duplicate_answer": True,
                "gemini_judgement": "Overall label: Not Hallucinating",
                "prompt_pack_version": prompt_pack_version(),
                "prompt_ids": prompt_ids(),
            }
            write(SEEDS_PATH, record, SEEDS_PATH.exists())
            print(f"{progress}: duplicate of an earlier sample, not judged")
            continue
        label, reason, judge_raw, web_verification = judge_seed(question, answer)
        label_counts[label] += 1
        record = {
            "seed_schema_version": SEED_SCHEMA_VERSION,
            "question_number": question_number,
            "sample_index": sample_index,
            "domain": domain,
            "question": question,
            "model_answer": answer,
            "qwen_answer": answer,
            "model_name": MODEL_NAME,
            "enable_thinking": ENABLE_THINKING,
            "max_new_tokens": SEED_MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "rng_seed": rng_seed,
            "duplicate_answer": False,
        }
        record = apply_seed_judgement(
            record, label, reason, judge_raw, active_judge_model(),
            web_verification=web_verification,
        )
        record["seed_id"] = seed_identifier(record)
        if features:
            record.update({f"gen_{name}": value for name, value in features.items()})
        write(SEEDS_PATH, record, True)
        print(f"{progress}: {label}" + (f" - {reason[:80]}" if reason else ""))

    total = sum(label_counts.values())
    hallucinating = label_counts["Hallucinating"]
    print(f"\nJudged {total} answers: {hallucinating} hallucinating, {total - hallucinating} clean")
    if total:
        print(f"Hallucination rate: {hallucinating / total:.1%}")
    if duplicate_count:
        print(f"Skipped {duplicate_count} duplicate answer(s).")
    print(f"Wrote {SEEDS_PATH}")
    if pilot:
        write_pilot_stage("seeds", n=len(question_items), judged=total, model=MODEL_NAME)
        print("Recorded 10-example seed-prompt debug in forecasting/results/pilot.json")
        print(
            f"\nNext: python forecasting/pipeline.py tree --pilot --fresh "
            f"--seeds {SEEDS_PATH} --out forecasting/cascade_tree_pilot.jsonl --levels 2"
        )
    else:
        print(
            f"\nNext: python forecasting/pipeline.py scale --limit 400 --skip-pilot"
            "\n  rejudges 400 saved answers, then grows the D/N/V tree on those halls"
        )


if __name__ == "__main__":
    main()
