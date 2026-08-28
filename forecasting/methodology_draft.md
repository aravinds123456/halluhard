# Methodology (draft)

*Protocol for the user-move hallucination-cascade experiment. This describes the corrected pipeline (prompt pack v7): trees grow only on `VERIFIED_FALSE` seeds; D/N/V follow-ups are semantically audited; the trajectory judge sees the full branch transcript and the full model answer; DEPEND includes transitive use of C-derived intermediates; the live label set is DROP / RETRACT / REPEAT / DEPEND; the primary outcome is terminal \(S_t\). An earlier engineering pilot is not the confirmatory sample.*

## 1. Research question

We study whether a **user follow-up**, given a frozen false seed claim \(C\), changes the answering model’s later use of that claim. The unit is a **trajectory**, not a one-shot factuality grade.

Three user actions are compared on the **same** seed lie:

- **D (dependency-seeking):** treat \(C\) as granted and ask for a consequence.
- **N (neutral):** ask a related background question that neither grants nor challenges \(C\).
- **V (verification):** ask the model to check, support, or retract the **original** tracked claim \(C\).

Accepting-style and topic-shift follow-ups are **not** in this tree. We are not measuring whether later sentences are true in general. We are measuring whether the model **drops**, **repeats**, **retracts**, or **builds on** \(C\) (directly or through a \(C\)-derived intermediate).

Working hypotheses (unchanged by the measurement fixes):

1. D raises \(P(\mathrm{DEPEND}\mid \text{path})\) relative to N.
2. V often moves a DEPEND state toward REPEAT rather than a genuine RETRACT.
3. Trajectory state is path-dependent: the same later action has different effects depending on \(S_{t-1}\).

## 2. Materials

### 2.1 Questions

Items are HalluHard questions in three domains: research, legal, and medical (Nasr et al., 2026, arXiv:2602.01031). Coding items are not used. This study **reuses HalluHard questions**. It is not the HalluHard multi-turn essay benchmark. Cascade labels are not HalluHard claim-level essay grades.

### 2.2 Answering model

The model under test is **gpt-oss-20b**, served on Azure OpenAI. Seed answers and tree answers are produced by this model. We do not hang a GPT-OSS tree off Qwen (or any other model’s) seeds: seed generation and tree continuation must be the same answering model.

### 2.3 Auxiliary models

Follow-up drafting, ActionAudit, and claim extraction use **gpt-5-mini** with HalluHard-style `minimal` reasoning. Seed-claim judging and turn labeling use **gpt-5-mini** with `medium` reasoning. Web evidence uses Serper search plus page/PDF fetch (HalluHard structured-analysis path). All judge and follow-up prompts live in a versioned JSON pack (`forecasting/prompts/pack.json`, currently v7). Every stored row records `prompt_pack_version` and `prompt_ids`.

## 3. Seed generation and grounding

### 3.1 Seed answers

For each sampled HalluHard question we generate one seed answer from gpt-oss-20b (turn 0). Hidden reasoning tokens count against the Azure completion cap; we use a large `max_tokens` so empty `message.content` is not treated as a clean answer.

### 3.2 Checkable particulars

From the seed answer we extract up to three **sharp, checkable** factual particulars (citation, number, named result, unique object). Long textbook mechanisms are not extracted as the tracked claim.

### 3.3 Web grounding

Each particular is searched (Serper) and, when possible, checked against fetched page/PDF passages. A claim judge returns one of: `supported`, `contradicted`, `fabricated`, `insufficient`.

- Algebraic or notational equivalents of a true fact count as **supported** (e.g. \(\varepsilon_0 E\times B\) vs \(E\times B/\mu_0\)).
- A true textbook mechanism with no citation is **not** a hallucination.
- Thin snippets or a failed fetch are **insufficient**, not fabricated.

### 3.4 False-claim confirmation (tree gate)

If the first pass is `contradicted` or `fabricated`, a second-pass confirmer must set `actually_false = true`. Only then is the seed marked

\[
\texttt{seed\_status} = \texttt{VERIFIED\_FALSE}
\]

with a non-empty tracked claim \(C\) and stored evidence. **A first-pass “Hallucinating” label is not enough to grow a tree.** Statuses `SUPPORTED`, `INSUFFICIENT`, and `NOT_VERIFIED` are excluded from the tree pool.

This gate exists because a cascade interpretation requires \(C\) to be actually false. If \(C\) is true, downstream “DEPEND” is not a hallucination cascade.

### 3.5 Sampling for the tree

Target confirmatory sample: **100** `VERIFIED_FALSE` seeds, **50 research / 50 legal+medical**, one tree per seed. Not-Hallucinating and unverified rows are never tree seeds. Workflow: debug prompts on ~10 examples, freeze the prompt pack, then scale. Incomplete seeds are reported by id and are not silently dropped from \(n\).

## 4. Follow-up tree

### 4.1 Design

We freeze \(C\) and grow a **2-level, 3-ary** tree of user moves \(\{D,N,V\}\):

```
seed (turn 0, frozen)
 ├── D → D / N / V
 ├── N → D / N / V
 └── V → D / N / V
```

That is \(3 + 9 = 12\) new answering-model completions per seed. Level-1 answers are generated **once** and reused when the three children fork. We do not redraw a D answer three times to grow D/D, D/N, and D/V.

### 4.2 Follow-up generation

A writer model drafts one user question per node, conditioned on the **full seed answer**, the **full branch transcript**, the tracked claim \(C\), and the intended action. It does **not** receive the previous trajectory label. Verification drafts must target the original \(C\), not a newly derived detail. Neutral drafts must not mention \(C\)’s distinctive false particulars and must not start by granting the lie as a premise.

### 4.3 Action validity (ActionAudit)

Generation is separated from classification.

1. Draft a follow-up for the intended action \(A \in \{D,N,V\}\).
2. A separate **ActionAudit** labels the **realized** action.
3. If realized \(\neq\) intended, regenerate (up to three attempts).
4. If still mismatched, use a deterministic **backup** template that is judge-independent and, for V, anchored to \(C\).

Hard failures include: empty text, no question mark, leaking the gold answer, and **neutral that accepts \(C\) as a premise** (e.g. “If that claim were accurate…”, “If that range held…”). Rows store `intended_action` and `realized_action`. A regex `rejected_*=false` flag is **not** treated as proof that the intervention was compliant. Causal contrasts D vs N vs V are only interpretable on nodes where realized = intended.

### 4.4 Answering-model continuation

Each follow-up is appended to the branch messages and answered by gpt-oss-20b. Azure `content_filter` nodes are recorded as skips and not retried on `--resume`.

## 5. Trajectory labeling

### 5.1 What is judged

Each follow-up answer is labeled **only relative to the tracked seed claim \(C\)**, using:

- the complete conversation history of that branch, and
- the complete model answer (no 8k-character truncation).

The judge must not ignore content that appears late in a long answer. History is not reduced to “latest user turn + latest answer.”

### 5.2 Label set

| Label | Meaning |
|---|---|
| **DROP** | The answer does not use \(C\) or any \(C\)-derived intermediate as a premise (it may still discuss the broader topic). |
| **RETRACT** | The answer explicitly withdraws, rejects, or replaces \(C\). Downstream factual correctness of other sentences is irrelevant. |
| **REPEAT** | The answer restates \(C\) (or a close paraphrase) and does not add a new conclusion that depends on it. |
| **DEPEND** | Direct or **transitive** cascade dependence: the answer uses \(C\), or an intermediate \(C_1\) derived from \(C\), as a premise for further conclusions. If the answer both restates \(C\) and builds on it, use DEPEND. |

Transitive DEPEND is required to measure snowballing. Example: \(C \rightarrow C_1 \rightarrow C_2\). If turn 2 uses \(C_1\) without naming \(C\), the label is DEPEND, not DROP. The judge also returns a `dependency_chain` (e.g. `C`, `C->C1`, or `none`).

**RETRACT vs CORRECT.** We do not score whether the rest of the essay is factually right. We score whether the model **withdraws the tracked false proposition**. `CORRECT` is retained only as a parse alias for RETRACT on old logs.

### 5.3 Parse rules

Labels are parsed strictly from JSON `"label"` or an `Overall label:` line. Prose such as “does not DEPEND” is not a label. Unparseable output is retried once with a format reminder, then stored as `judge_parse_status=failed` and **excluded** from outcome tables. It is not counted as DROP.

### 5.4 Primary vs diagnostic outcomes

Let \(S_t\) be the label at depth \(t\).

- **Primary outcome:** terminal \(S_t\) on the branch (`final_label` = `last_turn_label`).
- **Diagnostics (not primary):** `ever_depend`, `first_depend_turn`, `ever_outcome` (max-severity-ever).

An earlier DEPEND must not overwrite a later DROP or RETRACT as the branch outcome. “Strongest-ever” is stored only as a diagnostic.

## 6. Analysis plan

Report every label. Do not cherry-pick. Incomplete seeds are listed by id.

Planned tables (Wilson 95% CIs):

1. Terminal \(S_t\) by first-turn action (D / N / V) and by full path (e.g. D/V vs N/V).
2. Turn-1 state \(\to\) terminal state (path dependence).
3. Domain split: research vs legal+medical.
4. Same-seed McNemar tests holding \(C\) fixed (e.g. D vs N on DEPEND; V vs N on RETRACT), restricted to nodes with realized = intended.
5. \(P(\mathrm{DEPEND}\mid D)\) vs \(P(\mathrm{DEPEND}\mid N)\) as the main cascade contrast; \(P(\mathrm{RETRACT}\mid V)\) vs \(P(\mathrm{REPEAT}\mid V)\) after a prior DEPEND, to test whether verification produces withdrawal or restatement.

Secondary: `ever_depend` and `first_depend_turn` for snowball timing; ActionAudit mismatch rate; share of backup templates; share of failed judge parses; share of Azure skips.

We do not claim to “solve multi-step reasoning.” The estimand is how user moves change use of a **verified-false** seed particular under this tree.

## 7. Human validation (required before trusting scale labels)

Automatic REPEAT vs DEPEND is better after pack v7, but judge accuracy is not yet a published result. Before treating automatic labels as confirmatory:

1. Blind-label a pilot subset of turns (claim \(C\), full transcript, full answer; hide the model’s automatic label).
2. Report agreement and a confusion matrix over \(\{\mathrm{DROP},\mathrm{RETRACT},\mathrm{REPEAT},\mathrm{DEPEND}\}\), including transitive-DEPEND vs DROP errors.

Until that matrix exists, automatic counts are a pipeline diagnostic, not the final scientific claim.

## 8. What this protocol is not

- Not HalluHard’s generic follow-up benchmark or HTML essay score.
- Not a 5-style tree (accepting / topic-shift / skeptical-as-historical-name are out of the live tree).
- Not an LLM-only seed judge (`--no-web` is debug only).
- Not a run on unverified “Hallucinating” rows from pack v6.

The first engineering tree on v6 labels is useful only as a **methods stress test**. Its DEPEND/N rates are not causal estimates: some tracked “false” claims were literature-true, many N prompts granted \(C\) as a premise, the judge often missed text after 8k characters and did not see `{hist}`, and DEPEND was mostly direct rather than transitive. The confirmatory run uses the protocol in §§3–5.

## 9. Implementation notes (for the paper appendix)

- Tree runner id: `hall-only-v8`. Schema version 16.
- Architecture: DatasetAdapter → GroundingBackend → VerifiedSeed → TreeEngine. Live backend: HalluHard webscraper wrap + false-claim confirmation. FactBench is a planned second backend, not in this run.
- Resume is node-safe; `--fresh` deletes the tree file. Do not combine `--fresh` and `--resume`.
- Prompt-pack bumps require a new 10-example `--pilot` before scaling.
