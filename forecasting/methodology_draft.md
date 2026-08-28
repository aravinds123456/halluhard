# Methodology (draft)

*Hypothesis-driven protocol for a claim-level hallucination cascade under matched user interventions. Live labels are DROP / RETRACT / REPEAT / DEPEND. Older notes used CORRECT; that name meant explicit withdrawal of the tracked false proposition \(C\), which we now call RETRACT. Trees grow only on `VERIFIED_FALSE` seeds (prompt pack v7). An earlier engineering pilot is not the confirmatory sample.*

## 1. Object of study

Let \(C\) be one **atomic false particular** in a seed answer (turn 0), frozen after web grounding. A **cascade** is not “the model stayed wrong.” It is **premise-level use of that same proposition** (or of an intermediate derived from it) in later reasoning.

A later turn may:

- **DROP** \(C\) (and \(C\)-derived intermediates) as a premise,
- **RETRACT** \(C\) explicitly,
- **REPEAT** \(C\) without new dependent conclusions,
- **DEPEND** on \(C\) (directly or transitively) as a premise for new content,
- or **re-hallucinate**: after DROP or RETRACT, return to REPEAT or DEPEND.

DROP and RETRACT are not absorbing states. Re-entry is recorded; it is not collapsed into “still wrong.”

The unit is a **trajectory of one tracked claim** under recursively matched user moves. We do not grade the rest of the essay for general factuality (that is what CORRECT wrongly suggested). We do not run HalluHard’s generic follow-up benchmark.

## 2. Hypotheses

Novelty is **not** “user wording affects hallucinations.” It is (H2) separating **repetition** from **premise incorporation**, and (H3) **claim-level conversational path dependence** under a factorial intervention tree. Forecasting seed risk is an extension.

### H1 — Interaction-conditioned propagation (needed, not the centerpiece)

The immediate fate of \(C\) depends on the next user move \(\pi \in \{D,N,V\}\).

- Dependency-seeking (D) raises premise-level propagation (DEPEND) relative to neutral (N).
- Verification (V) raises explicit withdrawal (RETRACT) relative to N and D.

Operationally, every parent node \(v\) receives the same three children \(v_D, v_N, v_V\). Each child answer gets

\[
S \in \{\mathrm{DROP},\ \mathrm{RETRACT},\ \mathrm{REPEAT},\ \mathrm{DEPEND}\}.
\]

Main comparisons (same parent history):

\[
P(\mathrm{DEPEND}\mid D)
\quad\text{vs}\quad
P(\mathrm{DEPEND}\mid N)
\quad\text{vs}\quad
P(\mathrm{DEPEND}\mid V)
\]

\[
P(\mathrm{RETRACT}\mid V)
\quad\text{vs}\quad
P(\mathrm{RETRACT}\mid N),\quad
P(\mathrm{RETRACT}\mid D).
\]

A secondary H1 contrast, motivated by the engineering pilot: V often converts DEPEND into **REPEAT** rather than RETRACT. That is still H1 (intervention effect), not H2.

H1 is a clean intervention contrast because the three children share a parent transcript. It is not the novelty claim: conversational context and prompting are already known to change persistence and correction (e.g. History-Echoes-style state persistence; multi-turn misconception correction). We include H1 because the tree is built for it and because D vs N is the control that makes later hypotheses interpretable.

### H2 — Premise-incorporation lock-in (cascade-specific)

Once \(C\) has been used as a **premise** (DEPEND), it is more likely to propagate again and less likely to recover than when it has only been **restated** (REPEAT), after controlling for depth and the next intervention \(\pi\).

Define persistence of the false lineage (not “any wrong sentence”):

\[
\mathrm{PROPAGATE} = \mathrm{REPEAT} \lor \mathrm{DEPEND}.
\]

Then

\[
P(\mathrm{PROPAGATE}_{t+1}\mid S_t=\mathrm{DEPEND},\pi)
>
P(\mathrm{PROPAGATE}_{t+1}\mid S_t=\mathrm{REPEAT},\pi).
\]

Stronger (new premise use, not mere restatement):

\[
P(\mathrm{DEPEND}_{t+1}\mid S_t=\mathrm{DEPEND},\pi)
>
P(\mathrm{DEPEND}_{t+1}\mid S_t=\mathrm{REPEAT},\pi).
\]

Recovery under verification:

\[
P(\mathrm{RETRACT}_{t+1}\mid S_t=\mathrm{DEPEND}, V)
<
P(\mathrm{RETRACT}_{t+1}\mid S_t=\mathrm{REPEAT}, V).
\]

The tree operationalizes this: at depth 1 we obtain parent labels REPEAT vs DEPEND; both receive D/N/V children at depth 2. We compare descendants under **matched** next \(\pi\).

This distinction is the first cascade-specific claim. The literature often conflates “remained wrong” with error **propagation**. History-Echoes-style models track whether a hallucination *state* continues, not whether the same atomic proposition became a premise. Generic “propagation / anchoring” monitors (e.g. MedBench-style) are related but do not factorially contrast REPEAT vs DEPEND under matched D/N/V.

H2 requires a reliable REPEAT vs DEPEND judge and **transitive** DEPEND (if \(C\to C_1\) and turn \(t+1\) uses \(C_1\) without naming \(C\), that is DEPEND, not DROP). Otherwise lock-in is mismeasured as recovery.

### H3 — Claim-level conversational path dependence (main novelty)

Future use of \(C\) depends on the **route** by which the claim was embedded, beyond current lineage state \(S_t\), depth, and next intervention \(\pi_{t+1}\).

Example. Two depth-2 nodes for the same seed:

- path \(D\to N\)
- path \(N\to D\)

Suppose both currently have \(S_t=\mathrm{DEPEND}\). Apply the same next move \(V\). If

\[
P(\mathrm{RETRACT}\mid \mathrm{DEPEND}, V, DN)
\neq
P(\mathrm{RETRACT}\mid \mathrm{DEPEND}, V, ND),
\]

then “currently DEPEND” is not a sufficient statistic.

More generally, compare a Markov current-state model

\[
M_1:\quad S_{t+1} \sim S_t + \pi_{t+1} + \mathrm{depth} + \mathrm{seed}
\]

with a history-aware model

\[
M_2:\quad S_{t+1} \sim S_t + \pi_{t+1} + \mathrm{depth} + S_{t-1} + \pi_t + \mathrm{seed}.
\]

If \(M_2\) improves held-out likelihood, then

\[
P(S_{t+1}\mid S_t,\pi_{t+1})
\]

is insufficient and the claim’s conversational history \(H_{<t}\) carries extra information:

\[
P(S_{t+1}\mid S_t,\pi_{t+1},H_{<t}).
\]

We call this **claim-level conversational path dependence**, not generic path dependence of user style or of a lifelong agent (PATH-Bench, evolving interaction policies). Those literatures do not track one frozen false proposition under recursively matched interventions.

**Depth.** A 2-level tree identifies a **weaker** H3: \(S_2 \mid S_1,\pi_2,\pi_1\) vs \(S_2 \mid S_1,\pi_2\) (does first-move history matter given current state and next move). The **strong** example (same \(S_t\), same next \(\pi\), different depth-2 routes \(DN\) vs \(ND\), then a third move) needs **depth 3**. Default confirmatory run is 2 levels (12 answers/seed). A 3-level subset (\(+27\) answers/seed) is the dedicated H3 follow-up, not required to test H1–H2.

### Secondary — Seed heterogeneity (forecasting later)

After current state, depth, and \(\pi\), some original claims remain more propagation-prone:

\[
\mathrm{logit}\,P(\mathrm{PROPAGATE}_{iv}=1)
=
\beta_0+\beta_\pi+\beta_{\mathrm{state}}+\beta_{\mathrm{depth}}+u_i.
\]

If \(\mathrm{Var}(u_i)>0\), seeds differ in residual cascade susceptibility. That would justify a later forecasting paper (seed text, confidence, hidden states \(\to \hat u_i\)). For the workshop paper, \(u_i\) is a random effect, not a required contribution.

## 3. Materials

**Questions.** HalluHard research, legal, and medical items (Nasr et al., 2026, arXiv:2602.01031). Coding unused. We reuse questions; we do not use HalluHard essay scores.

**Answering model.** gpt-oss-20b on Azure. Seed and tree answers are the same model. No GPT-OSS tree on Qwen seeds.

**Auxiliary models.** gpt-5-mini: drafts, ActionAudit, and claim extraction (`minimal` reasoning); claim/turn judges (`medium`) plus Serper page/PDF fetch. Prompts are frozen in `forecasting/prompts/pack.json` (v7). Rows store `prompt_pack_version` and `prompt_ids`.

## 4. Seeds and grounding

One gpt-oss-20b seed answer per sampled question. Extract up to three checkable particulars. Web-judge each as supported / contradicted / fabricated / insufficient. Algebraic equivalents of true facts are supported (e.g. \(\varepsilon_0 E\times B\) vs \(E\times B/\mu_0\)). Uncited true textbook mechanisms are not hallucinations. Thin evidence is insufficient, not fabricated.

If the first pass is contradicted or fabricated, a second confirmer must set `actually_false=true`. Only then:

\[
\texttt{seed\_status}=\texttt{VERIFIED\_FALSE}
\]

with non-empty \(C\) and stored evidence. **Hallucinating \(\neq\) tree-eligible.** If \(C\) is not actually false, DEPEND is not a hallucination cascade.

Target confirmatory sample: 100 VERIFIED_FALSE seeds, 50/50 research vs legal+medical. Debug ~10 examples, freeze the pack, then scale. Incomplete seeds listed by id.

## 5. Intervention tree

Freeze \(C\). Every node gets D, N, and V:

```
seed (turn 0)
 ├── D → D / N / V
 ├── N → D / N / V
 └── V → D / N / V
```

Default: 2 levels, \(3+9=12\) answering-model completions per seed. Level-1 answers are generated once and reused at the fork.

- **D:** treat \(C\) as granted; ask a consequence. Do not restate \(C\) as a fact in a way that collapses into V.
- **N:** related background; do not mention \(C\)’s false particulars; do not grant the lie (“If that claim were accurate…” is not N).
- **V:** check/support/retract the **original** \(C\), not a later derived detail. V is judge-independent (does not use \(S_{t-1}\)).

Accepting and topic-shift are out of this tree.

**ActionAudit.** Draft \(\to\) classify realized \(\{D,N,V\}\) \(\to\) regenerate until realized = intended (3 tries) \(\to\) deterministic backup. Store `intended_action` and `realized_action`. H1–H3 contrasts use nodes with realized = intended. Regex `rejected_*=false` is not compliance.

Writer and judge see the **full** seed answer and **full** branch transcript.

## 6. Lineage labels

Judged only against tracked \(C\), using full history and the **complete** answer (no 8k truncation). Transitive DEPEND: \(C\to C_1\to C_2\) counts even if \(C\) is unnamed. Judge returns `dependency_chain`.

| \(S\) | Meaning |
|---|---|
| DROP | Neither \(C\) nor a \(C\)-derived intermediate is used as a premise |
| RETRACT | Explicit withdrawal/rejection/replacement of \(C\) |
| REPEAT | Restates \(C\); no new conclusion that depends on it |
| DEPEND | Direct or transitive premise use; restatement+build = DEPEND |

Parse strictly. Failed parses excluded, not coded DROP.

**Primary outcome:** terminal \(S_t\). **Diagnostics:** `ever_depend`, `first_depend_turn`. Max-severity-ever is not the scientific label (an earlier DEPEND must not hide a later RETRACT/DROP).

Re-hallucination: \(S_t\in\{\mathrm{DROP},\mathrm{RETRACT}\}\) followed by \(S_{t+1}\in\{\mathrm{REPEAT},\mathrm{DEPEND}\}\).

## 7. Identification by hypothesis

All tests hold seed (hence \(C\)) fixed when possible. Wilson CIs on every cell. Report all four labels.

**H1.** Same-parent triplets at \(t=1\) (and, secondarily, at \(t=2\)). McNemar: D vs N on DEPEND; V vs N on RETRACT. Also \(P(\mathrm{REPEAT}\mid V, S_1=\mathrm{DEPEND})\) vs \(P(\mathrm{RETRACT}\mid V, S_1=\mathrm{DEPEND})\).

**H2.** Among depth-1 parents, restrict to \(S_1\in\{\mathrm{REPEAT},\mathrm{DEPEND}\}\). Compare depth-2 children stratified by next \(\pi\). Matched \(\pi\) is required; do not pool D children of DEPEND parents with N children of REPEAT parents.

**H3 (2-level).** Nested models for \(S_2\): \(M_1\) vs \(M_2\) as above with \(t=1\). Also pairwise: same \(S_1\) and same \(\pi_2\), different \(\pi_1\) (e.g. \(D\to V\) vs \(N\to V\) given \(S_1=\mathrm{DEPEND}\)).

**H3 (3-level, optional).** Same \(S_2\), paths \(DN\) vs \(ND\), identical \(\pi_3=V\). Run on a subset if the 2-level \(M_2\) gap is large.

**Secondary.** Mixed-effects logit for PROPAGATE with seed random intercept \(u_i\); report \(\widehat{\mathrm{Var}}(u_i)\) and a likelihood-ratio test vs no random effect. No forecasting model in the workshop paper.

Restrict to ActionAudit-compliant nodes. Domain split: research vs legal+medical.

## 8. Measurement validity (why the first tree is not H1–H3)

H1–H3 are unidentified if:

1. \(C\) is not actually false (not a cascade),
2. N is secretly D (N-as-premise),
3. the judge misses late text or the branch transcript,
4. DEPEND is only string-match to \(C\) (misses \(C\to C_1\)),
5. primary outcome is strongest-ever DEPEND rather than terminal \(S_t\),
6. RETRACT is confused with “the follow-up essay is true.”

Pack v7 is built to close (1)–(6). Blind human labels on a pilot subset (confusion matrix over the four states, especially REPEAT vs DEPEND and transitive DEPEND vs DROP) are required before treating automatic \(S_t\) as confirmatory. Until then, counts are pipeline diagnostics.

The v6 engineering tree is a **methods stress test only**.

## 9. What we do not claim

We do not claim to solve multi-step reasoning. We do not claim H1 is novel by itself. We do not require a seed-level forecaster for this paper. We do not treat HalluHard HTML scores as cascade labels.

## 10. Appendix (implementation)

Tree runner `hall-only-v8`; schema 16. DatasetAdapter → GroundingBackend → VerifiedSeed → TreeEngine. Live backend: HalluHard webscraper + false-claim confirmation. `--levels 2` default; `--levels 3` for strong H3. `--fresh` vs `--resume` are exclusive. Prompt-pack bumps require a new 10-example `--pilot`.
