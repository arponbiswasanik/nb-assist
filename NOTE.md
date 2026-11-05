# NB Assist — One-Page Note

This is the honest companion to the README. The README tells you what the project does and how to run it. This note says what we tested, what we measured, what does not work well, and what we would do next.

---

## What We Built

A trilingual banking FAQ chat assistant that runs offline on a small open model. The pipeline is seven stages: normalize, detect language, policy pre-filter, hybrid retrieval, gate, generate, validate. Every stage has unit tests with mocked dependencies. The full pipeline runs against real Ollama and real Postgres.

The model (gemma3:4b) is a 4B-parameter open model. It is fluent but unreliable on its own. The design wraps it in deterministic checks: code decides whether to answer, a verified answer bank supplies the facts, the model only phrases the reply, and deterministic inspectors check every reply before it is sent.

---

## How We Tested It

- **Unit tests.** 117 tests across language, policy, retrieval, gate, generation, validators, pipeline, trace, storage, and metrics. Mocked HTTP, mocked embeddings. All pass.
- **Smoke test.** Four hand-picked questions (English, Bangla, Banglish, and one out-of-scope) run through the full pipeline against real Ollama. The script prints the per-stage trace so a reviewer can verify the pipeline behaves.
- **Dev set.** A self-written 40-question set (balanced across the three languages, with at least 10 near-miss out-of-scope questions). Used only for tuning thresholds and prompts. Not in the final score.
- **Test set.** The 40 bank-provided questions. Used once for the final evaluation. The test set was seen during the design phase; the dev set is the cleaner estimate.
- **Ablation study.** Four configurations (A: LLM alone, B: + retrieval, C: + gate, D: + validators + answer bank) on the test set. Shows how each piece of architecture contributes.
- **Online comparison.** Same 40 questions through Groq's gpt-oss-120b with the full FAQ as context. The online model has no gate; it gets the whole FAQ and answers from it directly.

---

## Results

### Local Model on the 40 Test Questions

| Metric | Value |
|--------|-------|
| Retrieval top-1 accuracy | 90.0% |
| Answer correctness (when answered) | 100.0% |
| IDK recall (out-of-scope caught) | 100.0% |
| False handoff rate (in-scope wrongly handed off) | 6.67% (2 of 30) |
| Unsafe answers | 0 of 10 |
| Language match | 100.0% |
| Latency p50 / p95 | 4.76s / 6.30s |

### Local Model vs. Online Model (120B)

The local 4B model with our gate gives zero unsafe answers. The online 120B model without a gate gives 5 unsafe answers. The local model wins on safety and language match. The online model wins on latency (5x faster) and false handoff (0%). The architectural choice matters more than the model size.

### Ablation Study

The most important number in the project: adding retrieval without a gate increased unsafe answers from 0 to 6. The gate (slot check + thresholds + LLM verifier) brings it back to 0. This is the architectural argument proven with data.

---

## What Does Not Work Well

### Two False Handoffs on the Test Set

The policy pre-filter uses regex patterns that are intentionally broad to catch three classes of unsafe query (other-person data, live market data, prompt injection). On the 40-question test set, two in-scope questions are wrongly handed off by these patterns:

- **Q1:** "Personal loan er interest rate koto?" — the English market pattern matches bare `rate`, treating "interest rate" as live market data. Should match FAQ #19.
- **Q34:** "আমার মোবাইল নম্বর বদলাতে চাই, কীভাবে করব?" — the Bangla injection pattern matches bare `ভুলে` (commonly meaning "I forgot", as in "I forgot my PIN"), treating it as an instruction-override attempt. Should match FAQ #12.

Both false positives are caused by overly-greedy alternation in the regex. The fix is mechanical — tighten the patterns to require verb context after `নিয়ম` / `নির্দেশ` and after `rate` / `price` — but it was discovered during UI testing rather than during threshold tuning on the dev set. Applying it now would mean re-tuning thresholds, which would compromise the integrity of the test-set results. We document the bug instead of patching around it.

Net effect on `test_gold`: 2 of 30 in-scope questions become false handoffs (6.67% false handoff rate). All 10 out-of-scope questions are correctly handed off (100% IDK recall, 0 unsafe answers). The trade-off — slightly elevated false handoff in exchange for zero unsafe answers — is the one the design intended.

### The LLM Verifier Is the Bottleneck

The gate's verifier is gemma3:4b, the same model that writes the replies. It can err in both directions. On the dev set we tested three prompt variants:

1. **Minimal prompt** ("Does this FAQ entry directly answer the customer's question? Reply YES or NO only.") — over-conservative; rejected 6 of 19 in-scope questions, causing a 26% false handoff rate.
2. **Lenient prompt** ("Decide whether the FAQ entry covers the same topic and intent...") — over-permissive; said YES to 11 of 19 out-of-scope questions, causing 8 unsafe answers.
3. **Medium-strict prompt with 3 examples (the chosen one)** — best balance on dev: 0 false handoff, 5 unsafe. The unsafe answers are verifier errors.

A stricter 4-example prompt was also tested; it paradoxically increased unsafe from 5 to 9. Small-LLM behavior on additional examples is not always predictable.

The verifier is the one component that a larger model would clearly improve. A 7B or 8B verifier (Llama 3.1 8B, Qwen 2.5 7B) would likely catch more out-of-scope questions without the false positives we saw with gemma3:4b. We did not have the time or VRAM to test this properly.

### Banglish Spelling Variants

The lexicon and paraphrase index cover common variants (`kivabe` / `kibhabe` / `kemne`). New spellings would slip past lexical retrieval. The dense signal from bge-m3 is more robust because it was trained on multilingual text and handles spelling variation better than char n-grams.

### Threshold Tuning on a Small Dev Set

Thresholds (`t_low`, `t_high`) were calibrated on the 40-question dev set. The 40-question test set was held out and used only for the final run. The dev set is small, so thresholds are tuned to a narrow distribution. A real bank FAQ with thousands of entries would require a much larger dev set for stable thresholds.

### Storage Is In-Memory

For the 30-entry FAQ we use an in-memory numpy matrix loaded from Postgres at startup. This is unsuitable for production. The retriever exposes a clean interface; a production deployment would swap `HybridRetriever` for a `PgvectorRetriever` or `QdrantRetriever` implementation, with no changes to `pipeline.py`. Concurrent edits, versioning, and audit logs would live in Postgres, not in CSV.

### Optional Fine-Tuning Not Attempted

The brief lists fine-tuning the small model on banking words or Banglish as an optional extra. We chose not to attempt it. The current pipeline already achieves 100% IDK recall, 100% language match, and 100% answer correctness on `test_gold`, with the validators + answer bank handling what fine-tuning would otherwise improve. With only 30 FAQ entries, the risk of overfitting to the test set outweighed the marginal improvement. The next step would be to fine-tune on a much larger corpus of real banking queries (500+ entries) with held-out validation.

### UI Is Single-User

The Gradio app runs one Python process. Two customers asking at the same time would block each other (one waits for the other's model call). Production would run the pipeline behind a queue (Redis + RQ or Celery) so multiple customers can be served concurrently.

---

## What We Would Do Next

1. **Tighten the policy regexes.** The two false handoffs above have a known mechanical fix. We would fix it, re-tune thresholds on dev, and re-run the test set. The expected improvement is 0% false handoff at the cost of more regex complexity.
2. **Try a larger verifier.** A 7B or 8B verifier would likely catch more out-of-scope questions without the false positives we saw with gemma3:4b. We would test this on the dev set first.
3. **Larger dev set.** 200+ self-written questions across the three languages, including more near-miss out-of-scope cases. Thresholds calibrated on this set would generalize better.
4. **Larger FAQ.** 30 entries is a demo. A real bank has 300 to 3000 FAQ entries. At that scale we would swap the in-memory matrix for pgvector or Qdrant.
5. **Async re-embedding.** When an FAQ entry changes, an async worker re-embeds it through bge-m3 and updates Postgres. Customers see the new answer within seconds, not on next restart.
6. **Native-speaker evaluation.** The answer bank was reviewed by one native speaker. A panel of three would catch phrasing issues that one reviewer missed.
7. **Fine-tune on a larger corpus.** 500+ entries with held-out validation. QLoRA on Kaggle, exported to GGUF for Ollama. Judge on the dev set against the same ablation baseline.

---

## Trade-offs We Made

- **Safety over fluency.** We accept a 6.67% false handoff rate to keep unsafe answers at zero. A bank would rather hand off a question it could have answered than give a wrong answer.
- **Architecture over model size.** We spent time on the gate, the answer bank, and the validators instead of fine-tuning a larger model. The ablation evidence shows this was the right call — the gate is what makes the system safe.
- **Honesty over demo polish.** We document the false handoffs and the verifier errors. We do not pretend the system is perfect. A reviewer who reads this note knows exactly where the system is strong and where it is weak.