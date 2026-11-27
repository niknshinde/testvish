# How AI tools were used

The brief asks for AI-assisted development and for that assistance to be documented. This project was built with
**Claude Code**, Anthropic's agentic coding CLI, running in a cloud session with this repository checked out. Claude
Code worked as the main implementer: it read the assignment PDF, planned, wrote the code, ran it, read the output
and iterated. This file records what it did and where its output had to change.

## Workflow

| Stage | What Claude Code did | Outcome |
|---|---|---|
| Requirements | Extracted the brief from the PDF and turned it into a module plan: simulator → perception → tracker → rules → alerts → index → memory → agent → CLI, plus docs, diagrams and report | Plan matched 1:1 to the deliverables table in the README |
| Architecture | Proposed the edge-pipeline + agent-on-top design and checked it against the brief's requirements (real-time alerts, frame indexing, Q&A, scalability). Weighed alternatives (LLM-per-frame alerting, a vector DB, CLIP/BLIP). | Deterministic alert path + LLM analysis layer (see ARCHITECTURE.md §3) |
| API research | Looked up the current Claude API and LangChain 1.x shapes before writing LLM code: JSON-schema structured outputs (`output_config.format`), refusal handling and server-side fallback, `create_agent` and `ChatAnthropic` fields (checked by inspecting the installed packages' signatures) | VLM and agent code written against the installed library versions instead of from memory |
| **AI-generated components** | Generated the **object-detection logic** (lexicons, span merging, action priorities, negation), the **13 alert rules** with thresholds and recommendation text, the **LangChain agent** and its prompt and tools, the **hashing embedder + synonym table**, the **simulated scenario**, and the tests | All in `drone_security/` and `tests/` |
| Cross-domain learning (indexing) | Designed the SQLite schema, picked FTS5 with BM25 and Porter stemming, explained Reciprocal Rank Fusion and implemented hybrid search | `index.py` + `test_index.py` |
| QA | Wrote 73 tests, ran the full simulated day, read the logs, and fixed the defects listed below | 73/73 passing |
| Docs | Wrote the README, spec, architecture, testing doc, Mermaid diagrams (rendered with headless Chromium), and the report (Markdown → HTML → PDF) | `docs/`, `report/` |

## Where the AI's first draft was wrong (and how it was caught)

The first full-day run printed 47 alerts, and most were noise. Reading that output and the tests produced these
fixes:

1. **False "person" detections.** The quiet-frame text "Visitor parking empty" matched the person noun *visitor*,
   which caused about 30 false night and loitering alerts. The noun was removed, and the quiet phrase was added to a
   parametrised negative test.
2. **Brittle re-identification.** People were keyed on *all* attributes, so the same intruder became three entities
   as they picked up a flashlight and a crowbar. That defeated the cooldown and duplicated CRITICAL alerts. The fix
   uses clothing-only signatures.
3. **Package "visits".** A package seen again after 17 min was treated as a new visit, so its dwell reset and the
   unattended-package rule never fired. Stationary objects no longer start visits.
4. **Swallowed escalation.** Battery CRITICAL (13 %) was suppressed by the cooldown from the MEDIUM (22 %) alert 8
   minutes earlier. A test exposed this, and the AlertManager now lets escalations through.
5. **Destination vs location.** "running across backyard toward tool shed" was put in the *tool shed* because the
   longest alias won. Zone resolution now ignores destinations.
6. **Misleading loitering.** A person *climbing* a fence for 2 minutes was reported as loitering. The rule is now
   restricted to stationary people.
7. **Weak search ranking.** While checking the report examples, the assistant noticed that "intruder with a tool"
   returned empty tool-shed patrol frames. It added stop-word removal, synonym expansion and an event-first prior,
   and wrote a regression test for it.
8. **Overstated performance claim.** The docs first said the text analyzer ran at about 20 µs/frame. Measuring it
   gave about 0.1 ms, and the docs were corrected.
9. **Unbounded context.** At 50k frames the agent briefing had grown to 28k characters (one line per hour). Hourly
   episodes older than 24 h are now rolled into one line, and the briefing stays at about 14k characters at 100k
   frames.

## Impact on the workflow

* **Speed.** A working pipeline, tests and docs took one working session instead of days. The biggest gain was the
  short loop: write, run the whole simulated day, read the alert list, fix, re-run.
* **Learning a new area.** Frame indexing (FTS5, BM25, embeddings, RRF) was implemented together with the explanation
  of *why* each piece is there, which is now in `index.py`'s docstring and ARCHITECTURE.md §2.5.
* **Verification still needed.** Every fix above came from running the code and reading the output critically. The
  AI's first draft compiled and looked plausible but produced noisy alerts. The scenario-based tests are what keep
  it honest.

## Limitations of this AI-assisted build

* The live Claude API paths (VLM and LangChain agent) were exercised only with mocked or scripted models, because no
  API key was available in the development environment. Run them with your own key before relying on them.
* The demo videos with voice-over have to be recorded by the candidate. `DEMO_SCRIPT.md` is an AI-drafted script
  and shot list.

## Candidate's notes

> _Add your own reflections here: which suggestions you accepted or changed, what you learned (e.g. LangChain,
> FTS5, VLMs), and anything you would have done differently. The evaluators explicitly ask for this._
