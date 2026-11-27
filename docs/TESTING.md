# Testing & Validation

```bash
pytest -q                  # 73 tests, ~30 s
pytest -q -m "not slow"    # skip the 100k-frame on-disk scale test
```

The latest run is recorded in [`test_results.txt`](test_results.txt) (73 passed).

## Strategy

1. **Ground-truth scenario.** `simulator.SCENARIO` scripts 38 events over a 24 h day. Each event is annotated with
   the outcome the system should produce (for example `"loitering HIGH at 00:01"` or
   `"log: Blue Ford F150 spotted at garage, 12:00."`). The day is generated deterministically (seeded), processed once
   per test session (`day_pipeline` fixture), and checked end to end. The check covers both directions: **every
   expected alert fires, and no unexpected ones do**. Authorised vehicles, wildlife and 1,400 routine patrol frames
   must stay silent.
2. **Unit tests per component** using small hand-written frames (the `frame` fixture), so each rule is tested in
   isolation with its boundary conditions.
3. **External dependencies are mocked, not skipped.** The Claude VLM is tested against a fake client (the request
   shape and the response mapping are both checked). The LangChain agent runs through the **real `create_agent`
   loop** with a scripted chat model that issues a real tool call. Both fallback paths (refusal or network error)
   are tested.
4. **Scale tests** on synthetic streams: throughput, query latency and the LLM context bound.

## Test catalogue

| File | What it verifies | Examples |
|---|---|---|
| `test_perception.py` (16) | Text detection logic + VLM mapping | make/model/colour/category; plate regex; person clothing; multi-object frames with zone carried between clauses; suspicious items; **negation** ("No vehicles at main gate" → nothing); animals; groups; VLM payload → `Detection`, alias → canonical zone, request uses JSON schema + image block; VLM refusal/error → text fallback |
| `test_tracker.py` (5) | Re-identification & visits | same truck = one entity with 2 visits after a 15 min gap; different cars stay distinct; plate links sightings; person dwell time; clothing-based re-ID ignores carried items |
| `test_rules.py` (14) | Every alert rule | **"alert triggered at midnight"**: loitering exactly once, message `Person loitering at main gate, 00:01.`; no loitering for brief presence; no night alert at noon; restricted zone at night is CRITICAL; breach + crowbar; repeat vehicle only on the 2nd visit; **authorised car never alerts**; vehicle dwell; unattended package timing; restricted zone after hours but not in business hours; cooldown suppresses duplicates; battery MEDIUM → CRITICAL escalation; docked drone is exempt; geofence / altitude / wind / telemetry gap |
| `test_pipeline.py` (9) | Full-day scenario | **"truck logged correctly"** (`Blue Ford F150 spotted at garage, 12:00.`, `... 11:58 (2nd visit today).`); expected alert set; no false alerts on authorised vehicles or wildlife; entity visits; every frame indexed; compact memory; JSONL round-trip; deterministic simulation |
| `test_index.py` (9) | Frame index | **"show all truck events"** → 6 rows in time order; frames by time window; attribute filters incl. normalised `f-150`; alert queries; FTS5 BM25; **semantic synonyms** ("pickup" → F150, "intruder breach" → climbing); hybrid ranking ("intruder with a tool" → the crowbar frames, not 180 quiet tool-shed frames); embedding properties; persistence to disk and reopen |
| `test_agent.py` (10) | Q&A and follow-ups | "What objects were in the video?"; visit count + **follow-up "When did it leave?"** (pronoun resolution); "around midnight"; time windows; critical alerts after 22:00; recommendations name the suspect truck; summary; free-text fallback; object-filter parsing; **LangChain agent executes a tool call and grounds its answer**; LLM error → offline answer |
| `test_summary.py` (4) | Summary & recommendations | single sentence with key facts; cross-event correlation links the 12:04 garage tampering to the F150's 2nd visit; prioritisation; empty state |
| `test_scalability.py` (4) | Scale | >1k frames/s on 20k frames; queries < 0.5 s; context < 20k chars; 100k frames to an on-disk DB (`slow`) |
| `test_cli.py` (2) | CLI smoke | `demo --offline` prints the expected log, alert, summary and Q&A sections; `run` → `query --object truck` → `ask` |

## Dynamic inputs & emergency responses

| Scenario | Input | Validated response |
|---|---|---|
| Midnight loiterer | person at gate 00:00:00–00:01:40 | HIGH night alert at 00:00, HIGH loitering at 00:01, **one** alert each (cooldown) |
| Intrusion with tool | climbing fence with crowbar, 22:49 | 2× CRITICAL (breach + suspicious object) with "trigger siren / call security" recommendation |
| Escalating emergency | battery 22 % → 13 % within 8 min | MEDIUM then CRITICAL even inside the cooldown window |
| Comms loss | 150 s telemetry dropout | `telemetry_gap` alert when the link returns |
| Drift | position outside geofence | HIGH geofence alert + return-to-fence recommendation |
| Noise | 1,400 quiet frames, cats/birds/squirrels, owner's car, gardener | no alerts |
| Ambiguous text | "running across backyard toward tool shed" | zone = backyard (destination ignored) |
| Missing landmark | "Man in grey jacket walking" | zone from telemetry/camera hint |
| Unseen long stream | 100k synthetic frames | runs at ~5k fps; context stays ~14k chars |

## Bugs found by tests during development

The first full-day run plus the tests exposed real defects, which were fixed:

* "Visitor parking empty" was detected as a *person* (the word "visitor"), which caused dozens of false night
  alerts. Fixed by removing that noun from the person lexicon.
* Re-identification used *all* attributes, so a person who picked up a crowbar became a new entity. That broke the
  cooldown and duplicated alerts. Fixed with clothing-only signatures.
* A package seen 17 min apart started a new "visit", which reset its dwell and stopped the unattended-package rule
  from ever firing. Fixed: stationary objects don't make visits.
* The low-battery CRITICAL alert was swallowed by the cooldown of the earlier MEDIUM alert. Fixed with escalation
  bypass.
* "running across backyard toward tool shed" was placed in the tool shed. Fixed with destination-aware zone
  resolution.
* Loitering fired on a person who was *climbing*. Fixed: loitering only applies to stationary people.
* Hybrid search for "intruder with a tool" returned quiet "Tool shed door closed" frames (BM25 matched "tool", and
  stop-words like "at" added noise). Fixed with stop-word removal, synonym expansion of the relaxed query, and an
  event-first prior. Covered by `test_hybrid_search`.

## What is not covered

* Live calls to the Claude API (VLM and agent) were **not** run in this environment, because no API key was
  available. Both paths are covered by mocked or scripted tests. To verify them live, export `ANTHROPIC_API_KEY` and
  run `python -m drone_security demo` and `python -m drone_security video --file <clip.mp4>`.
* Detection accuracy on real footage. There was no labelled drone dataset. See the report's "what I'd do next"
  section.
