# Demo video script (record with your own voice-over)

The brief requires screen recordings **with voice-over** for each goal. Below is a shot list with the command to run
and talking points. Total is about 8–10 minutes; you can record it as one video or split it by section.
Tip: run `python -m drone_security run --speed 3` for a "live" replay that is easier to narrate.

| # | Goal | On screen | Say (talking points) |
|---|---|---|---|
| 0 | Intro (30 s) | README top + `docs/diagrams/architecture.png` | Problem: a docked drone patrols one property; the agent turns its telemetry and video into logs, alerts, a searchable index and answers. Deterministic alert path, LLM analysis on top. |
| 1 | **Video processing / frame descriptions** (1 min) | `head -5 data/sample_day.jsonl`, then `python -m drone_security demo --offline` section 1 | Simulated frames are text descriptions with telemetry (time, zone, altitude, battery). The same `Detection` schema comes from the Claude VLM for real video (show `perception.py` `ClaudeVisionAnalyzer` + `video.py` change gating). |
| 2 | **Real-time logs & alerts** (2 min) | `python -m drone_security run --speed 3` | Point out `Person loitering at main gate, 00:01.` (night + dwell); the owner's Tesla logged `[authorised]` with no alert; `Blue Ford F150 spotted at garage, 12:00.`; `entered twice today`; the garage tampering CRITICAL; drone safety: telemetry gap, wind, geofence, battery MEDIUM→CRITICAL escalation; the evening intrusion. Mention 42 duplicates were suppressed. |
| 3 | **Frame indexing (cross-domain)** (1.5 min) | `python -m drone_security run --db out/index.db --quiet`, then `query --object truck`, `query --from 00:00 --to 00:05`, `query --search "intruder with a tool"` | SQLite schema (show `index_schema.png`); B-tree for structured queries, FTS5 BM25 for keywords, hashed embeddings for synonyms ("pickup" finds F150), RRF hybrid. |
| 4 | **Context summaries** (1 min) | `python -m drone_security summary` | One-sentence video summary (bonus) + hourly episodes; explain the three memory tiers and why the context stays ~14k chars regardless of video length. |
| 5 | **Agent recommendations & follow-up Q&A** (2 min) | `python -m drone_security chat` (with `ANTHROPIC_API_KEY` set for the Claude agent; otherwise offline). Ask: "What objects were in the video?", "How many times did the blue Ford F150 visit?", "When did it leave?", "What happened around midnight?", "What should I do next?" | Follow-up pronoun resolution; answers cite times and frame numbers; recommendations link the 12:04 garage tampering to the F150's 2nd visit (cross-event reasoning). If using the LLM, point out the tool calls. |
| 6 | **Scalability test** (1 min) | `python -m drone_security benchmark --frames 50000` | ~5k frames/s on one core, single-digit-ms queries, bounded context; a day at 1 fps (86k frames) takes ~20 s, so the VLM is the bottleneck, which is why change gating and a detector-first cascade come next. |
| 7 | **Tests** (30 s) | `pytest -q` | 73 tests: scenario ground truth, every rule, index queries, LangChain loop with a scripted model, VLM mocks, scale. |
| 8 | **Innovation & wrap-up** (30 s) | ARCHITECTURE §3 table | Cross-event correlation, escalation-aware alerting, hybrid search, bounded memory, graceful offline fallback, YAML-configurable site. Next steps from the report. |

Upload the recordings (e.g. Google Drive with "anyone with the link can view") and paste the links into
`report/REPORT.md` → *Videos*, then rebuild the PDF with `python scripts/build_report.py`.
