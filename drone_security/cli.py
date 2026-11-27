"""Command-line interface.

    python -m drone_security demo                       # end-to-end showcase (offline)
    python -m drone_security simulate --out data/sample_day.jsonl
    python -m drone_security run --input data/sample_day.jsonl --db out/index.db
    python -m drone_security query --db out/index.db --object truck
    python -m drone_security query --db out/index.db --from 00:00 --to 00:10
    python -m drone_security query --db out/index.db --search "person near gate"
    python -m drone_security ask "How many times did the blue Ford F150 come today?"
    python -m drone_security chat                       # multi-turn Q&A (LLM if ANTHROPIC_API_KEY set)
    python -m drone_security summary
    python -m drone_security benchmark --frames 50000
    python -m drone_security video --file patrol.mp4 --fps 1   # real video through the Claude VLM
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

from .agent import OfflineSecurityAgent, make_agent
from .alerts import console_sink
from .config import load_config
from .index import FrameIndex
from .models import Frame, Severity, Telemetry
from .pipeline import SecurityPipeline, read_stream, write_stream
from .simulator import simulate_day, synthetic_stream
from .summarizer import one_sentence_summary, recommendations

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "sample_day.jsonl"
BOLD, DIM, RESET, CYAN, GREEN = "\033[1m", "\033[2m", "\033[0m", "\033[36m", "\033[32m"


def _c(s: str, code: str, enabled: bool) -> str:
    return f"{code}{s}{RESET}" if enabled else s


def _records(path: str | None):
    p = Path(path) if path else DEFAULT_INPUT
    if not p.exists():
        return simulate_day()
    return list(read_stream(p))


def _build(args, sinks=None, log_sink=None) -> SecurityPipeline:
    cfg = load_config(getattr(args, "config", None))
    analyzer = None
    if getattr(args, "vlm", False):
        from .perception import ClaudeVisionAnalyzer

        analyzer = ClaudeVisionAnalyzer(cfg)
    db = getattr(args, "db", None) or ":memory:"
    if db != ":memory:" and Path(db).exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
    return SecurityPipeline(cfg, analyzer=analyzer, db_path=db, sinks=sinks, log_sink=log_sink)


def _print_header(title: str, colour: bool) -> None:
    print("\n" + _c(f"== {title} " + "=" * max(0, 70 - len(title)), BOLD, colour))


# ---------------------------------------------------------------------------
def cmd_simulate(args) -> None:
    day = date.fromisoformat(args.day) if args.day else None
    recs = simulate_day(day=day, seed=args.seed) if day else simulate_day(seed=args.seed)
    n = write_stream(recs, args.out)
    frames = sum(1 for r in recs if isinstance(r, Frame))
    print(f"Wrote {n} records ({frames} frames, {n - frames} telemetry samples) to {args.out}")


def cmd_run(args) -> SecurityPipeline:
    colour = not args.no_colour and sys.stdout.isatty() or args.colour
    show_logs = not args.quiet

    def log_sink(l):
        if show_logs:
            print(_c("[LOG]", CYAN, colour), l.message)

    p = _build(args, sinks=[console_sink(colour=colour)], log_sink=log_sink)
    delay = 1.0 / args.speed if args.speed else 0.0
    t0 = time.perf_counter()
    for rec in _records(args.input):
        if isinstance(rec, Telemetry):
            p.process_telemetry(rec)
        else:
            res = p.process_frame(rec)
            if args.show_frames and res.detections:
                print(_c(f"  frame#{rec.frame_id} {rec.ts:%H:%M:%S} [{rec.zone}] {rec.description}", DIM, colour))
            if delay and (res.detections or res.alerts):
                time.sleep(delay)
    p.index.flush()
    p.stats.seconds = time.perf_counter() - t0
    s = p.stats
    _print_header("Run complete", colour)
    print(f"{s.frames} frames, {s.telemetry} telemetry samples, {s.detections} detections, {s.logs} log lines, "
          f"{s.alerts} alerts ({p.alerts.suppressed} duplicates suppressed) in {s.seconds:.2f}s")
    if args.db:
        print(f"Index written to {args.db}")
    _print_header("Video summary", colour)
    print(one_sentence_summary(p))
    _print_header("Agent recommendations", colour)
    for r in recommendations(p):
        print(" -", r)
    return p


def cmd_query(args) -> None:
    if args.db and Path(args.db).exists():
        idx = FrameIndex(args.db)
    else:
        p = _build(argparse.Namespace(db=None, config=args.config))
        p.run(_records(args.input))
        idx = p.index
    if args.search:
        rows = idx.hybrid_search(args.search, limit=args.limit)
        for r in rows:
            print(f"{r['ts'][11:19]} frame#{r['frame_id']:<5} [{r['zone']}] {r['caption'] or r['description']}")
    elif args.alerts:
        for a in idx.alerts(min_severity=int(Severity.parse(args.min_severity))):
            print(f"{a['ts'][11:16]} {a['severity_name']:<8} {a['message']}")
    elif args.object or args.color or args.make or args.zone or args.obj_type:
        from .agent import parse_object_filters

        f = parse_object_filters(" ".join(filter(None, [args.object, args.color, args.make])),
                                 [z.name for z in load_config(args.config).zones])
        if args.obj_type:
            f["obj_type"] = args.obj_type
        if args.zone:
            f["zone"] = args.zone
        rows = idx.query_detections(start=args.start, end=args.end, limit=args.limit, **f)
        print(f"# filters: {f}")
        for r in rows:
            print(f"{r['ts'][11:19]} frame#{r['frame_id']:<5} {r['label']:<28} {r['action'] or '-':<11} {r['zone']}")
    elif args.start or args.end:
        for r in idx.frames_between(args.start or "00:00", args.end or "23:59", limit=args.limit):
            print(f"{r['ts'][11:19]} frame#{r['frame_id']:<5} [{r['zone']}] {r['description']}")
    else:
        print(json.dumps(idx.stats(), indent=2))
        for r in idx.object_counts():
            print(f"  {r['category']:<12} {r['entities']:>3} entities {r['detections']:>5} detections")


def _agent_for(args):
    p = _build(argparse.Namespace(db=None, config=args.config, vlm=False))
    p.run(_records(args.input))
    agent = OfflineSecurityAgent(p) if args.offline else make_agent(p)
    return p, agent


def cmd_ask(args) -> None:
    _, agent = _agent_for(args)
    print(f"[{type(agent).__name__}]")
    print(agent.ask(" ".join(args.question)))


def cmd_chat(args) -> None:
    p, agent = _agent_for(args)
    print(f"Security Analyst ready ({type(agent).__name__}). {p.stats.frames} frames loaded. "
          "Ask anything; empty line to quit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except EOFError:
            break
        if not q:
            break
        print("agent>", agent.ask(q))


def cmd_summary(args) -> None:
    p = _build(argparse.Namespace(db=None, config=args.config))
    p.run(_records(args.input))
    print(one_sentence_summary(p))
    print("\nHourly episodes:")
    for line in p.memory.episodic_lines():
        print("  " + line)


def cmd_benchmark(args) -> None:
    db = args.db or ":memory:"
    p = _build(argparse.Namespace(db=db, config=args.config))
    t0 = time.perf_counter()
    stats = p.run(synthetic_stream(args.frames, seed=args.seed))
    ingest = time.perf_counter() - t0
    timings = {}
    for name, fn in {
        "structured: all trucks": lambda: p.index.query_detections(category="truck", limit=10_000),
        "time window: 1 hour": lambda: p.index.frames_between("2026-01-01T01:00:00", "2026-01-01T02:00:00"),
        "FTS5: 'climbing fence'": lambda: p.index.search_text("climbing fence", limit=50),
        "vector: 'intruder'": lambda: p.index.semantic_search("intruder near the fence", limit=20),
        "hybrid: 'pickup at gate'": lambda: p.index.hybrid_search("pickup at gate", limit=20),
    }.items():
        fn()  # warm (vector cache)
        t = time.perf_counter()
        n = len(fn())
        timings[name] = ((time.perf_counter() - t) * 1000, n)
    print(f"Ingested {stats.frames:,} frames ({stats.detections:,} detections, {stats.alerts:,} alerts) "
          f"in {ingest:.2f}s -> {stats.frames / ingest:,.0f} frames/s")
    print(f"Context block size: {len(p.memory.context_block()):,} chars (bounded regardless of video length)")
    for name, (ms, n) in timings.items():
        print(f"  {name:<28} {ms:8.2f} ms  ({n} rows)")


def cmd_video(args) -> None:
    from .perception import ClaudeVisionAnalyzer
    from .video import load_telemetry_csv, video_to_frames

    cfg = load_config(args.config)
    start = datetime.fromisoformat(args.start) if args.start else datetime.now().replace(microsecond=0)
    tel = load_telemetry_csv(args.telemetry, start) if args.telemetry else None
    frames = video_to_frames(args.file, args.frames_dir, fps=args.fps, start=start, telemetry=tel,
                             default_zone=args.zone)
    print(f"{len(frames)} frames kept after change-gating")
    p = SecurityPipeline(cfg, analyzer=ClaudeVisionAnalyzer(cfg), db_path=args.db or ":memory:",
                         sinks=[console_sink()], log_sink=lambda l: print("[LOG]", l.message))
    for f in frames:
        res = p.process_frame(f)
        print(f"frame#{f.frame_id} {f.ts:%H:%M:%S}: {f.caption}")
    p.index.flush()
    print("\n" + one_sentence_summary(p))


def cmd_demo(args) -> None:
    colour = sys.stdout.isatty() or args.colour
    _print_header("1. Simulated input (telemetry + frame descriptions)", colour)
    recs = _records(args.input)
    frames = [r for r in recs if isinstance(r, Frame)]
    print(f"{len(recs)} records: {len(frames)} frames, {len(recs) - len(frames)} telemetry samples. Examples:")
    for f in frames[:2] + [fr for fr in frames if fr.frame_id in (514, 723, 1372)]:
        t = f.telemetry
        print(f"  Frame {f.frame_id} | Time {f.ts:%H:%M:%S} | Location {f.zone} | alt {t.altitude_m if t else '-'} m "
              f"| batt {t.battery_pct if t else '-'}% | \"{f.description}\"")
    _print_header("2. Real-time processing: logs + alerts", colour)
    ns = argparse.Namespace(input=args.input, db=args.db, config=args.config, vlm=False, quiet=False, speed=args.speed,
                            show_frames=False, no_colour=not colour, colour=colour)
    p = cmd_run(ns)
    _print_header("3. Frame index queries", colour)
    print(_c('> show all truck events', GREEN, colour))
    for r in p.index.query_detections(category="truck"):
        print(f"  {r['ts'][11:16]} frame#{r['frame_id']} {r['label']} {r['action'] or 'seen'} at {r['zone']}")
    print(_c("> frames between 00:00 and 00:02", GREEN, colour))
    for r in p.index.frames_between("00:00", "00:02"):
        print(f"  {r['ts'][11:19]} frame#{r['frame_id']} [{r['zone']}] {r['description']}")
    print(_c('> hybrid search "intruder with a tool"', GREEN, colour))
    for r in p.index.hybrid_search("intruder with a tool", limit=4):
        print(f"  {r['ts'][11:16]} frame#{r['frame_id']} [{r['zone']}] {r['description']}")
    _print_header("4. Agent Q&A (follow-up questions)", colour)
    agent = OfflineSecurityAgent(p) if args.offline else make_agent(p)
    print(f"(agent: {type(agent).__name__})")
    for q in ["What objects were in the video?", "How many times did the blue Ford F150 visit today?",
              "When did it leave?", "What happened around midnight?", "Any critical alerts after 22:00?",
              "What should I do next?"]:
        print(_c(f"\nQ: {q}", GREEN, colour))
        print(agent.ask(q))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="drone_security", description="Drone Security Analyst Agent prototype")
    ap.add_argument("--config", default=None, help="path to property.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate a simulated day (telemetry + frames) as JSONL")
    s.add_argument("--out", default=str(DEFAULT_INPUT))
    s.add_argument("--day", default=None)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_simulate)

    r = sub.add_parser("run", help="process a stream: logs, alerts, index, summary")
    r.add_argument("--input", default=None)
    r.add_argument("--db", default=None, help="SQLite index path (default: in-memory)")
    r.add_argument("--vlm", action="store_true", help="use the Claude VLM analyzer instead of the text parser")
    r.add_argument("--quiet", action="store_true", help="hide log lines, show alerts only")
    r.add_argument("--show-frames", action="store_true")
    r.add_argument("--speed", type=float, default=0, help="events per second for a 'live' replay (0 = max)")
    r.add_argument("--no-colour", action="store_true")
    r.add_argument("--colour", action="store_true")
    r.set_defaults(fn=cmd_run)

    q = sub.add_parser("query", help="query the frame index")
    q.add_argument("--db", default=None)
    q.add_argument("--input", default=None)
    q.add_argument("--object", default=None, help="e.g. truck, 'blue ford f150', person, deer")
    q.add_argument("--obj-type", default=None)
    q.add_argument("--color", default=None)
    q.add_argument("--make", default=None)
    q.add_argument("--zone", default=None)
    q.add_argument("--from", dest="start", default=None, help="HH:MM")
    q.add_argument("--to", dest="end", default=None, help="HH:MM")
    q.add_argument("--search", default=None, help="hybrid free-text search")
    q.add_argument("--alerts", action="store_true")
    q.add_argument("--min-severity", default="INFO")
    q.add_argument("--limit", type=int, default=50)
    q.set_defaults(fn=cmd_query)

    for name, fn, hlp in (("ask", cmd_ask, "ask the agent one question"), ("chat", cmd_chat, "interactive Q&A")):
        a = sub.add_parser(name, help=hlp)
        if name == "ask":
            a.add_argument("question", nargs="+")
        a.add_argument("--input", default=None)
        a.add_argument("--offline", action="store_true", help="force the rule-based agent")
        a.set_defaults(fn=fn)

    sm = sub.add_parser("summary", help="one-sentence video summary + hourly episodes")
    sm.add_argument("--input", default=None)
    sm.set_defaults(fn=cmd_summary)

    b = sub.add_parser("benchmark", help="scalability benchmark on a synthetic stream")
    b.add_argument("--frames", type=int, default=50_000)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--db", default=None)
    b.set_defaults(fn=cmd_benchmark)

    v = sub.add_parser("video", help="analyse a real video file with the Claude VLM")
    v.add_argument("--file", required=True)
    v.add_argument("--fps", type=float, default=1.0)
    v.add_argument("--frames-dir", default="out/frames")
    v.add_argument("--telemetry", default=None, help="CSV: t_offset_s,lat,lon,altitude_m,battery_pct,zone")
    v.add_argument("--start", default=None, help="ISO start time of the video")
    v.add_argument("--zone", default=None, help="zone to assume when no telemetry is given")
    v.add_argument("--db", default=None)
    v.set_defaults(fn=cmd_video)

    d = sub.add_parser("demo", help="end-to-end showcase")
    d.add_argument("--input", default=None)
    d.add_argument("--db", default=None)
    d.add_argument("--speed", type=float, default=0)
    d.add_argument("--offline", action="store_true")
    d.add_argument("--colour", action="store_true")
    d.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    main()
