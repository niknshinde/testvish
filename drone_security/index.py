"""Frame-by-frame index (the cross-domain piece).

Every processed frame is written to SQLite with its timestamp, telemetry,
description/caption and the structured detections found in it. Three
complementary query paths are offered:

1. **Structured queries** (B-tree indexes) - by time range, zone, object type,
   colour, make, action or tracked entity: "show all truck events",
   "everything at the main gate between 00:00 and 01:00".
2. **Full-text search** (SQLite FTS5, BM25 ranking, Porter stemming) over the
   description plus the detection labels: "hoodie gate".
3. **Vector search** over a lightweight local embedding (feature hashing +
   a small security-domain synonym table) so "pickup" finds "F150 truck" and
   "intruder" finds "person climbing fence". A sentence-transformer can be
   plugged in through the :class:`Embedder` protocol without schema changes.

``hybrid_search`` fuses (2) and (3) with Reciprocal Rank Fusion.

SQLite was chosen deliberately: zero-ops, ships with Python, FTS5 built in,
single-file artefact the drone dock can sync. The schema maps 1:1 onto Postgres
(+ pgvector) or OpenSearch for a fleet deployment.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from .models import Alert, Detection, Frame, LogEntry, Telemetry

_EPOCH = datetime(1970, 1, 1)


def epoch(ts: datetime) -> float:
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return (ts - _EPOCH).total_seconds()


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS frames (
    frame_id INTEGER PRIMARY KEY, ts TEXT NOT NULL, ts_epoch REAL NOT NULL, day TEXT, hhmm TEXT,
    zone TEXT, description TEXT, caption TEXT, image_path TEXT, labels TEXT,
    lat REAL, lon REAL, altitude_m REAL, battery_pct REAL
);
CREATE INDEX IF NOT EXISTS ix_frames_ts ON frames(ts_epoch);
CREATE INDEX IF NOT EXISTS ix_frames_zone ON frames(zone, ts_epoch);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, frame_id INTEGER NOT NULL REFERENCES frames(frame_id),
    ts TEXT, ts_epoch REAL, obj_type TEXT, category TEXT, color TEXT, make TEXT, model TEXT, plate TEXT,
    action TEXT, zone TEXT, entity_id TEXT, label TEXT, confidence REAL, attributes TEXT
);
CREATE INDEX IF NOT EXISTS ix_det_type ON detections(obj_type, category, ts_epoch);
CREATE INDEX IF NOT EXISTS ix_det_entity ON detections(entity_id, ts_epoch);
CREATE INDEX IF NOT EXISTS ix_det_ts ON detections(ts_epoch);
CREATE INDEX IF NOT EXISTS ix_det_frame ON detections(frame_id);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, ts_epoch REAL, frame_id INTEGER,
    message TEXT, obj_type TEXT, entity_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_logs_ts ON logs(ts_epoch);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY, ts TEXT, ts_epoch REAL, rule_id TEXT, severity INTEGER, severity_name TEXT,
    message TEXT, details TEXT, recommendation TEXT, frame_id INTEGER, zone TEXT, entity_id TEXT,
    evidence TEXT
);
CREATE INDEX IF NOT EXISTS ix_alerts_ts ON alerts(ts_epoch);

CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, ts_epoch REAL, drone_id TEXT, lat REAL, lon REAL,
    altitude_m REAL, battery_pct REAL, zone TEXT, wind_mps REAL, mode TEXT
);
CREATE INDEX IF NOT EXISTS ix_tel_ts ON telemetry(ts_epoch);

CREATE TABLE IF NOT EXISTS embeddings (frame_id INTEGER PRIMARY KEY, vec BLOB NOT NULL);

CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts USING fts5(
    description, labels, zone, frame_id UNINDEXED, tokenize='porter unicode61'
);
"""


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


SYNONYMS: dict[str, list[str]] = {
    "truck": ["pickup", "vehicle", "f150"], "pickup": ["truck", "vehicle"], "f150": ["truck", "ford", "pickup"],
    "car": ["vehicle", "sedan"], "sedan": ["car", "vehicle"], "van": ["vehicle", "delivery"],
    "suv": ["vehicle", "car"], "vehicle": ["car", "truck", "van"], "motorcycle": ["vehicle", "bike"],
    "person": ["people", "human", "someone"], "man": ["person"], "woman": ["person"], "people": ["person"],
    "intruder": ["person", "trespasser", "climbing", "breach"], "trespasser": ["intruder", "person"],
    "courier": ["delivery", "person"], "delivery": ["courier", "package", "van"],
    "package": ["parcel", "box", "delivery"], "parcel": ["package"],
    "loitering": ["lingering", "standing", "waiting", "suspicious"], "lingering": ["loitering"],
    "climbing": ["breach", "fence", "intruder"], "breach": ["climbing", "intrusion"],
    "deer": ["animal", "wildlife"], "fox": ["animal", "wildlife"], "dog": ["animal", "pet"],
    "cat": ["animal", "pet"], "animal": ["wildlife"], "crowbar": ["tool", "weapon", "suspicious"],
    "night": ["midnight", "dark"], "midnight": ["night"], "gate": ["entrance"], "entrance": ["gate"],
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


def _tokens(text: str) -> list[str]:
    toks = []
    for t in _TOKEN_RE.findall(text.lower()):
        t = t.replace("-", "")
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        toks.append(t)
    return toks


class HashingEmbedder:
    """Dependency-free text embedding: hashed unigrams + bigrams + synonym expansion, L2-normalised.

    Not a neural model, but captures lexical similarity and the domain synonyms
    that matter for security queries, and costs microseconds per frame.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _tokens(text)
            feats: list[tuple[str, float]] = [(t, 1.0) for t in toks]
            feats += [(f"{a}_{b}", 0.5) for a, b in zip(toks, toks[1:])]
            for t in toks:
                feats += [(s, 0.5) for s in SYNONYMS.get(t, [])]
            for f, w in feats:
                j, sign = self._bucket(f)
                out[i, j] += sign * w
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class FrameIndex:
    def __init__(self, path: str | Path = ":memory:", embedder: Embedder | None = None, batch_size: int = 500):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.embedder = embedder or HashingEmbedder()
        self.batch_size = batch_size
        self._pending: dict[str, list[tuple]] = {k: [] for k in
                                                 ("frames", "fts", "det", "logs", "alerts", "tel", "emb")}
        self._pending_text: list[tuple[int, str]] = []
        self._vec_cache: tuple[np.ndarray, np.ndarray] | None = None

    # ---- writes ---------------------------------------------------------
    def add_frame(self, frame: Frame, detections: Iterable[Detection] = ()) -> None:
        dets = list(detections)
        labels = " ".join(
            " ".join(filter(None, [d.obj_type, d.category, d.label, d.color, d.make, d.model, d.plate, d.action,
                                   " ".join(d.attributes)])) for d in dets)
        t = frame.telemetry
        self._pending["frames"].append((
            frame.frame_id, frame.ts.isoformat(), epoch(frame.ts), frame.ts.date().isoformat(),
            frame.ts.strftime("%H:%M"), frame.zone, frame.description, frame.caption, frame.image_path, labels,
            t.lat if t else None, t.lon if t else None, t.altitude_m if t else None, t.battery_pct if t else None))
        self._pending["fts"].append((frame.caption or frame.description, labels, frame.zone or "", frame.frame_id))
        for d in dets:
            self._pending["det"].append((
                d.frame_id, d.ts.isoformat(), epoch(d.ts), d.obj_type, d.category, d.color, d.make, d.model,
                d.plate, d.action, d.zone, d.entity_id, d.label, d.confidence, json.dumps(d.attributes)))
        self._pending_text.append((frame.frame_id, f"{frame.caption or frame.description} {labels} {frame.zone or ''}"))
        if len(self._pending["frames"]) >= self.batch_size:
            self.flush()

    def add_logs(self, logs: Iterable[LogEntry]) -> None:
        for l in logs:
            self._pending["logs"].append((l.ts.isoformat(), epoch(l.ts), l.frame_id, l.message, l.obj_type, l.entity_id))

    def add_alerts(self, alerts: Iterable[Alert]) -> None:
        for a in alerts:
            self._pending["alerts"].append((
                a.alert_id, a.ts.isoformat(), epoch(a.ts), a.rule_id, int(a.severity), a.severity.name, a.message,
                a.details, a.recommendation, a.frame_id, a.zone, a.entity_id, json.dumps(a.evidence_frames)))

    def add_telemetry(self, t: Telemetry) -> None:
        self._pending["tel"].append((t.ts.isoformat(), epoch(t.ts), t.drone_id, t.lat, t.lon, t.altitude_m,
                                     t.battery_pct, t.zone, t.wind_mps, t.mode))

    def flush(self) -> None:
        p = self._pending
        if self._pending_text:
            ids, texts = zip(*self._pending_text)
            vecs = self.embedder.embed(list(texts))
            p["emb"].extend((fid, v.astype(np.float32).tobytes()) for fid, v in zip(ids, vecs))
            self._pending_text.clear()
            self._vec_cache = None
        with self.conn:
            c = self.conn
            c.executemany("INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", p["frames"])
            c.executemany("INSERT INTO frames_fts(description, labels, zone, frame_id) VALUES (?,?,?,?)", p["fts"])
            c.executemany("INSERT INTO detections(frame_id, ts, ts_epoch, obj_type, category, color, make, model, "
                          "plate, action, zone, entity_id, label, confidence, attributes) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", p["det"])
            c.executemany("INSERT INTO logs(ts, ts_epoch, frame_id, message, obj_type, entity_id) VALUES (?,?,?,?,?,?)",
                          p["logs"])
            c.executemany("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", p["alerts"])
            c.executemany("INSERT INTO telemetry(ts, ts_epoch, drone_id, lat, lon, altitude_m, battery_pct, zone, "
                          "wind_mps, mode) VALUES (?,?,?,?,?,?,?,?,?,?)", p["tel"])
            c.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?)", p["emb"])
        for v in p.values():
            v.clear()

    def close(self) -> None:
        self.flush()
        self.conn.close()

    # ---- time helpers ---------------------------------------------------
    def default_day(self) -> datetime | None:
        row = self.conn.execute("SELECT MIN(ts) FROM frames").fetchone()
        return datetime.fromisoformat(row[0]) if row and row[0] else None

    def _resolve_time(self, value: str | datetime | None, *, end: bool = False) -> float | None:
        """Accept datetimes, ISO strings or 'HH:MM' (resolved against the first indexed day)."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return epoch(value)
        s = str(value).strip()
        m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
        if m:
            day = self.default_day() or datetime(1970, 1, 1)
            dt = datetime.combine(day.date(), time(int(m[1]), int(m[2]), int(m[3] or 0)))
            if end and not m[3]:
                dt += timedelta(seconds=59.999)
            return epoch(dt)
        return epoch(datetime.fromisoformat(s))

    # ---- structured queries ---------------------------------------------
    def query_detections(self, *, start=None, end=None, obj_type: str | None = None, category: str | None = None,
                         color: str | None = None, make: str | None = None, model: str | None = None,
                         action: str | None = None, zone: str | None = None, entity_id: str | None = None,
                         limit: int = 200) -> list[dict[str, Any]]:
        where, args = ["1=1"], []
        for col, val in (("d.obj_type", obj_type), ("d.category", category), ("d.color", color),
                         ("d.action", action), ("d.zone", zone), ("d.entity_id", entity_id)):
            if val:
                where.append(f"LOWER({col}) = LOWER(?)")
                args.append(val)
        for col, val in (("d.make", make), ("d.model", model)):
            if val:
                where.append(f"REPLACE(LOWER({col}),'-','') = REPLACE(LOWER(?),'-','')")
                args.append(val)
        s, e = self._resolve_time(start), self._resolve_time(end, end=True)
        if s is not None:
            where.append("d.ts_epoch >= ?"); args.append(s)
        if e is not None:
            where.append("d.ts_epoch <= ?"); args.append(e)
        sql = (f"SELECT d.*, f.description, f.caption FROM detections d JOIN frames f USING(frame_id) "
               f"WHERE {' AND '.join(where)} ORDER BY d.ts_epoch LIMIT ?")
        return [dict(r) for r in self.conn.execute(sql, (*args, limit))]

    def frames_between(self, start, end, zone: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        s, e = self._resolve_time(start), self._resolve_time(end, end=True)
        sql = "SELECT * FROM frames WHERE ts_epoch BETWEEN ? AND ?" + (" AND zone = ?" if zone else "")
        sql += " ORDER BY ts_epoch LIMIT ?"
        args = (s, e, zone, limit) if zone else (s, e, limit)
        return [dict(r) for r in self.conn.execute(sql, args)]

    def frame(self, frame_id: int) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
        if not r:
            return None
        out = dict(r)
        out["detections"] = [dict(d) for d in self.conn.execute(
            "SELECT obj_type, label, action, zone, entity_id FROM detections WHERE frame_id = ?", (frame_id,))]
        return out

    def alerts(self, *, min_severity: int = 0, start=None, end=None, rule_id: str | None = None,
               limit: int = 200) -> list[dict[str, Any]]:
        where, args = ["severity >= ?"], [min_severity]
        s, e = self._resolve_time(start), self._resolve_time(end, end=True)
        if s is not None:
            where.append("ts_epoch >= ?"); args.append(s)
        if e is not None:
            where.append("ts_epoch <= ?"); args.append(e)
        if rule_id:
            where.append("rule_id = ?"); args.append(rule_id)
        sql = f"SELECT * FROM alerts WHERE {' AND '.join(where)} ORDER BY ts_epoch LIMIT ?"
        return [dict(r) for r in self.conn.execute(sql, (*args, limit))]

    def logs(self, *, start=None, end=None, contains: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        where, args = ["1=1"], []
        s, e = self._resolve_time(start), self._resolve_time(end, end=True)
        if s is not None:
            where.append("ts_epoch >= ?"); args.append(s)
        if e is not None:
            where.append("ts_epoch <= ?"); args.append(e)
        if contains:
            where.append("LOWER(message) LIKE ?"); args.append(f"%{contains.lower()}%")
        sql = f"SELECT * FROM logs WHERE {' AND '.join(where)} ORDER BY ts_epoch LIMIT ?"
        return [dict(r) for r in self.conn.execute(sql, (*args, limit))]

    def object_counts(self) -> list[dict[str, Any]]:
        sql = ("SELECT obj_type, COALESCE(category, obj_type) AS category, COUNT(*) AS detections, "
               "COUNT(DISTINCT entity_id) AS entities, COUNT(DISTINCT frame_id) AS frames "
               "FROM detections GROUP BY obj_type, COALESCE(category, obj_type) ORDER BY detections DESC")
        return [dict(r) for r in self.conn.execute(sql)]

    def entities(self) -> list[dict[str, Any]]:
        sql = ("SELECT entity_id, obj_type, label, MIN(ts) AS first_seen, MAX(ts) AS last_seen, "
               "COUNT(*) AS detections, GROUP_CONCAT(DISTINCT zone) AS zones FROM detections "
               "WHERE entity_id IS NOT NULL GROUP BY entity_id ORDER BY first_seen")
        return [dict(r) for r in self.conn.execute(sql)]

    def stats(self) -> dict[str, Any]:
        q = lambda sql: self.conn.execute(sql).fetchone()[0]  # noqa: E731
        return {"frames": q("SELECT COUNT(*) FROM frames"), "detections": q("SELECT COUNT(*) FROM detections"),
                "alerts": q("SELECT COUNT(*) FROM alerts"), "logs": q("SELECT COUNT(*) FROM logs"),
                "telemetry": q("SELECT COUNT(*) FROM telemetry"),
                "first": q("SELECT MIN(ts) FROM frames"), "last": q("SELECT MAX(ts) FROM frames")}

    # ---- text / vector search -------------------------------------------
    def search_text(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        toks = [t for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 1]
        if not toks:
            return []
        sql = ("SELECT f.*, bm25(frames_fts) AS score FROM frames_fts JOIN frames f ON f.frame_id = frames_fts.frame_id "
               "WHERE frames_fts MATCH ? ORDER BY score LIMIT ?")
        rows = list(self.conn.execute(sql, (" ".join(f'"{t}"' for t in toks), limit)))   # AND
        if not rows and len(toks) > 1:
            rows = list(self.conn.execute(sql, (" OR ".join(f'"{t}"' for t in toks), limit)))
        return [dict(r) for r in rows]

    def _vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self._vec_cache is None:
            rows = self.conn.execute("SELECT frame_id, vec FROM embeddings ORDER BY frame_id").fetchall()
            if not rows:
                self._vec_cache = (np.zeros(0, dtype=np.int64), np.zeros((0, self.embedder.dim), dtype=np.float32))
            else:
                ids = np.array([r[0] for r in rows], dtype=np.int64)
                mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
                self._vec_cache = (ids, mat)
        return self._vec_cache

    def semantic_search(self, query: str, limit: int = 20, min_score: float = 0.05) -> list[dict[str, Any]]:
        self.flush()
        ids, mat = self._vectors()
        if not len(ids):
            return []
        q = self.embedder.embed([query])[0]
        scores = mat @ q
        top = np.argsort(-scores)[:limit]
        out = []
        for i in top:
            if scores[i] < min_score:
                break
            row = self.frame(int(ids[i]))
            if row:
                row["score"] = float(scores[i])
                out.append(row)
        return out

    def hybrid_search(self, query: str, limit: int = 10, k: int = 60) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion of BM25 (keywords) and vector similarity (synonyms/paraphrase)."""
        self.flush()
        fused: dict[int, float] = {}
        rows: dict[int, dict] = {}
        for ranked in (self.search_text(query, limit * 3), self.semantic_search(query, limit * 3)):
            for rank, r in enumerate(ranked):
                fused[r["frame_id"]] = fused.get(r["frame_id"], 0.0) + 1.0 / (k + rank + 1)
                rows.setdefault(r["frame_id"], r)
        best = sorted(fused, key=lambda f: -fused[f])[:limit]
        return [{**rows[f], "rrf": round(fused[f], 5)} for f in best]
