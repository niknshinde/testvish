"""Perception: turn a frame (text description or real image) into structured detections.

Two interchangeable analyzers implement :class:`FrameAnalyzer`:

* :class:`TextFrameAnalyzer` - deterministic, offline, ~20 us/frame. Parses the
  simulated frame descriptions ("Blue Ford F150 entering through main gate")
  with a lexicon + span-merging approach. This is the default and is what the
  test-suite exercises.
* :class:`ClaudeVisionAnalyzer` - a VLM path. Sends the real image (or the text
  description) to Claude with a JSON-schema constrained output and maps the
  result onto the same :class:`Detection` dataclass. Falls back to the text
  analyzer on any API error so the pipeline never stalls.

Because both produce identical ``Detection`` objects, everything downstream
(tracking, rules, indexing, the agent) is agnostic to where detections came from.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .config import SiteConfig
from .models import Detection, Frame

log = logging.getLogger(__name__)


class FrameAnalyzer(Protocol):
    def analyze(self, frame: Frame) -> list[Detection]: ...


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

COLORS = {
    "blue", "red", "white", "black", "silver", "grey", "gray", "green", "yellow", "orange",
    "brown", "beige", "maroon", "gold", "purple", "dark", "navy", "tan",
}
_COLOR_NORMALISE = {"gray": "grey", "navy": "blue"}

# model -> (make, category)
VEHICLE_MODELS: dict[str, tuple[str, str]] = {
    "f150": ("Ford", "truck"), "f-150": ("Ford", "truck"), "mustang": ("Ford", "sedan"),
    "transit": ("Ford", "van"), "explorer": ("Ford", "suv"),
    "camry": ("Toyota", "sedan"), "corolla": ("Toyota", "sedan"), "prius": ("Toyota", "sedan"),
    "tacoma": ("Toyota", "truck"), "rav4": ("Toyota", "suv"), "hilux": ("Toyota", "truck"),
    "civic": ("Honda", "sedan"), "accord": ("Honda", "sedan"), "cr-v": ("Honda", "suv"),
    "model 3": ("Tesla", "sedan"), "model y": ("Tesla", "suv"), "model s": ("Tesla", "sedan"),
    "silverado": ("Chevrolet", "truck"), "malibu": ("Chevrolet", "sedan"),
    "x5": ("BMW", "suv"), "sprinter": ("Mercedes", "van"), "altima": ("Nissan", "sedan"),
    "wrangler": ("Jeep", "suv"), "tucson": ("Hyundai", "suv"),
}
_MODEL_CANONICAL = {"f150": "F150", "f-150": "F150", "rav4": "RAV4", "cr-v": "CR-V", "x5": "X5"}
VEHICLE_MAKES = {
    "ford": "Ford", "toyota": "Toyota", "honda": "Honda", "tesla": "Tesla", "chevrolet": "Chevrolet",
    "chevy": "Chevrolet", "bmw": "BMW", "mercedes": "Mercedes", "nissan": "Nissan", "jeep": "Jeep",
    "hyundai": "Hyundai", "kia": "Kia", "audi": "Audi", "volkswagen": "Volkswagen", "vw": "Volkswagen",
}
VEHICLE_NOUNS = {
    "pickup truck": "truck", "pickup": "truck", "truck": "truck", "lorry": "truck",
    "delivery van": "van", "minivan": "van", "van": "van",
    "sedan": "sedan", "hatchback": "sedan", "coupe": "sedan", "car": "car",
    "suv": "suv", "jeep": "suv",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "scooter": "motorcycle",
    "bicycle": "bicycle", "bike": "bicycle", "bus": "bus", "vehicle": None, "vehicles": None,
}
PERSON_NOUNS = {
    "person", "people", "man", "men", "woman", "women", "individual", "intruder", "courier",
    "delivery person", "delivery driver", "driver", "mail carrier", "child", "kid", "teenager",
    "stranger", "gardener", "worker", "jogger", "guard",
}
ANIMAL_NOUNS = {"dog", "cat", "deer", "fox", "raccoon", "coyote", "bird", "birds", "squirrel"}
PACKAGE_NOUNS = {"package", "parcel", "box", "delivery box"}
SUSPICIOUS_OBJECTS = {
    "crowbar", "weapon", "gun", "knife", "bolt cutters", "ladder", "spray paint", "gas can",
    "balaclava", "ski mask", "mask",
}
PERSON_ATTR_NOUNS = {
    "hoodie", "jacket", "vest", "backpack", "flashlight", "torch", "hat", "cap", "uniform",
    "bag", "gloves", "coat", "shirt", "helmet", "umbrella", "phone",
}
PERSON_ATTR_ADJS = COLORS | {"hi-vis", "reflective", "hooded", "masked", "large", "small", "brown"}

# (canonical action, priority, patterns). Highest priority wins in an object's window.
ACTIONS: list[tuple[str, int, tuple[str, ...]]] = [
    ("climbing", 100, ("climbing", "climbs", "scaling", "jumping over", "jumps over", "hopping the")),
    ("tampering", 95, ("tampering", "prying", "forcing", "breaking", "cutting", "trying the door",
                       "trying door", "trying the handle", "trying the", "tries the", "jiggling", "smashing")),
    ("loitering", 80, ("loitering", "lingering", "pacing", "looking around", "still at", "still near",
                       "still standing", "hanging around", "peering")),
    ("entering", 70, ("entering", "enters", "entered", "arriving", "arrives", "pulls in", "pulling in",
                      "drives in", "driving in", "coming in", "approaching")),
    ("exiting", 70, ("exiting", "exits", "leaving", "leaves", "departing", "departs", "drives away",
                     "driving away", "pulling out", "pulls out", "drives off")),
    ("delivering", 60, ("delivering", "drops off", "dropping off", "dropped off", "left on", "left at",
                        "placing", "places")),
    ("parked", 50, ("parked", "stationary", "stopped", "idling", "engine running", "sitting")),
    ("running", 45, ("running", "fleeing", "sprinting")),
    ("walking", 40, ("walking", "walks", "crossing", "crosses", "strolling", "patrolling", "moving")),
    ("standing", 30, ("standing", "waiting", "stands", "waits")),
    ("carrying", 20, ("carrying", "holding")),
]

_NEGATION_RE = re.compile(r"\b(no|without|zero|nobody|nothing)\s+(\w+\s+){0,1}$")
_PLATE_RE = re.compile(r"\b(?:plate\s*[:#]?\s*)?([A-Z]{2,4}-\d{3,4})\b")
_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "several": 3, "group": 3}


def _alt(words: Iterable[str]) -> str:
    # longest first so "pickup truck" beats "pickup"
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_VEH_TOKEN_RE = re.compile(
    rf"\b({_alt(VEHICLE_MODELS)}|{_alt(VEHICLE_MAKES)}|{_alt(VEHICLE_NOUNS)})\b", re.I
)
_PERSON_RE = re.compile(rf"\b({_alt(PERSON_NOUNS)})\b", re.I)
_ANIMAL_RE = re.compile(rf"\b({_alt(ANIMAL_NOUNS)})\b", re.I)
_PACKAGE_RE = re.compile(rf"\b({_alt(PACKAGE_NOUNS)})\b", re.I)
_SUSP_RE = re.compile(rf"\b({_alt(SUSPICIOUS_OBJECTS)})\b", re.I)
_ATTR_RE = re.compile(rf"\b(?:({_alt(PERSON_ATTR_ADJS)})\s+)?({_alt(PERSON_ATTR_NOUNS)})(?:e?s)?\b", re.I)
_COLOR_RE = re.compile(rf"\b({_alt(COLORS)})\b", re.I)


@dataclass
class _Mention:
    obj_type: str
    start: int
    end: int
    tokens: list[str]


def _split_clauses(text: str) -> list[str]:
    parts = re.split(r";|(?<!\b[A-Z])\.\s+|\.$|\bwhile\b|\bmeanwhile\b", text)
    return [p.strip(" ,") for p in parts if p and p.strip(" ,")]


def _is_negated(clause: str, start: int) -> bool:
    return bool(_NEGATION_RE.search(clause[:start].lower()))


def _vehicle_mentions(clause: str) -> list[_Mention]:
    """Find vehicle token spans and merge adjacent ones ('Blue Ford F150 pickup truck')."""
    spans = [(m.start(), m.end(), m.group(1).lower()) for m in _VEH_TOKEN_RE.finditer(clause)]
    merged: list[_Mention] = []
    for s, e, tok in spans:
        if _is_negated(clause, s) and not merged:
            continue
        if merged:
            gap = clause[merged[-1].end:s]
            # merge when separated by at most 2 short words (colour, adjectives, '-')
            if len(gap.split()) <= 2 and not re.search(r"[,;]|\band\b|\bwith\b", gap):
                merged[-1].end = e
                merged[-1].tokens.append(tok)
                continue
        merged.append(_Mention("vehicle", s, e, [tok]))
    return merged


def _actions_in(window: str) -> str | None:
    w = window.lower()
    best: tuple[int, int, str] | None = None   # (priority, -position, action)
    for action, prio, pats in ACTIONS:
        for p in pats:
            idx = w.find(p)
            if idx >= 0:
                cand = (prio, -idx, action)
                if best is None or cand > best:
                    best = cand
    return best[2] if best else None


class TextFrameAnalyzer:
    """Lexicon-based detector for text frame descriptions (offline, deterministic)."""

    def __init__(self, config: SiteConfig):
        self.config = config

    def analyze(self, frame: Frame) -> list[Detection]:
        text = frame.caption or frame.description
        detections: list[Detection] = []
        zone = frame.zone
        for clause in _split_clauses(text):
            # a clause without a landmark inherits the zone of the previous clause
            zone = self.config.resolve_zone(clause) or zone
            detections.extend(self._analyze_clause(clause, frame, zone))
        return detections

    # -- internals ---------------------------------------------------------
    def _analyze_clause(self, clause: str, frame: Frame, clause_zone: str | None) -> list[Detection]:
        mentions: list[_Mention] = list(_vehicle_mentions(clause))
        for regex, obj_type in ((_PERSON_RE, "person"), (_ANIMAL_RE, "animal"), (_PACKAGE_RE, "package")):
            for m in regex.finditer(clause):
                if _is_negated(clause, m.start()):
                    continue
                # "delivery van" is a vehicle, "box truck" is a vehicle: skip overlaps
                if any(v.start <= m.start() < v.end for v in mentions if v.obj_type == "vehicle"):
                    continue
                mentions.append(_Mention(obj_type, m.start(), m.end(), [m.group(1).lower()]))
        mentions.sort(key=lambda m: m.start)

        plates = _PLATE_RE.findall(clause)
        suspicious = [m.group(1).lower() for m in _SUSP_RE.finditer(clause)]
        out: list[Detection] = []

        for i, m in enumerate(mentions):
            nxt = mentions[i + 1].start if i + 1 < len(mentions) else len(clause)
            prev_end = mentions[i - 1].end if i > 0 else 0
            window = clause[m.start:nxt]
            pre = clause[prev_end:m.start]
            zone = self.config.resolve_zone(window) or clause_zone
            action = _actions_in(window)
            det = Detection(frame_id=frame.frame_id, ts=frame.ts, obj_type=m.obj_type,
                            zone=zone, action=action, source_text=clause)

            if m.obj_type == "vehicle":
                self._fill_vehicle(det, m, pre, window, plates)
            elif m.obj_type == "person":
                self._fill_person(det, m, pre, window, suspicious)
            elif m.obj_type == "animal":
                det.category = m.tokens[0].rstrip("s") if m.tokens[0] != "birds" else "bird"
                det.confidence = 0.85
            elif m.obj_type == "package":
                det.category = "package"
                if det.action in (None, "carrying"):
                    det.action = "delivering" if "left" in window.lower() or "porch" in window.lower() else det.action
            out.append(det)

        # suspicious tool with nobody holding it -> standalone object detection
        if suspicious and not any(d.obj_type == "person" for d in out):
            for s in suspicious:
                out.append(Detection(frame_id=frame.frame_id, ts=frame.ts, obj_type="object",
                                     category=s, zone=clause_zone, attributes=["suspicious"],
                                     confidence=0.8, source_text=clause))
        return out

    @staticmethod
    def _fill_vehicle(det: Detection, m: _Mention, pre: str, window: str, plates: list[str]) -> None:
        make = model = category = None
        for tok in m.tokens:
            if tok in VEHICLE_MODELS:
                model = _MODEL_CANONICAL.get(tok, tok.title())
                make, cat = VEHICLE_MODELS[tok]
                category = category or cat
            elif tok in VEHICLE_MAKES and tok not in VEHICLE_NOUNS:
                make = VEHICLE_MAKES[tok]
            elif tok in VEHICLE_NOUNS and VEHICLE_NOUNS[tok]:
                category = VEHICLE_NOUNS[tok]
        # colour: inside the span or in the 3 words before it
        span_text = window[: m.end - m.start]
        colour_src = " ".join(pre.split()[-3:]) + " " + span_text
        colours = _COLOR_RE.findall(colour_src)
        det.color = _COLOR_NORMALISE.get(colours[-1].lower(), colours[-1].lower()) if colours else None
        det.make, det.model = make, model
        det.category = category or ("car" if (make or model) else "vehicle")
        det.plate = plates[0] if plates else None
        det.confidence = 0.95 if (make and model) else 0.85
        if "delivery" in window.lower():
            det.attributes.append("delivery")

    @staticmethod
    def _fill_person(det: Detection, m: _Mention, pre: str, window: str, suspicious: list[str]) -> None:
        noun = m.tokens[0]
        det.category = {"courier": "courier", "delivery person": "courier", "delivery driver": "courier",
                        "mail carrier": "courier", "child": "child", "kid": "child",
                        "intruder": "intruder"}.get(noun, "person")
        attrs: list[str] = []
        for am in _ATTR_RE.finditer(window):
            adj, noun_attr = am.group(1), am.group(2)
            attrs.append(f"{adj.lower()} {noun_attr.lower()}" if adj else noun_attr.lower())
        for word in ("masked", "hooded", "uniform", "hi-vis"):
            if re.search(rf"\b{word}\b", (pre + " " + window).lower()) and not any(word in a for a in attrs):
                attrs.append(word)
        count = None
        for w, n in _COUNT_WORDS.items():
            if re.search(rf"\b{w}\b", pre.lower() + " " + noun):
                count = n
        if noun in {"people", "men", "women"}:
            count = count or 2
        if count:
            attrs.append(f"group of {count}")
        attrs.extend(s for s in suspicious if s not in attrs)
        if suspicious:
            attrs.append("suspicious")
        det.attributes = list(dict.fromkeys(attrs))
        det.confidence = 0.9


# ---------------------------------------------------------------------------
# VLM path (Claude vision with JSON-schema constrained output)
# ---------------------------------------------------------------------------

DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "obj_type": {"type": "string", "enum": ["person", "vehicle", "animal", "package", "object"]},
                    "category": {"type": ["string", "null"]},
                    "color": {"type": ["string", "null"]},
                    "make": {"type": ["string", "null"]},
                    "model": {"type": ["string", "null"]},
                    "plate": {"type": ["string", "null"]},
                    "action": {"type": ["string", "null"]},
                    "zone": {"type": ["string", "null"]},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": ["obj_type", "category", "color", "make", "model", "plate", "action",
                             "zone", "attributes", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["caption", "detections"],
    "additionalProperties": False,
}

VLM_SYSTEM_PROMPT = """You are the perception module of a security drone that patrols a single private property.
For each frame, write a one-sentence factual caption, then list every person, vehicle, animal, package
or security-relevant object you can see.

Rules:
- Only report what is visible. Do not guess identities. Use null for unknown fields.
- vehicle: category is one of truck, van, sedan, suv, car, motorcycle, bicycle, bus; give colour, make,
  model and licence plate only when legible.
- person: put clothing, carried items and anything security-relevant (mask, crowbar, ladder, flashlight)
  in attributes.
- action is one of: entering, exiting, parked, loitering, standing, walking, running, climbing,
  tampering, delivering, carrying.
- zone must be one of: {zones}. Use the camera's zone hint when the frame doesn't show a landmark.
- confidence is your 0-1 confidence that the detection is correct."""


class ClaudeVisionAnalyzer:
    """VLM analyzer backed by Claude (vision + structured outputs).

    Works for real frames (``frame.image_path``) and for text-only frames, where
    it acts as an LLM parser of the description. Any API failure falls back to
    :class:`TextFrameAnalyzer` for that frame.
    """

    def __init__(self, config: SiteConfig, model: str = "claude-opus-5-5", effort: str = "low",
                 client=None, fallback: FrameAnalyzer | None = None):
        self.config = config
        self.model = model
        self.effort = effort
        self.fallback = fallback or TextFrameAnalyzer(config)
        if client is None:
            import anthropic  # optional dependency

            client = anthropic.Anthropic()
        self.client = client
        zones = ", ".join(z.name for z in config.zones)
        self.system_prompt = VLM_SYSTEM_PROMPT.format(zones=zones)

    def _content(self, frame: Frame) -> list[dict]:
        hint = (f"Frame {frame.frame_id} at {frame.ts:%H:%M:%S}. Camera zone hint: {frame.zone or 'unknown'}."
                + (f" Altitude {frame.telemetry.altitude_m:.0f} m." if frame.telemetry else ""))
        content: list[dict] = []
        if frame.image_path and Path(frame.image_path).exists():
            media_type = mimetypes.guess_type(frame.image_path)[0] or "image/jpeg"
            data = base64.standard_b64encode(Path(frame.image_path).read_bytes()).decode("utf-8")
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
            content.append({"type": "text", "text": hint + " Analyse this frame."})
        else:
            content.append({"type": "text", "text": hint + f" Simulated frame description: {frame.description}"})
        return content

    def analyze(self, frame: Frame) -> list[Detection]:
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            anthropic = None
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                messages=[{"role": "user", "content": self._content(frame)}],
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": DETECTION_SCHEMA}},
                # Server-side refusal fallback: if a safety classifier declines, the API retries on a
                # fallback model inside the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except Exception as exc:  # network, auth, rate-limit ... never stall the pipeline
            if anthropic is not None and isinstance(exc, anthropic.RateLimitError):
                log.warning("VLM rate-limited on frame %s; using text fallback", frame.frame_id)
            else:
                log.warning("VLM call failed on frame %s (%s); using text fallback", frame.frame_id, exc)
            return self.fallback.analyze(frame)

        if response.stop_reason == "refusal":
            log.warning("VLM declined frame %s; using text fallback", frame.frame_id)
            return self.fallback.analyze(frame)
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            log.warning("VLM returned non-JSON for frame %s; using text fallback", frame.frame_id)
            return self.fallback.analyze(frame)
        return self.parse_payload(frame, payload)

    def parse_payload(self, frame: Frame, payload: dict) -> list[Detection]:
        frame.caption = payload.get("caption") or frame.caption
        dets: list[Detection] = []
        valid_zones = {z.name for z in self.config.zones}
        for d in payload.get("detections", []):
            zone = d.get("zone")
            if zone not in valid_zones:
                zone = self.config.resolve_zone(zone or "") or frame.zone
            make, model = d.get("make"), d.get("model")
            if model and model.lower() in VEHICLE_MODELS:
                make = make or VEHICLE_MODELS[model.lower()][0]
                model = _MODEL_CANONICAL.get(model.lower(), model)
            dets.append(Detection(
                frame_id=frame.frame_id, ts=frame.ts, obj_type=d["obj_type"],
                category=d.get("category"), color=(d.get("color") or None) and d["color"].lower(),
                make=make, model=model, plate=d.get("plate"), action=d.get("action"), zone=zone,
                attributes=[a.lower() for a in d.get("attributes", [])],
                confidence=float(d.get("confidence", 0.8)), source_text=frame.caption or "",
            ))
        return dets
