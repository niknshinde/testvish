# Feature Spec: Drone Security Analyst Agent

## Problem

A docked drone patrols a fixed property every day and produces hours of video and a constant telemetry feed.
Nobody watches all of it. Owners today either get no alerts at all or get spammed by motion alerts
("motion detected" for every cat). Security-relevant patterns that build up across a day, such as the same unknown
truck coming back, or someone standing at the gate for minutes at midnight, are invisible to frame-by-frame motion
detection.

## Value to property owners

**The agent turns raw drone footage into a security analyst's notebook. It decides what matters, says why, keeps a
searchable record, and answers questions about it.**

* **Automated 24/7 monitoring.** Every frame is analysed. People, vehicles, animals and packages are identified and
  logged with time, place and what they were doing.
* **Alerts worth acting on.** Rules use context: dwell time, time of day, restricted zones, repeat visits and the
  owner's authorised vehicles. One loitering person produces one alert, not one per frame. The owner's car never
  alerts.
* **Pattern awareness.** The agent remembers across the day ("blue Ford F150 entered **twice** today") and links
  events together ("the person who tried the garage door arrived in that truck").
* **Instant answers.** "Show all truck events", "what happened around midnight?", "when did it leave?" are answered
  from an indexed, searchable record. No scrubbing through video.
* **Drone safety.** The same agent watches the drone itself (battery, geofence, altitude ceiling, wind, datalink
  gaps) so the patrol stays airworthy and compliant.

## Users & key scenarios

| User | Scenario | Outcome |
|---|---|---|
| Home owner | Away at night; someone stops at the gate | Push alert within one frame of the dwell threshold, with clip frame IDs and a recommended action |
| Home owner | "Was that truck here before?" | Entity history: visits, times, zones, frames |
| Facility manager | Morning review | One-sentence summary, hourly episodes, prioritised recommendations |
| Security operator | Incident investigation | Time-window and free-text search across all indexed frames |

## Key requirements

1. **R1: Real-time detection, logging and alerting.**
   Every frame is processed as it arrives (target: ≥ 1 fps per drone on edge hardware; the prototype reaches about
   5,000 fps without the VLM). Each new entity, visit, zone change or significant action produces a context log line.
   Configured rules raise severity-ranked alerts with a recommendation. Duplicates are suppressed (cooldown) but
   escalations always get through.
   *Acceptance:* a person who stays ≥ 60 s at the main gate at night raises `Person loitering at main gate, 00:01.`
   exactly once. A blue Ford F150 is logged as `Blue Ford F150 spotted at garage, 12:00.`. Authorised vehicles never
   alert.

2. **R2: Frame-by-frame index that you can query.**
   Every frame is stored with its timestamp, telemetry, description or caption and structured detections. The index
   answers queries by time range, object type or attribute (colour, make, model, plate, action, zone), by tracked
   entity, and by free text (keyword plus semantic) in well under a second at 100k frames.
   *Acceptance:* "show all truck events" returns all 6 F150 detections. Frames 00:00–00:01 return frames 1–5.
   "pickup" finds the F150 frames even though the word never appears in their text.

3. **R3: Context-aware analyst agent.**
   The agent answers natural-language questions and follow-ups, grounded in the index (it cites times and frame
   numbers). It produces a one-sentence video summary and prioritised recommendations, and it works with no network
   (offline mode). The LLM context stays bounded regardless of video length.
   *Acceptance:* "How many times did the blue Ford F150 visit?" → 2 (08:31, 11:58); follow-up "When did it
   leave?" → 08:41, 12:06. The context block for 100k frames is under 20k characters.

## Non-goals (prototype)

* Training or benchmarking a detector on real drone footage (simulated text frames stand in for perception; the VLM
  path is built in, but no curated video dataset was evaluated).
* Face recognition or identity resolution. The agent deliberately reports appearance only.
* A production notification channel (push/SMS) or a UI. Sinks are pluggable, and console and JSONL are provided.
