"""
plan_utils.py
-------------
Plan parsing, validation, and pretty-printing for the Gazebo executor.

extract_json(), validate_plan(), and parse_replan_response() are taken
verbatim from experiment_utils.py so that validation behaviour is identical
between the offline eval harness and the live executor.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from prompts import MIN_STANDOFF_M

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON EXTRACTION  (verbatim from experiment_utils.py)
# ---------------------------------------------------------------------------

def extract_json(text: str) -> list:
    text = re.sub(r"```[\w]*", "", text).strip("`").strip()
    for bad, good in {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-", "\u00a0": " ",
    }.items():
        text = text.replace(bad, good)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    text = re.sub(r'"args":\s*"\{\}"', '"args": {}', text)
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    return [item for item in parsed if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# PLAN VALIDATION  (verbatim from experiment_utils.py)
# ---------------------------------------------------------------------------

def validate_plan(plan: list, is_tail: bool = False) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not plan:
        return False, ["Plan is empty"]

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            errors.append(f"Step {i} is not a dict")

    if errors:
        return False, errors

    if not is_tail and plan[0].get("state") != "takeoff":
        errors.append("Must start with takeoff")
    if plan[-1].get("state") != "return_home":
        errors.append("Must end with return_home")

    valid_states = {"takeoff", "search", "approach", "map", "return_home"}
    approached: set[Any] = set()

    for i, step in enumerate(plan):
        s    = step.get("state", "")
        args = step.get("args", {})

        if s not in valid_states:
            errors.append(f"Unknown state '{s}' at step {i}")
            continue

        repeat = step.get("repeat", 1)
        if not isinstance(repeat, int) or repeat < 1:
            errors.append(f"Invalid repeat={repeat!r} at step {i}")

        if s == "takeoff" and "altitude" not in args:
            errors.append(f"takeoff at {i} missing altitude")

        if s == "search" and args.get("pattern") not in ("yaw_scan", "lawnmower"):
            errors.append(f"search at {i} invalid pattern: {args.get('pattern')}")

        if s == "approach":
            approached.add(args.get("object_id"))
            if "standoff_distance" not in args:
                errors.append(f"approach at {i} missing standoff_distance")

        if s == "map":
            oid = args.get("object_id")
            sd  = float(args.get("standoff_distance", 0))

            if oid not in approached and oid != "all":
                errors.append(
                    f"map at {i} (object_id={oid}) requires a preceding approach")

            if args.get("mode", "orbit") != "orbit":
                errors.append(
                    f"map at {i} invalid mode={args.get('mode')!r} (only 'orbit' supported)")

            if sd < MIN_STANDOFF_M:
                errors.append(f"map at {i} standoff={sd} < {MIN_STANDOFF_M}m")

            if "repeat" in args:
                errors.append(f"map at {i} has 'repeat' inside args (must be top-level)")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# REPLAN RESPONSE PARSING  (verbatim from experiment_utils.py)
# ---------------------------------------------------------------------------

def parse_replan_response(raw: str) -> tuple[list | None, str | None]:
    stripped = raw.strip()

    has_nominal   = bool(re.search(r'\bNOMINAL\b', stripped, re.IGNORECASE))
    has_json_plan = bool(re.search(r'\[\s*\{', stripped))

    if has_nominal and not has_json_plan:
        return None, None

    reason = None
    reason_match = re.search(r"REASON:\s*(.+)$", stripped, re.IGNORECASE | re.MULTILINE)
    if reason_match:
        reason = reason_match.group(1).strip()

    spans: list[tuple[int, int]] = []
    depth = 0
    span_start = -1
    for i, ch in enumerate(stripped):
        if ch == "[":
            if depth == 0:
                span_start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and span_start != -1:
                spans.append((span_start, i))
                span_start = -1

    if not spans:
        raise ValueError("No JSON array found in replanner response")

    best_start, best_end = max(spans, key=lambda s: s[1] - s[0])
    json_portion = stripped[best_start: best_end + 1]
    tail = extract_json(json_portion)
    return tail, reason


# ---------------------------------------------------------------------------
# PRETTY PRINTING
# ---------------------------------------------------------------------------

def print_plan(
    plan: list,
    logger=None,
    current_step: int = -1,
    label: str = "",
) -> None:
    emit  = logger.info if logger else print
    width = 62

    if label:
        emit("=" * width)
        emit(f"  {label}")
        emit("=" * width)

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            emit(f"  --- [{i}] <invalid step: {step!r}>")
            continue
        args_str = ", ".join(f"{k}={v}" for k, v in step.get("args", {}).items())
        rep      = step.get("repeat", 1)
        rep_str  = f" x{rep}" if rep > 1 else ""

        if current_step < 0:
            marker = "   "
        elif i < current_step:
            marker = "[v]"
        elif i == current_step:
            marker = ">>>"
        else:
            marker = "   "

        emit(f"  {marker} [{i}] {step['state']:22} {args_str}{rep_str}")

    if label:
        emit("=" * width)
