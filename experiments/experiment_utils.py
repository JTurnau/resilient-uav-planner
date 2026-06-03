"""
experiment_utils.py
-------------------
Shared utilities for UAV planning/replanning experiments.
 
Provides:
  - LLM client wrapper (OpenRouter / OpenAI) with a uniform interface
  - Planning prompt and plan generation
  - Replan prompt builder and response parser
  - Plan validator
  - MockWorldState for injecting synthetic failure scenarios
  - Result logging helpers
 
Models
------
All models are served via OpenRouter unless otherwise noted.
 
  google/gemini-2.5-flash  (reasoning="medium")
      Gemini 2.5 Flash with mid-level reasoning enabled.
 
  google/gemini-2.5-flash  (reasoning="none")
      Gemini 2.5 Flash with reasoning disabled — pure instruction-following
      baseline for direct comparison with the reasoning variant.
 
  qwen/qwen3-235b-a22b-thinking-2507  (reasoning="medium")
      Qwen3 235B thinking variant with mid-level reasoning enabled.
 
  qwen/qwen3-235b-a22b-2507  (reasoning="none")
      Qwen3 235B instruct variant. No reasoning parameter is passed.
 
  deepseek/deepseek-r1  (reasoning="medium")
      DeepSeek R1 with mid-level reasoning enabled.
 
  o4-mini  (via OpenAI, effort="medium")
      OpenAI o4-mini with reasoning effort set to "medium". Temperature is
      omitted (not supported by the o-series API).
 
Rate-limit / retry policy
--------------------------
All backends retry indefinitely on 429 and transient 5xx errors with
exponential backoff up to MAX_SINGLE_WAIT_S. Wall-clock time spent waiting
is included in the returned latency. Only genuine model-logic failures (bad
JSON, constraint violations) are surfaced as errors to callers.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------------------------

# Each entry defines the OpenRouter model ID and an optional reasoning
# parameter to pass via the "reasoning" field in the API request body.
# reasoning=None means no reasoning parameter is sent.
MODELS: dict[str, dict] = {
    "gemini-2.5-flash-thinking": {
        "backend":   "openrouter",
        "model_id":  "google/gemini-2.5-flash",
        "reasoning": "medium",
    },
    "gemini-2.5-flash-base": {
        "backend":   "openrouter",
        "model_id":  "google/gemini-2.5-flash",
        "reasoning": "none",
    },
    "qwen-235b-thinking": {
        "backend":   "openrouter",
        "model_id":  "qwen/qwen3-235b-a22b-thinking-2507",
        "reasoning": "medium",
    },
    "qwen-235b-instruct": {
        "backend":   "openrouter",
        "model_id":  "qwen/qwen3-235b-a22b-2507",
        "reasoning": None,
    },
    "deepseek-r1": {
        "backend":   "openrouter",
        "model_id":  "deepseek/deepseek-r1",
        "reasoning": "medium",
    },
    "o4-mini": {
        "backend":   "openai",
        "model_id":  "o4-mini",
        "reasoning": None,   # handled separately via reasoning={"effort": "medium"}
    },
}

# ---------------------------------------------------------------------------
# API KEYS
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# RETRY CONFIG
# ---------------------------------------------------------------------------

RETRY_WAIT_S:      int = 15
MAX_SINGLE_WAIT_S: int = 120

# ---------------------------------------------------------------------------
# CLIENT CACHE
# ---------------------------------------------------------------------------

_openrouter_client = None
_openai_client     = None


def _get_openrouter():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI as _OpenAI
        _openrouter_client = _OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI as _OpenAI
        _openai_client = _OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ---------------------------------------------------------------------------
# RATE-LIMIT / TRANSIENT ERROR DETECTION
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        kw in msg for kw in (
            "429", "quota", "rate", "resource_exhausted", "too many requests",
            "503", "service unavailable", "overloaded", "high traffic",
            "try again", "temporarily unavailable", "server error",
            "internal error", "connection", "timeout", "timed out",
        )
    )


# ---------------------------------------------------------------------------
# LLM QUERY
# ---------------------------------------------------------------------------

def query_llm(
    prompt: str,
    system: str,
    model_key: str,
    max_tokens: int = 4096,
) -> tuple[str, float]:
    """
    Call the LLM identified by model_key.
    Returns (response_text, latency_seconds).
 
    Retries indefinitely on all exceptions:
      - Rate-limit / transient errors: exponential backoff up to MAX_SINGLE_WAIT_S.
      - Other errors: fixed RETRY_WAIT_S wait before retry.
 
    OpenRouter models: the "reasoning" field is injected via extra_body when
    cfg["reasoning"] is not None.
 
    OpenAI o4-mini: reasoning_effort="medium" is passed directly; temperature
    is intentionally omitted (not supported by the o-series API).
    """
    
    cfg     = MODELS[model_key]
    backend = cfg["backend"]

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt if prompt else "Respond now."},
    ]

    t0 = time.monotonic()

    if backend == "openrouter":
        wait_s  = RETRY_WAIT_S
        attempt = 0

        extra_body: dict = {}
        if cfg.get("reasoning") is not None:
            extra_body["reasoning"] = {"effort": cfg["reasoning"]}

        while True:
            attempt += 1
            try:
                kwargs: dict[str, Any] = dict(
                    model=cfg["model_id"],
                    messages=messages,
                    max_tokens=max_tokens,
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body

                response = _get_openrouter().chat.completions.create(**kwargs)
                return response.choices[0].message.content, time.monotonic() - t0

            except Exception as exc:
                elapsed  = round(time.monotonic() - t0, 1)
                exc_kind = "Rate-limit" if _is_rate_limit_error(exc) else "Unexpected error"
                print(
                    f"  [{model_key}] {exc_kind} on attempt {attempt} "
                    f"(+{elapsed}s elapsed): {exc!r:.120}. "
                    f"Waiting {wait_s}s before retry..."
                )
                time.sleep(wait_s)
                if _is_rate_limit_error(exc):
                    wait_s = min(wait_s * 2, MAX_SINGLE_WAIT_S)

    if backend == "openai":
        wait_s  = RETRY_WAIT_S
        attempt = 0
        while True:
            attempt += 1
            try:
                response = _get_openai().chat.completions.create(
                    model=cfg["model_id"],
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    reasoning_effort="medium",
                    # temperature: not supported by o-series — intentionally omitted
                )
                return response.choices[0].message.content, time.monotonic() - t0

            except Exception as exc:
                elapsed  = round(time.monotonic() - t0, 1)
                exc_kind = "Rate-limit" if _is_rate_limit_error(exc) else "Unexpected error"
                print(
                    f"  [{model_key}] {exc_kind} on attempt {attempt} "
                    f"(+{elapsed}s elapsed): {exc!r:.120}. "
                    f"Waiting {wait_s}s before retry..."
                )
                time.sleep(wait_s)
                if _is_rate_limit_error(exc):
                    wait_s = min(wait_s * 2, MAX_SINGLE_WAIT_S)

    raise ValueError(f"Unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# CONSTRAINTS
# ---------------------------------------------------------------------------

MIN_STANDOFF_M = 5.0

_NO_REPLAN_AFTER: frozenset[str] = frozenset({"takeoff", "return_home"})

# ---------------------------------------------------------------------------
# PLANNING PROMPT
# ---------------------------------------------------------------------------

PLANNING_PROMPT = """
You are a UAV mission planner.
Given a natural language mission description, produce a JSON execution plan
as a sequence of parameterized skill invocations chosen from the vocabulary below.
The UAV operates in an unknown environment - objects are discovered only through
onboard perception during flight, not known in advance.

=== AVAILABLE SKILLS ===

Each plan step has the shape:
  { "state": <skill_name>, "args": { ... }, "repeat": <int, default 1> }

"repeat": N runs that skill to completion N times before advancing.

SKILL          ARGS                                   NOTES
-------------------------------------------------------------------------------
takeoff        altitude: float (m)                    ALWAYS the first step.
                                                      Infer a reasonable altitude
                                                      from context if not stated:
                                                      confined space  ->  2-3 m
                                                      open area       ->  5-8 m

search         pattern: "yaw_scan"                    Discovers objects via onboard
                        "lawnmower"                   perception during flight.
                                                      yaw_scan  - rotate in place;
                                                                  good for small areas
                                                                  or quick sweeps.
                                                      lawnmower - systematic grid
                                                                  coverage; use when
                                                                  thorough search of a
                                                                  large area is needed.

approach       object_id: int (starting at 1) | "all" Fly to standoff_distance from
               standoff_distance: float (m, min 5.0)  the specified target(s).
                                                      MUST appear before map for the
                                                      same object_id.

map            object_id: int (starting at 1) | "all" Orbit the target once per
               standoff_distance: float (m, min 5.0)  invocation. standoff >= 5.0 m.
               mode: "orbit"                          Use repeat: N to orbit N times.
                                                      approach for the same object_id
                                                      MUST precede this.

return_home    (no args)                              Always the last step.

=== CONSTRAINTS ===

  1. First step MUST be takeoff. Last step must be return_home.
  2. approach(object_id=X) MUST appear immediately before map(object_id=X).
  3. Do not include approach or map unless the mission explicitly involves object
     interaction.
  4. Do not add steps that are not implied by the mission description.
  5. standoff_distance must be >= 5.0 m for both approach and map.
  6. After search, assume objects may be found. The automatic replanner will
     adjust the plan based on what is actually discovered.

=== OUTPUT ===

Respond with a JSON array only. No explanation, no markdown, no backticks.
"""

# ---------------------------------------------------------------------------
# REPLAN PROMPT BUILDER
# ---------------------------------------------------------------------------

def build_replan_system_prompt(
    mission_intent:  str,
    completed_steps: list[dict],
    remaining_steps: list[dict],
    objects:         list["MockObject"],
    failure_context: str | None = None,
) -> str:

    def _fmt_steps(steps):
        if not steps:
            return "  (none)"
        lines = []
        for i, st in enumerate(steps):
            args_str = ", ".join(f"{k}={v}" for k, v in st.get("args", {}).items())
            rep      = st.get("repeat", 1)
            rep_str  = f" x{rep}" if rep > 1 else ""
            lines.append(f"  [{i}] {st['state']:22} {args_str}{rep_str}")
        return "\n".join(lines)

    def _build_manifest(objs):
        if not objs:
            return "  No objects discovered yet."
        data = [
            {
                "object_id":            o.id,
                "world_ned_x_m":        round(o.world_x, 2),
                "world_ned_y_m":        round(o.world_y, 2),
                "depth_at_detection_m": round(o.depth_m,  2),
            }
            for o in objs
        ]
        return (
            f"  {len(objs)} object(s) discovered:\n"
            "  ```json\n"
            + "  " + json.dumps(data, indent=2).replace("\n", "\n  ")
            + "\n  ```"
        )

    failure_section = ""
    if failure_context:
        failure_section = f"""
=== SECTION 3b - FAILURE / ANOMALY REPORT ===

The following event occurred during execution that may require you to revise
the remaining plan:

  {failure_context}

Consider whether the remaining plan still makes sense given this event.
"""

    return f"""\
=== SECTION 1 - BACKGROUND ===

You are a mid-flight autonomous UAV mission replanner.

After each skill completes (or fails), you receive a full snapshot of the
ongoing mission: the complete execution history, all steps still scheduled,
and all objects discovered so far.

Your job is to decide whether the remaining plan will fully satisfy the user's
mission by analyzing the execution history, remaining plan, and events / anomalies.

You do NOT re-plan from scratch, only the tail (steps yet to execute) is yours
to change. Completed steps are fixed history.
Do NOT make changes unless they are ABSOLUTELY NECESSARY to complete the
intended mission.


=== SECTION 2 - USER MISSION ===

  "{mission_intent}"

This is the exact mission the user requested. Use it as the ground truth
for what "success" means.


=== SECTION 3 - AVAILABLE SKILLS ===

Only the following skills may appear in a revised tail plan.
takeoff is already complete, do NOT include it again.

SKILL         ARGS                                   NOTES
------------------------------------------------------------------------------
search        pattern: "yaw_scan"                    Discovers objects via onboard
                       "lawnmower"                   perception during flight.
                                                     yaw_scan  - rotate in place;
                                                                 good for small areas
                                                                 or quick sweeps.
                                                     lawnmower - systematic grid
                                                                 coverage; use when
                                                                 thorough search of a
                                                                 large area is needed.

approach      object_id: int                         Fly to standoff_distance
              standoff_distance: float (m)           from a specific object.
                                                     MUST immediately precede
                                                     the map step for the same
                                                     object_id.
                                                     standoff >= 5.0 m

map           object_id: int                         One full orbit per
              standoff_distance: float (m, >= 5.0)   invocation.
              mode: "orbit"                          Use repeat: N to orbit N times.

return_home   (no args)                              Must be the last step.
{failure_section}
Required pattern for each object you intend to map:
  {{"state":"approach","args":{{"standoff_distance":D,"object_id":N}}}},
  {{"state":"map","args":{{"mode":"orbit","standoff_distance":D,"object_id":N}},"repeat":K}}

Hard constraints:
  - standoff_distance >= 5.0 for both approach and map
  - approach(object_id=X) MUST immediately precede map(object_id=X)
  - "repeat" is a TOP-LEVEL step field, never inside args
  - Last step MUST be return_home


=== SECTION 4 - MISSION STATUS ===

EXECUTION HISTORY (these steps are done - fixed, cannot be changed):
{_fmt_steps(completed_steps)}

REMAINING PLAN (scheduled - not yet executed):
{_fmt_steps(remaining_steps)}

DISCOVERED OBJECTS:
{_build_manifest(objects)}


=== SECTION 5 - OUTPUT DIRECTIVE ===

Do NOT repeat, echo, or summarize any part of this prompt in your response.

If the mission will be fully satisfied with no structural issues, respond with
exactly:

NOMINAL

ONLY if the remaining plan MUST be adjusted, output:
  1. A corrected JSON array of the complete revised tail (all steps from
     now until return_home). Every element must be a complete JSON object.
  2. Immediately after the JSON, on a new line starting with "REASON:", a
     single concise sentence explaining what was wrong and why your revision
     is necessary.

Do not include any other explanation, markdown, or backticks.
"""


# ---------------------------------------------------------------------------
# PLAN PARSING AND VALIDATION
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
                errors.append(f"map at {i} invalid mode={args.get('mode')!r} (only 'orbit' supported)")

            if sd < MIN_STANDOFF_M:
                errors.append(f"map at {i} standoff={sd} < {MIN_STANDOFF_M}m")

            if "repeat" in args:
                errors.append(f"map at {i} has 'repeat' inside args (must be top-level)")

    return len(errors) == 0, errors


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
# MOCK WORLD STATE
# ---------------------------------------------------------------------------

@dataclass
class MockObject:
    id:      int
    world_x: float
    world_y: float
    depth_m: float


@dataclass
class MockWorldState:
    objects:          list[MockObject]   = field(default_factory=list)
    completed_steps:  list[dict]         = field(default_factory=list)
    remaining_steps:  list[dict]         = field(default_factory=list)
    failure_context:  str | None         = None
    battery_pct:      float              = 100.0


def make_world(
    *,
    n_objects:       int   = 0,
    completed_steps: list  | None = None,
    remaining_steps: list  | None = None,
    failure_context: str   | None = None,
    battery_pct:     float = 100.0,
) -> MockWorldState:
    objs = [
        MockObject(id=i + 1, world_x=float(i * 5), world_y=0.0, depth_m=5.0)
        for i in range(n_objects)
    ]
    return MockWorldState(
        objects         = objs,
        completed_steps = completed_steps or [],
        remaining_steps = remaining_steps or [],
        failure_context = failure_context,
        battery_pct     = battery_pct,
    )


# ---------------------------------------------------------------------------
# PLAN GENERATION
# ---------------------------------------------------------------------------

def generate_plan(
    mission: str,
    model_key: str,
    max_attempts: int = 3,
    run_id: int = 1,
) -> dict:
    """
    Attempt to generate a valid plan for a mission.

    run_id is stamped onto the result dict so that records from multiple
    experimental runs can be distinguished when aggregated.
    """
    result = {
        "run_id":        run_id,
        "mission":       mission,
        "model":         model_key,
        "success":       False,
        "plan":          None,
        "attempts":      0,
        "latency_s":     0.0,
        "raw_responses": [],
        "errors":        [],
    }
    total_latency = 0.0

    for attempt in range(max_attempts):
        result["attempts"] += 1
        try:
            raw, latency = query_llm(mission, system=PLANNING_PROMPT, model_key=model_key)
            total_latency += latency
            result["raw_responses"].append(raw)
            plan = extract_json(raw)
            ok, errs = validate_plan(plan)
            if ok:
                result["success"]   = True
                result["plan"]      = plan
                result["latency_s"] = total_latency
                return result
            result["errors"].append(f"attempt {attempt+1}: " + "; ".join(errs))
        except Exception as exc:
            result["errors"].append(f"attempt {attempt+1}: {exc}")

    result["latency_s"] = total_latency
    return result


# ---------------------------------------------------------------------------
# REPLAN CALL
# ---------------------------------------------------------------------------

def run_replan(
    mission_intent:  str,
    world:           MockWorldState,
    model_key:       str,
    max_attempts:    int = 3,
    run_id:          int = 1,
) -> dict:
    """
    Call the replanner and return a result dict.

    run_id is stamped onto the result dict so that records from multiple
    experimental runs can be distinguished when aggregated.
    """
    system_prompt = build_replan_system_prompt(
        mission_intent  = mission_intent,
        completed_steps = world.completed_steps,
        remaining_steps = world.remaining_steps,
        objects         = world.objects,
        failure_context = world.failure_context,
    )

    result = {
        "run_id":        run_id,
        "mission":       mission_intent,
        "model":         model_key,
        "nominal":       False,
        "tail":          None,
        "reason":        None,
        "valid":         False,
        "attempts":      0,
        "latency_s":     0.0,
        "raw_responses": [],
        "errors":        [],
    }
    total_latency = 0.0
    current_system = system_prompt

    for attempt in range(max_attempts):
        result["attempts"] += 1
        try:
            raw, latency = query_llm("", system=current_system, model_key=model_key)
            total_latency += latency
            result["raw_responses"].append(raw)

            tail, reason = parse_replan_response(raw)

            if tail is None:
                result["nominal"]   = True
                result["valid"]     = True
                result["latency_s"] = total_latency
                return result

            ok, errs = validate_plan(tail, is_tail=True)
            if ok:
                result["tail"]      = tail
                result["reason"]    = reason
                result["valid"]     = True
                result["latency_s"] = total_latency
                return result

            err_str = "; ".join(errs)
            result["errors"].append(f"attempt {attempt+1}: {err_str}")
            current_system = (
                current_system
                + f"\n\n--- PREVIOUS ATTEMPT {attempt+1} WAS INVALID ---\n"
                + "Errors found:\n"
                + "\n".join(f"  - {e}" for e in errs)
                + "\n\nFix all errors and output the corrected JSON array "
                  "followed by REASON: <one sentence>."
            )

        except Exception as exc:
            result["errors"].append(f"attempt {attempt+1}: {exc}")

    result["latency_s"] = total_latency
    return result


# ---------------------------------------------------------------------------
# FORMATTING HELPERS
# ---------------------------------------------------------------------------

def fmt_plan(plan: list) -> str:
    if not plan:
        return "  (empty)"
    lines = []
    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            lines.append(f"  [{i}] <invalid: {step!r}>")
            continue
        args_str = ", ".join(f"{k}={v}" for k, v in step.get("args", {}).items())
        rep      = step.get("repeat", 1)
        rep_str  = f" x{rep}" if rep > 1 else ""
        lines.append(f"  [{i}] {step['state']:20} {args_str}{rep_str}")
    return "\n".join(lines)


def save_json(path: str, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")