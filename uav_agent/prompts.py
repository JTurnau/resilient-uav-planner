"""
prompts.py
----------
LLM prompts for UAV mission planning and replanning.

PLANNING_PROMPT and build_replan_system_prompt() are taken verbatim from
experiment_utils.py so that prompts issued by the Gazebo executor are
byte-for-byte identical to those used in the offline experiments.

The only executor-specific addition is that build_replan_system_prompt()
accepts SpottedObject instances in addition to MockObject - both expose
.id, .world_x, .world_y, .depth_m so no adapter is needed.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# CONSTRAINTS  (shared with plan_utils.py)
# ---------------------------------------------------------------------------

MIN_STANDOFF_M: float = 5.0

# Steps after which the replanner is never triggered (no new discoveries possible).
NO_REPLAN_AFTER: frozenset[str] = frozenset({"takeoff", "return_home"})

# ---------------------------------------------------------------------------
# PLANNING PROMPT  (verbatim from experiment_utils.py)
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
# REPLAN PROMPT BUILDER  (verbatim from experiment_utils.py)
# ---------------------------------------------------------------------------

def build_replan_system_prompt(
    mission_intent:  str,
    completed_steps: list[dict],
    remaining_steps: list[dict],
    objects:         list,          # SpottedObject or MockObject: .id .world_x .world_y .depth_m
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
