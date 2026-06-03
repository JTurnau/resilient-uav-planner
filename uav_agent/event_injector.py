"""
event_injector.py
-----------------
Synthetic fault injection for mid-flight replan testing.

Two event types are available:

  SyntheticEvent
    Injects a fault message into the replanner.  The LLM receives the
    remaining plan unchanged plus a FAILURE / ANOMALY REPORT section
    describing the fault.  Use this to test whether the model reacts
    correctly to explicitly reported failures (sensor loss, wind, battery).

  PlanTruncationEvent
    Silently removes a contiguous slice of steps from the remaining plan
    before the replanner runs, then triggers a replan with no failure
    message.  The LLM sees a shortened remaining plan alongside the full
    object list and execution history and must detect, without being told,
    that steps are missing and add them back.

    Use this to test silent-failure recovery: e.g. after object 1 is mapped
    the planner drops the approach+map steps for object 2, leaving only
    return_home.  The model should notice that object 2 appears in the
    discovered-objects manifest but has no corresponding steps and insert
    the missing approach+map pair.

Both types share the same trigger interface:
  on_step: int   - fires after the executor completes step N (0-indexed)
  at_time: float - fires this many wall-clock seconds after the drone arms

Exactly one trigger must be set per event.  Each event fires at most once.

-------------------------------------------------------------------------------
Example A - explicit fault report (SyntheticEvent)
-------------------------------------------------------------------------------

    INJECTED_EVENTS = [
        SyntheticEvent(
            on_step         = 2,
            failure_context = "Depth sensor returned NaN for 10 consecutive "
                              "frames - object distance unavailable.",
        ),
    ]

-------------------------------------------------------------------------------
Example B - silent plan truncation (PlanTruncationEvent)
-------------------------------------------------------------------------------

Scenario: the LLM generates a four-step tail after search completes:
    [2] approach   object_id=1, standoff_distance=6.0
    [3] map        object_id=1, standoff_distance=6.0, mode=orbit
    [4] approach   object_id=2, standoff_distance=6.0
    [5] map        object_id=2, standoff_distance=6.0, mode=orbit
    [6] return_home

After step 3 (map of object 1) completes, inject a truncation that removes
the two object-2 steps, leaving the remaining plan as just [return_home].
The replanner fires, sees object 2 in the manifest with no pending steps,
and must insert approach+map for object 2 before return_home.

    INJECTED_EVENTS = [
        PlanTruncationEvent(
            on_step      = 3,
            remove_states = ["approach", "map"],
            remove_count  = 2,
        ),
    ]

  remove_states : only steps whose "state" field matches one of these values
                  are eligible for removal.
  remove_count  : how many of those matching steps to drop, counting from
                  the front of the remaining plan.  Defaults to removing all
                  matching steps if not set.

-------------------------------------------------------------------------------
Leave INJECTED_EVENTS empty for a nominal flight with no injected faults.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mission_executor import MissionExecutorNode

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHARED TRIGGER BASE
# ---------------------------------------------------------------------------

@dataclass
class _BaseEvent:
    """
    Trigger bookkeeping shared by all event types.

    Exactly one of on_step / at_time must be set.
    """
    on_step:  int   | None = None
    at_time:  float | None = None

    _fired: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if (self.on_step is None) == (self.at_time is None):
            raise ValueError(
                f"{type(self).__name__}: exactly one of on_step / at_time must be set"
            )

    def _is_due(self, node: "MissionExecutorNode", elapsed: float) -> bool:
        if self.on_step is not None and node.current_step > self.on_step:
            return True
        if self.at_time is not None and elapsed >= self.at_time:
            return True
        return False

    def fire(self, node: "MissionExecutorNode") -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FAULT-MESSAGE EVENT
# ---------------------------------------------------------------------------

@dataclass
class SyntheticEvent(_BaseEvent):
    """
    Pauses execution and passes failure_context verbatim to the replanner
    as a FAILURE / ANOMALY REPORT.  The remaining plan is not modified.

    failure_context : the fault description shown to the LLM.
    """
    failure_context: str = ""

    def fire(self, node: "MissionExecutorNode") -> None:
        log.info(f"[INJECT] SyntheticEvent firing: {self.failure_context!r:.80}")
        node.inject_event(self.failure_context)


# ---------------------------------------------------------------------------
# SILENT PLAN TRUNCATION EVENT
# ---------------------------------------------------------------------------

@dataclass
class PlanTruncationEvent(_BaseEvent):
    """
    Silently removes steps from the remaining plan, then triggers a replan
    with no failure message.

    The LLM receives:
      - execution history up to the current step (unchanged)
      - the shortened remaining plan (steps removed)
      - the full discovered-objects manifest (unchanged)
      - no failure_context section

    It must infer from the mismatch between discovered objects and remaining
    steps that action is required and produce a corrected tail.

    remove_states : list of "state" values eligible for removal
                    (e.g. ["approach", "map"]).  Steps whose state is not
                    in this list are never touched.
    remove_count  : maximum number of matching steps to drop, counting from
                    the front of the remaining plan.  Set to None to remove
                    all matching steps.
    """
    remove_states: list[str] = field(default_factory=list)
    remove_count:  int | None = None

    def fire(self, node: "MissionExecutorNode") -> None:
        live_step      = node.current_step
        remaining      = node.plan[live_step:]
        to_remove      = set(self.remove_states)
        removed        = 0
        surviving      = []

        for step in remaining:
            state = step.get("state", "")
            if (
                state in to_remove
                and (self.remove_count is None or removed < self.remove_count)
            ):
                removed += 1
                log.info(
                    f"[INJECT] PlanTruncationEvent: dropping step "
                    f"state={state!r} args={step.get('args', {})}"
                )
            else:
                surviving.append(step)

        node.plan = list(node.plan[:live_step]) + surviving

        log.info(
            f"[INJECT] PlanTruncationEvent: removed {removed} step(s) - "
            f"{len(surviving)} step(s) remain in tail.  Triggering silent replan."
        )

        # Trigger a replan with no failure_context so the LLM must detect
        # the gap itself from the object manifest and execution history.
        node._replan_pending = True
        node._replanner.trigger(
            node,
            completed_step_name = node.plan[live_step - 1].get("state", "")
                                  if live_step > 0 else "",
            failure_context     = None,
        )


# ---------------------------------------------------------------------------
# INJECTOR
# ---------------------------------------------------------------------------

class EventInjector:
    """
    Attached to MissionExecutorNode.  Call check() from the control loop
    each tick; it fires any due events in declaration order.
    """

    def __init__(self, events: list[_BaseEvent]):
        self._events    = events
        self._arm_time: float | None = None

    def arm(self) -> None:
        """Call once when the vehicle arms so at_time offsets are anchored."""
        self._arm_time = time.monotonic()

    def check(self, node: "MissionExecutorNode") -> None:
        """Called every control-loop tick.  Fires any pending events."""
        if node._replan_pending:
            return

        elapsed = (
            time.monotonic() - self._arm_time
            if self._arm_time is not None
            else 0.0
        )

        for evt in self._events:
            if evt._fired:
                continue
            if evt._is_due(node, elapsed):
                evt._fired = True
                evt.fire(node)


# ---------------------------------------------------------------------------
# CONFIGURE EVENTS HERE
# ---------------------------------------------------------------------------

INJECTED_EVENTS: list[_BaseEvent] = [
    # --- Example A: explicit fault report ---
    #SyntheticEvent(
    #    on_step         = 2,
    #    failure_context = "(1) Perception lost track of object 1 during approach. "
    #                      "(2) Battery telemetry reports 12% remaining - critically low.",
    #),

    # --- Example B: silent plan truncation (object 2 steps removed) ---
    # After step 3 (map of object 1) completes, drop the approach+map for
    # object 2 so the remaining plan is only [return_home].  The replanner
    # must detect from the object manifest that object 2 is unhandled.
    # PlanTruncationEvent(
    #     on_step       = 3,
    #     remove_states = ["approach", "map"],
    #     remove_count  = 2,
    # ),
]