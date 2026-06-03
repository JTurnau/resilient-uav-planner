"""
replanner.py
------------
AutoReplanner: mid-flight plan revision triggered after each skill completes
or when a synthetic fault is injected.

When triggered, the executor is paused (held at its current position) while
the LLM makes a decision in a background thread.  If the LLM returns NOMINAL
the plan is unchanged; otherwise the revised tail replaces the remaining steps
and execution continues immediately.

Synthetic fault injection
-------------------------
Use EventInjector (see event_injector.py) to schedule faults at a specific
step index or wall-clock time.  The injector calls node.inject_event(message),
which suspends execution and triggers a replan with the supplied failure context.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from llm_client import query_llm, model_label
from plan_utils  import parse_replan_response, validate_plan, print_plan
from prompts     import build_replan_system_prompt, NO_REPLAN_AFTER

if TYPE_CHECKING:
    from mission_executor import MissionExecutorNode

log = logging.getLogger(__name__)


class AutoReplanner:
    """
    Triggers a background LLM replan call after each event.

    The node sets _replan_pending=True before calling trigger().
    The control loop holds at the current setpoint while _replan_pending is True.
    The background thread clears _replan_pending when done.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._running = False

    def trigger(
        self,
        node: "MissionExecutorNode",
        completed_step_name: str,
        failure_context: str | None = None,
    ) -> None:
        """
        Schedule a replan.  Skipped for steps where no new discoveries are
        possible (takeoff, return_home) unless a failure context is supplied -
        anomaly injection always replans.
        """
        if failure_context is None and completed_step_name in NO_REPLAN_AFTER:
            node.get_logger().info(
                f"[REPLAN] Skipping replan after '{completed_step_name}' "
                f"(no new discoveries possible)"
            )
            node._replan_pending = False
            return

        with self._lock:
            if self._running:
                node.get_logger().info(
                    "[REPLAN] Previous replan still running - skipping trigger"
                )
                node._replan_pending = False
                return
            self._running = True

        mission_intent  = node.mission_intent
        current_step    = node.current_step
        completed_steps = list(node.plan[:current_step])
        remaining_steps = list(node.plan[current_step:])
        objects         = list(node.ctx.objects)

        def _worker():
            try:
                system_prompt = build_replan_system_prompt(
                    mission_intent  = mission_intent,
                    completed_steps = completed_steps,
                    remaining_steps = remaining_steps,
                    objects         = objects,
                    failure_context = failure_context,
                )

                node.get_logger().info(
                    "\n" + "=" * 80 +
                    "\n[REPLAN PROMPT SENT TO LLM]\n" +
                    system_prompt +
                    "\n" + "=" * 80
                )
                node.get_logger().info(
                    f"[REPLAN] Querying {model_label()}  "
                    f"({len(objects)} object(s) known, "
                    f"{len(remaining_steps)} step(s) remaining)"
                )

                tail           = None
                reason         = None
                current_system = system_prompt

                for attempt in range(3):
                    try:
                        raw = query_llm("", system=current_system, max_tokens=4096)
                        node.get_logger().info(
                            f"[REPLAN] Raw response (attempt {attempt+1}):\n{raw}"
                        )

                        tail, reason = parse_replan_response(raw)

                        if tail is None:
                            node.get_logger().info(
                                "[REPLAN] LLM returned NOMINAL - plan unchanged"
                            )
                            return

                        ok, errs = validate_plan(tail, is_tail=True)
                        if ok:
                            break

                        err_str = "; ".join(errs)
                        node.get_logger().warn(
                            f"[REPLAN] Invalid tail (attempt {attempt+1}): {err_str}"
                        )
                        current_system = (
                            current_system
                            + f"\n\n--- PREVIOUS ATTEMPT {attempt+1} WAS INVALID ---\n"
                            + "Errors found:\n"
                            + "\n".join(f"  - {e}" for e in errs)
                            + "\n\nFix all errors and output the corrected JSON array "
                              "followed by REASON: <one sentence>."
                        )
                        tail   = None
                        reason = None

                    except Exception as exc:
                        node.get_logger().warn(
                            f"[REPLAN] Attempt {attempt+1} exception: {exc}"
                        )

                if tail is None:
                    node.get_logger().warn(
                        "[REPLAN] Could not produce a valid tail after 3 attempts - "
                        "leaving plan unchanged"
                    )
                    return

                live_step = node.current_step
                head      = list(node.plan[:live_step])
                node.plan = head + tail

                reason_str = f"  Reason: {reason}" if reason else ""
                print_plan(
                    node.plan,
                    logger       = node.get_logger(),
                    current_step = live_step,
                    label        = (
                        f"REVISED PLAN  ({len(tail)}-step tail from replanner)"
                        + (f"\n  {reason_str}" if reason_str else "")
                    ),
                )

            finally:
                node._replan_pending = False
                with self._lock:
                    self._running = False

        threading.Thread(target=_worker, daemon=True).start()
