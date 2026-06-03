"""
scenarios.py
------------
All scenario definitions for UAV planning and replanning offline evaluation.

Experiment 0 (Planning):  10 scenarios across L1, L2, L3
Experiment 1 (Replanning, isolated failures): 8 scenarios
Experiment 2 (Replanning, silent/implicit failures): 10 scenarios
Experiment 3 (Replanning, compound/complex failures): 16 scenarios

Total replanning scenarios: 34
"""

from dataclasses import dataclass, field
from typing import Optional
import json


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class PlanningScenario:
    id: str
    level: str
    mission_text: str
    reference_plan: list
    notes: str
    # Indicates whether the takeoff altitude in the reference plan was
    # explicitly stated in the mission_text (True) or must be inferred (False).
    # When False, the evaluator accepts any altitude in [2.0, 8.0] m.
    altitude_explicit: bool = True
    # Indicates whether standoff_distance values in the reference plan were
    # explicitly stated in the mission_text (True) or must be inferred (False).
    # When False, the evaluator accepts any standoff >= MIN_STANDOFF_M (5.0 m).
    standoff_explicit: bool = True
    # Indicates whether the search pattern in the reference plan was explicitly
    # named in the mission_text (True) or must be inferred from context (False).
    # When False, the evaluator accepts either valid pattern ("yaw_scan" or
    # "lawnmower") as correct, since the model is free to choose.
    search_explicit: bool = True


@dataclass
class ReplanScenario:
    id: str
    experiment: int
    category: str
    mission_text: str
    completed_steps: list
    remaining_steps: list
    objects: list          # list of dicts: {"id": int, "world_x": float, "world_y": float, "depth_m": float}
    failure_context: Optional[str]
    ground_truth: str      # "NOMINAL" or JSON string of reference tail
    is_nominal: bool
    allows_partial_credit: bool
    notes: str
    # Mirrors the same semantics as PlanningScenario flags, applied to the
    # mission_text of each replanning scenario.  The replanning evaluator uses
    # these when checking whether a model-generated tail matches the reference.
    #
    # altitude_explicit  — True if takeoff altitude is stated in mission_text.
    #                      All replanning missions omit altitude, so always False.
    #                      (Takeoff never appears in a replan tail anyway, but the
    #                      flag is kept for symmetry and future-proofing.)
    #
    # standoff_explicit  — True if a specific standoff distance (e.g. "6 meter
    #                      standoff") is stated in the mission_text.  When False
    #                      the evaluator accepts any standoff >= MIN_STANDOFF_M.
    #
    # search_explicit    — True if a search pattern ("yaw_scan" / "lawnmower")
    #                      is named in the mission_text.  All replanning missions
    #                      use generic language ("find", "search for"), so always
    #                      False.  The evaluator then accepts either valid pattern.
    altitude_explicit: bool = False   # no replanning mission names an altitude
    standoff_explicit: bool = False   # default; True only when distance is explicit
    search_explicit: bool = False     # no replanning mission names a pattern


# ---------------------------------------------------------------------------
# PLANNING SCENARIOS  (10 total)
# 3× L1, 3× L2, 4× L3
# ---------------------------------------------------------------------------

PLANNING_SCENARIOS: list[PlanningScenario] = [

    # --- L1: no target interaction ---

    PlanningScenario(
        id="P-L1-01",
        level="L1",
        mission_text="Take off to 5 meters and return home.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="Minimal L1: explicit altitude, no search, direct return. Tests that the model does not hallucinate extra steps.",
        altitude_explicit=True,
        standoff_explicit=True,
        search_explicit=True,   # no search step — irrelevant, True by default
    ),

    PlanningScenario(
        id="P-L1-02",
        level="L1",
        mission_text="Take off, do a quick scan of the area, then come back.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="L1 with yaw_scan search. Altitude is unspecified so any reasonable value (2-8 m) is accepted; reference uses 5 m. 'Quick scan' strongly implies yaw_scan but does not name it explicitly.",
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=False,  # "quick scan" implies yaw_scan but doesn't name it
    ),

    PlanningScenario(
        id="P-L1-03",
        level="L1",
        mission_text="Perform a thorough grid search of the area and return to base.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="L1 with lawnmower search. 'Thorough grid search' strongly implies lawnmower. No object interaction.",
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=True,   # "thorough grid search" unambiguously implies lawnmower
    ),

    # --- L2: single target ---

    PlanningScenario(
        id="P-L2-01",
        level="L2",
        mission_text="Search for objects and orbit the first one you find at a 5 meter standoff, then come home.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="L2: single object, one orbit pass, explicit 5 m standoff. Tests approach-then-map ordering for object_id=1.",
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=False,  # "Search for objects" — pattern unspecified
    ),

    PlanningScenario(
        id="P-L2-02",
        level="L2",
        mission_text="Scan for objects. If you find any, orbit the first one twice at 6 meters, then return.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 6.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 6.0, "mode": "orbit"}, "repeat": 2},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="L2: single object, two orbit passes (repeat: 2), explicit 6 m standoff. Tests repeat field on map step.",
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=False,  # "Scan for objects" — pattern unspecified
    ),

    PlanningScenario(
        id="P-L2-03",
        level="L2",
        mission_text="Do a thorough search of the field. Map the first object you find with one orbit at 8 meters standoff, then return home.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 8.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 8.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="L2: single object, with explicit 8 m standoff and lawnmower search. Tests that 'thorough' maps to lawnmower and standoff is propagated correctly.",
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=True,   # "thorough search" unambiguously implies lawnmower
    ),

    # --- L3: multi-target or complex ---

    PlanningScenario(
        id="P-L3-01",
        level="L3",
        mission_text="Search for objects and map every one you find with a single orbit at 5 meter standoff, then return home.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": "all", "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": "all", "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="L3 multi-target baseline: object_id='all', 1 orbit each, 5 m standoff. Pattern unspecified — either valid pattern accepted.",
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=False,  # "Search for objects" — pattern unspecified
    ),

    PlanningScenario(
        id="P-L3-02",
        level="L3",
        mission_text="Take off to 8 meters, do a lawnmower search, then orbit every object you find three times at 5 meters. Return home when done.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 8.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": "all", "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": "all", "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 3},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="L3: explicit altitude (8 m), lawnmower, 3 orbit passes (repeat: 3). Tests all three complexity axes simultaneously.",
        altitude_explicit=True,
        standoff_explicit=True,
        search_explicit=True,   # "lawnmower search" explicitly named
    ),

    PlanningScenario(
        id="P-L3-03",
        level="L3",
        mission_text="Find any objects and map each one twice at 7 meter standoff, then come home.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": "all", "standoff_distance": 7.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": "all", "standoff_distance": 7.0, "mode": "orbit"}, "repeat": 2},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="L3: 2 orbit passes, explicit 7 m standoff on both approach and map. Tests non-default repeat and non-default standoff together.",
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=False,  # "Find any objects" — pattern unspecified
    ),

    PlanningScenario(
        id="P-L3-04",
        level="L3",
        mission_text="Conduct a systematic grid survey of the area. For every object discovered, do three orbits at 6 meter standoff to collect a detailed map. Return to launch when complete.",
        reference_plan=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": "all", "standoff_distance": 6.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": "all", "standoff_distance": 6.0, "mode": "orbit"}, "repeat": 3},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        notes="L3 hardest: lawnmower + 3-orbit passes + 6 m standoff + formal language. All complexity axes engaged.",
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=True,   # "systematic grid survey" unambiguously implies lawnmower
    ),
]


# ---------------------------------------------------------------------------
# REPLANNING SCENARIOS  (34 total)
# ---------------------------------------------------------------------------

# ---- Helpers for building ground_truth JSON strings ----

def _tail_json(*steps) -> str:
    """Serialize a list of step dicts to a compact JSON string."""
    return json.dumps(list(steps))


def _approach(object_id, standoff_distance=5.0):
    return {"state": "approach", "args": {"object_id": object_id, "standoff_distance": standoff_distance}, "repeat": 1}


def _map(object_id, standoff_distance=5.0, repeat=1):
    return {"state": "map", "args": {"object_id": object_id, "standoff_distance": standoff_distance, "mode": "orbit"}, "repeat": repeat}


def _rh():
    return {"state": "return_home", "args": {}, "repeat": 1}


# ===========================================================================
# EXPERIMENT 1 — Isolated Failure Baseline  (8 scenarios)
# ===========================================================================

REPLAN_SCENARIOS: list[ReplanScenario] = [

    # --- EX: explicit REPLAN (3) ---

    ReplanScenario(
        id="E1-EX1",
        experiment=1,
        category="EX",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 10.0, "world_y": 3.0, "depth_m": 5.5},
        ],
        failure_context=(
            "Perception lost track of object 1 during approach. "
            "No confirmed detection in the last 10 seconds."
        ),
        ground_truth=_tail_json(
            _approach(1, 5.0), _map(1, 5.0),
            _approach(2, 5.0), _map(2, 5.0),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "Explicit tracking-loss failure mid-approach. "
            "Replanner must re-insert approach(1) before map(1), "
            "then continue with object 2 and return home. "
            "Tail: A(1,5) → M(1,5) → A(2,5) → M(2,5) → RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E1-EX2",
        experiment=1,
        category="EX",
        mission_text="Find objects and orbit each one twice, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 2},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 4.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 12.0, "world_y": 5.0, "depth_m": 5.2},
        ],
        failure_context=(
            "Setpoint timeout during 2/2 planned orbit pass of object 1. "
            "The drone aborted mid-arc and is now holding position. "
            "The map data from the aborted pass is unusable."
        ),
        ground_truth=_tail_json(
            _approach(1, 5.0), _map(1, 5.0, repeat=1),
            _approach(2, 5.0), _map(2, 5.0, repeat=2),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "Explicit mid-orbit abort; object 1 needs one more orbit pass. "
            "Remaining plan skipped back to approach(2). "
            "Replanner must insert A(1,5)→M(1,5)x1 before object 2's sequence. "
            "Tail: A(1,5) → M(1,5) → A(2,5) → M(2,5)x2 → RH."
        ),
        # Mission: "Find objects and orbit each one twice, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E1-EX3",
        experiment=1,
        category="EX",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
        ],
        failure_context=(
            "Battery telemetry reports 10% remaining - critically low."
        ),
        ground_truth=_tail_json(_rh()),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "Critically low battery (10%) while yet to map object 1. "
            "Remaining plan should be changed to RH for safety. "
            "Tests that the model treats critical battery as a hard abort trigger "
            "and does NOT attempt to continue the planned mapping step."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    # --- NOM: clean NOMINAL (3) ---

    ReplanScenario(
        id="E1-NOM1",
        experiment=1,
        category="NOM",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 4.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 9.0, "world_y": 3.0, "depth_m": 5.2},
        ],
        failure_context=None,
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Clean NOMINAL: object 1 mapped, remaining plan correctly covers object 2 "
            "with approach+map then return_home. No failure context. "
            "Tests false-positive rate - model must not hallucinate a replan."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E1-NOM2",
        experiment=1,
        category="NOM",
        mission_text="Find objects and orbit each one twice, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 2},
        ],
        remaining_steps=[
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 6.0, "world_y": 0.0, "depth_m": 5.0},
        ],
        failure_context=None,
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Clean NOMINAL: only one object found, both orbits completed (repeat:2 in history). "
            "Only return_home remains. No failure context. "
            "Tests that model correctly counts completed map passes and does not replan."
        ),
        # Mission: "Find objects and orbit each one twice, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E1-NOM3",
        experiment=1,
        category="NOM",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[],
        failure_context=None,
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Clean NOMINAL: search complete, zero objects in manifest, remaining plan is just RH. "
            "No failure context. Model must correctly infer that no objects means no mapping needed."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    # --- SW: soft-warning NOMINAL (2) ---

    ReplanScenario(
        id="E1-SW1",
        experiment=1,
        category="SW",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 4.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 9.0, "world_y": 2.0, "depth_m": 5.1},
        ],
        failure_context=(
            "Wind speed measured at 2.5 m/s - within normal operational tolerance."
        ),
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Soft-warning NOMINAL: wind 2.5 m/s is well within tolerance. "
            "Remaining plan is valid and complete. Model must not replan for a minor soft event."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E1-SW2",
        experiment=1,
        category="SW",
        mission_text="Find objects and orbit each one once at 6 meter standoff, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": "all", "standoff_distance": 6.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": "all", "standoff_distance": 6.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 7.0, "world_y": 1.0, "depth_m": 6.0},
        ],
        failure_context=(
            "Depth sensor noise of ±0.1 m detected on last distance estimate - "
            "within acceptable tolerance for standoff geometry."
        ),
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Soft-warning NOMINAL: depth sensor noise ±0.1 m is within tolerance. "
            "Remaining plan covers all discovered objects at correct standoff. "
            "Model must not treat minor sensor noise as a failure requiring replan."
        ),
        # Mission: "Find objects and orbit each one once at 6 meter standoff, then return home."
        # Standoff explicitly stated as 6 meters. No altitude, no pattern named.
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=False,
    ),


    # ===========================================================================
    # EXPERIMENT 2 — Silent / Implicit Failures  (10 scenarios)
    # failure_context is ALWAYS None
    # ===========================================================================

    # --- MM: manifest mismatch REPLAN (3) ---

    ReplanScenario(
        id="E2-MM1",
        experiment=2,
        category="MM",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 10.0, "world_y": 3.0, "depth_m": 5.2},
            {"id": 3, "world_x": 15.0, "world_y": 6.0, "depth_m": 5.4},
        ],
        failure_context=None,
        ground_truth=_tail_json(
            _approach(2, 5.0), _map(2, 5.0),
            _approach(3, 5.0), _map(3, 5.0),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "Silent MM: 3 objects in manifest, object 1 mapped, remaining plan covers only object 2. "
            "Object 3 is absent from remaining plan - no error message given. "
            "LLM must detect the gap by comparing manifest to plan. "
            "Tail: A(2,5) → M(2,5) → A(3,5) → M(3,5) → RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E2-MM2",
        experiment=2,
        category="MM",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 10.0, "world_y": 4.0, "depth_m": 5.3},
            {"id": 3, "world_x": 15.0, "world_y": 7.0, "depth_m": 5.5},
            {"id": 4, "world_x": 20.0, "world_y": 2.0, "depth_m": 5.1},
        ],
        failure_context=None,
        ground_truth=_tail_json(
            _approach(1, 5.0), _map(1, 5.0),
            _approach(2, 5.0), _map(2, 5.0),
            _approach(3, 5.0), _map(3, 5.0),
            _approach(4, 5.0), _map(4, 5.0),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "Silent MM (large gap): 4 objects in manifest, remaining plan covers only object 1. "
            "LLM must enumerate all 4 objects and extend the tail for objects 2, 3, 4. "
            "No failure context. Tail: A(1)→M(1)→A(2)→M(2)→A(3)→M(3)→A(4)→M(4)→RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E2-MM3",
        experiment=2,
        category="MM",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 3, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 3, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 11.0, "world_y": 4.0, "depth_m": 5.3},
            {"id": 3, "world_x": 15.0, "world_y": 5.0, "depth_m": 5.7},
        ],
        failure_context=None,
        ground_truth=_tail_json(
            _approach(2, 5.0), _map(2, 5.0),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "Silent MM: object 1 and 3 mapped and in history, remaining plan is just RH. "
            "Object 2 is in manifest but has no approach+map in the remaining plan. "
            "LLM must detect uncovered object 2 and insert A(2,5)→M(2,5) before RH. "
            "Tail: A(2,5) → M(2,5) → RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    # --- LC: lap/pass undercount REPLAN (3) ---

    ReplanScenario(
        id="E2-LC1",
        experiment=2,
        category="LC",
        mission_text="Find objects and orbit each one twice, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
        ],
        failure_context=None,
        ground_truth=_tail_json(
            _approach(1, 5.0), _map(1, 5.0, repeat=1),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "Silent LC: mission requires 2 orbits of object 1. "
            "History shows only 1 map pass (repeat:1). Remaining plan is just RH. "
            "LLM must count completed passes and add the missing one. "
            "Tail: A(1,5) → M(1,5)x1 → RH  (adding the missing second pass)."
        ),
        # Mission: "Find objects and orbit each one twice, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E2-LC2",
        experiment=2,
        category="LC",
        mission_text="Search for objects and do three orbit passes on each one you find, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 6.0, "world_y": 0.0, "depth_m": 5.0},
        ],
        failure_context=None,
        ground_truth=_tail_json(
            _approach(1, 5.0), _map(1, 5.0, repeat=2),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "Silent LC: mission requires 3 passes. History shows 1 done; remaining plan passes 1 more time (total 2). "
            "LLM must count: 1 done + 1 scheduled = 2 < 3 required. "
            "Must add approach(1) + map(1)x2 to cover the 2 missing passes. "
            "Tail: A(1,5) → M(1,5)x2 → RH."
        ),
        # Mission: "Search for objects and do three orbit passes on each one you find, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E2-LC3",
        experiment=2,
        category="LC",
        mission_text="Find objects and orbit each one twice at 6 meter standoff, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 6.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 6.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 6.0, "world_y": 0.0, "depth_m": 6.0},
            {"id": 2, "world_x": 13.0, "world_y": 5.0, "depth_m": 6.0},
        ],
        failure_context=None,
        ground_truth=_tail_json(
            _approach(1, 6.0), _map(1, 6.0, repeat=1),
            _approach(2, 6.0), _map(2, 6.0, repeat=2),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "Silent lap count failure with approach also missing: object 1 partially mapped (1 pass at 6 m). "
            "Object 2 is in manifest but has 0 passes and no approach in the remaining plan. "
            "LLM must re-add A(1,6)→M(1,6)x1 → A(2,6)→M(2,6)x2."
        ),
        # Mission: "Find objects and orbit each one twice at 6 meter standoff, then return home."
        # Standoff explicitly stated as 6 meters. No altitude, no pattern named.
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=False,
    ),

    # --- NOM: clean NOMINAL (4) ---

    ReplanScenario(
        id="E2-NOM1",
        experiment=2,
        category="NOM",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 10.0, "world_y": 3.0, "depth_m": 5.2},
        ],
        failure_context=None,
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Silent NOM: 2 objects in manifest, object 1 mapped, remaining plan correctly covers object 2. "
            "No failure context. Tests false-positive resistance in Exp 2."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E2-NOM2",
        experiment=2,
        category="NOM",
        mission_text="Find objects and orbit each one twice, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 2},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 2},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 4.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 9.0, "world_y": 4.0, "depth_m": 5.1},
        ],
        failure_context=None,
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Silent NOM: 2 objects, mission requires 2 passes each. "
            "Object 1 fully done (2 passes in history); remaining plan has object 2 with repeat:2. "
            "All pass counts match. No failure context."
        ),
        # Mission: "Find objects and orbit each one twice, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E2-NOM4",
        experiment=2,
        category="NOM",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 3, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 3, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 4.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 9.0, "world_y": 3.0, "depth_m": 5.1},
            {"id": 3, "world_x": 14.0, "world_y": 6.0, "depth_m": 5.3},
        ],
        failure_context=None,
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Silent NOM (all done): 3 objects in manifest, all 3 mapped in history (1 pass each), "
            "only RH remains. No failure context. Tests false-positive resistance when mission is fully complete."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),


    # ===========================================================================
    # EXPERIMENT 3 — Compound and Complex Failures  (16 scenarios)
    # ===========================================================================

    # --- 3a: Safety override (SO) — all REPLAN → [return_home] (4) ---

    ReplanScenario(
        id="E3-SO1",
        experiment=3,
        category="SO",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 10.0, "world_y": 3.0, "depth_m": 5.2},
            {"id": 3, "world_x": 15.0, "world_y": 5.0, "depth_m": 5.7},
        ],
        failure_context=(
            "(1) Battery telemetry reports 12% remaining - critically low. "
            "(2) A new object (ID=3) has been detected at approximately 18 m during flight."
        ),
        ground_truth=_tail_json(_rh()),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "SO1: critical battery (12%) + new object discovery. "
            "Battery is the safety override; new object is irrelevant. "
            "Correct tail is immediate return_home regardless of unmapped objects. "
            "Tail: RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-SO2",
        experiment=3,
        category="SO",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 10.0, "world_y": 4.0, "depth_m": 5.1},
        ],
        failure_context=(
            "(1) Battery at 11% - critically low. "
            "(2) Approach to object 2 succeeded; standoff geometry established at 5 m."
        ),
        ground_truth=_tail_json(_rh()),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "SO2: critical battery (11%) + approach success signal. "
            "Even though approach succeeded and map is next, battery forces immediate abort. "
            "Mission-continuation signal must not override safety. Tail: RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-SO3",
        experiment=3,
        category="SO",
        mission_text="Find objects and orbit each one twice, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 2},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 2},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 12.0, "world_y": 5.0, "depth_m": 5.3},
        ],
        failure_context=(
            "(1) Battery at 14% - critically low. "
            "(2) Wind speed 10 m/s causing roll oscillations."
        ),
        ground_truth=_tail_json(_rh()),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "SO3: two simultaneous safety signals (critical battery + wind oscillations). "
            "Both independently require abort; model must correctly identify both as safety overrides. "
            "Tail: RH."
        ),
        # Mission: "Find objects and orbit each one twice, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-SO4",
        experiment=3,
        category="SO",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 1.0, "depth_m": 5.0},
        ],
        failure_context=(
            "(1) Battery at 10% - critically low. "
            "(2) Object 1 visually confirmed at 5 m standoff; high-confidence detection."
        ),
        ground_truth=_tail_json(_rh()),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "SO4: critical battery (10%) + high-confidence detection of object 1. "
            "Tempting to proceed since object is right there, but battery is too low. "
            "Safety must override mission continuation. Tail: RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    # --- 3b: Hard vs. soft (HS) — REPLAN addressing only the hard constraint (3) ---

    ReplanScenario(
        id="E3-HS1",
        experiment=3,
        category="HS",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 10.0, "world_y": 3.0, "depth_m": 5.2},
        ],
        failure_context=(
            "(1) Obstacle avoidance triggered: uncharted obstacle at 8.0 m on the bearing to object 2. "
            "Object 2 is unreachable - standoff geometry cannot be established. "
            "(2) Visual odometry drift detected: positional uncertainty ±0.4 m - within accepted tolerance."
        ),
        ground_truth=_tail_json(_rh()),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "HS1: hard constraint (physically unreachable object 2) + soft warning (VO drift within tolerance). "
            "Correct response: skip object 2 entirely (it's unreachable), return home. "
            "Soft warning must NOT trigger any additional changes. Tail: RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-HS2",
        experiment=3,
        category="HS",
        mission_text="Find objects and orbit each one twice, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
        ],
        failure_context=(
            "(1) Motor controller fault on motor 3 - reduced thrust available; flight stability is degraded. "
            "(2) Wind gusts reaching 3.5 m/s - within operational tolerance."
        ),
        ground_truth=_tail_json(_rh()),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "HS2: hard constraint (motor fault, degraded stability) + soft warning (wind within tolerance). "
            "Motor fault is unrecoverable and requires immediate return. "
            "Wind is irrelevant. Tail: RH."
        ),
        # Mission: "Find objects and orbit each one twice, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-HS3",
        experiment=3,
        category="HS",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 12.0, "world_y": 5.0, "depth_m": 5.3},
        ],
        failure_context=(
            "(1) Object 2 not found - lawnmower search of the expected region yielded no detectable target. "
            "(2) GPS position estimates are experiencing intermittent drift of approximately ±0.2 m."
        ),
        ground_truth=_tail_json(_rh()),
        is_nominal=False,
        allows_partial_credit=False,
        notes=(
            "HS3: hard constraint (object 2 not found / unreachable) + soft warning (GPS position noise ±0.2 m). "
            "Object not present is an unrecoverable hard failure - skip and return. "
            "GPS position noise is irrelevant as the object cannot be approached. Tail: RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    # --- 3c: Efficiency trade-off (ET) — partial mission (5) ---

    ReplanScenario(
        id="E3-ET1",
        experiment=3,
        category="ET",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 3, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 3, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 8.0, "world_y": 2.0, "depth_m": 5.1},
            {"id": 3, "world_x": 52.0, "world_y": 15.0, "depth_m": 5.3},
        ],
        failure_context=(
            "(1) Battery at 20% - low but not critical. "
            "(2) Object 2 is 8 m from current position. Object 3 is 52 m from current position."
        ),
        ground_truth=_tail_json(
            _approach(2, 5.0), _map(2, 5.0),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=True,
        notes=(
            "ET1: battery 20%, object 2 at 8 m (reachable), object 3 at 52 m (energy-prohibitive). "
            "Correct trade-off: map object 2 (nearby), skip object 3 (too far), return home. "
            "Tail: A(2,5) → M(2,5) → RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-ET2",
        experiment=3,
        category="ET",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 3, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 3, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 9.0, "world_y": 3.0, "depth_m": 5.1},
            {"id": 3, "world_x": 55.0, "world_y": 20.0, "depth_m": 5.5},
        ],
        failure_context=(
            "(1) Battery at 28% - low. "
            "(2) Object 1 is 5 m from current position. "
            "Object 2 is 9 m from current position. "
            "Object 3 is 55 m from current position."
        ),
        ground_truth=_tail_json(
            _approach(1, 5.0), _map(1, 5.0),
            _approach(2, 5.0), _map(2, 5.0),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=True,
        notes=(
            "ET2: battery 28%, objects 1 (5 m) and 2 (9 m) reachable, object 3 (55 m) prohibitive. "
            "Map objects 1 and 2, skip 3. "
            "Tail: A(1,5)→M(1,5)→A(2,5)→M(2,5)→RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-ET3",
        experiment=3,
        category="ET",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 3, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 3, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 40.0, "world_y": 12.0, "depth_m": 5.2},
            {"id": 3, "world_x": 8.0, "world_y": 2.0, "depth_m": 5.0},
        ],
        failure_context=(
            "(1) Battery at 22% - low. "
            "(2) Object 2 is 40 m from current position. "
            "Object 3 is 8 m from current position."
        ),
        ground_truth=_tail_json(
            _approach(3, 5.0), _map(3, 5.0),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=True,
        notes=(
            "ET3: battery 22%, object 1 already done, object 2 at 40 m (skip), object 3 at 8 m (reachable). "
            "Non-trivial order: remaining plan has object 2 before 3, but object 2 should be skipped. "
            "Tail: A(3,5)→M(3,5)→RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-ET4",
        experiment=3,
        category="ET",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 3, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 3, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 8.0, "world_y": 2.0, "depth_m": 5.0},
            {"id": 3, "world_x": 60.0, "world_y": 18.0, "depth_m": 5.4},
        ],
        failure_context=(
            "(1) Battery at 30% - low."
            "(2) Approach to object 2 failed - attempted 1 time(s)."
            "(3) Object 2 is 8 m from current position. "
            "Object 3 is 60 m from current position."
        ),
        ground_truth=_tail_json(
            _approach(2, 5.0), _map(2, 5.0),
            _rh(),
        ),
        is_nominal=False,
        allows_partial_credit=True,
        notes=(
            "ET4: battery 30%, approach to obj 2 failed, object 2 at 8 m (map it), object 3 at 60 m (skip). "
            "Only one nearby object to map; must not attempt the far one at low battery. Should retry close object."
            "Tail: A(2,5)→M(2,5)→RH."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    # --- 3d: NOMINAL controls (4) ---

    ReplanScenario(
        id="E3-NOM1",
        experiment=3,
        category="NOM",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 10.0, "world_y": 3.0, "depth_m": 5.1},
        ],
        failure_context=(
            "Battery at 55%."
        ),
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Exp3 NOM1: healthy battery (55%) + mission proceeding normally. "
            "Remaining plan correctly covers object 2. Soft positive signal must not trigger replan."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-NOM2",
        experiment=3,
        category="NOM",
        mission_text="Find objects and orbit each one once at 5 meter standoff, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "lawnmower"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 4.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 9.0, "world_y": 3.0, "depth_m": 5.1},
        ],
        failure_context=(
            "Wind speed 2.0 m/s - within operational tolerance."
        ),
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Exp3 NOM2: wind 2.0 m/s  (within tolerance) + one object remaining to map. "
            "Soft environmental signal must not trigger replan. Remaining plan is valid."
        ),
        # Mission: "Find objects and orbit each one once at 5 meter standoff, then return home."
        # Standoff explicitly stated as 5 meters. No altitude, no pattern named.
        altitude_explicit=False,
        standoff_explicit=True,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-NOM3",
        experiment=3,
        category="NOM",
        mission_text="Search for objects and map every one you find with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "approach", "args": {"object_id": 2, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 2, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 10.0, "world_y": 3.0, "depth_m": 5.1},
        ],
        failure_context=(
            "(1) Low-amplitude IMU angular noise detected (0.02 rad) - within normal operating bounds. "
            "(2) Battery at 60% - healthy."
        ),
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Exp3 NOM3: two simultaneous soft signals (IMU jitter within tolerance + 60% battery). "
            "Neither signal warrants a replan. Tests that compound soft signals do not cause false positives."
        ),
        # Mission: "Search for objects and map every one you find with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),

    ReplanScenario(
        id="E3-NOM4",
        experiment=3,
        category="NOM",
        mission_text="Search for the first object you find and map it with one orbit, then return home.",
        completed_steps=[
            {"state": "takeoff", "args": {"altitude": 5.0}, "repeat": 1},
            {"state": "search", "args": {"pattern": "yaw_scan"}, "repeat": 1},
            {"state": "approach", "args": {"object_id": 1, "standoff_distance": 5.0}, "repeat": 1},
            {"state": "map", "args": {"object_id": 1, "standoff_distance": 5.0, "mode": "orbit"}, "repeat": 1},
        ],
        remaining_steps=[
            {"state": "return_home", "args": {}, "repeat": 1},
        ],
        objects=[
            {"id": 1, "world_x": 5.0, "world_y": 0.0, "depth_m": 5.0},
            {"id": 2, "world_x": 12.0, "world_y": 5.0, "depth_m": 5.3},
        ],
        failure_context=(
            "A new object (object 2) has been detected at approximately 12 m."
        ),
        ground_truth="NOMINAL",
        is_nominal=True,
        allows_partial_credit=False,
        notes=(
            "Exp3 NOM4: new object detected, but mission explicitly says 'first object only'. "
            "Object 1 mapped; remaining plan is just RH which is correct for this mission scope. "
            "Model must resist extending the plan for a newly detected object when the mission is scoped. "
            "Correct answer is NOMINAL."
        ),
        # Mission: "Search for the first object you find and map it with one orbit, then return home."
        # No altitude, no standoff, no pattern named.
        altitude_explicit=False,
        standoff_explicit=False,
        search_explicit=False,
    ),
]


# ---------------------------------------------------------------------------
# Sanity check counts
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Planning scenarios:   {len(PLANNING_SCENARIOS)}")
    exp_counts = {}
    for s in REPLAN_SCENARIOS:
        exp_counts[s.experiment] = exp_counts.get(s.experiment, 0) + 1
    for exp, count in sorted(exp_counts.items()):
        print(f"  Experiment {exp} replanning scenarios: {count}")
    print(f"Total replanning scenarios: {len(REPLAN_SCENARIOS)}")

    # Verify is_nominal consistency
    for s in REPLAN_SCENARIOS:
        assert s.is_nominal == (s.ground_truth == "NOMINAL"), \
            f"{s.id}: is_nominal={s.is_nominal} but ground_truth={s.ground_truth!r}"

    # Verify allows_partial_credit only on ET1-ET4
    for s in REPLAN_SCENARIOS:
        if s.allows_partial_credit:
            assert s.category == "ET" and s.id != "E3-ET5", \
                f"{s.id}: allows_partial_credit=True but not ET1-ET4"

    # Verify remaining_steps end with return_home
    for s in REPLAN_SCENARIOS:
        if s.remaining_steps:
            assert s.remaining_steps[-1]["state"] == "return_home", \
                f"{s.id}: remaining_steps does not end with return_home"

    # Verify no takeoff in remaining_steps
    for s in REPLAN_SCENARIOS:
        for step in s.remaining_steps:
            assert step["state"] != "takeoff", \
                f"{s.id}: takeoff found in remaining_steps"

    # Verify failure_context=None for all Exp 2
    for s in REPLAN_SCENARIOS:
        if s.experiment == 2:
            assert s.failure_context is None, \
                f"{s.id}: Exp 2 scenario has non-None failure_context"

    # Verify standoff_explicit=True only where mission text names a distance
    standoff_explicit_ids = {s.id for s in REPLAN_SCENARIOS if s.standoff_explicit}
    expected_standoff_explicit = {"E1-SW2", "E2-LC3", "E3-NOM2"}
    assert standoff_explicit_ids == expected_standoff_explicit, (
        f"standoff_explicit mismatch.\n"
        f"  Got:      {sorted(standoff_explicit_ids)}\n"
        f"  Expected: {sorted(expected_standoff_explicit)}"
    )

    # Verify search_explicit=False for all replan scenarios
    assert all(not s.search_explicit for s in REPLAN_SCENARIOS), \
        "Some replan scenario has search_explicit=True — no replan mission names a pattern."

    # Verify altitude_explicit=False for all replan scenarios
    assert all(not s.altitude_explicit for s in REPLAN_SCENARIOS), \
        "Some replan scenario has altitude_explicit=True — no replan mission names an altitude."

    print("All internal consistency checks passed.")