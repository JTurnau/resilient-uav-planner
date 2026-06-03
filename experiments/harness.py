"""
harness.py
----------
Evaluation harness for UAV planning and replanning experiments.
 
Usage:
    python harness.py             # run_id=1 (default)
    RUN_ID=2 python harness.py    # run_id=2
    RUN_ID=3 python harness.py    # run_id=3
 
Output files (run_id is embedded in every JSONL record):
    results_planning_raw.jsonl          ← appended across all runs
    results_replan_raw.jsonl            ← appended across all runs
    results_planning_summary_runN.json
    results_planning_raw_runN.csv
    results_planning_summary_runN.csv
    results_replan_summary_runN.json
    results_replan_raw_runN.csv
    results_replan_summary_runN.csv
 
Models
------
  gemini-2.5-flash-thinking   google/gemini-2.5-flash via OpenRouter,
                              reasoning effort "medium".
  gemini-2.5-flash-base       google/gemini-2.5-flash via OpenRouter,
                              reasoning disabled — pure instruction-following
                              baseline.
  qwen-235b-thinking          qwen/qwen3-235b-a22b-thinking-2507 via
                              OpenRouter, reasoning effort "medium".
  qwen-235b-instruct          qwen/qwen3-235b-a22b-2507 via OpenRouter,
                              no reasoning parameter.
  deepseek-r1                 deepseek/deepseek-r1 via OpenRouter,
                              reasoning effort "medium".
  o4-mini                     o4-mini via the OpenAI API,
                              reasoning_effort="medium". Temperature omitted
                              (not supported by the o-series API).
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from typing import Any

from experiment_utils import (
    query_llm, MODELS, PLANNING_PROMPT, MIN_STANDOFF_M,
    generate_plan, run_replan, build_replan_system_prompt,
    validate_plan, parse_replan_response, extract_json,
    MockObject, MockWorldState, make_world,
    fmt_plan, save_json,
    _is_rate_limit_error,
)
from scenarios import (
    PLANNING_SCENARIOS, REPLAN_SCENARIOS,
    PlanningScenario, ReplanScenario,
)

# ---------------------------------------------------------------------------
# RUN ID — increment this (or set env var RUN_ID) for each experimental run.
# Records from all runs can be appended to the same JSONL and separated later.
# ---------------------------------------------------------------------------

RUN_ID: int = int(os.environ.get("RUN_ID", "1"))

# ---------------------------------------------------------------------------
# Model list — all six models evaluated each run
# ---------------------------------------------------------------------------

ALL_MODELS = [
    "gemini-2.5-flash-thinking",
    "gemini-2.5-flash-base",
    "qwen-235b-thinking",
    "qwen-235b-instruct",
    "deepseek-r1",
    "o4-mini",
]

# ---------------------------------------------------------------------------
# Step matching
# ---------------------------------------------------------------------------

_ALT_FUZZY_MIN = 2.0
_ALT_FUZZY_MAX = 8.0


def _steps_match(
    a: dict,
    b: dict,
    altitude_explicit: bool = True,
    standoff_explicit: bool = True,
    search_explicit: bool = True,
) -> bool:
    if a.get("state") != b.get("state"):
        return False
    if a.get("repeat", 1) != b.get("repeat", 1):
        return False

    args_a = a.get("args", {})
    args_b = b.get("args", {})

    if set(args_a.keys()) != set(args_b.keys()):
        return False

    for k in args_a:
        va, vb = args_a[k], args_b[k]

        if k == "altitude":
            if altitude_explicit:
                try:
                    if abs(float(va) - float(vb)) > 0.01:
                        return False
                except (TypeError, ValueError):
                    if va != vb:
                        return False
            else:
                try:
                    if not (_ALT_FUZZY_MIN <= float(vb) <= _ALT_FUZZY_MAX):
                        return False
                except (TypeError, ValueError):
                    return False

        elif k == "standoff_distance":
            if standoff_explicit:
                try:
                    if abs(float(va) - float(vb)) > 0.01:
                        return False
                except (TypeError, ValueError):
                    if va != vb:
                        return False
            else:
                try:
                    if float(vb) < MIN_STANDOFF_M:
                        return False
                except (TypeError, ValueError):
                    return False

        elif k == "pattern":
            if search_explicit:
                if va != vb:
                    return False
            else:
                valid_patterns = {"yaw_scan", "lawnmower"}
                if vb not in valid_patterns:
                    return False

        else:
            if va != vb:
                return False

    return True


def _tails_match(
    plan_a: list,
    plan_b: list,
    altitude_explicit: bool = True,
    standoff_explicit: bool = True,
    search_explicit: bool = True,
) -> bool:
    if len(plan_a) != len(plan_b):
        return False
    return all(
        _steps_match(
            a, b,
            altitude_explicit=altitude_explicit,
            standoff_explicit=standoff_explicit,
            search_explicit=search_explicit,
        )
        for a, b in zip(plan_a, plan_b)
    )


# ---------------------------------------------------------------------------
# make_world_from_scenario
# ---------------------------------------------------------------------------

def _parse_battery_from_failure_context(failure_context: str | None) -> float:
    if not failure_context:
        return 100.0
    match = re.search(
        r'battery[^%\d]*?(\d+(?:\.\d+)?)\s*%',
        failure_context,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', failure_context)
    if match:
        return float(match.group(1))
    return 100.0


def make_world_from_scenario(scenario: ReplanScenario) -> MockWorldState:
    objects = [
        MockObject(
            id=o["id"],
            world_x=float(o["world_x"]),
            world_y=float(o["world_y"]),
            depth_m=float(o["depth_m"]),
        )
        for o in scenario.objects
    ]
    battery_pct = _parse_battery_from_failure_context(scenario.failure_context)
    return MockWorldState(
        objects=objects,
        completed_steps=scenario.completed_steps,
        remaining_steps=scenario.remaining_steps,
        failure_context=scenario.failure_context,
        battery_pct=battery_pct,
    )

# ---------------------------------------------------------------------------
# Approximate token counting
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.35))

# ---------------------------------------------------------------------------
# compute_planning_metrics
# ---------------------------------------------------------------------------

def compute_planning_metrics(
    scenario: PlanningScenario,
    model_key: str,
    result: dict,
) -> dict:
    validity_rate = result.get("success", False)
    plan = result.get("plan") or []

    tail_match = False
    if validity_rate and plan:
        tail_match = _tails_match(
            plan,
            scenario.reference_plan,
            altitude_explicit=scenario.altitude_explicit,
            standoff_explicit=scenario.standoff_explicit,
            search_explicit=scenario.search_explicit,
        )

    step_correct: list[bool] = []
    if validity_rate and plan and len(plan) == len(scenario.reference_plan):
        for a, b in zip(plan, scenario.reference_plan):
            step_correct.append(
                _steps_match(
                    a, b,
                    altitude_explicit=scenario.altitude_explicit,
                    standoff_explicit=scenario.standoff_explicit,
                    search_explicit=scenario.search_explicit,
                )
            )

    raw_responses = result.get("raw_responses", [])
    last_raw = raw_responses[-1] if raw_responses else ""

    approx_output_tokens = _approx_tokens(last_raw)
    approx_prompt_tokens = _approx_tokens(PLANNING_PROMPT + "\n" + scenario.mission_text)

    return {
        "run_id":                result.get("run_id", RUN_ID),
        "scenario_id":           scenario.id,
        "level":                 scenario.level,
        "model":                 model_key,
        "mission_text":          scenario.mission_text,
        "validity_rate":         validity_rate,
        "tail_match":            tail_match,
        "mission_success":       validity_rate and tail_match,
        "attempts":              result.get("attempts", 0),
        "needed_retry":          result.get("attempts", 1) > 1,
        "latency_s":             round(result.get("latency_s", 0.0), 3),
        "approx_prompt_tokens":  approx_prompt_tokens,
        "approx_output_tokens":  approx_output_tokens,
        "altitude_explicit":     scenario.altitude_explicit,
        "standoff_explicit":     scenario.standoff_explicit,
        "plan_length":           len(plan),
        "reference_length":      len(scenario.reference_plan),
        "length_match":          len(plan) == len(scenario.reference_plan),
        "step_correct":          step_correct,
        "steps_correct_count":   sum(step_correct),
        "errors":                result.get("errors", []),
        "raw_responses":         raw_responses,
        "plan":                  plan,
        "reference_plan":        scenario.reference_plan,
    }

# ---------------------------------------------------------------------------
# compute_replan_metrics
# ---------------------------------------------------------------------------

def compute_replan_metrics(
    scenario: ReplanScenario,
    model_key: str,
    result: dict,
) -> dict:
    gt_is_nominal = scenario.is_nominal
    model_nominal = result.get("nominal", False)
    decision_correct = (model_nominal == gt_is_nominal)

    is_tp = (not gt_is_nominal) and (not model_nominal)
    is_tn = gt_is_nominal and model_nominal
    is_fp = gt_is_nominal and (not model_nominal)
    is_fn = (not gt_is_nominal) and model_nominal

    constraint_valid = None
    tail_match = None
    partial_credit_score = None
    model_tail = None

    if not gt_is_nominal:
        try:
            reference_tail = json.loads(scenario.ground_truth)
        except Exception:
            reference_tail = []

        if not model_nominal:
            model_tail = result.get("tail") or []
            ok, _ = validate_plan(model_tail, is_tail=True)
            constraint_valid = ok

            if ok:
                tail_match = _tails_match(
                    model_tail, reference_tail,
                    altitude_explicit=True,
                    standoff_explicit=True,
                )
            else:
                tail_match = False

            if scenario.allows_partial_credit and reference_tail:
                common = sum(
                    1 for ref_step in reference_tail
                    if any(_steps_match(ref_step, m, altitude_explicit=True, standoff_explicit=True)
                           for m in model_tail)
                )
                partial_credit_score = round(common / len(reference_tail), 4)
        else:
            constraint_valid = False
            tail_match = False
            if scenario.allows_partial_credit:
                partial_credit_score = 0.0

    if gt_is_nominal:
        mission_success = decision_correct
    else:
        mission_success = decision_correct and bool(tail_match)

    raw_responses = result.get("raw_responses", [])
    last_raw = raw_responses[-1] if raw_responses else ""

    system_prompt_text = build_replan_system_prompt(
        mission_intent=scenario.mission_text,
        completed_steps=scenario.completed_steps,
        remaining_steps=scenario.remaining_steps,
        objects=[
            MockObject(id=o["id"], world_x=o["world_x"], world_y=o["world_y"], depth_m=o["depth_m"])
            for o in scenario.objects
        ],
        failure_context=scenario.failure_context,
    )
    approx_prompt_tokens = _approx_tokens(system_prompt_text)
    approx_output_tokens = _approx_tokens(last_raw)

    try:
        gt_tail = json.loads(scenario.ground_truth) if not scenario.is_nominal else None
    except Exception:
        gt_tail = None

    return {
        "run_id":                result.get("run_id", RUN_ID),
        "scenario_id":           scenario.id,
        "experiment":            scenario.experiment,
        "category":              scenario.category,
        "model":                 model_key,
        "mission_text":          scenario.mission_text,
        "is_nominal_gt":         gt_is_nominal,
        "ground_truth_tail":     gt_tail,
        "decision_correct":      decision_correct,
        "is_tp":                 is_tp,
        "is_tn":                 is_tn,
        "is_fp":                 is_fp,
        "is_fn":                 is_fn,
        "constraint_valid":      constraint_valid,
        "tail_match":            tail_match,
        "mission_success":       mission_success,
        "allows_partial_credit": scenario.allows_partial_credit,
        "partial_credit_score":  partial_credit_score,
        "attempts":              result.get("attempts", 0),
        "needed_retry":          result.get("attempts", 1) > 1,
        "latency_s":             round(result.get("latency_s", 0.0), 3),
        "approx_prompt_tokens":  approx_prompt_tokens,
        "approx_output_tokens":  approx_output_tokens,
        "model_said_nominal":    model_nominal,
        "model_tail":            model_tail,
        "reason":                result.get("reason"),
        "errors":                result.get("errors", []),
        "raw_response":          last_raw,
        "all_raw_responses":     raw_responses,
    }

# ---------------------------------------------------------------------------
# aggregate_planning_results
# ---------------------------------------------------------------------------

def aggregate_planning_results(records: list[dict]) -> dict:
    def _pct(num, den):
        return round(100.0 * num / den, 1) if den > 0 else 0.0

    def _mean(vals):
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    models = list(dict.fromkeys(r["model"] for r in records))
    levels = list(dict.fromkeys(r["level"] for r in records))

    summary = {}
    for model in models:
        model_records = [r for r in records if r["model"] == model]
        summary[model] = {
            "overall": {
                "n":                    len(model_records),
                "validity_rate":        _pct(sum(r["validity_rate"] for r in model_records), len(model_records)),
                "tail_match_rate":      _pct(sum(r["tail_match"] for r in model_records), len(model_records)),
                "mission_success_rate": _pct(sum(r["mission_success"] for r in model_records), len(model_records)),
                "mean_latency_s":       _mean([r["latency_s"] for r in model_records]),
                "mean_attempts":        _mean([r["attempts"] for r in model_records]),
                "retry_rate":           _pct(sum(r["needed_retry"] for r in model_records), len(model_records)),
                "mean_output_tokens":   _mean([r["approx_output_tokens"] for r in model_records]),
            },
        }
        for level in levels:
            lvl_records = [r for r in model_records if r["level"] == level]
            if not lvl_records:
                continue
            summary[model][level] = {
                "n":                    len(lvl_records),
                "validity_rate":        _pct(sum(r["validity_rate"] for r in lvl_records), len(lvl_records)),
                "tail_match_rate":      _pct(sum(r["tail_match"] for r in lvl_records), len(lvl_records)),
                "mission_success_rate": _pct(sum(r["mission_success"] for r in lvl_records), len(lvl_records)),
                "mean_latency_s":       _mean([r["latency_s"] for r in lvl_records]),
                "mean_attempts":        _mean([r["attempts"] for r in lvl_records]),
                "retry_rate":           _pct(sum(r["needed_retry"] for r in lvl_records), len(lvl_records)),
                "mean_output_tokens":   _mean([r["approx_output_tokens"] for r in lvl_records]),
            }

    return summary

# ---------------------------------------------------------------------------
# aggregate_replan_results
# ---------------------------------------------------------------------------

def aggregate_replan_results(records: list[dict]) -> dict:
    def _pct(num, den):
        return round(100.0 * num / den, 1) if den > 0 else 0.0

    def _mean(vals):
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    models = list(dict.fromkeys(r["model"] for r in records))
    experiments = sorted(set(r["experiment"] for r in records))

    summary = {}
    for model in models:
        model_records = [r for r in records if r["model"] == model]
        summary[model] = {}

        def _agg_block(recs):
            n = len(recs)
            tp = sum(r["is_tp"] for r in recs)
            tn = sum(r["is_tn"] for r in recs)
            fp = sum(r["is_fp"] for r in recs)
            fn = sum(r["is_fn"] for r in recs)
            replan_gt = [r for r in recs if not r["is_nominal_gt"]]
            tail_matches = [r for r in replan_gt if r.get("tail_match") is True]
            cv_recs = [r for r in replan_gt if r.get("constraint_valid") is True]
            ms = sum(r["mission_success"] for r in recs)
            pc_scores = [r["partial_credit_score"] for r in recs if r.get("partial_credit_score") is not None]
            return {
                "n":                       n,
                "decision_accuracy":       _pct(tp + tn, n),
                "FPR":                     _pct(fp, fp + tn) if (fp + tn) > 0 else 0.0,
                "FNR":                     _pct(fn, fn + tp) if (fn + tp) > 0 else 0.0,
                "constraint_valid_rate":   _pct(len(cv_recs), len(replan_gt)) if replan_gt else None,
                "tail_match_rate":         _pct(len(tail_matches), len(replan_gt)) if replan_gt else None,
                "mission_success_rate":    _pct(ms, n),
                "avg_partial_credit":      round(sum(pc_scores) / len(pc_scores), 4) if pc_scores else None,
                "mean_latency_s":          _mean([r["latency_s"] for r in recs]),
                "mean_attempts":           _mean([r["attempts"] for r in recs]),
                "retry_rate":              _pct(sum(r["needed_retry"] for r in recs), n),
                "mean_output_tokens":      _mean([r["approx_output_tokens"] for r in recs]),
                "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                "n_replan_gt":             len(replan_gt),
            }

        summary[model]["overall"] = _agg_block(model_records)

        for exp in experiments:
            exp_records = [r for r in model_records if r["experiment"] == exp]
            if not exp_records:
                continue
            exp_key = f"Exp{exp}"
            summary[model][exp_key] = _agg_block(exp_records)

            categories = list(dict.fromkeys(r["category"] for r in exp_records))
            for cat in categories:
                cat_records = [r for r in exp_records if r["category"] == cat]
                summary[model][exp_key][cat] = _agg_block(cat_records)

    return summary

# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def _write_planning_csv(records: list[dict], path: str) -> None:
    if not records:
        return
    flat_keys = [
        "run_id",
        "scenario_id", "level", "model", "mission_text",
        "validity_rate", "tail_match", "mission_success",
        "attempts", "needed_retry",
        "latency_s", "approx_prompt_tokens", "approx_output_tokens",
        "altitude_explicit", "standoff_explicit",
        "plan_length", "reference_length", "length_match",
        "steps_correct_count",
        "errors",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in flat_keys}
            if isinstance(row["errors"], list):
                row["errors"] = json.dumps(row["errors"])
            writer.writerow(row)
    print(f"  Saved: {path}")


def _write_replan_csv(records: list[dict], path: str) -> None:
    if not records:
        return
    flat_keys = [
        "run_id",
        "scenario_id", "experiment", "category", "model", "mission_text",
        "is_nominal_gt",
        "decision_correct", "is_tp", "is_tn", "is_fp", "is_fn",
        "constraint_valid", "tail_match", "mission_success",
        "allows_partial_credit", "partial_credit_score",
        "model_said_nominal", "reason",
        "attempts", "needed_retry",
        "latency_s", "approx_prompt_tokens", "approx_output_tokens",
        "errors",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in flat_keys}
            if isinstance(row["errors"], list):
                row["errors"] = json.dumps(row["errors"])
            writer.writerow(row)
    print(f"  Saved: {path}")


def _write_summary_csv(summary: dict, path: str, kind: str) -> None:
    rows = []
    for model, model_data in summary.items():
        for scope_key, metrics in model_data.items():
            if not isinstance(metrics, dict):
                continue
            row = {"model": model, "scope": scope_key}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    continue
                row[k] = v
            rows.append(row)

    if not rows:
        return

    all_keys: list[str] = ["model", "scope"]
    seen = set(all_keys)
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    print(f"  Saved: {path}")

# ---------------------------------------------------------------------------
# run_planning_experiments
# ---------------------------------------------------------------------------

def run_planning_experiments() -> list[dict]:
    records = []
    total = len(PLANNING_SCENARIOS) * len(ALL_MODELS)
    done = 0
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 0: Initial Plan Generation  [run_id={RUN_ID}]")
    print(f"  {len(PLANNING_SCENARIOS)} scenarios × {len(ALL_MODELS)} models = {total} runs")
    print(f"{'='*60}")

    for scenario in PLANNING_SCENARIOS:
        for model_key in ALL_MODELS:
            done += 1
            print(f"  [{done}/{total}] {scenario.id} | {model_key} ...", end=" ", flush=True)
            try:
                result = generate_plan(
                    scenario.mission_text,
                    model_key=model_key,
                    run_id=RUN_ID,
                )
                metrics = compute_planning_metrics(scenario, model_key, result)
                status = "✓" if metrics["mission_success"] else ("V" if metrics["validity_rate"] else "✗")
                print(f"{status}  ({metrics['latency_s']:.1f}s, {metrics['attempts']} attempt(s))")
            except Exception as exc:
                print(f"ERROR: {exc}")
                metrics = {
                    "run_id": RUN_ID,
                    "scenario_id": scenario.id, "level": scenario.level,
                    "model": model_key, "mission_text": scenario.mission_text,
                    "validity_rate": False, "tail_match": False,
                    "mission_success": False,
                    "attempts": 0, "needed_retry": False,
                    "latency_s": 0.0,
                    "approx_prompt_tokens": 0, "approx_output_tokens": 0,
                    "altitude_explicit": scenario.altitude_explicit,
                    "standoff_explicit": scenario.standoff_explicit,
                    "plan_length": 0, "reference_length": len(scenario.reference_plan),
                    "length_match": False, "step_correct": [], "steps_correct_count": 0,
                    "errors": [str(exc)], "raw_responses": [],
                    "plan": None, "reference_plan": scenario.reference_plan,
                }
            records.append(metrics)

    return records

# ---------------------------------------------------------------------------
# run_replan_experiments
# ---------------------------------------------------------------------------

def run_replan_experiments() -> list[dict]:
    records = []
    total = len(REPLAN_SCENARIOS) * len(ALL_MODELS)
    done = 0
    print(f"\n{'='*60}")
    print(f"EXPERIMENTS 1–3: Replanning  [run_id={RUN_ID}]")
    print(f"  {len(REPLAN_SCENARIOS)} scenarios × {len(ALL_MODELS)} models = {total} runs")
    print(f"{'='*60}")

    for scenario in REPLAN_SCENARIOS:
        for model_key in ALL_MODELS:
            done += 1
            print(f"  [{done}/{total}] {scenario.id} | {model_key} ...", end=" ", flush=True)
            try:
                world = make_world_from_scenario(scenario)
                result = run_replan(
                    mission_intent=scenario.mission_text,
                    world=world,
                    model_key=model_key,
                    run_id=RUN_ID,
                )
                metrics = compute_replan_metrics(scenario, model_key, result)
                status = "✓" if metrics["mission_success"] else (
                    "FP" if metrics["is_fp"] else (
                    "FN" if metrics["is_fn"] else (
                    "CV" if metrics["is_tp"] and metrics.get("constraint_valid") and not metrics.get("tail_match") else "✗"
                )))
                print(f"{status}  ({metrics['latency_s']:.1f}s, {metrics['attempts']} attempt(s))")
            except Exception as exc:
                print(f"ERROR: {exc}")
                metrics = {
                    "run_id": RUN_ID,
                    "scenario_id": scenario.id, "experiment": scenario.experiment,
                    "category": scenario.category, "model": model_key,
                    "mission_text": scenario.mission_text,
                    "is_nominal_gt": scenario.is_nominal,
                    "ground_truth_tail": None,
                    "decision_correct": False, "is_tp": False, "is_tn": False,
                    "is_fp": False, "is_fn": False,
                    "constraint_valid": None, "tail_match": None,
                    "mission_success": False,
                    "allows_partial_credit": scenario.allows_partial_credit,
                    "partial_credit_score": None,
                    "attempts": 0, "needed_retry": False,
                    "latency_s": 0.0,
                    "approx_prompt_tokens": 0, "approx_output_tokens": 0,
                    "model_said_nominal": False,
                    "model_tail": None, "reason": None,
                    "errors": [str(exc)],
                    "raw_response": "", "all_raw_responses": [],
                }
            records.append(metrics)

    return records

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def _print_summary_table(planning_summary: dict, replan_summary: dict) -> None:
    col_w = 28

    print(f"\n{'='*103}")
    print("PLANNING SUMMARY")
    print(f"{'='*103}")
    header = (
        f"{'Model':<{col_w}} | {'Scope':<8} | {'n':>3} | {'Valid%':>7} | "
        f"{'TailMatch%':>10} | {'MissionOK%':>10} | {'AvgLat(s)':>9} | {'Retries%':>8}"
    )
    print(header)
    print("-" * len(header))
    for model, data in planning_summary.items():
        for scope_key in ["L1", "L2", "L3", "overall"]:
            if scope_key not in data:
                continue
            d = data[scope_key]
            label = scope_key if scope_key != "overall" else "All"
            print(
                f"{model:<{col_w}} | {label:<8} | {d['n']:>3} | "
                f"{d['validity_rate']:>7.1f} | {d['tail_match_rate']:>10.1f} | "
                f"{d['mission_success_rate']:>10.1f} | {d['mean_latency_s']:>9.2f} | "
                f"{d['retry_rate']:>8.1f}"
            )
        print()

    print(f"\n{'='*118}")
    print("REPLANNING SUMMARY")
    print(f"{'='*118}")
    header2 = (
        f"{'Model':<{col_w}} | {'Scope':<6} | {'n':>3} | {'DecAcc':>7} | "
        f"{'FPR':>6} | {'FNR':>6} | {'ConstrV%':>8} | {'TailMatch%':>10} | "
        f"{'MissionOK%':>10} | {'AvgLat(s)':>9}"
    )
    print(header2)
    print("-" * len(header2))
    for model, data in replan_summary.items():
        for scope_key in ["Exp1", "Exp2", "Exp3", "overall"]:
            if scope_key not in data:
                continue
            d = data[scope_key]
            label = scope_key if scope_key != "overall" else "All"
            cv = f"{d['constraint_valid_rate']:>8.1f}" if d['constraint_valid_rate'] is not None else "     N/A"
            tm = f"{d['tail_match_rate']:>10.1f}" if d['tail_match_rate'] is not None else "       N/A"
            print(
                f"{model:<{col_w}} | {label:<6} | {d['n']:>3} | "
                f"{d['decision_accuracy']:>7.1f} | {d['FPR']:>6.1f} | {d['FNR']:>6.1f} | "
                f"{cv} | {tm} | "
                f"{d['mission_success_rate']:>10.1f} | {d['mean_latency_s']:>9.2f}"
            )
        print()

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("UAV Planning & Replanning Evaluation Harness")
    print(f"Run ID: {RUN_ID}  (set env var RUN_ID=N to change)")
    print(f"Models: {ALL_MODELS}")
    print(f"Planning scenarios: {len(PLANNING_SCENARIOS)}")
    print(f"Replanning scenarios: {len(REPLAN_SCENARIOS)}")

    planning_raw = run_planning_experiments()

    with open("results_planning_raw.jsonl", "a") as f:
        for rec in planning_raw:
            f.write(json.dumps(rec) + "\n")
    print("\n  Appended: results_planning_raw.jsonl")

    planning_summary = aggregate_planning_results(planning_raw)
    save_json(f"results_planning_summary_run{RUN_ID}.json", planning_summary)
    _write_planning_csv(planning_raw, f"results_planning_raw_run{RUN_ID}.csv")
    _write_summary_csv(planning_summary, f"results_planning_summary_run{RUN_ID}.csv", kind="planning")

    replan_raw = run_replan_experiments()

    with open("results_replan_raw.jsonl", "a") as f:
        for rec in replan_raw:
            f.write(json.dumps(rec) + "\n")
    print("\n  Appended: results_replan_raw.jsonl")

    replan_summary = aggregate_replan_results(replan_raw)
    save_json(f"results_replan_summary_run{RUN_ID}.json", replan_summary)
    _write_replan_csv(replan_raw, f"results_replan_raw_run{RUN_ID}.csv")
    _write_summary_csv(replan_summary, f"results_replan_summary_run{RUN_ID}.csv", kind="replan")

    _print_summary_table(planning_summary, replan_summary)


if __name__ == "__main__":
    main()