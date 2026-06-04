# resilient-uav-planner

**Language-Guided Adaptive Mission Replanning for Resilient Autonomous UAVs in Unknown Environments**

A framework in which a large language model (LLM) serves as both the initial mission planner and the dynamic replanner for an autonomous UAV, invoked at every skill boundary and event trigger over an accumulated context of execution history and discovered object observations. Plans are expressed as sequences of parameterized skill invocations from a fixed vocabulary, making them directly executable on a PX4/ROS2 flight stack without intermediate compilation.

> 📄 Paper: *coming soon*

---

## System Overview

<!-- TODO: Insert system architecture diagram here -->
> **[Placeholder]** System overview diagram coming soon — will illustrate the planning loop, execution context accumulation, replanning trigger flow, and PX4/ROS2 integration.

---

## Skill Vocabulary

Plans are expressed as JSON arrays of parameterized skill invocations. Each step takes the form:

```json
{ "state": "<skill_name>", "args": { ... }, "repeat": 1 }
```

`"repeat": N` runs that skill to completion N times before advancing.

| Skill | Args | Notes |
|---|---|---|
| `takeoff` | `altitude: float (m)` | Always the first step. Infer altitude from context: confined space → 2–3 m, open area → 5–8 m. |
| `search` | `pattern: "yaw_scan" \| "lawnmower"` | Discovers objects via onboard perception. `yaw_scan` rotates in place (quick sweeps); `lawnmower` does systematic grid coverage (large areas). |
| `approach` | `object_id: int \| "all"`, `standoff_distance: float (m, min 5.0)` | Fly to standoff distance from target(s). Must appear immediately before `map` for the same `object_id`. |
| `map` | `object_id: int \| "all"`, `standoff_distance: float (m, ≥ 5.0)`, `mode: "orbit"` | One full orbit per invocation. Use `repeat: N` to orbit N times. A preceding `approach` for the same `object_id` is required. |
| `return_home` | *(no args)* | Always the last step. |

**Example plan** — *"Search for objects, orbit each one twice, then return home."*

```json
[
  { "state": "takeoff",     "args": { "altitude": 6.0 } },
  { "state": "search",      "args": { "pattern": "yaw_scan" } },
  { "state": "approach",    "args": { "object_id": "all", "standoff_distance": 5.0 } },
  { "state": "map",         "args": { "mode": "orbit", "object_id": "all", "standoff_distance": 5.0 }, "repeat": 2 },
  { "state": "return_home", "args": {} }
]
```

---

## Offline Evaluation Results

Six models evaluated across three replanning experiment types (32 scenarios total). Results reported as overall across all experiments.

| Model | Decision Acc. (%) ↑ | FPR (%) ↓ | FNR (%) ↓ | Tail Match (%) ↑ | Mission Success (%) ↑ | Avg Latency (s) ↓ |
|----|----:|----:|----:|----:|----:|----:|
| Gemini 2.5 Flash (thinking) | 88.1 | 19.2 | 7.5 | 75.5 | 77.5 | 8.4 |
| Gemini 2.5 Flash (base) | 89.7 | 22.5 | 3.0 | 73.5 | 75.0 | 1.3 |
| Qwen3 235B (thinking) | 89.4 | 14.2 | 8.5 | 63.0 | 71.6 | 84.8 |
| Qwen3 235B (instruct) | 71.9 | 12.5 | 37.5 | 22.0 | 46.6 | 4.2 |
| DeepSeek R1 | 85.6 | 22.5 | 9.5 | 65.5 | 70.0 | 82.4 |
| o4-mini | 80.0 | 13.3 | 24.0 | 51.5 | 64.7 | 6.1 |

---

## Demo Videos

Four closed-loop demonstrations in Gazebo/PX4, spanning no-failure through compound simultaneous failures.

| Demo | Scenario | Description |
|---|---|---|
| Demo 1 | No failure | Nominal mission — takeoff, scan, orbit each object, return home |
| Demo 2 | Isolated failure | Battery critically low (10%) injected mid-mission |
| Demo 3 | Silent failure | Object 2 silently removed from tail after object 1 mapped — replanner must detect and restore |
| Demo 4 | Compound failure | Low battery + tracking loss injected simultaneously |

📂 [Demo videos on Google Drive](https://drive.google.com/drive/folders/15buozLWNcE_k_DTQkE_LhOT8YPfIuZYQ?usp=drive_link)

---

## Setup

### Offline Evaluations

The offline experiments (Experiments 1–4) have no ROS2 or simulation dependency. All you need is API access.

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/resilient-uav-planner.git
cd resilient-uav-planner
```

**2. Install dependencies**

```bash
pip install openai
```

**3. Export API keys**

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENAI_API_KEY="your-openai-key"        # only needed for o4-mini
```

**4. Run all experiments (planning & replanning)**

```bash
python harness.py
```

Results are printed to stdout and saved as JSON in `results/`.

---

### Full System (Gazebo + PX4 + ROS2)

The closed-loop system builds on the [Space Robotics course repo]([https://github.com/<space-robotics-repo-link>](https://github.com/DREAMS-lab/ses598-space-robotics-and-ai-2026)) — set that up first following its README.

The following files from this repo drop into the `assignments/terrain_mapping_drone_control/terrain_mapping_drone_control` directory of that codebase:

| File | Role |
|---|---|
| `nl_drone_control.py` | Language-guided autonomous UAV mission executor (ROS 2 / PX4) |
| `replanner.py` | AutoReplanner: mid-flight plan revision triggered after each skill completes
or when a synthetic fault is injected |
| `prompts.py` | LLM prompts for UAV mission planning and replanning |
| `skills.py` | All UAV skill implementations |
| `llm_client.py` | LLM client for the Gazebo mission executor |
| `event_injector.py` | Synthetic fault injection for mid-flight replan testing |
| `plan_utils.py` | Plan parsing, validation, and printing for the Gazebo executor |
| `geometry_tracker.py` | Detects cylindrical objects from a depth image using vertical line pairs |

Once the files are in place, launch as described in the base repo and provide a natural language mission string into `nl_drone_control.py` MISSIONS variable as the entry point.

<!---

## Citation

```bibtex
@article{turnau2025resilientuav,
  title   = {Language-Guided Adaptive Mission Replanning for Resilient Autonomous UAVs in Unknown Environments},
  author  = {Turnau, Justin},
  year    = {2025},
  note    = {Manuscript in preparation}
}
```
-->
---

## Author

**Justin Turnau** — Arizona State University  
[jturnau@asu.edu](mailto:jturnau@asu.edu)
