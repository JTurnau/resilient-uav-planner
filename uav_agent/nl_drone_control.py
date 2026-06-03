#!/usr/bin/env python3
"""
mission_executor.py
-------------------
Language-guided autonomous UAV mission executor (ROS 2 / PX4).

Architecture
------------
1. At startup, the planning LLM converts a natural-language mission string into
   a typed skill sequence (plan).
2. The executor is a fixed state machine that advances through plan steps,
   calling skill.tick() until it returns True, honouring repeat counts, and
   settling between transitions.
3. After every skill transition - or whenever an anomaly is injected - the
   mid-flight replanner queries the LLM with a full mission-status snapshot.
   The executor is held at its current position until the LLM responds.
   If the LLM says NOMINAL the existing plan continues unchanged; otherwise
   the remaining tail is replaced with the revised plan.
4. Synthetic faults can be injected at a specific step index or wall-clock
   time via INJECTED_EVENTS in event_injector.py.

Module layout
-------------
  prompts.py         - PLANNING_PROMPT + build_replan_system_prompt()
  llm_client.py      - query_llm() wrapper (Gemini / Cerebras)
  plan_utils.py      - extract_json(), validate_plan(), parse_replan_response(),
                       print_plan()
  skills.py          - all Skill subclasses + ExecutionContext + SpottedObject
  replanner.py       - AutoReplanner (background thread)
  event_injector.py  - SyntheticEvent / EventInjector + INJECTED_EVENTS config
  mission_executor.py - this file: ROS 2 node + entry point
"""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy,
    QoSProfile, QoSReliabilityPolicy,
)

from geometry_msgs.msg import Point
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint,
    VehicleCommand, VehicleOdometry, VehicleStatus,
)
from std_msgs.msg import Bool, Float32MultiArray, Int32MultiArray

from llm_client      import query_llm, model_label
from plan_utils       import extract_json, validate_plan, print_plan
from prompts          import PLANNING_PROMPT
from skills           import (
    SKILL_REGISTRY, ExecutionContext, SpottedObject,
    _settle,
)
from replanner        import AutoReplanner
from event_injector   import EventInjector, INJECTED_EVENTS


# ---------------------------------------------------------------------------
# MISSION(S) - edit to test different natural-language inputs
# ---------------------------------------------------------------------------

MISSIONS: list[str] = [
    "Take off to 6 meters and perform a quick scan of the area. Orbit each object found once at a 5-meter standoff, then return.",
    # "Search for objects and map every one you find.",
    # "Find any objects and map each one 2 times.",
    # "Approach and inspect the first object you see, then return.",
    # "Search for objects and do 2 vertical mapping laps of each one you find.",
]


# ---------------------------------------------------------------------------
# NODE
# ---------------------------------------------------------------------------

class MissionExecutorNode(Node):

    POSITION_THRESHOLD    = 0.3
    YAW_THRESHOLD         = 0.05
    OBJECT_MERGE_RADIUS   = 1.2   # m - new detections within this range merge
    TRANSITION_SETTLE_S   = 4.0

    def __init__(self, plan: list, mission_intent: str):
        super().__init__("mission_executor_node")

        self.plan           = plan
        self.mission_intent = mission_intent
        self.current_step   = 0
        self.tick_count     = 0

        # Skill instances (one per type, reused across steps)
        self._skill_instances: dict = {
            name: cls() for name, cls in SKILL_REGISTRY.items()
        }
        self._active_skill   = None
        self._repeat_count   = 0
        self._skill_repeats  = 1
        self._step_done      = False
        self._step_done_name = ""
        self._settle_state: dict = {}

        # Replanner / event injection
        self._replanner             = AutoReplanner()
        self._replan_pending: bool  = False
        self._injector              = EventInjector(INJECTED_EVENTS)

        # Shared execution state
        self.ctx = ExecutionContext()

        # Perception state
        self.spotted: list[SpottedObject]  = []
        self._spotted_id_counter           = 1    # IDs start at 1
        self.live_object_center: Point | None = None
        self._live_center_stamp: float        = 0.0
        self.live_object_height_px: float     = 0.0
        self._image_half_w: float             = 400.0
        self._image_half_h: float             = 300.0

        # Vehicle state
        self.odometry       = VehicleOdometry()
        self.vehicle_status = VehicleStatus()
        self.offboard_setpoint_counter = 0
        self.armed_and_offboard        = False
        self._arm_time: float | None   = None

        qos = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            durability  = QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history     = QoSHistoryPolicy.KEEP_LAST,
            depth       = 1,
        )

        # Publishers
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos)
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self.search_active_pub = self.create_publisher(
            Bool, "/mission/search_active", 10)

        # Subscribers
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry",
            self._odometry_cb, qos)
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._status_cb, qos)
        self.create_subscription(
            Float32MultiArray, "/geometry/confirmed_cylinder",
            self._confirmed_object_cb, 10)
        self.create_subscription(
            Point, "/geometry/cylinder_center",
            self._live_center_cb, 10)
        self.create_subscription(
            Int32MultiArray, "/geometry/image_size",
            self._image_size_cb, 10)
        self.create_subscription(
            Float32MultiArray, "/geometry/cylinder_info",
            self._object_info_cb, 10)

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info(
            f"Mission executor ready  [LLM: {model_label()}]"
        )

    # ------------------------------------------------------------------
    # PUBLIC API - anomaly injection
    # ------------------------------------------------------------------

    def inject_event(self, failure_context: str) -> None:
        """
        Immediately suspend execution and trigger a replan with the given
        failure context.  Safe to call from any thread.
        """
        if self._replan_pending:
            self.get_logger().warn(
                "[INJECT] Replan already pending - ignoring new event"
            )
            return
        self.get_logger().warn(
            f"[INJECT] Event injected: {failure_context}"
        )
        self._replan_pending = True
        completed_name = (
            self.plan[self.current_step - 1]["state"]
            if self.current_step > 0
            else "pre-flight"
        )
        self._replanner.trigger(
            self,
            completed_step_name = completed_name,
            failure_context     = failure_context,
        )

    # ------------------------------------------------------------------
    # Perception callbacks
    # ------------------------------------------------------------------

    def _odometry_cb(self, msg):
        self.odometry = msg

    def _status_cb(self, msg):
        self.vehicle_status = msg

    def _live_center_cb(self, msg: Point):
        self.live_object_center = msg
        self._live_center_stamp = time.monotonic()

    def _image_size_cb(self, msg: Int32MultiArray):
        if len(msg.data) >= 1:
            self._image_half_w = float(msg.data[0]) / 2.0
        if len(msg.data) >= 2:
            self._image_half_h = float(msg.data[1]) / 2.0

    def _object_info_cb(self, msg: Float32MultiArray):
        # layout: [width_px, height_px, depth_m, ...]
        if len(msg.data) >= 2:
            self.live_object_height_px = float(msg.data[1])

    def _confirmed_object_cb(self, msg: Float32MultiArray):
        """
        Fuse a confirmed object detection into the spotted list.
        Objects within OBJECT_MERGE_RADIUS of an existing entry are merged
        (best-depth wins); new objects are appended with IDs starting at 1.
        """
        if len(msg.data) < 4:
            return
        cx_px, cy_px, depth_m, width_px = (float(v) for v in msg.data[:4])
        if depth_m <= 0.0 or math.isnan(depth_m):
            return

        ned_x, ned_y, ned_z = self.current_pos()
        yaw             = self.current_yaw()
        pixel_error     = cx_px - self._image_half_w
        yaw_corrected   = yaw + 0.00005 * pixel_error * 5.0
        world_x = ned_x + depth_m * math.cos(yaw_corrected)
        world_y = ned_y + depth_m * math.sin(yaw_corrected)

        for obj in self.spotted:
            if math.hypot(world_x - obj.world_x, world_y - obj.world_y) \
                    < self.OBJECT_MERGE_RADIUS:
                if depth_m < obj.depth_m:
                    obj.depth_m  = depth_m
                    obj.ned_x    = ned_x
                    obj.ned_y    = ned_y
                    obj.ned_z    = ned_z
                    obj.yaw      = yaw_corrected
                    obj.world_x  = world_x
                    obj.world_y  = world_y
                return

        obj = SpottedObject(
            id       = self._spotted_id_counter,
            ned_x    = ned_x,
            ned_y    = ned_y,
            ned_z    = ned_z,
            yaw      = yaw_corrected,
            depth_m  = depth_m,
            world_x  = world_x,
            world_y  = world_y,
            px_cx    = cx_px,
            px_cy    = cy_px,
            width_px = width_px,
        )
        self.spotted.append(obj)
        self._spotted_id_counter += 1
        self.get_logger().info(
            f"[SPOTTED] Object ID={obj.id}  "
            f"depth={depth_m:.2f}m  world=({world_x:.2f},{world_y:.2f})"
        )

    # ------------------------------------------------------------------
    # PX4 helpers
    # ------------------------------------------------------------------

    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        for k in range(1, 8):
            setattr(msg, f"param{k}", float(params.get(f"param{k}", 0.0)))
        msg.target_system    = msg.target_component = 1
        msg.source_system    = msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position  = True
        msg.velocity  = msg.acceleration = msg.attitude = msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        msg          = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw      = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
        )

    def engage_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
        )

    def send_return_to_launch(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)

    def _publish_search_active(self, active: bool):
        msg = Bool()
        msg.data = active
        self.search_active_pub.publish(msg)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def current_pos(self) -> tuple[float, float, float]:
        try:
            return tuple(float(v) for v in self.odometry.position[:3])
        except Exception:
            return (0.0, 0.0, 0.0)

    def current_altitude(self) -> float:
        return -self.current_pos()[2]

    def current_yaw(self) -> float:
        try:
            q    = self.odometry.q
            siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
            cosy = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
            return math.atan2(siny, cosy)
        except Exception:
            return 0.0

    def at_position(self, tx, ty, tz) -> bool:
        x, y, z = self.current_pos()
        return math.sqrt((x-tx)**2 + (y-ty)**2 + (z-tz)**2) < self.POSITION_THRESHOLD

    def at_yaw(self, target: float) -> bool:
        diff = abs(self.current_yaw() - target)
        return min(diff, 2 * math.pi - diff) < self.YAW_THRESHOLD

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        self.publish_offboard_control_mode()
        self.tick_count += 1

        # Arm + engage offboard after 10 setpoints
        if self.offboard_setpoint_counter == 10 and not self.armed_and_offboard:
            self.engage_offboard_mode()
            self.arm()
            self.armed_and_offboard = True
            self._arm_time          = time.monotonic()
            self._injector.arm()
        self.offboard_setpoint_counter += 1

        if not self.armed_and_offboard:
            return
        if self.current_step >= len(self.plan):
            return

        # Check synthetic event injection every tick
        self._injector.check(self)

        step = self.plan[self.current_step]
        if not isinstance(step, dict):
            self.get_logger().error(
                f"[CONTROL] Step {self.current_step} is not a dict "
                f"({type(step).__name__}: {step!r}) - skipping"
            )
            self.current_step += 1
            return

        # ---- HOLD: transition settle or waiting for replanner ----
        if self._step_done:
            settled = _settle(
                self, self._settle_state, "_transition", self.TRANSITION_SETTLE_S
            )
            if settled:
                completed_name = self._step_done_name
                self.current_step  += 1
                self._step_done     = False
                self._settle_state  = {}
                self._active_skill  = None

                # Every skill transition is an event - trigger replan
                self._replan_pending = True
                self._replanner.trigger(
                    self,
                    completed_step_name = completed_name,
                )
                self.get_logger().info(
                    f"Step {self.current_step - 1} [{completed_name}] "
                    f"complete - advancing to step {self.current_step}"
                )
                print_plan(
                    self.plan,
                    logger       = self.get_logger(),
                    current_step = self.current_step,
                    label        = (
                        f"PLAN PROGRESS  "
                        f"(step {self.current_step}/{len(self.plan)})"
                    ),
                )
            return

        if self._replan_pending:
            # Hold at current setpoint while replanner thinks
            cx, cy, cz = self.current_pos()
            self.publish_trajectory_setpoint(x=cx, y=cy, z=cz, yaw=self.current_yaw())
            if self.tick_count % 20 == 0:
                next_name = self.plan[self.current_step].get("state", "?")
                self.get_logger().info(
                    f"[CONTROL] Holding - waiting for replanner before "
                    f"step {self.current_step} [{next_name}]"
                )
            return

        # ---- EXECUTE current skill ----
        skill_name = step["state"]
        if self._active_skill is None:
            skill = self._skill_instances.get(skill_name)
            if skill is None:
                self.get_logger().warn(
                    f"Unknown skill '{skill_name}' - skipping"
                )
                self._step_done      = True
                self._step_done_name = skill_name
                return
            repeat = step.get("repeat", 1)
            self._active_skill  = skill
            self._repeat_count  = 0
            self._skill_repeats = repeat
            skill.reset()
            self.get_logger().info(
                f"Step {self.current_step}: starting [{skill_name}]  "
                f"repeats={repeat}"
            )

        done = self._active_skill.tick(step.get("args", {}), self.ctx, self)

        if done:
            self._repeat_count += 1
            self.get_logger().info(
                f"[REPEAT] step={self.current_step} [{skill_name}]  "
                f"rep {self._repeat_count}/{self._skill_repeats} complete"
            )
            if self._repeat_count < self._skill_repeats:
                self._active_skill.reset()
            else:
                self.get_logger().info(
                    f"Step {self.current_step} [{skill_name}] "
                    f"all {self._skill_repeats} rep(s) done - settling"
                )
                self._step_done      = True
                self._step_done_name = skill_name
                self._settle_state   = {}


# ---------------------------------------------------------------------------
# PLAN GENERATION
# ---------------------------------------------------------------------------

def generate_plan(mission: str) -> list | None:
    print(f"\n{'='*60}")
    print(f"Mission: {mission}")
    print("="*60)
    for attempt in range(3):
        try:
            raw  = query_llm(mission, system=PLANNING_PROMPT, max_tokens=4096)
            print(f"  [RAW attempt {attempt+1}]:\n{raw}\n")
            plan = extract_json(raw)
            ok, errs = validate_plan(plan)
            if ok:
                print_plan(plan, label=f"INITIAL PLAN  ({len(plan)} steps)")
                return plan
            print(f"  [INVALID]: {'; '.join(errs)}")
        except Exception as exc:
            print(f"  [ERROR] {exc}")
    return None


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    if not MISSIONS:
        print("No missions defined - aborting.")
        return
    mission = MISSIONS[0]
    plan    = generate_plan(mission)
    if plan is None:
        print("Could not generate a valid plan - aborting.")
        return
    print(f"\nExecuting {len(plan)}-step plan for: {mission!r}")
    rclpy.init()
    node = MissionExecutorNode(plan, mission_intent=mission)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()