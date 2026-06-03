"""
skills.py
---------
All UAV skill implementations.

Each skill exposes:
    tick(args, ctx, node) -> bool   # True = this invocation is done
    reset()                          # called before every fresh run

Skills communicate exclusively through ExecutionContext (ctx) and the
MissionExecutorNode (node) - no inter-skill coupling.

Object IDs start at 1 (matching the planner vocabulary).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prompts import MIN_STANDOFF_M

if TYPE_CHECKING:
    from geometry_msgs.msg import Point
    from mission_executor import MissionExecutorNode


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class SpottedObject:
    """A detected object with world-frame position and perception metadata."""
    id:       int
    ned_x:    float
    ned_y:    float
    ned_z:    float
    yaw:      float       # drone yaw at detection
    depth_m:  float
    world_x:  float
    world_y:  float
    px_cx:    float       # centroid pixel x
    px_cy:    float       # centroid pixel y
    width_px: float


@dataclass
class ExecutionContext:
    """Shared state passed to every skill tick."""
    objects: list[SpottedObject] = field(default_factory=list)
    extras:  dict                = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SKILL BASE
# ---------------------------------------------------------------------------

class Skill:
    def tick(self, args: dict, ctx: ExecutionContext, node: "MissionExecutorNode") -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# HELPER: timed settle
# ---------------------------------------------------------------------------

def _settle(node, state_dict: dict, key: str, duration_s: float) -> bool:
    """Hold current position for duration_s. Returns True when time elapses."""
    ss = state_dict.setdefault(key, {})
    if "end_time" not in ss:
        cx, cy, cz = node.current_pos()
        yaw         = node.current_yaw()
        ss["end_time"] = time.monotonic() + duration_s
        ss["pos"]      = (cx, cy, cz)
        ss["yaw"]      = yaw
        node.get_logger().info(
            f"[SETTLE/{key}] Holding {duration_s:.1f}s at "
            f"({cx:.2f},{cy:.2f},{cz:.2f}) yaw={math.degrees(yaw):.1f} deg"
        )
    hx, hy, hz = ss["pos"]
    node.publish_trajectory_setpoint(x=hx, y=hy, z=hz, yaw=ss["yaw"])
    remaining = ss["end_time"] - time.monotonic()
    if node.tick_count % 10 == 0:
        node.get_logger().info(f"[SETTLE/{key}] {remaining:.1f}s remaining")
    return time.monotonic() >= ss["end_time"]


# ---------------------------------------------------------------------------
# TAKEOFF
# ---------------------------------------------------------------------------

class TakeoffSkill(Skill):
    def tick(self, args, ctx, node):
        altitude = float(args["altitude"])
        cx, cy, _ = node.current_pos()
        node.publish_trajectory_setpoint(x=cx, y=cy, z=-altitude, yaw=0.0)
        if node.tick_count % 10 == 0:
            node.get_logger().info(
                f"[TAKEOFF] target={altitude:.1f}m  "
                f"current={node.current_altitude():.2f}m"
            )
        return abs(node.current_altitude() - altitude) < node.POSITION_THRESHOLD


# ---------------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------------

class SearchYawScanSkill(Skill):
    def __init__(self):
        self._s: dict = {}

    def reset(self):
        self._s = {}

    def tick(self, args, ctx, node):
        s = self._s
        if "start_yaw" not in s:
            s["start_yaw"]   = node.current_yaw()
            s["accumulated"] = 0.0
            node._publish_search_active(True)
            node.get_logger().info("[SEARCH/yaw_scan] Starting 360 degree scan - gate OPEN")

        cx, cy, cz = node.current_pos()
        yaw_speed        = 0.05
        s["start_yaw"]   += yaw_speed
        s["accumulated"] += yaw_speed
        node.publish_trajectory_setpoint(x=cx, y=cy, z=cz, yaw=s["start_yaw"])

        if s["accumulated"] >= 2 * math.pi:
            node._publish_search_active(False)
            ctx.objects = list(node.spotted)
            node.get_logger().info(
                f"[SEARCH/yaw_scan] Complete - gate CLOSED.  "
                f"Spotted {len(ctx.objects)} object(s)."
            )
            return True
        return False


# ---------------------------------------------------------------------------
# APPROACH
# ---------------------------------------------------------------------------

class ApproachSkill(Skill):
    """
    Flies to standoff_distance from one or more objects.

    Phase pipeline per target:
      I - Initial settle so perception stabilises.
      A - Coarse fly: position standoff_distance behind the object.
      S - Short settle after coarse fly.
      B - Fine yaw: rotate until object is centred horizontally in frame.
      C - Depth match: advance/retreat along facing ray until depth == standoff.
      R - Recovery: fly back to object NED pos and restart from S.
    """

    CENTER_PX_TOLERANCE:    float = 20.0
    CENTER_CONFIRM_TICKS:   int   = 15
    CENTER_YAW_STEP_MAX:    float = 0.005
    CENTER_YAW_GAIN:        float = 0.00005
    CENTER_ADJUST_SETTLE_S: float = 0.6
    APPROACH_SETTLE_S:      float = 3.0
    INITIAL_SETTLE_S:       float = 4.0
    LOST_SIGHT_TIMEOUT_S:   float = 1.5
    MAX_RETRIES:            int   = 3
    DEPTH_STEP_M:           float = 0.3

    def __init__(self):
        self._s: dict = {}

    def reset(self):
        self._s = {}

    def _resolve_targets(self, args, ctx, node) -> list[SpottedObject]:
        oid = args.get("object_id")
        if oid is None or oid == "all":
            return list(ctx.objects)
        if oid == "first":
            return ctx.objects[:1]
        for obj in ctx.objects:
            if obj.id == int(oid):
                return [obj]
        node.get_logger().warn(
            f"[APPROACH] object_id={oid} not in context - skipping"
        )
        return []

    def tick(self, args, ctx, node):
        s = self._s

        if "targets" not in s:
            targets = self._resolve_targets(args, ctx, node)
            if not targets:
                node.get_logger().warn("[APPROACH] No targets - skipping skill")
                return True
            s.update(
                targets     = targets,
                target_idx  = -1,
                phase       = "I",
                retry_count = 0,
            )
            node.get_logger().info(
                f"[APPROACH] {len(targets)} target(s) - "
                f"initial settle {self.INITIAL_SETTLE_S:.0f}s"
            )

        standoff = float(args.get("standoff_distance", MIN_STANDOFF_M))

        if s["phase"] == "I":
            if _settle(node, s, "initial", self.INITIAL_SETTLE_S):
                s["target_idx"] = 0
                s["phase"]      = "A"
            return False

        idx = s["target_idx"]
        if idx >= len(s["targets"]):
            node.get_logger().info("[APPROACH] All targets approached")
            return True

        obj = s["targets"][idx]

        def sight_ok() -> bool:
            if node.live_object_center is None:
                return False
            return (time.monotonic() - node._live_center_stamp) < self.LOST_SIGHT_TIMEOUT_S

        def enter_recovery(reason: str):
            s["retry_count"] += 1
            if s["retry_count"] > self.MAX_RETRIES:
                node.get_logger().warn(
                    f"[APPROACH] Obj {obj.id}: max retries exceeded - skipping"
                )
                s["target_idx"] += 1
                s["phase"]       = "A"
                s["retry_count"] = 0
            else:
                node.get_logger().warn(
                    f"[APPROACH] Obj {obj.id}: {reason} "
                    f"(retry {s['retry_count']}/{self.MAX_RETRIES}) - recovering"
                )
                s["phase"] = "R"

        if s["phase"] == "A":
            ax = obj.world_x - standoff * math.cos(obj.yaw)
            ay = obj.world_y - standoff * math.sin(obj.yaw)
            node.publish_trajectory_setpoint(x=ax, y=ay, z=obj.ned_z, yaw=obj.yaw)
            pos_ok = node.at_position(ax, ay, obj.ned_z)
            yaw_ok = node.at_yaw(obj.yaw)
            if node.tick_count % 10 == 0:
                node.get_logger().info(
                    f"[APPROACH] Obj {obj.id} Phase A  pos_ok={pos_ok} yaw_ok={yaw_ok}"
                )
            if pos_ok and yaw_ok:
                s["phase"] = "S"
                s.pop("interphase", None)
                node.live_object_center = None
            return False

        if s["phase"] == "S":
            ss = s.setdefault("interphase", {})
            if "end_time" not in ss:
                ss["end_time"] = time.monotonic() + self.APPROACH_SETTLE_S
                cx, cy, cz     = node.current_pos()
                ss["pos"]      = (cx, cy, cz)
                ss["yaw"]      = obj.yaw
            hx, hy, hz = ss["pos"]
            node.publish_trajectory_setpoint(x=hx, y=hy, z=hz, yaw=ss["yaw"])
            remaining = ss["end_time"] - time.monotonic()
            if node.tick_count % 10 == 0:
                node.get_logger().info(
                    f"[APPROACH] Obj {obj.id} Phase S  settling {remaining:.1f}s"
                )
            if time.monotonic() >= ss["end_time"]:
                s.update(
                    phase               = "B",
                    center_yaw          = obj.yaw,
                    centered_ticks      = 0,
                    b_adjust_settle_end = 0.0,
                )
            return False

        if s["phase"] == "B":
            cx, cy, cz = node.current_pos()
            if not sight_ok():
                enter_recovery("sight lost during centering")
                return False
            now = time.monotonic()
            if now < s.get("b_adjust_settle_end", 0.0):
                node.publish_trajectory_setpoint(x=cx, y=cy, z=cz, yaw=s["center_yaw"])
                return False
            pixel_error = node.live_object_center.x - node._image_half_w
            if abs(pixel_error) <= self.CENTER_PX_TOLERANCE:
                s["centered_ticks"] += 1
            else:
                s["centered_ticks"] = 0
            if s["centered_ticks"] >= self.CENTER_CONFIRM_TICKS:
                s["approach_yaw"] = s["center_yaw"]
                s["phase"]        = "C"
                node.get_logger().info(
                    f"[APPROACH] Obj {obj.id} Phase B CENTRED  "
                    f"yaw={math.degrees(s['center_yaw']):.1f}degrees - Phase C"
                )
                return False
            yaw_delta = self.CENTER_YAW_GAIN * pixel_error
            yaw_delta = math.copysign(min(abs(yaw_delta), self.CENTER_YAW_STEP_MAX), yaw_delta)
            s["center_yaw"]          += yaw_delta
            s["b_adjust_settle_end"]  = now + self.CENTER_ADJUST_SETTLE_S
            node.publish_trajectory_setpoint(x=cx, y=cy, z=cz, yaw=s["center_yaw"])
            return False

        if s["phase"] == "C":
            if not sight_ok():
                enter_recovery("sight lost during depth matching")
                return False
            live_depth = node.live_object_center.z
            cx, cy, cz = node.current_pos()
            if node.tick_count % 10 == 0:
                node.get_logger().info(
                    f"[APPROACH] Obj {obj.id} Phase C  "
                    f"depth={live_depth:.2f}m  target={standoff:.1f}m"
                )
            if abs(live_depth - standoff) < 0.2:
                node.get_logger().info(
                    f"[APPROACH] Obj {obj.id} Phase C DEPTH MATCHED  "
                    f"depth={live_depth:.2f}m  standoff={standoff:.1f}m"
                )
                s["target_idx"] += 1
                s["phase"]       = "A"
                s["retry_count"] = 0
                node.live_object_center = None
                return s["target_idx"] >= len(s["targets"])
            direction = 1.0 if live_depth > standoff else -1.0
            step      = direction * self.DEPTH_STEP_M
            sx = cx + step * math.cos(s["approach_yaw"])
            sy = cy + step * math.sin(s["approach_yaw"])
            node.publish_trajectory_setpoint(x=sx, y=sy, z=obj.ned_z, yaw=s["approach_yaw"])
            return False

        if s["phase"] == "R":
            node.publish_trajectory_setpoint(
                x=obj.ned_x, y=obj.ned_y, z=obj.ned_z, yaw=obj.yaw
            )
            if node.at_position(obj.ned_x, obj.ned_y, obj.ned_z) and node.at_yaw(obj.yaw):
                s.update(phase="S", b_adjust_settle_end=0.0)
                s.pop("interphase", None)
                node.live_object_center = None
            return False

        return False


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------

class MapSkill(Skill):
    """Orbits one or all objects once per invocation (repeat: N for N orbits)."""

    ORBIT_YAW_STEP: float = 0.015

    def __init__(self):
        self._s: dict     = {}
        self._orbit: dict = {}

    def reset(self):
        self._s     = {}
        self._orbit = {}

    def _resolve_targets(self, args, ctx, node) -> list[SpottedObject]:
        oid = args.get("object_id")
        if oid is None or oid == "all":
            return list(ctx.objects)
        for obj in ctx.objects:
            if obj.id == int(oid):
                return [obj]
        node.get_logger().warn(f"[MAP] object_id={oid} not found - skipping")
        return []

    def tick(self, args, ctx, node):
        s = self._s

        if "targets" not in s:
            targets = self._resolve_targets(args, ctx, node)
            if not targets:
                node.get_logger().warn("[MAP] No targets - skipping skill")
                return True
            standoff = max(
                float(args.get("standoff_distance", MIN_STANDOFF_M)), MIN_STANDOFF_M
            )
            s.update(targets=targets, target_idx=0, standoff=standoff)
            self._orbit = {}
            node.get_logger().info(
                f"[MAP] {len(targets)} object(s)  standoff={standoff:.1f}m"
            )

        idx = s["target_idx"]
        if idx >= len(s["targets"]):
            node.get_logger().info("[MAP] All objects mapped")
            return True

        obj      = s["targets"][idx]
        standoff = s["standoff"]

        if not self._orbit or self._orbit.get("for_idx") != idx:
            cx, cy, _ = node.current_pos()
            actual_r  = math.hypot(cx - obj.world_x, cy - obj.world_y)
            radius    = max(actual_r, MIN_STANDOFF_M, standoff)
            self._orbit = dict(
                for_idx     = idx,
                angle       = math.atan2(cy - obj.world_y, cx - obj.world_x),
                accumulated = 0.0,
                z           = obj.ned_z,
                radius      = radius,
                cx          = obj.world_x,
                cy          = obj.world_y,
            )
            node.get_logger().info(
                f"[MAP] Obj {obj.id}  actual_dist={actual_r:.2f}m  "
                f"orbit_r={radius:.2f}m"
            )

        o = self._orbit
        o["angle"]       += self.ORBIT_YAW_STEP
        o["accumulated"] += self.ORBIT_YAW_STEP

        gx       = o["cx"] + o["radius"] * math.cos(o["angle"])
        gy       = o["cy"] + o["radius"] * math.sin(o["angle"])
        face_yaw = math.atan2(o["cy"] - gy, o["cx"] - gx)
        node.publish_trajectory_setpoint(x=gx, y=gy, z=o["z"], yaw=face_yaw)

        if node.tick_count % 10 == 0:
            pct = 100.0 * o["accumulated"] / (2 * math.pi)
            node.get_logger().info(
                f"[MAP] Obj {obj.id}  {pct:.0f}%  r={o['radius']:.2f}m"
            )

        if o["accumulated"] >= 2 * math.pi:
            node.get_logger().info(f"[MAP] Full orbit complete - Obj {obj.id}")
            s["target_idx"] += 1
            self._orbit      = {}
            return s["target_idx"] >= len(s["targets"])

        return False


# ---------------------------------------------------------------------------
# RETURN HOME
# ---------------------------------------------------------------------------

class ReturnHomeSkill(Skill):
    def __init__(self):
        self._sent = False

    def reset(self):
        self._sent = False

    def tick(self, args, ctx, node):
        if not self._sent:
            node.send_return_to_launch()
            self._sent = True
            node.get_logger().info(
                "[RETURN_HOME] RTL sent\n"
                "=== MISSION OBJECT SUMMARY ===\n"
                + _object_summary(ctx.objects)
            )
        return True


def _object_summary(objects: list[SpottedObject]) -> str:
    if not objects:
        return "  No objects spotted."
    lines = [f"  {len(objects)} object(s):"]
    for obj in objects:
        lines.append(
            f"    [ID {obj.id:02d}]  "
            f"drone_pos=({obj.ned_x:+.2f},{obj.ned_y:+.2f},{obj.ned_z:+.2f})  "
            f"yaw={math.degrees(obj.yaw):+.1f}degrees  depth={obj.depth_m:.2f}m  "
            f"world=({obj.world_x:+.2f},{obj.world_y:+.2f})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SKILL REGISTRY
# ---------------------------------------------------------------------------

SKILL_REGISTRY: dict[str, type[Skill]] = {
    "takeoff":     TakeoffSkill,
    "search":      SearchYawScanSkill,
    "approach":    ApproachSkill,
    "map":         MapSkill,
    "return_home": ReturnHomeSkill,
}