from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from so101_rl.envs import SO101PickPlaceEnv


@dataclass(frozen=True)
class ControlSpec:
    sim_hz: float
    policy_hz: float
    sim_timestep: float
    frame_skip: int
    action_repeat: int


@dataclass(frozen=True)
class StageFlags:
    reached_object: bool
    is_grasping: bool
    is_lifted: bool
    is_above_bowl: bool
    is_in_bowl: bool
    success: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "reached_object": self.reached_object,
            "is_grasping": self.is_grasping,
            "is_lifted": self.is_lifted,
            "is_above_bowl": self.is_above_bowl,
            "is_in_bowl": self.is_in_bowl,
            "success": self.success,
        }


@dataclass(frozen=True)
class TeacherObservation:
    vector: np.ndarray
    fields: dict[str, np.ndarray]


@dataclass(frozen=True)
class StudentObservation:
    images: dict[str, np.ndarray]
    proprio: dict[str, np.ndarray]


@dataclass(frozen=True)
class RolloutStep:
    teacher: TeacherObservation
    student: StudentObservation
    action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    flags: StageFlags
    info: dict[str, Any]


def control_spec(env: SO101PickPlaceEnv) -> ControlSpec:
    sim_hz = 1.0 / env.config.timestep
    policy_hz = env.config.control_hz
    return ControlSpec(
        sim_hz=sim_hz,
        policy_hz=policy_hz,
        sim_timestep=env.config.timestep,
        frame_skip=env.config.frame_skip,
        action_repeat=env.config.frame_skip,
    )


def teacher_observation(env: SO101PickPlaceEnv) -> TeacherObservation:
    info = env._build_info()
    joint_positions = env.data.qpos[env._joint_qpos_adrs].astype(np.float32).copy()
    joint_velocities = env.data.qvel[env._joint_qvel_adrs].astype(np.float32).copy()
    gripper_state = np.asarray([joint_positions[-1], joint_velocities[-1]], dtype=np.float32)

    ee_position = env.data.site_xpos[env._ee_site_id].astype(np.float32).copy()
    ee_quaternion = mat_to_quat(env.data.site_xmat[env._ee_site_id].reshape(3, 3))
    ee_pose = np.concatenate([ee_position, ee_quaternion], dtype=np.float32)

    object_pose = env.data.qpos[env._object_qpos_adr : env._object_qpos_adr + 7].astype(np.float32).copy()
    object_velocity = env.data.qvel[env._object_qvel_adr : env._object_qvel_adr + 6].astype(np.float32).copy()

    bowl_pose = np.concatenate(
        [
            env._receptacle_position.astype(np.float32).copy(),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    gripper_to_object = object_pose[:3] - ee_position
    object_to_bowl = bowl_pose[:3] - object_pose[:3]
    stage_flags = np.asarray(
        [
            float(info["object_between_jaws"]),
            float(info["has_gripper_contact"]),
            float(info["gripper_is_closed"]),
            float(info["is_grasping"]),
            float(info["is_lifted"]),
        ],
        dtype=np.float32,
    )

    fields = {
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "gripper_state": gripper_state,
        "end_effector_pose": ee_pose,
        "object_pose": object_pose,
        "object_velocity": object_velocity,
        "bowl_pose": bowl_pose,
        "gripper_to_object": gripper_to_object.astype(np.float32),
        "object_to_bowl": object_to_bowl.astype(np.float32),
        "stage_flags": stage_flags,
    }
    vector = np.concatenate(list(fields.values()), dtype=np.float32)
    return TeacherObservation(vector=vector, fields=fields)


def student_observation(
    observation: dict[str, np.ndarray],
    teacher: TeacherObservation,
) -> StudentObservation:
    images = {
        "cam_high": observation["observation.images.cam_high"],
        "cam_right_wrist": observation["observation.images.cam_right_wrist"],
    }
    proprio = {
        "joint_positions": teacher.fields["joint_positions"],
        "joint_velocities": teacher.fields["joint_velocities"],
        "gripper_state": teacher.fields["gripper_state"],
    }
    return StudentObservation(images=images, proprio=proprio)


def stage_flags(env: SO101PickPlaceEnv) -> StageFlags:
    object_position = env.data.site_xpos[env._object_site_id].astype(np.float32)
    ee_position = env.data.site_xpos[env._ee_site_id].astype(np.float32)
    object_velocity = env.data.qvel[env._object_qvel_adr : env._object_qvel_adr + 3].astype(np.float32)
    object_speed = float(np.linalg.norm(object_velocity))

    pre_grasp = object_position.copy()
    pre_grasp[2] += 0.045
    reach_distance = float(np.linalg.norm(ee_position - pre_grasp))
    reached_object = reach_distance <= 0.035

    lifted_height = env.config.table_top_z + env.config.object_half_height + 0.025
    is_lifted = bool(object_position[2] >= lifted_height)

    xy_distance = float(np.linalg.norm(object_position[:2] - env._receptacle_position[:2]))
    is_above_bowl = bool(
        xy_distance <= env.config.insertion_xy_threshold
        and object_position[2]
        >= env.config.table_top_z + env.config.receptacle_wall_height + env.config.object_half_height
    )

    upper_z = env.config.table_top_z + env.config.receptacle_base_half_height + env.config.receptacle_wall_height
    lower_z = env.config.table_top_z + env.config.receptacle_base_half_height
    is_in_bowl = bool(
        xy_distance <= env.config.insertion_xy_threshold
        and lower_z <= object_position[2] <= upper_z
        and object_speed <= 0.15
    )
    is_grasping = bool(_object_in_gripper_contacts(env) and is_lifted)
    success = is_in_bowl
    return StageFlags(
        reached_object=reached_object,
        is_grasping=is_grasping,
        is_lifted=is_lifted,
        is_above_bowl=is_above_bowl,
        is_in_bowl=is_in_bowl,
        success=success,
    )


def build_step(
    env: SO101PickPlaceEnv,
    observation: dict[str, np.ndarray],
    reward: float,
    terminated: bool,
    truncated: bool,
    action: np.ndarray,
    info: dict[str, Any],
) -> RolloutStep:
    teacher = teacher_observation(env)
    student = student_observation(observation, teacher)
    flags = stage_flags(env)
    merged_info = dict(info)
    merged_info.update(flags.as_dict())
    return RolloutStep(
        teacher=teacher,
        student=student,
        action=np.asarray(action, dtype=np.float32).copy(),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        flags=flags,
        info=merged_info,
    )


class ScriptedPickPlacePolicy:
    STAGES = (
        "open_gripper_wide",
        "move_above_object",
        "descend_to_object",
        "close_gripper",
        "lift_object",
        "move_above_bowl",
        "lower_to_bowl",
        "open_gripper",
        "retreat",
    )

    def __init__(self) -> None:
        self.stage_index = 0
        self.stage_step = 0
        self.closed_gripper_target = -0.05

    @property
    def stage_name(self) -> str:
        return self.STAGES[min(self.stage_index, len(self.STAGES) - 1)]

    def reset(self) -> None:
        self.stage_index = 0
        self.stage_step = 0

    def act(self, env: SO101PickPlaceEnv, flags: StageFlags) -> np.ndarray:
        object_position = env.data.site_xpos[env._object_site_id].astype(np.float32)
        bowl_position = env._receptacle_position.astype(np.float32)
        ee_position = env.data.site_xpos[env._ee_site_id].astype(np.float32)
        open_gripper_target = float(env._ctrl_range[-1, 1])
        gripper_target = open_gripper_target

        if self.stage_name == "open_gripper_wide":
            target = ee_position + np.asarray([0.0, 0.0, 0.03], dtype=np.float32)
            if abs(float(env.data.qpos[env._joint_qpos_adrs[-1]]) - open_gripper_target) < 0.03 or self.stage_step > 15:
                self._advance()
        elif self.stage_name == "move_above_object":
            target = object_position + np.asarray([0.0, 0.0, 0.10], dtype=np.float32)
            if np.linalg.norm(ee_position - target) < 0.02 or self.stage_step > 90:
                self._advance()
        elif self.stage_name == "descend_to_object":
            descent_start = 0.10
            descent_end = 0.022
            descent_progress = min(1.0, self.stage_step / 65.0)
            descent_height = descent_start + (descent_end - descent_start) * descent_progress
            target = object_position + np.asarray([0.0, 0.0, descent_height], dtype=np.float32)
            if np.linalg.norm(ee_position - target) < 0.012 or self.stage_step > 75:
                self._advance()
        elif self.stage_name == "close_gripper":
            target = object_position + np.asarray([0.0, 0.0, 0.02], dtype=np.float32)
            gripper_target = self.closed_gripper_target
            if flags.is_grasping or self.stage_step > 25:
                self._advance()
        elif self.stage_name == "lift_object":
            target = object_position + np.asarray([0.0, 0.0, 0.12], dtype=np.float32)
            gripper_target = self.closed_gripper_target
            if flags.is_lifted and self.stage_step > 10:
                self._advance()
        elif self.stage_name == "move_above_bowl":
            target = bowl_position + np.asarray([0.0, 0.0, 0.12], dtype=np.float32)
            gripper_target = self.closed_gripper_target
            if flags.is_above_bowl or self.stage_step > 90:
                self._advance()
        elif self.stage_name == "lower_to_bowl":
            target = bowl_position + np.asarray([0.0, 0.0, 0.055], dtype=np.float32)
            gripper_target = self.closed_gripper_target
            if self.stage_step > 45:
                self._advance()
        elif self.stage_name == "open_gripper":
            target = bowl_position + np.asarray([0.0, 0.0, 0.055], dtype=np.float32)
            gripper_target = open_gripper_target
            if flags.is_in_bowl or self.stage_step > 20:
                self._advance()
        else:
            target = bowl_position + np.asarray([0.0, 0.0, 0.12], dtype=np.float32)
            gripper_target = open_gripper_target

        position_gain = 2.5 if self.stage_name in {"open_gripper_wide", "descend_to_object"} else 5.0
        max_xyz_step = 0.010 if self.stage_name in {"open_gripper_wide", "descend_to_object"} else 0.025
        action = cartesian_delta_action(
            env,
            target,
            gripper_target,
            position_gain=position_gain,
            max_xyz_step=max_xyz_step,
        )
        self.stage_step += 1
        return action

    def _advance(self) -> None:
        if self.stage_index < len(self.STAGES) - 1:
            self.stage_index += 1
        self.stage_step = 0


def cartesian_delta_action(
    env: SO101PickPlaceEnv,
    target_position: np.ndarray,
    gripper_target: float,
    damping: float = 1e-3,
    position_gain: float = 5.0,
    max_xyz_step: float = 0.025,
) -> np.ndarray:
    ee_position = env.data.site_xpos[env._ee_site_id].astype(np.float64)
    delta = np.asarray(target_position, dtype=np.float64) - ee_position
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > max_xyz_step:
        delta *= max_xyz_step / delta_norm
    position_error = delta * position_gain

    jacp = np.zeros((3, env.model.nv), dtype=np.float64)
    jacr = np.zeros((3, env.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(env.model, env.data, jacp, jacr, env._ee_site_id)

    arm_dofs = env._joint_qvel_adrs[:5]
    jacobian = jacp[:, arm_dofs]
    system = jacobian @ jacobian.T + (damping * np.eye(3, dtype=np.float64))
    joint_delta = jacobian.T @ np.linalg.solve(system, position_error)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[:5] = np.clip(joint_delta / env._action_scale[:5], -1.0, 1.0).astype(np.float32)

    current_gripper = float(env.data.qpos[env._joint_qpos_adrs[-1]])
    gripper_delta = np.clip(gripper_target - current_gripper, -env._action_scale[-1], env._action_scale[-1])
    action[-1] = float(np.clip(gripper_delta / env._action_scale[-1], -1.0, 1.0))
    return action


def save_dataset(path: str | Path, episodes: list[dict[str, Any]], metadata: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, list[np.ndarray]] = {
        "teacher_observation": [],
        "teacher_action": [],
        "student_joint_positions": [],
        "student_joint_velocities": [],
        "student_gripper_state": [],
        "student_image_cam_high": [],
        "student_image_cam_right_wrist": [],
        "stage_flags": [],
        "success": [],
        "episode_id": [],
        "step_id": [],
    }

    for episode_id, episode in enumerate(episodes):
        for step_id, step in enumerate(episode["steps"]):
            arrays["teacher_observation"].append(step.teacher.vector)
            arrays["teacher_action"].append(step.action)
            arrays["student_joint_positions"].append(step.student.proprio["joint_positions"])
            arrays["student_joint_velocities"].append(step.student.proprio["joint_velocities"])
            arrays["student_gripper_state"].append(step.student.proprio["gripper_state"])
            arrays["student_image_cam_high"].append(step.student.images["cam_high"])
            arrays["student_image_cam_right_wrist"].append(step.student.images["cam_right_wrist"])
            arrays["stage_flags"].append(np.asarray(list(step.flags.as_dict().values()), dtype=np.int8))
            arrays["success"].append(np.asarray(step.flags.success, dtype=np.int8))
            arrays["episode_id"].append(np.asarray(episode_id, dtype=np.int32))
            arrays["step_id"].append(np.asarray(step_id, dtype=np.int32))

    stacked = {key: np.stack(values) for key, values in arrays.items()}
    np.savez_compressed(output_path, **stacked)

    sidecar = output_path.with_suffix(".json")
    summary = dict(metadata)
    summary["episodes"] = len(episodes)
    summary["steps"] = int(sum(len(episode["steps"]) for episode in episodes))
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def mat_to_quat(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / scale
        x = (m[2, 1] - m[1, 2]) * scale
        y = (m[0, 2] - m[2, 0]) * scale
        z = (m[1, 0] - m[0, 1]) * scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    quat = np.asarray([w, x, y, z], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def _object_in_gripper_contacts(env: SO101PickPlaceEnv) -> bool:
    object_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_object_geom")
    finger_geom_ids = _finger_geom_ids(env)
    if object_geom_id == -1 or not finger_geom_ids:
        return False

    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        if contact.geom1 == object_geom_id and contact.geom2 in finger_geom_ids:
            return True
        if contact.geom2 == object_geom_id and contact.geom1 in finger_geom_ids:
            return True
    return False


def _finger_geom_ids(env: SO101PickPlaceEnv) -> set[int]:
    geom_ids: set[int] = set()
    body_markers = ("gripper", "moving_jaw", "camera_mount")
    name_markers = ("jaw", "camera_box", "gripper")
    for geom_id in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(env.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if any(marker in geom_name.lower() for marker in name_markers):
            geom_ids.add(geom_id)
            continue
        if any(marker in body_name.lower() for marker in body_markers):
            if env.model.geom_contype[geom_id] != 0 or env.model.geom_conaffinity[geom_id] != 0:
                geom_ids.add(geom_id)
    return geom_ids
