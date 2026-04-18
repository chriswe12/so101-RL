from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ..config import PickPlaceEnvConfig
from ..model_source import load_robot_xml_assets, resolve_robot_model_source
from ..scene import build_pick_place_xml


JOINT_NAME_CANDIDATES = (
    ("shoulder_pan", "Rotation"),
    ("shoulder_lift", "Pitch"),
    ("elbow_flex", "Elbow"),
    ("wrist_flex", "Wrist_Pitch"),
    ("wrist_roll", "Wrist_Roll"),
    ("gripper", "Jaw"),
)


class SO101PickPlaceEnv(gym.Env[np.ndarray | dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: PickPlaceEnvConfig | None = None,
        render_mode: str | None = None,
        mjcf_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render_mode={render_mode!r}. Expected None or 'rgb_array'.")

        self.config = config or PickPlaceEnvConfig()
        self.render_mode = render_mode

        source = resolve_robot_model_source(mjcf_path=mjcf_path)
        robot_xml_text, asset_map = load_robot_xml_assets(source)
        task_xml = build_pick_place_xml(robot_xml_text, self.config)

        self.model = mujoco.MjModel.from_xml_string(task_xml, assets=asset_map)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.image_height,
            width=self.config.image_width,
        )

        self.joint_names = tuple(
            _resolve_name(self.model, mujoco.mjtObj.mjOBJ_JOINT, candidates)
            for candidates in JOINT_NAME_CANDIDATES
        )
        self.actuator_names = tuple(
            _resolve_name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidates)
            for candidates in JOINT_NAME_CANDIDATES
        )
        self._joint_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names],
            dtype=np.int32,
        )
        self._actuator_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_names],
            dtype=np.int32,
        )
        self._joint_qpos_adrs = self.model.jnt_qposadr[self._joint_ids]
        self._joint_qvel_adrs = self.model.jnt_dofadr[self._joint_ids]

        self._object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.config.object_joint_name
        )
        self._object_qpos_adr = int(self.model.jnt_qposadr[self._object_joint_id])
        self._object_qvel_adr = int(self.model.jnt_dofadr[self._object_joint_id])

        self._receptacle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.receptacle_body_name
        )
        self._receptacle_mocap_id = int(self.model.body_mocapid[self._receptacle_body_id])

        self._ee_site_name = _resolve_name(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            (self.config.ee_site_name, "ee_site", "gripperframe"),
        )
        self._wrist_camera_name = _resolve_name(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            (self.config.wrist_camera_name, "cam_right_wrist", "wrist_cam"),
        )

        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self._ee_site_name)
        self._object_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.config.object_site_name
        )
        self._receptacle_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.config.receptacle_site_name
        )
        self._fixed_jaw_inner_geom_id = _maybe_resolve_name(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            ("fixed_jaw_box4", "fixed_jaw_box3", "fixed_jaw_box5"),
        )
        self._moving_jaw_inner_geom_id = _maybe_resolve_name(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            ("moving_jaw_box2", "moving_jaw_box3", "moving_jaw_box1"),
        )

        self._home_qpos = np.asarray(self.config.home_qpos, dtype=np.float32)
        self._action_scale = np.asarray(self.config.action_scale, dtype=np.float32)
        self._ctrl_range = np.asarray(self.model.actuator_ctrlrange[self._actuator_ids], dtype=np.float32)
        self._ctrl_target = self._home_qpos.copy()
        self._receptacle_position = np.zeros(3, dtype=np.float32)
        self._grasp_hold_counter = 0
        self._action_indices = self._select_action_indices()

        self.episode_step = 0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self._action_indices),),
            dtype=np.float32,
        )

        float_box = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.joint_names),), dtype=np.float32)
        task_box = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        goal_box = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        image_box = spaces.Box(
            low=0,
            high=255,
            shape=(self.config.image_height, self.config.image_width, 3),
            dtype=np.uint8,
        )

        full_observation_space = spaces.Dict(
            {
                "observation.images.cam_high": image_box,
                "observation.images.cam_right_wrist": image_box,
                "observation.state": float_box,
                "observation.velocity": float_box,
                "observation.task_state": task_box,
                "achieved_goal": goal_box,
                "desired_goal": goal_box,
            }
        )
        teacher_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        if self.config.observation_mode == "teacher":
            self.observation_space = teacher_observation_space
        else:
            self.observation_space = full_observation_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_step = 0
        self._grasp_hold_counter = 0

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0

        joint_noise_scale = self.config.reset_ranges.joint_noise_scale
        if options and "joint_noise_scale" in options:
            joint_noise_scale = float(options["joint_noise_scale"])
        if self.config.task_mode in {"pick_only", "grasp_only", "lift_only"}:
            joint_noise_scale = self.config.pick_reset_joint_noise_scale

        arm_qpos = self._home_qpos.copy()
        if self.config.task_mode in {"pick_only", "grasp_only", "lift_only"}:
            arm_qpos = np.asarray(self.config.pick_home_qpos, dtype=np.float32).copy()
        arm_qpos[:5] += self.np_random.uniform(
            low=-joint_noise_scale,
            high=joint_noise_scale,
            size=5,
        ).astype(np.float32)
        if self.config.task_mode in {"pick_only", "grasp_only", "lift_only"}:
            arm_qpos[-1] = self._ctrl_range[-1, 1]
        arm_qpos = np.clip(arm_qpos, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
        self._ctrl_target = arm_qpos.copy()

        self.data.qpos[self._joint_qpos_adrs] = arm_qpos
        self.data.ctrl[self._actuator_ids] = self._ctrl_target
        mujoco.mj_forward(self.model, self.data)

        object_position = self._coerce_position_option(options, "object_position", "cube_position")
        auto_pick_only_object_reset = False
        if object_position is None:
            if self.config.task_mode in {"pick_only", "grasp_only", "lift_only"}:
                object_position = self._resolve_pick_reset_object_position(self._sample_pick_only_object_position())
                auto_pick_only_object_reset = True
            else:
                object_position = self._sample_object_position()
        self._set_object_pose(object_position)

        receptacle_position = self._coerce_position_option(options, "receptacle_position", "goal_position")
        if receptacle_position is None:
            if self.config.task_mode in {"pick_only", "grasp_only", "lift_only"}:
                receptacle_position = self._default_receptacle_position()
            else:
                receptacle_position = self._sample_receptacle_position(object_position)
        self._set_receptacle_pose(receptacle_position)

        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.config.settle_steps):
            self.data.ctrl[self._actuator_ids] = self._ctrl_target
            mujoco.mj_step(self.model, self.data)
            self._set_receptacle_pose(receptacle_position)
        if self.config.task_mode in {"pick_only", "grasp_only", "lift_only"} and auto_pick_only_object_reset:
            self._set_object_pose(
                self._resolve_pick_reset_object_position(self._sample_pick_only_object_position())
            )
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        return observation, self._build_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}.")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        full_action = np.zeros(len(self.joint_names), dtype=np.float32)
        full_action[self._action_indices] = action
        self._ctrl_target = np.clip(
            self._ctrl_target + (full_action * self._action_scale),
            self._ctrl_range[:, 0],
            self._ctrl_range[:, 1],
        )

        for _ in range(self.config.frame_skip):
            self.data.ctrl[self._actuator_ids] = self._ctrl_target
            mujoco.mj_step(self.model, self.data)
            self._set_receptacle_pose(self._receptacle_position)

        self.episode_step += 1
        info = self._build_info()
        observation = self._get_observation()
        reward = self._compute_reward(info)
        terminated = bool(info["success"])
        truncated = self.episode_step >= self.config.episode_length
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise RuntimeError("render() requires render_mode='rgb_array'.")
        return self._render_camera(self.config.top_camera_name)

    def close(self) -> None:
        self.renderer.close()

    def _get_observation(self) -> np.ndarray | dict[str, np.ndarray]:
        if self.config.observation_mode == "teacher":
            return self._get_teacher_observation()

        object_position = self.data.site_xpos[self._object_site_id].astype(np.float32).copy()
        receptacle_target = self.data.site_xpos[self._receptacle_site_id].astype(np.float32).copy()
        task_state = np.concatenate(
            [object_position, receptacle_target],
            dtype=np.float32,
        )
        return {
            "observation.images.cam_high": self._render_camera(self.config.top_camera_name),
            "observation.images.cam_right_wrist": self._render_camera(self._wrist_camera_name),
            "observation.state": self.data.qpos[self._joint_qpos_adrs].astype(np.float32).copy(),
            "observation.velocity": self.data.qvel[self._joint_qvel_adrs].astype(np.float32).copy(),
            "observation.task_state": task_state,
            "achieved_goal": object_position,
            "desired_goal": receptacle_target,
        }

    def _get_teacher_observation(self) -> np.ndarray:
        info = self._build_info()
        joint_positions = self.data.qpos[self._joint_qpos_adrs].astype(np.float32).copy()
        joint_velocities = self.data.qvel[self._joint_qvel_adrs].astype(np.float32).copy()
        gripper_state = np.asarray([joint_positions[-1], joint_velocities[-1]], dtype=np.float32)

        ee_position = self._current_ee_position()
        ee_quaternion = _mat_to_quat(self.data.site_xmat[self._ee_site_id].reshape(3, 3))
        ee_pose = np.concatenate([ee_position, ee_quaternion], dtype=np.float32)

        object_pose = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 7].astype(np.float32).copy()
        object_velocity = self.data.qvel[self._object_qvel_adr : self._object_qvel_adr + 6].astype(np.float32).copy()
        bowl_pose = np.concatenate(
            [
                self._receptacle_position.astype(np.float32).copy(),
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ],
            dtype=np.float32,
        )
        gripper_to_object = (object_pose[:3] - ee_position).astype(np.float32)
        object_to_bowl = (bowl_pose[:3] - object_pose[:3]).astype(np.float32)
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
        return np.concatenate(
            [
                joint_positions,
                joint_velocities,
                gripper_state,
                ee_pose,
                object_pose,
                object_velocity,
                bowl_pose,
                gripper_to_object,
                object_to_bowl,
                stage_flags,
            ],
            dtype=np.float32,
        )

    def _render_camera(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render().copy()

    def _set_object_pose(self, position: np.ndarray) -> None:
        qpos = self.data.qpos
        qvel = self.data.qvel
        qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = position
        yaw_noise = np.pi
        if self.config.task_mode in {"pick_only", "grasp_only", "lift_only"}:
            yaw_noise = self.config.pick_reset_yaw_noise
        qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = _yaw_to_quat(
            float(self.np_random.uniform(-yaw_noise, yaw_noise))
        )
        qvel[self._object_qvel_adr : self._object_qvel_adr + 6] = 0.0

    def _set_receptacle_pose(self, position: np.ndarray) -> None:
        self._receptacle_position = position.astype(np.float32)
        self.data.mocap_pos[self._receptacle_mocap_id] = self._receptacle_position
        self.data.mocap_quat[self._receptacle_mocap_id] = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

    def _sample_object_position(self) -> np.ndarray:
        table_height = self.config.table_top_z + self.config.object_half_height + 0.001
        return np.asarray(
            [
                self.np_random.uniform(*self.config.reset_ranges.object_x),
                self.np_random.uniform(*self.config.reset_ranges.object_y),
                table_height,
            ],
            dtype=np.float32,
        )

    def _sample_pick_only_object_position(self) -> np.ndarray:
        table_height = self.config.table_top_z + self.config.object_half_height + 0.001
        xy_noise = self.np_random.uniform(
            low=-self.config.pick_reset_xy_noise,
            high=self.config.pick_reset_xy_noise,
            size=2,
        ).astype(np.float32)
        return np.asarray(
            [
                self.config.pick_reset_object_xy[0] + xy_noise[0],
                self.config.pick_reset_object_xy[1] + xy_noise[1],
                table_height,
            ],
            dtype=np.float32,
        )

    def _resolve_pick_reset_object_position(self, base_position: np.ndarray) -> np.ndarray:
        candidate = np.asarray(base_position, dtype=np.float32).copy()
        for _ in range(6):
            self._set_object_pose(candidate)
            mujoco.mj_forward(self.model, self.data)
            if not self._object_in_gripper_contacts():
                return candidate
            candidate[0] -= 0.004
        return candidate

    def _sample_receptacle_position(self, object_position: np.ndarray) -> np.ndarray:
        table_height = self.config.table_top_z
        for _ in range(100):
            candidate = np.asarray(
                [
                    self.np_random.uniform(*self.config.reset_ranges.receptacle_x),
                    self.np_random.uniform(*self.config.reset_ranges.receptacle_y),
                    table_height,
                ],
                dtype=np.float32,
            )
            if (
                np.linalg.norm(candidate[:2] - object_position[:2])
                >= self.config.reset_ranges.min_receptacle_separation
            ):
                return candidate
        return candidate

    def _default_receptacle_position(self) -> np.ndarray:
        return np.asarray([0.22, -0.22, self.config.table_top_z], dtype=np.float32)

    def _build_info(self) -> dict[str, Any]:
        object_position = self.data.site_xpos[self._object_site_id].astype(np.float32).copy()
        receptacle_target = self.data.site_xpos[self._receptacle_site_id].astype(np.float32).copy()
        ee_position = self._current_ee_position()
        object_velocity = self.data.qvel[self._object_qvel_adr : self._object_qvel_adr + 3].astype(np.float32).copy()
        object_speed = float(np.linalg.norm(object_velocity))
        xy_distance = float(np.linalg.norm(object_position[:2] - receptacle_target[:2]))
        z_error = float(abs(object_position[2] - receptacle_target[2]))
        ee_to_object = float(np.linalg.norm(ee_position - object_position))
        lift = float(
            max(
                0.0,
                object_position[2] - (self.config.table_top_z + self.config.object_half_height),
            )
        )
        pre_grasp = object_position.copy()
        pre_grasp[2] += 0.045
        reach_distance = float(np.linalg.norm(ee_position - pre_grasp))
        inside_xy = xy_distance <= self.config.insertion_xy_threshold
        below_rim = object_position[2] <= (
            self.config.table_top_z
            + self.config.receptacle_base_half_height
            + self.config.receptacle_wall_height
            - 0.25 * self.config.object_half_height
        )
        above_floor = object_position[2] >= (
            self.config.table_top_z
            + self.config.receptacle_base_half_height
            + 0.5 * self.config.object_half_height
        )
        is_above_bowl = bool(
            inside_xy
            and object_position[2]
            >= self.config.table_top_z + self.config.receptacle_wall_height + self.config.object_half_height
        )
        is_in_bowl = bool(inside_xy and below_rim and above_floor and object_speed <= self.config.object_still_speed_threshold)
        is_lifted = bool(
            object_position[2]
            >= self.config.table_top_z + self.config.object_half_height + self.config.pick_success_height_margin
        )
        reached_object = bool(reach_distance <= self.config.reach_threshold)
        has_gripper_contact = self._object_in_gripper_contacts()
        gripper_qpos = float(self.data.qpos[self._joint_qpos_adrs[-1]])
        gripper_is_closed = gripper_qpos <= float(self._ctrl_range[-1, 1] - self.config.grasp_closed_margin)
        object_between_jaws = bool(
            abs(float(object_position[0] - ee_position[0])) <= self.config.object_between_jaws_xy_threshold
            and abs(float(object_position[1] - ee_position[1])) <= self.config.object_between_jaws_xy_threshold
            and abs(float(object_position[2] - ee_position[2])) <= self.config.object_between_jaws_z_threshold
        )
        is_grasping = bool(
            has_gripper_contact
            and gripper_is_closed
            and object_between_jaws
        )
        if is_grasping:
            self._grasp_hold_counter += 1
        else:
            self._grasp_hold_counter = 0

        if self.config.task_mode == "grasp_only":
            success = bool(self._grasp_hold_counter >= 1)
        elif self.config.task_mode == "lift_only":
            success = bool(is_lifted and self._grasp_hold_counter >= self.config.grasp_hold_steps)
        elif self.config.task_mode == "pick_only":
            success = bool(is_lifted and self._grasp_hold_counter >= self.config.grasp_hold_steps)
        else:
            success = is_in_bowl
        return {
            "is_success": success,
            "success": success,
            "reached_object": reached_object,
            "is_grasping": is_grasping,
            "is_lifted": is_lifted,
            "is_above_bowl": is_above_bowl,
            "is_in_bowl": is_in_bowl,
            "object_between_jaws": object_between_jaws,
            "has_gripper_contact": has_gripper_contact,
            "gripper_is_closed": gripper_is_closed,
            "object_to_receptacle_xy_distance": xy_distance,
            "object_to_receptacle_z_error": z_error,
            "ee_to_object_distance": ee_to_object,
            "reach_distance": reach_distance,
            "object_lift": lift,
            "object_inside_receptacle": bool(inside_xy),
            "object_speed": object_speed,
            "grasp_hold_steps": self._grasp_hold_counter,
            "gripper_qpos": gripper_qpos,
            "receptacle_position": self._receptacle_position.copy(),
            "receptacle_target_position": receptacle_target,
            "object_position": object_position,
            "ee_position": ee_position,
        }

    def _compute_reward(self, info: dict[str, Any]) -> float:
        if not self.config.dense_reward:
            return float(info["success"])

        gripper_open_fraction = float(
            (self._ctrl_range[-1, 1] - info["gripper_qpos"])
            / max(self._ctrl_range[-1, 1] - self._ctrl_range[-1, 0], 1e-6)
        )

        if self.config.task_mode == "grasp_only":
            reward = -0.20 * info["ee_to_object_distance"]
            if info["object_between_jaws"]:
                reward += 0.9
                reward += 1.2 * self.config.close_bonus_scale * gripper_open_fraction
            if info["has_gripper_contact"]:
                reward += 1.0
            if info["gripper_is_closed"]:
                reward += 1.25
            if info["is_grasping"]:
                reward += 1.5
            if info["success"]:
                reward += 4.0
            return float(reward)

        if self.config.task_mode == "lift_only":
            reward = -0.15 * info["ee_to_object_distance"]
            if info["object_between_jaws"]:
                reward += 0.8
                reward += self.config.close_bonus_scale * gripper_open_fraction
            if info["has_gripper_contact"]:
                reward += 1.0
            if info["gripper_is_closed"]:
                reward += 1.5
            if info["is_grasping"]:
                reward += 2.0
                reward += 3.0 * min(info["object_lift"], self.config.pick_success_height_margin)
                if info["is_lifted"]:
                    reward += 2.0
            if info["success"]:
                reward += 5.0
            return float(reward)

        if self.config.task_mode == "pick_only":
            reward = -0.35 * info["ee_to_object_distance"]
            reward -= 0.02 * np.linalg.norm(self._ctrl_target - self.data.qpos[self._joint_qpos_adrs])
            if info["object_between_jaws"]:
                reward += 0.8
                reward += 0.8 * self.config.close_bonus_scale * gripper_open_fraction
            if info["reached_object"]:
                reward += 0.25
            if info["has_gripper_contact"]:
                reward += 0.8
            if info["gripper_is_closed"]:
                reward += 1.0
            if info["is_grasping"]:
                reward += 1.0
                reward += 3.0 * min(info["object_lift"], self.config.pick_success_height_margin)
                if info["is_lifted"]:
                    reward += 3.0
            if info["success"]:
                reward += 5.0
            return float(reward)

        reward = -1.2 * info["object_to_receptacle_xy_distance"]
        reward -= 0.4 * info["object_to_receptacle_z_error"]
        reward -= 0.15 * info["ee_to_object_distance"]
        reward += min(info["object_lift"], self.config.lift_height)
        if info["object_inside_receptacle"]:
            reward += 1.5
        if info["success"]:
            reward += 5.0
        return float(reward)

    def _coerce_position_option(
        self,
        options: dict[str, Any] | None,
        *keys: str,
    ) -> np.ndarray | None:
        if not options:
            return None
        value = None
        for key in keys:
            if key in options:
                value = options[key]
                break
        if value is None:
            return None
        raw = np.asarray(value, dtype=np.float32)
        if raw.shape != (3,):
            joined = ", ".join(keys)
            raise ValueError(f"{joined} must have shape (3,), got {raw.shape}.")
        return raw

    def _select_action_indices(self) -> np.ndarray:
        if self.config.action_mode == "full":
            return np.arange(len(self.joint_names), dtype=np.int32)
        if self.config.action_mode == "gripper_only":
            return np.asarray([len(self.joint_names) - 1], dtype=np.int32)
        if self.config.action_mode == "lift_subset":
            return np.asarray([1, 2, 3, len(self.joint_names) - 1], dtype=np.int32)
        raise ValueError(f"Unsupported action_mode={self.config.action_mode!r}")

    def _object_in_gripper_contacts(self) -> bool:
        object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "task_object_geom")
        if object_geom_id == -1:
            return False

        finger_geom_ids = set()
        body_markers = ("gripper", "moving_jaw", "camera_mount")
        name_markers = ("jaw", "camera_box", "gripper")
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if any(marker in geom_name.lower() for marker in name_markers):
                finger_geom_ids.add(geom_id)
                continue
            if any(marker in body_name.lower() for marker in body_markers):
                if self.model.geom_contype[geom_id] != 0 or self.model.geom_conaffinity[geom_id] != 0:
                    finger_geom_ids.add(geom_id)

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.geom1 == object_geom_id and contact.geom2 in finger_geom_ids:
                return True
            if contact.geom2 == object_geom_id and contact.geom1 in finger_geom_ids:
                return True
        return False

    def _current_ee_position(self) -> np.ndarray:
        site_position = self.data.site_xpos[self._ee_site_id].astype(np.float32).copy()
        if self.config.ee_reward_closing_offset == 0.0:
            return site_position
        if self._fixed_jaw_inner_geom_id == -1 or self._moving_jaw_inner_geom_id == -1:
            return site_position

        fixed_point = self.data.geom_xpos[self._fixed_jaw_inner_geom_id]
        moving_point = self.data.geom_xpos[self._moving_jaw_inner_geom_id]
        closing_direction = fixed_point - moving_point
        closing_norm = float(np.linalg.norm(closing_direction))
        if closing_norm < 1e-8:
            return site_position

        closing_direction = closing_direction / closing_norm
        return site_position + closing_direction.astype(np.float32) * self.config.ee_reward_closing_offset



def _yaw_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _mat_to_quat(matrix: np.ndarray) -> np.ndarray:
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


def _resolve_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if mujoco.mj_name2id(model, obj_type, name) != -1:
            return name
    tried = ", ".join(candidates)
    raise ValueError(f"Could not resolve MuJoCo object for any of: {tried}")


def _maybe_resolve_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, candidates: tuple[str, ...]) -> int:
    for name in candidates:
        obj_id = mujoco.mj_name2id(model, obj_type, name)
        if obj_id != -1:
            return obj_id
    return -1
