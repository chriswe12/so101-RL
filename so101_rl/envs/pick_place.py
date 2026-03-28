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


class SO101PickPlaceEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
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

        self._home_qpos = np.asarray(self.config.home_qpos, dtype=np.float32)
        self._action_scale = np.asarray(self.config.action_scale, dtype=np.float32)
        self._ctrl_range = np.asarray(self.model.actuator_ctrlrange[self._actuator_ids], dtype=np.float32)
        self._ctrl_target = self._home_qpos.copy()
        self._receptacle_position = np.zeros(3, dtype=np.float32)

        self.episode_step = 0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.joint_names),),
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

        self.observation_space = spaces.Dict(
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_step = 0

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0

        joint_noise_scale = self.config.reset_ranges.joint_noise_scale
        if options and "joint_noise_scale" in options:
            joint_noise_scale = float(options["joint_noise_scale"])

        arm_qpos = self._home_qpos.copy()
        arm_qpos[:5] += self.np_random.uniform(
            low=-joint_noise_scale,
            high=joint_noise_scale,
            size=5,
        ).astype(np.float32)
        arm_qpos = np.clip(arm_qpos, self._ctrl_range[:, 0], self._ctrl_range[:, 1])
        self._ctrl_target = arm_qpos.copy()

        self.data.qpos[self._joint_qpos_adrs] = arm_qpos
        self.data.ctrl[self._actuator_ids] = self._ctrl_target

        object_position = self._coerce_position_option(options, "object_position", "cube_position")
        if object_position is None:
            object_position = self._sample_object_position()
        self._set_object_pose(object_position)

        receptacle_position = self._coerce_position_option(options, "receptacle_position", "goal_position")
        if receptacle_position is None:
            receptacle_position = self._sample_receptacle_position(object_position)
        self._set_receptacle_pose(receptacle_position)

        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.config.settle_steps):
            self.data.ctrl[self._actuator_ids] = self._ctrl_target
            mujoco.mj_step(self.model, self.data)
            self._set_receptacle_pose(receptacle_position)
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        return observation, self._build_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}.")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._ctrl_target = np.clip(
            self._ctrl_target + (action * self._action_scale),
            self._ctrl_range[:, 0],
            self._ctrl_range[:, 1],
        )

        for _ in range(self.config.frame_skip):
            self.data.ctrl[self._actuator_ids] = self._ctrl_target
            mujoco.mj_step(self.model, self.data)
            self._set_receptacle_pose(self._receptacle_position)

        self.episode_step += 1
        observation = self._get_observation()
        info = self._build_info()
        reward = self._compute_reward(info)
        terminated = bool(info["is_success"])
        truncated = self.episode_step >= self.config.episode_length
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise RuntimeError("render() requires render_mode='rgb_array'.")
        return self._render_camera(self.config.top_camera_name)

    def close(self) -> None:
        self.renderer.close()

    def _get_observation(self) -> dict[str, np.ndarray]:
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

    def _render_camera(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render().copy()

    def _set_object_pose(self, position: np.ndarray) -> None:
        qpos = self.data.qpos
        qvel = self.data.qvel
        qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = position
        qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = _yaw_to_quat(
            float(self.np_random.uniform(-np.pi, np.pi))
        )
        qvel[self._object_qvel_adr : self._object_qvel_adr + 6] = 0.0

    def _set_receptacle_pose(self, position: np.ndarray) -> None:
        self._receptacle_position = position.astype(np.float32)
        self.data.mocap_pos[self._receptacle_mocap_id] = self._receptacle_position
        self.data.mocap_quat[self._receptacle_mocap_id] = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

    def _sample_object_position(self) -> np.ndarray:
        table_height = self.config.table_top_z + self.config.object_half_height + 0.01
        return np.asarray(
            [
                self.np_random.uniform(*self.config.reset_ranges.object_x),
                self.np_random.uniform(*self.config.reset_ranges.object_y),
                table_height,
            ],
            dtype=np.float32,
        )

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

    def _build_info(self) -> dict[str, Any]:
        object_position = self.data.site_xpos[self._object_site_id].astype(np.float32).copy()
        receptacle_target = self.data.site_xpos[self._receptacle_site_id].astype(np.float32).copy()
        ee_position = self.data.site_xpos[self._ee_site_id].astype(np.float32).copy()
        xy_distance = float(np.linalg.norm(object_position[:2] - receptacle_target[:2]))
        z_error = float(abs(object_position[2] - receptacle_target[2]))
        ee_to_object = float(np.linalg.norm(ee_position - object_position))
        lift = float(max(0.0, object_position[2] - self.config.table_top_z))
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
        is_success = bool(inside_xy and below_rim and above_floor)
        return {
            "is_success": is_success,
            "object_to_receptacle_xy_distance": xy_distance,
            "object_to_receptacle_z_error": z_error,
            "ee_to_object_distance": ee_to_object,
            "object_lift": lift,
            "object_inside_receptacle": bool(inside_xy),
            "receptacle_position": self._receptacle_position.copy(),
            "receptacle_target_position": receptacle_target,
            "object_position": object_position,
            "ee_position": ee_position,
        }

    def _compute_reward(self, info: dict[str, Any]) -> float:
        if not self.config.dense_reward:
            return float(info["is_success"])

        reward = -1.2 * info["object_to_receptacle_xy_distance"]
        reward -= 0.4 * info["object_to_receptacle_z_error"]
        reward -= 0.15 * info["ee_to_object_distance"]
        reward += min(info["object_lift"], self.config.lift_height)
        if info["object_inside_receptacle"]:
            reward += 1.5
        if info["is_success"]:
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


def _yaw_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _resolve_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if mujoco.mj_name2id(model, obj_type, name) != -1:
            return name
    tried = ", ".join(candidates)
    raise ValueError(f"Could not resolve MuJoCo object for any of: {tried}")
