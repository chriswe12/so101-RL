from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResetRanges:
    object_x: tuple[float, float] = (0.10, 0.22)
    object_y: tuple[float, float] = (-0.10, 0.10)
    receptacle_x: tuple[float, float] = (0.12, 0.25)
    receptacle_y: tuple[float, float] = (-0.12, 0.12)
    joint_noise_scale: float = 0.02
    min_receptacle_separation: float = 0.08


@dataclass(frozen=True)
class PickPlaceEnvConfig:
    timestep: float = 0.002
    frame_skip: int = 10
    episode_length: int = 200
    settle_steps: int = 40
    lift_height: float = 0.035

    image_width: int = 128
    image_height: int = 128
    top_camera_name: str = "cam_high"
    wrist_camera_name: str = "wrist_cam"

    ee_site_name: str = "gripperframe"
    object_site_name: str = "object_site"
    receptacle_site_name: str = "receptacle_site"
    object_joint_name: str = "object_freejoint"
    receptacle_body_name: str = "target_receptacle"

    table_pos: tuple[float, float, float] = (0.19, 0.0, -0.03)
    table_size: tuple[float, float, float] = (0.24, 0.20, 0.02)
    object_radius: float = 0.014
    object_half_height: float = 0.02
    object_rgba: tuple[float, float, float, float] = (0.12, 0.78, 0.28, 1.0)
    receptacle_inner_radius: float = 0.045
    receptacle_wall_thickness: float = 0.004
    receptacle_wall_height: float = 0.055
    receptacle_base_half_height: float = 0.004
    receptacle_segments: int = 12
    receptacle_rgba: tuple[float, float, float, float] = (0.62, 0.24, 0.84, 1.0)
    insertion_margin: float = 0.004
    robot_arm_rgba: tuple[float, float, float, float] = (0.08, 0.08, 0.08, 1.0)
    robot_gripper_rgba: tuple[float, float, float, float] = (0.95, 0.95, 0.95, 1.0)

    home_qpos: tuple[float, ...] = (0.0, -1.57, 1.57, 1.57, -1.57, 0.0)
    action_scale: tuple[float, ...] = (0.08, 0.08, 0.08, 0.12, 0.12, 0.08)
    reset_ranges: ResetRanges = field(default_factory=ResetRanges)

    dense_reward: bool = True

    @property
    def control_hz(self) -> float:
        return 1.0 / (self.timestep * self.frame_skip)

    @property
    def table_top_z(self) -> float:
        return self.table_pos[2] + self.table_size[2]

    @property
    def receptacle_outer_radius(self) -> float:
        return self.receptacle_inner_radius + self.receptacle_wall_thickness

    @property
    def insertion_xy_threshold(self) -> float:
        return max(0.002, self.receptacle_inner_radius - self.object_radius - self.insertion_margin)
