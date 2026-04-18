from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal


@dataclass(frozen=True)
class ResetRanges:
    object_x: tuple[float, float] = (0.10, 0.22)
    object_y: tuple[float, float] = (-0.10, 0.10)
    receptacle_x: tuple[float, float] = (0.12, 0.25)
    receptacle_y: tuple[float, float] = (-0.25, -0.12)
    joint_noise_scale: float = 0.02
    min_receptacle_separation: float = 0.08


@dataclass(frozen=True)
class PickPlaceEnvConfig:
    task_mode: Literal["pick_place", "pick_only", "grasp_only", "lift_only"] = "pick_place"
    observation_mode: Literal["full", "teacher"] = "full"
    action_mode: Literal["full", "gripper_only", "lift_subset"] = "full"
    timestep: float = 0.002
    solver_integrator: str = "implicitfast"
    solver_type: str = "Newton"
    solver_iterations: int = 30
    solver_ls_iterations: int = 60
    solver_impratio: float = 10.0
    frame_skip: int = 10
    episode_length: int = 200
    settle_steps: int = 40
    lift_height: float = 0.035
    grasp_hold_steps: int = 5
    pick_reset_joint_noise_scale: float = 0.005
    pick_reset_xy_noise: float = 0.004
    pick_reset_yaw_noise: float = 0.15
    pick_reset_forward_offset: float = 0.045
    pick_reset_object_xy: tuple[float, float] = (0.185, 0.0)

    image_width: int = 640
    image_height: int = 480
    top_camera_name: str = "cam_high"
    wrist_camera_name: str = "wrist_cam"
    top_camera_fovy: float = 40.0
    wrist_camera_fovy: float = 70.0

    ee_site_name: str = "gripperframe"
    object_site_name: str = "object_site"
    receptacle_site_name: str = "receptacle_site"
    object_joint_name: str = "object_freejoint"
    receptacle_body_name: str = "target_receptacle"

    table_pos: tuple[float, float, float] = (0.19, 0.0, -0.03)
    table_size: tuple[float, float, float] = (0.48, 0.40, 0.02)
    table_friction: tuple[float, float, float] = (0.95, 0.02, 0.001)
    object_radius: float = 0.014
    object_half_height: float = 0.014
    object_mass: float = 0.030
    object_diaginertia: tuple[float, float, float] | None = None
    object_friction: tuple[float, float, float] = (1.0, 5e-3, 5e-4)
    object_condim: int = 6
    object_rgba: tuple[float, float, float, float] = (0.12, 0.78, 0.28, 1.0)
    contact_solref: tuple[float, float] = (0.01, 1.0)
    contact_solimp: tuple[float, float, float, float, float] | None = None
    gripper_friction: tuple[float, float, float] = (1.0, 5e-3, 5e-4)
    gripper_condim: int = 6
    receptacle_inner_radius: float = 0.045
    receptacle_wall_thickness: float = 0.004
    receptacle_wall_height: float = 0.055
    receptacle_base_half_height: float = 0.004
    receptacle_segments: int = 12
    receptacle_rgba: tuple[float, float, float, float] = (0.62, 0.24, 0.84, 1.0)
    insertion_margin: float = 0.004
    robot_arm_rgba: tuple[float, float, float, float] = (0.08, 0.08, 0.08, 1.0)
    robot_gripper_rgba: tuple[float, float, float, float] = (0.95, 0.95, 0.95, 1.0)
    overhead_mount_body_name: str = "overhead_cam_mount"
    overhead_mount_mesh_scale: float = 0.001
    overhead_mount_pos: tuple[float, float, float] = (0, 0.0, 0.0)
    overhead_mount_quat: tuple[float, float, float, float] = (0.707107, 0.707107, 0.0, 0.0)
    # Each STL is on its own child body of overhead_mount_body_name; pos/quat are in the mount frame.
    overhead_frame_arm_base_pos: tuple[float, float, float] = (0.075, -0.0025, 0.0525)
    overhead_frame_arm_base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    overhead_frame_cam_bottom_pos: tuple[float, float, float] = (-0.019, -0.0075, 0.075)
    overhead_frame_cam_bottom_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    overhead_frame_cam_middle_pos: tuple[float, float, float] = (0.0, 0.18, 0.112)
    overhead_frame_cam_middle_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    overhead_frame_cam_top_pos: tuple[float, float, float] = (0.0, 0.18, 0.112)
    overhead_frame_cam_top_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    top_camera_mount_frame_name: str = "overhead_frame_cam_top"
    # cam_mount_top.stl spans about 58.0 x 182.3 x 36.6 mm; +Y is vertical once mounted.
    top_camera_pos_in_mount: tuple[float, float, float] = (0.016313, 0.377338, -0.0005)
    top_camera_forward: tuple[float, float, float] = (0.422618, 0.0, -0.906308)

    home_qpos: tuple[float, ...] = (0.0, -1.57, 1.57, 1.57, -1.57, 0.0)
    pick_home_qpos: tuple[float, ...] = (-0.2575, -1.745, 1.595, 1.041, -1.42, 1.74533)
    action_scale: tuple[float, ...] = (0.08, 0.08, 0.08, 0.12, 0.12, 0.08)
    reset_ranges: ResetRanges = field(default_factory=ResetRanges)

    dense_reward: bool = True
    reach_threshold: float = 0.035
    grasp_lift_threshold: float = 0.02
    pick_success_height_margin: float = 0.06
    object_still_speed_threshold: float = 0.15
    grasp_closed_margin: float = 0.05
    ee_reward_closing_offset: float = 0.008
    object_between_jaws_xy_threshold: float = 0.015
    object_between_jaws_z_threshold: float = 0.03
    close_bonus_scale: float = 0.6

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
    def object_planar_extent(self) -> float:
        return math.sqrt(2.0) * self.object_radius

    @property
    def resolved_object_diaginertia(self) -> tuple[float, float, float]:
        if self.object_diaginertia is not None:
            return self.object_diaginertia

        full_x = 2.0 * self.object_radius
        full_y = 2.0 * self.object_radius
        full_z = 2.0 * self.object_half_height
        mass = self.object_mass
        ix = mass * (full_y * full_y + full_z * full_z) / 12.0
        iy = mass * (full_x * full_x + full_z * full_z) / 12.0
        iz = mass * (full_x * full_x + full_y * full_y) / 12.0
        return (ix, iy, iz)

    @property
    def insertion_xy_threshold(self) -> float:
        return max(0.002, self.receptacle_inner_radius - self.object_planar_extent - self.insertion_margin)
