from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from so101_rl.envs import SO101PickPlaceEnv

from pick_place_helpers import student_observation, teacher_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect teacher and student observation layouts.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=None,
        help="Optional path to a specific SO-101 MJCF file.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional .npz path for saving one observation sample.",
    )
    args = parser.parse_args()

    env = SO101PickPlaceEnv(render_mode="rgb_array", mjcf_path=args.mjcf_path)
    observation, _ = env.reset(seed=args.seed)

    teacher = teacher_observation(env)
    student = student_observation(observation, teacher)

    print(f"teacher_vector_shape={teacher.vector.shape}")
    for key, value in teacher.fields.items():
        print(f"teacher.{key}.shape={value.shape}")

    for key, image in student.images.items():
        print(f"student.images.{key}.shape={image.shape}")
    for key, value in student.proprio.items():
        print(f"student.proprio.{key}.shape={value.shape}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save,
            teacher_vector=teacher.vector,
            teacher_joint_positions=teacher.fields["joint_positions"],
            teacher_joint_velocities=teacher.fields["joint_velocities"],
            teacher_gripper_state=teacher.fields["gripper_state"],
            teacher_end_effector_pose=teacher.fields["end_effector_pose"],
            teacher_object_pose=teacher.fields["object_pose"],
            teacher_object_velocity=teacher.fields["object_velocity"],
            teacher_bowl_pose=teacher.fields["bowl_pose"],
            teacher_gripper_to_object=teacher.fields["gripper_to_object"],
            teacher_object_to_bowl=teacher.fields["object_to_bowl"],
            student_image_cam_high=student.images["cam_high"],
            student_image_cam_right_wrist=student.images["cam_right_wrist"],
            student_joint_positions=student.proprio["joint_positions"],
            student_joint_velocities=student.proprio["joint_velocities"],
            student_gripper_state=student.proprio["gripper_state"],
        )
        print(f"saved_sample={args.save}")

    env.close()


if __name__ == "__main__":
    main()
