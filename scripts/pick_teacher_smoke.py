from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from so101_rl.gl_backend import configure_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the pick-only teacher-observation environment.")
    parser.add_argument("--steps", type=int, default=20, help="Number of control steps to run.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument(
        "--policy",
        choices=("zero", "random"),
        default="zero",
        help="Simple control policy for the smoke rollout.",
    )
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=None,
        help="Optional path to a specific SO-101 MJCF file.",
    )
    parser.add_argument(
        "--task-mode",
        choices=("grasp_only", "lift_only", "pick_only", "pick_place"),
        default="pick_only",
        help="Task mode to instantiate.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("gripper_only", "lift_subset", "full"),
        default="full",
        help="Action mode to instantiate.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use the EGL backend for headless execution.",
    )
    args = parser.parse_args()

    configure_backend(headless=args.headless)
    from so101_rl.config import PickPlaceEnvConfig
    from so101_rl.envs import SO101PickPlaceEnv

    config = PickPlaceEnvConfig(
        task_mode=args.task_mode,
        observation_mode="teacher",
        action_mode=args.action_mode,
    )
    env = SO101PickPlaceEnv(config=config, render_mode="rgb_array", mjcf_path=args.mjcf_path)
    observation, info = env.reset(seed=args.seed)
    total_reward = 0.0

    for _ in range(args.steps):
        if args.policy == "random":
            action = env.action_space.sample()
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"obs_shape={observation.shape}")
    print(
        f"steps={env.episode_step} total_reward={total_reward:.3f} "
        f"success={info['success']} reach={info['reached_object']} between={info['object_between_jaws']} "
        f"closed={info['gripper_is_closed']} contact={info['has_gripper_contact']} "
        f"grasp={info['is_grasping']} lift={info['is_lifted']}"
    )
    env.close()


if __name__ == "__main__":
    main()
