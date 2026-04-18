from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .gl_backend import configure_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the SO-101 MuJoCo pick-and-place env.")
    parser.add_argument("--steps", type=int, default=100, help="Number of control steps to run.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument(
        "--policy",
        choices=("zero", "random"),
        default="zero",
        help="Simple control policy for the smoke rollout.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Optional .npz output path for saving the episode rollout.",
    )
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=None,
        help="Optional path to so_arm100.xml if you do not want to use environment variables.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use the EGL backend for headless execution.",
    )
    args = parser.parse_args()

    configure_backend(headless=args.headless)
    from .envs import SO101PickPlaceEnv
    from .record import EpisodeRecorder

    env = SO101PickPlaceEnv(render_mode="rgb_array", mjcf_path=args.mjcf_path)
    recorder = EpisodeRecorder() if args.record else None

    observation, info = env.reset(seed=args.seed)
    total_reward = 0.0

    for _ in range(args.steps):
        if args.policy == "random":
            action = env.action_space.sample()
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)

        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if recorder is not None:
            recorder.add(next_observation, action, reward, terminated, truncated, info)

        observation = next_observation
        if terminated or truncated:
            break

    if recorder is not None and args.record is not None:
        recorder.save(
            args.record,
            metadata={
                "seed": args.seed,
                "policy": args.policy,
                "total_reward": total_reward,
            },
        )

    print(
        f"steps={env.episode_step} total_reward={total_reward:.3f} "
        f"success={info.get('is_success', False)}"
    )
    print(f"obs_keys={sorted(observation.keys())}")
    env.close()


if __name__ == "__main__":
    main()
