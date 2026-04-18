from __future__ import annotations

import argparse
from pathlib import Path

from so101_rl.envs import SO101PickPlaceEnv

from pick_place_helpers import ScriptedPickPlacePolicy, build_step, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect scripted teacher rollouts with RGB, proprio, actions, and stage labels."
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of rollout episodes to collect.")
    parser.add_argument("--steps", type=int, default=220, help="Maximum control steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base reset seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scripted_teacher_dataset.npz"),
        help="Output .npz path.",
    )
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Keep only episodes that achieved success.",
    )
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=None,
        help="Optional path to a specific SO-101 MJCF file.",
    )
    args = parser.parse_args()

    env = SO101PickPlaceEnv(render_mode="rgb_array", mjcf_path=args.mjcf_path)
    accepted_episodes: list[dict[str, object]] = []

    for episode_index in range(args.episodes):
        observation, info = env.reset(seed=args.seed + episode_index)
        policy = ScriptedPickPlacePolicy()
        steps = [build_step(env, observation, 0.0, False, False, env.action_space.low * 0.0, info)]

        for _ in range(args.steps):
            action = policy.act(env, steps[-1].flags)
            observation, reward, terminated, truncated, info = env.step(action)
            steps.append(build_step(env, observation, reward, terminated, truncated, action, info))
            if terminated or truncated:
                break

        success = any(step.flags.success for step in steps)
        if not args.successful_only or success:
            accepted_episodes.append({"steps": steps, "success": success})

        print(
            f"episode={episode_index:03d} accepted={int((not args.successful_only) or success)} "
            f"success={int(success)} steps={len(steps)}"
        )

    metadata = {
        "base_seed": args.seed,
        "requested_episodes": args.episodes,
        "successful_only": args.successful_only,
    }
    output_path = save_dataset(args.output, accepted_episodes, metadata)
    print(f"saved_dataset={output_path}")
    env.close()


if __name__ == "__main__":
    main()
