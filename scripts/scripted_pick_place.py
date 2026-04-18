from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from so101_rl.envs import SO101PickPlaceEnv

from pick_place_helpers import ScriptedPickPlacePolicy, build_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one scripted pick-and-place rollout for hand-testing.")
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument("--steps", type=int, default=220, help="Maximum control steps.")
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=None,
        help="Optional path to a specific SO-101 MJCF file.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print one status line every N control steps.",
    )
    args = parser.parse_args()

    env = SO101PickPlaceEnv(render_mode="rgb_array", mjcf_path=args.mjcf_path)
    observation, info = env.reset(seed=args.seed)
    policy = ScriptedPickPlacePolicy()

    reward = 0.0
    terminated = False
    truncated = False
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    last_step = build_step(env, observation, reward, terminated, truncated, action, info)
    peak_flags = {key: False for key in last_step.flags.as_dict()}

    for step_index in range(args.steps):
        action = policy.act(env, last_step.flags)
        observation, reward, terminated, truncated, info = env.step(action)
        last_step = build_step(env, observation, reward, terminated, truncated, action, info)

        for key, value in last_step.flags.as_dict().items():
            peak_flags[key] = peak_flags[key] or value

        if step_index % max(args.print_every, 1) == 0 or terminated or truncated:
            object_position = last_step.info["object_position"]
            bowl_position = last_step.info["receptacle_target_position"]
            print(
                f"step={step_index:03d} stage={policy.stage_name} reward={reward:+.3f} "
                f"reach={int(last_step.flags.reached_object)} grasp={int(last_step.flags.is_grasping)} "
                f"lift={int(last_step.flags.is_lifted)} above={int(last_step.flags.is_above_bowl)} "
                f"in_bowl={int(last_step.flags.is_in_bowl)} success={int(last_step.flags.success)} "
                f"object=({object_position[0]:+.3f},{object_position[1]:+.3f},{object_position[2]:+.3f}) "
                f"bowl=({bowl_position[0]:+.3f},{bowl_position[1]:+.3f},{bowl_position[2]:+.3f})"
            )

        if terminated or truncated:
            break

    print(f"peak_flags={peak_flags}")
    print(f"final_success={last_step.flags.success}")
    env.close()


if __name__ == "__main__":
    main()
