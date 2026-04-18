from __future__ import annotations

import argparse
from pathlib import Path

from so101_rl.envs import SO101PickPlaceEnv

from pick_place_helpers import control_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the current SO-101 control interface and timing.")
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=None,
        help="Optional path to a specific SO-101 MJCF file.",
    )
    args = parser.parse_args()

    env = SO101PickPlaceEnv(render_mode="rgb_array", mjcf_path=args.mjcf_path)
    spec = control_spec(env)

    print("control_interface=joint_position_deltas")
    print(f"policy_action_shape={env.action_space.shape}")
    print(f"joint_names={env.joint_names}")
    print(f"action_scale={env.config.action_scale}")
    print(f"sim_timestep={spec.sim_timestep:.6f}")
    print(f"sim_hz={spec.sim_hz:.2f}")
    print(f"policy_hz={spec.policy_hz:.2f}")
    print(f"frame_skip={spec.frame_skip}")
    print(f"action_repeat={spec.action_repeat}")
    print("controller=q_target = q_current + clamp(delta_q)")
    print("low_level_controller=MuJoCo position actuators tracking q_target")
    env.close()


if __name__ == "__main__":
    main()
