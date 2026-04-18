from __future__ import annotations

import argparse
import time
from pathlib import Path

from so101_rl.gl_backend import configure_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO teacher policy on the pick-only task.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a Stable-Baselines3 PPO .zip file.")
    parser.add_argument("--trials", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--steps", type=int, default=220, help="Maximum control steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base reset seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions for policy inference.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the MuJoCo GUI while evaluating. Do not use MUJOCO_GL=egl with this mode.",
    )
    parser.add_argument(
        "--camera",
        choices=("free", "cam_high", "wrist_cam"),
        default="free",
        help="Viewer camera when running with --gui.",
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
        help="Task mode to instantiate. Must match the checkpoint training setup.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("gripper_only", "lift_subset", "full"),
        default="full",
        help="Action mode to instantiate. Must match the checkpoint training setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_backend(headless=not args.gui)
    import mujoco
    import mujoco.viewer
    import numpy as np

    from so101_rl.config import PickPlaceEnvConfig
    from so101_rl.envs import SO101PickPlaceEnv

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is required for PPO evaluation. "
            "Install it with `python3 -m pip install -e .[training]`."
        ) from exc

    config = PickPlaceEnvConfig(
        task_mode=args.task_mode,
        observation_mode="teacher",
        action_mode=args.action_mode,
    )
    env = SO101PickPlaceEnv(config=config, render_mode="rgb_array", mjcf_path=args.mjcf_path)
    model = PPO.load(str(args.checkpoint), device="auto")

    print(f"eval_task_mode={args.task_mode}")
    print(f"eval_action_mode={args.action_mode}")
    print(f"env_action_shape={env.action_space.shape}")

    metrics = {
        "success": 0,
        "reached_object": 0,
        "is_grasping": 0,
        "is_lifted": 0,
    }

    if args.gui:
        print("Launching MuJoCo viewer for PPO policy evaluation.")
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            _configure_viewer(viewer, env, args.camera)
            _run_trials(env, model, args, metrics, viewer)
    else:
        _run_trials(env, model, args, metrics, viewer=None)

    print(f"trials={args.trials}")
    for key, count in metrics.items():
        print(f"{key}_rate={count / max(args.trials, 1):.3f}")
    env.close()


def _run_trials(
    env,
    model,
    args: argparse.Namespace,
    metrics: dict[str, int],
    viewer,
) -> None:
    for trial_index in range(args.trials):
        if viewer is not None and not viewer.is_running():
            print("viewer_closed=1")
            break

        observation, info = env.reset(seed=args.seed + trial_index)
        peak = {key: False for key in metrics}

        if viewer is not None:
            viewer.sync()
            time.sleep(env.model.opt.timestep)

        for step_index in range(args.steps):
            if viewer is not None and not viewer.is_running():
                print("viewer_closed=1")
                return

            action, _ = model.predict(observation, deterministic=args.deterministic)
            observation, reward, terminated, truncated, info = env.step(action)

            for key in peak:
                peak[key] = peak[key] or bool(info[key])

            if viewer is not None:
                viewer.sync()
                time.sleep(env.model.opt.timestep * env.config.frame_skip)

            if terminated or truncated:
                break

        for key in metrics:
            metrics[key] += int(peak[key])

        print(
            f"trial={trial_index:03d} success={int(peak['success'])} "
            f"reach={int(peak['reached_object'])} grasp={int(peak['is_grasping'])} "
            f"lift={int(peak['is_lifted'])}"
        )


def _configure_viewer(
    viewer,
    env,
    camera_name: str,
) -> None:
    import mujoco
    import numpy as np

    if camera_name == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.12, 0.0, 0.08], dtype=np.float64)
        viewer.cam.distance = 0.55
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -25
        return

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)


if __name__ == "__main__":
    main()
