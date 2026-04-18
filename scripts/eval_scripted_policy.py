from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from so101_rl.envs import SO101PickPlaceEnv

from pick_place_helpers import ScriptedPickPlacePolicy, build_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple scripted trials and summarize stage metrics.")
    parser.add_argument("--trials", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--steps", type=int, default=220, help="Maximum control steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base reset seed.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the MuJoCo GUI while evaluation runs. Do not use MUJOCO_GL=egl with this mode.",
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
    args = parser.parse_args()

    env = SO101PickPlaceEnv(render_mode="rgb_array", mjcf_path=args.mjcf_path)
    metrics = {
        "reached_object": 0,
        "is_grasping": 0,
        "is_lifted": 0,
        "is_above_bowl": 0,
        "is_in_bowl": 0,
        "success": 0,
    }

    if args.gui:
        print("Launching MuJoCo viewer for scripted policy evaluation.")
        print("Use `unset MUJOCO_GL` before running this mode.")
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            _configure_viewer(viewer, env, args.camera)
            _run_trials(env, args, metrics, viewer)
    else:
        _run_trials(env, args, metrics, viewer=None)

    print(f"trials={args.trials}")
    for key, count in metrics.items():
        print(f"{key}_rate={count / max(args.trials, 1):.3f}")
    env.close()


def _run_trials(
    env: SO101PickPlaceEnv,
    args: argparse.Namespace,
    metrics: dict[str, int],
    viewer: mujoco.viewer.Handle | None,
) -> None:
    for trial_index in range(args.trials):
        if viewer is not None and not viewer.is_running():
            print("viewer_closed=1")
            break

        observation, info = env.reset(seed=args.seed + trial_index)
        policy = ScriptedPickPlacePolicy()
        step = build_step(env, observation, 0.0, False, False, env.action_space.low * 0.0, info)
        peak_flags = {key: False for key in metrics}

        if viewer is not None:
            viewer.sync()
            time.sleep(env.model.opt.timestep)

        for _ in range(args.steps):
            if viewer is not None and not viewer.is_running():
                print("viewer_closed=1")
                return

            action = policy.act(env, step.flags)
            observation, reward, terminated, truncated, info = env.step(action)
            step = build_step(env, observation, reward, terminated, truncated, action, info)
            for key, value in step.flags.as_dict().items():
                peak_flags[key] = peak_flags[key] or value

            if viewer is not None:
                viewer.sync()
                time.sleep(env.model.opt.timestep * env.config.frame_skip)

            if terminated or truncated:
                break

        for key in metrics:
            metrics[key] += int(peak_flags[key])

        print(
            f"trial={trial_index:03d} success={int(peak_flags['success'])} "
            f"reach={int(peak_flags['reached_object'])} grasp={int(peak_flags['is_grasping'])} "
            f"lift={int(peak_flags['is_lifted'])} above={int(peak_flags['is_above_bowl'])} "
            f"in_bowl={int(peak_flags['is_in_bowl'])}"
        )


def _configure_viewer(
    viewer: mujoco.viewer.Handle,
    env: SO101PickPlaceEnv,
    camera_name: str,
) -> None:
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
