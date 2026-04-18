from __future__ import annotations

import argparse
import time
from pathlib import Path

from .gl_backend import configure_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the SO-101 pick-and-place scene in the MuJoCo GUI.")
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=None,
        help="Optional path to a specific SO-101 MJCF file.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument(
        "--policy",
        choices=("hold", "random"),
        default="hold",
        help="How to drive the robot while the viewer is open.",
    )
    parser.add_argument(
        "--reset-every",
        type=float,
        default=0.0,
        help="Optional auto-reset period in seconds. 0 disables auto-reset.",
    )
    parser.add_argument(
        "--camera",
        choices=("free", "cam_high", "wrist_cam"),
        default="free",
        help="Launch the viewer using the free camera or one of the named scene cameras.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use the EGL backend instead of opening the MuJoCo GUI.",
    )
    args = parser.parse_args()

    configure_backend(headless=args.headless)
    import mujoco
    import mujoco.viewer
    import numpy as np

    from .envs import SO101PickPlaceEnv

    env = SO101PickPlaceEnv(render_mode="rgb_array", mjcf_path=args.mjcf_path)
    env.reset(seed=args.seed)

    if args.headless:
        print("viewer_headless=1")
        print("Use `python3 -m so101_rl.smoke --headless` for scripted headless checks.")
        env.close()
        return

    print("Launching MuJoCo viewer.")
    print("Scene contents: robot, table, green task object, purple round receptacle, top camera, wrist camera.")
    print("The free camera is interactive. Use the MuJoCo UI to inspect named cameras.")

    next_reset_time = time.monotonic() + args.reset_every if args.reset_every > 0 else None

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.geomgroup[3] = 1
        if args.camera == "free":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = np.array([0.12, 0.0, 0.08], dtype=np.float64)
            viewer.cam.distance = 0.55
            viewer.cam.azimuth = 160
            viewer.cam.elevation = -25
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)

        while viewer.is_running():
            if next_reset_time is not None and time.monotonic() >= next_reset_time:
                env.reset(seed=args.seed)
                next_reset_time = time.monotonic() + args.reset_every

            if args.policy == "random":
                action = env.action_space.sample()
                env.step(action)
            else:
                env.data.ctrl[env._actuator_ids] = env._ctrl_target
                mujoco.mj_step(env.model, env.data)
                env._set_receptacle_pose(env._receptacle_position)
                env.episode_step += 1

            viewer.sync()
            time.sleep(env.model.opt.timestep)

    env.close()


if __name__ == "__main__":
    main()
