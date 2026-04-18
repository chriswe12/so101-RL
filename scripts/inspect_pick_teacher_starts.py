from __future__ import annotations

import argparse
import time
from pathlib import Path

from so101_rl.gl_backend import configure_backend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a sequence of initial pick-only teacher reset scenes in the MuJoCo GUI."
    )
    parser.add_argument("--seed", type=int, default=0, help="Base reset seed.")
    parser.add_argument("--scenes", type=int, default=5, help="Number of reset scenes to inspect.")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.5,
        help="How long to hold each reset scene before advancing.",
    )
    parser.add_argument(
        "--camera",
        choices=("free", "cam_high", "wrist_cam"),
        default="free",
        help="Viewer camera to use.",
    )
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=None,
        help="Optional path to a specific SO-101 MJCF file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use the EGL backend and print reset states without opening the GUI.",
    )
    args = parser.parse_args()

    configure_backend(headless=args.headless)
    import mujoco
    import mujoco.viewer
    import numpy as np

    from so101_rl.config import PickPlaceEnvConfig
    from so101_rl.envs import SO101PickPlaceEnv

    config = PickPlaceEnvConfig(task_mode="pick_only", observation_mode="teacher")
    env = SO101PickPlaceEnv(config=config, render_mode="rgb_array", mjcf_path=args.mjcf_path)

    if args.headless:
        for scene_index in range(args.scenes):
            _, info = env.reset(seed=args.seed + scene_index)
            object_position = info["object_position"]
            ee_position = info["ee_position"]
            gripper_qpos = float(env.data.qpos[env._joint_qpos_adrs[-1]])
            xy_offset = object_position[:2] - ee_position[:2]
            print(
                f"scene={scene_index:03d} seed={args.seed + scene_index} "
                f"gripper_qpos={gripper_qpos:+.4f} "
                f"ee=({ee_position[0]:+.4f},{ee_position[1]:+.4f},{ee_position[2]:+.4f}) "
                f"object=({object_position[0]:+.4f},{object_position[1]:+.4f},{object_position[2]:+.4f}) "
                f"xy_offset=({xy_offset[0]:+.4f},{xy_offset[1]:+.4f})"
            )
        env.close()
        return

    print("Launching MuJoCo viewer for pick-only teacher reset inspection.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.geomgroup[3] = 1
        _configure_viewer(viewer, env, args.camera)

        for scene_index in range(args.scenes):
            if not viewer.is_running():
                break

            _, info = env.reset(seed=args.seed + scene_index)
            object_position = info["object_position"]
            ee_position = info["ee_position"]
            gripper_qpos = float(env.data.qpos[env._joint_qpos_adrs[-1]])
            xy_offset = object_position[:2] - ee_position[:2]

            print(
                f"scene={scene_index:03d} seed={args.seed + scene_index} "
                f"gripper_qpos={gripper_qpos:+.4f} "
                f"ee=({ee_position[0]:+.4f},{ee_position[1]:+.4f},{ee_position[2]:+.4f}) "
                f"object=({object_position[0]:+.4f},{object_position[1]:+.4f},{object_position[2]:+.4f}) "
                f"xy_offset=({xy_offset[0]:+.4f},{xy_offset[1]:+.4f})"
            )

            deadline = time.monotonic() + max(args.hold_seconds, 0.0)
            while viewer.is_running() and time.monotonic() < deadline:
                env.data.ctrl[env._actuator_ids] = env._ctrl_target
                mujoco.mj_step(env.model, env.data)
                env._set_receptacle_pose(env._receptacle_position)
                viewer.sync()
                time.sleep(env.model.opt.timestep)

    env.close()


def _configure_viewer(
    viewer,
    env,
    camera_name: str,
) -> None:
    import mujoco
    import numpy as np

    if camera_name == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.12, 0.0, 0.02], dtype=np.float64)
        viewer.cam.distance = 0.40
        viewer.cam.azimuth = 165
        viewer.cam.elevation = -35
        return

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)


if __name__ == "__main__":
    main()
