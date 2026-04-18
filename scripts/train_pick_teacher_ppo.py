from __future__ import annotations

import argparse
from pathlib import Path
import time

from so101_rl.gl_backend import configure_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO teacher policy on the SO-101 curriculum tasks.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--total-timesteps", type=int, default=300_000, help="Total PPO environment steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout length.")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size.")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO optimization epochs per update.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.03,
        help="Entropy coefficient. Higher keeps the Gaussian policy more exploratory.",
    )
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes for the MLP policy.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Run evaluation every N environment steps. Set 0 to disable.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes per evaluation run.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25_000,
        help="Save a checkpoint every N environment steps. Set 0 to disable.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/ppo_grasp_teacher"),
        help="Directory for checkpoints, best model, and tensorboard logs.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Stable-Baselines3 log interval in training iterations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device passed to PPO, for example auto, cpu, cuda.",
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
        help="Training task to use.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("gripper_only", "lift_subset", "full"),
        default="full",
        help="Action space subset to train.",
    )
    parser.add_argument(
        "--preview-gui",
        action="store_true",
        help="Open a MuJoCo viewer and preview a few reset scenes before training starts.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Preview the training env and exit without training.",
    )
    parser.add_argument(
        "--preview-scenes",
        type=int,
        default=3,
        help="Number of reset scenes to preview before training.",
    )
    parser.add_argument(
        "--preview-hold-seconds",
        type=float,
        default=2.5,
        help="How long to hold each preview scene in the viewer.",
    )
    parser.add_argument(
        "--camera",
        choices=("free", "cam_high", "wrist_cam"),
        default="free",
        help="Viewer camera for preview mode.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use the EGL backend for training and non-GUI preview flows.",
    )
    return parser.parse_args()


def make_env(seed: int, mjcf_path: Path | None, task_mode: str, action_mode: str):
    def _factory() -> SO101PickPlaceEnv:
        from so101_rl.config import PickPlaceEnvConfig
        from so101_rl.envs import SO101PickPlaceEnv

        config = PickPlaceEnvConfig(task_mode=task_mode, observation_mode="teacher", action_mode=action_mode)
        env = SO101PickPlaceEnv(config=config, render_mode="rgb_array", mjcf_path=mjcf_path)
        env.reset(seed=seed)
        return env

    return _factory


def main() -> None:
    args = parse_args()
    configure_backend(headless=args.headless or not args.preview_gui)
    if args.preview_gui:
        _preview_training_env(args)
        if args.preview_only:
            return

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is required for PPO training. "
            "Install it with `python3 -m pip install -e .[training]`."
        ) from exc

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    eval_log_dir = run_dir / "eval_logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_env(
                    seed=args.seed,
                    mjcf_path=args.mjcf_path,
                    task_mode=args.task_mode,
                    action_mode=args.action_mode,
                )(),
                filename=None,
                info_keywords=(
                    "success",
                    "reached_object",
                    "object_between_jaws",
                    "has_gripper_contact",
                    "gripper_is_closed",
                    "is_grasping",
                    "is_lifted",
                ),
            )
        ]
    )
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv(
        [
            lambda: Monitor(
                make_env(
                    seed=args.seed + 10_000,
                    mjcf_path=args.mjcf_path,
                    task_mode=args.task_mode,
                    action_mode=args.action_mode,
                )(),
                filename=None,
                info_keywords=(
                    "success",
                    "reached_object",
                    "object_between_jaws",
                    "has_gripper_contact",
                    "gripper_is_closed",
                    "is_grasping",
                    "is_lifted",
                ),
            )
        ]
    )
    eval_env = VecMonitor(eval_env)

    callbacks = []
    if args.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(eval_log_dir),
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )
        )
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="ppo_pick_teacher",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        policy_kwargs={"net_arch": list(args.net_arch)},
        seed=args.seed,
        device=args.device,
    )

    print(f"training_task_mode={args.task_mode}")
    print("training_observation_mode=teacher")
    print(f"training_action_mode={args.action_mode}")
    print("policy_architecture=MlpPolicy with default FlattenExtractor and net_arch for policy/value heads")
    print(f"net_arch={list(args.net_arch)}")
    print(f"total_timesteps={args.total_timesteps}")
    print(f"run_dir={run_dir}")

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None, log_interval=args.log_interval)
    final_model_path = run_dir / "ppo_pick_teacher_final"
    model.save(str(final_model_path))
    print(f"saved_final_model={final_model_path}.zip")

    train_env.close()
    eval_env.close()


def _preview_training_env(args: argparse.Namespace) -> None:
    import mujoco
    import mujoco.viewer
    import numpy as np

    env = make_env(
        seed=args.seed,
        mjcf_path=args.mjcf_path,
        task_mode=args.task_mode,
        action_mode=args.action_mode,
    )()
    print("Launching MuJoCo viewer for training-env preview.")
    print(f"preview_task_mode={args.task_mode}")
    print(f"preview_action_mode={args.action_mode}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        _configure_viewer(viewer, env, args.camera)
        for scene_index in range(args.preview_scenes):
            if not viewer.is_running():
                break

            observation, info = env.reset(seed=args.seed + scene_index)
            del observation
            print(
                f"scene={scene_index:03d} "
                f"gripper_qpos={info['gripper_qpos']:+.4f} "
                f"ee_to_object={info['ee_to_object_distance']:.4f} "
                f"between={int(info['object_between_jaws'])} "
                f"contact={int(info['has_gripper_contact'])} "
                f"closed={int(info['gripper_is_closed'])}"
            )

            deadline = time.monotonic() + max(args.preview_hold_seconds, 0.0)
            while viewer.is_running() and time.monotonic() < deadline:
                env.data.ctrl[env._actuator_ids] = env._ctrl_target
                mujoco.mj_step(env.model, env.data)
                env._set_receptacle_pose(env._receptacle_position)
                viewer.sync()
                time.sleep(env.model.opt.timestep)

    env.close()


def _configure_viewer(viewer, env, camera_name: str) -> None:
    import mujoco
    import numpy as np

    if camera_name == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.18, 0.0, 0.02], dtype=np.float64)
        viewer.cam.distance = 0.38
        viewer.cam.azimuth = 165
        viewer.cam.elevation = -35
        return

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)


if __name__ == "__main__":
    main()
