from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk

import numpy as np

from .gl_backend import configure_backend


def _rgb_to_ppm(image: np.ndarray) -> bytes:
    height, width, _ = image.shape
    header = f"P6 {width} {height} 255\n".encode("ascii")
    return header + image.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Display the exact camera tensors used by the environment.")
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=None,
        help="Optional path to a specific SO-101 MJCF file.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reset seed.")
    parser.add_argument(
        "--policy",
        choices=("zero", "random"),
        default="zero",
        help="Action policy while the camera feed viewer is open.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target UI refresh rate.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use the EGL backend and skip opening the Tk camera window.",
    )
    args = parser.parse_args()

    configure_backend(headless=args.headless)
    from .envs import SO101PickPlaceEnv

    env = SO101PickPlaceEnv(render_mode="rgb_array", mjcf_path=args.mjcf_path)
    observation, _ = env.reset(seed=args.seed)

    if args.headless:
        print("camera_viewer_headless=1")
        print(f"cam_high_shape={observation['observation.images.cam_high'].shape}")
        print(f"cam_right_wrist_shape={observation['observation.images.cam_right_wrist'].shape}")
        env.close()
        return

    root = tk.Tk()
    root.title("SO-101 Camera Feeds")

    info_text = (
        f"Exact observation tensors | cam_high={env.config.image_width}x{env.config.image_height} "
        f"| cam_right_wrist={env.config.image_width}x{env.config.image_height}"
    )
    tk.Label(root, text=info_text, anchor="w").pack(fill="x")

    frame = tk.Frame(root)
    frame.pack()

    top_label = tk.Label(frame, text="cam_high")
    top_label.grid(row=0, column=0)
    wrist_label = tk.Label(frame, text="cam_right_wrist")
    wrist_label.grid(row=0, column=1)

    top_image_label = tk.Label(frame)
    top_image_label.grid(row=1, column=0)
    wrist_image_label = tk.Label(frame)
    wrist_image_label.grid(row=1, column=1)

    interval_ms = max(1, int(round(1000.0 / max(args.fps, 1.0))))

    def _action() -> np.ndarray:
        if args.policy == "random":
            return env.action_space.sample()
        return np.zeros(env.action_space.shape, dtype=np.float32)

    def _update_labels(obs: dict[str, np.ndarray]) -> None:
        top_photo = tk.PhotoImage(data=_rgb_to_ppm(obs["observation.images.cam_high"]), format="PPM")
        wrist_photo = tk.PhotoImage(data=_rgb_to_ppm(obs["observation.images.cam_right_wrist"]), format="PPM")
        top_image_label.configure(image=top_photo)
        wrist_image_label.configure(image=wrist_photo)
        top_image_label.image = top_photo
        wrist_image_label.image = wrist_photo

    def _tick() -> None:
        nonlocal observation

        next_observation, _, terminated, truncated, _ = env.step(_action())
        if terminated or truncated:
            observation, _ = env.reset(seed=args.seed)
        else:
            observation = next_observation

        _update_labels(observation)
        root.after(interval_ms, _tick)

    def _close() -> None:
        env.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _close)
    _update_labels(observation)
    root.after(interval_ms, _tick)
    root.mainloop()


if __name__ == "__main__":
    main()
