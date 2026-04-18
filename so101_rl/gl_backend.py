from __future__ import annotations

import os


def configure_backend(*, headless: bool) -> None:
    """Select the MuJoCo OpenGL backend before creating windows or renderers."""
    if headless:
        os.environ["MUJOCO_GL"] = "egl"
