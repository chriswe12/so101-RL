"""SO-101 MuJoCo workspace package."""

from .config import PickPlaceEnvConfig

__all__ = ["PickPlaceEnvConfig", "SO101PickPlaceEnv"]


def __getattr__(name: str):
    if name == "SO101PickPlaceEnv":
        from .envs import SO101PickPlaceEnv

        return SO101PickPlaceEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
