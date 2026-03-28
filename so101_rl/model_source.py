from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path


DEFAULT_LOCAL_MODEL_DIRS = (
    Path("external/mujoco_menagerie/robotstudio_so101"),
    Path("external/SO-ARM100/Simulation/SO101"),
)
MODEL_FILENAMES = ("so101.xml", "so101_new_calib.xml", "so_arm100.xml")


@dataclass(frozen=True)
class RobotModelSource:
    mjcf_path: Path
    asset_dir: Path


def resolve_robot_model_source(mjcf_path: str | Path | None = None) -> RobotModelSource:
    candidate_paths: list[Path] = []
    candidate_dirs: list[Path] = []

    if mjcf_path is not None:
        candidate_paths.append(Path(mjcf_path).expanduser())

    env_mjcf = _read_env_path("SO101_MJCF_PATH")
    if env_mjcf is not None:
        candidate_paths.append(env_mjcf)

    env_menagerie = _read_env_path("SO101_MENAGERIE_DIR")
    if env_menagerie is not None:
        candidate_dirs.append(env_menagerie)

    env_trs = _read_env_path("SO101_TRS_DIR")
    if env_trs is not None:
        candidate_dirs.append(env_trs)

    for local_dir in DEFAULT_LOCAL_MODEL_DIRS:
        candidate_dirs.append(local_dir)

    robot_descriptions_path = _robot_descriptions_mjcf_path()
    if robot_descriptions_path is not None:
        candidate_paths.append(robot_descriptions_path)

    for directory in candidate_dirs:
        candidate_paths.extend(_candidate_files_for_dir(directory))

    checked_paths: list[Path] = []
    for candidate in candidate_paths:
        resolved = candidate.expanduser()
        checked_paths.append(resolved)
        if resolved.is_file():
            asset_dir = resolved.parent / "assets"
            if asset_dir.is_dir():
                return RobotModelSource(mjcf_path=resolved, asset_dir=asset_dir)

    checked_display = ", ".join(str(path) for path in checked_paths) or "<none>"
    raise FileNotFoundError(
        "Could not resolve the SO-101 Menagerie MJCF. "
        "Set SO101_MJCF_PATH to a concrete XML file, set SO101_MENAGERIE_DIR to "
        "robotstudio_so101, set SO101_TRS_DIR to Simulation/SO101, or install "
        f"robot_descriptions. Checked: {checked_display}"
    )


def load_robot_xml_assets(source: RobotModelSource) -> tuple[str, dict[str, bytes]]:
    xml_text = source.mjcf_path.read_text(encoding="utf-8")
    asset_map: dict[str, bytes] = {}

    # MuJoCo resolves asset and include paths relative to the MJCF parent.
    # Keep only those relative paths here; adding basename aliases causes
    # duplicate-name failures for Menagerie SO-101 assets.
    for file_path in sorted(source.mjcf_path.parent.rglob("*")):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(source.mjcf_path.parent).as_posix()
        asset_map[rel_path] = file_path.read_bytes()

    return xml_text, asset_map


def _candidate_files_for_dir(path: Path) -> list[Path]:
    path = path.expanduser()
    if not path:
        return []
    return [path / filename for filename in MODEL_FILENAMES]


def _read_env_path(name: str) -> Path | None:
    import os

    raw = os.environ.get(name)
    if not raw:
        return None
    return Path(raw).expanduser()


def _robot_descriptions_mjcf_path() -> Path | None:
    try:
        robot_descriptions = import_module("robot_descriptions")
        module = getattr(robot_descriptions, "so_arm100_mj_description", None)
        if module is not None and hasattr(module, "MJCF_PATH"):
            return Path(module.MJCF_PATH)
    except Exception:
        pass

    try:
        module = import_module("robot_descriptions.so_arm100_mj_description")
        if hasattr(module, "MJCF_PATH"):
            return Path(module.MJCF_PATH)
    except Exception:
        pass

    try:
        robot_descriptions = import_module("robot_descriptions")
        module = getattr(robot_descriptions, "robotstudio_so101_mj_description", None)
        if module is not None and hasattr(module, "MJCF_PATH"):
            return Path(module.MJCF_PATH)
    except Exception:
        pass

    try:
        module = import_module("robot_descriptions.robotstudio_so101_mj_description")
        if hasattr(module, "MJCF_PATH"):
            return Path(module.MJCF_PATH)
    except Exception:
        return None

    return None
