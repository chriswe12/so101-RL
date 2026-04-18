from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import gc
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from so101_rl.gl_backend import configure_backend
from so101_rl.model_source import load_robot_xml_assets, resolve_robot_model_source


ARM_ACTUATORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_ACTUATOR = "gripper"
ALL_ACTUATORS = ARM_ACTUATORS + (GRIPPER_ACTUATOR,)

DEFAULT_HOVER_ARM_QPOS = (-0.2575, -1.745, 0.895, 0.441, -1.42)
DEFAULT_GRASP_ARM_QPOS = (-0.2575, -1.75, 1.65, 1.1, -1.42)
DEFAULT_LIFT_ARM_QPOS = (-0.3, -1.75, 1.0, 0.6, -1.0)

FIXED_JAW_GEOM_CANDIDATES = ("fixed_jaw_box4", "fixed_jaw_box3", "fixed_jaw_box5")
MOVING_JAW_GEOM_CANDIDATES = ("moving_jaw_box2", "moving_jaw_box3", "moving_jaw_box1")


@dataclass(frozen=True)
class TuneConfig:
    # Tune first: mass and inertia.
    cube_half_size: tuple[float, float, float] = (0.014, 0.014, 0.014)
    cube_mass: float = 0.030
    cube_diaginertia: tuple[float, float, float] | None = None

    # Fixed motion used for all tuning passes.
    hover_arm_qpos: tuple[float, float, float, float, float] = DEFAULT_HOVER_ARM_QPOS
    grasp_arm_qpos: tuple[float, float, float, float, float] = DEFAULT_GRASP_ARM_QPOS
    lift_arm_qpos: tuple[float, float, float, float, float] = DEFAULT_LIFT_ARM_QPOS
    initial_gripper_qpos: float = 0.60
    closed_gripper_qpos: float = -0.03
    settle_steps: int = 40
    hover_steps: int = 20
    descend_steps: int = 70
    close_steps: int = 60
    hold_closed_steps: int = 20
    lift_steps: int = 120
    hold_lift_steps: int = 120

    # Tune second: controller stiffness.
    arm_kp: float = 998.22
    arm_kv: float = 2.731
    arm_forcerange: tuple[float, float] = (-2.94, 2.94)
    finger_kp: float = 998.22
    finger_kv: float = 2.731
    finger_forcerange: tuple[float, float] = (-2.94, 2.94)

    # Tune third: contact softness.
    contact_solref: tuple[float, float] = (0.01, 1.0)
    contact_solimp: tuple[float, float, float, float, float] | None = None

    # Tune fourth: solver stability.
    timestep: float = 0.002
    solver_integrator: str = "implicitfast"
    solver_type: str = "Newton"
    solver_iterations: int = 30
    solver_ls_iterations: int = 60
    solver_impratio: float = 10.0

    # Tune fifth: friction and condim.
    gripper_friction: tuple[float, float, float] = (1.0, 5e-3, 5e-4)
    cube_friction: tuple[float, float, float] = (1.0, 5e-3, 5e-4)
    gripper_condim: int = 6
    cube_condim: int = 6

    # Placement and presentation.
    cube_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cube_align: str = "world"
    surface_mode: str = "platform"
    floor_z: float = -0.20
    platform_half_size: tuple[float, float, float] = (0.05, 0.05, 0.015)
    platform_top_z: float = 0.013


@dataclass(frozen=True)
class ModelHandles:
    actuator_ids: np.ndarray
    joint_qpos_adrs: np.ndarray
    cube_joint_qpos_adr: int
    cube_joint_qvel_adr: int
    cube_geom_id: int
    cube_site_id: int
    gripper_site_id: int
    fixed_jaw_geom_id: int
    moving_jaw_geom_id: int
    support_body_id: int


@dataclass
class RolloutStats:
    initial_cube_height: float
    max_cube_height: float
    final_cube_height: float
    max_cube_speed: float = 0.0
    max_cube_gripper_contacts: int = 0
    min_cube_to_gripper_distance: float = float("inf")
    final_cube_to_gripper_distance: float = float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Menagerie SO101 grasp-tuning scene. "
            "Tune in this order: mass/inertia, controller stiffness, contact softness, "
            "solver stability, then friction/condim."
        )
    )
    parser.add_argument("--mjcf-path", type=Path, default=None, help="Optional path to a specific SO101 MJCF.")
    parser.add_argument("--dump-xml", type=Path, default=None, help="Optional path to save the generated MJCF.")
    parser.add_argument("--headless", action="store_true", help="Use EGL and run without the MuJoCo GUI.")
    parser.add_argument(
        "--camera",
        choices=("free", "wrist_cam"),
        default="free",
        help="Viewer camera to use when not headless.",
    )
    parser.add_argument("--print-every", type=int, default=10, help="Print diagnostics every N simulation steps.")

    parser.add_argument("--cube-half-size", type=float, nargs=3, metavar=("HX", "HY", "HZ"), default=None)
    parser.add_argument("--cube-mass", type=float, default=None)
    parser.add_argument("--cube-diaginertia", type=float, nargs=3, metavar=("IX", "IY", "IZ"), default=None)
    parser.add_argument("--cube-offset", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None)
    parser.add_argument("--cube-align", choices=("gripper", "world"), default=None)

    parser.add_argument(
        "--hover-arm-qpos",
        "--start-arm-qpos",
        dest="hover_arm_qpos",
        type=float,
        nargs=5,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5"),
        default=None,
    )
    parser.add_argument("--grasp-arm-qpos", type=float, nargs=5, metavar=("Q1", "Q2", "Q3", "Q4", "Q5"), default=None)
    parser.add_argument("--lift-arm-qpos", type=float, nargs=5, metavar=("Q1", "Q2", "Q3", "Q4", "Q5"), default=None)
    parser.add_argument("--initial-gripper-qpos", type=float, default=None)
    parser.add_argument("--closed-gripper-qpos", type=float, default=None)
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument("--hover-steps", "--preclose-steps", dest="hover_steps", type=int, default=None)
    parser.add_argument("--descend-steps", type=int, default=None)
    parser.add_argument("--close-steps", type=int, default=None)
    parser.add_argument("--hold-closed-steps", type=int, default=None)
    parser.add_argument("--lift-steps", type=int, default=None)
    parser.add_argument("--hold-lift-steps", type=int, default=None)

    parser.add_argument("--arm-kp", type=float, default=None)
    parser.add_argument("--arm-kv", type=float, default=None)
    parser.add_argument("--arm-forcerange", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--finger-kp", type=float, default=None)
    parser.add_argument("--finger-kv", type=float, default=None)
    parser.add_argument("--finger-forcerange", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None)

    parser.add_argument("--contact-solref", type=float, nargs=2, metavar=("TIMECONST", "DAMPRATIO"), default=None)
    parser.add_argument(
        "--contact-solimp",
        type=float,
        nargs=5,
        metavar=("A", "B", "C", "D", "E"),
        default=None,
    )

    parser.add_argument("--timestep", type=float, default=None)
    parser.add_argument("--solver-iterations", type=int, default=None)
    parser.add_argument("--solver-ls-iterations", type=int, default=None)
    parser.add_argument("--solver-impratio", type=float, default=None)

    parser.add_argument("--gripper-friction", type=float, nargs=3, metavar=("SLIDE", "TORSION", "ROLL"), default=None)
    parser.add_argument("--cube-friction", type=float, nargs=3, metavar=("SLIDE", "TORSION", "ROLL"), default=None)
    parser.add_argument("--gripper-condim", type=int, default=None)
    parser.add_argument("--cube-condim", type=int, default=None)
    parser.add_argument("--surface-mode", choices=("platform", "floor"), default=None)
    parser.add_argument("--platform-half-size", type=float, nargs=3, metavar=("HX", "HY", "HZ"), default=None)
    parser.add_argument("--platform-top-z", type=float, default=None)
    parser.add_argument("--floor-z", type=float, default=None)
    args = parser.parse_args()

    config = build_config(args)
    configure_backend(headless=args.headless)

    import mujoco

    source = resolve_robot_model_source(mjcf_path=args.mjcf_path)
    robot_xml_text, asset_map = load_robot_xml_assets(source)
    tuning_xml = build_tuning_xml(robot_xml_text, config)

    if args.dump_xml is not None:
        args.dump_xml.parent.mkdir(parents=True, exist_ok=True)
        args.dump_xml.write_text(tuning_xml, encoding="utf-8")

    model = None
    data = None
    try:
        model = mujoco.MjModel.from_xml_string(tuning_xml, assets=asset_map)
        data = mujoco.MjData(model)
        handles = resolve_handles(mujoco, model)

        ctrl_range = np.asarray(model.actuator_ctrlrange[handles.actuator_ids], dtype=np.float64)
        hover_target = clamp_ctrl_target(config, ctrl_range, config.hover_arm_qpos, config.initial_gripper_qpos)
        grasp_target = clamp_ctrl_target(config, ctrl_range, config.grasp_arm_qpos, config.initial_gripper_qpos)
        close_target = clamp_ctrl_target(config, ctrl_range, config.grasp_arm_qpos, config.closed_gripper_qpos)
        lift_target = clamp_ctrl_target(config, ctrl_range, config.lift_arm_qpos, config.closed_gripper_qpos)

        grasp_anchor_position, grasp_anchor_quaternion = compute_grasp_anchor_pose(
            mujoco,
            model,
            handles,
            grasp_target,
        )
        place_support_body(model, handles, config, grasp_anchor_position)

        reset_state(mujoco, data)
        data.qpos[handles.joint_qpos_adrs] = hover_target
        data.ctrl[handles.actuator_ids] = hover_target
        mujoco.mj_forward(model, data)

        cube_position, cube_quaternion = compute_cube_start_pose(
            grasp_anchor_position,
            grasp_anchor_quaternion,
            config,
        )
        set_cube_pose(data, handles, cube_position, cube_quaternion)
        mujoco.mj_forward(model, data)
        for _ in range(config.settle_steps):
            data.ctrl[handles.actuator_ids] = hover_target
            mujoco.mj_step(model, data)

        cube_height = float(data.site_xpos[handles.cube_site_id][2])
        stats = RolloutStats(
            initial_cube_height=cube_height,
            max_cube_height=cube_height,
            final_cube_height=cube_height,
        )
        update_stats(mujoco, model, data, handles, stats)

        print_tuning_summary(config)

        phases = tuple(
            phase
            for phase in (
                ("hover", hover_target, hover_target, config.hover_steps),
                ("descend", hover_target, grasp_target, config.descend_steps),
                ("close", grasp_target, close_target, config.close_steps),
                ("hold_closed", close_target, close_target, config.hold_closed_steps),
                ("lift", close_target, lift_target, config.lift_steps),
                ("hold_lift", lift_target, lift_target, config.hold_lift_steps),
            )
            if phase[3] > 0
        )

        if args.headless:
            run_headless_rollout(
                mujoco,
                model,
                data,
                handles,
                phases,
                stats,
                print_every=max(args.print_every, 1),
            )
        else:
            run_viewer_rollout(
                mujoco,
                model,
                data,
                handles,
                phases,
                stats,
                camera_name=args.camera,
                print_every=max(args.print_every, 1),
            )

        print_final_summary(stats)
    finally:
        # Explicitly drop MuJoCo objects before interpreter shutdown.
        del data
        del model
        gc.collect()
        sys.stdout.flush()
        sys.stderr.flush()


def build_config(args: argparse.Namespace) -> TuneConfig:
    defaults = TuneConfig()
    return TuneConfig(
        cube_half_size=tuple(args.cube_half_size) if args.cube_half_size is not None else defaults.cube_half_size,
        cube_mass=float(args.cube_mass) if args.cube_mass is not None else defaults.cube_mass,
        cube_diaginertia=tuple(args.cube_diaginertia) if args.cube_diaginertia is not None else defaults.cube_diaginertia,
        hover_arm_qpos=tuple(args.hover_arm_qpos) if args.hover_arm_qpos is not None else defaults.hover_arm_qpos,
        grasp_arm_qpos=tuple(args.grasp_arm_qpos) if args.grasp_arm_qpos is not None else defaults.grasp_arm_qpos,
        lift_arm_qpos=tuple(args.lift_arm_qpos) if args.lift_arm_qpos is not None else defaults.lift_arm_qpos,
        initial_gripper_qpos=(
            float(args.initial_gripper_qpos)
            if args.initial_gripper_qpos is not None
            else defaults.initial_gripper_qpos
        ),
        closed_gripper_qpos=(
            float(args.closed_gripper_qpos)
            if args.closed_gripper_qpos is not None
            else defaults.closed_gripper_qpos
        ),
        settle_steps=int(args.settle_steps) if args.settle_steps is not None else defaults.settle_steps,
        hover_steps=int(args.hover_steps) if args.hover_steps is not None else defaults.hover_steps,
        descend_steps=int(args.descend_steps) if args.descend_steps is not None else defaults.descend_steps,
        close_steps=int(args.close_steps) if args.close_steps is not None else defaults.close_steps,
        hold_closed_steps=(
            int(args.hold_closed_steps) if args.hold_closed_steps is not None else defaults.hold_closed_steps
        ),
        lift_steps=int(args.lift_steps) if args.lift_steps is not None else defaults.lift_steps,
        hold_lift_steps=int(args.hold_lift_steps) if args.hold_lift_steps is not None else defaults.hold_lift_steps,
        arm_kp=float(args.arm_kp) if args.arm_kp is not None else defaults.arm_kp,
        arm_kv=float(args.arm_kv) if args.arm_kv is not None else defaults.arm_kv,
        arm_forcerange=tuple(args.arm_forcerange) if args.arm_forcerange is not None else defaults.arm_forcerange,
        finger_kp=float(args.finger_kp) if args.finger_kp is not None else defaults.finger_kp,
        finger_kv=float(args.finger_kv) if args.finger_kv is not None else defaults.finger_kv,
        finger_forcerange=(
            tuple(args.finger_forcerange)
            if args.finger_forcerange is not None
            else defaults.finger_forcerange
        ),
        contact_solref=tuple(args.contact_solref) if args.contact_solref is not None else defaults.contact_solref,
        contact_solimp=tuple(args.contact_solimp) if args.contact_solimp is not None else defaults.contact_solimp,
        timestep=float(args.timestep) if args.timestep is not None else defaults.timestep,
        solver_integrator=defaults.solver_integrator,
        solver_type=defaults.solver_type,
        solver_iterations=(
            int(args.solver_iterations) if args.solver_iterations is not None else defaults.solver_iterations
        ),
        solver_ls_iterations=(
            int(args.solver_ls_iterations)
            if args.solver_ls_iterations is not None
            else defaults.solver_ls_iterations
        ),
        solver_impratio=(
            float(args.solver_impratio) if args.solver_impratio is not None else defaults.solver_impratio
        ),
        gripper_friction=(
            tuple(args.gripper_friction) if args.gripper_friction is not None else defaults.gripper_friction
        ),
        cube_friction=tuple(args.cube_friction) if args.cube_friction is not None else defaults.cube_friction,
        gripper_condim=int(args.gripper_condim) if args.gripper_condim is not None else defaults.gripper_condim,
        cube_condim=int(args.cube_condim) if args.cube_condim is not None else defaults.cube_condim,
        cube_offset=tuple(args.cube_offset) if args.cube_offset is not None else defaults.cube_offset,
        cube_align=str(args.cube_align) if args.cube_align is not None else defaults.cube_align,
        surface_mode=str(args.surface_mode) if args.surface_mode is not None else defaults.surface_mode,
        floor_z=float(args.floor_z) if args.floor_z is not None else defaults.floor_z,
        platform_half_size=(
            tuple(args.platform_half_size) if args.platform_half_size is not None else defaults.platform_half_size
        ),
        platform_top_z=float(args.platform_top_z) if args.platform_top_z is not None else defaults.platform_top_z,
    )


def build_tuning_xml(robot_xml_text: str, config: TuneConfig) -> str:
    root = ET.fromstring(robot_xml_text)
    root.set("model", "so101_grasp_tuning")

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("integrator", config.solver_integrator)
    option.set("solver", config.solver_type)
    option.set("timestep", f"{config.timestep:.6g}")
    option.set("iterations", str(config.solver_iterations))
    option.set("ls_iterations", str(config.solver_ls_iterations))
    option.set("impratio", f"{config.solver_impratio:.6g}")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    _maybe_add(asset, "material", name="tuning_cube_mat", rgba="0.12 0.78 0.28 1")
    _maybe_add(asset, "material", name="tuning_floor_mat", rgba="0.16 0.16 0.18 1")
    _maybe_add(asset, "material", name="tuning_platform_mat", rgba="0.25 0.25 0.28 1")

    _patch_gripper_contact_defaults(root, config)
    _patch_actuators(root, config)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("The source MJCF is missing a worldbody.")

    _maybe_add(
        worldbody,
        "light",
        name="tuning_light",
        pos="0.35 -0.20 0.55",
        dir="-0.6 0.2 -1.0",
        diffuse="0.95 0.95 0.95",
        specular="0.15 0.15 0.15",
    )
    _maybe_add(
        worldbody,
        "geom",
        name="tuning_floor",
        type="plane",
        pos=_fmt((0.0, 0.0, config.floor_z)),
        size="0 0 0.1",
        material="tuning_floor_mat",
        friction="0.8 0.02 0.001",
    )
    if config.surface_mode == "platform":
        support_body = ET.SubElement(
            worldbody,
            "body",
            name="tuning_support",
            pos=_fmt((0.0, 0.0, config.platform_top_z - config.platform_half_size[2])),
        )
        ET.SubElement(
            support_body,
            "geom",
            name="tuning_support_geom",
            type="box",
            size=_fmt(config.platform_half_size),
            material="tuning_platform_mat",
            friction="0.95 0.02 0.001",
        )

    cube_body = ET.SubElement(
        worldbody,
        "body",
        name="tuning_cube",
        pos="0 0 0.05",
    )
    ET.SubElement(cube_body, "freejoint", name="tuning_cube_freejoint")
    ET.SubElement(
        cube_body,
        "inertial",
        pos="0 0 0",
        mass=f"{config.cube_mass:.6g}",
        diaginertia=_fmt(resolve_cube_diaginertia(config)),
    )
    cube_geom_attrib = {
        "name": "tuning_cube_geom",
        "type": "box",
        "size": _fmt(config.cube_half_size),
        "material": "tuning_cube_mat",
        "condim": str(config.cube_condim),
        "friction": _fmt(config.cube_friction),
    }
    cube_geom = ET.SubElement(cube_body, "geom", attrib=cube_geom_attrib)
    cube_geom.set("solref", _fmt(config.contact_solref))
    if config.contact_solimp is not None:
        cube_geom.set("solimp", _fmt(config.contact_solimp))
    ET.SubElement(
        cube_body,
        "site",
        name="tuning_cube_site",
        type="sphere",
        pos="0 0 0",
        size="0.004",
        rgba="0.10 0.95 0.25 1",
    )

    return ET.tostring(root, encoding="unicode")


def _patch_gripper_contact_defaults(root: ET.Element, config: TuneConfig) -> None:
    for class_name in ("collision_gripper", "collision_gripper_mesh"):
        default = root.find(f".//default[@class='{class_name}']")
        if default is None:
            raise ValueError(f"The source MJCF is missing default class {class_name!r}.")
        geom = default.find("geom")
        if geom is None:
            geom = ET.SubElement(default, "geom")
        geom.set("condim", str(config.gripper_condim))
        geom.set("friction", _fmt(config.gripper_friction))
        geom.set("solref", _fmt(config.contact_solref))
        if config.contact_solimp is not None:
            geom.set("solimp", _fmt(config.contact_solimp))


def _patch_actuators(root: ET.Element, config: TuneConfig) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("The source MJCF is missing an actuator block.")

    for actuator_name in ARM_ACTUATORS:
        element = actuator.find(f"./position[@name='{actuator_name}']")
        if element is None:
            raise ValueError(f"Missing actuator {actuator_name!r} in source MJCF.")
        element.set("kp", f"{config.arm_kp:.6g}")
        element.set("kv", f"{config.arm_kv:.6g}")
        element.set("forcerange", _fmt(config.arm_forcerange))

    gripper = actuator.find(f"./position[@name='{GRIPPER_ACTUATOR}']")
    if gripper is None:
        raise ValueError(f"Missing actuator {GRIPPER_ACTUATOR!r} in source MJCF.")
    gripper.set("kp", f"{config.finger_kp:.6g}")
    gripper.set("kv", f"{config.finger_kv:.6g}")
    gripper.set("forcerange", _fmt(config.finger_forcerange))


def resolve_cube_diaginertia(config: TuneConfig) -> tuple[float, float, float]:
    if config.cube_diaginertia is not None:
        return config.cube_diaginertia

    hx, hy, hz = config.cube_half_size
    full_x = 2.0 * hx
    full_y = 2.0 * hy
    full_z = 2.0 * hz
    mass = config.cube_mass
    return (
        mass * (full_y * full_y + full_z * full_z) / 12.0,
        mass * (full_x * full_x + full_z * full_z) / 12.0,
        mass * (full_x * full_x + full_y * full_y) / 12.0,
    )


def resolve_handles(mujoco, model) -> ModelHandles:
    actuator_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ALL_ACTUATORS],
        dtype=np.int32,
    )
    joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ALL_ACTUATORS],
        dtype=np.int32,
    )
    if np.any(actuator_ids < 0) or np.any(joint_ids < 0):
        raise ValueError("Could not resolve the expected Menagerie joint and actuator names.")

    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tuning_cube_freejoint")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tuning_cube_geom")
    cube_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tuning_cube_site")
    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if min(cube_joint_id, cube_geom_id, cube_site_id, gripper_site_id) < 0:
        raise ValueError("Could not resolve the injected cube or gripperframe handles.")

    return ModelHandles(
        actuator_ids=actuator_ids,
        joint_qpos_adrs=model.jnt_qposadr[joint_ids],
        cube_joint_qpos_adr=int(model.jnt_qposadr[cube_joint_id]),
        cube_joint_qvel_adr=int(model.jnt_dofadr[cube_joint_id]),
        cube_geom_id=cube_geom_id,
        cube_site_id=cube_site_id,
        gripper_site_id=gripper_site_id,
        fixed_jaw_geom_id=resolve_geom_id(mujoco, model, FIXED_JAW_GEOM_CANDIDATES),
        moving_jaw_geom_id=resolve_geom_id(mujoco, model, MOVING_JAW_GEOM_CANDIDATES),
        support_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tuning_support"),
    )


def resolve_geom_id(mujoco, model, candidates: tuple[str, ...]) -> int:
    for name in candidates:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id != -1:
            return geom_id
    raise ValueError(f"Could not resolve any gripper geom from: {', '.join(candidates)}")


def clamp_ctrl_target(
    config: TuneConfig,
    ctrl_range: np.ndarray,
    arm_qpos: tuple[float, float, float, float, float],
    gripper_qpos: float,
) -> np.ndarray:
    del config
    target = np.asarray(arm_qpos + (gripper_qpos,), dtype=np.float64)
    return np.clip(target, ctrl_range[:, 0], ctrl_range[:, 1])


def reset_state(mujoco, data) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if data.act is not None:
        data.act[:] = 0.0
    if hasattr(data, "mocap_pos"):
        data.mocap_pos[:] = 0.0
        data.mocap_quat[:] = 0.0


def compute_grasp_anchor_pose(
    mujoco,
    model,
    handles: ModelHandles,
    grasp_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probe = mujoco.MjData(model)
    reset_state(mujoco, probe)
    probe.qpos[handles.joint_qpos_adrs] = grasp_target
    probe.ctrl[handles.actuator_ids] = grasp_target
    mujoco.mj_forward(model, probe)

    fixed_position = probe.geom_xpos[handles.fixed_jaw_geom_id].astype(np.float64).copy()
    moving_position = probe.geom_xpos[handles.moving_jaw_geom_id].astype(np.float64).copy()
    midpoint = 0.5 * (fixed_position + moving_position)
    gripper_quaternion = mat_to_quat(probe.site_xmat[handles.gripper_site_id].reshape(3, 3))
    return midpoint, gripper_quaternion


def place_support_body(
    model,
    handles: ModelHandles,
    config: TuneConfig,
    grasp_anchor_position: np.ndarray,
) -> None:
    if handles.support_body_id < 0 or config.surface_mode != "platform":
        return

    model.body_pos[handles.support_body_id, 0] = float(grasp_anchor_position[0])
    model.body_pos[handles.support_body_id, 1] = float(grasp_anchor_position[1])
    model.body_pos[handles.support_body_id, 2] = float(config.platform_top_z - config.platform_half_size[2])


def compute_cube_start_pose(
    grasp_anchor_position: np.ndarray,
    grasp_anchor_quaternion: np.ndarray,
    config: TuneConfig,
) -> tuple[np.ndarray, np.ndarray]:
    surface_top_z = config.floor_z
    if config.surface_mode == "platform":
        surface_top_z = config.platform_top_z

    cube_position = np.asarray(
        (
            grasp_anchor_position[0] + config.cube_offset[0],
            grasp_anchor_position[1] + config.cube_offset[1],
            surface_top_z + config.cube_half_size[2] + config.cube_offset[2],
        ),
        dtype=np.float64,
    )

    if config.cube_align == "world":
        cube_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    else:
        cube_quaternion = grasp_anchor_quaternion.astype(np.float64)

    return cube_position, cube_quaternion


def set_cube_pose(data, handles: ModelHandles, position: np.ndarray, quaternion: np.ndarray) -> None:
    qpos = data.qpos
    qvel = data.qvel
    qpos[handles.cube_joint_qpos_adr : handles.cube_joint_qpos_adr + 3] = position
    qpos[handles.cube_joint_qpos_adr + 3 : handles.cube_joint_qpos_adr + 7] = quaternion
    qvel[handles.cube_joint_qvel_adr : handles.cube_joint_qvel_adr + 6] = 0.0


def run_headless_rollout(
    mujoco,
    model,
    data,
    handles: ModelHandles,
    phases: tuple[tuple[str, np.ndarray, np.ndarray, int], ...],
    stats: RolloutStats,
    print_every: int,
) -> None:
    step_index = 0
    for phase_name, phase_start, phase_end, phase_steps in phases:
        for phase_step in range(phase_steps):
            target = interpolate_target(phase_start, phase_end, phase_step, phase_steps)
            data.ctrl[handles.actuator_ids] = target
            mujoco.mj_step(model, data)
            update_stats(mujoco, model, data, handles, stats)
            if step_index % print_every == 0 or phase_step == phase_steps - 1:
                print_step_summary(mujoco, model, data, handles, stats, phase_name, step_index)
            step_index += 1


def run_viewer_rollout(
    mujoco,
    model,
    data,
    handles: ModelHandles,
    phases: tuple[tuple[str, np.ndarray, np.ndarray, int], ...],
    stats: RolloutStats,
    camera_name: str,
    print_every: int,
) -> None:
    import mujoco.viewer

    print("Launching viewer. Close the window to stop the rollout early.")
    step_index = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_viewer_camera(mujoco, viewer, model, camera_name)
        viewer.opt.geomgroup[3] = 1

        for phase_name, phase_start, phase_end, phase_steps in phases:
            for phase_step in range(phase_steps):
                if not viewer.is_running():
                    return
                target = interpolate_target(phase_start, phase_end, phase_step, phase_steps)
                data.ctrl[handles.actuator_ids] = target
                mujoco.mj_step(model, data)
                update_stats(mujoco, model, data, handles, stats)
                if step_index % print_every == 0 or phase_step == phase_steps - 1:
                    print_step_summary(mujoco, model, data, handles, stats, phase_name, step_index)
                viewer.sync()
                time.sleep(model.opt.timestep)
                step_index += 1


def configure_viewer_camera(mujoco, viewer, model, camera_name: str) -> None:
    if camera_name == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array((0.18, 0.01, 0.10), dtype=np.float64)
        viewer.cam.distance = 0.68
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -16
        return

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)


def interpolate_target(start: np.ndarray, end: np.ndarray, phase_step: int, phase_steps: int) -> np.ndarray:
    if phase_steps <= 1:
        return end
    alpha = float(phase_step + 1) / float(phase_steps)
    return start + alpha * (end - start)


def update_stats(mujoco, model, data, handles: ModelHandles, stats: RolloutStats) -> None:
    cube_position = data.site_xpos[handles.cube_site_id].astype(np.float64)
    cube_velocity = data.qvel[handles.cube_joint_qvel_adr : handles.cube_joint_qvel_adr + 3].astype(np.float64)
    gripper_position = data.site_xpos[handles.gripper_site_id].astype(np.float64)

    cube_height = float(cube_position[2])
    cube_speed = float(np.linalg.norm(cube_velocity))
    cube_to_gripper_distance = float(np.linalg.norm(cube_position - gripper_position))
    cube_gripper_contacts = count_cube_gripper_contacts(mujoco, model, data, handles)

    stats.max_cube_height = max(stats.max_cube_height, cube_height)
    stats.final_cube_height = cube_height
    stats.max_cube_speed = max(stats.max_cube_speed, cube_speed)
    stats.max_cube_gripper_contacts = max(stats.max_cube_gripper_contacts, cube_gripper_contacts)
    stats.min_cube_to_gripper_distance = min(stats.min_cube_to_gripper_distance, cube_to_gripper_distance)
    stats.final_cube_to_gripper_distance = cube_to_gripper_distance


def count_cube_gripper_contacts(mujoco, model, data, handles: ModelHandles) -> int:
    contacts = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.geom1 == handles.cube_geom_id and is_gripper_geom(mujoco, model, int(contact.geom2)):
            contacts += 1
        elif contact.geom2 == handles.cube_geom_id and is_gripper_geom(mujoco, model, int(contact.geom1)):
            contacts += 1
    return contacts


def is_gripper_geom(mujoco, model, geom_id: int) -> bool:
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    if any(marker in geom_name.lower() for marker in ("jaw", "gripper", "camera_box")):
        return True
    if any(marker in body_name.lower() for marker in ("gripper", "moving_jaw", "camera_mount")):
        return True
    return False


def print_step_summary(
    mujoco,
    model,
    data,
    handles: ModelHandles,
    stats: RolloutStats,
    phase_name: str,
    step_index: int,
) -> None:
    cube_position = data.site_xpos[handles.cube_site_id].astype(np.float64)
    cube_velocity = data.qvel[handles.cube_joint_qvel_adr : handles.cube_joint_qvel_adr + 3].astype(np.float64)
    lift = cube_position[2] - stats.initial_cube_height
    contacts = count_cube_gripper_contacts(mujoco, model, data, handles)
    print(
        f"step={step_index:04d} phase={phase_name:>11} "
        f"cube=({cube_position[0]:+.4f},{cube_position[1]:+.4f},{cube_position[2]:+.4f}) "
        f"lift={lift:+.4f} speed={np.linalg.norm(cube_velocity):.4f} contacts={contacts}"
    )


def print_tuning_summary(config: TuneConfig) -> None:
    print("tuning.mass_inertia", f"mass={config.cube_mass:.6g}", f"diaginertia={resolve_cube_diaginertia(config)}")
    print(
        "tuning.motion",
        f"hover_arm_qpos={config.hover_arm_qpos}",
        f"grasp_arm_qpos={config.grasp_arm_qpos}",
        f"lift_arm_qpos={config.lift_arm_qpos}",
    )
    print(
        "tuning.controller",
        f"arm_kp={config.arm_kp:.6g}",
        f"arm_kv={config.arm_kv:.6g}",
        f"finger_kp={config.finger_kp:.6g}",
        f"finger_kv={config.finger_kv:.6g}",
    )
    print(
        "tuning.contact",
        f"solref={config.contact_solref}",
        f"solimp={config.contact_solimp}",
    )
    print(
        "tuning.solver",
        f"timestep={config.timestep:.6g}",
        f"integrator={config.solver_integrator}",
        f"solver={config.solver_type}",
        f"iterations={config.solver_iterations}",
        f"ls_iterations={config.solver_ls_iterations}",
    )
    print(
        "tuning.friction",
        f"gripper_friction={config.gripper_friction}",
        f"cube_friction={config.cube_friction}",
        f"gripper_condim={config.gripper_condim}",
        f"cube_condim={config.cube_condim}",
    )
    print(
        "tuning.surface",
        f"surface_mode={config.surface_mode}",
        f"floor_z={config.floor_z:.6g}",
        f"platform_top_z={config.platform_top_z:.6g}",
        f"platform_half_size={config.platform_half_size}",
    )


def print_final_summary(stats: RolloutStats) -> None:
    max_lift = stats.max_cube_height - stats.initial_cube_height
    final_lift = stats.final_cube_height - stats.initial_cube_height
    print(
        "summary",
        f"max_lift={max_lift:+.4f}",
        f"final_lift={final_lift:+.4f}",
        f"max_cube_speed={stats.max_cube_speed:.4f}",
        f"max_cube_gripper_contacts={stats.max_cube_gripper_contacts}",
        f"min_cube_to_gripper_distance={stats.min_cube_to_gripper_distance:.4f}",
        f"final_cube_to_gripper_distance={stats.final_cube_to_gripper_distance:.4f}",
    )


def mat_to_quat(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / scale
        x = (m[2, 1] - m[1, 2]) * scale
        y = (m[0, 2] - m[2, 0]) * scale
        z = (m[1, 0] - m[0, 1]) * scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    quat = np.asarray((w, x, y, z), dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _maybe_add(parent: ET.Element, tag: str, **attrib: str) -> None:
    name = attrib.get("name")
    if name is not None and any(child.tag == tag and child.get("name") == name for child in parent):
        return
    ET.SubElement(parent, tag, attrib=attrib)


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


if __name__ == "__main__":
    main()
