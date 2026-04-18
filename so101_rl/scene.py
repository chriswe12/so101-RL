from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .config import PickPlaceEnvConfig


def build_pick_place_xml(robot_xml_text: str, config: PickPlaceEnvConfig) -> str:
    root = ET.fromstring(robot_xml_text)
    root.set("model", "so101_pick_place")

    _ensure_option(root, config)
    _ensure_visual(root, config)
    _ensure_task_assets(root, config)
    _ensure_gripper_contact_defaults(root, config)
    _apply_robot_visual_theme(root, config)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("SO-101 MJCF is missing a <worldbody> block.")

    _ensure_world_decor(worldbody, config)
    _attach_overhead_mount(worldbody, config)
    _attach_top_camera(worldbody, config)
    _attach_gripper_site_and_camera(worldbody, config)
    _add_object_body(worldbody, config)
    _add_receptacle_body(worldbody, config)
    return ET.tostring(root, encoding="unicode")


def _ensure_option(root: ET.Element, config: PickPlaceEnvConfig) -> None:
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("integrator", config.solver_integrator)
    option.set("solver", config.solver_type)
    option.set("timestep", str(config.timestep))
    option.set("iterations", str(config.solver_iterations))
    option.set("ls_iterations", str(config.solver_ls_iterations))
    option.set("impratio", str(config.solver_impratio))


def _ensure_visual(root: ET.Element, config: PickPlaceEnvConfig) -> None:
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")

    global_element = visual.find("global")
    if global_element is None:
        global_element = ET.SubElement(visual, "global")
    global_element.set("offwidth", str(config.image_width))
    global_element.set("offheight", str(config.image_height))


def _ensure_task_assets(root: ET.Element, config: PickPlaceEnvConfig) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    _maybe_add(asset, "material", name="robot_arm_black", rgba=_fmt(config.robot_arm_rgba))
    _maybe_add(asset, "material", name="robot_gripper_white", rgba=_fmt(config.robot_gripper_rgba))
    _maybe_add(
        asset,
        "texture",
        name="task_ground",
        type="2d",
        builtin="checker",
        rgb1="0.23 0.24 0.28",
        rgb2="0.17 0.18 0.21",
        width="512",
        height="512",
    )
    _maybe_add(
        asset,
        "material",
        name="task_ground_mat",
        texture="task_ground",
        texrepeat="3 3",
        texuniform="true",
        reflectance="0.1",
    )
    _maybe_add(asset, "material", name="task_table_mat", rgba="0.03 0.03 0.03 1")
    _maybe_add(asset, "material", name="task_object_green", rgba=_fmt(config.object_rgba))
    _maybe_add(asset, "material", name="task_receptacle_purple", rgba=_fmt(config.receptacle_rgba))
    mesh_scale = (config.overhead_mount_mesh_scale,) * 3
    _maybe_add(
        asset,
        "mesh",
        name="overhead_arm_base",
        file="assets/camera_top/arm_base.stl",
        scale=_fmt(mesh_scale),
    )
    _maybe_add(
        asset,
        "mesh",
        name="overhead_cam_bottom",
        file="assets/camera_top/cam_mount_bottom.stl",
        scale=_fmt(mesh_scale),
    )
    _maybe_add(
        asset,
        "mesh",
        name="overhead_cam_middle",
        file="assets/camera_top/cam_mount_middle.stl",
        scale=_fmt(mesh_scale),
    )
    _maybe_add(
        asset,
        "mesh",
        name="overhead_cam_top",
        file="assets/camera_top/cam_mount_top.stl",
        scale=_fmt(mesh_scale),
    )


def _ensure_gripper_contact_defaults(root: ET.Element, config: PickPlaceEnvConfig) -> None:
    for class_name in ("collision_gripper", "collision_gripper_mesh"):
        default = root.find(f".//default[@class='{class_name}']")
        if default is None:
            continue

        geom = default.find("geom")
        if geom is None:
            geom = ET.SubElement(default, "geom")

        geom.set("condim", str(config.gripper_condim))
        geom.set("friction", _fmt(config.gripper_friction))
        geom.set("solref", _fmt(config.contact_solref))
        if config.contact_solimp is not None:
            geom.set("solimp", _fmt(config.contact_solimp))


def _apply_robot_visual_theme(root: ET.Element, config: PickPlaceEnvConfig) -> None:
    del config  # Theme values are already materialized in _ensure_task_assets.

    for geom in root.findall(".//geom"):
        if not _is_robot_visual_geom(geom):
            continue

        geom.attrib.pop("rgba", None)
        if _is_gripper_visual_geom(geom):
            geom.set("material", "robot_gripper_white")
        else:
            geom.set("material", "robot_arm_black")


def _ensure_world_decor(worldbody: ET.Element, config: PickPlaceEnvConfig) -> None:
    if not _has_named_child(worldbody, "light", "task_light"):
        ET.SubElement(
            worldbody,
            "light",
            name="task_light",
            pos="0.3 0.0 0.9",
            dir="-0.2 0 -1",
            diffuse="0.9 0.9 0.9",
            specular="0.2 0.2 0.2",
        )

    if not _has_named_child(worldbody, "geom", "task_floor"):
        ET.SubElement(
            worldbody,
            "geom",
            name="task_floor",
            type="plane",
            pos="0 0 -0.25",
            size="0 0 0.1",
            material="task_ground_mat",
            contype="0",
            conaffinity="0",
        )

    if not _has_named_child(worldbody, "geom", "task_table"):
        ET.SubElement(
            worldbody,
            "geom",
            name="task_table",
            type="box",
            pos=_fmt(config.table_pos),
            size=_fmt(config.table_size),
            material="task_table_mat",
            friction=_fmt(config.table_friction),
        )


def _attach_overhead_mount(worldbody: ET.Element, config: PickPlaceEnvConfig) -> None:
    base_body = worldbody.find(".//body[@name='base']")
    if base_body is None:
        raise ValueError("SO-101 MJCF is missing a base body; overhead camera mount injection failed.")

    if base_body.find(f"./body[@name='{config.overhead_mount_body_name}']") is not None:
        return

    mount_body = ET.SubElement(
        base_body,
        "body",
        name=config.overhead_mount_body_name,
        pos=_fmt(config.overhead_mount_pos),
        quat=_fmt(config.overhead_mount_quat),
    )
    for frame_name, geom_name, mesh_name, pos, quat in (
        (
            "overhead_frame_arm_base",
            "overhead_arm_base_geom",
            "overhead_arm_base",
            config.overhead_frame_arm_base_pos,
            config.overhead_frame_arm_base_quat,
        ),
        (
            "overhead_frame_cam_bottom",
            "overhead_cam_bottom_geom",
            "overhead_cam_bottom",
            config.overhead_frame_cam_bottom_pos,
            config.overhead_frame_cam_bottom_quat,
        ),
        (
            "overhead_frame_cam_middle",
            "overhead_cam_middle_geom",
            "overhead_cam_middle",
            config.overhead_frame_cam_middle_pos,
            config.overhead_frame_cam_middle_quat,
        ),
        (
            "overhead_frame_cam_top",
            "overhead_cam_top_geom",
            "overhead_cam_top",
            config.overhead_frame_cam_top_pos,
            config.overhead_frame_cam_top_quat,
        ),
    ):
        frame_body = ET.SubElement(
            mount_body,
            "body",
            name=frame_name,
            pos=_fmt(pos),
            quat=_fmt(quat),
        )
        ET.SubElement(
            frame_body,
            "geom",
            name=geom_name,
            type="mesh",
            mesh=mesh_name,
            material="robot_gripper_white",
            contype="0",
            conaffinity="0",
            mass="0",
        )


def _attach_top_camera(worldbody: ET.Element, config: PickPlaceEnvConfig) -> None:
    if worldbody.find(f".//camera[@name='{config.top_camera_name}']") is not None:
        return

    camera_body = _find_body(worldbody, (config.top_camera_mount_frame_name,))
    if camera_body is None:
        raise ValueError(
            f"SO-101 MJCF is missing the top camera mount frame {config.top_camera_mount_frame_name!r}."
        )

    camera_xyaxes = _compute_top_camera_xyaxes(camera_body, config)
    ET.SubElement(
        camera_body,
        "camera",
        name=config.top_camera_name,
        pos=_fmt(config.top_camera_pos_in_mount),
        xyaxes=_fmt(camera_xyaxes),
        fovy=f"{config.top_camera_fovy:.6g}",
    )


def _attach_gripper_site_and_camera(worldbody: ET.Element, config: PickPlaceEnvConfig) -> None:
    if worldbody.find(f".//site[@name='{config.ee_site_name}']") is None:
        site_body = _find_body(worldbody, ("gripper", "Fixed_Jaw", "camera_mount"))
        if site_body is None:
            raise ValueError("SO-101 MJCF is missing a gripper body; end-effector site injection failed.")
        site_attrib = {
            "name": config.ee_site_name,
            "size": "0.005",
            "rgba": "0.95 0.15 0.15 1",
        }
        if site_body.get("name") == "gripper":
            site_attrib["pos"] = "-0.0079 -0.000218 -0.098127"
        elif site_body.get("name") == "camera_mount":
            site_attrib["pos"] = "0.0 0.055 -0.045"
        else:
            site_attrib["pos"] = "0.002 -0.103 0.0"
        ET.SubElement(site_body, "site", attrib=site_attrib)

    if worldbody.find(f".//camera[@name='{config.wrist_camera_name}']") is None:
        camera_body = _find_body(worldbody, ("camera_mount", "gripper", "Fixed_Jaw"))
        if camera_body is None:
            raise ValueError("SO-101 MJCF is missing a camera mount or gripper body; wrist camera injection failed.")
        camera_attrib = {
            "name": config.wrist_camera_name,
            "fovy": f"{config.wrist_camera_fovy:.6g}",
        }
        if camera_body.get("name") == "camera_mount":
            camera_attrib["pos"] = "0.0 0.055 -0.045"
            camera_attrib["euler"] = "-0.57 0 0"
        else:
            camera_attrib["pos"] = "0.0 -0.055 0.028"
            camera_attrib["xyaxes"] = "1 0 0 0 0 -1"
        ET.SubElement(camera_body, "camera", attrib=camera_attrib)

def _add_object_body(worldbody: ET.Element, config: PickPlaceEnvConfig) -> None:
    if worldbody.find(".//body[@name='task_object']") is not None:
        return

    object_body = ET.SubElement(
        worldbody,
        "body",
        name="task_object",
        pos=_fmt((0.15, 0.0, config.table_top_z + config.object_half_height)),
    )
    ET.SubElement(object_body, "freejoint", name=config.object_joint_name)
    ET.SubElement(
        object_body,
        "inertial",
        pos="0 0 0",
        mass=_fmt((config.object_mass,)),
        diaginertia=_fmt(config.resolved_object_diaginertia),
    )
    object_geom = ET.SubElement(
        object_body,
        "geom",
        attrib={
            "name": "task_object_geom",
            "type": "box",
            "size": _fmt((config.object_radius, config.object_radius, config.object_half_height)),
            "condim": str(config.object_condim),
            "friction": _fmt(config.object_friction),
            "material": "task_object_green",
        },
    )
    object_geom.set("solref", _fmt(config.contact_solref))
    if config.contact_solimp is not None:
        object_geom.set("solimp", _fmt(config.contact_solimp))
    ET.SubElement(
        object_body,
        "site",
        name=config.object_site_name,
        pos="0 0 0",
        size="0.004",
        rgba="0.12 0.9 0.3 1",
    )


def _add_receptacle_body(worldbody: ET.Element, config: PickPlaceEnvConfig) -> None:
    if worldbody.find(f".//body[@name='{config.receptacle_body_name}']") is not None:
        return

    receptacle_body = ET.SubElement(
        worldbody,
        "body",
        name=config.receptacle_body_name,
        mocap="true",
        pos=_fmt((0.22, -0.22, config.table_top_z)),
    )
    ET.SubElement(
        receptacle_body,
        "site",
        name=config.receptacle_site_name,
        type="sphere",
        pos=_fmt((0.0, 0.0, config.object_half_height)),
        size="0.006",
        rgba="0.98 0.75 0.98 0.5",
    )

    ET.SubElement(
        receptacle_body,
        "geom",
        name="task_receptacle_base",
        type="cylinder",
        pos=_fmt((0.0, 0.0, config.receptacle_base_half_height)),
        size=_fmt((config.receptacle_outer_radius, config.receptacle_base_half_height)),
        material="task_receptacle_purple",
        friction="0.95 0.05 0.001",
        contype="1",
        conaffinity="1",
    )

    wall_radius = config.receptacle_inner_radius + 0.5 * config.receptacle_wall_thickness
    wall_z = config.receptacle_base_half_height + 0.5 * config.receptacle_wall_height
    tangent_half = max(
        0.004,
        np.pi * wall_radius / config.receptacle_segments * 0.6,
    )
    for index in range(config.receptacle_segments):
        theta = 2.0 * np.pi * index / config.receptacle_segments
        wall_x = wall_radius * np.cos(theta)
        wall_y = wall_radius * np.sin(theta)
        ET.SubElement(
            receptacle_body,
            "geom",
            name=f"task_receptacle_wall_{index}",
            type="box",
            pos=_fmt((wall_x, wall_y, wall_z)),
            euler=_fmt((0.0, 0.0, theta)),
            size=_fmt(
                (
                    0.5 * config.receptacle_wall_thickness,
                    tangent_half,
                    0.5 * config.receptacle_wall_height,
                )
            ),
            material="task_receptacle_purple",
            friction="0.95 0.05 0.001",
            contype="1",
            conaffinity="1",
        )


def _has_named_child(parent: ET.Element, tag: str, name: str) -> bool:
    return any(child.tag == tag and child.get("name") == name for child in parent)


def _is_robot_visual_geom(geom: ET.Element) -> bool:
    geom_class = geom.get("class")
    group = geom.get("group")
    if geom_class == "visual":
        return True
    if group == "2" and geom.get("contype") == "0" and geom.get("conaffinity") == "0":
        return True
    return False


def _is_gripper_visual_geom(geom: ET.Element) -> bool:
    tokens = (
        geom.get("name", ""),
        geom.get("mesh", ""),
        geom.get("material", ""),
    )
    jaw_markers = (
        "moving_jaw",
        "fixed_jaw",
        "wrist_roll_follower_so101_v1",
        "moving_jaw_so101_v1",
        "moving_jaw",
        "Fixed_Jaw",
        "Moving_Jaw",
        "camera_mount",
        "wrist_roll_follower_so101_camera_mount",
    )
    if any(marker in token for token in tokens for marker in jaw_markers):
        return True
    return False


def _find_body(worldbody: ET.Element, names: tuple[str, ...]) -> ET.Element | None:
    for name in names:
        body = worldbody.find(f".//body[@name='{name}']")
        if body is not None:
            return body
    return None


def _maybe_add(parent: ET.Element, tag: str, **attrib: str) -> None:
    name = attrib.get("name")
    if name is not None and _has_named_child(parent, tag, name):
        return
    ET.SubElement(parent, tag, attrib=attrib)


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def _compute_top_camera_xyaxes(
    camera_body: ET.Element,
    config: PickPlaceEnvConfig,
) -> tuple[float, float, float, float, float, float]:
    del camera_body

    mount_rotation = _quat_to_mat(config.overhead_mount_quat)
    frame_rotation = _quat_to_mat(config.overhead_frame_cam_top_quat)
    parent_rotation = mount_rotation @ frame_rotation
    parent_pos = np.asarray(config.overhead_mount_pos, dtype=np.float64) + (
        mount_rotation @ np.asarray(config.overhead_frame_cam_top_pos, dtype=np.float64)
    )
    del parent_pos
    forward_world = np.asarray(config.top_camera_forward, dtype=np.float64)
    forward_world /= np.linalg.norm(forward_world)

    up_hint_world = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(forward_world, up_hint_world))) > 0.98:
        up_hint_world = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)

    right_world = np.cross(forward_world, up_hint_world)
    right_world /= np.linalg.norm(right_world)
    up_world = np.cross(right_world, forward_world)
    up_world /= np.linalg.norm(up_world)

    right_local = parent_rotation.T @ right_world
    up_local = parent_rotation.T @ up_world
    return (
        float(right_local[0]),
        float(right_local[1]),
        float(right_local[2]),
        float(up_local[0]),
        float(up_local[1]),
        float(up_local[2]),
    )


def _quat_to_mat(quat: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
