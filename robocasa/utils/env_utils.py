from robocasa.utils.dataset_registry import (
    ATOMIC_TASK_DATASETS,
    COMPOSITE_TASK_DATASETS,
)
from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import (
    get_env_metadata_from_dataset,
)
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.transform_utils import rotate_2d_point
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import (
    array_to_string,
    find_elements,
)
import xml.etree.ElementTree as ET
import os
import robosuite
import robocasa
import yaml
import imageio
import numpy as np
import contextlib
from tqdm import tqdm
from termcolor import colored
from copy import deepcopy
import gymnasium as gym

from robocasa.utils.errors import PlacementError, SamplingError
import robocasa.macros as macros
import robocasa.utils.object_utils as OU
from robocasa.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

from robocasa.models.objects.objects import MJCFObject

_ROBOT_POS_OFFSETS: dict[str, list[float]] = {
    "GR1FloatingBody": [0, 0, 0.97],
    "GR1": [0, 0, 0.97],
    "GR1FixedLowerBody": [0, 0, 0.97],
    "G1FloatingBody": [0, -0.33, 0],
    "G1": [0, -0.33, 0],
    "G1FixedLowerBody": [0, -0.33, 0],
    "GoogleRobot": [0, 0, 0],
}

KITCHEN_SCENES_5X5 = [
    (layout, style) for layout in [11, 15, 18, 40, 50] for style in [14, 28, 34, 46, 58]
]

KITCHEN_SCENES_5X1 = [
    (layout, style) for layout in [11, 15, 18, 40, 50] for style in [34]
]


def create_env(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    seed=None,
    render_onscreen=False,
    translucent_robot=False,
    # robocasa-related configs
    split=None,
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
    **kwargs,
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robots if isinstance(robots, str) else robots[0],
    )

    if split == "target":
        obj_instance_split = "target"
        layout_ids = None
        style_ids = None
        layout_and_style_ids = list(zip(range(1, 11), range(1, 11)))
    elif split == "pretrain":
        obj_instance_split = "pretrain"
        layout_ids = -2
        style_ids = -2
        layout_and_style_ids = None
    elif split == "all":
        obj_instance_split = None
        layout_ids = -3
        style_ids = -3
        layout_and_style_ids = None
    elif split is None:
        pass
    else:
        raise ValueError('split must be either {None, "all", "pretrain", "target"}')

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=render_onscreen,
        has_offscreen_renderer=(not render_onscreen),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=(not render_onscreen),
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=translucent_robot,
        **kwargs,  # additional env keyword args
    )

    env = robosuite.make(**env_kwargs)
    return env


def convert_action(action):
    """
    Converts input action (np.array) to format expected by gym env (dict)
    """
    action = action.copy()
    output_action = {
        "action.end_effector_position": action[0:3],
        "action.end_effector_rotation": action[3:6],
        "action.gripper_close": action[6:7],
        "action.base_motion": action[7:11],
        "action.control_mode": action[11:12],
    }
    return output_action


def run_random_rollouts(
    env, num_rollouts, num_steps, video_path=None, camera_name="robot0_agentview_center"
):
    assert isinstance(env, gym.Env), f"Must use gym environment! Used type {type(env)}"
    video_writer = None
    if video_path is not None:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs, info = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            action = env.action_space.sample()
            # zero out the base actions to prevent excessive jitter
            action["action.base_motion"][:] = 0.0
            obs, reward, terminated, truncated, info = env.step(action)

            if video_writer is not None:
                video_img = env.sim.render(
                    height=512, width=768, camera_name=camera_name
                )[::-1]
                video_writer.append_data(video_img)

            if info["success"]:
                num_success_rollouts += 1
                break

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


def determine_face_dir(fixture_rot, ref_rot, epsilon=1e-2):
    delta = ref_rot - fixture_rot
    delta = ((delta + np.pi) % (2 * np.pi)) - np.pi

    # compare to the four cardinal angles
    if abs(delta - 0.0) < epsilon:
        return -1
    if abs(delta - (np.pi / 2)) < epsilon:
        return -2
    if abs(abs(delta) - np.pi) < epsilon:
        return 1
    if abs(delta + (np.pi / 2)) < epsilon:
        return 2


def get_current_layout_stool_rotations(env):
    """
    Automatically detect the current layout and extract unique stool rotation values (z_rot)
    from a YAML layout file associated with the environment.

    Args:
        env: The environment object (must have layout_id attribute)

    Returns:
        list: List of unique rotation values (floats) found in stool configurations
    """
    from robocasa.models.scenes.scene_registry import get_layout_path

    layout_path = get_layout_path(env.layout_id)

    with open(layout_path, "r") as file:
        layout_data = yaml.safe_load(file)

    unique_rots = set()

    if "floor_accessories" in layout_data:
        for accessory in layout_data["floor_accessories"]:
            if accessory.get("type") == "stool" and "z_rot" in accessory:
                unique_rots.add(accessory["z_rot"])

    if "room" in layout_data and "floor_accessories" in layout_data["room"]:
        for accessory in layout_data["room"]["floor_accessories"]:
            if accessory.get("type") == "stool" and "z_rot" in accessory:
                unique_rots.add(accessory["z_rot"])

    return sorted(list(unique_rots))


def categorize_stool_rotations(stool_rotations, ground_fixture_rot=None):
    """
    Categorize stool rotations into 4 cardinal directions, relative to the ground fixture rotation,
    using vector rotation and cosine similarity.

    Args:
        stool_rotations (list): List of rotation values in radians
        ground_fixture_rot (float): Rotation of the ground fixture in radians

    Returns:
        list: List of categorized rotation directions: [1, 2, -1, -2]
    """
    # Define canonical direction vectors
    category_vectors = {
        1: np.array([0, -1]),
        2: np.array([1, 0]),
        -1: np.array([0, 1]),
        -2: np.array([-1, 0]),
    }

    categorized_rotations = []

    for rotation in stool_rotations:
        # Adjust rotation relative to ground fixture
        if ground_fixture_rot is not None:
            rel_yaw = rotation - ground_fixture_rot
        else:
            rel_yaw = rotation

        # Create rotation matrix
        cos_yaw = np.cos(rel_yaw)
        sin_yaw = np.sin(rel_yaw)
        rot_matrix = np.array(
            [
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw],
            ]
        )

        # Rotate the base direction vector [0, 1]
        rotated_vec = rot_matrix @ np.array([0, 1])

        # Compare to each cardinal direction using dot product
        best_category = None
        best_similarity = -float("inf")  # cos(angle) ranges from -1 to 1

        for category, canonical_vec in category_vectors.items():
            similarity = np.dot(rotated_vec, canonical_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category

        categorized_rotations.append(best_category)

    return categorized_rotations


def get_island_group_counter_names(env):
    """
    Automatically detect all counter fixture names under all island_group groups in the current layout.
    Used to get bounding box combo of multiple counters.
    Args:
        env: The environment object (must have layout_id attribute)
    Returns:
        list: List of counter fixture names (str) under all island_group groups
    """
    from robocasa.models.scenes.scene_registry import get_layout_path

    layout_path = get_layout_path(env.layout_id)
    with open(layout_path, "r") as f:
        layout_data = yaml.safe_load(f)
    counter_names = []
    for group_key, group_val in layout_data.items():
        if group_key.startswith("island_group"):
            for subkey, subval in group_val.items():
                if isinstance(subval, list):
                    for fixture in subval:
                        if (
                            isinstance(fixture, dict)
                            and fixture.get("type") == "counter"
                            and "name" in fixture
                        ):
                            counter_names.append(fixture["name"] + "_" + group_key)
    return counter_names


def get_combined_counters_2d_bbox_corners(env, counter_names):
    """
    Used to get bounding box combo of multiple counters - useful for dining counters with multiple defined counters.
    """
    # Collect fixtures and their absolute ext sites
    fixtures = [env.get_fixture(name) for name in counter_names]
    all_pts = []
    for fx in fixtures:
        all_pts.extend(fx.get_ext_sites(all_points=False, relative=False))
    all_pts = np.asarray(all_pts, dtype=float)

    # Anchor = first fixture's local frame
    anchor = fixtures[0]
    a_p0, a_px, a_py, a_pz = anchor.get_ext_sites(all_points=False, relative=False)

    # Build anchor axes (u along p0->px, v along p0->py)
    u_axis = a_px[:2] - a_p0[:2]
    v_axis = a_py[:2] - a_p0[:2]
    u_norm = np.linalg.norm(u_axis) or 1.0
    v_norm = np.linalg.norm(v_axis) or 1.0
    u_axis = u_axis / u_norm
    v_axis = v_axis / v_norm

    # Project all points into anchor frame (origin at anchor p0)
    rel_xy = all_pts[:, :2] - a_p0[:2]
    proj_u = rel_xy @ u_axis
    proj_v = rel_xy @ v_axis

    min_u = float(np.min(proj_u))
    max_u = float(np.max(proj_u))
    min_v = float(np.min(proj_v))
    max_v = float(np.max(proj_v))

    z_min = float(np.min(all_pts[:, 2]))
    z_max = float(np.max(all_pts[:, 2]))

    # Reconstruct corners in world coords using anchor frame
    p0_world = a_p0.copy()
    p0_world[:2] = a_p0[:2] + u_axis * min_u + v_axis * min_v
    p0_world[2] = z_min

    px_world = a_px.copy()
    px_world[:2] = a_p0[:2] + u_axis * max_u + v_axis * min_v
    px_world[2] = z_min

    py_world = a_py.copy()
    py_world[:2] = a_p0[:2] + u_axis * min_u + v_axis * max_v
    py_world[2] = z_min

    pz_world = p0_world.copy()
    pz_world[2] = z_max

    abs_sites = np.array([p0_world, px_world, py_world, pz_world], dtype=float)

    return abs_sites


def compute_robot_base_placement_pose(env, ref_fixture, ref_object=None, offset=None):
    """
    steps:
    1. find the nearest counter to this fixture
    2. compute offset relative to this counter
    3. transform offset to global coordinates

    Args:
        ref_fixture (Fixture): reference fixture to place th robot near

        offset (list): offset to add to the base position

    """
    from robocasa.models.fixtures import (
        Counter,
        Stove,
        Stovetop,
        HousingCabinet,
        Fridge,
        fixture_is_type,
        FixtureType,
    )

    # step 1: find ground fixture closest to robot
    ground_fixture = None

    # manipulate drawer envs are exempt from dining counter/stool placement rules
    from robocasa.environments.kitchen.atomic.kitchen_drawer import (
        ManipulateDrawer,
    )

    manipulate_drawer_env = isinstance(env, ManipulateDrawer)

    if not fixture_is_type(ref_fixture, FixtureType.DINING_COUNTER):
        # get all base fixtures in the environment
        ground_fixtures = [
            fxtr
            for fxtr in env.fixtures.values()
            if isinstance(fxtr, Counter)
            or isinstance(fxtr, Stove)
            or isinstance(fxtr, Stovetop)
            or isinstance(fxtr, HousingCabinet)
            or isinstance(fxtr, Fridge)
        ]

        for fxtr in ground_fixtures:
            # get bounds of fixture
            point = ref_fixture.pos
            # if fxtr.name == "counter_corner_1_main_group_1":
            #     print("point:", point)
            #     p0, px, py, pz = fxtr.get_ext_sites(relative=False)
            #     print("p0:", p0)
            #     print("px:", px)
            #     print("py:", py)
            #     print("pz:", pz)
            #     print()

            if not OU.point_in_fixture(point=point, fixture=fxtr, only_2d=True):
                continue
            ground_fixture = fxtr
            break

    # set the stool fixture as the ref fixture itself if cannot find fixture containing ref
    if ground_fixture is None:
        if fixture_is_type(ref_fixture, FixtureType.STOOL):
            stool_only = True
        else:
            stool_only = False
        ground_fixture = ref_fixture
    else:
        stool_only = False
    # assert base_fixture is not None

    # step 2: compute offset relative to this counter
    ground_to_ref, _ = OU.get_rel_transform(ground_fixture, ref_fixture)

    # find the reference fixture to dining counter if it exists
    ref_to_fixture = None
    if (
        fixture_is_type(ground_fixture, FixtureType.DINING_COUNTER)
        and not manipulate_drawer_env
    ):
        if hasattr(env, "object_cfgs") and env.object_cfgs is not None:
            for cfg in env.object_cfgs:
                placement = cfg.get("placement", None)
                if placement is None:
                    continue
                fixture_id = placement.get("fixture", None)
                if fixture_id is None:
                    continue
                fixture = env.get_fixture(
                    id=fixture_id,
                    ref=placement.get("ref", None),
                    full_name_check=True if cfg["type"] == "fixture" else False,
                )
                if fixture_is_type(fixture, FixtureType.DINING_COUNTER):
                    sample_region_kwargs = placement.get("sample_region_kwargs", {})
                    ref_to_fixture = sample_region_kwargs.get("ref", None)
                    if ref_to_fixture is None:
                        continue
                    # in case ref_to_fixture is a string, get the corresponding fixture object
                    ref_to_fixture = env.get_fixture(ref_to_fixture)

    face_dir = 1  # 1 is facing front of fixture, -1 is facing south end of fixture

    if (
        fixture_is_type(ground_fixture, FixtureType.ISLAND)
        or fixture_is_type(ground_fixture, FixtureType.DINING_COUNTER)
        and manipulate_drawer_env
    ):
        island_group_counter_names = get_island_group_counter_names(env)
        if len(island_group_counter_names) > 1:
            # Ensure the ground fixture is the anchor by putting it first
            if ground_fixture.name in island_group_counter_names:
                island_group_counter_names = [
                    ground_fixture.name,
                    *[
                        n
                        for n in island_group_counter_names
                        if n != ground_fixture.name
                    ],
                ]
            abs_sites = get_combined_counters_2d_bbox_corners(
                env, island_group_counter_names
            )
        else:
            abs_sites = ground_fixture.get_ext_sites(relative=False)
        abs_sites = np.vstack(abs_sites)
        rel_yaw = ground_fixture.rot + np.pi / 2

        cos_yaw = np.cos(rel_yaw)
        sin_yaw = np.sin(rel_yaw)
        rot_matrix = np.array(
            [
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw],
            ]
        )

        # Rotate each point in abs_sites
        rotated_abs_sites = np.array([rot_matrix @ site[:2] for site in abs_sites])
        ref_point = ref_fixture.pos

        # Rotate ref_point (assuming it's a 3D point tuple or np.array)
        if isinstance(ref_point, tuple):
            ref_xy = np.array([ref_point[0], ref_point[1]])
        else:
            ref_xy = np.array(ref_point[:2])  # in case it's a full np.array

        rotated_ref_point = rot_matrix @ ref_xy

        dist1 = abs(rotated_ref_point[0] - rotated_abs_sites[0][0])
        dist2 = abs(rotated_ref_point[0] - rotated_abs_sites[2][0])
        dist3 = abs(rotated_ref_point[1] - rotated_abs_sites[1][1])
        dist4 = abs(rotated_ref_point[1] - rotated_abs_sites[0][1])

        if fixture_is_type(ground_fixture, FixtureType.ISLAND):
            min_dist = min(dist1, dist2, dist3, dist4)
            if min_dist == dist1:
                face_dir = 1
            elif min_dist == dist2:
                face_dir = -1
            elif min_dist == dist3:
                face_dir = 2
            else:
                face_dir = -2
        else:
            if dist1 < dist2:
                face_dir = 1
            else:
                face_dir = -1

    if (
        fixture_is_type(ground_fixture, FixtureType.DINING_COUNTER)
        or stool_only
        and not manipulate_drawer_env
    ):
        stool_rotations = get_current_layout_stool_rotations(env)

        # for dining counters, can face either north of south end of fixture
        if ref_object is not None:
            # choose the end that is closest to the ref object
            ref_point = env.object_placements[ref_object][0]
            categorized_stool_rotations = categorize_stool_rotations(
                stool_rotations, ground_fixture.rot
            )
        else:
            ### find the side closest to the ref fixture ###
            ref_point = ref_fixture.pos
            categorized_stool_rotations = None

        if fixture_is_type(ref_fixture, FixtureType.DRAWER):
            ref_fxtr_is_drawer = True
        else:
            ref_fxtr_is_drawer = False

        if (
            ref_to_fixture is not None
            and fixture_is_type(ref_to_fixture, FixtureType.STOOL)
            and ref_object is None
            and not ref_fxtr_is_drawer
        ):
            face_dir = determine_face_dir(ground_fixture.rot, ref_to_fixture.rot)
        elif fixture_is_type(ref_fixture, FixtureType.STOOL) and ref_object is None:
            face_dir = determine_face_dir(ground_fixture.rot, ref_fixture.rot)
        else:
            island_group_counter_names = get_island_group_counter_names(env)
            if len(island_group_counter_names) > 1:
                # Ensure the ground fixture is the anchor by putting it first
                if ground_fixture.name in island_group_counter_names:
                    island_group_counter_names = [
                        ground_fixture.name,
                        *[
                            n
                            for n in island_group_counter_names
                            if n != ground_fixture.name
                        ],
                    ]
                abs_sites = get_combined_counters_2d_bbox_corners(
                    env, island_group_counter_names
                )
            else:
                abs_sites = ground_fixture.get_ext_sites(relative=False)
            abs_sites = np.vstack(abs_sites)
            rel_yaw = ground_fixture.rot + np.pi / 2

            cos_yaw = np.cos(rel_yaw)
            sin_yaw = np.sin(rel_yaw)
            rot_matrix = np.array(
                [
                    [cos_yaw, -sin_yaw],
                    [sin_yaw, cos_yaw],
                ]
            )

            # Rotate each point in abs_sites
            rotated_abs_sites = np.array([rot_matrix @ site[:2] for site in abs_sites])

            # Rotate ref_point (assuming it's a 3D point tuple or np.array)
            if isinstance(ref_point, tuple):
                ref_xy = np.array([ref_point[0], ref_point[1]])
            else:
                ref_xy = np.array(ref_point[:2])  # in case it's a full np.array

            rotated_ref_point = rot_matrix @ ref_xy

            dist1 = abs(rotated_ref_point[0] - rotated_abs_sites[0][0])
            dist2 = abs(rotated_ref_point[0] - rotated_abs_sites[2][0])
            dist3 = abs(rotated_ref_point[1] - rotated_abs_sites[1][1])
            dist4 = abs(rotated_ref_point[1] - rotated_abs_sites[0][1])

            if fixture_is_type(ground_fixture, FixtureType.ISLAND):
                min_dist = min(dist1, dist2, dist3, dist4)
                if min_dist == dist1:
                    face_dir = 1
                elif min_dist == dist2:
                    face_dir = -1
                elif min_dist == dist3:
                    face_dir = 2
                else:
                    face_dir = -2
            else:
                if dist1 < dist2:
                    face_dir = 1
                else:
                    face_dir = -1

                # these dining counters only have 1 accesssible side for robot to spawn
                one_accessible_layout_ids = [11, 27, 35, 49, 60]
                if env.layout_id in one_accessible_layout_ids:
                    stool_rotations = get_current_layout_stool_rotations(env)
                    categorized_stool_rotations = categorize_stool_rotations(
                        stool_rotations, ground_fixture.rot
                    )
                    face_dir = categorized_stool_rotations[0]

    fixture_ext_sites = ground_fixture.get_ext_sites(relative=True)
    fixture_to_robot_offset = np.zeros(3)

    # set x offset
    fixture_to_robot_offset[0] = ground_to_ref[0]

    # y direction it's facing from perspective of host fixture
    if face_dir == 1:  # north
        fixture_p = fixture_ext_sites[0]
        fixture_to_robot_offset[1] = fixture_p[1] - 0.20
    elif face_dir == -1:  # south
        fixture_p = fixture_ext_sites[2]
        fixture_to_robot_offset[1] = fixture_p[1] + 0.20
    elif face_dir == 2:  # west
        fixture_p = fixture_ext_sites[1]
        fixture_to_robot_offset[0] = fixture_p[0] + 0.20
    elif face_dir == -2:  # east
        fixture_p = fixture_ext_sites[0]
        fixture_to_robot_offset[0] = fixture_p[0] - 0.20

    if offset is not None:
        ox, oy = offset
        if face_dir == 1:
            rx, ry = ox, oy
        elif face_dir == -1:
            rx, ry = -ox, -oy
        elif face_dir == 2:
            rx, ry = oy, ox
        elif face_dir == -2:
            rx, ry = -oy, -ox

        fixture_to_robot_offset[0] += rx
        fixture_to_robot_offset[1] += ry
    elif ref_object is not None:
        sampler = env.placement_initializer.samplers[f"{ref_object}_Sampler"]
        if face_dir == -1 or face_dir == 1:
            fixture_to_robot_offset[0] += np.mean(sampler.x_range)
        if face_dir == 2 or face_dir == -2:
            fixture_to_robot_offset[1] += np.mean(sampler.y_range)

    if (
        isinstance(ground_fixture, HousingCabinet)
        or isinstance(ground_fixture, Fridge)
        or "stack" in ground_fixture.name
    ):
        fixture_to_robot_offset[1] += face_dir * -0.10

    # move back a bit for the stools
    if fixture_is_type(ground_fixture, FixtureType.DINING_COUNTER):
        abs_sites = ground_fixture.get_ext_sites(relative=False)
        if fixture_is_type(ref_to_fixture, FixtureType.STOOL):
            stool = ref_to_fixture
        else:
            stool = env.get_fixture(FixtureType.STOOL)

        stool_rotations = get_current_layout_stool_rotations(env)

        def rotation_matrix_z(theta):
            """
            Return the 3x3 rotation matrix that rotates a vector about the Z axis by theta radians.
            """
            c = np.cos(theta)
            s = np.sin(theta)
            return np.array(
                [
                    [c, -s, 0.0],
                    [s, c, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

        if stool is not None and not ref_fxtr_is_drawer:
            abs_sites = ground_fixture.get_ext_sites(relative=False)
            ref_sites = stool.get_ext_sites(relative=False)

            # Apply rotation to both sets of points
            fixture_Rz = rotation_matrix_z(stool.rot + np.pi)
            stool_Rz = rotation_matrix_z(stool.rot + np.pi)
            fixture_sites = [fixture_Rz @ p for p in abs_sites]
            stool_sites = [stool_Rz @ p for p in ref_sites]

            if ref_object is not None:
                # Determine if we should take min or max y based on stool orientation
                normalized_rot = (
                    (stool.rot + np.pi) % (2 * np.pi)
                ) - np.pi  # normalize
                angle = normalized_rot % (2 * np.pi)

                if np.isclose(angle, np.pi / 2, atol=0.2) or np.isclose(
                    angle, 3 * np.pi / 2, atol=0.2
                ):
                    stool_back_site = max(stool_sites, key=lambda p: p[1])
                else:
                    stool_back_site = min(stool_sites, key=lambda p: p[1])

                stool_y = stool_back_site[1]

                # Find fixture site closest in Y to stool back
                fixture_y_diffs = [abs(p[1] - stool_y) for p in fixture_sites]
                closest_fixture_site = fixture_sites[np.argmin(fixture_y_diffs)]
                fixture_y = closest_fixture_site[1]

                delta_y = abs(stool_y - fixture_y)

                if face_dir == 1 and face_dir in categorized_stool_rotations:
                    fixture_to_robot_offset[1] -= delta_y
                elif face_dir == -1 and face_dir in categorized_stool_rotations:
                    fixture_to_robot_offset[1] += delta_y
                elif face_dir == 2 and face_dir in categorized_stool_rotations:
                    fixture_to_robot_offset[0] += delta_y
                elif face_dir == -2 and face_dir in categorized_stool_rotations:
                    fixture_to_robot_offset[0] -= delta_y
            elif fixture_is_type(ref_to_fixture, FixtureType.STOOL):
                if face_dir == 1:
                    fixture_to_robot_offset[1] -= abs(
                        fixture_sites[0][1] - stool_sites[0][1]
                    )
                elif face_dir == -1:
                    fixture_to_robot_offset[1] += abs(
                        fixture_sites[2][1] - stool_sites[2][1]
                    )
                elif face_dir == 2:
                    fixture_to_robot_offset[0] += abs(
                        fixture_sites[1][1] - stool_sites[2][1]
                    )
                elif face_dir == -2:
                    fixture_to_robot_offset[0] -= abs(
                        fixture_sites[0][1] - stool_sites[2][1]
                    )

    # apply robot-specific offset relative to the base fixture for x,y dims
    robot_model = env.robots[0].robot_model
    robot_class_name = robot_model.__class__.__name__
    if robot_class_name in _ROBOT_POS_OFFSETS:
        for dimension in range(0, 2):
            if dimension == 1:
                fixture_to_robot_offset[dimension] += (
                    _ROBOT_POS_OFFSETS[robot_class_name][dimension] * face_dir
                )
            else:
                fixture_to_robot_offset[dimension] += _ROBOT_POS_OFFSETS[
                    robot_class_name
                ][dimension]

    # step 3: transform offset to global coordinates
    robot_base_pos = np.zeros(3)
    robot_base_pos[0:2] = OU.get_pos_after_rel_offset(
        ground_fixture, fixture_to_robot_offset
    )[0:2]

    # apply robot-specific absolutely for z dim
    if robot_class_name in _ROBOT_POS_OFFSETS:
        robot_base_pos[2] = _ROBOT_POS_OFFSETS[robot_class_name][2]
    robot_base_ori = np.array([0, 0, ground_fixture.rot + np.pi / 2])
    if face_dir == -1:
        robot_base_ori[2] += np.pi
    elif face_dir == -2:
        robot_base_ori[2] = ground_fixture.rot
    elif face_dir == 2:
        robot_base_ori[2] = ground_fixture.rot + np.pi

    return robot_base_pos, robot_base_ori


def _check_cfg_is_valid(cfg):
    """
    check a object / fixture config for correctness. called by _get_placement_initializer
    """
    VALID_CFG_KEYS = set(
        {
            "type",
            "name",
            "model",
            "obj_groups",
            "exclude_obj_groups",
            "graspable",
            "cookable",
            "washable",
            "microwavable",
            "dishwashable",
            "fridgable",
            "freezable",
            "max_size",
            "object_scale",
            "placement",
            "info",
            "init_robot_here",
            "reset_region",
            "rotate_upright",
            "auxiliary_object_config",
            "auxiliary_obj_enable",
            "auxiliary_obj_placement",
        }
    )

    VALID_PLACEMENT_KEYS = set(
        {
            "size",
            "pos",
            "offset",
            "margin",
            "rotation_axis",
            "rotation",
            "ensure_object_boundary_in_range",
            "ensure_valid_placement",
            "ensure_valid_auxiliary_placement",
            "sample_args",
            "sample_region_kwargs",
            "reuse_region_from",
            "ref_obj",
            "fixture",
            "try_to_place_in",
            "anchor_to",
            "object",
            "try_to_place_in_kwargs",
        }
    )

    for k in cfg:
        assert (
            k in VALID_CFG_KEYS
        ), f"got invaild key \"{k}\" in {cfg['type']} config {cfg['name']}"
    placement = cfg.get("placement", None)
    if placement is None:
        return
    for k in cfg["placement"]:
        assert (
            k in VALID_PLACEMENT_KEYS
        ), f"got invaild key \"{k}\" in placement config for {cfg['name']}"


def get_single_fixture_sampler(env, cfg, z_offset=0.003):
    """
    Creates a SequentialCompositeSampler for a single fixture,
    skipping if it's an auxiliary fixture without valid placement.
    """

    if type(cfg) == list:
        return _get_placement_initializer(env, cfg, z_offset)
    else:
        return _get_placement_initializer(env, [cfg], z_offset)


def _get_placement_initializer(env, cfg_list, z_offset=0.01):
    """
    Creates a placement initializer for the objects/fixtures based on the specifications in the configurations list.

    Args:
        cfg_list (list): list of object configurations

        z_offset (float): offset in z direction if not specified in cfg

    Returns:
        SequentialCompositeSampler: placement initializer
    """

    from robocasa.models.fixtures import (
        fixture_is_type,
        FixtureType,
    )

    placement_initializer = SequentialCompositeSampler(name="SceneSampler", rng=env.rng)

    def has_auxiliary_pair(cfg_list, name):
        from robocasa.models.fixtures.fixture import Fixture

        """
        Returns true if fixture is a base fixture type eligible for an auxiliary,
        and its corresponding auxiliary name also exists in cfg_list
        """
        if not isinstance(name, str):
            return False

        all_names = {cfg["name"] for cfg in cfg_list}

        parts = name.rsplit("_", 2)
        if len(parts) != 3:
            return False
        base_type, group, group_id = parts
        suffix = f"{group}_{group_id}"

        if base_type not in Fixture.BASE_TO_AUXILIARY_FIXTURES:
            return False

        aux_name = f"{name}_auxiliary_{suffix}"

        return aux_name in all_names

    for (obj_i, cfg) in enumerate(cfg_list):
        _check_cfg_is_valid(cfg)

        if cfg["type"] == "fixture":
            mj_obj = env.fixtures[cfg["name"]]
        elif cfg["type"] == "object":
            mj_obj = env.objects[cfg["name"]]
        else:
            raise ValueError

        placement = cfg.get("placement", None)
        if placement is None:
            continue

        fixture_id = placement.get("fixture", None)
        reference_object = None
        anchor_to = placement.pop("anchor_to", None)

        if anchor_to is not None:
            if cfg["type"] == "fixture":
                anchor_model = env.fixtures[anchor_to]
            elif cfg["type"] == "object":
                anchor_model = env.objects[anchor_to]
            else:
                raise ValueError
            offset = anchor_model.anchor_offset
            new_placement = dict(
                size=(0.0, 0.0),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                sample_args=dict(
                    reference=anchor_to,
                    on_top=False,
                    use_reference_quat=True,
                ),
                offset=(offset[0], offset[1], offset[2] + 0.001),
                rotation=0.0,
            )
            new_placement.update(placement)
            placement = new_placement

        rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))

        if hasattr(mj_obj, "mirror_placement") and mj_obj.mirror_placement:
            rotation = [-rotation[1], -rotation[0]]

        aux_x_offset = 0
        if has_auxiliary_pair(cfg_list, cfg["name"]):
            if hasattr(mj_obj, "anchor_offset"):
                aux_x_offset = mj_obj.anchor_offset[0]

        ensure_object_boundary_in_range = placement.get(
            "ensure_object_boundary_in_range", True
        )
        ensure_valid_placement = placement.get("ensure_valid_placement", True)
        ensure_valid_auxiliary_placement = placement.get(
            "ensure_valid_auxiliary_placement"
        )
        rotation_axis = placement.get("rotation_axis", "z")
        sampler_kwargs = dict(
            name="{}_Sampler".format(cfg["name"]),
            mujoco_objects=mj_obj,
            rng=env.rng,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            ensure_valid_auxiliary_placement=ensure_valid_auxiliary_placement,
            rotation_axis=rotation_axis,
            rotation=rotation,
        )

        if anchor_to is not None:
            target_size = placement.get("size", None)

            x_range = np.array([-target_size[0] / 2, target_size[0] / 2]) + offset[0]
            y_range = np.array([-target_size[1] / 2, target_size[1] / 2]) + offset[1]

            # ref pos doesnt matter since we already have a reference object
            # arbitrarily set to 0
            ref_pos = [0, 0, 0]
            # use reference rotation of base object!
            ref_rot = placement_initializer.samplers[
                f"{anchor_to}_Sampler"
            ].reference_rot
            this_z_offset = offset[2]
        # infer and fill in rest of configs now
        elif fixture_id is None:
            target_size = placement.get("size", None)
            if target_size is None:
                target_size = (0.0, 0.0)
            x_range = np.array([-target_size[0] / 2, target_size[0] / 2])
            y_range = np.array([-target_size[1] / 2, target_size[1] / 2])
            ref_pos = [0, 0, 0]
            ref_rot = 0.0
            this_z_offset = z_offset
        else:
            fixture = env.get_fixture(
                id=fixture_id,
                ref=placement.get("ref", None),
                full_name_check=True if cfg["type"] == "fixture" else False,
            )
            reuse_region_from = placement.get("reuse_region_from", None)
            sample_region_kwargs = placement.get("sample_region_kwargs", {})
            ref_fixture = sample_region_kwargs.get("ref", None)

            # this checks if the reference fixture and dining counter are facing different directions
            ref_dining_counter_mismatch = False
            if fixture_is_type(fixture, FixtureType.DINING_COUNTER) and fixture_is_type(
                ref_fixture, FixtureType.STOOL
            ):
                if abs(abs(ref_fixture.rot) - abs(fixture.rot)) > 0.01:
                    ref_dining_counter_mismatch = True

            ref_obj_name = placement.get("ref_obj", None)

            if ref_obj_name is not None and cfg["name"] != ref_obj_name:
                ref_obj_cfg = find_object_cfg_by_name(env, ref_obj_name)
                reset_region = ref_obj_cfg["reset_region"]
            else:
                if cfg.get("reset_region", None) is not None:
                    reset_region = cfg["reset_region"]
                else:
                    if (
                        ensure_object_boundary_in_range
                        and ensure_valid_placement
                        and rotation_axis == "z"
                    ):
                        sample_region_kwargs["min_size"] = mj_obj.size
                    try:
                        if reuse_region_from is None:
                            reset_region = fixture.sample_reset_region(
                                env=env, **sample_region_kwargs
                            )
                        else:
                            # find and re-use sampling region from another object
                            reset_region = None
                            for this_obj_config in cfg_list:
                                if this_obj_config["name"] == reuse_region_from:
                                    reset_region = this_obj_config["reset_region"]
                                    break
                            assert (
                                reset_region is not None
                            ), "Could not find reset region to reuse"

                    except SamplingError:
                        raise PlacementError("Cannot initialize placement.")
                reference_object = fixture.name

            cfg["reset_region"] = reset_region
            outer_size = reset_region["size"]
            if fixture_is_type(fixture, FixtureType.TOASTER) or fixture_is_type(
                fixture, FixtureType.BLENDER
            ):
                default_margin = 0.0
            else:
                default_margin = 0.04
            margin = placement.get("margin", default_margin)
            outer_size = (outer_size[0] - margin, outer_size[1] - margin)
            assert outer_size[0] > 0 and outer_size[1] > 0

            target_size = placement.get("size", None)
            offset = placement.get("offset", (0.0, 0.0))
            inner_xpos, inner_ypos = placement.get("pos", (None, None))

            if ref_dining_counter_mismatch:
                rel_yaw = fixture.rot - ref_fixture.rot

                target_size = T.rotate_2d_point(target_size, rot=rel_yaw)
                target_size = (abs(target_size[0]), abs(target_size[1]))

                # rotate the pos tuple
                # treat "ref" as sentinel 5 (or -5) so it survives rotation
                placeholder = 5.0
                raw_pos = placement.get("pos", (None, None))
                sampler_kwargs["rotation"] -= rel_yaw

                numeric_pos = []
                for v in raw_pos:
                    if v == None:
                        v = 0.0
                    if v == "ref":
                        numeric_pos.append(placeholder)
                    else:
                        numeric_pos.append(float(v))
                rotated = T.rotate_2d_point(np.array(numeric_pos), rot=rel_yaw)

                def unpack(v):
                    if abs(abs(v) - placeholder) < 1e-2:
                        return "ref"
                    return float(np.clip(v, -1.0, 1.0))

                inner_xpos, inner_ypos = unpack(rotated[0]), unpack(rotated[1])

            stool_orientation = False

            # make sure the offset is relative to the reference fixture
            if fixture_is_type(fixture, FixtureType.DINING_COUNTER) and fixture_is_type(
                ref_fixture, FixtureType.STOOL
            ):
                rel_yaw = np.pi - (fixture.rot - ref_fixture.rot)
                offset = T.rotate_2d_point(offset, rot=rel_yaw)
                epsilon = 1e-2
                off0 = 0.0 if abs(offset[0]) < epsilon else offset[0]
                off1 = 0.0 if abs(offset[1]) < epsilon else offset[1]
                offset = (float(off0), float(off1))

                stool_orientation = True
                inner_xpos_og = inner_xpos
                inner_ypos_og = inner_ypos

            if target_size is not None:
                target_size = deepcopy(list(target_size))
                for size_dim in [0, 1]:
                    if target_size[size_dim] == "obj":
                        target_size[size_dim] = mj_obj.size[size_dim] + 0.005
                        if size_dim == 0:
                            target_size[size_dim] += aux_x_offset
                    if target_size[size_dim] == "obj.x":
                        target_size[size_dim] = mj_obj.size[0] + 0.005 + aux_x_offset
                    if target_size[size_dim] == "obj.y":
                        target_size[size_dim] = mj_obj.size[1] + 0.005
                inner_size = np.min((outer_size, target_size), axis=0)
            else:
                inner_size = outer_size

            # center inner region within outer region
            if inner_xpos == "ref":
                # compute optimal placement of inner region to match up with the reference fixture
                x_halfsize = outer_size[0] / 2 - inner_size[0] / 2
                if x_halfsize == 0.0:
                    inner_xpos = 0.0
                else:
                    ref_fixture = env.get_fixture(sample_region_kwargs["ref"])
                    ref_pos = ref_fixture.pos
                    fixture_to_ref = OU.get_rel_transform(fixture, ref_fixture)[0]
                    outer_to_ref = fixture_to_ref - reset_region["offset"]
                    inner_xpos = outer_to_ref[0] / x_halfsize
                    inner_xpos = np.clip(inner_xpos, a_min=-1.0, a_max=1.0)
            elif inner_xpos is None:
                inner_xpos_og = 0.0
                inner_xpos = 0.0

            if inner_ypos == "ref":
                # compute optimal placement of inner region to match up with the reference fixture
                y_halfsize = outer_size[1] / 2 - inner_size[1] / 2
                if y_halfsize == 0.0:
                    inner_ypos = 0.0
                else:
                    ref_fixture = env.get_fixture(sample_region_kwargs["ref"])
                    ref_pos = ref_fixture.pos
                    fixture_to_ref = OU.get_rel_transform(fixture, ref_fixture)[0]
                    outer_to_ref = fixture_to_ref - reset_region["offset"]
                    inner_ypos = outer_to_ref[1] / y_halfsize
                    inner_ypos = np.clip(inner_ypos, a_min=-1.0, a_max=1.0)
            elif inner_ypos is None:
                inner_ypos_og = 0.0
                inner_ypos = 0.0

            # make sure that the orientation is around stool reference
            if stool_orientation and not ref_dining_counter_mismatch:
                # only skip if both coordinates are "ref"
                if not (inner_xpos_og == "ref" and inner_ypos_og == "ref"):
                    rel_yaw = np.pi - (fixture.rot - ref_fixture.rot)
                    vec = np.array(
                        [
                            0.0 if inner_xpos_og == "ref" else inner_xpos_og,
                            0.0 if inner_ypos_og == "ref" else inner_ypos_og,
                        ]
                    )

                    cos_yaw = np.cos(rel_yaw)
                    sin_yaw = np.sin(rel_yaw)
                    rot_matrix = np.array(
                        [
                            [cos_yaw, -sin_yaw],
                            [sin_yaw, cos_yaw],
                        ]
                    )
                    rotated = rot_matrix @ vec

                    # Update only the non-"ref" values
                    if inner_xpos_og != "ref":
                        inner_xpos = float(np.clip(rotated[0], -1.0, 1.0))
                    if inner_ypos_og != "ref":
                        inner_ypos = float(np.clip(rotated[1], -1.0, 1.0))

            # offset for inner region
            intra_offset = (
                (outer_size[0] / 2 - inner_size[0] / 2) * inner_xpos + offset[0],
                (outer_size[1] / 2 - inner_size[1] / 2) * inner_ypos + offset[1],
            )

            # center surface point of entire region
            ref_pos = fixture.pos + [0, 0, reset_region["offset"][2]]
            ref_rot = fixture.rot

            # x, y, and rotational ranges for randomization
            x_range = (
                np.array([-inner_size[0] / 2, inner_size[0] / 2])
                + reset_region["offset"][0]
                + intra_offset[0]
            )
            y_range = (
                np.array([-inner_size[1] / 2, inner_size[1] / 2])
                + reset_region["offset"][1]
                + intra_offset[1]
            )
            this_z_offset = offset[2] if len(offset) == 3 else z_offset

        x_range_og = x_range.copy()
        if has_auxiliary_pair(cfg_list, cfg["name"]):
            x_range[1] = x_range.mean()

        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                reference_object=reference_object,
                reference_pos=ref_pos,
                reference_rot=ref_rot,
                z_offset=this_z_offset,
                x_range=x_range,
                y_range=y_range,
                **sampler_kwargs,
            ),
            sample_args=placement.get("sample_args", None),
        )

        x_range = x_range_og

        # optional: visualize the sampling region
        if macros.SHOW_SITES is True and (
            target_size[0] > 0.0 and target_size[1] > 0.0
        ):
            """
            show outer reset region
            """
            pos_to_vis = deepcopy(ref_pos)
            pos_to_vis[:2] += T.rotate_2d_point(
                [reset_region["offset"][0], reset_region["offset"][1]], rot=ref_rot
            )
            size_to_vis = np.concatenate(
                [
                    np.abs(
                        T.rotate_2d_point(
                            [outer_size[0] / 2, outer_size[1] / 2], rot=ref_rot
                        )
                    ),
                    [0.001],
                ]
            )
            site_str = """<site type="box" rgba="0 0 1 0.4" size="{size}" pos="{pos}" name="reset_region_outer_{postfix}"/>""".format(
                pos=array_to_string(pos_to_vis),
                size=array_to_string(size_to_vis),
                postfix=cfg["name"],
            )
            site_tree = ET.fromstring(site_str)
            env.model.worldbody.append(site_tree)

            """
            show inner reset region
            """
            pos_to_vis = deepcopy(ref_pos)
            pos_to_vis[:2] += T.rotate_2d_point(
                [np.mean(x_range), np.mean(y_range)], rot=ref_rot
            )
            size_to_vis = np.concatenate(
                [
                    np.abs(
                        T.rotate_2d_point(
                            [
                                (x_range[1] - x_range[0]) / 2,
                                (y_range[1] - y_range[0]) / 2,
                            ],
                            rot=ref_rot,
                        )
                    ),
                    [0.002],
                ]
            )
            site_str = """<site type="box" rgba="1 0 0 0.4" size="{size}" pos="{pos}" name="reset_region_inner_{postfix}"/>""".format(
                pos=array_to_string(pos_to_vis),
                size=array_to_string(size_to_vis),
                postfix=cfg["name"],
            )
            site_tree = ET.fromstring(site_str)
            env.model.worldbody.append(site_tree)

    return placement_initializer


def init_robot_base_pose(env):
    """
    helper function to initialize robot base pose
    """
    # set robot position
    if env.init_robot_base_ref is not None:
        ref_fixture = env.get_fixture(env.init_robot_base_ref)
    else:
        fixtures = list(env.fixtures.values())
        valid_ref_fixture_classes = [
            "CoffeeMachine",
            "Toaster",
            "ToasterOven",
            "Stove",
            "Stovetop",
            "SingleCabinet",
            "HingeCabinet",
            "OpenCabinet",
            "Drawer",
            "Microwave",
            "Sink",
            "Hood",
            "Oven",
            "Fridge",
            "Dishwasher",
        ]
        while True:
            ref_fixture = env.rng.choice(fixtures)
            fxtr_class = type(ref_fixture).__name__
            if fxtr_class not in valid_ref_fixture_classes:
                continue
            break

    ref_object = None
    for cfg in env.object_cfgs:
        if cfg.get("init_robot_here", None) is True:
            ref_object = cfg.get("name")
            break

    robot_base_pos, robot_base_ori = compute_robot_base_placement_pose(
        env,
        ref_fixture=ref_fixture,
        ref_object=ref_object,
    )

    return robot_base_pos, robot_base_ori


def find_object_cfg_by_name(env, name):
    """
    Finds and returns the object configuration with the given name.

    Args:
        name (str): name of the object configuration to find

    Returns:
        dict: object configuration with the given name
    """
    for cfg in env.object_cfgs:
        if cfg["name"] == name:
            return cfg
    raise ValueError


def create_obj(env, cfg):
    """
    Helper function for creating objects.
    Called by _create_objects()
    """
    from robocasa.models.fixtures import (
        fixture_is_type,
        FixtureType,
    )

    if "info" in cfg:
        """
        if cfg has "info" key in it, that means it is storing meta data already
        that indicates which object we should be using.
        set the obj_groups to this path to do deterministic playback
        """
        mjcf_path = cfg["info"]["mjcf_path"]
        # replace with correct base path
        mjcf_path = mjcf_path.replace(
            "\\", "/"
        )  # replace windows backslashes with forward slashes
        new_base_path = os.path.join(robocasa.models.assets_root, "objects")
        new_path = os.path.join(new_base_path, mjcf_path.split("/objects/")[-1])
        obj_groups = new_path
        exclude_obj_groups = None
    else:
        obj_groups = cfg.get("obj_groups", "all")
        exclude_obj_groups = cfg.get("exclude_obj_groups", None)

    if not isinstance(obj_groups, list) and isinstance(obj_groups, tuple):
        obj_groups = list(obj_groups)

    graspable = cfg.get("graspable", None)
    washable = cfg.get("washable", None)
    microwavable = cfg.get("microwavable", None)
    cookable = cfg.get("cookable", None)
    freezable = cfg.get("freezable", None)
    fridgable = cfg.get("fridgable", None)
    dishwashable = cfg.get("dishwashable", None)
    if "placement" in cfg and "fixture" in cfg["placement"]:
        ref_fixture = cfg["placement"]["fixture"]
        if fixture_is_type(ref_fixture, FixtureType.SINK):
            washable = True
        elif fixture_is_type(ref_fixture, FixtureType.DISHWASHER):
            dishwashable = True
        elif fixture_is_type(ref_fixture, FixtureType.MICROWAVE):
            microwavable = True
        elif fixture_is_type(ref_fixture, FixtureType.STOVE):
            if any(
                cat in obj_groups
                for cat in ["pan", "kettle_non_electric", "pot", "saucepan", "cookware"]
            ):
                cookable = False
            else:
                cookable = True
        elif fixture_is_type(ref_fixture, FixtureType.OVEN):
            # hack for cake, it is bakeable but not cookable.
            # we don't have a bakeable category, so using cookable category in place
            # however, cakes are not cookable in general, only bakable.
            # so we are making an exception here.
            if any(
                cat in obj_groups
                for cat in ["oven_tray", "pan", "pot", "saucepan", "cake"]
            ):
                cookable = False
            else:
                cookable = True
        elif fixture_is_type(ref_fixture, FixtureType.FRIDGE):
            fridgable = True

    object_kwargs, object_info = env.sample_object(
        obj_groups,
        exclude_groups=exclude_obj_groups,
        graspable=graspable,
        washable=washable,
        microwavable=microwavable,
        cookable=cookable,
        fridgable=fridgable,
        freezable=freezable,
        dishwashable=dishwashable,
        max_size=cfg.get("max_size", (None, None, None)),
        object_scale=cfg.get("object_scale", None),
        rotate_upright=cfg.get("rotate_upright", False),
    )
    info = object_info

    object = MJCFObject(name=cfg["name"], **object_kwargs)

    return object, info


@contextlib.contextmanager
def no_collision(sim):
    """
    A context manager that temporarily disables all collision interactions in the simulation.
    Args:
        sim (MjSim): The simulation object where collision interactions will be temporarily disabled.
    Yields:
        None: The function yields control back to the caller while collisions remain disabled.
    Upon exiting the context, the original collision settings are restored.
    """
    original_contype = sim.model.geom_contype.copy()
    original_conaffinity = sim.model.geom_conaffinity.copy()
    sim.model.geom_contype[:] = 0
    sim.model.geom_conaffinity[:] = 0
    try:
        yield
    finally:
        sim.model.geom_contype = original_contype
        sim.model.geom_conaffinity = original_conaffinity


def detect_robot_collision(env):
    """
    Checks if the robot has a collision with any placed fixtures/objects.
    Returns:
        bool: True if a collision is detected between the robot and any other fixtures/objects, False otherwise.
    """
    if env.robot_geom_ids is None:
        env.robot_geom_ids = set()
        robot_geoms = find_elements(
            root=env.robots[0].robot_model.root, tags="geom", return_first=False
        )
        for robot_geom in robot_geoms:
            env.robot_geom_ids.add(env.sim.model.geom_name2id(robot_geom.get("name")))
    for i in range(env.sim.data.ncon):
        geom1 = env.sim.data.contact[i].geom1
        geom2 = env.sim.data.contact[i].geom2
        if (geom1 in env.robot_geom_ids and geom2 not in env.robot_geom_ids) or (
            geom2 in env.robot_geom_ids and geom1 not in env.robot_geom_ids
        ):
            return True
    return False


def generate_random_robot_pos(env, anchor_pos, anchor_ori, pos_dev_x, pos_dev_y):
    local_deviation = env.rng.uniform(
        low=(-pos_dev_x, -pos_dev_y),
        high=(pos_dev_x, pos_dev_y),
    )
    local_deviation = np.concatenate((local_deviation, [0.0]))
    global_deviation = np.matmul(
        T.euler2mat(anchor_ori + [0, 0, np.pi / 2]), -local_deviation
    )
    return anchor_pos + global_deviation


def set_robot_to_position(env, global_pos):
    local_pos = np.matmul(
        T.matrix_inverse(T.euler2mat(env.init_robot_base_ori_anchor)), global_pos
    )
    undo_pos = np.matmul(
        T.matrix_inverse(T.euler2mat(env.init_robot_base_ori_anchor)),
        [-10.0, -10.0, 0.0],
    )

    # check the axis of the mobile base joints to determine which
    # joint moves in the x direction and which moves in the y
    x_jnt_name, y_jnt_name = (
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_side",
    )
    if (
        env.sim.model.jnt_axis[
            env.sim.model.joint_name2id("mobilebase0_joint_mobile_forward")
        ]
        == np.array([0, 1, 0])
    ).all():
        x_jnt_name, y_jnt_name = (
            "mobilebase0_joint_mobile_side",
            "mobilebase0_joint_mobile_forward",
        )

    with no_collision(env.sim):
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(x_jnt_name)] = (
            undo_pos[0] + local_pos[0]
        )
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(y_jnt_name)] = (
            undo_pos[1] + local_pos[1]
        )

        env.sim.forward()


def set_robot_base(
    env,
    anchor_pos,
    anchor_ori,
    rot_dev,
    pos_dev_x,
    pos_dev_y,
):
    """
    Sets the initial state of the robot by randomizing its position and orientation within defined deviation limits.
    The deviation limits are provided by `self.robot_spawn_position_deviation_x`, `self.robot_spawn_position_deviation_y`,
    and `self.robot_spawn_rotation_deviation`.
    Raises:
        RandomizationError: If the robot cannot be placed without collisions.
    """
    assert len(env.robots) == 1
    # assert isinstance(self.robots[0].robot_model, PandaOmron) or isinstance(
    #     self.robots[0].robot_model, GR1FloatingBody
    # )

    with no_collision(env.sim):
        env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")
        ] = env.rng.uniform(-rot_dev, rot_dev)
        # l_elbow_pitch_id = self.sim.model.get_joint_qpos_addr("robot0_l_elbow_pitch")
        # l_elbow_pitch_range = self.sim.model.jnt_range[l_elbow_pitch_id]
        # self.sim.data.qpos[
        #     self.sim.model.get_joint_qpos_addr("robot0_l_elbow_pitch")
        # ] = (l_elbow_pitch_range[0] * 0.6 + l_elbow_pitch_range[1] * 0.4)
        env.sim.forward()

    initial_state_copy = env.sim.get_state()

    # Try to move for 100 times to resolve any collision
    found_valid = False
    import time

    # t1 = time.time()
    cur_dev_pos_x = pos_dev_x
    cur_dev_pos_y = pos_dev_y
    while found_valid is not True:
        # try up to 50 times
        for attempt_position in range(50):
            robot_pos = generate_random_robot_pos(
                env=env,
                anchor_pos=anchor_pos,
                anchor_ori=anchor_ori,
                pos_dev_x=cur_dev_pos_x,
                pos_dev_y=cur_dev_pos_y,
            )
            set_robot_to_position(env, robot_pos)
            env.sim.forward()
            if not detect_robot_collision(env):
                found_valid = True
                break

            env.sim.set_state(initial_state_copy)

        # print(time.time() - t1, attempt_position)

        # if valid position not found, increase range by 10 cm for x and 5 cm for y
        cur_dev_pos_x += 0.10
        cur_dev_pos_y += 0.05

    return robot_pos


if __name__ == "__main__":
    # select random task to run rollouts for
    env_name = np.random.choice(
        list(ATOMIC_TASK_DATASETS) + list(COMPOSITE_TASK_DATASETS)
    )
    env = create_env(env_name=env_name)
    info = run_random_rollouts(
        env, num_rollouts=3, num_steps=100, video_path="/tmp/target.mp4"
    )
