from robocasa.utils.dataset_registry import (
    get_ds_path,
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)
from robocasa.scripts.dataset_scripts.playback_dataset import (
    get_env_metadata_from_dataset,
)
from robosuite.controllers import load_composite_controller_config
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import (
    array_to_string,
    find_elements,
)
import xml.etree.ElementTree as ET
import os
import robosuite
import robocasa
import imageio
import numpy as np
import contextlib
from tqdm import tqdm
from termcolor import colored
from copy import deepcopy

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


def run_random_rollouts(env, num_rollouts, num_steps, video_path=None):
    video_writer = None
    if video_path is not None:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    info = {}
    num_success_rollouts = 0
    for rollout_i in tqdm(range(num_rollouts)):
        obs = env.reset()
        for step_i in range(num_steps):
            # sample and execute random action
            action = np.random.uniform(low=env.action_spec[0], high=env.action_spec[1])
            # hack for panda robot: TODO remove
            action[-5:-1] = 0.0
            obs, _, _, _ = env.step(action)

            if video_writer is not None:
                video_img = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(video_img)

            if env._check_success():
                num_success_rollouts += 1
                break

    if video_writer is not None:
        video_writer.close()
        print(colored(f"Saved video of rollouts to {video_path}", color="yellow"))

    info["num_success_rollouts"] = num_success_rollouts

    return info


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

    # set the base fixture as the ref fixture itself if cannot find fixture containing ref
    if ground_fixture is None:
        ground_fixture = ref_fixture
    # assert base_fixture is not None

    # step 2: compute offset relative to this counter
    ground_to_ref, _ = OU.get_rel_transform(ground_fixture, ref_fixture)

    face_dir = 1  # 1 is facing front of fixture, -1 is facing south end of fixture
    if fixture_is_type(ground_fixture, FixtureType.DINING_COUNTER):
        # for dining counters, can face either north of south end of fixture
        if ref_object is not None:
            # choose the end that is closest to the ref object
            ref_point = env.object_placements[ref_object][0]
        else:
            ### find the side closest to the ref fixture ###
            ref_point = ref_fixture.pos

        abs_sites = ground_fixture.get_ext_sites(relative=False)
        dist1 = np.linalg.norm(ref_point - abs_sites[0])
        dist2 = np.linalg.norm(ref_point - abs_sites[2])
        if dist1 < dist2:
            face_dir = 1
        else:
            face_dir = -1

    fixture_ext_sites = ground_fixture.get_ext_sites(relative=True)
    fixture_to_robot_offset = np.zeros(3)

    # set x offset
    fixture_to_robot_offset[0] = ground_to_ref[0]

    # set y offset
    if face_dir == 1:
        fixture_p0 = fixture_ext_sites[0]
        fixture_to_robot_offset[1] = fixture_p0[1] - 0.20
    elif face_dir == -1:
        fixture_py = fixture_ext_sites[2]
        fixture_to_robot_offset[1] = fixture_py[1] + 0.20

    if offset is not None:
        fixture_to_robot_offset[0] += offset[0]
        fixture_to_robot_offset[1] += offset[1]
    elif ref_object is not None:
        sampler = env.placement_initializer.samplers[f"{ref_object}_Sampler"]
        fixture_to_robot_offset[0] += np.mean(sampler.x_range)

    if (
        isinstance(ground_fixture, HousingCabinet)
        or isinstance(ground_fixture, Fridge)
        or "stack" in ground_fixture.name
    ):
        fixture_to_robot_offset[1] += face_dir * -0.10

    # move back a bit for the stools
    if fixture_is_type(ground_fixture, FixtureType.DINING_COUNTER):
        abs_sites = ground_fixture.get_ext_sites(relative=False)
        stool = env.get_fixture(FixtureType.STOOL)
        dist1 = np.linalg.norm(stool.pos - abs_sites[0])
        dist2 = np.linalg.norm(stool.pos - abs_sites[2])
        if (dist1 < dist2 and face_dir == 1) or (dist1 > dist2 and face_dir == -1):
            fixture_to_robot_offset[1] += -0.15 * face_dir

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
            "freezable",
            "max_size",
            "object_scale",
            "placement",
            "info",
            "init_robot_here",
            "reset_region",
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
            "sample_args",
            "sample_region_kwargs",
            "ref_obj",
            "fixture",
            "try_to_place_in",
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


def _get_placement_initializer(env, cfg_list, z_offset=0.01):

    """
    Creates a placement initializer for the objects/fixtures based on the specifications in the configurations list

    Args:
        cfg_list (list): list of object configurations

        z_offset (float): offset in z direction

    Returns:
        SequentialCompositeSampler: placement initializer

    """

    placement_initializer = SequentialCompositeSampler(name="SceneSampler", rng=env.rng)

    for (obj_i, cfg) in enumerate(cfg_list):
        # check cfg keys are set up correctly
        _check_cfg_is_valid(cfg)

        # determine which object is being placed
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

        rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
        if hasattr(mj_obj, "mirror_placement") and mj_obj.mirror_placement:
            rotation = [-rotation[1], -rotation[0]]

        # set up placement sampler kwargs
        sampler_kwargs = dict(
            name="{}_Sampler".format(cfg["name"]),
            mujoco_objects=mj_obj,
            rng=env.rng,
            ensure_object_boundary_in_range=placement.get(
                "ensure_object_boundary_in_range", True
            ),
            ensure_valid_placement=placement.get("ensure_valid_placement", True),
            rotation_axis=placement.get("rotation_axis", "z"),
            rotation=rotation,
        )

        # infer and fill in rest of configs now
        if fixture_id is None:
            target_size = placement.get("size", None)
            x_range = np.array([-target_size[0] / 2, target_size[0] / 2])
            y_range = np.array([-target_size[1] / 2, target_size[1] / 2])

            ref_pos = [0, 0, 0]
            ref_rot = 0.0
        else:
            # get fixture to place object on
            fixture = env.get_fixture(
                id=fixture_id,
                ref=placement.get("ref", None),
                full_name_check=True if cfg["type"] == "fixture" else False,  # hack
            )

            # calculate the total available space where object could be placed
            sample_region_kwargs = placement.get("sample_region_kwargs", {})
            ref_obj_name = placement.get("ref_obj", None)
            if ref_obj_name is not None and cfg["name"] != ref_obj_name:
                ref_obj_cfg = find_object_cfg_by_name(env, ref_obj_name)
                reset_region = ref_obj_cfg["reset_region"]
            else:
                reset_region = fixture.sample_reset_region(
                    env=env, **sample_region_kwargs
                )
                reference_object = fixture.name
            cfg["reset_region"] = reset_region
            outer_size = reset_region["size"]
            margin = placement.get("margin", 0.04)
            outer_size = (outer_size[0] - margin, outer_size[1] - margin)

            # calculate the size of the inner region where object will actually be placed
            target_size = placement.get("size", None)
            if target_size is not None:
                target_size = deepcopy(list(target_size))
                for size_dim in [0, 1]:
                    if target_size[size_dim] == "obj":
                        target_size[size_dim] = mj_obj.size[size_dim] + 0.005
                    if target_size[size_dim] == "obj.x":
                        target_size[size_dim] = mj_obj.size[0] + 0.005
                    if target_size[size_dim] == "obj.y":
                        target_size[size_dim] = mj_obj.size[1] + 0.005
                inner_size = np.min((outer_size, target_size), axis=0)
            else:
                inner_size = outer_size

            inner_xpos, inner_ypos = placement.get("pos", (None, None))
            offset = placement.get("offset", (0.0, 0.0))

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
                inner_ypos = 0.0

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

        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                reference_object=reference_object,
                reference_pos=ref_pos,
                reference_rot=ref_rot,
                z_offset=z_offset,
                x_range=x_range,
                y_range=y_range,
                **sampler_kwargs,
            ),
            sample_args=placement.get("sample_args", None),
        )

        # optional: visualize the sampling region
        if macros.SHOW_SITES is True:
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
                postfix=str(obj_i),
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
                postfix=str(obj_i),
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
    object_kwargs, object_info = env.sample_object(
        obj_groups,
        exclude_groups=exclude_obj_groups,
        graspable=cfg.get("graspable", None),
        washable=cfg.get("washable", None),
        microwavable=cfg.get("microwavable", None),
        cookable=cfg.get("cookable", None),
        freezable=cfg.get("freezable", None),
        max_size=cfg.get("max_size", (None, None, None)),
        object_scale=cfg.get("object_scale", None),
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


def generate_random_robot_pos(anchor_pos, anchor_ori, pos_dev_x, pos_dev_y):
    local_deviation = np.random.uniform(
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
    with no_collision(env.sim):
        env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_side")
        ] = (undo_pos[0] + local_pos[0])
        env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_forward")
        ] = (undo_pos[1] + local_pos[1])

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
        # l_elbow_pitch_id = self.sim.model.joint_name2id("robot0_l_elbow_pitch")
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
        list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS)
    )
    env = create_env(env_name=env_name)
    info = run_random_rollouts(
        env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
    )
