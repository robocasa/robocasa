from robocasa.utils.dataset_registry import (
    get_ds_path,
    SINGLE_STAGE_TASK_DATASETS,
    MULTI_STAGE_TASK_DATASETS,
)
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset
from robosuite.controllers import load_composite_controller_config
import os
import robosuite
import robocasa
import imageio
import numpy as np
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
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
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
        translucent_robot=False,
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
            obj_pos = env.object_placements[ref_object][0]
            abs_sites = ground_fixture.get_ext_sites(relative=False)
            dist1 = np.linalg.norm(obj_pos - abs_sites[0])
            dist2 = np.linalg.norm(obj_pos - abs_sites[2])
            if dist1 < dist2:
                face_dir = 1
            else:
                face_dir = -1
        else:
            face_dir = env.rng.choice([-1, 1])

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
        if fixture_id is not None:
            # get fixture to place object on
            fixture = env.get_fixture(
                id=fixture_id,
                ref=placement.get("ref", None),
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
                    ref_fixture = env.get_fixture(
                        placement["sample_region_kwargs"]["ref"]
                    )
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
                    ref_fixture = env.get_fixture(
                        placement["sample_region_kwargs"]["ref"]
                    )
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
            rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
        else:
            target_size = placement.get("size", None)
            x_range = np.array([-target_size[0] / 2, target_size[0] / 2])
            y_range = np.array([-target_size[1] / 2, target_size[1] / 2])
            rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
            ref_pos = [0, 0, 0]
            ref_rot = 0.0

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

        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="{}_Sampler".format(cfg["name"]),
                mujoco_objects=mj_obj,
                x_range=x_range,
                y_range=y_range,
                rotation=rotation,
                ensure_object_boundary_in_range=placement.get(
                    "ensure_object_boundary_in_range", True
                ),
                ensure_valid_placement=placement.get("ensure_valid_placement", True),
                reference_pos=ref_pos,
                reference_rot=ref_rot,
                z_offset=z_offset,
                rng=env.rng,
                rotation_axis=placement.get("rotation_axis", "z"),
            ),
            sample_args=placement.get("sample_args", None),
        )

    return placement_initializer


def init_robot_base_pose(env):
    """
    helper function to initialize robot base pose
    """
    # set robot position
    if env.init_robot_base_pos is not None:
        ref_fixture = env.get_fixture(env.init_robot_base_pos)
    else:
        fixtures = list(env.fixtures.values())
        valid_ref_fixture_classes = [
            "CoffeeMachine",
            "Toaster",
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
    robot_model = env.robots[0].robot_model
    robot_model.set_base_xpos(robot_base_pos)
    robot_model.set_base_ori(robot_base_ori)


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


if __name__ == "__main__":
    # select random task to run rollouts for
    env_name = np.random.choice(
        list(SINGLE_STAGE_TASK_DATASETS) + list(MULTI_STAGE_TASK_DATASETS)
    )
    env = create_env(env_name=env_name)
    info = run_random_rollouts(
        env, num_rollouts=3, num_steps=100, video_path="/tmp/test.mp4"
    )
