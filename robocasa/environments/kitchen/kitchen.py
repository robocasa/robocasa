import numpy as np
import xml.etree.ElementTree as ET
import random
from copy import deepcopy
import os
from scipy.spatial.transform import Rotation

from robosuite.models.tasks import ManipulationTask
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.mjcf_utils import xml_path_completion, find_elements, array_to_string
from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T
from robosuite.utils.errors import RandomizationError

import robocasa
from robocasa.models.arenas import KitchenArena
from robocasa.models.objects.kitchen_objects import sample_kitchen_object
from robocasa.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robocasa.models.objects.objects import MJCFObject
import robocasa.utils.kitchen_utils as KU
import robocasa.utils.object_utils as OU
# from robocasa.models.objects.fixtures import (
#     FixtureType, Fixture, fixture_is_type, Counter, Stove, Stovetop, HousingCabinet, Fridge, Dishwasher
# )
from robocasa.models.objects.fixtures import *
# from robocasa.models.arenas.layout_utils import *
from robocasa.utils.texture_swap import (
    get_random_textures,
    replace_cab_textures,
    replace_counter_top_texture,
    replace_wall_texture,
    replace_floor_texture,
)
import robocasa.macros as macros


class Kitchen(ManipulationEnv):
    EXCLUDE_LAYOUTS = []
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True, # for backwards compatibility
        reward_scale=1.0, # for backwards compatibility
        reward_shaping=False, # for backwards compatibility
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        renderer="mujoco",
        renderer_config=None,
        init_robot_base_pos=None,
        seed=None,
        scene_split=None,
        layout_and_style_ids=None,
        layout_ids=None,
        style_ids=None,
        generative_textures=None,
        obj_registries=("objaverse",),
        obj_instance_split=None,
        use_distractors=False,
        translucent_robot=False,
        randomize_cameras=False,
    ):
        self.init_robot_base_pos = init_robot_base_pos

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.obj_registries = obj_registries
        self.obj_instance_split = obj_instance_split
        
        # set layouts and styles
        self.scene_split = scene_split
        if scene_split is not None:
            assert layout_ids is None and style_ids is None and layout_and_style_ids is None
            self.layout_and_style_ids = KU.SCENE_SPLITS[scene_split]
        elif layout_and_style_ids is not None:
            assert layout_ids is None and style_ids is None
            self.layout_and_style_ids = layout_and_style_ids
        else:
            if layout_ids is None or layout_ids == -1 or layout_ids == [-1]:
                layout_ids = list(range(10))
            elif layout_ids == -2 or layout_ids == [-2]: # NOT involving islands/wall stacks
                layout_ids = [0, 2, 4, 5, 7]
            elif layout_ids == -3 or layout_ids == [-3]: # involving islands/wall stacks
                layout_ids = [1, 3, 6, 8, 9]
            elif layout_ids == -4 or layout_ids == [-4]: #layouts with dinings areas
                layout_ids = [1, 3, 6, 7, 8, 9]
            if not isinstance(layout_ids, list):
                layout_ids = [layout_ids]
            
            if style_ids is None or style_ids == -1 or style_ids == [-1]:
                style_ids = list(range(11))
            if not isinstance(style_ids, list):
                style_ids = [style_ids]
            
            self.layout_and_style_ids = []
            for l in layout_ids:
                for s in style_ids:
                    self.layout_and_style_ids.append([l, s])

        # remove excluded layouts
        self.layout_and_style_ids = [(l, s) for (l, s) in self.layout_and_style_ids if l not in self.EXCLUDE_LAYOUTS]
        
        assert generative_textures in [None, False, "100p"]
        self.generative_textures = generative_textures

        self.use_distractors = use_distractors
        self.translucent_robot = translucent_robot
        self.randomize_cameras = randomize_cameras

        # intialize cameras
        self._cam_configs = deepcopy(KU.CAM_CONFIGS)

        initial_qpos = None
        if isinstance(robots, str):
            robots = [robots]
        if robots[0] == "PandaMobile":
            initial_qpos=(-0.01612974, -1.03446714, -0.02397936, -2.27550888, 0.03932365, 1.51639493, 0.69615947),

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            composite_controller_configs={"type": "HYBRID_MOBILE_BASE"},
            base_types=base_types,
            gripper_types=gripper_types,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=True,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # determine sample layout and style
        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            self.layout_id = self._ep_meta["layout_id"]
            self.style_id = self._ep_meta["style_id"]
        else:
            layout_id, style_id = self.rng.choice(self.layout_and_style_ids)
            self.layout_id = int(layout_id)
            self.style_id = int(style_id)

        if macros.VERBOSE:
            print("layout: {}, style: {}".format(self.layout_id, self.style_id))
        
        # to be set later inside edit_model_xml function
        self._curr_gen_fixtures = None

        # setup scene
        if self.layout_id not in KU.LAYOUTS:
            raise ValueError("Invalid layout id: \"{}\"".format(self.layout_id))
        self.mujoco_arena = KitchenArena(
            xml_path_completion(KU.LAYOUTS[self.layout_id], root=robocasa.models.assets_root),
            style=self.style_id,
        )
        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])
        self.set_cameras() # setup cameras

        # setup rendering for this layout
        if self.renderer == "mjviewer":
            camera_config = KU.LAYOUT_CAMS.get(self.layout_id, KU.DEFAULT_LAYOUT_CAM)
            self.renderer_config = {"cam_config": camera_config}

        # setup fixtures
        self.fixture_cfgs = self.mujoco_arena.get_fixture_cfgs()
        self.fixtures = {cfg["name"]: cfg["model"] for cfg in self.fixture_cfgs}

        # setup scene, robots, objects
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.fixtures.values()),
        )

        # if applicable: initialize the fixture locations
        fxtr_placement_initializer = self._get_placement_initializer(self.fixture_cfgs, z_offset=0.0)
        fxtr_placements = None
        for i in range(10):
            try:
                fxtr_placements = fxtr_placement_initializer.sample()
            except RandomizationError as e:
                if macros.VERBOSE:
                    print("Ranomization error in initial placement. Try #{}".format(i))
                continue
            break
        if fxtr_placements is None:
            if macros.VERBOSE:
                print("Could not place fixtures. Trying again with self._load_model()")
            self._load_model()
            return
        self.fxtr_placements = fxtr_placements
        # Loop through all objects and reset their positions
        for obj_pos, obj_quat, obj in fxtr_placements.values():
            assert isinstance(obj, Fixture)
            obj.set_pos(obj_pos)
            
            # hacky code to set orientation
            obj.set_euler(T.mat2euler(T.quat2mat(T.convert_quat(obj_quat, "xyzw"))))
        
        # setup internal references related to fixtures
        self._setup_kitchen_references()

        # set robot position
        if self.init_robot_base_pos is not None:
            ref_fixture = self.get_fixture(self.init_robot_base_pos)
        else:
            choices = [name for (name, fxtr) in self.fixtures.items() if not isinstance(fxtr, Wall)]
            fixture_name = self.rng.choice(choices)
            ref_fixture = self.fixtures[fixture_name]        
        self.planned_robot_base_xpos, self.planned_robot_base_ori = \
            self.compute_robot_base_placement_pose(ref_fixture=ref_fixture)
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(self.planned_robot_base_xpos)
        robot_model.set_base_ori(self.planned_robot_base_ori)

        # helper function for creating objects
        def _create_obj(cfg):
            obj_groups = cfg.get("obj_groups", "all")
            exclude_obj_groups = cfg.get("exclude_obj_groups", None)
            object_kwargs, object_info = self.sample_object(
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
            if "name" not in cfg:
                cfg["name"] = "obj_{}".format(obj_num+1)
            info = object_info

            object = MJCFObject(
                name=cfg["name"],
                **object_kwargs
            )

            return object, info
        
        # add objects
        self.objects = {}
        if "object_cfgs" in self._ep_meta:
            self.object_cfgs = self._ep_meta["object_cfgs"]
            for obj_num, cfg in enumerate(self.object_cfgs):
                model, info = _create_obj(cfg)
                cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])
        else:
            self.object_cfgs = self._get_obj_cfgs()
            addl_obj_cfgs = []
            for obj_num, cfg in enumerate(self.object_cfgs):
                cfg["type"] = "object"
                model, info = _create_obj(cfg)
                cfg["info"] = info
                self.objects[model.name] = model
                self.model.merge_objects([model])

                try_to_place_in = cfg["placement"].get("try_to_place_in", None)
                if try_to_place_in and ("in_container" in cfg["info"]["groups_containing_sampled_obj"]):
                    container_cfg = {}
                    container_cfg["name"] = cfg["name"] + "_container"
                    container_cfg["obj_groups"] = try_to_place_in
                    container_cfg["placement"] = deepcopy(cfg["placement"])
                    container_cfg["type"] = "object"

                    container_kwargs = cfg["placement"].get("container_kwargs", None)
                    if container_kwargs is not None:
                        for k, v in container_kwargs.values():
                            container_cfg[k] = v

                    # # increase size for conainer placement
                    # container_cfg["placement"]["size"] = (
                    #     container_cfg["placement"]["size"][0] + 0.15,
                    #     container_cfg["placement"]["size"][1] + 0.15,
                    # )

                    # add in the new object to the model
                    addl_obj_cfgs.append(container_cfg)
                    model, info = _create_obj(container_cfg)
                    container_cfg["info"] = info
                    self.objects[model.name] = model
                    self.model.merge_objects([model])
                    
                    # modify object config to lie inside of container
                    cfg["placement"] = dict(
                        size=(0.01, 0.01),
                        ensure_object_boundary_in_range=False,
                        sample_args=dict(
                            reference=container_cfg["name"],
                        ),
                    )

            # prepend the new object configs in
            self.object_cfgs = addl_obj_cfgs + self.object_cfgs

            # # remove objects that didn't get created
            # self.object_cfgs = [cfg for cfg in self.object_cfgs if "model" in cfg]
        self.placement_initializer = self._get_placement_initializer(self.object_cfgs)

        object_placements = None
        for i in range(1):
            try:
                object_placements = self.placement_initializer.sample(placed_objects=self.fxtr_placements)
            except RandomizationError as e:
                if macros.VERBOSE:
                    print("Ranomization error in initial placement. Try #{}".format(i))
                continue
            
            break
        if object_placements is None:
            if macros.VERBOSE:
                print("Could not place objects. Trying again with self._load_model()")
            self._load_model()
            return
        self.object_placements = object_placements

    def _setup_kitchen_references(self):
        """
        setup fixtures (and their references). this function is called within load_model function for kitchens
        """
        serialized_refs = self._ep_meta.get("fixture_refs", {})
        # unserialize refs
        self.fixture_refs = {k: self.get_fixture(v) for (k, v) in serialized_refs.items()}

    def _reset_observables(self):
        if self.hard_reset:
            self._observables = self._setup_observables()
    
    def compute_robot_base_placement_pose(self, ref_fixture, offset=None):
        """
        steps:
        1. find the nearest counter to this fixture
        2. compute offset relative to this counter
        3. transform offset to global coordinates
        """
        # step 1: find vase fixture closest to robot
        base_fixture = None

        # get all base fixtures in the environment
        base_fixtures = [
            fxtr for fxtr in self.fixtures.values()
            if isinstance(fxtr, Counter)
            or isinstance(fxtr, Stove)
            or isinstance(fxtr, Stovetop)
            or isinstance(fxtr, HousingCabinet)
            or isinstance(fxtr, Fridge)
        ]
        
        for fxtr in base_fixtures:
            # get bounds of fixture
            point = ref_fixture.pos
            if not OU.point_in_fixture(point=point, fixture=fxtr, only_2d=True):
                continue
            base_fixture = fxtr
            break
        
        # set the base fixture as the ref fixture itself if cannot find fixture containing ref
        if base_fixture is None:
            base_fixture = ref_fixture
        # assert base_fixture is not None

        # step 2: compute offset relative to this counter
        base_to_ref, _ = OU.get_rel_transform(base_fixture, ref_fixture)
        cntr_y = base_fixture.get_ext_sites(relative=True)[0][1]
        base_to_edge = [
            base_to_ref[0],
            cntr_y - 0.20,
            0,
        ]
        if offset is not None:
            base_to_edge[0] += offset[0]
            base_to_edge[1] += offset[1]

        if isinstance(base_fixture, HousingCabinet) or isinstance(base_fixture, Fridge) or "stack" in base_fixture.name:
            base_to_edge[1] -= 0.10

        # step 3: transform offset to global coordinates
        robot_base_pos = np.zeros(3)
        robot_base_pos[0:2] = OU.get_pos_after_rel_offset(base_fixture, base_to_edge)[0:2]
        robot_base_ori = np.array([0, 0, base_fixture.rot + np.pi / 2])

        return robot_base_pos, robot_base_ori

    def _get_placement_initializer(self, cfg_list, z_offset=0.01):
        placement_initializer = SequentialCompositeSampler(name="SceneSampler", rng=self.rng)
        
        for (obj_i, cfg) in enumerate(cfg_list):
            # determine which object is being placed
            if cfg["type"] == "fixture":
                mj_obj = self.fixtures[cfg["name"]]
            elif cfg["type"] == "object":
                mj_obj = self.objects[cfg["name"]]
            else:
                raise ValueError

            placement = cfg.get("placement", None)
            if placement is None:
                continue
            fixture_id = placement.get("fixture", None)
            if fixture_id is not None:
                # get fixture to place object on
                fixture = self.get_fixture(
                    id=fixture_id,
                    ref=placement.get("ref", None),
                )

                # calculate the total available space where object could be placed
                sample_region_kwargs = placement.get("sample_region_kwargs", {})
                reset_region = fixture.sample_reset_region(env=self, **sample_region_kwargs)
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
                        ref_fixture = self.get_fixture(placement["sample_region_kwargs"]["ref"])
                        ref_pos = ref_fixture.pos
                        fixture_to_ref = OU.get_rel_transform(fixture, ref_fixture)[0]
                        outer_to_ref = fixture_to_ref - reset_region["offset"]
                        inner_xpos = outer_to_ref[0] / x_halfsize
                        inner_xpos = np.clip(inner_xpos, a_min=-1.0, a_max=1.0)
                elif inner_xpos is None:
                    inner_xpos = 0.0
                
                if inner_ypos is None:
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
                x_range = np.array([-inner_size[0] / 2, inner_size[0] / 2]) + reset_region["offset"][0] + intra_offset[0]
                y_range = np.array([-inner_size[1] / 2, inner_size[1] / 2]) + reset_region["offset"][1] + intra_offset[1]
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
                pos_to_vis[:2] += T.rotate_2d_point([reset_region["offset"][0], reset_region["offset"][1]], rot=ref_rot)
                size_to_vis = np.concatenate([
                    np.abs(T.rotate_2d_point([outer_size[0] / 2, outer_size[1] / 2], rot=ref_rot)),
                    [0.001]
                ])
                site_str = """<site type="box" rgba="0 0 1 0.4" size="{size}" pos="{pos}" name="reset_region_outer_{postfix}"/>""".format(
                    pos=array_to_string(pos_to_vis),
                    size=array_to_string(size_to_vis),
                    postfix=str(obj_i),
                )
                site_tree = ET.fromstring(site_str)
                self.model.worldbody.append(site_tree)

                """
                show inner reset region
                """
                pos_to_vis = deepcopy(ref_pos)
                pos_to_vis[:2] += T.rotate_2d_point([np.mean(x_range), np.mean(y_range)], rot=ref_rot)
                size_to_vis = np.concatenate([
                    np.abs(T.rotate_2d_point([(x_range[1] - x_range[0]) / 2, (y_range[1] - y_range[0]) / 2], rot=ref_rot)),
                    [0.002]
                ])
                site_str = """<site type="box" rgba="1 0 0 0.4" size="{size}" pos="{pos}" name="reset_region_inner_{postfix}"/>""".format(
                    pos=array_to_string(pos_to_vis),
                    size=array_to_string(size_to_vis),
                    postfix=str(obj_i),
                )
                site_tree = ET.fromstring(site_str)
                self.model.worldbody.append(site_tree)
            
            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="{}_Sampler".format(cfg["name"]),
                    mujoco_objects=mj_obj,
                    x_range=x_range,
                    y_range=y_range,
                    rotation=rotation,
                    ensure_object_boundary_in_range=placement.get("ensure_object_boundary_in_range", True),
                    ensure_valid_placement=placement.get("ensure_valid_placement", True),
                    reference_pos=ref_pos,
                    reference_rot=ref_rot,
                    z_offset=z_offset,
                    rng=self.rng,
                    rotation_axis=placement.get("rotation_axis", "z")
                ),
                sample_args=placement.get("sample_args", None),
            )
        
        return placement_initializer

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset and self.placement_initializer is not None:
            # use pre-computed object placements
            object_placements = self.object_placements

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                    
        # step through a few timesteps to settle objects
        action = np.zeros(12) # apply empty action

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(10 * int(self.control_timestep / self.model_timestep)):
            self.sim.step1()
            self._pre_action(action, policy_step)
            self.sim.step2()
            policy_step = False
    
    def _get_obj_cfgs(self):
        return []
    
    def get_ep_meta(self):
        """
        Returns a dictionary containing episode meta data
        """
        def copy_dict_for_json(orig_dict):
            new_dict = {}
            for (k, v) in orig_dict.items():
                if isinstance(v, dict):
                    new_dict[k] = copy_dict_for_json(v)
                elif isinstance(v, Fixture):
                    new_dict[k] = v.name
                else:
                    new_dict[k] = v
            return new_dict

        ep_meta = super().get_ep_meta()
        ep_meta["layout_id"] = self.layout_id
        ep_meta["style_id"] = self.style_id
        ep_meta["object_cfgs"] = [copy_dict_for_json(cfg) for cfg in self.object_cfgs]
        ep_meta["fixtures"] = {k: {"cls": v.__class__.__name__} for (k, v) in self.fixtures.items()}
        ep_meta["gen_textures"] = self._curr_gen_fixtures or {}
        ep_meta["lang"] = ""
        ep_meta["fixture_refs"] = dict({k: v.name for (k, v) in self.fixture_refs.items()})
        ep_meta["cam_configs"] = deepcopy(self._cam_configs)

        return ep_meta

    def find_object_cfg_by_name(self, name):
        for cfg in self.object_cfgs:
            if cfg["name"] == name:
                return cfg
        raise ValueError
    
    def set_cameras(self):
        self._cam_configs = deepcopy(KU.CAM_CONFIGS)
        if self.randomize_cameras:
            self._randomize_cameras()

        for (cam_name, cam_cfg) in self._cam_configs.items():
            if cam_cfg.get("parent_body", None) is not None:
                continue

            self.mujoco_arena.set_camera(
                camera_name=cam_name,
                pos=cam_cfg["pos"],
                quat=cam_cfg["quat"],
                camera_attribs=cam_cfg.get("camera_attribs", None),
            )

    def _randomize_cameras(self):
        for camera in self._cam_configs:
            if "agentview" in camera:
                pos_noise = np.random.normal(loc=0, scale=0.05, size=(1, 3))[0]
                euler_noise = np.random.normal(loc=0, scale=3, size=(1, 3))[0]
            elif "eye_in_hand" in camera:
                pos_noise = np.zeros_like(pos_noise)
                euler_noise = np.zeros_like(euler_noise)
            else:
                # skip randomization for cameras not implemented
                continue
                
            old_pos = self._cam_configs[camera]["pos"]
            new_pos = [pos + n for pos,n in zip(old_pos, pos_noise)]
            self._cam_configs[camera]["pos"] = list(new_pos)

            old_euler = Rotation.from_quat(self._cam_configs[camera]["quat"]).as_euler("xyz", degrees=True)
            new_euler = [eul + n for eul,n in zip(old_euler, euler_noise)]
            new_quat = Rotation.from_euler("xyz", new_euler, degrees=True).as_quat()
            self._cam_configs[camera]["quat"] = list(new_quat)

    def edit_model_xml(self, xml_str):
        """
        This function postprocesses the model.xml collected from a MuJoCo demonstration
        for retrospective model changes.
        Args:
            xml_str (str): Mujoco sim demonstration XML file as string
        Returns:
            str: Post-processed xml file as string
        """
        xml_str = super().edit_model_xml(xml_str)

        tree = ET.fromstring(xml_str)
        root = tree
        worldbody = root.find("worldbody")
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        robocasa_path_split = os.path.split(robocasa.__file__)[0].split("/")

        # replace robocasa-specific asset paths
        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue

            old_path_split = old_path.split("/")
            # maybe replace all paths to robosuite assets
            if ("models/assets/fixtures" in old_path) \
                or ("models/assets/textures" in old_path) \
                or ("models/assets/objects/objaverse" in old_path):
                if "/robosuite/" in old_path:
                    check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite"]
                elif "/robocasa/" in old_path:
                    check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robocasa"]
                else:
                    raise ValueError
                
                ind = max(check_lst)  # last occurrence index
                new_path_split = robocasa_path_split + old_path_split[ind + 1 :]

                new_path = "/".join(new_path_split)
                elem.set("file", new_path)
        
        # set cameras
        for cam_name, cam_config in self._cam_configs.items():
            parent_body = cam_config.get("parent_body", None)

            cam_root = worldbody
            if parent_body is not None:
                cam_root = find_elements(root=worldbody, tags="body", attribs={"name": parent_body})
            
            cam = find_elements(root=cam_root, tags="camera", attribs={"name": cam_name})

            if cam is None:
                cam = ET.Element("camera")
                cam.set("mode", "fixed")
                cam.set("name", cam_name)
                cam_root.append(cam)

            cam.set("pos", array_to_string(cam_config["pos"]))
            cam.set("quat", array_to_string(cam_config["quat"]))
            for (k, v) in cam_config.get("camera_attribs", {}).items():
                 cam.set(k, v)

        # result = ET.tostring(root, encoding="utf8").decode("utf8")
        result = ET.tostring(root).decode("utf8")

        # replace with generative textures
        if (self.generative_textures is not None) and (self.generative_textures is not False):
            # sample textures
            assert self.generative_textures == "100p"
            self._curr_gen_fixtures = get_random_textures()
            
            cab_tex = self._curr_gen_fixtures["cab_tex"]
            counter_tex = self._curr_gen_fixtures["counter_tex"]
            wall_tex = self._curr_gen_fixtures["wall_tex"]
            floor_tex = self._curr_gen_fixtures["floor_tex"]

            result = replace_cab_textures(result, new_cab_texture_file=cab_tex)
            result = replace_counter_top_texture(result, new_counter_top_texture_file=counter_tex)
            result = replace_wall_texture(result, new_wall_texture_file=wall_tex)
            result = replace_floor_texture(result, new_floor_texture_file=floor_tex)

        return result

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        self.obj_body_id = {}
        for (name, model) in self.objects.items():
            self.obj_body_id[name] = self.sim.model.body_name2id(model.root_body)
        
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled
        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information

        # Get robot prefix and define observables modality
        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        # for conversion to relative gripper frame
        @sensor(modality=modality)
        def world_pose_in_gripper(obs_cache):
            return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
        sensors = [world_pose_in_gripper]
        names = ["world_pose_in_gripper"]
        actives = [False]

        # add ground-truth poses (absolute and relative to eef) for all objects
        for obj_name in self.obj_body_id:
            obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj_name, modality=modality)
            sensors += obj_sensors
            names += obj_sensor_names
            actives += [True] * len(obj_sensors)

        # Create observables
        for name, s, active in zip(names, sensors, actives):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                active=active,
            )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.
        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors
        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        ### TODO: this was stolen from pick-place - do we want to move this into utils to share it? ###
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):            
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        # Check if stove is turned on or not
        self.update_state()

        # base_mode = (action[-1] > 0.0)
        # if base_mode:
        #     info["actions_abs"] = action
        # else:
        #     controller = self.robots[0].controller["right"]
        #     eef_goal_pos = np.array(controller.goal_pos)
        #     eef_goal_ori = np.array(controller.goal_ori)
        #     # convert to actions relative to robot base
        #     base_pos, base_ori = self.robots[0].get_base_pose()
        #     ac_pos, ac_ori = OU.compute_rel_transform(base_pos, base_ori, eef_goal_pos, eef_goal_ori)
        #     ac_ori = Rotation.from_matrix(ac_ori).as_rotvec()
        #     info["actions_abs"] = np.hstack([
        #         ac_pos,
        #         ac_ori,
        #         action[6:],
        #     ])

        return reward, done, info
    
    def convert_rel_to_abs_action(self, rel_action):
        # if moving mobile base, there is no notion of absolute actions.
        # use relative actions instead.
        
        # if moving arm, get the absolute action
        robot = self.robots[0]
        robot.control(rel_action, policy_step=True)
        ac_pos = robot.composite_controller.controllers["right"].goal_origin_to_eef_pos
        ac_ori = robot.composite_controller.controllers["right"].goal_origin_to_eef_ori
        ac_ori = Rotation.from_matrix(ac_ori).as_rotvec()
        action_abs = np.hstack([
            ac_pos,
            ac_ori,
            rel_action[6:],
        ])
        return action_abs
        
    def update_state(self):
        super().update_state()

        for fixtr in self.fixtures.values():
            fixtr.update_state(self)

    def visualize(self, vis_settings):
        """
        In addition to super call, make the robot semi-transparent

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        robot_model = self.robots[0].robot_model
        visual_geom_names = robot_model.visual_geoms

        for name in visual_geom_names:
            rgba = self.sim.model.geom_rgba[self.sim.model.geom_name2id(name)]
            if self.translucent_robot:
                rgba[-1] = 0.20
            else:
                rgba[-1] = 1.0

    def reward(self, action=None):
        reward = 0
        return reward
    
    def _check_success(self):    
        return False

    def sample_object(
        self, groups, exclude_groups=None,
        graspable=None, microwavable=None, washable=None, cookable=None, freezable=None,
        split=None, obj_registries=None,
        max_size=(None, None, None), object_scale=None,
    ):        
        return sample_kitchen_object(
            groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            freezable=freezable,
            rng=self.rng,
            obj_registries=(obj_registries or self.obj_registries),
            split=(split or self.obj_instance_split),
            max_size=max_size,
            object_scale=object_scale,
        )
    
    def _is_fxtr_valid(self, fxtr, size):
        for region in fxtr.get_reset_regions(self).values():
            if region["size"][0] >= size[0] and region["size"][1] >= size[1]:
                return True
        return False

    def get_fixture(self, id, ref=None, size=(0.2, 0.2)):
        """search fixture by id (name, object, or type)"""
        # case 1: id refers to fixture object directly
        if isinstance(id, Fixture):
            return id
        # case 2: id refers to exact name of fixture
        elif id in self.fixtures.keys():
            return self.fixtures[id]
        
        if ref is None:
            # find all fixtures with names containing given name
            if isinstance(id, FixtureType) or isinstance(id, int):
                matches = [name for (name, fxtr) in self.fixtures.items() if fixture_is_type(fxtr, id)]
            else:
                matches = [name for name in self.fixtures.keys() if id in name]
            if id == FixtureType.COUNTER or id == FixtureType.COUNTER_NON_CORNER:
                matches = [name for name in matches if self._is_fxtr_valid(self.fixtures[name], size) ]
            assert len(matches) > 0
            # sample random key
            key = random.sample(matches, 1)[0]
            return self.fixtures[key]
        else:
            ref_fixture = self.get_fixture(ref)

            assert isinstance(id, FixtureType)             
            cand_fixtures = []
            for fxtr in self.fixtures.values():
                if not fixture_is_type(fxtr, id):
                    continue
                if fxtr is ref_fixture:
                    continue               
                if id == FixtureType.COUNTER:
                    fxtr_is_valid = self._is_fxtr_valid(fxtr, size)
                    if not fxtr_is_valid:
                        continue
                cand_fixtures.append(fxtr)

            # first, try to find fixture "containing" the reference fixture
            for fxtr in cand_fixtures:
                if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                    return fxtr
            # if no fixture contains reference fixture, sample all close fixtures
            dists = [OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures]
            min_dist = np.min(dists)
            close_fixtures = [fxtr for (fxtr, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10]
            return random.sample(close_fixtures, 1)[0]
        
    def register_fixture_ref(self, ref_name, fn_kwargs):
        if ref_name not in self.fixture_refs:
            self.fixture_refs[ref_name] = self.get_fixture(**fn_kwargs)
        return self.fixture_refs[ref_name]
    
    def get_obj_lang(self, obj_name="obj", get_preposition=False):
        obj_cfg = None
        for cfg in self.object_cfgs:
            if cfg["name"] == obj_name:
                obj_cfg = cfg
                break
        lang = obj_cfg["info"]["cat"].replace("_", " ")
        
        if not get_preposition:
            return lang

        if lang in ["bowl", "pot", "pan"]:
            preposition = "in"
        elif lang in ["plate"]:
            preposition = "on"
        else:
            raise ValueError
        
        return lang, preposition


class KitchenDemo(Kitchen):
    def __init__(
        self,
        init_robot_base_pos="cab_main_main_group",
        obj_groups="all",
        num_objs=1,
        *args,
        **kwargs
    ):
        self.obj_groups = obj_groups
        self.num_objs = num_objs

        super().__init__(init_robot_base_pos=init_robot_base_pos, *args, **kwargs)

    def _get_obj_cfgs(self):
        cfgs = []

        for i in range(self.num_objs):
            cfgs.append(dict(
                name="obj_{}".format(i),
                obj_groups=self.obj_groups,
                placement=dict(
                    fixture="counter_main_main_group",
                    sample_region_kwargs=dict(
                        ref="cab_main_main_group",
                    ),
                    size=(1.0, 1.0),
                    pos=(0.0, -1.0),
                ),
            ))

        return cfgs
