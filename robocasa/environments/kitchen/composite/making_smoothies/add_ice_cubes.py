from robocasa.environments.kitchen.kitchen import *


class AddIceCubes(Kitchen):
    """
    AddIceCubes: composite task for the making smoothies activity.
    Simulates the task of adding ice cubes to the blender.
    Steps:
        1. Pick up the ice cubes from the bowl on the counter.
        2. Put the ice cubes in the blender.
    """

    EXCLUDE_LAYOUTS = [21, 27]
    # update yaml to disable the lid since we are placing object in blender
    _BLENDER_PLACEMENT_UPDATE_DICT = {"aux_fixture_config": {"enable": False}}

    def __init__(
        self, enable_fixtures=None, update_fxtr_cfg_dict=None, *args, **kwargs
    ):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["blender"]

        update_fxtr_cfg_dict = update_fxtr_cfg_dict or {}
        update_fxtr_cfg_dict["blender"] = self._BLENDER_PLACEMENT_UPDATE_DICT
        super().__init__(
            enable_fixtures=enable_fixtures,
            update_fxtr_cfg_dict=update_fxtr_cfg_dict,
            *args,
            **kwargs,
        )

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.blender = self.register_fixture_ref(
            "blender", dict(id=FixtureType.BLENDER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.blender)
        )

        self.init_robot_base_ref = self.counter

        if "refs" in self._ep_meta:
            self.num_ice_cubes = self._ep_meta["refs"]["num_ice_cubes"]
        else:
            self.num_ice_cubes = int(self.rng.choice([2, 3]))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        number_to_word = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            0: "zero",
        }

        number = self.num_ice_cubes
        word_representation = number_to_word.get(number)
        ep_meta["lang"] = f"Put the {word_representation} ice cubes in the blender."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["num_ice_cubes"] = self.num_ice_cubes

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.35),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(ref=self.blender, loc="left_right"),
                ),
                object_scale=[1.0, 1.0, 0.75],
                init_robot_here=True,
            )
        )
        for i in range(self.num_ice_cubes):
            cfgs.append(
                dict(
                    name="ice_cube" + str(i),
                    obj_groups="ice_cube",
                    graspable=True,
                    placement=dict(
                        object="bowl",
                        size=(1.0, 1.0),
                    ),
                    object_scale=0.8,
                )
            )
        return cfgs

    def _check_success(self):
        ice_in_blender = all(
            [
                OU.obj_inside_of(self, f"ice_cube{i}", self.blender, th=0.01)
                for i in range(self.num_ice_cubes)
            ]
        )
        gripper_objs_far = all(
            [
                OU.gripper_obj_far(self, f"ice_cube{i}")
                for i in range(self.num_ice_cubes)
            ]
        )
        return ice_in_blender and gripper_objs_far
