from robocasa.environments.kitchen.kitchen import *


class AddSweetener(Kitchen):
    """
    AddSweetener: composite task for the making smoothies activity.
    Simulates the task of adding sweetener to the blender.
    Steps:
        1. Pick up the sugar cubes from the plate on the counter.
        2. Put the sugar cubes in the blender.
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
            self.num_sugar_cubes = self._ep_meta["refs"]["num_sugar_cubes"]
        else:
            self.num_sugar_cubes = int(self.rng.choice([2, 3]))

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

        number = self.num_sugar_cubes
        word_representation = number_to_word.get(number)
        ep_meta["lang"] = f"Put the {word_representation} sugar cubes in the blender."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["num_sugar_cubes"] = self.num_sugar_cubes
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("fruit"),
                object_scale=0.80,
                placement=dict(
                    fixture=self.blender,
                    size=(0.40, 0.40),
                    pos=(0, 0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.35),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(ref=self.blender, loc="left_right"),
                ),
                init_robot_here=True,
            )
        )
        for i in range(self.num_sugar_cubes):
            cfgs.append(
                dict(
                    name="sugar_cube" + str(i),
                    obj_groups="sugar_cube",
                    graspable=True,
                    placement=dict(
                        object="plate",
                        size=(0.5, 0.5),
                    ),
                    object_scale=1.25,
                )
            )
        return cfgs

    def _check_success(self):
        sugar_in_blender = all(
            [
                OU.obj_inside_of(self, f"sugar_cube{i}", self.blender, th=0.01)
                for i in range(self.num_sugar_cubes)
            ]
        )
        gripper_objs_far = all(
            [
                OU.gripper_obj_far(self, f"sugar_cube{i}")
                for i in range(self.num_sugar_cubes)
            ]
        )
        return sugar_in_blender and gripper_objs_far
