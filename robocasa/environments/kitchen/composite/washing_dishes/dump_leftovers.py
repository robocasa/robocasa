from robocasa.environments.kitchen.kitchen import *


class DumpLeftovers(Kitchen):
    """
    Dump Leftovers: composite task for Washing Dishes activity.

    Simulates the process of clearing leftover food items from a bowl before washing it.

    Steps:
        1. Dump the leftover food items from the bowl onto the counter.
        2. Place the empty bowl in the sink for washing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        leftover1_lang = self.get_obj_lang("leftover1")
        leftover2_lang = self.get_obj_lang("leftover2")

        if leftover1_lang == leftover2_lang:
            leftovers_text = f"{leftover1_lang}s"
        else:
            leftovers_text = f"{leftover1_lang} and {leftover2_lang}"

        ep_meta["lang"] = (
            f"Dump the {leftovers_text} from the bowl so the bowl can be washed. "
            f"Place the bowl in the sink after clearing the leftovers."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                init_robot_here=True,
                graspable=True,
                object_scale=[1, 1, 0.75],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="leftover1",
                obj_groups=("fruit", "vegetable", "cooked_food"),
                graspable=True,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="leftover2",
                obj_groups=("fruit", "vegetable", "cooked_food"),
                graspable=True,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        leftover1_off_bowl = not OU.check_obj_in_receptacle(self, "leftover1", "bowl")
        leftover2_off_bowl = not OU.check_obj_in_receptacle(self, "leftover1", "bowl")
        leftovers_dumped = leftover1_off_bowl and leftover2_off_bowl

        bowl_in_sink = OU.obj_inside_of(self, "bowl", self.sink)
        gripper_far = OU.gripper_obj_far(self, obj_name="bowl")

        return leftovers_dumped and bowl_in_sink and gripper_far
