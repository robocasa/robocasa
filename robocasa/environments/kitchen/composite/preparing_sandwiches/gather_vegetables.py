from robocasa.environments.kitchen.kitchen import *


class GatherVegetables(Kitchen):
    """
    Gather Vegetables to Cutting Board: composite task for Preparing Sandwiches.

    Simulates the task of gathering 2 vegetables from the sink
    and placing them on the cutting board on the counter.

    Steps:
        1) Pick up the 2 vegetables from the sink.
        2) Place them on the cutting board.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg1_lang = self.get_obj_lang("veg1")
        veg2_lang = self.get_obj_lang("veg2")
        ep_meta["lang"] = (
            f"Pick up the {veg1_lang} and {veg2_lang} from the sink "
            f"and place them on the cutting board on the counter."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="veg1",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.15),
                    pos=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="veg2",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.15),
                    pos=(-1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.40, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        vegetables = ["veg1", "veg2"]
        all_veggies_on_board = all(
            OU.check_obj_in_receptacle(self, veg, "cutting_board") for veg in vegetables
        )

        all_veggies_far = all(
            OU.gripper_obj_far(self, obj_name=veg) for veg in vegetables
        )

        return all_veggies_on_board and all_veggies_far
