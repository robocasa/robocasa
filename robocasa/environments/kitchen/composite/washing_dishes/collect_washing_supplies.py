from robocasa.environments.kitchen.kitchen import *


class CollectWashingSupplies(Kitchen):
    """
    Collect Washing Supplies: a composite task for Washing Dishes activity.

    Locate and retrieve cleaning supplies from the cabinet and place them
    within reach of the sink for washing.

    Steps:
        1. Locate the cabinet containing the cleaning items.
        2. Retrieve the first cleaning item and place it near the sink.
        3. Retrieve the second cleaning item and place it near the sink.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET, ref=self.sink)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        supply1_lang = self.get_obj_lang("supply1")
        supply2_lang = self.get_obj_lang("supply2")
        ep_meta["lang"] = (
            f"Locate and retrieve the {supply1_lang} and {supply2_lang} from the cabinet. "
            f"Place them next to the sink."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="supply1",
                obj_groups="cleaner",
                graspable=True,
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.3, 0.3),
                    pos=(-0.5, -0.9),
                ),
            )
        )

        # Ensure unique object names and add one sponge
        cfgs.append(
            dict(
                name="supply2",
                obj_groups="cleaner",
                graspable=True,
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.3, 0.3),
                    pos=(0.5, -0.9),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        supply1_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["supply1"]])
        supply2_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["supply2"]])

        dist_threshhold = 0.20

        supply1_dist = OU.obj_fixture_bbox_min_dist(self, "supply1", self.sink)
        supply2_dist = OU.obj_fixture_bbox_min_dist(self, "supply2", self.sink)

        supplies_close = (
            supply1_dist < dist_threshhold and supply2_dist < dist_threshhold
        )
        supplies_on_counter = OU.check_obj_fixture_contact(
            self, "supply1", self.counter
        ) and OU.check_obj_fixture_contact(self, "supply2", self.counter)
        gripper_far = OU.gripper_obj_far(
            self, obj_name="supply1"
        ) and OU.gripper_obj_far(self, obj_name="supply2")

        return supplies_close and supplies_on_counter and gripper_far
