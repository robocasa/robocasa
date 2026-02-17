from robocasa.environments.kitchen.kitchen import *


class LineUpCondiments(Kitchen):
    """
    Line Up Condiments: composite task for Arranging Condiments activity.

    Task: Take 2 condiments from the cabinet and place them next to each other on the counter near the stove.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Take two condiments from the cabinet and move them on the counter near the stove."
            "Make sure the condiments are lined up with each other, next to each other."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="condiment1",
                obj_groups=("condiment", "shaker", "salt_and_pepper_shaker"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment2",
                obj_groups=("condiment", "shaker", "salt_and_pepper_shaker"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        condiment1_on_counter = OU.check_obj_any_counter_contact(self, "condiment1")
        condiment2_on_counter = OU.check_obj_any_counter_contact(self, "condiment2")

        condiment1_stove_dist = OU.obj_fixture_bbox_min_dist(
            self, "condiment1", self.stove
        )
        condiment2_stove_dist = OU.obj_fixture_bbox_min_dist(
            self, "condiment2", self.stove
        )
        condiments_near_stove = (condiment1_stove_dist < 0.35) and (
            condiment2_stove_dist < 0.35
        )

        condiment1_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["condiment1"]]
        )[:2]
        condiment2_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["condiment2"]]
        )[:2]

        x_distance = abs(condiment1_pos[0] - condiment2_pos[0])
        y_distance = abs(condiment1_pos[1] - condiment2_pos[1])

        x_spacing_correct = x_distance <= 0.2
        y_spacing_correct = y_distance <= 0.05

        gripper_far = OU.gripper_obj_far(self, "condiment1") and OU.gripper_obj_far(
            self, "condiment2"
        )

        return (
            condiment1_on_counter
            and condiment2_on_counter
            and condiments_near_stove
            and x_spacing_correct
            and y_spacing_correct
            and gripper_far
        )
