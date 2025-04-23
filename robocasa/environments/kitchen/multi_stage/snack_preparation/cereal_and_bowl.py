from robocasa.environments.kitchen.kitchen import *


class CerealAndBowl(Kitchen):
    """
    Cereal And Bowl: composite task for Snack Preparation activity.

    Simulates the preparation of a cereal snack.

    Steps:
        Pick the cereal and bowl from the cabinet and place them on the counter.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the bowl and
            cereal are picked.
    """

    def __init__(self, cab_id=FixtureType.CABINET_DOUBLE_DOOR, *args, **kwargs):
        # use double door cabinet as default to have space for the bowl and cereal
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Open the cabinet. Pick the cereal and bowl from the cabinet and place them on the counter. "
            "Then close the cabinet."
        )
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.close_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        # make sure bowl and cereal show up on diff sides randomly
        direction = self.rng.choice([1.0, -1.0])

        cfgs.append(
            dict(
                name="cereal",
                obj_groups="boxed_food",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.30, 0.30),
                    pos=(1.0 * direction, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(-1.0 * direction, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="milk",
                obj_groups="milk",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.5, 0.30),
                    pos=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        box_on_counter = OU.check_obj_fixture_contact(self, "cereal", self.counter)
        bowl_on_counter = OU.check_obj_fixture_contact(self, "bowl", self.counter)

        door_state = self.cab.get_door_state(env=self)
        cabinet_closed = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                cabinet_closed = False
                break
        return box_on_counter and bowl_on_counter and cabinet_closed
