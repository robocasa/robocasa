from robocasa.environments.kitchen.kitchen import *


class ArrangeBreadBasket(Kitchen):
    """
    Arrange Bread Basket: composite task for Setting The Table activity.

    Simulates the task of arranging the bread basket.

    Steps:
        Pick the bread from the cabinet and place it in the bowl. Then move the bowl
        to the dining counter.

    Restricted to layouts which have a dining table (long counter area with
    stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the bread is
            picked.
    """

    EXCLUDE_LAYOUTS = [0, 2, 4, 5]

    def __init__(self, cab_id=FixtureType.CABINET, *args, **kwargs):

        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter_small = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        bread_name = self.get_obj_lang("bread")
        ep_meta["lang"] = (
            f"Open the cabinet, pick up the {bread_name} from the cabinet and place it in the bowl. "
            "Then move the bowl to the dining counter."
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

        cfgs.append(
            dict(
                name="bread",
                obj_groups="bread",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter_small,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining2",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                    offset=(0.05, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="bowl")
        bread_in_bowl = OU.check_obj_in_receptacle(self, "bread", "bowl")
        bowl_on_counter = OU.check_obj_fixture_contact(self, "bowl", self.dining_table)

        return gripper_obj_far and bread_in_bowl and bowl_on_counter
