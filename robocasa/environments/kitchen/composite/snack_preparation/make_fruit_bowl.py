from robocasa.environments.kitchen.kitchen import *


class MakeFruitBowl(Kitchen):
    """
    Make Fruit Bowl: composite task for Snack Preparation activity.

    Simulates the preparation of a fruit bowl snack.

    Steps:
        Pick the fruit from the cabinet and place them in the bowl.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the fruit are
            picked.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, cab_id=FixtureType.CABINET_DOUBLE_DOOR, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        fruit1_name = self.get_obj_lang("fruit1")
        fruit2_name = self.get_obj_lang("fruit2")
        ep_meta["lang"] = (
            "Open the cabinet. "
            f"Pick the {fruit1_name} and {fruit2_name} from the cabinet and place them into the bowl. "
            "Then close the cabinet."
        )

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                    size=(0.80, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit1",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(-0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit2",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fruit1_in_bowl = OU.check_obj_in_receptacle(self, "fruit1", "bowl")
        fruit2_in_bowl = OU.check_obj_in_receptacle(self, "fruit2", "bowl")
        cabinet_closed = self.cab.is_closed(env=self)
        return fruit1_in_bowl and fruit2_in_bowl and cabinet_closed
