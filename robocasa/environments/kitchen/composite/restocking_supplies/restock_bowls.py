from robocasa.environments.kitchen.kitchen import *


class RestockBowls(Kitchen):
    """
    Restock Bowls: composite task for Restocking Supplies activity.

    Simulates the task of restocking bowls.

    Steps:
        Restock two bowls from the counter to the cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet to which the bowls are
            restocked.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, cab_id=FixtureType.CABINET_DOUBLE_DOOR, *args, **kwargs):
        # use double door cabinet as default to have space for two bowls
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.5, 0.4))
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_name_1 = self.get_obj_lang("obj1")
        obj_name_2 = self.get_obj_lang("obj2")

        ep_meta["lang"] = (
            "Open the cabinet. "
            f"Pick the bowls from the counter and place them in the cabinet. "
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
        cfgs.append(
            dict(
                name="obj1",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.5, 0.4)),
                    size=(0.75, 0.50),
                    pos=(0.0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.5, 0.4)),
                    size=(0.75, 0.50),
                    pos=(0.0, -1),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj1_inside_cab = OU.obj_inside_of(self, "obj1", self.cab)
        obj2_inside_cab = OU.obj_inside_of(self, "obj2", self.cab)
        cab_closed = self.cab.is_closed(env=self)
        return obj1_inside_cab and obj2_inside_cab and cab_closed
