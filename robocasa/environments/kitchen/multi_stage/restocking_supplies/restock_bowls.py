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

    def __init__(self, cab_id=FixtureType.CABINET_DOUBLE_DOOR, *args, **kwargs):
        # use double door cabinet as default to have space for two bowls
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

        obj_name_1 = self.get_obj_lang("obj1")
        obj_name_2 = self.get_obj_lang("obj2")

        ep_meta["lang"] = (
            "Open the cabinet. "
            f"Pick the {obj_name_1} and the {obj_name_2} from the counter and place it in the cabinet directly in front. "
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
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                    size=(0.50, 0.50),
                    pos=(-0.5, -1),
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
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                    size=(0.50, 0.50),
                    pos=(0.5, -1),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj1_inside_cab = OU.obj_inside_of(self, "obj1", self.cab)
        obj2_inside_cab = OU.obj_inside_of(self, "obj2", self.cab)

        door_state = self.cab.get_door_state(env=self)
        door_closed = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                door_closed = False
                break

        return obj1_inside_cab and obj2_inside_cab and door_closed
