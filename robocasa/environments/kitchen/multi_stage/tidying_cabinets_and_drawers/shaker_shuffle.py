from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.single_stage.kitchen_drawer import *

# Inherit from ManipulateDrawer class since the class starts off with the drawer open
class ShakerShuffle(ManipulateDrawer):
    """
    Shaker Shuffle: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of reorganizing the pantry by only placing the shakers in
    the drawer.

    Steps:
        Open the cabinet. Place the shakers in the drawer. Close the cabinet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=self.drawer)
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick and place the shaker into the drawer. Then close the cabinet."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="shaker1",
                obj_groups="shaker",
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="shaker2",
                obj_groups="shaker",
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment",
                obj_groups="condiment_bottle",
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        # make sure only the shakers were placed in the drawer!
        shaker_in_drawer = (
            OU.obj_inside_of(self, "shaker1", self.drawer)
            and OU.obj_inside_of(self, "shaker2", self.drawer)
            and not OU.obj_inside_of(self, "condiment", self.drawer)
        )

        door_state = self.cab.get_door_state(env=self)

        door_closed = True
        for joint_p in door_state.values():

            if joint_p > 0.05:
                door_closed = False
                break

        return shaker_in_drawer and door_closed
