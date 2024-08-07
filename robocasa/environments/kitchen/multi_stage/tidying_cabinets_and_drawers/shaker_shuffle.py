from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.single_stage.kitchen_drawer import *


class ShakerShuffle(ManipulateDrawer):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_TOP, ref=self.drawer)
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick place the shaker into the drawer. Then, Close the cabinet."
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.set_door_state(min=0.9, max=1.0, env=self, rng=self.rng)

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
