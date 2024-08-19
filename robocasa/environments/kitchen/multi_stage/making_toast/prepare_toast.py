from robocasa.environments.kitchen.kitchen import *


class PrepareToast(Kitchen):
    """
    Prepare Toast: composite task for Making Toast activity.

    Simulates the task of preparing toast.

    Steps:
        Open the cabinet, pick the bread, place it on the cutting board, pick the jam,
        place it on the counter, and close the cabinet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_TOP, ref=FixtureType.TOASTER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = """open the cabinet, pick the bread, place it on the cutting board, 
        pick the jam, place it on the counter, and close the cabinet."""
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
                name="obj",
                obj_groups=("bread"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.30),
                    pos=(0, -0.1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.5, 0.5),
                    pos=(0.0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="obj2",
                obj_groups="jam",
                placement=dict(
                    fixture=self.cab,
                    size=(0.3, 0.15),
                    pos=(0.0, -1.0),
                    offset=(-0.05, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj3",
                obj_groups="knife",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.3),
                    pos=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                    offset=(-0.05, 0.05),
                ),
            )
        )

        return cfgs

    def _check_door_closed(self):
        door_state = self.cab.get_door_state(env=self)

        success = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                success = False
                break

        return success

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self)
        jam_on_counter = OU.check_obj_fixture_contact(self, "obj2", self.counter)
        bread_on_cutting_board = OU.check_obj_in_receptacle(self, "obj", "container")
        cutting_board_on_counter = OU.check_obj_fixture_contact(
            self, "container", self.counter
        )
        cabinet_closed = self._check_door_closed()
        return (
            jam_on_counter
            and gripper_obj_far
            and bread_on_cutting_board
            and cutting_board_on_counter
            and cabinet_closed
        )
