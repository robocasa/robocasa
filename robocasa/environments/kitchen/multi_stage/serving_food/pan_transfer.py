from robocasa.environments.kitchen.kitchen import *


class PanTransfer(Kitchen):
    """
    Pan Transfer: composite task for Serving Food activity.

    Simulates the task of transferring vegetables from a pan to a plate.

    Steps:
        Pick up the pan and dump the vegetables in it onto the plate.
        Then, return the pan to the stove.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_pos = self.stove
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the pan and dump the vegetables in it onto the plate. "
            "Then return the pan to the stove."
        )
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.stove,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="pan",
                    container_kwargs=dict(
                        rotation=[
                            (-3 * np.pi / 8, -np.pi / 4),
                            (np.pi / 4, 3 * np.pi / 8),
                        ],
                    ),
                ),
            )
        )
        # cfgs.append(dict(
        #     name="vegetable2",
        #     obj_groups="vegetable",
        #     placement=dict(
        #         size=(0.01, 0.01),
        #         ensure_object_boundary_in_range=False,
        #         sample_args=dict(
        #             reference="vegetable_container"
        #         )
        #     ),
        # ))

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                exclude_obj_groups=["plate", "pan", "vegetable"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.20),
                    pos=(0.5, 0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        vegetable_on_plate = OU.check_obj_in_receptacle(self, "vegetable", "plate")
        pan_on_stove = OU.check_obj_fixture_contact(
            self, "vegetable_container", self.stove
        )
        gripper_obj_far = OU.gripper_obj_far(
            self, "vegetable_container"
        ) and OU.gripper_obj_far(self, "vegetable")

        return vegetable_on_plate and pan_on_stove and gripper_obj_far
