from robocasa.environments.kitchen.kitchen import *


class TransportCookware(Kitchen):
    """
    Transport Cookware: composite task for Washing Dishes activity.

    Stimulates transporting cookware (pots and pans) from the stove to the sink.

    Steps:
        1. Identify cookware (pots and pans) on the stove.
        2. Transport cookware to the sink.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))

        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Move the pot and pan from the stove into the sink."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pot",
                obj_groups=("pot"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.20, 0.20),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pan",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.20, 0.20),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Success criteria for pot & pan near the sink:
        1. Both pot and pan are within the sink.
        2. The robot gripper is far from both pot and pan.
        """

        pot_in_sink = OU.obj_inside_of(self, "pot", self.sink)
        pan_in_sink = OU.obj_inside_of(self, "pan", self.sink)
        within_sink = pot_in_sink and pan_in_sink

        gripper_far_pot = OU.gripper_obj_far(self, obj_name="pot")
        gripper_far_pan = OU.gripper_obj_far(self, obj_name="pan")

        return within_sink and gripper_far_pot and gripper_far_pan
