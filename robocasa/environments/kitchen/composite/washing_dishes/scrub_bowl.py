from robocasa.environments.kitchen.kitchen import *


class ScrubBowl(Kitchen):
    """
    Scrub Bowl Task: composite task for Washing Dishes activity.

    Simulates scrubbing a bowl in the sink using a sponge.

    Steps:
        1. Pick up the sponge from the counter near the sink.
        2. Scrub the bowl inside the sink using the sponge.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Use the sponge to scrub the bowl inside the sink."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.success_contact_time = 0

    def _get_obj_cfgs(self):
        return [
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.5, 0.3),
                    pos=(0.0, 1.0),
                ),
            ),
            dict(
                name="sponge",
                obj_groups="sponge",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.25, 0.3),
                    pos=("ref", -1.0),
                ),
            ),
        ]

    def update_state(self):
        super().update_state()
        if OU.check_obj_scrubbed(self, "sponge", "bowl"):
            self.success_contact_time += 1

    def _check_success(self):
        gripper_far_from_bowl = OU.gripper_obj_far(self, obj_name="bowl")
        return self.success_contact_time >= 15 and gripper_far_from_bowl
