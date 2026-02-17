from robocasa.environments.kitchen.kitchen import *


class SanitizeSink(Kitchen):
    """
    Sanitize Sink with Sponge Sweep: composite task for Sanitize Surfaces activity.

    Simulates sweeping a sponge across the sink to sanitize.

    Steps:
        1. Pick up the sponge from the counter.
        2. Sweep the sponge along the sink's interior in long strokes for at least 5 timesteps.
        3. When finished, drop the sponge.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the sponge from the counter and rub it across the sink. "
            "When finished, drop the sponge."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.sink_contact_timer = 0
        self.swept_sink = False
        self.sink_contact_positions = []

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="sponge",
                obj_groups="sponge",
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.2, 0.2),
                    pos=("ref", -0.8),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        """
        Success is achieved if:
        - At least 5 unique contacts have been recorded
        - The overall sweep range is at least 0.01
        - The gripper is not holding the sponge and it's 0.15 units far from the sponge
        """
        sponge_grasped = OU.check_obj_grasped(self, "sponge", threshold=0.6)
        sponge_contact_sink = OU.check_obj_fixture_contact(self, "sponge", self.sink)
        gripper_obj_far = OU.gripper_obj_far(self, "sponge", th=0.15)

        sponge_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["sponge"]])
        sponge_xy = sponge_pos[:2]

        if sponge_contact_sink and sponge_grasped:
            new_contact = True
            for recorded_xy in self.sink_contact_positions:
                if np.linalg.norm(sponge_xy - recorded_xy) < 0.025:
                    new_contact = False
                    break
            if new_contact:
                self.sink_contact_timer += 1
                self.sink_contact_positions.append(sponge_xy)

        sweep_range = 0.0
        if self.sink_contact_positions:
            positions = np.array(self.sink_contact_positions)
            min_xy = positions.min(axis=0)
            max_xy = positions.max(axis=0)
            sweep_range = np.linalg.norm(max_xy - min_xy)

        return (
            (self.sink_contact_timer >= 5) and (sweep_range >= 0.1) and gripper_obj_far
        )
