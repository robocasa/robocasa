from robocasa.environments.kitchen.kitchen import *


class WipeTable(Kitchen):
    """
    Wipe Table: composite task for Sanitize Surfaces activity.

    Simulates wiping the counter using a sponge by pressing it against the surface.

    Steps:
        1. Pick up the sponge from the counter.
        2. Wipe the counter by pressing it against the surface.
        3. Once the wiping is complete, place the sponge down.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, full_depth_region=True)
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the sponge from the counter and wipe the counter by gripping it and pressing it against the surface. "
            "Once finished, release the sponge."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.table_contact_positions = []
        self.table_contact_timer = 0

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
                        full_depth_region=True,
                    ),
                    size=(0.5, 0.2),
                    pos=(1.0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        sponge_grasped = OU.check_obj_grasped(self, "sponge", threshold=0.6)
        sponge_contact_table = OU.check_obj_any_counter_contact(self, "sponge")
        gripper_obj_far = OU.gripper_obj_far(self, "sponge", th=0.15)

        sponge_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["sponge"]])
        sponge_xy = sponge_pos[:2]

        # Only record contact positions when sponge is actually in contact with table and grasped
        if sponge_contact_table and sponge_grasped:
            new_contact = True
            for recorded_xy in self.table_contact_positions:
                if np.linalg.norm(sponge_xy - recorded_xy) < 0.02:
                    new_contact = False
                    break
            if new_contact:
                self.table_contact_positions.append(sponge_xy)
                self.table_contact_timer += 1

        sweep_range = 0.0
        if self.table_contact_positions:
            positions = np.array(self.table_contact_positions)
            min_xy = positions.min(axis=0)
            max_xy = positions.max(axis=0)
            sweep_range = np.linalg.norm(max_xy - min_xy)

        return (
            (self.table_contact_timer >= 5) and (sweep_range >= 0.1) and gripper_obj_far
        )
