from robocasa.environments.kitchen.kitchen import *


class ScrubCuttingBoard(Kitchen):
    """
    Scrub Cutting Board: composite task for Sanitizing Cutting Board activity.

    Simulates scrubbing a cutting board using a sponge in a circular motion to ensure
    thorough cleaning and removal of food particles.

    Steps:
        1. Pick up the sponge from the counter.
        2. Scrub the cutting board.
        3. Once the scrubbing is complete, place the sponge down.
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
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Pick up the sponge from the counter and clean the cutting board by briefly "
                "scrubbing or pressing down on the cutting board. Once finished, release the sponge."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.board_contact_positions = []
        self.board_contact_timer = 0
        self.sponge_contact_height = None

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="sponge",
                obj_groups="sponge",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(0.3, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def update_state(self):
        sponge_grasped = OU.check_obj_grasped(self, "sponge")
        super().update_state()
        sponge_body = self.obj_body_id["sponge"]
        board_body = self.obj_body_id["cutting_board"]

        sponge_geoms = [
            self.sim.model.geom_id2name(gid)
            for gid in range(self.sim.model.ngeom)
            if self.sim.model.geom_bodyid[gid] == sponge_body
        ]
        board_geoms = [
            self.sim.model.geom_id2name(gid)
            for gid in range(self.sim.model.ngeom)
            if self.sim.model.geom_bodyid[gid] == board_body
        ]

        contact = self.check_contact(sponge_geoms, board_geoms)
        sponge_pos = np.array(self.sim.data.body_xpos[sponge_body])
        sponge_xy = sponge_pos[:2]

        if contact and sponge_grasped:
            new_contact = True
            for recorded_xy in self.board_contact_positions:
                if np.linalg.norm(sponge_xy - recorded_xy) < 0.02:
                    new_contact = False
                    break
            if new_contact:
                self.board_contact_positions.append(sponge_xy)
                self.board_contact_timer += 1

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, "sponge", th=0.15)

        sweep_range = 0.0
        if self.board_contact_positions:
            positions = np.array(self.board_contact_positions)
            min_xy = positions.min(axis=0)
            max_xy = positions.max(axis=0)
            sweep_range = np.linalg.norm(max_xy - min_xy)

        return (
            (self.board_contact_timer >= 5) and (sweep_range >= 0.1) and gripper_obj_far
        )
