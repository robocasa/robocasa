from robocasa.environments.kitchen.kitchen import *


class PrepareSoupServing(Kitchen):
    def __init__(self, cab_id=FixtureType.CABINET_TOP, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.cabinet = self.register_fixture_ref(
            "cab", dict(id=self.cab_id, ref=self.stove)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_pos = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Open the cabinet and move the ladle to the pot. Then close the cabinet."
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.cabinet.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="ladle",
                obj_groups="ladle",
                graspable=True,
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                    rotation=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pot",
                obj_groups=("pot"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl1",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        ladle_in_pot = OU.check_obj_in_receptacle(self, "ladle", "pot")

        door_state = self.cabinet.get_door_state(env=self)
        door_closed = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                door_closed = False
                break

        return ladle_in_pot and door_closed
