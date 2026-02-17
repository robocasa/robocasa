from robocasa.environments.kitchen.kitchen import *


class PlaceLidToBoil(Kitchen):
    """
    Place Lid to Boil: composite task for Boiling activity.

    Simulates the task of placing a lid on a saucepan on the stove to boil the water faster.

    Steps:
        1. Place the lid on top of the saucepan on the stove to boil the water faster.
    """

    def __init__(self, stove_id=FixtureType.STOVE, *args, **kwargs):
        self.stove_id = stove_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=self.stove_id))
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )

        valid_front_knobs = [
            k
            for (k, v) in self.stove.knob_joints.items()
            if v is not None and k.startswith("front")
        ]
        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            self.knob = self.rng.choice(valid_front_knobs)

        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Place the lid on the saucepan on the stove to boil the water faster."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(mode="high", knob=self.knob, env=self, rng=self.rng)
        self.cabinet.open_door(self)
        OU.add_obj_liquid_site(self, "saucepan", [0.5, 0.6, 1.0, 0.3])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="saucepan",
                obj_groups="saucepan_with_lid",
                graspable=True,
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
                auxiliary_obj_enable=True,
                auxiliary_obj_placement=dict(
                    fixture=self.cabinet,
                    size=(0.3, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        saucepan_loc = self.stove.check_obj_location_on_stove(self, "saucepan")
        saucepan_on_burner = saucepan_loc == self.knob

        lid_on_saucepan = OU.check_obj_in_receptacle(
            self, "saucepan_auxiliary", "saucepan", th=0.02
        )
        gripper_far = OU.gripper_obj_far(self, obj_name="saucepan_auxiliary")

        return all([saucepan_on_burner, lid_on_saucepan, gripper_far])
