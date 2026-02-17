from robocasa.environments.kitchen.kitchen import *


class BoilCorn(Kitchen):
    """
    Boil Corn: composite task for Boiling activity.

    Simulates the task of boiling corn by placing two corns in a saucepan with water
    and then placing the lid to start the boiling process.

    Steps:
        1. Place two corns in the saucepan with water
        2. Place the lid on the saucepan to start boiling
    """

    def __init__(self, stove_id=FixtureType.STOVE, *args, **kwargs):
        self.stove_id = stove_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=self.stove_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            # Choose a front burner for the saucepan
            valid_front_knobs = [
                k
                for (k, v) in self.stove.knob_joints.items()
                if v is not None and k.startswith("front")
            ]
            self.knob = self.rng.choice(valid_front_knobs)

        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Place both pieces of corn from the plate into the saucepan. "
            "Then place the lid on the saucepan to start boiling the corn."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "saucepan", [0.5, 0.6, 1.0, 0.3])
        self.stove.set_knob_state(mode="high", knob=self.knob, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="saucepan",
                obj_groups="saucepan_with_lid",
                object_scale=1.15,
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
                auxiliary_obj_enable=True,
            )
        )

        cfgs.append(
            dict(
                name="corn_plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.4, 0.35),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="corn1",
                obj_groups="corn",
                object_scale=0.95,
                placement=dict(
                    object="corn_plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="corn2",
                obj_groups="corn",
                object_scale=0.95,
                placement=dict(
                    object="corn_plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        saucepan_loc = self.stove.check_obj_location_on_stove(self, "saucepan")
        saucepan_on_burner = saucepan_loc == self.knob

        corn1_in_pot = OU.check_obj_in_receptacle(self, "corn1", "saucepan")
        corn2_in_pot = OU.check_obj_in_receptacle(self, "corn2", "saucepan")

        lid_on_saucepan = OU.check_obj_in_receptacle(
            self, "saucepan_auxiliary", "saucepan", th=0.02
        )

        gripper_far = OU.gripper_obj_far(self, obj_name="saucepan_auxiliary")

        return all(
            [
                saucepan_on_burner,
                corn1_in_pot,
                corn2_in_pot,
                lid_on_saucepan,
                gripper_far,
            ]
        )
