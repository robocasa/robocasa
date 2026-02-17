from robocasa.environments.kitchen.kitchen import *


class SweetenHotChocolate(Kitchen):
    """
    Sweeten Hot Chocolate: composite task for Preparing Hot Chocolate activity.
    Simulates the task of adding sugar to a saucepan with hot chocolate.

    Steps:
        Pick the sugar, and place it in the saucepan.
        Repeat for each sugar cube on the counter.
    """

    def __init__(self, *args, **kwargs):
        self.knob_id = "random"
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )

        self.init_robot_base_ref = self.stove

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob_id"]
            self.num_sugar = self._ep_meta["refs"]["num_sugar"]
        else:
            self.num_sugar = int(self.rng.choice([1, 2, 3]))
            valid_knobs = [
                k
                for (k, v) in self.stove.knob_joints.items()
                if v is not None and not k.startswith("rear")
            ]
            self.knob = (
                self.rng.choice(valid_knobs)
                if self.knob_id == "random"
                else self.knob_id
            )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the sugar cube(s) and place them in the saucepan with hot chocolate."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["num_sugar"] = self.num_sugar
        ep_meta["refs"]["knob_id"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")
        liquid_geom_id = self.sim.model.geom_name2id("saucepan_liquid")
        self.sim.model.geom_rgba[liquid_geom_id] = [0.58, 0.37, 0.24, 0.92]

    def _get_obj_cfgs(self):
        cfgs = []
        for i in range(self.num_sugar):
            cfgs.append(
                dict(
                    name=f"sugar{i}",
                    obj_groups="sugar_cube",
                    placement=dict(
                        fixture=self.counter,
                        size=(0.25, 0.25),
                        sample_region_kwargs=dict(ref=self.stove, loc="left_right"),
                        pos=("ref", -1.0),
                    ),
                )
            )

        cfgs.append(
            dict(
                name="saucepan",
                obj_groups="saucepan",
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=[self.knob]),
                    size=(0.05, 0.05),
                ),
            )
        )

        return cfgs

    def _check_success(self):

        sugar_in_saucepan = all(
            [
                OU.check_obj_in_receptacle(self, f"sugar{i}", "saucepan")
                for i in range(self.num_sugar)
            ]
        )
        gripper_far = OU.gripper_obj_far(self, "saucepan")

        return sugar_in_saucepan and gripper_far
