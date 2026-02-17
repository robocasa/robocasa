from robocasa.environments.kitchen.kitchen import *


class TurnOffSimmeredSauceHeat(Kitchen):
    """
    Turn Off Simmered Sauce Heat: composite task for Simmering Sauces activity.
    Simulates the task of turning off the heat for a sauce once it is simmered.
    Steps:
        Turn off the stove knob once the sauce has simmered.
    """

    # randomize amount of time to simmer so policy doesnt wait a fixed length. Min must be longer
    # than the average time it takesto simply turn on the knob because we want the policy to wait
    MIN_SIMMER_SAUCE_TIMESTEPS = 450
    MAX_SIMMER_SAUCE_TIMESTEPS = 500
    START_SAUCE_COLOR = np.array([255.0, 120.0, 60.0, 100.0]) / 255.0
    FINISHED_SAUCE_COLOR = np.array([178.0, 25.0, 7.0, 100.0]) / 255.0

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.stove

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
            self.simmer_timesteps = self._ep_meta["refs"]["simmer_timesteps"]
        else:
            self.simmer_timesteps = self.rng.uniform(
                self.MIN_SIMMER_SAUCE_TIMESTEPS, self.MAX_SIMMER_SAUCE_TIMESTEPS
            )
            valid_knobs = [
                k for (k, v) in self.stove.knob_joints.items() if v is not None
            ]
            if self.knob_id == "random":
                self.knob = self.rng.choice(list(valid_knobs))
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob_id

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Wait until the marinara sauce is simmered to a deep red color and then turn off the stove."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"].update(knob=self.knob, simmer_timesteps=self.simmer_timesteps)
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.stove.set_knob_state(mode="on", knob=self.knob, env=self, rng=self.rng)
        self.task_timestep = 0
        liquid_geom_id = self.sim.model.geom_name2id("saucepan_liquid")
        self.sim.model.geom_rgba[liquid_geom_id] = self.START_SAUCE_COLOR
        self.off_too_early = False

    def update_state(self):
        super().update_state()
        color_diff = self.FINISHED_SAUCE_COLOR - self.START_SAUCE_COLOR
        finished_percent = min(1, self.task_timestep / self.simmer_timesteps)

        # linearly interpolate
        new_sauce_color = self.START_SAUCE_COLOR + finished_percent * color_diff
        liquid_geom_id = self.sim.model.geom_name2id("saucepan_liquid")
        self.sim.model.geom_rgba[liquid_geom_id] = new_sauce_color
        self.task_timestep += 1

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="saucepan",
                obj_groups=("saucepan"),
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.05),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        knob_off = not self.stove.is_burner_on(env=self, burner_loc=self.knob)

        if self.off_too_early:
            return False

        if knob_off and self.task_timestep < self.simmer_timesteps:
            self.off_too_early = True

        return knob_off and self.simmer_timesteps < self.task_timestep
