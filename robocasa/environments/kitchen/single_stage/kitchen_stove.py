from robocasa.environments.kitchen.kitchen import *


class ManipulateStoveKnob(Kitchen):
    """
    Class encapsulating the atomic manipulate stove knob tasks.

    Args:
        knob_id (str): The stove knob id to manipulate. If set to "random", a random knob will be selected.

        behavior (str): "turn_on" or "turn_off". Used to define the desired
            stove knob manipulation behavior for the task.
    """

    def __init__(self, knob_id="random", behavior="turn_on", *args, **kwargs):
        assert behavior in ["turn_on", "turn_off"]
        self.behavior = behavior
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the stove knob tasks
        This includes the stove and the stove knob to manipulate, and the burner to place the cookware on.
        """
        super()._setup_kitchen_references()
        self.stove = self.get_fixture(FixtureType.STOVE)
        if "task_refs" in self._ep_meta:
            self.knob = self._ep_meta["task_refs"]["knob"]
            self.cookware_burner = self._ep_meta["task_refs"]["cookware_burner"]
        else:
            valid_knobs = [
                k for (k, v) in self.stove.knob_joints.items() if v is not None
            ]
            if self.knob_id == "random":
                self.knob = self.rng.choice(list(valid_knobs))
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob
            self.cookware_burner = (
                self.knob
                if self.rng.uniform() <= 0.50
                else self.rng.choice(valid_knobs)
            )
        self.init_robot_base_pos = self.stove

    def get_ep_meta(self):
        """
        Get the episode metadata for the stove knob tasks.
        This includes the language description of the task and the task references.
        """
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"{self.behavior.replace('_', ' ')} the {self.knob.replace('_', ' ')} burner of the stove"
        ep_meta["task_refs"] = dict(
            knob=self.knob,
            cookware_burner=self.cookware_burner,
        )
        return ep_meta

    def _setup_scene(self):
        """
        Reset the environment internal state for the stove knob tasks.
        This includes setting the stove knob state based on the behavior.
        """
        super()._setup_scene()

        if self.behavior == "turn_on":
            self.stove.set_knob_state(
                mode="off", knob=self.knob, env=self, rng=self.rng
            )
        elif self.behavior == "turn_off":
            self.stove.set_knob_state(mode="on", knob=self.knob, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the stove knob tasks.
        This includes the object placement configurations.
        Place the cookware on the stove burner.

        Returns:
            list: List of object configurations
        """
        cfgs = []

        cfgs.append(
            dict(
                name="cookware",
                obj_groups=("cookware"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.cookware_burner],
                    ),
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the stove knob manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        knobs_state = self.stove.get_knobs_state(env=self)
        knob_value = knobs_state[self.knob]

        knob_on = 0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35

        if self.behavior == "turn_on":
            success = knob_on
        elif self.behavior == "turn_off":
            success = not knob_on

        return success


class TurnOnStove(ManipulateStoveKnob):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="turn_on", *args, **kwargs)


class TurnOffStove(ManipulateStoveKnob):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="turn_off", *args, **kwargs)
