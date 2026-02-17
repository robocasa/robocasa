from robocasa.environments.kitchen.kitchen import *


class AirDryFruit(Kitchen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink
        if "refs" in self._ep_meta:
            self.num_fruit = self._ep_meta["refs"]["num_fruit"]
        else:
            self.num_fruit = int(self.rng.choice([1, 2, 3]))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        fruit0 = self.get_obj_lang("fruit0")
        fruit_lang = fruit0
        for i in range(1, self.num_fruit):
            fi = self.get_obj_lang(f"fruit{i}")
            fruit_lang += f", {fi}"
        pronoun = "it" if self.num_fruit == 1 else "them"
        ep_meta["lang"] = (
            f"Pick the {fruit_lang} from the sink and place {pronoun} on the cutting board to airdry. "
            "Then turn off the sink faucet."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["num_fruit"] = self.num_fruit
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.sink.set_handle_state(mode="on", env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        for i in range(self.num_fruit):
            cfgs.append(
                dict(
                    name=f"fruit{i}",
                    obj_groups="fruit",
                    graspable=True,
                    washable=True,
                    placement=dict(
                        fixture=self.sink,
                        size=(0.25, 0.25),
                        pos=(0.0, 1.0),
                    ),
                )
            )

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
                    size=(0.40, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        handle_state = self.sink.get_handle_state(env=self)
        water_on = handle_state["water_on"]

        fruit_on_board = all(
            [
                OU.check_obj_in_receptacle(self, f"fruit{i}", "cutting_board")
                for i in range(self.num_fruit)
            ]
        )
        gripper_far = all(
            [
                OU.gripper_obj_far(self, obj_name=f"fruit{i}", th=0.15)
                for i in range(self.num_fruit)
            ]
        )

        return (not water_on) and fruit_on_board and gripper_far
