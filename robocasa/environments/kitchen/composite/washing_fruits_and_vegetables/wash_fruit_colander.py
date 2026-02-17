from robocasa.environments.kitchen.kitchen import *


class WashFruitColander(Kitchen):
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
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                fruit_lang=fruit_lang
            )
        else:
            ep_meta[
                "lang"
            ] = f"Put the colander in the sink, put the {fruit_lang} in the colander, and turn on the sink faucet and pour water over the colander."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["num_fruit"] = self.num_fruit
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

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
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.30, 0.30),
                        pos=("ref", -1.0),
                    ),
                )
            )

        cfgs.append(
            dict(
                name="colander",
                obj_groups="colander",
                graspable=True,
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
        fruit_in_colander = all(
            [
                OU.check_obj_in_receptacle(self, f"fruit{i}", "colander")
                for i in range(self.num_fruit)
            ]
        )
        return self.sink.check_obj_under_water(self, "colander") and fruit_in_colander
