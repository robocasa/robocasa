from robocasa.environments.kitchen.kitchen import *


class VeggieBoil(Kitchen):
    def __init__(self, *args, **kwargs):
        self.pot_filled = False
        self.filled_time = 0
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter_sink = self.register_fixture_ref(
            "counter_sink", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter_stove = self.register_fixture_ref(
            "counter_stove", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_name = self.get_obj_lang("food")
        ep_meta[
            "lang"
        ] = f"""Pick up the pot and place it in the sink. Then turn on the sink and let the pot fill up with water. Then turn the sink off and move the pot to the stove.
        Lastly, turn on the stove and place the {food_name} in the pot for boiling."""
        return ep_meta

    def _reset_internal(self):
        self.pot_filled = False
        self.filled_time = 0
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"pot",
                obj_groups="pot",
                placement=dict(
                    fixture=self.counter_sink,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.5, 0.5)
                    ),
                    size=(0.05, 0.05),
                    pos=("ref", -0.55),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"food",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.counter_stove,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="nn",
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter_stove,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="nn",
                    ),
                    size=(0.30, 0.30),
                    pos=(0, 1.0),
                ),
            )
        )

        # mitigate randomization errors
        if self.counter_sink != self.counter_stove:

            cfgs.append(
                dict(
                    name="distr_counter2",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.counter_sink,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.30, 0.30),
                        pos=(0, 1.0),
                    ),
                )
            )

        return cfgs

    def _check_obj_location_on_stove(self, obj_name, threshold=0.08):

        knobs_state = self.stove.get_knobs_state(env=self)
        obj = self.objects[obj_name]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])[0:2]
        obj_on_stove = OU.check_obj_fixture_contact(self, obj_name, self.stove)
        if obj_on_stove:
            for location, site in self.stove.burner_sites.items():
                if site is not None:
                    burner_pos = np.array(
                        self.sim.data.get_site_xpos(site.get("name"))
                    )[0:2]
                    dist = np.linalg.norm(burner_pos - obj_pos)

                    obj_on_site = dist < threshold
                    knob_on = (
                        (0.35 <= np.abs(knobs_state[location]) <= 2 * np.pi - 0.35)
                        if location in knobs_state
                        else False
                    )

                    if obj_on_site and knob_on:
                        return location

        return None

    def _check_success(self):
        pot_in_sink = OU.obj_inside_of(self, "pot", self.sink)
        water_on = self.sink.get_handle_state(env=self)["water_on"]

        if pot_in_sink and water_on:
            self.filled_time += 1
            self.pot_filled = self.filled_time > 10
        else:
            self.filled_time = 0

        vegetables_in_pot = OU.check_obj_in_receptacle(self, "food", "pot")
        pot_on_stove = self._check_obj_location_on_stove("pot") is not None
        return self.pot_filled and vegetables_in_pot and not water_on and pot_on_stove
