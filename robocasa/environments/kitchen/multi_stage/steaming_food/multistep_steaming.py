from robocasa.environments.kitchen.kitchen import *


class MultistepSteaming(Kitchen):
    """
    Multistep Steaming: composite task for Steaming Food activity.

    Simulates the task of steaming a vegetable.

    Steps:
        Turn on the sink. Then move the vegetable from the counter to the sink.
        Turn of the sink. Move the vegetable from the sink to the pot next to the
        stove. Finally, move the pot to the stove burner.

    Args:
        knob_id (str): The id of the knob who's burner the pot will be placed on.
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        # Flags to keep track of the task progress
        self.water_was_turned_on = False
        self.vegetable_was_in_sink = False
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.stove_counter = self.register_fixture_ref(
            "stove_counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_pos = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_name = self.get_obj_lang("vegetable1")
        ep_meta["lang"] = (
            "Turn on the sink faucet. "
            f"Then move the {vegetable_name} from the counter to the sink. "
            "Turn off the sink. Move the vegetable from the sink to the pot next to the stove. "
            f"Finally, move the pot to the {self.knob.replace('_', ' ')} burner."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)

        valid_knobs = self.stove.get_knobs_state(env=self).keys()
        if self.knob_id == "random":
            self.knob = self.rng.choice(list(valid_knobs))
        else:
            assert self.knob_id in valid_knobs
            self.knob = self.knob

        self.stove.set_knob_state(mode="off", knob=self.knob, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pot",
                obj_groups="pot",
                placement=dict(
                    fixture=self.stove_counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.05, 0.05),
                    pos=("ref", -0.7),
                    rotation=np.pi / 2,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.4, 0.4),
                    pos=(-1.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_obj_location_on_stove(self, obj_name, threshold=0.08):
        """
        Check if the object is on any of the burners of the stove
        Returns the location of the burner if the object is on it. Otherwise, returns None
        """

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
                    if obj_on_site:
                        return location

        return None

    def _check_success(self):

        handle_state = self.sink.get_handle_state(env=self)
        water_on = handle_state["water_on"]

        if water_on:
            self.water_was_turned_on = True

        pot_loc = self._check_obj_location_on_stove(obj_name="pot")
        pot_on_burner = pot_loc == self.knob

        vegetable_in_sink = OU.obj_inside_of(self, "vegetable1", self.sink)

        # make sure the vegetable was in the sink when the water was turned
        if vegetable_in_sink and water_on:
            self.vegetable_was_in_sink = True

        vegetable_in_pot = OU.check_obj_in_receptacle(self, "vegetable1", "pot")

        return (
            self.water_was_turned_on
            and self.vegetable_was_in_sink
            and (not water_on)
            and pot_on_burner
            and vegetable_in_pot
        )
