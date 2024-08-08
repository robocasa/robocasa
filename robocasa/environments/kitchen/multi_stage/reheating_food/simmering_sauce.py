from robocasa.environments.kitchen.kitchen import *


class SimmeringSauce(Kitchen):
    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.5, 0.4))
        )
        self.init_robot_base_pos = self.stove

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
            valid_knobs = [
                k for (k, v) in self.stove.knob_joints.items() if v is not None
            ]
            if self.knob_id == "random":
                self.knob = random.sample(valid_knobs, 1)[0]
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Place the pan on the {self.knob.replace('_', ' ')} burner on the stove. Then place the tomato and the onion in the pan and turn on the {self.knob.replace('_', ' ')} burner."
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                placement=dict(
                    fixture=self.counter,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(ref=self.stove, top_size=(0.50, 0.40)),
                    size=(0.25, 0.05),
                    pos=("ref", 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tomato",
                obj_groups="tomato",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.35, 0.2),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="onion",
                obj_groups="onion",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.35, 0.2),
                    pos=("ref", 0.0),
                ),
            )
        )

        return cfgs

    def _check_obj_location_on_stove(self, obj_name, threshold=0.08):

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
        pan_on_stove = self._check_obj_location_on_stove("pan") is not None
        tomato_in_pan = OU.check_obj_in_receptacle(self, "tomato", "pan")
        onion_in_pan = OU.check_obj_in_receptacle(self, "onion", "pan")

        knobs_state = self.stove.get_knobs_state(env=self)
        knob_value = knobs_state[self.knob]
        knob_on = 0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35

        return pan_on_stove and tomato_in_pan and onion_in_pan and knob_on
