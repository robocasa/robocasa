from robocasa.environments.kitchen.kitchen import *


class CheeseMixing(Kitchen):
    """
    Mix Cheese: composite task for Mixing Ingredients activity.
    Simulates the task of melting cheese in a pot on the stove and stirring it with a spatula.
    Steps:
        1) Put the cheese in the pot filled with milk on the stove
        2) Stir the cheese in the pot using the spatula
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )

        self.init_robot_base_ref = self.counter

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob_id"]
        else:
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
        ep_meta["lang"] = f"Put the cheese in the pot filled with milk and stir."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob_id"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.success_time = 0
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")
        liquid_geom_id = self.sim.model.geom_name2id("pot_liquid")
        self.sim.model.geom_rgba[liquid_geom_id] = [1, 1, 1, 0.8]

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="spatula",
                obj_groups="spatula",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.2, 0.4),
                    pos=("ref", -1.0),
                    rotation=0,
                ),
                object_scale=[1.0, 1.0, 2.0],
            )
        )

        cfgs.append(
            dict(
                name="cheese_plate",
                obj_groups="plate",
                init_robot_here=True,
                object_scale=[0.75, 0.75, 1.0],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cheese",
                obj_groups="cheese",
                init_robot_here=True,
                placement=dict(
                    object="cheese_plate",
                    size=(0.65, 0.65),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pot",
                obj_groups="pot",
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

        cheese_in_pot = OU.check_obj_in_receptacle(self, "cheese", "pot")

        pot_loc = (
            self.stove.check_obj_location_on_stove(
                env=self, obj_name="pot", threshold=0.15
            )
            == self.knob
        )

        objects_stirred = self._detect_stirring(["cheese"])

        if pot_loc and cheese_in_pot and objects_stirred:
            self.success_time += 1

        return self.success_time > 2

    def _detect_stirring(self, obj_names, movement_threshold=0.15):
        all_objects_stirred = True
        spatula = self.objects["spatula"]
        contact_with_spatula = False

        for obj_name in obj_names:
            obj = self.objects[obj_name]
            obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
            prev_obj_pos = getattr(self, f"prev_obj_pos_{obj_name}", obj_pos)

            movement_vector = obj_pos - prev_obj_pos
            movement_magnitude = np.linalg.norm(movement_vector[:2]) * 1e2
            movement_detected = movement_magnitude > movement_threshold

            inside_pot = OU.check_obj_in_receptacle(self, obj_name, "pot")

            if not contact_with_spatula:
                contact_with_spatula = self.check_contact(spatula, obj)

            if not (movement_detected and inside_pot and contact_with_spatula):
                all_objects_stirred = False

            setattr(self, f"prev_obj_pos_{obj_name}", obj_pos)

        return all_objects_stirred
