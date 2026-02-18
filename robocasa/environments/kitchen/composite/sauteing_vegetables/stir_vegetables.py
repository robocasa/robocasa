from robocasa.environments.kitchen.kitchen import *


class StirVegetables(Kitchen):
    """
    Stir the Vegetables: composite task for Sauteing Vegetables activity.

    Simulates the task of retrieving a spatula from the counter and using it to stir vegetables in a pot.

    Steps:
        1. Retrieve the spatula from the counter.
        2. Use the spatula to simulate stirring the vegetables inside the pot.
        3. Stir the vegetables for 5 timesteps.
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

        self.init_robot_base_ref = self.stove

        if "refs" in self._ep_meta:
            self.knob = self._ep_meta["refs"]["knob"]
        else:
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
        veg1_lang = self.get_obj_lang("veg1")
        veg2_lang = self.get_obj_lang("veg2")
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions).format(
                veg1_lang=veg1_lang, veg2_lang=veg2_lang
            )
        else:
            ep_meta["lang"] = (
                f"Put the {veg1_lang} and {veg2_lang} in the pot. "
                f"Retrieve the spatula and lightly stir the vegetables in the pot."
            )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.success_time = 0
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="spatula",
                obj_groups="spatula",
                object_scale=[1, 1, 2],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.stove),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
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

        def veg_cfg(name, offset):
            return dict(
                name=name,
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove, loc="left_right", top_size=(0.3, 0.3)
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", offset),
                ),
            )

        cfgs.append(veg_cfg("veg1", -1.0))
        cfgs.append(veg_cfg("veg2", -0.7))

        return cfgs

    def update_state(self):
        super().update_state()
        # Check if the vegetables are being stirred and update success time accordingly
        veg1_in_pot = OU.check_obj_in_receptacle(self, "veg1", "pot")
        veg2_in_pot = OU.check_obj_in_receptacle(self, "veg2", "pot")

        pot_loc = (
            self.stove.check_obj_location_on_stove(
                env=self, obj_name="pot", threshold=0.15
            )
            == self.knob
        )

        spatula_grasped = OU.check_obj_grasped(self, "spatula")
        if spatula_grasped:
            objects_stirred = self._detect_stirring(["veg1", "veg2"])
            if pot_loc and veg1_in_pot and veg2_in_pot and objects_stirred:
                self.success_time += 1

    def _check_success(self):
        return self.success_time >= 5

    def _detect_stirring(self, obj_names, movement_threshold=0.05):
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
