from robocasa.environments.kitchen.kitchen import *


class PlaceMeatInMarinade(Kitchen):
    """
    Place Meat in Marinade: composite task for Preparing Marinade activity.

    Simulates the task of placing meat from the fridge onto a pan of marinade on the stove.

    Steps:
        Retrieve a piece of meat from the fridge and place it on the pan with marinade on the stove.
    """

    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

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
        meat_lang = self.get_obj_lang("meat")

        ep_meta[
            "lang"
        ] = f"Retrieve the {meat_lang} from the fridge and place it on the saucepan with marinade on the stove. "
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["knob"] = self.knob
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "pan", [0.78, 0.39, 0.20, 0.5])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups=("saucepan", "pan"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.05, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                exclude_obj_groups=("shrimp"),
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.4, 0.25),
                    pos=(0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="fridge_distractor",
                exclude_obj_groups=("meat"),
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.4, 0.3),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_meat_far = OU.gripper_obj_far(self, obj_name="meat")
        meat_in_pan = OU.check_obj_in_receptacle(self, "meat", "pan")
        pan_on_stove = OU.check_obj_fixture_contact(self, "pan", self.stove)

        return meat_in_pan and gripper_meat_far and pan_on_stove
