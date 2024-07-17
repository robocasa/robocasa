from robocasa.environments.kitchen.kitchen import *


class AssembleCookingArray(Kitchen):
    def __init__(self, knob_id="random", *args, **kwargs):
        self.knob_id = knob_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref( "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40]))
        self.cab = self.register_fixture_ref( "cab", dict(id=FixtureType.CABINET_TOP, ref=self.counter))
        self.init_robot_base_pos = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_name = self.get_obj_lang("meat")
        condiment_name = self.get_obj_lang("condiment")
        vegetable_name = self.get_obj_lang("vegetable")
        ep_meta["lang"] = f"Move the {meat_name} onto the pan on the stove. Then, move the {condiment_name} and {vegetable_name} from the cabinet to the counter where the plate is."
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()

        valid_knobs = self.stove.get_knobs_state(env=self).keys()
        if self.knob_id == "random":
            self.knob = self.rng.choice(valid_knobs)
        else:
            assert self.knob_id in valid_knobs
            self.knob = self.knob

        self.stove.set_knob_state(mode="off", knob=self.knob, env=self, rng=self.rng)
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        
        cfgs.append(dict(
            name="pan",
            obj_groups=("pan"),
            placement=dict(
                fixture=self.stove,
                ensure_object_boundary_in_range=False,
                size=(0.05, 0.05),
            ),
        ))


        cfgs.append(dict(
            name="meat",
            obj_groups="meat",
            graspable=True,
            heatable=True,
            placement=dict(
                fixture=self.counter,
                loc="nn",
                sample_region_kwargs=dict(
                    ref=self.stove,
                ),
                size=(0.30, 0.30),
                pos=("ref", -1.0),
                try_to_place_in="container",
            ),
        ))

        cfgs.append(dict(
            name="condiment",
            obj_groups="condiment",
            graspable=True,
            placement=dict(
                fixture=self.cab,
                size=(0.50, 0.20),
                pos=(-1.0, -1.0),
            ),
        ))

        cfgs.append(dict(
            name="vegetable",
            obj_groups="vegetable",
            graspable=True,
            placement=dict(
                fixture=self.cab,
                size=(0.50, 0.20),
                pos=(1.0, -1.0),
            ),
        ))

        return cfgs

    def _check_success(self):
        meat_in_pan = OU.check_obj_in_receptacle(self, "meat", "pan", th=0.07)
        gripper_vegetable_far = OU.gripper_obj_far(self, obj_name="vegetable")
        gripper_condiment_far = OU.gripper_obj_far(self, obj_name="condiment")
        gripper_meat_far = OU.gripper_obj_far(self, obj_name="meat")
        vegetable_on_counter = OU.check_obj_fixture_contact(self, "vegetable", self.counter)
        condiment_on_counter = OU.check_obj_fixture_contact(self, "condiment", self.counter)
        return meat_in_pan and gripper_vegetable_far and gripper_condiment_far and gripper_meat_far and vegetable_on_counter and condiment_on_counter