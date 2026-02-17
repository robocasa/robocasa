from robocasa.environments.kitchen.kitchen import *


class MicrowaveThawingFridge(Kitchen):
    """
    Microwave Thawing Fridge: composite task for Defrosting Food activity.
    Simulates the task of defrosting food in a microwave.

    Steps:
        Pick the frozen food from the fridge and place it in the microwave.
        Then, close the microwave door and turn it on.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.init_robot_base_ref = self.fridge

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self)
        self.microwave.open_door(env=self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_lang = self.get_obj_lang("meat")
        ep_meta[
            "lang"
        ] = f"Transport the {meat_lang} from the fridge to the microwave and turn on the microwave."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.275),
                    pos=(0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.fridge,
                    size=(0.3, 0.25),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        obj_in_microwave = OU.obj_inside_of(self, "meat", self.microwave)

        microwave_on = self.microwave.get_state()["turned_on"]
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="meat")

        return obj_in_microwave and gripper_obj_far and microwave_on
