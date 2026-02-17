from robocasa.environments.kitchen.kitchen import *


class ReheatMeal(Kitchen):
    """
    Reheat Meal: composite task for Microwaving Food activity.

    Simulates the task of reheating a meal in a tupperware container containing meat and vegetables.

    Steps:
        1. Place the tupperware containing meat and vegetables in the microwave
        2. Close the microwave door
        3. Press the start button to begin microwaving
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

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_lang = self.get_obj_lang("meat")
        veg_lang = self.get_obj_lang("vegetable")
        ep_meta["lang"] = (
            f"Place the tupperware containing the {meat_lang} and {veg_lang} meal in the microwave. "
            "Close the microwave door and press the start button to reheat the meal."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)
        self.microwave.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="tupperware",
                obj_groups="tupperware",
                object_scale=[1.5, 1.5, 1],
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.30, 0.35),
                    pos=(0, -1.0),
                    rotation=(np.pi / 2),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                microwavable=True,
                placement=dict(
                    object="tupperware",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                microwavable=True,
                placement=dict(
                    object="tupperware",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        tupperware_in_microwave = OU.obj_inside_of(self, "tupperware", self.microwave)
        meat_in_tupperware = OU.check_obj_in_receptacle(self, "meat", "tupperware")
        vegetable_in_tupperware = OU.check_obj_in_receptacle(
            self, "vegetable", "tupperware"
        )

        door_closed = self.microwave.is_closed(self)
        if not door_closed:
            return False

        microwave_on = self.microwave.get_state()["turned_on"]
        gripper_far = OU.gripper_obj_far(self, obj_name="tupperware")

        return (
            tupperware_in_microwave
            and meat_in_tupperware
            and vegetable_in_tupperware
            and microwave_on
            and gripper_far
        )
