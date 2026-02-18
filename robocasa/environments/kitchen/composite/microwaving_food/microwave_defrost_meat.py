from robocasa.environments.kitchen.kitchen import *


class MicrowaveDefrostMeat(Kitchen):
    """
    Microwave Defrost Meat: composite task for Microwaving Food activity.

    Simulates the task of defrosting meat in the microwave.

    Steps:
        1. Place the tupperware containing meat in the microwave
        2. Close the microwave door
        3. Press the start button to begin microwaving
    """

    EXCLUDE_LAYOUTS = Kitchen.FREEZER_EXCLUDED_LAYOUTS

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
        ep_meta["lang"] = (
            f"Place the tupperware containing the frozen {meat_lang} in the microwave. "
            "Close the microwave door and press the start button to defrost the meat."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self, compartment="freezer")
        self.microwave.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="tupperware",
                obj_groups="tupperware",
                object_scale=[1.25, 1.25, 1],
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        compartment="freezer",
                        z_range=(1.0, 1.5),
                    ),
                    size=(0.30, 0.30),
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

        return cfgs

    def _check_success(self):
        tupperware_in_microwave = OU.obj_inside_of(self, "tupperware", self.microwave)
        meat_in_tupperware = OU.check_obj_in_receptacle(self, "meat", "tupperware")

        door_closed = self.microwave.is_closed(self)
        if not door_closed:
            return False

        microwave_on = self.microwave.get_state()["turned_on"]
        gripper_far = OU.gripper_obj_far(self, obj_name="tupperware")

        return (
            tupperware_in_microwave
            and meat_in_tupperware
            and microwave_on
            and gripper_far
        )
