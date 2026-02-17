from robocasa.environments.kitchen.kitchen import *


class FilterMicrowavableItem(Kitchen):
    """
    Filter Microwavable Item: composite task for Microwaving Food activity.

    Simulates the task of filtering microwavable items from a bowl containing both
    microwavable (meat) and non-microwavable (fruit) items.

    Steps:
        1. Take the bowl containing meat and fruit from the counter
        2. Remove the fruit item from the bowl and place it on the plate
        3. Place the bowl with only meat in the microwave
        4. Close the microwave door
        5. Press the start button to begin microwaving
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.microwave)
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_lang = self.get_obj_lang("meat")
        fruit_lang = self.get_obj_lang("fruit")

        ep_meta["lang"] = (
            f"Remove the {fruit_lang} from the bowl and place it on the small plate. "
            f"Then place the bowl with only the {meat_lang} in the microwave, "
            f"close the door, and press the start button to microwave the {meat_lang}."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.microwave.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                object_scale=[1, 1, 0.75],
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(1.0, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                microwavable=True,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit",
                obj_groups="fruit",
                graspable=True,
                microwavable=False,
                placement=dict(
                    object="bowl",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=[0.75, 0.75, 1],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        bowl_in_microwave = OU.obj_inside_of(self, "bowl", self.microwave)
        meat_in_bowl = OU.check_obj_in_receptacle(self, "meat", "bowl")

        fruit_on_plate = OU.check_obj_in_receptacle(self, "fruit", "plate")
        door_closed = self.microwave.is_closed(self)

        if not door_closed:
            return False

        microwave_on = self.microwave.get_state()["turned_on"]
        gripper_far = OU.gripper_obj_far(self, obj_name="bowl")

        return (
            bowl_in_microwave
            and meat_in_bowl
            and fruit_on_plate
            and microwave_on
            and gripper_far
        )
