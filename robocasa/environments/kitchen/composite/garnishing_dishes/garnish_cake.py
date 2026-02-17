from robocasa.environments.kitchen.kitchen import *


class GarnishCake(Kitchen):
    """
    Garnish Cake: composite task for Garnishing Dishes activity.

    Simulates the task of garnishing a cake with fruits.

    Steps:
        1. Take one cherry from the fruit plate and place it on top of the cake, or next to the cake
        2. Take one strawberry from the fruit plate and place it on the plate next to the cake
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self, obj_registries=("objaverse", "lightwheel", "aigen"), *args, **kwargs
    ):
        obj_registries = list(obj_registries or [])
        # make sure to use aigen objects to access the strawberry + cherry
        if "aigen" not in obj_registries:
            obj_registries.append("aigen")
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Place one cherry either on top of the cake or on the cake plate, next to the cake. "
            "Then place one strawberry on the cake plate, next to the cake."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cake_plate",
                obj_groups="plate",
                object_scale=1.2,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit_plate",
                obj_groups="plate",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(1.0, 0.35),
                    pos=(0, "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cake",
                obj_groups="cake",
                object_scale=1.85,
                placement=dict(
                    object="cake_plate",
                    size=(0.01, 0.01),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cherry1",
                obj_groups="cherry",
                placement=dict(
                    object="fruit_plate",
                    size=(0.8, 0.8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cherry2",
                obj_groups="cherry",
                placement=dict(
                    object="fruit_plate",
                    size=(0.8, 0.8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="strawberry1",
                obj_groups="strawberry",
                graspable=True,
                placement=dict(
                    object="fruit_plate",
                    size=(0.8, 0.8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="strawberry2",
                obj_groups="strawberry",
                graspable=True,
                placement=dict(
                    object="fruit_plate",
                    size=(0.8, 0.8),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        cherry1_in_contact = OU.check_obj_in_receptacle(
            self, "cherry1", "cake"
        ) or OU.check_obj_in_receptacle(self, "cherry1", "cake_plate")
        cherry2_in_contact = OU.check_obj_in_receptacle(
            self, "cherry2", "cake"
        ) or OU.check_obj_in_receptacle(self, "cherry2", "cake_plate")
        exactly_one_cherry_in_contact = (
            sum([cherry1_in_contact, cherry2_in_contact]) == 1
        )

        strawberry1_on_plate = OU.check_obj_in_receptacle(
            self, "strawberry1", "cake_plate"
        )
        strawberry2_on_plate = OU.check_obj_in_receptacle(
            self, "strawberry2", "cake_plate"
        )
        strawberry_count_on_plate = sum([strawberry1_on_plate, strawberry2_on_plate])
        exactly_one_strawberry_on_plate = strawberry_count_on_plate == 1

        plate_on_counter = OU.check_obj_fixture_contact(
            self, "cake_plate", self.dining_counter
        )
        gripper_away = OU.gripper_obj_far(self, "cake_plate")

        return (
            exactly_one_cherry_in_contact
            and exactly_one_strawberry_on_plate
            and plate_on_counter
            and gripper_away
        )
