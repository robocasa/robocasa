from robocasa.environments.kitchen.kitchen import *


class PlaceMicrowaveSafeItem(Kitchen):
    """
    Place Microwave Safe Item: composite task for Microwaving Food activity.

    Simulates the task of identifying a microwave-safe and non-microwave-safe drink item.
    The robot must identify which drinkware item is safe for microwaving and place it in the microwave.

    Steps:
        1. Identify microwave-safe drink item
        2. Identify non-microwave-safe drink item
        3. Place microwave-safe item in the microwave
        4. Close the microwave door
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

        if "refs" in self._ep_meta:
            self.safe_item = self._ep_meta["refs"]["microwave_safe_item"]
            self.unsafe_item = self._ep_meta["refs"]["microwave_unsafe_item"]
        else:
            microwave_safe_items = ["cup", "mug", "glass_cup"]
            microwave_unsafe_items = [
                "bottled_water",
                "bottled_drink",
                "liquor",
                "milk",
                "wine",
                "beer",
            ]

            self.safe_item = self.rng.choice(microwave_safe_items)
            self.unsafe_item = self.rng.choice(microwave_unsafe_items)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Place the microwavable item in the microwave. "
            "Then, close the microwave door and press the start button."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"].update(
            microwave_safe_item=self.safe_item,
            microwave_unsafe_item=self.unsafe_item,
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.microwave.open_door(self)
        OU.add_obj_liquid_site(self, "microwave_safe_item", (0.49, 0.12, 0.025, 0.4))

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="microwave_safe_item",
                obj_groups=self.safe_item,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="microwave_unsafe_item",
                obj_groups=self.unsafe_item,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    reuse_region_from="microwave_safe_item",
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        safe_item_in_microwave = OU.obj_inside_of(
            self, "microwave_safe_item", self.microwave
        )
        unsafe_item_not_in_microwave = not OU.obj_inside_of(
            self, "microwave_unsafe_item", self.microwave
        )

        door_closed = self.microwave.is_closed(self)
        if not door_closed:
            return False

        microwave_on = self.microwave.get_state()["turned_on"]

        return safe_item_in_microwave and unsafe_item_not_in_microwave and microwave_on
