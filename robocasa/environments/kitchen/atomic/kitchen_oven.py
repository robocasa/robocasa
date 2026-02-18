from robocasa.environments.kitchen.kitchen import *


class PreheatOven(Kitchen):
    """
    Class encapsulating the atomic oven preheating task.
    """

    EXCLUDE_LAYOUTS = Kitchen.OVEN_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.init_robot_base_ref = self.oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Preheat the oven by turning the temperature knob."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _check_success(self):
        return self.oven.get_state()["temperature"] >= 0.25


class SlideOvenRack(Kitchen):
    """
    Atomic task for sliding an oven rack.
    """

    EXCLUDE_LAYOUTS = Kitchen.OVEN_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.init_robot_base_ref = self.oven
        if "rack_level" in self._ep_meta:
            self.should_pull = self._ep_meta["should_pull"]
            self.rack_level = self._ep_meta["rack_level"]
        else:
            self.should_pull = self.rng.random() > 0.5
            self.rack_level = 1 if self.rng.random() > 0.5 else 0

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        direction = "out" if self.should_pull else "in"
        if self.oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta["lang"] = f"Fully slide the {rack_pos} oven rack {direction}."
        else:
            ep_meta["lang"] = f"Fully slide the oven rack {direction}."
        ep_meta["should_pull"] = self.should_pull
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.oven.open_door(self)

        if not self.should_pull:
            self.oven.slide_rack(self, rack_level=self.rack_level)
        else:
            self.oven.slide_rack(self, value=0.50, rack_level=self.rack_level)

    def _check_success(self):
        oven_state = self.oven.get_state(rack_level=self.rack_level)

        movable_keys = [k for k in oven_state if k.startswith("rack")]
        if not movable_keys:
            return False

        key = movable_keys[0]
        current_pos = oven_state[key]

        if current_pos is None:
            return False

        if self.should_pull:
            return current_pos >= 0.95
        else:
            return current_pos <= 0.05
