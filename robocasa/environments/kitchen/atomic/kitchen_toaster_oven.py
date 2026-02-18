from robocasa.environments.kitchen.kitchen import *


class AdjustToasterOvenTemperature(Kitchen):
    """
    Class encapsulating atomic task for adjusting the toaster oven temperature.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        if "initial_temp" in self._ep_meta:
            self.initial_temp = self._ep_meta["initial_temp"]
        else:
            self.initial_temp = float(self.rng.random())

        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        direction = "Increase" if self.should_increase else "Decrease"
        ep_meta["lang"] = f"{direction.capitalize()} the toaster oven temperature."
        ep_meta["initial_temp"] = self.initial_temp
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.toaster_oven.set_temperature(env=self, val=self.initial_temp)
        self.should_increase = self.initial_temp < 0.5

    def _check_success(self):
        toaster_oven_state = self.toaster_oven.get_state(self)
        current_temp = toaster_oven_state["temperature"]
        temp_diff = current_temp - self.initial_temp

        if self.should_increase:
            return temp_diff >= 0.15
        else:
            return temp_diff <= -0.15


class TurnOnToasterOven(Kitchen):
    """
    Class encapsulating atomic task for turning on the toaster oven by setting the timer.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Turn on the toaster oven by setting the timer."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

    def _check_success(self):
        return self.toaster_oven.get_state(self)["time"] >= 0.1


class SlideToasterOvenRack(Kitchen):
    """
    Class encapsulating the atomic task for sliding rack in or out of the toaster oven.
    """

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["toaster_oven"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.init_robot_base_ref = self.toaster_oven
        if "rack_level" in self._ep_meta:
            self.should_pull = self._ep_meta["should_pull"]
            self.rack_level = self._ep_meta["rack_level"]
        else:
            self.should_pull = self.rng.random() > 0.5
            self.rack_level = 1 if self.rng.random() > 0.5 else 0

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        direction = "out" if self.should_pull else "in"

        if "rack" in self.chosen_toaster_receptacle:
            receptacle_type = "rack"
        else:
            receptacle_type = "tray"

        if self.toaster_oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Fully slide the toaster oven {rack_pos} {receptacle_type} {direction}."
        else:
            ep_meta[
                "lang"
            ] = f"Fully slide the toaster oven {receptacle_type} {direction}."
        ep_meta["should_pull"] = self.should_pull
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.toaster_oven.open_door(self)

        if not self.should_pull:
            self.chosen_toaster_receptacle = self.toaster_oven.slide_rack(
                self, rack_level=self.rack_level
            )
        else:
            self.chosen_toaster_receptacle = self.toaster_oven.slide_rack(
                self, value=0.50, rack_level=self.rack_level
            )

    def _check_success(self):
        toaster_oven_state = self.toaster_oven.get_state(rack_level=self.rack_level)

        movable_keys = [
            k
            for k in toaster_oven_state
            if k.startswith("rack") or k.startswith("tray")
        ]

        key = movable_keys[0]
        current_pos = toaster_oven_state[key]

        if current_pos is None:
            return False

        if self.should_pull:
            return current_pos >= 0.95
        else:
            return current_pos <= 0.05
