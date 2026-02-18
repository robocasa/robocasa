from robocasa.environments.kitchen.atomic.kitchen_drawer import *
from robocasa.environments.kitchen.kitchen import *


class ChooseMeasuringCup(ManipulateDrawer):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="open", *args, **kwargs)

    def _setup_kitchen_references(self):

        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer)
        )
        if "refs" in self._ep_meta:
            self.use_tablespoon = self._ep_meta["refs"]["use_tablespoon"]
        else:
            self.use_tablespoon = self.rng.random() > 0.5

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        size_lang = "larger" if self.use_tablespoon else "smaller"
        ep_meta[
            "lang"
        ] = f"Open the {self.drawer_side} drawer and place the {size_lang} measuring cup on the counter."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["use_tablespoon"] = self.use_tablespoon
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer tasks. This includes the object placement configurations.
        Place the object inside the drawer and 1-4 distractors on the counter.
        Returns:
            list: List of object configurations.
        """
        cfgs = []

        measuring_cup_info = self.sample_object(
            groups=["measuring_cup"], obj_registries=self.obj_registries
        )

        # use the same object instance
        measuring_cup_inst = measuring_cup_info[1]["mjcf_path"]

        cfgs.append(
            dict(
                name="measuring_cup_small",
                obj_groups=measuring_cup_inst,
                placement=dict(
                    fixture=self.drawer,
                    size=(0.5, 0.4),
                    pos=(-1, 0),
                ),
                object_scale=0.75,
            )
        )

        cfgs.append(
            dict(
                name="measuring_cup_large",
                obj_groups=measuring_cup_inst,
                placement=dict(
                    fixture=self.drawer,
                    size=(0.5, 0.4),
                    pos=(-1, 0),
                ),
                object_scale=1.4,
            )
        )

        return cfgs

    def _check_success(self):

        large_cup_on_counter = OU.check_obj_fixture_contact(
            self, "measuring_cup_large", self.counter
        )
        small_cup_on_counter = OU.check_obj_fixture_contact(
            self, "measuring_cup_small", self.counter
        )

        if self.use_tablespoon:
            return (
                large_cup_on_counter
                and not small_cup_on_counter
                and OU.gripper_obj_far(self, "measuring_cup_large")
            )
        return (
            small_cup_on_counter
            and not large_cup_on_counter
            and OU.gripper_obj_far(self, "measuring_cup_small")
        )
