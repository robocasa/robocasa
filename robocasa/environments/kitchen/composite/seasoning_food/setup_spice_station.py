from robocasa.environments.kitchen.kitchen import *


class SetUpSpiceStation(Kitchen):
    """
    Set Up Spice Station: composite task for Seasoning Food.

    Simulates the task of organizing spices neatly near the stove.

    Steps:
        Pick up spice bottles in the cabinet.
        Place them neatly upright near the stove for easy access.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter_stove = self.register_fixture_ref(
            "counter_stove", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        spice = self.get_obj_lang("spice")
        bottle = self.get_obj_lang("bottle")
        shaker = self.get_obj_lang("shaker")

        ep_meta[
            "lang"
        ] = f"Collect the {spice}, {bottle}, and {shaker} from the cabinet and place them next to the stove."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="spice",
                obj_groups=("turmeric", "paprika"),
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="shaker",
                obj_groups="shaker",
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bottle",
                obj_groups=("ketchup", "mayonnaise"),
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        stove_threshold = 0.2

        d_spice = OU.obj_fixture_bbox_min_dist(self, "spice", self.stove)
        d_bottle = OU.obj_fixture_bbox_min_dist(self, "bottle", self.stove)
        d_shaker = OU.obj_fixture_bbox_min_dist(self, "shaker", self.stove)

        spice_near = d_spice <= stove_threshold
        bottle_near = d_bottle <= stove_threshold
        shaker_near = d_shaker <= stove_threshold

        on_counter_spice = OU.check_obj_any_counter_contact(self, "spice")
        on_counter_bottle = OU.check_obj_any_counter_contact(self, "bottle")
        on_counter_shaker = OU.check_obj_any_counter_contact(self, "shaker")

        gripper_far = (
            OU.gripper_obj_far(self, obj_name="spice")
            and OU.gripper_obj_far(self, obj_name="bottle")
            and OU.gripper_obj_far(self, obj_name="shaker")
        )

        return (
            spice_near
            and bottle_near
            and shaker_near
            and on_counter_spice
            and on_counter_bottle
            and on_counter_shaker
            and gripper_far
        )
