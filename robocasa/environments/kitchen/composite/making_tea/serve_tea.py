from robocasa.environments.kitchen.kitchen import *


class ServeTea(Kitchen):
    """Serve Tea: composite task for Making Tea activity.
    Simulates the task of serving tea in a teacup on a dining table.
    Steps:
        1. Pick the teacup from the microwave.
        2. Navigate to the dining table.
        3. Place the teacup on the saucer on the dining table.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self):
        super()._setup_scene()
        self.microwave.open_door(self)
        OU.add_obj_liquid_site(self, "teacup", (0.757, 0.561, 0.4, 1.0))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the cup of tea, navigate to the dining table, and place it on the saucer."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="teacup",
                obj_groups="mug",
                placement=dict(
                    fixture=self.microwave,
                    size=(0.3, 0.2),
                ),
            )
        )
        cfgs.append(
            dict(
                name="saucer",
                obj_groups="plate",
                # saucer = small plate
                object_scale=[0.6, 0.6, 1],
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.5, 0.3),
                    pos=("ref", "ref"),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        teacup_on_saucer = OU.check_obj_in_receptacle(self, "teacup", "saucer")
        saucer_on_table = OU.check_obj_fixture_contact(
            self, "saucer", self.dining_table
        )

        return (
            teacup_on_saucer and OU.gripper_obj_far(self, "teacup") and saucer_on_table
        )
