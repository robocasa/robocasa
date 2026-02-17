from robocasa.environments.kitchen.kitchen import *


class BowlAndCup(Kitchen):
    """
    Bowl And Cup: composite task for Clearing Table activity.

    Simulates the process of efficiently clearing the table.

    Steps:
        Place the cup inside the bowl on the dining table and move it to any counter.

    Restricted to layouts with a dining table.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool, size=(0.50, 0.35)),
        )

        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Place the cup inside the bowl on the dining table and move the bowl to any counter."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"cup",
                obj_groups=["cup"],
                graspable=True,
                washable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.50, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"bowl",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    ref_obj="cup",
                    size=(0.50, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        cup_in_bowl = OU.check_obj_in_receptacle(self, "cup", "bowl")
        bowl_on_counter = any(
            [
                OU.check_obj_fixture_contact(self, "bowl", fxtr)
                for (_, fxtr) in self.fixtures.items()
                if isinstance(fxtr, Counter) and fxtr != self.dining_table
            ]
        )
        return cup_in_bowl and bowl_on_counter and OU.gripper_obj_far(self, "bowl")
