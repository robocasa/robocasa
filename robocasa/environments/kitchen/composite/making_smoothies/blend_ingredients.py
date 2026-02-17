from robocasa.environments.kitchen.kitchen import *


class BlendIngredients(Kitchen):
    """
    BlendIngredients: composite task for the making smoothies activity.
    Simulates the task of blending ingredients in the blender.
    Steps:
        1. Open the blender lid.
        2. Pick up the fruit from the counter.
        3. Put the fruit in the blender.
        4. Close the blender lid.
        5. Start the blender."""

    EXCLUDE_LAYOUTS = [21, 27]
    EXCLUDE_STYLES = Kitchen.PROBLEMATIC_BLENDER_LID_STYLES

    def __init__(self, enable_fixtures=None, *args, **kwargs):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["blender"]
        super().__init__(enable_fixtures=enable_fixtures, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.blender = self.register_fixture_ref(
            "blender", dict(id=FixtureType.BLENDER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.blender)
        )

        self.init_robot_base_ref = self.blender

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        fruit_lang = OU.get_obj_lang(self, "fruit")
        ep_meta[
            "lang"
        ] = f"Open the blender lid, put the {fruit_lang} in the blender, close the lid, and start the blender."

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="fruit",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.2, 0.2),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(ref=self.blender, loc="left_right"),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        fruit_in_blender = OU.obj_inside_of(self, "fruit", self.blender, th=0.01)
        blender_state = self.blender.get_state()
        return fruit_in_blender and blender_state["turned_on"]
