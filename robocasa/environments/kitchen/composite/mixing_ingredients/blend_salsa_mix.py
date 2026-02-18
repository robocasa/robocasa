from robocasa.environments.kitchen.kitchen import *


class BlendSalsaMix(Kitchen):
    """
    Blend Salsa Mix: composite task for Mixing Ingredients activity.

    Simulates the task of making salsa by blending vegetables in a blender.

    Steps:
        1. Open the blender lid
        2. Place tomato, chili pepper, and onion in the blender
        3. Close the blender lid
        4. Turn on the blender to create salsa
    """

    EXCLUDE_LAYOUTS = [21, 27]
    EXCLUDE_STYLES = Kitchen.PROBLEMATIC_BLENDER_LID_STYLES

    def __init__(
        self,
        enable_fixtures=None,
        obj_registries=("objaverse", "lightwheel", "aigen"),
        *args,
        **kwargs,
    ):
        enable_fixtures = enable_fixtures or []
        enable_fixtures = list(enable_fixtures) + ["blender"]
        obj_registries = list(obj_registries or [])
        # make sure to use aigen objects to access the chili pepper
        if "aigen" not in obj_registries:
            obj_registries.append("aigen")
        super().__init__(
            enable_fixtures=enable_fixtures,
            obj_registries=obj_registries,
            *args,
            **kwargs,
        )

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
        ep_meta["lang"] = (
            "Open the blender lid and place the tomato, chili pepper, and onion in the blender. "
            "Then place the lid back on and press the power button to make salsa."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.1,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.blender, loc="left_right"),
                    size=(0.5, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tomato",
                obj_groups="tomato",
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="chili_pepper",
                obj_groups="chili_pepper",
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="onion",
                obj_groups="onion",
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        tomato_in_blender = OU.obj_inside_of(self, "tomato", self.blender, th=0.01)
        chili_pepper_in_blender = OU.obj_inside_of(
            self, "chili_pepper", self.blender, th=0.01
        )
        onion_in_blender = OU.obj_inside_of(self, "onion", self.blender, th=0.01)

        # Check if blender is turned on
        blender_state = self.blender.get_state()
        blender_on = blender_state["turned_on"]

        return (
            tomato_in_blender
            and chili_pepper_in_blender
            and onion_in_blender
            and blender_on
        )
