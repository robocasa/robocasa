from robocasa.environments.kitchen.kitchen import *


class PrepareCocktailStation(Kitchen):
    """
    Prepare Cocktail Station: composite task for Serving Beverages activity.

    Simulates the task of preparing a cocktail station on the dining counter.

    Steps:
        1. There is a bowl on the dining counter.
        2. Grab a lemon wedge from the fridge.
        3. Place the lemon wedge on the dining counter with the bowl.
        4. Get a liquor and glass cup from the cabinet.
        5. Place the liquor and glass cup in the same area as the bowl and lemon wedge.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "There is a bowl on the dining counter. Grab a lemon wedge from the fridge and place it "
            "in the bowl. Then get the liquor bottle and glass cup "
            "from the cabinet and place them next to the bowl."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bowl",
                obj_groups=("bowl"),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.5, 0.3),
                    pos=("ref", "ref"),
                ),
            )
        )
        cfgs.append(
            dict(
                name="lemon_wedge",
                obj_groups=("lemon_wedge"),
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=-1,
                    ),
                    size=(0.20, 0.15),
                    pos=(0.0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="liquor",
                obj_groups=("liquor"),
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="glass_cup",
                obj_groups=("glass_cup"),
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        bowl_pos = self.sim.data.body_xpos[self.obj_body_id["bowl"]][:2]
        lemon_pos = self.sim.data.body_xpos[self.obj_body_id["lemon_wedge"]][:2]
        lemon_in_bowl = OU.check_obj_in_receptacle(self, "lemon_wedge", "bowl")

        liquor_on_counter = OU.check_obj_fixture_contact(
            self, "liquor", self.dining_counter
        )
        glass_cup_on_counter = OU.check_obj_fixture_contact(
            self, "glass_cup", self.dining_counter
        )

        liquor_pos = self.sim.data.body_xpos[self.obj_body_id["liquor"]][:2]
        glass_cup_pos = self.sim.data.body_xpos[self.obj_body_id["glass_cup"]][:2]
        liquor_near_bowl = np.linalg.norm(bowl_pos - liquor_pos) <= 0.35
        glass_cup_near_bowl = np.linalg.norm(bowl_pos - glass_cup_pos) <= 0.35

        gripper_far = (
            OU.gripper_obj_far(self, obj_name="lemon_wedge")
            and OU.gripper_obj_far(self, obj_name="liquor")
            and OU.gripper_obj_far(self, obj_name="glass_cup")
        )

        return (
            lemon_in_bowl
            and liquor_on_counter
            and liquor_near_bowl
            and glass_cup_on_counter
            and glass_cup_near_bowl
            and gripper_far
        )
