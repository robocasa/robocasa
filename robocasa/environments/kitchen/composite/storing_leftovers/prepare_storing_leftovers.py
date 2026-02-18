from robocasa.environments.kitchen.kitchen import *


class PrepareStoringLeftovers(Kitchen):
    """
    Prepare Storing Leftovers: composite task for Storing Leftovers activity.

    Simulates the process of retrieving storage supplies (tupperware and aluminum foil)
    from a cabinet and placing them on the dining counter near leftover food to prepare
    for storing leftovers.

    Steps:
        1. Retrieve the tupperware from the cabinet.
        2. Retrieve the aluminum foil from the cabinet.
        3. Place both items on the dining counter near the plate with leftover food.
    """

    EXCLUDE_LAYOUTS = (
        Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS + Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )

        if "refs" in self._ep_meta:
            self.aluminum_foil_x_pos = self._ep_meta["refs"]["aluminum_foil_x_pos"]
            self.tupperware_x_pos = self._ep_meta["refs"]["tupperware_x_pos"]
        else:
            self.aluminum_foil_x_pos = self.rng.choice([-1.0, 1.0])
            self.tupperware_x_pos = -self.aluminum_foil_x_pos

        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        food1_lang = self.get_obj_lang("food1")
        food2_lang = self.get_obj_lang("food2")

        if food1_lang == food2_lang:
            food_text = f"{food1_lang}s"
        else:
            food_text = f"{food1_lang} and {food2_lang}"

        ep_meta["lang"] = (
            f"Retrieve the tupperware and aluminum foil from the cabinet "
            f"and place them on the dining counter near the plate with {food_text} "
            f"to prepare for storing leftovers."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["aluminum_foil_x_pos"] = self.aluminum_foil_x_pos
        ep_meta["refs"]["tupperware_x_pos"] = self.tupperware_x_pos
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="tupperware",
                obj_groups="tupperware",
                object_scale=[1.3, 1.3, 1.2],
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.30, 0.30),
                    pos=(self.tupperware_x_pos, -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        cfgs.append(
            dict(
                name="aluminum_foil",
                obj_groups="aluminum_foil",
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.50, 0.15),
                    pos=(self.aluminum_foil_x_pos, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.40, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food1",
                obj_groups=("meat", "cooked_food", "vegetable"),
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food2",
                obj_groups=("meat", "cooked_food", "vegetable"),
                graspable=True,
                placement=dict(
                    object="plate",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        tupperware_on_dining = OU.check_obj_fixture_contact(
            self, "tupperware", self.dining_counter
        )
        foil_on_dining = OU.check_obj_fixture_contact(
            self, "aluminum_foil", self.dining_counter
        )

        dist_threshold = 0.3
        tupperware_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["tupperware"]]
        )
        foil_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["aluminum_foil"]])
        plate_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["plate"]])

        tupperware_close_to_plate = (
            np.linalg.norm(tupperware_pos - plate_pos) < dist_threshold
        )
        foil_close_to_plate = np.linalg.norm(foil_pos - plate_pos) < dist_threshold

        gripper_far = all(
            OU.gripper_obj_far(self, obj) for obj in ["tupperware", "aluminum_foil"]
        )

        return (
            tupperware_on_dining
            and foil_on_dining
            and tupperware_close_to_plate
            and foil_close_to_plate
            and gripper_far
        )
