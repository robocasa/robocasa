from robocasa.environments.kitchen.kitchen import *


class DivideBuffetTrays(Kitchen):
    """
    Divide Buffet Trays: composite task for Arranging Buffet.

    Simulates gathering vegetables and meat items from the fridge and placing them
    on separate trays on the dining counter for buffet organization.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        if "stool1" in self.fixture_refs:
            self.stool1 = self.fixture_refs["stool1"]
            self.stool2 = self.fixture_refs["stool2"]
        else:
            registered_stool_ids = set()
            self.stool1 = None
            self.stool2 = None

            while len(registered_stool_ids) < 2:
                for fixture in self.fixtures.values():
                    if isinstance(fixture, robocasa.models.fixtures.accessories.Stool):
                        fixture_id = id(fixture)
                        if fixture_id not in registered_stool_ids:
                            registered_stool_ids.add(fixture_id)
                            if self.stool1 is None:
                                self.stool1 = fixture
                            elif self.stool2 is None:
                                self.stool2 = fixture
                                break

            self.fixture_refs["stool1"] = self.stool1
            self.fixture_refs["stool2"] = self.stool2

        if "refs" in self._ep_meta:
            self.meat_rack_index = self._ep_meta["refs"]["meat_rack_index"]
            self.veg_rack_index = self._ep_meta["refs"]["veg_rack_index"]
        else:
            self.meat_rack_index = -1 if self.rng.random() < 0.5 else -2
            self.veg_rack_index = -2 if self.meat_rack_index == -1 else -1

        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool1)
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        veg1 = self.get_obj_lang("vegetable1")
        veg2 = self.get_obj_lang("vegetable2")
        meat1 = self.get_obj_lang("meat1")
        meat2 = self.get_obj_lang("meat2")

        if veg1 == veg2:
            veg_text = f"{veg1}s"
        else:
            veg_text = f"{veg1} and {veg2}"

        if meat1 == meat2:
            meat_text = f"{meat1}s"
        else:
            meat_text = f"{meat1} and {meat2}"

        ep_meta["lang"] = (
            f"Gather the {veg_text} from the fridge and place them on a tray on the dining counter. "
            f"Then gather the {meat_text} from the fridge and place them on the other tray."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["meat_rack_index"] = self.meat_rack_index
        ep_meta["refs"]["veg_rack_index"] = self.veg_rack_index
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.veg_rack_index,
                    ),
                    size=(0.50, 0.20),
                    pos=(-0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups="vegetable",
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.veg_rack_index,
                    ),
                    size=(0.50, 0.20),
                    pos=(0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat1",
                obj_groups="meat",
                exclude_obj_groups=("shrimp"),
                object_scale=[1, 1, 1.5],
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.meat_rack_index,
                    ),
                    size=(0.50, 0.20),
                    pos=(-0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat2",
                obj_groups="meat",
                exclude_obj_groups=("shrimp"),
                object_scale=[1, 1, 1.5],
                graspable=True,
                fridgable=True,
                placement=dict(
                    fixture=self.fridge,
                    sample_region_kwargs=dict(
                        rack_index=self.meat_rack_index,
                    ),
                    size=(0.5, 0.20),
                    pos=(0.25, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tray1",
                obj_groups=("tongs"),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.60, 0.40),
                    pos=("ref", "ref"),
                    rotation=(np.pi / 2, np.pi / 2),
                    try_to_place_in="tray",
                ),
            )
        )

        cfgs.append(
            dict(
                name="tray2",
                obj_groups=("tongs"),
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.60, 0.40),
                    pos=("ref", "ref"),
                    rotation=(np.pi / 2, np.pi / 2),
                    try_to_place_in="tray",
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_cup",
                obj_groups="cup",
                placement=dict(
                    fixture=self.dining_counter,
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor_cup_2",
                obj_groups="cup",
                placement=dict(
                    fixture=self.dining_counter,
                    reuse_region_from="distractor_cup",
                    size=(1.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        tray_names = ["tray1_container", "tray2_container"]

        meat_tray = None
        veg_tray = None

        for tray in tray_names:
            has_meats = all(
                OU.check_obj_in_receptacle(self, meat, tray)
                for meat in ["meat1", "meat2"]
            )
            has_vegs = all(
                OU.check_obj_in_receptacle(self, veg, tray)
                for veg in ["vegetable1", "vegetable2"]
            )
            if has_meats:
                meat_tray = tray
            if has_vegs:
                veg_tray = tray

        if meat_tray is None or veg_tray is None or meat_tray == veg_tray:
            return False

        meat_tray_vegs = any(
            OU.check_obj_in_receptacle(self, veg, meat_tray)
            for veg in ["vegetable1", "vegetable2"]
        )
        veg_tray_meats = any(
            OU.check_obj_in_receptacle(self, meat, veg_tray)
            for meat in ["meat1", "meat2"]
        )

        gripper_far = (
            OU.gripper_obj_far(self, obj_name="vegetable1")
            and OU.gripper_obj_far(self, obj_name="vegetable2")
            and OU.gripper_obj_far(self, obj_name="meat1")
            and OU.gripper_obj_far(self, obj_name="meat2")
            and OU.gripper_obj_far(self, obj_name="tray1_container")
            and OU.gripper_obj_far(self, obj_name="tray2_container")
        )

        return not meat_tray_vegs and not veg_tray_meats and gripper_far
