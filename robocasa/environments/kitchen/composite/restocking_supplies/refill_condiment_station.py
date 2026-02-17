from robocasa.environments.kitchen.kitchen import *
from robocasa.models.objects.kitchen_objects import get_cats_by_type, OBJ_CATEGORIES


class RefillCondimentStation(Kitchen):
    """
    Refill Condiment Station: composite task for Restocking Supplies activity.
    Simulates the task of refilling the condiment station on the dining table.
    Steps:
        1. Open the cabinet
        2. Take the condiment from the cabinet
        3. Place the condiment on the dining table
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_WITH_DOOR)
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_table = self.register_fixture_ref(
            "dining_table", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.init_robot_base_ref = self.cab
        if "refs" in self._ep_meta:
            self.condiment_cat = self._ep_meta["refs"]["condiment_cat"]
        else:
            condiment_cats = get_cats_by_type(
                types=["condiment"], obj_registries=self.obj_registries
            )
            graspable_condiment_cats = []
            for cat in condiment_cats:
                graspable = False
                for reg in self.obj_registries:
                    if (
                        OBJ_CATEGORIES[cat].get(reg) is not None
                        and OBJ_CATEGORIES[cat][reg].graspable
                    ):
                        graspable = True
                        break
                if graspable:
                    graspable_condiment_cats.append(cat)
            self.condiment_cat = self.rng.choice(graspable_condiment_cats)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        condiment_name = self.get_obj_lang("condiment")
        ep_meta[
            "lang"
        ] = f"The dining table condiment station is running low on {condiment_name}. Refill the condiment station by opening the cabinet and taking the {condiment_name} from the cabinet and placing it on the dining table."
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["condiment_cat"] = self.condiment_cat
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="condiment",
                obj_groups=self.condiment_cat,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups="condiment",
                placement=dict(
                    fixture=self.cab,
                    size=(0.60, 0.30),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dining_table_condiment",
                obj_groups=self.condiment_cat,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.6, 0.40),
                    pos=("ref", "ref"),
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, obj_name="condiment")
        condiment_on_dining_table = OU.check_obj_fixture_contact(
            self, "condiment", self.dining_table
        )
        return gripper_obj_far and condiment_on_dining_table
