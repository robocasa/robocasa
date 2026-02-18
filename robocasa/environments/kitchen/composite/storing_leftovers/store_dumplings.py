from robocasa.environments.kitchen.kitchen import *


class StoreDumplings(Kitchen):
    """
    Store Dumplings: composite task for Storing Leftovers activity.

    Simulates the process of storing dumplings from both a pan and a plate into containers on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        kwargs["obj_registries"] = ["aigen", "objaverse", "lightwheel"]
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

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

        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool1)
        )

        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))

        plate_choices = ["dumpling_plate1", "dumpling_plate2"]
        if "plate_for_3" in self._ep_meta:
            self.plate_for_3 = self._ep_meta["plate_for_3"]
            self.plate_for_4 = self._ep_meta["plate_for_4"]
        else:
            self.plate_for_3 = self.rng.choice(plate_choices)
            self.plate_for_4 = self.rng.choice(plate_choices)

        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["plate_for_3"] = self.plate_for_3
        ep_meta["plate_for_4"] = self.plate_for_4

        ep_meta["lang"] = (
            f"Place two dumplings into each of the tupperware containers "
            f"and then place the containers in the fridge."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="dumpling_plate1",
                obj_groups="plate",
                object_scale=1.25,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dumpling_plate2",
                obj_groups="plate",
                object_scale=1.25,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dumpling1",
                obj_groups="dumpling",
                placement=dict(
                    object="dumpling_plate1",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dumpling2",
                obj_groups="dumpling",
                placement=dict(
                    object="dumpling_plate2",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dumpling3",
                obj_groups="dumpling",
                placement=dict(
                    object=str(self.plate_for_3),
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dumpling4",
                obj_groups="dumpling",
                placement=dict(
                    object=str(self.plate_for_4),
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tupperware1",
                obj_groups="tupperware",
                object_scale=[1.5, 1.5, 1],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="tupperware2",
                obj_groups="tupperware",
                object_scale=[1.5, 1.5, 1],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
                    rotation=0,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        dumplings = ["dumpling1", "dumpling2", "dumpling3", "dumpling4"]

        count_in_t1 = sum(
            OU.check_obj_in_receptacle(self, d, "tupperware1") for d in dumplings
        )
        count_in_t2 = sum(
            OU.check_obj_in_receptacle(self, d, "tupperware2") for d in dumplings
        )
        # each container must have exactly 2 dumplings
        containers_ok = (count_in_t1 == 2) and (count_in_t2 == 2)

        t1_in_fridge = self.fridge.check_rack_contact(self, "tupperware1")
        t2_in_fridge = self.fridge.check_rack_contact(self, "tupperware2")
        fridge_ok = t1_in_fridge and t2_in_fridge

        all_objects = dumplings + ["tupperware1", "tupperware2"]
        gripper_far = all(OU.gripper_obj_far(self, obj_name=obj) for obj in all_objects)

        return containers_ok and fridge_ok and gripper_far
