from robocasa.environments.kitchen.kitchen import *


class PlaceEqualIceCubes(Kitchen):
    """
    Place Equal Ice Cubes: composite task for Adding Ice to Beverages activity.

    Simulates placing 4 ice cubes into 2 glass cups, with 2 ice cubes in each cup.

    Steps:
        1. Pick up an ice cube from the bowl.
        2. Place it in one of the glass cups.
        3. Repeat until each glass cup has 2 ice cubes.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
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
        self.init_robot_base_ref = self.dining_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "There are four ice cubes in the ice bowl. "
            "Place two ice cubes in each glass of water."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "glass_cup1", [0.3, 0.4, 0.5, 0.25])
        OU.add_obj_liquid_site(self, "glass_cup2", [0.3, 0.4, 0.5, 0.25])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="ice_bowl",
                obj_groups="bowl",
                object_scale=[1.25, 1.25, 0.75],
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool1),
                    size=(1.0, 0.4),
                    pos=(0, -1.0),
                ),
            )
        )

        for i in range(1, 5):
            cfgs.append(
                dict(
                    name=f"ice_cube{i}",
                    obj_groups="ice_cube",
                    object_scale=0.9,
                    placement=dict(
                        object="ice_bowl",
                        size=(1.0, 1.0),
                    ),
                )
            )

        cfgs.append(
            dict(
                name="glass_cup1",
                obj_groups="glass_cup",
                object_scale=[1.3, 1.3, 1],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool1),
                    size=(0.4, 0.25),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="glass_cup2",
                obj_groups="glass_cup",
                object_scale=[1.3, 1.3, 1],
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool2),
                    size=(0.4, 0.25),
                    pos=("ref", "ref"),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        def cup_ok(cup_name):
            in_cup = [
                i
                for i in range(1, 5)
                if OU.check_obj_in_receptacle(self, f"ice_cube{i}", cup_name, th=0.5)
            ]
            if not in_cup:
                return False  # need at least one directly in the cup

            # any remaining cubes that are touching a cube that's in the cup
            touching = []
            for i in range(1, 5):
                if i in in_cup:
                    continue
                for j in in_cup:
                    if self.check_contact(
                        self.objects[f"ice_cube{i}"], self.objects[f"ice_cube{j}"]
                    ):
                        touching.append(i)
                        break

            total = len(in_cup) + len(touching)
            return total == 2

        cup1_ok = cup_ok("glass_cup1")
        cup2_ok = cup_ok("glass_cup2")
        gripper_far = all(
            OU.gripper_obj_far(self, f"ice_cube{i}", th=0.15) for i in range(1, 5)
        )

        return cup1_ok and cup2_ok and gripper_far
