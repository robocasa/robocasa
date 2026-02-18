from robocasa.environments.kitchen.kitchen import *


class ServeMealJuice(Kitchen):
    """
    Serve Meal Juice: composite task for Serving Beverages activity.

    Simulates the process of serving orange juice glasses next to plates of food
    on the dining counter for a meal.

    Steps:
        1. Pick up orange juice glasses from the counter.
        2. Place one orange juice glass next to each plate on the dining counter.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cabinet = self.register_fixture_ref("sink", dict(id=FixtureType.CABINET))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )

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

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the orange juice cups from the counter and "
            "place one next to each plate on the dining counter."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "cup1", [1.0, 0.5, 0.0, 0.8])
        OU.add_obj_liquid_site(self, "cup2", [1.0, 0.5, 0.0, 0.8])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate1",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.35, 0.3),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate2",
                obj_groups="plate",
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.35, 0.3),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food_plate1",
                obj_groups=("cooked_food", "chicken_drumstick"),
                exclude_obj_groups=("pizza"),
                placement=dict(
                    object="plate1",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food_plate2",
                obj_groups=("cooked_food", "chicken_drumstick"),
                exclude_obj_groups=("pizza"),
                placement=dict(
                    object="plate2",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cup1",
                obj_groups="cup",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(0.5, 0.2),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cup2",
                obj_groups="cup",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cup1",
                    size=(0.5, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        plate_names = ["plate1", "plate2"]
        juice_names = ["cup1", "cup2"]

        left_right_threshold = 0.25
        front_back_threshold = 0.15

        assigned_plates = set()
        success = True

        for juice_name in juice_names:
            juice_obj = self.objects[juice_name]
            best_candidate = None
            best_score = float("inf")

            for idx, plate_name in enumerate(plate_names):
                if idx in assigned_plates:
                    continue

                plate_obj = self.objects[plate_name]
                plate_pos_global = np.array(
                    self.sim.data.body_xpos[self.obj_body_id[plate_obj.name]]
                )
                juice_pos_global = np.array(
                    self.sim.data.body_xpos[self.obj_body_id[juice_obj.name]]
                )

                stool_rot = (
                    -self.stool1.rot + np.pi if idx == 0 else -self.stool2.rot + np.pi
                )

                plate_x, plate_y = OU.transform_global_to_local(
                    plate_pos_global[0],
                    plate_pos_global[1],
                    stool_rot,
                )
                juice_x, juice_y = OU.transform_global_to_local(
                    juice_pos_global[0],
                    juice_pos_global[1],
                    stool_rot,
                )

                x_dist = abs(juice_x - plate_x)
                y_dist = abs(juice_y - plate_y)
                score = x_dist + y_dist

                if (
                    x_dist < left_right_threshold
                    and y_dist < front_back_threshold
                    and score < best_score
                ):
                    best_candidate = idx
                    best_score = score

            juice_on_table = OU.check_obj_fixture_contact(
                self, juice_name, self.dining_counter
            )
            if best_candidate is not None and juice_on_table:
                assigned_plates.add(best_candidate)
            else:
                success = False

        all_juice_far = all(
            OU.gripper_obj_far(self, obj_name=f"cup{i+1}") for i in range(2)
        )

        return success and all_juice_far
