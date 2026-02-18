from robocasa.environments.kitchen.kitchen import *


class SetupWineGlasses(Kitchen):
    """
    Set Up Wine Glasses: composite task for Setting The Table activity.

    Simulates the task of placing a wine glass next to each plate on the dining counter.

    Steps:
        1) Pick up the wine glasses from the cabinet by the sink.
        2) Place a wine glass on the right or left side of each plate.
    """

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self, obj_registries=("objaverse", "lightwheel", "aigen"), *args, **kwargs
    ):
        super().__init__(obj_registries=obj_registries, *args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
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
            "Retrieve the wine glasses from the counter and "
            "place one next to each plate."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()

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
                    size=(0.30, 0.30),
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
                    size=(0.30, 0.30),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"wine_glass1",
                obj_groups="wine_glass",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(0.5, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"wine_glass2",
                obj_groups="wine_glass",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from=f"wine_glass1",
                    size=(0.5, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        plate_names = [f"plate{i+1}" for i in range(2)]
        wine_glass_names = [f"wine_glass{i+1}" for i in range(2)]

        left_right_threshold = 0.25
        front_back_threshold = 0.10

        assigned_plates = set()
        success = True

        # For each wine glass, compare it against the available plates.
        for wine_glass_name in wine_glass_names:
            wine_glass_obj = self.objects[wine_glass_name]
            best_candidate = None
            best_score = float("inf")
            for idx, plate_name in enumerate(plate_names):
                if idx in assigned_plates:
                    continue
                plate_obj = self.objects[plate_name]
                plate_pos_global = np.array(
                    self.sim.data.body_xpos[self.obj_body_id[plate_obj.name]]
                )
                glass_pos_global = np.array(
                    self.sim.data.body_xpos[self.obj_body_id[wine_glass_obj.name]]
                )

                plate_x, plate_y = OU.transform_global_to_local(
                    plate_pos_global[0],
                    plate_pos_global[1],
                    -self.stool1.rot + np.pi,
                )
                wine_glass_x, wine_glass_y = OU.transform_global_to_local(
                    glass_pos_global[0],
                    glass_pos_global[1],
                    -self.stool1.rot + np.pi,
                )

                x_dist = abs(wine_glass_x - plate_x)
                y_dist = abs(wine_glass_y - plate_y)
                score = x_dist + y_dist

                if (
                    x_dist < left_right_threshold
                    and y_dist < front_back_threshold
                    and score < best_score
                ):
                    best_candidate = idx
                    best_score = score

            # Check that the wine glass is in contact with the dining counter.
            wine_glass_on_table = OU.check_obj_fixture_contact(
                self, wine_glass_name, self.dining_counter
            )
            if best_candidate is not None and wine_glass_on_table:
                assigned_plates.add(best_candidate)
            else:
                success = False

        all_glasses_far = all(
            OU.gripper_obj_far(self, obj_name=f"wine_glass{i+1}") for i in range(2)
        )

        return success and all_glasses_far
