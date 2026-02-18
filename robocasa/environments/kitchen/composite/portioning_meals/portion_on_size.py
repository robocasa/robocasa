from robocasa.environments.kitchen.kitchen import *


class PortionOnSize(Kitchen):
    """
    Portion On Size: composite task for Portioning Meals activity.

    Simulates the task of placing bowls next to plates based on the meal size.
    The task involves placing a small bowl next to a small plate with appetizer,
    and a large bowl next to a large plate with a full meal (meat and cooked food).

    Steps:
        1. Take bowls from the cabinet
        2. Place small bowl next to small plate with appetizer
        3. Place large bowl next to large plate with meal
    """

    EXCLUDE_LAYOUTS = (
        Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS
        + Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS
        + [22, 58]
    )

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

        if "refs" in self._ep_meta:
            self.bowl_positions = self._ep_meta["refs"]["bowl_positions"]
        else:
            available_positions = [-1.0, 1.0]
            self.rng.shuffle(available_positions)
            self.bowl_positions = {
                "bowl_small": available_positions[0],
                "bowl_large": available_positions[1],
            }

        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )

        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool1)
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["bowl_positions"] = self.bowl_positions

        ep_meta["lang"] = (
            "Take the smaller bowl and place it next to the appetizer plate on the dining counter, "
            "and place the larger bowl next to the plate with cooked food."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="plate_small",
                obj_groups="plate",
                object_scale=0.75,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool1,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate_large",
                obj_groups="plate",
                object_scale=1.25,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(
                        ref=self.stool2,
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", "ref"),
                ),
            )
        )

        cfgs.append(
            dict(
                name="appetizer",
                obj_groups=("cheese", "fruit", "bread", "croissant"),
                placement=dict(
                    object="plate_small",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cooked_food",
                obj_groups="cooked_food",
                exclude_obj_groups=("kebab_skewer", "pizza"),
                placement=dict(
                    object="plate_large",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    object="plate_large",
                    size=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl_small",
                obj_groups="bowl",
                object_scale=0.75,
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.35, 0.4),
                    pos=(self.bowl_positions["bowl_small"], -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl_large",
                obj_groups="bowl",
                object_scale=1.1,
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.6, 0.4),
                    pos=(self.bowl_positions["bowl_large"], -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        lateral_threshold_small = 0.3
        forward_threshold_small = 0.05

        lateral_threshold_large = 0.4
        forward_threshold_large = 0.1

        small_bowl_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["bowl_small"]]
        )
        large_bowl_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["bowl_large"]]
        )
        small_plate_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["plate_small"]]
        )
        large_plate_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["plate_large"]]
        )

        small_bowl_x, small_bowl_y = OU.transform_global_to_local(
            small_bowl_pos[0], small_bowl_pos[1], -self.stool1.rot + np.pi
        )
        small_plate_x, small_plate_y = OU.transform_global_to_local(
            small_plate_pos[0], small_plate_pos[1], -self.stool1.rot + np.pi
        )

        small_x_dist = abs(small_bowl_x - small_plate_x)
        small_y_dist = abs(small_bowl_y - small_plate_y)
        small_bowl_near_small_plate = (
            small_x_dist < lateral_threshold_small
            and small_y_dist < forward_threshold_small
        )

        large_bowl_x, large_bowl_y = OU.transform_global_to_local(
            large_bowl_pos[0], large_bowl_pos[1], -self.stool2.rot + np.pi
        )
        large_plate_x, large_plate_y = OU.transform_global_to_local(
            large_plate_pos[0], large_plate_pos[1], -self.stool2.rot + np.pi
        )

        small_bowl_on_counter = OU.check_obj_any_counter_contact(self, "bowl_small")
        large_bowl_on_counter = OU.check_obj_any_counter_contact(self, "bowl_large")

        large_x_dist = abs(large_bowl_x - large_plate_x)
        large_y_dist = abs(large_bowl_y - large_plate_y)
        large_bowl_near_large_plate = (
            large_x_dist < lateral_threshold_large
            and large_y_dist < forward_threshold_large
        )

        gripper_far = OU.gripper_obj_far(self, "bowl_small") and OU.gripper_obj_far(
            self, "bowl_large"
        )

        return (
            small_bowl_near_small_plate
            and large_bowl_near_large_plate
            and gripper_far
            and small_bowl_on_counter
            and large_bowl_on_counter
        )
