from robocasa.environments.kitchen.kitchen import *


class SetupBowls(Kitchen):
    """
    Place Bowl in Front of Each Stool: composite task for Setting The Table activity.

    Simulates the task of placing a bowl in front of each stool.

    Steps:
        1) Pick up the bowls from the cabinet.
        2) Place a bowl in front of each stool.
    """

    EXCLUDE_LAYOUTS = (
        Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS
        + Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS
        + [35]
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )

        if "refs" in self._ep_meta:
            self.stools = []
            self.stool_positions = []
            self.stool_rotations = []
            stool_index = 1
            while f"stool{stool_index}" in self._ep_meta["refs"]:
                stool_name = self._ep_meta["refs"][f"stool{stool_index}"]
                stool_rot = self._ep_meta["refs"][f"stool{stool_index}_rot"]
                stool_pos = np.array(self._ep_meta["refs"][f"stool{stool_index}_pos"])

                if stool_name in self.fixtures:
                    stool_ref = self.fixtures[stool_name]
                    self.stools.append(stool_ref)
                    self.stool_positions.append(stool_pos)
                    self.stool_rotations.append(stool_rot)
                stool_index += 1
        else:
            self.num_stools = sum(
                1
                for fixture in self.fixtures.values()
                if isinstance(fixture, robocasa.models.fixtures.accessories.Stool)
            )
            self.stools = []
            self.stool_positions = []
            self.stool_rotations = []
            processed_fixture_ids = set()
            stool_index = 1

            unique_stool_fixtures = []
            for fixture in self.fixtures.values():
                if len(unique_stool_fixtures) == self.num_stools:
                    break

                if isinstance(fixture, robocasa.models.fixtures.accessories.Stool):
                    fixture_id = id(fixture)
                    if fixture_id not in processed_fixture_ids:
                        stool_pos = np.array(fixture.pos)

                        if not any(
                            np.allclose(stool_pos, existing_pos, atol=0.05)
                            for existing_pos in self.stool_positions
                        ):
                            unique_stool_fixtures.append(fixture)
                            self.stool_positions.append(stool_pos)
                            self.stool_rotations.append(fixture.rot)
                            processed_fixture_ids.add(fixture_id)

            for i, fixture in enumerate(unique_stool_fixtures):
                stool_name = f"stool{i+1}"
                self.fixture_refs[stool_name] = fixture
                self.stools.append(fixture)

        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick up the bowls from the cabinet and place each bowl in front of a stool on the dining counter."

        ep_meta["refs"] = ep_meta.get("refs", {})
        for i, stool in enumerate(self.stools):
            ep_meta["refs"][f"stool{i+1}"] = stool.name
            ep_meta["refs"][f"stool{i+1}_rot"] = stool.rot
            ep_meta["refs"][f"stool{i+1}_pos"] = list(stool.pos)

        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"bowl1",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.3),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"bowl2",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.cabinet,
                    size=(1.0, 0.3),
                    pos=(1.0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        lateral_threshold = 0.15
        forward_threshold = 0.3

        assigned_stools = set()
        success = True

        bowl_names = ["bowl1", "bowl2"]
        bowl_positions = {
            name: np.array(self.sim.data.body_xpos[self.obj_body_id[name]])
            for name in bowl_names
        }

        for bowl_name, bowl_pos in bowl_positions.items():
            closest_stool_idx = None
            min_x_dist = float("inf")
            min_y_dist = float("inf")
            for idx, stool_pos in enumerate(self.stool_positions):
                stool_rot = self.stool_rotations[idx]
                bowl_x, bowl_y = OU.transform_global_to_local(
                    bowl_pos[0],
                    bowl_pos[1],
                    -stool_rot + np.pi,
                )
                stool_x, stool_y = OU.transform_global_to_local(
                    stool_pos[0],
                    stool_pos[1],
                    -stool_rot + np.pi,
                )

                x_dist = abs(bowl_x - stool_x)
                y_dist = abs(bowl_y - stool_y)

                if (
                    idx not in assigned_stools
                    and x_dist < lateral_threshold
                    and y_dist < forward_threshold
                ):
                    if x_dist < min_x_dist and y_dist < min_y_dist:
                        closest_stool_idx = idx
                        min_x_dist = x_dist
                        min_y_dist = y_dist

            bowl_on_counter = OU.check_obj_any_counter_contact(self, bowl_name)

            if closest_stool_idx is not None and bowl_on_counter:
                assigned_stools.add(closest_stool_idx)
            else:
                success = False

        all_bowls_far = all(
            OU.gripper_obj_far(self, obj_name=name) for name in bowl_names
        )

        return success and all_bowls_far
