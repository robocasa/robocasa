from robocasa.environments.kitchen.kitchen import *


class GatherTableware(Kitchen):
    """
    Gather Glasses: composite task for Arranging Cabinets activity.

    Simulates the task of sorting glasses and bowls in the cabinets.

    Steps:
        Gather all objects into one cabinet and sort the mugs and bowl to opposite sides.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        if "cab1" in self.fixture_refs:
            self.cab = self.fixture_refs["cab1"]
            self.cab2 = self.fixture_refs["cab2"]
        else:
            while True:
                self.cab = self.get_fixture(FixtureType.CABINET)

                valid_cab_config_found = False
                for _ in range(100):  # 20 attempts
                    # sample until at least 2 different cabinets are selected
                    self.cab2 = self.get_fixture(FixtureType.CABINET)
                    if (
                        self.cab2 != self.cab
                    ):  # We only check for 2 different cabinets as there might only be two cabinets
                        valid_cab_config_found = True
                        break

                if valid_cab_config_found:
                    break

            self.fixture_refs["cab1"] = self.cab
            self.fixture_refs["cab2"] = self.cab2

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = "Gather all objects into one cabinet and sort the glasses and bowls to opposite sides."
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(self)
        self.cab2.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="glass1",
                graspable=True,
                obj_groups=["mug"],
                placement=dict(
                    fixture=self.cab,
                    size=(0.8, 0.8),
                    pos=(0.7, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="glass2",
                graspable=True,
                obj_groups=["mug"],
                placement=dict(
                    fixture=self.cab,
                    size=(0.60, 0.60),
                    pos=(-0.7, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="glass3",
                obj_groups=["mug"],
                graspable=True,
                placement=dict(
                    fixture=self.cab2,
                    size=(0.40, 0.40),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.80, 0.80),
                    pos=(0, 0),
                ),
            )
        )

        return cfgs

    def dist_between_obj(self, obj_name1, obj_name2):
        obj1 = self.objects[obj_name1]
        obj2 = self.objects[obj_name2]

        obj1_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj1.name]])
        obj2_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj2.name]])
        return np.linalg.norm(obj1_pos - obj2_pos)

    def _check_success(self):

        # must make sure the cleaner is on the counter and close to the sink
        glass1_glass2 = self.dist_between_obj("glass1", "glass2")
        glass2_glass3 = self.dist_between_obj("glass2", "glass3")
        glass3_glass1 = self.dist_between_obj("glass3", "glass1")

        bowl1_glass1 = self.dist_between_obj("bowl", "glass1")
        bowl1_glass2 = self.dist_between_obj("bowl", "glass2")
        bowl1_glass3 = self.dist_between_obj("bowl", "glass3")

        max_glass = max(glass1_glass2, glass2_glass3, glass3_glass1)
        max_bowl = max(bowl1_glass1, bowl1_glass2, bowl1_glass3)

        far_from_obj1 = OU.gripper_obj_far(self, obj_name="glass1")
        far_from_obj2 = OU.gripper_obj_far(self, obj_name="glass2")
        far_from_obj3 = OU.gripper_obj_far(self, obj_name="glass3")

        return (
            far_from_obj1
            and far_from_obj2
            and far_from_obj3
            and max_glass * 1.5 < max_bowl
        )
