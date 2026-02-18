from robocasa.environments.kitchen.kitchen import *


class CategorizeCondiments(Kitchen):
    """
    Categorizes Condiments: composite task for Arranging Condiments activity.

    Task: Put the shaker from the counter next to the shaker in the cabinet
    and the condiment bottle from the counter next to the condiment bottle in the cabinet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = "Put the shaker and condiment bottle from the counter next to their counterparts in the cabinet."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj1",
                obj_groups=["condiment_bottle", "ketchup", "syrup_bottle"],
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(0.50, 0.2),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="shaker",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(0.50, 0.2),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cab_obj1",
                obj_groups=["condiment_bottle", "ketchup", "syrup_bottle"],
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cab_obj2",
                obj_groups="shaker",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.2),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter1",
                obj_groups="all",
                exclude_obj_groups=["condiment"],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _xy_dist(self, obj_name1, obj_name2):
        obj1_pos = self.sim.data.body_xpos[
            self.obj_body_id[self.objects[obj_name1].name]
        ][:2]
        obj2_pos = self.sim.data.body_xpos[
            self.obj_body_id[self.objects[obj_name2].name]
        ][:2]
        return np.linalg.norm(obj1_pos - obj2_pos)

    def _check_success(self):
        obj1_in_cab = OU.obj_inside_of(self, "obj1", self.cab)
        obj2_in_cab = OU.obj_inside_of(self, "obj2", self.cab)

        bottles_close = self._xy_dist("obj1", "cab_obj1") <= 0.15
        shakers_close = self._xy_dist("obj2", "cab_obj2") <= 0.15

        gripper_far = OU.gripper_obj_far(self, "obj1") and OU.gripper_obj_far(
            self, "obj2"
        )

        return (
            obj1_in_cab
            and obj2_in_cab
            and bottles_close
            and shakers_close
            and gripper_far
        )
