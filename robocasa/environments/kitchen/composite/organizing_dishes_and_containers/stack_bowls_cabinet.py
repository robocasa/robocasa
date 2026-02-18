from robocasa.environments.kitchen.kitchen import *


class StackBowlsCabinet(Kitchen):
    """
    Stack Bowls Cabinet: composite task for Organizing Dishes and Containers.

    Simulates the task of picking up bowls from the dining counter, stacking them, and placing them in the cabinet.

    Steps:
        Pick up the first bowl from the dining counter.
        Pick up the second bowl and place it on top of the first bowl.
        Move the stacked bowls to the cabinet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.init_robot_base_ref = self.cabinet

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "Pick up the bowls on the counter and stack them on top of one another in the open cabinet. "
                "Place the smaller bowl on top of the larger bowl."
            )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bowl1",
                obj_groups="bowl",
                object_scale=0.7,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(1.0, 0.60),
                    pos=("ref", -0.8),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl2",
                obj_groups="bowl",
                object_scale=0.95,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(1.0, 0.60),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        bowl1_in_cabinet = OU.obj_inside_of(self, "bowl1", self.cabinet)
        bowl2_in_cabinet = OU.obj_inside_of(self, "bowl2", self.cabinet)

        bowl2_in_bowl1 = OU.check_obj_in_receptacle(self, "bowl2", "bowl1")
        bowl1_in_bowl2 = OU.check_obj_in_receptacle(self, "bowl1", "bowl2")

        gripper_bowl1_far = OU.gripper_obj_far(self, obj_name="bowl1")
        gripper_bowl2_far = OU.gripper_obj_far(self, obj_name="bowl2")

        return (
            bowl1_in_cabinet
            and bowl2_in_cabinet
            and (bowl2_in_bowl1 or bowl1_in_bowl2)
            and gripper_bowl1_far
            and gripper_bowl2_far
        )
