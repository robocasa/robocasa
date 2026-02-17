from robocasa.environments.kitchen.kitchen import *


class OrganizeCondiments(Kitchen):
    """
    Organize Condiments: composite task for Arranging Condiments activity.

    Task: Move the 3 condiments from the counter into the cabinet, leaving the distractor on the counter.
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
        ep_meta[
            "lang"
        ] = "There are various items on the counter. Only move the condiments/shakers into the cabinet."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cab.open_door(self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="condiment1",
                obj_groups=("condiment"),
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(1.0, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment2",
                obj_groups=("shaker", "salt_and_pepper_shaker"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="condiment1",
                    size=(1.0, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment3",
                obj_groups=("condiment", "shaker", "salt_and_pepper_shaker"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="condiment1",
                    size=(1.0, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distractor",
                exclude_obj_groups=("condiment", "shaker", "salt_and_pepper_shaker"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1.0, 0.5),
                    reuse_region_from="condiment1",
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        condiment1_in_cab = OU.obj_inside_of(self, "condiment1", self.cab)
        condiment2_in_cab = OU.obj_inside_of(self, "condiment2", self.cab)
        condiment3_in_cab = OU.obj_inside_of(self, "condiment3", self.cab)

        distractor_on_counter = OU.check_obj_fixture_contact(
            self, "distractor", self.counter
        )

        gripper_far = (
            OU.gripper_obj_far(self, "condiment1")
            and OU.gripper_obj_far(self, "condiment2")
            and OU.gripper_obj_far(self, "condiment3")
            and OU.gripper_obj_far(self, "distractor")
        )

        return (
            condiment1_in_cab
            and condiment2_in_cab
            and condiment3_in_cab
            and distractor_on_counter
            and gripper_far
        )
