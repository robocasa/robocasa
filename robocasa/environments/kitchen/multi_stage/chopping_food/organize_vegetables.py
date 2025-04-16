from robocasa.environments.kitchen.kitchen import *


class OrganizeVegetables(Kitchen):
    """
    Organize Vegetables: composite task for Chopping Food activity.

    Simulates the task of organizing vegetables on cutting boards.

    Steps:
        Place the vegetables on separate cutting boards
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(1.0, 0.4))
        )

        # self.counter = self.get_fixture(FixtureType.DINING_COUNTER, ref=self.sink)
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_name_1 = self.get_obj_lang("vegetable1")
        obj_name_2 = self.get_obj_lang("vegetable2")

        ep_meta[
            "lang"
        ] = f"Place the {obj_name_1} and the {obj_name_2} on separate cutting boards."

        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board1",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(0.05, 0.05),
                    rotation=np.pi / 2,
                    pos=(-0.6, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="cutting_board2",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(0.05, 0.05),
                    rotation=0,
                    pos=(0.5, -0.4),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=["vegetable", "fruit"],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(0.40, 0.40),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups=["vegetable", "fruit"],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(0.40, 0.40),
                    pos=(0, -0.5),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Make sure vegetables are on different cutting boards
        """
        vegetable1_cutting_board_contact1 = OU.check_obj_in_receptacle(
            self, "vegetable1", "cutting_board1"
        )
        vegetable2_cutting_board_contact1 = OU.check_obj_in_receptacle(
            self, "vegetable2", "cutting_board1"
        )
        vegetable1_cutting_board_contact2 = OU.check_obj_in_receptacle(
            self, "vegetable1", "cutting_board2"
        )
        vegetable2_cutting_board_contact2 = OU.check_obj_in_receptacle(
            self, "vegetable2", "cutting_board2"
        )

        return (
            vegetable1_cutting_board_contact1 and vegetable2_cutting_board_contact2
        ) or (vegetable2_cutting_board_contact1 and vegetable1_cutting_board_contact2)
