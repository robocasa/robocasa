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
            "counter",
            dict(id=FixtureType.COUNTER, size=(0.9, 0.4), full_depth_region=True),
        )

        # self.counter = self.get_fixture(FixtureType.DINING_COUNTER, ref=self.sink)
        self.init_robot_base_ref = self.counter

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
                    sample_region_kwargs=dict(
                        top_size=(1.0, 0.5),
                        full_depth_region=True,
                    ),
                    size=(1.0, 0.4),
                    rotation=np.pi / 2,
                    pos=(0.0, -1.0),
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
                    reuse_region_from="cutting_board1",
                    size=(1.0, 0.4),
                    rotation=0,
                    pos=(0.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=["vegetable", "fruit"],
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board1",
                    size=(0.40, 0.35),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups=["vegetable", "fruit"],
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board1",
                    size=(0.40, 0.35),
                    pos=(0, -1.0),
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
