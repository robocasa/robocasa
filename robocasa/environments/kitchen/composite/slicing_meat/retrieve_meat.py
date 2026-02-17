from robocasa.environments.kitchen.kitchen import *


class RetrieveMeat(Kitchen):
    """
    Retrieve Meat: composite task for Slicing Meat activity.
    Simulates the task of retrieving meat from the fridge and placing it on the cutting board.

    Steps:
        1. Pick the meat from the fridge.
        2. Place the meat on the cutting board.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )
        self.init_robot_base_ref = self.fridge

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_lang = self.get_obj_lang("meat")
        ep_meta[
            "lang"
        ] = f"Retrieve the {meat_lang} from the fridge and place it on the cutting board."

        return ep_meta

    def _setup_scene(self):
        """
        Reset the environment internal state for the door tasks.
        This includes setting the door state based on the behavior.
        """
        self.fridge.open_door(env=self)
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=FixtureType.FRIDGE,
                    size=(0.5, 0.25),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="dist_obj",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=FixtureType.FRIDGE,
                    size=(0.5, 0.25),
                    pos=(0, -1.0),
                    sample_region_kwargs=dict(),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.4),
                    pos=(-0.6, -0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                # graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.3, 0.3),
                    pos=(-0.6, -0.5),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        meat_on_board = OU.check_obj_in_receptacle(self, "meat", "receptacle")

        gripper_far = OU.gripper_obj_far(self, "meat")

        return meat_on_board and gripper_far
