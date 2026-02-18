from robocasa.environments.kitchen.kitchen import *


class BreadSelection(Kitchen):
    """
    Bread Selection: composite task for Making Toast activity.

    Simulates the task of setting up ingredients for making a bread snack.

    Steps:
        Place a croissant and a jar of jam on the cutting board.
    """

    EXCLUDE_LAYOUTS = Kitchen.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.9, 0.6))
        )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta["lang"] = (
                "From the different types of pastries on the counter, select a croissant and place it on the cutting board. "
                "Then retrieve a jar of jam from the cabinet and place it alongside the croissant on the cutting board."
            )
        return ep_meta

    def _setup_scene(self):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.6),
                    pos=(0.0, -1.0),
                    rotation=[np.pi / 2, np.pi / 2],
                ),
            )
        )

        # Three types of pastries
        cfgs.append(
            dict(
                name="distr_pastry",
                obj_groups=("baguette", "cupcake"),
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(1.0, 0.6),
                    pos=(0.0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        cfgs.append(
            dict(
                name="croissant",
                obj_groups="croissant",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(1.0, 0.6),
                    pos=(0.0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        # Jar of jam in the cabinet
        cfgs.append(
            dict(
                name="jam",
                obj_groups="jam",
                placement=dict(fixture=self.cab, size=(1.0, 0.20), pos=(0, -1.0)),
            )
        )

        # Additional distractor on the counter
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(1.0, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        # Check that both the croissant and jam are on the cutting board
        return (
            OU.check_obj_in_receptacle(self, "croissant", "cutting_board")
            and OU.gripper_obj_far(self, obj_name="croissant")
            and OU.check_obj_in_receptacle(self, "jam", "cutting_board")
        )
