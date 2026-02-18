from robocasa.environments.kitchen.kitchen import *


class CleanBoard(Kitchen):
    """
    Clean Board: composite task for "Slicing Meat" activity.

    Simulates the act of cleaning the cutting board after cutting meat.


    Steps:
        Pick the knife and place it in the sink.
        Place the cutting board in the sink.
        Place the spray from the cabinet right above the counter on the counter.

    """

    # I found that sliding the cutting board off the counter slightly to pick it up to be easiest.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.fridge)
        )
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET, ref=self.counter)
        )
        self.sink = self.register_fixture_ref(
            "sink", dict(id=FixtureType.SINK, ref=self.counter)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta[
            "lang"
        ] = f"Move the cutting board and knife to the sink. Then move spray from the {self.cabinet.nat_lang} to the counter."

        return ep_meta

    def _setup_scene(self):
        """
        Reset the environment internal state for the door tasks.
        This includes setting the door state based on the behavior.
        """
        self.cabinet.open_door(env=self)
        super()._setup_scene()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="spray",
                obj_groups="spray",
                graspable=True,
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.3, 0.3),
                    pos=(-0.6, -0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="cutting_board",
                graspable=False,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.5),
                    pos=(0.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="knife",
                obj_groups="knife",
                object_scale=1.5,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=(0.0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):

        knife_in_sink = OU.obj_inside_of(self, "knife", self.sink)
        board_in_sink = OU.obj_inside_of(self, "receptacle", self.sink)

        spray_on_counter = OU.check_obj_fixture_contact(self, "spray", self.counter)
        gripper_far = (
            OU.gripper_obj_far(self, "knife")
            and OU.gripper_obj_far(self, "receptacle")
            and OU.gripper_obj_far(self, "spray")
        )

        return knife_in_sink and board_in_sink and spray_on_counter and gripper_far
