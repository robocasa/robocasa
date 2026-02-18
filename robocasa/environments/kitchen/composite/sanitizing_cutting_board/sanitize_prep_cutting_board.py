from robocasa.environments.kitchen.kitchen import *


class SanitizePrepCuttingBoard(Kitchen):
    """
    Sanitize Prep Cutting Board: composite task for Sanitizing Cutting Board activity.

    Simulates the task of preparing cleaning supplies for sanitizing a cutting board.

    Steps:
        1. Grab the spray bottle from the cabinet and place it next to the cutting board.
        2. Grab the dish brush from next to the sink and place it next to the cutting board.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cabinet = self.register_fixture_ref(
            "cabinet", dict(id=FixtureType.CABINET)
        )
        self.counter_cab = self.register_fixture_ref(
            "counter_cab", dict(id=FixtureType.COUNTER, ref=self.cabinet)
        )
        self.counter_sink = self.register_fixture_ref(
            "counter_sink", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter_cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Grab the spray bottle from the {self.cabinet.nat_lang}, then place it next to the cutting board for sanitizing. "
            "Then grab the dish brush from the countertop next to the sink and place it next to the cutting board."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.cabinet.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter_cab,
                    sample_region_kwargs=dict(
                        ref=self.cabinet,
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="spray",
                obj_groups="spray",
                placement=dict(
                    fixture=self.cabinet,
                    size=(0.3, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dish_brush",
                obj_groups="dish_brush",
                object_scale=[1, 1, 1.2],
                placement=dict(
                    fixture=self.counter_sink,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.3, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        cutting_board_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["cutting_board"]]
        )
        spray_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["spray"]])
        dish_brush_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["dish_brush"]]
        )

        spray_distance = np.linalg.norm(spray_pos[:2] - cutting_board_pos[:2])
        brush_distance = np.linalg.norm(dish_brush_pos[:2] - cutting_board_pos[:2])

        spray_near_board = spray_distance <= 0.35
        brush_near_board = brush_distance <= 0.35

        spray_on_counter = OU.check_obj_any_counter_contact(self, "spray")
        brush_on_counter = OU.check_obj_any_counter_contact(self, "dish_brush")
        cutting_board_on_counter = OU.check_obj_any_counter_contact(
            self, "cutting_board"
        )

        gripper_far_spray = OU.gripper_obj_far(self, "spray")
        gripper_far_brush = OU.gripper_obj_far(self, "dish_brush")

        return (
            spray_near_board
            and brush_near_board
            and cutting_board_on_counter
            and gripper_far_spray
            and gripper_far_brush
            and spray_on_counter
            and brush_on_counter
        )
