from robocasa.environments.kitchen.kitchen import *


class RinseCuttingBoard(Kitchen):
    """
    Rinse Cutting Board: composite task for Sanitizing Cutting Board activity.

    Simulates the task of rinsing a cutting board under hot water.

    Steps:
        Turn on the sink and ensure the water is running hot.
        After 100 timesteps of rinsing, turn off the sink.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Turn on the sink faucet with hot water and rinse the cutting board for ten seconds, then turn off the faucet. "
            "Push the cutting board to reach the water if necessary."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.rinsing_time = 0

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                object_scale=[1, 1, 1.5],
                placement=dict(
                    fixture=self.sink,
                    size=(1.0, 0.7),
                    pos=(0, 1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        handle_state = self.sink.get_handle_state(self)
        water_on = handle_state["water_on"]
        water_temp = handle_state["water_temp"]

        water_hot = water_temp >= 0.6

        # Check if cutting board is under water
        board_under_water = self.sink.check_obj_under_water(self, "cutting_board")

        if water_on and water_hot and board_under_water:
            self.rinsing_time += 1

        if self.rinsing_time >= 100:
            can_turn_off = True
        else:
            can_turn_off = False

        return not water_on and can_turn_off
