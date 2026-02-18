from robocasa.environments.kitchen.kitchen import *


class RinseSinkBasin(Kitchen):
    """
    Rinse the Sink Basin: composite task for Washing Dishes activity.

    Simulates the process of rinsing the sink basin using the sink's spout.

    Steps:
        1. Turn on the sink
        2. Move the spout left and right to rinse all locations of the sink basin.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.washed_loc = [False, False, False]

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink = self.get_fixture(FixtureType.SINK)
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):

        ep_meta = super().get_ep_meta()

        if self.use_novel_instructions:
            ep_meta["lang"] = self.rng.choice(self.novel_instructions)
        else:
            ep_meta[
                "lang"
            ] = "Turn on the sink and manuever the spout to wash all locations of the sink basin."
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()
        self.sink.set_handle_state(mode="off", env=self, rng=self.rng)
        self.washed_loc = [False, False, False]

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.get_fixture(FixtureType.COUNTER, ref=self.sink),
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        handle_state = self.sink.get_handle_state(env=self)

        if handle_state["water_on"]:
            if handle_state["spout_ori"] == "left":
                self.washed_loc[0] = True
            elif handle_state["spout_ori"] == "center":
                self.washed_loc[1] = True
            elif handle_state["spout_ori"] == "right":
                self.washed_loc[2] = True

        return all(self.washed_loc)
