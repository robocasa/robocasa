from robocasa.environments.kitchen.kitchen import *


class FreezeIceTray(Kitchen):
    """
    Freeze Ice Tray: composite task for Managing Freezer Space activity.

    Simulates the task of placing an ice tray in the freezer to make ice.

    Steps:
        1. Take the ice tray from next to the sink
        2. Place it in the freezer
        3. Close the freezer door
    """

    EXCLUDE_LAYOUTS = Kitchen.FREEZER_EXCLUDED_LAYOUTS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.fridge = self.register_fixture_ref("fridge", dict(id=FixtureType.FRIDGE))
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"The ice cube tray has been filled with water. Take the tray "
            "and place it in the freezer to make ice. Close the freezer door after placing the tray."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.fridge.open_door(env=self, compartment="freezer")

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="ice_tray",
                obj_groups="ice_cube_tray",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    rotation=np.pi / 2,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        ice_tray_in_freezer = self.fridge.check_rack_contact(
            self,
            "ice_tray",
            compartment="freezer",
        )

        freezer_door_closed = self.fridge.is_closed(self, compartment="freezer")

        return ice_tray_in_freezer and freezer_door_closed
