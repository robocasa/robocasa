import math

from robocasa.environments.kitchen.kitchen import *


class DryDrinkware(Kitchen):
    """
    Dry Drinkware: composite task for Washing Dishes activity.

    Simulates the task of drying drinkware.

    Steps:
        Pick the mug from the counter and place it upside down in the open cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet in which the mug is placed.
    """

    def __init__(self, cab_id=FixtureType.CABINET, *args, **kwargs):
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.5))
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "A wet mug is on the counter and needs to be dried. "
            "Pick it up and place it upside down in the open cabinet."
        )
        return ep_meta

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.cab.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []
        x_positions = [-1, 1]
        self.rng.shuffle(x_positions)

        cfgs.append(
            dict(
                name="mug",
                obj_groups="mug",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.3),
                    pos=(x_positions[0], -1.0),
                ),
            )
        )
        # cfgs.append(dict(
        #     name="cup",
        #     obj_groups="cup",
        #     placement=dict(
        #         fixture=self.counter,
        #         sample_region_kwargs=dict(
        #             ref=self.cab,
        #         ),
        #         size=(0.3, 0.3),
        #         pos=(x_positions[1], -1.0)
        #     ),
        # ))

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    def _check_success(self):
        mug_rot = self.sim.data.xquat[self.obj_body_id["mug"]]
        # make sure the mug is placed upside down
        mug_rot = self.euler_from_quaternion(*mug_rot)
        return (
            OU.gripper_obj_far(self, obj_name="mug")
            and np.abs(mug_rot[2]) > 3
            and OU.check_obj_fixture_contact(self, "mug", self.cab)
        )
