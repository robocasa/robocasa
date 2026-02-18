from robocasa.environments.kitchen.atomic.kitchen_drawer import *
from robocasa.environments.kitchen.kitchen import *
from scipy.spatial.transform import Rotation as R


class OrganizeMeasuringCups(Kitchen):
    """
    Organize Measuring Cups: composite task for Measuring Ingredients activity.

    Simulates the organization of measuring cups by their size.

    Steps:
        Organize the measuring cups left to right, smallest to largest and
        make sure they are rotated straight
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):

        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=FixtureType.CABINET)
        )
        self.init_robot_base_ref = self.counter

        if "refs" in self._ep_meta:
            self.cup_sides = self._ep_meta["refs"]["cup_sides"]
        else:
            self.cup_sides = self.rng.choice(
                ["left", "right"], size=3, replace=True
            ).tolist()

    def get_ep_meta(self):

        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Organize the measuring cups left to right, smallest to largest. The "
            "measuring cups must be rotated so the handle points directly outward "
            "towards the front of the counter. Make sure the measuring cups are upright."
        )
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["cup_sides"] = self.cup_sides
        return ep_meta

    def _get_obj_cfgs(self):

        cfgs = []

        measuring_cup_info = self.sample_object(
            groups=["measuring_cup"], obj_registries=self.obj_registries
        )

        # use the same object instance
        measuring_cup_inst = measuring_cup_info[1]["mjcf_path"]

        sizes = [(0.8, "small"), (1.0, "medium"), (1.2, "large")]

        for size in sizes:
            scale, name = size
            cup_side = self.cup_sides.pop(0)
            cup_side_sign = -1.0 if cup_side == "left" else 1.0
            cfgs.append(
                dict(
                    name=f"measuring_cup_{name}",
                    obj_groups=measuring_cup_inst,
                    placement=dict(
                        fixture=self.counter,
                        size=(0.6, 0.4),
                        pos=(0, -1),
                        rotation=(
                            cup_side_sign * np.deg2rad(35),
                            cup_side_sign * np.deg2rad(60),
                        ),
                        # offset=offset,
                    ),
                    object_scale=scale,
                )
            )
        cfgs[1]["init_robot_here"] = True
        cfgs[1]["placement"]["reuse_region_from"] = "measuring_cup_small"
        cfgs[2]["placement"]["reuse_region_from"] = "measuring_cup_small"

        return cfgs

    def _check_success(self):
        pos_thresholds = [0.35, 0.15]
        rot_threshold = 30

        sizes = ["small", "medium", "large"]
        obj_ids = [self.obj_body_id[f"measuring_cup_{size}"] for size in sizes]
        global_pos = [self.sim.data.body_xpos[oid] for oid in obj_ids]
        local_xy_pos = [
            OU.get_fixture_to_point_rel_offset(self.counter, gp)[:2]
            for gp in global_pos
        ]

        diffs = [
            local_xy_pos[i + 1] - local_xy_pos[i] for i in range(len(local_xy_pos) - 1)
        ]
        # no need to reorder mujoco quaternion from wxyz to xyzw if scalar_first=True
        z_rots = [
            R.from_quat(self.sim.data.body_xquat[oid], scalar_first=True).as_euler(
                "xyz", degrees=True
            )[2]
            for oid in obj_ids
        ]
        cups_rotated_properly = all(
            [
                abs(zrot - np.rad2deg(self.counter.rot)) < rot_threshold
                for zrot in z_rots
            ]
        )
        cups_upright = all(
            [
                OU.check_obj_upright(self, f"measuring_cup_{size}", th=60)
                for size in sizes
            ]
        )
        cups_on_counter = all(
            [
                OU.check_obj_fixture_contact(
                    self,
                    f"measuring_cup_{size}",
                    self.counter,
                )
                for size in sizes
            ]
        )

        for i, (dx, dy) in enumerate(diffs):
            if not (0 < dx < pos_thresholds[0]) or not (
                0 < abs(dy) < pos_thresholds[1]
            ):
                return False

        gripper_far = all(
            [
                OU.gripper_obj_far(self, f"measuring_cup_{size}", th=0.15)
                for size in sizes
            ]
        )

        return (
            cups_rotated_properly and cups_on_counter and cups_upright and gripper_far
        )
