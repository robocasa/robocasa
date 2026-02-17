from robocasa.environments.kitchen.kitchen import *


class MakeChocolateMilk(Kitchen):
    """
    Make Chocolate Milk: composite task for Mixing Ingredients activity.

    Simulates the task of placing one chocolate piece and one sugar cube in a warm glass of milk to create chocolate flavored milk.

    Steps:
        1. Pick up one chocolate piece and one sugar cube from the plate
        2. Place the chocolate piece and sugar cube in the glass of milk
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, full_depth_region=True)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Place one chocolate piece and one sugar cube in the warm glass of milk to make chocolate flavored milk."
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        OU.add_obj_liquid_site(self, "glass_milk", [1.0, 1.0, 1.0, 0.9])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="glass_milk",
                obj_groups="glass_cup",
                object_scale=[1.35, 1.35, 1.0],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(full_depth_region=True),
                    size=(0.3, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                object_scale=1.1,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="glass_milk",
                    size=(0.7, 0.35),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="chocolate1",
                obj_groups="chocolate",
                object_scale=[1, 1, 2],
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="chocolate2",
                obj_groups="chocolate",
                object_scale=[1, 1, 2],
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="sugar_cube",
                obj_groups="sugar_cube",
                object_scale=1.1,
                placement=dict(
                    object="plate",
                    size=(0.75, 0.75),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        def is_tilted(obj_name, min_deg=25.0):
            xmat = np.array(self.sim.data.xmat[self.obj_body_id[obj_name]]).reshape(
                3, 3
            )
            local_z = xmat[:, 2]
            world_z = np.array([0.0, 0.0, 1.0])
            cosang = np.clip(np.dot(local_z, world_z), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cosang))
            return angle_deg >= min_deg

        chocolate1_in_glass = OU.check_obj_in_receptacle(
            self, "chocolate1", "glass_milk"
        ) and is_tilted("chocolate1")
        chocolate2_in_glass = OU.check_obj_in_receptacle(
            self, "chocolate2", "glass_milk"
        ) and is_tilted("chocolate2")

        exactly_one_chocolate = (chocolate1_in_glass and not chocolate2_in_glass) or (
            chocolate2_in_glass and not chocolate1_in_glass
        )

        sugar_in_glass = OU.check_obj_in_receptacle(self, "sugar_cube", "glass_milk")
        gripper_far_glass = OU.gripper_obj_far(self, "glass_milk")

        return exactly_one_chocolate and sugar_in_glass and gripper_far_glass
