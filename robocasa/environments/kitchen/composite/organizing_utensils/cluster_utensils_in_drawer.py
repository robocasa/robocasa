from robocasa.environments.kitchen.kitchen import *


class ClusterUtensilsInDrawer(Kitchen):
    """
    Cluster Utensils In Drawer: composite task for Organizing Utensils activity.

    Simulates the task of organizing utensils by clustering them by type in a drawer.
    There are drawer utensils in the drawer (fork on one side, spoon on the other)
    and additional utensils on the counter that need to be placed with their matching types.

    Steps:
        1. Pick up the fork from the counter and place it on the fork side of the drawer.
        2. Pick up the spoon from the counter and place it on the spoon side of the drawer.
        3. Ensure utensils are clustered by type on opposite sides of the drawer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER, size=(0.5, 0.5))
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer)
        )

        if "fork_side" in self._ep_meta:
            self.fork_side = self._ep_meta["fork_side"]
            self.spoon_side = self._ep_meta["spoon_side"]
        else:
            sides = ["left", "right"]
            fork_choice = self.rng.choice(sides)
            spoon_choice = "right" if fork_choice == "left" else "left"
            self.fork_side = fork_choice
            self.spoon_side = spoon_choice

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["fork_side"] = self.fork_side
        ep_meta["spoon_side"] = self.spoon_side

        ep_meta["lang"] = (
            f"Pick up the fork from the counter and place it on the {self.fork_side} side of the drawer. "
            f"Then, pick up the spoon from the counter and place it on the {self.spoon_side} side of the drawer. "
            f"Make sure that the utensils are clustered by type."
        )
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.drawer.open_door(env=self)

    def _get_obj_cfgs(self):
        cfgs = []

        # Determine positions based on randomized sides
        fork_pos = (-1.0, -0.25) if self.fork_side == "left" else (1.0, -0.25)
        spoon_pos = (1.0, -0.25) if self.spoon_side == "right" else (-1.0, -0.25)

        cfgs.append(
            dict(
                name="drawer_fork",
                obj_groups="fork",
                placement=dict(
                    fixture=self.drawer,
                    size=(0.1, 0.3),
                    pos=fork_pos,
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="drawer_spoon",
                obj_groups="spoon",
                placement=dict(
                    fixture=self.drawer,
                    size=(0.1, 0.3),
                    pos=spoon_pos,
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="counter_fork",
                obj_groups="fork",
                init_robot_here=True,
                object_scale=[1, 1, 2.5],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    rotation=0,
                ),
            )
        )

        cfgs.append(
            dict(
                name="counter_spoon",
                obj_groups="spoon",
                object_scale=[1, 1, 2.5],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    rotation=0,
                ),
            )
        )

        return cfgs

    def _check_success(self):
        utensil_names = ["drawer_fork", "drawer_spoon", "counter_fork", "counter_spoon"]
        all_in_drawer = all(
            OU.obj_inside_of(self, name, self.drawer) for name in utensil_names
        )

        all_not_on_counter = all(
            not OU.check_obj_any_counter_contact(self, name) for name in utensil_names
        )

        threshold = 0.1

        drawer_fork_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["drawer_fork"]]
        )
        counter_fork_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["counter_fork"]]
        )
        drawer_spoon_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["drawer_spoon"]]
        )
        counter_spoon_pos = np.array(
            self.sim.data.body_xpos[self.obj_body_id["counter_spoon"]]
        )

        drawer_fork_x, drawer_fork_y = OU.transform_global_to_local(
            drawer_fork_pos[0], drawer_fork_pos[1], -self.drawer.rot
        )
        counter_fork_x, counter_fork_y = OU.transform_global_to_local(
            counter_fork_pos[0], counter_fork_pos[1], -self.drawer.rot
        )
        drawer_spoon_x, drawer_spoon_y = OU.transform_global_to_local(
            drawer_spoon_pos[0], drawer_spoon_pos[1], -self.drawer.rot
        )
        counter_spoon_x, counter_spoon_y = OU.transform_global_to_local(
            counter_spoon_pos[0], counter_spoon_pos[1], -self.drawer.rot
        )

        fork_x_distance = abs(drawer_fork_x - counter_fork_x)
        spoon_x_distance = abs(drawer_spoon_x - counter_spoon_x)

        forks_close = fork_x_distance < threshold
        spoons_close = spoon_x_distance < threshold

        gripper_far_fork = OU.gripper_obj_far(self, obj_name="counter_fork")
        gripper_far_spoon = OU.gripper_obj_far(self, obj_name="counter_spoon")
        gripper_far = gripper_far_fork and gripper_far_spoon

        return (
            all_in_drawer
            and all_not_on_counter
            and forks_close
            and spoons_close
            and gripper_far
        )
