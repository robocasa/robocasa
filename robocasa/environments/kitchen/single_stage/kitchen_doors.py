from robocasa.environments.kitchen.kitchen import *


class ManipulateDoor(Kitchen):
    """
    Class encapsulating the atomic manipulate door tasks.

    Args:
        behavior (str): "open" or "close". Used to define the desired
            door manipulation behavior for the task.

        door_id (str): The door fixture id to manipulate.
    """

    def __init__(
        self, behavior="open", door_id=FixtureType.DOOR_TOP_HINGE, *args, **kwargs
    ):
        self.door_id = door_id
        assert behavior in ["open", "close"]
        self.behavior = behavior
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the door tasks.
        """
        super()._setup_kitchen_references()
        self.door_fxtr = self.register_fixture_ref("door_fxtr", dict(id=self.door_id))
        self.init_robot_base_pos = self.door_fxtr

    def get_ep_meta(self):
        """
        Get the episode metadata for the door tasks.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        if isinstance(self.door_fxtr, Microwave):
            door_fxtr_name = "microwave"
            door_name = "door"
        elif isinstance(self.door_fxtr, SingleCabinet):
            door_fxtr_name = "cabinet"
            door_name = "door"
        elif isinstance(self.door_fxtr, HingeCabinet):
            door_fxtr_name = "cabinet"
            door_name = "doors"
        elif isinstance(self.door_fxtr, Drawer):
            door_fxtr_name = "drawer"
            door_name = "doors"
        ep_meta["lang"] = f"{self.behavior} the {door_fxtr_name} {door_name}"
        return ep_meta

    def _reset_internal(self):
        """
        Reset the environment internal state for the door tasks.
        This includes setting the door state based on the behavior.
        """
        if self.behavior == "open":
            self.door_fxtr.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        elif self.behavior == "close":
            self.door_fxtr.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        # set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened
        super()._reset_internal()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Get the address for the door hinge joint
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(
            self.door_fxtr.joints[0]
        )
        self.gripper_qpos_joint1_addr = self.sim.model.get_joint_qpos_addr(
            self.robots[0].gripper_joints['right'][0]
        )
        self.gripper_qpos_joint2_addr = self.sim.model.get_joint_qpos_addr(
            self.robots[0].gripper_joints['right'][1]
        )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Add door angle to the observables

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        @sensor(modality="object")
        def hinge_angle(obs_cache):
            # Return normalized door angle instead of raw angle
            door_state = self.door_fxtr.get_door_state(env=self)
            # Convert dict values to list before creating numpy array
            return np.array(list(door_state.values()))

        observables["hinge_angle"] = Observable(
            name="hinge_angle",
            sensor=hinge_angle,
            sampling_rate=self.control_freq,
            active=True,
        )

        @sensor(modality="object")
        def handle_pos_quat(obs_cache):
            # Return handle rotation
            try:
                handle_xpos = self.sim.data.get_geom_xpos(self.door_fxtr.contact_geoms[12])
                handle_xmat = self.sim.data.get_geom_xmat(self.door_fxtr.contact_geoms[12])
                handle_quat = Rotation.from_matrix(handle_xmat).as_quat()
                return np.array(handle_xpos.tolist() + handle_quat.tolist())
            except Exception as e:
                print(f"Error getting handle rotation: {e}")
                return np.zeros(9)

        observables["handle_pos_quat"] = Observable(
            name="handle_pos_quat",
            sensor=handle_pos_quat,
            sampling_rate=self.control_freq,
            active=True,
        )

        @sensor(modality="object")
        def gripper_pos_quat_angle(obs_cache):
            # Return gripper angle
            eef_pos = self.sim.data.get_body_xpos(self.robots[0].gripper["right"].bodies[1])
            eef_quat = self.sim.data.get_body_xquat(self.robots[0].gripper["right"].bodies[1])
            joint1 = self.sim.data.qpos[self.gripper_qpos_joint1_addr]
            joint2 = self.sim.data.qpos[self.gripper_qpos_joint2_addr]
            return np.array(eef_pos.tolist() + eef_quat.tolist() + [joint1])

        observables["gripper_pos_quat_angle"] = Observable(
            name="gripper_pos_quat_angle",
            sensor=gripper_pos_quat_angle,
            sampling_rate=self.control_freq,
            active=True,
        )

        # @sensor(modality="object")
        # def gripper_finger1_pos(obs_cache):
        #     # Return gripper finger1 position
        #     finger1_xpos = self.sim.data.get_geom_xpos(self.robots[0].gripper["right"].contact_geoms[1])
        #     return np.array(finger1_xpos)

        # observables["gripper_finger1_pos"] = Observable(
        #     name="gripper_finger1_pos",
        #     sensor=gripper_finger1_pos,
        #     sampling_rate=self.control_freq,
        #     active=True,
        # )

        # @sensor(modality="object")
        # def gripper_finger1_quat(obs_cache):
        #     # Return gripper finger1 rotation
        #     finger1_xmat = self.sim.data.get_geom_xmat(self.robots[0].gripper["right"].contact_geoms[1])
        #     finger1_quat = Rotation.from_matrix(finger1_xmat).as_quat()
        #     return np.array(finger1_quat)

        # observables["gripper_finger1_quat"] = Observable(
        #     name="gripper_finger1_quat",
        #     sensor=gripper_finger1_quat,
        #     sampling_rate=self.control_freq,
        #     active=True,
        # )

        # @sensor(modality="object")
        # def gripper_finger2_pos(obs_cache):
        #     # Return gripper finger2 position
        #     finger2_xpos = self.sim.data.get_geom_xpos(self.robots[0].gripper["right"].contact_geoms[3])
        #     return np.array(finger2_xpos)

        # observables["gripper_finger2_pos"] = Observable(
        #     name="gripper_finger2_pos",
        #     sensor=gripper_finger2_pos,
        #     sampling_rate=self.control_freq,
        #     active=True,
        # )

        # @sensor(modality="object")
        # def gripper_finger2_quat(obs_cache):
        #     # Return gripper finger2 rotation
        #     finger2_xmat = self.sim.data.get_geom_xmat(self.robots[0].gripper["right"].contact_geoms[3])
        #     finger2_quat = Rotation.from_matrix(finger2_xmat).as_quat()
        #     return np.array(finger2_quat)

        # observables["gripper_finger2_quat"] = Observable(
        #     name="gripper_finger2_quat",
        #     sensor=gripper_finger2_quat,
        #     sampling_rate=self.control_freq,
        #     active=True,
        # )

        return observables

    def _check_success(self):
        """
        Check if the door manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        door_state = self.door_fxtr.get_door_state(env=self)

        success = True
        for joint_p in door_state.values():
            if self.behavior == "open":
                if joint_p < 0.90:
                    success = False
                    break
            elif self.behavior == "close":
                if joint_p > 0.05:
                    success = False
                    break

        return success

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the door tasks. This includes the object placement configurations.
        Place one object inside the door fixture and 1-4 distractors on the counter.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="door_obj",
                obj_groups="all",
                graspable=True,
                microwavable=(True if isinstance(self.door_fxtr, Microwave) else None),
                placement=dict(
                    fixture=self.door_fxtr,
                    size=(0.30, 0.30),
                    pos=(None, -1.0),
                ),
            )
        )

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i+1}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(
                            FixtureType.COUNTER, ref=self.door_fxtr
                        ),
                        sample_region_kwargs=dict(
                            ref=self.door_fxtr,
                        ),
                        size=(1.0, 0.50),
                        pos=(None, -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )

        return cfgs


class OpenDoor(ManipulateDoor):
    def __init__(self, *args, **kwargs):
        super().__init__(behavior="open", *args, **kwargs)


class OpenSingleDoor(OpenDoor):
    def __init__(self, door_id=FixtureType.DOOR_TOP_HINGE_SINGLE, *args, **kwargs):
        super().__init__(door_id=door_id, *args, **kwargs)


class OpenDoubleDoor(OpenDoor):
    def __init__(self, door_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):
        super().__init__(door_id=door_id, *args, **kwargs)


class CloseDoor(ManipulateDoor):
    def __init__(self, behavior=None, *args, **kwargs):
        super().__init__(behavior="close", *args, **kwargs)


class CloseSingleDoor(CloseDoor):
    def __init__(self, door_id=FixtureType.DOOR_TOP_HINGE_SINGLE, *args, **kwargs):
        super().__init__(door_id=door_id, *args, **kwargs)


class CloseDoubleDoor(CloseDoor):
    def __init__(self, door_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs):
        super().__init__(door_id=door_id, *args, **kwargs)
