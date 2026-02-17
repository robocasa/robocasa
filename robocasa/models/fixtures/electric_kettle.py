from robocasa.models.fixtures import Fixture
import numpy as np


class ElectricKettle(Fixture):
    """
    Electric Kettle fixture class
    """

    def __init__(
        self,
        xml="fixtures/electric_kettles/ElectricKettle003",
        name="electric_kettle",
        *args,
        **kwargs,
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )
        self._lid = 0.0
        self._lid_speed = 0.5
        self._turned_on = False

        self._num_steps_on = 0
        self._cooldown_time = 0
        self._last_lid_update = None
        self._target_lid_angle = None

        # Initialize joint names dictionary
        joint_prefix = self._get_joint_prefix()
        self._joint_names = {
            "lid": f"{joint_prefix}lid_joint",
            "switch": f"{joint_prefix}switch_joint",
            "lid_button": f"{joint_prefix}button_lid_joint",
        }

    def _get_joint_prefix(self):
        return f"{self.naming_prefix}"

    def set_lid(self, env, lid_val=1.0, gradual=False):
        """
        Sets the state of the lid

        Args:
            lid_val (float): normalized value between 0 (closed) and 1 (open)
            gradual (bool): if True, the lid will move smoothly to its target position over time
        """
        joint_name = self._joint_names["lid"]
        self._lid = np.clip(lid_val, 0.0, 1.0)

        if gradual:
            self._target_lid_angle = self._lid
            self._last_lid_update = env.sim.data.time
            self._lid_speed = 1.5
        else:
            self.set_joint_state(
                env=env,
                min=self._lid,
                max=self._lid,
                joint_names=[joint_name],
            )

    def set_power_state(self, env, power_on):
        """
        Sets the power state using the switch

        Args:
            power_on (bool): if True, power is on, otherwise power is off
        """
        self._turned_on = power_on
        joint_name = self._joint_names["switch"]
        value = 1.0 if power_on else 0.0

        self.set_joint_state(
            env=env,
            min=value,
            max=value,
            joint_names=[joint_name],
        )

    def set_joint_state(self, min, max, env, joint_names):
        """
        Overrides base set_joint_state. Directly sets joint position using qpos without relying on actuators.

        Args:
            min (float): minimum normalized joint value (0 to 1)
            max (float): maximum normalized joint value (0 to 1)
            env (MujocoEnv): simulation environment
            joint_names (list of str): names of joints to update
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        for j_name in joint_names:
            info = self._joint_infos[j_name]
            joint_min, joint_max = info["range"]

            # Compute desired qpos from normalized min/max
            if joint_min >= 0:
                qpos_min = joint_min + (joint_max - joint_min) * min
                qpos_max = joint_min + (joint_max - joint_min) * max
            else:
                # For joints with negative ranges (e.g., -0.02 to 0.02)
                qpos_min = joint_max - (joint_max - joint_min) * max
                qpos_max = joint_max - (joint_max - joint_min) * min

            # Choose deterministic midpoint for stability (avoid bouncing)
            desired_qpos = 0.5 * (qpos_min + qpos_max)

            env.sim.data.set_joint_qpos(j_name, desired_qpos)
            env.sim.forward()

    def update_state(self, env):
        """
        Update the state of the electric kettle
        """
        # Update all joint states
        for name, jn in self._joint_names.items():
            if jn in env.sim.model.joint_names:
                state = self.get_joint_state(env, [jn])[jn]
                state = np.clip(state, 0.0, 1.0)
                setattr(self, f"_{name}", state)

        if self._lid < 0.9 and self._lid_speed == 0.5:
            self._target_lid_angle = None
            self._last_lid_update = None

        # Handle gradual lid movement when button is pressed
        if self._target_lid_angle is not None and self._last_lid_update is not None:
            joint_name = self._joint_names["lid"]
            if joint_name in env.sim.model.joint_names:
                current_angle = self.get_joint_state(env, [joint_name])[joint_name]
                time_elapsed = env.sim.data.time - self._last_lid_update

                angle_change = min(
                    time_elapsed * self._lid_speed,
                    abs(self._target_lid_angle - current_angle),
                )
                if self._target_lid_angle < current_angle:
                    angle_change = -angle_change

                new_angle = current_angle + angle_change
                new_angle = np.clip(new_angle, 0.0, 1.0)
                self.set_joint_state(
                    env=env, min=new_angle, max=new_angle, joint_names=[joint_name]
                )

                if abs(new_angle - self._target_lid_angle) < 0.001:
                    self._target_lid_angle = None
                    self._last_lid_update = None
                else:
                    self._last_lid_update = env.sim.data.time

        # Handle button-triggered lid opening
        if self._lid_button > 0.1 and self._lid <= 0.01:
            self.set_lid(env, lid_val=1.0, gradual=True)

        # Handle switch/power state
        switch_open_val = 1.0

        if self._switch >= 0.95 and not self._turned_on:
            self._turned_on = True
            self._num_steps_on += 1
            self.set_joint_state(
                env=env,
                min=switch_open_val,
                max=switch_open_val,
                joint_names=[self._joint_names["switch"]],
            )
        elif self._turned_on and self._num_steps_on < 500:
            self._num_steps_on += 1
            self.set_joint_state(
                env=env,
                min=switch_open_val,
                max=switch_open_val,
                joint_names=[self._joint_names["switch"]],
            )
        elif self._turned_on and self._num_steps_on >= 500:
            self._turned_on = False
            self._cooldown_time += 1
            self._num_steps_on = 0

        if self._cooldown_time > 0 and self._cooldown_time < 10:
            self._cooldown_time += 1
            new_switch_state = self._switch - 0.1
            if new_switch_state < 0.0:
                new_switch_state = 0.0
            new_switch_state = np.clip(new_switch_state, 0.0, 1.0)
            self.set_joint_state(
                env=env,
                min=new_switch_state,
                max=new_switch_state,
                joint_names=[self._joint_names["switch"]],
            )
            self._switch = new_switch_state
        elif self._cooldown_time >= 10:
            self._cooldown_time = 0

        # ensures lid stays open, and doesn't close on its own
        if self._lid > 0.90:
            if self._target_lid_angle is None:
                self._target_lid_angle = 1.0
                self._last_lid_update = env.sim.data.time
                self._lid_speed = 0.5

    def get_state(self, env):
        """
        Returns a dictionary representing the state of the electric kettle.
        """
        st = {}
        for name, jn in self._joint_names.items():
            if jn in env.sim.model.joint_names:
                st[name] = getattr(self, f"_{name}", None)
        st["turned_on"] = self._turned_on
        return st

    @property
    def nat_lang(self):
        return "electric kettle"
