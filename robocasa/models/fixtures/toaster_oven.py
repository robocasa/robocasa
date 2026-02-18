from robocasa.models.fixtures import Fixture
import numpy as np
import re


class ToasterOven(Fixture):
    """
    Toaster Oven fixture class
    """

    def __init__(
        self,
        xml="fixtures/toaster_ovens/ToasterOven055",
        name="toaster_oven",
        *args,
        **kwargs,
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

        self._door = 0.0
        self._doneness = 0.0
        self._function = 0.0
        self._temperature = 0.0
        self._time = 0.0
        self._rack = {}
        self._tray = {}

        self._door_target = None
        self._last_time_update = None

        joint_prefix = self._get_joint_prefix()
        self._joint_names = {
            "door": f"{joint_prefix}door_joint",
            "doneness": f"{joint_prefix}knob_doneness_joint",
            "function": f"{joint_prefix}knob_function_joint",
            "temperature": f"{joint_prefix}knob_temp_joint",
            "time": f"{joint_prefix}knob_time_joint",
            "rack": f"{joint_prefix}rack0_joint",
            "tray": f"{joint_prefix}tray0_joint",
        }

    def _get_joint_prefix(self):
        return f"{self.naming_prefix}"

    def get_reset_region_names(self):
        return ("rack0", "rack1", "tray0", "tray1")

    def get_reset_regions(self, env=None, rack_level=0, z_range=None):
        """
        Returns one reset region at the desired rack/tray level.
        Supports:
            rack_level=0 → bottom (rack0 or tray0)
            rack_level=1 → top if exists, else bottom
        Also updates joint name routing and initializes _rack/_tray tracking dicts.
        """

        rack_regions, tray_regions = [], []

        for key, reg in self._regions.items():
            m = re.fullmatch(r"(rack|tray)(\d+)", key)
            if not m:
                continue
            typ, idx = m.group(1), int(m.group(2))
            p0, px, py, pz = reg["p0"], reg["px"], reg["py"], reg["pz"]
            hx, hy, hz = (px[0] - p0[0]) / 2, (py[1] - p0[1]) / 2, (pz[2] - p0[2]) / 2
            center = p0 + np.array([hx, hy, hz])
            entry = (idx, key, p0, px, py, pz)
            if typ == "rack":
                rack_regions.append(entry)
            else:
                tray_regions.append(entry)

        def pick(regions, level):
            if not regions:
                return None
            regions.sort(key=lambda x: x[0])
            return regions[1] if level == 1 and len(regions) > 1 else regions[0]

        region = pick(rack_regions, rack_level)
        if region:
            level = region[0]
            self._joint_names["rack"] = f"{self._get_joint_prefix()}rack{level}_joint"
            self._rack[f"rack{level}"] = 0.0
        else:
            region = pick(tray_regions, rack_level)
            if not region:
                raise ValueError(f"No rack or tray reset regions found for {self.name}")
            level = region[0]
            self._joint_names["tray"] = f"{self._get_joint_prefix()}tray{level}_joint"
            self._tray[f"tray{level}"] = 0.0

        idx, key, p0, px, py, pz = region
        offset = (
            float(np.mean((p0[0], px[0]))),
            float(np.mean((p0[1], py[1]))),
            float(p0[2]),
        )
        size = (float(px[0] - p0[0]), float(py[1] - p0[1]))
        height = float(pz[2] - p0[2])
        return {key: {"offset": offset, "size": size, "height": height}}

    def set_doneness(self, env, val):
        self._doneness = np.clip(val, 0.0, 1.0)
        self.set_joint_state(
            env=env, min=val, max=val, joint_names=[self._joint_names["doneness"]]
        )

    def set_function(self, env, val):
        self._function = np.clip(val, 0.0, 1.0)
        self.set_joint_state(
            env=env, min=val, max=val, joint_names=[self._joint_names["function"]]
        )

    def set_temperature(self, env, val):
        self._temperature = np.clip(val, 0.0, 1.0)
        self.set_joint_state(
            env=env, min=val, max=val, joint_names=[self._joint_names["temperature"]]
        )

    def set_time(self, env, val):
        self._time = np.clip(val, 0.0, 1.0)
        self.set_joint_state(
            env=env, min=val, max=val, joint_names=[self._joint_names["time"]]
        )

    def has_multiple_rack_levels(self):
        """
        Returns True if there are multiple rack or tray levels, False if only one exists.
        """

        rack_levels = set()
        tray_levels = set()

        for key in self._regions:
            m = re.fullmatch(r"(rack|tray)(\d+)", key)
            if m:
                typ, idx = m.group(1), int(m.group(2))
                if typ == "rack":
                    rack_levels.add(idx)
                else:
                    tray_levels.add(idx)

        return len(rack_levels) > 1 or len(tray_levels) > 1

    def open_door(self, env, min=1.0, max=1.0):
        """
        helper function to open the door. calls set_door_state function
        """
        super().open_door(env=env, min=min, max=max)

    def close_door(self, env, min=0.0, max=0.0):
        """
        helper function to close the door. calls set_door_state function
        """
        super().close_door(env=env, min=min, max=max)

    def slide_rack(self, env, value=1.0, rack_level=0):
        """
        Slides the rack/tray at the specified level, with fallback to level 0 if the target level doesn't exist.

        Args:
            env: The environment object
            value (float): normalized value between 0 (closed) and 1 (open)
            rack_level (int): which level to target (0 = bottom, 1 = top)
        """
        door_pos = self.get_joint_state(env, [self._joint_names["door"]])[
            self._joint_names["door"]
        ]
        if door_pos <= 0.99:
            self.open_door(env)

        joint_prefix = self._get_joint_prefix()

        # Try rack{level}, fallback to rack0
        for level in [rack_level, 0]:
            joint = f"{joint_prefix}rack{level}_joint"
            if joint in env.sim.model.joint_names:
                name = f"rack{level}"
                if not isinstance(self._rack, dict):
                    self._rack = {}
                self._rack[name] = value
                self.set_joint_state(env=env, min=value, max=value, joint_names=[joint])
                return name

        for level in [rack_level, 0]:
            joint = f"{joint_prefix}tray{level}_joint"
            if joint in env.sim.model.joint_names:
                name = f"tray{level}"
                if not isinstance(self._tray, dict):
                    self._tray = {}
                self._tray[name] = value
                self.set_joint_state(env=env, min=value, max=value, joint_names=[joint])
                return name

        raise ValueError(
            f"No rack or tray found for level {rack_level}, and fallback to level 0 also failed."
        )

    def update_state(self, env):
        if not isinstance(self._rack, dict):
            self._rack = {}
        if not isinstance(self._tray, dict):
            self._tray = {}

        # Clear canonical keys to avoid stale values
        for k in ["rack0", "rack1"]:
            self._rack.pop(k, None)
        for k in ["tray0", "tray1"]:
            self._tray.pop(k, None)

        # Check every joint in the model for matching rack/tray joints
        for joint_name in env.sim.model.joint_names:
            if "rack" in joint_name and self.name in joint_name:
                val = self.get_joint_state(env, [joint_name])[joint_name]
                key = joint_name.replace("_joint", "")
                self._rack[key] = val
                if key.endswith("rack0"):
                    self._rack["rack0"] = val
                elif key.endswith("rack1"):
                    self._rack["rack1"] = val
            elif "tray" in joint_name and self.name in joint_name:
                val = self.get_joint_state(env, [joint_name])[joint_name]
                key = joint_name.replace("_joint", "")
                self._tray[key] = val
                if key.endswith("tray0"):
                    self._tray["tray0"] = val
                elif key.endswith("tray1"):
                    self._tray["tray1"] = val

        # Update non-rack/tray joints from predefined names
        for name, jn in self._joint_names.items():
            # safeguard to keep door from falling on its own
            if name == "door" and self.get_joint_state(env, [jn])[jn] < 0.01:
                self.set_joint_state(env=env, min=0.0, max=0.0, joint_names=[jn])
            if name in ["rack", "tray"]:
                continue  # already handled above
            if jn in env.sim.model.joint_names:
                val = self.get_joint_state(env, [jn])[jn]
                if name == "time" and val > 0:
                    if self._last_time_update is None:
                        self._last_time_update = env.sim.data.time
                    else:
                        elapsed = env.sim.data.time - self._last_time_update
                        new_val = max(0, val - elapsed / 3000)
                        self.set_joint_state(
                            env=env, min=new_val, max=new_val, joint_names=[jn]
                        )
                else:
                    self._last_time_update = None
                setattr(self, f"_{name}", val)

    def check_rack_contact(self, env, obj_name, rack_level=0):
        """
        Checks whether object is touching the specified rack or tray level.
        If level 1 is requested but only level 0 exists, falls back to level 0.

        Args:
            env: The simulation environment
            obj_name (str): Name of the object to check contact with
            rack_level (int): 0 for bottom, 1 for top (falls back to 0 if top does not exist)
        """
        joint_prefix = self._get_joint_prefix()

        for level in [rack_level, 0]:
            joint = f"{joint_prefix}rack{level}_joint"
            if joint in env.sim.model.joint_names:
                contact_name = joint.replace("_joint", "")
                break
        else:
            for level in [rack_level, 0]:
                joint = f"{joint_prefix}tray{level}_joint"
                if joint in env.sim.model.joint_names:
                    contact_name = joint.replace("_joint", "")
                    break
            else:
                raise RuntimeError(
                    f"No rack or tray found at level {rack_level}, and fallback to level 0 failed."
                )

        body_id = env.sim.model.body_name2id(contact_name)

        shelf_geoms = [
            env.sim.model.geom_id2name(i)
            for i in range(env.sim.model.ngeom)
            if env.sim.model.geom_bodyid[i] == body_id
        ]
        obj_body = env.obj_body_id[obj_name]
        obj_geoms = [
            env.sim.model.geom_id2name(gid)
            for gid in range(env.sim.model.ngeom)
            if env.sim.model.geom_bodyid[gid] == obj_body
        ]
        return env.check_contact(shelf_geoms, obj_geoms)

    def rack_or_tray(self, env):
        """
        Returns whether this toaster oven model has rack0 or tray0.

        Returns:
            str: "rack" if rack0 exists, "tray" if tray0 exists.
        """
        joint_prefix = self._get_joint_prefix()

        if f"{joint_prefix}rack0_joint" in env.sim.model.joint_names:
            return "rack"
        elif f"{joint_prefix}tray0_joint" in env.sim.model.joint_names:
            return "tray"
        else:
            raise ValueError(
                "Neither rack0 nor tray0 found in this toaster oven model."
            )

    def get_state(self, rack_level=0):
        """
        Returns the state of the toaster oven.

        Args:
            env: the environment
            rack_level (int): 0 (bottom) or 1 (top). If 1 is not available, falls back to 0.

        Returns:
            dict: current state of the selected rack or tray level
        """
        state = {}

        target_keys = [f"rack{rack_level}", f"tray{rack_level}"]
        fallback_keys = ["rack0", "tray0"]

        for key in target_keys + fallback_keys:
            if key in self._rack:
                state[key] = self._rack.get(key, None)
                break
            elif key in self._tray:
                state[key] = self._tray.get(key, None)
                break
        else:
            raise ValueError(
                f"No rack or tray state available for rack_level={rack_level}"
            )

        for name in ["door", "doneness", "function", "temperature", "time"]:
            state[name] = getattr(self, f"_{name}", None)

        return state

    @property
    def nat_lang(self):
        return "toaster oven"
