import numpy as np

from robocasa.models.arenas.layout_utils import *
from robocasa.models.objects.fixtures.cabinets import *
from robocasa.models.objects.fixtures.others import Box

# fixtures that can be used to form a stack
STACKABLE = {
    "single_cabinet": SingleCabinet,
    "hinge_cabinet": HingeCabinet,
    "drawer": Drawer,
    "panel_cabinet": PanelCabinet,
    "box": Box,
}


class FixtureStack:
    def __init__(
        self,
        config,
        scene_fixtures,
        scene_configs,
        scene_style,
        base_height=0.05,
        base_overhang=0.07,
        default_texture=None,
        rng=None,
    ):
        self._check_config_syntax(config)
        self.config = config

        # for relative positioning purposes
        self.origin_offset = np.array([0, 0, 0])
        self._scale = 1

        # for base
        self.base_height = base_height
        self.base_overhang = base_overhang

        self.scene_style = scene_style
        self.scene_configs = scene_configs
        self.scene_fixtures = scene_fixtures
        self.default_texture = default_texture
        self.fixtures = list()

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        self._create_stack()

    def _create_stack(self):
        self.size = self.config["size"]
        width_stack, depth, height_stack = self.size

        if "pos" in self.config:
            self.pos = self.config["pos"]
        else:
            self.pos = get_relative_position(
                self,
                self.config,
                self.scene_fixtures[self.config["align_to"]],
                self.scene_configs[self.config["align_to"]],
            )

        x_stack, y, z_stack = self.pos
        # incremented as levels are stacked
        z_current = z_stack - height_stack / 2
        fxtr_count = 0

        # add base for stack
        if self.base_height > 0:
            depth_base = depth - self.base_overhang
            z_base = z_stack - height_stack / 2 - self.base_height / 2
            base_name = self.config["name"] + "_base"

            base_config = load_default_config(self.scene_style, {"type": "box"})
            base_config["pos"] = [x_stack, y + self.base_overhang / 2, z_base]
            base_config["size"] = [width_stack, depth_base, self.base_height]
            base_config["name"] = base_name
            base_config["type"] = STACKABLE["box"]
            for k in ["group_origin", "group_pos", "group_z_rot"]:
                if k in self.config:
                    base_config[k] = self.config[k]

            if self.default_texture is not None:
                base_config["texture"] = self.default_texture
            base_config["rng"] = self.rng
            base = initialize_fixture(base_config, self.scene_fixtures)
            self.scene_fixtures[base_name] = base
            self.scene_configs[base_name] = base_config
            self.fixtures.append(base)

        # initialize layers, going from bottom to top
        for i, level in enumerate(self.config["levels"]):
            if type(level) == list:
                # two fixtures in this level
                fixtures = [
                    [level[0], [x_stack - width_stack / 4, width_stack / 2]],
                    [level[1], [x_stack + width_stack / 4, width_stack / 2]],
                ]
            else:
                fixtures = [[level, [x_stack, width_stack]]]

            for fixture, (x, width) in fixtures:
                if fixture not in STACKABLE:
                    raise ValueError(
                        '"{}" is not a stackable fixture type'.format(fixture)
                    )

                # find size and position of fixture
                height = height_stack * self.config["percentages"][i]
                z = z_current + height / 2

                # get fixture config for fixture
                fxtr_config = load_default_config(self.scene_style, {"type": fixture})
                fxtr_config["pos"] = [x, y, z]
                fxtr_config["size"] = [width, depth, height]

                if "cabinet" in fixture:
                    fxtr_config["panel_config"] = fxtr_config.get("panel_config", {})
                    fxtr_config["panel_config"]["handle_vpos"] = "top"

                fxtr_count += 1
                fxtr_name = self.config["name"] + "_" + str(fxtr_count)
                fxtr_config["name"] = fxtr_name
                fxtr_config["type"] = STACKABLE[fixture]

                # add in additional configurations as specified
                if "configs" in self.config:
                    cfg_fixture = None
                    if fixture in self.config["configs"]:
                        cfg_fixture = fixture
                    elif f"{fixture}_{i + 1}" in self.config["configs"]:
                        cfg_fixture = f"{fixture}_{i + 1}"

                    if cfg_fixture is not None:
                        for k, v in self.config["configs"][cfg_fixture].items():
                            fxtr_config[k] = v

                # account for group rotations
                for k in ["group_origin", "group_pos", "group_z_rot"]:
                    if k in self.config:
                        fxtr_config[k] = self.config[k]

                # for texture randomization, ensures all cabinets have the same color
                if self.default_texture is not None and "texture" in fxtr_config:
                    fxtr_config["texture"] = self.default_texture

                # initialize fixture and add to scene fixtures and configs
                fixture = initialize_fixture(
                    fxtr_config, self.scene_fixtures, rng=self.rng
                )
                self.scene_fixtures[fxtr_name] = fixture
                self.scene_configs[fxtr_name] = fxtr_config
                self.fixtures.append(fixture)

            # increment z to move up a level
            z_current += height

    @staticmethod
    def _check_config_syntax(config):
        if "size" not in config or None in config["size"]:
            raise ValueError(
                "Size for stacks must be specified explicitely, " "received:",
                config["size"],
            )
        if "levels" not in config or "percentages" not in config:
            raise ValueError(
                'Both "levels" and "percentages" ' "must be specified for fixture stack"
            )
        if len(config["levels"]) != len(config["percentages"]):
            raise ValueError(
                '"levels" and "percentages" must have '
                "the same length for fixture stacks"
            )
        for level in config["levels"]:
            if type(level) == list and len(level) != 2:
                raise ValueError(
                    "There can be at most 2 fixtures per level " "in fixture stacks"
                )
