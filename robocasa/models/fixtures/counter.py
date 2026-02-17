import math
import random
from copy import deepcopy

import numpy as np
from robosuite.utils.mjcf_utils import array_to_string as a2s
from robosuite.utils.mjcf_utils import find_elements, new_geom
from robosuite.utils.mjcf_utils import string_to_array as s2a
from robosuite.utils.mjcf_utils import xml_path_completion

import robocasa
from robocasa.models.fixtures.fixture import (
    ProcGenFixture,
    get_texture_name_from_file,
)
from robocasa.utils.object_utils import (
    get_fixture_to_point_rel_offset,
    get_pos_after_rel_offset,
    get_rel_transform,
    project_point_to_segment,
)

SIDES = ["left", "right", "front", "back"]


class Counter(ProcGenFixture):
    """
    Initializes a counter fixture.

    Args:
        name (str): name of the counter

        size (tuple): size of the counter (width, depth, height)

        overhang (float): amount which the top of the counter overhangs the base

        top_texture (str): path to the texture file for the top of the counter

        top_thickness (float): thickness of the top of the counter

        half_top (list): list of booleans to specify if the counter is half-sized and extends right or left

        base_texture (str): path to the texture file for the base of the counter

        base_color (list): list of 4 floats specifying the color of the base

        base_opening (list): list of booleans to specify if the counter has an opening in the front or back

        hollow (list): list of booleans to specify if the should exclude the front, back, or the whole the base.
                        Both hollow and base_opening cannot be set at the same time if they are, base_opening takes precedence.

        interior_obj (ProcGenFixture): object to be placed inside the counter

        obj_y_percent (float): Percentage of the counter's depth taken up by the interior object

        obj_x_percent (float): Percentage of the counter's width taken up by the interior object
    """

    def __init__(
        self,
        name="counter",
        size=(0.72, 0.60, 0.60),
        overhang=0,
        # top
        top_texture=None,
        top_thickness=0.03,
        half_top=[False, False],  # for aligning corner counters
        # base, can use both
        base_texture=None,
        base_color=None,
        base_opening=[False, False],
        # to add bottom row cabinets inside
        # [back, front]
        hollow=[False, True],
        # for sinks, cooktops
        interior_obj=None,
        obj_y_percent=0.5,
        obj_x_percent=0.5,
        # island_only flag to override dining counter classification
        island_only=False,
        *args,
        **kwargs,
    ):
        self.has_opening = interior_obj is not None
        if self.has_opening:
            xml = "fixtures/counters/counter_with_opening/model.xml"
        else:
            xml = "fixtures/counters/counter/model.xml"
        self.interior_obj = None

        self.size = size
        self.th = top_thickness
        self.overhang = overhang
        self.half_top = half_top
        self.island_only = island_only

        # Remove island_only from kwargs to avoid passing it to parent class
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("island_only", None)

        super().__init__(
            xml=xml,
            name=name,
            *args,
            **kwargs_copy,
        )

        assert len(hollow) == 2
        self.hollow = hollow

        if len(base_opening) != 2 or sum(base_opening) > 1:
            raise ValueError("Invalid value for `base_opening`:", base_opening)
        self.base_opening = base_opening
        if sum(self.base_opening) == 1:
            # overwrites hollow parameter
            self.hollow = [False, False]

        if interior_obj is None:
            self._make_counter()
        else:
            if type(interior_obj) == str:
                raise ValueError("Sink must be initialized before CounterSink")
            self.interior_obj = interior_obj
            self.obj_x_percent = obj_x_percent
            self.obj_y_percent = obj_y_percent

            top_padding = self._place_interior_obj()
            self._make_counter_with_opening(top_padding)

        # set top texture
        self._set_texture(top_texture, base_texture, base_color)

        # set sites
        x, y, z = np.array(self.size) / 2

        main_p0 = np.array([-x, -y + self.overhang, -z])
        main_p1 = np.array([x, y, z])
        self.update_regions(
            {
                "main": {
                    "pos": (main_p0 + main_p1) / 2,
                    "halfsize": (main_p1 - main_p0) / 2,
                }
            },
            update_elem=True,
        )

    def _set_texture(self, top_texture, base_texture, base_color):
        # set top and bottom textures
        self.top_texture = xml_path_completion(
            top_texture, root=robocasa.models.assets_root
        )
        self.base_texture = xml_path_completion(
            base_texture, root=robocasa.models.assets_root
        )

        # set top texture and materials
        texture = find_elements(
            self.root, tags="texture", attribs={"name": "tex_top_2d"}, return_first=True
        )
        tex_name = get_texture_name_from_file(self.top_texture) + "_2d"
        texture.set("name", tex_name)
        texture.set("file", self.top_texture)
        material = find_elements(
            self.root,
            tags="material",
            attribs={"name": "{}_counter_top".format(self.name)},
            return_first=True,
        )
        material.set("texture", tex_name)

        texture = find_elements(
            self.root, tags="texture", attribs={"name": "tex_base"}, return_first=True
        )
        tex_name = get_texture_name_from_file(self.base_texture)
        texture.set("name", tex_name)
        texture.set("file", self.base_texture)
        material = find_elements(
            self.root,
            tags="material",
            attribs={"name": "{}_counter_base".format(self.name)},
            return_first=True,
        )
        material.set("texture", tex_name)

        # need to look at this later, not sure why
        prefix = self.naming_prefix if self.name != "counter" else ""

        # set base color
        if base_color is not None:
            self.base_color = base_color
            base_material = find_elements(
                self.root,
                "material",
                attribs={"name": prefix + "counter_base"},
                return_first=True,
            )

            if len(self.base_color) == 3:
                self.base_color.append(1)
            base_material.set("rgba", a2s(self.base_color))

    def _get_counter_geoms(self):
        """
        searches for geoms corresponding to each of the four components of the counter.
        Currently does not return collision geoms for top because does not account for the chunking!
        """

        geoms = dict()
        for side in SIDES:
            geoms["base" + "_" + side] = list()

        if self.has_opening:
            for side in SIDES:
                geoms["top" + "_" + side] = list()
        else:
            geoms["top"] = list()

        for geom_name in geoms.keys():
            for postfix in ["", "_visual"]:
                g = find_elements(
                    root=self._obj,
                    tags="geom",
                    attribs={"name": "{}_{}{}".format(self.name, geom_name, postfix)},
                    return_first=True,
                )
                geoms[geom_name].append(g)
        return geoms

    def _place_interior_obj(self):
        """
        calculates and sets the position of the sink,
        calculates and returns the sizes of padding around the sink.

        x_percent/y_percent specifies at what percent of the fixture's entire width/depth
        the center of the sink should be placed.

        Returns:
            list: list of padding values [left, right, front, back],
            which are distance between the edge of the counter and (the edge of the interior object + gap)
        """

        x_percent, y_percent = self.obj_x_percent, self.obj_y_percent

        depth_padding = self.overhang + 0.015  # also add the thickness of cabinet doors

        # remove overhang from consideration for placement
        top_size = [
            self.size[0],
            self.size[1] - depth_padding,
            self.size[2],
        ]  # remove the depth_padding

        # respect boundaires: limit range of x_percent and y_percent so interior object doesn't overflow
        gap = 0.02
        max_x_percent = (top_size[0] - gap - self.interior_obj.width / 2) / top_size[0]
        x_percent = np.clip(x_percent, 1 - max_x_percent, max_x_percent)
        max_y_percent = (top_size[1] - gap - self.interior_obj.depth / 2) / top_size[1]
        y_percent = np.clip(y_percent, 1 - max_y_percent, max_y_percent)

        # calculate and set the position of sink
        interior_origin = [
            self.pos[0] + (x_percent - 0.50) * top_size[0],
            self.pos[1] + depth_padding / 2 + (y_percent - 0.50) * top_size[1],
            self.pos[2] + top_size[2] / 2 - self.interior_obj.height / 2,
        ]

        self.interior_obj.set_origin(interior_origin)

        # calculate the size of padding around the sink
        left_pad = x_percent * top_size[0] - self.interior_obj.width / 2
        right_pad = (1 - x_percent) * top_size[0] - self.interior_obj.width / 2
        front_pad = (
            y_percent * top_size[1] - self.interior_obj.depth / 2 + depth_padding
        )  # add the depth_padding to the front
        back_pad = (1 - y_percent) * top_size[1] - self.interior_obj.depth / 2

        return [left_pad, right_pad, front_pad, back_pad]

    def _get_chunks(self, pos, size, chunk_size=0.5):
        """
        Breaks the top width into chunks of size `chunk_size`

        Args:
            pos (np.array): position of the center of the top

            size (np.array): size of the top

            chunk_size (float): size of each chunk
        """
        top_x_sizes = np.array(math.ceil(size[0] / chunk_size) * [chunk_size])
        top_x_sizes[-1] = size[0] - np.sum(top_x_sizes[:-1])

        chunk_sizes = [np.array([x_size, size[1], size[2]]) for x_size in top_x_sizes]
        left_pos = pos - np.array([size[0] / 2, 0, 0])
        chunk_positions = []
        for i in range(len(top_x_sizes)):
            chunk_positions.append(
                np.array(
                    [
                        left_pos[0]
                        + np.sum(top_x_sizes[: max(0, i)])
                        + top_x_sizes[i] / 2,
                        pos[1],
                        pos[2],
                    ]
                )
            )
        return chunk_positions, chunk_sizes

    def _make_counter(self):
        """
        Creates a contigious counter with a top and possibly a base.
        """
        w, d, h = np.array(self.size)
        th = self.th

        geoms = dict()

        # get the base geoms
        for side in SIDES:
            geoms["base" + "_" + side] = list()
        for geom_name in geoms.keys():
            for postfix in ["", "_visual"]:
                g = find_elements(
                    root=self._obj,
                    tags="geom",
                    attribs={"name": "{}_{}{}".format(self.name, geom_name, postfix)},
                    return_first=True,
                )
                geoms[geom_name].append(g)

        # counters with half-top can only either be right or left
        assert sum(self.half_top) < 2
        if sum(self.half_top) == 0:
            pos = np.array([0, 0, h / 2 - th / 2])
            size = np.array([w, d, th])
        else:
            # used for aligning corner counters
            size = np.array([w / 2, d, th])
            if self.half_top[1]:
                # keep right half
                pos = np.array([w / 4, 0, h / 2 - th / 2])
            else:
                # keep left half
                pos = np.array([-w / 4, 0, h / 2 - th / 2])

        half_size = size / 2

        geom_name = self._name + "_top"
        g_vis = new_geom(
            name=geom_name + "_visual",
            type="box",
            size=half_size,
            pos=pos,
            group=1,
            material=self._name + "_counter_top",
            density=10,
            conaffinity=0,
            contype=0,
            mass=1e-8,
        )
        self._obj.append(g_vis)
        # manually update visual geoms registry
        self._visual_geoms.append("top_visual")

        # break the top width into chunks
        chunk_positions, chunk_sizes = self._get_chunks(pos, size, chunk_size=0.5)
        rgba = a2s(np.concatenate((np.random.uniform(low=0, high=1, size=3), [0.5])))
        for i in range(len(chunk_sizes)):
            g = new_geom(
                name=geom_name + "_{}".format(i),
                type="box",
                size=chunk_sizes[i] / 2,
                pos=chunk_positions[i],
                group=0,
                density=10,
                rgba=rgba,
            )
            self._obj.append(g)
            # manually update contact geoms registry
            self._contact_geoms.append("top_{}".format(i))

        base_size, base_pos = self._get_base_dimensions()

        for side in SIDES:
            for elem in geoms["base_{}".format(side)]:
                # remove appropriate part of the base if based on hollow specification
                if side == "front" and self.hollow[1]:
                    self._remove_element(elem)
                elif side == "back" and self.hollow[0]:
                    self._remove_element(elem)
                # houses bottom row cabinets
                # remove right and left base geoms if any of hollow is True
                elif sum(self.base_opening) == 0 and sum(self.hollow) > 0:
                    self._remove_element(elem)
                else:
                    elem.set("pos", a2s(base_pos[side]))
                    elem.set("size", a2s(base_size[side]))

    def _make_counter_with_opening(self, padding):
        """
        calculates and set the size and position of each component in the counter

        coordinate system change may or may not be necessary. currently all other fixtures
        use Mujoco's center-based coordinate system

        Args:
            padding (list): list of padding values [left, right, front, back],

            which are distance between the edge of the counter and (the edge of the interior object + gap)
        """
        w, d, h = self.size
        th = self.th
        side_th = 0.2 if sum(self.base_opening) == 1 else 0.0002
        left_pad, right_pad, front_pad, back_pad = padding

        if front_pad < self.overhang:
            raise ValueError(
                "Overhang value is too large, must be lower "
                "than front padding ({:.2f})".format(front_pad)
            )

        # calculate the (full) size of each component
        top_size = dict(
            back=[self.interior_obj.width, back_pad, h - th],
            front=[self.interior_obj.width, front_pad - self.overhang, h - th],
            left=[left_pad, d - self.overhang, h - th],
            right=[right_pad, d - self.overhang, h - th],
        )

        # all coordinates are the bottom-left corners
        # the origin is the bottom-left corner of the entire fixture
        top_pos = dict(
            back=[left_pad, front_pad + self.interior_obj.depth, -th],
            front=[left_pad, self.overhang, -th],
            left=[0, self.overhang, -th],
            right=[left_pad + self.interior_obj.width, self.overhang, -th],
        )

        base_size = dict(
            back=[w, 0.0002, h - th],
            front=[w, 0.0002, h - th],
            left=[side_th, d - self.overhang, h - th],
            right=[side_th, d - self.overhang, h - th],
        )
        base_pos = dict(
            back=[0.0, d, -th],
            front=[0.0, -d + 2 * self.overhang, -th],
            left=[-w + side_th, self.overhang, -th],
            right=[w - side_th, self.overhang, -th],
        )

        # move panel to middle of the island this will create an opening under the counter in the front/back
        if self.base_opening[0]:
            base_pos["front"] = [0, 0, -th]
        elif self.base_opening[1]:
            base_pos["back"] = [0, 0, -th]

        geoms = dict()
        # get the base geoms
        for side in SIDES:
            geoms["base" + "_" + side] = list()
        for geom_name in geoms.keys():
            for postfix in ["", "_visual"]:
                g = find_elements(
                    root=self._obj,
                    tags="geom",
                    attribs={"name": "{}_{}{}".format(self.name, geom_name, postfix)},
                    return_first=True,
                )
                geoms[geom_name].append(g)

        rgba = a2s(np.concatenate((np.random.uniform(low=0, high=1, size=3), [0.5])))
        for side in SIDES:
            # convert coordinate of bottom-left corner to coordinate of center
            # the origin is now the center of the entire fixture
            top_pos[side][0] = top_pos[side][0] + top_size[side][0] / 2 - w / 2
            top_pos[side][1] = top_pos[side][1] + top_size[side][1] / 2 - d / 2
            top_pos[side][2] /= 2

            base_pos[side] = np.array(base_pos[side]) / 2

            for elem in geoms["base_{}".format(side)]:
                # remove appropriate part of the base, based on hollow specification
                if side == "front" and self.hollow[1]:
                    self._remove_element(elem)
                elif side == "back" and self.hollow[0]:
                    self._remove_element(elem)
                elif sum(self.base_opening) == 0 and sum(self.hollow) > 0:
                    self._remove_element(elem)
                else:
                    elem.set("pos", a2s(base_pos[side]))
                    # divide sizes by 2 according to mujoco convention
                    elem.set("size", a2s(np.array(base_size[side]) / 2))

            # for elem in geoms["top_{}".format(side)]:
            pos = np.array(top_pos[side])
            size = np.array(top_size[side])
            pos[2] = (h - th) / 2
            size[2] = th

            # account for overhang
            if side != "back":
                size[1] += self.overhang
                pos[1] -= self.overhang / 2

            geom_name = self._name + "_top_{}".format(side)

            half_size = size / 2

            g_vis = new_geom(
                name=geom_name + "_visual",
                type="box",
                size=half_size,
                pos=pos,
                group=1,
                material=self._name + "_counter_top",
                density=10,
                conaffinity=0,
                contype=0,
                mass=1e-8,
            )
            self._obj.append(g_vis)
            # manually update visual geoms registry
            self._visual_geoms.append("top_{}_visual".format(side))

            chunk_positions, chunk_sizes = self._get_chunks(pos, size, chunk_size=0.5)
            for i in range(len(chunk_sizes)):
                g = new_geom(
                    name=geom_name + "_{}".format(i),
                    type="box",
                    size=chunk_sizes[i] / 2,
                    pos=chunk_positions[i],
                    group=0,
                    density=10,
                    rgba=rgba,
                )
                self._obj.append(g)
                self._contact_geoms.append("top_{}_{}".format(side, i))

    def _get_base_dimensions(self):
        """
        Returns the size and position of the base components. Considers the base opening specifications

        Returns:
            tuple: tuple of dictionaries containing the size and position of each base component
        """
        # divide everything by 2 per mujoco convention
        x, y, z = np.array(self.size) / 2
        overhang = self.overhang / 2
        th = self.th / 2
        side_th = 0.1 if sum(self.base_opening) == 1 else 0.0001

        base_size = dict(
            back=[x - side_th * 2, th, z - th],
            front=[x - side_th * 2, th, z - th],
            left=[side_th, y - overhang, z - th],
            right=[side_th, y - overhang, z - th],
        )
        base_pos = dict(
            back=[0, y - th, -th],
            front=[0, -y + 2 * overhang + th, -th],
            left=[-x + side_th, overhang, -th],
            right=[x - side_th, overhang, -th],
        )

        if self.base_opening[0]:
            # move panel to middle of the island
            base_pos["front"] = [0, 0, -th]
        elif self.base_opening[1]:
            base_pos["back"] = [0, 0, -th]

        return base_size, base_pos

    # to overwrite Fixture class default
    @property
    def width(self):
        return self.size[0]

    @property
    def depth(self):
        return self.size[1]

    @property
    def height(self):
        return self.size[2]

    def set_pos(self, pos):
        super().set_pos(pos)
        # we have to set the postion of the interior object as well
        if self.interior_obj is not None:
            self._place_interior_obj()

    def get_reset_regions(
        self,
        env,
        ref=None,
        loc="nn",
        top_size=(0.40, 0.25),
        ref_rot_flag=False,
        full_depth_region=False,
    ):
        """
        returns dictionary of reset regions, each region defined as offsets and size

        Args:
            env (Kitchen): the kitchen environment which contains this counter

            ref (str): reference fixture used in determining sampling location

            loc (str): sampling method, one of ["nn", "left", "right", "left_right", "any"]
                        nn: chooses the closest top geom to the reference fixture
                        left: chooses the any top geom within 0.3 distance of the left side of the reference fixture
                        right: chooses the any top geom within 0.3 distance of the right side of the reference fixture
                        left_right: chooses the any top geom within 0.3 distance of the left or right side of the reference fixture
                        any: chooses any top geom

            top_size (tuple): minimum size of the top region to return

            ref_rot_flag (bool): if True, the counter's fixture is rotated by the reference fixture's rotation

            full_depth_region (bool): if True, the island counter will sample regions with full depth accessibility,
                (ie. excliuding stove/sink strip regions), for accessibility


        Returns:
            dict: dictionary of reset regions
        """
        all_geoms = []
        for (k, v) in self._get_counter_geoms().items():
            # only reset on top geoms
            if not k.startswith("top"):
                continue
            geom = v[-1]
            top_pos = s2a(geom.get("pos"))
            this_top_size = s2a(geom.get("size")) * 2
            # make sure region is sufficiently large
            if this_top_size[0] >= top_size[0] and this_top_size[1] >= top_size[1]:
                all_geoms.append(geom)

        is_island_group = hasattr(self, "name") and "island_group" in self.name
        if full_depth_region and is_island_group and len(all_geoms) > 1:
            region_sizes = [s2a(g.get("size")) * 2 for g in all_geoms]
            areas = [sz[0] * sz[1] for sz in region_sizes]
            sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i])
            min_area = areas[sorted_indices[0]]
            next_min_area = areas[sorted_indices[1]] if len(areas) > 1 else min_area
            if min_area < 0.8 * next_min_area:
                all_geoms = [
                    g for i, g in enumerate(all_geoms) if i != sorted_indices[0]
                ]
        reset_regions = {}

        if ref is None:
            geom_i = 0
            for g in all_geoms:
                top_pos = s2a(g.get("pos"))
                top_half_size = s2a(g.get("size"))
                offset = (top_pos[0], top_pos[1], self.size[2] / 2)
                size = (top_half_size[0] * 2, top_half_size[1] * 2)
                reset_regions[f"geom_{geom_i}"] = dict(size=size, offset=offset)
                geom_i += 1
        else:
            ref_fixture = env.get_fixture(ref)

            ### find an appropriate geom to sample ###
            counter_top_geom_info = {}
            for g in all_geoms:
                g_info = {}

                # get global coordinates of geom center and four corners
                g_pos_in_counter_frame = s2a(g.get("pos"))
                g_halfsize = s2a(g.get("size"))

                # use a slightly smaller bounding box
                SC = 0.99
                g_pos = get_pos_after_rel_offset(self, s2a(g.get("pos")))
                g_pos_c1 = get_pos_after_rel_offset(
                    self,
                    g_pos_in_counter_frame
                    + [-SC * g_halfsize[0], -SC * g_halfsize[1], 0],
                )
                g_pos_c2 = get_pos_after_rel_offset(
                    self,
                    g_pos_in_counter_frame
                    + [-SC * g_halfsize[0], SC * g_halfsize[1], 0],
                )
                g_pos_c3 = get_pos_after_rel_offset(
                    self,
                    g_pos_in_counter_frame
                    + [SC * g_halfsize[0], -SC * g_halfsize[1], 0],
                )
                g_pos_c4 = get_pos_after_rel_offset(
                    self,
                    g_pos_in_counter_frame
                    + [SC * g_halfsize[0], SC * g_halfsize[1], 0],
                )

                g_info["pos"] = g_pos
                g_info["pos_c1"] = g_pos_c1
                g_info["pos_c2"] = g_pos_c2
                g_info["pos_c3"] = g_pos_c3
                g_info["pos_c4"] = g_pos_c4
                g_info["halfsize"] = g_halfsize

                ref_fixture_pos = ref_fixture.pos

                x_min = np.min([g_pos_c1[0], g_pos_c2[0], g_pos_c3[0], g_pos_c4[0]])
                x_max = np.max([g_pos_c1[0], g_pos_c2[0], g_pos_c3[0], g_pos_c4[0]])
                y_min = np.min([g_pos_c1[1], g_pos_c2[1], g_pos_c3[1], g_pos_c4[1]])
                y_max = np.max([g_pos_c1[1], g_pos_c2[1], g_pos_c3[1], g_pos_c4[1]])

                fixture_in_2d_bounds = (
                    ref_fixture_pos[0] >= x_min
                    and ref_fixture_pos[0] <= x_max
                    and ref_fixture_pos[1] >= y_min
                    and ref_fixture_pos[1] <= y_max
                )
                g_info["fixture_in_2d_bounds"] = fixture_in_2d_bounds

                if fixture_in_2d_bounds:
                    g_info["dist_to_fixture"] = 0.0
                else:
                    g_info["dist_to_fixture"] = np.min(
                        [
                            project_point_to_segment(
                                ref_fixture_pos, g_pos_c1, g_pos_c2
                            )[1],
                            project_point_to_segment(
                                ref_fixture_pos, g_pos_c2, g_pos_c4
                            )[1],
                            project_point_to_segment(
                                ref_fixture_pos, g_pos_c3, g_pos_c4
                            )[1],
                            project_point_to_segment(
                                ref_fixture_pos, g_pos_c1, g_pos_c3
                            )[1],
                        ]
                    )

                rel_offset = get_fixture_to_point_rel_offset(ref_fixture, g_pos)
                g_info["fixture_to_geom_rel_offset"] = rel_offset

                counter_top_geom_info[g] = g_info

            valid_geoms = []

            geom_containing_fixture = None
            for g, g_info in counter_top_geom_info.items():
                fixture_in_2d_bounds = g_info["fixture_in_2d_bounds"]
                if fixture_in_2d_bounds:
                    geom_containing_fixture = g
                    break

            if loc == "any":
                valid_geoms = all_geoms
            elif loc == "nn":
                min_dist = None
                chosen_top = None
                for g, g_info in counter_top_geom_info.items():
                    g_dist = g_info["dist_to_fixture"]
                    if min_dist is None or g_dist < min_dist:
                        chosen_top = g
                        min_dist = g_dist
                valid_geoms.append(chosen_top)
            elif loc in ["left_right", "right", "left"]:
                if geom_containing_fixture is not None:
                    valid_geoms.append(g)
                else:
                    # add all regions to the left
                    if loc in ["left_right", "left"]:
                        min_dist = None
                        chosen_top = None
                        for g, g_info in counter_top_geom_info.items():
                            offset = g_info["fixture_to_geom_rel_offset"]
                            fixture_in_2d_bounds = g_info["fixture_in_2d_bounds"]
                            g_dist = g_info["dist_to_fixture"]
                            if fixture_in_2d_bounds:
                                valid_geoms.append(g)
                            elif offset[0] < -0.30 and (
                                min_dist is None or g_dist < min_dist
                            ):
                                chosen_top = g
                                min_dist = g_dist
                        if chosen_top is not None:
                            valid_geoms.append(chosen_top)

                    # add all regions to the right
                    if loc in ["left_right", "right"]:
                        min_dist = None
                        chosen_top = None
                        for g, g_info in counter_top_geom_info.items():
                            offset = g_info["fixture_to_geom_rel_offset"]
                            fixture_in_2d_bounds = g_info["fixture_in_2d_bounds"]
                            g_dist = g_info["dist_to_fixture"]
                            if fixture_in_2d_bounds:
                                valid_geoms.append(g)
                            elif offset[0] > 0.30 and (
                                min_dist is None or g_dist < min_dist
                            ):
                                chosen_top = g
                                min_dist = g_dist
                        if chosen_top is not None:
                            valid_geoms.append(chosen_top)
            else:
                raise ValueError

            geom_i = 0

            if len(valid_geoms) > 0:
                # remove duplicates as needed
                valid_geoms = list(set(valid_geoms))

            for g in valid_geoms:
                top_pos = s2a(g.get("pos"))
                top_half_size = s2a(g.get("size"))
                offset = [top_pos[0], top_pos[1], self.size[2] / 2]
                size = [top_half_size[0] * 2, top_half_size[1] * 2]

                if (
                    loc in ["left", "right", "left_right"]
                    and geom_containing_fixture is not None
                ):
                    # set size and offset appropriately
                    fixture_size = ref_fixture.size

                    g_pos_c1 = counter_top_geom_info[g]["pos_c1"]
                    g_pos_c3 = counter_top_geom_info[g]["pos_c3"]

                    point_to_fixture = get_fixture_to_point_rel_offset(
                        ref_fixture, g_pos_c1, rot=self.rot
                    )
                    fixture_x = np.abs(point_to_fixture[0])

                    if loc in ["left", "left_right"]:
                        x1 = top_pos[0] - top_half_size[0]
                        x2 = (
                            top_pos[0]
                            - top_half_size[0]
                            + fixture_x
                            - fixture_size[0] / 2
                        )
                        this_offset = [np.mean((x1, x2)), offset[1], offset[2]]
                        this_size = [x2 - x1, size[1]]
                        if this_size[0] > 0.20:
                            reset_regions[f"geom_{geom_i}"] = dict(
                                size=this_size, offset=this_offset
                            )
                            geom_i += 1

                    if loc in ["right", "left_right"]:
                        x1 = (
                            top_pos[0]
                            - top_half_size[0]
                            + fixture_x
                            + fixture_size[0] / 2
                        )
                        x2 = top_pos[0] + top_half_size[0]
                        this_offset = [np.mean((x1, x2)), offset[1], offset[2]]
                        this_size = [x2 - x1, size[1]]
                        if this_size[0] > 0.20:
                            reset_regions[f"geom_{geom_i}"] = dict(
                                size=this_size, offset=this_offset
                            )
                            geom_i += 1
                else:
                    min_x = top_pos[0] - top_half_size[0]
                    max_x = top_pos[0] + top_half_size[0]

                    ref_pos, _ = get_rel_transform(self, ref_fixture)
                    if ref_rot_flag is False:
                        if min_x <= ref_pos[0] <= max_x:
                            new_size = min(ref_pos[0] - min_x, max_x - ref_pos[0]) * 2
                            # ensure that new size is not too small
                            if new_size >= 0.20:
                                # TODO: recompute so that region is adjcent to ref fixture
                                offset[0] = ref_pos[0]
                                size[0] = new_size

                    reset_regions[f"geom_{geom_i}"] = dict(size=size, offset=offset)
                    geom_i += 1

        return reset_regions
