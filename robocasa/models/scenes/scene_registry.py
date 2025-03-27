from collections import OrderedDict
from enum import IntEnum
from robosuite.utils.mjcf_utils import xml_path_completion
import robocasa
import numpy as np


class LayoutType(IntEnum):
    """
    Enum for available layouts in RoboCasa environment
    """

    LAYOUT0 = 0
    LAYOUT1 = 1
    LAYOUT2 = 2
    LAYOUT3 = 3
    LAYOUT4 = 4
    LAYOUT5 = 5
    LAYOUT6 = 6
    LAYOUT7 = 7
    LAYOUT8 = 8
    LAYOUT9 = 9

    LAYOUT101 = 101
    LAYOUT102 = 102
    LAYOUT103 = 103
    LAYOUT104 = 104
    LAYOUT105 = 105
    LAYOUT106 = 106
    LAYOUT107 = 107
    LAYOUT108 = 108
    LAYOUT109 = 109
    LAYOUT110 = 110
    LAYOUT111 = 111
    LAYOUT112 = 112
    LAYOUT113 = 113
    LAYOUT114 = 114
    LAYOUT115 = 115
    LAYOUT116 = 116
    LAYOUT117 = 117
    LAYOUT118 = 118
    LAYOUT119 = 119
    LAYOUT120 = 120
    LAYOUT121 = 121
    LAYOUT122 = 122
    LAYOUT123 = 123
    LAYOUT124 = 124
    LAYOUT125 = 125
    LAYOUT126 = 126
    LAYOUT127 = 127
    LAYOUT128 = 128
    LAYOUT129 = 129
    LAYOUT130 = 130
    LAYOUT131 = 131
    LAYOUT132 = 132
    LAYOUT133 = 133
    LAYOUT134 = 134
    LAYOUT135 = 135
    LAYOUT136 = 136
    LAYOUT137 = 137
    LAYOUT138 = 138
    LAYOUT139 = 139
    LAYOUT140 = 140
    LAYOUT141 = 141
    LAYOUT142 = 142
    LAYOUT143 = 143
    LAYOUT144 = 144
    LAYOUT145 = 145
    LAYOUT146 = 146
    LAYOUT147 = 147
    LAYOUT148 = 148
    LAYOUT149 = 149
    LAYOUT150 = 150

    # negative values correspond to groups (see LAYOUT_GROUPS_TO_IDS)
    ALL = -1
    NO_ISLAND = -2
    ISLAND = -3
    DINING = -4


LAYOUT_GROUPS_TO_IDS = {
    -1: list(range(10)),  # all
    -2: [0, 2, 4, 5, 7],  # no island
    -3: [1, 3, 6, 8, 9],  # island
    -4: [1, 3, 6, 7, 8, 9],  # dining
    -20: list(range(101, 121)),
    -35: list(range(101, 136)),
}


class StyleType(IntEnum):
    """
    Enums for available styles in RoboCasa environment
    """

    INDUSTRIAL = 0
    SCANDANAVIAN = 1
    COASTAL = 2
    MODERN_1 = 3
    MODERN_2 = 4
    TRADITIONAL_1 = 5
    TRADITIONAL_2 = 6
    FARMHOUSE = 7
    RUSTIC = 8
    MEDITERRANEAN = 9
    TRANSITIONAL_1 = 10
    TRANSITIONAL_2 = 11

    # negative values correspond to groups
    ALL = -1


STYLE_GROUPS_TO_IDS = {
    -1: list(range(12)),  # all
}


def get_layout_path(layout_id):
    """
    Get corresponding blueprint filepath (yaml) for a layout

    Args:
        layout_id (int or LayoutType): layout id (int or enum)

    Return:
        str: yaml path for specified layout
    """
    if (
        isinstance(layout_id, int)
        or isinstance(layout_id, np.int64)
        or isinstance(layout_id, np.int32)
    ):
        layout_int_to_name = dict(
            map(lambda item: (item.value, item.name.lower()), LayoutType)
        )
        layout_name = layout_int_to_name[layout_id]
    elif isinstance(layout_id, LayoutType):
        layout_name = layout_id.name.lower()
    else:
        raise ValueError

    # special case: if name starts with one letter, capitalize it
    if layout_name[1] == "_":
        layout_name = layout_name.capitalize()

    return xml_path_completion(
        f"scenes/kitchen_layouts/{layout_name}.yaml",
        root=robocasa.models.assets_root,
    )


def get_style_path(style_id):
    """
    Get corresponding blueprint filepath (yaml) for a style

    Args:
        style_id (int or StyleType): style id (int or enum)

    Return:
        str: yaml path for specified style
    """
    if (
        isinstance(style_id, int)
        or isinstance(style_id, np.int64)
        or isinstance(style_id, np.int32)
    ):
        style_int_to_name = dict(
            map(lambda item: (item.value, item.name.lower()), StyleType)
        )
        style_name = style_int_to_name[style_id]
    elif isinstance(style_id, StyleType):
        style_name = style_id.name.lower()
    else:
        raise ValueError

    return xml_path_completion(
        f"scenes/kitchen_styles/{style_name}.yaml",
        root=robocasa.models.assets_root,
    )


def unpack_layout_ids(layout_ids):
    if layout_ids is None:
        layout_ids = LayoutType.ALL

    if not isinstance(layout_ids, list):
        layout_ids = [layout_ids]

    all_layout_ids = []
    for id in layout_ids:
        if isinstance(id, dict):
            all_layout_ids.append(id)
        else:
            id = int(id)
            if id < 0:
                all_layout_ids += LAYOUT_GROUPS_TO_IDS[id]
            else:
                all_layout_ids.append(id)
    return all_layout_ids


def unpack_style_ids(style_ids):
    if style_ids is None:
        style_ids = StyleType.ALL

    if not isinstance(style_ids, list):
        style_ids = [style_ids]

    all_style_ids = []
    for id in style_ids:
        if isinstance(id, dict):
            all_style_ids.append(id)
        else:
            id = int(id)
            if id < 0:
                all_style_ids += STYLE_GROUPS_TO_IDS[id]
            else:
                all_style_ids.append(id)
    return all_style_ids
