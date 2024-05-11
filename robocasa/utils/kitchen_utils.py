
LAYOUTS = {
    0: "kitchen_layouts/one_wall_small_arena.yaml",
    1: "kitchen_layouts/one_wall_large_arena.yaml",
    2: "kitchen_layouts/L_shaped_small_arena.yaml",
    3: "kitchen_layouts/L_shaped_large_arena.yaml",
    4: "kitchen_layouts/galley_arena.yaml",
    5: "kitchen_layouts/U_shaped_small_arena.yaml",
    6: "kitchen_layouts/U_shaped_large_arena.yaml",
    7: "kitchen_layouts/G_shaped_small_arena.yaml",
    8: "kitchen_layouts/G_shaped_large_arena.yaml",
    9: "kitchen_layouts/wraparound_arena.yaml",

    -1: "kitchen_layouts/playground.yaml",
}

# default free cameras for different kitchen layouts
LAYOUT_CAMS = {
    0: dict(
        lookat=[2.26593463, -1.00037131, 1.38769295],
        distance=3.0505089839567323,
        azimuth=90.71563812375285,
        elevation=-12.63948837207208
    ),
    1: dict(
        lookat=[2.66147999, -1.00162429, 1.2425155],
        distance=3.7958766287746255,
        azimuth=89.75784013699234,
        elevation=-15.177406642875091,
    ),
    2: dict(
        lookat=[3.02344359, -1.48874618, 1.2412914],
        distance=3.6684844368165512,
        azimuth=51.67880851867874,
        elevation=-13.302619131542388,
    ),
    3: dict(
        lookat=[11.44842548, -11.47664723, 11.24115989],
        distance=43.923271794728187,
        azimuth=227.12928449329333,
        elevation=-16.495686334624907,
    ),
    4: dict(
        lookat=[1.6, -1.0, 1.0],
        distance=5,
        azimuth=89.70301806083651,
        elevation=-18.02177994296577,
    ),
}

DEFAULT_LAYOUT_CAM = {
    "lookat": [2.25, -1, 1.05312667],
    "distance": 5,
    "azimuth": 89.70301806083651,
    "elevation": -18.02177994296577,
}

SCENE_SPLITS = {
    "5_scenes": [[3, 0], [0, 4], [4, 5], [5, 4], [6, 1]], # split for sampling from 5 percent of kitchens
}

CAM_CONFIGS = dict(
    robot0_agentview_center=dict(
        pos=[-0.6, 0.0, 1.15],
        quat=[0.636945903301239, 0.3325185477733612, -0.3199238181114197, -0.6175596117973328],
        parent_body="base0_support",
    ),
    robot0_agentview_left=dict(
        pos=[-0.5, 0.35, 1.05],
        quat=[0.55623853, 0.29935253, -0.37678665, -0.6775092],
        camera_attribs=dict(fovy="60"),
        parent_body="base0_support",
    ),
    robot0_agentview_right=dict(
        pos=[-0.5, -0.35, 1.05],
        quat=[0.6775091886520386, 0.3767866790294647, -0.2993525564670563, -0.55623859167099],
        camera_attribs=dict(fovy="60"),
        parent_body="base0_support",
    ),
    robot0_frontview=dict(
        pos=[-0.50, 0, 0.95],
        quat=[0.6088936924934387, 0.3814677894115448, -0.3673907518386841, -0.5905545353889465],
        camera_attribs=dict(fovy="60"),
        parent_body="base0_support",
    ),

    robot0_eye_in_hand=dict(
        pos=[0.05, 0, 0],
        quat=[0, 0.707107, 0.707107, 0],
        parent_body="robot0_right_hand",
    ),
)