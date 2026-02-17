import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import (
    array_to_string,
    string_to_array,
    xml_path_completion,
)

from robocasa.models.objects.objects import MJCFObject
from scipy.spatial.transform import Rotation as R
import mujoco


def obj_inside_of(env, obj_name, fixture_id, partial_check=False, th=0.05):
    """
    whether an object (another mujoco object) is inside of fixture. applies for most fixtures
    """
    from robocasa.models.fixtures import Fixture

    obj = env.objects[obj_name]
    fixture = env.get_fixture(fixture_id)
    assert isinstance(obj, MJCFObject)
    assert isinstance(fixture, Fixture)

    # step 1: calculate fxiture points
    fixtr_int_regions = fixture.get_int_sites(relative=False)
    for (fixtr_p0, fixtr_px, fixtr_py, fixtr_pz) in fixtr_int_regions.values():
        u = fixtr_px - fixtr_p0
        v = fixtr_py - fixtr_p0
        w = fixtr_pz - fixtr_p0

        # get the position and quaternion of object
        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
        obj_quat = T.convert_quat(
            env.sim.data.body_xquat[env.obj_body_id[obj.name]], to="xyzw"
        )

        if partial_check:
            obj_points_to_check = [obj_pos]
            th = 0.0
        else:
            # calculate 8 boundary points of object
            obj_points_to_check = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)

        inside_of = True
        for obj_p in obj_points_to_check:
            check1 = (
                np.dot(u, fixtr_p0) - th <= np.dot(u, obj_p) <= np.dot(u, fixtr_px) + th
            )
            check2 = (
                np.dot(v, fixtr_p0) - th <= np.dot(v, obj_p) <= np.dot(v, fixtr_py) + th
            )
            check3 = (
                np.dot(w, fixtr_p0) - th <= np.dot(w, obj_p) <= np.dot(w, fixtr_pz) + th
            )

            if not (check1 and check2 and check3):
                inside_of = False
                break

        if inside_of is True:
            return True

    return False


# used for cabinets, cabinet panels, counters, etc.
def set_geom_dimensions(sizes, positions, geoms, rotated=False):
    """
    set the dimensions of geoms in a fixture

    Args:
        sizes (dict): dictionary of sizes for each side

        positions (dict): dictionary of positions for each side

        geoms (dict): dictionary of geoms for each side

        rotated (bool): whether the fixture is rotated. Fixture may be rotated to make texture appear uniform
                        due to mujoco texture conventions
    """
    if rotated:
        # rotation trick to make texture appear uniform
        # see .xml file
        for side in sizes.keys():
            # not the best detection method, TODO: check euler
            if "door" in side or "trim" in side:
                sizes[side] = [sizes[side][i] for i in [0, 2, 1]]

    # set sizes and positions of all geoms
    for side in positions.keys():
        for geom in geoms[side]:
            if geom is None:
                raise ValueError("Did not find geom:", side)
            geom.set("pos", array_to_string(positions[side]))
            geom.set("size", array_to_string(sizes[side]))


def get_rel_transform(fixture_A, fixture_B):
    """
    Gets fixture_B's position and rotation relative to fixture_A's frame
    """
    A_trans = np.array(fixture_A.pos)
    B_trans = np.array(fixture_B.pos)

    A_rot = np.array([0, 0, fixture_A.rot])
    B_rot = np.array([0, 0, fixture_B.rot])

    A_mat = T.euler2mat(A_rot)
    B_mat = T.euler2mat(B_rot)

    T_WA = np.vstack((np.hstack((A_mat, A_trans[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((B_mat, B_trans[:, None])), [0, 0, 0, 1]))

    T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

    return T_AB[:3, 3], T_AB[:3, :3]


def transform_global_to_local(global_x, global_y, rot):
    """
    Transforms a global movement vector from a global frame (with rotation `rot`)
    into local coordinates (assumed to be rotation 0).
    Args:
        global_x (float): Movement along local x-axis
        global_y (float): Movement along local y-axis
        rot (float): Rotation of the local frame (in radians)
    Returns:
        (float, float): Transformed (local_x, local_y)
    """
    cos_yaw = np.cos(rot)
    sin_yaw = np.sin(rot)

    local_x = cos_yaw * global_x - sin_yaw * global_y
    local_y = sin_yaw * global_x + cos_yaw * global_y

    return local_x, local_y


def compute_rel_transform(A_pos, A_mat, B_pos, B_mat):
    """
    Gets B's position and rotation relative to A's frame
    """
    T_WA = np.vstack((np.hstack((A_mat, A_pos[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((B_mat, B_pos[:, None])), [0, 0, 0, 1]))

    T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

    return T_AB[:3, 3], T_AB[:3, :3]


def get_fixture_to_point_rel_offset(fixture, point, rot=None):
    """
    get offset relative to fixture's frame, given a global point
    """
    global_offset = point - fixture.pos
    if rot is None:
        rot = fixture.rot
    T_WF = T.euler2mat([0, 0, rot])
    rel_offset = np.matmul(np.linalg.inv(T_WF), global_offset)
    return rel_offset


def get_pos_after_rel_offset(fixture, offset):
    """
    get global position of a fixture, after applying offset relative to center of fixture
    """
    fixture_rot = np.array([0, 0, fixture.rot])
    fixture_mat = T.euler2mat(fixture_rot)

    return fixture.pos + np.dot(fixture_mat, offset)


def project_point_to_line(P, A, B):
    """
    logic copied from here: https://stackoverflow.com/a/61342198
    """
    AP = P - A
    AB = B - A
    result = A + np.dot(AP, AB) / np.dot(AB, AB) * AB

    return result


def project_point_to_segment(point, seg_start, seg_end):
    """
    Projects a point onto a line segment, and clamps it to the segment bounds.

    Args:
        point (np.ndarray): The point to project, shape (2,) or (3,)
        seg_start (np.ndarray): Start point of the segment, same shape as point
        seg_end (np.ndarray): End point of the segment, same shape as point

    Returns:
        np.ndarray: The closest point on the segment to the input point
        float: The distance to the segment

    From ChatGPT
    """
    # Convert to numpy arrays in case they aren't already
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq == 0.0:
        # Segment is a point
        return seg_start, 0.0

    # Compute t: projection factor along the segment
    t = np.dot(point - seg_start, seg_vec) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)

    # Compute the closest point
    closest_point = seg_start + t_clamped * seg_vec
    dist = np.linalg.norm(point - closest_point)

    return closest_point, dist


def point_in_fixture(point, fixture, only_2d=False):
    """
    check if point is inside of the exterior bounding boxes of the fixture

    Args:
        point (np.array): point to check

        fixture (Fixture): fixture object

        only_2d (bool): whether to check only in 2D
    """
    p0, px, py, pz = fixture.get_ext_sites(relative=False)
    th = 0.00
    u = px - p0
    v = py - p0
    w = pz - p0
    check1 = np.dot(u, p0) - th <= np.dot(u, point) <= np.dot(u, px) + th
    check2 = np.dot(v, p0) - th <= np.dot(v, point) <= np.dot(v, py) + th
    check3 = np.dot(w, p0) - th <= np.dot(w, point) <= np.dot(w, pz) + th

    if only_2d:
        return check1 and check2
    else:
        return check1 and check2 and check3


def obj_in_region(
    obj,
    obj_pos,
    obj_quat,
    p0,
    px,
    py,
    pz=None,
):
    """
    check if object is in the region defined by the points.
    Uses either the objects bounding box or the object's horizontal radius
    """
    from robocasa.models.fixtures import Fixture

    if isinstance(obj, MJCFObject) or isinstance(obj, Fixture):
        obj_points = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
    else:
        radius = obj.horizontal_radius
        obj_points = obj_pos + np.array(
            [
                [radius, 0, 0],
                [-radius, 0, 0],
                [0, radius, 0],
                [0, -radius, 0],
            ]
        )

    u = px - p0
    v = py - p0
    w = pz - p0 if pz is not None else None

    for point in obj_points:
        check1 = np.dot(u, p0) <= np.dot(u, point) <= np.dot(u, px)
        check2 = np.dot(v, p0) <= np.dot(v, point) <= np.dot(v, py)

        if not check1 or not check2:
            return False

        if w is not None:
            check3 = np.dot(w, p0) <= np.dot(w, point) <= np.dot(w, pz)
            if not check3:
                return False

    return True


def fixture_pairwise_dist(f1, f2):
    """
    Gets the distance between two fixtures by finding the minimum distance between their exterior bounding box points
    """
    f1_points = f1.get_ext_sites(all_points=True, relative=False)
    f2_points = f2.get_ext_sites(all_points=True, relative=False)

    all_dists = [np.linalg.norm(p1 - p2) for p1 in f1_points for p2 in f2_points]
    return np.min(all_dists)


def obj_fixture_bbox_min_dist(env, obj_name, fixture):
    """
    Gets the minimum distance between a fixture and an object by computing the minimal axis-aligned bounding separation.
    """
    fix_pts = fixture.get_ext_sites(all_points=True, relative=False)
    fix_coords = np.array(fix_pts)
    fix_min = fix_coords.min(axis=0)
    fix_max = fix_coords.max(axis=0)

    body_id = env.obj_body_id[obj_name]
    trans = env.sim.data.body_xpos[body_id]
    rot_quat = env.sim.data.body_xquat[body_id]

    obj = env.objects[obj_name]
    obj_pts = obj.get_bbox_points(trans=trans, rot=rot_quat)
    obj_coords = np.array(obj_pts)
    obj_min = obj_coords.min(axis=0)
    obj_max = obj_coords.max(axis=0)

    sep = np.zeros(3)
    for i, axis in enumerate(["x", "y", "z"]):
        if fix_max[i] < obj_min[i]:
            sep[i] = obj_min[i] - fix_max[i]
        elif obj_max[i] < fix_min[i]:
            sep[i] = fix_min[i] - obj_max[i]
        else:
            sep[i] = 0.0

    return np.linalg.norm(sep)


def check_fxtr_contact(env, pos):
    """
    Check if the point is in contact with any fixture

    Args:
        pos (tuple): The position of the point to check

    Returns:
        bool: True if the point is in contact with any fixture, False otherwise
    """
    from robocasa.models.fixtures import (
        Counter,
        Stove,
        Stovetop,
        HousingCabinet,
        SingleCabinet,
        HingeCabinet,
        Fridge,
        Wall,
        Floor,
    )

    fxtrs = [
        fxtr
        for fxtr in env.fixtures.values()
        if isinstance(fxtr, Counter)
        or isinstance(fxtr, Stove)
        or isinstance(fxtr, Stovetop)
        or isinstance(fxtr, HousingCabinet)
        or isinstance(fxtr, SingleCabinet)
        or isinstance(fxtr, HingeCabinet)
        or isinstance(fxtr, Fridge)
        or (isinstance(fxtr, Wall) and not isinstance(fxtr, Floor))
    ]

    for fxtr in fxtrs:
        # get bounds of fixture
        if point_in_fixture(point=pos, fixture=fxtr, only_2d=True):
            return True
    return False


def point_outside_scene(env, pos):
    from robocasa.models.fixtures import Floor

    walls = [fxtr for (name, fxtr) in env.fixtures.items() if isinstance(fxtr, Floor)]
    return not any(
        [point_in_fixture(point=pos, fixture=wall, only_2d=True) for wall in walls]
    )


def check_obj_scrubbed(env, sponge_name, obj_name):
    """
    Determine if the sponge is scrubbing the object by checking contact and movement.
    """
    # Check if sponge is in contact with bowl
    in_contact = check_obj_in_receptacle(env, sponge_name, obj_name)

    if not in_contact:
        return False

    sponge_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[sponge_name]])
    prev_sponge_pos = getattr(env, "prev_sponge_pos", sponge_pos)

    movement_vector = sponge_pos - prev_sponge_pos

    in_contact = env.check_contact(env.objects[sponge_name], env.objects[obj_name])

    sponge_still_inside = check_obj_in_receptacle(env, sponge_name, obj_name)
    env.prev_sponge_pos = sponge_pos

    scrubbing = (
        in_contact and sponge_still_inside and np.linalg.norm(movement_vector) > 0.001
    )

    return scrubbing


def object_contact_with_liquid(env, obj_name, liquid_receptacle_name):
    key = f"{liquid_receptacle_name}_liquid"
    liquid_pos = None
    liquid_size = None
    cylinder = False
    if key in env.sim.model._geom_name2id:
        liquid_geom_id = env.sim.model.geom_name2id(key)
        liquid_pos = env.sim.data.geom_xpos[liquid_geom_id]
        liquid_size = env.sim.model.geom_size[liquid_geom_id]
        liquid_mat = env.sim.data.geom_xmat[liquid_geom_id]
        cylinder = (
            env.sim.model.geom_type[liquid_geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
    else:
        liquid_site_id = env.sim.model.site_name2id(key)
        liquid_pos = env.sim.data.site_xpos[liquid_site_id]
        liquid_size = env.sim.model.site_size[liquid_site_id]
        liquid_mat = env.sim.data.site_xmat[liquid_site_id]
        cylinder = (
            env.sim.model.site_type[liquid_site_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        )

    if cylinder:
        liquid_size = [liquid_size[0], liquid_size[0], liquid_size[1]]

    obj_pos = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
    obj_quat = T.convert_quat(
        env.sim.data.body_xquat[env.obj_body_id[obj_name]], to="xyzw"
    )
    liquid_quat = T.mat2quat(
        np.asarray(liquid_mat.copy(), dtype=np.float32).reshape(3, 3)
    )

    obj_bbox = np.array(
        env.objects[obj_name].get_bbox_points(trans=obj_pos, rot=obj_quat)
    )
    liquid_bbox = get_rotated_bbox_points(liquid_pos, liquid_quat, liquid_size)

    return objs_intersect_bbox(obj_bbox, liquid_bbox)


def get_rotated_bbox_points(obj_pos, rot, half_size, bbox_center=None):
    if bbox_center is None:
        bbox_center = np.array([0, 0, 0])
    bbox_offsets = [
        bbox_center + half_size * np.array([-1, -1, -1]),  # p0
        bbox_center + half_size * np.array([1, -1, -1]),  # px
        bbox_center + half_size * np.array([-1, 1, -1]),  # py
        bbox_center + half_size * np.array([-1, -1, 1]),  # pz
        bbox_center + half_size * np.array([1, 1, 1]),
        bbox_center + half_size * np.array([-1, 1, 1]),
        bbox_center + half_size * np.array([1, -1, 1]),
        bbox_center + half_size * np.array([1, 1, -1]),
    ]

    if rot is not None:
        rot = T.quat2mat(rot)
    else:
        rot = np.eye(3)
    points = [(np.matmul(rot, p)) + obj_pos for p in bbox_offsets]
    return points


def objs_intersect_bbox(obj_points, other_obj_points):
    face_normals = [
        obj_points[1] - obj_points[0],
        obj_points[2] - obj_points[0],
        obj_points[3] - obj_points[0],
        other_obj_points[1] - other_obj_points[0],
        other_obj_points[2] - other_obj_points[0],
        other_obj_points[3] - other_obj_points[0],
    ]

    intersect = True

    # noramlize length of normals
    for normal in face_normals:
        normal = np.array(normal) / np.linalg.norm(normal)

        obj_projs = [np.dot(p, normal) for p in obj_points]
        other_obj_projs = [np.dot(p, normal) for p in other_obj_points]

        # see if gap detected
        if np.min(other_obj_projs) > np.max(obj_projs) or np.min(obj_projs) > np.max(
            other_obj_projs
        ):
            intersect = False
            break
    return intersect


def objs_intersect(
    obj,
    obj_pos,
    obj_quat,
    other_obj,
    other_obj_pos,
    other_obj_quat,
):
    """
    check if two objects intersect
    """
    from robocasa.models.fixtures import Fixture

    bbox_check = (isinstance(obj, MJCFObject) or isinstance(obj, Fixture)) and (
        isinstance(other_obj, MJCFObject) or isinstance(other_obj, Fixture)
    )
    if bbox_check:
        obj_points = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
        other_obj_points = other_obj.get_bbox_points(
            trans=other_obj_pos, rot=other_obj_quat
        )

        face_normals = [
            obj_points[1] - obj_points[0],
            obj_points[2] - obj_points[0],
            obj_points[3] - obj_points[0],
            other_obj_points[1] - other_obj_points[0],
            other_obj_points[2] - other_obj_points[0],
            other_obj_points[3] - other_obj_points[0],
        ]

        intersect = True

        # noramlize length of normals
        for normal in face_normals:
            normal = np.array(normal) / np.linalg.norm(normal)

            obj_projs = [np.dot(p, normal) for p in obj_points]
            other_obj_projs = [np.dot(p, normal) for p in other_obj_points]

            # see if gap detected
            if np.min(other_obj_projs) > np.max(obj_projs) or np.min(
                obj_projs
            ) > np.max(other_obj_projs):
                intersect = False
                break
    else:
        """
        old code from placement_samplers.py
        """
        obj_x, obj_y, obj_z = obj_pos
        other_obj_x, other_obj_y, other_obj_z = other_obj_pos
        xy_collision = (
            np.linalg.norm((obj_x - other_obj_x, obj_y - other_obj_y))
            <= other_obj.horizontal_radius + obj.horizontal_radius
        )
        if obj_z > other_obj_z:
            z_collision = (
                obj_z - other_obj_z <= other_obj.top_offset[-1] - obj.bottom_offset[-1]
            )
        else:
            z_collision = (
                other_obj_z - obj_z <= obj.top_offset[-1] - other_obj.bottom_offset[-1]
            )

        if xy_collision and z_collision:
            intersect = True
        else:
            intersect = False

    return intersect


def normalize_joint_value(raw, joint_min, joint_max):
    """
    normalize raw value to be between 0 and 1
    """
    return (raw - joint_min) / (joint_max - joint_min)


def check_obj_in_receptacle(env, obj_name, receptacle_name, th=None):
    """
    check if object is in receptacle object based on threshold
    """
    obj = env.objects[obj_name]
    recep = env.objects[receptacle_name]
    obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]])
    recep_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[receptacle_name]])
    if th is None:
        th = recep.horizontal_radius * 0.7
    obj_in_recep = (
        env.check_contact(obj, recep)
        and np.linalg.norm(obj_pos[:2] - recep_pos[:2]) < th
    )
    return obj_in_recep


def check_obj_upright(env, obj_name, th=15):
    obj_rot = env.sim.data.xquat[env.obj_body_id[obj_name]]
    r = R.from_quat([obj_rot[1], obj_rot[2], obj_rot[3], obj_rot[0]])
    obj_rot_euler = r.as_euler("xyz", degrees=True)
    obj_upright = abs(obj_rot_euler[1]) < th and abs(obj_rot_euler[0]) < th
    return obj_upright


def check_fxtr_upright(env, fixture_name, th=15):
    """
    Check if the fixture is upright based on its rotation.
    """
    fixture_rot = env.sim.data.get_body_xquat(fixture_name)
    r = R.from_quat([fixture_rot[1], fixture_rot[2], fixture_rot[3], fixture_rot[0]])
    fixture_rot_euler = r.as_euler("xyz", degrees=True)
    fixture_upright = abs(fixture_rot_euler[1]) < th and abs(fixture_rot_euler[0]) < th
    return fixture_upright


def check_obj_fixture_contact(env, obj_name, fixture_name):
    """
    check if object is in contact with fixture
    """
    obj = env.objects[obj_name]
    fixture = env.get_fixture(fixture_name)
    return env.check_contact(obj, fixture)


def check_obj_any_counter_contact(env, obj_name):
    from robocasa.models.fixtures import Counter

    """
    check if the object is in contact with any counter fixture in the environment.
    """
    for fixture in env.fixtures.values():
        if isinstance(fixture, Counter):
            if check_obj_fixture_contact(env, obj_name, fixture):
                return True
    return False


def gripper_obj_far(env, obj_name="obj", th=0.25):
    """
    check if gripper is far from object based on distance defined by threshold
    """
    obj_pos = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
    gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    gripper_obj_far = np.linalg.norm(gripper_site_pos - obj_pos) > th
    return gripper_obj_far


def gripper_fxtr_far(env, fixture_name, th=0.25):
    """
    check if gripper is far from fixture based on distance defined by threshold
    """
    fixture_pos = env.sim.data.get_body_xpos(fixture_name)
    gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    gripper_fxtr_far = np.linalg.norm(gripper_site_pos - fixture_pos) > th
    return gripper_fxtr_far


def check_obj_grasped(env, obj_name, threshold=0.035):
    """
    Check if the gripper has grasped the object by analyzing contact and proximity
    """
    obj = env.objects[obj_name]
    robot = env.robots[0]

    gripper_joints = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
    gripper_joint_positions = [
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint)]
        for joint in gripper_joints
    ]

    gripper_closed = all(pos < threshold for pos in gripper_joint_positions)

    if "right" in robot.gripper:
        gripper = robot.gripper["right"]
    else:
        raise AttributeError("Gripper dictionary does not contain a 'right' key.")

    return env.check_contact(gripper, obj) and gripper_closed


def obj_cos(env, obj_name="obj", ref=(0, 0, 1)):
    def cos(u, v):
        return np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-10)

    obj_id = env.obj_body_id[obj_name]
    obj_quat = T.convert_quat(np.array(env.sim.data.body_xquat[obj_id]), to="xyzw")
    obj_mat = T.quat2mat(obj_quat)

    return cos(obj_mat[:, 2], np.array(ref))


def quaternion_to_euler(env, quat):
    """
    Converts a quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat (array-like): Quaternion [x, y, z, w]

    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    rotation = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
    return rotation.as_euler("xyz", degrees=False)


def get_obj_lang(env, obj_name="obj", get_preposition=False):
    """
    gets a formatted language string for the object (replaces underscores with spaces)

    Args:
        obj_name (str): name of object
        get_preposition (bool): if True, also returns preposition for object

    Returns:
        str: language string for object
    """
    obj_cfg = None
    for cfg in env.object_cfgs:
        if cfg["name"] == obj_name:
            obj_cfg = cfg
            break
    lang = obj_cfg["info"]["cat"].replace("_", " ")

    # replace some phrases
    if lang == "kettle electric":
        lang = "electric kettle"
    elif lang == "kettle non electric":
        lang = "kettle"
    elif lang == "bread flat":
        lang = "bread"
    elif lang == "oil and vinegar bottle":
        lang = "oil/vinegar bottle"
    elif lang == "salt and pepper shaker":
        lang = "salt/pepper shaker"
    elif lang == "jug wide opening":
        lang = "jug"

    if not get_preposition:
        return lang

    if lang in ["bowl", "pot", "pan"]:
        preposition = "in"
    elif lang in ["plate"]:
        preposition = "on"
    else:
        raise ValueError

    return lang, preposition


def add_obj_liquid_site(env, obj_name, liquid_rgba):
    key = f"{obj_name}_liquid"
    if key in env.sim.model._geom_name2id:
        liquid_geom_id = env.sim.model.geom_name2id(key)
        env.sim.model.geom_rgba[liquid_geom_id] = liquid_rgba
    else:
        liquid_site_id = env.sim.model.site_name2id(key)
        env.sim.model.site_rgba[liquid_site_id] = liquid_rgba
