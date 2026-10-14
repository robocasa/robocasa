import numpy as np
from robosuite.utils import transform_utils as TT

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

SUBTASK_KNIFE = "PickPlaceDrawerToCounter"   # bank atomic (knife drawer -> board)
SUBTASK_MEAT = "PickPlaceCounterToCounter"   # extra (meat plate -> board)
SUBTASK_NAV = "NavigateKitchen"

CARRY_NAV_LANG = "navigate to the cutting board"
DONE_LANG = "task complete"


@register_labeler
class SetUpCuttingStationLabeler(BaseSubtaskLabeler):
    """Labeler for the SetUpCuttingStation composite task.

    A knife sits in an already-open drawer, a steak/fish sits on a plate, and a
    cutting board (`receptacle`) sits on the counter. The robot moves the knife
    from the drawer onto the board, then the meat from the plate onto the board.
    Two subtasks: PickPlaceDrawerToCounter (knife) and PickPlaceCounterToCounter
    (meat), in grasp order. The drawer starts open (opened at scene setup), so
    there is no open-drawer subtask, and the cutting board is never moved.

    Detection off privileged sim state: each item's grasp onset is its first
    contact-grasp (fallback lift/displace) and its place ends fully-ungrasped
    (later of the contact-grasp end +pad and the finger-width open). NavigateKitchen
    is inserted only when the base actually drives (usually it does not).
    """

    ENV_NAME = "SetUpCuttingStation"

    KNIFE = "knife"
    MEAT = "meat"

    MOVE_TH = 0.10
    Z_LIFT = 0.03
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    REST_W = 10

    NAV_MOVING_TH = 0.004
    NAV_SMOOTH_W = 7
    MIN_NAV_DIST = 0.65

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.bids = {n: env.obj_body_id[n] for n in (self.KNIFE, self.MEAT)}
        self.geoms = {n: env.objects[n].contact_geoms for n in (self.KNIFE, self.MEAT)}
        try:
            self.meat_lang = env.get_obj_lang(self.MEAT)
        except Exception:
            self.meat_lang = "meat"
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        # Which side the drawer is on relative to the robot at spawn (base frame:
        # +x forward, +y left) -> "the left/right drawer", matching the atomic
        # drawer tasks' wording. setup() runs at the spawn state (states[0]).
        self.drawer_side = "left"
        try:
            bp = np.array(env.sim.data.body_xpos[self.base_body_id])
            bq = np.array(env.sim.data.body_xquat[self.base_body_id])  # wxyz
            lat = float((TT.quat2mat(bq[[1, 2, 3, 0]]).T @ (np.array(env.drawer.pos) - bp))[1])
            self.drawer_side = "left" if lat > 0 else "right"
        except Exception:
            pass
        self.extra_subtasks = {SUBTASK_MEAT}
        self.min_segment_frames = 10
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        d = {
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
        }
        for n in (self.KNIFE, self.MEAT):
            d[f"pos_{n}"] = env.sim.data.body_xpos[self.bids[n]].copy()
            d[f"grasp_{n}"] = int(env._check_grasp(gripper=self.gripper, object_geoms=self.geoms[n]))
        return d

    # ---- event helpers -------------------------------------------------

    def _grasp_onset(self, grasp_arr, obj):
        gi = np.where(grasp_arr > 0)[0]
        if gi.size:
            return int(gi[0])
        z0 = float(np.median(obj[: self.REST_W, 2]))
        lifted = np.where(obj[:, 2] > z0 + self.Z_LIFT)[0]
        if lifted.size:
            return int(lifted[0])
        disp = np.linalg.norm(obj[:, :2] - obj[0, :2], axis=1)
        w = np.where(disp > self.MOVE_TH)[0]
        return int(w[0]) if w.size else 0

    def _fw_open(self, fw, fw_open, grasp_t, hi):
        seg = fw[grasp_t:hi]
        if seg.size < 4:
            return None
        held = float(np.min(seg[: min(25, seg.size)]))
        level = held + self.REL_FRAC * (fw_open - held)
        above = seg > level
        for t in range(1, seg.size - 2):
            if above[t] and above[t + 1] and above[t + 2] and not above[t - 1]:
                return grasp_t + t
        return None

    def _release_for(self, grasp_arr, obj, fw, fw_open, grasp_t, hi):
        spd = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        mv = np.where(spd[grasp_t:hi] > self.MOTION_TH)[0]
        setdown = int(grasp_t + int(mv[-1]) + 2) if mv.size else int(grasp_t + 1)
        fwo = self._fw_open(fw, fw_open, grasp_t, hi)
        gc = np.where(grasp_arr[grasp_t:hi] > 0)[0]
        if gc.size:
            rel = grasp_t + int(gc[-1]) + self.RELEASE_PAD
            if fwo is not None:
                rel = max(rel, fwo)
        elif fwo is not None:
            rel = fwo
        else:
            rel = setdown
        return int(min(max(rel, grasp_t + 1), hi))

    def _nav_end(self, base_speed, base_xy, a, b):
        if b - a < 2 or base_xy is None:
            return a
        seg = base_speed[a:b]
        moving = np.where(seg > self.NAV_MOVING_TH)[0]
        if not moving.size:
            return a
        nav_end = min(a + int(moving[-1]) + 1, b)
        disp = float(np.linalg.norm(base_xy[min(nav_end, b - 1)] - base_xy[a]))
        return nav_end if disp >= self.MIN_NAV_DIST else a

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        base_xy = np.array([h["base_xy"] for h in history])
        pos = {n: np.array([h[f"pos_{n}"] for h in history]) for n in (self.KNIFE, self.MEAT)}
        grasp = {n: np.array([h[f"grasp_{n}"] for h in history]) for n in (self.KNIFE, self.MEAT)}

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            bs = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([bs, bs[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        meta = {
            self.KNIFE: dict(
                sub=SUBTASK_KNIFE,
                pick=f"pick up the knife from the {self.drawer_side} drawer",
                place="place the knife on the cutting board",
                src_nav=f"navigate to the {self.drawer_side} drawer",
            ),
            self.MEAT: dict(
                sub=SUBTASK_MEAT,
                pick=f"pick up the {self.meat_lang} from the plate",
                place=f"place the {self.meat_lang} on the cutting board",
                src_nav="navigate to the plate",
            ),
        }
        g = {n: self._grasp_onset(grasp[n], pos[n]) for n in (self.KNIFE, self.MEAT)}
        order = sorted((self.KNIFE, self.MEAT), key=lambda n: g[n])
        g0, g1 = g[order[0]], g[order[1]]
        r0 = self._release_for(grasp[order[0]], pos[order[0]], fw, fw_open, g0, max(g1, g0 + 1))
        r1 = self._release_for(grasp[order[1]], pos[order[1]], fw, fw_open, g1, T)
        rel = {order[0]: r0, order[1]: r1}

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang):
            nav_end = self._nav_end(base_speed, base_xy, cursor, end)
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        for n in order:
            m = meta[n]
            maybe_nav(g[n], m["src_nav"])
            emit(g[n], m["sub"], STAGE_PICK, m["pick"])
            maybe_nav(rel[n], CARRY_NAV_LANG)
            emit(rel[n], m["sub"], STAGE_PLACE, m["place"])
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "order": ",".join(order),
            "grasp": [int(g[order[0]]), int(g[order[1]])],
            "release": [r0, r1],
            "base_max": round(float(np.linalg.norm(base_xy - base_xy[0], axis=1).max()), 2)
            if base_xy is not None else 0.0,
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        issues = super().validate(result)
        if getattr(result, "debug", None) and result.debug.get("manual"):
            return issues
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        for sub, label in ((SUBTASK_KNIFE, "knife"), (SUBTASK_MEAT, "meat")):
            m = names == sub
            npi = int(np.sum(m & (stages == STAGE_PICK)))
            npl = int(np.sum(m & (stages == STAGE_PLACE)))
            if npi != 1 or npl != 1:
                issues.append(f"{label}: expected 1 pick + 1 place, found {npi} + {npl}")
        return issues
