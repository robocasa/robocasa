import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# The atomic bank has no bare "pick up" or "scrub" skill, so both are declared
# extra_subtasks (non-bank names).
SUBTASK_PICK = "PickUpSponge"
SUBTASK_SCRUB = "ScrubCuttingBoard"
SUBTASK_NAV = "NavigateKitchen"

PICK_LANG = "pick up the sponge from the counter"
SCRUB_LANG = "scrub the cutting board"
NAV_LANG = "navigate to the cutting board"
DONE_LANG = "task complete"


@register_labeler
class ScrubCuttingBoardLabeler(BaseSubtaskLabeler):
    """Labeler for the ScrubCuttingBoard composite task.

    A sponge and a cutting board sit on the counter; the robot spawns at the
    sponge (`init_robot_here`). The two subtasks are: pick up the sponge, then
    scrub the cutting board (press/rub the sponge over the board, then release
    it). Success needs 5+ distinct board contacts spread >= 0.1 m with the sponge
    finally released.

    Signals off privileged sim state: the sponge's grasp onset is its first
    contact-grasp (fallback lift/displace); the scrub runs from the grasp (or the
    end of the drive to the board) until the sponge is released -- the later of the
    contact-grasp end (+pad) and the finger-width gripper-open. The sponge<->board
    contact burst marks the actual scrubbing (used to place the optional drive and
    to sanity-check that scrubbing happened). Some layouts put the board far enough
    to require a base drive, so NavigateKitchen is inserted only when the base
    moves.
    """

    ENV_NAME = "ScrubCuttingBoard"

    SPONGE = "sponge"
    BOARD = "cutting_board"

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
        self.sponge_bid = env.obj_body_id[self.SPONGE]
        self.sponge_geoms = env.objects[self.SPONGE].contact_geoms
        self.board_geoms = env.objects[self.BOARD].contact_geoms
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        try:
            self.base_body_id = env.sim.model.body_name2id("mobilebase0_base")
        except Exception:
            self.base_body_id = None
        self.extra_subtasks = {SUBTASK_PICK, SUBTASK_SCRUB}
        self.min_segment_frames = 15
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        base_xy = (
            env.sim.data.body_xpos[self.base_body_id][:2].copy()
            if self.base_body_id is not None
            else np.array([np.nan, np.nan])
        )
        try:
            board = int(env.check_contact(self.sponge_geoms, self.board_geoms))
        except Exception:
            board = 0
        return {
            "pos": env.sim.data.body_xpos[self.sponge_bid].copy(),
            "base_xy": base_xy,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.sponge_geoms)),
            "board": board,
        }

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
        pos = np.array([h["pos"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history])
        grasp = np.array([h["grasp"] for h in history])
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())
        board = np.array([h["board"] for h in history])

        has_base = np.isfinite(base_xy).all()
        if has_base and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            bs = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([bs, bs[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        grasp0 = self._grasp_onset(grasp, pos)
        release = self._release_for(grasp, pos, fw, fw_open, grasp0, T)
        # First sponge<->board contact after the grasp marks where scrubbing
        # starts (the drive to the board, if any, ends here).
        bc = np.where(board[grasp0:release] > 0)[0]
        board_first = int(grasp0 + int(bc[0])) if bc.size else grasp0

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

        emit(grasp0, SUBTASK_PICK, STAGE_PICK, PICK_LANG)
        maybe_nav(board_first, NAV_LANG)
        emit(release, SUBTASK_SCRUB, STAGE_EXECUTE, SCRUB_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        debug = {
            "grasp0": grasp0,
            "board_first": board_first,
            "release": release,
            "board_frames": int(board.sum()),
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
        names = set(str(v) for v in result.subtask_name[firsts])
        for req in (SUBTASK_PICK, SUBTASK_SCRUB):
            if req not in names:
                issues.append(f"missing {req} subtask")
        # The scrub should have actually scrubbed (sponge touched the board).
        if getattr(result, "debug", None) and result.debug.get("board_frames", 0) < 3:
            issues.append("sponge barely contacted the board (scrub not detected)")
        return issues
