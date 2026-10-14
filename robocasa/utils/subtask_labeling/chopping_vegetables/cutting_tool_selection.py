import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# The robot picks the correct cutting tool (knife or peeler, by the food on the
# board) from the open drawer and places it on the cutting board. A single
# pick-place; the bank atomic splits into pick/place stages.
SUBTASK_TOOL = "PickPlaceDrawerToCounter"

DONE_LANG = "task complete"

# food -> correct tool (mirrors CuttingToolSelection._CUTTING_MAP).
CUTTING_MAP = {
    "garlic": "knife",
    "eggplant": "peeler",
    "broccoli": "knife",
    "carrot": "peeler",
    "cucumber": "peeler",
    "potato": "peeler",
    "avocado": "knife",
}


@register_labeler
class CuttingToolSelectionLabeler(BaseSubtaskLabeler):
    """Labeler for the CuttingToolSelection composite task.

    Task (non-mobile): a knife and a peeler sit in an already-open drawer; a food
    item sits on the cutting board. The robot selects the correct tool for that
    food (CUTTING_MAP) and places it on the cutting board. The drawer is never
    closed (it only opens further as the robot reaches in). Target split only --
    no pretrain release.

    Doc-canonical decomposition (2 subtasks):
      1. PickPlaceDrawerToCounter [pick]   -> pick the correct tool from the drawer
      2. PickPlaceDrawerToCounter [place]  -> place it on the cutting board
    plus a terminal done. No leading navigation -- the demo opens on the pick.

    The moved tool is the one with the larger total displacement (the chosen tool
    travels ~0.4-0.6 m; the other is only nudged, <0.2 m -- a clean ~2x+ gap).
    Events: t_pick = the tool departs its rest in the drawer; t_place = the
    gripper finger-width opens after the pickup (the release). Validation is
    structural plus checks that the moved tool is the correct one for the food and
    that it ends on the cutting board.
    """

    ENV_NAME = "CuttingToolSelection"

    DEPART_TOL = 0.10
    REST_W = 10
    FW_RANGE_MIN = 0.03
    # The pick ends when the tool is lifted out of the drawer (its height rises
    # this far above its resting level). The tool is first shuffled horizontally
    # inside the drawer (the drawer is pulled further open) while still at drawer
    # height -- that in-drawer phase is part of "picking from the drawer", so the
    # boundary is the lift-out, not the first horizontal displacement.
    Z_LIFT_TOL = 0.04
    # The release finger-open must occur after the tool has arrived within this of
    # its final (board) pose. This filters early grip-adjustment opens (the robot
    # briefly re-opens to reseat the grasp just after lifting the tool out of the
    # drawer), which would otherwise collapse the place.
    ARRIVE_TOL = 0.15
    # The placed tool ends within this xy of the cutting board.
    BOARD_TOL = 0.15

    TOOLS = ["peeler", "knife"]

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.tool_body_ids = [env.obj_body_id[t] for t in self.TOOLS]
        self.tool_lang = []
        for t in self.TOOLS:
            try:
                self.tool_lang.append(env.get_obj_lang(t))
            except Exception:
                self.tool_lang.append(t)
        self.food = ep_meta.get("refs", {}).get("food")
        self.correct_tool = CUTTING_MAP.get(self.food)
        # This task's lang carries no drawer side, and env.drawer_side is re-randomized on
        # each reset (wrong on ~half of demos, and disagrees between surfaces). Derive the
        # side from geometry at the spawn state: the drawer's position in the robot base
        # frame (+y = left), the same way set_up_cutting_station does.
        self.drawer_side = None
        try:
            from robosuite.utils import transform_utils as _TT
            bid = env.sim.model.body_name2id("mobilebase0_base")
            bp = np.array(env.sim.data.body_xpos[bid])
            bq = np.array(env.sim.data.body_xquat[bid])  # wxyz
            lat = float((_TT.quat2mat(bq[[1, 2, 3, 0]]).T @ (np.array(env.drawer.pos) - bp))[1])
            self.drawer_side = "left" if lat > 0 else "right"
        except Exception:
            self.drawer_side = getattr(env, "drawer_side", None)
        self.board_body_id = env.obj_body_id["food_container"]
        self.finger_qadr = [
            env.sim.model.get_joint_qpos_addr(j) for j in env.robots[0].gripper["right"].joints
        ]
        self.min_segment_frames = 30
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        fw = float(sum(abs(env.sim.data.qpos[a]) for a in self.finger_qadr))
        return {
            "tools": np.stack(
                [env.sim.data.body_xpos[b].copy() for b in self.tool_body_ids]
            ),
            "board": env.sim.data.body_xpos[self.board_body_id].copy(),
            "fw": fw,
        }

    # ---- event helpers -------------------------------------------------

    def _depart_time(self, obj, end):
        # Lift onset: first frame the tool rises Z_LIFT_TOL above its resting
        # height (= lifted clear of the drawer, the start of the carry). Falls
        # back to the first horizontal displacement if no clear lift is seen.
        z_rest = np.median(obj[: self.REST_W, 2])
        zrise = obj[:, 2] - z_rest
        end = max(int(end), 1)
        cand = np.where((zrise > self.Z_LIFT_TOL)[:end])[0]
        if cand.size:
            return int(cand[0])
        start = np.median(obj[: self.REST_W], axis=0)
        moved = np.linalg.norm(obj - start, axis=1) > self.DEPART_TOL
        cand = np.where(moved[:end])[0]
        return int(cand[0]) if cand.size else 0

    def _release_time(self, obj, fw, t_dep, T):
        # Release = the first gripper finger-open after the tool has arrived near
        # its final (board) pose. Gating on arrival (rather than just "after the
        # pickup") skips the early grip-adjustment open that fires while the tool
        # is still by the drawer.
        final = np.median(obj[-self.REST_W:], axis=0)
        d2final = np.linalg.norm(obj - final, axis=1)
        near = np.where(d2final[:T] < self.ARRIVE_TOL)[0]
        near = near[near > t_dep]
        t_near = int(near[0]) if near.size else t_dep
        lo, hi = float(np.min(fw)), float(np.max(fw))
        if hi - lo > self.FW_RANGE_MIN:
            mid = lo + 0.5 * (hi - lo)
            opens = np.where((fw[:-1] < mid) & (fw[1:] >= mid))[0] + 1
            opens = opens[opens >= t_near]
            if opens.size:
                return int(opens[0])
        # Fallback: the tool's permanent arrival near its final pose.
        uns = np.where(d2final[:T] >= self.ARRIVE_TOL)[0]
        return int(uns[-1]) + 1 if uns.size else max(t_dep + 1, 1)

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        tools = np.array([h["tools"] for h in history])  # (T, 2, 3)
        board = np.array([h["board"] for h in history])
        fw = np.array([h["fw"] for h in history])

        disp_max = []
        for i in range(len(self.TOOLS)):
            o = tools[:, i, :]
            disp_max.append(float(np.linalg.norm(o - np.median(o[: self.REST_W], axis=0), axis=1).max()))
        mover = int(np.argmax(disp_max))
        other = 1 - mover
        obj = tools[:, mover, :]

        t_dep = self._depart_time(obj, T)
        t_rel = self._release_time(obj, fw, t_dep, T)

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        lang = self.tool_lang[mover]
        drawer = f"{self.drawer_side} open drawer" if self.drawer_side else "open drawer"
        emit(t_dep, SUBTASK_TOOL, STAGE_PICK, f"pick up the {lang} from the {drawer}")
        emit(t_rel, SUBTASK_TOOL, STAGE_PLACE, f"place the {lang} on the cutting board")
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        final_xy = np.median(obj[-self.REST_W:, :2], axis=0) if T else None
        board_xy = np.median(board[-self.REST_W:, :2], axis=0) if T else None
        on_board = (
            bool(np.linalg.norm(final_xy - board_xy) < self.BOARD_TOL)
            if final_xy is not None else False
        )
        debug = {
            "t_pick": int(t_dep),
            "t_place": int(t_rel),
            "mover": self.TOOLS[mover],
            "disp_mover": round(disp_max[mover], 3),
            "disp_other": round(disp_max[other], 3),
            "on_board": int(on_board),
            "n_segments": len(segments),
        }
        return self.build_result(T, segments, debug=debug)

    # ---- structural validation ----------------------------------------

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        n_pick = int(np.sum(stages == STAGE_PICK))
        n_place = int(np.sum(stages == STAGE_PLACE))
        if n_pick != 1:
            issues.append(f"expected 1 pick stage, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 place stage, found {n_place}")
        d = result.debug
        # The moved tool must be the correct one for the food.
        if self.correct_tool is not None and d.get("mover") != self.correct_tool:
            issues.append(
                f"moved {d.get('mover')} but correct tool for {self.food} is "
                f"{self.correct_tool}"
            )
        if d.get("on_board") == 0:
            issues.append("tool did not end on the cutting board")
        return issues
