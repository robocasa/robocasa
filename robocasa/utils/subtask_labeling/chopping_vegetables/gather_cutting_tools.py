import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Open the drawer, then take the two cutting tools (a peeler and a knife) out of it
# and set them on the cutting board. OpenDrawer + PickPlaceDrawerToCounter are bank
# atomics (the cutting board is a receptacle ON the counter) -- no extras.
SUBTASK_OPEN = "OpenDrawer"
SUBTASK_MOVE = "PickPlaceDrawerToCounter"

DONE_LANG = "task complete"

TOOLS = ("peeler", "knife")


@register_labeler
class GatherCuttingToolsLabeler(BaseSubtaskLabeler):
    """Labeler for the GatherCuttingTools composite task.

    Task: a peeler and a knife sit in a CLOSED drawer; a cutting board sits on the
    counter beside it. Open the drawer, then place both tools on the cutting board.
    The base never drives (mobile: No).

    Doc-canonical count = 3 (`mobile: No`): OpenDrawer + one PickPlaceDrawerToCounter
    per tool. OpenDrawer + peeler[pick,place] + knife[pick,place] + done = 6 segments;
    expected_done_idx = 5. The tool ORDER varies (resolved from which moves first).

    Boundaries:
      - t_open: the drawer slide joint reaches its open plateau (OPEN_FRAC of its own
        peak, sustained). The tools sit INSIDE the drawer and slide OUT with it, so
        each tool's rest pose is taken AFTER the drawer is open (median around t_open)
        -- the drawer-ride is not counted as the tool's carry.
      - per tool (first-move / relative-release split): t_move = the tool first
        displaces MOVE_TOL from its post-open pose; t_placed = the gripper RELEASES it.
    """

    ENV_NAME = "GatherCuttingTools"

    DIVERGE_TH = 0.05  # peeler<->knife separation growing this far past its rest ==
    # the first tool has been lifted out (the two slide together inside the drawer)
    OPEN_FRAC = 0.90  # slide at this fraction of its pre-grab peak == main swing done
    OPEN_SUSTAIN = 5
    OPEN_MIN = 0.02  # drawer slide peak below this (m) == it never really opened
    MOVE_TOL = 0.10  # tool displaced this far from its post-open pose == being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in TOOLS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # env.drawer_side is re-randomized on each reset (wrong on ~a third of demos and it
        # disagrees between surfaces); take the side from the demo's recorded lang
        # ("...drawer on the {left/right}..."), the ground truth.
        import re
        _m = re.search(r"drawer on the (left|right)", (ep_meta or {}).get("lang", "") or "")
        self.drawer_side = _m.group(1) if _m else (getattr(env, "drawer_side", "") or "")

        drawer = env.drawer
        self.drawer_adr = None
        for jname in (f"{drawer.name}_slidejoint", f"{drawer.name}_slide"):
            try:
                self.drawer_adr = env.sim.model.get_joint_qpos_addr(jname)
                break
            except Exception:
                continue
        if self.drawer_adr is None:
            # fall back to any slide-like joint under the drawer's name prefix
            for n in env.sim.model.joint_names:
                if n and n.startswith(drawer.name) and "slide" in n.lower():
                    self.drawer_adr = env.sim.model.get_joint_qpos_addr(n)
                    break

        self.extra_subtasks = set()
        self.expected_done_idx = 5
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "drawer": float(env.sim.data.qpos[self.drawer_adr]) if self.drawer_adr is not None else 0.0,
        }
        for o in TOOLS:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        return d

    @staticmethod
    def _sustained_start(mask, start, sustain):
        run = 0
        for t in range(int(start), len(mask)):
            if mask[t]:
                run += 1
                if run >= sustain:
                    return t - sustain + 1
            else:
                run = 0
        return None

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in TOOLS}
        grip = np.array([h["grip"] for h in history])
        drawer = np.abs(np.array([h["drawer"] for h in history]))

        # open the drawer: the two tools sit inside the drawer and slide OUT together
        # while it opens, so their separation stays ~constant until the FIRST tool is
        # lifted out. End OpenDrawer at the LAST drawer motion BEFORE that divergence:
        # this captures the whole opening even when it comes in stages (a fixed
        # fraction-of-peak gate stops at the first pause), and excludes the later
        # nudges the slide gets while a tool is being grabbed (which inflate the peak
        # and would push the boundary far too late).
        rel = np.linalg.norm(opos[TOOLS[0]] - opos[TOOLS[1]], axis=1)
        rel0 = float(np.median(rel[: self.REST_W]))
        div = np.where(rel > rel0 + self.DIVERGE_TH)[0]
        first_diverge = int(div[0]) if div.size else T
        # peak openness reached BEFORE the first tool is lifted -- excludes the nudges
        # the slide gets while a tool is grabbed (which inflate a global peak).
        peak_pre = float(drawer[:first_diverge].max()) if first_diverge > 0 else float(drawer.max())
        has_open = peak_pre >= self.OPEN_MIN
        if has_open:
            # OpenDrawer ends at the END OF THE MAIN OPENING SWING: the first frame the
            # slide reaches OPEN_FRAC of that pre-grab peak and stays. NOT the last
            # tiny reach-nudge before the grab (that left the pick too short), and the
            # pre-grab peak handles a slide opened in stages (each stage counts).
            t_open = self._sustained_start(drawer >= self.OPEN_FRAC * peak_pre, 0, self.OPEN_SUSTAIN)
            if t_open is None:
                t_open = int(np.argmax(drawer[:first_diverge])) if first_diverge > 0 else int(np.argmax(drawer))
            t_open = int(min(t_open, first_diverge))
        else:
            t_open = 0

        # each tool's rest pose is its post-open (still-in-drawer) pose
        w0 = min(t_open, max(T - self.REST_W, 0))
        rest = {o: np.median(opos[o][w0: w0 + self.REST_W], axis=0) for o in TOOLS}
        disp = {o: np.linalg.norm(opos[o] - rest[o], axis=1) for o in TOOLS}
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in TOOLS}

        first_move = {}
        for o in TOOLS:
            fm = self._sustained_start(disp[o] >= self.MOVE_TOL, t_open, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(TOOLS, key=lambda o: first_move[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        side = f"{self.drawer_side} " if self.drawer_side else ""
        emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, f"open the {side}drawer")

        for o in order:
            # rest pose at the START of THIS tool's turn: the tool may drift while the
            # previous tool is removed (reaching into the drawer nudges the slide and
            # the tools with it), so measuring displacement from the stale post-open
            # pose fires early and collapses the pick. Re-anchor to the tool's settled
            # pose over the window just after `cursor`.
            r = np.median(opos[o][cursor: min(cursor + self.REST_W, T)], axis=0)
            dsp = np.linalg.norm(opos[o] - r, axis=1)
            t_move = self._sustained_start(dsp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
            if t_move is None:
                mv = np.where(dsp[cursor:] >= self.MOVE_TOL)[0]
                t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
            t_move = max(t_move, cursor)

            hold_gd = float(np.median(gd[o][t_move: min(t_move + self.HOLD_W, T)]))
            released = gd[o] > hold_gd + self.RELEASE_MARGIN
            t_placed = self._sustained_start(released, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
            if t_placed is None:
                t_placed = T - 1

            emit(t_move, SUBTASK_MOVE, STAGE_PICK, f"pick up the {o} from the drawer")
            emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {o} on the cutting board")

        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not has_open:
            review.append(f"drawer never opened (peak {peak_pre:.3f}) -- verify")

        debug = {
            "t_open": t_open, "drawer_peak": round(peak_pre, 3),
            "order": order, "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]

        def count(name, stage=None):
            m = names == name
            if stage is not None:
                m = m & (stages == stage)
            return int(np.sum(m))

        if count(SUBTASK_OPEN) != 1:
            issues.append(f"expected 1 OpenDrawer, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 2:
            issues.append(f"expected 2 tool picks, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 2:
            issues.append(f"expected 2 tool places, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        return issues
