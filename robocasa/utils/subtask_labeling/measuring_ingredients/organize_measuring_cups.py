import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# Three measuring cups on the counter are rearranged left-to-right, smallest-to-largest
# and rotated handle-out. Each cup is a counter -> counter reposition, which has no bank
# name -> PickPlaceCounterToCounter (extension). Fixed base; no nav, no close.
SUBTASK_MOVE = "PickPlaceCounterToCounter"

DONE_LANG = "task complete"

CUPS = ("measuring_cup_small", "measuring_cup_medium", "measuring_cup_large")
DEST = {"small": "on the left", "medium": "in the middle", "large": "on the right"}


@register_labeler
class OrganizeMeasuringCupsLabeler(BaseSubtaskLabeler):
    """Labeler for the OrganizeMeasuringCups composite task.

    Task (fixed-base): three measuring cups (small/medium/large) sit on the counter in a
    jumble. Reposition them left-to-right smallest-to-largest, each rotated handle-out and
    upright. Robot starts at the counter; every reposition is local.

    Doc-canonical decomposition (3 high-level: one reposition per cup). Emitted per cup in
    GRASP order:
      PickPlaceCounterToCounter [pick]  -> pick up the {size} measuring cup from the counter
      PickPlaceCounterToCounter [place] -> place the {size} measuring cup {on the left|in the middle|on the right}
    plus a terminal done. The destination is fixed by size (small->left, large->right).

    Detection is GRASP-based, not displacement-based: the sort is largely a ROTATE-IN-PLACE
    (a cup is picked up, spun so the handle points out, and set back down with almost no
    translation), so a cup's displacement can be < 5 cm. Every cup, however, is grasped
    once. For each cup the manipulation is its longest sustained gripper-contact run:
    t_grasp = its start (end of the reach / the pick); the place ends at t_release =
    the gripper's LAST CONTACT with the cup (last frame within CONTACT_TH before it
    withdraws for good), so `done` begins only once the gripper has pulled clear.
    """

    ENV_NAME = "OrganizeMeasuringCups"

    GRASP_TH = 0.07      # gripper<->cup dist below which the cup is in hand. Kept tight:
                         # a looser tol (0.11) false-grasps a NEIGHBOUR the gripper only
                         # hovers past while sorting an adjacent cup (demo_32). The few
                         # "2/3 cups" reviews are genuine -- a cup that starts already sorted.
    GRASP_SUSTAIN = 4
    CONTACT_TH = 0.10    # the place ends when the gripper separates from the cup: the first
                         # sustained frame (after the grasp minimum) that gd rises past this.
                         # Looser than GRASP_TH (0.07) because the gripper keeps steadying the
                         # cup as it lets go -- gd creeps from its grasp minimum (~0.04) up past
                         # ~0.10 as the fingers finally clear the rim. We take the FIRST sustained
                         # crossing, not the last dip below it: the destination slots are only
                         # ~10 cm apart, so a "last-contact" rule would keep the place alive while
                         # the gripper reaches the ADJACENT next cup and swallow its pick.
    REL_SUSTAIN = 3
    REL_MARGIN = 0.03    # fallback release when the gripper never cleanly separates to
                         # CONTACT_TH before grabbing the next (adjacent) cup: the first frame
                         # gd rises this far above the grasp minimum == the let-go.
    MIN_PICK_GAP = 8     # an intermediate place may never end within this many frames of the
                         # next cup's grasp, so every following pick keeps a visible segment.
    REST_W = 15

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in CUPS}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.size = {o: o.rsplit("_", 1)[1] for o in CUPS}
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {"grip": env.sim.data.site_xpos[self.eef_site_id].copy()}
        for o in CUPS:
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

    @staticmethod
    def _longest_run(mask, min_len):
        """(start, end_exclusive) of the longest True run >= min_len, or None."""
        best = None
        best_len = 0
        run_start = None
        for t in range(len(mask)):
            if mask[t]:
                if run_start is None:
                    run_start = t
            else:
                if run_start is not None:
                    L = t - run_start
                    if L >= min_len and L > best_len:
                        best_len, best = L, (run_start, t)
                    run_start = None
        if run_start is not None:
            L = len(mask) - run_start
            if L >= min_len and L > best_len:
                best = (run_start, len(mask))
        return best

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in CUPS}
        grip = np.array([h["grip"] for h in history])
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in CUPS}

        # each cup's manipulation = its longest sustained in-hand run (for its start/order)
        grasp = {}
        for o in CUPS:
            run = self._longest_run(gd[o] < self.GRASP_TH, self.GRASP_SUSTAIN)
            grasp[o] = run[0] if run is not None else None  # t_grasp or None
        moved = [o for o in CUPS if grasp[o] is not None]
        order = sorted(moved, key=lambda o: grasp[o])

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        n_moved = 0
        for i, o in enumerate(order):
            size = self.size[o]
            t_grasp = max(int(grasp[o]), cursor)
            # place ends at the LAST gripper-cup contact within this cup's turn (bounded by
            # the next cup's grasp so a pass-by near a placed cup can't extend it), so `done`
            # begins only once the gripper has truly let go of the last cup.
            win_end = int(grasp[order[i + 1]]) if i + 1 < len(order) else T
            seg = gd[o][t_grasp:win_end]
            # the place ends when the gripper SEPARATES from the cup: the first sustained
            # frame, after the grasp minimum (the set-down), that gd rises past CONTACT_TH.
            # For the last cup the withdrawal is monotonic so this is also its last contact.
            gmin_i = int(np.argmin(seg)) if seg.size else 0
            gmin = float(seg[gmin_i]) if seg.size else 0.0
            rel = self._sustained_start(seg > self.CONTACT_TH, gmin_i, self.REL_SUSTAIN)
            if rel is None:
                # adjacent cups: the gripper never cleanly clears CONTACT_TH before grabbing
                # the next cup. Fall back to the release ONSET (gd first rises REL_MARGIN above
                # the grasp minimum) so the place ends at let-go, not at the next cup's grasp.
                rel = self._sustained_start(seg > gmin + self.REL_MARGIN, gmin_i, self.REL_SUSTAIN)
            if rel is not None:
                t_release = t_grasp + int(rel)
            else:
                below = np.where(seg < self.CONTACT_TH)[0]
                t_release = t_grasp + int(below[-1]) + 1 if below.size else t_grasp + 1
            # never let an intermediate place run into the next cup's pick (guardrail).
            if i + 1 < len(order):
                t_release = min(t_release, win_end - self.MIN_PICK_GAP)
            t_release = int(min(max(t_release, t_grasp + 1), T))
            emit(t_grasp, SUBTASK_MOVE, STAGE_PICK, f"pick up the {size} measuring cup from the counter")
            emit(t_release, SUBTASK_MOVE, STAGE_PLACE, f"place the {size} measuring cup {DEST[size]}")
            n_moved += 1

        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        review = []
        if n_moved != 3:
            review.append(f"only {n_moved}/3 cups grasped -- verify (a cup may already be sorted)")
        result = self.build_result(T, segments, debug={
            "order": order, "n_moved": n_moved,
            "grasp_start": {o: int(grasp[o]) for o in moved},
            "n_segments": len(segments),
        })
        result.review = review
        return result

    def validate(self, result):
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        stages = result.subtask_stage[firsts]
        npi = int(np.sum(stages == STAGE_PICK))
        npl = int(np.sum(stages == STAGE_PLACE))
        if npi != npl:
            issues.append(f"pick/place mismatch: {npi} picks, {npl} places")
        return issues
