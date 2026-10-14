import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE

# The straw is lifted out of the (already-open) drawer and dropped into the smoothie
# cup on the counter. Drawer -> counter/cup has no bank name -> PickPlaceDrawerToCounter
# (extension). The robot starts at the drawer; the initial base creep to the drawer is
# the leading approach (not a subtask) and there is no nav between pick and place.
SUBTASK_MOVE = "PickPlaceDrawerToCounter"

DONE_LANG = "task complete"


@register_labeler
class PlaceStrawLabeler(BaseSubtaskLabeler):
    """Labeler for the PlaceStraw composite task.

    Task: a straw sits in an already-open drawer; a smoothie cup sits on the counter by
    the drawer. Pick the straw out of the drawer and put it in the cup. The drawer is
    not closed (success only requires the straw in the cup). The robot starts at the
    drawer.

    Doc-canonical decomposition (2 high-level subtasks: pick, place). Emitted:
      PickPlaceDrawerToCounter [pick]  -> pick up the straw from the {side} drawer
      PickPlaceDrawerToCounter [place] -> put the straw in the smoothie cup
    plus a terminal done.

    Detection (robust to the straw being wiggled inside the drawer before lift-out):
    t_release = first frame the straw is at the cup (within PLACED_TOL of its final
    pose) with the gripper let go; t_pick = the start of the LAST contiguous run in
    which the straw is lifted out of the drawer before that release.
    """

    ENV_NAME = "PlaceStraw"

    LIFT_TOL = 0.05      # height above rest for the straw to count as lifted out
    PLACED_TOL = 0.10    # straw within this of its final pose counts as delivered
    RELEASE_TH = 0.12    # gripper-straw dist above which the gripper has let go

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.straw_body_id = env.obj_body_id["straw"]
        self.cup_body_id = env.obj_body_id["cup"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # drawer_side is re-sampled on replay (random_index in ManipulateDrawer) so it is
        # unreliable -- take the side from the demo's own recorded lang.
        import re
        m = re.search(r"from the (left|right) drawer", ep_meta.get("lang", ""))
        self.side = m.group(1) if m else (getattr(env, "drawer_side", "") or "")
        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = 2
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "straw": env.sim.data.body_xpos[self.straw_body_id].copy(),
            "cup": env.sim.data.body_xpos[self.cup_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
        }

    def resolve(self, history):
        T = len(history)
        straw = np.array([h["straw"] for h in history])
        grip = np.array([h["grip"] for h in history])

        final = straw[-1]
        d_final = np.linalg.norm(straw - final, axis=1)
        d_grip = np.linalg.norm(straw - grip, axis=1)
        delivered = (d_final < self.PLACED_TOL) & (d_grip > self.RELEASE_TH)
        rel = np.where(delivered)[0]
        t_release = int(rel[0]) if rel.size else T - 1

        lifted = straw[:, 2] > straw[0, 2] + self.LIFT_TOL
        window = lifted[:t_release]
        if window.any():
            grounded = np.where(~window)[0]
            t_pick = int(grounded[-1]) + 1 if grounded.size else 0
        else:
            t_pick = max(int(t_release) - 1, 1)
        t_pick = min(max(t_pick, 1), t_release)

        side = f"{self.side} " if self.side else ""
        segments = [
            (t_pick, SUBTASK_MOVE, STAGE_PICK, f"pick up the straw from the {side}drawer"),
            (t_release, SUBTASK_MOVE, STAGE_PLACE, "put the straw in the smoothie cup"),
            (T, DONE, STAGE_DONE, DONE_LANG),
        ]
        self.expected_done_idx = len(segments) - 1

        review = []
        if t_release >= T - 1 and not rel.size:
            review.append("straw never registered delivered to the cup")
        result = self.build_result(T, segments, debug={
            "t_pick": int(t_pick), "t_release": int(t_release),
            "side": self.side, "n_segments": len(segments),
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
        if int(np.sum(stages == STAGE_PICK)) != 1:
            issues.append("expected 1 pick stage")
        if int(np.sum(stages == STAGE_PLACE)) != 1:
            issues.append("expected 1 place stage")
        return issues
