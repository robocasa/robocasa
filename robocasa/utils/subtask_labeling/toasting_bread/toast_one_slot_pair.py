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


# Subtask 1: relocate the bread. The atomic-task bank only has Counter<->X
# pick-and-place entries, none for toaster-slot to toaster-slot, so we use the
# generic "PickPlace" skill name (declared as an extra below) and flag the gap.
SUBTASK_MOVE = "PickPlace"
PICK_LANG = "pick up the bread slice from the broken toaster slot"
PLACE_LANG = "place the bread slice in the working toaster slot-pair"

# Subtask 2: TurnOnToaster is an exact bank atomic task.
SUBTASK_TOAST = "TurnOnToaster"
TOAST_LANG = "push down the lever of the toaster to turn it on"

DONE_LANG = "task complete"


@register_labeler
class ToastOneSlotPairLabeler(BaseSubtaskLabeler):
    """Labeler for the ToastOneSlotPair composite task.

    The toaster has a broken slot-pair and a working slot-pair. One bread slice
    (bread_broken) starts in the broken pair; another (bread_working) sits in
    the working pair. The robot relocates bread_broken into the working pair and
    presses that pair's lever to toast both slices.

    Two bank subtasks (matching the doc-set count of 2):
      - PickPlace   : relocate bread_broken -> split into pick / place stages
      - TurnOnToaster : press the lever      -> single 'execute' stage
    giving 4 segments: pick, place, turn-on-toaster, done.

    Labels come from object trajectories plus the lever joint value (all in the
    replayed sim state, so no stale fixture references):
      - t_lift:    bread_broken first lifts off the broken slot (pick -> place)
      - t_placed:  the first frame AFTER t_lift where bread_broken is at the
                   working slot and the gripper has let go     (place -> press)
      - t_toasted: the working lever crosses its 'on' point     (press -> done)

    t_lift uses the *first* lift-off (not the last contiguous run, as in
    AddIceCubes): bread_broken ends at the same height as it started -- both
    toaster slots sit at the same level -- so a "last run" would wrongly catch
    the lowering motion. t_placed is forced to come after t_lift because the
    two slot-pairs are close enough that bread_broken can start within
    WORKING_TOL of bread_working.
    """

    ENV_NAME = "ToastOneSlotPair"

    # Height gained above its resting pose for bread_broken to count as lifted.
    LIFT_TOL = 0.04
    # Horizontal distance to bread_working within which bread_broken counts as
    # delivered to the working slot-pair.
    WORKING_TOL = 0.12
    # Gripper-bread distance above which the gripper counts as released.
    RELEASE_TH = 0.10
    # Normalized lever joint value at/above which the toaster is on.
    LEVER_ON_TH = 0.9

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        working = ep_meta["refs"]["working_slot_pair"]
        self.working_idx = 0 if working == "left" else 1
        # Lever joint for the working slot-pair (a string name, stable across
        # the model reload; its value is read live from the sim state).
        self.lever_jn = env.toaster._joint_names.get(f"lever_{self.working_idx}")
        # 3 segments + terminal "done" = index 3.
        self.expected_done_idx = 3
        # "PickPlace" is not a bank entry (no toaster->toaster atomic task).
        self.extra_subtasks = {SUBTASK_MOVE}

    def per_step(self, env, t: int) -> dict:
        bread_broken = np.array(
            env.sim.data.body_xpos[env.obj_body_id["bread_broken"]]
        )
        bread_working = np.array(
            env.sim.data.body_xpos[env.obj_body_id["bread_working"]]
        )
        gripper = np.array(
            env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        )
        lever = 0.0
        if self.lever_jn is not None:
            lever = float(
                env.toaster.get_joint_state(env, [self.lever_jn])[self.lever_jn]
            )
        return {
            "bread_broken": bread_broken,
            "bread_working": bread_working,
            "gripper": gripper,
            "lever": lever,
        }

    def resolve(self, history):
        T = len(history)
        bb = np.stack([h["bread_broken"] for h in history])  # (T, 3)
        bw = np.stack([h["bread_working"] for h in history])  # (T, 3)
        gp = np.stack([h["gripper"] for h in history])  # (T, 3)
        lever = np.array([h["lever"] for h in history])  # (T,)

        # t_toasted: the working lever reaches its 'on' position.
        on = np.where(lever >= self.LEVER_ON_TH)[0]
        t_toasted = int(on[0]) if on.size else T

        # t_lift: first frame bread_broken is lifted off the broken slot.
        lifted = bb[:, 2] > bb[0, 2] + self.LIFT_TOL
        lift_frames = np.where(lifted)[0]
        t_lift = int(lift_frames[0]) if lift_frames.size else T

        # t_placed: first frame AFTER the lift where bread_broken sits at the
        # working slot (close to bread_working) and the gripper has let go.
        d_work = np.linalg.norm(bb[:, :2] - bw[:, :2], axis=1)
        d_grip = np.linalg.norm(bb - gp, axis=1)
        delivered = (d_work < self.WORKING_TOL) & (d_grip > self.RELEASE_TH)
        delivered[: t_lift + 1] = False  # placement can only follow the lift
        placed = np.where(delivered[:t_toasted])[0]
        t_placed = int(placed[0]) if placed.size else t_toasted

        segments = [
            (t_lift, SUBTASK_MOVE, STAGE_PICK, PICK_LANG),
            (t_placed, SUBTASK_MOVE, STAGE_PLACE, PLACE_LANG),
            (t_toasted, SUBTASK_TOAST, STAGE_EXECUTE, TOAST_LANG),
            (T, DONE, STAGE_DONE, DONE_LANG),
        ]
        return self.build_result(
            T,
            segments,
            debug={
                "t_lift": t_lift,
                "t_placed": t_placed,
                "t_toasted": t_toasted,
                "lever": lever,
            },
        )
