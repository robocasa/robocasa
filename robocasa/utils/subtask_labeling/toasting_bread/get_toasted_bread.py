import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    STAGE_PRESS,
    STAGE_WAIT,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Subtask 1: TurnOnToaster, split into PRESS (actively pushing the lever) and
# WAIT (lever held down while the toaster heats, until it pops back up).
SUBTASK_TOAST = "TurnOnToaster"
PRESS_LANG = "push down the lever of the toaster"
WAIT_LANG = "wait for the toaster to finish and the lever to pop up"

# Subtasks 2 + 4: PickPlaceToasterToCounter, split across a navigation between
# them (composite-task rule: pick-place WITH navigation in between = three
# subtasks Pick / Navigate / Place instead of the usual one).
SUBTASK_MOVE = "PickPlaceToasterToCounter"
PICK_LANG = "pick up the bread from the toaster"
PLACE_LANG = "place the bread on the plate"

# Subtask 3: NavigateKitchen (bank atomic). Robot drives from the toaster
# fixture to the dining counter while holding the bread.
SUBTASK_NAV = "NavigateKitchen"
NAV_LANG = "navigate to the plate on the dining counter"

DONE_LANG = "task complete"


@register_labeler
class GetToastedBreadLabeler(BaseSubtaskLabeler):
    """Labeler for the GetToastedBread composite task.

    Task: push the toaster's lever to start it, wait for the lever to pop back
    up, then move the bread to the plate on the dining counter.

    Four subtasks (doc-set count = 4), expanded into five active stages plus a
    terminal 'done':
      - TurnOnToaster              -> 'press' stage (push the lever down)
      - TurnOnToaster              -> 'wait'  stage (hold/idle until pop-up)
      - PickPlaceToasterToCounter  -> 'pick'  stage (reach + grasp + lift-off)
      - NavigateKitchen            -> 'navigate' stage (drive to dining counter)
      - PickPlaceToasterToCounter  -> 'place' stage (lower + release on plate)
    plus a terminal 'done', giving 6 segments.

    Each boundary is detected from its own signal -- no boundary cascades
    if one upstream detection fails. This is important for sim edge cases
    where e.g. the toaster lever stays stuck pushed (never pops back up):
    the demo can still complete fine and we still want clean labels.

    Boundaries:
      - t_press_done: working lever first reaches PRESS_FRAC of its peak
                  pushed-down value, restricted to the press-then-pop window.
                  Marks the moment active pushing ends and idling begins.
      - t_popped: working lever returns near 0 and stays there for POP_SUSTAIN
                  frames after first being pushed. May not detect if the
                  lever is stuck (sim bug) -- in that case we fall back for
                  the wait/pick boundary.
      - t_wait_done: t_popped if it fired; otherwise the first frame of a
                  GRIP_GRASP_SUSTAIN-long run where grip-to-bread distance
                  sits below GRIP_GRASP_TH (a tight threshold; this is the
                  actual grasp moment, not the gripper merely hovering
                  near the bread during the wait). If that also doesn't
                  fire, falls back to t_lift - PICK_FALLBACK_FRAMES.
                  Clamped to t_lift.
      - t_lift:   bread Z first rises LIFT_TOL above its rest level. Rest
                  level is the median bread Z over [0, t_pushed_first) --
                  the bread's undisturbed pre-press height. (Using the
                  post-press window doesn't work: the lever compresses the
                  bread during the press, then the natural pop-up returns
                  it to its initial height -- which would falsely trigger
                  lift detection if measured from the compressed level.)
      - t_nav_done: the frame just after the LAST frame in [t_lift, T)
                  where the smoothed mobile-base XY speed (norm of the
                  forward + side joint deltas, smoothed over NAV_SMOOTH_W
                  frames) exceeded NAV_MOVING_TH. Marks the handoff from
                  "base driving" to "arm placing". Robust to bursty drives
                  with pauses. Falls back to bread-near-plate (NAV_DONE_TOL)
                  if the base never crossed the moving threshold.
      - t_placed: bread is within PLACED_TOL of the plate AND the gripper has
                  released, AFTER t_nav_done.

    The "working" side (left vs right) is inferred from which lever shows the
    largest pushed-down displacement across the demo -- both joints are
    recorded per frame and resolved at the end.
    """

    ENV_NAME = "GetToastedBread"

    # Joint travel above which the lever counts as pushed. Empirically lev
    # range is 0..~0.05 in the pushed state.
    LEVER_PUSHED_TH = 0.025
    # Joint travel at/below which the lever counts as released (popped up).
    LEVER_RELEASED_TH = 0.005
    # Number of consecutive frames at the released level required to call
    # the pop event (avoids noise during the gradual climb back to 0).
    POP_SUSTAIN = 10
    # Fraction of the per-demo peak pushed-down value at which the press
    # stage is considered complete (lever is fully pushed; active pressing
    # ends here, wait stage begins).
    PRESS_FRAC = 0.9
    # Height gained above the bread's undisturbed pre-press rest height
    # for it to count as lifted by the agent. The lever's natural pop-up
    # only returns the bread to its pre-press height, so any rise above
    # that level must be the agent lifting it out.
    LIFT_TOL = 0.05
    # Tight gripper-to-bread distance for detecting an actual grasp (used
    # as the wait->pick fallback when the lever-pop detector fails). The
    # gripper sits at d~0.10-0.20 during the wait phase but drops to
    # d~0.06 once it actually closes on the bread; this threshold is set
    # tightly to fire only at the grasp event.
    GRIP_GRASP_TH = 0.065
    # Required consecutive sub-threshold frames to commit to grasp (avoids
    # firing on a brief flutter where the gripper passes close to the bread).
    GRIP_GRASP_SUSTAIN = 30
    # Final fallback: if neither lever pop nor grip-grasp fires, place
    # t_wait_done this many frames before t_lift (the typical pick-reach
    # duration in normal demos is ~150 frames).
    PICK_FALLBACK_FRAMES = 120
    # Mobile-base XY speed (m/frame, smoothed) above which the base counts
    # as actively driving. t_nav_done is one frame past the LAST frame in
    # [t_lift, T) whose smoothed speed exceeds this threshold.
    NAV_MOVING_TH = 0.005
    # Window over which base-speed is smoothed before thresholding.
    NAV_SMOOTH_W = 5
    # Fallback bread-to-plate XY distance for declaring nav done when the
    # base-speed signal isn't usable (e.g. base never moved).
    NAV_DONE_TOL = 0.30
    # XY distance to the plate body within which the bread counts as delivered.
    PLACED_TOL = 0.10
    # Gripper-to-bread distance above which the gripper counts as released.
    RELEASE_TH = 0.12

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.bread_body_id = env.obj_body_id["obj"]
        self.plate_body_id = env.obj_body_id["plate"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        # Toaster fixtures vary across scenes: single-slot ones have a single
        # *_lever_joint, two-slot ones have side{L,R} variants, and the fixture
        # group can be named toaster_main_group, toaster_left_group,
        # toaster_right_group, and may carry a numeric suffix (pretrain scenes
        # place multiple toasters). Track every lever on any toaster group; the
        # working one is the lever whose value moves the most during the demo,
        # so distractor toasters are ignored automatically. The toaster-OVEN is
        # excluded since its joints are door/knob/rack, never *_lever_joint.
        njnt = env.sim.model.njnt
        lever_names = []
        for j in range(njnt):
            try:
                nm = env.sim.model.joint_id2name(j)
            except Exception:
                nm = env.sim.model.joint(j).name
            if (
                nm
                and nm.startswith("toaster_")
                and "group" in nm
                and nm.endswith("_lever_joint")
            ):
                lever_names.append(nm)
        if not lever_names:
            raise RuntimeError(
                "No toaster lever joints found in env -- expected at least one "
                "joint matching 'toaster_*group*_lever_joint'."
            )
        self.lever_qpos_addrs = [
            env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)]
            for nm in lever_names
        ]
        self.lever_names = lever_names
        # Mobile-base XY joints (PandaOmron). Used to detect when base
        # driving stops -> place stage begins. Missing on a non-mobile env
        # variant; we fall back to bread-near-plate in resolve() if so.
        self.base_qpos_addrs = []
        for nm in (
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
        ):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_addrs.append(env.sim.model.jnt_qposadr[jid])
        # 5 active segments + terminal 'done' at index 5.
        self.expected_done_idx = 5

    def per_step(self, env, t: int) -> dict:
        return {
            "t": t,
            "bread": env.sim.data.body_xpos[self.bread_body_id].copy(),
            "plate": env.sim.data.body_xpos[self.plate_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "levers": np.array(
                [env.sim.data.qpos[a] for a in self.lever_qpos_addrs]
            ),
            "base_xy": np.array(
                [env.sim.data.qpos[a] for a in self.base_qpos_addrs]
            ),
        }

    def resolve(self, history):
        T = len(history)
        bread = np.array([h["bread"] for h in history])
        plate = np.array([h["plate"] for h in history])
        grip = np.array([h["grip"] for h in history])
        levers = np.array([h["levers"] for h in history])

        # Which lever was used? The one with the largest pushed-down peak.
        working_side = int(np.argmax(levers.max(axis=0)))
        lev = levers[:, working_side]
        working_name = self.lever_names[working_side]

        # t_pushed_first: first time the working lever crosses the pushed
        # threshold. Used as the start point for the pop search.
        pushed = np.where(lev >= self.LEVER_PUSHED_TH)[0]
        if pushed.size == 0:
            t_pushed_first = 0
        else:
            t_pushed_first = int(pushed[0])

        # t_popped: first frame at or after t_pushed_first where the lever
        # sits at released level for POP_SUSTAIN consecutive frames. Avoids
        # picking up the gradual return-to-0 spline mid-climb.
        released = lev < self.LEVER_RELEASED_TH
        t_popped = T - 1
        run = 0
        for t in range(t_pushed_first, T):
            if released[t]:
                run += 1
                if run >= self.POP_SUSTAIN:
                    t_popped = t - self.POP_SUSTAIN + 1
                    break
            else:
                run = 0

        # t_press_done: first frame in [t_pushed_first, t_popped) where the
        # lever reaches PRESS_FRAC of its peak pushed-down value -- active
        # pressing ends here, wait stage begins. Restricted to the actual
        # push window so we don't pick up post-pop noise.
        search_end = max(t_popped, t_pushed_first + 1)
        seg = lev[t_pushed_first:search_end]
        if seg.size:
            peak_lev = float(seg.max())
            press_done_th = peak_lev * self.PRESS_FRAC
            above = np.where(seg >= press_done_th)[0]
            t_press_done = (
                t_pushed_first + int(above[0]) if above.size else t_pushed_first
            )
        else:
            t_press_done = t_pushed_first
        # Make sure press strictly precedes wait, and wait strictly precedes
        # the pop boundary.
        t_press_done = min(max(t_press_done, t_pushed_first), max(t_popped - 1, 0))

        # XY distances: bread-to-plate and gripper-to-bread (used below for
        # the lift / reach / placed boundaries).
        plate_dist_xy = np.linalg.norm(bread[:, :2] - plate[:, :2], axis=1)
        grip_bread_dist = np.linalg.norm(bread - grip, axis=1)

        # t_lift: bread Z rises LIFT_TOL above its undisturbed pre-press
        # rest height. Rest height is the median bread Z over [0,
        # t_pushed_first) -- before the lever was first pushed and before
        # the bread was compressed. This is robust to stuck-lever sim
        # edge cases (rest height never depends on t_popped).
        if t_pushed_first > 0:
            rest_z = float(np.median(bread[:t_pushed_first, 2]))
        else:
            rest_z = float(bread[0, 2])
        lift_mask = bread[t_press_done:, 2] > rest_z + self.LIFT_TOL
        lf = np.where(lift_mask)[0]
        t_lift = (t_press_done + int(lf[0])) if lf.size else T - 1

        # t_wait_done: lever pop if detected (t_popped < T-1). Otherwise
        # detect the actual grasp as a sustained sub-threshold run of
        # grip-to-bread distance. If even that doesn't fire, fall back to
        # t_lift - PICK_FALLBACK_FRAMES so pick has a sane default length.
        # Clamped to t_lift so wait never overruns pick.
        wait_via = "pop"
        if t_popped < T - 1:
            t_wait_done = t_popped
        else:
            tight = grip_bread_dist[t_press_done:] < self.GRIP_GRASP_TH
            t_grasp = None
            run = 0
            for i in range(len(tight)):
                if tight[i]:
                    run += 1
                    if run >= self.GRIP_GRASP_SUSTAIN:
                        t_grasp = t_press_done + i - self.GRIP_GRASP_SUSTAIN + 1
                        break
                else:
                    run = 0
            if t_grasp is not None:
                t_wait_done = t_grasp
                wait_via = "grasp"
            else:
                t_wait_done = max(t_press_done, t_lift - self.PICK_FALLBACK_FRAMES)
                wait_via = "fallback"
        t_wait_done = min(max(t_wait_done, t_press_done), t_lift)

        # t_nav_done: switch control modes from base to arm. The mobile
        # base often drives in bursts with brief pauses, so the right
        # boundary is the frame AFTER the LAST base-motion burst, not the
        # first stop. Compute smoothed base XY speed, find the last frame
        # in [t_lift, T) above NAV_MOVING_TH, and place t_nav_done one
        # frame after it. Falls back to bread-near-plate if no base joints
        # were found or the base never crossed the moving threshold.
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_addrs else None
        t_nav_done = T - 1
        used_fallback = True
        if base_xy is not None and base_xy.shape[1] >= 2 and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy[:, :2], axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            kernel = np.ones(W) / W
            base_speed = np.convolve(inst, kernel, mode="same")
            post = base_speed[t_lift:]
            moving = np.where(post > self.NAV_MOVING_TH)[0]
            if moving.size:
                t_nav_done = min(T - 1, t_lift + int(moving[-1]) + 1)
                used_fallback = False
        if used_fallback:
            near = plate_dist_xy[t_lift:] < self.NAV_DONE_TOL
            nf = np.where(near)[0]
            t_nav_done = (t_lift + int(nf[0])) if nf.size else T - 1

        # t_placed: bread within PLACED_TOL of plate AND gripper released,
        # searched after t_nav_done.
        delivered = (plate_dist_xy < self.PLACED_TOL) & (
            grip_bread_dist > self.RELEASE_TH
        )
        df = np.where(delivered[t_nav_done:])[0]
        t_placed = (t_nav_done + int(df[0])) if df.size else T - 1

        segments = [
            (t_press_done, SUBTASK_TOAST, STAGE_PRESS, PRESS_LANG),
            (t_wait_done, SUBTASK_TOAST, STAGE_WAIT, WAIT_LANG),
            (t_lift, SUBTASK_MOVE, STAGE_PICK, PICK_LANG),
            (t_nav_done, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG),
            (t_placed, SUBTASK_MOVE, STAGE_PLACE, PLACE_LANG),
            (T, DONE, STAGE_DONE, DONE_LANG),
        ]

        debug = {
            "working_lever": working_name,
            "t_pushed_first": t_pushed_first,
            "t_press_done": t_press_done,
            "t_popped": t_popped,
            "t_wait_done": t_wait_done,
            "t_lift": t_lift,
            "t_nav_done": t_nav_done,
            "t_placed": t_placed,
            "wait_via": wait_via,
            "nav_done_via_fallback": int(used_fallback),
        }
        return self.build_result(T, segments, debug=debug)
