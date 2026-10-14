import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_NAVIGATE,
    STAGE_PICK,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Subtask 1: open the oven door (bank atomic, single EXECUTE stage).
SUBTASK_OPEN = "OpenOven"
OPEN_LANG = "open the oven door"

# Subtask 2: slide the oven rack out to reach the tray (bank atomic).
SUBTASK_RACK = "SlideOvenRack"
RACK_OUT_LANG = "slide the oven rack out"

# Subtasks 3 + 5: move the cookie tray oven -> counter, split across an optional
# navigation into pick / (NavigateKitchen) / place. No bank atomic covers
# oven -> counter (the bank only has the reverse, PickPlaceCounterToOven), so
# the PickPlace name is a declared extension; unchanged across pick/place. The
# moved object is the oven_tray itself (a receptacle) -- the two cookies ride on
# it and are not manipulated separately.
SUBTASK_MOVE = "PickPlaceOvenToCounter"
PICK_LANG = "pick up the tray of cookies from the oven"
PLACE_LANG = "place the tray of cookies on the counter"

# NavigateKitchen (bank atomic): carry the tray from the oven to the counter.
# Adaptive -- present only when the base actually drives (some scenes place the
# cooling counter within arm reach of the oven).
SUBTASK_NAV = "NavigateKitchen"
NAV_LANG = "navigate to the counter"

DONE_LANG = "task complete"


@register_labeler
class CoolBakedCookiesLabeler(BaseSubtaskLabeler):
    """Labeler for the CoolBakedCookies composite task.

    Task: open the (closed) oven, slide the rack out, take the tray of baked
    cookies out of the oven and set it on the counter to cool. Unlike
    CoolBakedCake, the task ENDS once the tray is on the counter -- there is no
    return trip, no rack slide-in, and no door close (the door and rack are both
    left open at the end). That is why the doc-canonical count is 5, not 8.

    Doc-canonical count = 5 (`mobile: Yes`). The full decomposition, in the order
    the events physically must occur:
      1. OpenOven               [execute]  -> swing the door open
      2. SlideOvenRack          [execute]  -> pull the rack (with the tray) out
      3. PickPlaceOvenToCounter [pick]     -> reach + grasp + lift the tray out
      4. NavigateKitchen        [navigate] -> carry the tray to the counter (opt.)
      5. PickPlaceOvenToCounter [place]    -> lower + release on the counter
    plus a terminal 'done'.

    The moved object is the oven_tray itself; the two cookies sit on it and ride
    along (their Z tracks the tray's), so pick/place are resolved from the tray
    pose alone.

    NavigateKitchen is ADAPTIVE (MIN_NAV_DIST): scenes with the counter within
    arm reach never drive, so the nav vanishes (4 active subtasks); far-counter
    scenes drive and it appears (5). The segment count therefore varies and
    expected_done_idx is set per demo; validation is structural.

    Each boundary is computed from its OWN signal (door joint / rack joint /
    tray pose / base speed) so one failed detection never cascades into the rest.

    Boundaries:
      - t_open_done:  door joint first crosses OPEN_ABS (absolute, robust to the
                  door creeping the last few percent open passively after the
                  active push -- a 0.9*peak threshold would land that creep).
      - t_rackout_done: the working rack (auto-detected as the rack joint with
                  the largest travel -- the tray may sit on rack0 OR rack1) first
                  reaches OUT_DONE_FRAC of its out travel.
      - t_lift:   tray Z first rises LIFT_TOL above its rest height AND stays up
                  for LIFT_SUSTAIN frames (a sustained rise rejects any transient
                  bob when the rack jerks out). Pick -> Place boundary.
      - t_nav_done (carry): frame after the LAST base-motion burst in
                  [t_lift, t_placed], gated on MIN_NAV_DIST net displacement.
      - t_placed: the tray is set on the counter and released -- it has settled
                  within PLACE_Z_TOL of its final resting height AND the gripper
                  has let go (grip->tray distance > RELEASE_TH). There is no
                  target plate/container object to measure against (the tray goes
                  directly on the counter), so placement is read from the tray's
                  own settle + the release, not a distance-to-target. RELEASE_TH
                  is larger than for a small object because the eef site stays
                  ~0.15 from the (large) tray body center even while grasping it.
    """

    ENV_NAME = "CoolBakedCookies"

    # --- door (closed ~0, open ~1.15) ---
    OPEN_ABS = 0.55  # door counts as open above this (well below the ~1.15 peak)
    # --- rack (in ~0, out ~0.25-0.33) ---
    RACK_MOVE_TH = 0.05  # min rack travel for a real SlideOvenRack to have happened
    OUT_DONE_FRAC = 0.80  # rack is "out" once it reaches this fraction of its travel
    # --- tray pick/place ---
    GRASP_TH = 0.16  # grip->tray distance for an actual grasp (tray body is large,
    # so the grasp distance to its center is bigger than for a small object)
    LIFT_TOL = 0.08  # tray Z rise above rest to count as lifted out (rise is ~0.35)
    LIFT_SUSTAIN = 20  # frames the tray must stay lifted (rejects any transient bob)
    RELEASE_TH = 0.20  # grip->tray distance above which the gripper has released.
    # The held/released gap is ~[0.14, 0.29]; 0.20 sits inside it.
    PLACE_Z_TOL = 0.06  # tray within this of its final settled Z == set on counter
    REST_W = 10  # window for rest-height / final-pose medians
    # --- mobile base ---
    NAV_MOVING_TH = 0.005  # smoothed base XY speed (m/frame) counting as driving
    NAV_SMOOTH_W = 5
    MIN_NAV_DIST = 0.40  # net base displacement to emit a NavigateKitchen

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.tray_body_id = env.obj_body_id["oven_tray"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]

        # Oven door joint + all rack joints on this oven. The tray sits on rack0
        # in single-rack ovens but on rack1 in some multi-rack ovens, so we track
        # every rack joint and let resolve() pick the one that actually moves.
        oven = env.oven
        self.door_qpos_adr = env.sim.model.get_joint_qpos_addr(
            oven._joint_names["door"]
        )
        prefix = oven.naming_prefix
        rack_names = [
            n
            for n in env.sim.model.joint_names
            if n.startswith(prefix) and "rack" in n and n.endswith("_joint")
        ]
        self.rack_qpos_adrs = [
            env.sim.model.get_joint_qpos_addr(n) for n in rack_names
        ]
        self.rack_names = rack_names

        # Mobile-base XY joints (PandaOmron). Missing on a fixed-base variant.
        self.base_qpos_adrs = []
        for nm in (
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
        ):
            try:
                jid = env.sim.model.joint_name2id(nm)
            except Exception:
                continue
            self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[jid])

        self.extra_subtasks = {SUBTASK_MOVE}
        self.expected_done_idx = None  # set per demo in resolve()
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        return {
            "door": float(env.sim.data.qpos[self.door_qpos_adr]),
            "racks": np.array(
                [float(env.sim.data.qpos[a]) for a in self.rack_qpos_adrs]
            ),
            "tray": env.sim.data.body_xpos[self.tray_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array(
                [float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]
            ),
        }

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _first(mask):
        w = np.where(mask)[0]
        return int(w[0]) if w.size else None

    @staticmethod
    def _sustained_start(mask, start, sustain):
        """First frame >= start that begins a run of `sustain` consecutive True
        values in `mask`, or None."""
        run = 0
        for t in range(int(start), len(mask)):
            if mask[t]:
                run += 1
                if run >= sustain:
                    return t - sustain + 1
            else:
                run = 0
        return None

    def _nav_end(self, base_speed, base_xy, a, b):
        """Frame after the last base-motion burst in [a, b), or a if the base
        did not drive at least MIN_NAV_DIST over the window."""
        if base_xy is None or b - a < 2:
            return a
        seg = base_speed[a:b]
        moving = np.where(seg > self.NAV_MOVING_TH)[0]
        if not moving.size:
            return a
        nav_end = min(a + int(moving[-1]) + 1, b)
        disp = float(
            np.linalg.norm(base_xy[min(nav_end, len(base_xy) - 1)] - base_xy[a])
        )
        return nav_end if disp >= self.MIN_NAV_DIST else a

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        door = np.array([h["door"] for h in history])
        racks = np.array([h["racks"] for h in history])  # (T, n_rack)
        tray = np.array([h["tray"] for h in history])
        grip = np.array([h["grip"] for h in history])

        base_xy = (
            np.array([h["base_xy"] for h in history])
            if self.base_qpos_adrs
            else None
        )
        if base_xy is not None and T >= 2:
            inst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            W = self.NAV_SMOOTH_W
            base_speed = np.convolve(inst, np.ones(W) / W, mode="same")
            base_speed = np.concatenate([base_speed, base_speed[-1:]])
        else:
            base_speed = np.zeros(T)
            base_xy = None

        # -- door open boundary (no close: the door is left open at the end) --
        t_open_done = self._first(door > self.OPEN_ABS)
        if t_open_done is None:
            t_open_done = 0

        # -- working rack (largest travel) --
        if racks.shape[1] > 0:
            travel = racks.max(axis=0) - racks.min(axis=0)
            j = int(np.argmax(travel))
            rack = racks[:, j]
            r_in = float(np.median(rack[: self.REST_W]))
            rrange = float(rack.max() - r_in)
            working_rack = self.rack_names[j]
        else:
            rack = np.zeros(T)
            r_in = 0.0
            rrange = 0.0
            working_rack = None
        rack_moves = rrange >= self.RACK_MOVE_TH
        out_amt = rack - r_in

        if rack_moves:
            t_rackout_done = self._first(out_amt >= self.OUT_DONE_FRAC * rrange)
            if t_rackout_done is None:
                t_rackout_done = 0
        else:
            # No rack slide at all: the pick starts right after the door opens.
            t_rackout_done = t_open_done

        # -- tray pick / lift / place --
        grip_tray = np.linalg.norm(grip - tray, axis=1)
        pick_start = t_rackout_done

        rest_z = float(np.median(tray[: max(pick_start, self.REST_W), 2]))
        # Sustained lift: reject any transient bob when the rack jerks out.
        lifted = tray[:, 2] > rest_z + self.LIFT_TOL
        t_lift = self._sustained_start(lifted, pick_start, self.LIFT_SUSTAIN)
        if t_lift is None:  # fallback: first bare crossing
            lf = np.where(lifted[pick_start:])[0]
            t_lift = pick_start + int(lf[0]) if lf.size else T - 1

        # (grasp frame is informational -- the pick spans reach+grasp+lift)
        gf = np.where(grip_tray[pick_start:] < self.GRASP_TH)[0]
        t_grasp = pick_start + int(gf[0]) if gf.size else t_lift

        # Placement: no target plate exists (the tray goes on the counter), so
        # read it from the tray settling at its final height AND the release.
        final_z = float(np.median(tray[-self.REST_W:, 2]))
        settled = np.abs(tray[:, 2] - final_z) < self.PLACE_Z_TOL
        released = grip_tray > self.RELEASE_TH
        delivered = settled & released
        df = np.where(delivered[t_lift:])[0]
        t_placed = t_lift + int(df[0]) if df.size else T - 1

        # -- assemble contiguous segments in physical order --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang):
            nav_end = self._nav_end(
                base_speed, base_xy, cursor, int(min(max(end, cursor), T))
            )
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        emit(t_open_done, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        if rack_moves:
            emit(t_rackout_done, SUBTASK_RACK, STAGE_EXECUTE, RACK_OUT_LANG)
        emit(t_lift, SUBTASK_MOVE, STAGE_PICK, PICK_LANG)
        maybe_nav(t_placed, NAV_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, PLACE_LANG)
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        debug = {
            "working_rack": working_rack,
            "rack_moves": int(rack_moves),
            "rrange": round(rrange, 3),
            "t_open_done": t_open_done,
            "t_rackout_done": t_rackout_done,
            "t_grasp": t_grasp,
            "t_lift": t_lift,
            "t_placed": t_placed,
            "final_z": round(final_z, 3),
            "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
        return result

    # ---- structural validation ----------------------------------------

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
            issues.append(f"expected 1 OpenOven, found {count(SUBTASK_OPEN)}")
        n_pick = count(SUBTASK_MOVE, STAGE_PICK)
        n_place = count(SUBTASK_MOVE, STAGE_PLACE)
        if n_pick != 1:
            issues.append(f"expected 1 tray pick, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 tray place, found {n_place}")
        # SlideOvenRack: exactly 1 (slide-out only; the rack is left out -- there
        # is no slide-in in this task). 0 only if the rack never moved (unobserved).
        n_rack = count(SUBTASK_RACK)
        if n_rack not in (0, 1):
            issues.append(f"expected 0-1 SlideOvenRack, found {n_rack}")
        # This task never closes the oven; a CloseOven label would be a bug.
        if count("CloseOven") != 0:
            issues.append("unexpected CloseOven (task leaves the oven open)")
        return issues
