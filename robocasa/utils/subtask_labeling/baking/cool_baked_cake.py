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

# Subtasks 2 + 7: slide the oven rack out (to reach the cake) then back in
# (before closing the door). Same bank atomic, different lang per direction.
SUBTASK_RACK = "SlideOvenRack"
RACK_OUT_LANG = "slide the oven rack out"
RACK_IN_LANG = "slide the oven rack in"

# Subtasks 3 + 5: move the cake oven -> plate, split across an optional
# navigation into pick / (NavigateKitchen) / place. No bank atomic covers
# oven -> counter (the bank only has the reverse, PickPlaceCounterToOven), so
# the PickPlace name is a declared extension; unchanged across pick/place.
SUBTASK_MOVE = "PickPlaceOvenToCounter"

# NavigateKitchen (bank atomic): carry the cake to the plate, and the drive back
# to the oven to slide the rack in / close the door. Adaptive -- present only
# when the base actually drives (many scenes place the plate within arm reach).
SUBTASK_NAV = "NavigateKitchen"
NAV_TO_PLATE_LANG = "navigate to the plate on the counter"
NAV_TO_OVEN_LANG = "navigate back to the oven"

# Subtask 8: close the oven door (bank atomic, single EXECUTE stage).
SUBTASK_CLOSE = "CloseOven"
CLOSE_LANG = "close the oven door"

DONE_LANG = "task complete"


@register_labeler
class CoolBakedCakeLabeler(BaseSubtaskLabeler):
    """Labeler for the CoolBakedCake composite task.

    Task: take the baked cake out of the (closed) oven, place it on the plate on
    the counter to cool, then slide the rack back in and close the oven door.

    Doc-canonical count = 8 (`mobile: Yes`). The full decomposition, in the order
    the events physically must occur:
      1. OpenOven               [execute]  -> swing the door open
      2. SlideOvenRack          [execute]  -> pull the rack (with the cake) out
      3. PickPlaceOvenToCounter [pick]     -> reach + grasp + lift the cake out
      4. NavigateKitchen        [navigate] -> carry the cake to the plate (opt.)
      5. PickPlaceOvenToCounter [place]    -> lower + release on the plate
      6. NavigateKitchen        [navigate] -> drive back to the oven (opt.)
      7. SlideOvenRack          [execute]  -> push the (empty) rack back in
      8. CloseOven              [execute]  -> swing the door shut
    plus a terminal 'done'.

    The two NavigateKitchen subtasks are ADAPTIVE (MIN_NAV_DIST): in most scenes
    the oven and plate are within arm reach so the base never drives and both
    navs vanish (6 active subtasks); in far-plate scenes the base drives out and
    back and both appear (8 active subtasks). So the segment count varies and
    expected_done_idx is set per demo; validation is structural.

    Each boundary is computed from its OWN signal (door joint / rack joint /
    cake pose / base speed) so one failed detection never cascades into the rest.

    Boundaries:
      - t_open_done:  door joint first crosses OPEN_ABS (absolute, robust to the
                  door creeping the last few percent open passively after the
                  active push -- a 0.9*peak threshold would land that creep).
      - t_rackout_done: the working rack (auto-detected as the rack joint with
                  the largest travel -- the tray may sit on rack0 OR rack1) first
                  reaches OUT_DONE_FRAC of its out travel.
      - t_lift:   cake Z first rises LIFT_TOL above its rest height AND stays up
                  for LIFT_SUSTAIN frames (a sustained rise -- the rack jerking
                  out bobs the cake up for a few frames, which is not the pick).
                  Pick -> Place boundary.
      - t_nav_done (carry): frame after the LAST base-motion burst in
                  [t_lift, t_placed], gated on MIN_NAV_DIST net displacement.
      - t_placed: cake within PLACED_TOL xy of the plate AND gripper released,
                  after the carry.
      - t_door_descend: LAST frame the door is >= DOOR_OPEN_FRAC of its peak --
                  i.e. the door starts closing right after. Separates the rack
                  slide-in (before) from the door close (after).
      - SlideOvenRack[in] is CONDITIONAL: present only if the rack shows a
                  distinct inward motion (>= a fraction of its out travel) in
                  [t_placed, t_door_descend]. In some demos the robot leaves the
                  rack part way out and the door close alone pushes it in -- there
                  is then no separate slide-in subtask (drop belongs to CloseOven).
      - t_rackin_done: when present, the frame the rack first reaches its inward
                  plateau (within RACK_SETTLE of the window min).
      - t_nav_done (return): base-motion burst before the slide-in / close.
      - t_close_done: door returns near closed after it was last open. CloseOven
                  ends here; any tail is 'done'.
    """

    ENV_NAME = "CoolBakedCake"

    # --- door (closed ~0, open ~1.15) ---
    OPEN_ABS = 0.55  # door counts as open above this (well below the ~1.15 peak)
    CLOSE_ABS = 0.15  # door counts as closed below this
    DOOR_OPEN_FRAC = 0.90  # >= this fraction of peak counts as fully open (for the
    # door-descent detector: the last such frame is where the door starts closing)
    # --- rack (in ~0, out ~0.25-0.33) ---
    RACK_MOVE_TH = 0.05  # min rack travel for a real SlideOvenRack to have happened
    OUT_DONE_FRAC = 0.80  # rack is "out" once it reaches this fraction of its travel
    # Min inward rack motion after the place to count as an active
    # SlideOvenRack[in]; below this the rack merely stayed out and the door close
    # alone pushes it in (no separate slide-in subtask). Taken as a FRACTION of
    # the rack's out travel (with a small absolute floor) -- some ovens have a
    # short rack (travel ~0.16), so a fixed absolute threshold would misread a
    # real half-range slide-in as "no slide-in".
    RACK_SLIDE_MIN_FRAC = 0.30
    RACK_SLIDE_MIN_ABS = 0.04
    RACK_SETTLE = 0.03  # rack is at its inward plateau within this of the window min
    # --- cake pick/place ---
    GRASP_TH = 0.05  # grip->cake distance for an actual grasp
    LIFT_TOL = 0.06  # cake Z rise above rest to count as lifted out
    LIFT_SUSTAIN = 20  # frames the cake must stay lifted (rejects the transient bob
    # when the rack jerks out -- that momentary bounce is not the pick-up)
    PLACED_TOL = 0.12  # cake-to-plate xy within which the cake is delivered
    RELEASE_TH = 0.12  # grip->cake distance above which the gripper has released
    REST_W = 10  # window for rest-height / final-pose medians
    # --- mobile base ---
    NAV_MOVING_TH = 0.005  # smoothed base XY speed (m/frame) counting as driving
    NAV_SMOOTH_W = 5
    MIN_NAV_DIST = 0.40  # net base displacement to emit a NavigateKitchen

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.cake_body_id = env.obj_body_id["cake"]
        self.plate_body_id = env.obj_body_id["container"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.cake_lang = env.get_obj_lang("cake")
        except Exception:
            self.cake_lang = "cake"

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
            "cake": env.sim.data.body_xpos[self.cake_body_id].copy(),
            "plate": env.sim.data.body_xpos[self.plate_body_id].copy(),
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
    def _last(mask):
        w = np.where(mask)[0]
        return int(w[-1]) if w.size else None

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
        disp = float(np.linalg.norm(base_xy[min(nav_end, len(base_xy) - 1)] - base_xy[a]))
        return nav_end if disp >= self.MIN_NAV_DIST else a

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        door = np.array([h["door"] for h in history])
        racks = np.array([h["racks"] for h in history])  # (T, n_rack)
        cake = np.array([h["cake"] for h in history])
        plate = np.array([h["plate"] for h in history])
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

        # -- door open / close boundaries --
        t_open_done = self._first(door > self.OPEN_ABS)
        if t_open_done is None:
            t_open_done = 0
        t_close_start = self._last(door > self.OPEN_ABS)
        if t_close_start is None:
            t_close_start = T - 1
        cf = np.where(door[t_close_start:] < self.CLOSE_ABS)[0]
        t_close_done = t_close_start + int(cf[0]) if cf.size else T - 1

        # Door-descent start: the last frame the door is (near) fully open -- the
        # active close begins right after. Separates the rack slide-in from the
        # door close.
        peak_door = float(door.max())
        t_door_descend = self._last(door >= self.DOOR_OPEN_FRAC * peak_door)
        if t_door_descend is None:
            t_door_descend = t_close_start

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

        # -- cake pick / lift / place --
        grip_cake = np.linalg.norm(grip - cake, axis=1)
        cake_plate_xy = np.linalg.norm(cake[:, :2] - plate[:, :2], axis=1)
        pick_start = t_rackout_done

        rest_z = float(np.median(cake[: max(pick_start, self.REST_W), 2]))
        # Sustained lift: reject the transient bob when the rack jerks out.
        lifted = cake[:, 2] > rest_z + self.LIFT_TOL
        t_lift = self._sustained_start(lifted, pick_start, self.LIFT_SUSTAIN)
        if t_lift is None:  # fallback: first bare crossing
            lf = np.where(lifted[pick_start:])[0]
            t_lift = pick_start + int(lf[0]) if lf.size else T - 1

        # (grasp frame is informational -- the pick spans reach+grasp+lift)
        gf = np.where(grip_cake[pick_start:] < self.GRASP_TH)[0]
        t_grasp = pick_start + int(gf[0]) if gf.size else t_lift

        delivered = (cake_plate_xy < self.PLACED_TOL) & (grip_cake > self.RELEASE_TH)
        df = np.where(delivered[t_lift:])[0]
        t_placed = t_lift + int(df[0]) if df.size else T - 1

        # -- rack slide-in (conditional): only if the rack shows a distinct inward
        # motion between the place and the door close. Otherwise the robot left
        # the rack out and the door-close alone pushes it in (belongs to Close).
        t_rackin_done = None
        if rack_moves:
            a = min(t_placed, T - 1)
            b = max(t_door_descend, a + 1)
            seg = out_amt[a:b]
            win_min = float(seg.min())
            slide_min = max(self.RACK_SLIDE_MIN_ABS, self.RACK_SLIDE_MIN_FRAC * rrange)
            if (out_amt[a] - win_min) >= slide_min:
                rf = np.where(seg <= win_min + self.RACK_SETTLE)[0]
                t_rackin_done = a + int(rf[0]) if rf.size else b

        # -- assemble contiguous segments in physical order --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        def maybe_nav(end, lang):
            nav_end = self._nav_end(base_speed, base_xy, cursor, int(min(max(end, cursor), T)))
            if nav_end > cursor:
                emit(nav_end, SUBTASK_NAV, STAGE_NAVIGATE, lang)

        emit(t_open_done, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        if rack_moves:
            emit(t_rackout_done, SUBTASK_RACK, STAGE_EXECUTE, RACK_OUT_LANG)
        emit(t_lift, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.cake_lang} from the oven")
        maybe_nav(t_placed, NAV_TO_PLATE_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.cake_lang} on the plate")
        # Return drive (if any) happens before the slide-in / door close.
        maybe_nav(t_rackin_done if t_rackin_done is not None else t_door_descend,
                  NAV_TO_OVEN_LANG)
        if t_rackin_done is not None:
            emit(t_rackin_done, SUBTASK_RACK, STAGE_EXECUTE, RACK_IN_LANG)
        emit(t_close_done, SUBTASK_CLOSE, STAGE_EXECUTE, CLOSE_LANG)
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if rack_moves and t_rackin_done is None:
            review.append(
                "no active rack slide-in -- the robot left the rack out and the "
                "door close pushed it in (no separate SlideOvenRack[in])"
            )

        debug = {
            "working_rack": working_rack,
            "rack_moves": int(rack_moves),
            "rrange": round(rrange, 3),
            "t_open_done": t_open_done,
            "t_rackout_done": t_rackout_done,
            "t_grasp": t_grasp,
            "t_lift": t_lift,
            "t_placed": t_placed,
            "t_door_descend": t_door_descend,
            "t_rackin_done": -1 if t_rackin_done is None else t_rackin_done,
            "t_close_start": t_close_start,
            "t_close_done": t_close_done,
            "n_segments": len(segments),
        }
        result = self.build_result(T, segments, debug=debug)
        result.review = review
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
        if count(SUBTASK_CLOSE) != 1:
            issues.append(f"expected 1 CloseOven, found {count(SUBTASK_CLOSE)}")
        n_pick = count(SUBTASK_MOVE, STAGE_PICK)
        n_place = count(SUBTASK_MOVE, STAGE_PLACE)
        if n_pick != 1:
            issues.append(f"expected 1 cake pick, found {n_pick}")
        if n_place != 1:
            issues.append(f"expected 1 cake place, found {n_place}")
        # SlideOvenRack: 1 (slide-out; the robot leaves the rack out and the door
        # close pushes it in) or 2 (slide-out + a distinct slide-in). 0 only if the
        # rack never moved (cake reachable without pulling the rack -- unobserved).
        n_rack = count(SUBTASK_RACK)
        if n_rack not in (0, 1, 2):
            issues.append(f"expected 0-2 SlideOvenRack, found {n_rack}")
        return issues
