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


# Open the cabinet, take the (non-electric) kettle out of it, carry it across to
# the sink and set it down in the sink. OpenCabinet + NavigateKitchen are bank
# atomics; the cabinet->sink pick-place has no exact bank name (the bank only has
# CabinetToCounter and CounterToSink), so it is an extra_subtask.
SUBTASK_OPEN = "OpenCabinet"
SUBTASK_MOVE = "PickPlaceCabinetToSink"
SUBTASK_NAV = "NavigateKitchen"

OPEN_LANG = "open the cabinet door"
NAV_LANG = "carry the kettle to the sink"
DONE_LANG = "task complete"


@register_labeler
class FillKettleLabeler(BaseSubtaskLabeler):
    """Labeler for the FillKettle composite task.

    Task: the kettle sits inside a (closed, double-door) cabinet by the sink.
    Open the cabinet, pick the kettle out of it, carry it to the sink and set it
    down inside the sink.

    Doc-canonical count = 4 (`mobile: Yes`): OpenCabinet + the kettle's pick-place,
    which is INTERRUPTED by a navigation from the cabinet to the sink ->
    PickPlaceCabinetToSink[pick] -> NavigateKitchen -> PickPlaceCabinetToSink[place].
    5 segments with the nav (expected_done_idx = 4). The nav is ADAPTIVE (gated on
    MIN_NAV_DIST) -- the cabinet and sink are adjacent, so a few demos transfer the
    kettle mostly with the arm and barely drive the base (4 segments).

    Boundaries (each from its OWN signal):
      - t_open:   the cabinet door (max fraction over the two hinge leaves) first
                  reaches OPEN_ABS of fully-open and stays -> OpenCabinet ends. It
                  starts closed (~0) and swings to ~1.0.
      - t_carry:  the kettle first displaces MOVE_TOL from its rest (in-cabinet)
                  pose and stays -> it has been grasped and is being pulled out /
                  carried. Ends the pick. (The kettle is extracted horizontally --
                  its Z barely changes -- so boundaries use kettle DISPLACEMENT, not
                  a Z-lift.)
      - t_arrive: the kettle first settles within ARRIVE_TOL of its final (in-sink)
                  pose and stays -> the carry/nav is done.
      - t_placed: the gripper RELEASES the kettle -- gripper->kettle distance rises
                  above the carry (hold) level -- after the kettle has arrived. The
                  kettle is large / handle-grabbed, so the place runs to the release,
                  not to first-settle (same as BoilPot / CoolKettle).
    """

    ENV_NAME = "FillKettle"

    OPEN_ABS = 0.85  # door fraction (of fully open) at/above which it counts as open
    OPEN_SUSTAIN = 5
    MOVE_TOL = 0.12  # kettle displaced this far from rest == grasped & being extracted
    MOVE_SUSTAIN = 5
    BASE_MOVE_TH = 0.003  # per-frame base speed above this == the base is driving
    DESCEND_DZ = 0.03  # kettle Z dropping this far below its held (carry) level == descending
    DESCEND_SUSTAIN = 5
    RELEASE_MARGIN = 0.05  # grip->kettle rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    REST_W = 20
    # The base must drive at least this far (PEAK displacement from the carry
    # start) to emit a nav. Peak, not net: several demos swing the base out ~0.5 m
    # then partly back, so a net-displacement gate wrongly dropped their nav (like
    # BoilPot's demo_93). The truly arm-only transfers sit at exactly 0.0 base
    # motion; the smallest real drive is ~0.33 -- so 0.20 lands in the clean gap.
    MIN_NAV_DIST = 0.20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.kettle_body_id = env.obj_body_id["obj"]
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        try:
            self.kettle_lang = env.get_obj_lang("obj")
        except Exception:
            self.kettle_lang = "kettle"

        # Cabinet door hinge joints (1 for a single door, 2 for a hinge/double
        # cabinet). Track each as a fraction of its own fully-open range and let
        # resolve() use the most-open leaf. `door_joint_names` reads a build-time
        # dict; fall back to a sim-model scan by the cabinet's name prefix so a
        # reloaded model still finds them. A few scenes place the kettle in an
        # OpenCabinet (open shelves, NO door) -- there the list stays empty and
        # self.has_door is False (handled as an anomaly in resolve/validate).
        cab = env.cab
        joint_names = list(cab.door_joint_names)
        if not joint_names:
            pref = cab.name
            joint_names = [
                n
                for n in env.sim.model.joint_names
                if n and n.startswith(pref) and ("door" in n.lower() or "hinge" in n.lower())
            ]
        self.door_adrs, self.door_scales = [], []
        for j in joint_names:
            try:
                adr = env.sim.model.get_joint_qpos_addr(j)
                jid = env.sim.model.joint_name2id(j)
                lo, hi = env.sim.model.jnt_range[jid]
            except Exception:
                continue
            scale = max(abs(float(hi)), abs(float(lo)), 1e-6)
            self.door_adrs.append(adr)
            self.door_scales.append(scale)
        self.has_door = len(self.door_adrs) > 0

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
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        doors = np.array(
            [abs(float(env.sim.data.qpos[a])) / s
             for a, s in zip(self.door_adrs, self.door_scales)]
        )
        return {
            "doors": doors,
            "kettle": env.sim.data.body_xpos[self.kettle_body_id].copy(),
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array(
                [float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]
            ),
        }

    # ---- helpers -------------------------------------------------------

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

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        doorv = np.array([h["doors"] for h in history])  # (T, n_doors)
        kettle = np.array([h["kettle"] for h in history])
        grip = np.array([h["grip"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history])
            if self.base_qpos_adrs
            else None
        )

        # open the cabinet: the door(s) swing open. Double-door (hinge) cabinets
        # are opened one leaf at a time, so OpenCabinet ends when the LAST leaf that
        # gets opened reaches its open threshold (a max-over-doors signal would end
        # at the FIRST leaf and mislabel the second-door swing as the pick). Only
        # leaves that actually open (reach OPEN_ABS) count -- a double cabinet where
        # the kettle is reachable through one door leaves the other closed. Skipped
        # entirely for a doorless (open-shelf) cabinet -- an anomaly, flagged.
        if self.has_door and doorv.shape[1] > 0:
            open_frames = []
            for i in range(doorv.shape[1]):
                fr = self._sustained_start(doorv[:, i] >= self.OPEN_ABS, 0, self.OPEN_SUSTAIN)
                if fr is not None:
                    open_frames.append(fr)
            t_open = max(open_frames) if open_frames else int(np.argmax(doorv.max(axis=1)))
        else:
            t_open = 0

        # pick -> carry: the kettle is grasped and pulled out of the cabinet (its
        # displacement from the in-cabinet rest pose grows). The kettle moves
        # horizontally, so use displacement rather than a Z-lift.
        rest = np.median(kettle[: self.REST_W], axis=0)
        kdisp = np.linalg.norm(kettle - rest, axis=1)
        t_carry = self._sustained_start(kdisp >= self.MOVE_TOL, t_open, self.MOVE_SUSTAIN)
        if t_carry is None:
            mv = np.where(kdisp >= self.MOVE_TOL)[0]
            t_carry = int(mv[0]) if mv.size else max(t_open, T - 1)
        t_carry = max(t_carry, t_open)

        # carry -> place: the kettle is carried (at height) over to the sink, then
        # LOWERED into it. The place is that final descent + release; the nav is the
        # transport up to it. Split at the DESCENT ONSET -- the first frame, after
        # the base has parked, where the kettle starts dropping below its held
        # (carry) height toward the sink. (A kettle-3D-arrival split ended the nav
        # only once the descent was nearly done: the base parks and the arm keeps
        # lowering the kettle well afterward -- demo_44 placed ~5 s late.)
        kz = kettle[:, 2]
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)
        moving = np.where(binst[t_carry:] > self.BASE_MOVE_TH)[0]
        t_base_stop = min(t_carry + int(moving[-1]) + 1, T - 1) if moving.size else t_carry
        # the kettle's held height right before the set-down (median Z over the
        # window ending where the drive stops)
        w0 = max(t_carry, t_base_stop - self.REST_W)
        hold_z = float(np.median(kz[w0: t_base_stop + 1])) if t_base_stop > w0 else float(kz[t_base_stop])
        t_arrive = self._sustained_start(kz < hold_z - self.DESCEND_DZ, t_base_stop, self.DESCEND_SUSTAIN)
        if t_arrive is None:
            t_arrive = t_base_stop
        t_arrive = max(t_arrive, t_carry)

        # place ends at the gripper RELEASE (grip->kettle rising above the carry
        # hold level), after the kettle has begun its set-down.
        gd = np.linalg.norm(grip - kettle, axis=1)
        hold_gd = float(np.median(gd[t_carry: max(t_arrive, t_carry + 1)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_arrive, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        if self.has_door:
            emit(t_open, SUBTASK_OPEN, STAGE_EXECUTE, OPEN_LANG)
        emit(t_carry, SUBTASK_MOVE, STAGE_PICK, f"pick up the {self.kettle_lang} from the cabinet")
        # nav: emitted if the base actually drove between the extraction and the
        # arrival (cabinet and sink are adjacent, so a few demos are arm-only). Use
        # the PEAK base displacement over the carry -- see MIN_NAV_DIST.
        if base_xy is not None and t_arrive > t_carry:
            off = base_xy[t_carry: t_arrive] - base_xy[t_carry]
            nav_disp = float(np.max(np.linalg.norm(off, axis=1)))
        else:
            nav_disp = 0.0
        if t_arrive > cursor and nav_disp >= self.MIN_NAV_DIST:
            emit(t_arrive, SUBTASK_NAV, STAGE_NAVIGATE, NAV_LANG)
        emit(t_placed, SUBTASK_MOVE, STAGE_PLACE, f"place the {self.kettle_lang} in the sink")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if not self.has_door:
            review.append(
                "cabinet has no door (open shelves) -- no OpenCabinet subtask; verify"
            )

        debug = {
            "t_open": t_open,
            "t_carry": t_carry,
            "t_arrive": t_arrive,
            "t_placed": t_placed,
            "nav_disp": round(nav_disp, 3),
            "door_peak": round(float(doorv.max()), 2) if doorv.size else 0.0,
            "has_door": self.has_door,
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

        exp_open = 1 if getattr(self, "has_door", True) else 0
        if count(SUBTASK_OPEN) != exp_open:
            issues.append(f"expected {exp_open} OpenCabinet, found {count(SUBTASK_OPEN)}")
        if count(SUBTASK_MOVE, STAGE_PICK) != 1:
            issues.append(f"expected 1 kettle pick, found {count(SUBTASK_MOVE, STAGE_PICK)}")
        if count(SUBTASK_MOVE, STAGE_PLACE) != 1:
            issues.append(f"expected 1 kettle place, found {count(SUBTASK_MOVE, STAGE_PLACE)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1):
            issues.append(f"expected 0-1 NavigateKitchen, found {n_nav}")
        return issues
