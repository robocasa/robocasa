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


# Carry the kettle (from the already-open cabinet) and the pot (from the counter)
# onto two stove burners, then turn both burners on. Each carry may be interrupted
# by a navigation. PickPlaceCounterToStove, NavigateKitchen and TurnOnStove are
# bank atomics; the cabinet->stove kettle move has no exact bank name (the bank has
# CabinetToCounter, not CabinetToStove) so it is an extra_subtask.
SUBTASK_KETTLE = "PickPlaceCabinetToStove"   # extra: kettle, cabinet -> stove
SUBTASK_POT = "PickPlaceCounterToStove"      # bank atomic: pot, counter -> stove
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_TURNON = "TurnOnStove"

DONE_LANG = "task complete"

# object name -> (subtask name, source phrase, fallback lang)
OBJ_INFO = {
    "obj2": (SUBTASK_KETTLE, "cabinet", "kettle"),  # kettle in the cabinet
    "obj": (SUBTASK_POT, "counter", "pot"),         # pot on the counter
}


@register_labeler
class HeatMultipleWaterLabeler(BaseSubtaskLabeler):
    """Labeler for the HeatMultipleWater composite task.

    Task: heat water in a pot and a kettle. The kettle sits in an ALREADY-OPEN
    cabinet, the pot on the counter, and every stove knob starts off. Carry the
    kettle onto a burner, carry the pot onto another burner, then turn both burners
    on.

    Doc-canonical count = 4 (`mobile: Yes`): kettle move + pot move + two burner
    turn-ons. Each object's pick-place may be INTERRUPTED by a navigation (the
    fixtures are spread around the stove) -> per move: [pick] -> [NavigateKitchen]
    -> [place]. Full structure (navs adaptive):
      kettle[pick] -> [nav] -> kettle[place] -> pot[pick] -> [nav] -> pot[place] ->
      TurnOnStove(A) -> TurnOnStove(B) -> done.

    The two objects are carried in whichever ORDER the demo uses (resolved from
    which moves first -- the kettle leads in practice). Each object is large /
    handle-grabbed (like BoilPot's pot), so boundaries come from the object's
    DISPLACEMENT and the gripper RELEASE, not a Z-lift or a tight grasp:
      - obj_move (pick end): the object first displaces MOVE_TOL from its rest pose
                 (after the previous object was released) -> being carried.
      - obj_arrive (nav end / place start): the DESCENT ONSET -- the first frame,
                 after the base parks during this carry, where the object starts
                 dropping onto the burner (same idea as FillKettle).
      - obj_placed (place end): the gripper RELEASES the object (grip->obj distance
                 rises above the carry hold level), after it arrives.
      - a NavigateKitchen is emitted for a carry iff the base's PEAK displacement
                 over [obj_move, obj_arrive] >= MIN_NAV_DIST.
      - turn-ons: the two knobs whose normalized angle enters the "on" range
                 [KNOB_ON, 2*pi-KNOB_ON] (matches stove.is_burner_on); every knob
                 starts off. Ordered by when each turns on; the burner loc is spoken.
    """

    ENV_NAME = "HeatMultipleWater"

    MOVE_TOL = 0.10  # object displaced this far from rest == grasped & being carried
    MOVE_SUSTAIN = 5
    HOLD_W = 40  # window after obj_move used to estimate the carry (hold) grip dist
    RELEASE_MARGIN = 0.08  # grip->obj rising this far above carry level == let go
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 20  # a carry lasts at least this many frames before a release counts
    BASE_MOVE_TH = 0.003  # per-frame base speed above this == the base is driving
    DESCEND_DZ = 0.03  # object Z dropping this far below its carry height == descending
    DESCEND_SUSTAIN = 5
    MIN_NAV_DIST = 0.20  # peak base displacement over a carry to emit a nav
    KNOB_ON = 0.35
    REST_W = 20

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.obj_body_ids = {o: env.obj_body_id[o] for o in OBJ_INFO}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.obj_lang = {}
        for o, (_, _, fallback) in OBJ_INFO.items():
            try:
                self.obj_lang[o] = env.get_obj_lang(o)
            except Exception:
                self.obj_lang[o] = fallback

        # stove knob joints (all real burners) -- resolve() picks the two turned on
        stove = env.stove
        prefix = stove.naming_prefix
        self.knob_locs, self.knob_adrs = [], []
        for loc, joint in stove.knob_joints.items():
            if joint is None:
                continue
            try:
                adr = env.sim.model.get_joint_qpos_addr(f"{prefix}knob_{loc}_joint")
            except Exception:
                continue
            self.knob_locs.append(loc)
            self.knob_adrs.append(adr)

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

        self.extra_subtasks = {SUBTASK_KETTLE}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "knobs": np.array([float(env.sim.data.qpos[a]) for a in self.knob_adrs]),
            "base_xy": np.array(
                [float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]
            ),
        }
        for o in OBJ_INFO:
            d[o] = env.sim.data.body_xpos[self.obj_body_ids[o]].copy()
        return d

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

    def _carry(self, opos, grip, base_xy, binst, cursor, T):
        """Resolve one object's carry starting at/after `cursor`.

        Returns dict with t_move, t_arrive, t_placed, nav_disp (peak base disp over
        the carry). Boundaries: displacement from rest (move), descent-onset after
        the base parks (arrive), gripper release (placed).
        """
        rest = np.median(opos[: self.REST_W], axis=0)
        disp = np.linalg.norm(opos - rest, axis=1)
        t_move = self._sustained_start(disp >= self.MOVE_TOL, cursor, self.MOVE_SUSTAIN)
        if t_move is None:
            mv = np.where(disp[cursor:] >= self.MOVE_TOL)[0]
            t_move = cursor + int(mv[0]) if mv.size else min(cursor + 1, T - 1)
        t_move = max(t_move, cursor)

        gd = np.linalg.norm(grip - opos, axis=1)
        hold_gd = float(np.median(gd[t_move: min(t_move + self.HOLD_W, T)]))
        released = gd > hold_gd + self.RELEASE_MARGIN
        t_placed = self._sustained_start(released, t_move + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_placed is None:
            t_placed = T - 1

        # descent onset (nav -> place split), within [t_move, t_placed]
        oz = opos[:, 2]
        moving = np.where(binst[t_move: t_placed] > self.BASE_MOVE_TH)[0]
        t_base_stop = min(t_move + int(moving[-1]) + 1, t_placed) if moving.size else t_move
        w0 = max(t_move, t_base_stop - self.REST_W)
        hold_z = float(np.median(oz[w0: t_base_stop + 1])) if t_base_stop > w0 else float(oz[t_base_stop])
        t_arrive = self._sustained_start(oz < hold_z - self.DESCEND_DZ, t_base_stop, self.DESCEND_SUSTAIN)
        if t_arrive is None:
            t_arrive = t_base_stop
        t_arrive = int(min(max(t_arrive, t_move), t_placed))

        if base_xy is not None and t_arrive > t_move:
            off = base_xy[t_move: t_arrive] - base_xy[t_move]
            nav_disp = float(np.max(np.linalg.norm(off, axis=1)))
        else:
            nav_disp = 0.0
        return {"t_move": t_move, "t_arrive": t_arrive, "t_placed": t_placed, "nav_disp": nav_disp}

    # ---- main resolve --------------------------------------------------

    def resolve(self, history):
        T = len(history)
        opos = {o: np.array([h[o] for h in history]) for o in OBJ_INFO}
        grip = np.array([h["grip"] for h in history])
        knobs = np.array([h["knobs"] for h in history])
        base_xy = (
            np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        )
        if base_xy is not None and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)

        # order the two objects by which moves first
        first_move = {}
        for o in OBJ_INFO:
            rest = np.median(opos[o][: self.REST_W], axis=0)
            disp = np.linalg.norm(opos[o] - rest, axis=1)
            fm = self._sustained_start(disp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            first_move[o] = fm if fm is not None else T
        order = sorted(OBJ_INFO, key=lambda o: first_move[o])

        carries = []
        cursor = 0
        for o in order:
            c = self._carry(opos[o], grip, base_xy, binst, cursor, T)
            c["obj"] = o
            carries.append(c)
            cursor = c["t_placed"]

        # turn-ons: the knobs whose normalized angle enters the on-range (all start
        # off). Ordered by first-on frame; take the (up to) two turned on.
        norm = np.mod(knobs, 2 * np.pi)
        on = (norm >= self.KNOB_ON) & (norm <= 2 * np.pi - self.KNOB_ON)  # (T, n)
        turn_events = []
        for j, loc in enumerate(self.knob_locs):
            w = np.where(on[:, j])[0]
            if w.size:
                turn_events.append((int(w[0]), loc))
        turn_events.sort()

        # -- assemble --
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        for c in carries:
            o = c["obj"]
            name, source, _ = OBJ_INFO[o]
            lang = self.obj_lang[o]
            emit(c["t_move"], name, STAGE_PICK, f"pick up the {lang} from the {source}")
            if c["t_arrive"] > cursor and c["nav_disp"] >= self.MIN_NAV_DIST:
                emit(c["t_arrive"], SUBTASK_NAV, STAGE_NAVIGATE, f"carry the {lang} to the stove")
            emit(c["t_placed"], name, STAGE_PLACE, f"place the {lang} on a stove burner")

        for t_on, loc in turn_events[:2]:
            loc_txt = loc.replace("_", " ")
            emit(t_on, SUBTASK_TURNON, STAGE_EXECUTE, f"turn on the {loc_txt} burner")
        emit(T, DONE, STAGE_DONE, DONE_LANG)

        self.expected_done_idx = len(segments) - 1

        review = []
        if len(turn_events) != 2:
            review.append(f"expected 2 burners turned on, found {len(turn_events)}")

        debug = {
            "order": order,
            "carries": [(c["obj"], c["t_move"], c["t_arrive"], c["t_placed"], round(c["nav_disp"], 2)) for c in carries],
            "burners": [loc for _, loc in turn_events[:2]],
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

        for nm, label in ((SUBTASK_KETTLE, "kettle"), (SUBTASK_POT, "pot")):
            if count(nm, STAGE_PICK) != 1:
                issues.append(f"expected 1 {label} pick, found {count(nm, STAGE_PICK)}")
            if count(nm, STAGE_PLACE) != 1:
                issues.append(f"expected 1 {label} place, found {count(nm, STAGE_PLACE)}")
        if count(SUBTASK_TURNON) != 2:
            issues.append(f"expected 2 TurnOnStove, found {count(SUBTASK_TURNON)}")
        n_nav = count(SUBTASK_NAV)
        if n_nav not in (0, 1, 2):
            issues.append(f"expected 0-2 NavigateKitchen, found {n_nav}")
        return issues
