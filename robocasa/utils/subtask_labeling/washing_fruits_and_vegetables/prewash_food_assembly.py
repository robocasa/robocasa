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


# Put the produce from the cabinet into the bowl, carry the bowl into the sink, then turn on
# the sink faucet. food -> bowl is PickPlaceCabinetToBowl (ext); bowl -> sink is
# PickPlaceCounterToSink (bank); TurnOnSinkFaucet is bank; NavigateKitchen is bank. Mobile.
SUBTASK_NAV = "NavigateKitchen"
SUBTASK_FAUCET = "TurnOnSinkFaucet"
FAUCET_LANG = "turn on the sink faucet"
DONE_LANG = "task complete"

# obj -> (bank name, source phrase, carry-nav dest, place dest)
DEST = {
    "food": ("PickPlaceCabinetToBowl", "the cabinet", "the bowl", "in the bowl"),
    "bowl": ("PickPlaceCounterToSink", "the counter", "the sink", "in the sink"),
}
ORDER = ("food", "bowl")
FOOD_BANK = "PickPlaceCabinetToBowl"
BOWL_BANK = "PickPlaceCounterToSink"


@register_labeler
class PrewashFoodAssemblyLabeler(BaseSubtaskLabeler):
    """Labeler for the PrewashFoodAssembly composite task.

    Task (mobile): pick the produce from the cabinet and put it in the bowl; pick the bowl and
    carry it to the sink; turn on the sink faucet. Structure (doc count = 5 + the carry nav,
    broken out per the user's convention):
        pick up the {food} from the cabinet  (PickPlaceCabinetToBowl, pick)
        place the {food} in the bowl         (PickPlaceCabinetToBowl, place)   -- no nav
        pick up the bowl from the counter    (PickPlaceCounterToSink, pick)
        carry the bowl to the sink           (NavigateKitchen)                 -- only the bowl navs
        place the bowl in the sink           (PickPlaceCounterToSink, place)
        turn on the sink faucet              (TurnOnSinkFaucet)

    Everything is anchored on the BOWL's motion, because after the food is dropped into the bowl
    the gripper immediately goes to the bowl (same spot) so the food's gripper-release is not a
    reliable "food placed" signal (it runs late and swallows the whole bowl carry -- that was the
    old "carry the {food} to the bowl" mislabel). Instead: the bowl's first lift (t_bmove) and
    its release in the sink (t_bplace) are the anchors; the food's place ends when the FOOD comes
    to rest in the bowl, just before the bowl is lifted. The single kitchen-nav belongs to the
    bowl leg (cabinet->bowl is a local pick-place and never navs).
    """

    ENV_NAME = "PrewashFoodAssembly"

    MOVE_TOL = 0.10
    MOVE_SUSTAIN = 5
    HOLD_W = 40
    RELEASE_MARGIN = 0.08
    RELEASE_SUSTAIN = 5
    MIN_CARRY = 15
    REST_W = 15
    BASE_MOVE_TH = 0.008
    MIN_NAV_DIST = 0.30
    MIN_PICK_GAP = 12
    SETTLE_TH = 0.0015
    FOOD_RELEASE_MARGIN = 0.05   # gripper separation from the food that marks "placed in bowl"

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.names = [o for o in ORDER if o in env.obj_body_id]
        self.obj_body_ids = {o: env.obj_body_id[o] for o in self.names}
        self.eef_site_id = env.robots[0].eef_site_id["right"]
        self.sink = getattr(env, "sink", None)
        self.base = {}
        for o in self.names:
            try:
                self.base[o] = env.get_obj_lang(o)
            except Exception:
                self.base[o] = o
        self.base_qpos_adrs = []
        for nm in ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"):
            try:
                self.base_qpos_adrs.append(env.sim.model.jnt_qposadr[env.sim.model.joint_name2id(nm)])
            except Exception:
                pass
        self.extra_subtasks = {"PickPlaceCabinetToBowl"}
        self.expected_done_idx = None
        self.min_segment_frames = 10

    def per_step(self, env, t: int) -> dict:
        try:
            won = 1.0 if self.sink.get_handle_state(env).get("water_on", False) else 0.0
        except Exception:
            won = 0.0
        d = {
            "grip": env.sim.data.site_xpos[self.eef_site_id].copy(),
            "base_xy": np.array([float(env.sim.data.qpos[a]) for a in self.base_qpos_adrs]),
            "water_on": won,
        }
        for o in self.names:
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
        opos = {o: np.array([h[o] for h in history]) for o in self.names}
        grip = np.array([h["grip"] for h in history])
        won = np.array([h["water_on"] for h in history])
        base_xy = np.array([h["base_xy"] for h in history]) if self.base_qpos_adrs else None
        has_base = base_xy is not None and base_xy.ndim == 2 and base_xy.shape[1] > 0
        if has_base and T >= 2:
            binst = np.linalg.norm(np.diff(base_xy, axis=0), axis=1)
            binst = np.concatenate([binst, binst[-1:]])
        else:
            binst = np.zeros(T)
        gd = {o: np.linalg.norm(grip - opos[o], axis=1) for o in self.names}

        def base_disp(a, b):
            return (float(np.max(np.linalg.norm(base_xy[a: b] - base_xy[a], axis=1)))
                    if has_base and b > a else 0.0)

        def first_sustained_move(p):
            rest = np.median(p[: self.REST_W], axis=0)
            dsp = np.linalg.norm(p - rest, axis=1)
            t = self._sustained_start(dsp >= self.MOVE_TOL, 0, self.MOVE_SUSTAIN)
            if t is None:
                mv = np.where(dsp >= self.MOVE_TOL)[0]
                t = int(mv[0]) if mv.size else T - 1
            return int(t)

        review = []

        # --- bowl anchors: first lift and release in the sink ---
        t_bmove = first_sustained_move(opos["bowl"])
        hold_b = float(np.median(gd["bowl"][t_bmove: min(t_bmove + self.HOLD_W, T)]))
        t_bplace = self._sustained_start(gd["bowl"] > hold_b + self.RELEASE_MARGIN,
                                         t_bmove + self.MIN_CARRY, self.RELEASE_SUSTAIN)
        if t_bplace is None:
            t_bplace = T - 1
        if t_bmove >= T - 5:
            review.append("bowl never registered a lift -- verify the carry to the sink")

        # --- food: pick, then placed into the bowl. Its place ends when the gripper RELEASES the
        # food into the bowl -- detected as the gripper separating from the food (gd rises off its
        # grasp baseline). The search is bounded to [lo, hi] = before the bowl is picked up, so it
        # can only catch the food's release into the bowl, never the bowl's release at the sink
        # (the old "last food motion" rule chased post-placement settling jiggle and ran late,
        # swallowing most of the bowl pick -- that was the mislabel). Falls back to the food's
        # first sustained rest, then to the window cap, if the gripper never clearly separates. ---
        t_fmove = first_sustained_move(opos["food"])
        fspeed = np.concatenate([[0.0], np.linalg.norm(np.diff(opos["food"], axis=0), axis=1)])
        lo = min(t_fmove + self.MIN_CARRY, max(t_bmove - 1, t_fmove + 1))
        hi = max(t_bmove - self.MIN_PICK_GAP, lo + 1)
        hold_f = float(np.median(gd["food"][t_fmove: min(t_fmove + self.HOLD_W, T)]))
        t_fplace = self._sustained_start(gd["food"] > hold_f + self.FOOD_RELEASE_MARGIN,
                                         lo, self.RELEASE_SUSTAIN)
        if t_fplace is None or t_fplace > hi:
            t_rest = self._sustained_start(fspeed < self.SETTLE_TH, lo, self.MOVE_SUSTAIN)
            t_fplace = t_rest if (t_rest is not None and t_rest <= hi) else hi
        t_fplace = int(min(max(t_fplace, lo), hi))

        w = self.base["food"]
        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        # food: cabinet -> bowl (local pick-place, never a nav)
        emit(t_fmove, FOOD_BANK, STAGE_PICK, f"pick up the {w} from the cabinet")
        emit(t_fplace, FOOD_BANK, STAGE_PLACE, f"place the {w} in the bowl")
        # bowl: counter -> sink, with the single kitchen-nav
        emit(t_bmove, BOWL_BANK, STAGE_PICK, "pick up the bowl from the counter")
        moving = np.where(binst[t_bmove: t_bplace] > self.BASE_MOVE_TH)[0]
        t_barrive = min(t_bmove + int(moving[-1]) + 1, t_bplace) if moving.size else t_bmove
        t_barrive = min(t_barrive, max(t_bplace - self.MIN_PICK_GAP, cursor))
        if t_barrive > cursor and base_disp(t_bmove, t_barrive) >= self.MIN_NAV_DIST:
            emit(t_barrive, SUBTASK_NAV, STAGE_NAVIGATE, "carry the bowl to the sink")
        emit(t_bplace, BOWL_BANK, STAGE_PLACE, "place the bowl in the sink")

        # TurnOnSinkFaucet: turned on and left running to the end (reach + turn-on + running).
        emit(T - 1, SUBTASK_FAUCET, STAGE_EXECUTE, FAUCET_LANG)
        onf = np.where(won > 0.5)[0]
        if onf.size == 0:
            review.append("sink water never registered on -- verify TurnOnSinkFaucet")

        emit(T, DONE, STAGE_DONE, DONE_LANG)
        self.expected_done_idx = len(segments) - 1

        result = self.build_result(T, segments, debug={
            "t_fmove": t_fmove, "t_fplace": t_fplace, "t_bmove": t_bmove,
            "t_bplace": t_bplace, "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {"PickPlaceCabinetToBowl"}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = result.subtask_name[firsts]
        stages = result.subtask_stage[firsts]
        if int(np.sum(stages == STAGE_PICK)) != 2:
            issues.append("expected 2 picks")
        if int(np.sum(stages == STAGE_PLACE)) != 2:
            issues.append("expected 2 places")
        if int(np.sum(names == SUBTASK_FAUCET)) != 1:
            issues.append("expected 1 TurnOnSinkFaucet")
        return issues
