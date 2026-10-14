import numpy as np

from robocasa.utils.subtask_labeling.base import (
    BaseSubtaskLabeler,
    STAGE_DONE,
    STAGE_EXECUTE,
    STAGE_PLACE,
    register_labeler,
)
from robocasa.utils.subtask_labeling.subtask_bank import DONE


# Turn the sink water on, hold the sponge under the running water to soak it, then drop it in
# the sink. TurnOnSinkFaucet is a bank atomic; the soak and the drop have no bank names
# (extra_subtasks). Non-mobile. (Doc-canonical count = 2 (soak + drop); the faucet turn-on is
# a distinct robot action here -- water starts off and is turned on -- so it is labeled too.)
SUBTASK_FAUCET = "TurnOnSinkFaucet"
SUBTASK_SOAK = "SoakSponge"
SUBTASK_DROP = "DropSponge"

FAUCET_LANG = "turn on the sink faucet"
SOAK_LANG = "soak the sponge under the running water"
DROP_LANG = "drop the sponge in the sink"
DONE_LANG = "task complete"


@register_labeler
class SoakSpongeLabeler(BaseSubtaskLabeler):
    """Labeler for the SoakSponge composite task.

    Turn on the sink water, hold the sponge under the running water for a few seconds, then
    drop it in the sink. TurnOnSinkFaucet [0, t_on] -> SoakSponge [t_on, soak_end] (grab +
    hold under water) -> DropSponge [soak_end, release].

    t_on = first water-on frame; soak_end = one frame after the sponge leaves the water;
    release = the gripper opening (finger-width / grasp-contact end).
    """

    ENV_NAME = "SoakSponge"

    SPONGE = "obj"
    MOTION_TH = 0.004
    RELEASE_PAD = 8
    REL_FRAC = 0.5
    MIN_DROP = 12   # floor duration for the drop (carved from the soak tail if needed)
    DROP_PAD = 3

    def setup(self, env, ep_meta: dict) -> None:
        super().setup(env, ep_meta)
        self.sponge_bid = env.obj_body_id[self.SPONGE]
        self.sponge_geoms = env.objects[self.SPONGE].contact_geoms
        self.sink = getattr(env, "sink", None)
        self.gripper = env.robots[0].gripper["right"]
        self.fadr = [env.sim.model.get_joint_qpos_addr(j) for j in self.gripper.joints]
        self.extra_subtasks = {SUBTASK_SOAK, SUBTASK_DROP}
        self.min_segment_frames = 12
        self.expected_done_idx = None

    def per_step(self, env, t: int) -> dict:
        try:
            won = 1.0 if self.sink.get_handle_state(env).get("water_on", False) else 0.0
        except Exception:
            won = 0.0
        try:
            uw = 1.0 if self.sink.check_obj_under_water(env, self.SPONGE) else 0.0
        except Exception:
            uw = 0.0
        return {
            "pos": env.sim.data.body_xpos[self.sponge_bid].copy(),
            "water_on": won,
            "uw": uw,
            "fw": float(sum(abs(env.sim.data.qpos[a]) for a in self.fadr)),
            "grasp": int(env._check_grasp(gripper=self.gripper, object_geoms=self.sponge_geoms)),
        }

    def _fw_open(self, fw, fw_open, grasp_t, hi):
        seg = fw[grasp_t:hi]
        if seg.size < 4:
            return None
        held = float(np.min(seg[: min(25, seg.size)]))
        level = held + self.REL_FRAC * (fw_open - held)
        above = seg > level
        for t in range(1, seg.size - 2):
            if above[t] and above[t + 1] and above[t + 2] and not above[t - 1]:
                return grasp_t + t
        return None

    def _release(self, grasp_arr, obj, fw, fw_open, grasp_t, hi):
        spd = np.linalg.norm(np.diff(obj, axis=0), axis=1)
        mv = np.where(spd[grasp_t:hi] > self.MOTION_TH)[0]
        setdown = int(grasp_t + int(mv[-1]) + 2) if mv.size else int(grasp_t + 1)
        fwo = self._fw_open(fw, fw_open, grasp_t, hi)
        gc = np.where(grasp_arr[grasp_t:hi] > 0)[0]
        if gc.size:
            rel = grasp_t + int(gc[-1]) + self.RELEASE_PAD
            if fwo is not None:
                rel = max(rel, fwo)
        elif fwo is not None:
            rel = fwo
        else:
            rel = setdown
        return int(min(max(rel, grasp_t + 1), hi))

    def resolve(self, history):
        T = len(history)
        pos = np.array([h["pos"] for h in history])
        won = np.array([h["water_on"] for h in history])
        uw = np.array([h["uw"] for h in history])
        grasp = np.array([h["grasp"] for h in history])
        fw = np.array([h["fw"] for h in history])
        fw_open = float(fw.max())

        onf = np.where(won > 0.5)[0]
        t_on = int(onf[0]) if onf.size else 0
        uwf = np.where(uw > 0.5)[0]
        has_soak = uwf.size > 0
        soak_end_nat = int(max(int(uwf[-1]) + 1 if has_soak else t_on + 1, t_on + 1))
        release = self._release(grasp, pos, fw, fw_open, max(soak_end_nat, 1), T)

        # drop = release the sponge in the sink. The sponge is often let go right as it leaves
        # the water (release ~ end of soak), so carve the drop out of the tail of the soak so
        # it is always a real segment (with a terminal 'done'), not a collapse.
        sp = np.concatenate([np.linalg.norm(np.diff(pos, axis=0), axis=1), [0.0]])
        mv_tail = np.where(sp[soak_end_nat:T] > self.MOTION_TH)[0]
        motion_end = (soak_end_nat + int(mv_tail[-1]) + 1 + self.DROP_PAD
                      if mv_tail.size else soak_end_nat + self.MIN_DROP)
        drop_end = int(min(max(motion_end, release), T - 1))
        drop_start = int(max(min(soak_end_nat, drop_end - self.MIN_DROP), t_on + 1))

        segments = []
        cursor = 0

        def emit(end, name, stage, lang):
            nonlocal cursor
            end = int(min(max(end, cursor), T))
            segments.append((end, name, stage, lang))
            cursor = end

        review = []
        if onf.size and t_on > 1:
            emit(t_on, SUBTASK_FAUCET, STAGE_EXECUTE, FAUCET_LANG)
        elif not onf.size:
            review.append("sink water never registered on -- check TurnOnSinkFaucet")
        emit(drop_start, SUBTASK_SOAK, STAGE_EXECUTE, SOAK_LANG)
        emit(drop_end, SUBTASK_DROP, STAGE_PLACE, DROP_LANG)
        segments.append((T, DONE, STAGE_DONE, DONE_LANG))
        self.expected_done_idx = len(segments) - 1

        if not has_soak:
            review.append("sponge never registered under running water -- check the soak")
        result = self.build_result(T, segments, debug={
            "t_on": t_on, "drop_start": drop_start, "drop_end": drop_end, "release": release,
            "has_soak": has_soak, "n_segments": len(segments)})
        result.review = review
        return result

    def validate(self, result):
        self.extra_subtasks = {SUBTASK_SOAK, SUBTASK_DROP}
        issues = super().validate(result)
        idx = result.subtask_idx
        if idx.size == 0:
            return issues
        firsts = np.concatenate([[0], np.where(np.diff(idx) != 0)[0] + 1])
        names = set(str(v) for v in result.subtask_name[firsts])
        for req in (SUBTASK_SOAK, SUBTASK_DROP):
            if req not in names:
                issues.append(f"missing {req} subtask")
        return issues
