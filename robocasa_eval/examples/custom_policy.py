"""Starting point for a model adapter; this is not a useful trained policy.

Use factory ``robocasa_eval.examples.custom_policy:create_policy`` only for
interface/integration tests. Replace predict_batch with real model inference,
including the model's normalization and conversion to G1 SONIC actions.
"""

from pathlib import Path

import numpy as np

from robocasa_eval.policy import ACTION_DIMS, BasePolicy, EpisodeContext


class ExamplePolicy(BasePolicy):
    """Illustrate safe history ownership when batches and slots are reused."""

    def __init__(self, *, device, seed, workspace, action_horizon=1):
        if isinstance(action_horizon, bool) or not isinstance(action_horizon, int) or action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer.")
        self.device = device
        self.seed = seed
        self.workspace = Path(workspace).expanduser().resolve()
        self.action_horizon = action_horizon
        self._episodes = {}

    def reset(self, context: EpisodeContext):
        if context.episode_id in self._episodes:
            raise ValueError("Episode is already active.")
        # A model needing temporal history can store a bounded deque here.
        self._episodes[context.episode_id] = {"context": context, "inferences": 0}

    def predict_batch(self, observations, contexts):
        if len(observations) != len(contexts):
            raise ValueError("Observations and contexts must have the same batch length.")
        output = []
        for observation, context in zip(observations, contexts):
            state = self._episodes.get(context.episode_id)
            if state is None or state["context"] != context:
                raise ValueError("Prediction requires the original active episode context.")
            state["inferences"] += 1
            # Replace these placeholder zeros with denormalized model output.
            # A zero SONIC latent is not guaranteed to be a safe or idle action.
            output.append({
                key: np.zeros((self.action_horizon, dim), dtype=np.float32)
                for key, dim in ACTION_DIMS.items()
            })
        return output

    def end_episode(self, context: EpisodeContext):
        state = self._episodes.get(context.episode_id)
        if state is None or state["context"] != context:
            raise ValueError("Ending an unknown or changed episode context.")
        del self._episodes[context.episode_id]

    def close(self):
        self._episodes.clear()


def create_policy(*, device, seed, workspace, action_horizon=1):
    return ExamplePolicy(
        device=device, seed=seed, workspace=workspace, action_horizon=action_horizon
    )
