from types import SimpleNamespace

from robocasa.scripts import collect_sonic_demos as collector


def test_reset_with_retry_clears_episode_metadata_before_reset(monkeypatch):
    events = []
    observation = {"state": "reset"}

    class FakeEnvironment:
        def reset(self):
            events.append("reset")
            return observation

    class FakeBase:
        def unset_ep_meta(self):
            events.append("unset_ep_meta")

    monkeypatch.setattr(
        collector,
        "_apply_runtime",
        lambda base, args: events.append("apply_runtime"),
    )

    result = collector.reset_with_retry(
        FakeEnvironment(),
        FakeBase(),
        SimpleNamespace(),
    )

    assert result is observation
    assert events == ["unset_ep_meta", "reset", "apply_runtime"]


def test_reset_with_retry_clears_episode_metadata_before_every_attempt(monkeypatch):
    events = []
    observation = {"state": "reset"}
    reset_attempts = 0

    class FakeEnvironment:
        def reset(self):
            nonlocal reset_attempts
            reset_attempts += 1
            events.append("reset")
            if reset_attempts == 1:
                raise collector.mujoco.FatalError("rank-deficient Hessian")
            return observation

    class FakeBase:
        def unset_ep_meta(self):
            events.append("unset_ep_meta")

    monkeypatch.setattr(
        collector,
        "_apply_runtime",
        lambda base, args: events.append("apply_runtime"),
    )

    result = collector.reset_with_retry(
        FakeEnvironment(),
        FakeBase(),
        SimpleNamespace(),
        tries=2,
    )

    assert result is observation
    assert events == [
        "unset_ep_meta",
        "reset",
        "unset_ep_meta",
        "reset",
        "apply_runtime",
    ]
