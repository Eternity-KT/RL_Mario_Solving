"""Boundary cases that are hard to trigger reliably inside the actual game."""

import gymnasium as gym
import numpy as np
import pytest

from src.env.wrappers import ActionRepeatMaxPool, GrayResizeObservation


class ReusedFrameEnv(gym.Env):
    """A tiny simulator whose frames deliberately share a mutable buffer."""

    def __init__(self, *, end_at=None, truncate=False):
        self.observation_space = gym.spaces.Box(0, 255, (4, 6, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.screen = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.end_at = end_at
        self.truncate = truncate
        self.calls = 0
        self.received_reset = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.received_reset = (seed, options)
        self.calls = 0
        self.screen.fill(0)
        return self.screen, {"reset_marker": True}

    def step(self, action):
        self.calls += 1
        # Alternating red and blue catches accidental observation aliasing.
        self.screen[:] = (200, 0, 0) if self.calls % 2 else (0, 0, 150)
        ended = self.calls == self.end_at
        return (
            self.screen, float(self.calls), ended and not self.truncate,
            ended and self.truncate,
            {"reward_components": {"progress": float(self.calls)}},
        )


def test_repeat_copies_frames_sums_rewards_and_pools_last_two():
    base = ReusedFrameEnv()
    env = ActionRepeatMaxPool(base, repeat=4)
    env.reset(seed=17, options={"test": True})
    observation, reward, terminated, truncated, info = env.step(0)
    assert base.calls == 4
    assert base.received_reset == (17, {"test": True})
    np.testing.assert_array_equal(observation[0, 0], [200, 0, 150])
    assert reward == 10.0
    assert not terminated and not truncated
    assert info["frames_this_step"] == 4
    assert info["episode_emulator_frames"] == 4
    assert info["reward_components_sum"] == {"progress": 10.0}
    base.screen.fill(255)
    np.testing.assert_array_equal(observation[0, 0], [200, 0, 150])


@pytest.mark.parametrize("truncate", [False, True])
def test_repeat_stops_at_first_end_and_preserves_flags(truncate):
    base = ReusedFrameEnv(end_at=2, truncate=truncate)
    env = ActionRepeatMaxPool(base, repeat=4)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert base.calls == 2
    assert reward == 3.0
    assert terminated is (not truncate)
    assert truncated is truncate
    assert info["frames_this_step"] == 2
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)


def test_reset_and_short_step_cannot_pool_frames_from_previous_episode():
    base = ReusedFrameEnv()
    env = ActionRepeatMaxPool(base, repeat=4)
    env.reset()
    env.step(0)
    base.end_at = 1
    reset_observation, info = env.reset()
    assert info["episode_emulator_frames"] == 0
    observation, _, terminated, _, info = env.step(0)
    assert terminated
    np.testing.assert_array_equal(observation[0, 0], [200, 0, 0])
    assert not reset_observation.any()
    assert info["episode_emulator_frames"] == 1


def test_repeat_one_does_not_pool_previous_decision():
    env = ActionRepeatMaxPool(ReusedFrameEnv(), repeat=1)
    env.reset()
    env.step(0)
    observation, *_ = env.step(0)
    np.testing.assert_array_equal(observation[0, 0], [0, 0, 150])


def test_grayscale_non_square_size_and_uint8_range():
    env = GrayResizeObservation(ReusedFrameEnv(), screen_size=(3, 5))
    observation, _ = env.reset()
    assert observation.shape == (3, 5)
    assert observation.dtype == np.uint8
    assert observation.flags.c_contiguous
    assert env.observation_space.contains(observation)
    white = np.full((4, 6, 3), 255, dtype=np.uint8)
    assert np.all(env.observation(white) == 255)


@pytest.mark.parametrize("repeat", [0, -1, 1.5, True])
def test_invalid_repeat(repeat):
    with pytest.raises(ValueError):
        ActionRepeatMaxPool(ReusedFrameEnv(), repeat=repeat)
