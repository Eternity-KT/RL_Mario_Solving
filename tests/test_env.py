"""Integration checks use the real emulator, without training an agent."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.env import make_mario_env


@pytest.mark.integration
@pytest.mark.parametrize("action_type, action_count", [
    ("SIMPLE_MOVEMENT", 7), ("RIGHT_ONLY", 5), ("RIGHT_JUMP", 2),
])
def test_mario_observations_and_action_spaces(action_type, action_count):
    with make_mario_env(action_type=action_type) as env:
        observation, info = env.reset(seed=42)
        assert observation.shape == (4, 84, 84)
        assert observation.dtype == np.uint8
        assert env.observation_space.contains(observation)
        assert env.action_space.n == action_count
        assert "x_pos" in info and "flag_get" in info
        # Reset padding repeats the initial observation rather than old frames.
        assert np.all(observation == observation[0])
        snapshot = observation.copy()
        next_observation, reward, terminated, truncated, info = env.step(1)
        np.testing.assert_array_equal(observation, snapshot)
        np.testing.assert_array_equal(next_observation[:-1], snapshot[1:])
        assert env.observation_space.contains(next_observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert 1 <= info["frames_this_step"] <= 4


@pytest.mark.integration
def test_time_limit_counts_agent_decisions_and_reset_discards_history():
    with make_mario_env(max_episode_steps=2) as env:
        initial, _ = env.reset(seed=42)
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated and not truncated
        _, _, terminated, truncated, info = env.step(0)
        assert not terminated and truncated
        assert info["episode_emulator_frames"] == 8
        assert info["episode"]["l"] == 2
        reset_observation, info = env.reset(seed=42)
        np.testing.assert_array_equal(reset_observation, initial)
        assert info["episode_emulator_frames"] == 0


@pytest.mark.integration
def test_native_render_is_rgb_and_network_observation_is_grayscale():
    with make_mario_env(render_mode="rgb_array", screen_size=(40, 60), frame_stack=2) as env:
        observation, _ = env.reset(seed=3)
        frame = env.render()
        assert observation.shape == (2, 40, 60)
        assert frame.shape == (240, 256, 3)
        assert frame.dtype == np.uint8


@pytest.mark.integration
def test_seeded_action_trace_is_reproducible():
    def trace(env):
        env.reset(seed=31)
        env.action_space.seed(81)
        result = []
        for _ in range(30):
            observation, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            result.append((observation.copy(), reward, terminated, truncated, info["x_pos"]))
            if terminated or truncated:
                env.reset()
        return result

    with make_mario_env() as env:
        first, second = trace(env), trace(env)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]


@pytest.mark.integration
def test_reward_hook_runs_once_per_decision_and_sees_external_truncation():
    calls = []

    class RewardProbe(gym.Wrapper):
        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            calls.append((reward, truncated, info["frames_this_step"]))
            return observation, reward + 7.0, terminated, truncated, info

    with make_mario_env(max_episode_steps=1, reward_wrapper=RewardProbe) as env:
        env.reset()
        _, reward, _, truncated, info = env.step(0)
        assert calls == [(info["default_reward"], True, 4)]
        assert truncated
        assert reward == info["default_reward"] + 7.0
        assert info["episode"]["r"] == reward


@pytest.mark.integration
def test_gymnasium_contract():
    with make_mario_env() as env:
        # Native JoypadSpace does not record constructor arguments, so the
        # checker's env.spec.make() close/recreation probe cannot rebuild it.
        # Reset/step/seed/space checks still run against the full pipeline.
        check_env(env, skip_render_check=True, skip_close_check=True)


@pytest.mark.parametrize("kwargs", [
    {"world": 0}, {"stage": 5}, {"version": 4},
    {"action_type": "SIMPLE"}, {"frame_skip": 0},
    {"frame_stack": True}, {"screen_size": (0, 84)},
    {"max_episode_steps": 0}, {"render_mode": "bad"},
])
def test_invalid_factory_configuration(kwargs):
    with pytest.raises(ValueError):
        make_mario_env(**kwargs)
