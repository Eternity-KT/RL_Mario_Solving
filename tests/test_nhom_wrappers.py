"""Bài kiểm tra BỔ SUNG của nhóm cho wrappers (không thuộc bộ công khai của đề; Người 4 quản lý).

Bịt các chỗ mà tests/test_wrappers.py không bắt được (đã thử cài sai mà test công khai vẫn PASS):
  - FrameSkip max-pool trên MỌI khung thay vì chỉ HAI khung cuối;
  - FrameSkip không dừng khi lượt bị cắt (truncated), không trả phần thưởng float32;
  - GrayResize khai báo observation_space sai kiểu / sai cận;
  - ScalarizeReward không đọc số mục tiêu từ reward_space, làm mất các khóa info khác.
"""
import gymnasium as gym
import numpy as np
import pytest

from mario_ppo.wrappers import FrameSkip, GrayResize, ScalarizeReward


class _Env(gym.Env):
    """Môi trường giả: khung thứ k (k = 0, 1, 2, ...) chỉ có điểm ảnh (k, k) sáng bằng 10 * k;
    phần thưởng véc-tơ float64 bằng k * [1..n_obj]."""

    observation_space = gym.spaces.Box(0, 255, (240, 256, 3), np.uint8)
    action_space = gym.spaces.Discrete(7)

    def __init__(self, end_at=None, trunc_at=None, n_obj=5, with_reward_space=False):
        self.end_at, self.trunc_at, self.n_obj, self.k = end_at, trunc_at, n_obj, 0
        if with_reward_space:
            self.reward_space = gym.spaces.Box(-np.inf, np.inf, (n_obj,), np.float32)

    def _frame(self):
        f = np.zeros((240, 256, 3), np.uint8)
        f[self.k, self.k] = 10 * self.k
        return f

    def reset(self, seed=None, options=None):
        self.k = 0
        return self._frame(), {}

    def step(self, action):
        self.k += 1
        reward = self.k * np.arange(1, self.n_obj + 1, dtype=np.float64)
        terminated = self.end_at is not None and self.k >= self.end_at
        truncated = self.trunc_at is not None and self.k >= self.trunc_at
        return self._frame(), reward, terminated, truncated, {"x_pos": self.k}


def test_frameskip_max_pools_only_the_last_two_frames():
    env = FrameSkip(_Env(), skip=4)
    env.reset()
    obs, *_ = env.step(0)
    assert obs[3, 3, 0] == 30 and obs[4, 4, 0] == 40  # khung 3 và 4 (hai khung cuối)
    assert obs[1, 1, 0] == 0 and obs[2, 2, 0] == 0  # khung 1 và 2 không được góp vào


def test_frameskip_single_frame_is_returned_unchanged():
    env = FrameSkip(_Env(end_at=1), skip=4)
    env.reset()
    obs, _, terminated, _, _ = env.step(0)
    assert terminated and env.unwrapped.k == 1
    expected = np.zeros((240, 256, 3), np.uint8)
    expected[1, 1] = 10
    assert np.array_equal(obs, expected)


def test_frameskip_stops_when_truncated():
    env = FrameSkip(_Env(trunc_at=2), skip=4)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    assert env.unwrapped.k == 2 and info["x_pos"] == 2
    assert np.allclose(reward, (1 + 2) * np.arange(1, 6))


def test_frameskip_reward_is_float32_vector():
    env = FrameSkip(_Env(), skip=4)
    env.reset()
    _, reward, *_ = env.step(0)
    assert isinstance(reward, np.ndarray) and reward.dtype == np.float32 and reward.shape == (5,)


@pytest.mark.parametrize("size", [84, 64])
def test_grayresize_declares_uint8_box_matching_observations(size):
    env = GrayResize(_Env(), size=size)
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (size, size) and space.dtype == np.uint8
    assert np.all(space.low == 0) and np.all(space.high == 255)
    obs, _ = env.reset()
    assert space.contains(obs)
    obs, *_ = env.step(0)
    assert space.contains(obs)


def test_scalarize_reward_reads_objective_count_from_reward_space():
    env = ScalarizeReward(_Env(n_obj=3, with_reward_space=True), [1.0, 2.0, 3.0])
    env.reset()
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(1 * 1 + 2 * 2 + 3 * 3)
    with pytest.raises(ValueError):
        ScalarizeReward(_Env(n_obj=3, with_reward_space=True), [1.0] * 5)


def test_scalarize_reward_keeps_other_info_keys():
    env = ScalarizeReward(_Env(), [1.0] * 5)
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["x_pos"] == 1  # train.py / evaluate.py đọc x_pos từ info
    assert np.shape(info["vector_reward"]) == (5,)
