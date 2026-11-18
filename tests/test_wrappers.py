"""Kiểm tra các wrapper tiền xử lý và cắt lượt khi Mario bị kẹt."""
import gymnasium as gym
import numpy as np
import pytest

from mario_ppo.wrappers import FrameSkip, GrayResize, NoProgressTruncation, ScalarizeReward


class _FakeMarioEnv(gym.Env):
    """Môi trường giả: khung hình thứ k có mọi điểm ảnh bằng k, phần thưởng véc-tơ 5 chiều bằng k * [1..5]."""

    observation_space = gym.spaces.Box(0, 255, (240, 256, 3), np.uint8)
    action_space = gym.spaces.Discrete(7)

    def __init__(self, end_at=None, pattern=None):
        self.end_at, self.pattern, self.k = end_at, pattern, 0
        self.actions = []

    def _frame(self):
        if self.pattern is not None:
            return self.pattern[self.k % len(self.pattern)].copy()
        return np.full((240, 256, 3), self.k, np.uint8)

    def reset(self, seed=None, options=None):
        self.k, self.actions = 0, []
        return self._frame(), {}

    def step(self, action):
        self.k += 1
        self.actions.append(int(action))
        reward = self.k * np.arange(1, 6, dtype=np.float32)
        terminated = self.end_at is not None and self.k >= self.end_at
        return self._frame(), reward, terminated, False, {"x_pos": self.k}


def test_frameskip_repeats_action_and_sums_vector_reward():
    env = FrameSkip(_FakeMarioEnv(), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(3)
    assert env.unwrapped.actions == [3, 3, 3, 3]
    assert np.allclose(reward, (1 + 2 + 3 + 4) * np.arange(1, 6))
    assert not terminated and not truncated
    assert info["x_pos"] == 4


def test_frameskip_max_pools_last_two_frames():
    a = np.zeros((240, 256, 3), np.uint8)
    b = np.zeros((240, 256, 3), np.uint8)
    a[0, 0] = 200  # chỉ có ở khung chẵn
    b[1, 1] = 100  # chỉ có ở khung lẻ
    env = FrameSkip(_FakeMarioEnv(pattern=[a, b]), skip=4)
    env.reset()
    obs, *_ = env.step(0)
    assert obs.shape == (240, 256, 3) and obs.dtype == np.uint8
    assert obs[0, 0, 0] == 200 and obs[1, 1, 0] == 100


def test_frameskip_stops_at_episode_end():
    env = FrameSkip(_FakeMarioEnv(end_at=2), skip=4)
    env.reset()
    _, reward, terminated, _, _ = env.step(1)
    assert terminated
    assert env.unwrapped.actions == [1, 1]
    assert np.allclose(reward, 3 * np.arange(1, 6))


def test_frameskip_pools_reused_frames_until_truncation():
    class _ReusedFrame(_FakeMarioEnv):
        frame = np.zeros((240, 256, 3), np.uint8)

        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            self.frame[:] = obs
            return self.frame, reward.astype(np.float64), False, self.k == 3, info

    frames = np.zeros((4, 240, 256, 3), np.uint8)
    frames[1, 0, 0] = 250  # Khung thứ nhất không được lấy max.
    frames[2, 0, 1] = 100
    frames[3, 0, 2] = 200
    env = FrameSkip(_ReusedFrame(pattern=frames), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)
    assert not terminated and truncated and info["x_pos"] == 3
    assert env.unwrapped.actions == [1, 1, 1]
    assert obs[0, :3, 0].tolist() == [0, 100, 200]
    assert reward.dtype == np.float32
    assert np.allclose(reward, 6 * np.arange(1, 6))


def test_grayresize_shape_dtype_and_brightness():
    env = GrayResize(FrameSkip(_FakeMarioEnv(), skip=1), size=84)
    assert env.observation_space.shape == (84, 84)
    obs, _ = env.reset()
    assert obs.shape == (84, 84) and obs.dtype == np.uint8
    for _ in range(50):
        obs, *_ = env.step(0)
    # ảnh xám đều 50 -> độ sáng xấp xỉ 50
    assert abs(float(obs.mean()) - 50.0) <= 1.0


def test_grayresize_uses_luminance_not_channel_mean():
    class _Red(_FakeMarioEnv):
        def _frame(self):
            f = np.zeros((240, 256, 3), np.uint8)
            f[..., 0] = 255  # đỏ thuần
            return f
    obs, _ = GrayResize(_Red(), size=84).reset()
    # Y = 0,299 R + 0,587 G + 0,114 B -> khoảng 76
    assert 70 <= float(obs.mean()) <= 82


def test_scalarize_reward_dot_product_and_info():
    w = [1.0, 0.5, 2.0, 0.0, -1.0]
    env = ScalarizeReward(FrameSkip(_FakeMarioEnv(), skip=1), w)
    env.reset()
    _, reward, _, _, info = env.step(0)
    vec = np.arange(1, 6, dtype=np.float32)
    assert isinstance(reward, float)
    assert reward == pytest.approx(float(np.dot(w, vec)))
    assert np.allclose(info["vector_reward"], vec)


def test_scalarize_reward_rejects_wrong_weight_count():
    with pytest.raises(ValueError):
        ScalarizeReward(_FakeMarioEnv(), [1.0, 2.0])


class _ProgressEnv(_FakeMarioEnv):
    def __init__(self, positions, end_at=None):
        super().__init__(end_at=end_at)
        self.positions = positions

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return obs, dict(info, x_pos=self.positions[0])

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, dict(info, x_pos=self.positions[self.k])


def test_no_progress_truncates_at_threshold_and_resets():
    env = NoProgressTruncation(_ProgressEnv([50, 52, 53, 51, 55]), max_steps=2)
    env.reset(seed=7)
    for _ in range(3):
        assert env.step(0)[2:4] == (False, False)
    assert env.step(0)[2:4] == (False, True)
    env.reset(seed=7)
    assert env.step(0)[2:4] == (False, False)


def test_no_progress_starts_counting_after_first_known_position():
    env = NoProgressTruncation(_FakeMarioEnv(), max_steps=1)
    env.reset()
    assert env.step(0)[2:4] == (False, False)
    assert env.step(0)[2:4] == (False, True)


def test_no_progress_preserves_native_termination_at_threshold():
    env = NoProgressTruncation(_ProgressEnv([50, 50, 50], end_at=2), max_steps=2)
    env.reset()
    assert env.step(0)[2:4] == (False, False)
    assert env.step(0)[2:4] == (True, False)
