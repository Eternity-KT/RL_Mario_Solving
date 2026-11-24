"""Wrapper Gymnasium cho mo-supermario-v0.

Phần CÓ SẴN (không cần sửa): chuyển hành động, sửa lỗi của môi trường gốc, xuất phát ngẫu nhiên, ghi khung hình.
Phần tiền xử lý: FrameSkip, GrayResize, ScalarizeReward. Có thể thêm wrapper riêng của nhóm ở cuối tệp
(ví dụ cắt lượt khi Mario đứng yên, thưởng khi chạm cờ, ...) và bật chúng trong env.py.
"""
import cv2
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

# Các tập hành động có thể chọn qua cấu hình env.actions. Hành động 0 luôn là NOOP.
ACTION_SETS = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "simple_down": SIMPLE_MOVEMENT + [["down"]],  # thêm DOWN để chui ống
    "complex": COMPLEX_MOVEMENT,
}

_BUTTON_BITS = {"right": 0x80, "left": 0x40, "down": 0x20, "up": 0x10,
                "start": 0x08, "select": 0x04, "B": 0x02, "A": 0x01, "NOOP": 0x00}


# ---------------------------------------------------------------------------------------------------------
# CÓ SẴN
# ---------------------------------------------------------------------------------------------------------

class ButtonActions(gym.ActionWrapper):
    """Discrete(256) (mặt nạ bit của tay cầm NES) -> Discrete(len(movements))."""

    def __init__(self, env, movements=SIMPLE_MOVEMENT):
        super().__init__(env)
        self._masks = [sum(_BUTTON_BITS[b] for b in combo) for combo in movements]
        self.action_space = gym.spaces.Discrete(len(self._masks))

    def action(self, action):
        return self._masks[int(action)]


class DeathPenaltyFix(gym.Wrapper):
    """Môi trường gốc kết thúc lượt ngay khi Mario bắt đầu chết, trước khi số mạng giảm, nên thành phần
    phần thưởng 'death' luôn bằng 0. Wrapper này trả lại phần phạt -25 (đã co giãn như các thành phần khác).
    Bọc trực tiếp môi trường gốc (trước FrameSkip)."""

    DEATH_INDEX = 2  # thứ tự mục tiêu: [x_pos, time, death, coin, enemy]

    def __init__(self, env):
        super().__init__(env)
        self.penalty = -25.0 * env.unwrapped.reward_space.shape[0] / 150.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and not info.get("flag_get", False):
            reward = np.array(reward, dtype=np.float32, copy=True)
            if reward[self.DEATH_INDEX] == 0.0:
                reward[self.DEATH_INDEX] = self.penalty
        return obs, reward, terminated, truncated, info


class XPosWrapFix(gym.Wrapper):
    """Tọa độ x 16 bit có thể tràn (lâu đài mê cung đưa Mario về bên trái x = 0), khiến môi trường trả một
    phần thưởng x khổng lồ trong một khung. Bỏ các bước nhảy x không thể có. Bọc môi trường gốc."""

    MAX_DX = 16

    def __init__(self, env):
        super().__init__(env)
        self.scale = env.unwrapped.reward_space.shape[0] / 150.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if abs(reward[0]) > self.MAX_DX * self.scale:
            reward = np.array(reward, dtype=np.float32, copy=True)
            reward[0] = 0.0
        x = info.get("x_pos")
        if x is not None and x >= 0x8000:
            info = dict(info, x_pos=int(x) - 0x10000)
        return obs, reward, terminated, truncated, info


class RandomNoopStart(gym.Wrapper):
    """Mỗi lượt bắt đầu sau một số ngẫu nhiên 0..max_noops bước NOOP, để chính sách không học thuộc một
    nhịp thời gian duy nhất. Script đánh giá dùng cùng ý tưởng với các số NOOP cố định theo seed."""

    NOOP = 0

    def __init__(self, env, max_noops):
        super().__init__(env)
        self.max_noops = max_noops

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(int(self.np_random.integers(0, self.max_noops + 1))):
            obs, _, terminated, truncated, info = self.env.step(self.NOOP)
            if terminated or truncated:
                return self.env.reset()
        return obs, info


class RecordFrames(gym.Wrapper):
    """Giữ mọi khung hình RGB của lượt hiện tại trong self.frames (để ghi video)."""

    def __init__(self, env):
        super().__init__(env)
        self.frames = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs.copy()]
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self.frames.append(out[0].copy())
        return out


# ---------------------------------------------------------------------------------------------------------
# Tiền xử lý
# ---------------------------------------------------------------------------------------------------------

class FrameSkip(gym.Wrapper):
    """Cho một quyết định của tác tử kéo dài qua nhiều khung hình NES.

    Dùng `FrameSkip(env, skip=4)` sau wrapper chuyển hành động. Gọi `reset()`
    để bắt đầu lượt, rồi `step(action)` cho mỗi quyết định của tác tử.
    """

    def __init__(self, env, skip=4):
        """Nhận môi trường `env` trả ảnh RGB và phần thưởng véc-tơ.

        `skip` là số khung tối đa cho mỗi hành động, mặc định là 4. Hàm lưu
        `skip` và trả về `None`, hoặc báo `ValueError` nếu `skip` không hợp lệ.
        """
        super().__init__(env)
        if not isinstance(skip, (int, np.integer)) or skip < 1:
            raise ValueError("skip must be a positive integer")
        self.skip = skip

    def step(self, action):
        """Nhận `action` hợp lệ trong `action_space` và lặp tối đa `skip` lần.

        Trả `(obs, total_reward, terminated, truncated, info)`. `obs` là ảnh
        lấy max theo từng pixel của hai khung cuối để giảm nhấp nháy sprite.
        Nếu chỉ chạy một khung, dùng ảnh của khung đó. `total_reward` là tổng
        véc-tơ `float32`. Hai cờ và `info` lấy từ khung cuối. Dừng lặp ngay
        khi một trong hai cờ bật.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        total_reward = np.array(reward, dtype=np.float32, copy=True)
        previous_obs = None
        for _ in range(1, self.skip):
            if terminated or truncated:
                break
            # Trình giả lập có thể ghi đè bộ đệm ảnh ở lần step tiếp theo.
            previous_obs = obs.copy()
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += np.asarray(reward, dtype=np.float32)
        if previous_obs is not None:
            obs = np.maximum(previous_obs, obs)
        return obs, total_reward, terminated, truncated, info


class GrayResize(gym.ObservationWrapper):
    """Giảm kích thước ảnh đầu vào của mạng bằng ảnh xám có kích thước cố định.

    Dùng `GrayResize(env, size=84)` trước khi xếp chồng khung hình. Gymnasium
    gọi `observation()` tự động khi `reset()` và `step()`.
    """

    def __init__(self, env, size=84):
        """Nhận môi trường `env` trả ảnh RGB và cạnh `size` tính bằng pixel.

        Khai báo `observation_space` là `Box` có shape `(size, size)`, kiểu
        `uint8`, giá trị từ 0 đến 255. Hàm khởi tạo trả về `None`.
        """
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(0, 255, (size, size), dtype=np.uint8)

    def observation(self, obs):
        """Nhận ảnh RGB `obs` kiểu `uint8`, shape `(H, W, 3)`.

        Trả ảnh xám `uint8` có shape `(size, size)`. OpenCV tính độ sáng từ
        ba kênh bằng `COLOR_RGB2GRAY` rồi thu nhỏ ảnh bằng `INTER_AREA`.
        """
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ScalarizeReward(gym.Wrapper):
    """Gộp các mục tiêu của Mario thành một phần thưởng vô hướng cho PPO.

    Dùng `ScalarizeReward(env, weights=[1.0, 1.0, 2.0, 0.05, 0.05])` sau
    `FrameSkip` và các wrapper huấn luyện. Trọng số lần lượt ứng với tiến
    ngang, thời gian, chết, nhặt xu và hạ địch.
    """

    def __init__(self, env, weights):
        """Nhận môi trường `env` và véc-tơ một chiều `weights` có một trọng số cho mỗi mục tiêu.

        Lấy số mục tiêu từ `env.unwrapped.reward_space.shape[0]` hoặc dùng 5
        nếu thiếu `reward_space`. Hàm lưu trọng số dưới dạng `float32` và trả
        về `None`, hoặc báo `ValueError` nếu kích thước véc-tơ không hợp lệ.
        """
        super().__init__(env)
        reward_space = getattr(env.unwrapped, "reward_space", None)
        num_objectives = reward_space.shape[0] if reward_space is not None else 5
        self.weights = np.array(weights, dtype=np.float32)
        if self.weights.shape != (num_objectives,):
            raise ValueError(f"weights must be a vector of length {num_objectives}")

    def step(self, action):
        """Nhận `action` hợp lệ và chuyển véc-tơ phần thưởng thành tích vô hướng với `weights`.

        Trả `(obs, reward, terminated, truncated, info)`. `reward` là `float`
        của Python. `info["vector_reward"]` chứa bản sao véc-tơ `float32`
        để có thể xem đóng góp của từng mục tiêu.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        vector_reward = np.array(reward, dtype=np.float32, copy=True)
        scalar_reward = float(np.dot(self.weights, vector_reward))
        return obs, scalar_reward, terminated, truncated, dict(info, vector_reward=vector_reward)


# ---------------------------------------------------------------------------------------------------------
# Wrapper bổ sung của nhóm (tùy chọn) - thêm ở đây
# ---------------------------------------------------------------------------------------------------------

class NoProgressTruncation(gym.Wrapper):
    """Cắt lượt khi Mario bị kẹt và không tiến sang phải đủ lâu.

    Dùng `make_env("1-1", max_no_progress_steps=200)` để bật khi huấn luyện,
    hoặc đặt `max_no_progress_steps: 200` trong mục `env` của cấu hình YAML.
    """

    def __init__(self, env, max_steps=200):
        """Nhận môi trường `env` cung cấp `info["x_pos"]` khi chơi.

        `max_steps` là ngưỡng số bước tác tử liên tiếp thiếu tiến triển. Hàm
        khởi tạo bộ đếm và trả về `None`, hoặc báo `ValueError` nếu ngưỡng
        không phải số nguyên dương.
        """
        super().__init__(env)
        if not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.max_steps = max_steps
        self.steps_without_progress = 0
        self.max_x = None

    def reset(self, **kwargs):
        """Nhận tham số reset của môi trường, ví dụ `seed` hoặc `options`.

        Trả `(obs, info)` đầu lượt. Xóa bộ đếm và lấy mốc `x_pos` từ `info`
        nếu có. Nếu thiếu, dùng vị trí đầu tiên quan sát được khi chơi.
        """
        obs, info = self.env.reset(**kwargs)
        self.steps_without_progress = 0
        self.max_x = info.get("x_pos")
        return obs, info

    def step(self, action):
        """Nhận `action` hợp lệ và kiểm tra vị trí ngang sau một bước tác tử.

        Trả `(obs, reward, terminated, truncated, info)`. Khi `x_pos` vượt
        mốc trước quá 2 pixel, cập nhật mốc và đặt lại bộ đếm. Khi chạm ngưỡng
        `max_steps`, đặt `truncated=True`. Nếu môi trường đã báo kết thúc
        hoặc cắt lượt, trả nguyên hai cờ của môi trường.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = info.get("x_pos")
        if not (terminated or truncated) and x is not None:
            # mo-supermario-v0 chỉ trả x_pos từ bước đầu tiên.
            if self.max_x is None or x > self.max_x + 2:
                self.max_x = x
                self.steps_without_progress = 0
            else:
                self.steps_without_progress += 1
                truncated = self.steps_without_progress >= self.max_steps
        return obs, reward, terminated, truncated, info
