"""Image preprocessing with explicit action-repeat and frame ownership."""

from collections import deque

import gymnasium as gym
import numpy as np
from PIL import Image


class ActionRepeatMaxPool(gym.Wrapper):
    """Repeat an action, sum rewards, and pool the final two executed RGB frames.

    Frames are copied because NES observations can alias a reusable emulator
    buffer. Pooling uses only frames from this call, including early endings.
    Native termination and truncation flags are preserved unchanged.
    """

    def __init__(self, env: gym.Env, repeat: int = 4):
        """Khởi tạo wrapper lặp hành động và max-pooling.

        Args:
            env: Môi trường Gymnasium cung cấp ảnh NumPy. Trong pipeline
                Mario, ảnh là RGB uint8 có shape (height, width, 3).
            repeat: Số lần gọi env.step(action) tối đa cho mỗi bước của
                wrapper. Phải là số nguyên dương, không nhận bool.

        Raises:
            ValueError: repeat không phải số nguyên dương.

        Notes:
            Giữ nguyên observation_space và action_space của env.
            Chưa reset game tại đây; phải gọi reset() trước step().
        """
        super().__init__(env)
        if type(repeat) is not int or repeat < 1:
            raise ValueError("repeat must be a positive integer")
        self.repeat = repeat
        self._episode_frames = 0
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        """Reset game và bắt đầu lại bộ đếm frame của episode.

        Args:
            seed: Seed được truyền nguyên vẹn xuống env.reset(). None
                giữ cách quản lý RNG mặc định của môi trường bên trong.
            options: Dictionary tùy chọn reset hoặc None, được truyền
                xuống môi trường bên trong; wrapper này không diễn giải.

        Returns:
            tuple[np.ndarray, dict]: (observation, info). observation là
            bản sao ảnh đầu episode, cùng shape/dtype với ảnh native.
            info giữ các trường native và đặt frames_this_step cùng
            episode_emulator_frames về 0.

        Notes:
            Cho phép gọi step() lại sau khi episode kết thúc. Ảnh trả về
            không dùng chung bộ nhớ với bộ đệm ảnh có thể bị giả lập ghi đè.
            Seed ở đây không đặt seed riêng cho action_space.sample().
        """
        observation, info = self.env.reset(seed=seed, options=options)
        self._episode_frames = 0
        self._needs_reset = False
        info = dict(info)
        info.update(frames_this_step=0, episode_emulator_frames=0)
        return np.array(observation, copy=True), info

    def step(self, action):
        """Chạy một bước agent bằng cách lặp action tối đa repeat frame.

        Args:
            action: Một hành động hợp lệ trong action_space của env,
                được giữ nguyên qua các frame. Sau JoypadSpace, đây là
                chỉ số nguyên của tổ hợp nút bấm.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Bao gồm:
                observation: Max theo từng phần tử của hai ảnh cuối thực
                    sự đã chạy; nếu chỉ chạy một frame thì dùng ảnh đó.
                    Shape/dtype giữ nguyên, không chia sẻ bộ nhớ ảnh native.
                reward: Tổng reward các frame đã chạy, không chiết khấu
                    và không clipping thêm ở wrapper này.
                terminated: Cờ kết thúc theo luật môi trường bên trong.
                truncated: Cờ ngắt ngoài của môi trường bên trong.
                info: Info của frame cuối, bổ sung frames_this_step,
                    episode_emulator_frames, default_reward và
                    reward_components_sum. Các thành phần reward được
                    cộng trước clipping native nên tổng có thể khác reward.

        Raises:
            gym.error.ResetNeeded: Chưa gọi reset(), hoặc môi trường bên
                trong đã trả terminated/truncated ở lượt trước.

        Notes:
            Dừng lặp ngay khi một trong hai cờ kết thúc đúng và giữ nguyên
            cả hai cờ. Không lấy ảnh từ bước trước hoặc episode trước để
            pooling. Wrapper TimeLimit nằm ngoài lớp này có thể đặt thêm
            truncated; caller vẫn phải reset khi một trong hai cờ đúng.
        """
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before stepping this environment")

        frames = deque(maxlen=2)
        reward_sum = 0.0
        components_sum: dict[str, float] = {}
        for frames_executed in range(1, self.repeat + 1):
            observation, reward, terminated, truncated, info = self.env.step(action)
            frames.append(np.array(observation, copy=True))
            reward_sum += float(reward)
            self._episode_frames += 1
            for name, value in info.get("reward_components", {}).items():
                components_sum[name] = components_sum.get(name, 0.0) + float(value)
            if terminated or truncated:
                self._needs_reset = True
                break

        pooled = frames[-1] if len(frames) == 1 else np.maximum(frames[0], frames[1])
        info = dict(info)
        info.update(
            frames_this_step=frames_executed,
            episode_emulator_frames=self._episode_frames,
            default_reward=reward_sum,
            reward_components_sum=components_sum,
        )
        return pooled, reward_sum, terminated, truncated, info


class GrayResizeObservation(gym.ObservationWrapper):
    """Convert pooled RGB to grayscale, then downsample with Pillow BOX.

    Output is a C-contiguous uint8 array with shape (height, width). Pixel
    normalization belongs to the neural network, not the replay buffer.
    """

    def __init__(self, env: gym.Env, screen_size: tuple[int, int] = (84, 84)):
        """Khai báo phép đổi ảnh RGB sang ảnh xám có kích thước cố định.

        Args:
            env: Môi trường có observation_space là Box uint8, shape
                (height, width, 3), với thứ tự kênh RGB.
            screen_size: Kích thước đầu ra (height, width), gồm hai số
                nguyên dương; không dùng thứ tự (width, height) của Pillow.

        Raises:
            ValueError: screen_size không hợp lệ hoặc observation_space
                không phải Box RGB uint8 có ba chiều như yêu cầu.

        Notes:
            Đặt observation_space đầu ra thành Box 0–255, uint8,
            shape (height, width). Không thay đổi action_space, reward
            hoặc cờ kết thúc của môi trường.
        """
        super().__init__(env)
        if (
            len(screen_size) != 2
            or any(type(value) is not int or value < 1 for value in screen_size)
        ):
            raise ValueError("screen_size must contain positive (height, width) integers")
        if (
            not isinstance(env.observation_space, gym.spaces.Box)
            or len(env.observation_space.shape) != 3
            or env.observation_space.shape[-1] != 3
            or env.observation_space.dtype != np.uint8
        ):
            raise ValueError("GrayResizeObservation requires an RGB uint8 Box")
        self.screen_size = tuple(screen_size)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.screen_size, dtype=np.uint8
        )

    def observation(self, observation):
        """Đổi một ảnh RGB thành ảnh xám rồi resize bằng bộ lọc BOX.

        Args:
            observation: Ảnh NumPy RGB uint8, shape (height, width, 3),
                phù hợp observation_space của môi trường bên trong.

        Returns:
            np.ndarray: Ảnh xám uint8 trong khoảng 0–255, C-contiguous,
            shape self.screen_size tức (height, width). Không còn chiều
            kênh màu và không dùng chung bộ nhớ với ảnh đầu vào.

        Notes:
            Không sửa ảnh đầu vào. ObservationWrapper tự gọi hàm này cho
            ảnh trả về từ cả reset() và step(); FrameStackObservation
            bên ngoài mới thêm chiều stack. Hàm không chuẩn hóa về [0, 1].
        """
        height, width = self.screen_size
        image = Image.fromarray(observation).convert("L")
        image = image.resize((width, height), resample=Image.Resampling.BOX)
        return np.array(image, dtype=np.uint8, copy=True)
