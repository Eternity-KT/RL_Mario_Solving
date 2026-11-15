"""Construct the single shared Gymnasium pipeline used by the team."""

from collections.abc import Callable

import gymnasium as gym
import gym_super_mario_bros  # noqa: F401 -- registers Gymnasium environments
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from gymnasium.wrappers import FrameStackObservation, RecordEpisodeStatistics, TimeLimit
from nes_py.wrappers import JoypadSpace

from .wrappers import ActionRepeatMaxPool, GrayResizeObservation

_ACTIONS = {
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
    "RIGHT_ONLY": RIGHT_ONLY,
    "RIGHT_JUMP": [["right"], ["right", "A"]],
}


def make_mario_env(
    world: int = 1,
    stage: int = 1,
    *,
    version: int = 0,
    action_type: str = "SIMPLE_MOVEMENT",
    frame_skip: int = 4,
    frame_stack: int = 4,
    screen_size: tuple[int, int] = (84, 84),
    max_episode_steps: int | None = 4500,
    render_mode: str | None = None,
    reward_wrapper: Callable[[gym.Env], gym.Env] | None = None,
) -> gym.Env:
    """Tạo môi trường Mario dùng chung cho train, đánh giá và demo.

    Args:
        world: Số world, là số nguyên từ 1 đến 8.
        stage: Số màn trong world, là số nguyên từ 1 đến 4.
        version: Biến thể ROM đã đăng ký, từ 0 đến 3; 0 là ROM chuẩn.
        action_type: "SIMPLE_MOVEMENT" cho 7 hành động, "RIGHT_ONLY" cho
            5 hành động, hoặc "RIGHT_JUMP" cho 2 hành động (0: right,
            1: right+A). Phải dùng đúng tên, không nhận tên viết tắt.
        frame_skip: Số frame giả lập tối đa lặp lại cùng một hành động.
            Phải là số nguyên dương; mặc định là 4.
        frame_stack: Số ảnh sau tiền xử lý được ghép theo chiều đầu tiên.
            Phải là số nguyên dương; mặc định là 4.
        screen_size: Kích thước ảnh sau resize theo thứ tự (height, width),
            gồm hai số nguyên dương; mặc định là (84, 84).
        max_episode_steps: Giới hạn số bước quyết định của agent trong một
            episode, không phải số frame giả lập. None tắt giới hạn ngoài;
            luật kết thúc trong game vẫn có hiệu lực. Mặc định là 4500.
        render_mode: None để chạy không cửa sổ, "human" để dùng cửa sổ
            native qua render(), hoặc "rgb_array" để render() trả ảnh RGB.
        reward_wrapper: Hàm nhận môi trường đã repeat và giới hạn bước,
            trả về môi trường Gymnasium được bọc reward. None giữ reward
            native. Hook này được gọi một lần khi tạo môi trường, còn
            step() của wrapper reward chạy một lần mỗi bước agent.

    Returns:
        gym.Env: Môi trường đơn theo API Gymnasium. reset() trả (obs, info);
        step() trả (obs, reward, terminated, truncated, info). obs là mảng
        NumPy uint8, khoảng 0–255, shape (frame_stack, height, width), không
        có batch dimension. reward là tổng reward của các frame đã chạy,
        hoặc reward do wrapper tùy chọn trả về. render() dùng ảnh native,
        độc lập với ảnh xám đầu vào mạng.

    Raises:
        ValueError: Giá trị cấu hình nằm ngoài miền hỗ trợ hoặc kích thước
            ảnh/số frame/giới hạn bước không hợp lệ.
        TypeError: reward_wrapper không gọi được hoặc trả về đối tượng
            không phải môi trường Gymnasium.
        gym.error.Error: Gymnasium không thể tạo môi trường đã yêu cầu.

    Notes:
        Thứ tự wrapper: Mario -> Joypad -> repeat/max-pool -> TimeLimit
        -> reward tùy chọn -> grayscale/resize -> stack -> thống kê episode.
        Stack đi từ ảnh cũ đến mới; lúc reset lặp lại ảnh đầu tiên.
        Gọi env.reset(seed=...) để đặt seed môi trường; gọi riêng
        env.action_space.seed(...) nếu cần tái lập random action.
        Reset khi terminated hoặc truncated đúng và luôn close() sau khi
        dùng, hoặc dùng context manager. Nếu bọc wrapper thất bại, factory
        đóng môi trường đã tạo rồi truyền lỗi ra ngoài. Không chuẩn hóa
        ảnh sang float hay tự gắn reward shaping tại đây.

    Examples:
        >>> with make_mario_env() as env:
        ...     obs, info = env.reset(seed=42)
        ...     obs, reward, terminated, truncated, info = env.step(1)
    """
    for name, value, low, high in (
        ("world", world, 1, 8),
        ("stage", stage, 1, 4),
        ("version", version, 0, 3),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if action_type not in _ACTIONS:
        raise ValueError(f"action_type must be one of {tuple(_ACTIONS)}")
    for name, value in (("frame_skip", frame_skip), ("frame_stack", frame_stack)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_episode_steps is not None and (
        type(max_episode_steps) is not int or max_episode_steps < 1
    ):
        raise ValueError("max_episode_steps must be positive or None")
    if render_mode not in (None, "human", "rgb_array"):
        raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
    if len(screen_size) != 2 or any(
        type(value) is not int or value < 1 for value in screen_size
    ):
        raise ValueError("screen_size must contain positive (height, width) integers")

    # Disable the registration's frame-based TimeLimit. Our limit is applied
    # after action repetition, so changing frame_skip cannot change its units.
    env = gym.make(
        f"SuperMarioBros-{world}-{stage}-v{version}",
        render_mode=render_mode,
        max_episode_steps=-1,
    )
    try:
        env = JoypadSpace(env, _ACTIONS[action_type])
        env = ActionRepeatMaxPool(env, repeat=frame_skip)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        if reward_wrapper is not None:
            wrapped_env = reward_wrapper(env)
            if not isinstance(wrapped_env, gym.Env):
                raise TypeError("reward_wrapper must return a Gymnasium environment")
            env = wrapped_env
        env = GrayResizeObservation(env, screen_size=screen_size)
        env = FrameStackObservation(env, stack_size=frame_stack, padding_type="reset")
        env = RecordEpisodeStatistics(env)
    except Exception:
        # A failed user-supplied wrapper must not leak a native emulator handle.
        env.close()
        raise
    return env
