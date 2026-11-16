"""Kiểm thử môi trường Mario do Người 1 xây dựng (src/env/make_env.py).

Chạy giả lập NES thật, không huấn luyện agent. Nếu chưa cài
gymnasium / nes_py / gym_super_mario_bros hoặc chưa có src.env.make_env,
cả module được skip kèm thông báo rõ ràng để CI không bị đỏ.

Chạy nhanh:  python -m pytest tests/test_env.py -m "not slow" -v
Chạy đủ:     python -m pytest tests/test_env.py -v
"""

import math

import numpy as np
import pytest

# --- Kiểm tra phụ thuộc: thiếu thư viện thì skip cả module -----------------
_HINT = "cài bằng: pip install -r requirements-dev.txt"
gym = pytest.importorskip("gymnasium", reason=f"Chưa cài gymnasium ({_HINT})")
pytest.importorskip("nes_py", reason=f"Chưa cài nes_py ({_HINT})")
pytest.importorskip(
    "gym_super_mario_bros", reason=f"Chưa cài gym_super_mario_bros ({_HINT})"
)

try:
    from src.env.make_env import make_mario_env
except ModuleNotFoundError as exc:  # pragma: no cover - chỉ khi Người 1 chưa bàn giao
    # Chỉ skip khi chính module của Người 1 chưa tồn tại; lỗi import khác
    # bên trong make_env.py (bug thật) vẫn phải làm test đỏ.
    if exc.name not in ("src", "src.env", "src.env.make_env"):
        raise
    pytest.skip(
        f"Chưa có src.env.make_env (Người 1 chưa bàn giao): {exc}",
        allow_module_level=True,
    )

from gymnasium.utils.env_checker import check_env  # noqa: E402

pytestmark = pytest.mark.integration

# --- Hằng số theo đặc tả Người 1 / configs/env.json ------------------------
FRAME_STACK = 4
SCREEN = (84, 84)
OBS_SHAPE = (FRAME_STACK, *SCREEN)
FRAME_SKIP = 4
NATIVE_RENDER_SHAPE = (240, 256, 3)
# Reward native của gym_super_mario_bros bị clip trong [-15, 15] mỗi frame.
MAX_ABS_REWARD_PER_STEP = 15.0 * FRAME_SKIP
INFO_KEYS = ("x_pos", "y_pos", "flag_get", "life", "time", "coins", "score")
SEED = 42
ACTION_RIGHT = 1  # SIMPLE_MOVEMENT[1] == ["right"]


@pytest.fixture(scope="module")
def env():
    """Môi trường mặc định (SIMPLE_MOVEMENT, 4x84x84), dùng chung cho module."""
    environment = make_mario_env()
    yield environment
    environment.close()


def _assert_valid_obs(environment, observation):
    """Kiểm tra quan sát đúng shape, dtype uint8, miền 0–255 và thuộc space."""
    assert isinstance(observation, np.ndarray)
    assert observation.shape == OBS_SHAPE
    assert observation.dtype == np.uint8
    assert observation.min() >= 0 and observation.max() <= 255
    assert environment.observation_space.contains(observation)


# --- Reset / không gian quan sát / hành động --------------------------------
def test_reset_returns_obs_and_info(env):
    """reset() theo API Gymnasium trả (obs, info) với obs (4, 84, 84) uint8."""
    result = env.reset(seed=SEED)
    assert isinstance(result, tuple) and len(result) == 2
    observation, info = result
    _assert_valid_obs(env, observation)
    assert isinstance(info, dict)
    # Ảnh không được toàn đen/toàn một màu: tiền xử lý thật sự giữ nội dung.
    assert observation.max() > observation.min()


def test_observation_space_spec(env):
    """observation_space là Box uint8, shape (4, 84, 84), miền [0, 255]."""
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == OBS_SHAPE
    assert space.dtype == np.uint8
    assert np.all(space.low == 0) and np.all(space.high == 255)


def test_default_action_space_is_simple_movement(env):
    """Mặc định SIMPLE_MOVEMENT: Discrete(7)."""
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 7


@pytest.mark.parametrize(
    "action_type, action_count",
    [("SIMPLE_MOVEMENT", 7), ("RIGHT_ONLY", 5), ("RIGHT_JUMP", 2)],
)
def test_action_types_and_observations(action_type, action_count):
    """Mỗi action_type cho đúng số hành động; stack dịch đúng từ cũ đến mới."""
    with make_mario_env(action_type=action_type) as environment:
        observation, info = environment.reset(seed=SEED)
        _assert_valid_obs(environment, observation)
        assert environment.action_space.n == action_count
        assert "x_pos" in info and "flag_get" in info
        # Lúc reset, stack lặp lại ảnh đầu tiên chứ không lấy ảnh cũ.
        assert np.all(observation == observation[0])
        snapshot = observation.copy()
        next_observation, reward, terminated, truncated, info = environment.step(1)
        # step() không được sửa tại chỗ quan sát đã trả trước đó.
        np.testing.assert_array_equal(observation, snapshot)
        np.testing.assert_array_equal(next_observation[:-1], snapshot[1:])
        _assert_valid_obs(environment, next_observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert 1 <= info["frames_this_step"] <= FRAME_SKIP


# --- step() ------------------------------------------------------------------
def test_step_returns_gymnasium_five_tuple(env):
    """step() trả (obs, reward, terminated, truncated, info) – 5 giá trị."""
    env.reset(seed=SEED)
    result = env.step(ACTION_RIGHT)
    assert isinstance(result, tuple) and len(result) == 5
    observation, reward, terminated, truncated, info = result
    _assert_valid_obs(env, observation)
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)


def test_step_info_contains_game_state(env):
    """info chứa x_pos, flag_get, life... với kiểu dữ liệu hợp lý."""
    env.reset(seed=SEED)
    _, _, _, _, info = env.step(ACTION_RIGHT)
    missing = [key for key in INFO_KEYS if key not in info]
    assert not missing, f"info thiếu các khóa: {missing}"
    assert int(info["x_pos"]) >= 0
    assert isinstance(info["flag_get"], (bool, np.bool_))
    assert int(info["life"]) >= 0


def test_reward_is_finite_and_bounded(env):
    """Reward là số hữu hạn, nằm trong biên clip native nhân frame_skip."""
    env.reset(seed=SEED)
    for _ in range(20):
        _, reward, terminated, truncated, _ = env.step(ACTION_RIGHT)
        assert math.isfinite(float(reward))
        assert abs(float(reward)) <= MAX_ABS_REWARD_PER_STEP
        if terminated or truncated:
            break


def test_moving_right_increases_x_pos(env):
    """Giữ nút phải vài bước thì x_pos tăng (hành động thật sự tác động)."""
    _, info = env.reset(seed=SEED)
    start_x = int(info["x_pos"])
    for _ in range(15):
        _, _, terminated, truncated, info = env.step(ACTION_RIGHT)
        if terminated or truncated:
            break
    assert int(info["x_pos"]) > start_x


# --- Frame skip / frame stack ------------------------------------------------
def test_frame_skip_runs_four_emulator_frames(env):
    """Mỗi bước agent chạy đúng frame_skip frame giả lập (khi chưa kết thúc)."""
    env.reset(seed=SEED)
    for step in range(1, 4):
        _, _, terminated, truncated, info = env.step(ACTION_RIGHT)
        assert not (terminated or truncated)
        assert info["frames_this_step"] == FRAME_SKIP
        assert info["episode_emulator_frames"] == FRAME_SKIP * step


def test_frame_stack_frames_differ_after_moving(env):
    """Sau vài bước di chuyển, các frame trong stack khác nhau."""
    env.reset(seed=SEED)
    for _ in range(FRAME_STACK + 2):
        observation, _, terminated, truncated, _ = env.step(ACTION_RIGHT)
        assert not (terminated or truncated)
    assert not np.array_equal(observation[0], observation[-1])
    distinct = {observation[i].tobytes() for i in range(FRAME_STACK)}
    assert len(distinct) > 1


# --- Giới hạn bước, reset, render ------------------------------------------
def test_time_limit_counts_agent_decisions_and_reset_discards_history():
    """max_episode_steps tính theo bước agent; reset xóa lịch sử episode."""
    with make_mario_env(max_episode_steps=2) as environment:
        initial, _ = environment.reset(seed=SEED)
        _, _, terminated, truncated, _ = environment.step(0)
        assert not terminated and not truncated
        _, _, terminated, truncated, info = environment.step(0)
        assert not terminated and truncated
        assert info["episode_emulator_frames"] == 2 * FRAME_SKIP
        assert info["episode"]["l"] == 2
        reset_observation, info = environment.reset(seed=SEED)
        np.testing.assert_array_equal(reset_observation, initial)
        assert info["episode_emulator_frames"] == 0


def test_native_render_is_rgb_and_network_observation_is_grayscale():
    """render() trả ảnh RGB native; quan sát mạng là ảnh xám theo cấu hình."""
    with make_mario_env(
        render_mode="rgb_array", screen_size=(40, 60), frame_stack=2
    ) as environment:
        observation, _ = environment.reset(seed=3)
        frame = environment.render()
        assert observation.shape == (2, 40, 60)
        assert frame.shape == NATIVE_RENDER_SHAPE
        assert frame.dtype == np.uint8


def test_reward_hook_runs_once_per_decision_and_sees_external_truncation():
    """reward_wrapper chạy một lần mỗi bước agent và thấy truncated của TimeLimit."""
    calls = []

    class RewardProbe(gym.Wrapper):
        """Wrapper thăm dò: ghi lại reward gốc và cộng thêm 7."""

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            calls.append((reward, truncated, info["frames_this_step"]))
            return observation, reward + 7.0, terminated, truncated, info

    with make_mario_env(max_episode_steps=1, reward_wrapper=RewardProbe) as environment:
        environment.reset()
        _, reward, _, truncated, info = environment.step(0)
        assert calls == [(info["default_reward"], True, FRAME_SKIP)]
        assert truncated
        assert reward == info["default_reward"] + 7.0
        assert info["episode"]["r"] == reward


@pytest.mark.parametrize(
    "kwargs",
    [
        {"world": 0}, {"stage": 5}, {"version": 4},
        {"action_type": "SIMPLE"}, {"frame_skip": 0},
        {"frame_stack": True}, {"screen_size": (0, 84)},
        {"max_episode_steps": 0}, {"render_mode": "bad"},
    ],
)
def test_invalid_factory_configuration(kwargs):
    """Cấu hình sai bị từ chối bằng ValueError trước khi tạo giả lập."""
    with pytest.raises(ValueError):
        make_mario_env(**kwargs)


# --- Kiểm thử chậm -----------------------------------------------------------
@pytest.mark.slow
def test_gymnasium_contract():
    """Bộ check_env chính thức của Gymnasium chạy qua toàn pipeline."""
    with make_mario_env() as environment:
        # JoypadSpace không lưu tham số khởi tạo nên bỏ qua bước tạo lại env.
        check_env(environment, skip_render_check=True, skip_close_check=True)


@pytest.mark.slow
def test_seeded_action_trace_is_reproducible():
    """Cùng seed env + seed action_space cho cùng chuỗi quan sát/reward."""

    def trace(environment):
        environment.reset(seed=31)
        environment.action_space.seed(81)
        result = []
        for _ in range(30):
            observation, reward, terminated, truncated, info = environment.step(
                environment.action_space.sample()
            )
            result.append(
                (observation.copy(), reward, terminated, truncated, info["x_pos"])
            )
            if terminated or truncated:
                environment.reset()
        return result

    with make_mario_env() as environment:
        first, second = trace(environment), trace(environment)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]


@pytest.mark.slow
def test_random_agent_1000_steps_without_error(env):
    """Random agent chạy 1.000 bước liên tục (DoD tuần 1), tự reset khi hết episode."""
    env.action_space.seed(SEED)
    observation, _ = env.reset(seed=SEED)
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        assert observation.shape == OBS_SHAPE and observation.dtype == np.uint8
        assert math.isfinite(float(reward))
        if terminated or truncated:
            observation, _ = env.reset()
            _assert_valid_obs(env, observation)


@pytest.mark.slow
def test_reset_after_terminated_episode():
    """Đứng yên đến khi Mario chết (terminated) rồi reset và chơi tiếp được."""
    with make_mario_env(max_episode_steps=None) as environment:
        environment.reset(seed=SEED)
        terminated = truncated = False
        # Goomba đầu tiên đi tới giết Mario đứng yên; game timer là chặn trên.
        for _ in range(3000):
            _, _, terminated, truncated, info = environment.step(0)  # NOOP
            if terminated or truncated:
                break
        assert terminated, "Mario đứng yên nhưng episode không kết thúc"
        assert not info["flag_get"]
        # Sau khi kết thúc phải reset; step tiếp mà không reset là lỗi.
        with pytest.raises(gym.error.ResetNeeded):
            environment.step(0)
        observation, info = environment.reset()
        _assert_valid_obs(environment, observation)
        assert info["episode_emulator_frames"] == 0
        observation, _, terminated, truncated, _ = environment.step(ACTION_RIGHT)
        _assert_valid_obs(environment, observation)
        assert not (terminated or truncated)
