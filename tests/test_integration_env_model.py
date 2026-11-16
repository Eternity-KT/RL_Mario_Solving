"""Integration test verifying DuelingDQN directly accepts observations from make_mario_env().

This test exercises the live NES emulator and full Gymnasium pipeline.
If the NES emulator (nes-py / gym-super-mario-bros) is not installed in the
running environment, the test is cleanly skipped.
"""

import pytest
import torch

# Skip cleanly if emulator packages are not installed in the running interpreter
pytest.importorskip("gymnasium", reason="gymnasium not installed")
pytest.importorskip("nes_py", reason="nes_py not installed")
pytest.importorskip("gym_super_mario_bros", reason="gym_super_mario_bros not installed")

try:
    from src.env.make_env import make_mario_env
except ModuleNotFoundError as exc:
    pytest.skip(f"src.env.make_env not available: {exc}", allow_module_level=True)

from src.models import DuelingDQN

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "action_type, expected_actions",
    [
        ("SIMPLE_MOVEMENT", 7),
        ("RIGHT_ONLY", 5),
        ("RIGHT_JUMP", 2),
    ],
)
def test_dueling_dqn_with_real_mario_env(action_type, expected_actions):
    """Verify DuelingDQN seamlessly receives real observations from make_mario_env()."""
    with make_mario_env(action_type=action_type) as env:
        # Verify action space
        assert env.action_space.n == expected_actions

        # Step 1: Reset with seed
        obs, info = env.reset(seed=42)

        # Step 2: Validate real observation contract
        assert obs.shape == (4, 84, 84)
        assert obs.dtype.name == "uint8"
        assert 0 <= obs.min() <= 255
        assert 0 <= obs.max() <= 255

        # Step 3: Instantiate model using dynamic env.action_space.n (not hard-coded)
        model = DuelingDQN(in_channels=obs.shape[0], num_actions=env.action_space.n)
        model.eval()

        # Step 4: Convert observation to torch.Tensor and add batch dimension: (1, 4, 84, 84)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        assert obs_tensor.shape == (1, 4, 84, 84)
        assert obs_tensor.dtype == torch.uint8

        # Step 5: Forward pass
        with torch.no_grad():
            q_values = model(obs_tensor)

        assert q_values.shape == (1, expected_actions)
        assert q_values.dtype == torch.float32
        assert torch.isfinite(q_values).all()

        # Step 6: Select action and execute one step
        action = int(q_values.argmax(dim=-1).item())
        assert 0 <= action < expected_actions

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Step 7: Verify next observation forward pass
        next_tensor = torch.from_numpy(next_obs).unsqueeze(0)
        with torch.no_grad():
            next_q = model(next_tensor)

        assert next_q.shape == (1, expected_actions)
        assert torch.isfinite(next_q).all()
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
