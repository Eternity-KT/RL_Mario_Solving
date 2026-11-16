"""Unit tests for DuelingDQN architecture.

These tests are fully deterministic and isolated from the NES emulator,
exercising only the PyTorch neural network module and its contracts.
"""

import pytest
import torch

from src.models import DuelingDQN


def test_default_initialization():
    """Verify default initialization: 4 input channels and 7 discrete actions."""
    model = DuelingDQN()
    assert model.in_channels == 4
    assert model.num_actions == 7
    assert isinstance(model.feature_network, torch.nn.Sequential)
    assert isinstance(model.value_stream, torch.nn.Sequential)
    assert isinstance(model.advantage_stream, torch.nn.Sequential)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_forward_batch_sizes(batch_size):
    """Verify forward pass with varying batch sizes using raw uint8 observations."""
    model = DuelingDQN(in_channels=4, num_actions=7)
    model.eval()

    # Raw pixel input from environment: uint8 [0, 255]
    state = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        output = model(state)

    assert output.shape == (batch_size, 7)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("num_actions", [2, 5, 7])
def test_action_counts_parameterized(num_actions):
    """Verify model supports configured action spaces: 2 (RIGHT_JUMP), 5 (RIGHT_ONLY), 7 (SIMPLE)."""
    batch_size = 2
    model = DuelingDQN(in_channels=4, num_actions=num_actions)
    model.eval()

    state = torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        output = model(state)

    assert output.shape == (batch_size, num_actions)
    assert output.dtype == torch.float32


def test_output_properties():
    """Verify output is a floating-point tensor with all finite values on the correct device."""
    model = DuelingDQN()
    model.eval()

    state = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        output = model(state)

    assert output.is_floating_point()
    assert output.device == state.device
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


def test_backbone_layer_shapes():
    """Verify intermediate spatial dimensions and flattened features of Nature CNN backbone."""
    batch_size = 3
    model = DuelingDQN(in_channels=4, num_actions=7)
    model.eval()

    state_normalized = torch.rand(batch_size, 4, 84, 84, dtype=torch.float32)

    # Conv1: Conv2d(4, 32, 8, 4) + ReLU -> (B, 32, 20, 20)
    conv1_out = model.feature_network[1](model.feature_network[0](state_normalized))
    assert conv1_out.shape == (batch_size, 32, 20, 20)

    # Conv2: Conv2d(32, 64, 4, 2) + ReLU -> (B, 64, 9, 9)
    conv2_out = model.feature_network[3](model.feature_network[2](conv1_out))
    assert conv2_out.shape == (batch_size, 64, 9, 9)

    # Conv3: Conv2d(64, 64, 3, 1) + ReLU -> (B, 64, 7, 7)
    conv3_out = model.feature_network[5](model.feature_network[4](conv2_out))
    assert conv3_out.shape == (batch_size, 64, 7, 7)

    # Flatten -> (B, 3136)
    flatten_out = model.feature_network[6](conv3_out)
    assert flatten_out.shape == (batch_size, 3136)


def test_dueling_aggregation_formula():
    """Verify Dueling aggregation matches manual calculation: Q = V + (A - mean(A))."""
    model = DuelingDQN(in_channels=4, num_actions=7)
    model.eval()

    state = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        actual_q = model(state)

        # Manually compute streams from backbone
        norm_x = state.float().div(255.0)
        features = model.feature_network(norm_x)
        values = model.value_stream(features)
        advantages = model.advantage_stream(features)
        expected_q = values + (advantages - advantages.mean(dim=-1, keepdim=True))

    torch.testing.assert_close(actual_q, expected_q)


def test_mean_centered_property():
    """Verify the mathematical identity: mean_a(Q(s, a)) == V(s) for each sample."""
    model = DuelingDQN(in_channels=4, num_actions=7)
    model.eval()

    state = torch.randint(0, 256, (5, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        actual_q = model(state)

        norm_x = state.float().div(255.0)
        features = model.feature_network(norm_x)
        values = model.value_stream(features)  # (5, 1)

        # Mean Q across action dimension: (5, 1)
        mean_q_over_actions = actual_q.mean(dim=-1, keepdim=True)

    # Mathematically: mean_a(V(s) + A(s, a) - mean_a'(A(s, a'))) = V(s) + 0 = V(s)
    torch.testing.assert_close(mean_q_over_actions, values, rtol=1e-5, atol=1e-6)


def test_backward_pass():
    """Verify gradient propagation through all learnable parameters."""
    model = DuelingDQN(in_channels=4, num_actions=7)
    model.train()
    model.zero_grad()

    state = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    q_values = model(state)
    loss = q_values.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for parameter '{name}' is None"
        assert torch.isfinite(param.grad).all(), f"Gradient for parameter '{name}' has non-finite values"


@pytest.mark.parametrize(
    "invalid_state, expected_exc, match_msg",
    [
        ("not_a_tensor", TypeError, "state must be a torch.Tensor"),
        (torch.zeros(4, 84, 84), ValueError, "state must have 4 dimensions"),
        (torch.zeros(0, 4, 84, 84), ValueError, "Batch size must be positive"),
        (torch.zeros(2, 3, 84, 84), ValueError, "Expected input with 4 channels"),
        (torch.zeros(2, 4, 80, 84), ValueError, "Expected input spatial dimensions"),
        (torch.zeros(2, 4, 84, 90), ValueError, "Expected input spatial dimensions"),
    ],
)
def test_invalid_inputs(invalid_state, expected_exc, match_msg):
    """Verify informative errors on invalid input types, dimensions, channels, and sizes."""
    model = DuelingDQN(in_channels=4, num_actions=7)
    with pytest.raises(expected_exc, match=match_msg):
        model(invalid_state)


def test_device_cpu():
    """Verify CPU operation and device preservation."""
    model = DuelingDQN(in_channels=4, num_actions=7).to("cpu")
    state = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8, device="cpu")
    output = model(state)
    assert output.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available on this system")
def test_device_cuda():
    """Verify CUDA operation and device preservation when CUDA is available."""
    device = torch.device("cuda:0")
    model = DuelingDQN(in_channels=4, num_actions=7).to(device)
    state = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8, device=device)
    output = model(state)
    assert output.device.type == "cuda"
    assert output.device == device


def test_raw_pixel_scale_semantics():
    """Verify that uint8 [0, 255] and float32 [0.0, 255.0] produce identical outputs.

    This asserts the unified contract: inputs represent raw pixel values on scale [0, 255].
    """
    model = DuelingDQN(in_channels=4, num_actions=7)
    model.eval()

    state_uint8 = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    state_float_raw = state_uint8.to(dtype=torch.float32)

    with torch.no_grad():
        out_from_uint8 = model(state_uint8)
        out_from_float = model(state_float_raw)

    torch.testing.assert_close(out_from_uint8, out_from_float)
