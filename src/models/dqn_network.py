"""Dueling Deep Q-Network (Dueling DQN) architecture for Super Mario Bros.

References:
    - Mnih et al., 2015: "Human-level control through deep reinforcement learning" (Nature CNN).
    - Wang et al., 2016: "Dueling Network Architectures for Deep Reinforcement Learning" (ICML).
"""

import numbers
import numpy as np
import torch
import torch.nn as nn


def _is_positive_integer(value: object) -> bool:
    """Validate that value is a positive integer (rejects bool, floats, and strings)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, np.integer, numbers.Integral)):
        return int(value) >= 1
    return False


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network for discrete action reinforcement learning.

    Decomposes state-action value Q(s, a) into state value V(s) and action advantage A(s, a):
        Q(s, a) = V(s) + (A(s, a) - mean_{a'}(A(s, a')))

    Designed for 84x84 visual observations with channel-first order (B, in_channels, 84, 84).
    The contract assumes inputs represent raw pixel values in the range [0, 255]
    (typically torch.uint8 from the environment, or floating-point tensor on the same scale),
    which are converted to float32 and normalized by dividing 255.0 exactly once in forward().
    """

    def __init__(self, in_channels: int = 4, num_actions: int = 7) -> None:
        """Initialize the DuelingDQN network.

        Args:
            in_channels: Number of stacked frame channels (positive integer). Default is 4.
            num_actions: Number of discrete actions in the action space (positive integer,
                supports Python int or Gymnasium space.n / np.integer).
                Default is 7 (SIMPLE_MOVEMENT). Also supports 5 (RIGHT_ONLY) or 2 (RIGHT_JUMP).

        Raises:
            ValueError: If in_channels or num_actions is not a positive integer (rejects bool).
        """
        super().__init__()

        if not _is_positive_integer(in_channels):
            raise ValueError(f"in_channels must be a positive integer, got {in_channels!r}")
        if not _is_positive_integer(num_actions):
            raise ValueError(f"num_actions must be a positive integer, got {num_actions!r}")

        self.in_channels = int(in_channels)
        self.num_actions = int(num_actions)

        # Nature CNN feature extractor backbone (Mnih et al., 2015)
        # Input: (B, in_channels, 84, 84)
        # Conv1: floor((84 - 8) / 4) + 1 = 20 -> (B, 32, 20, 20)
        # Conv2: floor((20 - 4) / 2) + 1 = 9  -> (B, 64, 9, 9)
        # Conv3: floor((9 - 3) / 1) + 1 = 7   -> (B, 64, 7, 7)
        # Flatten: 64 * 7 * 7 = 3136 features
        self.feature_network = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Value stream: V(s) -> (B, 1)
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Advantage stream: A(s, a) -> (B, num_actions)
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass computing Q-values for all discrete actions.

        Args:
            state: Tensor with shape (B, in_channels, 84, 84). Expected in raw pixel scale
                [0, 255] as torch.uint8 or floating-point tensor.

        Returns:
            torch.Tensor: Q-values with shape (B, num_actions), float32.

        Raises:
            TypeError: If state is not a torch.Tensor.
            ValueError: If state does not have 4 dimensions, has non-positive batch size,
                has mismatched in_channels, or spatial dimensions differ from 84x84.
        """
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"state must be a torch.Tensor, got {type(state).__name__}")

        if state.ndim != 4:
            raise ValueError(
                f"state must have 4 dimensions (B, C, H, W), got {state.ndim}D tensor "
                f"with shape {tuple(state.shape)}"
            )

        batch_size, channels, height, width = state.shape

        if batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if channels != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, got {channels} "
                f"in shape {tuple(state.shape)}"
            )
        if height != 84 or width != 84:
            raise ValueError(
                f"Expected input spatial dimensions (84, 84), got ({height}, {width}) "
                f"in shape {tuple(state.shape)}"
            )

        # Single normalization step: raw pixel scale [0, 255] -> float32 [0.0, 1.0]
        # Preserves caller device without silent cross-device transfer
        x = state.to(dtype=torch.float32).div(255.0)

        features = self.feature_network(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Dueling formula (Wang et al., 2016, Eq. 9):
        # Q(s, a) = V(s) + (A(s, a) - mean_{a'}(A(s, a')))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values
