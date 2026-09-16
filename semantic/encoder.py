import torch
import torch.nn as nn


class SemanticEncoder(nn.Module):
    """
    Converts high-dimensional robotic kinematic data
    into a compact latent semantic representation.
    """

    def __init__(
        self,
        input_dim,
        latent_dim=16,
        hidden_dim=128
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.network(x)
