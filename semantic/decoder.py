import torch.nn as nn


class SemanticDecoder(nn.Module):
    """
    Reconstructs the original robotic representation
    from the semantic latent representation.
    """

    def __init__(
        self,
        latent_dim=16,
        hidden_dim=128,
        output_dim=76
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.network(z)
