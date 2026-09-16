import torch


class AdaptiveSemanticCompressor:
    """
    Applies adaptive compression to a semantic latent representation.

    compression_level:
        0.25 -> retain approximately 25% of latent features
        0.50 -> retain approximately 50%
        0.75 -> retain approximately 75%
        1.00 -> retain all features
    """

    def __init__(self):
        self.levels = [0.25, 0.50, 0.75, 1.00]

    def compress(self, latent, compression_level):
        """
        Retain the most important latent dimensions based on
        average absolute activation.
        """

        if compression_level not in self.levels:
            raise ValueError(
                f"Unsupported compression level: "
                f"{compression_level}"
            )

        latent_dim = latent.shape[-1]

        keep_dim = max(
            1,
            int(latent_dim * compression_level)
        )

        importance = torch.mean(
            torch.abs(latent),
            dim=tuple(range(latent.dim() - 1))
        )

        indices = torch.topk(
            importance,
            k=keep_dim
        ).indices

        compressed = latent[..., indices]

        return compressed, indices
