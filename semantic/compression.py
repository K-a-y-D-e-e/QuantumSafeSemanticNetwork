
import torch


class AdaptiveSemanticCompressor:
    """
    Compresses semantic latent representations by retaining
    the most important latent dimensions.
    """

    def __init__(self, latent_dim=16):
        self.latent_dim = latent_dim
        self.levels = [0.25, 0.50, 0.75, 1.00]

    def compress(self, latent, compression_level):
        """
        Compress the latent representation.

        Returns:
            compressed: Selected latent features
            indices: Positions of selected features
        """

        if compression_level not in self.levels:
            raise ValueError(
                f"Unsupported compression level: "
                f"{compression_level}"
            )

        keep_dim = max(
            1,
            int(self.latent_dim * compression_level)
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

    def decompress(self, compressed, indices):
        """
        Reconstruct the original latent dimension.

        Missing latent features are filled with zeros.
        """

        full_shape = list(compressed.shape)
        full_shape[-1] = self.latent_dim

        reconstructed = torch.zeros(
            full_shape,
            dtype=compressed.dtype,
            device=compressed.device
        )

        reconstructed[..., indices] = compressed

        return reconstructed