"""
Utilities for loading trained semantic encoder/decoder checkpoints.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch

from semantic.encoder import SemanticEncoder
from semantic.decoder import SemanticDecoder


DEFAULT_ENCODER_PATH = "semantic_encoder.pth"
DEFAULT_DECODER_PATH = "semantic_decoder.pth"
DEFAULT_INPUT_DIM = 76
DEFAULT_LATENT_DIM = 16
DEFAULT_HIDDEN_DIM = 128


def load_semantic_models(
    encoder_path: str = DEFAULT_ENCODER_PATH,
    decoder_path: str = DEFAULT_DECODER_PATH,
    input_dim: int = DEFAULT_INPUT_DIM,
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    device: Optional[torch.device] = None,
) -> Tuple[SemanticEncoder, SemanticDecoder]:
    """
    Load encoder and decoder weights from disk.

    Raises FileNotFoundError if checkpoint files are missing.
    """
    if device is None:
        device = torch.device("cpu")

    encoder_file = Path(encoder_path)
    decoder_file = Path(decoder_path)

    if not encoder_file.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_file}")
    if not decoder_file.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_file}")

    encoder = SemanticEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    decoder = SemanticDecoder(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
    ).to(device)

    encoder.load_state_dict(
        torch.load(encoder_file, map_location=device, weights_only=True)
    )
    decoder.load_state_dict(
        torch.load(decoder_file, map_location=device, weights_only=True)
    )

    encoder.eval()
    decoder.eval()

    return encoder, decoder
