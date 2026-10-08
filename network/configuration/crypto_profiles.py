"""
Cryptographic profiles for network-condition evaluation.

These profiles describe the communication overhead associated with
classical and post-quantum cryptographic configurations.

The profiles separate:

1. Network/message-size overhead
2. Cryptographic computational overhead

This distinction is important because cryptographic CPU time should
not automatically be treated as network transmission time.
"""


# ---------------------------------------------------------------------------
# Classical cryptographic profile
# ---------------------------------------------------------------------------

CLASSICAL_CRYPTO_PROFILE = {
    "profile": "CLASSICAL",

    "key_exchange": {
        "algorithm": "X25519",
        "public_key_bytes": 32,
        "ciphertext_bytes": 0,
        "shared_secret_bytes": 32,
    },

    "authentication": {
        "algorithm": "Ed25519",
        "public_key_bytes": 32,
        "signature_bytes": 64,
    },

    "bulk_cipher": {
        "algorithm": "AES-256-GCM",
        "authentication_tag_bytes": 16,
    },

    "computational_overhead_us": {
        "key_exchange": 27.41,
        "signing": 29.31,
        "verification": 82.65,
    },
}


# ---------------------------------------------------------------------------
# Post-quantum cryptographic profile
# ---------------------------------------------------------------------------

PQC_CRYPTO_PROFILE = {
    "profile": "PQC",

    "key_exchange": {
        "algorithm": "ML-KEM-768",
        "public_key_bytes": 1184,
        "ciphertext_bytes": 1088,
        "shared_secret_bytes": 32,
    },

    "authentication": {
        "algorithm": "ML-DSA-65",
        "public_key_bytes": 1952,
        "signature_bytes": 3309,
    },

    "bulk_cipher": {
        "algorithm": "AES-256-GCM",
        "authentication_tag_bytes": 16,
    },

    "computational_overhead_us": {
        "key_exchange": 224.15,
        "signing": 2249.55,
        "verification": 592.32,
    },
}


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

CRYPTO_PROFILES = {
    "CLASSICAL": CLASSICAL_CRYPTO_PROFILE,
    "PQC": PQC_CRYPTO_PROFILE,
}


# ---------------------------------------------------------------------------
# Profile access
# ---------------------------------------------------------------------------

def get_crypto_profile(profile_name):
    """
    Return a copy of the requested cryptographic profile.
    """

    if profile_name not in CRYPTO_PROFILES:
        raise ValueError(
            f"Unknown crypto profile: {profile_name}"
        )

    return CRYPTO_PROFILES[profile_name].copy()


# ---------------------------------------------------------------------------
# Network-overhead calculations
# ---------------------------------------------------------------------------

def calculate_session_setup_overhead(profile_name):
    """
    Calculate the modeled session-establishment message overhead.

    This represents the public-key and key-exchange ciphertext material
    that needs to cross the network.

    It does not represent every byte of a real protocol handshake.
    """

    profile = CRYPTO_PROFILES[profile_name]

    key_exchange = profile["key_exchange"]
    authentication = profile["authentication"]

    return (
        key_exchange["public_key_bytes"]
        + key_exchange["ciphertext_bytes"]
        + authentication["public_key_bytes"]
        + authentication["signature_bytes"]
    )


def get_application_data_overhead(profile_name):
    """
    Return the per-encrypted-data authentication-tag overhead.

    AES-256-GCM contributes the same authentication-tag size to both
    Classical and PQC profiles.
    """

    profile = CRYPTO_PROFILES[profile_name]

    return profile["bulk_cipher"]["authentication_tag_bytes"]
