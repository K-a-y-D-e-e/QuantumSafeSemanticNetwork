from classical_crypto import (
    X25519KeyExchange,
    Ed25519Signature
)


print("========================================")
print(" CLASSICAL CRYPTOGRAPHY TEST")
print("========================================")


# ============================================================
# X25519
# ============================================================

print("\n[1] X25519 KEY EXCHANGE")

alice_public, alice_private = (
    X25519KeyExchange.generate_keypair()
)

bob_public, bob_private = (
    X25519KeyExchange.generate_keypair()
)

alice_secret = X25519KeyExchange.derive_shared_secret(
    alice_private,
    bob_public
)

bob_secret = X25519KeyExchange.derive_shared_secret(
    bob_private,
    alice_public
)

print(
    f"Shared secrets match: "
    f"{alice_secret == bob_secret}"
)

print(f"Public key size: {len(alice_public.public_bytes_raw())} bytes")
print(f"Shared secret size: {len(alice_secret)} bytes")


# ============================================================
# Ed25519
# ============================================================

print("\n[2] ED25519 DIGITAL SIGNATURE")

public_key, private_key = (
    Ed25519Signature.generate_keypair()
)

message = (
    b"FLOW_ID=17; "
    b"PRIORITY=HIGH; "
    b"LATENCY=1ms"
)

signature = Ed25519Signature.sign(
    private_key,
    message
)

valid = Ed25519Signature.verify(
    public_key,
    message,
    signature
)

print(f"Message size: {len(message)} bytes")
print(f"Public key size: {len(public_key.public_bytes_raw())} bytes")
print(f"Signature size: {len(signature)} bytes")
print(f"Signature valid: {valid}")


# ============================================================
# TAMPERING TEST
# ============================================================

print("\n[3] TAMPERING TEST")

tampered_message = (
    b"FLOW_ID=17; "
    b"PRIORITY=LOW; "
    b"LATENCY=1ms"
)

tampered_valid = Ed25519Signature.verify(
    public_key,
    tampered_message,
    signature
)

print(f"Tampered message valid: {tampered_valid}")