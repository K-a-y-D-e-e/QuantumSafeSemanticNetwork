from ml_dsa import MLDSA


dsa = MLDSA("ML-DSA-65")

print("=== ML-DSA KEY GENERATION ===")

public_key, secret_key = dsa.generate_keypair()

print(f"Public key size: {len(public_key)} bytes")
print(f"Secret key size: {len(secret_key)} bytes")


# Original control message
message = b"FLOW_ID=17; PRIORITY=HIGH; LATENCY=1ms"

print("\n=== SIGNING ===")

signature = dsa.sign(
    secret_key,
    message
)

print(f"Message size: {len(message)} bytes")
print(f"Signature size: {len(signature)} bytes")


# Verify original message
print("\n=== VERIFY ORIGINAL MESSAGE ===")

valid = dsa.verify(
    public_key,
    message,
    signature
)

print(f"Signature valid: {valid}")


# Modify the message after signing
tampered_message = b"FLOW_ID=17; PRIORITY=LOW; LATENCY=1ms"

print("\n=== VERIFY TAMPERED MESSAGE ===")

tampered_valid = dsa.verify(
    public_key,
    tampered_message,
    signature
)

print(f"Signature valid: {tampered_valid}")