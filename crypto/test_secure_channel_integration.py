from secure_channel import SecureChannel


print("========================================")
print(" QUANTUM-SAFE SECURE CHANNEL TEST")
print("========================================")


# --------------------------------------------------
# NODE A
# --------------------------------------------------

print("\n[1] Initializing Node A")

node_a = SecureChannel()

node_a.initialize()

print("Node A ML-KEM key pair generated")
print("Node A ML-DSA key pair generated")


# --------------------------------------------------
# NODE B
# --------------------------------------------------

print("\n[2] Initializing Node B")

node_b = SecureChannel()

node_b.initialize()

print("Node B ML-KEM key pair generated")
print("Node B ML-DSA key pair generated")


# --------------------------------------------------
# ML-KEM SESSION ESTABLISHMENT
# --------------------------------------------------

print("\n[3] Establishing secure session")

ciphertext, node_b_secret = node_b.establish_session(
    node_a.public_key
)

node_a_secret = node_a.receive_session(
    ciphertext
)

print(
    f"Shared secrets match: "
    f"{node_a_secret == node_b_secret}"
)

# --------------------------------------------------
# AES-GCM ENCRYPTION
# --------------------------------------------------

print("\n[4] Encrypting application data")

message = (
    b"FLOW_ID=17; "
    b"PRIORITY=HIGH; "
    b"LATENCY=1ms"
)

nonce, encrypted = node_a.encrypt(message)

print(f"Original message: {message}")
print(f"Encrypted size: {len(encrypted)} bytes")


# --------------------------------------------------
# AES-GCM DECRYPTION
# --------------------------------------------------

print("\n[5] Decrypting application data")

decrypted = node_b.decrypt(
    nonce,
    encrypted
)

print(f"Decrypted message: {decrypted}")


# --------------------------------------------------
# ML-DSA CONTROL MESSAGE
# --------------------------------------------------

print("\n[6] Signing control message")

control_message = (
    b"ESTABLISH_FLOW; "
    b"FLOW_ID=17; "
    b"PRIORITY=HIGH"
)

signature = node_a.sign_control_message(
    control_message
)

print(f"Signature size: {len(signature)} bytes")


# --------------------------------------------------
# ML-DSA VERIFICATION
# --------------------------------------------------

print("\n[7] Verifying control message")

valid = node_b.verify_control_message(
    control_message,
    signature,
    node_a.dsa_public_key
)

print(f"Control message valid: {valid}")


# --------------------------------------------------
# TAMPERING TEST
# --------------------------------------------------

print("\n[8] Testing tampered control message")

tampered_message = (
    b"ESTABLISH_FLOW; "
    b"FLOW_ID=17; "
    b"PRIORITY=LOW"
)

tampered_valid = node_b.verify_control_message(
    tampered_message,
    signature,
    node_a.dsa_public_key
)

print(
    f"Tampered message valid: "
    f"{tampered_valid}"
)