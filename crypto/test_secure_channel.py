from ml_kem import MLKEM
from aes_gcm import AESGCMCipher


# --------------------------------------------------
# 1. ML-KEM session establishment
# --------------------------------------------------

kem = MLKEM("ML-KEM-768")

print("=== ML-KEM SESSION ESTABLISHMENT ===")

public_key, secret_key = kem.generate_keypair()

ciphertext, bob_shared_secret = kem.encapsulate(public_key)

alice_shared_secret = kem.decapsulate(
    secret_key,
    ciphertext
)

print(f"Shared secrets match: {alice_shared_secret == bob_shared_secret}")


# --------------------------------------------------
# 2. AES-GCM session
# --------------------------------------------------

print("\n=== AES-256-GCM SESSION ===")

aes = AESGCMCipher(alice_shared_secret)

message = b"Robot command: MOVE_FORWARD"

nonce, encrypted = aes.encrypt(message)

print(f"Original message: {message.decode()}")
print(f"Ciphertext size: {len(encrypted)} bytes")

decrypted = aes.decrypt(
    nonce,
    encrypted
)

print(f"Decrypted message: {decrypted.decode()}")