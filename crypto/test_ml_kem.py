from ml_kem import MLKEM


kem = MLKEM("ML-KEM-768")

print("Generating Alice's key pair...")
public_key, secret_key = kem.generate_keypair()

print(f"Public key size: {len(public_key)} bytes")
print(f"Secret key size: {len(secret_key)} bytes")

print("\nBob encapsulating...")
ciphertext, bob_shared_secret = kem.encapsulate(public_key)

print(f"Ciphertext size: {len(ciphertext)} bytes")
print(f"Bob shared secret size: {len(bob_shared_secret)} bytes")

print("\nAlice decapsulating...")
alice_shared_secret = kem.decapsulate(
    secret_key,
    ciphertext
)

print(f"Alice shared secret size: {len(alice_shared_secret)} bytes")

print("\nShared secrets match:")
print(alice_shared_secret == bob_shared_secret)