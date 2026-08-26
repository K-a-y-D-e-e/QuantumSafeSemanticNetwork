import os
import time
import statistics

import oqs

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ITERATIONS = 100


def benchmark(operation, iterations=ITERATIONS):
    times = []

    for _ in range(iterations):
        start = time.perf_counter_ns()
        operation()
        end = time.perf_counter_ns()

        times.append((end - start) / 1_000)

    return {
        "average": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }


def print_result(name, result):
    print(f"{name:<30} "
          f"avg: {result['average']:>10.2f} µs | "
          f"median: {result['median']:>10.2f} µs | "
          f"min: {result['min']:>10.2f} µs | "
          f"max: {result['max']:>10.2f} µs")


print("=" * 80)
print("POST-QUANTUM CRYPTOGRAPHY PERFORMANCE BENCHMARK")
print("=" * 80)

print(f"\nIterations per operation: {ITERATIONS}")


# ============================================================
# ML-KEM
# ============================================================

print("\n" + "=" * 80)
print("ML-KEM-768")
print("=" * 80)

kem_algorithm = "ML-KEM-768"

with oqs.KeyEncapsulation(kem_algorithm) as kem:

    keygen_result = benchmark(
        lambda: kem.generate_keypair()
    )

    public_key = kem.generate_keypair()

    ciphertext, shared_secret = kem.encap_secret(
        public_key
    )

    encap_result = benchmark(
        lambda: kem.encap_secret(public_key)
    )

    decap_result = benchmark(
        lambda: kem.decap_secret(ciphertext)
    )

    print_result("Key Generation", keygen_result)
    print_result("Encapsulation", encap_result)
    print_result("Decapsulation", decap_result)

    print("\nML-KEM sizes:")
    print(f"Public key:     {len(public_key):>6} bytes")
    print(f"Ciphertext:     {len(ciphertext):>6} bytes")
    print(f"Shared secret:  {len(shared_secret):>6} bytes")


# ============================================================
# ML-DSA
# ============================================================

print("\n" + "=" * 80)
print("ML-DSA-65")
print("=" * 80)

dsa_algorithm = "ML-DSA-65"

message = b"FLOW_ID=17; PRIORITY=HIGH; LATENCY=1ms"

with oqs.Signature(dsa_algorithm) as signer:

    dsa_keygen_result = benchmark(
        lambda: signer.generate_keypair()
    )

    dsa_public_key = signer.generate_keypair()

    dsa_secret_key = signer.export_secret_key()

    signature = signer.sign(message)

    sign_result = benchmark(
        lambda: signer.sign(message)
    )

    verify_result = benchmark(
        lambda: signer.verify(
            message,
            signature,
            dsa_public_key
        )
    )

    print_result("Key Generation", dsa_keygen_result)
    print_result("Signing", sign_result)
    print_result("Verification", verify_result)

    print("\nML-DSA sizes:")
    print(f"Public key:     {len(dsa_public_key):>6} bytes")
    print(f"Secret key:     {len(dsa_secret_key):>6} bytes")
    print(f"Signature:      {len(signature):>6} bytes")


# ============================================================
# AES-256-GCM
# ============================================================

print("\n" + "=" * 80)
print("AES-256-GCM")
print("=" * 80)

aes_key = os.urandom(32)
aes = AESGCM(aes_key)

payload = os.urandom(1024)

nonce = os.urandom(12)

encrypted = aes.encrypt(
    nonce,
    payload,
    None
)

aes_encrypt_result = benchmark(
    lambda: aes.encrypt(
        os.urandom(12),
        payload,
        None
    )
)

aes_decrypt_result = benchmark(
    lambda: aes.decrypt(
        nonce,
        encrypted,
        None
    )
)

print_result("Encryption (1 KB)", aes_encrypt_result)
print_result("Decryption (1 KB)", aes_decrypt_result)

print("\nAES-GCM payload:")
print(f"Plaintext:      {len(payload):>6} bytes")
print(f"Ciphertext:     {len(encrypted):>6} bytes")
print(f"Overhead:       {len(encrypted) - len(payload):>6} bytes")


print("\n" + "=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)