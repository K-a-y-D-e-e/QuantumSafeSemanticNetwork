import time
import statistics
import oqs

from cryptography.hazmat.primitives.asymmetric import (
    x25519,
    ed25519
)


WARMUP = 100
ITERATIONS = 1000


def benchmark(operation):
    # Warm-up
    for _ in range(WARMUP):
        operation()

    times = []

    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        operation()
        end = time.perf_counter_ns()

        times.append((end - start) / 1_000)

    times.sort()

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times),
        "p95": times[int(0.95 * len(times))],
        "p99": times[int(0.99 * len(times))],
        "min": times[0],
        "max": times[-1],
    }


def print_result(name, result):
    print(
        f"{name:<25}"
        f"Mean: {result['mean']:>10.2f} µs | "
        f"Median: {result['median']:>10.2f} µs | "
        f"P95: {result['p95']:>10.2f} µs | "
        f"P99: {result['p99']:>10.2f} µs"
    )


print("=" * 110)
print("CLASSICAL vs POST-QUANTUM CRYPTOGRAPHY BENCHMARK")
print("=" * 110)

print(f"\nWarm-up iterations : {WARMUP}")
print(f"Measured iterations: {ITERATIONS}")


# ============================================================
# X25519
# ============================================================

print("\n" + "=" * 110)
print("X25519 — CLASSICAL KEY ESTABLISHMENT")
print("=" * 110)

x_public_a, x_private_a = (
    x25519.X25519PrivateKey.generate(),
    None
)

x_private_a = x_public_a
x_public_a = x_private_a.public_key()

x_private_b = x25519.X25519PrivateKey.generate()
x_public_b = x_private_b.public_key()

x25519_keygen = benchmark(
    lambda: x25519.X25519PrivateKey.generate()
)

x25519_exchange = benchmark(
    lambda: x_private_a.exchange(x_public_b)
)

print_result("Key Generation", x25519_keygen)
print_result("Shared Secret", x25519_exchange)

print("\nSizes:")
print(
    f"Public key:    "
    f"{len(x_public_a.public_bytes_raw())} bytes"
)
print("Shared secret: 32 bytes")


# ============================================================
# ML-KEM-768
# ============================================================

print("\n" + "=" * 110)
print("ML-KEM-768 — POST-QUANTUM KEY ESTABLISHMENT")
print("=" * 110)

kem_algorithm = "ML-KEM-768"

with oqs.KeyEncapsulation(kem_algorithm) as kem:

    kem_public_key = kem.generate_keypair()

    kem_ciphertext, kem_secret = (
        kem.encap_secret(kem_public_key)
    )

    kem_keygen = benchmark(
        lambda: kem.generate_keypair()
    )

    kem_encap = benchmark(
        lambda: kem.encap_secret(kem_public_key)
    )

    kem_decap = benchmark(
        lambda: kem.decap_secret(kem_ciphertext)
    )

    print_result("Key Generation", kem_keygen)
    print_result("Encapsulation", kem_encap)
    print_result("Decapsulation", kem_decap)

    print("\nSizes:")
    print(f"Public key:    {len(kem_public_key)} bytes")
    print(f"Ciphertext:    {len(kem_ciphertext)} bytes")
    print(f"Shared secret: {len(kem_secret)} bytes")


# ============================================================
# ED25519
# ============================================================

print("\n" + "=" * 110)
print("Ed25519 — CLASSICAL DIGITAL SIGNATURE")
print("=" * 110)

message = (
    b"FLOW_ID=17; "
    b"PRIORITY=HIGH; "
    b"LATENCY=1ms"
)

ed_private = ed25519.Ed25519PrivateKey.generate()
ed_public = ed_private.public_key()

ed_signature = ed_private.sign(message)

ed25519_keygen = benchmark(
    lambda: ed25519.Ed25519PrivateKey.generate()
)

ed25519_sign = benchmark(
    lambda: ed_private.sign(message)
)

ed25519_verify = benchmark(
    lambda: ed_public.verify(
        ed_signature,
        message
    )
)

print_result("Key Generation", ed25519_keygen)
print_result("Signing", ed25519_sign)
print_result("Verification", ed25519_verify)

print("\nSizes:")
print(
    f"Public key: {len(ed_public.public_bytes_raw())} bytes"
)
print(f"Signature:  {len(ed_signature)} bytes")


# ============================================================
# ML-DSA-65
# ============================================================

print("\n" + "=" * 110)
print("ML-DSA-65 — POST-QUANTUM DIGITAL SIGNATURE")
print("=" * 110)

dsa_algorithm = "ML-DSA-65"

with oqs.Signature(dsa_algorithm) as dsa:

    dsa_public_key = dsa.generate_keypair()

    dsa_secret_key = dsa.export_secret_key()

    dsa_signature = dsa.sign(message)

    dsa_keygen = benchmark(
        lambda: dsa.generate_keypair()
    )

    dsa_sign = benchmark(
        lambda: dsa.sign(message)
    )

    dsa_verify = benchmark(
        lambda: dsa.verify(
            message,
            dsa_signature,
            dsa_public_key
        )
    )

    print_result("Key Generation", dsa_keygen)
    print_result("Signing", dsa_sign)
    print_result("Verification", dsa_verify)

    print("\nSizes:")
    print(f"Public key: {len(dsa_public_key)} bytes")
    print(f"Secret key: {len(dsa_secret_key)} bytes")
    print(f"Signature:  {len(dsa_signature)} bytes")


print("\n" + "=" * 110)
print("BENCHMARK COMPLETE")
print("=" * 110)