import oqs


class MLKEM:
    def __init__(self, algorithm="ML-KEM-768"):
        self.algorithm = algorithm

    def generate_keypair(self):
        kem = oqs.KeyEncapsulation(self.algorithm)
        public_key = kem.generate_keypair()
        secret_key = kem.export_secret_key()

        kem.free()

        return public_key, secret_key

    def encapsulate(self, public_key):
        kem = oqs.KeyEncapsulation(self.algorithm)

        ciphertext, shared_secret = kem.encap_secret(public_key)

        kem.free()

        return ciphertext, shared_secret

    def decapsulate(self, secret_key, ciphertext):
        kem = oqs.KeyEncapsulation(
            self.algorithm,
            secret_key
        )

        shared_secret = kem.decap_secret(ciphertext)

        kem.free()

        return shared_secret