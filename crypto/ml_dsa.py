import oqs


class MLDSA:
    def __init__(self, algorithm="ML-DSA-65"):
        self.algorithm = algorithm

    def generate_keypair(self):
        signer = oqs.Signature(self.algorithm)

        public_key = signer.generate_keypair()
        secret_key = signer.export_secret_key()

        signer.free()

        return public_key, secret_key

    def sign(self, secret_key, message):
        signer = oqs.Signature(
            self.algorithm,
            secret_key
        )

        signature = signer.sign(message)

        signer.free()

        return signature

    def verify(self, public_key, message, signature):
        verifier = oqs.Signature(self.algorithm)

        valid = verifier.verify(
            message,
            signature,
            public_key
        )

        verifier.free()

        return valid