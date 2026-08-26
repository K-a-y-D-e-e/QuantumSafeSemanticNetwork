from cryptography.hazmat.primitives.asymmetric import x25519, ed25519


class X25519KeyExchange:
    @staticmethod
    def generate_keypair():
        private_key = x25519.X25519PrivateKey.generate()
        public_key = private_key.public_key()

        return public_key, private_key

    @staticmethod
    def derive_shared_secret(private_key, peer_public_key):
        return private_key.exchange(peer_public_key)


class Ed25519Signature:
    @staticmethod
    def generate_keypair():
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        return public_key, private_key

    @staticmethod
    def sign(private_key, message):
        return private_key.sign(message)

    @staticmethod
    def verify(public_key, message, signature):
        try:
            public_key.verify(signature, message)
            return True
        except Exception:
            return False