from ml_kem import MLKEM
from ml_dsa import MLDSA
from aes_gcm import AESGCMCipher


class SecureChannel:
    def __init__(
        self,
        kem_algorithm="ML-KEM-768",
        dsa_algorithm="ML-DSA-65"
    ):
        self.kem = MLKEM(kem_algorithm)
        self.dsa = MLDSA(dsa_algorithm)

        self.aes = None

        self.public_key = None
        self.secret_key = None

        self.dsa_public_key = None
        self.dsa_secret_key = None

    def initialize(self):
        self.public_key, self.secret_key = (
            self.kem.generate_keypair()
        )

        self.dsa_public_key, self.dsa_secret_key = (
            self.dsa.generate_keypair()
        )

    def establish_session(self, peer_public_key):
        ciphertext, shared_secret = (
            self.kem.encapsulate(peer_public_key)
        )

        self.aes = AESGCMCipher(shared_secret)

        return ciphertext, shared_secret

    def receive_session(self, ciphertext):
        shared_secret = self.kem.decapsulate(
            self.secret_key,
            ciphertext
        )

        self.aes = AESGCMCipher(shared_secret)

        return shared_secret

    def encrypt(self, plaintext, associated_data=None):
        if self.aes is None:
            raise RuntimeError("Secure session not established")

        return self.aes.encrypt(
            plaintext,
            associated_data
        )

    def decrypt(self, nonce, ciphertext, associated_data=None):
        if self.aes is None:
            raise RuntimeError("Secure session not established")

        return self.aes.decrypt(
            nonce,
            ciphertext,
            associated_data
        )

    def sign_control_message(self, message):
        return self.dsa.sign(
            self.dsa_secret_key,
            message
        )

    def verify_control_message(
        self,
        message,
        signature,
        public_key
    ):
        return self.dsa.verify(
            public_key,
            message,
            signature
        )