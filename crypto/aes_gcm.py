import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class AESGCMCipher:
    def __init__(self, key):
        if len(key) != 32:
            raise ValueError("AES-256 requires a 32-byte key")

        self.aes = AESGCM(key)

    def encrypt(self, plaintext, associated_data=None):
        nonce = os.urandom(12)

        ciphertext = self.aes.encrypt(
            nonce,
            plaintext,
            associated_data
        )

        return nonce, ciphertext

    def decrypt(self, nonce, ciphertext, associated_data=None):
        return self.aes.decrypt(
            nonce,
            ciphertext,
            associated_data
        )