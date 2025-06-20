import pytest
from src.sdes import sdes_encrypt, key_schedule


def test_key_schedule_length():
    # Given a valid 10-bit key, key_schedule should return two 8-bit subkeys.
    key = "1010000010"
    K1, K2 = key_schedule(key)
    assert len(K1) == 8
    assert len(K2) == 8


def test_sdes_encrypt_known_vector():
    # Correct S-DES output for key="1010000010" and plaintext="11010111" is "10101000"
    key = "1010000010"
    plaintext = "11010111"
    expected_cipher = "10101000"
    result = sdes_encrypt(plaintext, key)
    assert result == expected_cipher
