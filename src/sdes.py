from typing import List

# Permutation tables (1-based indexing)
P10    = [3, 5, 2, 7, 4, 10, 1, 9, 8, 6]
P8     = [6, 3, 7, 4, 8, 5, 10, 9]
IP     = [2, 6, 3, 1, 4, 8, 5, 7]
IP_INV = [4, 1, 3, 5, 7, 2, 8, 6]
EP     = [4, 1, 2, 3, 2, 3, 4, 1]
P4     = [2, 4, 3, 1]

S0 = [
    [1, 0, 3, 2],
    [3, 2, 1, 0],
    [0, 2, 1, 3],
    [3, 1, 3, 2]
]

S1 = [
    [0, 1, 2, 3],
    [2, 0, 1, 3],
    [3, 0, 1, 0],
    [2, 1, 0, 3]
]


def permute(bitstr: str, table: List[int]) -> str:
    """
    Apply a permutation table (1-based indices) to the input bitstring.
    """
    return "".join(bitstr[i - 1] for i in table)


def left_shift(bits: str, n: int) -> str:
    """
    Perform a circular left shift by n positions on the bit string.
    """
    return bits[n:] + bits[:n]


def key_schedule(key: str) -> (str, str):
    """
    Given a 10-bit key, generate the two 8-bit subkeys (K1, K2).
    """
    # Apply P10 permutation
    permuted = permute(key, P10)
    left, right = permuted[:5], permuted[5:]

    # Left shift each half by 1
    left1 = left_shift(left, 1)
    right1 = left_shift(right, 1)
    # First subkey K1 via P8
    K1 = permute(left1 + right1, P8)

    # Left shift each half of the result by 2
    left2 = left_shift(left1, 2)
    right2 = left_shift(right1, 2)
    # Second subkey K2 via P8
    K2 = permute(left2 + right2, P8)

    return K1, K2


def sbox_lookup(bits: str, sbox: List[List[int]]) -> str:
    """
    bits: 4-bit string.
    sbox: 4×4 table.
    Returns a 2-bit string output of the S-box lookup.
    """
    row = int(bits[0] + bits[3], 2)
    col = int(bits[1:3], 2)
    val = sbox[row][col]
    return format(val, "02b")


def fk(left: str, right: str, subkey: str) -> (str, str):
    """
    One round of the Feistel function.
    left: 4-bit string
    right: 4-bit string
    subkey: 8-bit string
    Returns (new_left, new_right) after applying fk.
    """
    # Expand and permute right half
    expanded = permute(right, EP)  # 8 bits
    # XOR with subkey
    xored = format(int(expanded, 2) ^ int(subkey, 2), "08b")
    # Split into two 4-bit halves
    left4, right4 = xored[:4], xored[4:]
    # Apply S-boxes
    s0_out = sbox_lookup(left4, S0)
    s1_out = sbox_lookup(right4, S1)
    # Combine and apply P4
    p4_out = permute(s0_out + s1_out, P4)
    # XOR with the left half
    new_left = format(int(left, 2) ^ int(p4_out, 2), "04b")
    # The right half remains the same before the swap
    return new_left, right


def sdes_encrypt(plaintext: str, key: str) -> str:
    """
    Encrypt an 8-bit plaintext with a 10-bit key using S-DES.
    Returns the 8-bit ciphertext.
    """
    if len(plaintext) != 8 or len(key) != 10:
        raise ValueError("plaintext must be 8 bits and key must be 10 bits.")

    # Generate subkeys K1 and K2
    K1, K2 = key_schedule(key)

    # Initial Permutation (IP)
    ip_bits = permute(plaintext, IP)
    left, right = ip_bits[:4], ip_bits[4:]

    # Round 1
    f1_left, _ = fk(left, right, K1)
    # Swap halves
    left2, right2 = right, f1_left

    # Round 2
    f2_left, _ = fk(left2, right2, K2)
    preoutput = f2_left + right2

    # Inverse Initial Permutation (IP_INV)
    ciphertext = permute(preoutput, IP_INV)
    return ciphertext
