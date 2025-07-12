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




## S-AES Implementation

# 4-bit S-Box and inverse
SBOX = {
    0x0: 0x9, 0x1: 0x4, 0x2: 0xA, 0x3: 0xB,
    0x4: 0xD, 0x5: 0x1, 0x6: 0x8, 0x7: 0x5,
    0x8: 0x6, 0x9: 0x2, 0xA: 0x0, 0xB: 0x3,
    0xC: 0xC, 0xD: 0xE, 0xE: 0xF, 0xF: 0x7
}

SBOX_INV = {v: k for k, v in SBOX.items()}

# GF(2^4) multiplication
def mul(a, b):
    p = 0
    for _ in range(4):
        if b & 1:
            p ^= a
        hi_bit_set = a & 0x8
        a <<= 1
        if hi_bit_set:
            a ^= 0b10011  # x^4 + x + 1 irreducible polynomial
        b >>= 1
    return p & 0xF

# SubNib (4-bit substitution)
def sub_nib(b):
    return (SBOX[b >> 4] << 4) | SBOX[b & 0xF]

# ShiftRows (swap lower nibbles of 2 bytes)
def shift_rows(s):
    return ((s & 0xF000) | (s & 0x0F00) >> 8 |
            (s & 0x00F0) << 8 | (s & 0x000F))

# MixColumns
def mix_columns(s):
    s0 = (s >> 12) & 0xF
    s1 = (s >> 8) & 0xF
    s2 = (s >> 4) & 0xF
    s3 = s & 0xF
    t0 = mul(1, s0) ^ mul(4, s2)
    t1 = mul(1, s1) ^ mul(4, s3)
    t2 = mul(4, s0) ^ mul(1, s2)
    t3 = mul(4, s1) ^ mul(1, s3)
    return (t0 << 12) | (t1 << 8) | (t2 << 4) | t3

# Key expansion (produces 3 round keys from 16-bit key)
def key_expansion(key):
    RCON1, RCON2 = 0x80, 0x30
    w = [0] * 6
    w[0] = (key >> 8) & 0xFF
    w[1] = key & 0xFF
    w[2] = w[0] ^ RCON1 ^ sub_nib(w[1])
    w[3] = w[2] ^ w[1]
    w[4] = w[2] ^ RCON2 ^ sub_nib(w[3])
    w[5] = w[4] ^ w[3]
    return [(w[0] << 8) | w[1], (w[2] << 8) | w[3], (w[4] << 8) | w[5]]

# Encrypt 16-bit plaintext using 16-bit key
def s_aes_encrypt(plaintext : str, key : str):
    plaintext = int(plaintext, 2)
    key = int(key, 2)
    keys = key_expansion(key)
    state = plaintext ^ keys[0]
    state = sub_nib(state >> 8) << 8 | sub_nib(state & 0xFF)
    state = shift_rows(state)
    state = mix_columns(state)
    state ^= keys[1]
    state = sub_nib(state >> 8) << 8 | sub_nib(state & 0xFF)
    state = shift_rows(state)
    ciphertext = state ^ keys[2]
    return format(ciphertext, "016b")  # Return as 16-bit binary string
