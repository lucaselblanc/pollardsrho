/******************************************************************************************************
 * This file is part of the Pollard's Rho distribution: (https://github.com/lucaselblanc/pollardsrho) *
 * Copyright (c) 2024, 2025 Lucas Leblanc.                                                            *
 * Distributed under the MIT software license, see the accompanying.                                  *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.                               *
 ******************************************************************************************************/

/*****************************************
 * Pollard's Rho Algorithm for SECP256K1 *
 * Written by Lucas Leblanc              *
******************************************/

/* --- AINDA EM TESTES --- */

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ uint64_t P_CONST[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

__constant__ uint64_t N_CONST[4] = {
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

__constant__ uint64_t GX_CONST[4] = {
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL
};

__constant__ uint64_t GY_CONST[4] = {
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL
};

__constant__ uint64_t R_MOD_P[4] = {
    0x00000001000003D1ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

__constant__ uint64_t R2_MOD_P[4] = {
    0x000007A2000E90A1ULL,
    0x0000000100000000ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

__constant__ uint64_t R2_MOD_N[4] = {
    0x896CF21467D7D140ULL,
    0x741496C20E7CF878ULL,
    0xE697F5E45BCD07C6ULL,
    0x9D671CD581C69BC5ULL
};

__constant__ uint64_t MU_P = 0xD2253531ULL;
__constant__ uint64_t MU_N = 0x5588B13FULL;

__constant__ uint64_t ZERO[4]  = {0ULL, 0ULL, 0ULL, 0ULL};
__constant__ uint64_t ONE[4]   = {1ULL, 0ULL, 0ULL, 0ULL};
__constant__ uint64_t TWO[4]   = {2ULL, 0ULL, 0ULL, 0ULL};
__constant__ uint64_t THREE[4] = {3ULL, 0ULL, 0ULL, 0ULL};
__constant__ uint64_t SEVEN[4] = {7ULL, 0ULL, 0ULL, 0ULL};

__constant__ uint64_t ONE_MONT[4] = {0x00000001000003D1ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL};

__constant__ uint64_t SEVEN_MONT[4] = {0x0000000700001A97ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL};

typedef struct {
    uint64_t x[4];
    uint64_t y[4];
    int infinity;
} ECPoint;

typedef struct {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
    int infinity;
} ECPointJacobian;

__device__ int bignum_cmp(const uint64_t *a, const uint64_t *b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ int bignum_is_zero(const uint64_t *a) {
    for (int i = 0; i < 4; i++) {
        if (a[i] != 0ULL) return 0;
    }
    return 1;
}

__device__ int bignum_is_odd(const uint64_t *a) {
    return a[0] & 1ULL;
}

__device__ void bignum_copy(uint64_t *dst, const uint64_t *src) {
    for (int i = 0; i < 4; i++) {
        dst[i] = src[i];
    }
}

__device__ void bignum_zero(uint64_t *a) {
    for (int i = 0; i < 4; i++) {
        a[i] = 0ULL;
    }
}

__device__ int bignum_is_one(const uint64_t *a) {
    if (a[0] != 1ULL) return 0;
    for (int i = 1; i < 4; i++) {
        if (a[i] != 0ULL) return 0;
    }
    return 1;
}

__device__ void bignum_set_ui(uint64_t *a, uint64_t val) {
    bignum_zero(a);
    a[0] = val;
}

__device__ uint64_t bignum_add_carry(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t carry = 0ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t temp = a[i] + b[i] + carry;
        carry = (temp < a[i]) || (carry && temp == a[i]);
        result[i] = temp;
    }
    return carry;
}

__device__ uint64_t bignum_sub_borrow(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t borrow = 0ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t temp = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow);
        result[i] = temp;
    }
    return borrow;
}

__device__ void bignum_shr1(uint64_t *result, const uint64_t *a) {
    uint64_t carry = 0ULL;
    for (int i = 3; i >= 0; i--) {
        uint64_t next = (a[i] & 1ULL) << 63;
        result[i] = (a[i] >> 1) | carry;
        carry = next;
    }
}

__device__ void bignum_mul_full(uint64_t *result_high, uint64_t *result_low,
                                const uint64_t *a, const uint64_t *b) {
    uint64_t temp_low[8] = {0};
    uint64_t temp_high[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0ULL;
        for (int j = 0; j < 4; j++) {
            uint64_t a_hi = a[i] >> 32, a_lo = a[i] & 0xFFFFFFFFULL;
            uint64_t b_hi = b[j] >> 32, b_lo = b[j] & 0xFFFFFFFFULL;

            uint64_t p0 = a_lo * b_lo;
            uint64_t p1 = a_lo * b_hi;
            uint64_t p2 = a_hi * b_lo;
            uint64_t p3 = a_hi * b_hi;

            uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
            uint64_t high = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);

            uint64_t low = (mid << 32) | (p0 & 0xFFFFFFFFULL);

            uint64_t sum_low = temp_low[i + j] + low + carry;
            carry = (sum_low < temp_low[i + j]) || (sum_low < low);
            temp_low[i + j] = sum_low;

            temp_high[i + j] += high + carry;
        }
        temp_high[i + 4] += carry;
    }

    for (int i = 0; i < 4; i++) {
        result_low[i] = temp_low[i];
        result_high[i] = temp_high[i];
    }
}

__device__ void montgomery_reduce_p(uint64_t *result,
                                    const uint64_t *input_high,
                                    const uint64_t *input_low) {
    uint64_t temp[8];

    for (int i = 0; i < 4; i++) {
        temp[i]     = input_low[i];
        temp[i + 4] = input_high[i];
    }

    for (int i = 0; i < 4; i++) {
        uint64_t ui = temp[i] * (uint64_t)MU_P;

        uint64_t carry = 0ULL;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod =
                (unsigned __int128)ui * (unsigned __int128)P_CONST[j] +
                (unsigned __int128)temp[i + j] +
                (unsigned __int128)carry;

            temp[i + j] = (uint64_t)prod;
            carry       = (uint64_t)(prod >> 64);
        }

        for (int j = i + 4; j < 8; j++) {
            unsigned __int128 tmp = (unsigned __int128)temp[j] +
                                    (unsigned __int128)carry;

            temp[j] = (uint64_t)tmp;
            carry   = (uint64_t)(tmp >> 64);
        }
    }

    for (int i = 0; i < 4; i++) {
        result[i] = temp[i + 4];
    }

    if (bignum_cmp(result, (const uint64_t*)P_CONST) >= 0) {
        bignum_sub_borrow(result, result, (const uint64_t*)P_CONST);
    }
}

__device__ void to_montgomery_p(uint64_t *result, const uint64_t *a) {
    uint64_t high[4], low[4];
    bignum_mul_full(high, low, a, (uint64_t*)R2_MOD_P);
    montgomery_reduce_p(result, high, low);
}

__device__ void from_montgomery_p(uint64_t *result, const uint64_t *a) {
    uint64_t zero[4] = {0, 0, 0, 0};
    bignum_zero(zero);
    montgomery_reduce_p(result, zero, a);
}

__device__ void mod_add_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t temp[4];
    uint64_t carry = bignum_add_carry(temp, a, b);

    if (carry || bignum_cmp(temp, (uint64_t*)P_CONST) >= 0) {
        bignum_sub_borrow(result, temp, (uint64_t*)P_CONST);
    } else {
        bignum_copy(result, temp);
    }
}

__device__ void mod_sub_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t temp[4];
    uint64_t borrow = bignum_sub_borrow(temp, a, b);

    if (borrow) {
        bignum_add_carry(result, temp, (uint64_t*)P_CONST);
    } else {
        bignum_copy(result, temp);
    }
}

__device__ void mod_mul_mont_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t high[4], low[4];
    bignum_mul_full(high, low, a, b);
    montgomery_reduce_p(result, high, low);
}

__device__ void mod_sqr_mont_p(uint64_t out[4], const uint64_t in[4]) {
    // out = in^2 mod P
    mod_mul_mont_p(out, in, in);
}

// ----------- ALMOST INV -----------

struct __uint256_t {
    __uint64_t limb[4];
};

struct __uint512_t {
    __uint64_t limb[8];
};

struct __uint1024_t {
    __uint64_t limb[16];
};

struct uint256_t_sign {
    __uint256_t magnitude;
    int sign;
};

struct uint1024_t_sign {
    __uint1024_t magnitude;
    int sign;
};

struct frac1024_t {
    __uint1024_t num;
    unsigned int exp;
    int sign;
};

__constant__ __uint512_t mu = {
     {
       0x1000003d1ULL,
       0x0ULL,
       0x1ULL,
       0x0ULL,
       0x0ULL,
       0x0ULL,
       0x0ULL,
       0x0ULL
     }
};

__device__ __uint256_t add_256(const __uint256_t &a, const __uint256_t &b) {
    __uint256_t res;
    __uint64_t carry = 0;

    for (int i = 0; i < 4; i++) {
        __uint64_t tmp = a.limb[i] + b.limb[i] + carry;
        carry = (tmp < a.limb[i] || (carry && tmp == a.limb[i])) ? 1 : 0;
        res.limb[i] = tmp;
    }

    return res;
}

__device__ __uint256_t sub_256(const __uint256_t &a, const __uint256_t &b) {
    __uint256_t res;
    __uint64_t borrow = 0;

    for (int i = 0; i < 4; i++) {
        __uint64_t tmp = a.limb[i] - b.limb[i] - borrow;
        borrow = (a.limb[i] < b.limb[i] || (borrow && a.limb[i] == b.limb[i])) ? 1 : 0;
        res.limb[i] = tmp;
    }

    return res;
}

__device__ bool borrow_256(const __uint256_t &a, const __uint256_t &b) {
    __uint64_t borrow = 0;

    for (int i = 0; i < 4; i++) {
        __uint64_t tmp = a.limb[i] - b.limb[i] - borrow;
        borrow = (a.limb[i] < b.limb[i] || (borrow && a.limb[i] == b.limb[i])) ? 1 : 0;
    }

    return borrow;
}

__device__ __uint512_t add_512(const __uint512_t &a, const __uint512_t &b) {
    __uint512_t res;
    __uint64_t carry = 0;

    for (int i = 0; i < 8; i++) {
        __uint64_t tmp = a.limb[i] + b.limb[i] + carry;
        carry = (tmp < a.limb[i] || (carry && tmp == a.limb[i])) ? 1 : 0;
        res.limb[i] = tmp;
    }

    return res;
}

__device__ __uint512_t sub_512(const __uint512_t &a, const __uint512_t &b) {
    __uint512_t res;
    __uint64_t borrow = 0;

    for (int i = 0; i < 8; i++) {
        __uint64_t tmp = a.limb[i] - b.limb[i] - borrow;
        borrow = (a.limb[i] < b.limb[i] || (borrow && a.limb[i] == b.limb[i])) ? 1 : 0;
        res.limb[i] = tmp;
    }
    return res;
}

__device__ bool borrow_512(const __uint512_t &a, const __uint512_t &b) {
    __uint64_t borrow = 0;

    for (int i = 0; i < 8; i++) {
        __uint64_t tmp = a.limb[i] - b.limb[i] - borrow;
        borrow = (a.limb[i] < b.limb[i] || (borrow && a.limb[i] == b.limb[i])) ? 1 : 0;
    }

    return borrow;
}

__device__ __uint1024_t add_1024(const __uint1024_t &a, const __uint1024_t &b) {
    __uint1024_t res;
    __uint64_t carry = 0;

    for (int i = 0; i < 16; i++) {
        __uint64_t tmp = a.limb[i] + b.limb[i] + carry;
        carry = (tmp < a.limb[i] || (carry && tmp == a.limb[i])) ? 1 : 0;
        res.limb[i] = tmp;
    }

    return res;
}

__device__ __uint1024_t sub_1024(const __uint1024_t &a, const __uint1024_t &b) {
    __uint1024_t res;
    __uint64_t borrow = 0;

    for (int i = 0; i < 16; i++) {
        __uint64_t tmp = a.limb[i] - b.limb[i] - borrow;
        borrow = (a.limb[i] < b.limb[i] || (borrow && a.limb[i] == b.limb[i])) ? 1 : 0;
        res.limb[i] = tmp;
    }

    return res;
}

__device__ bool borrow_1024(const __uint1024_t &a, const __uint1024_t &b) {
    __uint64_t borrow = 0;

    for (int i = 0; i < 16; i++) {
        __uint64_t tmp = a.limb[i] - b.limb[i] - borrow;
        borrow = (a.limb[i] < b.limb[i] || (borrow && a.limb[i] == b.limb[i])) ? 1 : 0;
    }

    return borrow;
}

__device__ unsigned int bit_length_256(const __uint256_t &x) {
    for (int i = 3; i >= 0; i--) {
        if (x.limb[i] != 0) {
            return i * 64 + (64 - __clzll(x.limb[i]));
        }
    }
    return 0;
}

__device__ __uint256_t make_power_of_two_256(unsigned int t) {
    __uint256_t res = {};
    if (t == 0) return res;
    
    unsigned int limb_index = t / 64;
    unsigned int bit_index = t % 64;
    
    if (limb_index < 4) {
        res.limb[limb_index] = 1ULL << bit_index;
    }
    
    return res;
}

__device__ __uint256_t truncate(const __uint256_t &f, unsigned int t) {
    __uint256_t res = {};
    if (t == 0) return res;
    if (t >= 256) return f;

    unsigned int full_limbs = t / 64;
    unsigned int remaining_bits = t % 64;

    for (int i = 0; i < 4; i++) {
        if (i < (int)full_limbs) {
            res.limb[i] = f.limb[i];
        } else if (i == (int)full_limbs && remaining_bits > 0) {
            __uint64_t mask = (1ULL << remaining_bits) - 1;
            res.limb[i] = f.limb[i] & mask;
        } else {
            res.limb[i] = 0;
        }
    }

    unsigned int sign_bit_pos = t - 1;
    unsigned int sign_limb = sign_bit_pos / 64;
    unsigned int sign_bit = sign_bit_pos % 64;
    
    bool negative = false;
    if (sign_limb < 4) {
        negative = (res.limb[sign_limb] & (1ULL << sign_bit)) != 0;
    }

    if (negative) {
        __uint256_t pow2 = make_power_of_two_256(t);
        res = sub_256(res, pow2);
    }

    return res;
}

__device__ int sign(const __uint256_t &x, unsigned int t) {
    if (t == 0) return 1;

    unsigned int sign_bit_pos = t - 1;
    unsigned int sign_limb = sign_bit_pos / 64;
    unsigned int sign_bit = sign_bit_pos % 64;
    
    if (sign_limb >= 4) return 1;
    
    bool neg = (x.limb[sign_limb] & (1ULL << sign_bit)) != 0;
    return neg ? -1 : 1;
}

__device__ void normalize_sign_uint256(uint256_t_sign &x) {
    bool is_zero = true;
    for (int i = 0; i < 4; i++) {
        if (x.magnitude.limb[i] != 0) {
            is_zero = false;
            break;
        }
    }
    if (is_zero) x.sign = 1;
}

__device__ uint256_t_sign signed_add_256(const uint256_t_sign &a, const uint256_t_sign &b) {
    uint256_t_sign res;
    if (a.sign == b.sign) {
        res.magnitude = add_256(a.magnitude, b.magnitude);
        res.sign = a.sign;
    } else {
        bool borrow = borrow_256(a.magnitude, b.magnitude);
        if (!borrow) {
            res.magnitude = sub_256(a.magnitude, b.magnitude);
            res.sign = a.sign;
        } else {
            res.magnitude = sub_256(b.magnitude, a.magnitude);
            res.sign = b.sign;
        }
    }
    normalize_sign_uint256(res);
    return res;
}

__device__ uint256_t_sign signed_div2_floor_256(const uint256_t_sign &a_in) {
    uint256_t_sign r = a_in;
    
    bool is_neg = (r.sign < 0);
    bool is_odd = (r.magnitude.limb[0] & 1);
    
    if (is_neg && is_odd) {
        __uint256_t one = {{1,0,0,0}};
        r.magnitude = add_256(r.magnitude, one);
    }
    
    __uint64_t carry = 0;
    for (int i = 3; i >= 0; i--) {
        __uint64_t new_carry = r.magnitude.limb[i] & 1;
        r.magnitude.limb[i] = (r.magnitude.limb[i] >> 1) | (carry << 63);
        carry = new_carry;
    }
    
    bool is_zero = true;
    for (int i = 0; i < 4; i++) {
        if (r.magnitude.limb[i] != 0) {
            is_zero = false;
            break;
        }
    }
    if (is_zero) r.sign = 1;
    
    return r;
}

__device__ __uint1024_t lshift_1024(const __uint1024_t &x, unsigned int shift) {
    __uint1024_t res = {};
    if (shift >= 1024) return res;

    unsigned int limb_shift = shift / 64;
    unsigned int bit_shift = shift % 64;

    for (int i = 15; i >= 0; i--) {
        if (i - (int)limb_shift < 0) continue;
        res.limb[i] = x.limb[i - limb_shift] << bit_shift;
        if (i - (int)limb_shift - 1 >= 0 && bit_shift != 0)
            res.limb[i] |= x.limb[i - limb_shift - 1] >> (64 - bit_shift);
    }
    return res;
}

__device__ __uint1024_t rshift_1024(const __uint1024_t &x, unsigned int shift) {
    __uint1024_t res = {};
    if (shift >= 1024) return res;

    unsigned int limb_shift = shift / 64;
    unsigned int bit_shift = shift % 64;

    for (int i = 0; i < 16; i++) {
        if (i + (int)limb_shift >= 16) continue;
        res.limb[i] = x.limb[i + limb_shift] >> bit_shift;
        if (i + (int)limb_shift + 1 < 16 && bit_shift != 0)
            res.limb[i] |= x.limb[i + limb_shift + 1] << (64 - bit_shift);
    }
    return res;
}

__device__ __uint128_t mul_64_128(const __uint64_t &a, const __uint64_t &b) {
    return (__uint128_t)a * b;
}

__device__ __uint256_t mul_64_256(const __uint64_t &a, const __uint64_t &b) {
    __uint128_t prod = mul_64_128(a, b);
    __uint256_t result = {};
    result.limb[0] = (__uint64_t)prod;
    result.limb[1] = (__uint64_t)(prod >> 64);
    return result;
}

__device__ __uint256_t mul_128_256(const __uint128_t &a, const __uint128_t &b) {
    __uint64_t a0 = (__uint64_t)a;
    __uint64_t a1 = (__uint64_t)(a >> 64);
    __uint64_t b0 = (__uint64_t)b;
    __uint64_t b1 = (__uint64_t)(b >> 64);
    __uint128_t p00 = mul_64_128(a0, b0);
    __uint128_t p01 = mul_64_128(a0, b1);
    __uint128_t p10 = mul_64_128(a1, b0);
    __uint128_t p11 = mul_64_128(a1, b1);
    
    __uint256_t result = {};
    
    result.limb[0] = (__uint64_t)p00;
    
    __uint128_t middle_sum = p01 + p10 + (p00 >> 64);
    result.limb[1] = (__uint64_t)middle_sum;
    
    __uint128_t high_sum = p11 + (middle_sum >> 64);
    result.limb[2] = (__uint64_t)high_sum;
    result.limb[3] = (__uint64_t)(high_sum >> 64);
    
    return result;
}

__device__ __uint512_t mul_256_512(const __uint256_t &a, const __uint256_t &b) {
    __uint512_t res = {};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = mul_64_128(a.limb[i], b.limb[j]);

            __uint64_t carry = 0;
            int pos = i + j;

            if (pos < 8) {
                __uint64_t tmp_lo = res.limb[pos] + (__uint64_t)prod;
                carry = (tmp_lo < res.limb[pos]) ? 1 : 0;
                res.limb[pos] = tmp_lo;
            }

            pos++;
            if (pos < 8) {
                __uint64_t tmp_hi = res.limb[pos] + (__uint64_t)(prod >> 64) + carry;
                carry = (tmp_hi < res.limb[pos]) ? 1 : 0;
                res.limb[pos] = tmp_hi;
            }

            pos++;
            while (carry && pos < 8) {
                __uint64_t tmp = res.limb[pos] + carry;
                carry = (tmp < res.limb[pos]) ? 1 : 0;
                res.limb[pos] = tmp;
                pos++;
            }
        }
    }

    return res;
}

__device__ __uint1024_t mul_512_1024(const __uint512_t &a, const __uint512_t &b) {
    __uint1024_t res = {};

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            __uint128_t prod = mul_64_128(a.limb[i], b.limb[j]);

            __uint64_t carry = 0;
            int pos = i + j;

            if (pos < 16) {
                __uint64_t tmp_lo = res.limb[pos] + (__uint64_t)prod;
                carry = (tmp_lo < res.limb[pos]) ? 1 : 0;
                res.limb[pos] = tmp_lo;
            }

            pos++;
            if (pos < 16) {
                __uint64_t tmp_hi = res.limb[pos] + (__uint64_t)(prod >> 64) + carry;
                carry = (tmp_hi < res.limb[pos]) ? 1 : 0;
                res.limb[pos] = tmp_hi;
            }

            pos++;
            while (carry && pos < 16) {
                __uint64_t tmp = res.limb[pos] + carry;
                carry = (tmp < res.limb[pos]) ? 1 : 0;
                res.limb[pos] = tmp;
                pos++;
            }
        }
    }

    return res;
}

__device__ __uint512_t mul_256_512_512(const __uint512_t &a, const __uint256_t &b) {
    __uint512_t res = {};

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = mul_64_128(a.limb[i], b.limb[j]);

            __uint64_t carry = 0;
            int pos = i + j;

            if (pos < 8) {
                __uint64_t tmp_lo = res.limb[pos] + (__uint64_t)prod;
                carry = (tmp_lo < res.limb[pos]) ? 1 : 0;
                res.limb[pos] = tmp_lo;
            }

            pos++;
            if (pos < 8) {
                __uint64_t tmp_hi = res.limb[pos] + (__uint64_t)(prod >> 64) + carry;
                carry = (tmp_hi < res.limb[pos]) ? 1 : 0;
                res.limb[pos] = tmp_hi;
            }

            pos++;
            while (carry && pos < 8) {
                __uint64_t tmp = res.limb[pos] + carry;
                carry = (tmp < res.limb[pos]) ? 1 : 0;
                res.limb[pos] = tmp;
                pos++;
            }
        }
    }

    return res;
}

__device__ __uint256_t barrett_reduction(const __uint1024_t &x, const __uint256_t &p) {
    // q1 = x >> (k-1) onde k = 256, então shift = 255
    __uint1024_t q1 = rshift_1024(x, 255);
    
    // q2 = q1 * mu (multiplicação 1024 x 512 = 1536 bits, truncamento)
    __uint1024_t q2 = {};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8 && i + j < 16; j++) {
            __uint128_t prod = mul_64_128(q1.limb[i], mu.limb[j]);
            
            __uint64_t carry = 0;
            int pos = i + j;
            
            if (pos < 16) {
                __uint64_t tmp_lo = q2.limb[pos] + (__uint64_t)prod;
                carry = (tmp_lo < q2.limb[pos]) ? 1 : 0;
                q2.limb[pos] = tmp_lo;
            }
            
            pos++;
            if (pos < 16) {
                __uint64_t tmp_hi = q2.limb[pos] + (__uint64_t)(prod >> 64) + carry;
                carry = (tmp_hi < q2.limb[pos]) ? 1 : 0;
                q2.limb[pos] = tmp_hi;
            }
            
            pos++;
            while (carry && pos < 16) {
                __uint64_t tmp = q2.limb[pos] + carry;
                carry = (tmp < q2.limb[pos]) ? 1 : 0;
                q2.limb[pos] = tmp;
                pos++;
            }
        }
    }
    
    // q3 = q2 >> (k+1) onde k = 256, então shift = 257
    __uint512_t q3 = {};
    for (int i = 0; i < 8; i++) {
        int src_pos = i + 4; // shift de 257 = 4*64 + 1
        if (src_pos < 16) {
            q3.limb[i] = (q2.limb[src_pos] >> 1);
            if (src_pos + 1 < 16) {
                q3.limb[i] |= (q2.limb[src_pos + 1] << 63);
            }
        }
    }
    
    // r = x - q3 * p
    __uint512_t qp = mul_256_512_512(q3, p);
    __uint512_t x_low = {};
    for (int i = 0; i < 8; i++) {
        x_low.limb[i] = x.limb[i];
    }
    
    __uint512_t r = sub_512(x_low, qp);
    
    __uint512_t p_512 = {};
    for (int i = 0; i < 4; i++) {
        p_512.limb[i] = p.limb[i];
    }
    
    for (int iter = 0; iter < 2; iter++) {
        bool ge = !borrow_512(r, p_512);
        if (ge) {
            r = sub_512(r, p_512);
        }
    }
    
    __uint256_t result = {};
    for (int i = 0; i < 4; i++) {
        result.limb[i] = r.limb[i];
    }
    return result;
}

__device__ __uint256_t mulmod_256(const __uint256_t &a, const __uint256_t &b, const __uint256_t &mod) {
    __uint512_t prod = mul_256_512(a, b);
    __uint256_t result = barrett_reduction(prod, mod);

    return result;
}

__device__ void normalize_frac(frac1024_t &x) {
    bool is_zero = true;
    for (int i = 0; i < 16; i++) {
        if (x.num.limb[i] != 0) {
            is_zero = false;
            break;
        }
    }
    if (is_zero) {
        x.sign = 1;
        x.exp = 0;
    }
}

__device__ bool frac_has_overflow(const frac1024_t &x) {
    for (int i = 12; i < 16; i++) {
        if (x.num.limb[i] != 0) return true;
    }
    return false;
}

__device__ void frac_normalize_overflow(frac1024_t &x) {
    if (!frac_has_overflow(x)) {
        return;
    }
    
    int overflow_limbs = 0;
    for (int i = 15; i >= 4; i--) {
        if (x.num.limb[i] != 0) {
            overflow_limbs = i - 3;
            break;
        }
    }
    
    int shift_amount;
    if (overflow_limbs <= 1) {
        shift_amount = 64;
    } else if (overflow_limbs <= 2) {
        shift_amount = 128;
    } else {
        shift_amount = overflow_limbs * 64;
    }
    
    x.num = rshift_1024(x.num, shift_amount);
    x.exp += shift_amount;
    
    #ifdef DEBUG_OVERFLOW
    printf("DEBUG: Overflow normalizado - limbs extras: %d, shift: %d\n", 
           overflow_limbs, shift_amount);
    
    if (frac_has_overflow(x)) {
        printf("ERRO: Overflow persistente após normalização!\n");
    }
    #endif
}

__device__ __uint1024_t frac_align_num(const frac1024_t &b, unsigned int target_exp) {
    if (target_exp >= b.exp) {
        unsigned int shift = target_exp - b.exp;
        return rshift_1024(b.num, shift);
    } else {
        unsigned int shift = b.exp - target_exp;
        return lshift_1024(b.num, shift);
    }
}

__device__ void frac_add(frac1024_t &res, const frac1024_t &a, const frac1024_t &b) {
    if (a.exp >= b.exp) {
        __uint1024_t bnum_al = frac_align_num(b, a.exp);
        
        if (a.sign == b.sign) {
            res.num = add_1024(a.num, bnum_al);
            res.sign = a.sign;
        } else {
            if (!borrow_1024(a.num, bnum_al)) {
                res.num = sub_1024(a.num, bnum_al);
                res.sign = a.sign;
            } else {
                res.num = sub_1024(bnum_al, a.num);
                res.sign = b.sign;
            }
        }
        res.exp = a.exp;
    } else {
        __uint1024_t anum_al = frac_align_num(a, b.exp);
        
        if (a.sign == b.sign) {
            res.num = add_1024(anum_al, b.num);
            res.sign = a.sign;
        } else {
            if (!borrow_1024(anum_al, b.num)) {
                res.num = sub_1024(anum_al, b.num);
                res.sign = a.sign;
            } else {
                res.num = sub_1024(b.num, anum_al);
                res.sign = b.sign;
            }
        }
        res.exp = b.exp;
    }
    
    normalize_frac(res);
    frac_normalize_overflow(res);
}

__device__ void frac_div2(frac1024_t &x) {
    x.exp += 1;
}

__device__ __uint1024_t frac_to_scaled_int(const frac1024_t &frac, unsigned int scale_exp) {
    if (scale_exp >= frac.exp) {
        unsigned int shift = scale_exp - frac.exp;
        return lshift_1024(frac.num, shift);
    } else {
        unsigned int shift = frac.exp - scale_exp;
        return rshift_1024(frac.num, shift);
    }
}

__device__ __uint256_t adjust_sign(__uint256_t value, int sign, const __uint256_t &mod) {
    bool nonzero = false;
    for (int i = 0; i < 4; i++) {
        if (value.limb[i] != 0) {
            nonzero = true;
            break;
        }
    }

    if (sign < 0 && nonzero) {
        return sub_256(mod, value);
    }
    return value;
}

__device__ __uint256_t modexp_256(__uint256_t base, unsigned int exp, const __uint256_t &mod) {
    __uint256_t result = {{1, 0, 0, 0}};

    for (int i = 0; i < 32; i++) {
        if (exp & 1) {
            result = mulmod_256(result, base, mod);
        }
        exp >>= 1;
        if (exp == 0) break;
        base = mulmod_256(base, base, mod);
    }

    return result;
}

__device__ void divsteps2(
    unsigned int n,
    unsigned int t,
    int &delta,
    uint256_t_sign &f,
    uint256_t_sign &g,
    frac1024_t &u,
    frac1024_t &v,
    frac1024_t &q,
    frac1024_t &r
) {

    u.num = {}; u.num.limb[0] = 1; u.exp = 0; u.sign = 1;
    v.num = {}; v.exp = 0; v.sign = 1;
    q.num = {}; q.exp = 0; q.sign = 1;
    r.num = {}; r.num.limb[0] = 1; r.exp = 0; r.sign = 1;

    f.magnitude = truncate(f.magnitude, t);
    g.magnitude = truncate(g.magnitude, t);

    while (n > 0) {
        f.magnitude = truncate(f.magnitude, t);
        
        bool g_odd = (g.magnitude.limb[0] & 1);
        
        if (delta > 0 && g_odd) {
            delta = -delta;
            
            uint256_t_sign f_old = f, g_old = g;
            frac1024_t u_old = u, v_old = v, q_old = q, r_old = r;
            
            f = g_old;
            g = f_old;
            g.sign = -g.sign;
            
            u = q_old;
            v = r_old;
            q = u_old; q.sign = -q.sign;
            r = v_old; r.sign = -r.sign;
        }
        
        bool g0 = (g.magnitude.limb[0] & 1);
        delta = delta + 1;
        
        if (g0) {
            uint256_t_sign f_contrib = {f.magnitude, f.sign};
            g = signed_add_256(g, f_contrib);
            
            frac1024_t q_old = q;
            frac_add(q, q_old, u);
            frac1024_t r_old = r;
            frac_add(r, r_old, v);
        }
        
        g = signed_div2_floor_256(g);
        frac_div2(q);
        frac_div2(r);
        
        n--;
        t--;
        
        if (t > 0) {
            g.magnitude = truncate(g.magnitude, t);
        }
    }
}

__device__ unsigned int iterations(unsigned int d) {
    return (d < 46) ? ((49*d + 80) / 17) : ((49*d + 57) / 17);
}

__device__ __uint256_t recip2(const __uint256_t &f_in, const __uint256_t &g_in) {
    if ((f_in.limb[0] & 1) == 0) {
        __uint256_t zero = {};
        return zero;
    }

    unsigned int d_f = bit_length_256(f_in);
    unsigned int d_g = bit_length_256(g_in);
    unsigned int d = (d_f > d_g) ? d_f : d_g;

    unsigned int m = iterations(d);

    __uint256_t f_plus_1 = add_256(f_in, (__uint256_t){{1, 0, 0, 0}});
    uint256_t_sign f_plus_1_signed = {f_plus_1, 1};
    uint256_t_sign half_f_plus_1_signed = signed_div2_floor_256(f_plus_1_signed);
    
    __uint256_t precomp = modexp_256(
        half_f_plus_1_signed.magnitude,
        (m > 0 ? m - 1 : 0),
        f_in
    );

    uint256_t_sign f_s = {f_in, 1};
    uint256_t_sign g_s = {g_in, 1};
    int delta = 1;

    frac1024_t u, v, q, r;
    divsteps2(m, m + 1, delta, f_s, g_s, u, v, q, r);
    
    unsigned int shift_amount = (m > 0 ? m - 1 : 0);
    
    __uint1024_t v_scaled = frac_to_scaled_int(v, shift_amount);
    
    __uint256_t V_magnitude = {};
    for (int i = 0; i < 4; i++) {
        V_magnitude.limb[i] = v_scaled.limb[i];
    }
    
    int fm_sign = sign(f_s.magnitude, m + 1);
    int combined_sign = v.sign * fm_sign;
    
    __uint256_t V_mod = adjust_sign(V_magnitude, combined_sign, f_in);
    
    __uint256_t inv = mulmod_256(V_mod, precomp, f_in);
    return inv;
}

// ------------- FINAL -------------

__device__ void jacobian_init(ECPointJacobian *point) {
    bignum_zero(point->X);
    bignum_zero(point->Y);
    bignum_copy(point->Z, ONE_MONT);
    point->infinity = 0;
}

__device__ void jacobian_set_infinity(ECPointJacobian *point) {
    bignum_copy(point->X, ONE_MONT);
    bignum_copy(point->Y, ONE_MONT);
    bignum_zero(point->Z);
    point->infinity = 1;
}

__device__ int jacobian_is_infinity(const ECPointJacobian *point) {
    return point->infinity || bignum_is_zero(point->Z);
}

__device__ void affine_to_jacobian(ECPointJacobian *jac, const ECPoint *aff) {
    if (aff->infinity) {
        jacobian_set_infinity(jac);
        return;
    }
    
    bignum_copy(jac->X, aff->x);
    bignum_copy(jac->Y, aff->y);
    bignum_copy(jac->Z, ONE_MONT);
    jac->infinity = 0;
}

__device__ void jacobian_to_affine(ECPoint *aff, const ECPointJacobian *jac) {
    if (jacobian_is_infinity(jac)) {
        bignum_zero(aff->x);
        bignum_zero(aff->y);
        aff->infinity = 1;
        return;
    }

    uint64_t z_norm[4], z_inv[4], z_inv_sqr[4], z_inv_cube[4];

    from_montgomery_p(z_norm, jac->Z);

    //mod_inverse_p(z_inv, z_norm);
    mod_mul_mont_p(z_inv_sqr, z_inv, z_inv);
    mod_mul_mont_p(z_inv_cube, z_inv_sqr, z_inv);
    mod_mul_mont_p(aff->x, jac->X, z_inv_sqr);
    mod_mul_mont_p(aff->y, jac->Y, z_inv_cube);

    from_montgomery_p(aff->x, aff->x);
    from_montgomery_p(aff->y, aff->y);

    aff->infinity = 0;
}

__device__ void jacobian_double(ECPointJacobian *result, const ECPointJacobian *point) {
    if (jacobian_is_infinity(point) || bignum_is_zero(point->Y)) {
        jacobian_set_infinity(result);
        return;
    }

    uint64_t A[4], B[4], C[4], D[4], E[4];
    uint64_t X2[4];

    mod_sqr_mont_p(A, point->Y);
    mod_mul_mont_p(B, point->X, A);
    mod_add_p(B, B, B);
    mod_add_p(B, B, B);

    mod_sqr_mont_p(C, A);
    mod_add_p(C, C, C);
    mod_add_p(C, C, C);
    mod_add_p(C, C, C);

    mod_sqr_mont_p(X2, point->X);
    mod_add_p(D, X2, X2);
    mod_add_p(D, D, X2);

    mod_sqr_mont_p(result->X, D);
    mod_sub_p(result->X, result->X, B);
    mod_sub_p(result->X, result->X, B);

    mod_sub_p(E, B, result->X);
    mod_mul_mont_p(result->Y, D, E);
    mod_sub_p(result->Y, result->Y, C);

    mod_mul_mont_p(result->Z, point->Y, point->Z);
    mod_add_p(result->Z, result->Z, result->Z);

    result->infinity = 0;
}

__device__ void jacobian_add(ECPointJacobian *result, const ECPointJacobian *P, const ECPointJacobian *Q) {

    int P_infinity = jacobian_is_infinity(P);
    int Q_infinity = jacobian_is_infinity(Q);

    if (P_infinity) {
        bignum_copy(result->X, Q->X);
        bignum_copy(result->Y, Q->Y);
        bignum_copy(result->Z, Q->Z);
        result->infinity = Q->infinity;
        return;
    }

    if (Q_infinity) {
        bignum_copy(result->X, P->X);
        bignum_copy(result->Y, P->Y);
        bignum_copy(result->Z, P->Z);
        result->infinity = P->infinity;
        return;
    }

    uint64_t U1[4], U2[4], S1[4], S2[4], H[4], I[4], J[4], r[4], V[4];
    uint64_t Z1Z1[4], Z2Z2[4], Z1Z2[4], temp1[4], temp2[4];

    mod_sqr_mont_p(Z1Z1, P->Z);
    mod_sqr_mont_p(Z2Z2, Q->Z);
    mod_mul_mont_p(U1, P->X, Z2Z2);
    mod_mul_mont_p(U2, Q->X, Z1Z1);
    mod_mul_mont_p(temp1, Q->Z, Z2Z2);
    mod_mul_mont_p(S1, P->Y, temp1);
    mod_mul_mont_p(temp2, P->Z, Z1Z1);
    mod_mul_mont_p(S2, Q->Y, temp2);
    mod_sub_p(H, U2, U1);
    mod_sub_p(r, S2, S1);

    int is_H_zero = (bignum_cmp(H, ZERO) == 0);
    int is_r_zero = (bignum_cmp(r, ZERO) == 0);

    if (is_H_zero) {
        if (is_r_zero) {
            jacobian_double(result, P);
        } else {
            jacobian_set_infinity(result);
        }
        return;
    }

    mod_add_p(I, H, H);
    mod_sqr_mont_p(I, I);
    mod_mul_mont_p(J, H, I);
    mod_mul_mont_p(V, U1, I);
    mod_add_p(r, r, r);
    mod_sqr_mont_p(result->X, r);
    mod_sub_p(result->X, result->X, J);
    mod_sub_p(result->X, result->X, V);
    mod_sub_p(result->X, result->X, V);
    mod_sub_p(temp1, V, result->X);
    mod_mul_mont_p(result->Y, r, temp1);
    mod_mul_mont_p(temp2, S1, J);
    mod_add_p(temp2, temp2, temp2);
    mod_sub_p(result->Y, result->Y, temp2);
    mod_add_p(Z1Z2, P->Z, Q->Z);
    mod_sqr_mont_p(Z1Z2, Z1Z2);
    mod_sub_p(Z1Z2, Z1Z2, Z1Z1);
    mod_sub_p(Z1Z2, Z1Z2, Z2Z2);
    mod_mul_mont_p(result->Z, Z1Z2, H);

    result->infinity = 0;
}

__device__ void scalar_reduce_n(uint64_t *r, const uint64_t *k) {
    uint64_t t[4];
    uint64_t borrow = bignum_sub_borrow(t, k, (uint64_t*)N_CONST);

    if (borrow == 0) {
        bignum_copy(r, t);
    } else {
        bignum_copy(r, k);
    }
}

__device__ void jacobian_scalar_mult(ECPointJacobian *result, const uint64_t *scalar, const ECPointJacobian *point) {
    if (bignum_is_zero(scalar) || jacobian_is_infinity(point)) {
        jacobian_set_infinity(result);
        return;
    }

    ECPointJacobian R0, R1;
    jacobian_set_infinity(&R0);
    R1 = *point;

    uint64_t k[4];
    bignum_copy(k, scalar);

    scalar_reduce_n(k, k);

    int msb = 255;
    while (msb >= 0) {
        int word = 3 - (msb / 64);
        int bit  = 63 - (msb % 64);
        if ((k[word] >> bit) & 1ULL) break;
        msb--;
    }

    for (int i = msb; i >= 0; i--) {
        int word = 3 - (i / 64);
        int bit  = 63 - (i % 64);
        int kbit = (k[word] >> bit) & 1ULL;

        if (kbit == 0) {
            ECPointJacobian temp;
            jacobian_add(&temp, &R1, &R0);
            R1 = temp;
            jacobian_double(&R0, &R0);
        } else {
            ECPointJacobian temp;
            jacobian_add(&temp, &R0, &R1);
            R0 = temp;
            jacobian_double(&R1, &R1);
        }
    }

    *result = R0;
}

__device__ void point_from_montgomery(ECPoint *result, const ECPoint *point_mont) {
    if (point_mont->infinity) {
        result->infinity = 1;
        bignum_zero(result->x);
        bignum_zero(result->y);
        return;
    }
    
    from_montgomery_p(result->x, point_mont->x);
    from_montgomery_p(result->y, point_mont->y);
    result->infinity = 0;
}

__device__ void kernel_point_init(ECPoint *point) {
    bignum_zero(point->x);
    bignum_zero(point->y);
    point->infinity = 0;
}

__device__ void kernel_point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q) {
    ECPointJacobian P_jac, Q_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    affine_to_jacobian(&Q_jac, Q);    
    jacobian_add(&R_jac, &P_jac, &Q_jac);   
    jacobian_to_affine(R, &R_jac);
}

__device__ void kernel_point_double(ECPoint *R, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    jacobian_double(&R_jac, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

__device__ void kernel_scalar_mult(ECPoint *R, const uint64_t *k, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;

    affine_to_jacobian(&P_jac, P);
    jacobian_scalar_mult(&R_jac, k, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

__device__ int kernel_point_is_valid(const ECPoint *point) {
    if (point->infinity) return 1;

    uint64_t lhs[4], rhs[4];

    mod_sqr_mont_p(lhs, point->y);
    mod_sqr_mont_p(rhs, point->x);
    mod_mul_mont_p(rhs, rhs, point->x);
    mod_add_p(rhs, rhs, SEVEN_MONT);

    return (bignum_cmp(lhs, rhs) == 0);
}

__device__ void kernel_get_compressed_public_key(unsigned char *out, const ECPoint *public_key) {
    unsigned char prefix = (public_key->y[0] & 1ULL) ? 0x03 : 0x02;
    out[0] = prefix;

    for (int i = 0; i < 4; i++) {
        uint64_t word = public_key->x[3 - i];
        out[1 + i*8 + 0] = (word >> 56) & 0xFF;
        out[1 + i*8 + 1] = (word >> 48) & 0xFF;
        out[1 + i*8 + 2] = (word >> 40) & 0xFF;
        out[1 + i*8 + 3] = (word >> 32) & 0xFF;
        out[1 + i*8 + 4] = (word >> 24) & 0xFF;
        out[1 + i*8 + 5] = (word >> 16) & 0xFF;
        out[1 + i*8 + 6] = (word >> 8)  & 0xFF;
        out[1 + i*8 + 7] = word & 0xFF;
    }
}

__global__ void generate_public_key(unsigned char *out, const uint64_t *PRIV_KEY) {
    ECPoint pub;
    ECPoint G;
    ECPointJacobian G_jac, pub_jac;

    to_montgomery_p(G.x, GX_CONST);
    to_montgomery_p(G.y, GY_CONST);
    G.infinity = 0;

    affine_to_jacobian(&G_jac, &G);
    jacobian_scalar_mult(&pub_jac, PRIV_KEY, &G_jac);
    jacobian_to_affine(&pub, &pub_jac);

    kernel_get_compressed_public_key(out, &pub);
}

__global__ void point_init(ECPoint *point) {
    kernel_point_init(point);
}

__global__ void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q) {
    kernel_point_add(R, P, Q);
}

__global__ void point_double(ECPoint *R, const ECPoint *P) {
    kernel_point_double(R, P);
}

__global__ void scalar_mult(ECPoint *R, const uint64_t *k, const ECPoint *P) {
    kernel_scalar_mult(R, k, P);
}

__global__ void point_is_valid(int *result, const ECPoint *point) {
    *result = kernel_point_is_valid(point);
}

__global__ void get_compressed_public_key(unsigned char *out, const ECPoint *pub) {
    kernel_get_compressed_public_key(out, pub);
}

__global__ void test_mod_inverse(const __uint256_t* f, const __uint256_t* g, __uint256_t* result) {
    *result = recip2(*f, *g);
}

int main() {

    // f = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    __uint256_t f = {};
    f.limb[0] = 0xFFFFFFFFFFFFFFFFULL;
    f.limb[1] = 0xFFFFFFFFFFFFFFFFULL;
    f.limb[2] = 0xFFFFFFFFFFFFFFFFULL;
    f.limb[3] = 0xFFFFFFFEFFFFFC2FULL;

    // g = 0x33e7665705359f04f28b88cf897c603c9
    __uint256_t g = {};
    g.limb[0] = 0x05359f04f28b88cfULL;
    g.limb[1] = 0x33e76657897c603cULL;
    g.limb[2] = 0x0000000000000009ULL;
    g.limb[3] = 0x0000000000000000ULL;

    /* g ≡ 1 (mod f): */
    //Hex: 7FDB62ED2D6FA0874ABD664C95B7CEF2ED79CC82D13FF3AC8E9766AA21BEBEAE (bebeae kkk)
    //Dec: 57831354042695616917422878622316954017183908093256327737334808907053491207854

    __uint256_t result_host;
    __uint256_t *f_device, *g_device, *result_device;
    cudaMalloc(&f_device, sizeof(__uint256_t));
    cudaMalloc(&g_device, sizeof(__uint256_t));
    cudaMalloc(&result_device, sizeof(__uint256_t));

    cudaMemcpy(f_device, &f, sizeof(__uint256_t), cudaMemcpyHostToDevice);
    cudaMemcpy(g_device, &g, sizeof(__uint256_t), cudaMemcpyHostToDevice);

    test_mod_inverse<<<1,1>>>(f_device, g_device, result_device);

    cudaMemcpy(&result_host, result_device, sizeof(__uint256_t), cudaMemcpyDeviceToHost);

    printf("Resultado limb[3] (mais alto): %016llx\n", (unsigned long long)result_host.limb[3]);
    printf("Resultado limb[2]: %016llx\n", (unsigned long long)result_host.limb[2]);
    printf("Resultado limb[1]: %016llx\n", (unsigned long long)result_host.limb[1]);
    printf("Resultado limb[0] (mais baixo): %016llx\n", (unsigned long long)result_host.limb[0]);

    cudaFree(f_device);
    cudaFree(g_device);
    cudaFree(result_device);

    return 0;
}