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

#define MAX_W 16
#define MAX_PRECOMP (1 << (MAX_W-1))

#include "secp256k1.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <thread>
#include <chrono>

#ifdef __CUDA_ARCH__
__device__ __constant__ uint64_t P_CONST[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
__device__ __constant__ uint64_t N_CONST[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
__device__ __constant__ uint64_t GX_CONST[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
__device__ __constant__ uint64_t GY_CONST[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };
__device__ __constant__ uint64_t R2_MOD_P[4] = { 0x000007A2000E90A1, 0x0000000000000001, 0x0ULL, 0x0ULL };
__device__ __constant__ uint64_t ZERO_MONT[4] = { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL };
__device__ __constant__ uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
__device__ __constant__ uint64_t SEVEN_MONT[4] = {0x700001AB7ULL, 0x0ULL, 0x0ULL, 0x0ULL};
__device__ __constant__ uint64_t P_CONST_MINUS_2[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
__device__ __constant__ uint64_t LAMBDA_N[4] = { 0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL, 0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL };
__device__ __constant__ uint64_t BETA_P[4] = { 0x7AE96A2B657C0710ULL, 0x6E64479EAC3434E9ULL, 0x9CF0497512F58995ULL, 0xB315ECECBB640683ULL };
__device__ __constant__ uint64_t MINUS_B1[4] = { 0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL, 0x0000000000000000ULL, 0x0000000000000000ULL };
__device__ __constant__ uint64_t MINUS_B2[4] = { 0xD765CDA83DB1562CULL, 0x8A280AC50774346DULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
__device__ __constant__ uint64_t G1[4] = { 0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL, 0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL };
__device__ __constant__ uint64_t G2[4] = { 0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL, 0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL };
__device__ __constant__ uint64_t MU_P = 0xD838091DD2253531ULL;
__device__ ECPointJacobian precomp_g[MAX_PRECOMP];
__device__ ECPointJacobian precomp_g_phi[MAX_PRECOMP];
__device__ int window_size = 4;
#else
constexpr uint64_t P_CONST[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
constexpr uint64_t N_CONST[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
constexpr uint64_t GX_CONST[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
constexpr uint64_t GY_CONST[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };
constexpr uint64_t R2_MOD_P[4] = { 0x000007A2000E90A1, 0x0000000000000001, 0x0ULL, 0x0ULL };
constexpr uint64_t ZERO_MONT[4] = { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL };
constexpr uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
constexpr uint64_t SEVEN_MONT[4] = {0x700001AB7ULL, 0x0ULL, 0x0ULL, 0x0ULL};
constexpr uint64_t P_CONST_MINUS_2[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
constexpr uint64_t LAMBDA_N[4] = { 0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL, 0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL };
constexpr uint64_t BETA_P[4] = { 0x7AE96A2B657C0710ULL, 0x6E64479EAC3434E9ULL, 0x9CF0497512F58995ULL, 0xB315ECECBB640683ULL };
constexpr uint64_t MINUS_B1[4] = { 0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL, 0x0000000000000000ULL, 0x0000000000000000ULL };
constexpr uint64_t MINUS_B2[4] = { 0xD765CDA83DB1562CULL, 0x8A280AC50774346DULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
constexpr uint64_t G1[4] = { 0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL, 0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL };
constexpr uint64_t G2[4] = { 0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL, 0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL };
constexpr uint64_t MU_P = 0xD838091DD2253531ULL;
ECPointJacobian precomp_g[MAX_PRECOMP];
ECPointJacobian precomp_g_phi[MAX_PRECOMP];
int window_size = 4;
#endif

#ifdef __CUDA_ARCH__
struct uint128_t {
    uint64_t lo;
    uint64_t hi;

    __host__ __device__
    uint128_t(uint64_t x = 0) : lo(x), hi(0) {}

    __host__ __device__
    uint128_t operator*(const uint128_t& other) const {
        uint128_t res;
        res.lo = lo * other.lo;
        res.hi = __umul64hi(lo, other.lo);
        return res;
    }

    __host__ __device__
    uint128_t operator*(uint64_t other) const {
        return *this * uint128_t(other);
    }

    __host__ __device__
    uint128_t& operator*=(const uint128_t& other) {
        *this = *this * other;
        return *this;
    }

    __host__ __device__
    uint128_t& operator*=(uint64_t other) {
        *this = *this * other;
        return *this;
    }

    __host__ __device__
    uint128_t operator+(const uint128_t& other) const {
        uint128_t res;
        res.lo = lo + other.lo;
        res.hi = hi + other.hi + (res.lo < lo ? 1 : 0);
        return res;
    }

    __host__ __device__
    uint128_t operator+(uint64_t other) const {
        return *this + uint128_t(other);
    }

    __host__ __device__
    uint128_t& operator+=(const uint128_t& other) {
        uint64_t old_lo = lo;
        lo += other.lo;
        hi += other.hi + (lo < old_lo ? 1 : 0);
        return *this;
    }

    __host__ __device__
    uint128_t& operator+=(uint64_t other) {
        *this += uint128_t(other);
        return *this;
    }

    __host__ __device__
    uint128_t operator-(const uint128_t& other) const {
        uint128_t res;
        res.lo = lo - other.lo;
        res.hi = hi - other.hi - (lo < other.lo ? 1 : 0);
        return res;
    }

    __host__ __device__
    uint128_t operator-(uint64_t other) const {
        return *this - uint128_t(other);
    }

    __host__ __device__
    uint128_t& operator-=(const uint128_t& other) {
        uint64_t old_lo = lo;
        lo -= other.lo;
        hi -= other.hi + (old_lo < other.lo ? 1 : 0);
        return *this;
    }

    __host__ __device__
    uint128_t& operator-=(uint64_t other) {
        *this -= uint128_t(other);
        return *this;
    }

    __host__ __device__
    uint128_t operator>>(const unsigned int n) const {
        uint128_t res;
        if (n == 0) return *this;
        else if (n < 64) {
            res.lo = (lo >> n) | (hi << (64 - n));
            res.hi = hi >> n;
        } else if (n < 128) {
            res.lo = hi >> (n - 64);
            res.hi = 0;
        } else {
            res.lo = 0;
            res.hi = 0;
        }
        return res;
    }

    __host__ __device__
    uint128_t& operator>>=(const unsigned int n) {
        *this = *this >> n;
        return *this;
    }

    __host__ __device__
    uint128_t operator<<(const unsigned int n) const {
        uint128_t res;
        if (n == 0) return *this;
        else if (n < 64) {
            res.hi = (hi << n) | (lo >> (64 - n));
            res.lo = lo << n;
        } else if (n < 128) {
            res.hi = lo << (n - 64);
            res.lo = 0;
        } else {
            res.lo = 0;
            res.hi = 0;
        }
        return res;
    }

    __host__ __device__
    uint128_t& operator<<=(const unsigned int n) {
        *this = *this << n;
        return *this;
    }

    __host__ __device__
    operator uint64_t() const {
        return lo;
    }

    __host__ __device__
    bool operator<(const uint128_t& other) const {
        return (hi < other.hi) || (hi == other.hi && lo < other.lo);
    }

    __host__ __device__
    bool operator<=(const uint128_t& other) const {
        return (hi < other.hi) || (hi == other.hi && lo <= other.lo);
    }

    __host__ __device__
    bool operator==(const uint128_t& other) const {
        return hi == other.hi && lo == other.lo;
    }

    __host__ __device__
    bool operator!=(const uint128_t& other) const {
        return !(*this == other);
    }

    __host__ __device__
    bool operator<(uint64_t other) const { return *this < uint128_t(other); }

    __host__ __device__
    bool operator<=(uint64_t other) const { return *this <= uint128_t(other); }

    __host__ __device__
    bool operator==(uint64_t other) const { return *this == uint128_t(other); }

    __host__ __device__
    bool operator!=(uint64_t other) const { return *this != uint128_t(other); }
};
#else
using uint128_t = unsigned __int128;
#endif

__host__ __device__ void montgomery_reduce_p(uint64_t *result, const uint64_t *input_high, const uint64_t *input_low) {
    uint64_t temp[8];
    for (int i = 0; i < 4; i++) {
        temp[i]     = input_low[i];
        temp[i + 4] = input_high[i];
    }

    for (int i = 0; i < 4; i++) {
        uint64_t ui = (uint64_t)((uint128_t)temp[i] * (uint128_t)MU_P);
        uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)ui * (uint128_t)P_CONST[j]
                                   + (uint128_t)temp[i + j] + carry;
            temp[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        uint128_t s = (uint128_t)temp[i + 4] + carry;
        temp[i + 4] = (uint64_t)s;
        carry = s >> 64;
        for (int j = i + 5; j < 8; ++j) {
            uint128_t sum = (uint128_t)temp[j] + carry;
            temp[j] = (uint64_t)sum;
            carry = sum >> 64;
        }
    }

    for (int i = 0; i < 4; i++) result[i] = temp[i + 4];

    uint64_t diff[4];
    uint128_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint128_t sub = (uint128_t)result[i] - (uint128_t)P_CONST[i] - borrow;
        diff[i] = (uint64_t)sub;
        borrow = (sub >> 127) & 1;
    }

    if (borrow == 0) {
        for (int i = 0; i < 4; i++) result[i] = diff[i];
    }
}

__host__ __device__ void to_montgomery_p(uint64_t *result, const uint64_t *a) {
    uint64_t a_local[4];
    for (int i = 0; i < 4; i++) a_local[i] = a[i];

    uint64_t temp[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint128_t carry = 0;

        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)a_local[i] * (uint128_t)R2_MOD_P[j]
                                   + (uint128_t)temp[i + j]
                                   + carry;
            temp[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }

        uint128_t sum = (uint128_t)temp[i + 4] + carry;
        temp[i + 4] = (uint64_t)sum;
        carry = sum >> 64;

        for (int k = i + 5; k < 8; k++) {
            uint128_t s = (uint128_t)temp[k] + carry;
            temp[k] = (uint64_t)s;
            carry = s >> 64;
        }
    }

    uint64_t low[4], high[4];
    for (int i = 0; i < 4; i++) {
        low[i] = temp[i];
        high[i] = temp[i + 4];
    }

    montgomery_reduce_p(result, high, low);
}

__host__ __device__ void from_montgomery_p(uint64_t *result, const uint64_t *a) {
    uint64_t zero[4] = {0, 0, 0, 0};
    montgomery_reduce_p(result, zero, a);
}

__host__ __device__ void mod_add_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t temp[4];
    uint128_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint128_t s = (uint128_t)a[i] + (uint128_t)b[i] + carry;
        temp[i] = (uint64_t)s;
        carry = s >> 64;
    }

    bool ge = (bool)carry;
    if (!ge) {
        for (int i = 3; i >= 0; --i) {
            if (temp[i] > P_CONST[i]) { ge = true; break; }
            if (temp[i] < P_CONST[i]) { ge = false; break; }
        }
    }

    if (ge) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            uint128_t sub = (uint128_t)temp[i] - (uint128_t)P_CONST[i] - borrow;
            result[i] = (uint64_t)sub;
            borrow = ((uint128_t)temp[i] < (uint128_t)P_CONST[i] + borrow) ? 1 : 0;
        }
    } else {
        for (int i = 0; i < 4; ++i) result[i] = temp[i];
    }
}

__host__ __device__ void mod_sub_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t temp[4];
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint128_t sub = (uint128_t)a[i] - (uint128_t)b[i] - borrow;
        temp[i] = (uint64_t)sub;
        borrow = ((uint128_t)a[i] < (uint128_t)b[i] + borrow) ? 1 : 0;
    }

    if (borrow) {
        uint128_t carry = 0;
        for (int i = 0; i < 4; ++i) {
            uint128_t s = (uint128_t)temp[i] + (uint128_t)P_CONST[i] + carry;
            result[i] = (uint64_t)s;
            carry = s >> 64;
        }
    } else {
        for (int i = 0; i < 4; ++i) result[i] = temp[i];
    }
}

__host__ __device__ void mod_mul_mont_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t high[4], low[4];

    uint64_t temp[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0ULL;
        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)a[i] * (uint128_t)b[j] + (uint128_t)temp[i + j] + (uint128_t)carry;
            temp[i + j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        temp[i + 4] = carry;
    }

    for (int i = 0; i < 4; i++) {
        low[i] = temp[i];
        high[i] = temp[i + 4];
    }

    montgomery_reduce_p(result, high, low);
}

__host__ __device__ void mod_sqr_mont_p(uint64_t *out, const uint64_t *in) {
    mod_mul_mont_p(out, in, in);
}

__host__ __device__ void scalar_reduce_n(uint64_t *r, const uint64_t *k) {
    bool ge = false;
    for (int i = 3; i >= 0; i--) {
        if (k[i] > N_CONST[i]) { ge = true; break; }
        if (k[i] < N_CONST[i]) { ge = false; break; }
    }

    if (ge) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t temp = k[i] - N_CONST[i] - borrow;
            borrow = (k[i] < N_CONST[i] + borrow) ? 1 : 0;
            r[i] = temp;
        }
    } else {
        for (int i = 0; i < 4; i++) {
            r[i] = k[i];
        }
    }
}

__host__ __device__ void mod_exp_mont_p(uint64_t *res, const uint64_t *base, const uint64_t *exp) {
    uint64_t one[4] = {0};
    one[0] = 1ULL;
    to_montgomery_p(res, one);

    uint64_t acc[4];
    for (int i = 0; i < 4; i++) acc[i] = base[i];

    for (int word = 3; word >= 0; word--) {
        for (int bit = 63; bit >= 0; bit--) {
            mod_sqr_mont_p(res, res);

            if ((exp[word] >> bit) & 1ULL) {
                mod_mul_mont_p(res, res, acc);
            }
        }
    }
}

__host__ __device__ void sqrt_mod_p(uint64_t y[4], const uint64_t v[4]) {
    uint64_t exp[4];

    exp[0] = 0x0FFFFFFFFFFFFFFF;
    exp[1] = 0xFFFFFFFFFFFFFFFF;
    exp[2] = 0xFFFFFFFFFFFFFFFF;
    exp[3] = 0x3FFFFFFF0000000C;

    mod_exp_mont_p(y, v, exp);
}

__host__ __device__ void jacobian_init(ECPointJacobian *point) {
    for (int i = 0; i < 4; i++) {
        point->X[i] = 0;
        point->Y[i] = 0;
        point->Z[i] = ONE_MONT[i];
    }
    point->infinity = 0;
}

__host__ __device__ void jacobian_set_infinity(ECPointJacobian *point) {
    for (int i = 0; i < 4; i++) {
        point->X[i] = ONE_MONT[i];
        point->Y[i] = ONE_MONT[i];
        point->Z[i] = 0;
    }
    point->infinity = 1;
}

__host__ __device__ int jacobian_is_infinity(const ECPointJacobian *point) {
    uint64_t z_zero = 0;
    for (int i = 0; i < 4; i++) {
        z_zero |= point->Z[i];
    }
    return point->infinity || (z_zero == 0);
}

//X/Z ONLY
__host__ __device__ void jacobian_to_affine(ECPoint *aff, const ECPointJacobian *jac) {
    if (jacobian_is_infinity(jac)) {
        for (int i = 0; i < 4; i++) aff->x[i] = aff->y[i] = 0;
        aff->infinity = 1;
        return;
    }

    uint64_t Z_inv[4], Z_inv2[4];

    mod_exp_mont_p(Z_inv, jac->Z, P_CONST_MINUS_2);
    mod_sqr_mont_p(Z_inv2, Z_inv);
    mod_mul_mont_p(aff->x, jac->X, Z_inv2);
    from_montgomery_p(aff->x, aff->x);

    uint64_t X3[4], aX[4], rhs[4];
    mod_sqr_mont_p(X3, aff->x);
    mod_mul_mont_p(X3, X3, aff->x);
    mod_mul_mont_p(aX, aff->x, ZERO_MONT);
    mod_add_p(rhs, X3, aX);
    mod_add_p(rhs, rhs, SEVEN_MONT);

    sqrt_mod_p(aff->y, rhs);

    aff->infinity = 0;
}

//X/Z ONLY
__host__ __device__ void jacobian_double(ECPointJacobian *result, const ECPointJacobian *point) {
    if (jacobian_is_infinity(point)) {
        jacobian_set_infinity(result);
        return;
    }

    uint64_t XX[4], ZZ[4], w[4], B[4];

    mod_sqr_mont_p(XX, point->X);
    mod_sqr_mont_p(ZZ, point->Z);
    mod_add_p(w, XX, XX);
    mod_add_p(w, w, XX);
    mod_mul_mont_p(B, point->X, ZZ);
    mod_sqr_mont_p(result->X, w);
    mod_add_p(B, B, B);
    mod_sub_p(result->X, result->X, B);
    mod_add_p(result->Z, point->X, point->Z);
    mod_sqr_mont_p(result->Z, result->Z);
    mod_sub_p(result->Z, result->Z, XX);
    mod_sub_p(result->Z, result->Z, ZZ);

    result->infinity = 0;
}

//X/Z ONLY
__host__ __device__ void jacobian_add(ECPointJacobian *result, const ECPointJacobian *P, const ECPointJacobian *Q) {
    int P_infinity = jacobian_is_infinity(P);
    int Q_infinity = jacobian_is_infinity(Q);

    if (P_infinity) {
        for (int i = 0; i < 4; i++) {
            result->X[i] = Q->X[i];
            result->Z[i] = Q->Z[i];
        }
        result->infinity = Q->infinity;
        return;
    }
    if (Q_infinity) {
        for (int i = 0; i < 4; i++) {
            result->X[i] = P->X[i];
            result->Z[i] = P->Z[i];
        }
        result->infinity = P->infinity;
        return;
    }

    uint64_t U1[4], U2[4], H[4], I[4], J[4], V[4], Z1Z2[4];
    uint64_t Z1Z1[4], Z2Z2[4];
    mod_sqr_mont_p(Z1Z1, P->Z);
    mod_sqr_mont_p(Z2Z2, Q->Z);
    mod_mul_mont_p(U1, P->X, Z2Z2);
    mod_mul_mont_p(U2, Q->X, Z1Z1);
    mod_sub_p(H, U2, U1);

    uint64_t H_zero = 0;
    for (int i = 0; i < 4; i++) H_zero |= H[i];
    if (H_zero == 0) {
        jacobian_double(result, P);
        return;
    }

    mod_add_p(I, H, H);
    mod_sqr_mont_p(I, I);
    mod_mul_mont_p(J, H, I);
    mod_mul_mont_p(V, U1, I);
    mod_add_p(H, V, V);
    mod_add_p(H, H, J);
    mod_sub_p(result->X, I, H);
    mod_add_p(Z1Z2, P->Z, Q->Z);
    mod_sqr_mont_p(Z1Z2, Z1Z2);
    mod_sub_p(Z1Z2, Z1Z2, Z1Z1);
    mod_sub_p(Z1Z2, Z1Z2, Z2Z2);
    mod_mul_mont_p(result->Z, Z1Z2, H);

    result->infinity = 0;
}

__host__ __device__ void endomorphism_map(ECPointJacobian *R, const ECPointJacobian *P) {
    mod_mul_mont_p(R->X, P->X, BETA_P);
    for (int i = 0; i < 4; i++) {
        R->Y[i] = P->Y[i];
        R->Z[i] = P->Z[i];
    }
    R->infinity = P->infinity;
}

__host__ __device__ void init_precomp_g() {
    int w = window_size;
    int d_normal = (256 + w - 1) / w;
    int d_phi    = (128 + w - 1) / w;
    int tableSize = (1 << w) - 1;

    uint64_t GX_mont[4];
    to_montgomery_p(GX_mont, GX_CONST);

    ECPointJacobian G, GE;
    for (int i = 0; i < 4; i++) {
        G.X[i] = GX_mont[i];
        G.Z[i] = ONE_MONT[i];
        GE.X[i] = GX_mont[i];
        GE.Z[i] = ONE_MONT[i];
    }

    G.infinity  = 0;
    GE.infinity = 0;

    endomorphism_map(&GE, &GE);

    ECPointJacobian P_j[MAX_W];
    ECPointJacobian P_je[MAX_W];

    P_j[0]  = G;
    P_je[0] = GE;

    for (int j = 1; j < w; j++) {
        P_j[j]  = P_j[j-1];
        P_je[j] = P_je[j-1];

        for (int i = 0; i < d_normal; i++) {
            jacobian_double(&P_j[j], &P_j[j]);
        }
        for (int i = 0; i < d_phi; i++) {
            jacobian_double(&P_je[j], &P_je[j]);
        }
    }

    for (int i = 1; i <= tableSize; i++) {
        jacobian_set_infinity(&precomp_g[i-1]);
        for (int j = 0; j < w; j++) {
            if ((i >> j) & 1) {
                ECPointJacobian tmp;
                jacobian_add(&tmp, &precomp_g[i-1], &P_j[j]);
                precomp_g[i-1] = tmp;
            }
        }

        jacobian_set_infinity(&precomp_g_phi[i-1]);
        for (int j = 0; j < w; j++) {
            if ((i >> j) & 1) {
                ECPointJacobian tmp;
                jacobian_add(&tmp, &precomp_g_phi[i-1], &P_je[j]);
                precomp_g_phi[i-1] = tmp;
            }
        }
    }
}

__host__ __device__ void jacobian_scalar_mult(ECPointJacobian *result, const uint64_t *scalar) {
    uint64_t k[4];
    scalar_reduce_n(k, scalar);

    int w = window_size;
    int d = (128 + w - 1) / w;

    jacobian_set_infinity(result);

    for (int col = d-1; col >= 0; col--) {
        if (col != d-1) {
            for (int i = 0; i < w; i++) {
                jacobian_double(result, result);
            }
        }

        int idx = 0;
        for (int row = 0; row < w; row++) {
            int bit_index = row*d + col;
            int limb = bit_index / 64;
            int shift = bit_index % 64;
            uint64_t bit = (k[limb] >> shift) & 1;
            idx |= (bit << row);
        }

        if (idx != 0) {
            ECPointJacobian tmp;
            jacobian_add(&tmp, result, &precomp_g[idx-1]);
            *result = tmp;
        }
    }
}

__host__ __device__ void jacobian_scalar_mult_phi(ECPointJacobian *result, const uint64_t *scalar) {
    uint64_t k[4];
    scalar_reduce_n(k, scalar);

    int w = window_size;
    int d = (128 + w - 1) / w;

    jacobian_set_infinity(result);

    for (int col = d-1; col >= 0; col--) {
        if (col != d-1) {
            for (int i = 0; i < w; i++) {
                jacobian_double(result, result);
            }
        }

        int idx = 0;
        for (int row = 0; row < w; row++) {
            int bit_index = row*d + col;
            int limb = bit_index / 64;
            int shift = bit_index % 64;
            uint64_t bit = (k[limb] >> shift) & 1;
            idx |= (bit << row);
        }

        if (idx != 0) {
            ECPointJacobian tmp;
            jacobian_add(&tmp, result, &precomp_g_phi[idx-1]);
            *result = tmp;
        }
    }
}

__host__ __device__ void scalar_mul_shift_var(uint64_t r[4], const uint64_t a[4], const uint64_t b[4], int shift) {
    uint64_t t[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)a[i] * b[j];
            prod += t[i+j];
            prod += carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i+4] = carry;
    }

    int word_shift = shift / 64;
    int bit_shift  = shift % 64;
    for (int i = 0; i < 4; i++) {
        uint128_t v = 0;
        if (i + word_shift < 8) v = uint128_t(t[i + word_shift]);
        if (bit_shift && i + word_shift + 1 < 8) {
            v += uint128_t(t[i + word_shift + 1]) << 64;
        }
        r[i] = (uint64_t)(v >> bit_shift);
    }
}

__host__ __device__ void scalar_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t t[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)a[i] * b[j];
            prod += t[i+j];
            prod += carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i+4] = carry;
    }

    scalar_reduce_n(r, t);
}

__host__ __device__ void scalar_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t tmp = a[i] + b[i] + carry;
        carry = (tmp < a[i]) || (carry && tmp == a[i]);
        r[i] = tmp;
    }
    scalar_reduce_n(r, r);
}

__host__ __device__ void scalar_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t bi = b[i] + borrow;
        uint64_t ri = a[i] - bi;
        borrow = (a[i] < bi);
        r[i] = ri;
    }

    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t tmp = r[i] + N_CONST[i] + carry;
            carry = (tmp < r[i]);
            r[i] = tmp;
        }
    }

    scalar_reduce_n(r, r);
}

__host__ __device__ int scalar_is_zero(const uint64_t a[4]) {
    uint64_t acc = 0;
    for (int i = 0; i < 4; i++) {
        acc |= a[i];
    }
    return acc == 0;
}

__host__ __device__ void scalar_neg(uint64_t r[4], const uint64_t a[4]) {
    if (scalar_is_zero(a)) {
        r[0] = r[1] = r[2] = r[3] = 0;
        return;
    }
    uint64_t tmp[4];
    scalar_sub(tmp, N_CONST, a);
    scalar_reduce_n(r, tmp);
}

__host__ __device__ void scalar_split_lambda(uint64_t r1[4], uint64_t r2[4], const uint64_t k[4]) {
    uint64_t c1[4], c2[4], t1[4], t2[4];

    scalar_mul_shift_var(c1, k, G1, 384);
    scalar_mul_shift_var(c2, k, G2, 384);
    scalar_mul(t1, c1, MINUS_B1);
    scalar_mul(t2, c2, MINUS_B2);
    scalar_add(r2, t1, t2);
    scalar_mul(t1, r2, LAMBDA_N);
    scalar_neg(t1, t1);
    scalar_add(r1, t1, k);
}

__host__ __device__ void jacobian_scalar_mult_glv(ECPointJacobian *R, const uint64_t k[4]) {
    uint64_t r1[4], r2[4];
    scalar_split_lambda(r1, r2, k);
    ECPointJacobian P1, P2;
    jacobian_scalar_mult(&P1, r1);
    jacobian_scalar_mult_phi(&P2, r2);
    jacobian_add(R, &P1, &P2);
}

__host__ __device__ void point_from_montgomery(ECPoint *result, const ECPoint *point_mont) {
    if (point_mont->infinity) {
        result->infinity = 1;
        for (int i = 0; i < 4; i++) {
            result->x[i] = 0;
            result->y[i] = 0;
        }
        return;
    }
    from_montgomery_p(result->x, point_mont->x);
    from_montgomery_p(result->y, point_mont->y);
    result->infinity = 0;
}

__host__ __device__ void point_init_jacobian(ECPointJacobian *P) {
    for (int i = 0; i < 4; i++) {
        P->X[i] = 0;
        P->Y[i] = 0;
        P->Z[i] = 0;
    }
    P->infinity = 1;
}

__host__ __device__ void point_add_jacobian(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q) {
    jacobian_add(R, P, Q);
}

__host__ __device__ void point_double_jacobian(ECPointJacobian *R, const ECPointJacobian *P) {
    jacobian_double(R, P);
}

__host__ __device__ void scalar_mult_jacobian(ECPointJacobian *R, const uint64_t *k) {
    jacobian_scalar_mult_glv(R, k);
}

__host__ __device__ void get_compressed_public_key(unsigned char *out, const ECPoint *public_key) {
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
        out[1 + i*8 + 6] = (word >> 8) & 0xFF;
        out[1 + i*8 + 7] = word & 0xFF;
    }
}

__host__ __device__ void generate_public_key(unsigned char *out, const uint64_t *PRIV_KEY) {
    ECPoint pub;
    ECPointJacobian pub_jac;

    jacobian_scalar_mult_glv(&pub_jac, PRIV_KEY);
    jacobian_to_affine(&pub, &pub_jac);
    get_compressed_public_key(out, &pub);
}

int main() {
    const std::string expected_pubkey = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    uint64_t PRIV_KEY[4] = {1, 0, 0, 0};
    unsigned char pubkey_compressed[33];

    #ifdef __CUDA_ARCH__
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        size_t maxPoints = freeMem / 128;
        int w = 4;
        while (w < MAX_W && (1ull << (w-1)) < maxPoints) {
            w++;
        }
        if (w > MAX_W) w = MAX_W;
        window_size = w;
    #else
        window_size = 8;
    #endif

    init_precomp_g();

    generate_public_key(pubkey_compressed, PRIV_KEY);

    std::cout << "Compressed Public Key: ";
    for (int i = 0; i < 33; ++i) {
        printf("%02x", pubkey_compressed[i]);
    }
    std::cout << std::endl;
    std::cout << "A chave pública esperada é: " << expected_pubkey << std::endl;

    return 0;
}

/*
int main() {
    const std::string expected_pubkey = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    uint64_t PRIV_KEY[4] = {1, 0, 0, 0};
    unsigned char pubkey_compressed[33];

    ECPointJacobian pub_jac;

    #ifdef __CUDA_ARCH__
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        size_t maxPoints = freeMem / 128;
        int w = 4;
        while (w < MAX_W && (1ull << (w-1)) < maxPoints) {
            w++;
        }
        if (w > MAX_W) w = MAX_W;
        window_size = w;
    #else
        window_size = 8;
    #endif

    init_precomp_g();

    int count = 0;
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < 100000000; i++) {
        scalar_mult_jacobian(&pub_jac, PRIV_KEY);
        count++;

        auto now = std::chrono::steady_clock::now();
        if(std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= 10) {
        std::cout << "Chaves geradas = " << count << std::endl;
        start = now;
        }
    }

    generate_public_key(pubkey_compressed, PRIV_KEY);
    std::cout << "Compressed Public Key: ";
    for (int i = 0; i < 33; ++i) {
        printf("%02x", pubkey_compressed[i]);
    }
    std::cout << std::endl;
    std::cout << "A chave pública esperada é: " << expected_pubkey << std::endl;

    return 0;
}
*/
