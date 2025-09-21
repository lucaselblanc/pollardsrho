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

#include "secp256k1.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>

#ifdef __CUDA_ARCH__
__device__ __constant__ uint64_t P_CONST[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
__device__ __constant__ uint64_t N_CONST[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
__device__ __constant__ uint64_t GX_CONST[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
__device__ __constant__ uint64_t GY_CONST[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };
__device__ __constant__ uint64_t R2_MOD_P[4] = { 0x000007A2000E90A1, 0x0000000000000001, 0x0ULL, 0x0ULL };
__device__ __constant__ uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
__device__ __constant__ uint64_t P_CONST_MINUS_2[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
__device__ __constant__ uint64_t MU_P = 0xD838091DD2253531ULL;
#else
const uint64_t P_CONST[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
const uint64_t N_CONST[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
const uint64_t GX_CONST[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
const uint64_t GY_CONST[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };
const uint64_t R2_MOD_P[4] = { 0x000007A2000E90A1, 0x0000000000000001, 0x0ULL, 0x0ULL };
const uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
const uint64_t P_CONST_MINUS_2[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
const uint64_t MU_P = 0xD838091DD2253531ULL;
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

__host__ __device__ void affine_to_jacobian(ECPointJacobian *jac, const ECPoint *aff) {
    if (aff->infinity) {
        jacobian_set_infinity(jac);
        return;
    }
    for (int i = 0; i < 4; i++) {
        jac->X[i] = aff->x[i];
        jac->Y[i] = aff->y[i];
        jac->Z[i] = ONE_MONT[i];
    }
    jac->infinity = 0;
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

__host__ __device__ void jacobian_to_affine(ECPoint *aff, const ECPointJacobian *jac) {
    if (jacobian_is_infinity(jac)) {
        for (int i = 0; i < 4; i++) aff->x[i] = aff->y[i] = 0;
        aff->infinity = 1;
        return;
    }

    uint64_t Z_inv[4], Z_inv2[4], Z_inv3[4];

    mod_exp_mont_p(Z_inv, jac->Z, P_CONST_MINUS_2);
    mod_sqr_mont_p(Z_inv2, Z_inv);
    mod_mul_mont_p(Z_inv3, Z_inv2, Z_inv);
    mod_mul_mont_p(aff->x, jac->X, Z_inv2);
    mod_mul_mont_p(aff->y, jac->Y, Z_inv3);
    from_montgomery_p(aff->x, aff->x);
    from_montgomery_p(aff->y, aff->y);

    aff->infinity = 0;
}

__host__ __device__ void jacobian_double(ECPointJacobian *result, const ECPointJacobian *point) {
    uint64_t y_zero = 0;
    for (int i = 0; i < 4; i++) {
        y_zero |= point->Y[i];
    }

    if (jacobian_is_infinity(point) || y_zero == 0) {
        jacobian_set_infinity(result);
        return;
    }

    uint64_t A[4], B[4], C[4], D[4], E[4], X2[4];
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

__host__ __device__ void jacobian_add(ECPointJacobian *result, const ECPointJacobian *P, const ECPointJacobian *Q) {
    int P_infinity = jacobian_is_infinity(P);
    int Q_infinity = jacobian_is_infinity(Q);

    if (P_infinity) {
        for (int i = 0; i < 4; i++) {
            result->X[i] = Q->X[i];
            result->Y[i] = Q->Y[i];
            result->Z[i] = Q->Z[i];
        }
        result->infinity = Q->infinity;
        return;
    }
    if (Q_infinity) {
        for (int i = 0; i < 4; i++) {
            result->X[i] = P->X[i];
            result->Y[i] = P->Y[i];
            result->Z[i] = P->Z[i];
        }
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

    uint64_t h_zero = 0;
    uint64_t r_zero = 0;
    for (int i = 0; i < 4; i++) {
        h_zero |= H[i];
        r_zero |= r[i];
    }

    int is_H_zero = (h_zero == 0);
    int is_r_zero = (r_zero == 0);

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

__host__ __device__ void jacobian_scalar_mult(ECPointJacobian *result, const uint64_t *scalar, const ECPointJacobian *point) {
    if (jacobian_is_infinity(point)) {
        jacobian_set_infinity(result);
        return;
    }

    uint64_t k[4];
    scalar_reduce_n(k, scalar);

    ECPointJacobian R0, R1;
    jacobian_set_infinity(&R0);
    R1 = *point;

    auto cswap = [](ECPointJacobian &a, ECPointJacobian &b, uint64_t swap) {
        swap = 0 - swap;
        for (int i = 0; i < 4; i++) {
            uint64_t tmp;

            tmp = (a.X[i] ^ b.X[i]) & swap;
            a.X[i] ^= tmp;
            b.X[i] ^= tmp;

            tmp = (a.Y[i] ^ b.Y[i]) & swap;
            a.Y[i] ^= tmp;
            b.Y[i] ^= tmp;

            tmp = (a.Z[i] ^ b.Z[i]) & swap;
            a.Z[i] ^= tmp;
            b.Z[i] ^= tmp;
        }
        int tmp_inf = (a.infinity ^ b.infinity) & (int)swap;
        a.infinity ^= tmp_inf;
        b.infinity ^= tmp_inf;
    };

    for (int i = 255; i >= 0; i--) {
        int word = i / 64;
        int bit  = i % 64;
        uint64_t kbit = (k[word] >> bit) & 1ULL;

        cswap(R0, R1, kbit);

        ECPointJacobian R0_new, R1_new;
        jacobian_double(&R0_new, &R0);
        jacobian_add(&R1_new, &R0, &R1);

        R0 = R0_new;
        R1 = R1_new;

        cswap(R0, R1, kbit);
    }

    *result = R0;
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

__host__ __device__ void point_init(ECPoint *point) {
    for (int i = 0; i < 4; i++) {
        point->x[i] = 0;
        point->y[i] = 0;
    }
    point->infinity = 0;
}

__host__ __device__ void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q) {
    ECPointJacobian P_jac, Q_jac, R_jac;
    affine_to_jacobian(&P_jac, P);
    affine_to_jacobian(&Q_jac, Q);
    jacobian_add(&R_jac, &P_jac, &Q_jac);
    jacobian_to_affine(R, &R_jac);
}

__host__ __device__ void point_double(ECPoint *R, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    affine_to_jacobian(&P_jac, P);
    jacobian_double(&R_jac, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

__host__ __device__ void scalar_mult(ECPoint *R, const uint64_t *k, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    affine_to_jacobian(&P_jac, P);
    jacobian_scalar_mult(&R_jac, k, &P_jac);
    jacobian_to_affine(R, &R_jac);
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
    ECPoint G;
    ECPointJacobian G_jac, pub_jac;

    to_montgomery_p(G.x, GX_CONST);
    to_montgomery_p(G.y, GY_CONST);
    G.infinity = 0;

    for (int i = 0; i < 4; i++) G_jac.Z[i] = ONE_MONT[i];
    G_jac.infinity = 0;

    for (int i = 0; i < 4; i++) {
        G_jac.X[i] = G.x[i];
        G_jac.Y[i] = G.y[i];
    }

    jacobian_scalar_mult(&pub_jac, PRIV_KEY, &G_jac);
    jacobian_to_affine(&pub, &pub_jac);
    get_compressed_public_key(out, &pub);
}

/*
int main() {
    const std::string expected_pubkey = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    uint64_t PRIV_KEY[4] = {1, 0, 0, 0};
    unsigned char pubkey_compressed[33];

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

#include <chrono>

__global__ void keygen_kernel(const uint64_t* priv_keys,
                              unsigned long long* counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char local_pubkey[33];
    uint64_t local_priv[4];
    for (int i = 0; i < 4; i++) local_priv[i] = priv_keys[i];

    for (int i = 0; i < BATCH; i++) {
        generate_public_key(local_pubkey, local_priv);
        atomicAdd(counter, 1ULL);
    }
}

int main() {
    const std::string expected_pubkey =
        "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    uint64_t h_priv_keys[4] = {1, 0, 0, 0};
    unsigned long long h_counter = 0ULL;

    uint64_t* d_priv_keys;
    unsigned long long* d_counter;

    cudaMalloc((void**)&d_priv_keys, 4 * sizeof(uint64_t));
    cudaMalloc((void**)&d_counter, sizeof(unsigned long long));

    cudaMemcpy(d_priv_keys, h_priv_keys, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counter, &h_counter, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

    int threads= prop.maxThreadsPerBlock / 2
    int blocks = 32;

    auto start = std::chrono::high_resolution_clock::now();
    auto last_report = start;

    while (true) {
        keygen_kernel<<<blocks, threads>>>(d_priv_keys, d_counter);
        cudaDeviceSynchronize();

        auto now = std::chrono::high_resolution_clock::now();
        double total_elapsed = std::chrono::duration<double>(now - start).count();
        double since_last = std::chrono::duration<double>(now - last_report).count();

        if (since_last >= 10.0) {
            cudaMemcpy(&h_counter, d_counter, sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost);
            std::cout << "Tempo: " << (int)total_elapsed
                      << "s, total de chaves geradas = " << h_counter << std::endl;
            last_report = now;
        }
    }

    cudaFree(d_priv_keys);
    cudaFree(d_counter);
    return 0;
}