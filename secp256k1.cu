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

#include <iostream>
#include <tuple>
#include <boost/multiprecision/cpp_int.hpp>
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

using boost::multiprecision::cpp_int;
using BigInt = cpp_int;
using std::make_tuple;
using std::tuple;
using std::get;
using std::pair;
using std::make_pair;

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

constexpr int MAX_BITS = 256;

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

__device__ __host__ int bignum_is_zero(const uint64_t *a) {
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

__device__ __host__ void bignum_zero(uint64_t *a) {
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

__host__ void bignum_mul_full(uint64_t *result_high, uint64_t *result_low,
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

__host__ void montgomery_reduce_p(uint64_t *result,
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

__host__ void to_montgomery_p(uint64_t *result, const uint64_t *a) {
    uint64_t high[4], low[4];
    bignum_mul_full(high, low, a, (uint64_t*)R2_MOD_P);
    montgomery_reduce_p(result, high, low);
}

__host__ void from_montgomery_p(uint64_t *result, const uint64_t *a) {
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

__host__ void mod_mul_mont_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t high[4], low[4];
    bignum_mul_full(high, low, a, b);
    montgomery_reduce_p(result, high, low);
}

__device__ void mod_sqr_mont_p(uint64_t out[4], const uint64_t in[4]) {
    // out = in^2 mod P
    mod_mul_mont_p(out, in, in);
}

/* ----------- ALMOST INVERSE C++ -----------
    Based on the Paper Almost-Inverse/Bernstein-Yang, REF: https://eprint.iacr.org/2019/266.pdf
*/

BigInt div2_floor(const BigInt &a) {
    return a / 2;
}

BigInt truncate(const BigInt& f, int t) {
    BigInt mask = (BigInt(1) << t) - 1;
    BigInt result = f & mask;
    BigInt bound = BigInt(1) << (t - 1);
    BigInt over = (result >= bound) ? 1 : 0;
    result -= over * (BigInt(1) << t);
    return result;
}

int bit_length(const BigInt &x) {  
    int msb = 0;  
    for (int i = 0; i < MAX_BITS; ++i) {  
        msb = ((x >> i) & 1) ? i + 1 : msb;  
    }  
    return msb;  
}

auto divsteps2(int n, int t, int delta, BigInt f, BigInt g) {
    f = truncate(f, t);
    g = truncate(g, t);

    BigInt scale = BigInt(1) << n;
    BigInt U = scale;
    BigInt V = 0;
    BigInt Q = 0;
    BigInt R = scale;

    for (int i = 0; i < n; ++i) {
        f = truncate(f, t);

        int g_odd = (g & 1) != 0;
        int delta_pos = (delta > 0) ? 1 : 0;
        int swap_mask = delta_pos & g_odd;

        BigInt new_f = swap_mask ? g : f;
        BigInt new_g = swap_mask ? -f : g;
        BigInt new_U = swap_mask ? Q : U;
        BigInt new_Q = swap_mask ? -U : Q;
        BigInt new_V = swap_mask ? R : V;
        BigInt new_R = swap_mask ? -V : R;

        f = new_f;
        g = new_g;
        U = new_U;
        Q = new_Q;
        V = new_V;
        R = new_R;

        delta = delta * (1 - 2 * swap_mask) + 1;

        BigInt tmpg = g + g_odd * f;
        g = div2_floor(tmpg);
        Q = div2_floor(Q + g_odd * U);
        R = div2_floor(R + g_odd * V);

        --t;
        g = truncate(g, t);
    }

    auto UV = make_pair(U, V);
    auto QR = make_pair(Q, R);
    auto P  = make_pair(UV, QR);
    return make_tuple(delta, f, g, P);
}

int iterations(int d) {
    return (d < 46) ? (49 * d + 80) / 17 : (49 * d + 57) / 17;
}

BigInt recip2(BigInt f, BigInt g) {
    if ((f & 1) == 0) throw std::invalid_argument("f must be odd");

    int d = std::max(bit_length(f), bit_length(g));
    int m = iterations(d);

    BigInt base = (f + 1) / 2;
    BigInt precomp = boost::multiprecision::powm(base, m - 1, f);

    auto result = divsteps2(m, m + 1, 1, f, g);
    BigInt fm = get<1>(result);
    auto P = get<3>(result);

    BigInt scale = BigInt(1) << m;
    BigInt V_scaled = P.first.second;
    BigInt V_int = (V_scaled * (BigInt(1) << (m - 1))) / scale;
    if (fm < 0) V_int = -V_int;

    BigInt inv = (V_int * precomp) % f;
    if (inv < 0) inv += f;

    return inv;
}

// ------------- END C++ -------------

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

__host__ int jacobian_is_infinity(const ECPointJacobian *point) {
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

cpp_int array_to_bigint(const uint64_t in[], size_t n_words) {
    cpp_int result = 0;
    for (size_t i = 0; i < n_words; ++i) {
        cpp_int limb = in[i];
        result += limb << (64 * i);
    }
    return result;
}

__host__ void almost_inverse(uint64_t out[4], const uint64_t f[4], const uint64_t g[4]) {
    BigInt temp_f = array_to_bigint(f, 4);
    BigInt temp_g = array_to_bigint(g, 4);
    BigInt temp_res = recip2(temp_f, temp_g);

    for (int i = 0; i < 4; i++) {
        out[i] = (temp_res >> (64 * i)) & 0xFFFFFFFFFFFFFFFFULL;
    }
}

__host__ void jacobian_to_affine(ECPoint *aff, const ECPointJacobian *jac) {
    if (jacobian_is_infinity(jac)) {
        bignum_zero(aff->x);
        bignum_zero(aff->y);
        aff->infinity = 1;
        return;
    }

    uint64_t z_norm[4], z_inv[4], z_inv_sqr[4], z_inv_cube[4];

    from_montgomery_p(z_norm, jac->Z);

    almost_inverse(z_norm, z_inv, P_CONST);

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

__host__ void kernel_point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q) {
    ECPointJacobian P_jac, Q_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    affine_to_jacobian(&Q_jac, Q);    
    jacobian_add(&R_jac, &P_jac, &Q_jac);   
    jacobian_to_affine(R, &R_jac);
}

__host__ void kernel_point_double(ECPoint *R, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    jacobian_double(&R_jac, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

__host__ void kernel_scalar_mult(ECPoint *R, const uint64_t *k, const ECPoint *P) {
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

__host__ void kernel_get_compressed_public_key(unsigned char *out, const ECPoint *public_key) {
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

__host__ void generate_public_key(unsigned char *out, const uint64_t *PRIV_KEY) {
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

__host__ void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q) {
    kernel_point_add(R, P, Q);
}

__host__ void point_double(ECPoint *R, const ECPoint *P) {
    kernel_point_double(R, P);
}

__global__ void scalar_mult(ECPoint *R, const uint64_t *k, const ECPoint *P) {
    kernel_scalar_mult(R, k, P);
}

__global__ void point_is_valid(int *result, const ECPoint *point) {
    *result = kernel_point_is_valid(point);
}

__host__ void get_compressed_public_key(unsigned char *out, const ECPoint *pub) {
    kernel_get_compressed_public_key(out, pub);
}

__host__ void test_mod_inverse(const __uint256_t* f, const __uint256_t* g, __uint256_t* result) {
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
