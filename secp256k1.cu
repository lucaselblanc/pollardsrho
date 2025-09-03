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

__constant__ unsigned int P_CONST[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ unsigned int N_CONST[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ unsigned int GX_CONST[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

__constant__ unsigned int GY_CONST[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

__constant__ unsigned int R_MOD_P[8] = {
    0x000003D1, 0x00000001, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

__constant__ unsigned int R2_MOD_P[8] = {
    0x000E90A1, 0x000007A2, 0x00000001, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

__constant__ unsigned int R2_MOD_N[8] = {
    0x67D7D140, 0x896CF214, 0x0E7CF878, 0x741496C2,
    0x5BCD07C6, 0xE697F5E4, 0x81C69BC5, 0x9D671CD5
};

__constant__ unsigned int MU_P = 0xD2253531;
__constant__ unsigned int MU_N = 0x5588B13F;

__constant__ unsigned int ZERO[8]  = {0, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int ONE[8]   = {1, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int TWO[8]   = {2, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int THREE[8] = {3, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int SEVEN[8] = {7, 0, 0, 0, 0, 0, 0, 0};

__constant__ unsigned int ONE_MONT[8] = {
    0x000003D1, 0x00000001, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

__constant__ unsigned int SEVEN_MONT[8] = {
    0x00001A97, 0x00000007, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

typedef struct {
    unsigned int x[8];
    unsigned int y[8];
    int infinity;
} ECPoint;

typedef struct {
    unsigned int X[8];
    unsigned int Y[8];
    unsigned int Z[8];
    int infinity;
} ECPointJacobian;

__device__ int bignum_cmp(const unsigned int *a, const unsigned int *b) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ int bignum_is_zero(const unsigned int *a) {
    for (int i = 0; i < 8; i++) {
        if (a[i] != 0) return 0;
    }
    return 1;
}

__device__ int bignum_is_odd(const unsigned int *a) {
    return a[0] & 1;
}

__device__ void bignum_copy(unsigned int *dst, const unsigned int *src) {
    for (int i = 0; i < 8; i++) {
        dst[i] = src[i];
    }
}

__device__ void bignum_zero(unsigned int *a) {
    for (int i = 0; i < 8; i++) {
        a[i] = 0;
    }
}

__device__ int bignum_is_one(const unsigned int *a) {
    if (a[0] != 1u) return 0;
    for (int i = 1; i < 8; ++i) {
        if (a[i] != 0u) return 0;
    }
    return 1;
}

__device__ void bignum_set_ui(unsigned int *a, unsigned int val) {
    bignum_zero(a);
    a[0] = val;
}

__device__ unsigned int bignum_add_carry(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned long long carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (unsigned long long)a[i] + b[i];
        result[i] = (unsigned int)carry;
        carry >>= 32;
    }
    return (unsigned int)carry;
}

__device__ unsigned int bignum_sub_borrow(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    long long carry = 0;
    for (int i = 0; i < 8; i++) {
        long long tmp = (long long)a[i] - (long long)b[i] - carry;
        if (tmp < 0) {
            result[i] = (unsigned int)(tmp + (1ULL << 32));
            carry = 1;
        } else {
            result[i] = (unsigned int)tmp;
            carry = 0;
        }
    }
    return (unsigned int)carry;
}

__device__ void bignum_shr1(unsigned int *result, const unsigned int *a) {
    unsigned int carry = 0;
    for (int i = 7; i >= 0; i--) {
        unsigned int new_carry = a[i] & 1;
        result[i] = (a[i] >> 1) | (carry << 31);
        carry = new_carry;
    }
}

__device__ void bignum_mul_full(unsigned int *result_high, unsigned int *result_low, 
                                const unsigned int *a, const unsigned int *b) {
    unsigned int temp[16];
    
    for (int i = 0; i < 16; i++) {
        temp[i] = 0;
    }
    
    for (int i = 0; i < 8; i++) {
        unsigned long long carry = 0;
        for (int j = 0; j < 8; j++) {
            unsigned long long prod = (unsigned long long)a[i] * b[j] + temp[i + j] + carry;
            temp[i + j] = (unsigned int)prod;
            carry = prod >> 32;
        }
        temp[i + 8] = (unsigned int)carry;
    }
    
    for (int i = 0; i < 8; i++) {
        result_low[i] = temp[i];
        result_high[i] = temp[i + 8];
    }
}

__device__ void montgomery_reduce_p(unsigned int *result, const unsigned int *input_high, const unsigned int *input_low) {
    unsigned int temp[16];
    
    for (int i = 0; i < 8; i++) {
        temp[i] = input_low[i];
        temp[i + 8] = input_high[i];
    }
    
    for (int i = 0; i < 8; i++) {
        unsigned int ui = (temp[i] * MU_P) & 0xFFFFFFFF;
        unsigned long long carry = 0;
        
        for (int j = 0; j < 8; j++) {
            unsigned long long prod = (unsigned long long)ui * P_CONST[j] + temp[i + j] + carry;
            temp[i + j] = (unsigned int)prod;
            carry = prod >> 32;
        }
        
        for (int j = i + 8; j < 16; j++) {
            unsigned long long tmp = (unsigned long long)temp[j] + carry;
            temp[j] = (unsigned int)tmp;
            carry = tmp >> 32;
        }
    }
    
    for (int i = 0; i < 8; i++) {
        result[i] = temp[i + 8];
    }
    
    if (bignum_cmp(result, P_CONST) >= 0) {
        bignum_sub_borrow(result, result, P_CONST);
    }
}

__device__ void to_montgomery_p(unsigned int *result, const unsigned int *a) {
    unsigned int high[8], low[8];
    bignum_mul_full(high, low, a, R2_MOD_P);
    montgomery_reduce_p(result, high, low);
}

__device__ void from_montgomery_p(unsigned int *result, const unsigned int *a) {
    unsigned int zero[8];
    bignum_zero(zero);
    montgomery_reduce_p(result, zero, a);
}

__device__ void mod_add_p(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int temp[8];
    unsigned int carry = bignum_add_carry(temp, a, b);
    
    if (carry || bignum_cmp(temp, P_CONST) >= 0) {
        bignum_sub_borrow(result, temp, P_CONST);
    } else {
        bignum_copy(result, temp);
    }
}

__device__ void mod_sub_p(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int temp[8];
    unsigned int borrow = bignum_sub_borrow(temp, a, b);
    
    if (borrow) {
        bignum_add_carry(result, temp, P_CONST);
    } else {
        bignum_copy(result, temp);
    }
}

__device__ void mod_mul_mont_p(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int high[8], low[8];
    bignum_mul_full(high, low, a, b);
    montgomery_reduce_p(result, high, low);
}

__device__ void mod_sqr_mont_p(unsigned int *result, const unsigned int *a) {
    mod_mul_mont_p(result, a, a);
}

/*
//Divstep / Almost Inverse
__device__ void mod_inverse_p(uint32_t *result, const uint32_t *a_normal) {
    const uint32_t p[8] = {
        0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };

    if (bignum_is_zero(a_normal)) {
        bignum_zero(result);
        return;
    }

    auto shr1 = [] __device__ (uint32_t *x) {
        uint32_t carry = 0u;
        for (int k = 7; k >= 0; --k) {
            uint32_t next = (x[k] & 1u) << 31;
            x[k] = (x[k] >> 1) | carry;
            carry = next;
        }
    };

    auto add_cond = [] __device__ (uint32_t *dst, const uint32_t *src, uint32_t mask) {
        uint64_t carry = 0;
        for (int t = 0; t < 8; ++t) {
            uint64_t addend = (uint64_t)(src[t] & mask);
            uint64_t sum = (uint64_t)dst[t] + addend + carry;
            dst[t] = (uint32_t)sum;
            carry = sum >> 32;
        }
    };

    int32_t delta = 1;
    uint32_t u[8], v[8], q[8], r[8];
    uint32_t temp_u[8], temp_q[8], q_minus_p[8];

    bignum_copy(u, a_normal);

    if (bignum_cmp(u, p) >= 0) {
        bignum_sub_borrow(u, u, p);
    }

    bignum_copy(v, p);
    bignum_set_ui(q, 1);
    bignum_zero(r);

    for (int i = 0; i < 128; ++i) {
        #pragma unroll 4
        for (int j = 0; j < 4; ++j) {
            uint32_t v_odd = v[0] & 1u;
            int32_t swap_flag = ((delta > 0) & (int32_t)v_odd);
            int32_t m = -swap_flag;
            uint32_t mask = (uint32_t)m;
            uint32_t inv_mask = ~mask;

            for (int t = 0; t < 8; ++t) {
                temp_u[t] = u[t];
                temp_q[t] = q[t];
            }

            for (int t = 0; t < 8; ++t) {
                u[t] = (v[t] & mask) | (u[t] & inv_mask);
                v[t] = (temp_u[t] & mask) | (v[t] & inv_mask);
                q[t] = (r[t] & mask) | (q[t] & inv_mask);
                r[t] = (temp_q[t] & mask) | (r[t] & inv_mask);
            }

            int32_t temp_delta = delta;
            int32_t neg = -temp_delta;
            delta = (neg & m) | (delta & ~m);
            delta++;

            uint32_t v_odd_mask = 0u - v_odd;
            add_cond(v, u, v_odd_mask);
            add_cond(r, q, v_odd_mask);

            shr1(v);

            uint32_t r_odd = r[0] & 1u;
            uint32_t r_odd_mask = 0u - r_odd;
            add_cond(r, p, r_odd_mask);
            shr1(r);
        }
    }

    uint32_t borrow = 0u;
    for (int t = 0; t < 8; ++t) {
        uint64_t tmp = (uint64_t)p[t] + (uint64_t)borrow;
        uint32_t qi = q[t];
        uint32_t diff = (uint32_t)((uint64_t)qi - tmp);
        q_minus_p[t] = diff;
        borrow = (qi < tmp) ? 1u : 0u;
    }

    uint32_t sel_mask = 0u - (borrow ^ 1u);
    uint32_t sel_inv = ~sel_mask;
    for (int t = 0; t < 8; ++t) {
        q[t] = (q_minus_p[t] & sel_mask) | (q[t] & sel_inv);
    }

    bignum_copy(result, q);
    to_montgomery_p(result, q);
}
*/

__device__ void mod_inverse_p(uint32_t *result, const uint32_t *a_normal) {
    // p = modulus (little-word order: word0 = LSW)
    const uint32_t p[8] = {
        0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };

    // Se a == 0 -> zero
    if (bignum_is_zero(a_normal)) {
        bignum_zero(result);
        return;
    }

    // Exponente = p - 2 (armazenado em little-word order)
    uint32_t exp[8];
    // copia p para exp
    for (int i = 0; i < 8; ++i) exp[i] = p[i];
    // subtrai 2
    uint64_t borrow = 0;
    uint64_t sub = (uint64_t)2;
    for (int i = 0; i < 8; ++i) {
        uint64_t wi = (uint64_t)exp[i];
        uint64_t v = wi - (sub & 0xFFFFFFFFull) - borrow;
        exp[i] = (uint32_t)v;
        borrow = ((wi < (sub & 0xFFFFFFFFull)) || (borrow && wi == (sub & 0xFFFFFFFFull))) ? 1 : 0;
        sub = 0; // só subtrair 2 na primeira iteração
    }

    // base_mont = a_normal converted to Montgomery
    unsigned int base_mont[8];
    to_montgomery_p(base_mont, a_normal); // base em MONT

    // result_mont = 1 (na representação MONT, 1 == R mod p == ONE_MONT)
    unsigned int res_mont[8];
    bignum_copy(res_mont, ONE_MONT);

    // Exponenciação binária: do MSB -> LSB do exp
    for (int bit_idx = 255; bit_idx >= 0; --bit_idx) {
        // res = res^2
        unsigned int tmp[8];
        mod_sqr_mont_p(tmp, res_mont);
        bignum_copy(res_mont, tmp);

        int word = bit_idx / 32;   // palavra (0 = LSW)
        int bit  = bit_idx % 32;
        if ((exp[word] >> bit) & 1u) {
            // res = res * base
            mod_mul_mont_p(tmp, res_mont, base_mont);
            bignum_copy(res_mont, tmp);
        }
    }

    // res_mont é a^(p-2) em MONTGOMERY -> copiar para result (uint32_t*)
    for (int i = 0; i < 8; ++i) result[i] = res_mont[i];
}

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

    unsigned int z_norm[8], z_inv[8], z_inv_sqr[8], z_inv_cube[8];

    from_montgomery_p(z_norm, jac->Z);

    mod_inverse_p(z_inv, z_norm);
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

    unsigned int A[8], B[8], C[8], D[8], E[8];
    unsigned int X2[8];

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

    unsigned int P_infinity = jacobian_is_infinity(P);
    unsigned int Q_infinity = jacobian_is_infinity(Q);

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

    unsigned int U1[8], U2[8], S1[8], S2[8], H[8], I[8], J[8], r[8], V[8];
    unsigned int Z1Z1[8], Z2Z2[8], Z1Z2[8], temp1[8], temp2[8];

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
            return;
        } else {
            jacobian_set_infinity(result);
            return;
        }
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

__device__ void scalar_reduce_n(unsigned int *r, const unsigned int *k) {
    unsigned int t[8];
    unsigned int borrow = bignum_sub_borrow(t, k, N_CONST);

    if (borrow == 0) {
        bignum_copy(r, t);
    } else {
        bignum_copy(r, k);
    }
}

__device__ void jacobian_scalar_mult(ECPointJacobian *result, const unsigned int *scalar, const ECPointJacobian *point) {
    if (bignum_is_zero(scalar) || jacobian_is_infinity(point)) {
        jacobian_set_infinity(result);
        return;
    }

    ECPointJacobian R0, R1;
    jacobian_set_infinity(&R0);
    R1 = *point;

    unsigned int k[8];
    bignum_copy(k, scalar);

    scalar_reduce_n(k, k);

    //(MSB > LSB)
    int msb = 255;
    while (msb >= 0) {
        int word = 7 - (msb / 32);
        int bit  = 31 - (msb % 32);
        if ((k[word] >> bit) & 1) break;
        msb--;
    }

    for (int i = msb; i >= 0; i--) {
        int word = 7 - (i / 32);
        int bit  = 31 - (i % 32);
        int kbit = (k[word] >> bit) & 1;

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

__device__ void kernel_scalar_mult(ECPoint *R, const unsigned int *k, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    jacobian_scalar_mult(&R_jac, k, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

__device__ int kernel_point_is_valid(const ECPoint *point) {
    if (point->infinity) return 1;

    unsigned int lhs[8], rhs[8];
    
    mod_sqr_mont_p(lhs, point->y);
    mod_sqr_mont_p(rhs, point->x);
    mod_mul_mont_p(rhs, rhs, point->x);
    mod_add_p(rhs, rhs, SEVEN_MONT);

    return (bignum_cmp(lhs, rhs) == 0);
}

__device__ void kernel_get_compressed_public_key(unsigned char *out, const ECPoint *public_key) {
    unsigned char prefix = (public_key->y[0] & 1) ? 0x03 : 0x02;
    out[0] = prefix;

    for (int i = 0; i < 8; i++) {
        unsigned int word = public_key->x[7 - i];
        out[1 + i*4]     = (word >> 24) & 0xFF;
        out[1 + i*4 + 1] = (word >> 16) & 0xFF;
        out[1 + i*4 + 2] = (word >> 8)  & 0xFF;
        out[1 + i*4 + 3] = word & 0xFF;
    }
}

__global__ void generate_public_key(unsigned char *out, unsigned int *PRIV_KEY) {
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

__global__ void scalar_mult(ECPoint *R, const unsigned int *k, const ECPoint *P) {
    kernel_scalar_mult(R, k, P);
}

__global__ void point_is_valid(int *result, const ECPoint *point) {
    *result = kernel_point_is_valid(point);
}

__global__ void get_compressed_public_key(unsigned char *out, const ECPoint *pub) {
    kernel_get_compressed_public_key(out, pub);
}

/*
//test
__global__ void test_inverse_kernel(unsigned int *a, unsigned int *result) {
    mod_inverse_p(result, a);
}
*/

/*
int main() {

    //MSB
    unsigned int h_priv[8] = {
        0x00000000, 0x00000000, 0x00000000, 0x00000003, 0x3e766570, 0x5359f04f, 0x28b88cf8, 0x97c603c9
    };

    unsigned int h_result[8];
    unsigned int *d_priv, *d_result;

    cudaMalloc(&d_priv, sizeof(h_priv));
    cudaMalloc(&d_result, sizeof(h_result));

    cudaMemcpy(d_priv, h_priv, sizeof(h_priv), cudaMemcpyHostToDevice);

    test_inverse_kernel<<<1,1>>>(d_priv, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, sizeof(h_result), cudaMemcpyDeviceToHost);

    printf("Inverse a mod P (big-endian words):\n");
    for (int i = 7; i >= 0; i--) {
        printf("%08X ", h_result[i]);
    }
    printf("\n");

    cudaFree(d_priv);
    cudaFree(d_result);

    return 0;
}
*/

//TESTING BEGIN

__global__ void debug_inverse() {
    uint32_t a[8] = {
        0x12345678u, 0x9abcdef0u, 0x0fedcba9u, 0x87654321u,
        0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u
    };

    uint32_t inv_mont[8];
    uint32_t a_mont[8];
    uint32_t prod_mont[8], prod_norm[8];

    // calcula inverso
    mod_inverse_p(inv_mont, a);

    // a em montgomery
    to_montgomery_p(a_mont, a);

    // produto = a * inv(a) em montgomery
    mod_mul_mont_p(prod_mont, a_mont, inv_mont);

    // converte de volta para normal
    from_montgomery_p(prod_norm, prod_mont);

    // log
    printf("a (normal): ");
    for (int i = 7; i >= 0; --i) printf("%08X", a[i]);
    printf("\n");

    printf("inv(a) (mont): ");
    for (int i = 7; i >= 0; --i) printf("%08X", inv_mont[i]);
    printf("\n");

    printf("a * inv(a) mod p (normal): ");
    for (int i = 7; i >= 0; --i) printf("%08X", prod_norm[i]);
    printf("\n");

    if (bignum_is_one(prod_norm)) {
        printf("VERIFICADO: inverso correto!\\n");
    } else {
        printf("ERRO: inverso incorreto!\\n");
    }
}

int main() {
    debug_inverse<<<1,1>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}

//TESTING END

/*
int main() {
    
    //MSB
    unsigned int h_priv[8] = {
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001
    };

    //pub key: 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798

    unsigned int *d_priv;
    unsigned char *d_out, h_out[33];

    cudaMalloc(&d_priv, sizeof(h_priv));
    cudaMalloc(&d_out, 33);

    cudaMemcpy(d_priv, h_priv, sizeof(h_priv), cudaMemcpyHostToDevice);

    generate_public_key<<<1,1>>>(d_out, d_priv);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 33, cudaMemcpyDeviceToHost);

    printf("Public key compressed: ");
    for (int i = 0; i < 33; i++) {
        printf("%02X", h_out[i]);
    }
    printf("\n");

    cudaFree(d_priv);
    cudaFree(d_out);
    return 0;
}
*/