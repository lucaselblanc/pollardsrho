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

__device__ void mod_sqr_mont_p(uint64_t *result, const uint64_t *a) {
    mod_mul_mont_p(result, a, a);
}

/*
static __device__ __forceinline__ bool is_zero_4(const uint64_t *x) {
    return (x[0] | x[1] | x[2] | x[3]) == 0ULL;
}

static __device__ __forceinline__ void copy_4(uint64_t *dst, const uint64_t *src) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
}

static __device__ __forceinline__ void set_ui_4(uint64_t *dst, uint64_t g) {
    dst[0] = g; dst[1] = 0ULL; dst[2] = 0ULL; dst[3] = 0ULL;
}

static __device__ __forceinline__ void zero_4(uint64_t *dst) {
    dst[0] = dst[1] = dst[2] = dst[3] = 0ULL;
}

static __device__ __forceinline__ void shr1_4(uint64_t *x) {
    uint64_t carry = 0ULL;
    for (int k = 0; k < 4; ++k) {
        uint64_t next = (k < 3) ? ((x[k+1] & 1ULL) << 63) : 0ULL;
        uint64_t cur = x[k];
        x[k] = (cur >> 1) | carry;
        carry = next;
    }
}

static __device__ __forceinline__ void add_cond_4(uint64_t *dst, const uint64_t *src, uint64_t mask) {
    uint64_t carry = 0ULL;
    for (int t = 0; t < 4; ++t) {
        uint64_t s = src[t] & mask;
        uint64_t old = dst[t];
        uint64_t tmp = old + s;
        uint64_t carry1 = (tmp < old) ? 1ULL : 0ULL;
        uint64_t sum = tmp + carry;
        uint64_t carry2 = (sum < tmp) ? 1ULL : 0ULL;
        dst[t] = sum;
        carry = carry1 | carry2;
    }
}

//Almost Inverse - Bernstein-Yang Variant
__device__ void mod_inverse_p(uint64_t *result, const uint64_t *a_normal) {

    const uint64_t p[4] = {
        0xFFFFFFFEFFFFFC2FULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };

    if (is_zero_4(a_normal)) {
        zero_4(result);
        return;
    }

    int32_t delta = 1;
    const int d = 256;
    const int m = (49*d + 57 + 16) / 17;
    uint64_t f[4], g[4], q[4], r[4];
    uint64_t temp_q[4];

    copy_4(f, a_normal);
    copy_4(g, p);
    set_ui_4(q, 1ULL);
    zero_4(r);

    for (int i = 0; i < m; ++i) {
        uint64_t g_odd = g[0] & 1ULL;

        int32_t swap_flag = (delta > 0 && g_odd) ? 1 : 0;
        uint64_t mask = (uint64_t)0 - (uint64_t)swap_flag;
        uint64_t inv_mask = ~mask;

        copy_4(temp_q, q);
        for (int t = 0; t < 4; ++t) {
            uint64_t ft = f[t], gt = g[t], qt = q[t], rt = r[t], tmpq = temp_q[t];
            f[t] = (gt & mask) | (ft & inv_mask);
            g[t] = (ft & mask) | (gt & inv_mask);
            q[t] = (rt & mask) | (qt & inv_mask);
            r[t] = (tmpq & mask) | (rt & inv_mask);
        }

        if (swap_flag) {
            delta = 1 - delta;
        } else {
            delta = delta + 1;
        }

        uint64_t g_odd_mask = 0ULL - g_odd;
        add_cond_4(g, f, g_odd_mask);
        add_cond_4(r, q, g_odd_mask);

        shr1_4(g);

        uint64_t r_odd = r[0] & 1ULL;
        uint64_t r_odd_mask = 0ULL - r_odd;
        add_cond_4(r, p, r_odd_mask);
        shr1_4(r);
    }
    
    //Aqui deve conter a normalização...

    //to_montgomery_p(result, q);
}
*/

static __device__ __forceinline__ bool is_zero_4(const uint64_t *x) {
    return (x[0] | x[1] | x[2] | x[3]) == 0ULL;
}

static __device__ __forceinline__ void copy_4(uint64_t *dst, const uint64_t *src) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
}

static __device__ __forceinline__ void set_ui_4(uint64_t *dst, uint64_t g) {
    dst[0] = g; dst[1] = 0ULL; dst[2] = 0ULL; dst[3] = 0ULL;
}

static __device__ __forceinline__ void zero_4(uint64_t *dst) {
    dst[0] = dst[1] = dst[2] = dst[3] = 0ULL;
}

static __device__ __forceinline__ void shr1_4(uint64_t *x) {
    uint64_t carry = 0ULL;
    for (int k = 0; k < 4; ++k) {
        uint64_t next = (k < 3) ? ((x[k+1] & 1ULL) << 63) : 0ULL;
        uint64_t cur = x[k];
        x[k] = (cur >> 1) | carry;
        carry = next;
    }
}

static __device__ __forceinline__ void sub_and_shr1_4(uint64_t *res, const uint64_t *a, const uint64_t *b) {
    uint64_t tmp[4];
    uint64_t borrow = 0;
    for(int i=0;i<4;i++){
        __uint128_t t = (__uint128_t)a[i] - b[i] - borrow;
        tmp[i] = (uint64_t)t;
        borrow = (t >> 127) & 1ULL;
    }
    shr1_4(tmp);
    for(int i=0;i<4;i++) res[i] = tmp[i];
}

static __device__ __forceinline__ void add_cond_4(uint64_t *dst, const uint64_t *src, uint64_t mask) {
    uint64_t carry = 0ULL;
    for (int t = 0; t < 4; ++t) {
        uint64_t s = src[t] & mask;
        uint64_t old = dst[t];
        uint64_t tmp = old + s;
        uint64_t carry1 = (tmp < old) ? 1ULL : 0ULL;
        uint64_t sum = tmp + carry;
        uint64_t carry2 = (sum < tmp) ? 1ULL : 0ULL;
        dst[t] = sum;
        carry = carry1 | carry2;
    }
}

static __device__ __forceinline__ void mul_4x4(uint64_t *res_low, uint64_t *res_high,
                                               const uint64_t *a, const uint64_t *b) {
    uint64_t tmp[8] = {0};
    for(int i=0;i<4;i++) {
        uint64_t carry = 0;
        for(int j=0;j<4;j++) {
            __uint128_t prod = (__uint128_t)a[i]*b[j] + tmp[i+j] + carry;
            tmp[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        tmp[i+4] = carry;
    }
    for(int k=0;k<4;k++) res_low[k] = tmp[k];
    for(int k=0;k<4;k++) res_high[k] = tmp[k+4];
}

static __device__ __forceinline__ void transition_matrix_4(int32_t *delta, uint64_t *u, uint64_t *v, uint64_t t[4], int N) {
    uint64_t m00[4], m01[4], m10[4], m11[4];
    set_ui_4(m00, 1ULL);
    zero_4(m01);
    zero_4(m10);
    set_ui_4(m11, 1ULL);

    uint64_t u_copy[4], v_copy[4];
    copy_4(u_copy, u);
    copy_4(v_copy, v);

    for (int i = 0; i < N; i++) {
        uint64_t v_odd = v_copy[0] & 1ULL;
        if (*delta > 0 && v_odd) {
            int32_t delta_new = 1 - *delta;
            uint64_t tmp[4];
            copy_4(tmp, u_copy);
            copy_4(u_copy, v_copy);
            sub_and_shr1_4(v_copy, tmp, v_copy);

            uint64_t tmp_m[4];
            copy_4(tmp_m, m10); mul_4x4(m10, tmp, m10, &m10[0]);
            copy_4(tmp_m, m11); mul_4x4(m11, tmp, m11, &m11[0]);
            copy_4(m00, tmp_m);
            copy_4(m01, tmp_m);

            *delta = delta_new;
        } else if (v_odd) {
            *delta += 1;
            uint64_t tmp[4];
            sub_and_shr1_4(v_copy, v_copy, u_copy);

            uint64_t tmp_m[4];
            copy_4(tmp_m, m10); mul_4x4(m10, tmp, m10, &m10[0]);
            copy_4(tmp_m, m11); mul_4x4(m11, tmp, m11, &m11[0]);
        } else {
            *delta += 1;
            shr1_4(v_copy);
            mul_4x4(m00, m00, m00, &m00[0]);
            mul_4x4(m01, m01, m01, &m01[0]);
        }
    }

    copy_4(&t[0], m00);
    copy_4(&t[1], m01);
    copy_4(&t[2], m10);
    copy_4(&t[3], m11);
}

static __device__ __forceinline__ void div2n_4(uint64_t *res, const uint64_t *x_low, const uint64_t *x_high, const uint64_t *p, const uint64_t *p_inv, int N) {

    uint64_t x_mod[4];
    copy_4(x_mod, x_low);

    uint64_t m[4] = {0};
    m[0] = (x_mod[0] * p_inv[0]) & ((1ULL << N) - 1);
    m[1] = m[2] = m[3] = 0ULL;

    uint64_t borrow = 0;
    for(int i=0;i<4;i++){
        __uint128_t t = (__uint128_t)x_low[i] - (__uint128_t)m[i]*p[i] - borrow;
        res[i] = (uint64_t)t;
        borrow = (t >> 127) & 1ULL;
    }

    for(int i=0;i<N;i++) shr1_4(res);
}

static __device__ __forceinline__ void update_x1x2_optimized_ver2_4(uint64_t *x1, uint64_t *x2, const uint64_t t[4], const uint64_t *p, const uint64_t *p_inv, int N) {

    uint64_t x1n_low[4], x1n_high[4], x2n_low[4], x2n_high[4];
    uint64_t tmp_low[4], tmp_high[4];

    mul_4x4(x1n_low, x1n_high, x1, &t[0]);
    mul_4x4(tmp_low, tmp_high, x2, &t[1]);
    uint64_t carry = 0;
    for(int i=0;i<4;i++){
        __uint128_t sum = (__uint128_t)x1n_low[i] + tmp_low[i] + carry;
        x1n_low[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }

    mul_4x4(x2n_low, x2n_high, x1, &t[2]);
    mul_4x4(tmp_low, tmp_high, x2, &t[3]);
    carry = 0;
    for(int i=0;i<4;i++){
        __uint128_t sum = (__uint128_t)x2n_low[i] + tmp_low[i] + carry;
        x2n_low[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }

    div2n_4(x1, x1n_low, x1n_high, p, p_inv, N);
    div2n_4(x2, x2n_low, x2n_high, p, p_inv, N);

    for(int i=0;i<4;i++){
        if((int64_t)x1[i] < 0) x1[i] += p[i];
        if((int64_t)x2[i] < 0) x2[i] += p[i];
    }
}

static __device__ __forceinline__ void normalize_4(uint64_t *res, uint64_t *v, int32_t sign, const uint64_t *p) {
    if(v[3] >> 63) add_cond_4(v, p, 0xFFFFFFFFFFFFFFFFULL);
    if(sign == -1){
        for(int i=0;i<4;i++) v[i] = ~v[i]+1;
    }
    if(v[3] >> 63) add_cond_4(v, p, 0xFFFFFFFFFFFFFFFFULL);
    copy_4(res,v);
}

__device__ void mod_inverse_p(uint64_t *result, const uint64_t *a_normal) {

    const uint64_t p[4] = {
        0xFFFFFFFEFFFFFC2FULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };

    const int N = 17;
    const uint64_t p_inv[4] = {0x00000001000003D1ULL, 0ULL, 0ULL, 0ULL};

    if (is_zero_4(a_normal)) { 
        zero_4(result); 
        return; 
    }

    int32_t delta = 1;
    const int d = 256;
    const int m = (49*d + 57 + 16)/17;

    uint64_t f[4], g[4], x1[4], x2[4];
    copy_4(f, a_normal);
    copy_4(g, p);
    zero_4(x1);
    set_ui_4(x2, 1ULL);

    uint64_t t[4];

    for(int i = 0; i < m; i++) {

        uint64_t g_odd = g[0] & 1ULL;
        int32_t swap_flag = (delta > 0 && g_odd) ? 1 : 0;

        uint64_t mask = 0ULL - (uint64_t)swap_flag;
        uint64_t inv_mask = ~mask;

        uint64_t f_temp[4], g_temp[4], x1_temp[4], x2_temp[4];
        copy_4(f_temp, f); copy_4(g_temp, g);
        copy_4(x1_temp, x1); copy_4(x2_temp, x2);

        for(int t_idx = 0; t_idx < 4; t_idx++){
            f[t_idx]  = (g_temp[t_idx] & mask) | (f_temp[t_idx] & inv_mask);
            g[t_idx]  = (f_temp[t_idx] & mask) | (g_temp[t_idx] & inv_mask);
            x1[t_idx] = (x2_temp[t_idx] & mask) | (x1_temp[t_idx] & inv_mask);
            x2[t_idx] = (x1_temp[t_idx] & mask) | (x2_temp[t_idx] & inv_mask);
        }

        if (swap_flag) {
            delta = 1 - delta;
        } else {
            delta = delta + 1;
        }

        uint64_t g_odd_mask = 0ULL - g_odd;
        add_cond_4(g, f, g_odd_mask);
        add_cond_4(x2, x1, g_odd_mask);

        shr1_4(g);

        transition_matrix_4(&delta, f, g, t, N);

        update_x1x2_optimized_ver2_4(x1, x2, t, p, p_inv, N);
    }

    normalize_4(result, x2, (delta > 0) ? 1 : -1, p);
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

    uint64_t z_norm[4], z_inv[4], z_inv_sqr[4], z_inv_cube[4];

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

__global__ void test_inverse_kernel(uint64_t *a, uint64_t *result) {
    mod_inverse_p(result, a);
}

int main() {

    //LSB 130 Bits:
    uint64_t h_priv[4] = {
    0x28b88cf897c603c9ULL,
    0x3e7665705359f04fULL,
    0x0000000000000003ULL,
    0x0000000000000000ULL
    };

    uint64_t h_result[4];
    uint64_t *d_priv = nullptr;
    uint64_t *d_result = nullptr;

    cudaMalloc((void**)&d_priv, 4 * sizeof(uint64_t));
    cudaMalloc((void**)&d_result, 4 * sizeof(uint64_t));
    cudaMemcpy(d_priv, h_priv, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    test_inverse_kernel<<<1,1>>>(d_priv, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    //Print in MSB
    printf("%016llx%016llx%016llx%016llx\n",
           h_result[3], h_result[2], h_result[1], h_result[0]);

    cudaFree(d_priv);
    cudaFree(d_result);

    cudaDeviceReset();
    return 0;
}