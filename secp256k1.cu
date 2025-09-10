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

//mod_inverse

struct uint256_t {
    __uint128_t low;
    __uint128_t high;

    __device__ __host__ uint256_t() : low(0), high(0) {}
    __device__ __host__ uint256_t(__uint128_t l) : low(l), high(0) {}
    __device__ __host__ uint256_t(__uint128_t h, __uint128_t l) : high(h), low(l) {}
};

struct Fraction256 {
    uint256_t num;
    uint256_t den;
    bool negative;

    __device__ Fraction256() : num(0), den(1), negative(false) {}
    __device__ Fraction256(int64_t n) : num(n<0 ? uint256_t(-n) : uint256_t(n)), den(1), negative(n<0) {}
};

__device__ bool is_zero(const uint256_t& a) {
    return a.high == 0 && a.low == 0;
}

__device__ bool is_odd(const uint256_t& a) {
    return (a.low & 1) != 0;
}

__device__ bool is_negative(const uint256_t& a) {
    return (a.high & ((__uint128_t)1 << 127)) != 0;
}

__device__ uint256_t negate256(const uint256_t& a) {
    uint256_t res;
    res.low = ~a.low + 1;
    res.high = ~a.high + (res.low == 0 ? 1 : 0);
    return res;
}

__device__ bool lt256(const uint256_t& a, const uint256_t& b) {
    if(a.high < b.high) return true;
    if(a.high > b.high) return false;
    return a.low < b.low;
}

__device__ uint256_t mul256(const uint256_t& a, const uint256_t& b) {
    __uint128_t a_lo = a.low;
    __uint128_t a_hi = a.high;
    __uint128_t b_lo = b.low;
    __uint128_t b_hi = b.high;
    __uint128_t lo_lo = a_lo * b_lo;
    __uint128_t lo_hi = a_lo * b_hi;
    __uint128_t hi_lo = a_hi * b_lo;
    __uint128_t hi_hi = a_hi * b_hi;
    __uint128_t mid = lo_hi + hi_lo;
    bool carry_mid = (mid < lo_hi);

    __uint128_t res_low = lo_lo + (mid << 64);
    bool carry_low = (res_low < lo_lo);

    __uint128_t res_high = hi_hi + (mid >> 64) + (carry_mid ? 1 : 0) + (carry_low ? 1 : 0);

    return uint256_t(res_high, res_low);
}

__device__ uint256_t mul256_hi(const uint256_t& a, const uint256_t& b) {
    __uint128_t a_lo = a.low;
    __uint128_t a_hi = a.high;
    __uint128_t b_lo = b.low;
    __uint128_t b_hi = b.high;

    __uint128_t lo_lo = a_lo * b_lo;
    __uint128_t lo_hi = a_lo * b_hi;
    __uint128_t hi_lo = a_hi * b_lo;
    __uint128_t hi_hi = a_hi * b_hi;

    __uint128_t mid = lo_hi + hi_lo;
    bool carry_mid = (mid < lo_hi);

    __uint128_t carry_low = (lo_lo + (mid << 64)) < lo_lo;

    __uint128_t high = hi_hi + (mid >> 64) + (carry_mid ? 1 : 0) + (carry_low ? 1 : 0);
    __uint128_t low = lo_lo + (mid << 64);

    return uint256_t(high, low);
}

__device__ uint256_t sub256(const uint256_t& a, const uint256_t& b) {
    uint256_t res;
    res.low = a.low - b.low;
    res.high = a.high - b.high - (a.low < b.low ? 1 : 0);
    return res;
}

__device__ uint256_t mod256(const uint256_t& x, const uint256_t& m) {
    //0x100000000000000000000000000000000000000000000000000000001000003d1 => {2^512 / secp256k1 p}
    uint256_t MU;
    MU.high = (__uint128_t)0x1000000000000000ULL << 64;
    MU.low  = 0x00000000000000001000003d1ULL;

    uint256_t q = mul256_hi(x, MU);
    uint256_t r = sub256(x, mul256(q, m));
    if(lt256(r, m)) return r;
    return sub256(r, m);
}

__device__ uint256_t add256(const uint256_t& a, const uint256_t& b) {
    uint256_t res;
    res.low = a.low + b.low;
    res.high = a.high + b.high + (res.low < a.low ? 1 : 0);
    return res;
}

__device__ uint256_t shiftr256(const uint256_t& a, int s) {
    if (s == 0) return a;
    if (s >= 256) return uint256_t(0);
    if (s < 128)
        return uint256_t(a.high >> s, (a.low >> s) | (a.high << (128 - s)));
    else
        return uint256_t(0, a.high >> (s - 128));
}

__device__ uint256_t shiftl256(const uint256_t& a, int s) {
    if (s == 0) return a;
    if (s >= 256) return uint256_t(0);
    if (s < 128)
        return uint256_t((a.high << s) | (a.low >> (128 - s)), a.low << s);
    else
        return uint256_t(a.low << (s - 128), 0);
}

__device__ uint256_t div2_uint256(uint256_t x) {
    if(x.low & 1) x = add256(x, uint256_t(1));
    return shiftr256(x,1);
}

__device__ Fraction256 add_frac(const Fraction256& a, const Fraction256& b) {
    Fraction256 res;
    uint256_t na = mul256(a.num, b.den);
    uint256_t nb = mul256(b.num, a.den);

    if (a.negative == b.negative) {
        res.num = add256(na, nb);
        res.negative = a.negative;
    } else {
        if (lt256(na, nb)) { res.num = sub256(nb, na); res.negative = b.negative; }
        else { res.num = sub256(na, nb); res.negative = a.negative; }
    }
    res.den = mul256(a.den, b.den);
    return res;
}

__device__ Fraction256 div2_frac(const Fraction256& a) {
    Fraction256 res = a;

    if ((res.num.low & 1) == 0) {
        res.num = shiftr256(res.num, 1);
    } else {
        res.den = mul256(res.den, uint256_t(2));
    }
    return res;
}

__device__ uint256_t truncate_cuda(uint256_t f, int t) {
    if(t == 0) return uint256_t(0);
    if(t >= 256) return f;

    uint256_t mask;
    if(t <= 128) mask = uint256_t(0, ((__uint128_t)1 << t) - 1);
    else {
        __uint128_t high_mask = ((__uint128_t)1 << (t - 128)) - 1;
        __uint128_t low_mask = ~((__uint128_t)0) >> (256 - t);
        mask = uint256_t(high_mask, low_mask);
    }

    uint256_t res = uint256_t(f.high & mask.high, f.low & mask.low);

    bool neg;
    if(t <= 128) neg = res.low >= ((__uint128_t)1 << (t-1));
    else neg = (res.high & ((__uint128_t)1 << (t-129))) != 0;

    if(neg) {
        if(t <= 128) res.low -= ((__uint128_t)1 << t);
        else res.high -= ((__uint128_t)1 << (t-128));
    }
    return res;
}

__device__ void divsteps2_cuda(int n, int t, int* delta, uint256_t* f, uint256_t* g,
                               Fraction256* u, Fraction256* v, Fraction256* q, Fraction256* r) {
    *f = truncate_cuda(*f, t);
    *g = truncate_cuda(*g, t);
    *u = Fraction256(1); *v = Fraction256(0); *q = Fraction256(0); *r = Fraction256(1);

    while(n>0) {
        *f = truncate_cuda(*f, t);
        if(*delta>0 && is_odd(*g)) {
            int temp_delta = -*delta;
            uint256_t temp_f = *g;
            uint256_t temp_g = negate256(*f);
            Fraction256 temp_u = *q;
            Fraction256 temp_v = *r;
            Fraction256 temp_q = *u; temp_q.negative = !temp_q.negative;
            Fraction256 temp_r = *v; temp_r.negative = !temp_r.negative;
            *delta = temp_delta; *f = temp_f; *g = temp_g;
            *u = temp_u; *v = temp_v; *q = temp_q; *r = temp_r;
        }
        bool g0 = is_odd(*g);
        *delta = 1 + *delta;
        if(g0) {
            *g = shiftr256(add256(*g, *f),1);
            *q = div2_frac(add_frac(*q,*u));
            *r = div2_frac(add_frac(*r,*v));
        } else {
            *g = shiftr256(*g,1);
            *q = div2_frac(*q);
            *r = div2_frac(*r);
        }
        n--; t--;
        *g = truncate_cuda(*g, t);
    }
}

__device__ uint256_t powmod256(uint256_t base, uint256_t exp, uint256_t mod) {
    uint256_t result(1);
    while(!is_zero(exp)) {
        if(is_odd(exp)) result = mod256(mul256(result,base),mod);
        base = mod256(mul256(base,base),mod);
        exp = shiftr256(exp,1);
    }
    return result;
}

__device__ int iterations_cuda(int d) {
    if(d < 46) return (49 * d + 80) / 17;
    else       return (49 * d + 57) / 17;
}

__device__ void almost_inverse_p(uint256_t f, uint256_t g, uint256_t* result) {
    int d = 256; 
    int m = iterations_cuda(d);

    uint256_t precomp = powmod256(div2_uint256(add256(f, uint256_t(1))), uint256_t(m-1), f);

    int delta = 1;
    uint256_t f_work = f, g_work = g;
    Fraction256 u,v,q,r;
    divsteps2_cuda(m, m+1, &delta, &f_work, &g_work, &u,&v,&q,&r);

    uint256_t V_int = shiftl256(v.num, m-1);
    if(v.negative ^ is_negative(f_work)) V_int = negate256(V_int);

    *result = mod256(mul256(V_int,precomp), f);
}

//Final

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

__global__ void test_mod_inverse(uint256_t f, uint256_t g, uint256_t* result) {
    almost_inverse_p(f, g, result);
}

int main() {

    // f = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    uint256_t f(
        (((__uint128_t)0xFFFFFFFFFFFFFFFFULL) << 64) | (__uint128_t)0xFFFFFFFFFFFFFFFFULL,
        (((__uint128_t)0xFFFFFFFFFFFFFFFFULL) << 64) | (__uint128_t)0xFFFFFFFEFFFFFC2FULL
    );

    // g = 0x33e7665705359f04f28b88cf897c603c9
    uint256_t g(
        (__uint128_t)0x33e7665705359f04ULL,  // high 128 bits
        ((__uint128_t)0xf << 64) | (__uint128_t)0x28b88cf897c603c9ULL
    );

    /* g ≡ 1 (mod f): */
    //Hex: 7FDB62ED2D6FA0874ABD664C95B7CEF2ED79CC82D13FF3AC8E9766AA21BEBEAE (bebeae kkk)
    //Dec: 57831354042695616917422878622316954017183908093256327737334808907053491207854

    uint256_t result_host;
    uint256_t* result_device;
    cudaMalloc(&result_device, sizeof(uint256_t));

    test_mod_inverse<<<1,1>>>(f, g, result_device);

    cudaMemcpy(&result_host, result_device, sizeof(uint256_t), cudaMemcpyDeviceToHost);

    printf("Resultado high: %llx\n", (unsigned long long)result_host.high);
    printf("Resultado low : %llx\n", (unsigned long long)result_host.low);

    cudaFree(result_device);
    return 0;
}
