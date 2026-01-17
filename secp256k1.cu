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

/* --- AINDA EM TESTES DE OTIMIZAÇÃO --- */

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
__device__ ECPointJacobian* preCompG;
__device__ ECPointJacobian* preCompGphi;
__device__ ECPointJacobian* jacNorm;
__device__ ECPointJacobian* jacEndo;
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
ECPointJacobian* preCompG = nullptr;
ECPointJacobian* preCompGphi = nullptr;
ECPointJacobian* jacNorm = nullptr;
ECPointJacobian* jacEndo = nullptr;
#endif

#ifdef __CUDA_ARCH__
struct uint128_t {
    uint64_t lo;
    uint64_t hi;

    __host__ __device__ uint128_t(uint64_t x = 0) : lo(x), hi(0) {}
    __host__ __device__ uint128_t operator*(const uint128_t& other) const { uint128_t res; res.lo = lo * other.lo; res.hi = __umul64hi(lo, other.lo); return res; }
    __host__ __device__ uint128_t operator*(uint64_t other) const { return *this * uint128_t(other); }
    __host__ __device__ uint128_t& operator*=(const uint128_t& other) { *this = *this * other; return *this; }
    __host__ __device__ uint128_t& operator*=(uint64_t other) { *this = *this * other; return *this; }
    __host__ __device__ uint128_t operator+(const uint128_t& other) const { uint128_t res; res.lo = lo + other.lo; res.hi = hi + other.hi + (res.lo < lo ? 1 : 0); return res; }
    __host__ __device__ uint128_t operator+(uint64_t other) const { return *this + uint128_t(other); }
    __host__ __device__ uint128_t& operator+=(const uint128_t& other) { uint64_t old_lo = lo; lo += other.lo; hi += other.hi + (lo < old_lo ? 1 : 0); return *this; }
    __host__ __device__ uint128_t& operator+=(uint64_t other) { *this += uint128_t(other); return *this; }
    __host__ __device__ uint128_t operator-(const uint128_t& other) const { uint128_t res; res.lo = lo - other.lo; res.hi = hi - other.hi - (lo < other.lo ? 1 : 0); return res; }
    __host__ __device__ uint128_t operator-(uint64_t other) const { return *this - uint128_t(other); }
    __host__ __device__ uint128_t& operator-=(const uint128_t& other) { uint64_t old_lo = lo; lo -= other.lo; hi -= other.hi + (old_lo < other.lo ? 1 : 0); return *this; }
    __host__ __device__ uint128_t& operator-=(uint64_t other) { *this -= uint128_t(other); return *this; }
    __host__ __device__ uint128_t operator>>(const unsigned int n) const { uint128_t res; if (n == 0) return *this; else if (n < 64) { res.lo = (lo >> n) | (hi << (64 - n)); res.hi = hi >> n; } else if (n < 128) { res.lo = hi >> (n - 64); res.hi = 0; } else { res.lo = 0; res.hi = 0; } return res; }
    __host__ __device__ uint128_t& operator>>=(const unsigned int n) { *this = *this >> n; return *this; }
    __host__ __device__ uint128_t operator<<(const unsigned int n) const { uint128_t res; if (n == 0) return *this; else if (n < 64) { res.hi = (hi << n) | (lo >> (64 - n)); res.lo = lo << n; } else if (n < 128) { res.hi = lo << (n - 64); res.lo = 0; } else { res.lo = 0; res.hi = 0; } return res; }
    __host__ __device__ uint128_t& operator<<=(const unsigned int n) { *this = *this << n; return *this; }
    __host__ __device__ operator uint64_t() const { return lo; }
    __host__ __device__ bool operator<(const uint128_t& other) const { return (hi < other.hi) || (hi == other.hi && lo < other.lo); }
    __host__ __device__ bool operator<=(const uint128_t& other) const { return (hi < other.hi) || (hi == other.hi && lo <= other.lo); }
    __host__ __device__ bool operator==(const uint128_t& other) const { return hi == other.hi && lo == other.lo; }
    __host__ __device__ bool operator!=(const uint128_t& other) const { return !(*this == other); }
    __host__ __device__ bool operator<(uint64_t other) const { return *this < uint128_t(other); }
    __host__ __device__ bool operator<=(uint64_t other) const { return *this <= uint128_t(other); }
    __host__ __device__ bool operator==(uint64_t other) const { return *this == uint128_t(other); }
    __host__ __device__ bool operator!=(uint64_t other) const { return *this != uint128_t(other); }
};
#else
using uint128_t = unsigned __int128;
#endif

__host__ __device__ void montgomeryReduceP(uint64_t *result, const uint64_t *inputHigh, const uint64_t *inputLow) {
    uint64_t temp[8];
    for (int i = 0; i < 4; i++) {
        temp[i]     =  inputLow[i];
        temp[i + 4] = inputHigh[i];
    }

    for (int i = 0; i < 4; i++) {
        uint64_t ui = (uint64_t)((uint128_t)temp[i] * (uint128_t)MU_P);
        uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)ui * (uint128_t)P_CONST[j] + (uint128_t)temp[i + j] + carry;
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

__host__ __device__ void toMontgomeryP(uint64_t *result, const uint64_t *a) {
    uint64_t aLocal[4];
    for (int i = 0; i < 4; i++) aLocal[i] = a[i];

    uint64_t temp[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint128_t carry = 0;

        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)aLocal[i] * (uint128_t)R2_MOD_P[j] + (uint128_t)temp[i + j] + carry;
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

    montgomeryReduceP(result, high, low);
}

__host__ __device__ void fromMontgomeryP(uint64_t *result, const uint64_t *a) {
    uint64_t zero[4] = {0, 0, 0, 0};
    montgomeryReduceP(result, zero, a);
}

__host__ __device__ void modAddP(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint128_t carry = 0;

    for (int i = 0; i < 4; ++i) {
        uint128_t s = (uint128_t)a[i] + (uint128_t)b[i] + carry;
        result[i] = (uint64_t)s;
        carry = s >> 64;
    }

    bool ge = (bool)carry;
    if (!ge) {
        for (int i = 3; i >= 0; --i) {
            if (result[i] > P_CONST[i]) { ge = true; break; }
            if (result[i] < P_CONST[i]) { ge = false; break; }
        }
    }

    if (ge) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            uint128_t sub = (uint128_t)result[i] - (uint128_t)P_CONST[i] - borrow;
            result[i] = (uint64_t)sub;
            borrow = ((uint128_t)result[i] < (uint128_t)P_CONST[i] + borrow) ? 1 : 0;
        }
    }
}

__host__ __device__ void modSubP(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t borrow = 0;

    for (int i = 0; i < 4; ++i) {
        uint128_t sub = (uint128_t)a[i] - (uint128_t)b[i] - borrow;
        result[i] = (uint64_t)sub;
        borrow = ((uint128_t)a[i] < (uint128_t)b[i] + borrow) ? 1 : 0;
    }

    if (borrow) {
        uint128_t carry = 0;
        for (int i = 0; i < 4; ++i) {
            uint128_t s = (uint128_t)result[i] + (uint128_t)P_CONST[i] + carry;
            result[i] = (uint64_t)s;
            carry = s >> 64;
        }
    }
}

__host__ __device__ void modMulMontP(uint64_t *result, const uint64_t *a, const uint64_t *b) {
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

    montgomeryReduceP(result, high, low);
}

__host__ __device__ void modSqrMontP(uint64_t *out, const uint64_t *in) {
    modMulMontP(out, in, in);
}

__host__ __device__ void scalarReduceN(uint64_t *r, const uint64_t *k) {
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

__host__ __device__ void modExpMontP(uint64_t *res, const uint64_t *base, const uint64_t *exp) {
    uint64_t one[4] = {0};
    one[0] = 1ULL;
    toMontgomeryP(res, one);

    uint64_t acc[4];
    for (int i = 0; i < 4; i++) acc[i] = base[i];

    for (int word = 3; word >= 0; word--) {
        for (int bit = 63; bit >= 0; bit--) {
            modSqrMontP(res, res);

            if ((exp[word] >> bit) & 1ULL) {
                modMulMontP(res, res, acc);
            }
        }
    }
}

__host__ __device__ void sqrtModP(uint64_t y[4], const uint64_t v[4]) {
    uint64_t exp[4];

    exp[0] = 0x0FFFFFFFFFFFFFFF;
    exp[1] = 0xFFFFFFFFFFFFFFFF;
    exp[2] = 0xFFFFFFFFFFFFFFFF;
    exp[3] = 0x3FFFFFFF0000000C;

    modExpMontP(y, v, exp);
}

__host__ __device__ void jacobianInit(ECPointJacobian *point) {
    for (int i = 0; i < 4; i++) {
        point->X[i] = 0;
        point->Y[i] = 0;
        point->Z[i] = ONE_MONT[i];
    }
    point->infinity = 0;
}

__host__ __device__ void jacobianSetInfinity(ECPointJacobian *point) {
    for (int i = 0; i < 4; i++) {
        point->X[i] = ONE_MONT[i];
        point->Y[i] = ONE_MONT[i];
        point->Z[i] = 0;
    }
    point->infinity = 1;
}

__host__ __device__ int jacobianIsInfinity(const ECPointJacobian *point) {
    uint64_t z_zero = 0;
    for (int i = 0; i < 4; i++) {
        z_zero |= point->Z[i];
    }
    return point->infinity || (z_zero == 0);
}

__host__ __device__ void jacobianToAffine(ECPoint *aff, const ECPointJacobian *jac) {

    if (jacobianIsInfinity(jac)) {
        for (int i = 0; i < 4; i++)
            aff->x[i] = aff->y[i] = 0;
        aff->infinity = 1;
        return;
    }

    uint64_t zInv[4];
    uint64_t zInv2[4];
    uint64_t zInv3[4];

    modExpMontP(zInv, jac->Z, P_CONST_MINUS_2);
    modMulMontP(zInv2, zInv, zInv);
    modMulMontP(zInv3, zInv2, zInv);
    modMulMontP(aff->x, jac->X, zInv2);
    fromMontgomeryP(aff->x, aff->x);
    modMulMontP(aff->y, jac->Y, zInv3);
    fromMontgomeryP(aff->y, aff->y);

    aff->infinity = 0;
}

__host__ __device__ void jacobianDouble(ECPointJacobian *R, const ECPointJacobian *P) {
    if (jacobianIsInfinity(P) ||
        ((P->Y[0] | P->Y[1] | P->Y[2] | P->Y[3]) == 0)) {
        jacobianSetInfinity(R);
        return;
    }

    uint64_t XX[4], YY[4], YYYY[4], S[4], M[4], T[4];

    modMulMontP(XX, P->X, P->X);
    modMulMontP(YY, P->Y, P->Y);
    modMulMontP(YYYY, YY, YY);
    modMulMontP(S, P->X, YY);
    modAddP(S, S, S);
    modAddP(S, S, S);
    modAddP(M, XX, XX);
    modAddP(M, M, XX);
    modMulMontP(R->X, M, M);
    modSubP(R->X, R->X, S);
    modSubP(R->X, R->X, S);
    modSubP(T, S, R->X);
    modMulMontP(R->Y, M, T);
    modAddP(YYYY, YYYY, YYYY);
    modAddP(YYYY, YYYY, YYYY);
    modAddP(YYYY, YYYY, YYYY);
    modSubP(R->Y, R->Y, YYYY);
    modMulMontP(R->Z, P->Y, P->Z);
    modAddP(R->Z, R->Z, R->Z);

    R->infinity = 0;
}

__host__ __device__ void jacobianAdd(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q) {
    if (jacobianIsInfinity(P)) { *R = *Q; return; }
    if (jacobianIsInfinity(Q)) { *R = *P; return; }

    uint64_t Z1Z1[4], Z2Z2[4], U1[4], U2[4];
    uint64_t Z1Z1Z1[4], Z2Z2Z2[4], S1[4], S2[4];
    uint64_t H[4], Rr[4], HH[4], HHH[4], V[4];

    modMulMontP(Z1Z1, P->Z, P->Z);
    modMulMontP(Z2Z2, Q->Z, Q->Z);
    modMulMontP(U1, P->X, Z2Z2);
    modMulMontP(U2, Q->X, Z1Z1);
    modMulMontP(Z1Z1Z1, Z1Z1, P->Z);
    modMulMontP(Z2Z2Z2, Z2Z2, Q->Z);
    modMulMontP(S1, P->Y, Z2Z2Z2);
    modMulMontP(S2, Q->Y, Z1Z1Z1);
    modSubP(H, U2, U1);
    modSubP(Rr, S2, S1);

    if ((H[0] | H[1] | H[2] | H[3]) == 0) {
        if ((Rr[0] | Rr[1] | Rr[2] | Rr[3]) == 0) {
            jacobianDouble(R, P);
        } else {
            jacobianSetInfinity(R);
        }
        return;
    }

    modMulMontP(HH, H, H);
    modMulMontP(HHH, HH, H);
    modMulMontP(V, U1, HH);
    modMulMontP(R->X, Rr, Rr);
    modSubP(R->X, R->X, HHH);
    modSubP(R->X, R->X, V);
    modSubP(R->X, R->X, V);
    modSubP(R->Y, V, R->X);
    modMulMontP(R->Y, Rr, R->Y);
    modMulMontP(S1, S1, HHH);
    modSubP(R->Y, R->Y, S1);
    modMulMontP(R->Z, P->Z, Q->Z);
    modMulMontP(R->Z, R->Z, H);

    R->infinity = 0;
}

__host__ __device__ void endomorphismMap(ECPointJacobian *R, const ECPointJacobian *P) {
    modMulMontP(R->X, P->X, BETA_P);
    for (int i = 0; i < 4; i++) {
        R->Y[i] = P->Y[i];
        R->Z[i] = P->Z[i];
    }
    R->infinity = P->infinity;
}

__host__ __device__ void initPrecompG(int windowSize) {
    int dnorm = (256 + windowSize - 1) / windowSize;
    int dphi = (128 + windowSize - 1) / windowSize;
    int tableSize = (1 << windowSize) - 1;

    uint64_t gxMont[4], gyMont[4];
    toMontgomeryP(gxMont, GX_CONST);
    toMontgomeryP(gyMont, GY_CONST);

    ECPointJacobian g, ge;
    for (int i = 0; i < 4; i++) {
        g.X[i] = gxMont[i];
        g.Y[i] = gyMont[i];
        g.Z[i] = ONE_MONT[i];

        ge.X[i] = gxMont[i];
        ge.Y[i] = gyMont[i];
        ge.Z[i] = ONE_MONT[i];
    }

    g.infinity  = 0;
    ge.infinity = 0;

    endomorphismMap(&ge, &ge);

    jacNorm[0]  = g;
    jacEndo[0] = ge;

    for (int j = 1; j < windowSize; j++) {
        jacNorm[j]  = jacNorm[j-1];
        jacEndo[j] = jacEndo[j-1];

        for (int i = 0; i < dnorm; i++) {
            jacobianDouble(&jacNorm[j], &jacNorm[j]);
        }
        for (int i = 0; i < dphi; i++) {
            jacobianDouble(&jacEndo[j], &jacEndo[j]);
        }
    }

    for (int i = 1; i <= tableSize; i++) {
        jacobianSetInfinity(&preCompG[i-1]);
        for (int j = 0; j < windowSize; j++) {
            if ((i >> j) & 1) {
                ECPointJacobian tmp;
                jacobianAdd(&tmp, &preCompG[i-1], &jacNorm[j]);
                preCompG[i-1] = tmp;
            }
        }

        jacobianSetInfinity(&preCompGphi[i-1]);
        for (int j = 0; j < windowSize; j++) {
            if ((i >> j) & 1) {
                ECPointJacobian tmp;
                jacobianAdd(&tmp, &preCompGphi[i-1], &jacEndo[j]);
                preCompGphi[i-1] = tmp;
            }
        }
    }
}

__host__ __device__ void jacobianScalarMult(ECPointJacobian *result, const uint64_t *scalar, int nBits, int windowSize) {
    int d = (nBits + windowSize - 1) / windowSize;

    jacobianSetInfinity(result);

    for (int col = d - 1; col >= 0; col--) {
        if (col != d - 1) {
            for (int i = 0; i < windowSize; i++) {
                jacobianDouble(result, result);
            }
        }

        int idx = 0;
        for (int row = 0; row < windowSize; row++) {
            int bitIndex = row * d + col;
            if (bitIndex >= nBits) continue;
            int limb = bitIndex / 64;
            int shift = bitIndex % 64;
            uint64_t bit = (scalar[limb] >> shift) & 1ULL;
            idx |= (bit << row);
        }

        if (idx != 0) {
            ECPointJacobian tmp;
            jacobianAdd(&tmp, result, &preCompG[idx - 1]);
            *result = tmp;
        }
    }
}

__host__ __device__ void jacobianScalarMultPhi(ECPointJacobian *result, const uint64_t *scalar, int nBits, int windowSize) {
    int d = (nBits + windowSize - 1) / windowSize;

    jacobianSetInfinity(result);

    for (int col = d - 1; col >= 0; col--) {
        if (col != d - 1) {
            for (int i = 0; i < windowSize; i++) {
                jacobianDouble(result, result);
            }
        }

        int idx = 0;
        for (int row = 0; row < windowSize; row++) {
            int bitIndex = row * d + col;
            if (bitIndex >= nBits) continue;
            int limb = bitIndex / 64;
            int shift = bitIndex % 64;
            uint64_t bit = (scalar[limb] >> shift) & 1ULL;
            idx |= (bit << row);
        }

        if (idx != 0) {
            ECPointJacobian tmp;
            jacobianAdd(&tmp, result, &preCompGphi[idx - 1]);
            *result = tmp;
        }
    }
}

__host__ __device__ void scalarMulShiftVar(uint64_t r[4], const uint64_t a[4], const uint64_t b[4], int shift) {
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

__host__ __device__ void scalarMul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
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

    scalarReduceN(r, t);
}

__host__ __device__ void scalarAdd(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t tmp = a[i] + b[i] + carry;
        carry = (tmp < a[i]) || (carry && tmp == a[i]);
        r[i] = tmp;
    }
    scalarReduceN(r, r);
}

__host__ __device__ void scalarSub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
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

    scalarReduceN(r, r);
}

__host__ __device__ int scalarIsZero(const uint64_t a[4]) {
    uint64_t acc = 0;
    for (int i = 0; i < 4; i++) {
        acc |= a[i];
    }
    return acc == 0;
}

__host__ __device__ void scalarNeg(uint64_t r[4], const uint64_t a[4]) {
    if (scalarIsZero(a)) {
        r[0] = r[1] = r[2] = r[3] = 0;
        return;
    }
    uint64_t tmp[4];
    scalarSub(tmp, N_CONST, a);
    scalarReduceN(r, tmp);
}

__host__ __device__ void scalarSplitLambda(uint64_t r1[4], uint64_t r2[4], const uint64_t k[4]) {
    uint64_t c1[4], c2[4], t1[4], t2[4];

    scalarMulShiftVar(c1, k, G1, 384);
    scalarMulShiftVar(c2, k, G2, 384);
    scalarMul(t1, c1, MINUS_B1);
    scalarMul(t2, c2, MINUS_B2);
    scalarAdd(r2, t1, t2);
    scalarMul(t1, r2, LAMBDA_N);
    scalarNeg(t1, t1);
    scalarAdd(r1, t1, k);
}

__host__ __device__ void jacobianScalarMultGlv(ECPointJacobian *R, const uint64_t k[4], int nBits, int windowSize) {
    uint64_t r1[4], r2[4];
    scalarSplitLambda(r1, r2, k);
    ECPointJacobian P1, P2;
    jacobianScalarMult(&P1, r1, nBits, windowSize);
    jacobianScalarMultPhi(&P2, r2, nBits, windowSize);
    jacobianAdd(R, &P1, &P2);
}

__host__ __device__ void pointInitJacobian(ECPointJacobian *P) { for (int i = 0; i < 4; i++) { P->X[i] = 0; P->Y[i] = 0; P->Z[i] = 0; } P->infinity = 1; }
__host__ __device__ void pointAddJacobian(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q) { jacobianAdd(R, P, Q); }
__host__ __device__ void pointDoubleJacobian(ECPointJacobian *R, const ECPointJacobian *P) { jacobianDouble(R, P); }
__host__ __device__ void scalarMultJacobian(ECPointJacobian *R, const uint64_t *k, int nBits, int windowSize) { jacobianScalarMultGlv(R, k, nBits, windowSize); }
__host__ __device__ void serializePublicKey(unsigned char *out, const ECPoint *publicKey) {
    unsigned char prefix = (publicKey->y[0] & 1ULL) ? 0x03 : 0x02;
    out[0] = prefix;

    for (int i = 0; i < 4; i++) {
        uint64_t word = publicKey->x[3 - i];
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

__host__ __device__ void generatePublicKey(unsigned char *out, const uint64_t *PRIV_KEY, int nBits, int windowSize) {
    ECPoint pub;
    ECPointJacobian pub_jac;

    jacobianScalarMultGlv(&pub_jac, PRIV_KEY, nBits, windowSize);
    jacobianToAffine(&pub, &pub_jac);
    serializePublicKey(out, &pub);
}

__host__ __device__ void decompressPublicKey(ECPoint* out, const unsigned char compressed[33]) {
    unsigned char prefix = compressed[0];

    for (int i = 0; i < 4; i++) {
        uint64_t word = 0;
        for (int j = 0; j < 8; j++) {
            word = (word << 8) | compressed[1 + i * 8 + j];
        }
        out->x[3 - i] = word;
    }

    uint64_t x_mont[4], x2[4], x3[4], rhs[4];

    toMontgomeryP(x_mont, out->x);
    modMulMontP(x2, x_mont, x_mont);
    modMulMontP(x3, x2, x_mont);
    modAddP(rhs, x3, SEVEN_MONT);

    uint64_t y_mont[4];
    sqrtModP(y_mont, rhs);
    fromMontgomeryP(out->y, y_mont);

    uint64_t is_odd   = out->y[0] & 1ULL;
    uint64_t want_odd = (prefix == 0x03);

    if (is_odd != want_odd) {
        modSubP(out->y, P_CONST, out->y);
    }

    out->infinity = 0;
}