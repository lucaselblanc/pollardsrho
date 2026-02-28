/******************************************************************************************************
 * This file is part of the Pollard's Rho distribution: (https://github.com/lucaselblanc/pollardsrho) *
 * Copyright (c) 2024, 2026 Lucas Leblanc.                                                            *
 * Distributed under the MIT software license, see the accompanying.                                  *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.                               *
 ******************************************************************************************************/

/*****************************************
 * Pollard's Rho Algorithm for SECP256K1 *
 * Written by Lucas Leblanc              *
******************************************/

#ifndef EC_SECP256K1_H
#define EC_SECP256K1_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

struct uint256_t {
    uint64_t limbs[4];
};

typedef struct {
    uint64_t x[4];
    uint64_t y[4];
    int infinity;
} ECPointAffine;

typedef struct {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
    int infinity;
} ECPointJacobian;

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

#ifdef __CUDA_ARCH__
    __device__ extern ECPointJacobian* preCompG;
    __device__ extern ECPointJacobian* preCompGphi;
    __device__ extern ECPointJacobian* preCompH;
    __device__ extern ECPointJacobian* preCompHphi;
    __device__ extern ECPointJacobian* jacNorm;
    __device__ extern ECPointJacobian* jacEndo;
    __device__ extern ECPointJacobian* jacNormH;
    __device__ extern ECPointJacobian* jacEndoH;
#else
    extern ECPointJacobian* preCompG;
    extern ECPointJacobian* preCompGphi;
    extern ECPointJacobian* preCompH;
    extern ECPointJacobian* preCompHphi;
    extern ECPointJacobian* jacNorm;
    extern ECPointJacobian* jacEndo;
    extern ECPointJacobian* jacNormH;
    extern ECPointJacobian* jacEndoH;
#endif

uint256_t almostinverse(uint256_t base, uint256_t mod);

__host__ __device__ void affineToJacobian(ECPointJacobian *jac, const ECPointAffine *aff);
__host__ __device__ void decompressPublicKey(ECPointAffine* out, const unsigned char compressed[33]);
__host__ __device__ void endomorphismMap(ECPointJacobian *R, const ECPointJacobian *P);
__host__ __device__ void fromMontgomeryP(uint64_t *result, const uint64_t *a);
__host__ __device__ void generatePublicKey(ECPointJacobian *preCompTable, ECPointJacobian *preCompTablePhi, unsigned char *out, const uint64_t *PRIV_KEY, int windowSize);
__host__ __device__ void initPreCompG(int windowSize);
__host__ __device__ void initPreCompH(const ECPointJacobian *h, int windowSize);
__host__ __device__ void jacobianScalarMult(ECPointJacobian *result, ECPointJacobian *preCompTable, const uint64_t *scalar, int windowSize);
__host__ __device__ void jacobianScalarMultGlv(ECPointJacobian *R, ECPointJacobian *preCompTable, ECPointJacobian *preCompTablePhi, const uint64_t k[4], int windowSize);
__host__ __device__ void jacobianDouble(ECPointJacobian *R, const ECPointJacobian *P);
__host__ __device__ void jacobianAdd(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q);
__host__ __device__ void jacobianToAffine(ECPointAffine *aff, const ECPointJacobian *jac);
__host__ __device__ void jacobianSetInfinity(ECPointJacobian *point);
__host__ __device__ bool jacobianIsInfinity(const ECPointJacobian *P);
__host__ __device__ void modMulMontP(uint64_t *result, const uint64_t *a, const uint64_t *b);
__host__ __device__ void modSubP(uint64_t *result, const uint64_t *a, const uint64_t *b);
__host__ __device__ void modAddP(uint64_t *result, const uint64_t *a, const uint64_t *b);
__host__ __device__ void modExpMontP(uint64_t *res, const uint64_t *base, const uint64_t *exp);
__host__ __device__ void pointInitJacobian(ECPointJacobian *P);
__host__ __device__ void pointAddJacobian(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q);
__host__ __device__ void pointDoubleJacobian(ECPointJacobian *R, const ECPointJacobian *P);
__host__ __device__ void scalarReduceN(uint64_t *r, const uint64_t *k);
__host__ __device__ void scalarMultJacobian(ECPointJacobian *R, ECPointJacobian *preCompTable, ECPointJacobian *preCompTablePhi, const uint64_t *k, int windowSize);
__host__ __device__ void scalarMul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
__host__ __device__ void scalarAdd(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
__host__ __device__ void scalarSub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
__host__ __device__ void scalarNeg(uint64_t r[4], const uint64_t a[4]);
__host__ __device__ int scalarIsZero(const uint64_t a[4]);
__host__ __device__ void serializePublicKey(unsigned char *out, const ECPointAffine *publicKey);
__host__ __device__ void toMontgomeryP(uint64_t *result, const uint64_t *a);

#endif /* EC_SECP256K1_H */