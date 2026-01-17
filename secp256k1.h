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
} ECPoint;

typedef struct {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
    int infinity;
} ECPointJacobian;

#ifdef __CUDA_ARCH__
    __device__ extern ECPointJacobian* preCompG;
    __device__ extern ECPointJacobian* preCompGphi;
    __device__ extern ECPointJacobian* jacNorm;
    __device__ extern ECPointJacobian* jacEndo;
#else
    extern ECPointJacobian* preCompG;
    extern ECPointJacobian* preCompGphi;
    extern ECPointJacobian* jacNorm;
    extern ECPointJacobian* jacEndo;
#endif

uint256_t almostinverse(uint256_t base, uint256_t mod);

__host__ __device__ void pointInitJacobian(ECPointJacobian *P);
__host__ __device__ void pointAddJacobian(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q);
__host__ __device__ void pointDoubleJacobian(ECPointJacobian *R, const ECPointJacobian *P);
__host__ __device__ void scalarMultJacobian(ECPointJacobian *R, const uint64_t *k, int windowSize);
__host__ __device__ void scalarMul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
__host__ __device__ void serializePublicKey(unsigned char *out, const ECPoint *publicKey);
__host__ __device__ void generatePublicKey(unsigned char *out, const uint64_t *PRIV_KEY, int windowSize);
__host__ __device__ void decompressPublicKey(ECPoint* out, const unsigned char compressed[33]);
__host__ __device__ void jacobianToAffine(ECPoint *aff, const ECPointJacobian *jac);
__host__ __device__ void initPrecompG(int windowSize);

#endif /* EC_SECP256K1_H */
