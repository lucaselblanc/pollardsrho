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

__host__ __device__ void point_init_jacobian(ECPointJacobian *P);
__host__ __device__ void point_add_jacobian(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q);
__host__ __device__ void point_double_jacobian(ECPointJacobian *R, const ECPointJacobian *P);
__host__ __device__ void scalar_mult_jacobian(ECPointJacobian *R, const uint64_t *k, int n_bits);
__host__ __device__ void get_compressed_public_key(unsigned char *out, const ECPoint *public_key);
__host__ __device__ void init_precomp_g();

#endif /* EC_SECP256K1_H */