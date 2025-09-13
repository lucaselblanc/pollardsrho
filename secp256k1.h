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

#include <cuda_runtime.h>

typedef struct {
    uint64_t x[4];
    uint64_t y[4];
    int infinity;
} ECPoint;

__global__ void point_init(ECPoint *point);
__global__ void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q);
__global__ void point_double(ECPoint *R, const ECPoint *P);
__global__ void scalar_mult(ECPoint *R, const unsigned int *k, const ECPoint *P);
__global__ void point_is_valid(int *result, const ECPoint *point);
__global__ void get_compressed_public_key(unsigned char *out, const ECPoint *pub);

//test
__global__ void test_mod_inverse(__uint256_t f, __uint256_t g, __uint256_t* result);

#endif /* EC_SECP256K1_H */
