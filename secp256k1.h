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

#ifndef EC_SECP256K1_H
#define EC_SECP256K1_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned int x[8];
    unsigned int y[8];
    int infinity;
} ECPoint;

void point_init(ECPoint *point);
void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q);
void point_double(ECPoint *R, const ECPoint *P);
void scalar_mult(ECPoint *R, const char *k_hex, const ECPoint *P);
int point_is_valid(const ECPoint *point);
void get_compressed_public_key(unsigned char *out, const ECPoint *public_key);

#ifdef __cplusplus
}
#endif

#endif /* EC_SECP256K1_H */
