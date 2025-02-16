#ifndef EC_H
#define EC_H

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

#include <gmp.h>

typedef struct { mpz_t x, y; int infinity; } ECPoint;

void point_init(ECPoint *point);
void point_clear(ECPoint *point);
void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q, const mpz_t p);
void point_double(ECPoint *R, const ECPoint *P, const mpz_t p);
void scalar_mult(ECPoint *R, const mpz_t k, const ECPoint *P, const mpz_t p);
void get_compressed_public_key(char *out, const ECPoint *public_key);

int point_is_valid(const ECPoint *point, const mpz_t p);

#endif /* EC_H */