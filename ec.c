#include "ec.h"
#include <gmp.h>
#include <string.h>

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

const char *P_STR = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F";
const char *N_STR = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";
const char *GX_STR = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
const char *GY_STR = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";

void point_init(ECPoint *point) {
    mpz_init(point->x);
    mpz_init(point->y);
    point->infinity = 0;
}

void point_clear(ECPoint *point) {
    mpz_clear(point->x);
    mpz_clear(point->y);
}

void modular_inverse(mpz_t result, const mpz_t a, const mpz_t mod) {
    mpz_invert(result, a, mod);
}

void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q, const mpz_t p) {

    if (P->infinity) {
        mpz_set(R->x, Q->x);
        mpz_set(R->y, Q->y);
        R->infinity = Q->infinity;
        return;
    }

    if (Q->infinity) {
        mpz_set(R->x, P->x);
        mpz_set(R->y, P->y);
        R->infinity = P->infinity;
        return;
    }

    if (mpz_cmp(P->x, Q->x) == 0) {
        mpz_t neg_y;
        mpz_init(neg_y);
        mpz_neg(neg_y, Q->y);
        mpz_mod(neg_y, neg_y, p);

        if (mpz_cmp(P->y, neg_y) == 0) {
            R->infinity = 1;
            mpz_clear(neg_y);
            return;
        }
        mpz_clear(neg_y);
    }

    mpz_t lambda, temp1, temp2;
    mpz_inits(lambda, temp1, temp2, NULL);

    if (mpz_cmp(P->x, Q->x) == 0 && mpz_cmp(P->y, Q->y) == 0) {

        if (mpz_cmp_ui(P->y, 0) == 0) {
            R->infinity = 1;
            mpz_clears(lambda, temp1, temp2, NULL);
            return;
        }

        mpz_mul(temp1, P->x, P->x);
        mpz_mul_ui(temp1, temp1, 3);
        mpz_mod(temp1, temp1, p);

        mpz_mul_ui(temp2, P->y, 2);
        mpz_mod(temp2, temp2, p);
        modular_inverse(temp2, temp2, p);

        mpz_mul(lambda, temp1, temp2);
        mpz_mod(lambda, lambda, p);
    } else {

        mpz_sub(temp1, Q->y, P->y);
        mpz_mod(temp1, temp1, p);

        mpz_sub(temp2, Q->x, P->x);
        mpz_mod(temp2, temp2, p);
        modular_inverse(temp2, temp2, p);

        mpz_mul(lambda, temp1, temp2);
        mpz_mod(lambda, lambda, p);
    }

    mpz_mul(temp1, lambda, lambda);
    mpz_sub(temp1, temp1, P->x);
    mpz_sub(temp1, temp1, Q->x);
    mpz_mod(R->x, temp1, p);

    mpz_sub(temp1, P->x, R->x);
    mpz_mul(temp1, lambda, temp1);
    mpz_sub(temp1, temp1, P->y);
    mpz_mod(R->y, temp1, p);

    R->infinity = 0;

    mpz_clears(lambda, temp1, temp2, NULL);
}

void point_double(ECPoint *R, const ECPoint *P, const mpz_t p) {

    if (mpz_cmp_ui(P->y, 0) == 0) {
        R->infinity = 1;
        return;
    }

    mpz_t lambda, temp1, temp2;
    mpz_inits(lambda, temp1, temp2, NULL);
    mpz_mul(temp1, P->x, P->x);
    mpz_mul_ui(temp1, temp1, 3);
    mpz_mod(temp1, temp1, p);
    mpz_mul_ui(temp2, P->y, 2);
    mpz_mod(temp2, temp2, p);
    modular_inverse(temp2, temp2, p);
    mpz_mul(lambda, temp1, temp2);
    mpz_mod(lambda, lambda, p);
    mpz_mul(temp1, lambda, lambda);
    mpz_sub(temp1, temp1, P->x);
    mpz_sub(temp1, temp1, P->x);
    mpz_mod(R->x, temp1, p);
    mpz_sub(temp1, P->x, R->x);
    mpz_mul(temp1, lambda, temp1);
    mpz_sub(temp1, temp1, P->y);
    mpz_mod(R->y, temp1, p);

    R->infinity = 0;

    mpz_clears(lambda, temp1, temp2, NULL);
}

void scalar_mult(ECPoint *R, const mpz_t k, const ECPoint *P, const mpz_t p) {
    ECPoint Q;
    point_init(&Q);
    point_init(R);

    mpz_set(Q.x, P->x);
    mpz_set(Q.y, P->y);
    Q.infinity = P->infinity;

    R->infinity = 1;
    mpz_set_ui(R->x, 0);
    mpz_set_ui(R->y, 0);

    mpz_t N;
    mpz_init_set_str(N, N_STR, 16);

    mpz_t current_k;
    mpz_init(current_k);
    mpz_mod(current_k, k, N);

    while (mpz_cmp_ui(current_k, 0) > 0) {
        if (mpz_odd_p(current_k)) {
            if (R->infinity) {
                mpz_set(R->x, Q.x);
                mpz_set(R->y, Q.y);
                R->infinity = Q.infinity;
            } else {
                ECPoint temp;
                point_init(&temp);
                point_add(&temp, R, &Q, p);
                mpz_set(R->x, temp.x);
                mpz_set(R->y, temp.y);
                R->infinity = temp.infinity;
                point_clear(&temp);
            }
        }

        ECPoint tempQ;
        point_init(&tempQ);
        point_add(&tempQ, &Q, &Q, p);
        mpz_set(Q.x, tempQ.x);
        mpz_set(Q.y, tempQ.y);
        Q.infinity = tempQ.infinity;
        point_clear(&tempQ);

        mpz_fdiv_q_2exp(current_k, current_k, 1);
    }

    mpz_clear(current_k);
    mpz_clear(N);
    point_clear(&Q);
}

int point_is_valid(const ECPoint *point, const mpz_t p) {
    if (point->infinity) return 1;

    mpz_t lhs, rhs;
    mpz_inits(lhs, rhs, NULL);
    mpz_mul(lhs, point->y, point->y);
    mpz_mod(lhs, lhs, p);
    mpz_mul(rhs, point->x, point->x);
    mpz_mul(rhs, rhs, point->x);
    mpz_add_ui(rhs, rhs, 7);
    mpz_mod(rhs, rhs, p);

    int is_valid = (mpz_cmp(lhs, rhs) == 0);

    mpz_clears(lhs, rhs, NULL);
    return is_valid;
}

void get_compressed_public_key(char *out, const ECPoint *public_key) {
    char prefix = (mpz_tstbit(public_key->y, 0) == 0) ? 0x02 : 0x03;
    out[0] = prefix;
    size_t x_len = mpz_sizeinbase(public_key->x, 2) / 8 + 1;
    unsigned char x_bytes[x_len];
    mpz_export(x_bytes, &x_len, 1, 1, 0, 0, public_key->x);
    memcpy(out + 1, x_bytes, x_len);
}