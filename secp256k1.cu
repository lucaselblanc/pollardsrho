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

#include "secp256k1.h"
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <stdint.h>

using std::make_tuple;
using std::tuple;
using std::get;
using std::pair;
using std::make_pair;

BigInt f("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

const uint64_t P_CONST[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
const uint64_t N_CONST[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
const uint64_t GX_CONST[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
const uint64_t GY_CONST[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };
const uint64_t R2_MOD_P[4] = { 0x000007A2000E90A1, 0x0000000000000001, 0x0ULL, 0x0ULL };
const uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
const uint64_t SEVEN_MONT[4] = { 0x0000007000001AB7, 0x0ULL, 0x0ULL, 0x0ULL };
const uint64_t P_CONST_MINUS_2[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
const uint64_t MU_P = 0xD838091DD2253531ULL;

typedef struct {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
    int infinity;
} ECPointJacobian;

void montgomery_reduce_p(uint64_t *result, const uint64_t *input_high, const uint64_t *input_low) {
    uint64_t temp[8];
    for (int i = 0; i < 4; i++) {
        temp[i]     = input_low[i];
        temp[i + 4] = input_high[i];
    }

    for (int i = 0; i < 4; i++) {
        uint64_t ui = (uint64_t)((unsigned __int128)temp[i] * (unsigned __int128)MU_P);
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)ui * (unsigned __int128)P_CONST[j]
                                   + (unsigned __int128)temp[i + j] + carry;
            temp[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        unsigned __int128 s = (unsigned __int128)temp[i + 4] + carry;
        temp[i + 4] = (uint64_t)s;
        carry = s >> 64;
        for (int j = i + 5; j < 8; ++j) {
            unsigned __int128 sum = (unsigned __int128)temp[j] + carry;
            temp[j] = (uint64_t)sum;
            carry = sum >> 64;
        }
    }

    for (int i = 0; i < 4; i++) result[i] = temp[i + 4];

    uint64_t diff[4];
    unsigned __int128 borrow = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 sub = (unsigned __int128)result[i] - (unsigned __int128)P_CONST[i] - borrow;
        diff[i] = (uint64_t)sub;
        borrow = (sub >> 127) & 1;
    }

    if (borrow == 0) {
        for (int i = 0; i < 4; i++) result[i] = diff[i];
    }
}

void to_montgomery_p(uint64_t *result, const uint64_t *a) {
    uint64_t a_local[4];
    for (int i = 0; i < 4; i++) a_local[i] = a[i];

    uint64_t temp[8] = {0};

    for (int i = 0; i < 4; i++) {
        unsigned __int128 carry = 0;

        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a_local[i] * (unsigned __int128)R2_MOD_P[j]
                                   + (unsigned __int128)temp[i + j]
                                   + carry;
            temp[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }

        unsigned __int128 sum = (unsigned __int128)temp[i + 4] + carry;
        temp[i + 4] = (uint64_t)sum;
        carry = sum >> 64;

        for (int k = i + 5; k < 8; k++) {
            unsigned __int128 s = (unsigned __int128)temp[k] + carry;
            temp[k] = (uint64_t)s;
            carry = s >> 64;
        }
    }

    uint64_t low[4], high[4];
    for (int i = 0; i < 4; i++) {
        low[i] = temp[i];
        high[i] = temp[i + 4];
    }

    montgomery_reduce_p(result, high, low);
}

void from_montgomery_p(uint64_t *result, const uint64_t *a) {
    uint64_t zero[4] = {0, 0, 0, 0};
    montgomery_reduce_p(result, zero, a);
}

void mod_add_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t temp[4];
    unsigned __int128 carry = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 s = (unsigned __int128)a[i] + (unsigned __int128)b[i] + carry;
        temp[i] = (uint64_t)s;
        carry = s >> 64;
    }

    bool ge = (bool)carry;
    if (!ge) {
        for (int i = 3; i >= 0; --i) {
            if (temp[i] > P_CONST[i]) { ge = true; break; }
            if (temp[i] < P_CONST[i]) { ge = false; break; }
        }
    }

    if (ge) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned __int128 sub = (unsigned __int128)temp[i] - (unsigned __int128)P_CONST[i] - borrow;
            result[i] = (uint64_t)sub;
            borrow = ((unsigned __int128)temp[i] < (unsigned __int128)P_CONST[i] + borrow) ? 1 : 0;
        }
    } else {
        for (int i = 0; i < 4; ++i) result[i] = temp[i];
    }
}

void mod_sub_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t temp[4];
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 sub = (unsigned __int128)a[i] - (unsigned __int128)b[i] - borrow;
        temp[i] = (uint64_t)sub;
        borrow = ((unsigned __int128)a[i] < (unsigned __int128)b[i] + borrow) ? 1 : 0;
    }

    if (borrow) {
        unsigned __int128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            unsigned __int128 s = (unsigned __int128)temp[i] + (unsigned __int128)P_CONST[i] + carry;
            result[i] = (uint64_t)s;
            carry = s >> 64;
        }
    } else {
        for (int i = 0; i < 4; ++i) result[i] = temp[i];
    }
}

void mod_mul_mont_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t high[4], low[4];

    uint64_t temp[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0ULL;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a[i] * (unsigned __int128)b[j] + (unsigned __int128)temp[i + j] + (unsigned __int128)carry;
            temp[i + j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        temp[i + 4] = carry;
    }

    for (int i = 0; i < 4; i++) {
        low[i] = temp[i];
        high[i] = temp[i + 4];
    }

    montgomery_reduce_p(result, high, low);
}

void mod_sqr_mont_p(uint64_t *out, const uint64_t *in) {
    mod_mul_mont_p(out, in, in);
}

void scalar_reduce_n(uint64_t *r, const uint64_t *k) {
    bool ge = false;
    for (int i = 3; i >= 0; i--) {
        if (k[i] > N_CONST[i]) { ge = true; break; }
        if (k[i] < N_CONST[i]) { ge = false; break; }
    }

    if (ge) {
        BigInt k_big = 0;
        BigInt n_big = 0;
        for (int i = 3; i >= 0; i--) {
            k_big <<= 64;
            k_big |= BigInt(k[i]);
            n_big <<= 64;
            n_big |= BigInt(N_CONST[i]);
        }

        BigInt result = k_big % n_big;

        for (int i = 0; i < 4; i++) {
            r[i] = static_cast<uint64_t>(result & 0xFFFFFFFFFFFFFFFFULL);
            result >>= 64;
        }
    } else {
        for (int i = 0; i < 4; i++) {
            r[i] = k[i];
        }
    }
}

BigInt uint64_array_to_bigint(const uint64_t *a) {
    BigInt result = 0;

    for (int i = 3; i >= 0; i--) {
        result <<= 64;
        result |= BigInt(a[i]);
    }
    return result;
}

void bigint_to_uint64_array(uint64_t *result, const BigInt &value) {
    BigInt temp = value;

    for (int i = 3; i >= 0; i--) {
        result[i] = static_cast<uint64_t>(temp & 0xFFFFFFFFFFFFFFFFULL);
        temp >>= 64;
    }
}

void jacobian_init(ECPointJacobian *point) {
    for (int i = 0; i < 4; i++) {
        point->X[i] = 0;
        point->Y[i] = 0;
        point->Z[i] = ONE_MONT[i];
    }
    point->infinity = 0;
}

void jacobian_set_infinity(ECPointJacobian *point) {
    for (int i = 0; i < 4; i++) {
        point->X[i] = ONE_MONT[i];
        point->Y[i] = ONE_MONT[i];
        point->Z[i] = 0;
    }
    point->infinity = 1;
}

int jacobian_is_infinity(const ECPointJacobian *point) {
    uint64_t z_zero = 0;
    for (int i = 0; i < 4; i++) {
        z_zero |= point->Z[i];
    }
    return point->infinity || (z_zero == 0);
}

void affine_to_jacobian(ECPointJacobian *jac, const ECPoint *aff) {
    if (aff->infinity) {
        jacobian_set_infinity(jac);
        return;
    }
    for (int i = 0; i < 4; i++) {
        jac->X[i] = aff->x[i];
        jac->Y[i] = aff->y[i];
        jac->Z[i] = ONE_MONT[i];
    }
    jac->infinity = 0;
}

void mod_exp_mont_p(uint64_t *res, const uint64_t *base, const uint64_t *exp) {
    uint64_t one[4] = {0};
    one[0] = 1ULL;
    to_montgomery_p(res, one);

    uint64_t acc[4];
    for (int i = 0; i < 4; i++) acc[i] = base[i];

    for (int word = 3; word >= 0; word--) {
        for (int bit = 63; bit >= 0; bit--) {
            mod_sqr_mont_p(res, res);

            if ((exp[word] >> bit) & 1ULL) {
                mod_mul_mont_p(res, res, acc);
            }
        }
    }
}

void jacobian_to_affine(ECPoint *aff, const ECPointJacobian *jac) {
    if (jacobian_is_infinity(jac)) {
        for (int i = 0; i < 4; i++) aff->x[i] = aff->y[i] = 0;
        aff->infinity = 1;
        return;
    }

    uint64_t Z_inv[4], Z_inv2[4], Z_inv3[4];

    mod_exp_mont_p(Z_inv, jac->Z, P_CONST_MINUS_2);
    mod_sqr_mont_p(Z_inv2, Z_inv);
    mod_mul_mont_p(Z_inv3, Z_inv2, Z_inv);
    mod_mul_mont_p(aff->x, jac->X, Z_inv2);
    mod_mul_mont_p(aff->y, jac->Y, Z_inv3);
    from_montgomery_p(aff->x, aff->x);
    from_montgomery_p(aff->y, aff->y);

    aff->infinity = 0;
}

void jacobian_double(ECPointJacobian *result, const ECPointJacobian *point) {
    uint64_t y_zero = 0;
    for (int i = 0; i < 4; i++) {
        y_zero |= point->Y[i];
    }

    if (jacobian_is_infinity(point) || y_zero == 0) {
        jacobian_set_infinity(result);
        return;
    }

    uint64_t A[4], B[4], C[4], D[4], E[4], X2[4];
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

void jacobian_add(ECPointJacobian *result, const ECPointJacobian *P, const ECPointJacobian *Q) {
    int P_infinity = jacobian_is_infinity(P);
    int Q_infinity = jacobian_is_infinity(Q);

    if (P_infinity) {
        for (int i = 0; i < 4; i++) {
            result->X[i] = Q->X[i];
            result->Y[i] = Q->Y[i];
            result->Z[i] = Q->Z[i];
        }
        result->infinity = Q->infinity;
        return;
    }
    if (Q_infinity) {
        for (int i = 0; i < 4; i++) {
            result->X[i] = P->X[i];
            result->Y[i] = P->Y[i];
            result->Z[i] = P->Z[i];
        }
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

    uint64_t h_zero = 0;
    uint64_t r_zero = 0;
    for (int i = 0; i < 4; i++) {
        h_zero |= H[i];
        r_zero |= r[i];
    }

    int is_H_zero = (h_zero == 0);
    int is_r_zero = (r_zero == 0);

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

void jacobian_scalar_mult(ECPointJacobian *result, const uint64_t *scalar, const ECPointJacobian *point) {
    if (jacobian_is_infinity(point)) {
        jacobian_set_infinity(result);
        return;
    }

    uint64_t k[4];
    scalar_reduce_n(k, scalar);

    ECPointJacobian R0, R1;
    jacobian_set_infinity(&R0);
    R1 = *point;

    auto cswap = [](ECPointJacobian &a, ECPointJacobian &b, uint64_t swap) {
        swap = 0 - swap;
        for (int i = 0; i < 4; i++) {
            uint64_t tmp;

            tmp = (a.X[i] ^ b.X[i]) & swap;
            a.X[i] ^= tmp;
            b.X[i] ^= tmp;

            tmp = (a.Y[i] ^ b.Y[i]) & swap;
            a.Y[i] ^= tmp;
            b.Y[i] ^= tmp;

            tmp = (a.Z[i] ^ b.Z[i]) & swap;
            a.Z[i] ^= tmp;
            b.Z[i] ^= tmp;
        }
        int tmp_inf = (a.infinity ^ b.infinity) & (int)swap;
        a.infinity ^= tmp_inf;
        b.infinity ^= tmp_inf;
    };

    for (int i = 255; i >= 0; i--) {
        int word = i / 64;
        int bit  = i % 64;
        uint64_t kbit = (k[word] >> bit) & 1ULL;

        cswap(R0, R1, kbit);

        ECPointJacobian R0_new, R1_new;
        jacobian_double(&R0_new, &R0);
        jacobian_add(&R1_new, &R0, &R1);

        R0 = R0_new;
        R1 = R1_new;

        cswap(R0, R1, kbit);
    }

    *result = R0;
}

void point_from_montgomery(ECPoint *result, const ECPoint *point_mont) {
    if (point_mont->infinity) {
        result->infinity = 1;
        for (int i = 0; i < 4; i++) {
            result->x[i] = 0;
            result->y[i] = 0;
        }
        return;
    }
    from_montgomery_p(result->x, point_mont->x);
    from_montgomery_p(result->y, point_mont->y);
    result->infinity = 0;
}

void point_init(ECPoint *point) {
    for (int i = 0; i < 4; i++) {
        point->x[i] = 0;
        point->y[i] = 0;
    }
    point->infinity = 0;
}

void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q) {
    ECPointJacobian P_jac, Q_jac, R_jac;
    affine_to_jacobian(&P_jac, P);
    affine_to_jacobian(&Q_jac, Q);
    jacobian_add(&R_jac, &P_jac, &Q_jac);
    jacobian_to_affine(R, &R_jac);
}

void point_double(ECPoint *R, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    affine_to_jacobian(&P_jac, P);
    jacobian_double(&R_jac, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

void scalar_mult(ECPoint *R, const uint64_t *k, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    affine_to_jacobian(&P_jac, P);
    jacobian_scalar_mult(&R_jac, k, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

void get_compressed_public_key(unsigned char *out, const ECPoint *public_key) {
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
        out[1 + i*8 + 6] = (word >> 8) & 0xFF;
        out[1 + i*8 + 7] = word & 0xFF;
    }
}

void generate_public_key(unsigned char *out, const uint64_t *PRIV_KEY) {
    ECPoint pub;
    ECPoint G;
    ECPointJacobian G_jac, pub_jac;

    to_montgomery_p(G.x, GX_CONST);
    to_montgomery_p(G.y, GY_CONST);
    G.infinity = 0;

    for (int i = 0; i < 4; i++) G_jac.Z[i] = ONE_MONT[i];
    G_jac.infinity = 0;

    for (int i = 0; i < 4; i++) {
        G_jac.X[i] = G.x[i];
        G_jac.Y[i] = G.y[i];
    }

    jacobian_scalar_mult(&pub_jac, PRIV_KEY, &G_jac);
    jacobian_to_affine(&pub, &pub_jac);
    get_compressed_public_key(out, &pub);
}

BigInt test_mod_inverse(const BigInt &g, const BigInt &f) {
    return recip2(g, f);
}

void debug_print_traces(const uint64_t PRIV_KEY[4], unsigned char out_pubkey[33]) {
    auto print_u64_4 = [](const char *label, const uint64_t v[4]) {
        printf("%s = 0x", label);
        for (int i = 3; i >= 0; --i) printf("%016llx", (unsigned long long)v[i]);
        printf("\n");
    };

    auto print_u64_4_words = [](const char *label, const uint64_t v[4]) {
        printf("%s :", label);
        for (int i = 3; i >= 0; --i) printf(" %016llx", (unsigned long long)v[i]);
        printf("\n");
    };

    printf("=== Constantes (originais) ===\n");
    print_u64_4("P_CONST", P_CONST);
    print_u64_4("N_CONST", N_CONST);
    print_u64_4("GX_CONST", GX_CONST);
    print_u64_4("GY_CONST", GY_CONST);
    print_u64_4("R2_MOD_P", R2_MOD_P);
    print_u64_4("ONE_MONT", ONE_MONT);
    print_u64_4("SEVEN_MONT", SEVEN_MONT);
    printf("\n");

    // 1) Teste de conversão Montgomery <-> normal
    uint64_t gx_m[4], gy_m[4], gx_back[4], gy_back[4];
    to_montgomery_p(gx_m, GX_CONST);
    to_montgomery_p(gy_m, GY_CONST);
    print_u64_4("GX (mont)", gx_m);
    print_u64_4("GY (mont)", gy_m);

    from_montgomery_p(gx_back, gx_m);
    from_montgomery_p(gy_back, gy_m);
    print_u64_4("GX (from mont)", gx_back);
    print_u64_4("GY (from mont)", gy_back);
    printf("\n");

    // 2) Testes aritméticos mod p (usando representações em Montgomery)
    uint64_t tmp_m[4], tmp_norm[4];

    // add
    mod_add_p(tmp_m, gx_m, gy_m);
    print_u64_4_words("GX+GY (mont)", tmp_m);
    from_montgomery_p(tmp_norm, tmp_m);
    print_u64_4("GX+GY (normal)", tmp_norm);

    // sub
    mod_sub_p(tmp_m, gx_m, gy_m);
    print_u64_4_words("GX-GY (mont)", tmp_m);
    from_montgomery_p(tmp_norm, tmp_m);
    print_u64_4("GX-GY (normal)", tmp_norm);

    // mul
    uint64_t mul_m[4], mul_norm[4];
    mod_mul_mont_p(mul_m, gx_m, gy_m);
    print_u64_4_words("GX*GY (mont)", mul_m);
    from_montgomery_p(mul_norm, mul_m);
    print_u64_4("GX*GY (normal)", mul_norm);

    // sqr
    uint64_t sq_m[4], sq_norm[4];
    mod_sqr_mont_p(sq_m, gx_m);
    print_u64_4_words("GX^2 (mont)", sq_m);
    from_montgomery_p(sq_norm, sq_m);
    print_u64_4("GX^2 (normal)", sq_norm);

    // Comparação com BigInt (verificação independente)
    {
        BigInt gx_big = uint64_array_to_bigint(GX_CONST);
        BigInt gy_big = uint64_array_to_bigint(GY_CONST);
        BigInt p_big  = f;
        BigInt expect_mul = (gx_big * gy_big) % p_big;
        BigInt expect_sqr = (gx_big * gx_big) % p_big;
        std::cout << "Expected GX*GY (bigint mod p) = 0x" << std::hex << expect_mul << std::endl;
        std::cout << "Expected GX^2 (bigint mod p) = 0x" << std::hex << expect_sqr << std::endl;
    }
    printf("\n");

    // 3) Teste de redução de escalar (n)
    uint64_t all_ones[4] = { 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
    uint64_t reduced[4];
    scalar_reduce_n(reduced, all_ones);
    print_u64_4("Scalar reduce(all_ones) ->", reduced);

    // teste com PRIV_KEY fornecida
    uint64_t k_copy[4];
    for (int i = 0; i < 4; ++i) k_copy[i] = PRIV_KEY[i];
    scalar_reduce_n(reduced, k_copy);
    print_u64_4("Scalar reduce(PRIV_KEY) ->", reduced);
    printf("\n");

    // 4) Jacobian / operações sobre pontos
    ECPoint G_aff;
    to_montgomery_p(G_aff.x, GX_CONST);
    to_montgomery_p(G_aff.y, GY_CONST);
    G_aff.infinity = 0;
    print_u64_4("G_aff.x (mont)", G_aff.x);
    print_u64_4("G_aff.y (mont)", G_aff.y);

    ECPointJacobian G_jac;
    affine_to_jacobian(&G_jac, &G_aff);
    print_u64_4_words("G_jac.X", G_jac.X);
    print_u64_4_words("G_jac.Y", G_jac.Y);
    print_u64_4_words("G_jac.Z", G_jac.Z);

    // double
    ECPointJacobian dbl;
    jacobian_double(&dbl, &G_jac);
    print_u64_4_words("dbl.X (mont)", dbl.X);
    print_u64_4_words("dbl.Y (mont)", dbl.Y);
    print_u64_4_words("dbl.Z (mont)", dbl.Z);

    uint64_t dblx_n[4], dbly_n[4], dblz_n[4];
    from_montgomery_p(dblx_n, dbl.X);
    from_montgomery_p(dbly_n, dbl.Y);
    from_montgomery_p(dblz_n, dbl.Z);
    print_u64_4("dbl.X (normal)", dblx_n);
    print_u64_4("dbl.Y (normal)", dbly_n);
    print_u64_4("dbl.Z (normal)", dblz_n);
    printf("\n");

    // add (G + G) -> deve igualar double
    ECPointJacobian addres;
    jacobian_add(&addres, &G_jac, &G_jac);
    print_u64_4_words("addres.X (mont)", addres.X);
    print_u64_4_words("addres.Y (mont)", addres.Y);
    print_u64_4_words("addres.Z (mont)", addres.Z);

    uint64_t addx_n[4], addy_n[4], addz_n[4];
    from_montgomery_p(addx_n, addres.X);
    from_montgomery_p(addy_n, addres.Y);
    from_montgomery_p(addz_n, addres.Z);
    print_u64_4("addres.X (normal)", addx_n);
    print_u64_4("addres.Y (normal)", addy_n);
    print_u64_4("addres.Z (normal)", addz_n);

    // comparar double vs add (normal)
    int same = 1;
    for (int i = 0; i < 4; ++i) {
        if (dblx_n[i] != addx_n[i] || dbly_n[i] != addy_n[i]) { same = 0; break; }
    }
    printf("double == add (normal coords)? %s\n", same ? "YES" : "NO");
    printf("\n");

    // 5) Multiplicação escalar (k * G) e conversão final para affine -> chave compactada
    ECPointJacobian result_jac;
    jacobian_scalar_mult(&result_jac, PRIV_KEY, &G_jac);
    print_u64_4_words("scalar_mult result_jac.X (mont)", result_jac.X);
    print_u64_4_words("scalar_mult result_jac.Y (mont)", result_jac.Y);
    print_u64_4_words("scalar_mult result_jac.Z (mont)", result_jac.Z);

    ECPoint result_aff;
    jacobian_to_affine(&result_aff, &result_jac); // usa recip2 internamente (ok)
    print_u64_4("result_aff.x (normal)", result_aff.x);
    print_u64_4("result_aff.y (normal)", result_aff.y);

    // preencher out_pubkey com a chave compactada e imprimir
    get_compressed_public_key(out_pubkey, &result_aff);
    printf("Compressed Public Key (computed): ");
    for (int i = 0; i < 33; ++i) printf("%02x", out_pubkey[i]);
    printf("\n");

    // 6) Comparações finais com valores esperados óbvios
    {
        // valor esperado da chave pública para k=1 (hex string já no seu código)
        const char *expected = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
        printf("Expected compressed pubkey (reference): %s\n", expected);
    }

    printf("=== fim do debug_print_traces ===\n");
}

int main() {

    BigInt g("0x33e7665705359f04f28b88cf897c603c9");

    /* g ≡ 1 (mod f): */
    BigInt result = test_mod_inverse(g, f);
    const std::string expected_inverse = "7fdb62ed2d6fa0874abd664c95b7cef2ed79cc82d13ff3ac8e9766aa21bebeae";

    std::cout << std::hex << result << std::endl;
    std::cout << "Inverso esperado para g: " << expected_inverse << std::endl;

    const std::string expected_pubkey = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";

    uint64_t PRIV_KEY[4] = {1, 0, 0, 0};
    unsigned char pubkey_compressed[33];

    generate_public_key(pubkey_compressed, PRIV_KEY);

    std::cout << "Compressed Public Key: ";
    for (int i = 0; i < 33; ++i) {
        printf("%02x", pubkey_compressed[i]);
    }
    std::cout << std::endl;

    std::cout << "A chave pública esperada é: " << expected_pubkey << std::endl;

    unsigned char out_pubkey[33];

    debug_print_traces(PRIV_KEY, out_pubkey);

    return 0;
}