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

BigInt f = BigInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
const uint64_t P_CONST[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
const uint64_t N_CONST[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
const uint64_t GX_CONST[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
const uint64_t GY_CONST[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };
const uint64_t R_MOD_P[4] = { 0x00000001000003D1ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL };
const uint64_t R2_MOD_P[4] = { 0x000007A2000E90A1ULL, 0x0000000100000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL };
const uint64_t R2_MOD_N[4] = { 0x896CF21467D7D140ULL, 0x741496C20E7CF878ULL, 0xE697F5E45BCD07C6ULL, 0x9D671CD581C69BC5ULL };
const uint64_t MU_P = 0xD2253531ULL;
const uint64_t MU_N = 0x5588B13FULL;
const uint64_t ZERO[4] = {0ULL, 0ULL, 0ULL, 0ULL};
const uint64_t ONE[4] = {1ULL, 0ULL, 0ULL, 0ULL};
const uint64_t TWO[4] = {2ULL, 0ULL, 0ULL, 0ULL};
const uint64_t THREE[4] = {3ULL, 0ULL, 0ULL, 0ULL};
const uint64_t SEVEN[4] = {7ULL, 0ULL, 0ULL, 0ULL};
const uint64_t ONE_MONT[4] = {0x00000001000003D1ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL};
const uint64_t SEVEN_MONT[4] = {0x0000000700001A97ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL};

constexpr int MAX_BITS = 256;

typedef struct {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
    int infinity;
} ECPointJacobian;

void montgomery_reduce_p(uint64_t *result, const uint64_t *input_high, const uint64_t *input_low) {
    uint64_t temp[8];
    for (int i = 0; i < 4; i++) {
        temp[i] = input_low[i];
        temp[i + 4] = input_high[i];
    }
    
    for (int i = 0; i < 4; i++) {
        uint64_t ui = temp[i] * MU_P;
        uint64_t carry = 0ULL;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)ui * (unsigned __int128)P_CONST[j] + (unsigned __int128)temp[i + j] + (unsigned __int128)carry;
            temp[i + j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        for (int j = i + 4; j < 8 && carry; j++) {
            unsigned __int128 tmp = (unsigned __int128)temp[j] + (unsigned __int128)carry;
            temp[j] = (uint64_t)tmp;
            carry = (uint64_t)(tmp >> 64);
        }
    }
    
    for (int i = 0; i < 4; i++) {
        result[i] = temp[i + 4];
    }
    
    uint64_t cmp = 0;
    for (int i = 3; i >= 0; i--) {
        uint64_t gt = (result[i] > P_CONST[i]) ? 1 : 0;
        uint64_t lt = (result[i] < P_CONST[i]) ? 1 : 0;
        cmp = gt - lt + (cmp & (1 - (gt | lt)));
    }
    
    uint64_t mask = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t borrow = 0ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t temp_val = result[i] - (P_CONST[i] & mask) - borrow;
        borrow = (result[i] < (P_CONST[i] & mask) + borrow) ? 1 : 0;
        result[i] = temp_val;
    }
}

void to_montgomery_p(uint64_t *result, const uint64_t *a) {
    uint64_t high[4], low[4];
    
    uint64_t temp[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0ULL;
        for (int j = 0; j < 4; j++) {
            unsigned __int128 prod = (unsigned __int128)a[i] * (unsigned __int128)R2_MOD_P[j] + (unsigned __int128)temp[i + j] + (unsigned __int128)carry;
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

void from_montgomery_p(uint64_t *result, const uint64_t *a) {
    uint64_t zero[4] = {0, 0, 0, 0};
    montgomery_reduce_p(result, zero, a);
}

void mod_add_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t temp[4];
    uint64_t carry = 0ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t sum = a[i] + b[i] + carry;
        carry = (sum < a[i]) || (carry && sum == a[i]);
        temp[i] = sum;
    }
    
    uint64_t cmp = 0;
    for (int i = 3; i >= 0; i--) {
        uint64_t gt = (temp[i] > P_CONST[i]) ? 1 : 0;
        uint64_t lt = (temp[i] < P_CONST[i]) ? 1 : 0;
        cmp = gt - lt + (cmp & (1 - (gt | lt)));
    }
    
    uint64_t reduce_mask = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t borrow = 0ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t sub = temp[i] - (P_CONST[i] & reduce_mask) - borrow;
        borrow = (temp[i] < (P_CONST[i] & reduce_mask) + borrow) ? 1 : 0;
        result[i] = sub;
    }
}

void mod_sub_p(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t temp[4];
    uint64_t borrow = 0ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t sub = a[i] - b[i] - borrow;
        borrow = (a[i] < b[i] + borrow) ? 1 : 0;
        temp[i] = sub;
    }
    
    uint64_t mask = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t carry = 0ULL;
    for (int i = 0; i < 4; i++) {
        uint64_t add = temp[i] + (P_CONST[i] & mask) + carry;
        carry = (add < temp[i]) || (carry && add == temp[i]);
        result[i] = add;
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
    uint64_t cmp = 0;
    for (int i = 3; i >= 0; i--) {
        uint64_t gt = (k[i] > N_CONST[i]) ? 1 : 0;
        uint64_t lt = (k[i] < N_CONST[i]) ? 1 : 0;
        cmp = gt - lt + (cmp & (1 - (gt | lt)));
    }
    
    if (cmp >= 0) {
        BigInt k_big = 0, n_big = 0;
        for (int i = 0; i < 4; i++) {
            k_big += BigInt(k[i]) << (i * 64);
            n_big += BigInt(N_CONST[i]) << (i * 64);
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
    for (int i = 0; i < 4; i++) {
        result += BigInt(a[i]) << (i * 64);
    }
    return result;
}

void bigint_to_uint64_array(uint64_t *result, const BigInt &value) {
    BigInt temp = value;
    for (int i = 0; i < 4; i++) {
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

void jacobian_to_affine(ECPoint *aff, const ECPointJacobian *jac) {
    if (jacobian_is_infinity(jac)) {
        for (int i = 0; i < 4; i++) {
            aff->x[i] = 0;
            aff->y[i] = 0;
        }
        aff->infinity = 1;
        return;
    }
    
    uint64_t z_norm[4], z_inv[4], z_inv_sqr[4], z_inv_cube[4];
    from_montgomery_p(z_norm, jac->Z);

    cpp_int g = 0;  

    for (int i = 0; i < 4; ++i) {  
        g <<= 64;  
        g |= z_norm[i];  
    }  

    cpp_int temp_g = recip2(g, f);

    for (int i = 3; i >= 0; --i) {
        z_inv[i] = static_cast<uint64_t>(temp_g & 0xFFFFFFFFFFFFFFFFULL);
        temp_g >>= 64;
    }

    to_montgomery_p(z_inv, z_inv);
    mod_mul_mont_p(z_inv_sqr, z_inv, z_inv);
    mod_mul_mont_p(z_inv_cube, z_inv_sqr, z_inv);
    mod_mul_mont_p(aff->x, jac->X, z_inv_sqr);
    mod_mul_mont_p(aff->y, jac->Y, z_inv_cube);
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
    uint64_t scalar_zero = 0;
    for (int i = 0; i < 4; i++) {
        scalar_zero |= scalar[i];
    }
    
    if (scalar_zero == 0 || jacobian_is_infinity(point)) {
        jacobian_set_infinity(result);
        return;
    }
    
    ECPointJacobian R0, R1;
    jacobian_set_infinity(&R0);
    R1 = *point;
    
    uint64_t k[4];
    scalar_reduce_n(k, scalar);
    
    int msb = -1;
    for (int i = 3; i >= 0; i--) {
        if (k[i] != 0) {
            for (int bit = 63; bit >= 0; bit--) {
                if ((k[i] >> bit) & 1ULL) {
                    msb = i * 64 + bit;
                    goto found_msb;
                }
            }
        }
    }
    found_msb:
    
    if (msb < 0) {
        jacobian_set_infinity(result);
        return;
    }
    
    for (int i = msb; i >= 0; i--) {
        int word = i / 64;
        int bit = i % 64;
        int kbit = (word >= 4 || word < 0) ? 0 : ((k[word] >> bit) & 1ULL);
        
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

int point_is_valid(const ECPoint *point) {
    if (point->infinity) return 1;
    
    uint64_t lhs[4], rhs[4];
    mod_sqr_mont_p(lhs, point->y);
    mod_sqr_mont_p(rhs, point->x);
    mod_mul_mont_p(rhs, rhs, point->x);
    mod_add_p(rhs, rhs, SEVEN_MONT);
    
    uint64_t diff = 0;
    for (int i = 0; i < 4; i++) {
        diff |= (lhs[i] ^ rhs[i]);
    }
    return (diff == 0);
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
    
    affine_to_jacobian(&G_jac, &G);
    jacobian_scalar_mult(&pub_jac, PRIV_KEY, &G_jac);
    jacobian_to_affine(&pub, &pub_jac);
    get_compressed_public_key(out, &pub);
}

BigInt test_mod_inverse(const BigInt &g, const BigInt &f) {
    return recip2(g, f);
}

int main() {

    BigInt g = BigInt("0x33e7665705359f04f28b88cf897c603c9");
    
    /* g ≡ 1 (mod f): */
    BigInt result = test_mod_inverse(g, f);
    
    std::cout << std::hex << result << std::endl;
    
    return 0;
}
