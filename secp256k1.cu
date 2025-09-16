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

#include <iostream>
#include <tuple>
#include <boost/multiprecision/cpp_int.hpp>
#include <stdio.h>
#include <stdint.h>

using boost::multiprecision::cpp_int;
using BigInt = cpp_int;
using std::make_tuple;
using std::tuple;
using std::get;
using std::pair;
using std::make_pair;

const BigInt P_CONST = BigInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
const BigInt N_CONST = BigInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
const BigInt GX_CONST = BigInt("0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
const BigInt GY_CONST = BigInt("0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
const BigInt R_MOD_P = BigInt("0x1000003D1");
const BigInt R2_MOD_P = BigInt("0x7A2000E90A1");
const BigInt R2_MOD_N = BigInt("0x9D671CD581C69BC5E697F5E45BCD07C6741496C20E7CF878896CF21467D7D140");
const uint64_t MU_P = 0xD2253531ULL;
const uint64_t MU_N = 0x5588B13FULL;

const BigInt ZERO = BigInt("0");
const BigInt ONE = BigInt("1");
const BigInt TWO = BigInt("2");
const BigInt THREE = BigInt("3");
const BigInt SEVEN = BigInt("7");

const BigInt ONE_MONT = BigInt("0x1000003D1");
const BigInt SEVEN_MONT = BigInt("0x700001A97");

constexpr int MAX_BITS = 256;

typedef struct {
    BigInt x;
    BigInt y;
    int infinity;
} ECPoint;

typedef struct {
    BigInt X;
    BigInt Y;
    BigInt Z;
    int infinity;
} ECPointJacobian;

int bignum_cmp(const BigInt &a, const BigInt &b) {
    if (a > b) return 1;
    if (a < b) return -1;
    return 0;
}

int bignum_is_zero(const BigInt &a) {
    return a == 0 ? 1 : 0;
}

int bignum_is_odd(const BigInt &a) {
    return (a & 1) == 1 ? 1 : 0;
}

void bignum_copy(BigInt &dst, const BigInt &src) {
    dst = src;
}

void bignum_zero(BigInt &a) {
    a = 0;
}

int bignum_is_one(const BigInt &a) {
    return a == 1 ? 1 : 0;
}

void bignum_set_ui(BigInt &a, uint64_t val) {
    a = val;
}

BigInt bignum_add_carry(const BigInt &a, const BigInt &b) {
    return a + b;
}

BigInt bignum_sub_borrow(const BigInt &a, const BigInt &b) {
    return a - b;
}

void bignum_shr1(BigInt &result, const BigInt &a) {
    result = a >> 1;
}

void bignum_mul_full(BigInt &result_high, BigInt &result_low, const BigInt &a, const BigInt &b) {
    BigInt full_result = a * b;
    result_low = full_result & ((BigInt(1) << 256) - 1);
    result_high = full_result >> 256;
}

void montgomery_reduce_p(BigInt &result, const BigInt &input_high, const BigInt &input_low) {
    BigInt temp = (input_high << 256) | input_low;
    
    for (int i = 0; i < 4; i++) {
        BigInt ui = ((temp & ((BigInt(1) << 64) - 1)) * MU_P) & ((BigInt(1) << 64) - 1);
        temp = temp + ui * P_CONST;
        temp = temp >> 64;
    }
    
    result = temp & ((BigInt(1) << 256) - 1);
    if (result >= P_CONST) {
        result = result - P_CONST;
    }
}

void to_montgomery_p(BigInt &result, const BigInt &a) {
    BigInt high, low;
    bignum_mul_full(high, low, a, R2_MOD_P);
    montgomery_reduce_p(result, high, low);
}

void from_montgomery_p(BigInt &result, const BigInt &a) {
    BigInt zero = 0;
    montgomery_reduce_p(result, zero, a);
}

void mod_add_p(BigInt &result, const BigInt &a, const BigInt &b) {
    BigInt temp = a + b;
    if (temp >= P_CONST) {
        result = temp - P_CONST;
    } else {
        result = temp;
    }
}

void mod_sub_p(BigInt &result, const BigInt &a, const BigInt &b) {
    if (a >= b) {
        result = a - b;
    } else {
        result = a + P_CONST - b;
    }
}

void mod_mul_mont_p(BigInt &result, const BigInt &a, const BigInt &b) {
    BigInt high, low;
    bignum_mul_full(high, low, a, b);
    montgomery_reduce_p(result, high, low);
}

void mod_sqr_mont_p(BigInt &out, const BigInt &in) {
    mod_mul_mont_p(out, in, in);
}

/*
    Based on the Paper Almost-Inverse/Bernstein-Yang, REF: https://eprint.iacr.org/2019/266.pdf
*/

BigInt div2_floor(const BigInt &a) {
    return a / 2;
}

BigInt truncate(const BigInt& f, int t) {
    BigInt mask = (BigInt(1) << t) - 1;
    BigInt result = f & mask;
    BigInt bound = BigInt(1) << (t - 1);
    BigInt over = (result >= bound) ? 1 : 0;
    result -= over * (BigInt(1) << t);
    return result;
}

int bit_length(const BigInt &x) {  
    int msb = 0;  
    for (int i = 0; i < MAX_BITS; ++i) {  
        msb = ((x >> i) & 1) ? i + 1 : msb;  
    }  
    return msb;  
}

auto divsteps2(int n, int t, int delta, BigInt f, BigInt g) {
    f = truncate(f, t);
    g = truncate(g, t);

    BigInt scale = BigInt(1) << n;
    BigInt U = scale;
    BigInt V = 0;
    BigInt Q = 0;
    BigInt R = scale;

    for (int i = 0; i < n; ++i) {
        f = truncate(f, t);

        int g_odd = (g & 1) != 0;
        int delta_pos = (delta > 0) ? 1 : 0;
        int swap_mask = delta_pos & g_odd;

        BigInt new_f = swap_mask ? g : f;
        BigInt new_g = swap_mask ? -f : g;
        BigInt new_U = swap_mask ? Q : U;
        BigInt new_Q = swap_mask ? -U : Q;
        BigInt new_V = swap_mask ? R : V;
        BigInt new_R = swap_mask ? -V : R;

        f = new_f;
        g = new_g;
        U = new_U;
        Q = new_Q;
        V = new_V;
        R = new_R;

        delta = delta * (1 - 2 * swap_mask) + 1;

        BigInt tmpg = g + g_odd * f;
        g = div2_floor(tmpg);
        Q = div2_floor(Q + g_odd * U);
        R = div2_floor(R + g_odd * V);

        --t;
        g = truncate(g, t);
    }

    auto UV = make_pair(U, V);
    auto QR = make_pair(Q, R);
    auto P  = make_pair(UV, QR);
    return make_tuple(delta, f, g, P);
}

int iterations(int d) {
    return (d < 46) ? (49 * d + 80) / 17 : (49 * d + 57) / 17;
}

BigInt recip2(BigInt g, BigInt f) {
    if ((f & 1) == 0) throw std::invalid_argument("f must be odd");

    int d = std::max(bit_length(f), bit_length(g));
    int m = iterations(d);

    BigInt base = (f + 1) / 2;
    BigInt precomp = boost::multiprecision::powm(base, m - 1, f);

    auto result = divsteps2(m, m + 1, 1, f, g);
    BigInt fm = get<1>(result);
    auto P = get<3>(result);

    BigInt scale = BigInt(1) << m;
    BigInt V_scaled = P.first.second;
    BigInt V_int = (V_scaled * (BigInt(1) << (m - 1))) / scale;
    if (fm < 0) V_int = -V_int;

    BigInt inv = (V_int * precomp) % f;
    if (inv < 0) inv += f;

    return inv;
}

void jacobian_init(ECPointJacobian *point) {
    point->X = 0;
    point->Y = 0;
    point->Z = ONE_MONT;
    point->infinity = 0;
}

void jacobian_set_infinity(ECPointJacobian *point) {
    point->X = ONE_MONT;
    point->Y = ONE_MONT;
    point->Z = 0;
    point->infinity = 1;
}

int jacobian_is_infinity(const ECPointJacobian *point) {
    return point->infinity || point->Z == 0;
}

void affine_to_jacobian(ECPointJacobian *jac, const ECPoint *aff) {
    if (aff->infinity) {
        jacobian_set_infinity(jac);
        return;
    }
    
    jac->X = aff->x;
    jac->Y = aff->y;
    jac->Z = ONE_MONT;
    jac->infinity = 0;
}

void jacobian_to_affine(ECPoint *aff, const ECPointJacobian *jac) {
    if (jacobian_is_infinity(jac)) {
        aff->x = 0;
        aff->y = 0;
        aff->infinity = 1;
        return;
    }
    
    BigInt z_norm, z_inv, z_inv_sqr, z_inv_cube;
    
    from_montgomery_p(z_norm, jac->Z);
    
    z_inv = recip2(z_norm, P_CONST);
    
    mod_mul_mont_p(z_inv_sqr, z_inv, z_inv);
    mod_mul_mont_p(z_inv_cube, z_inv_sqr, z_inv);
    mod_mul_mont_p(aff->x, jac->X, z_inv_sqr);
    mod_mul_mont_p(aff->y, jac->Y, z_inv_cube);
    
    from_montgomery_p(aff->x, aff->x);
    from_montgomery_p(aff->y, aff->y);
    
    aff->infinity = 0;
}

void jacobian_double(ECPointJacobian *result, const ECPointJacobian *point) {
    if (jacobian_is_infinity(point) || point->Y == 0) {
        jacobian_set_infinity(result);
        return;
    }
    
    BigInt A, B, C, D, E, X2;
    
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
        result->X = Q->X;
        result->Y = Q->Y;
        result->Z = Q->Z;
        result->infinity = Q->infinity;
        return;
    }
    
    if (Q_infinity) {
        result->X = P->X;
        result->Y = P->Y;
        result->Z = P->Z;
        result->infinity = P->infinity;
        return;
    }
    
    BigInt U1, U2, S1, S2, H, I, J, r, V;
    BigInt Z1Z1, Z2Z2, Z1Z2, temp1, temp2;
    
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
    
    int is_H_zero = (H == 0);
    int is_r_zero = (r == 0);
    
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

void scalar_reduce_n(BigInt &r, const BigInt &k) {
    if (k >= N_CONST) {
        r = k % N_CONST;
    } else {
        r = k;
    }
}

void jacobian_scalar_mult(ECPointJacobian *result, const BigInt &scalar, const ECPointJacobian *point) {
    if (scalar == 0 || jacobian_is_infinity(point)) {
        jacobian_set_infinity(result);
        return;
    }
    
    ECPointJacobian R0, R1;
    jacobian_set_infinity(&R0);
    R1 = *point;
    
    BigInt k;
    scalar_reduce_n(k, scalar);
    
    int msb = msb_index(k);
    
    for (int i = msb; i >= 0; i--) {
        int kbit = bit_test(k, i) ? 1 : 0;
        
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
        result->x = 0;
        result->y = 0;
        return;
    }
    
    from_montgomery_p(result->x, point_mont->x);
    from_montgomery_p(result->y, point_mont->y);
    result->infinity = 0;
}

void point_init(ECPoint *point) {
    point->x = 0;
    point->y = 0;
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

void scalar_mult(ECPoint *R, const BigInt &k, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    jacobian_scalar_mult(&R_jac, k, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

int point_is_valid(const ECPoint *point) {
    if (point->infinity) return 1;
    
    BigInt lhs, rhs;
    
    mod_sqr_mont_p(lhs, point->y);
    mod_sqr_mont_p(rhs, point->x);
    mod_mul_mont_p(rhs, rhs, point->x);
    mod_add_p(rhs, rhs, SEVEN_MONT);
    
    return (lhs == rhs) ? 1 : 0;
}

void get_compressed_public_key(unsigned char *out, const ECPoint *public_key) {
    unsigned char prefix = (public_key->y & 1) == 1 ? 0x03 : 0x02;
    out[0] = prefix;
    
    BigInt x_copy = public_key->x;
    for (int i = 0; i < 32; i++) {
        out[1 + i] = static_cast<unsigned char>(x_copy & 0xFF);
        x_copy >>= 8;
    }
    
    for (int i = 0; i < 16; i++) {
        unsigned char temp = out[1 + i];
        out[1 + i] = out[1 + 31 - i];
        out[1 + 31 - i] = temp;
    }
}

void generate_public_key(unsigned char *out, const BigInt &PRIV_KEY) {
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

void test_mod_inverse(const BigInt &g, const BigInt &f, BigInt &result) {
    result = recip2(g, f);
}

int main() {
    BigInt f = BigInt("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    BigInt g = BigInt("0x33e7665705359f04f28b88cf897c603c9");
    
    /* g ≡ 1 (mod f): */
    BigInt result;
    test_mod_inverse(g, f, result);
    
    std::cout << std::hex << result << std::endl;
    
    return 0;
}