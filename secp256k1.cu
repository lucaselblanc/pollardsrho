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

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

__constant__ unsigned int P_CONST[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ unsigned int N_CONST[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ unsigned int GX_CONST[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

__constant__ unsigned int GY_CONST[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

__constant__ unsigned int R_MOD_P[8] = {
    0x000003D1, 0x00000001, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

__constant__ unsigned int R2_MOD_P[8] = {
    0x000E90A1, 0x000007A2, 0x00000001, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

__constant__ unsigned int R2_MOD_N[8] = {
    0x67D7D140, 0x896CF214, 0x0E7CF878, 0x741496C2,
    0x5BCD07C6, 0xE697F5E4, 0x81C69BC5, 0x9D671CD5
};

__constant__ unsigned int MU_P = 0xD2253531;
__constant__ unsigned int MU_N = 0x5588B13F;

__constant__ unsigned int ZERO[8]  = {0, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int ONE[8]   = {1, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int TWO[8]   = {2, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int THREE[8] = {3, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int SEVEN[8] = {7, 0, 0, 0, 0, 0, 0, 0};

__constant__ unsigned int ONE_MONT[8] = {
    0x000003D1, 0x00000001, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

__constant__ unsigned int SEVEN_MONT[8] = {
    0x00001A97, 0x00000007, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

typedef struct {
    unsigned int x[8];
    unsigned int y[8];
    int infinity;
} ECPoint;

typedef struct {
    unsigned int X[8];
    unsigned int Y[8];
    unsigned int Z[8];
    int infinity;
} ECPointJacobian;

__device__ int bignum_cmp(const unsigned int *a, const unsigned int *b) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ int bignum_is_zero(const unsigned int *a) {
    for (int i = 0; i < 8; i++) {
        if (a[i] != 0) return 0;
    }
    return 1;
}

__device__ int bignum_is_odd(const unsigned int *a) {
    return a[0] & 1;
}

__device__ void bignum_copy(unsigned int *dst, const unsigned int *src) {
    for (int i = 0; i < 8; i++) {
        dst[i] = src[i];
    }
}

__device__ void bignum_zero(unsigned int *a) {
    for (int i = 0; i < 8; i++) {
        a[i] = 0;
    }
}

__device__ void bignum_set_ui(unsigned int *a, unsigned int val) {
    bignum_zero(a);
    a[0] = val;
}

__device__ unsigned int bignum_add_carry(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned long long carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (unsigned long long)a[i] + b[i];
        result[i] = (unsigned int)carry;
        carry >>= 32;
    }
    return (unsigned int)carry;
}

__device__ unsigned int bignum_sub_borrow(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    long long carry = 0;
    for (int i = 0; i < 8; i++) {
        long long tmp = (long long)a[i] - (long long)b[i] - carry;
        if (tmp < 0) {
            result[i] = (unsigned int)(tmp + (1ULL << 32));
            carry = 1;
        } else {
            result[i] = (unsigned int)tmp;
            carry = 0;
        }
    }
    return (unsigned int)carry;
}

__device__ void bignum_shr1(unsigned int *result, const unsigned int *a) {
    unsigned int carry = 0;
    for (int i = 7; i >= 0; i--) {
        unsigned int new_carry = a[i] & 1;
        result[i] = (a[i] >> 1) | (carry << 31);
        carry = new_carry;
    }
}

__device__ void bignum_mul_full(unsigned int *result_high, unsigned int *result_low, 
                                const unsigned int *a, const unsigned int *b) {
    unsigned int temp[16];
    
    for (int i = 0; i < 16; i++) {
        temp[i] = 0;
    }
    
    for (int i = 0; i < 8; i++) {
        unsigned long long carry = 0;
        for (int j = 0; j < 8; j++) {
            unsigned long long prod = (unsigned long long)a[i] * b[j] + temp[i + j] + carry;
            temp[i + j] = (unsigned int)prod;
            carry = prod >> 32;
        }
        temp[i + 8] = (unsigned int)carry;
    }
    
    for (int i = 0; i < 8; i++) {
        result_low[i] = temp[i];
        result_high[i] = temp[i + 8];
    }
}

__device__ void montgomery_reduce_p(unsigned int *result, const unsigned int *input_high, const unsigned int *input_low) {
    unsigned int temp[16];
    
    for (int i = 0; i < 8; i++) {
        temp[i] = input_low[i];
        temp[i + 8] = input_high[i];
    }
    
    for (int i = 0; i < 8; i++) {
        unsigned int ui = (temp[i] * MU_P) & 0xFFFFFFFF;
        unsigned long long carry = 0;
        
        for (int j = 0; j < 8; j++) {
            unsigned long long prod = (unsigned long long)ui * P_CONST[j] + temp[i + j] + carry;
            temp[i + j] = (unsigned int)prod;
            carry = prod >> 32;
        }
        
        for (int j = i + 8; j < 16; j++) {
            unsigned long long tmp = (unsigned long long)temp[j] + carry;
            temp[j] = (unsigned int)tmp;
            carry = tmp >> 32;
        }
    }
    
    for (int i = 0; i < 8; i++) {
        result[i] = temp[i + 8];
    }
    
    if (bignum_cmp(result, P_CONST) >= 0) {
        bignum_sub_borrow(result, result, P_CONST);
    }
}

__device__ void to_montgomery_p(unsigned int *result, const unsigned int *a) {
    unsigned int high[8], low[8];
    bignum_mul_full(high, low, a, R2_MOD_P);
    montgomery_reduce_p(result, high, low);
}

__device__ void from_montgomery_p(unsigned int *result, const unsigned int *a) {
    unsigned int zero[8];
    bignum_zero(zero);
    montgomery_reduce_p(result, zero, a);
}

__device__ void mod_add_p(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int temp[8];
    unsigned int carry = bignum_add_carry(temp, a, b);
    
    if (carry || bignum_cmp(temp, P_CONST) >= 0) {
        bignum_sub_borrow(result, temp, P_CONST);
    } else {
        bignum_copy(result, temp);
    }
}

__device__ void mod_sub_p(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int temp[8];
    unsigned int borrow = bignum_sub_borrow(temp, a, b);
    
    if (borrow) {
        bignum_add_carry(result, temp, P_CONST);
    } else {
        bignum_copy(result, temp);
    }
}

__device__ void mod_mul_mont_p(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int high[8], low[8];
    bignum_mul_full(high, low, a, b);
    montgomery_reduce_p(result, high, low);
}

__device__ void mod_sqr_mont_p(unsigned int *result, const unsigned int *a) {
    mod_mul_mont_p(result, a, a);
}

__device__ void mod_inverse_p(unsigned int *result, const unsigned int *a_normal) {
    
    if (bignum_is_zero(a_normal)) {
        bignum_zero(result);
        return;
    }

    const unsigned int exp_const[8] = {
        0xFFFFFC2D, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };

    unsigned int a_hat[8];
    to_montgomery_p(a_hat, a_normal);

    unsigned int exp[8];
    bignum_copy(exp, exp_const);

    int bitlen = 0;
    for (int bit = 255; bit >= 0; bit--) {
        int word = bit >> 5;
        int wbit = bit & 31;
        if ( (exp[word] >> wbit) & 1U ) { bitlen = bit + 1; break; }
    }
    if (bitlen == 0) {
        bignum_copy(result, ONE_MONT);
        return;
    }

    const int WINDOW = 4;
    const int WSIZE = (1 << WINDOW); 
    unsigned int pow_table[WSIZE][8];
    bignum_copy(pow_table[1], a_hat);

    for (int i = 2; i < WSIZE; i++) {
        mod_mul_mont_p(pow_table[i], pow_table[i-1], a_hat);
    }

    unsigned int acc[8];
    bignum_copy(acc, ONE_MONT);

    int i = bitlen - 1;
    while (i >= 0) {
        int word = i >> 5;
        int wbit = i & 31;
        unsigned int curbit = (exp[word] >> wbit) & 1U;

        if (!curbit) {
            mod_sqr_mont_p(acc, acc);
            i--;
            continue;
        }

        int l = 1;
        int max_l = (i + 1 < WINDOW) ? (i + 1) : WINDOW;

        unsigned int wval = 0;
        for (int k = 0; k < max_l; k++) {
            int bitpos = i - k;
            int w = bitpos >> 5;
            int wb = bitpos & 31;
            unsigned int b = (exp[w] >> wb) & 1U;
            wval |= (b << k);
        }

        while (l < max_l) {
            l++;
        }

        int chosen_l = 1;
        for (int try_l = max_l; try_l >= 1; try_l--) {
            unsigned int val = 0;
            for (int k = 0; k < try_l; k++) {
                int bitpos = i - k;
                int w = bitpos >> 5;
                int wb = bitpos & 31;
                unsigned int b = (exp[w] >> wb) & 1U;
                val |= (b << k);
            }
            if ( (val & ((1U << (try_l-1)))) != 0 ) {
                chosen_l = try_l;
                wval = val;
                break;
            }
        }

        for (int s = 0; s < chosen_l; s++) {
            mod_sqr_mont_p(acc, acc);
        }

        mod_mul_mont_p(acc, acc, pow_table[wval]);

        i -= chosen_l;
    }

    bignum_copy(result, acc);

    unsigned int chk[8];
    mod_mul_mont_p(chk, a_hat, result);
    if (bignum_cmp(chk, ONE_MONT) != 0) {
        bignum_zero(result);
        return;
    }

    return;
}

__device__ void jacobian_init(ECPointJacobian *point) {
    bignum_zero(point->X);
    bignum_zero(point->Y);
    bignum_copy(point->Z, ONE_MONT);
    point->infinity = 0;
}

__device__ void jacobian_set_infinity(ECPointJacobian *point) {
    bignum_copy(point->X, ONE_MONT);
    bignum_copy(point->Y, ONE_MONT);
    bignum_zero(point->Z);
    point->infinity = 1;
}

__device__ int jacobian_is_infinity(const ECPointJacobian *point) {
    return point->infinity || bignum_is_zero(point->Z);
}

__device__ void affine_to_jacobian(ECPointJacobian *jac, const ECPoint *aff) {
    if (aff->infinity) {
        jacobian_set_infinity(jac);
        return;
    }
    
    bignum_copy(jac->X, aff->x);
    bignum_copy(jac->Y, aff->y);
    bignum_copy(jac->Z, ONE_MONT);
    jac->infinity = 0;
}

__device__ void jacobian_to_affine(ECPoint *aff, const ECPointJacobian *jac) {
    if (jacobian_is_infinity(jac)) {
        bignum_zero(aff->x);
        bignum_zero(aff->y);
        aff->infinity = 1;
        return;
    }

    unsigned int z_norm[8], z_inv[8], z_inv_sqr[8], z_inv_cube[8];

    from_montgomery_p(z_norm, jac->Z);

    mod_inverse_p(z_inv, z_norm);
    mod_mul_mont_p(z_inv_sqr, z_inv, z_inv);
    mod_mul_mont_p(z_inv_cube, z_inv_sqr, z_inv);

    mod_mul_mont_p(aff->x, jac->X, z_inv_sqr);
    mod_mul_mont_p(aff->y, jac->Y, z_inv_cube);

    aff->infinity = 0;
}

__device__ void jacobian_double(ECPointJacobian *result, const ECPointJacobian *point) {
    if (jacobian_is_infinity(point) || bignum_is_zero(point->Y)) {
        jacobian_set_infinity(result);
        return;
    }

    unsigned int A[8], B[8], C[8], D[8], E[8], F[8];
    unsigned int X2[8]; 
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
    mod_sqr_mont_p(E, D);
    mod_sub_p(F, E, B);
    mod_sub_p(F, F, B);
    bignum_copy(result->X, F);
    mod_sub_p(result->Y, B, F);
    mod_mul_mont_p(result->Y, D, result->Y);
    mod_sub_p(result->Y, result->Y, C);
    mod_mul_mont_p(result->Z, point->Y, point->Z);
    mod_add_p(result->Z, result->Z, result->Z);

    result->infinity = 0;
}

__device__ void jacobian_add(ECPointJacobian *result, const ECPointJacobian *P, const ECPointJacobian *Q) {
    if (jacobian_is_infinity(P)) {
        bignum_copy(result->X, Q->X);
        bignum_copy(result->Y, Q->Y);
        bignum_copy(result->Z, Q->Z);
        result->infinity = Q->infinity;
        return;
    }

    if (jacobian_is_infinity(Q)) {
        bignum_copy(result->X, P->X);
        bignum_copy(result->Y, P->Y);
        bignum_copy(result->Z, P->Z);
        result->infinity = P->infinity;
        return;
    }

    unsigned int U1[8], U2[8], S1[8], S2[8], H[8], I[8], J[8], r[8], V[8];
    unsigned int Z1Z1[8], Z2Z2[8], temp[8];

    mod_sqr_mont_p(Z1Z1, P->Z);
    mod_sqr_mont_p(Z2Z2, Q->Z);
    mod_mul_mont_p(U1, P->X, Z2Z2);
    mod_mul_mont_p(U2, Q->X, Z1Z1);
    mod_mul_mont_p(temp, Q->Z, Z2Z2);
    mod_mul_mont_p(S1, P->Y, temp);
    mod_mul_mont_p(temp, P->Z, Z1Z1);
    mod_mul_mont_p(S2, Q->Y, temp);

    if (bignum_cmp(U1, U2) == 0) {
        if (bignum_cmp(S1, S2) == 0) {
            jacobian_double(result, P);
            return;
        } else {
            jacobian_set_infinity(result);
            return;
        }
    }

    mod_sub_p(H, U2, U1);
    mod_sqr_mont_p(I, H);
    mod_add_p(I, I, I);
    mod_add_p(I, I, I);
    mod_mul_mont_p(J, H, I);
    mod_sub_p(r, S2, S1);
    mod_add_p(r, r, r);
    mod_mul_mont_p(V, U1, I);
    mod_sqr_mont_p(result->X, r);
    mod_sub_p(result->X, result->X, J);
    mod_sub_p(result->X, result->X, V);
    mod_sub_p(result->X, result->X, V);
    mod_sub_p(result->Y, V, result->X);
    mod_mul_mont_p(result->Y, r, result->Y);
    mod_mul_mont_p(S1, S1, J);
    mod_add_p(S1, S1, S1);
    mod_sub_p(result->Y, result->Y, S1);

    unsigned int Z1_plus_Z2[8], Z1Z2_term[8];
    mod_add_p(Z1_plus_Z2, P->Z, Q->Z);
    mod_sqr_mont_p(Z1_plus_Z2, Z1_plus_Z2);
    mod_sub_p(Z1_plus_Z2, Z1_plus_Z2, Z1Z1);
    mod_sub_p(Z1_plus_Z2, Z1_plus_Z2, Z2Z2);
    mod_mul_mont_p(result->Z, Z1_plus_Z2, H);

    result->infinity = 0;
}

__device__ void scalar_reduce_n(unsigned int *r, const unsigned int *k) {
    unsigned int t[8];
    unsigned int borrow = bignum_sub_borrow(t, k, N_CONST);

    if (borrow == 0) {
        bignum_copy(r, t);
    } else {
        bignum_copy(r, k);
    }
}

__device__ void jacobian_scalar_mult(ECPointJacobian *result, const unsigned int *scalar, const ECPointJacobian *point) {
    ECPointJacobian Q;
    
    if (bignum_is_zero(scalar)) {
        jacobian_set_infinity(result);
        return;
    }
    
    if (jacobian_is_infinity(point)) {
        jacobian_set_infinity(result);
        return;
    }
    
    jacobian_set_infinity(result);
    Q = *point;

    unsigned int k[8];
    bignum_copy(k, scalar);

    int bit_found = 0;
    for (int i = 7; i >= 0; i--) {
        if (k[i] != 0) {
            bit_found = 1;
            break;
        }
    }
    
    if (!bit_found) {
        jacobian_set_infinity(result);
        return;
    }

    while (!bignum_is_zero(k)) {
        if (bignum_is_odd(k)) {
            if (jacobian_is_infinity(result)) {
                *result = Q;
            } else {
                ECPointJacobian temp;
                jacobian_add(&temp, result, &Q);
                *result = temp;
            }
        }
        
        ECPointJacobian temp;
        jacobian_double(&temp, &Q);
        Q = temp;
        
        bignum_shr1(k, k);
    }
}

__device__ void point_from_montgomery(ECPoint *result, const ECPoint *point_mont) {
    if (point_mont->infinity) {
        result->infinity = 1;
        bignum_zero(result->x);
        bignum_zero(result->y);
        return;
    }
    
    from_montgomery_p(result->x, point_mont->x);
    from_montgomery_p(result->y, point_mont->y);
    result->infinity = 0;
}

__device__ void kernel_point_init(ECPoint *point) {
    bignum_zero(point->x);
    bignum_zero(point->y);
    point->infinity = 0;
}

__device__ void kernel_point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q) {
    ECPointJacobian P_jac, Q_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    affine_to_jacobian(&Q_jac, Q);    
    jacobian_add(&R_jac, &P_jac, &Q_jac);   
    jacobian_to_affine(R, &R_jac);
}

__device__ void kernel_point_double(ECPoint *R, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    jacobian_double(&R_jac, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

__device__ void kernel_scalar_mult(ECPoint *R, const unsigned int *k, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    jacobian_scalar_mult(&R_jac, k, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

__device__ int kernel_point_is_valid(const ECPoint *point) {
    if (point->infinity) return 1;

    unsigned int lhs[8], rhs[8];
    
    mod_sqr_mont_p(lhs, point->y);
    mod_sqr_mont_p(rhs, point->x);
    mod_mul_mont_p(rhs, rhs, point->x);
    mod_add_p(rhs, rhs, SEVEN_MONT);

    return (bignum_cmp(lhs, rhs) == 0);
}

__device__ void kernel_get_compressed_public_key(unsigned char *out, const ECPoint *public_key) {
    ECPoint public_key_normal;
    point_from_montgomery(&public_key_normal, public_key);
    
    unsigned char prefix = (public_key_normal.y[0] & 1) ? 0x03 : 0x02;
    out[0] = prefix;
    
    for (int i = 0; i < 8; i++) {
        unsigned int word = public_key_normal.x[7-i];
        out[1 + i*4] = (word >> 24) & 0xFF;
        out[1 + i*4 + 1] = (word >> 16) & 0xFF;
        out[1 + i*4 + 2] = (word >> 8) & 0xFF;
        out[1 + i*4 + 3] = word & 0xFF;
    }
}

__global__ void generate_public_key(unsigned char *out, unsigned int *PRIV_KEY) {
    ECPoint pub;
    ECPoint G;
    ECPointJacobian G_jac, pub_jac;

    to_montgomery_p(G.x, GX_CONST);
    to_montgomery_p(G.y, GY_CONST);
    G.infinity = 0;

    affine_to_jacobian(&G_jac, &G);
    jacobian_scalar_mult(&pub_jac, PRIV_KEY, &G_jac);
    jacobian_to_affine(&pub, &pub_jac);

    kernel_get_compressed_public_key(out, &pub);
}

__global__ void point_init(ECPoint *point) {
    kernel_point_init(point);
}

__global__ void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q) {
    kernel_point_add(R, P, Q);
}

__global__ void point_double(ECPoint *R, const ECPoint *P) {
    kernel_point_double(R, P);
}

__global__ void scalar_mult(ECPoint *R, const unsigned int *k, const ECPoint *P) {
    kernel_scalar_mult(R, k, P);
}

__global__ void point_is_valid(int *result, const ECPoint *point) {
    *result = kernel_point_is_valid(point);
}

__global__ void get_compressed_public_key(unsigned char *out, const ECPoint *pub) {
    kernel_get_compressed_public_key(out, pub);
}

int main() {
    unsigned int h_priv[8] = {
        0x97c603c9, 0x28b88cf8, 0x5359f04f, 0x3e766570, 0x00000003, 0x00000000, 0x00000000, 0x00000000
    };

    unsigned int *d_priv;
    unsigned char *d_out, h_out[33];

    cudaMalloc(&d_priv, sizeof(h_priv));
    cudaMalloc(&d_out, 33);

    cudaMemcpy(d_priv, h_priv, sizeof(h_priv), cudaMemcpyHostToDevice);

    generate_public_key<<<1,1>>>(d_out, d_priv);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 33, cudaMemcpyDeviceToHost);

    printf("Public key compressed: ");
    for (int i = 0; i < 33; i++) {
        printf("%02X", h_out[i]);
    }
    printf("\n");

    cudaFree(d_priv);
    cudaFree(d_out);
    return 0;
}