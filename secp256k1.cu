/******************************************************************************************************
 * This file is part of the Pollard's Rho distribution: (https://github.com/lucaselblanc/pollardsrho) *
 * Copyright (c) 2024, 2026 Lucas Leblanc.                                                            *
 * Distributed under the MIT software license, see the accompanying.                                  *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.                               *
 ******************************************************************************************************/

/*****************************************
 * Pollard's Rho Algorithm for SECP256K1 *
 * Written by Lucas Leblanc              *
******************************************/

#include "secp256k1.h"

    __device__ __constant__ __align__(16) uint64_t P_CONST[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
    __device__ __constant__ __align__(16) uint64_t N_CONST[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
    __device__ __constant__ __align__(16) uint256_t N_STRUCT = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
    __device__ __constant__ __align__(16) uint64_t GX_CONST[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
    __device__ __constant__ __align__(16) uint64_t GY_CONST[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };
    __device__ __constant__ __align__(16) uint64_t R2_MOD_P[4] = { 0x000007A2000E90A1, 0x0000000000000001, 0x0ULL, 0x0ULL };
    __device__ __constant__ __align__(16) uint64_t ZERO_MONT[4] = { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL };
    __device__ __constant__ __align__(16) uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
    __device__ __constant__ __align__(16) uint64_t SEVEN_MONT[4] = {0x700001AB7ULL, 0x0ULL, 0x0ULL, 0x0ULL};
    __device__ __constant__ __align__(16) uint64_t SUB2_FP[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
    __device__ __constant__ __align__(16) uint64_t LAMBDA_N[4] = { 0xDF02967C1B23BD72ULL, 0xA5261C028812645AULL, 0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL };
    __device__ __constant__ __align__(16) uint64_t BETA_P[4] = { 0xB315ECECBB640683ULL, 0x9CF0497512F58995ULL, 0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL };
    __device__ __constant__ __align__(16) uint64_t MINUS_B1[4] = { 0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL, 0x0ULL, 0x0ULL };
    __device__ __constant__ __align__(16) uint64_t MINUS_B2[4] = { 0x8A280AC50774346DULL, 0xD765CDA83DB1562CULL, 0xCFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
    __device__ __constant__ __align__(16) uint64_t G1[4] = { 0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL, 0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL };
    __device__ __constant__ __align__(16) uint64_t G2[4] = { 0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL, 0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL };
    __device__ __constant__ __align__(16) uint64_t MU_P = 0xD838091DD2253531ULL;
    __device__ ECPointJacobian* preCompG;
    __device__ ECPointJacobian* preCompGphi;
    __device__ ECPointJacobian* preCompH;
    __device__ ECPointJacobian* preCompHphi;
    __device__ ECPointJacobian* jacNorm;
    __device__ ECPointJacobian* jacEndo;
    __device__ ECPointJacobian* jacNormH;
    __device__ ECPointJacobian* jacEndoH;

extern const uint64_t P_CONST_HOST[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
extern const uint64_t N_CONST_HOST[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
extern const uint256_t N_STRUCT_HOST = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
extern const uint64_t GX_CONST_HOST[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
extern const uint64_t GY_CONST_HOST[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };
extern const uint64_t R2_MOD_P_HOST[4] = { 0x000007A2000E90A1, 0x0000000000000001, 0x0ULL, 0x0ULL };
extern const uint64_t ZERO_MONT_HOST[4] = { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL };
extern const uint64_t ONE_MONT_HOST[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
extern const uint64_t SEVEN_MONT_HOST[4] = {0x700001AB7ULL, 0x0ULL, 0x0ULL, 0x0ULL};
extern const uint64_t SUB2_FP_HOST[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
extern const uint64_t LAMBDA_N_HOST[4] = { 0xDF02967C1B23BD72ULL, 0xA5261C028812645AULL, 0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL };
extern const uint64_t BETA_P_HOST[4] = { 0xB315ECECBB640683ULL, 0x9CF0497512F58995ULL, 0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL };
extern const uint64_t MINUS_B1_HOST[4] = { 0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL, 0x0ULL, 0x0ULL };
extern const uint64_t MINUS_B2_HOST[4] = { 0x8A280AC50774346DULL, 0xD765CDA83DB1562CULL, 0xCFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
extern const uint64_t G1_HOST[4] = { 0xE893209A45DBB031ULL, 0x3DAA8A1471E8CA7FULL, 0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL };
extern const uint64_t G2_HOST[4] = { 0x1571B4AE8AC47F71ULL, 0x221208AC9DF506C6ULL, 0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL };
extern const uint64_t MU_P_HOST = 0xD838091DD2253531ULL;
ECPointJacobian* preCompG_HOST;
ECPointJacobian* preCompGphi_HOST;
ECPointJacobian* preCompH_HOST;
ECPointJacobian* preCompHphi_HOST;
ECPointJacobian* jacNorm_HOST;
ECPointJacobian* jacEndo_HOST;
ECPointJacobian* jacNormH_HOST;
ECPointJacobian* jacEndoH_HOST;

extern "C" void defGpuPointers(
    ECPointJacobian* d_G, ECPointJacobian* d_Gphi,
    ECPointJacobian* d_H, ECPointJacobian* d_Hphi,
    ECPointJacobian* d_jN, ECPointJacobian* d_jNH,
    ECPointJacobian* d_jE, ECPointJacobian* d_jEH
) {
    cudaMemcpyToSymbol(preCompG, &d_G, sizeof(ECPointJacobian*));
    cudaMemcpyToSymbol(preCompGphi, &d_Gphi, sizeof(ECPointJacobian*));
    cudaMemcpyToSymbol(preCompH, &d_H, sizeof(ECPointJacobian*));
    cudaMemcpyToSymbol(preCompHphi, &d_Hphi, sizeof(ECPointJacobian*));
    cudaMemcpyToSymbol(jacNorm, &d_jN, sizeof(ECPointJacobian*));
    cudaMemcpyToSymbol(jacNormH, &d_jNH, sizeof(ECPointJacobian*));
    cudaMemcpyToSymbol(jacEndo, &d_jE, sizeof(ECPointJacobian*));
    cudaMemcpyToSymbol(jacEndoH, &d_jEH, sizeof(ECPointJacobian*));
}

__global__ void cudaGetSymbols() {
    if (blockIdx.x == 0x7FFFFFFF) {
        (void)P_CONST[0];
        (void)N_CONST[0];
        (void)N_STRUCT.limbs[0];
        (void)GX_CONST[0];
        (void)GY_CONST[0];
        (void)R2_MOD_P[0];
        (void)ZERO_MONT[0];
        (void)ONE_MONT[0];
        (void)SEVEN_MONT[0];
        (void)SUB2_FP[0];
        (void)LAMBDA_N[0];
        (void)BETA_P[0];
        (void)MINUS_B1[0];
        (void)MINUS_B2[0];
        (void)G1[0];
        (void)G2[0];
        (void)MU_P;
        (void)preCompG;
        (void)preCompGphi;
        (void)preCompH;
        (void)preCompHphi;
        (void)jacNorm;
        (void)jacNormH;
        (void)jacEndo;
        (void)jacEndoH;
    }
}

void cudaLoadSymbols() {
    void* _ = (void*)cudaGetSymbols;
}

__host__ __device__ void montgomeryReduceP(uint64_t *result, const uint64_t *inputHigh, const uint64_t *inputLow) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&P_REF)[4] = P_CONST;
        const uint64_t &MU_REF = MU_P;
    #else
        const uint64_t (&P_REF)[4] = P_CONST_HOST;
        const uint64_t &MU_REF = MU_P_HOST;
    #endif

    uint64_t temp[8];
    for (int i = 0; i < 4; i++) {
        temp[i]     =  inputLow[i];
        temp[i + 4] = inputHigh[i];
    }

    uint64_t extra = 0;

    for (int i = 0; i < 4; i++) {
        uint64_t ui = (uint64_t)((uint128_t)temp[i] * (uint128_t)MU_REF);
        uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)ui * (uint128_t)P_REF[j] + (uint128_t)temp[i + j] + carry;
            temp[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }

        uint128_t s = (uint128_t)temp[i + 4] + carry;
        temp[i + 4] = (uint64_t)s;
        carry = s >> 64;

        for (int j = i + 5; j < 8; ++j) {
            uint128_t sum = (uint128_t)temp[j] + carry;
            temp[j] = (uint64_t)sum;
            carry = sum >> 64;
        }

        extra += (uint64_t)carry;
    }

    for (int i = 0; i < 4; i++) result[i] = temp[i + 4];

    uint64_t diff[4];
    uint128_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint128_t sub = (uint128_t)result[i] - (uint128_t)P_REF[i] - borrow;
        diff[i] = (uint64_t)sub;
        borrow = (sub >> 127) & 1;
    }

    if (extra != 0 || borrow == 0) {
        for (int i = 0; i < 4; i++) result[i] = diff[i];
    }
}

__host__ __device__ void toMontgomeryP(uint64_t *result, const uint64_t *a) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&R2_REF)[4] = R2_MOD_P;
    #else
        const uint64_t (&R2_REF)[4] = R2_MOD_P_HOST;
    #endif

    uint64_t aLocal[4];
    for (int i = 0; i < 4; i++) aLocal[i] = a[i];

    uint64_t temp[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint128_t carry = 0;

        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)aLocal[i] * (uint128_t)R2_REF[j] + (uint128_t)temp[i + j] + carry;
            temp[i + j] = (uint64_t)prod;
            carry = prod >> 64;
        }

        uint128_t sum = (uint128_t)temp[i + 4] + carry;
        temp[i + 4] = (uint64_t)sum;
        carry = sum >> 64;

        for (int k = i + 5; k < 8; k++) {
            uint128_t s = (uint128_t)temp[k] + carry;
            temp[k] = (uint64_t)s;
            carry = s >> 64;
        }
    }

    uint64_t low[4], high[4];
    for (int i = 0; i < 4; i++) {
        low[i] = temp[i];
        high[i] = temp[i + 4];
    }

    montgomeryReduceP(result, high, low);
}

__host__ __device__ void fromMontgomeryP(uint64_t *result, const uint64_t *a) {
    uint64_t zero[4] = {0, 0, 0, 0};
    montgomeryReduceP(result, zero, a);
}

__host__ __device__ void modAddP(uint64_t *result, const uint64_t *a, const uint64_t *b) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&P_REF)[4] = P_CONST;
    #else
        const uint64_t (&P_REF)[4] = P_CONST_HOST;
    #endif

    uint128_t carry = 0;
    uint64_t res[4];

    for (int i = 0; i < 4; ++i) {
        uint128_t s = (uint128_t)a[i] + b[i] + carry;
        res[i] = (uint64_t)s;
        carry = s >> 64;
    }

    bool ge = (carry != 0);
    if (!ge) {
        for (int i = 3; i >= 0; --i) {
            if (res[i] > P_REF[i]) { ge = true; break; }
            if (res[i] < P_REF[i]) { ge = false; break; }
            if (i == 0) ge = true;
        }
    }

    if (ge) {
        uint128_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            uint128_t sub = (uint128_t)res[i] - P_REF[i] - borrow;
            result[i] = (uint64_t)sub;
            borrow = (sub >> 127) & 1;
        }
    } else {
        for (int i = 0; i < 4; i++) result[i] = res[i];
    }
}

__host__ __device__ void modSubP(uint64_t *result, const uint64_t *a, const uint64_t *b) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&P_REF)[4] = P_CONST;
    #else
        const uint64_t (&P_REF)[4] = P_CONST_HOST;
    #endif

    uint128_t borrow = 0;
    uint64_t res[4];

    for (int i = 0; i < 4; ++i) {
        uint128_t sub = (uint128_t)a[i] - b[i] - borrow;
        res[i] = (uint64_t)sub;
        borrow = (sub >> 127) & 1;
    }

    if (borrow) {
        uint128_t carry = 0;
        for (int i = 0; i < 4; ++i) {
            uint128_t s = (uint128_t)res[i] + P_REF[i] + carry;
            result[i] = (uint64_t)s;
            carry = s >> 64;
        }
    } else {
        for (int i = 0; i < 4; i++) result[i] = res[i];
    }
}

__host__ __device__ void modMulMontP(uint64_t *result, const uint64_t *a, const uint64_t *b) {
    uint64_t temp[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128_t prod =
                (uint128_t)a[i] * (uint128_t)b[j]
              + (uint128_t)temp[i + j]
              + (uint128_t)carry;

            temp[i + j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        temp[i + 4] += carry;
    }

    montgomeryReduceP(result, temp + 4, temp);
}

__host__ __device__ void modSqrMontP(uint64_t *out, const uint64_t *in) {
    modMulMontP(out, in, in);
}

__host__ __device__ void scalarReduceN(uint64_t *r, const uint64_t *k) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&N_REF)[4] = N_CONST;
    #else
        const uint64_t (&N_REF)[4] = N_CONST_HOST;
    #endif

    bool ge = true;
    for (int i = 3; i >= 0; i--) {
        if (k[i] > N_REF[i]) { ge = true; break; }
        if (k[i] < N_REF[i]) { ge = false; break; }
    }

    if (ge) {
        uint128_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint128_t res = (uint128_t)k[i] - N_REF[i] - borrow;
            r[i] = (uint64_t)res;
            borrow = (res >> 127) & 1;
        }
    } else {
        for (int i = 0; i < 4; i++) r[i] = k[i];
    }
}

__host__ __device__ void modExpMontP(uint64_t *res, const uint64_t *base, const uint64_t *exp)
{
    uint64_t one[4] = {1,0,0,0};

    toMontgomeryP(res, one);

    uint64_t acc[4];
    for (int i = 0; i < 4; i++)
        acc[i] = base[i];

    for (int word = 3; word >= 0; word--) {
        for (int bit = 63; bit >= 0; bit--) {
            modSqrMontP(res, res);
            if ((exp[word] >> bit) & 1ULL) {
                modMulMontP(res, res, acc);
            }
        }
    }
}

__host__ __device__ void sqrtModP(uint64_t y[4], const uint64_t v[4]) {

    uint64_t exp[4];

    exp[0] = 0xFFFFFFFFBFFFFF0CULL;
    exp[1] = 0xFFFFFFFFFFFFFFFFULL;
    exp[2] = 0xFFFFFFFFFFFFFFFFULL;
    exp[3] = 0x3FFFFFFFFFFFFFFFULL;

    modExpMontP(y, v, exp);
}

__host__ __device__ void jacobianSetInfinity(ECPointJacobian *point) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&ONE_REF)[4] = ONE_MONT;
    #else
        const uint64_t (&ONE_REF)[4] = ONE_MONT_HOST;
    #endif

    for (int i = 0; i < 4; i++) {
        point->X[i] = ONE_REF[i];
        point->Y[i] = ONE_REF[i];
        point->Z[i] = 0;
    }
    point->infinity = 1;
}

__host__ __device__ bool jacobianIsInfinity(const ECPointJacobian *point) {
    uint64_t z_zero = 0;
    for (int i = 0; i < 4; i++) {
        z_zero |= point->Z[i];
    }
    return point->infinity || (z_zero == 0);
}

__host__ __device__ void jacobianToAffine(ECPointAffine *aff, const ECPointJacobian *jac) {

   #ifdef __CUDA_ARCH__
        const uint64_t (&SUB2_REF)[4] = SUB2_FP;
    #else
        const uint64_t (&SUB2_REF)[4] = SUB2_FP_HOST;
    #endif

    if (jacobianIsInfinity(jac)) {
        for (int i = 0; i < 4; i++)
            aff->x[i] = aff->y[i] = 0;
        aff->infinity = 1;
        return;
    }

    uint64_t zInv[4];
    uint64_t zInv2[4];
    uint64_t zInv3[4];

    modExpMontP(zInv, jac->Z, SUB2_REF);
    modMulMontP(zInv2, zInv, zInv);
    modMulMontP(zInv3, zInv2, zInv);
    modMulMontP(aff->x, jac->X, zInv2);
    fromMontgomeryP(aff->x, aff->x);
    modMulMontP(aff->y, jac->Y, zInv3);
    fromMontgomeryP(aff->y, aff->y);

    aff->infinity = 0;
}

__host__ __device__ void affineToJacobian(ECPointJacobian *jac, const ECPointAffine *aff) {

   #ifdef __CUDA_ARCH__
        const uint64_t (&ONE_REF)[4] = ONE_MONT;
    #else
        const uint64_t (&ONE_REF)[4] = ONE_MONT_HOST;
    #endif

    if (aff->infinity) {
        jacobianSetInfinity(jac);
        return;
    }

    toMontgomeryP(jac->X, aff->x);
    toMontgomeryP(jac->Y, aff->y);

    for (int i = 0; i < 4; i++) { jac->Z[i] = ONE_REF[i]; }

    jac->infinity = 0;
}

__host__ __device__ void jacobianDouble(ECPointJacobian *R, const ECPointJacobian *P) {
    if (jacobianIsInfinity(P) || ((P->Y[0] | P->Y[1] | P->Y[2] | P->Y[3]) == 0)) {
        jacobianSetInfinity(R);
        return;
    }

    uint64_t A[4], B[4], C[4], D[4], E[4], F[4], newZ[4], tmp[4];

    modMulMontP(newZ, P->Y, P->Z);
    modAddP(newZ, newZ, newZ);
    modMulMontP(A, P->X, P->X);
    modMulMontP(B, P->Y, P->Y);
    modMulMontP(C, B, B);
    modMulMontP(D, P->X, B);
    modAddP(D, D, D);
    modAddP(D, D, D);
    modAddP(E, A, A);
    modAddP(E, E, A);
    modMulMontP(F, E, E);
    modSubP(R->X, F, D);
    modSubP(R->X, R->X, D);
    modSubP(tmp, D, R->X);
    modMulMontP(R->Y, E, tmp);
    modAddP(C, C, C);
    modAddP(C, C, C);
    modAddP(C, C, C);
    modSubP(R->Y, R->Y, C);

    for(int i=0; i<4; i++) R->Z[i] = newZ[i];
    R->infinity = 0;
}

__host__ __device__ void jacobianAdd(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q) {
    if (jacobianIsInfinity(P)) { *R = *Q; return; }
    if (jacobianIsInfinity(Q)) { *R = *P; return; }

    uint64_t Z1Z1[4], Z2Z2[4], U1[4], U2[4], S1[4], S2[4];
    uint64_t H[4], I[4], J[4], r[4], V[4], SJ[4], newZ[4], newX[4], newY[4];

    modMulMontP(Z1Z1, P->Z, P->Z);
    modMulMontP(Z2Z2, Q->Z, Q->Z);
    modMulMontP(U1, P->X, Z2Z2);
    modMulMontP(U2, Q->X, Z1Z1);
    modMulMontP(S1, P->Y, Z2Z2);
    modMulMontP(S1, S1, Q->Z);
    modMulMontP(S2, Q->Y, Z1Z1);
    modMulMontP(S2, S2, P->Z);
    modSubP(H, U2, U1);
    modSubP(r, S2, S1);

    if ((H[0] | H[1] | H[2] | H[3]) == 0) {
        if ((r[0] | r[1] | r[2] | r[3]) == 0) {
            jacobianDouble(R, P);
        } else {
            jacobianSetInfinity(R);
        }
        return;
    }

    modMulMontP(I, H, H);
    modMulMontP(J, I, H);
    modMulMontP(V, U1, I);
    modMulMontP(newX, r, r);
    modSubP(newX, newX, J);
    modSubP(newX, newX, V);
    modSubP(newX, newX, V);
    modSubP(newY, V, newX);
    modMulMontP(newY, newY, r);
    modMulMontP(SJ, S1, J);
    modSubP(newY, newY, SJ);
    modMulMontP(newZ, P->Z, Q->Z);
    modMulMontP(newZ, newZ, H);

    for(int i=0; i<4; i++) {
        R->X[i] = newX[i];
        R->Y[i] = newY[i];
        R->Z[i] = newZ[i];
    }
    R->infinity = 0;
}

__host__ __device__ void endomorphismMap(ECPointJacobian *R, const ECPointJacobian *P) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&BETA_REF)[4] = BETA_P;
    #else
        const uint64_t (&BETA_REF)[4] = BETA_P_HOST;
    #endif

    uint64_t beta_mont[4];
    toMontgomeryP(beta_mont, BETA_REF);
    modMulMontP(R->X, P->X, beta_mont);

    if (R != P) {
        for (int i = 0; i < 4; i++) {
            R->Y[i] = P->Y[i];
            R->Z[i] = P->Z[i];
        }
        R->infinity = P->infinity;
    }
}

__host__ __device__ void initPreCompG(int windowSize) {

    #ifdef __CUDA_ARCH__
	const uint64_t (&GX_REF)[4]  = GX_CONST;
        const uint64_t (&GY_REF)[4]  = GY_CONST;
        const uint64_t (&ONE_REF)[4] = ONE_MONT;
	ECPointJacobian* &preCompG_REF    = preCompG;
    	ECPointJacobian* &preCompGphi_REF = preCompGphi;
    	ECPointJacobian* &jacNorm_REF     = jacNorm;
    	ECPointJacobian* &jacEndo_REF     = jacEndo;
    #else
	const uint64_t (&GX_REF)[4]  = GX_CONST_HOST;
    	const uint64_t (&GY_REF)[4]  = GY_CONST_HOST;
    	const uint64_t (&ONE_REF)[4] = ONE_MONT_HOST;
	ECPointJacobian* &preCompG_REF    = preCompG_HOST;
    	ECPointJacobian* &preCompGphi_REF = preCompGphi_HOST;
    	ECPointJacobian* &jacNorm_REF     = jacNorm_HOST;
    	ECPointJacobian* &jacEndo_REF     = jacEndo_HOST;
    #endif

    int dnorm = (128 + windowSize - 1) / windowSize;
    int dphi = (128 + windowSize - 1) / windowSize;
    int tableSize = (1 << windowSize) - 1;

    uint64_t gxMont[4], gyMont[4];
    toMontgomeryP(gxMont, GX_REF);
    toMontgomeryP(gyMont, GY_REF);

    ECPointJacobian g, ge;
    for (int i = 0; i < 4; i++) {
        g.X[i]  = gxMont[i];
        g.Y[i]  = gyMont[i];
        g.Z[i]  = ONE_REF[i];
        ge.X[i] = gxMont[i];
        ge.Y[i] = gyMont[i];
        ge.Z[i] = ONE_REF[i];
    }
    g.infinity  = 0;
    ge.infinity = 0;

    endomorphismMap(&ge, &ge);

    jacNorm_REF[0]  = g;
    jacEndo_REF[0] = ge;

    for (int j = 1; j < windowSize; j++) {
        jacNorm_REF[j]  = jacNorm_REF[j - 1];
        jacEndo_REF[j]  = jacEndo_REF[j - 1];

        for (int i = 0; i < dnorm; i++)
            jacobianDouble(&jacNorm_REF[j], &jacNorm_REF[j]);

        for (int i = 0; i < dphi; i++)
            jacobianDouble(&jacEndo_REF[j], &jacEndo_REF[j]);
    }

    for (int i = 1; i <= tableSize; i++) {
        jacobianSetInfinity(&preCompG_REF[i - 1]);

        for (int j = 0; j < windowSize; j++) {
            if ((i >> j) & 1) {
                ECPointJacobian tmp;
                jacobianAdd(&tmp, &preCompG_REF[i - 1], &jacNorm_REF[j]);
                preCompG_REF[i - 1] = tmp;
            }
        }

        jacobianSetInfinity(&preCompGphi_REF[i - 1]);

        for (int j = 0; j < windowSize; j++) {
            if ((i >> j) & 1) {
                ECPointJacobian tmp;
                jacobianAdd(&tmp, &preCompGphi_REF[i - 1], &jacEndo_REF[j]);
                preCompGphi_REF[i - 1] = tmp;
            }
        }
    }
}

__host__ __device__ void initPreCompH(const ECPointJacobian *h, int windowSize) {

    #ifdef __CUDA_ARCH__
        ECPointJacobian* &preCompH_REF    = preCompH;
        ECPointJacobian* &preCompHphi_REF = preCompHphi;
        ECPointJacobian* &jacNormH_REF    = jacNormH;
        ECPointJacobian* &jacEndoH_REF    = jacEndoH;
    #else
        ECPointJacobian* &preCompH_REF    = preCompH_HOST;
        ECPointJacobian* &preCompHphi_REF = preCompHphi_HOST;
        ECPointJacobian* &jacNormH_REF    = jacNormH_HOST;
        ECPointJacobian* &jacEndoH_REF    = jacEndoH_HOST;
    #endif

    int dnorm = (128 + windowSize - 1) / windowSize;
    int dphi = (128 + windowSize - 1) / windowSize;
    int tableSize = (1 << windowSize) - 1;

    ECPointJacobian H, HE;
    for (int i = 0; i < 4; i++) {
        H.X[i]  = h->X[i];
        H.Y[i]  = h->Y[i];
        H.Z[i]  = h->Z[i];
        HE.X[i] = h->X[i];
        HE.Y[i] = h->Y[i];
        HE.Z[i] = h->Z[i];
    }
    H.infinity  = h->infinity;
    HE.infinity = h->infinity;

    endomorphismMap(&HE, &HE);

    jacNormH_REF[0] = H;
    jacEndoH_REF[0] = HE;

    for (int j = 1; j < windowSize; j++) {
        jacNormH_REF[j]  = jacNormH_REF[j - 1];
        jacEndoH_REF[j]  = jacEndoH_REF[j - 1];

        for (int i = 0; i < dnorm; i++)
            jacobianDouble(&jacNormH_REF[j], &jacNormH_REF[j]);

        for (int i = 0; i < dphi; i++)
            jacobianDouble(&jacEndoH_REF[j], &jacEndoH_REF[j]);
    }

    for (int i = 1; i <= tableSize; i++) {
        jacobianSetInfinity(&preCompH_REF[i - 1]);

        for (int j = 0; j < windowSize; j++) {
            if ((i >> j) & 1) {
                ECPointJacobian tmp;
                jacobianAdd(&tmp, &preCompH_REF[i - 1], &jacNormH_REF[j]);
                preCompH_REF[i - 1] = tmp;
            }
        }

        jacobianSetInfinity(&preCompHphi_REF[i - 1]);

        for (int j = 0; j < windowSize; j++) {
            if ((i >> j) & 1) {
                ECPointJacobian tmp;
                jacobianAdd(&tmp, &preCompHphi_REF[i - 1], &jacEndoH_REF[j]);
                preCompHphi_REF[i - 1] = tmp;
            }
        }
    }
}

__host__ __device__ void scalarMulShiftVar(uint64_t r[4], const uint64_t a[4], const uint64_t b[4], int shift) {
    uint64_t t[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)a[i] * b[j];
            prod += t[i+j];
            prod += carry;
            t[i+j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }
        t[i+4] = carry;
    }

    int word_shift = shift / 64;
    int bit_shift  = shift % 64;
    for (int i = 0; i < 4; i++) {
        uint128_t v = 0;
        if (i + word_shift < 8) v = uint128_t(t[i + word_shift]);
        if (bit_shift && i + word_shift + 1 < 8) {
            v += uint128_t(t[i + word_shift + 1]) << 64;
        }
        r[i] = (uint64_t)(v >> bit_shift);
    }
}

__host__ __device__ void scalarMul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&N_REF)[4] = N_CONST;
    #else
        const uint64_t (&N_REF)[4] = N_CONST_HOST;
    #endif

    uint64_t t[8] = {0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;

        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)a[i] * b[j] + t[i + j] + carry;
            t[i + j] = (uint64_t)prod;
            carry = (uint64_t)(prod >> 64);
        }

        t[i + 4] = carry;
    }

    static const uint64_t N_C[4] = {
        0x402DA1732FC9BEBFULL,
        0x4551231950B75FC4ULL,
        0x0000000000000001ULL,
        0x0000000000000000ULL
    };

    uint64_t acc[8] = {t[0], t[1], t[2], t[3], 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;

        for (int j = 0; j < 4; j++) {
            uint128_t prod = (uint128_t)t[i + 4] * N_C[j] + ( (i+j < 4) ? acc[i + j] : 0 ) + carry;

            if (i + j < 4) {
                acc[i + j] = (uint64_t)prod;
            } else if (i + j == 4) {
                acc[4] += (uint64_t)prod;
            }

            carry = (uint64_t)(prod >> 64);
        }

        if (i < 3) acc[i + 4] += carry;
    }

    uint128_t c = 0;

    for (int j = 0; j < 4; j++) {
        c = (uint128_t)acc[j] + (uint128_t)acc[4] * N_C[j] + (uint64_t)(c >> 64);
        acc[j] = (uint64_t)c;
    }

    for (int iter = 0; iter < 3; iter++) {
        uint64_t borrow = 0;
        uint64_t diff[4];
        for (int i = 0; i < 4; i++) {
            uint128_t sub = (uint128_t)acc[i] - N_REF[i] - borrow;
            diff[i] = (uint64_t)sub;
            borrow = (sub >> 127) & 1;
        }

        if (!borrow) {
            for(int i=0; i<4; i++) acc[i] = diff[i];
        } else {
            break;
        }
    }

    for (int i = 0; i < 4; i++) r[i] = acc[i];
}

__host__ __device__ void scalarAdd(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&N_REF)[4] = N_CONST;
    #else
        const uint64_t (&N_REF)[4] = N_CONST_HOST;
    #endif

    uint128_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint128_t sum = (uint128_t)a[i] + (uint128_t)b[i] + carry;
        r[i] = (uint64_t)sum;
        carry = sum >> 64;
    }

    if (carry) {
        uint128_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint128_t res = (uint128_t)r[i] - N_REF[i] - borrow;
            r[i] = (uint64_t)res;
            borrow = (res >> 127) & 1;
        }
    } else {
        scalarReduceN(r, r);
    }
}

__host__ __device__ void scalarSub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&N_REF)[4] = N_CONST;
    #else
        const uint64_t (&N_REF)[4] = N_CONST_HOST;
    #endif

    uint128_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint128_t sub = (uint128_t)a[i] - (uint128_t)b[i] - borrow;
        r[i] = (uint64_t)sub;
        borrow = (sub >> 127) & 1;
    }

    if (borrow) {
        uint128_t carry = 0;
        for (int i = 0; i < 4; i++) {
            uint128_t sum = (uint128_t)r[i] + N_REF[i] + carry;
            r[i] = (uint64_t)sum;
            carry = sum >> 64;
        }
    }
}

__host__ __device__ int scalarIsZero(const uint64_t a[4]) {
    uint64_t acc = 0;
    for (int i = 0; i < 4; i++) {
        acc |= a[i];
    }
    return acc == 0;
}

__host__ __device__ void scalarNeg(uint64_t r[4], const uint64_t a[4]) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&N_REF)[4] = N_CONST;
    #else
        const uint64_t (&N_REF)[4] = N_CONST_HOST;
    #endif

    if (scalarIsZero(a)) {
        for(int i=0; i<4; i++) r[i] = 0;
        return;
    }

    uint128_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint128_t sub = (uint128_t)N_REF[i] - (uint128_t)a[i] - borrow;
        r[i] = (uint64_t)sub;
        borrow = (sub >> 127) & 1;
    }
}

__host__ __device__ void scalarSplitLambda(uint64_t r1[4], uint64_t r2[4], const uint64_t k[4]) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&G1_REF)[4] = G1;
        const uint64_t (&G2_REF)[4] = G2;
        const uint64_t (&B1_REF)[4] = MINUS_B1;
	const uint64_t (&B2_REF)[4] = MINUS_B2;
	const uint64_t (&LAMBDA_REF)[4] = LAMBDA_N;
    #else
        const uint64_t (&G1_REF)[4] = G1_HOST;
        const uint64_t (&G2_REF)[4] = G2_HOST;
        const uint64_t (&B1_REF)[4] = MINUS_B1_HOST;
	const uint64_t (&B2_REF)[4] = MINUS_B2_HOST;
	const uint64_t (&LAMBDA_REF)[4] = LAMBDA_N_HOST;
    #endif

    uint64_t c1[4], c2[4], t1[4], t2[4];

    scalarMulShiftVar(c1, k, G1_REF, 384);
    scalarMulShiftVar(c2, k, G2_REF, 384);
    scalarMul(t1, c1, B1_REF);
    scalarMul(t2, c2, B2_REF);
    scalarAdd(r2, t1, t2);
    scalarMul(t1, r2, LAMBDA_REF);
    scalarNeg(t1, t1);
    scalarAdd(r1, t1, k);
}

__host__ __device__ void jacobianScalarMultPhi(ECPointJacobian *result, ECPointJacobian *preCompTable, ECPointJacobian *preCompTablePhi, const uint64_t *scalar, int windowSize) {
    uint64_t r1[4], r2[4];
    int d = (128 + windowSize - 1) / windowSize;

    scalarSplitLambda(r1, r2, scalar);
    jacobianSetInfinity(result);

    for (int col = d - 1; col >= 0; col--) {
        if (!jacobianIsInfinity(result)) {
            jacobianDouble(result, result);
        }

        int idx1 = 0;
        int idx2 = 0;

        for (int row = 0; row < windowSize; row++) {
            int bitIndex = row * d + col;
            if (bitIndex >= 128) continue;
            int limb = bitIndex / 64;
            int shift = bitIndex % 64;
            uint64_t bit1 = (r1[limb] >> shift) & 1ULL;
            uint64_t bit2 = (r2[limb] >> shift) & 1ULL;
            idx1 |= (bit1 << row);
            idx2 |= (bit2 << row);
        }

        if (idx1 != 0) {
            ECPointJacobian tmp;
            jacobianAdd(&tmp, result, &preCompTable[idx1 - 1]);
            *result = tmp;
        }

        if (idx2 != 0) {
            ECPointJacobian tmp;
            jacobianAdd(&tmp, result, &preCompTablePhi[idx2 - 1]);
            *result = tmp;
        }
    }
}

__host__ __device__ void pointAddJacobian(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q) { jacobianAdd(R, P, Q); }
__host__ __device__ void pointDoubleJacobian(ECPointJacobian *R, const ECPointJacobian *P) { jacobianDouble(R, P); }
__host__ __device__ void serializePublicKey(unsigned char *out, const ECPointAffine *publicKey) {
    unsigned char prefix = (publicKey->y[0] & 1ULL) ? 0x03 : 0x02;
    out[0] = prefix;

    for (int i = 0; i < 4; i++) {
        uint64_t word = publicKey->x[3 - i];
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

__host__ __device__ void generatePublicKey(ECPointJacobian *preCompTable, ECPointJacobian *preCompTablePhi, unsigned char *out, const uint64_t *PRIV_KEY, int windowSize) {
    ECPointAffine pub;
    ECPointJacobian pub_jac;

    jacobianScalarMultPhi(&pub_jac, preCompTable, preCompTablePhi, PRIV_KEY, windowSize);
    jacobianToAffine(&pub, &pub_jac);
    serializePublicKey(out, &pub);
}

__host__ __device__ void decompressPublicKey(ECPointAffine* out, const unsigned char compressed[33]) {

    #ifdef __CUDA_ARCH__
        const uint64_t (&SEVEN_REF)[4] = SEVEN_MONT;
	const uint64_t (&P_REF)[4] = P_CONST;
    #else
        const uint64_t (&SEVEN_REF)[4] = SEVEN_MONT_HOST;
	const uint64_t (&P_REF)[4] = P_CONST_HOST;
    #endif

    unsigned char prefix = compressed[0];

    for (int i = 0; i < 4; i++) {
        uint64_t word = 0;
        for (int j = 0; j < 8; j++) {
            word = (word << 8) | compressed[1 + i * 8 + j];
        }
        out->x[3 - i] = word;
    }

    uint64_t x_mont[4], x2[4], x3[4], rhs[4];

    toMontgomeryP(x_mont, out->x);
    modMulMontP(x2, x_mont, x_mont);
    modMulMontP(x3, x2, x_mont);
    modAddP(rhs, x3, SEVEN_REF);

    uint64_t y_mont[4];
    sqrtModP(y_mont, rhs);
    fromMontgomeryP(out->y, y_mont);

    uint64_t is_odd   = out->y[0] & 1ULL;
    uint64_t want_odd = (prefix == 0x03);

    if (is_odd != want_odd) {
        modSubP(out->y, P_REF, out->y);
    }

    out->infinity = 0;
}
