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

#ifndef EC_SECP256K1_H
#define EC_SECP256K1_H

#include <boost/multiprecision/cpp_int.hpp>
#include <stdint.h>
#include "parallel_hashmap/phmap.h"
#include <condition_variable>
#include <openssl/sha.h>
#include <fstream>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <limits>
#include <climits>
#include <ctime>
#include <cmath>
#include <cstring>
#include <tuple>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

struct uint256_t {
    uint64_t limbs[4];
};

typedef boost::multiprecision::cpp_int BigInt;

typedef struct {
    uint64_t x[4];
    uint64_t y[4];
    int infinity;
} ECPointAffine;

typedef struct {
    uint64_t X[4];
    uint64_t Y[4];
    uint64_t Z[4];
    int infinity;
} ECPointJacobian;

using uint128_t = unsigned __int128;

extern ECPointJacobian* preCompG;
extern ECPointJacobian* preCompGphi;
extern ECPointJacobian* preCompH;
extern ECPointJacobian* preCompHphi;
extern ECPointJacobian* jacNorm;
extern ECPointJacobian* jacEndo;
extern ECPointJacobian* jacNormH;
extern ECPointJacobian* jacEndoH;

uint256_t modinv(uint256_t base, uint256_t mod);

void affineToJacobian(ECPointJacobian *jac, const ECPointAffine *aff);
void decompressPublicKey(ECPointAffine* out, const unsigned char compressed[33]);
void endomorphismMap(ECPointJacobian *R, const ECPointJacobian *P);
void fromMontgomeryP(uint64_t *result, const uint64_t *a);
void generatePublicKey(ECPointJacobian *preCompTable, ECPointJacobian *preCompTablePhi, unsigned char *out, const uint64_t *PRIV_KEY, int windowSize);
void initPreCompG(int windowSize);
void initPreCompH(const ECPointJacobian *h, int windowSize);
void jacobianScalarMultPhi(ECPointJacobian *result, ECPointJacobian *preCompTable, ECPointJacobian *preCompTablePhi, const uint64_t *scalar, int windowSize);
void jacobianDouble(ECPointJacobian *R, const ECPointJacobian *P);
void jacobianAdd(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q);
void jacobianToAffine(ECPointAffine *aff, const ECPointJacobian *jac);
void jacobianSetInfinity(ECPointJacobian *point);
bool jacobianIsInfinity(const ECPointJacobian *P);
void modMulMontP(uint64_t *result, const uint64_t *a, const uint64_t *b);
void modSubP(uint64_t *result, const uint64_t *a, const uint64_t *b);
void modAddP(uint64_t *result, const uint64_t *a, const uint64_t *b);
void modExpMontP(uint64_t *res, const uint64_t *base, const uint64_t *exp);
void pointInitJacobian(ECPointJacobian *P);
void pointAddJacobian(ECPointJacobian *R, const ECPointJacobian *P, const ECPointJacobian *Q);
void pointDoubleJacobian(ECPointJacobian *R, const ECPointJacobian *P);
void scalarReduceN(uint64_t *r, const uint64_t *k);
void scalarMul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
void scalarAdd(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
void scalarSub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]);
void scalarNeg(uint64_t r[4], const uint64_t a[4]);
int scalarIsZero(const uint64_t a[4]);
void serializePublicKey(unsigned char *out, const ECPointAffine *publicKey);
void toMontgomeryP(uint64_t *result, const uint64_t *a);

#endif /* EC_SECP256K1_H */