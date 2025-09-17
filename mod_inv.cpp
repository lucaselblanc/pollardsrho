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

//Based on the Paper Almost-Inverse/Bernstein-Yang, REF: https://eprint.iacr.org/2019/266.pdf

#include <iostream>
#include <tuple>
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::cpp_int;
using BigInt = cpp_int;
using std::make_tuple;
using std::tuple;
using std::get;
using std::pair;
using std::make_pair;

constexpr int MAX_BITS = 256;

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

auto divsteps2(int n, int t, int delta, BigInt g, BigInt f) {
    g = truncate(g, t);
    f = truncate(f, t);

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
    return make_tuple(delta, g, f, P);
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

    auto result = divsteps2(m, m + 1, 1, g, f);
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
