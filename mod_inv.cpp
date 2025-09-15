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

BigInt div2_floor(const BigInt &a) {
    if (a >= 0) return a >> 1;
    BigInt na = -a;
    return - ((na + 1) >> 1);
}

BigInt truncate(const BigInt& f, int t) {
    if (t == 0) return BigInt(0);
    BigInt mask = (BigInt(1) << t) - 1;
    BigInt result = f & mask;
    BigInt bound = (BigInt(1) << (t - 1));
    if (result >= bound) result -= (BigInt(1) << t);
    return result;
}

int bit_length(const BigInt& x) {
    if (x == 0) return 0;
    BigInt a = x >= 0 ? x : -x;
    int bits = 0;
    BigInt tmp = a;
    while (tmp > 0) {
        tmp = tmp >> 1;
        ++bits;
    }
    return bits;
}

BigInt mod_pow(BigInt base, BigInt exp, const BigInt &mod) {
    base %= mod;
    if (base < 0) base += mod;
    BigInt res = 1 % mod;
    while (exp > 0) {
        if ((exp & 1) != 0) res = (res * base) % mod;
        exp >>= 1;
        base = (base * base) % mod;
    }
    return res;
}

auto divsteps2(int n, int t, int delta, BigInt f, BigInt g) {
    f = truncate(f, t);
    g = truncate(g, t);

    BigInt scale = BigInt(1) << n;
    BigInt U = scale;
    BigInt V = 0;
    BigInt Q = 0;
    BigInt R = scale;

    while (n > 0) {
        f = truncate(f, t);
        bool g_odd = ((g & 1) != 0);
        if (delta > 0 && g_odd) {
            delta = -delta;
            BigInt oldf = f;
            f = g;
            g = -oldf;
            std::swap(U, Q);
            std::swap(V, R);
            U = -U;
            V = -V;
        }
        int g0 = (int)((g & 1) != 0 ? 1 : 0);
        delta = 1 + delta;
        BigInt tmpg = g + BigInt(g0) * f;
        g = div2_floor(tmpg);
        Q = div2_floor(Q + BigInt(g0) * U);
        R = div2_floor(R + BigInt(g0) * V);
        --n;
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

BigInt recip2(BigInt f, BigInt g) {
    if ((f & 1) == 0) {
        throw std::invalid_argument("f must be odd");
    }
    int d = std::max(bit_length(f), bit_length(g));
    int m = iterations(d);

    BigInt base = (f + 1) / 2;
    BigInt precomp = mod_pow(base, BigInt(m - 1), f);

    auto result = divsteps2(m, m + 1, 1, f, g);
    int delta = get<0>(result);

    BigInt fm = get<1>(result);
    BigInt gm = get<2>(result);
    auto P = get<3>(result);
    BigInt U_scaled = P.first.first;
    BigInt V_scaled = P.first.second;
    BigInt Q_scaled = P.second.first;
    BigInt R_scaled = P.second.second;
    BigInt V_int = (V_scaled >> 1);

    if (fm < 0) V_int = -V_int;
    BigInt inv = (V_int * precomp) % f;

    if (inv < 0) inv += f;
    return inv;
}

int main() {
    BigInt f = 12345;
    BigInt g = 6789;
    try {
        BigInt inv = recip2(f, g);
        std::cout << "Inverso: " << inv << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Erro: " << e.what() << std::endl;
    }
    return 0;
}