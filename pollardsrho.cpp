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
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>
#include <ctime>
#include <mutex>
#include <cstring>
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::cpp_int;

ECPointJacobian G;
ECPointJacobian H;

const cpp_int N_("0xBFD25E8CD0364141BAAEDCE6AF48A03BFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFF");
const uint256_t N = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
const uint256_t GX = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
const uint256_t GY = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };

int windowSize = 16; //Default value used only if getfcw() detection cannot access the processor for some reason, it can happen on different platforms like termux for example.

void uint256_to_uint64_array(uint64_t* out, const uint256_t& value) {
    for(int i = 0; i < 4; i++) {
        out[i] = value.limbs[i];
    }
}

void getfcw() {
    int w = 4;
    for (int idx = 0;; idx++) {
        std::ifstream levelFile("/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/level");
        if (!levelFile.is_open()) break;
        int L;
        levelFile >> L;

        std::ifstream typeFile("/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/type");
        std::string type;
        typeFile >> type;
        if (type != "Data" && type != "Unified") continue;

        std::ifstream sizeFile("/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/size");
        std::string sizeStr;
        sizeFile >> sizeStr;
        if (sizeStr.empty()) continue;

        size_t mult = 1;
        if (sizeStr.back() == 'K') mult = 1024;
        else if (sizeStr.back() == 'M') mult = 1024*1024;

        try { //Adjust the table to fit in the processor's L2/L3 cache (more fast), avoiding jumping to RAM.
            size_t size = std::stoul(sizeStr.substr(0, sizeStr.size()-1)) * mult;

            if (L == 2)
            {
                size_t maxPoints = size / 128;
                if (maxPoints == 0) continue;
                w = static_cast<int>(std::floor(std::log2(maxPoints)));
                break;
            }
        }
        catch(const std::invalid_argument& e) {
            std::cout << "Warning: " << e.what() << std::endl;
            continue;
        }
        catch(const std::out_of_range& e) {
            std::cout << "Warning: " << e.what() << std::endl;
            continue;
        }
    }

    if(w > 4) {
        windowSize = w;
    }
}

void init_secp256k1() {

    getfcw();

    #ifdef __CUDACC__
        cudaMalloc(&preCompG, sizeof(ECPointJacobian) * (1ULL << windowSize));
        cudaMalloc(&preCompGphi, sizeof(ECPointJacobian) * (1ULL << windowSize));
        cudaMalloc(&jacNorm, sizeof(ECPointJacobian) * windowSize);
        cudaMalloc(&jacEndo, sizeof(ECPointJacobian) * windowSize);
    #else
        preCompG = new ECPointJacobian[1ULL << windowSize];
        preCompGphi = new ECPointJacobian[1ULL << windowSize];
        jacNorm = new ECPointJacobian[windowSize];
        jacEndo = new ECPointJacobian[windowSize];
    #endif

    initPrecompG(windowSize);

    uint64_t gx_arr[4], gy_arr[4];
    uint256_to_uint64_array(gx_arr, GX);
    uint256_to_uint64_array(gy_arr, GY);

    for (int i = 0; i < 4; i++) {
        G.X[i] = gx_arr[i];
        G.Y[i] = gy_arr[i];
        G.Z[i] = (i == 3) ? 1 : 0;
    }
    G.infinity = 0;

    #ifndef __CUDACC__
        delete[] jacNorm;
        delete[] jacEndo;
    #endif
}

cpp_int random_mod_n(const cpp_int& N, std::mt19937_64& rng) {
    cpp_int r = 0;

    for (int i = 0; i < 4; i++) {
        r <<= 64;
        r |= cpp_int(rng());
    }

    r %= N;
    return r;
}

uint256_t cppint_to_uint256(const cpp_int& r) {
    uint256_t result{};
    cpp_int tmp = r;

    for (int i = 3; i >= 0; i--) {
        result.limbs[i] = static_cast<uint64_t>(tmp & 0xFFFFFFFFFFFFFFFFULL);
        tmp >>= 64;
    }

    return result;
}

uint256_t sub_uint256(const uint256_t& a, const uint256_t& b) {
    uint256_t result{};
    uint64_t borrow = 0;

    for (int i = 3; i >= 0; --i) {
        uint64_t ai = a.limbs[i];
        uint64_t bi = b.limbs[i];
        uint64_t temp = ai - bi - borrow;
        borrow = ((ai < bi + borrow) ? 1 : 0);
        result.limbs[i] = temp;
    } return result;
}

int compare_uint256(const uint256_t& a, const uint256_t& b) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return 1;
        if (a.limbs[i] < b.limbs[i]) return -1;
    } return 0;
}

uint256_t uint256_from_uint32(uint32_t value) {
    uint256_t r{};
    r.limbs[3] = value;
    for(int i = 0; i < 3; i++) r.limbs[i] = 0;
    return r;
}

uint256_t add_uint256(const uint256_t& a, const uint256_t& b) {
    uint256_t result{};
    uint64_t carry = 0;
    for(int i = 3; i >= 0; i--) {
        uint64_t sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i] || (carry && sum == a.limbs[i])) ? 1 : 0;
        result.limbs[i] = sum;
    } return result;
}

struct Buffers {
    ECPointJacobian* d_R;
    ECPointJacobian* d_G;
    ECPointJacobian* d_H;
    uint64_t* d_k;

    Buffers() {
        #ifdef __CUDACC__
            cudaMalloc(&d_R, sizeof(ECPointJacobian));
            cudaMalloc(&d_G, sizeof(ECPointJacobian));
            cudaMalloc(&d_H, sizeof(ECPointJacobian));
            cudaMalloc(&d_k, sizeof(uint64_t) * 4);
            cudaMemcpy(d_G, &G, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
            cudaMemcpy(d_H, &H, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
        #endif
    }

    ~Buffers() {
        #ifdef __CUDACC__
            cudaFree(d_R);
            cudaFree(d_G);
            cudaFree(d_H);
            cudaFree(d_k);
        #endif
    }
};

void f(ECPointJacobian& R, uint256_t& a, uint256_t& b, Buffers& buffers) {
    uint64_t x_low64 = R.X[3];
    unsigned int op = static_cast<unsigned int>(x_low64 % 3);

    #ifdef __CUDACC__
        cudaMemcpy(buffers.d_R, &R, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
    #endif

    switch (op) {
        case 0:
        #ifdef __CUDACC__
            pointAddJacobian(buffers.d_R, buffers.d_R, buffers.d_G);
        #else
            pointAddJacobian(&R, &R, &G);
        #endif
            a = add_uint256(a, uint256_from_uint32(1));
            if (compare_uint256(a, N) >= 0)
                a = sub_uint256(a, N);
            break;

        case 1:
        #ifdef __CUDACC__
            pointAddJacobian(buffers.d_R, buffers.d_R, buffers.d_H);
        #else
            pointAddJacobian(&R, &R, &H);
        #endif
            b = add_uint256(b, uint256_from_uint32(1));
            if (compare_uint256(b, N) >= 0)
                b = sub_uint256(b, N);
            break;

        default:
        #ifdef __CUDACC__
            pointDoubleJacobian(buffers.d_R, buffers.d_R);
        #else
            pointDoubleJacobian(&R, &R);
        #endif
            a = add_uint256(a, a);
            if (compare_uint256(a, N) >= 0)
                a = sub_uint256(a, N);

            b = add_uint256(b, b);
            if (compare_uint256(b, N) >= 0)
                b = sub_uint256(b, N);
        break;
    }

    #ifdef __CUDACC__
        cudaMemcpy(&R, buffers.d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
    #endif
}

uint64_t hashed_dp(const ECPointJacobian& P) {
    uint64_t h = 0;
    for(int i = 0; i < 4; i++) {
        h ^= P.X[i] + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2);
    }
    return h;
}

bool DP(const ECPointJacobian& P, int LSB) {
    return (P.X[3] & ((1ULL << LSB) - 1)) == 0;
}

uint256_t prho(std::string target_pubkey_hex, int key_range, int hares, bool test_mode) {

    std::atomic<bool> search_in_progress(true);

    auto start_time = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::tm start_tm = *std::localtime(&start_time_t);

    auto hex_to_bytes = [](const std::string& hex) -> std::vector<unsigned char> {
        std::vector<unsigned char> bytes(hex.size() / 2);
        for (size_t i = 0; i < bytes.size(); i++)
            bytes[i] = static_cast<unsigned char>(
                std::stoi(hex.substr(2 * i, 2), nullptr, 16));
        return bytes;
    };

    auto target_pubkey = hex_to_bytes(target_pubkey_hex);

    auto uint_256_to_hex = [](const uint256_t& value) -> std::string {
        std::ostringstream oss;
        for(int i = 0; i < 4; i++) {
            oss << std::setw(16) << std::setfill('0') << std::hex << value.limbs[i];
        }
        return oss.str();
    };

    ECPoint target_affine{};
    getCompressedPublicKey(target_pubkey.data(), &target_affine);

    for (int i = 0; i < 4; i++) {
        H.X[i] = target_affine.x[i];
        H.Y[i] = target_affine.y[i];
        H.Z[i] = (i == 3) ? 1 : 0;
    }
    H.infinity = target_affine.infinity;

    unsigned long long tested_keys = 0;
    uint256_t current_coeff{};

    uint256_t min_scalar{}, max_scalar{};
    {
        int limb_index = 3 - ((key_range - 1) / 64);
        int bit_in_limb = (key_range - 1) % 64;
        min_scalar.limbs[limb_index] = 1ULL << bit_in_limb;

        int full_limbs = key_range / 64;
        int rem_bits = key_range % 64;
        for (int i = 4 - full_limbs; i < 4; i++)
            max_scalar.limbs[i] = 0xFFFFFFFFFFFFFFFFULL;
        if (rem_bits)
            max_scalar.limbs[4 - full_limbs - 1] = (1ULL << rem_bits) - 1;
    }

    std::cout << "Started at: " << std::put_time(&start_tm, "%H:%M:%S") << std::endl;
    if(test_mode) { std::cout << "Test Mode: True" << std::endl; }
    else          { std::cout << "Test Mode: False" << std::endl; }

    std::cout << "key_range: " << key_range << std::endl;
    std::cout << "min_range: " << uint_256_to_hex(min_scalar) << std::endl;
    std::cout << "max_range: " << uint_256_to_hex(max_scalar) << std::endl;

    struct HareState {
        ECPointJacobian R;
        uint256_t a, b;
        Buffers* buffers;
    };

    std::unordered_map<uint64_t, HareState> dp_table;
    std::vector<HareState> hares_state(hares);

    std::mt19937_64 rng(std::random_device{}());

    for (int i = 0; i < hares; i++) {

        hares_state[i].buffers = new Buffers();
        hares_state[i].a = cppint_to_uint256(random_mod_n(N_, rng));
        hares_state[i].b = cppint_to_uint256(random_mod_n(N_, rng));

        uint64_t a_arr[4], b_arr[4];
        uint256_to_uint64_array(a_arr, hares_state[i].a);
        uint256_to_uint64_array(b_arr, hares_state[i].b);

        ECPointJacobian Ra{}, Rb{};

        #ifdef __CUDACC__
        cudaMemcpy(hares_state[i].buffers->d_k, a_arr, sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
        scalarMultJacobian(hares_state[i].buffers->d_R,
                           hares_state[i].buffers->d_k,
                           key_range,
                           windowSize);
        cudaMemcpy(&Ra, hares_state[i].buffers->d_R,
                   sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);

        cudaMemcpy(hares_state[i].buffers->d_G, &H,
                   sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
        cudaMemcpy(hares_state[i].buffers->d_k, b_arr,
                   sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
        scalarMultJacobian(hares_state[i].buffers->d_R,
                           hares_state[i].buffers->d_k,
                           key_range,
                           windowSize);
        cudaMemcpy(&Rb, hares_state[i].buffers->d_R,
                   sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
        #else
        scalarMultJacobian(&Ra, a_arr, key_range, windowSize);
        scalarMultJacobian(&Rb, b_arr, key_range, windowSize);
        #endif
        pointAddJacobian(&hares_state[i].R, &Ra, &Rb);
    }

    auto last_print = std::chrono::steady_clock::now();
    HareState* h = nullptr;

    if (test_mode) {
        ECPointJacobian INIT_JUMP{};
        uint256_t jump{};
        jump.limbs[3] = 1ULL << (key_range / 2); //Initial jump of 2^50% of the key range

        uint64_t jump_arr[4];
        uint256_to_uint64_array(jump_arr, jump);
        scalarMultJacobian(&INIT_JUMP, jump_arr, key_range, windowSize);

        for (int i = 0; i < hares; i++) {

            h = &hares_state[i];

            h->b = add_uint256(h->b, jump);
            if (compare_uint256(h->b, N) >= 0) h->b = sub_uint256(h->b, N);

            pointAddJacobian(&h->R, &h->R, &INIT_JUMP);
        }
    }

    uint256_t k{};

    while (search_in_progress.load()) {
        for (int i = 0; i < hares; i++) {

            tested_keys++;

            h = &hares_state[i];

            auto pointsEqual = [](const ECPointJacobian& A, const ECPointJacobian& B) -> bool {
                if (A.infinity != B.infinity) return false;
                if (A.infinity) return true;

                for (int i = 0; i < 4; i++) {
                    if (A.X[i] != B.X[i]) return false;
                    if (A.Z[i] != B.Z[i]) return false;
                }
                return true;
            };

            f(h->R, h->a, h->b, *h->buffers);

            if (compare_uint256(h->a, N) >= 0) h->a = sub_uint256(h->a, N);
            if (compare_uint256(h->b, N) >= 0) h->b = sub_uint256(h->b, N);

            current_coeff = h->a;

            if (!DP(h->R, 5)) continue;

            uint64_t hash = hashed_dp(h->R);

            if (!dp_table.count(hash)) {
                dp_table[hash] = *h;
                continue;
            }

            auto& other = dp_table[hash];

            if (!pointsEqual(h->R, other.R)) continue;

            uint256_t diff_coeff_a =
                (compare_uint256(h->a, other.a) >= 0)
                    ? sub_uint256(h->a, other.a)
                    : sub_uint256(N, sub_uint256(other.a, h->a));

            uint256_t diff_coeff_b =
                (compare_uint256(other.b, h->b) >= 0)
                    ? sub_uint256(other.b, h->b)
                    : sub_uint256(N, sub_uint256(h->b, other.b));

            std::cout << "\033[32mCollision found!\033[0m" << std::endl;
            std::cout << "Difference of the coefficient a: " << uint_256_to_hex(diff_coeff_a) << std::endl;
            std::cout << "Difference of the coefficient b: " << uint_256_to_hex(diff_coeff_b) << std::endl;
            std::cout << "Candidate a: " << uint_256_to_hex(h->a) << std::endl;
            std::cout << "Candidate b: " << uint_256_to_hex(h->b) << std::endl;

            uint256_t inv_diff_coeff_b = almostinverse(diff_coeff_b, N);

            uint64_t a_s[4], b_s[4], k_s[4];

            uint256_to_uint64_array(a_s, diff_coeff_a);
            uint256_to_uint64_array(b_s, inv_diff_coeff_b);

            scalarMul(k_s, a_s, b_s);

            for (int i = 0; i < 4; i++) {
                k.limbs[i] = k_s[i];
            }

            for (auto& hs : hares_state) delete hs.buffers;

            search_in_progress.store(false);
        }

        auto now = std::chrono::steady_clock::now();
        if (now - last_print >= std::chrono::seconds(10)) {
            std::cout << "\rCurrent coeff: " << uint_256_to_hex(current_coeff) << std::endl;
            std::cout << "\rTotal keys tested: " << tested_keys << std::endl;
            last_print = now;
        }
    }

    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Total duration: " << std::setw(2) << std::setfill('0') << duration.count() / 3600 << ":"
    << std::setw(2) << std::setfill('0') << (duration.count() % 3600) / 60 << ":"
    << std::setw(2) << std::setfill('0') << duration.count() % 60 << std::endl;
    return k;
}

int main(int argc, char* argv[]) {

    if (argc < 3 || argc > 4) {
        std::cerr << "Uso: " << argv[0] << " <Compressed Public Key> <Key Range> <Optional Test Mode: --t to true>" << std::endl;
        return 1;
    }

    init_secp256k1();

    std::string pub_key_hex(argv[1]);
    int key_range = std::stoi(argv[2]);
    bool test_mode = (argc == 4 && std::string(argv[3]) == "--t");

    std::cout << "Press 'Ctrl \\' to Quit\n";
    std::cout << "Auto Window-Size for secp256k1: " << windowSize << std::endl;

    uint256_t found_key = prho(pub_key_hex, key_range, 3, test_mode);

    auto uint_256_to_hex = [](const uint256_t& value) -> std::string {
        std::ostringstream oss;
        for(int i = 0; i < 4; i++) {
            oss << std::setw(16) << std::setfill('0') << std::hex << value.limbs[i];
        }
        return oss.str();
    };

    std::cout << "Chave privada encontrada: " << uint_256_to_hex(found_key) << std::endl;

    return 0;

}
