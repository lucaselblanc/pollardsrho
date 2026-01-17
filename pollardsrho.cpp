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

const cpp_int N_("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
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
    cpp_int r;
    cpp_int limit = (cpp_int(1) << 256) - ((cpp_int(1) << 256) % N);

    do {
        r = 0;
        for (int i = 0; i < 4; i++) {
            r <<= 64;
            r |= cpp_int(rng());
        }
    } while (r >= limit);

    return r % N;
}

uint256_t cppint_to_uint256(const cpp_int& v) {
    uint256_t out{};
    cpp_int tmp = v;

    for (int i = 0; i < 4; i++) {
        out.limbs[i] = static_cast<uint64_t>(tmp & cpp_int(0xFFFFFFFFFFFFFFFFULL));
        tmp >>= 64;
    }

    return out;
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
    }
    return 0;
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
            if (compare_uint256(a, N) >= 0) a = sub_uint256(a, N);
            break;

        case 1:
        #ifdef __CUDACC__
            pointAddJacobian(buffers.d_R, buffers.d_R, buffers.d_H);
        #else
            pointAddJacobian(&R, &R, &H);
        #endif
            b = add_uint256(b, uint256_from_uint32(1));
            if (compare_uint256(b, N) >= 0) b = sub_uint256(b, N);
            break;

        default:
        #ifdef __CUDACC__
            pointDoubleJacobian(buffers.d_R, buffers.d_R);
        #else
            pointDoubleJacobian(&R, &R);
        #endif
            a = add_uint256(a, a);
            if (compare_uint256(a, N) >= 0) a = sub_uint256(a, N);

            b = add_uint256(b, b);
            if (compare_uint256(b, N) >= 0) b = sub_uint256(b, N);
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

uint256_t prho(std::string target_pubkey_hex, int key_range, int walkers, bool test_mode) {

    std::atomic<bool> search_in_progress(true);
    std::atomic<unsigned long long> total_iters{0};
    std::mutex k_mutex;
    std::mutex dp_mutex;
    std::mutex io_mutex;

    auto start_time = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::tm start_tm{};
    localtime_r(&start_time_t, &start_tm);

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
    decompressPublicKey(&target_affine, target_pubkey.data());

    for (int i = 0; i < 4; i++) {
        H.X[i] = target_affine.x[i];
        H.Y[i] = target_affine.y[i];
        H.Z[i] = (i == 3) ? 1 : 0;
    }
    H.infinity = target_affine.infinity;

    uint256_t k{};
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

    struct WalkState {
        ECPointJacobian R;
        uint256_t a, b;
        Buffers* buffers;
        uint32_t walk_id;
    };

    std::unordered_map<uint64_t, WalkState> dp_table;
    std::vector<WalkState> walkers_state(walkers);

    std::mt19937_64 rng(std::random_device{}());

    for (int i = 0; i < walkers; i++) {
        walkers_state[i].buffers = new Buffers();
        walkers_state[i].a = cppint_to_uint256(random_mod_n(N_, rng));
        walkers_state[i].b = cppint_to_uint256(random_mod_n(N_, rng));
        walkers_state[i].walk_id = i;

        uint64_t a_arr[4], b_arr[4];
        uint256_to_uint64_array(a_arr, walkers_state[i].a);
        uint256_to_uint64_array(b_arr, walkers_state[i].b);

        ECPointJacobian Ra{}, Rb{};

        #ifdef __CUDACC__
            cudaMemcpy(walkers_state[i].buffers->d_k, a_arr, sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
            scalarMultJacobian(walkers_state[i].buffers->d_R, walkers_state[i].buffers->d_k, windowSize);
            cudaMemcpy(&Ra, walkers_state[i].buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
            cudaMemcpy(walkers_state[i].buffers->d_G, &H, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
            cudaMemcpy(walkers_state[i].buffers->d_k, b_arr, sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
            scalarMultJacobian(walkers_state[i].buffers->d_R, walkers_state[i].buffers->d_k, windowSize);
            cudaMemcpy(&Rb, walkers_state[i].buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
        #else
            scalarMultJacobian(&Ra, a_arr, windowSize);
            scalarMultJacobian(&Rb, b_arr, windowSize);
        #endif
            pointAddJacobian(&walkers_state[i].R, &Ra, &Rb);
    }

    auto last_print = std::chrono::steady_clock::now();

    if (test_mode) {
        ECPointJacobian INIT_JUMP{};
        uint256_t jump{};
 
        //Initial jump of 2^(key_range/2) of the key range
        int bit_index = key_range / 2;
        int limb_index = 3 - (bit_index / 64);
        int bit_in_limb = bit_index % 64;

        jump.limbs[limb_index] = 1ULL << bit_in_limb;

        uint64_t jump_arr[4];
        uint256_to_uint64_array(jump_arr, jump);
        scalarMultJacobian(&INIT_JUMP, jump_arr, windowSize);

        for (int i = 0; i < walkers; i++) {
            WalkState* w = &walkers_state[i];

            w->b = add_uint256(w->b, jump);
            if (compare_uint256(w->b, N) >= 0) w->b = sub_uint256(w->b, N);

            pointAddJacobian(&w->R, &w->R, &INIT_JUMP);
        }
    }

    auto pointsEqual = [](const ECPointJacobian& A, const ECPointJacobian& B) -> bool {
        if (A.infinity != B.infinity) return false;
        if (A.infinity) return true;

        for (int i = 0; i < 4; i++) {
            if (A.X[i] != B.X[i]) return false;
            if (A.Z[i] != B.Z[i]) return false;
        }
        return true;
    };

    auto worker = [&](int start, int end) {
        while (search_in_progress.load(std::memory_order_acquire)) {
            for (int i = start; i < end && search_in_progress.load(std::memory_order_acquire); i++) {
                total_iters.fetch_add(1, std::memory_order_relaxed);

                WalkState* w = &walkers_state[i];

                f(w->R, w->a, w->b, *w->buffers);

                if (compare_uint256(w->a, N) >= 0) w->a = sub_uint256(w->a, N);
                if (compare_uint256(w->b, N) >= 0) w->b = sub_uint256(w->b, N);
                if (!DP(w->R, 5)) continue;

                uint64_t hash = hashed_dp(w->R);

                {
                    std::lock_guard<std::mutex> lock(dp_mutex);

                    auto dp_entry = dp_table.find(hash);
                    if (dp_entry == dp_table.end()) {
                        WalkState wst = *w;
                        wst.buffers = nullptr;
                        dp_table.emplace(hash, wst);

                        continue;
                    }

                    if (dp_entry->second.walk_id == w->walk_id) continue;
                    if (!pointsEqual(w->R, dp_entry->second.R)) continue;

                    uint256_t diff_coeff_a = (compare_uint256(w->a, dp_entry->second.a) >= 0) ? sub_uint256(w->a, dp_entry->second.a) : sub_uint256(N, sub_uint256(dp_entry->second.a, w->a));
                    uint256_t diff_coeff_b = (compare_uint256(dp_entry->second.b, w->b) >= 0) ? sub_uint256(dp_entry->second.b, w->b) : sub_uint256(N, sub_uint256(w->b, dp_entry->second.b));
                    uint256_t inv_diff_coeff_b = almostinverse(diff_coeff_b, N);

                    std::cout << "\033[32mCollision found!\033[0m" << std::endl;
                    std::cout << "Difference of the coefficient A: " << uint_256_to_hex(diff_coeff_a) << std::endl;
                    std::cout << "Difference of the coefficient B: " << uint_256_to_hex(diff_coeff_b) << std::endl;
                    std::cout << "Scalar Coefficient A (Non-Key): " << uint_256_to_hex(w->a) << std::endl;
                    std::cout << "Scalar Coefficient B (Non-Key): " << uint_256_to_hex(w->b) << std::endl;

                    uint64_t a_s[4], b_s[4], k_s[4];
                    uint256_to_uint64_array(a_s, diff_coeff_a);
                    uint256_to_uint64_array(b_s, inv_diff_coeff_b);

                    scalarMul(k_s, a_s, b_s);
                    std::lock_guard<std::mutex> lk(k_mutex);
                    for (int j = 0; j < 4; j++) k.limbs[j] = k_s[j];
                    search_in_progress.store(false, std::memory_order_release);
                }
            }
        }
    };

    std::thread progress_thread([&]() {
        while (search_in_progress.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_print >= std::chrono::seconds(10)) {
                std::lock_guard<std::mutex> lock(io_mutex);
                std::cout << "\rTotal Iterations: " << total_iters.load() << std::endl;
                last_print = now;
            }
        }
    });

    unsigned int threads_count = std::thread::hardware_concurrency();
    if (threads_count == 0) threads_count = 2;
    threads_count = std::min<unsigned int>(threads_count, walkers);

    std::vector<std::thread> threads;
    if (walkers == 0) return k;
    int chunk = walkers / threads_count;

    for (unsigned int t = 0; t < threads_count; t++) {
        int start = t * chunk;
        int end   = (t == threads_count - 1) ? walkers : start + chunk;
        threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) th.join();
    for (auto& w : walkers_state) delete w.buffers;

    search_in_progress.store(false, std::memory_order_release);
    progress_thread.join();

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
