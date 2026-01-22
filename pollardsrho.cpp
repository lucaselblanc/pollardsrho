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
const uint64_t P[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
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

    preCompG = new ECPointJacobian[1ULL << windowSize];
    preCompGphi = new ECPointJacobian[1ULL << windowSize];
    jacNorm = new ECPointJacobian[windowSize];
    jacEndo = new ECPointJacobian[windowSize];

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
}

uint256_t rng_mersenne_twister(const uint256_t& min_scalar, const uint256_t& max_scalar, int key_range, std::mt19937_64& rng) {
    uint256_t r{};
    uint256_t range{};

    bool carry = false;
    for (int i = 3; i >= 0; i--) {
        uint64_t prev = max_scalar.limbs[i];
        range.limbs[i] = max_scalar.limbs[i] - min_scalar.limbs[i] - (carry ? 1 : 0);
        carry = (prev < min_scalar.limbs[i] + (carry ? 1 : 0));
    }

    range.limbs[3] += 1;

    auto greater_equal = [](const uint256_t& a, const uint256_t& b) -> bool {
        for (int i = 0; i < 4; i++) {
            if (a.limbs[i] > b.limbs[i]) return true;
            if (a.limbs[i] < b.limbs[i]) return false;
        }
        return true;
    };

    int limb_index = 3 - key_range / 64;
    int bit_in_limb = key_range % 64;

    do {
        for (int i = 0; i < 4; i++) r.limbs[i] = rng();
        if (bit_in_limb > 0) {
            r.limbs[limb_index] &= (1ULL << bit_in_limb) - 1;
        } else {
            r.limbs[limb_index] = 0;
        }

        for (int i = 0; i < limb_index; i++) r.limbs[i] = 0;

    } while (greater_equal(r, range));

    carry = false;
    for (int i = 3; i >= 0; i--) {
        uint64_t prev = r.limbs[i];
        r.limbs[i] += min_scalar.limbs[i] + (carry ? 1 : 0);
        carry = (r.limbs[i] < prev);
    }

    return r;
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
};

uint64_t hashed_dp(const ECPointJacobian& P) {
    uint64_t h = P.X[3];
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return h;
}

bool DP(const ECPointJacobian& P, int DP_BITS) {
    return (P.X[0] & ((1ULL << DP_BITS) - 1)) == 0;
}

void f(ECPointJacobian& R, uint256_t& a, uint256_t& b, Buffers& buffers) {
    //Teske 2^k Method
    uint64_t h = hashed_dp(R);

    constexpr uint32_t S = 32;

    uint32_t partition = h & (S - 1);
    uint32_t pc_idx = (h >> 6) & ((1U << windowSize) - 1);
    uint32_t add_a = 1 + ((h >> 16) & 0x7F);
    uint32_t add_b = 1 + ((h >> 24) & 0x7F);

    if (partition < S / 3) {
        pointAddJacobian(&R, &R, &preCompG[pc_idx]);

        a = add_uint256(a, uint256_from_uint32(add_a));
        if (compare_uint256(a, N) >= 0) a = sub_uint256(a, N);
    } 
    else if (partition < 2 * S / 3) {
        pointAddJacobian(&R, &R, &H);

        b = add_uint256(b, uint256_from_uint32(add_b));
        if (compare_uint256(b, N) >= 0) b = sub_uint256(b, N);
    } 
    else {
        pointDoubleJacobian(&R, &R);

        a = add_uint256(a, a);
        if (compare_uint256(a, N) >= 0) a = sub_uint256(a, N);

        b = add_uint256(b, b);
        if (compare_uint256(b, N) >= 0) b = sub_uint256(b, N);
    }
}

uint256_t prho(std::string target_pubkey_hex, int key_range, int walkers, bool test_mode) {

    std::atomic<bool> search_in_progress(true);
    std::atomic<unsigned long long> total_iters{0};
    std::mutex k_mutex;
    std::mutex dp_mutex;

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
    std::cout << "Key Range: " << key_range << std::endl;
    std::cout << "Min Range: " << uint_256_to_hex(min_scalar) << std::endl;
    std::cout << "Max Range: " << uint_256_to_hex(max_scalar) << std::endl;

    struct WalkState {
        ECPointJacobian R;
        uint256_t a, b;
        Buffers* buffers;
        uint32_t walk_id;
        bool negate;
        std::mt19937_64 rng;
    };

    struct DPEntry {
        ECPointJacobian R;
        uint256_t a, b;
        uint32_t walk_id;
        bool negate;
    };

    std::unordered_map<uint64_t, std::vector<DPEntry>> dp_table;
    std::vector<WalkState> walkers_state(walkers);

    for (int i = 0; i < walkers; i++) {
        std::seed_seq seed{
            std::random_device{}(),
            std::random_device{}(),
            static_cast<uint32_t>(i),
            static_cast<uint32_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count())
        };

        walkers_state[i].rng.seed(seed);
        walkers_state[i].buffers = new Buffers();
        walkers_state[i].a = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
        walkers_state[i].b = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
        walkers_state[i].walk_id = i;

        uint64_t a_arr[4], b_arr[4];
        uint256_to_uint64_array(a_arr, walkers_state[i].a);
        uint256_to_uint64_array(b_arr, walkers_state[i].b);

        ECPointJacobian Ra{}, Rb{};

        scalarMultJacobian(&Ra, a_arr, windowSize);
        ECPointJacobian TEMP_G = G;
        G = H;
        scalarMultJacobian(&Rb, b_arr, windowSize);
        G = TEMP_G;
        pointAddJacobian(&walkers_state[i].R, &Ra, &Rb);
    }

    auto last_print = std::chrono::steady_clock::now();

    /* This doesn't reduce the search to half the key range, it just centers the walkers before starting the random walk */
    if (test_mode) {
        uint256_t min_jump{}, max_jump{};

        // Initial jump of 2^(key_range/2)
        int bit_index = key_range / 2;
        int limb_index = 3 - (bit_index / 64);
        int bit_in_limb = bit_index % 64;
        min_jump.limbs[limb_index] = 1ULL << bit_in_limb;
        max_jump = max_scalar;

        for (int i = 0; i < walkers; i++) {
            WalkState* w = &walkers_state[i];

            w->a = rng_mersenne_twister(min_jump, max_jump, key_range, w->rng);
            w->b = rng_mersenne_twister(min_jump, max_jump, key_range, w->rng);

            uint64_t a_arr[4], b_arr[4];
            uint256_to_uint64_array(a_arr, w->a);
            uint256_to_uint64_array(b_arr, w->b);

            ECPointJacobian Ra{}, Rb{};
            scalarMultJacobian(&Ra, a_arr, windowSize);

            ECPointJacobian TEMP_G = G;
            G = H;
            scalarMultJacobian(&Rb, b_arr, windowSize);
            G = TEMP_G;

            pointAddJacobian(&w->R, &Ra, &Rb);
        }
    }

    auto pointsEqual = [](const ECPointJacobian& A, const ECPointJacobian& B) -> bool {
   	    if (A.infinity || B.infinity) return A.infinity && B.infinity;

    	uint64_t Z1Z1[4], Z2Z2[4];
    	uint64_t U1[4], U2[4];
    	uint64_t Z1Z1Z1[4], Z2Z2Z2[4];
    	uint64_t S1[4], S2[4];

    	modMulMontP(Z1Z1, A.Z, A.Z);
    	modMulMontP(Z2Z2, B.Z, B.Z);
    	modMulMontP(U1, A.X, Z2Z2);
    	modMulMontP(U2, B.X, Z1Z1);

    	if (memcmp(U1, U2, sizeof(U1)) != 0) return false;

    	modMulMontP(Z1Z1Z1, Z1Z1, A.Z);
    	modMulMontP(Z2Z2Z2, Z2Z2, B.Z);
    	modMulMontP(S1, A.Y, Z2Z2Z2);
    	modMulMontP(S2, B.Y, Z1Z1Z1);

    	return memcmp(S1, S2, sizeof(S1)) == 0;
    };

    auto worker = [&](int start, int end) {
        while (search_in_progress.load(std::memory_order_acquire)) {
            for (int i = start; i < end && search_in_progress.load(std::memory_order_acquire); i++) {
                total_iters.fetch_add(1, std::memory_order_relaxed);

                WalkState* w = &walkers_state[i];

                f(w->R, w->a, w->b, *w->buffers);

                if (compare_uint256(w->a, N) >= 0) w->a = sub_uint256(w->a, N);
                if (compare_uint256(w->b, N) >= 0) w->b = sub_uint256(w->b, N);
                if (!DP(w->R, 8)) continue;

                uint256_t diff_a {};
                uint256_t diff_b {};
                uint256_t inv_diff_b{};

                w->negate = false;

                if (!w->R.infinity && (w->R.Y[0] & 1)) {
                    modSubP(w->R.Y, P, w->R.Y);
                    if ((w->a.limbs[0] | w->a.limbs[1] | w->a.limbs[2] | w->a.limbs[3]) != 0) w->a = sub_uint256(N, w->a);
                    if ((w->b.limbs[0] | w->b.limbs[1] | w->b.limbs[2] | w->b.limbs[3]) != 0) w->b = sub_uint256(N, w->b);
                    w->negate = true;
                }

                uint64_t _ = hashed_dp(w->R);

                std::lock_guard<std::mutex> lock(dp_mutex);
                auto& dps = dp_table[_];

                for (const auto& in : dps) {

                    if (in.walk_id == w->walk_id) continue;
                    if (!pointsEqual(w->R, in.R)) continue;

                    if ((w->negate ^ in.negate) == 0)
                    {
                        diff_a = (compare_uint256(w->a, in.a) >= 0) ? sub_uint256(w->a, in.a) : sub_uint256(N, sub_uint256(in.a, w->a));
                        diff_b = (compare_uint256(in.b, w->b) >= 0) ? sub_uint256(in.b, w->b) : sub_uint256(N, sub_uint256(w->b, in.b));
                    }
                    else
                    {
                        diff_a = (compare_uint256(add_uint256(w->a, in.a), N) >= 0) ? sub_uint256(add_uint256(w->a, in.a), N) : add_uint256(w->a, in.a); 
                        diff_b = (compare_uint256(add_uint256(w->b, in.b), N) >= 0) ? sub_uint256(N, sub_uint256(add_uint256(w->b, in.b), N)) : sub_uint256(N, add_uint256(w->b, in.b));
                    }

                    if ((diff_b.limbs[0] | diff_b.limbs[1] | diff_b.limbs[2] | diff_b.limbs[3]) == 0) continue;

                    inv_diff_b = almostinverse(diff_b, N);

                    uint64_t a_s[4], b_s[4], k_s[4];
                    uint256_to_uint64_array(a_s, diff_a);
                    uint256_to_uint64_array(b_s, inv_diff_b);
                    scalarMul(k_s, a_s, b_s);

                    {
                        std::lock_guard<std::mutex> lk(k_mutex);
                        for (int j = 0; j < 4; j++) k.limbs[j] = k_s[j];
                    }

                    std::cout << "\033[32mCollision found!\033[0m" << std::endl;
                    std::cout << "Difference of the coefficient A: " << uint_256_to_hex(diff_a) << std::endl;
                    std::cout << "Difference of the coefficient B: " << uint_256_to_hex(diff_b) << std::endl;
                    std::cout << "Scalar Coefficient A (Non-Key): " << uint_256_to_hex(w->a) << std::endl;
                    std::cout << "Scalar Coefficient B (Non-Key): " << uint_256_to_hex(w->b) << std::endl;

                    search_in_progress.store(false, std::memory_order_release);
                    return;
                }

                DPEntry entry;
                entry.R = w->R;
                entry.a = w->a;
                entry.b = w->b;
                entry.walk_id = w->walk_id;
                entry.negate = w->negate;

                dps.push_back(std::move(entry));
            }
        }
    };

    std::thread progress_thread([&]() {

        const long double M = ldexpl(1.0L, key_range);

        while (search_in_progress.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_print >= std::chrono::seconds(10)) {
                long double k = (long double)total_iters.load(std::memory_order_relaxed);
                long double x = (k * k) / (2.0L * M);
                long double prob = 1.0L - expl(-x);
                prob *= 100.0L;

                std::cout << "\r\033[K"
                << "Total Iterations: " << total_iters.load() << "\n"
                << "Collision Probability: "
                << std::fixed << std::setprecision(8)
                << (prob) << "...%\n"
                << std::flush;

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

    uint256_t found_key = prho(pub_key_hex, key_range, 2, test_mode);

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