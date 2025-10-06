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
#include <unordered_set>
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

struct uint256_t {
    uint64_t limbs[4];
};

ECPointJacobian G;
ECPointJacobian H;

const uint256_t N = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
const uint256_t GX = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
const uint256_t GY = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };

int windowSize = 4;

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

        //Adjust the table to fit in the processor's L3 cache, avoiding jumping to RAM.
        try {
            size_t size = std::stoul(sizeStr.substr(0, sizeStr.size()-1)) * mult;

            if (L == 3)
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

    #ifdef __CUDACC__
        ECPointJacobian* dg = nullptr;
        cudaError_t err = cudaMalloc(&dg, sizeof(ECPointJacobian));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        cudaMemcpy(dg, &G, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
        pointDoubleJacobian(dg, dg);
        cudaDeviceSynchronize();
        cudaMemcpy(&H, dg, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
        cudaFree(dg);
        cudaFree(jacNorm);
        cudaFree(jacEndo);
    #else
        ECPointJacobian* tmp = new ECPointJacobian();
        *tmp = G;
        pointDoubleJacobian(tmp, tmp);
        H = *tmp;
        delete tmp;
        delete[] jacNorm;
        delete[] jacEndo;
    #endif
}

class XorShift64 {
    uint64_t state;
    public:
    XorShift64(uint64_t seed = 0) {
        if(seed == 0) {
           seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        }
        state = seed;
    }

    uint64_t next() {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 2685821657736338717ull;
    }

    uint64_t next_between(uint64_t min, uint64_t max) {
        if (min == max) return min;
        uint64_t range = max - min;
        if (range == UINT64_MAX) {
            return next();
        } else {
            return min + (next() % (range + 1));
        }
    }

};

class PKG {
    XorShift64 gen;
    uint64_t min_low, max_low;
    uint64_t min_mid_low, max_mid_low;
    uint64_t min_mid_high, max_mid_high;
    uint64_t min_high, max_high;

    public:
    PKG(const uint256_t& min_scalar, const uint256_t& max_scalar) : gen(std::random_device{}()) {
        min_high     = min_scalar.limbs[0];
        max_high     = max_scalar.limbs[0];
        min_mid_high = min_scalar.limbs[1];
        max_mid_high = max_scalar.limbs[1];
        min_mid_low  = min_scalar.limbs[2];
        max_mid_low  = max_scalar.limbs[2];
        min_low      = min_scalar.limbs[3];
        max_low      = max_scalar.limbs[3];
    }

    uint256_t generate() {
        uint64_t high     = gen.next_between(min_high, max_high);
        uint64_t mid_high = gen.next_between(min_mid_high, max_mid_high);
        uint64_t mid_low  = gen.next_between(min_mid_low, max_mid_low);
        uint64_t low      = gen.next_between(min_low, max_low);

        uint256_t result{};
        result.limbs[0] = high;
        result.limbs[1] = mid_high;
        result.limbs[2] = mid_low;
        result.limbs[3] = low;

        return result;
    }
};

uint256_t mask_for_bits(int bits) {
    uint256_t mask{};
    int full_limbs = bits / 64;
    int rem_bits   = bits % 64;
    for(int i = 0; i < 4; i++) mask.limbs[i] = 0;
    for(int i = 3; i > 3 - full_limbs; i--) mask.limbs[i] = 0xFFFFFFFFFFFFFFFFULL;
    if(full_limbs < 4 && rem_bits > 0) mask.limbs[3 - full_limbs] = (1ULL << rem_bits) - 1;
    return mask;
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

uint256_t left_shift_uint256(const uint256_t& a, int shift) {
    uint256_t result{};
    int limb_shift = shift / 64;
    int bit_shift  = shift % 64;
    for(int i = 0; i < 4; i++) result.limbs[i] = 0;
    for(int i = 3; i >= 0; i--) {
    if(i - limb_shift >= 0) result.limbs[i] |= a.limbs[i - limb_shift] << bit_shift;
    if(bit_shift != 0 && i - limb_shift - 1 >= 0) result.limbs[i] |= a.limbs[i - limb_shift - 1] >> (64 - bit_shift);
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

uint256_t f(ECPointJacobian& R, uint256_t k, int key_range, Buffers& buffers) {
    const uint256_t mask = mask_for_bits(key_range);

    uint64_t x_low64 = R.X[3];

    unsigned int op = static_cast<unsigned int>(x_low64 % 3);

    XorShift64 rng(x_low64);
    uint64_t jump1 = rng.next();
    uint64_t jump2 = rng.next();
    uint64_t jump3 = rng.next();
    uint64_t jump4 = rng.next();

    uint256_t jump{};
    jump.limbs[0] = 0;
    jump.limbs[1] = 0;
    jump.limbs[2] = 0;
    jump.limbs[3] = jump1;

    #ifdef __CUDACC__
        cudaMemcpy(buffers.d_R, &R, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
    #endif

    switch(op) {
        case 0:
            #ifdef __CUDACC__
                pointAddJacobian(buffers.d_R, buffers.d_R, buffers.d_G);
            #else
                pointAddJacobian(&R, &R, &G);
            #endif
            k = add_uint256(k, jump);
            break;
        case 1:
            #ifdef __CUDACC__
                pointAddJacobian(buffers.d_R, buffers.d_R, buffers.d_H);
            #else
                pointAddJacobian(&R, &R, &H);
            #endif
            k = add_uint256(k, jump);
            break;
        default:
            #ifdef __CUDACC__
                pointDoubleJacobian(buffers.d_R, buffers.d_R);
            #else
                pointDoubleJacobian(&R, &R);
            #endif
            k = left_shift_uint256(k, 1);
            break;
    }

    for(int i = 0; i < 4; i++) k.limbs[i] &= mask.limbs[i];

    #ifdef __CUDACC__
        cudaMemcpy(&R, buffers.d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
    #endif

    return k;
}

uint64_t hashed_dp(const ECPointJacobian& P) {
    uint64_t h = 0;
    for(int i = 0; i < 4; i++) {
        h ^= P.X[i] + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2);
    }
    return h;
}

bool DP(const ECPointJacobian& P, int LSB) {
    return (P.X[0] & ((1ULL << LSB) - 1)) == 0;
}

uint256_t prho(std::string target_pubkey_hex, int key_range, int hares, bool test_mode) {
    auto uint_256_to_hex = [](const uint256_t& value) -> std::string {
        std::ostringstream oss;
        for(int i = 0; i < 4; i++) {
            oss << std::setw(16) << std::setfill('0') << std::hex << value.limbs[i];
        }
        return oss.str();
    };

    auto bytes_to_hex = [](const unsigned char* bytes, size_t length) -> std::string {
        std::string hex_str;
        hex_str.reserve(length * 2);
        const char* hex_chars = "0123456789abcdef";
        for (size_t i = 0; i < length; ++i) {
            hex_str.push_back(hex_chars[bytes[i] >> 4]);
            hex_str.push_back(hex_chars[bytes[i] & 0x0F]);
        }
        return hex_str;
    };

    auto hex_to_bytes = [](const std::string& hex) -> std::vector<unsigned char> {
        std::vector<unsigned char> bytes;
        for (size_t i = 0; i < hex.length(); i += 2) {
            unsigned char byte = (unsigned char) std::stoi(hex.substr(i, 2), nullptr, 16);
            bytes.push_back(byte);
        }
        return bytes;
    };

    auto start_time = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::tm start_tm = *std::localtime(&start_time_t);

    std::cout << "Started at: " << std::put_time(&start_tm, "%H:%M:%S") << std::endl;
    if(test_mode) { std::cout << "Test Mode: True" << std::endl; }
    else          { std::cout << "Test Mode: False" << std::endl; }

    uint256_t min_scalar{};
    uint256_t max_scalar{};

    std::unordered_set<uint64_t> dp_table;

    int num_limbs = (key_range + 63) / 64;
    int limb_index = 3 - ((key_range - 1) / 64);
    int bit_in_limb = (key_range - 1) % 64;
    min_scalar.limbs[limb_index] = 1ULL << bit_in_limb;
    int full_limbs = key_range / 64;
    int rem_bits = key_range % 64;
    for (int i = 4 - full_limbs; i < 4; i++) max_scalar.limbs[i] = 0xFFFFFFFFFFFFFFFFULL;
    if (rem_bits != 0) max_scalar.limbs[4 - full_limbs - 1] = (1ULL << rem_bits) - 1;

    std::cout << "key_range: " << key_range << std::endl;
    std::cout << "min_range: " << uint_256_to_hex(min_scalar) << std::endl;
    std::cout << "max_range: " << uint_256_to_hex(max_scalar) << std::endl;

    std::atomic<bool> search_in_progress(true);
    std::atomic<bool> should_sync(false);
    std::mutex pgrs;
    std::string P_key;

    uint256_t p_key{};
    uint256_t found_key{};

    unsigned long long total_keys = 0;
    const unsigned long long SYNC_INTERVAL = 5000000;

    auto target_pubkey = hex_to_bytes(target_pubkey_hex);

    struct HareState {
        uint256_t k1, k2;
        ECPointJacobian R;
        int speed;
        Buffers* buffers;
    };

    PKG pkg(min_scalar, max_scalar);
    uint256_t current_key = pkg.generate();

    std::vector<HareState> hare_states(hares);

    for (int i = 0; i < hares; ++i) {
        hare_states[i].buffers = new Buffers();

        #ifdef __CUDACC__
            ECPointJacobian* d_point = nullptr;
            cudaError_t err = cudaMalloc(&d_point, sizeof(ECPointJacobian));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                exit(1);
            }

            pointInitJacobian(d_point);
            cudaMemcpy(&hare_states[i].R, d_point, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
            cudaFree(d_point);
        #else
            ECPointJacobian tmp;
            pointInitJacobian(&tmp);
            hare_states[i].R = tmp;
        #endif

        hare_states[i].R = G;
        hare_states[i].R.infinity = 0;
        hare_states[i].speed = 0;
        hare_states[i].k1 = pkg.generate();
        hare_states[i].k2 = pkg.generate();
    }

    std::thread log_thread([&]() {
        try {
            while(search_in_progress.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                std::lock_guard<std::mutex> lock(pgrs);

                if(search_in_progress.load()) {
                    std::cout << "\rCurrent private key: " << uint_256_to_hex(p_key) << std::endl;
                    std::cout << "\rLast tested public key: " << P_key << std::endl;
                    std::cout << "\rTotal keys tested: " << total_keys << std::endl;
                }
                else
                {
                    std::cout << "Initializing, wait..." << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in log_thread: " << e.what() << std::endl;
        }
    });

    try {
        while (search_in_progress.load()) {

            for (int i = 0; i < hares; ++i) {

                HareState& hare = hare_states[i];
                hare.speed = (i == 0) ? 1 : (i + 1);
                total_keys += 2 * hares;

                current_key = hare.k1;
                p_key = current_key;

                if (test_mode) {
                    hare.k1 = add_uint256(hare.k1, uint256_from_uint32(hare.speed));
                    hare.k2 = add_uint256(hare.k2, uint256_from_uint32(hare.speed));
                    for (int j = 2; j < 4; j++) {
                        hare.k1.limbs[j] = 0;
                        hare.k2.limbs[j] = 0;
                    }
                } else {
                    hare.k1 = f(hare.R, hare.k1, key_range, *hare.buffers);
                    hare.k2 = f(hare.R, hare.k2, key_range, *hare.buffers);
                }

                if (compare_uint256(hare.k1, min_scalar) < 0)
                    hare.k1 = add_uint256(hare.k1, min_scalar);
                if (compare_uint256(hare.k2, min_scalar) < 0)
                    hare.k2 = add_uint256(hare.k2, min_scalar);

                should_sync.store(total_keys % SYNC_INTERVAL == 0);

                ECPointJacobian pub1_jac{}, pub2_jac{};

                uint64_t k1_array[4], k2_array[4];
                uint256_to_uint64_array(k1_array, hare.k1);
                uint256_to_uint64_array(k2_array, hare.k2);

                #ifdef __CUDACC__
                    cudaMemcpy(hare.buffers->d_k, k1_array + (4 - num_limbs), sizeof(uint64_t) * num_limbs, cudaMemcpyHostToDevice);
                    scalarMultJacobian(hare.buffers->d_R, hare.buffers->d_k, key_range, windowSize);
                    if (should_sync.load()) cudaDeviceSynchronize();
                    cudaMemcpy(&pub1_jac, hare.buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
                    cudaMemcpy(hare.buffers->d_k, k2_array + (4 - num_limbs), sizeof(uint64_t) * num_limbs, cudaMemcpyHostToDevice);
                    scalarMultJacobian(hare.buffers->d_R, hare.buffers->d_k, key_range, windowSize);
                    if (should_sync.load()) cudaDeviceSynchronize();
                    cudaMemcpy(&pub2_jac, hare.buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
                #else
                    scalarMultJacobian(&pub1_jac, k1_array, key_range, windowSize);
                    scalarMultJacobian(&pub2_jac, k2_array, key_range, windowSize);
                #endif

                bool xfilled = true;
                for (int j = 0; j < 4; j++) {
                    if (pub1_jac.X[j] != 0 || pub2_jac.X[j] != 0) {
                        xfilled = false;
                        break;
                    }
                }

                if (!xfilled) {
                    ECPoint pub1{}, pub2{};
                    for (int j = 0; j < 4; j++) {
                        pub1.x[j] = pub1_jac.X[j];
                        pub1.y[j] = pub1_jac.Y[j];
                        pub2.x[j] = pub2_jac.X[j];
                        pub2.y[j] = pub2_jac.Y[j];
                    }
                    pub1.infinity = pub1_jac.infinity;
                    pub2.infinity = pub2_jac.infinity;

                    int LSB = 5;
                    if (!DP(pub1_jac, LSB)) continue;
                    if (!DP(pub2_jac, LSB)) continue;
                    if (dp_table.insert(hashed_dp(pub1_jac)).second) continue;
                    if (dp_table.insert(hashed_dp(pub2_jac)).second) continue;

                    std::cout << "Collision detected at DP!" << std::endl;

                    unsigned char compressed1[33], compressed2[33];

                    #ifdef __CUDACC__
                        unsigned char* d_compressed = nullptr;
                        ECPoint* d_pub_comp = nullptr;
                        cudaMalloc(&d_compressed, 33);
                        cudaMalloc(&d_pub_comp, sizeof(ECPoint));
                        cudaMemcpy(d_pub_comp, &pub1, sizeof(ECPoint), cudaMemcpyHostToDevice);
                        getCompressedPublicKey(d_compressed, d_pub_comp);
                        if (should_sync.load()) cudaDeviceSynchronize();
                        cudaMemcpy(compressed1, d_compressed, 33, cudaMemcpyDeviceToHost);
                        cudaMemcpy(d_pub_comp, &pub2, sizeof(ECPoint), cudaMemcpyHostToDevice);
                        getCompressedPublicKey(d_compressed, d_pub_comp);
                        if (should_sync.load()) cudaDeviceSynchronize();
                        cudaMemcpy(compressed2, d_compressed, 33, cudaMemcpyDeviceToHost);
                        cudaFree(d_compressed);
                        cudaFree(d_pub_comp);
                    #else
                        getCompressedPublicKey(compressed1, &pub1);
                        getCompressedPublicKey(compressed2, &pub2);
                    #endif

                    std::string current_pubkey_hex_R = bytes_to_hex((unsigned char*)compressed1, 33);
                    P_key = current_pubkey_hex_R;

                    if (!test_mode) {
                        bool x_equal = true;
                        for (int j = 0; j < 4; j++) {
                            if (pub1.x[j] != pub2.x[j]) {
                                x_equal = false;
                                break;
                            }
                        }

                        if (x_equal && compare_uint256(hare.k1, hare.k2) != 0) {

                            uint256_t tortoise_key = hare.k2;
                            uint256_t hare_key = hare.k1;
                            uint256_t d = (compare_uint256(hare_key, tortoise_key) >= 0) ? sub_uint256(hare_key, tortoise_key) : sub_uint256(N, sub_uint256(tortoise_key, hare_key));

                            uint64_t d_array[4];
                            uint256_to_uint64_array(d_array, d);

                            ECPointJacobian test_point_jac{};
                            #ifdef __CUDACC__
                                cudaMemcpy(hare.buffers->d_k, d_array + (4 - num_limbs), sizeof(uint64_t) * num_limbs, cudaMemcpyHostToDevice);
                                scalarMultJacobian(hare.buffers->d_R, hare.buffers->d_k, key_range, windowSize);
                                cudaDeviceSynchronize();
                                cudaMemcpy(&test_point_jac, hare.buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
                            #else
                                scalarMultJacobian(&test_point_jac, d_array + (4 - num_limbs), key_range, windowSize);
                            #endif

                            if (test_point_jac.infinity == 1) {

                                found_key = add_uint256(tortoise_key, d);
                                if (compare_uint256(found_key, N) >= 0) {
                                    found_key = sub_uint256(found_key, N);
                                }

                                uint64_t found_key_array[4];
                                uint256_to_uint64_array(found_key_array, found_key);

                                ECPointJacobian verify_point_jac{};
                                #ifdef __CUDACC__
                                    cudaMemcpy(hare.buffers->d_k, found_key_array + (4 - num_limbs), sizeof(uint64_t) * num_limbs, cudaMemcpyHostToDevice);
                                    scalarMultJacobian(hare.buffers->d_R, hare.buffers->d_k, key_range, windowSize);
                                    cudaDeviceSynchronize();
                                    cudaMemcpy(&verify_point_jac, hare.buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
                                #else
                                    scalarMultJacobian(&verify_point_jac, found_key_array + (4 - num_limbs), key_range, windowSize);
                                #endif

                                ECPoint verify_point{};
                                for (int j = 0; j < 4; j++) {
                                    verify_point.x[j] = verify_point_jac.X[j];
                                    verify_point.y[j] = verify_point_jac.Y[j];
                                }

                                verify_point.infinity = verify_point_jac.infinity;

                                unsigned char compressed_verify[33];
                                #ifdef __CUDACC__
                                    unsigned char* d_compressed_verify = nullptr;
                                    ECPoint* d_verify_comp = nullptr;
                                    cudaMalloc(&d_compressed_verify, 33);
                                    cudaMalloc(&d_verify_comp, sizeof(ECPoint));
                                    cudaMemcpy(d_verify_comp, &verify_point, sizeof(ECPoint), cudaMemcpyHostToDevice);
                                    getCompressedPublicKey(d_compressed_verify, d_verify_comp);
                                    cudaDeviceSynchronize();
                                    cudaMemcpy(compressed_verify, d_compressed_verify, 33, cudaMemcpyDeviceToHost);
                                    cudaFree(d_compressed_verify);
                                    cudaFree(d_verify_comp);
                                #else
                                    getCompressedPublicKey(compressed_verify, &verify_point);
                                #endif

                                if (memcmp(compressed_verify, target_pubkey.data(), 33) == 0) {
                                    std::cout << "\033[33mDP detected for hare " << i << " at k1: " << uint_256_to_hex(hare.k1) << "\033[0m" << std::endl;
                                    std::cout << "\033[33mDP detected for hare " << i << " at k2: " << uint_256_to_hex(hare.k2) << "\033[0m" << std::endl;
                                    std::cout << "A multiplicação satisfaz a equação (d * G ≡ 0)" << std::endl;
                                    std::cout << "Private Key Found: " << uint_256_to_hex(found_key) << std::endl;

                                    search_in_progress.store(false);
                                }
                            }
                        }
                    }

                    if (memcmp(compressed1, target_pubkey.data(), 33) == 0 ||
                        memcmp(compressed2, target_pubkey.data(), 33) == 0) {

                        found_key = (memcmp(compressed1, target_pubkey.data(), 33) == 0) ? hare.k1 : hare.k2;
                        std::cout << "\033[32mPrivate key found!\033[0m" << std::endl;
                        std::cout << "Private Key: " << uint_256_to_hex(found_key) << std::endl;

                        search_in_progress.store(false);
                    }
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    for (int i = 0; i < hares; ++i) {
        delete hare_states[i].buffers;
    }

    log_thread.join();

    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Total duration: " << std::setw(2) << std::setfill('0') << duration.count() / 3600 << ":"
    << std::setw(2) << std::setfill('0') << (duration.count() % 3600) / 60 << ":"
    << std::setw(2) << std::setfill('0') << duration.count() % 60 << std::endl;

    return found_key;
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

    if(test_mode && key_range > 20){
        std::cout << "Use a range of 20 bits or less for test mode!" << std::endl;
        return 1;
    }

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
