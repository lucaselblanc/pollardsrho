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

/*
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>
#include <ctime>
#include <mutex>

#include <sys/sysinfo.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "secp256k1.h"

struct uint256_t {
    uint32_t limbs[8];
};

ECPoint G;
ECPoint H;

const uint256_t N = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

const uint256_t GX = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

const uint256_t GY = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

void uint256_to_uint32_array(unsigned int* out, const uint256_t& value) {
    for (int i = 0; i < 8; i++) {
        out[i] = value.limbs[i];
    }
}

void init_secp256k1() {

    uint256_to_uint32_array(G.x, GX);
    uint256_to_uint32_array(G.y, GY);
    G.infinity = 0;
    
    ECPoint* d_G = nullptr;
    cudaMalloc(&d_G, sizeof(ECPoint));
    cudaMemcpy(d_G, &G, sizeof(ECPoint), cudaMemcpyHostToDevice);
    
    point_double<<<1,1>>>(d_G, d_G);
    cudaDeviceSynchronize();
    cudaMemcpy(&H, d_G, sizeof(ECPoint), cudaMemcpyDeviceToHost);
    
    cudaFree(d_G);
}

class PKG {
    std::mt19937_64 gen;
    uint64_t min_low, max_low;
    uint64_t min_mid_low, max_mid_low;
    uint64_t min_mid_high, max_mid_high;
    uint64_t min_high, max_high;

public:
    PKG(const uint256_t& min_scalar, const uint256_t& max_scalar) : gen(std::random_device{}()) {
        min_low      = (static_cast<uint64_t>(min_scalar.limbs[6]) << 32) | min_scalar.limbs[7];
        max_low      = (static_cast<uint64_t>(max_scalar.limbs[6]) << 32) | max_scalar.limbs[7];

        min_mid_low  = (static_cast<uint64_t>(min_scalar.limbs[4]) << 32) | min_scalar.limbs[5];
        max_mid_low  = (static_cast<uint64_t>(max_scalar.limbs[4]) << 32) | max_scalar.limbs[5];

        min_mid_high = (static_cast<uint64_t>(min_scalar.limbs[2]) << 32) | min_scalar.limbs[3];
        max_mid_high = (static_cast<uint64_t>(max_scalar.limbs[2]) << 32) | max_scalar.limbs[3];

        min_high     = (static_cast<uint64_t>(min_scalar.limbs[0]) << 32) | min_scalar.limbs[1];
        max_high     = (static_cast<uint64_t>(max_scalar.limbs[0]) << 32) | max_scalar.limbs[1];
    }

    uint256_t generate() {
        uint64_t low      = std::uniform_int_distribution<uint64_t>(min_low, max_low)(gen);
        uint64_t mid_low  = std::uniform_int_distribution<uint64_t>(min_mid_low, max_mid_low)(gen);
        uint64_t mid_high = std::uniform_int_distribution<uint64_t>(min_mid_high, max_mid_high)(gen);
        uint64_t high     = std::uniform_int_distribution<uint64_t>(min_high, max_high)(gen);

        uint256_t result;
        result.limbs[7] = static_cast<uint32_t>(low & 0xFFFFFFFF);
        result.limbs[6] = static_cast<uint32_t>(low >> 32);
        result.limbs[5] = static_cast<uint32_t>(mid_low & 0xFFFFFFFF);
        result.limbs[4] = static_cast<uint32_t>(mid_low >> 32);
        result.limbs[3] = static_cast<uint32_t>(mid_high & 0xFFFFFFFF);
        result.limbs[2] = static_cast<uint32_t>(mid_high >> 32);
        result.limbs[1] = static_cast<uint32_t>(high & 0xFFFFFFFF);
        result.limbs[0] = static_cast<uint32_t>(high >> 32);

        return result;
    }
};

size_t get_memory_bytes() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram * info.mem_unit;
    }
    return 0;
}

double get_memory_gb() {
    return static_cast<double>(get_memory_bytes()) / (1024 * 1024 * 1024);
}

double TOTAL_RAM = (get_memory_gb() / 2); // div 2 >> 50% of ram
size_t NUM_JUMPS;
size_t jump_index = 0;
bool precomputed_done = false;

std::vector<uint256_t> precomputed_jumps;

uint256_t mask_for_bits(int bits) {
    uint256_t mask{};
    int full_limbs = bits / 32;
    int rem_bits = bits % 32;
    for(int i = 0; i < 8; i++) mask.limbs[i] = 0;
    for(int i = 7; i > 7 - full_limbs; i--) mask.limbs[i] = 0xFFFFFFFF;
    if(full_limbs < 8 && rem_bits > 0) mask.limbs[7 - full_limbs] = (1U << rem_bits) - 1;
    return mask;
}

void precompute_jumps(int key_range) {
    std::mt19937_64 rng(std::random_device{}());
    precomputed_jumps.resize(NUM_JUMPS);

    uint256_t max_value = mask_for_bits(key_range);

    for (size_t i = 0; i < NUM_JUMPS; ++i) {
        uint64_t r0 = rng();
        uint64_t r1 = rng();
        uint64_t r2 = rng();
        uint64_t r3 = rng();

        uint256_t r{};
        r.limbs[7] = static_cast<uint32_t>(r0 & 0xFFFFFFFF);
        r.limbs[6] = static_cast<uint32_t>(r0 >> 32);
        r.limbs[5] = static_cast<uint32_t>(r1 & 0xFFFFFFFF);
        r.limbs[4] = static_cast<uint32_t>(r1 >> 32);
        r.limbs[3] = static_cast<uint32_t>(r2 & 0xFFFFFFFF);
        r.limbs[2] = static_cast<uint32_t>(r2 >> 32);
        r.limbs[1] = static_cast<uint32_t>(r3 & 0xFFFFFFFF);
        r.limbs[0] = static_cast<uint32_t>(r3 >> 32);

        for(int j = 0; j < 8; j++) {
            r.limbs[j] &= max_value.limbs[j];
        }

        if (r.limbs[0]==0 && r.limbs[1]==0 && r.limbs[2]==0 && r.limbs[3]==0 &&
            r.limbs[4]==0 && r.limbs[5]==0 && r.limbs[6]==0 && r.limbs[7]==0) {
            r.limbs[7] = 1;
        }

        precomputed_jumps[i] = r;

        if (i % (NUM_JUMPS / 100) == 0) {
            int progress = static_cast<int>((i * 101) / NUM_JUMPS);
            std::cout << "\rLoading Jumps: " << progress << "% of total jumps: " 
                      << NUM_JUMPS << " Using: " << TOTAL_RAM << " GB ram - " << std::flush;
        }
    }

    precomputed_done = true;
}

uint256_t sub_uint256(const uint256_t& a, const uint256_t& b) {
    uint256_t result{};
    uint64_t borrow = 0;

    for (int i = 0; i < 8; ++i) {
        uint64_t ai = a.limbs[i];
        uint64_t bi = b.limbs[i];
        uint64_t temp = ai - bi - borrow;

        borrow = ((ai < bi + borrow) ? 1 : 0);
        result.limbs[i] = temp;
    }

    return result;
}

int compare_uint256(const uint256_t& a, const uint256_t& b) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return 1;
        if (a.limbs[i] < b.limbs[i]) return -1;
    }
    return 0;
}

uint256_t uint256_from_uint32(uint32_t value) {
    uint256_t r{};
    r.limbs[0] = value;
    for(int i=1;i<8;i++) r.limbs[i] = 0;
    return r;
}

uint256_t add_uint256(const uint256_t& a, const uint256_t& b) {
    uint256_t result{};
    uint64_t carry = 0;
    for(int i = 7; i >= 0; i--) {
        uint64_t sum = static_cast<uint64_t>(a.limbs[i]) + b.limbs[i] + carry;
        result.limbs[i] = static_cast<uint32_t>(sum & 0xFFFFFFFF);
        carry = sum >> 32;
    }
    return result;
}

uint256_t left_shift_uint256(const uint256_t& a, int shift) {
    uint256_t result{};
    if(shift == 0) return a;
    int limb_shift = shift / 32;
    int bit_shift = shift % 32;
    for(int i = 0; i < 8; i++) result.limbs[i] = 0;

    for(int i = 0; i < 8; i++) {
        if(i + limb_shift < 8) {
            result.limbs[i] |= a.limbs[i + limb_shift] << bit_shift;
        }
        if(bit_shift != 0 && i + limb_shift + 1 < 8) {
            result.limbs[i] |= a.limbs[i + limb_shift + 1] >> (32 - bit_shift);
        }
    }
    return result;
}

uint256_t f(ECPoint& R, uint256_t k, int key_range) {
    const uint256_t mask = mask_for_bits(key_range);

    uint256_t x_coord{};
    for(int i = 0; i < 8; i++) x_coord.limbs[i] = R.x[i];

    uint64_t x_low64 = (static_cast<uint64_t>(x_coord.limbs[6]) << 32) | x_coord.limbs[7];
    unsigned int op = static_cast<unsigned int>(x_low64 % 3);
    size_t idx = static_cast<size_t>(x_low64 % NUM_JUMPS);

    ECPoint* d_R = nullptr;
    ECPoint* d_G = nullptr;
    ECPoint* d_H = nullptr;

    cudaMalloc(&d_R, sizeof(ECPoint));
    cudaMalloc(&d_G, sizeof(ECPoint));
    cudaMalloc(&d_H, sizeof(ECPoint));
    cudaMemcpy(d_R, &R, sizeof(ECPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_G, &G, sizeof(ECPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, &H, sizeof(ECPoint), cudaMemcpyHostToDevice);

    switch(op) {
        case 0:
            point_add<<<1,1>>>(d_R, d_R, d_G);
            k = add_uint256(k, precomputed_jumps[idx]);
            break;
        case 1:
            point_add<<<1,1>>>(d_R, d_R, d_H);
            k = add_uint256(k, precomputed_jumps[idx]);
            break;
        default:
            point_double<<<1,1>>>(d_R, d_R);
            k = left_shift_uint256(k, 1);
            break;
    }

    for(int i = 0; i < 8; i++) k.limbs[i] &= mask.limbs[i];

    cudaDeviceSynchronize();
    cudaMemcpy(&R, d_R, sizeof(ECPoint), cudaMemcpyDeviceToHost);
    cudaFree(d_R);
    cudaFree(d_G);
    cudaFree(d_H);

    return k;
}

uint256_t prho(std::string target_pubkey_hex, int key_range, int hares, bool test_mode) {

    auto uint_256_to_hex = [](const uint256_t& value) -> std::string {
        std::ostringstream oss;
        for(int i = 0; i < 8; i++) {
            oss << std::setw(8) << std::setfill('0') << std::hex << value.limbs[i];
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

    uint256_t min_scalar = left_shift_uint256(uint256_from_uint32(1), key_range - 1);
    uint256_t max_scalar = sub_uint256(left_shift_uint256(uint256_from_uint32(1), key_range), uint256_from_uint32(1));

    std::cout << "key_range: " << key_range << std::endl;
    std::cout << "min_range: " << uint_256_to_hex(min_scalar) << std::endl;
    std::cout << "max_range: " << uint_256_to_hex(max_scalar) << std::endl;
    std::atomic<unsigned int> keys_ps{0};
    std::atomic<bool> search_in_progress(true);
    std::mutex pgrs;
    std::string P_key;

    uint256_t p_key{};
    uint256_t found_key{};

    auto target_pubkey = hex_to_bytes(target_pubkey_hex);

    struct HareState {
        uint256_t k1, k2;
        ECPoint R;
        int speed;
    };

    PKG pkg(min_scalar, max_scalar);
    uint256_t current_key = pkg.generate();

    std::vector<HareState> hare_states(hares);

    for (int i = 0; i < hares; ++i) {
        ECPoint* d_point = nullptr;
        cudaMalloc(&d_point, sizeof(ECPoint));
        point_init<<<1,1>>>(d_point);
        cudaDeviceSynchronize();
        cudaMemcpy(&hare_states[i].R, d_point, sizeof(ECPoint), cudaMemcpyDeviceToHost);
        cudaFree(d_point);

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
                    std::cout << "\rTotal keys tested: " << keys_ps << std::endl;
                    keys_ps.store(0, std::memory_order_relaxed);
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

                current_key = hare.k1;
                p_key = current_key;

                keys_ps.fetch_add(1, std::memory_order_relaxed);

                // Linear Search Test Mode "brute-force", is recommended for ranges <= 20, larger ranges may cause hares to enter infinite loops/cycles.
                if (test_mode)
                {
                    hare.k1 = add_uint256(hare.k1, uint256_from_uint32(hare.speed));
                    hare.k2 = add_uint256(hare.k2, uint256_from_uint32(hare.speed));

                    for(int i = 2; i < 8; i++) {
                        hare.k1.limbs[i] = 0;
                        hare.k2.limbs[i] = 0;
                    }
                } else {
                    // Pollard's rho random walk
                    hare.k1 = f(hare.R, hare.k1, key_range);
                    hare.k2 = f(hare.R, hare.k2, key_range);
                }

                if (compare_uint256(hare.k1, min_scalar) < 0)
                { hare.k1 = add_uint256(hare.k1, min_scalar);}
                if (compare_uint256(hare.k2, min_scalar) < 0)
                { hare.k2 = add_uint256(hare.k2, min_scalar); }

                ECPoint pub1, pub2;
                ECPoint* d_pub1 = nullptr;
                ECPoint* d_G = nullptr;
                unsigned int* d_k1 = nullptr;

                cudaMalloc(&d_pub1, sizeof(ECPoint));
                cudaMalloc(&d_G, sizeof(ECPoint));
                cudaMalloc(&d_k1, sizeof(unsigned int) * 8);

                unsigned int k1_array[8];
                uint256_to_uint32_array(k1_array, hare.k1);

                cudaMemcpy(d_G, &G, sizeof(ECPoint), cudaMemcpyHostToDevice);
                cudaMemcpy(d_k1, k1_array, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice);

                scalar_mult<<<1,1>>>(d_pub1, d_k1, d_G);
                cudaDeviceSynchronize();
                cudaMemcpy(&pub1, d_pub1, sizeof(ECPoint), cudaMemcpyDeviceToHost);
                cudaFree(d_pub1);
                cudaFree(d_G);
                cudaFree(d_k1);

                ECPoint* d_pub2 = nullptr;
                ECPoint* d_G2 = nullptr;
                unsigned int* d_k2 = nullptr;

                cudaMalloc(&d_pub2, sizeof(ECPoint));
                cudaMalloc(&d_G2, sizeof(ECPoint));
                cudaMalloc(&d_k2, sizeof(unsigned int) * 8);

                unsigned int k2_array[8];
                uint256_to_uint32_array(k2_array, hare.k2);
                cudaMemcpy(d_G2, &G, sizeof(ECPoint), cudaMemcpyHostToDevice);
                cudaMemcpy(d_k2, k2_array, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice);

                scalar_mult<<<1,1>>>(d_pub2, d_k2, d_G2);
                cudaDeviceSynchronize();
                cudaMemcpy(&pub2, d_pub2, sizeof(ECPoint), cudaMemcpyDeviceToHost);
                cudaFree(d_pub2);
                cudaFree(d_G2);
                cudaFree(d_k2);

                unsigned char compressed1[33], compressed2[33];
                unsigned char* d_compressed1 = nullptr;
                ECPoint* d_pub1_comp = nullptr;

                cudaMalloc(&d_compressed1, 33);
                cudaMalloc(&d_pub1_comp, sizeof(ECPoint));
                cudaMemcpy(d_pub1_comp, &pub1, sizeof(ECPoint), cudaMemcpyHostToDevice);

                get_compressed_public_key<<<1,1>>>(d_compressed1, d_pub1_comp);
                cudaDeviceSynchronize();
                cudaMemcpy(compressed1, d_compressed1, 33, cudaMemcpyDeviceToHost);
                cudaFree(d_compressed1);
                cudaFree(d_pub1_comp);

                unsigned char* d_compressed2 = nullptr;
                ECPoint* d_pub2_comp = nullptr;

                cudaMalloc(&d_compressed2, 33);
                cudaMalloc(&d_pub2_comp, sizeof(ECPoint));
                cudaMemcpy(d_pub2_comp, &pub2, sizeof(ECPoint), cudaMemcpyHostToDevice);
                get_compressed_public_key<<<1,1>>>(d_compressed2, d_pub2_comp);
                cudaDeviceSynchronize();
                cudaMemcpy(compressed2, d_compressed2, 33, cudaMemcpyDeviceToHost);
                cudaFree(d_compressed2);
                cudaFree(d_pub2_comp);

                std::string current_pubkey_hex_R = bytes_to_hex((unsigned char*)compressed1, 33);
                std::string current_pubkey_hex_R1 = bytes_to_hex((unsigned char*)compressed2, 33);
                P_key = current_pubkey_hex_R;

                int LSB = 5;
                auto DP = [LSB](const ECPoint& point) -> bool {
    for (int i = 0; i < LSB; i++) {
        if ((point.x[0] >> i) & 1) return false;
    }
    return true;
};

                // Caso onde x é par:
                if (DP(pub1) && DP(pub2) && !test_mode) {

                    bool x_equal = true;
                    for (int j = 0; j < 8; j++) {
                        if (pub1.x[j] != pub2.x[j]) {
                            x_equal = false;
                            break;
                        }
                    }

                    if (x_equal && compare_uint256(hare.k1, hare.k2) != 0) {

                        /*
                            Calcular a diferença (d) entre os pontos pubkey1 e pubkey2:
                            d = k1 - k2 tal que P1 = k1 * G e P2 = k2 * G

                            Verificar se: (d * G ≡ 0), caso verdadeiro: found_key = (k2 + d) % n;
                        */

                        uint256_t tortoise_key = hare.k2;
                        uint256_t hare_key = hare.k1;
                        uint256_t d = (compare_uint256(hare_key, tortoise_key) >= 0) ? 
                                    sub_uint256(hare_key, tortoise_key) : 
                                    sub_uint256(N, sub_uint256(tortoise_key, hare_key));

                        ECPoint* d_test_point = nullptr;
                        ECPoint* d_G_test = nullptr;
                        unsigned int* d_d = nullptr;

                        cudaMalloc(&d_test_point, sizeof(ECPoint));
                        cudaMalloc(&d_G_test, sizeof(ECPoint));
                        cudaMalloc(&d_d, sizeof(unsigned int) * 8);

                        unsigned int d_array[8];
                        uint256_to_uint32_array(d_array, d);

                        cudaMemcpy(d_G_test, &G, sizeof(ECPoint), cudaMemcpyHostToDevice);
                        cudaMemcpy(d_d, d_array, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice);

                        scalar_mult<<<1,1>>>(d_test_point, d_d, d_G_test);
                        cudaDeviceSynchronize();

                        ECPoint test_point;
                        cudaMemcpy(&test_point, d_test_point, sizeof(ECPoint), cudaMemcpyDeviceToHost);

                        cudaFree(d_test_point);
                        cudaFree(d_G_test);
                        cudaFree(d_d);

                        if (test_point.infinity == 1) {

                            found_key = add_uint256(tortoise_key, d);
                            if (compare_uint256(found_key, N) >= 0) {
                                found_key = sub_uint256(found_key, N);
                            }

                            ECPoint* d_verify_point = nullptr;
                            ECPoint* d_G_verify = nullptr;
                            unsigned int* d_found_key = nullptr;

                            cudaMalloc(&d_verify_point, sizeof(ECPoint));
                            cudaMalloc(&d_G_verify, sizeof(ECPoint));
                            cudaMalloc(&d_found_key, sizeof(unsigned int) * 8);

                            unsigned int found_key_array[8];
                            uint256_to_uint32_array(found_key_array, found_key);

                            cudaMemcpy(d_G_verify, &G, sizeof(ECPoint), cudaMemcpyHostToDevice);
                            cudaMemcpy(d_found_key, found_key_array, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice);

                            scalar_mult<<<1,1>>>(d_verify_point, d_found_key, d_G_verify);
                            cudaDeviceSynchronize();

                            ECPoint verify_point;
                            cudaMemcpy(&verify_point, d_verify_point, sizeof(ECPoint), cudaMemcpyDeviceToHost);

                            cudaFree(d_verify_point);
                            cudaFree(d_G_verify);
                            cudaFree(d_found_key);

                            unsigned char* d_compressed_verify = nullptr;
                            ECPoint* d_verify_comp = nullptr;

                            cudaMalloc(&d_compressed_verify, 33);
                            cudaMalloc(&d_verify_comp, sizeof(ECPoint));

                            cudaMemcpy(d_verify_comp, &verify_point, sizeof(ECPoint), cudaMemcpyHostToDevice);

                            get_compressed_public_key<<<1,1>>>(d_compressed_verify, d_verify_comp);
                            cudaDeviceSynchronize();

                            unsigned char compressed_verify[33];
                            cudaMemcpy(compressed_verify, d_compressed_verify, 33, cudaMemcpyDeviceToHost);

                            cudaFree(d_compressed_verify);
                            cudaFree(d_verify_comp);

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

                    found_key = (current_pubkey_hex_R == target_pubkey_hex) ? hare.k1 : hare.k2;
                    std::cout << "\033[32mPrivate key found!\033[0m" << std::endl;
                    std::cout << "Private Key: " << uint_256_to_hex(found_key) << std::endl;

                    search_in_progress.store(false);
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    log_thread.join();

    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Total duration: " << std::setw(2) << std::setfill('0') << duration.count() / 3600 << ":"
    << std::setw(2) << std::setfill('0') << (duration.count() % 3600) / 60 << ":"
    << std::setw(2) << std::setfill('0') << duration.count() % 60 << std::endl;

    return found_key;
}
*/

/*
int main(int argc, char* argv[]) {

    if (argc < 3 || argc > 4) {
        std::cerr << "Uso: " << argv[0] << " <Compressed Public Key> <Key Range> <Optional Test Mode: --t to true>" << std::endl;
        return 1;
    }

    init_secp256k1();

    NUM_JUMPS = (get_memory_bytes() / 2) / 32; // div 2 >> 50% of ram

    std::string pub_key_hex(argv[1]);
    int key_range = std::stoi(argv[2]);
    bool test_mode = (argc == 4 && std::string(argv[3]) == "--t");

    if(test_mode && key_range > 20)
    {
        std::cout << "Use a range of 20 bits or less for test mode!" << std::endl;
        return 1;
    } else if (!test_mode && key_range > 20) {

        if(NUM_JUMPS != 0) { precompute_jumps(key_range); };

        while (!precomputed_done) {
            std::this_thread::sleep_for(std::chrono::minutes(1));
        }
    }

    uint256_t found_key = prho(pub_key_hex, key_range, 3, test_mode);

    auto uint_256_to_hex = [](const uint256_t& value) -> std::string {
        std::ostringstream oss;
        for(int i = 0; i < 8; i++) {
            oss << std::setw(8) << std::setfill('0') << std::hex << value.limbs[i];
        }
        return oss.str();
    };

    std::cout << "Chave privada encontrada: " << uint_256_to_hex(found_key) << std::endl;

    return 0;
}
*/