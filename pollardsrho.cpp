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
#include "secp256k1.h"
#include <sys/sysinfo.h>
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
    uint32_t limbs[8];
};

ECPointJacobian G;
ECPointJacobian H;

//constexpr uint64_t N_CONST[4] = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
//constexpr uint64_t GX_CONST[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
//constexpr uint64_t GY_CONST[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };

const uint256_t N = {
    0xBFD25E8C, 0xD0364141, 0xBAAEDCE6, 0xAF48A03B,
    0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF
};

const uint256_t GX = {
    0x59F2815B, 0x16F81798, 0x029BFCDB, 0x2DCE28D9,
    0x55A06295, 0xCE870B07, 0x79BE667E, 0xF9DCBBAC
};

const uint256_t GY = {
    0x9C47D08F, 0xFB10D4B8, 0xFD17B448, 0xA6855419,
    0x5DA4FBFC, 0x0E1108A8, 0x483ADA77, 0x26A3C465
};

void uint256_to_uint64_array(uint64_t* out, const uint256_t& value) {
    out[3] = (static_cast<uint64_t>(value.limbs[6]) << 32) | value.limbs[7];
    out[2] = (static_cast<uint64_t>(value.limbs[4]) << 32) | value.limbs[5];
    out[1] = (static_cast<uint64_t>(value.limbs[2]) << 32) | value.limbs[3];
    out[0] = (static_cast<uint64_t>(value.limbs[0]) << 32) | value.limbs[1];
}

/*
void init_secp256k1() {
    uint64_t gx_arr[4], gy_arr[4];
    uint256_to_uint64_array(gx_arr, GX);
    uint256_to_uint64_array(gy_arr, GY);

    for(int i = 0; i < 4; i++) {
        G.X[i] = gx_arr[i];
        G.Y[i] = gy_arr[i];
        G.Z[i] = (i == 3) ? 1 : 0;
    }
    G.infinity = 0;

    ECPointJacobian* d_G = nullptr;
    #ifdef __CUDACC__
    cudaMalloc(&d_G, sizeof(ECPointJacobian));
    cudaMemcpy(d_G, &G, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
    #endif

    point_double_jacobian(d_G, d_G);
    #ifdef __CUDACC__
    cudaDeviceSynchronize();
    cudaMemcpy(&H, d_G, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
    cudaFree(d_G);
    #endif
}
*/

void init_secp256k1() {
    // Inicializa G
    uint64_t gx_arr[4], gy_arr[4];
    uint256_to_uint64_array(gx_arr, GX);
    uint256_to_uint64_array(gy_arr, GY);
    for(int i = 0; i < 4; i++) {
        G.X[i] = gx_arr[i];
        G.Y[i] = gy_arr[i];
        G.Z[i] = (i == 3) ? 1 : 0;
    }
    G.infinity = 0;

    #ifdef __CUDACC__
    ECPointJacobian* d_G = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_G, sizeof(ECPointJacobian));
    if(err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    cudaMemcpy(d_G, &G, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
    point_double_jacobian(d_G, d_G);
    cudaDeviceSynchronize();
    cudaMemcpy(&H, d_G, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
    cudaFree(d_G);
    #else
    // CPU: passa ponteiros normais
    point_double_jacobian(&G, &G);
    H = G;
    #endif
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

double TOTAL_RAM = (get_memory_gb() / 2);
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

// OTIMIZAÇÃO 1 e 4: Comparação de coordenadas e buffers persistentes
struct PersistentBuffers {
    ECPointJacobian* d_R;
    ECPointJacobian* d_G;
    ECPointJacobian* d_H;
    uint64_t* d_k;

    PersistentBuffers() {
        #ifdef __CUDACC_
        cudaMalloc(&d_R, sizeof(ECPointJacobian));
        cudaMalloc(&d_G, sizeof(ECPointJacobian));
        cudaMalloc(&d_H, sizeof(ECPointJacobian));
        cudaMalloc(&d_k, sizeof(uint64_t) * 4);

        // Copiar G e H uma única vez
        cudaMemcpy(d_G, &G, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
        cudaMemcpy(d_H, &H, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
        #endif
    }

    ~PersistentBuffers() {
        #ifdef __CUDACC_
        cudaFree(d_R);
        cudaFree(d_G);
        cudaFree(d_H);
        cudaFree(d_k);
        #endif
    }
};

// OTIMIZAÇÃO 1: Comparar coordenadas X em Jacobiano
bool compare_jacobian_x(const ECPointJacobian& p1, const ECPointJacobian& p2) {
    for(int i = 0; i < 4; i++) {
        if(p1.X[i] != p2.X[i] || p1.Z[i] != p2.Z[i]) {
            return false;
        }
    }
    return true;
}

// OTIMIZAÇÃO 4: f otimizado com buffers persistentes
uint256_t f_optimized(ECPointJacobian& R, uint256_t k, int key_range, PersistentBuffers& buffers) {
    const uint256_t mask = mask_for_bits(key_range);

    uint256_t x_coord{};
    for(int i = 0; i < 8; i++) {
        x_coord.limbs[7-i] = static_cast<uint32_t>(R.X[3-i/2] >> ((i%2) * 32));
    }

    uint64_t x_low64 = (static_cast<uint64_t>(x_coord.limbs[6]) << 32) | x_coord.limbs[7];
    unsigned int op = static_cast<unsigned int>(x_low64 % 3);
    size_t idx = static_cast<size_t>(x_low64 % NUM_JUMPS);

    // Usar buffers persistentes
    #ifdef __CUDACC_
    cudaMemcpy(buffers.d_R, &R, sizeof(ECPointJacobian), cudaMemcpyHostToDevice);
    #endif

    switch(op) {
        case 0:
            point_add_jacobian(buffers.d_R, buffers.d_R, buffers.d_G);
            k = add_uint256(k, precomputed_jumps[idx]);
            break;
        case 1:
            point_add_jacobian(buffers.d_R, buffers.d_R, buffers.d_H);
            k = add_uint256(k, precomputed_jumps[idx]);
            break;
        default:
            point_double_jacobian(buffers.d_R, buffers.d_R);
            k = left_shift_uint256(k, 1);
            break;
    }

    for(int i = 0; i < 8; i++) k.limbs[i] &= mask.limbs[i];

    // Não sincronizar aqui - será feito periodicamente
    #ifdef __CUDACC_
    cudaMemcpy(&R, buffers.d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
    #endif

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

    std::atomic<unsigned long long> keys_ps{0};
    std::atomic<unsigned long long> iteration_counter{0}; // OTIMIZAÇÃO 3
    std::atomic<bool> search_in_progress(true);
    std::mutex pgrs;
    std::string P_key;

    uint256_t p_key{};
    uint256_t found_key{};

    auto target_pubkey = hex_to_bytes(target_pubkey_hex);

    struct HareState {
        uint256_t k1, k2;
        ECPointJacobian R;
        int speed;
        PersistentBuffers* buffers; // OTIMIZAÇÃO 2: Buffers persistentes
    };

    PKG pkg(min_scalar, max_scalar);
    uint256_t current_key = pkg.generate();

    std::vector<HareState> hare_states(hares);

    // OTIMIZAÇÃO 2: Alocar buffers persistentes para cada hare
    for (int i = 0; i < hares; ++i) {
        hare_states[i].buffers = new PersistentBuffers();

        ECPointJacobian* d_point = nullptr;
        #ifdef __CUDACC_
        cudaMalloc(&d_point, sizeof(ECPointJacobian));
        #endif
        point_init_jacobian(d_point);
        #ifdef __CUDACC_
        cudaMemcpy(&hare_states[i].R, d_point, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
        cudaFree(d_point);
        #endif

        hare_states[i].R = G;
        hare_states[i].R.infinity = 0;
        hare_states[i].speed = 0;
        hare_states[i].k1 = pkg.generate();
        hare_states[i].k2 = pkg.generate();
    }

    // OTIMIZAÇÃO 6: Log periódico
    std::thread log_thread([&]() {
        try {
            while(search_in_progress.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                std::lock_guard<std::mutex> lock(pgrs);

                if(search_in_progress.load()) {
                    std::cout << "\rCurrent private key: " << uint_256_to_hex(p_key) << std::endl;
                    std::cout << "\rLast tested public key: " << P_key << std::endl;
                    std::cout << "\rTotal keys tested: " << keys_ps << std::endl;
                    std::cout << "\rIterations: " << iteration_counter.load() << std::endl;
                    keys_ps.store(0, std::memory_order_relaxed);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in log_thread: " << e.what() << std::endl;
        }
    });

    const unsigned long long SYNC_INTERVAL = 5000000; // OTIMIZAÇÃO 3: Sincronizar a cada 5M iterações

    try {
        while (search_in_progress.load()) {

            for (int i = 0; i < hares; ++i) {

                HareState& hare = hare_states[i];
                hare.speed = (i == 0) ? 1 : (i + 1);

                current_key = hare.k1;
                p_key = current_key;

                keys_ps.fetch_add(1, std::memory_order_relaxed);
                unsigned long long iter = iteration_counter.fetch_add(1, std::memory_order_relaxed);

                if (test_mode)
                {
                    hare.k1 = add_uint256(hare.k1, uint256_from_uint32(hare.speed));
                    hare.k2 = add_uint256(hare.k2, uint256_from_uint32(hare.speed));

                    for(int j = 2; j < 8; j++) {
                        hare.k1.limbs[j] = 0;
                        hare.k2.limbs[j] = 0;
                    }
                } else {
                    // OTIMIZAÇÃO 4: Usar f otimizado
                    hare.k1 = f_optimized(hare.R, hare.k1, key_range, *hare.buffers);
                    hare.k2 = f_optimized(hare.R, hare.k2, key_range, *hare.buffers);
                }

                if (compare_uint256(hare.k1, min_scalar) < 0)
                { hare.k1 = add_uint256(hare.k1, min_scalar);}
                if (compare_uint256(hare.k2, min_scalar) < 0)
                { hare.k2 = add_uint256(hare.k2, min_scalar); }

                // OTIMIZAÇÃO 3: Sincronização periódica
                bool should_sync = (iter % SYNC_INTERVAL == 0);

                // OTIMIZAÇÃO 2: Reutilizar buffers
                ECPointJacobian pub1_jac, pub2_jac;

                uint64_t k1_array[4], k2_array[4];
                uint256_to_uint64_array(k1_array, hare.k1);
                uint256_to_uint64_array(k2_array, hare.k2);

                #ifdef __CUDACC_
                cudaMemcpy(hare.buffers->d_k, k1_array, sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
                #endif
                scalar_mult_jacobian(hare.buffers->d_R, hare.buffers->d_k);
                #ifdef __CUDACC_
                if(should_sync) cudaDeviceSynchronize(); // OTIMIZAÇÃO 3
                cudaMemcpy(&pub1_jac, hare.buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
                cudaMemcpy(hare.buffers->d_k, k2_array, sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
                #endif
                scalar_mult_jacobian(hare.buffers->d_R, hare.buffers->d_k);
                #ifdef __CUDACC_
                if(should_sync) cudaDeviceSynchronize(); // OTIMIZAÇÃO 3
                cudaMemcpy(&pub2_jac, hare.buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
                #endif
                // OTIMIZAÇÃO 1: Comparar primeiro coordenadas X em Jacobiano
                bool potential_collision = compare_jacobian_x(pub1_jac, pub2_jac);

                // Somente se houver potencial colisão, fazer conversão completa
                if(potential_collision || should_sync) {
                    ECPoint pub1, pub2;
                    for(int j = 0; j < 4; j++) {
                        pub1.x[j] = pub1_jac.X[j];
                        pub1.y[j] = pub1_jac.Y[j];
                        pub2.x[j] = pub2_jac.X[j];
                        pub2.y[j] = pub2_jac.Y[j];
                    }
                    pub1.infinity = pub1_jac.infinity;
                    pub2.infinity = pub2_jac.infinity;

                    // OTIMIZAÇÃO 5: Somente gerar chaves comprimidas em verificação
                    unsigned char compressed1[33], compressed2[33];
                    unsigned char* d_compressed = nullptr;
                    ECPoint* d_pub_comp = nullptr;
                    #ifdef __CUDACC_
                    cudaMalloc(&d_compressed, 33);
                    cudaMalloc(&d_pub_comp, sizeof(ECPoint));
                    cudaMemcpy(d_pub_comp, &pub1, sizeof(ECPoint), cudaMemcpyHostToDevice);
                    #endif
                    get_compressed_public_key(d_compressed, d_pub_comp);
                    #ifdef __CUDACC_
                    if(should_sync) cudaDeviceSynchronize();
                    cudaMemcpy(compressed1, d_compressed, 33, cudaMemcpyDeviceToHost);
                    cudaMemcpy(d_pub_comp, &pub2, sizeof(ECPoint), cudaMemcpyHostToDevice);
                    #endif
                    get_compressed_public_key(d_compressed, d_pub_comp);
                    #ifdef __CUDACC_
                    if(should_sync) cudaDeviceSynchronize();
                    cudaMemcpy(compressed2, d_compressed, 33, cudaMemcpyDeviceToHost);
                    cudaFree(d_compressed);
                    cudaFree(d_pub_comp);
                    #endif

                    std::string current_pubkey_hex_R = bytes_to_hex((unsigned char*)compressed1, 33);
                    P_key = current_pubkey_hex_R;

                    int LSB = 5;
                    auto DP = [LSB](const ECPoint& point) -> bool {
                        for (int i = 0; i < LSB; i++) {
                            if ((point.x[0] >> i) & 1) return false;
                        }
                        return true;
                    };

                    if (DP(pub1) && DP(pub2) && !test_mode && potential_collision) {

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
                            uint256_t d = (compare_uint256(hare_key, tortoise_key) >= 0) ? 
                                        sub_uint256(hare_key, tortoise_key) : 
                                        sub_uint256(N, sub_uint256(tortoise_key, hare_key));

                            uint64_t d_array[4];
                            uint256_to_uint64_array(d_array, d);
                            #ifdef __CUDACC_
                            cudaMemcpy(hare.buffers->d_k, d_array, sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
                            #endif
                            scalar_mult_jacobian(hare.buffers->d_R, hare.buffers->d_k);
                            #ifdef __CUDACC_
                            cudaDeviceSynchronize();
                            #endif
                            ECPointJacobian test_point_jac;
                            #ifdef __CUDACC_
                            cudaMemcpy(&test_point_jac, hare.buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
                            #endif
                            if (test_point_jac.infinity == 1) {

                                found_key = add_uint256(tortoise_key, d);
                                if (compare_uint256(found_key, N) >= 0) {
                                    found_key = sub_uint256(found_key, N);
                                }

                                uint64_t found_key_array[4];
                                uint256_to_uint64_array(found_key_array, found_key);
                                #ifdef __CUDACC_
                                cudaMemcpy(hare.buffers->d_k, found_key_array, sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
                                #endif
                                scalar_mult_jacobian(hare.buffers->d_R, hare.buffers->d_k);
                                #ifdef __CUDACC_
                                cudaDeviceSynchronize();
                                #endif
                                ECPointJacobian verify_point_jac;
                                #ifdef __CUDACC_
                                cudaMemcpy(&verify_point_jac, hare.buffers->d_R, sizeof(ECPointJacobian), cudaMemcpyDeviceToHost);
                                #endif
                                ECPoint verify_point;
                                for(int j = 0; j < 4; j++) {
                                    verify_point.x[j] = verify_point_jac.X[j];
                                    verify_point.y[j] = verify_point_jac.Y[j];
                                }
                                verify_point.infinity = verify_point_jac.infinity;

                                unsigned char* d_compressed_verify = nullptr;
                                ECPoint* d_verify_comp = nullptr;
                                #ifdef __CUDACC_
                                cudaMalloc(&d_compressed_verify, 33);
                                cudaMalloc(&d_verify_comp, sizeof(ECPoint));
                                cudaMemcpy(d_verify_comp, &verify_point, sizeof(ECPoint), cudaMemcpyHostToDevice);
                                #endif
                                get_compressed_public_key(d_compressed_verify, d_verify_comp);
                                #ifdef __CUDACC_
                                cudaDeviceSynchronize();
                                #endif
                                unsigned char compressed_verify[33];
                                #ifdef __CUDACC_
                                cudaMemcpy(compressed_verify, d_compressed_verify, 33, cudaMemcpyDeviceToHost);
                                cudaFree(d_compressed_verify);
                                cudaFree(d_verify_comp);
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

                    // Verificação contra target
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

    // Liberar buffers persistentes
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

    NUM_JUMPS = (get_memory_bytes() / 2) / 32;

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