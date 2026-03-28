/******************************************************************************************************
* This file is part of the Pollard's Rho distribution: (https://github.com/lucaselblanc/pollardsrho)  *
* Copyright (c) 2024, 2026 Lucas Leblanc.                                                             *
* Distributed under the MIT software license, see the accompanying.                                   *
* file COPYING or https://www.opensource.org/licenses/mit-license.php.                                *
******************************************************************************************************/

/*****************************************
* Pollard's Rho Algorithm for SECP256K1  *
* Written by Lucas Leblanc               *
******************************************/

/* --- LAMBDA RHO VERSION (λρ) --- */

#include "secp256k1.h"
#include "parallel_hashmap/phmap.h"
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
#include <ctime>
#include <cmath>
#include <cstring>

constexpr uint256_t N = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
constexpr uint256_t P = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
constexpr uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
constexpr uint64_t SUB2_FP[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
constexpr uint64_t ZERO[4] = {0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL};

const std::string& RED = "\033[91m";
const std::string& GREEN = "\033[92m";
const std::string& BLUE = "\033[94m";
const std::string& CYAN = "\033[38;5;39m";
const std::string& DARK_PINK = "\033[38;2;140;70;140m";
const std::string& PINK = "\033[35m";
const std::string& ORANGE = "\033[38;2;255;128;0m";
const std::string& RESET = "\033[0m";

struct Buffers {
    uint64_t* scalarStepsG;
    uint64_t* scalarStepsH;
};

struct WalkState {
    uint256_t a, b;
    std::mt19937_64 rng;
    ECPointJacobian R;
    Buffers* buffers;
    uint32_t walk_id;
    uint64_t snapshot_steps;
    uint64_t snapshot_x[4];
    uint64_t prev_x1[4];
    uint64_t prev_x2[4];
};

struct MurmurHash3 {
    //MurmurHash3<Avalanche constants>
    size_t operator()(uint64_t x) const {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }
};

struct DPEntry {
    uint256_t a;
    uint256_t b;
    uint64_t x;
};

void loading_bar(uint64_t current, uint64_t total, const std::string& label) {
    if (total == 0) return;
    float percent = (float)current / total;
    int barWidth = 40;
    int pos = barWidth * percent;

    std::cout << "\033[2A\r\033[J";

    std::cout << label << "\n";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << BLUE << "█" << RESET;
        else std::cout << BLUE << "▒" << RESET;
    }

    std::cout << " " << std::fixed << std::setprecision(1) << (percent * 100.0) << "%" << std::endl;
}

int windowSize = 12; //Default value used only if getfcw() detection cannot access the processor for some reason, it can happen on different platforms like termux for example.

void uint256_to_uint64_array(uint64_t* out, const uint256_t& value) {
    out[0] = value.limbs[0];
    out[1] = value.limbs[1];
    out[2] = value.limbs[2];
    out[3] = value.limbs[3];
}

std::string uint256_to_hex(const uint256_t& value) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for(int i = 3; i >= 0; i--) {
        oss << std::setw(16) << value.limbs[i];
    }
    return oss.str();
}

std::string gradient_zeros(std::string hex, const std::string& color_1, const std::string& color_2) {
    size_t first_nonzero = hex.find_first_not_of('0');
    if (std::string::npos == first_nonzero) { return color_1 + hex + RESET; }
    std::string leading_zeros = hex.substr(0, first_nonzero);
    std::string value = hex.substr(first_nonzero);
    return color_1 + leading_zeros + RESET + color_2 + value + RESET;
}

std::vector<unsigned char> hex_to_bytes(const std::string& hex) {
    if (hex.length() % 2 != 0) {
        throw std::invalid_argument("A string hexadecimal deve ter n par de caracteres");
    }

    std::vector<unsigned char> bytes(hex.size() / 2);
    for (size_t i = 0; i < bytes.size(); i++) {
        bytes[i] = static_cast<unsigned char>(
            std::stoi(hex.substr(2 * i, 2), nullptr, 16)
        );
    }
    return bytes;
}

uint256_t mod_add_N(const uint256_t& a, const uint256_t& b) {
    uint256_t res;
    uint64_t carry = 0;
    for(int i = 0; i < 4; i++) {
        uint64_t sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]) || (carry && sum == a.limbs[i]);
        res.limbs[i] = sum;
    }
    bool ge = true;
    for (int i = 3; i >= 0; i--) {
        if (res.limbs[i] > N.limbs[i]) break;
        if (res.limbs[i] < N.limbs[i]) { ge = false; break; }
    }
    if (ge) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t diff = res.limbs[i] - N.limbs[i] - borrow;
            borrow = (res.limbs[i] < N.limbs[i]) || (borrow && res.limbs[i] == N.limbs[i]);
            res.limbs[i] = diff;
        }
    }
    return res;
}

uint256_t mod_neg_N(const uint256_t& a) {
    bool is_zero = (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3]) == 0;
    if (is_zero) return a;
    uint256_t res;
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t diff = N.limbs[i] - a.limbs[i] - borrow;
        borrow = (N.limbs[i] < a.limbs[i]) || (borrow && N.limbs[i] == a.limbs[i]);
        res.limbs[i] = diff;
    }
    return res;
}

uint256_t mod_sub_N(const uint256_t& a, const uint256_t& b) {
    return mod_add_N(a, mod_neg_N(b));
}

void getfcw(int key_range) {
    int w = 4;
    double exp_steps = std::pow(2, key_range / 2.0);
    size_t l2Size = 0;
    size_t l3Size = 0;
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

            if(exp_steps < size) {
                if (L == 3) l3Size = size;
            }
            else {
                if (L == 2) l2Size = size;
            }
        }
        catch(const std::invalid_argument& e) {
            std::cout << ORANGE << "Warning: " << e.what() << RESET << std::endl;
            continue;
        }
        catch(const std::out_of_range& e) {
            std::cout << ORANGE << "Warning: " << e.what() << RESET << std::endl;
            continue;
        }
    }

    size_t lSize = 0;

    if (l2Size > 0) lSize = l2Size;
    else if (l3Size > 0) lSize = l3Size;

    if (lSize > 0)
    {
        size_t maxPoints = lSize / 128;
        if (maxPoints > 0) {
            w = static_cast<int>(std::floor(std::log2(maxPoints)));
        }
    }

    if(w > 4) {
        windowSize = w;
    }
}

void initScalarSteps(uint64_t* steps, int windowSize) {
    int tableSize = (1 << windowSize) - 1;
    for (int i = 1; i <= tableSize; i++) {
        uint256_t val = { (uint64_t)i, 0, 0, 0 };
        int idx = i - 1;
        for(int k=0; k<4; k++) {
            steps[idx * 4 + k] = val.limbs[k];
        }
    }
}

void init_secp256k1(int key_range) {
    getfcw(key_range);

    preCompG = new ECPointJacobian[1ULL << windowSize];
    preCompGphi = new ECPointJacobian[1ULL << windowSize];
    preCompH = new ECPointJacobian[1ULL << windowSize]; //internal use in secp256k1.h
    preCompHphi = new ECPointJacobian[1ULL << windowSize]; //internal use in secp256k1.h
    jacNorm = new ECPointJacobian[windowSize]; //internal use in secp256k1.h
    jacNormH = new ECPointJacobian[windowSize]; //internal use in secp256k1.h
    jacEndo = new ECPointJacobian[windowSize]; //internal use in secp256k1.h
    jacEndoH = new ECPointJacobian[windowSize]; //internal use in secp256k1.h

    initPreCompG(windowSize);
}

uint256_t add_uint256(const uint256_t& a, const uint256_t& b) {
    uint256_t result{};
    uint64_t carry = 0;
    for(int i = 0; i < 4; i++) {
        uint64_t sum = a.limbs[i] + b.limbs[i] + carry;
        if (carry) carry = (sum <= a.limbs[i]);
        else carry = (sum < a.limbs[i]);
        result.limbs[i] = sum;
    }
    return result;
}

uint256_t sub_uint256(const uint256_t& a, const uint256_t& b) {
    uint256_t result{};
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t ai = a.limbs[i];
        uint64_t bi = b.limbs[i];
        uint64_t res = ai - bi - borrow;
        if (borrow) borrow = (ai <= bi);
        else borrow = (ai < bi);
        result.limbs[i] = res;
    }
    return result;
}

int compare_uint256(const uint256_t& a, const uint256_t& b) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return 1;
        if (a.limbs[i] < b.limbs[i]) return -1;
    }
    return 0;
}

uint256_t rng_mersenne_twister(const uint256_t& min_scalar, const uint256_t& max_scalar, int key_range, std::mt19937_64& rng) {
    uint256_t r{};
    uint256_t range = sub_uint256(max_scalar, min_scalar);
    uint256_t one = {1, 0, 0, 0};
    range = add_uint256(range, one);

    int max_limb_idx = (key_range - 1) / 64;
    int bits_in_last_limb = key_range % 64;

    do {
        for (int i = 0; i < 4; i++) r.limbs[i] = rng();
        for (int i = max_limb_idx + 1; i < 4; i++) r.limbs[i] = 0;
        if (bits_in_last_limb > 0) {
            r.limbs[max_limb_idx] &= (1ULL << bits_in_last_limb) - 1;
        }
    } while (compare_uint256(r, range) >= 0);

    return add_uint256(r, min_scalar);
}

uint32_t get_step_idx(const uint64_t* x, uint32_t N_STEPS) {
    uint64_t combined = x[0] ^ (x[1] << 1) ^ (x[2] << 2) ^ (x[3] << 3);
    MurmurHash3 hasher;
    return static_cast<uint32_t>(hasher(combined) % N_STEPS);
}

bool DP(const uint64_t* affine_x, int DP_BITS) {
    return (affine_x[0] & ((1ULL << DP_BITS) - 1)) == 0;
}

void batchJacobianToAffine(ECPointAffine* aff_out, const ECPointJacobian* jac_in, int count, uint64_t* scratch_prefix, uint64_t* scratch_inv) {
    if (count <= 0) return;
    if (jacobianIsInfinity(&jac_in[0])) {
        memcpy(&scratch_prefix[0], ONE_MONT, 32);
    } else {
        memcpy(&scratch_prefix[0], jac_in[0].Z, 32);
    }

    for (int i = 1; i < count; i++) {
        if (jacobianIsInfinity(&jac_in[i])) {
            memcpy(&scratch_prefix[i * 4], &scratch_prefix[(i - 1) * 4], 32);
        } else {
            modMulMontP(&scratch_prefix[i * 4], &scratch_prefix[(i - 1) * 4], jac_in[i].Z);
        }
    }

    uint64_t total_inv[4];
    modExpMontP(total_inv, &scratch_prefix[(count - 1) * 4], SUB2_FP);
    uint64_t current_inv[4];
    memcpy(current_inv, total_inv, 32);

    for (int i = count - 1; i > 0; i--) {
        if (jacobianIsInfinity(&jac_in[i])) {
            continue;
        }
        modMulMontP(&scratch_inv[i * 4], current_inv, &scratch_prefix[(i - 1) * 4]);
        modMulMontP(current_inv, current_inv, jac_in[i].Z);
    }

    if (!jacobianIsInfinity(&jac_in[0])) {
        memcpy(&scratch_inv[0], current_inv, 32);
    }

    for (int i = 0; i < count; i++) {
        if (jacobianIsInfinity(&jac_in[i])) {
            aff_out[i].infinity = 1;
            memset(aff_out[i].x, 0, 32);
            memset(aff_out[i].y, 0, 32);
            continue;
        }

        uint64_t* z_inv = &scratch_inv[i * 4];
        uint64_t z2[4], z3[4], tmp_mont[4];

        modMulMontP(z2, z_inv, z_inv);
        modMulMontP(tmp_mont, jac_in[i].X, z2);
        fromMontgomeryP(aff_out[i].x, tmp_mont);

        aff_out[i].infinity = 0;
    }
}

uint256_t prho(std::string target_pubkey_hex, int key_range, const int WALKERS, const int DP_BITS) {
    std::atomic<bool> search_in_progress(true);
    std::atomic<unsigned long long> total_iters{0};
    std::atomic<unsigned long long> total_cycles{0};
    uint256_t k{};
    auto cores = std::thread::hardware_concurrency();
    auto start_time = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::tm start_tm{};
    localtime_r(&start_time_t, &start_tm);
    const uint32_t N_STEPS = 2048;
    auto target_pubkey = hex_to_bytes(target_pubkey_hex);
    uint256_t min_scalar{}, max_scalar{};
    {
        int limb = (key_range - 1) / 64;
        int bit = (key_range - 1) % 64;
        min_scalar.limbs[limb] = 1ULL << bit;
        for (int i = 0; i < limb; i++) max_scalar.limbs[i] = ~0ULL;
        max_scalar.limbs[limb] = (bit == 63) ? ~0ULL : (1ULL << (bit + 1)) - 1;
    }

    std::cout << CYAN << "Started at: " << RESET << PINK << std::put_time(&start_tm, "%H:%M:%S") << RESET << std::endl;
    std::cout << CYAN << "WALKERS: " << RESET << PINK << WALKERS << RESET << std::endl;
    std::cout << CYAN << "DP BITS: " << RESET << PINK << DP_BITS << RESET << std::endl;
    std::cout << CYAN << "Key Range: " << RESET << PINK << (key_range) << RESET << std::endl;
    std::cout << CYAN << "Min Range: " << RESET << gradient_zeros(uint256_to_hex(min_scalar), DARK_PINK, PINK) << std::endl;
    std::cout << CYAN << "Max Range: " << RESET << gradient_zeros(uint256_to_hex(max_scalar), DARK_PINK, PINK) << std::endl;
    std::cout << BLUE << "---------------------------------------------------------------------------" << RESET;
    std::cout << "\n\n\n\n";

    ECPointAffine target_affine{};
    ECPointJacobian target_affine_jac{};
    decompressPublicKey(&target_affine, target_pubkey.data());
    affineToJacobian(&target_affine_jac, &target_affine);
    initPreCompH(&target_affine_jac, windowSize);

    ECPointJacobian G_OFFSET;
    jacobianScalarMultPhi(&G_OFFSET, preCompG, preCompGphi, max_scalar.limbs, windowSize);
    if (!jacobianIsInfinity(&G_OFFSET)) { modSubP(G_OFFSET.Y, ZERO, G_OFFSET.Y); }

    struct StepLocal { ECPointJacobian point; uint256_t a; uint256_t b; };
    std::vector<StepLocal> localStepTable(N_STEPS);
    std::mt19937_64 salt(target_affine.x[0]);
    uint256_t stepSize = {};
    stepSize.limbs[(key_range / 2) / 64] = 1ULL << ((key_range / 2) % 64);

    for (int i = 0; i < N_STEPS; i++) {
        localStepTable[i].a = rng_mersenne_twister(uint256_t{0}, stepSize, key_range / 2, salt);
        localStepTable[i].b = uint256_t{};
        uint64_t a_tmp[4];
        uint256_to_uint64_array(a_tmp, localStepTable[i].a);
        jacobianScalarMultPhi(&localStepTable[i].point, preCompG, preCompGphi, a_tmp, windowSize);
        ECPointAffine aff_step;
        jacobianToAffine(&aff_step, &localStepTable[i].point);
        affineToJacobian(&localStepTable[i].point, &aff_step);
    }

    phmap::parallel_flat_hash_map<uint64_t, std::vector<DPEntry>, MurmurHash3, phmap::priv::hash_default_eq<uint64_t>, std::allocator<std::pair<const uint64_t, std::vector<DPEntry>>>, 8, std::mutex > dp_table;
    std::vector<WalkState> walkers_state(WALKERS);
    std::vector<uint64_t> sharedScalarStepsG((1ULL << windowSize) * 4);
    std::vector<uint64_t> sharedScalarStepsH((1ULL << windowSize) * 4);
    initScalarSteps(sharedScalarStepsG.data(), windowSize);
    initScalarSteps(sharedScalarStepsH.data(), windowSize);

    std::string header = "\033[96m[!] Loading Walkers... \033[0m";
    for (int i = 0; i < WALKERS; i++) {
        walkers_state[i].rng.seed(std::random_device{}() ^ (uint64_t)i);
        walkers_state[i].buffers = new Buffers();
        walkers_state[i].buffers->scalarStepsG = sharedScalarStepsG.data();
        walkers_state[i].buffers->scalarStepsH = sharedScalarStepsH.data();
        walkers_state[i].a = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
        walkers_state[i].b = uint256_t{};
        if (i % 2 != 0) {
            walkers_state[i].b.limbs[0] = 1;
        }

        walkers_state[i].walk_id = i;
        walkers_state[i].snapshot_steps = 0;
        memset(walkers_state[i].snapshot_x, 0, 32);
        memset(walkers_state[i].prev_x1, 0, 32);
        memset(walkers_state[i].prev_x2, 0, 32);

        uint64_t a_arr[4];
        uint256_to_uint64_array(a_arr, walkers_state[i].a);
        ECPointJacobian Ra;
        jacobianScalarMultPhi(&Ra, preCompG, preCompGphi, a_arr, windowSize);

        if (i % 2 == 0) {
            walkers_state[i].R = Ra;
        } else {
            pointAddJacobian(&walkers_state[i].R, &Ra, &target_affine_jac);
        }

        if (i % 32 == 0 || i == WALKERS - 1) {
            loading_bar(i + 1, WALKERS, header);
        }
    }

    auto last_print = std::chrono::steady_clock::now();

    auto worker = [&](int id_start, int id_end) {
        try {
            int local_count = id_end - id_start;
            std::vector<ECPointJacobian> jac_batch(local_count);
            std::vector<ECPointAffine> aff_batch(local_count);
            std::vector<uint64_t> scratch_prefix(local_count * 4);
            std::vector<uint64_t> scratch_inv(local_count * 4);

            for (int i = 0; i < local_count; i++) {
                jac_batch[i] = walkers_state[id_start + i].R;
            }
            batchJacobianToAffine(aff_batch.data(), jac_batch.data(), local_count, scratch_prefix.data(), scratch_inv.data());

            while (search_in_progress.load(std::memory_order_acquire)) {
                total_iters.fetch_add(local_count, std::memory_order_relaxed);
                for (int i = 0; i < local_count; i++) {
                    WalkState* w = &walkers_state[id_start + i];
                    memcpy(w->prev_x2, w->prev_x1, 32);
                    memcpy(w->prev_x1, aff_batch[i].x, 32);

                    uint32_t step_idx = get_step_idx(aff_batch[i].x, N_STEPS);
                    pointAddJacobian(&w->R, &w->R, &localStepTable[step_idx].point);
                    scalarAdd(w->a.limbs, w->a.limbs, localStepTable[step_idx].a.limbs);

                    if (compare_uint256(w->a, max_scalar) >= 0) {
                        w->a = mod_sub_N(w->a, max_scalar);
                        pointAddJacobian(&w->R, &w->R, &G_OFFSET);
                    }

                    jac_batch[i] = w->R;
                }
                batchJacobianToAffine(aff_batch.data(), jac_batch.data(), local_count, scratch_prefix.data(), scratch_inv.data());

                for (int i = 0; i < local_count; i++) {
                    WalkState* w = &walkers_state[id_start + i];
                    if (memcmp(aff_batch[i].x, w->snapshot_x, 32) == 0 || memcmp(aff_batch[i].x, w->prev_x1, 32) == 0 || memcmp(aff_batch[i].x, w->prev_x2, 32) == 0) {
                        MurmurHash3 hasher;
                        uint32_t idx = static_cast<uint32_t>(hasher(aff_batch[i].x[0] ^ 0xABCDEFULL) % N_STEPS);
                        pointAddJacobian(&w->R, &w->R, &localStepTable[idx].point);
                        scalarAdd(w->a.limbs, w->a.limbs, localStepTable[idx].a.limbs);

                        if (compare_uint256(w->a, max_scalar) >= 0) {
                            w->a = mod_sub_N(w->a, max_scalar);
                            pointAddJacobian(&w->R, &w->R, &G_OFFSET);
                        }

                        ECPointAffine aff_jump;
                        jacobianToAffine(&aff_jump, &w->R);
                        aff_batch[i] = aff_jump;
                        w->snapshot_steps = 0;
                        memset(w->snapshot_x, 0xFF, 32);
                        memset(w->prev_x1, 0xFE, 32);
                        memset(w->prev_x2, 0xFD, 32);
                        total_cycles.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }

                    w->snapshot_steps++;
                    if ((w->snapshot_steps & (w->snapshot_steps - 1)) == 0) {
                        memcpy(w->snapshot_x, aff_batch[i].x, 32);
                    }

                    if (!DP(aff_batch[i].x, DP_BITS)) continue;

                    w->snapshot_steps = 0;
                    try {
                        DPEntry found_dp;
                        bool cl = false;
                        dp_table.lazy_emplace_l(aff_batch[i].x[0], [&](auto& bucket) {
                            auto& dps = bucket.second;
                            for (const auto& entry : dps) {
                                if (entry.x == aff_batch[i].x[1]) {
                                    bool same_state = (compare_uint256(w->a, entry.a) == 0) && (compare_uint256(w->b, entry.b) == 0);
                                    if (!same_state) {
                                        found_dp = entry;
                                        cl = true;
                                    }
                                    return;
                                }
                            }
                            if(!cl) {
                                DPEntry new_entry;
                                new_entry.x = aff_batch[i].x[1];
                                new_entry.a = w->a;
                                new_entry.b = w->b;
                                dps.push_back(new_entry);
                            }
                        }, [&](auto bucket) {
                            DPEntry entry;
                            entry.x = aff_batch[i].x[1];
                            entry.a = w->a;
                            entry.b = w->b;
                            bucket(aff_batch[i].x[0], std::vector<DPEntry>{entry});
                        } );

                        if (cl) {
                            uint256_t da, db, inv_db;
                            if (compare_uint256(w->a, found_dp.a) >= 0) da = sub_uint256(w->a, found_dp.a);
                            else da = sub_uint256(N, sub_uint256(found_dp.a, w->a));
                            if (compare_uint256(found_dp.b, w->b) >= 0) db = sub_uint256(found_dp.b, w->b);
                            else db = sub_uint256(N, sub_uint256(w->b, found_dp.b));

                            if (!scalarIsZero(db.limbs)) {
                                inv_db = almostinverse(db, N);
                                uint64_t res_k[4];
                                scalarMul(res_k, da.limbs, inv_db.limbs);
                                unsigned char test_pub[33];
                                generatePublicKey(preCompG, preCompGphi, test_pub, res_k, windowSize);
                                if (memcmp(test_pub, target_pubkey.data(), 33) == 0) {
                                    memcpy(k.limbs, res_k, 32);
                                    search_in_progress.store(false, std::memory_order_release);
                                    break;
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << RED << "\n[Thread Error]: " << e.what() << RESET << std::endl;
                        search_in_progress.store(false);
                        return;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << RED << "\n[Thread Error]: " << e.what() << RESET << std::endl;
            search_in_progress.store(false);
        } catch (...) {
            std::cerr << RED << "\n[Thread Error]:" << RESET << std::endl;
            search_in_progress.store(false);
        }
    };

    std::thread progress_thread([&]() {
        const long double M = ldexpl(1.0L, key_range);
        while (search_in_progress.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_print >= std::chrono::seconds(10)) {
                long double k = (long double)total_iters.load(std::memory_order_relaxed);
                long double x = std::log2((k * k) / (2.0L * M));
                long double d = (x <= 1.0L) ? 0.0L : x;
                long double prob = (1.0L - expl(-d)) * 100.0L;
                std::cout << CYAN << "\033[3A\r" << "\033[2KTotal Ops/10s: " << RESET << GREEN << total_iters.load() << RESET << "\n" << CYAN << "\033[2KSelf-Collision Cycles: " << RESET << GREEN << total_cycles.load() << RESET << "\n" << CYAN << "\033[2KCollision Probability: " << RESET << GREEN << std::fixed << std::setprecision(8) << (prob) << "...%\n" << RESET << std::flush;
                last_print = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    std::vector<std::thread> threads;
    if (WALKERS == 0) {
        search_in_progress.store(false, std::memory_order_release);
        progress_thread.join();
        return k;
    }

    if (cores == 0) cores = 2;
    cores = std::min<unsigned int>(cores, WALKERS);
    int chunk = WALKERS / cores;

    for (unsigned int t = 0; t < cores; t++) {
        int start = t * chunk;
        int end = (t == cores - 1) ? WALKERS : start + chunk;
        threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }

    search_in_progress.store(false, std::memory_order_release);
    if(progress_thread.joinable()) progress_thread.join();

    for (auto& w : walkers_state) {
        if (w.buffers != nullptr) {
            delete w.buffers;
            w.buffers = nullptr;
        }
    }

    dp_table.clear();
    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\n" << std::endl;
    std::cout << CYAN << "Total duration: " << RESET << PINK << std::setw(2) << std::setfill('0') << duration.count() / 3600 << ":" << std::setw(2) << std::setfill('0') << (duration.count() % 3600) / 60 << ":" << std::setw(2) << std::setfill('0') << duration.count() % 60 << RESET;
    std::cout << "\n";

    return k;
}

void save_key(const std::string& pub_key_hex, const uint256_t& priv_key) {
    std::ofstream outfile("DISCRETE_LOGS_SOLVED", std::ios::app);
    if (outfile.is_open()) {
        outfile << pub_key_hex << " : " << uint256_to_hex(priv_key) << "\n";
        outfile.close();
        std::cout << ORANGE << "[INFO!] " << RESET << "Chave salva com sucesso em " << RESET << BLUE << "DISCRETE_LOGS_SOLVED" << RESET << std::endl;
    } else {
        std::cerr << RED << "[ERROR!] Nao foi possivel abrir o arquivo para salvar a chave!" << RESET << std::endl;
    }
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << CYAN << "Uso: " << RESET << argv[0] << PINK << " <Compressed Public Key(Hex)> " << "<Key Range(int)> " << "<Walkers(int)>" << RESET << std::endl;
        return 1;
    }

    init_secp256k1(std::stoi(argv[2]));

    std::string pub_key_hex(argv[1]);
    int key_range = std::stoi(argv[2]);
    int walkers = std::stoi(argv[3]);
    int dp = 0;

    if (argc >= 5) {
        try {
            dp = std::stoi(argv[4]);
        } catch (...) {
            std::cerr << RED << "Unknown error parsing arguments!" << RESET << std::endl;
        }
    }

    std::cout << BLUE << "---------------------------------------------------------------------------" << RESET << std::endl;

    if (dp <= 0) {
        std::cerr << ORANGE << "[INFO] " << RESET << GREEN << "Setting DP automatically..." << RESET << std::endl;
    	dp = (int)std::round(std::sqrt(key_range));
    }

    std::cout << ORANGE << "[INFO] " << RESET << GREEN << "Press 'Ctrl Z' to Quit\n" << RESET;
    std::cout << ORANGE << "[INFO] " << RESET << GREEN << "Auto Window-Size for secp256k1: " << RESET << PINK << windowSize << RESET << std::endl;
    std::cout << BLUE << "---------------------------------------------------------------------------" << RESET << std::endl;

    uint256_t found_key = prho(pub_key_hex, key_range, walkers, dp);

    std::cout << GREEN << "[SUCCESS!] " << RESET << "Collision Found!" << std::endl;

    unsigned char test_pub[33];
    auto target_pubkey = hex_to_bytes(pub_key_hex);
    generatePublicKey(preCompG, preCompGphi, test_pub, found_key.limbs, windowSize);

    if (memcmp(test_pub, target_pubkey.data(), 33) != 0) {
        std::cout << RED << "[ERROR!] Incorrect Public Key: " << RESET;
        for (int i = 0; i < 33; i++) printf("%02x", test_pub[i]);
        std::cout << std::dec << std::endl;
    }
    else
    {
        std::cout << GREEN << "[SUCCESS!] " << RESET << "Public Key Match: " << PINK;
        for (int i = 0; i < 33; i++) printf("%02x", test_pub[i]);
        std::cout << RESET << std::dec << std::endl;
    }

    save_key(pub_key_hex, found_key);

    std::cout << GREEN << "Private key found: " << RESET << gradient_zeros(uint256_to_hex(found_key), DARK_PINK, PINK) << std::endl;

    double key_val = 0;
    for (int i = 0; i < 4; i++) {
        key_val += (double)found_key.limbs[i] * std::pow(2.0, i * 64);
    }

    double range_start = std::pow(2.0, key_range - 1);
    double range_end = std::pow(2.0, key_range);
    double relative_pos = (key_val - range_start) / (range_end - range_start);
    double percentage = relative_pos * 100.0;

    std::cout << CYAN << "% of the Range: " << RESET << PINK << std::fixed << std::setprecision(2) << percentage << "%" << RESET << std::endl;

    delete[] preCompG;
    delete[] preCompGphi;
    delete[] preCompH;
    delete[] preCompHphi;
    delete[] jacNorm;
    delete[] jacNormH;
    delete[] jacEndo;
    delete[] jacEndoH;
    return 0;
}
