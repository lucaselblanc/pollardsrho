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

/* --- ENDOMORPHISM VERSION (ϕ) --- */

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
constexpr uint256_t LAMBDA = { 0xDF02967C1B23BD72ULL, 0xA5261C028812645AULL, 0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL };
constexpr uint256_t BETA = { 0xB315ECECBB640683ULL, 0x9CF0497512F58995ULL, 0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL };
constexpr uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
constexpr uint64_t SUB2_FP[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
constexpr uint64_t BETA_MONT[4] = { 0xACFAA7CF3D9205F3ULL, 0x03FDE1630E28013DULL, 0xF8E98978D02E3905ULL, 0x7A4A36AEBCBB3D53ULL };
constexpr uint64_t HALF_P[4] = { 0xFFFFFFFF7FFFFE18ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL };

struct Buffers {
    uint64_t* scalarStepsG;
    uint64_t* scalarStepsH;
};

struct WalkState {
    uint256_t a1, a2, b1, b2;
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
    uint256_t a1, a2, b1, b2;
    uint64_t x;
};

void loading_bar(uint64_t current, uint64_t total, const std::string& label) {
    if (total == 0) return;
    float percent = (float)current / total;
    int barWidth = 40;
    int pos = barWidth * percent;
    std::cout << "\033[2A\r\033[J" << label << "\n";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "█";
        else std::cout << "▒";
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

int compare_uint256(const uint256_t& a, const uint256_t& b) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return 1;
        if (a.limbs[i] < b.limbs[i]) return -1;
    }
    return 0;
}

int compare_uint256_arrays(const uint64_t* a, const uint64_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
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
            std::cout << "Warning: " << e.what() << std::endl;
            continue;
        }
        catch(const std::out_of_range& e) {
            std::cout << "Warning: " << e.what() << std::endl;
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
    preCompH = new ECPointJacobian[1ULL << windowSize];
    preCompHphi = new ECPointJacobian[1ULL << windowSize];
    jacNorm = new ECPointJacobian[windowSize]; //internal use in secp256k1
    jacNormH = new ECPointJacobian[windowSize]; //internal use in secp256k1
    jacEndo = new ECPointJacobian[windowSize]; //internal use in secp256k1
    jacEndoH = new ECPointJacobian[windowSize]; //internal use in secp256k1

    initPreCompG(windowSize);
}

uint256_t rng_mersenne_twister(const uint256_t& min_scalar, const uint256_t& max_scalar, int key_range, std::mt19937_64& rng) {
    uint256_t r{};
    uint256_t range = mod_sub_N(max_scalar, min_scalar);
    uint256_t one = {1, 0, 0, 0};
    range = mod_add_N(range, one);

    int max_limb_idx = (key_range - 1) / 64;
    int bits_in_last_limb = key_range % 64;

    do {
        for (int i = 0; i < 4; i++) r.limbs[i] = rng();
        for (int i = max_limb_idx + 1; i < 4; i++) r.limbs[i] = 0;
        if (bits_in_last_limb > 0) r.limbs[max_limb_idx] &= (1ULL << bits_in_last_limb) - 1;
    } while (compare_uint256(r, range) >= 0);

    return mod_add_N(r, min_scalar);
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
        modMulMontP(z3, z2, z_inv);
        modMulMontP(tmp_mont, jac_in[i].X, z2);
        fromMontgomeryP(aff_out[i].x, tmp_mont);
        modMulMontP(tmp_mont, jac_in[i].Y, z3);
        fromMontgomeryP(aff_out[i].y, tmp_mont);

        aff_out[i].infinity = 0;
    }
}

void normalize_oeq6(ECPointAffine& aff, uint256_t& a1, uint256_t& a2, uint256_t& b1, uint256_t& b2) {
    if (aff.infinity) return;

    uint64_t x_coords[3][4];
    memcpy(x_coords[0], aff.x, 32);

    uint64_t x_mont[4];
    toMontgomeryP(x_mont, aff.x);
    modMulMontP(x_mont, x_mont, BETA_MONT);
    fromMontgomeryP(x_coords[1], x_mont);
    modMulMontP(x_mont, x_mont, BETA_MONT);
    fromMontgomeryP(x_coords[2], x_mont);

    int best_idx = 0;
    for (int i = 1; i < 3; i++) {
        if (compare_uint256_arrays(x_coords[i], x_coords[best_idx]) < 0) {
            best_idx = i;
        }
    }

    if (best_idx == 1) {
        uint256_t old_a1 = a1;
        uint256_t old_b1 = b1;
        a1 = mod_neg_N(a2);
        a2 = mod_sub_N(old_a1, a2);
        b1 = mod_neg_N(b2);
        b2 = mod_sub_N(old_b1, b2);
    }
    else if (best_idx == 2) {
        uint256_t old_a1 = a1;
        uint256_t old_b1 = b1;
        a1 = mod_sub_N(a2, old_a1);
        a2 = mod_neg_N(old_a1);
        b1 = mod_sub_N(b2, old_b1);
        b2 = mod_neg_N(old_b1);
    }

    memcpy(aff.x, x_coords[best_idx], 32);

    if (compare_uint256_arrays(aff.y, HALF_P) > 0) {
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t diff = P.limbs[i] - aff.y[i] - borrow;
            borrow = (P.limbs[i] < aff.y[i]) || (borrow && P.limbs[i] == aff.y[i]);
            aff.y[i] = diff;
        }

        a1 = mod_neg_N(a1); a2 = mod_neg_N(a2);
        b1 = mod_neg_N(b1); b2 = mod_neg_N(b2);
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
    uint256_t mns{}, mxs{};
    {
        int limb_idx = key_range / 64;
        int bit_idx  = key_range % 64;
        min_scalar.limbs[limb_idx] = 1ULL << bit_idx;
        for (int i = 0; i < limb_idx; i++) max_scalar.limbs[i] = 0xFFFFFFFFFFFFFFFFULL;
        max_scalar.limbs[limb_idx] = (bit_idx == 63) ? 0xFFFFFFFFFFFFFFFFULL : (1ULL << (bit_idx + 1)) - 1;
        int mns_limb = (key_range - 1) / 64;
        int mns_bit  = (key_range - 1) % 64;
        mns.limbs[mns_limb] = 1ULL << mns_bit;
        for (int i = 0; i <= limb_idx; i++) mxs.limbs[i] = max_scalar.limbs[i];
        mxs.limbs[limb_idx] >>= 1;
    }

    std::cout << "Started at: " << std::put_time(&start_tm, "%H:%M:%S") << std::endl;
    std::cout << "WALKERS: " << WALKERS << std::endl;
    std::cout << "DP BITS: " << DP_BITS << std::endl;
    std::cout << "Key Range: " << (key_range) << std::endl;
    std::cout << "Min Range: " << uint256_to_hex(mns) << std::endl;
    std::cout << "Max Range: " << uint256_to_hex(mxs) << std::endl;
    std::cout << "\n\n\n\n";

    ECPointAffine target_affine{};
    ECPointJacobian target_affine_jac{};
    decompressPublicKey(&target_affine, target_pubkey.data());
    affineToJacobian(&target_affine_jac, &target_affine);
    initPreCompH(&target_affine_jac, windowSize);

    struct StepLocal {
        ECPointJacobian point;
        uint256_t a1, a2, b1, b2;
    };

    std::vector<StepLocal> localStepTable(N_STEPS);
    std::mt19937_64 salt(target_affine.x[0]);

    uint256_t stepSize = {};
    stepSize.limbs[(key_range / 2) / 64] = 1ULL << ((key_range / 2) % 64);
    uint256_t step_min = {};
    uint256_t step_max = stepSize;

    for (int i = 0; i < N_STEPS; i++) {
        localStepTable[i].a1 = rng_mersenne_twister(step_min, step_max, key_range / 2, salt);
        localStepTable[i].a2 = {0};
        localStepTable[i].b1 = rng_mersenne_twister(step_min, step_max, key_range / 2, salt);
        localStepTable[i].b2 = {0};

        uint64_t a_tmp[4], b_tmp[4];
        uint256_to_uint64_array(a_tmp, localStepTable[i].a1);
        uint256_to_uint64_array(b_tmp, localStepTable[i].b1);
        ECPointJacobian aiG, biH;
        jacobianScalarMultPhi(&aiG, preCompG, preCompGphi, a_tmp, windowSize);
        jacobianScalarMultPhi(&biH, preCompH, preCompHphi, b_tmp, windowSize);
        pointAddJacobian(&localStepTable[i].point, &aiG, &biH);
        ECPointAffine aff_step;
        jacobianToAffine(&aff_step, &localStepTable[i].point);
        affineToJacobian(&localStepTable[i].point, &aff_step);
    }

    phmap::parallel_flat_hash_map<uint64_t, std::vector<DPEntry>, MurmurHash3, 
    phmap::priv::hash_default_eq<uint64_t>, std::allocator<std::pair<const uint64_t, std::vector<DPEntry>>>, 8, std::mutex > dp_table;

    std::vector<WalkState> walkers_state(WALKERS);
    std::vector<uint64_t> sharedScalarStepsG((1ULL << windowSize) * 4);
    std::vector<uint64_t> sharedScalarStepsH((1ULL << windowSize) * 4);
    initScalarSteps(sharedScalarStepsG.data(), windowSize);
    initScalarSteps(sharedScalarStepsH.data(), windowSize);

    std::string header = "\033[32m[!] Loading Walkers... \033[0m";

    for (int i = 0; i < WALKERS; i++) {
        walkers_state[i].rng.seed(std::random_device{}() ^ (uint64_t)i);
        walkers_state[i].buffers = new Buffers();
        walkers_state[i].buffers->scalarStepsG = sharedScalarStepsG.data();
        walkers_state[i].buffers->scalarStepsH = sharedScalarStepsH.data();
        walkers_state[i].a1 = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
        walkers_state[i].a2 = {0};
        walkers_state[i].b1 = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
        walkers_state[i].b2 = {0};
        walkers_state[i].walk_id = i;
        walkers_state[i].snapshot_steps = 0;
        memset(walkers_state[i].snapshot_x, 0, 32);
        memset(walkers_state[i].prev_x1, 0, 32);
        memset(walkers_state[i].prev_x2, 0, 32);

        uint64_t a_arr[4], b_arr[4];
        uint256_to_uint64_array(a_arr, walkers_state[i].a1);
        uint256_to_uint64_array(b_arr, walkers_state[i].b1);

        ECPointJacobian Ra, Rb;
        jacobianScalarMultPhi(&Ra, preCompG, preCompGphi, a_arr, windowSize);
        jacobianScalarMultPhi(&Rb, preCompH, preCompHphi, b_arr, windowSize);
        pointAddJacobian(&walkers_state[i].R, &Ra, &Rb);

        if (i % 32 == 0 || i == WALKERS - 1) loading_bar(i + 1, WALKERS, header);
    }

    auto last_print = std::chrono::steady_clock::now();
    auto worker = [&](int id_start, int id_end) {
        int local_count = id_end - id_start;
        std::vector<ECPointJacobian> jac_batch(local_count);
        std::vector<ECPointAffine> aff_batch(local_count);
        std::vector<uint64_t> scratch_prefix(local_count * 4), scratch_inv(local_count * 4);

        for (int i = 0; i < local_count; i++) jac_batch[i] = walkers_state[id_start + i].R;
        batchJacobianToAffine(aff_batch.data(), jac_batch.data(), local_count, scratch_prefix.data(), scratch_inv.data());

        while (search_in_progress.load(std::memory_order_acquire)) {
            total_iters.fetch_add(local_count, std::memory_order_relaxed);

            for (int i = 0; i < local_count; i++) {
                WalkState* w = &walkers_state[id_start + i];
                ECPointAffine normalized_aff = aff_batch[i];
                uint256_t temp_a1 = w->a1, temp_a2 = w->a2, temp_b1 = w->b1, temp_b2 = w->b2;

                normalize_oeq6(normalized_aff, temp_a1, temp_a2, temp_b1, temp_b2);

                uint32_t step_idx = get_step_idx(normalized_aff.x, N_STEPS);

                pointAddJacobian(&w->R, &w->R, &localStepTable[step_idx].point);
                w->a1 = mod_add_N(w->a1, localStepTable[step_idx].a1);
                w->a2 = mod_add_N(w->a2, localStepTable[step_idx].a2);
                w->b1 = mod_add_N(w->b1, localStepTable[step_idx].b1);
                w->b2 = mod_add_N(w->b2, localStepTable[step_idx].b2);

                jac_batch[i] = w->R;
            }

            batchJacobianToAffine(aff_batch.data(), jac_batch.data(), local_count, scratch_prefix.data(), scratch_inv.data());

            for (int i = 0; i < local_count; i++) {
                WalkState* w = &walkers_state[id_start + i];
                ECPointAffine aff_temp = aff_batch[i];
                uint256_t a1_t = w->a1, a2_t = w->a2, b1_t = w->b1, b2_t = w->b2;

                normalize_oeq6(aff_temp, a1_t, a2_t, b1_t, b2_t);

                if (memcmp(aff_temp.x, w->snapshot_x, 32) == 0 ||
                    memcmp(aff_temp.x, w->prev_x1, 32) == 0 ||
                    memcmp(aff_temp.x, w->prev_x2, 32) == 0) {

                    MurmurHash3 hasher;
                    uint32_t idx = static_cast<uint32_t>(hasher(aff_temp.x[0] ^ 0xABCDEFULL) % N_STEPS);

                    pointAddJacobian(&w->R, &w->R, &localStepTable[idx].point);
                    w->a1 = mod_add_N(w->a1, localStepTable[idx].a1);
                    w->a2 = mod_add_N(w->a2, localStepTable[idx].a2);
                    w->b1 = mod_add_N(w->b1, localStepTable[idx].b1);
                    w->b2 = mod_add_N(w->b2, localStepTable[idx].b2);

                    ECPointAffine aff_jump;
                    jacobianToAffine(&aff_jump, &w->R);

                    aff_batch[i] = aff_jump;

                    w->snapshot_steps = 0;
                    memset(w->snapshot_x, 0xFF, 32);
                    memset(w->prev_x1, 0xFE, 32);
                    memset(w->prev_x2, 0xFD, 32);

                    total_cycles.fetch_add(1, std::memory_order_relaxed);
                    jac_batch[i] = w->R;
                    continue;
                }

                w->snapshot_steps++;
                if ((w->snapshot_steps & (w->snapshot_steps - 1)) == 0) {
                    memcpy(w->snapshot_x, aff_temp.x, 32);
                }

                if (!DP(aff_temp.x, DP_BITS)) {
                    jac_batch[i] = w->R;
                    continue;
                }

                w->snapshot_steps = 0;

                DPEntry found_dp;
                bool cl = false;

                dp_table.lazy_emplace_l(aff_temp.x[0],
                    [&](auto& bucket) {
                        for (const auto& entry : bucket.second) {
                            if (entry.x == aff_temp.x[1]) {
                                if (compare_uint256(a1_t, entry.a1) != 0 || compare_uint256(a2_t, entry.a2) != 0) {
                                    found_dp = entry; cl = true;
                                }
                                return;
                            }
                        }
                        if(!cl) bucket.second.push_back({a1_t, a2_t, b1_t, b2_t, aff_temp.x[1]});
                    },
                    [&](auto bucket) {
                        bucket(aff_temp.x[0], std::vector<DPEntry>{{a1_t, a2_t, b1_t, b2_t, aff_temp.x[1]}});
                    }
                );

                if (cl) {
                    uint64_t res_a[4];
                    scalarMul(res_a, a2_t.limbs, LAMBDA.limbs);
                    uint256_t A_w = mod_add_N(a1_t, {res_a[0], res_a[1], res_a[2], res_a[3]});
                    scalarMul(res_a, b2_t.limbs, LAMBDA.limbs);
                    uint256_t B_w = mod_add_N(b1_t, {res_a[0], res_a[1], res_a[2], res_a[3]});
                    scalarMul(res_a, found_dp.a2.limbs, LAMBDA.limbs);
                    uint256_t A_dp = mod_add_N(found_dp.a1, {res_a[0], res_a[1], res_a[2], res_a[3]});
                    scalarMul(res_a, found_dp.b2.limbs, LAMBDA.limbs);
                    uint256_t B_dp = mod_add_N(found_dp.b1, {res_a[0], res_a[1], res_a[2], res_a[3]});
                    uint256_t da = mod_sub_N(A_w, A_dp);
                    uint256_t db = mod_sub_N(B_dp, B_w);

                    if (!scalarIsZero(db.limbs)) {
                        uint256_t inv_db = almostinverse(db, N);
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

                jac_batch[i] = w->R;
            }
        }
    };

    std::thread progress_thread([&]() {
        const long double M = ldexpl(1.0L, key_range);
        while (search_in_progress.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_print >= std::chrono::seconds(10)) {
                long double k_val = (long double)total_iters.load(std::memory_order_relaxed);
                long double x = (k_val * k_val) / (2.0L * M) / (WALKERS * (1 << DP_BITS));
                long double d = (x <= 1.0L) ? 0.0L : std::floor(std::log2(x));
                long double prob = (1.0L - expl(-d)) * 100.0L;

                std::cout << "\033[3A\r"
                << "\033[2KTotal Ops/10s: " << total_iters.load() << "\n"
                << "\033[2KSelf-Collision Cycles:  " << total_cycles.load() << "\n"
                << "\033[2KCollision Probability: " << std::fixed << std::setprecision(8) << (prob) << "...%\n"
                << std::flush;
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
        int end   = (t == cores - 1) ? WALKERS : start + chunk;
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
    std::cout << "Total duration: " << std::setw(2) << std::setfill('0') << duration.count() / 3600 << ":"
    << std::setw(2) << std::setfill('0') << (duration.count() % 3600) / 60 << ":"
    << std::setw(2) << std::setfill('0') << duration.count() % 60 << std::endl;

    return k;
}

void save_key(const std::string& pub_key_hex, const uint256_t& priv_key) {
    std::ofstream outfile("DISCRETE_LOGS_SOLVED", std::ios::app);
    if (outfile.is_open()) {
        outfile << pub_key_hex << " : " << uint256_to_hex(priv_key) << "\n";
        outfile.close();
        std::cout << "\033[35m[INFO!] Chave salva com sucesso em DISCRETE_LOGS_SOLVED\033[0m" << std::endl;
    } else {
        std::cerr << "\033[31m[ERROR!] Nao foi possivel abrir o arquivo para salvar a chave!\033[0m" << std::endl;
    }
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <Compressed Public Key(Hex)> <Key Range(int)> <Walkers(int)>" << std::endl;
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
            std::cerr << "\033[31m[!] Unknown error parsing arguments!\033[0m" << std::endl;
        }
    }

    if (dp <= 0) {
        std::cerr << "Setting DP automatically..." << std::endl;
        dp = (int)std::round(std::sqrt(key_range));
    }

    std::cout << "Press 'Ctrl Z' to Quit\n";
    std::cout << "Auto Window-Size for secp256k1: " << windowSize << std::endl;

    uint256_t found_key = prho(pub_key_hex, key_range, walkers, dp);

    std::cout << "\n\n\033[32m[SUCCESS!] Collision Found!\033[0m" << std::endl;

    unsigned char test_pub[33];
    auto target_pubkey = hex_to_bytes(pub_key_hex);
    generatePublicKey(preCompG, preCompGphi, test_pub, found_key.limbs, windowSize);

    if (memcmp(test_pub, target_pubkey.data(), 33) != 0) {
        std::cout << "\033[31m[ERROR!] Incorrect Public Key:\033[0m" << std::endl;
        for (int i = 0; i < 33; i++) printf("%02x", test_pub[i]);
        std::cout << std::dec << std::endl;
    }
    else
    {
        std::cout << "\033[32m[SUCCESS!] Public Key Match:\033[0m" << std::endl;
        for (int i = 0; i < 33; i++) printf("%02x", test_pub[i]);
        std::cout << std::dec << std::endl;
    }

    save_key(pub_key_hex, found_key);

    std::cout << "Private key found: " << uint256_to_hex(found_key) << std::endl;

    double key_val = 0;
    for (int i = 0; i < 4; i++) {
        key_val += (double)found_key.limbs[i] * std::pow(2.0, i * 64);
    }

    double range_start = std::pow(2.0, key_range - 1);
    double range_end = std::pow(2.0, key_range);
    double relative_pos = (key_val - range_start) / (range_end - range_start);
    double percentage = relative_pos * 100.0;

    std::cout << "% of the Range: " << std::fixed << std::setprecision(2) << percentage << "%" << std::endl;

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
