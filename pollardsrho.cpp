/******************************************************************************************************
 * This file is part of the Pollard's Rho distribution: (https://github.com/lucaselblanc/pollardsrho) *
 * Copyright (c) 2024, 2026 Lucas Leblanc.                                                            *
 * Distributed under the MIT software license, see the accompanying.                                  *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.                               *
 ******************************************************************************************************/

/*****************************************
 * Pollard's Rho Algorithm for SECP256K1 *
 * Written by Lucas Leblanc              *
******************************************/

/* --- AINDA EM TESTES --- */

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

const uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
const uint64_t SUB_P2[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
const uint256_t N = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
const uint256_t P = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };

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

size_t ram_size() {
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (size_t)pages * page_size;
}

int windowSize = 16; //Default value used only if getfcw() detection cannot access the processor for some reason, it can happen on different platforms like termux for example.

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

void getfcw() {
    int w = 4;
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
            //if (L == 2) l2Size = size;
            if (L == 3) l3Size = size;
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

void init_secp256k1() {
    getfcw();

    preCompG = new ECPointJacobian[1ULL << windowSize];
    preCompGphi = new ECPointJacobian[1ULL << windowSize];
    preCompH = new ECPointJacobian[1ULL << windowSize];
    preCompHphi = new ECPointJacobian[1ULL << windowSize]; //internal use in secp256k1
    jacNorm = new ECPointJacobian[windowSize]; //internal use in secp256k1
    jacNormH = new ECPointJacobian[windowSize]; //internal use in secp256k1
    jacEndo = new ECPointJacobian[windowSize]; //internal use in secp256k1
    jacEndoH = new ECPointJacobian[windowSize]; //internal use in secp256k1

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
            memcpy(&scratch_prefix[i * 4], &scratch_prefix[(i-1) * 4], 32);
        } else {
            modMulMontP(&scratch_prefix[i * 4], &scratch_prefix[(i-1) * 4], jac_in[i].Z);
        }
    }

    uint64_t total_inv[4];
    modExpMontP(total_inv, &scratch_prefix[(count-1) * 4], SUB_P2);

    uint64_t current_inv[4];
    memcpy(current_inv, total_inv, 32);

    for (int i = count - 1; i > 0; i--) {
        if (jacobianIsInfinity(&jac_in[i])) continue;

        modMulMontP(&scratch_inv[i * 4], current_inv, &scratch_prefix[(i-1) * 4]);
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
        uint64_t z2[4];
        uint64_t x_mont[4];
        modMulMontP(z2, &scratch_inv[i * 4], &scratch_inv[i * 4]);
        modMulMontP(x_mont, jac_in[i].X, z2);
        fromMontgomeryP(aff_out[i].x, x_mont);
        aff_out[i].infinity = 0;
    }
}

uint256_t prho(std::string target_pubkey_hex, int key_range, const int DP_BITS, bool NEGATION_MAP) {
    std::atomic<bool> search_in_progress(true);
    std::atomic<unsigned long long> total_iters{0};
    uint256_t k{};

    auto cores = std::thread::hardware_concurrency();
    auto start_time = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::tm start_tm{};
    localtime_r(&start_time_t, &start_tm);

    const int WALKERS = []() {
        size_t ram = ram_size() / (1024 * 1024 * 1024);
        if (ram >= 32) return 8192;
        if (ram >= 16) return 4096;
        if (ram >= 8)  return 2048;
        if (ram <= 4)  return 1024;
        return 512;
    }();

    const int N_STEPS = 2048;

    auto target_pubkey = hex_to_bytes(target_pubkey_hex);
    uint256_t min_scalar{}, max_scalar{};
    uint256_t mns{}, mxs{};
    {
        int limb_idx = key_range / 64;
        int bit_idx  = key_range % 64;
        min_scalar.limbs[limb_idx] = 1ULL << bit_idx;
        for (int i = 0; i < limb_idx; i++) {
            max_scalar.limbs[i] = 0xFFFFFFFFFFFFFFFFULL;
        }
        max_scalar.limbs[limb_idx] = (bit_idx == 63) ? 0xFFFFFFFFFFFFFFFFULL : (1ULL << (bit_idx + 1)) - 1;
        int mns_limb = (key_range - 1) / 64;
        int mns_bit  = (key_range - 1) % 64;
        mns.limbs[mns_limb] = 1ULL << mns_bit;
        for (int i = 0; i <= limb_idx; i++) {
            mxs.limbs[i] = max_scalar.limbs[i];
        }
        mxs.limbs[limb_idx] >>= 1; 
    }

    std::cout << "Started at: " << std::put_time(&start_tm, "%H:%M:%S") << std::endl;
    std::cout << "WALKERS: " << WALKERS << std::endl;
    std::cout << "DP BITS: " << DP_BITS << std::endl;
    NEGATION_MAP == 0 ? std::cout << "NEGATION MAP = false" << std::endl : std::cout << "NEGATION MAP = true" << std::endl;
    std::cout << "Key Range: " << (key_range) << std::endl;
    std::cout << "Min Range: " << uint256_to_hex(mns) << std::endl;
    std::cout << "Max Range: " << uint256_to_hex(mxs) << std::endl;
    std::cout << "\n\n";

    ECPointAffine target_affine{};
    ECPointJacobian target_affine_jac{};
    decompressPublicKey(&target_affine, target_pubkey.data());
    affineToJacobian(&target_affine_jac, &target_affine);
    initPreCompH(&target_affine_jac, windowSize);

    struct StepLocal {
        ECPointJacobian point;
        uint256_t a;
        uint256_t b;
    };

    std::vector<StepLocal> localStepTable(N_STEPS);
    std::mt19937_64 salt(target_affine.x[0]);

    uint256_t stepSize = {};

    stepSize.limbs[(key_range / 2) / 64] = 1ULL << ((key_range / 2) % 64);

    uint256_t step_min = {};
    uint256_t step_max = stepSize;

    for (int i = 0; i < N_STEPS; i++) {
        localStepTable[i].a = rng_mersenne_twister(step_min, step_max, key_range / 2, salt);
        localStepTable[i].b = rng_mersenne_twister(step_min, step_max, key_range / 2, salt);

        uint64_t a_tmp[4], b_tmp[4];
        uint256_to_uint64_array(a_tmp, localStepTable[i].a);
        uint256_to_uint64_array(b_tmp, localStepTable[i].b);
        ECPointJacobian aiG, biH;
        jacobianScalarMult(&aiG, preCompG, a_tmp, windowSize);
        jacobianScalarMult(&biH, preCompH, b_tmp, windowSize);
        pointAddJacobian(&localStepTable[i].point, &aiG, &biH);
        ECPointAffine aff_step;
        jacobianToAffine(&aff_step, &localStepTable[i].point);
        affineToJacobian(&localStepTable[i].point, &aff_step);
    }

    phmap::parallel_flat_hash_map<uint64_t,
    std::vector<DPEntry>, MurmurHash3,
    phmap::priv::hash_default_eq<uint64_t>,
    std::allocator<std::pair<const uint64_t,
    std::vector<DPEntry>>>, 4,
    std::mutex > dp_table;

    std::vector<WalkState> walkers_state(WALKERS);
    std::vector<uint64_t> sharedScalarStepsG((1ULL << windowSize) * 4);
    std::vector<uint64_t> sharedScalarStepsH((1ULL << windowSize) * 4);
    initScalarSteps(sharedScalarStepsG.data(), windowSize);
    initScalarSteps(sharedScalarStepsH.data(), windowSize);

    for (int i = 0; i < WALKERS; i++) {
        walkers_state[i].rng.seed(std::random_device{}() ^ (uint64_t)i);
        walkers_state[i].buffers = new Buffers();
        walkers_state[i].buffers->scalarStepsG = sharedScalarStepsG.data();
        walkers_state[i].buffers->scalarStepsH = sharedScalarStepsH.data();
        walkers_state[i].a = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
        walkers_state[i].b = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
        walkers_state[i].walk_id = i;
        walkers_state[i].snapshot_steps = 0;
        memset(walkers_state[i].snapshot_x, 0, 32);

        if(NEGATION_MAP) {
            memset(walkers_state[i].prev_x1, 0, 32);
            memset(walkers_state[i].prev_x2, 0, 32);
        }

        uint64_t a_arr[4], b_arr[4];
        uint256_to_uint64_array(a_arr, walkers_state[i].a);
        uint256_to_uint64_array(b_arr, walkers_state[i].b);

        ECPointJacobian Ra, Rb;
        jacobianScalarMult(&Ra, preCompG, a_arr, windowSize);
        jacobianScalarMult(&Rb, preCompH, b_arr, windowSize);
        pointAddJacobian(&walkers_state[i].R, &Ra, &Rb);
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

                    if(NEGATION_MAP) {
                        memcpy(w->prev_x2, w->prev_x1, 32);
                        memcpy(w->prev_x1, aff_batch[i].x, 32);
                    }

                    uint32_t step_idx = get_step_idx(aff_batch[i].x, N_STEPS);
                    pointAddJacobian(&w->R, &w->R, &localStepTable[step_idx].point);
                    scalarAdd(w->a.limbs, w->a.limbs, localStepTable[step_idx].a.limbs);
                    scalarAdd(w->b.limbs, w->b.limbs, localStepTable[step_idx].b.limbs);
                    if (compare_uint256(w->a, N) >= 0) w->a = sub_uint256(w->a, N);
                    if (compare_uint256(w->b, N) >= 0) w->b = sub_uint256(w->b, N);
                    jac_batch[i] = w->R;
                }

                batchJacobianToAffine(aff_batch.data(), jac_batch.data(), local_count, scratch_prefix.data(), scratch_inv.data());

                for (int i = 0; i < local_count; i++) {
                    WalkState* w = &walkers_state[id_start + i];

                    if(NEGATION_MAP) {
                        if (aff_batch[i].infinity == 0) {
                            uint256_t y_val;
                            memcpy(y_val.limbs, aff_batch[i].y, 32);
                            uint256_t p_minus_y = sub_uint256(P, y_val);

                            if (compare_uint256(y_val, p_minus_y) > 0) {
                                memcpy(aff_batch[i].y, p_minus_y.limbs, 32);

                                if ((w->a.limbs[0] | w->a.limbs[1] | w->a.limbs[2] | w->a.limbs[3]) != 0) { w->a = sub_uint256(N, w->a); }
                                if ((w->b.limbs[0] | w->b.limbs[1] | w->b.limbs[2] | w->b.limbs[3]) != 0) { w->b = sub_uint256(N, w->b); }

                                uint256_t jac_y;
                                memcpy(jac_y.limbs, w->R.Y, 32);
                                uint256_t p_minus_jac_y = sub_uint256(P, jac_y);
                                memcpy(w->R.Y, p_minus_jac_y.limbs, 32);
                                jac_batch[i] = w->R;
                            }
                        }
                    }

                    if (memcmp(aff_batch[i].x, w->snapshot_x, 32) == 0 ||
                       (NEGATION_MAP ? (memcmp(aff_batch[i].x, w->prev_x1, 32) == 0 ||
                        memcmp(aff_batch[i].x, w->prev_x2, 32) == 0) : false)) {
                        w->a = rng_mersenne_twister(min_scalar, max_scalar, key_range, w->rng);
                        w->b = rng_mersenne_twister(min_scalar, max_scalar, key_range, w->rng);

                        uint64_t a_arr[4], b_arr[4];
                        uint256_to_uint64_array(a_arr, w->a);
                        uint256_to_uint64_array(b_arr, w->b);

                        ECPointJacobian Ra, Rb;
                        jacobianScalarMult(&Ra, preCompG, a_arr, windowSize);
                        jacobianScalarMult(&Rb, preCompH, b_arr, windowSize);
                        pointAddJacobian(&w->R, &Ra, &Rb);

                        w->snapshot_steps = 0;
                        memset(w->snapshot_x, 0, 32);
                        ECPointAffine aff_respawn;
                        jacobianToAffine(&aff_respawn, &w->R);
                        memcpy(aff_batch[i].x, aff_respawn.x, 32);
                        std::cout << "\033[33mWarning: Self-Collision Cycle Detected!\033[0m" << std::endl;
                        std::cout << "\033[33mThe walker " << w->walk_id << " was killed by hitting his own tail!\033[0m" << std::endl;
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

                        dp_table.lazy_emplace_l(aff_batch[i].x[0],
                            [&](auto& bucket) {
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

                                if(!cl)
                                {
                                    DPEntry new_entry;
                                    new_entry.x = aff_batch[i].x[1];
                                    new_entry.a = w->a; new_entry.b = w->b;
                                    dps.push_back(new_entry);
                                }
                            },
                            [&](auto bucket) {
                                DPEntry entry;
                                entry.x = aff_batch[i].x[1];
                                entry.a = w->a; entry.b = w->b;
                                bucket(aff_batch[i].x[0], std::vector<DPEntry>{entry});
                            }
                        );

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
                        std::cerr << "\n[Thread Error]: " << e.what() << std::endl;
                        search_in_progress.store(false);
                        return;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "\n[Thread Error]: " << e.what() << std::endl;
            search_in_progress.store(false);
        } catch (...) {
            std::cerr << "\n[Thread Error]:" << std::endl;
            search_in_progress.store(false);
        }
    };

    std::thread progress_thread([&]() {
        const long double M = ldexpl(1.0L, key_range);
        while (search_in_progress.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_print >= std::chrono::seconds(10)) {
                long double k_val = (long double)total_iters.load(std::memory_order_relaxed);
                long double x = (k_val * k_val) / (2.0L * M);
                long double prob = 1.0L - expl(-x);
                prob *= 100.0L;

                std::cout << "\033[2A\r\033[J"
                << "Total Iterations: " << total_iters.load() << "\n"
                << "Collision Probability: "
                << std::fixed << std::setprecision(8)
                << (prob) << "...%\n"
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
        std::cerr << "Uso: " << argv[0] << " <Compressed Public Key(Hex)> <Key Range(int)>" << std::endl;
        return 1;
    }

    init_secp256k1();

    std::string pub_key_hex(argv[1]);
    int key_range = std::stoi(argv[2]);
    int dp = 0;
    bool NEGATION_MAP = false;

    if (argc >= 4) {
        try {
            dp = std::stoi(argv[3]);
            if (argc >= 5) {
                if (std::string(argv[4]) == "NEGATION_MAP_TRUE") {
                    NEGATION_MAP = true;
                }
            }
        } catch (...) {
            std::cerr << "\033[31m[!] Unknown error parsing arguments!\033[0m" << std::endl;
        }
    }

    if (dp <= 0) {
        std::cerr << "Setting DP automatically..." << std::endl;
        dp = (int)std::floor(key_range / 2.0);
    }

    std::cout << "Press 'Ctrl Z' to Quit\n";
    std::cout << "Auto Window-Size for secp256k1: " << windowSize << std::endl;

    uint256_t found_key = prho(pub_key_hex, key_range, dp, NEGATION_MAP);

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