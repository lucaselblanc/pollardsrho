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

const uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
const uint64_t P_CONST_MINUS_2[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
const uint64_t P[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
const uint256_t N = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };

struct Buffers {
    ECPointJacobian* d_R;
    ECPointJacobian* d_G;
    ECPointJacobian* d_H;
    uint64_t* d_k;
    uint64_t* scalarStepsG;
    uint64_t* scalarStepsH;
};

struct WalkState {
    ECPointJacobian R;
    uint256_t a, b;
    Buffers* buffers;
    uint32_t walk_id;
    bool negate;
    std::mt19937_64 rng;
};

struct DPEntry {
    uint64_t x[4];
    uint256_t a;
    uint256_t b;
    int walk_id;
};

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
    preCompHphi = new ECPointJacobian[1ULL << windowSize];
    jacNorm = new ECPointJacobian[windowSize];
    jacNormH = new ECPointJacobian[windowSize];
    jacEndo = new ECPointJacobian[windowSize];
    jacEndoH = new ECPointJacobian[windowSize];

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

uint32_t get_step_idx(const uint64_t* x) {
    //MurmurHash3<Avalanche constants>
    uint64_t h = x[0] ^ (x[1] << 1) ^ (x[2] << 2) ^ (x[3] << 3);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdLLU;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53LLU;
    h ^= h >> 33;
    return (uint32_t)h % 32;
}

bool DP(const uint64_t* affine_x, int DP_BITS) {
    return (affine_x[0] & ((1ULL << DP_BITS) - 1)) == 0;
}

void batchJacobianToAffine(ECPointAffine* aff_out, const ECPointJacobian* jac_in, int count) {
    if (count <= 0) return;
    std::vector<uint64_t[4]> scratch_prefix(count);
    std::vector<uint64_t[4]> scratch_inv(count); 

    if (jacobianIsInfinity(&jac_in[0])) {
        memcpy(scratch_prefix[0], ONE_MONT, 32);
    } else {
        memcpy(scratch_prefix[0], jac_in[0].Z, 32);
    }

    for (int i = 1; i < count; i++) {
        if (jacobianIsInfinity(&jac_in[i])) {
            memcpy(scratch_prefix[i], scratch_prefix[i-1], 32);
        } else {
            modMulMontP(scratch_prefix[i], scratch_prefix[i-1], jac_in[i].Z);
        }
    }

    uint64_t total_inv[4];
    modExpMontP(total_inv, scratch_prefix[count-1], P_CONST_MINUS_2);

    uint64_t current_inv[4];
    memcpy(current_inv, total_inv, 32);

    for (int i = count - 1; i > 0; i--) {
        if (jacobianIsInfinity(&jac_in[i])) {
            continue;
        }

        modMulMontP(scratch_inv[i], current_inv, scratch_prefix[i-1]);
        modMulMontP(current_inv, current_inv, jac_in[i].Z);
    }

    if (!jacobianIsInfinity(&jac_in[0])) {
        memcpy(scratch_inv[0], current_inv, 32);
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

        modMulMontP(z2, scratch_inv[i], scratch_inv[i]);
        modMulMontP(x_mont, jac_in[i].X, z2);
        fromMontgomeryP(aff_out[i].x, x_mont);

        aff_out[i].infinity = 0;
    }
}

uint256_t prho(std::string target_pubkey_hex, int key_range, int walkers, bool test_mode) {
    std::atomic<bool> search_in_progress(true);
    std::atomic<unsigned long long> total_iters{0};
    std::mutex dp_mutexes[32];
    uint256_t k{};

    auto start_time = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::tm start_tm{};
    localtime_r(&start_time_t, &start_tm);

    const int DP_BITS = 8;
    const int LOCAL_NUM_STEPS = 32; //sweet spot, increasing this doesn't bring any advantages.

    auto target_pubkey = hex_to_bytes(target_pubkey_hex);

    uint256_t min_scalar{}, max_scalar{};
    {
        int limb_index = (key_range - 1) / 64; 
        int bit_in_limb = (key_range - 1) % 64;
        min_scalar.limbs[limb_index] = 1ULL << bit_in_limb;
        for (int i = 0; i < limb_index; i++) {
            max_scalar.limbs[i] = 0xFFFFFFFFFFFFFFFFULL;
        }
        max_scalar.limbs[limb_index] = (1ULL << (bit_in_limb + 1)) - 1;
    }

    std::cout << "Started at: " << std::put_time(&start_tm, "%H:%M:%S") << std::endl;
    if(test_mode) { std::cout << "Test Mode: True" << std::endl; }
    else          { std::cout << "Test Mode: False" << std::endl; }
    std::cout << "DP BITS: " << DP_BITS << std::endl;
    std::cout << "Key Range: " << key_range << std::endl;
    std::cout << "Min Range: " << uint256_to_hex(min_scalar) << std::endl;
    std::cout << "Max Range: " << uint256_to_hex(max_scalar) << std::endl;

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
    std::vector<StepLocal> localStepTable(LOCAL_NUM_STEPS);
    std::mt19937_64 salt(time(NULL));
    for (int i = 0; i < LOCAL_NUM_STEPS; i++) {
        localStepTable[i].a = rng_mersenne_twister(min_scalar, max_scalar, key_range, salt);
        localStepTable[i].b = rng_mersenne_twister(min_scalar, max_scalar, key_range, salt);

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

    std::unordered_map<uint64_t, std::vector<DPEntry>> dp_table;
    std::vector<WalkState> walkers_state(walkers);
    std::vector<uint64_t> sharedScalarStepsG((1ULL << windowSize) * 4);
    std::vector<uint64_t> sharedScalarStepsH((1ULL << windowSize) * 4);
    initScalarSteps(sharedScalarStepsG.data(), windowSize);
    initScalarSteps(sharedScalarStepsH.data(), windowSize);

    for (int i = 0; i < walkers; i++) {
        walkers_state[i].rng.seed(time(NULL) + i);
        walkers_state[i].buffers = new Buffers();
        walkers_state[i].buffers->scalarStepsG = sharedScalarStepsG.data();
        walkers_state[i].buffers->scalarStepsH = sharedScalarStepsH.data();
        walkers_state[i].a = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
        walkers_state[i].b = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
        walkers_state[i].walk_id = i;

        uint64_t a_arr[4], b_arr[4];
        uint256_to_uint64_array(a_arr, walkers_state[i].a);
        uint256_to_uint64_array(b_arr, walkers_state[i].b);

        ECPointJacobian Ra, Rb;
        jacobianScalarMult(&Ra, preCompG, a_arr, windowSize);
        jacobianScalarMult(&Rb, preCompH, b_arr, windowSize);
        pointAddJacobian(&walkers_state[i].R, &Ra, &Rb);
    }

    // This doesn't reduce the search to half the key range, it just centers the walkers before starting the random walk
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
            jacobianScalarMult(&Ra, preCompG, a_arr, windowSize);
            jacobianScalarMult(&Rb, preCompH, b_arr, windowSize);
            pointAddJacobian(&w->R, &Ra, &Rb);
        }
    }

    auto last_print = std::chrono::steady_clock::now();

    if (test_mode) {
        uint256_t min_jump{}, max_jump{};
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
            jacobianScalarMult(&Ra, preCompG, a_arr, windowSize);
            jacobianScalarMult(&Rb, preCompH, b_arr, windowSize);
            pointAddJacobian(&w->R, &Ra, &Rb);
        }
    }

    auto worker = [&](int id_start, int id_end) {
        int local_count = id_end - id_start;

        std::vector<ECPointJacobian> jac_batch(local_count);
        std::vector<ECPointAffine> aff_batch(local_count);

        while (search_in_progress.load(std::memory_order_acquire)) {
            for (int i = 0; i < local_count; i++) {
                WalkState* w = &walkers_state[id_start + i];
                total_iters.fetch_add(1, std::memory_order_relaxed);
                uint32_t step_idx = get_step_idx(aff_batch[i].x);
                pointAddJacobian(&w->R, &w->R, &localStepTable[step_idx].point);
                scalarAdd(w->a.limbs, w->a.limbs, localStepTable[step_idx].a.limbs);
                if (compare_uint256(w->a, N) >= 0) w->a = sub_uint256(w->a, N);
                scalarAdd(w->b.limbs, w->b.limbs, localStepTable[step_idx].b.limbs);
                if (compare_uint256(w->b, N) >= 0) w->b = sub_uint256(w->b, N);
                jac_batch[i] = w->R;
            }

            batchJacobianToAffine(aff_batch.data(), jac_batch.data(), local_count);

            for (int i = 0; i < local_count; i++) {
                if (!DP(aff_batch[i].x, DP_BITS)) continue;

                WalkState* w = &walkers_state[id_start + i];
                uint64_t table_idx = aff_batch[i].x[0]; 

                std::lock_guard<std::mutex> lock(dp_mutexes[table_idx % 32]);
                auto& dps = dp_table[table_idx];

                for (const auto& entry : dps) {
                    if (entry.walk_id == w->walk_id){
                        std::cout << "A walker hit himself in a cycle: " << w->walk_id << std::endl;
                        continue;
                    }

                    if (memcmp(aff_batch[i].x, entry.x, 32) == 0) {
                        if (compare_uint256(w->a, entry.a) == 0 && compare_uint256(w->b, entry.b) == 0) continue;

                        uint256_t da, db, inv_db;

                        if (compare_uint256(w->a, entry.a) >= 0) {
                            da = sub_uint256(w->a, entry.a);
                        }
                        else {
                            da = sub_uint256(N, sub_uint256(entry.a, w->a));
                        }

                        if (compare_uint256(entry.b, w->b) >= 0) {
                            db = sub_uint256(entry.b, w->b);
                        }
                        else {
                            db = sub_uint256(N, sub_uint256(w->b, entry.b));
                        }

                        if (!scalarIsZero(db.limbs)) {
                            auto valid_key = [&](uint64_t* k_limbs) -> bool {
                                unsigned char test_pub[33];
                                generatePublicKey(preCompG, preCompGphi, test_pub, k_limbs, windowSize);
                                return memcmp(test_pub, target_pubkey.data(), 33) == 0;
                            };

                            inv_db = almostinverse(db, N);
                            uint64_t res_k[4];
                            scalarMul(res_k, da.limbs, inv_db.limbs);
                            memcpy(k.limbs, res_k, 32);

                            if(!valid_key(k.limbs)){ continue; };

                            search_in_progress.store(false, std::memory_order_release);
                            return;
                        }
                    }
                }

                DPEntry new_entry;
                memcpy(new_entry.x, aff_batch[i].x, 32);
                new_entry.a = w->a;
                new_entry.b = w->b;
                new_entry.walk_id = w->walk_id;
                dps.push_back(new_entry);
            }
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

                std::cout << "\r\033[K"
                << "Total Iterations: " << total_iters.load() << "\n"
                << "Collision Probability: "
                << std::fixed << std::setprecision(8)
                << (prob) << "...%\n"
                << std::flush;

                last_print = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    std::vector<std::thread> threads;
    if (walkers == 0) {
        search_in_progress.store(false);
        progress_thread.join();
        return k;
    }
    unsigned int cores = std::thread::hardware_concurrency();
    if (cores == 0) cores = 2;
    cores = std::min<unsigned int>(cores, walkers);
    int chunk = walkers / cores;

    for (unsigned int t = 0; t < cores; t++) {
        int start = t * chunk;
        int end   = (t == cores - 1) ? walkers : start + chunk;
        threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) th.join();
    for (auto& w : walkers_state) delete w.buffers;

    search_in_progress.store(false, std::memory_order_release);
    if(progress_thread.joinable()) progress_thread.join();

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
        std::cout << "\033[34m[INFO!] Chave salva com sucesso em DISCRETE_LOGS_SOLVED\033[0m" << std::endl;
    } else {
        std::cerr << "\033[31m[ERROR!] Nao foi possivel abrir o arquivo para salvar a chave!\033[0m" << std::endl;
    }
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

    unsigned int cores = std::thread::hardware_concurrency();
    int walkers = (cores <= 8) ? 2048 : 4096;
    uint256_t found_key = prho(pub_key_hex, key_range, walkers, test_mode);

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

    std::cout << "Chave privada encontrada: " << uint256_to_hex(found_key) << std::endl;
    return 0;
}
