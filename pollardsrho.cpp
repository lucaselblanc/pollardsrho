#include "ec.h"
#include <gmpxx.h>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <sys/sysinfo.h>
#include <boost/multiprecision/cpp_int.hpp>

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

using namespace boost::multiprecision;

ECPoint G;
ECPoint H;
mpz_t P, GX, GY, N;

void init_secp256k1() {
    mpz_inits(P, N, GX, GY, NULL);

    mpz_set_str(P, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16);
    mpz_set_str(N, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16);
    mpz_set_str(GX, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16);
    mpz_set_str(GY, "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16);

    point_init(&G);
    mpz_set(G.x, GX);
    mpz_set(G.y, GY);
    G.infinity = 0;

    point_init(&H);
    mpz_set(H.x, GX);
    mpz_set(H.y, GY);
    H.infinity = 0;
}

class PKG {
    std::mt19937_64 gen;
    uint64_t min_low, max_low, min_mid_low, max_mid_low;
    uint64_t min_mid_high, max_mid_high, min_high, max_high;

    public: PKG(uint256_t min_scalar, uint256_t max_scalar) : gen(std::random_device{}()),
    min_low(static_cast<uint64_t>(min_scalar & 0xFFFFFFFFFFFFFFFF)),
    max_low(static_cast<uint64_t>(max_scalar & 0xFFFFFFFFFFFFFFFF)),
    min_mid_low(static_cast<uint64_t>((min_scalar >> 64) & 0xFFFFFFFFFFFFFFFF)),
    max_mid_low(static_cast<uint64_t>((max_scalar >> 64) & 0xFFFFFFFFFFFFFFFF)),
    min_mid_high(static_cast<uint64_t>((min_scalar >> 128) & 0xFFFFFFFFFFFFFFFF)),
    max_mid_high(static_cast<uint64_t>((max_scalar >> 128) & 0xFFFFFFFFFFFFFFFF)),
    min_high(static_cast<uint64_t>((min_scalar >> 192) & 0xFFFFFFFFFFFFFFFF)),
    max_high(static_cast<uint64_t>((max_scalar >> 192) & 0xFFFFFFFFFFFFFFFF)) {}

    uint256_t generate() {
        uint64_t low = std::uniform_int_distribution<uint64_t>(min_low, max_low)(gen);
        uint64_t mid_low = std::uniform_int_distribution<uint64_t>(min_mid_low, max_mid_low)(gen);
        uint64_t mid_high = std::uniform_int_distribution<uint64_t>(min_mid_high, max_mid_high)(gen);
        uint64_t high = std::uniform_int_distribution<uint64_t>(min_high, max_high)(gen);

        return (static_cast<uint256_t>(high) << 192) | (static_cast<uint256_t>(mid_high) << 128) | (static_cast<uint256_t>(mid_low) << 64) | low;
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

void precompute_jumps(int key_range) {
    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, std::random_device{}());

    precomputed_jumps.resize(NUM_JUMPS);

    for (size_t i = 0; i < NUM_JUMPS; ++i) {
        mpz_t r;
        mpz_init(r);
        mpz_urandomb(r, rng, key_range);

        char r_char[78];
        mpz_get_str(r_char, 10, r);

        precomputed_jumps[i] = uint256_t(r_char);

        mpz_clear(r);

        if (i % (NUM_JUMPS / 100) == 0) {
            int progress = static_cast<int>((i * 101) / NUM_JUMPS);
            std::cout << "\rLoading Jumps: " << progress << "% of total jumps: " << NUM_JUMPS << " Using: " << TOTAL_RAM << " GB ram - " << std::flush;
        }
    }

    gmp_randclear(rng);

    precomputed_done = true;
}

std::atomic<size_t> jump(0);

uint256_t f(ECPoint& R, uint256_t k, int key_range) {
    const uint256_t mask = (uint256_t(1) << key_range) - 1;

    unsigned long op = mpz_fdiv_ui(R.x, 3UL);
    size_t idx = static_cast<size_t>(mpz_fdiv_ui(R.x, (unsigned long)NUM_JUMPS));

    switch (op) {
        case 0: // class 1
            point_add(&R, &R, &G, P);
            k = (k + precomputed_jumps[idx]) & mask;
            break;

        case 1: // class 2
            point_add(&R, &R, &H, P);
            k = (k + precomputed_jumps[idx]) & mask;
            break;

        default: // class 3
            point_double(&R, &R, P);
            k = (k << 1) & mask;
            break;
    }

    return k;
}

/*-- The algorithm is more efficient for ranges >= 57 bits, as the search is more distributed over larger ranges --*/
uint256_t prho(std::string target_pubkey_hex, int key_range, int hares, bool test_mode) {

    auto uint_256_to_hex = [](const uint256_t& value) -> std::string {
        std::ostringstream oss;
        oss << std::setw(64) << std::setfill('0') << std::hex << value;
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

    auto uint256_to_mpz = [](mpz_t private_key, uint256_t value) {
        /*Retornando 0 pois futuramente esse lambda será removido,
        por favor comente a linha: //mpz_init_set_ui(private_key, 0); 
        e descomente o restante se for utilizar a função prho,
        a função uint256_to_mpz é ineficiente e deve ser
        substituída em breve, assim como o restante das funções gmp */

        mpz_init_set_ui(private_key, 0);

        /*
        const int limb_bits = sizeof(mp_limb_t) * 8;
        const int limb_bytes = sizeof(mp_limb_t);

        std::vector<uint8_t> bytes;
        export_bits(value, std::back_inserter(bytes), 8);

        size_t num_limbs = (bytes.size() + limb_bytes - 1) / limb_bytes;

        mpz_init2(private_key, 256);

        mp_limb_t* limbs = mpz_limbs_write(private_key, num_limbs);
        mp_limb_t carry = 0;

        size_t byte_index = 0;

        for (size_t i = 0; i < num_limbs; ++i) {

            mp_limb_t temp = carry;

            for (size_t j = 0; j < limb_bytes && byte_index < bytes.size(); ++j, ++byte_index) {
                temp = (temp << 8) | bytes[byte_index];
            }

            limbs[i] = temp;
            carry = temp >> (limb_bits - 8);
        }

        if (carry) {
            limbs[num_limbs - 1] = carry;
            num_limbs++;
        }

        private_key->_mp_size = num_limbs;
        */
    };

    auto start_time = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::tm start_tm = *std::localtime(&start_time_t);

    std::cout << "Started at: " << std::put_time(&start_tm, "%H:%M:%S") << std::endl;
    if(test_mode) { std::cout << "Test Mode: True" << std::endl; }
    else          { std::cout << "Test Mode: False" << std::endl; }

    uint256_t min_scalar = (uint256_t(1) << (key_range - 1));  
    uint256_t max_scalar = (uint256_t(1) << key_range) - 1;

    std::cout << "key_range: " << key_range << std::endl;
    std::cout << "min_range: " << uint_256_to_hex(min_scalar) << std::endl;
    std::cout << "max_range: " << uint_256_to_hex(max_scalar) << std::endl;

    std::atomic<unsigned int> keys_ps{0};
    std::atomic<bool> search_in_progress(true);
    std::mutex pgrs;

    std::string P_key;
    uint256_t p_key = uint256_t(0);
    uint256_t found_key = uint256_t(0);

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
        point_init(&hare_states[i].R);
        mpz_set(hare_states[i].R.x, G.x);
        mpz_set(hare_states[i].R.y, G.y);
        hare_states[i].R.infinity = 0;
        hare_states[i].speed = (i == 0) ? 1 : (i + 1);
        hare_states[i].k1 = pkg.generate();
        hare_states[i].k2 = pkg.generate();
    }

    //unsigned int threads = std::thread::hardware_concurrency();
    //omp_set_num_threads(threads);

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

    //#pragma omp parallel
    //{
        try {
            while (search_in_progress.load()) {

                //#pragma omp for
                for (int i = 0; i < hares; ++i) {

                    HareState& hare = hare_states[i];

                    hare.speed = (i == 0) ? 1 : (i + 1);

                    current_key = hare.k1;
                    p_key = current_key;

                    keys_ps.fetch_add(1, std::memory_order_relaxed);

                    // Test mode is recommended for ranges <= 20, larger ranges may cause hares to enter infinite loops/cycles.
                    if (test_mode)
                    {
                        /*-- Warning linear search, prone to infinite loops/cycles! --*/
                        hare.k1 = (hare.k1 + hare.speed) % (uint256_t(1) << 64);
                        hare.k2 = (hare.k2 + hare.speed) % (uint256_t(1) << 64);
                    } else {
                        // Pollard's rho random walk
                        hare.k1 = f(hare.R, hare.k1, key_range);
                        hare.k2 = f(hare.R, hare.k2, key_range);
                    }

                    if (hare.k1 < min_scalar) hare.k1 += min_scalar;
                    if (hare.k2 < min_scalar) hare.k2 += min_scalar;

                    ECPoint pub1, pub2;
                    point_init(&pub1);
                    point_init(&pub2);

                    mpz_t k1_mpz, k2_mpz;
                    mpz_init(k1_mpz);
                    mpz_init(k2_mpz);
                    uint256_to_mpz(k1_mpz, hare.k1);
                    uint256_to_mpz(k2_mpz, hare.k2);

                    scalar_mult(&pub1, k1_mpz, &hare.R, P);
                    scalar_mult(&pub2, k2_mpz, &hare.R, P);

                    char compressed1[33], compressed2[33];
                    get_compressed_public_key(compressed1, &pub1);
                    get_compressed_public_key(compressed2, &pub2);

                    std::string current_pubkey_hex_R = bytes_to_hex((unsigned char*)compressed1, 33);
                    std::string current_pubkey_hex_R1 = bytes_to_hex((unsigned char*)compressed2, 33);
                    P_key = current_pubkey_hex_R;

                    int LSB = 5;
                    auto DP = [LSB](const ECPoint& point) -> bool {
                        for (int i = 0; i < LSB; i++) {
                        if (mpz_tstbit(point.x, i) != 0) return false;
                        }
                        return true;
                    };

                    // Caso onde x é par:
                    if (DP(pub1) && DP(pub2) && !test_mode) {

                        //#pragma omp critical
                        //{
                            if (mpz_cmp(pub1.x, pub2.x) == 0 && mpz_cmp(pub1.y, pub2.y) == 0 && hare.k1 != hare.k2) {

                                /*
                                    Calcular a diferença (d) entre os pontos pubkey1 e pubkey2:
                                    d = k1 - k2 tal que P1 = k1 * G e P2 = k2 * G

                                    Verificar se: (d * G ≡ 0), caso verdadeiro: found_key = (k2 + d) % n;
                                */

                                mpz_class N_mpz(N);
                                uint256_t N_uint(N_mpz.get_str());

                                uint256_t key = DP(pub1) ? hare.k1 : hare.k2;
                                uint256_t d = (hare.k1 >= hare.k2) ? (hare.k1 - hare.k2) : (N_uint - (hare.k2 - hare.k1));

                                mpz_t d_mpz;
                                mpz_init(d_mpz);
                                uint256_to_mpz(d_mpz, d);

                                ECPoint test_point;
                                scalar_mult(&test_point, d_mpz, &G, P);

                                if (test_point.infinity == 1) {

                                    found_key = (key + d) % N_uint;

                                    ECPoint verify_point;
                                    mpz_t found_key_mpz;
                                    mpz_init(found_key_mpz);
                                    uint256_to_mpz(found_key_mpz, found_key);

                                    scalar_mult(&verify_point, found_key_mpz, &G, P);

                                    char compressed[33];
                                    get_compressed_public_key(compressed, &verify_point);

                                    if (memcmp(compressed, target_pubkey.data(), 33) == 0) {

                                        std::cout << "\033[33mDP detected for hare " << i << " at k1: " << uint_256_to_hex(hare.k1) << "\033[0m" << std::endl;
                                        std::cout << "\033[33mDP detected for hare " << i << " at k2: " << uint_256_to_hex(hare.k2) << "\033[0m" << std::endl;
                                        std::cout << "A multiplicação satisfaz a equação (d * G ≡ 0)" << std::endl;
                                        std::cout << "Private Key Found: " << uint_256_to_hex(found_key) << std::endl;

                                        search_in_progress.store(false);
                                    }
                                    else
                                    {
                                        //std::cout << "K não corresponde: " << uint_256_to_hex(found_key) << std::endl;
                                    }

                                    mpz_clear(d_mpz);
                                    mpz_clear(found_key_mpz);
                                }
                            }
                        //}
                    }

                    if (memcmp(compressed1, target_pubkey.data(), 33) == 0 ||
                        memcmp(compressed2, target_pubkey.data(), 33) == 0) {

                        //#pragma omp critical
                        //{
                            found_key = (current_pubkey_hex_R == target_pubkey_hex) ? hare.k1 : hare.k2;
                            std::cout << "\033[32mPrivate key found!\033[0m" << std::endl;
                            std::cout << "Private Key: " << uint_256_to_hex(found_key) << std::endl;

                            search_in_progress.store(false);
                        //}
                    }

                    mpz_clears(k1_mpz, k2_mpz, NULL);
                }

                //if(!search_in_progress.load()) {
                    //#pragma omp cancel parallel
                    //#pragma omp barrier
                //}
            }
        }
        catch (const std::exception& e) {
            //#pragma omp cancel parallel
            //#pragma omp barrier
            //#pragma omp critical
            std::cerr << "Exception: " << e.what() << std::endl;
        }
    //}

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

    uint256_t found_key = prho(pub_key_hex, key_range, 512, test_mode);

    auto uint_256_to_hex = [](const uint256_t& value) -> std::string {
        std::ostringstream oss;
        oss << std::setw(64) << std::setfill('0') << std::hex << value;
        return oss.str();
    };

    std::cout << "Chave privada encontrada: " << uint_256_to_hex(found_key) << std::endl;

    return 0;
}

