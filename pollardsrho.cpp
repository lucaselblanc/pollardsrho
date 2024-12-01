#include <secp256k1.h>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_set>
#include <omp.h>
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

std::string uint256ToHex(const uint256_t& value) {
    std::ostringstream oss;
    oss << std::setw(64) << std::setfill('0') << std::hex << value;
    return oss.str();
}

uint256_t bytesToUint256(const unsigned char* bytes) {
    uint256_t value = 0;
    for (size_t i = 0; i < 32; ++i) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

std::vector<unsigned char> hexToBytes(const std::string& hex) {
    std::vector<unsigned char> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        unsigned char byte = (unsigned char) std::stoi(hex.substr(i, 2), nullptr, 16);
        bytes.push_back(byte);
    }
    return bytes;
}

std::string bytesToHex(const unsigned char* bytes, size_t length) {
    std::string hex_str;
    hex_str.reserve(length * 2);
    const char* hex_chars = "0123456789abcdef";
    for (size_t i = 0; i < length; ++i) {
        hex_str.push_back(hex_chars[bytes[i] >> 4]);
        hex_str.push_back(hex_chars[bytes[i] & 0x0F]);
    }
    return hex_str;
}

std::tuple<std::string, std::string, std::string> privateKeyToPublicKey(

    /*Gargalos principais da função:
    secp256k1_ec_pubkey_create(ctx, &pubkey, private_key)
    secp256k1_ec_pubkey_serialize(ctx, serialized_G, &serialized_G_len, &G, SECP256K1_EC_COMPRESSED)
    secp256k1_ec_pubkey_tweak_mul(ctx, &result_pubkey, private_key.data()

    Para melhorar a eficiência do algoritmo, essas funções da secp256k1 deveriam ser escritas manualmente
    https://github.com/JeanLucPons/Kangaroo/tree/master/SECPK1 */

    const std::string& private_key_hex, secp256k1_context* ctx, const secp256k1_pubkey& G) {

    if (private_key_hex.size() != 64) {
        throw std::runtime_error("Invalid private key size!");
    }

    std::array<unsigned char, 32> private_key{};
    for (size_t i = 0; i < 32; ++i) {
        private_key[i] = std::stoi(private_key_hex.substr(i * 2, 2), nullptr, 16);
    }

    secp256k1_pubkey result_pubkey = G;
    if (!secp256k1_ec_pubkey_tweak_mul(ctx, &result_pubkey, private_key.data())) {
        throw std::runtime_error("Error multiplying public key with private key!");
    }

    if (!secp256k1_ec_pubkey_create(ctx, &result_pubkey, private_key.data())) {
        throw std::runtime_error("Error creating public key!");
    }

    std::array<unsigned char, 33> pubkey_compact{};
    size_t pubkey_len = pubkey_compact.size();
    secp256k1_ec_pubkey_serialize(ctx, pubkey_compact.data(), &pubkey_len, &result_pubkey, SECP256K1_EC_COMPRESSED);

    std::string compressed_key_hex = bytesToHex(pubkey_compact.data(), pubkey_len);

    unsigned char pubkey_full[65];
    size_t pubkey_full_len = sizeof(pubkey_full);
    secp256k1_ec_pubkey_serialize(ctx, pubkey_full, &pubkey_full_len, &result_pubkey, SECP256K1_EC_UNCOMPRESSED);

    std::string x_hex = bytesToHex(pubkey_full + 1, 32);
    std::string y_hex = bytesToHex(pubkey_full + 33, 32);

    return std::make_tuple(compressed_key_hex, x_hex, y_hex);
}

/* 
Não utilizado na implementação, apenas para testes simples:
https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm */
int64_t modular_inverse(int64_t a, int64_t m) {

    std::cout << "Base: " << a << ", Module: " << m << std::endl;

    if (m <= 0 || a <= 0) return 0;

    if (m == 1) return 1;

    if (std::gcd(a, m) != 1) {
        return 0;
    }

    int64_t m0 = m, t, q;
    int64_t x0 = 0, x1 = 1;

    a = (a % m + m) % m;

    while (a > 1) {
        q = a / m;
        t = m;
        m = a % m;
        a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }

    x1 = (x1 + m0) % m0;

    return x1;
}

class PrivateKeyGen {
    std::mt19937_64 gen;
    uint64_t min_low, max_low, min_high, max_high;

    public: PrivateKeyGen(uint256_t min_scalar, uint256_t max_scalar)
    : gen(std::random_device{}()),
    min_low(static_cast<uint64_t>(min_scalar & 0xFFFFFFFFFFFFFFFF)),
    max_low(static_cast<uint64_t>(max_scalar & 0xFFFFFFFFFFFFFFFF)),
    min_high(static_cast<uint64_t>((min_scalar >> 64) & 0xFFFFFFFFFFFFFFFF)),
    max_high(static_cast<uint64_t>((max_scalar >> 64) & 0xFFFFFFFFFFFFFFFF)) {}

    uint256_t generate() {
        uint64_t low = std::uniform_int_distribution<uint64_t>(min_low, max_low)(gen);
        uint64_t high = std::uniform_int_distribution<uint64_t>(min_high, max_high)(gen);
        return (static_cast<uint256_t>(high) << 64) | low;
    }
};

/******************
 **[MULTI-THREADS]*
 ******************/
//https://en.wikipedia.org/wiki/Pollard%27s_rho_algorithm#Further_reading
uint256_t prho(secp256k1_context* ctx, const secp256k1_pubkey& G, const secp256k1_pubkey& target_pubkey, int key_range, int hares) {
    uint256_t min_scalar = (uint256_t(1) << (key_range - 1));  
    uint256_t max_scalar = (uint256_t(1) << key_range) - 1;
    uint256_t keys_ps;

    //SECP256K1 n
    uint256_t n = uint256_t("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

    std::cout << "key_range: " << key_range << std::endl;
    std::cout << "min_range: " << uint256ToHex(min_scalar) << std::endl;
    std::cout << "max_range: " << uint256ToHex(max_scalar) << std::endl;

    std::atomic<bool> search_in_progress(true);
    std::mutex pgrs;

    std::string P_key;
    uint256_t p_key = uint256_t(0);
    uint256_t found_key = uint256_t(0);

    std::thread log_thread([&]() {
    try {
         for (uint256_t j = uint256_t(0); j < max_scalar; ++j) {
             std::this_thread::sleep_for(std::chrono::seconds(10));
             std::lock_guard<std::mutex> lock(pgrs);
             if(search_in_progress){        
             std::cout << "\rCurrent private key: " << uint256ToHex(p_key) << std::endl;
             std::cout << "\rLast tested public key: " << P_key << std::endl;
             std::cout << "\rTotal keys tested: " << keys_ps << std::endl; }
         }
    } catch (const std::exception& e) {
         std::cerr << "Error in log_thread: " << e.what() << std::endl;
    }});

    unsigned int threads = std::thread::hardware_concurrency();
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        std::string current_pubkey_hex_R;
        std::string current_pubkey_hex_R1;
        std::string compressed_key_hex_R, compressed_key_hex_R1;
        std::string x_hex_R, y_hex_R, x_hex_R1, y_hex_R1;

        PrivateKeyGen pkg(min_scalar, max_scalar);
        uint256_t current_key = pkg.generate();

        struct HareState {
            uint256_t k1, k2;
            secp256k1_pubkey R;//G Point
            std::unordered_set<std::string> cicle;
            int speed;
        };

        std::vector<HareState> hare_states(hares);

        for (int k = 0; k < hares; ++k)
        {
            HareState& hare = hare_states[k];
            std::memcpy(&hare.R, &G, sizeof(secp256k1_pubkey));
        }

        unsigned char target_pubkey_serialized[33];
        size_t target_pubkey_len = sizeof(target_pubkey_serialized);
        if (!secp256k1_ec_pubkey_serialize(ctx, target_pubkey_serialized, &target_pubkey_len, &target_pubkey, SECP256K1_EC_COMPRESSED)) {
             std::cerr << "Failed to serialize target public key!" << std::endl;
             throw std::runtime_error("Error serializing target public key!");
        }

        try {

            for (uint256_t j = uint256_t(0); j < max_scalar; ++j) {

                for (int i = 0; i < hares; ++i) {

                    HareState& hare = hare_states[i];
                    hare.k1 = pkg.generate();
                    hare.k2 = pkg.generate();
                    hare.speed = (i == 0) ? 1 : (i + 1);
                    current_key = hare.k1;
                    p_key = current_key;
                    keys_ps++;

                    hare.k1 = (hare.k1 + hare.speed) % (uint256_t(1) << 64);
                    hare.k2 = (hare.k2 + hare.speed) % (uint256_t(1) << 64);

                    if (hare.k1 < min_scalar) hare.k1 += min_scalar;
                    if (hare.k2 < min_scalar) hare.k2 += min_scalar;

                    std::string k1_hex = uint256ToHex(hare.k1);
                    std::tie(compressed_key_hex_R, x_hex_R, y_hex_R) = privateKeyToPublicKey(k1_hex, ctx, hare.R);
                    current_pubkey_hex_R = compressed_key_hex_R;
                    P_key = current_pubkey_hex_R;

                    std::string k2_hex = uint256ToHex(hare.k2);
                    std::tie(compressed_key_hex_R1, x_hex_R1, y_hex_R1) = privateKeyToPublicKey(k2_hex, ctx, hare.R);
                    current_pubkey_hex_R1 = compressed_key_hex_R1;

                    /* Verificar colisões não triviais utilizando ciclos de floyd:
                    https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare */
                    if (hare.cicle.find(current_pubkey_hex_R) != hare.cicle.end() && 
                        hare.cicle.find(current_pubkey_hex_R1) != hare.cicle.end()) {

                        /*
                           Calcular a diferença (d) entre os pontos pubkey1 e pubkey2:
                           d = k1 - k2 tal que P1 = k1 * G e P2 = k2 * G
  
                           Verificar se: (d * G ≡ 0), caso verdadeiro: found_key = (k2 + d) % n;
                        */

                        std::vector<unsigned char> pubkey_bytes_R = hexToBytes(current_pubkey_hex_R);
                        std::vector<unsigned char> pubkey_bytes_R1 = hexToBytes(current_pubkey_hex_R1);

                        secp256k1_pubkey pubkey1, pubkey2;

                        if (!secp256k1_ec_pubkey_parse(ctx, &pubkey1, pubkey_bytes_R.data(), pubkey_bytes_R.size())) {}
                        if (!secp256k1_ec_pubkey_parse(ctx, &pubkey2, pubkey_bytes_R1.data(), pubkey_bytes_R1.size())) {}

                        unsigned char serialized_pubkey1[33], serialized_pubkey2[33];
                        size_t len = 33;
                        if (!secp256k1_ec_pubkey_serialize(ctx, serialized_pubkey1, &len, &pubkey1, SECP256K1_EC_COMPRESSED)) {}
                        if (!secp256k1_ec_pubkey_serialize(ctx, serialized_pubkey2, &len, &pubkey2, SECP256K1_EC_COMPRESSED)) {}

                        uint256_t p1 = bytesToUint256(serialized_pubkey1);
                        uint256_t p2 = bytesToUint256(serialized_pubkey2);
                        uint256_t d = (p1 - p2) % n;
                        uint256_t found_key = (hare.k2 + d) % n;

                        unsigned char d_bytes[32] = {0};
                        for (int i = 0; i < 32; i++) {
                            d_bytes[31 - i] = static_cast<unsigned char>((d >> (8 * i)).convert_to<uint64_t>() & 0xFF);
                        }

                        secp256k1_pubkey point = G;
                        if (secp256k1_ec_pubkey_tweak_mul(ctx, &point, d_bytes)) {
                            unsigned char point_bytes[33];
                            secp256k1_ec_pubkey_serialize(ctx, point_bytes, &len, &point, SECP256K1_EC_COMPRESSED);

                            std::vector<unsigned char> zeros(len, 0);
                            if(memcmp(point_bytes, zeros.data(), len) == 0)
                            {
                                 std::cout << "\033[33mCycle detected for hare " << i << " at k1: " << uint256ToHex(hare.k1) << "\033[0m" << std::endl;
                                 std::cout << "\033[33mCycle detected for hare " << i << " at k2: " << uint256ToHex(hare.k2) << "\033[0m" << std::endl;
                                 std::cout << "A multiplicação satisfaz a equação (d * G ≡ 0)" << std::endl;
                                 std::cout << "Private Key Found: " << uint256ToHex(found_key) << std::endl;

                                 #pragma omp critical
                                 search_in_progress.store(false);
                            }
                            else
                            {
                                std::cout << "\033[33mCycle detected for hare " << i << " at k1: " << uint256ToHex(hare.k1) << "\033[0m" << std::endl;
                                std::cout << "\033[33mCycle detected for hare " << i << " at k2: " << uint256ToHex(hare.k2) << "\033[0m" << std::endl;
                                std::cout << "A multiplicação não satisfaz a equação (d * G ≡ 0)" << std::endl;
                                std::cout << "Invalid Key Found: " << uint256ToHex(found_key) << std::endl;
                            }
                        }
                    } else {
                        hare.cicle.insert(current_pubkey_hex_R);
                        hare.cicle.insert(current_pubkey_hex_R1);
                    }

                    if (current_pubkey_hex_R == bytesToHex(target_pubkey_serialized, target_pubkey_len)
                        || current_pubkey_hex_R1 == bytesToHex(target_pubkey_serialized, target_pubkey_len)) {

                        std::cout << "\033[32mPrivate key found by hare " << i << "!\033[0m" << std::endl;

                        if (compressed_key_hex_R == bytesToHex(target_pubkey_serialized, target_pubkey_len)) {
                            found_key = hare.k1;
                            std::cout << "Private Key Found: " << uint256ToHex(found_key) << std::endl;
                        }
                        else if (compressed_key_hex_R1 == bytesToHex(target_pubkey_serialized, target_pubkey_len)) {
                            found_key = hare.k2;
                            std::cout << "Private Key Found: " << uint256ToHex(found_key) << std::endl;
                        }

                        #pragma omp critical
                        search_in_progress.store(false);
                    }
                }

                if(!search_in_progress.load())
                {
                     #pragma omp cancel parallel
                     #pragma omp barrier
                     
                     #pragma omp critical
                     if (ctx) {
                         secp256k1_context_destroy(ctx);
                         ctx = nullptr;
                     }

                     exit(0);
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in thread: " << e.what() << std::endl;

            #pragma omp cancel parallel
            #pragma omp barrier

            #pragma omp critical
            if (ctx) {
                secp256k1_context_destroy(ctx);
                ctx = nullptr;
            }

            exit(0);
        }
    }

    return found_key;
}

int main(int argc, char* argv[]) {
    //Test
    // int64_t a = 2, m = 5;
    // int64_t result = modular_inverse(a, m);
    // if(result != 0)
    // {
    //     std::cout << "A inversa de " << a << " modulo " << m << " = " << result << std::endl;
    // } else {   
    //     std::cout << "Não existe inversão modular para números não coprimos" << std::endl; 
    // }

    if (argc != 3) {
        std::cerr << ">>: " << argv[0] << " <Compressed Public Key> <Key Range>" << std::endl;
        return 1;
    }
    
    std::string pub_key_hex(argv[1]);
    int key_range = std::stoi(argv[2]);

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx) {
        std::cerr << "Error to creating context SECP25651!" << std::endl;
        return 1;
    }

    unsigned char pub_key_bytes[33];
    for (size_t i = 0; i < 33; ++i) {
        pub_key_bytes[i] = std::stoi(pub_key_hex.substr(2 * i, 2), nullptr, 16);
    }

    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_parse(ctx, &pubkey, pub_key_bytes, sizeof(pub_key_bytes))) {
        std::cerr << "Error to parsing public key!" << std::endl;
        secp256k1_context_destroy(ctx);
        return 1;
    }

    //Teste atual com 1032 lebres:
    prho(ctx, pubkey, pubkey, key_range, 1032);

    return 0;
}
