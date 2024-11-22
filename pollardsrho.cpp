#include <iostream>
#include <secp256k1.h>
#include <unordered_map>
#include <random>
#include <cstring>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <boost/multiprecision/cpp_int.hpp>

/* Pollard's Rho Algorithm for SECP256K1 */
/* Written by Lucas Leblanc*/

using namespace boost::multiprecision;

uint256_t generate_random_private_key(uint256_t min_scalar, uint256_t max_scalar) {
    std::random_device rd;
    std::mt19937_64 gen(rd());

    uint64_t low = std::uniform_int_distribution<uint64_t>(
        static_cast<uint64_t>(min_scalar & 0xFFFFFFFFFFFFFFFF),
        static_cast<uint64_t>(max_scalar & 0xFFFFFFFFFFFFFFFF)
    )(gen);

    uint64_t high = std::uniform_int_distribution<uint64_t>(
        static_cast<uint64_t>((min_scalar >> 64) & 0xFFFFFFFFFFFFFFFF),
        static_cast<uint64_t>((max_scalar >> 64) & 0xFFFFFFFFFFFFFFFF)
    )(gen);

    return (static_cast<uint256_t>(high) << 64) | low;
}

std::string bytesToHex(const unsigned char* bytes, size_t length) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < length; ++i) {
        ss << std::setw(2) << static_cast<int>(bytes[i]);
    }
    return ss.str();
}

std::string uint256_to_hex(const uint256_t& value) {
    std::ostringstream oss;
    oss << std::setw(64) << std::setfill('0') << std::hex << value;
    return oss.str();
}

std::tuple<std::string, std::string, std::string> privateKeyToPublicKey(const std::string& private_key_hex) {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx) {
        throw std::runtime_error("Error to creating SECP256K1 context!");
    }

    unsigned char private_key[32];
    for (size_t i = 0; i < 32; ++i) {
        private_key[i] = std::stoi(private_key_hex.substr(2 * i, 2), nullptr, 16);
    }

    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, private_key)) {
        secp256k1_context_destroy(ctx);
        throw std::runtime_error("Error to creating public key!");
    }

    unsigned char pubkey_compact[33];
    size_t pubkey_len = sizeof(pubkey_compact);
    secp256k1_ec_pubkey_serialize(ctx, pubkey_compact, &pubkey_len, &pubkey, SECP256K1_EC_COMPRESSED);

    unsigned char pubkey_full[65];
    size_t pubkey_full_len = sizeof(pubkey_full);
    secp256k1_ec_pubkey_serialize(ctx, pubkey_full, &pubkey_full_len, &pubkey, SECP256K1_EC_UNCOMPRESSED);

    std::string x_hex = bytesToHex(pubkey_full + 1, 32);
    std::string y_hex = bytesToHex(pubkey_full + 33, 32);

    std::string compressed_key_hex = bytesToHex(pubkey_compact, pubkey_len);

    secp256k1_context_destroy(ctx);

    return std::make_tuple(compressed_key_hex, x_hex, y_hex);
}

bool point_to_hex(secp256k1_context* ctx, const secp256k1_pubkey& point, std::string& x_hex, std::string& y_hex) {
    unsigned char output[65];
    size_t output_len = sizeof(output);

    if (!secp256k1_ec_pubkey_serialize(ctx, output, &output_len, &point, SECP256K1_EC_UNCOMPRESSED)) {
        return false;
    }

    x_hex = bytesToHex(output + 1, 32);
    y_hex = bytesToHex(output + 33, 32);
    return true;
}

int64_t modular_inverse(int64_t a, int64_t m) {
    std::cout << "Base: " << a << ", Module: " << m << std::endl;

    if (m == 0) return 0;

    int64_t m0 = m, t, q;
    int64_t x0 = 0, x1 = 1;

    if (m == 1) return 1;

    while (a > 1) {

        if (m == 0) { return 0; }

        q = a / m;

        t = m;
        m = a % m;
        a = t;

        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }

    if (a != 1) return 0;

    if (x1 < 0) x1 += m0;

    return x1;
}

uint256_t serialize(const secp256k1_pubkey& point, secp256k1_context* ctx) {
    unsigned char serialized[33];
    size_t len = sizeof(serialized);

    try {
        if (!secp256k1_ec_pubkey_serialize(ctx, serialized, &len, &point, SECP256K1_EC_COMPRESSED)) {
            throw std::runtime_error("Failed to serialize public key");
        }

        return serialized[32];
    } catch (const std::exception& e) {
        std::cerr << "Exception caught during serialization: " << e.what() << std::endl;
        return uint256_t(0);
    }
}

class Random256Generator {
    std::mt19937_64 gen;
    uint64_t min_low, max_low, min_high, max_high;

public:
    Random256Generator(uint256_t min_scalar, uint256_t max_scalar)
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

uint256_t prho(secp256k1_context* ctx, const secp256k1_pubkey& G, const secp256k1_pubkey& target_pubkey, int key_range, int hares) {
    uint256_t min_scalar = (uint256_t(1) << (key_range - 1));  
    uint256_t max_scalar = (uint256_t(1) << key_range) - 1;
    uint256_t keys_ps;

    //SECP256K1 n
    uint256_t n = uint256_t("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

    std::cout << "key_range: " << key_range << std::endl;
    std::cout << "min_range: " << uint256_to_hex(min_scalar) << std::endl;
    std::cout << "max_range: " << uint256_to_hex(max_scalar) << std::endl;

    //Não remover:
    //uint256_t current_key = generate_random_private_key(min_scalar, max_scalar); 
    Random256Generator rng(min_scalar, max_scalar);
    uint256_t current_key = rng.generate();

    std::atomic<bool> search_in_progress(true);
    std::string current_pubkey_hex_R, current_pubkey_hex_R1;
    std::string compressed_key_hex_R, compressed_key_hex_R1;
    std::string x_hex_R, y_hex_R, x_hex_R1, y_hex_R1;
    std::mutex pgrs;

    std::thread log_thread([&]() {
    try {
         while (search_in_progress) {
             std::this_thread::sleep_for(std::chrono::seconds(10));
             std::lock_guard<std::mutex> lock(pgrs);
             std::cout << "\rCurrent private key: " << uint256_to_hex(current_key) << std::endl;
             std::cout << "\rLast tested public key: " << current_pubkey_hex_R << std::endl;
             std::cout << "\rTotal keys tested: " << keys_ps << std::endl;
         }
    } catch (const std::exception& e) {
         std::cerr << "Error in log_thread: " << e.what() << std::endl;
    }});

    struct HareState {
        uint256_t k1, k2;
        secp256k1_pubkey R, R1;
        int speed;
    };

    std::vector<HareState> hare_states(hares);

    for (int i = 0; i < hares; ++i)
    {
        hare_states[i].R = G;
        hare_states[i].R1 = G;
    }

    try {
        while (true) {
            for (int i = 0; i < hares; ++i) {

                HareState& hare = hare_states[i];
                /* Alternativa, não remover:
                hare_states[i].k1 = generate_random_private_key(min_scalar, max_scalar);
                hare_states[i].k2 = generate_random_private_key(min_scalar, max_scalar); */
                hare_states[i].k1 = rng.generate();
                hare_states[i].k2 = rng.generate();
                hare_states[i].speed = (i == 0) ? 1 : (i + 1);
                current_key = hare_states[i].k1;
                keys_ps++;

                unsigned char target_pubkey_serialized[33];
                size_t target_pubkey_len = sizeof(target_pubkey_serialized);
                if (!secp256k1_ec_pubkey_serialize(ctx, target_pubkey_serialized, &target_pubkey_len, &target_pubkey, SECP256K1_EC_COMPRESSED)) {
                    std::cerr << "Failed to serialize target public key!" << std::endl;
                    throw std::runtime_error("Error serializing target public key!");
                }

                if(serialize(hare.R, ctx) != 0 && serialize(hare.R1, ctx) != 0)
                {
                    hare.k1 = (hare.k1 + hare.speed) % (uint256_t(1) << 64);
                    hare.k2 = (hare.k2 + hare.speed) % (uint256_t(1) << 64);
                }

                if (hare.k1 < min_scalar) hare.k1 += min_scalar;
                if (hare.k2 < min_scalar) hare.k2 += min_scalar;

                if (!secp256k1_ec_pubkey_create(ctx, &hare.R, reinterpret_cast<const unsigned char*>(&hare.k1))) {
                    std::cerr << "Failed to create public key for hare " << i << " with k1: " << uint256_to_hex(hare.k1) << std::endl;
                    throw std::runtime_error("Error updating public key!");
                }

                if (!secp256k1_ec_pubkey_create(ctx, &hare.R1, reinterpret_cast<const unsigned char*>(&hare.k2))) {
                    std::cerr << "Failed to create public key for hare " << i << " with k2: " << uint256_to_hex(hare.k2) << std::endl;
                    throw std::runtime_error("Error updating public key!");
                }

                std::string k1_hex = uint256_to_hex(hare.k1);
                std::tie(compressed_key_hex_R, x_hex_R, y_hex_R) = privateKeyToPublicKey(k1_hex);
                current_pubkey_hex_R = compressed_key_hex_R;

                std::string k2_hex = uint256_to_hex(hare.k2);
                std::tie(compressed_key_hex_R1, x_hex_R1, y_hex_R1) = privateKeyToPublicKey(k2_hex);
                current_pubkey_hex_R1 = compressed_key_hex_R1;

                if (current_pubkey_hex_R == bytesToHex(target_pubkey_serialized, target_pubkey_len)
                     || current_pubkey_hex_R1 == bytesToHex(target_pubkey_serialized, target_pubkey_len)) {

                     std::cout << "\033[32mPrivate key found by hare " << i << "!\033[0m" << std::endl;
                     search_in_progress = false;
                     if (log_thread.joinable()) log_thread.join();

                     if (compressed_key_hex_R == bytesToHex(target_pubkey_serialized, target_pubkey_len)) {
                         return hare.k1;
                     }
                     else if (compressed_key_hex_R1 == bytesToHex(target_pubkey_serialized, target_pubkey_len))
                     {
                         return hare.k2;
                     }
                }

                //Verificar colisões não triviais, a espera de um milagre:
                if (!x_hex_R.empty() && !x_hex_R1.empty() && x_hex_R == x_hex_R1) {

                     for (int j = i + 1; j < hares; ++j) {

                          if (hare_states[i].k1 != hare_states[j].k2) {

                              uint256_t d = 0;
                             
                              while (true) {
                                  secp256k1_pubkey current_point = hare_states[i].R;
                                  secp256k1_pubkey next_point;

                                  unsigned char tweak[32];

                                  unsigned char serialized_G[33];
                                  size_t serialized_G_len = sizeof(serialized_G);
                                  if (!secp256k1_ec_pubkey_serialize(ctx, serialized_G, &serialized_G_len, &G, SECP256K1_EC_COMPRESSED)) {
                                      std::cerr << "Error serializing point G" << std::endl;
                                      return uint256_t(0);
                                  }

                                  std::memcpy(tweak, serialized_G, 32);

                                  if (!secp256k1_ec_pubkey_tweak_add(ctx, &next_point, tweak)) {
                                      std::cerr << "Error calculating point addition" << std::endl;
                                      break;
                                  }

                                  std::string x_current, y_current;
                                  if (!point_to_hex(ctx, next_point, x_current, y_current)) {
                                      std::cerr << "Failed to convert point to hex" << std::endl;
                                      break;
                                  }

                                  //d = (x1, y1) * G + (x2,y2) * (-G)
                                  if (x_current == x_hex_R1 && y_current == y_hex_R1) {
                                      break;
                                  }

                                  current_point = next_point;
                                  d++;

                                  if (d >= n) {
                                      std::cerr << "Failed to find difference between k1 and k2" << std::endl;
                                      break;
                                  }
                              }

                              //Verify if d * G ≡ 0
                              if ((d % n) == 0) {
                                   std::cout << "\033[32mCollision between hare " << i << " and hare " << j << "!\033[0m" << std::endl;
                                   std::cout << "hare k1: " << hare_states[i].k1 << std::endl;
                                   std::cout << "hare k2: " << hare_states[j].k2 << std::endl;

                                   search_in_progress = false;
                                   if (log_thread.joinable()) log_thread.join();

                                   return (hare_states[j].k2 + d) % n;
                              }
                              else
                              {
                                  std::cerr << "(d * G ≡ 0) is false" << std::endl;

                                  search_in_progress = false;
                                  if (log_thread.joinable()) log_thread.join();

                                  return uint256_t(0);
                              }
                          }
                     }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception: " << e.what() << std::endl;
        std::cerr.flush();
        return uint256_t(0);
    }
}

int main(int argc, char* argv[]) {
    //Test
    // int64_t a = 0x1fa5ee5, m = 0x1ff52b5;
    // int64_t result = modular_inverse(a, m);
    // std::cout << "A inversa de " << a << " modulo " << m << " = " << result << std::endl;

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

    //Teste atual com 16 lebres, valores maiores seriam eficientes apenas com multi-threads.
    auto private_key = prho(ctx, pubkey, pubkey, key_range, 16);

    std::cout << "Private Key Found: " << uint256_to_hex(private_key) << std::endl;

    secp256k1_context_destroy(ctx);
    return 0;
}