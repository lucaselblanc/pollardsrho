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

/* --- POLLARD'S RHO LAMBDA (ρλ) --- */

#include "secp256k1.h"

constexpr uint256_t N = { 0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL };
constexpr uint64_t ONE_MONT[4] = { 0x00000001000003D1ULL, 0x0ULL, 0x0ULL, 0x0ULL };
constexpr uint64_t SUB2_FP[4] = { 0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };

const std::string& RED = "\033[91m";
const std::string& GREEN = "\033[92m";
const std::string& BLUE = "\033[94m";
const std::string& CYAN = "\033[38;5;39m";
const std::string& DARK_PINK = "\033[38;2;140;70;140m";
const std::string& PINK = "\033[35m";
const std::string& ORANGE = "\033[38;2;255;128;0m";
const std::string& RESET = "\033[0m";
const char* BASE_58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

unsigned long long ops;
long double sqrtM, kFactor;
int snaptime_sec = 20;

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
    uint64_t x[4];
};

using DPTable = phmap::parallel_flat_hash_map<uint64_t, std::vector<DPEntry>, MurmurHash3, phmap::priv::hash_default_eq<uint64_t>, std::allocator<std::pair<const uint64_t, std::vector<DPEntry>>>, 8, std::mutex >;

const char SNAPOINT_CONSTANTS[8] = {'P', 'R', 'H', 'O', 'C', 'K', '1', '\0'};
const uint64_t SNAPOINT_HASH_OFFSET = 1469598103934665603ULL;
const uint64_t SNAPOINT_HASH_PRIME = 1099511628211ULL;

bool snapoint_walkers_state(
    bool save_snapoint,
    const std::string& snapoint_file,
    const std::string& target_pubkey_hex,
    int& key_range,
    int& walkers,
    int& dp_bits,
    int saved_window_size,
    uint32_t n_steps,
    std::vector<WalkState>& walkers_state,
    DPTable& dp_table,
    unsigned long long& total_iters,
    unsigned long long& total_cycles
)

{
    if (snapoint_file.empty()) return save_snapoint;
    if (snaptime_sec <= 0) return false;

    auto update_checksum = [](uint64_t& checksum, const void* data, size_t size) {
        const unsigned char* bytes = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < size; i++) {
            checksum ^= bytes[i];
            checksum *= SNAPOINT_HASH_PRIME;
        }
    };

    auto snapoint_directory = [](const std::string& file) {
        size_t slash = file.find_last_of('/');
        if (slash == std::string::npos) return std::string(".");
        if (slash == 0) return std::string("/");
        return file.substr(0, slash);
    };

    auto sync_snapoint_directory = [&](const std::string& file) {
        int directory_fd = ::open(snapoint_directory(file).c_str(), O_RDONLY | O_DIRECTORY);
        if (directory_fd < 0) return false;
        bool synced = (::fsync(directory_fd) == 0);
        ::close(directory_fd);
        return synced;
    };

    if (save_snapoint) {
        std::string temporary_file = snapoint_file + ".tmp." + std::to_string(static_cast<unsigned long long>(::getpid()));
        int snapoint_fd = ::open(temporary_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (snapoint_fd < 0) return false;

        uint64_t checksum = SNAPOINT_HASH_OFFSET;
        auto write_bytes = [&](const void* data, size_t size, bool include_in_checksum = true) {
            if (include_in_checksum) update_checksum(checksum, data, size);
            const unsigned char* bytes = static_cast<const unsigned char*>(data);
            while (size > 0) {
                ssize_t written = ::write(snapoint_fd, bytes, size);
                if (written < 0) {
                    if (errno == EINTR) continue;
                    return false;
                }
                if (written == 0) return false;
                bytes += written;
                size -= static_cast<size_t>(written);
            }
            return true;
        };

        auto write_value = [&](const auto& value) {
            return write_bytes(&value, sizeof(value));
        };

        auto write_string = [&](const std::string& value) {
            if (value.size() > std::numeric_limits<uint32_t>::max()) return false;
            uint32_t string_size = static_cast<uint32_t>(value.size());
            return write_value(string_size) && (string_size == 0 || write_bytes(value.data(), string_size));
        };

        auto write_uint256 = [&](const uint256_t& value) {
            return write_bytes(value.limbs, sizeof(value.limbs));
        };

        auto write_jacobian = [&](const ECPointJacobian& point) {
            int32_t infinity = point.infinity;
            return write_bytes(point.X, sizeof(point.X)) &&
                   write_bytes(point.Y, sizeof(point.Y)) &&
                   write_bytes(point.Z, sizeof(point.Z)) &&
                   write_value(infinity);
        };

        auto write_dp_entry = [&](const DPEntry& entry) {
            return write_uint256(entry.a) && write_uint256(entry.b) && write_value(entry.x);
        };

        uint32_t version = 1;
        int32_t saved_key_range = key_range;
        int32_t saved_walkers = walkers;
        int32_t saved_dp_bits = dp_bits;
        int32_t snapoint_window_size = saved_window_size;
        uint64_t saved_total_iters = total_iters;
        uint64_t saved_total_cycles = total_cycles;
        uint64_t walkers_count = static_cast<uint64_t>(walkers_state.size());
        uint64_t dp_bucket_count = static_cast<uint64_t>(dp_table.size());

        bool ok =
            write_bytes(SNAPOINT_CONSTANTS, sizeof(SNAPOINT_CONSTANTS)) &&
            write_value(version) &&
            write_string(target_pubkey_hex) &&
            write_value(saved_key_range) &&
            write_value(saved_walkers) &&
            write_value(saved_dp_bits) &&
            write_value(snapoint_window_size) &&
            write_value(n_steps) &&
            write_value(saved_total_iters) &&
            write_value(saved_total_cycles) &&
            write_value(walkers_count
        );

        for (size_t i = 0; ok && i < walkers_state.size(); i++) {
            WalkState& walker = walkers_state[i];
            ok =
                write_value(walker.walk_id) &&
                write_uint256(walker.a) &&
                write_uint256(walker.b) &&
                write_jacobian(walker.R) &&
                write_value(walker.snapshot_steps) &&
                write_bytes(walker.snapshot_x, sizeof(walker.snapshot_x)) &&
                write_bytes(walker.prev_x1, sizeof(walker.prev_x1)) &&
                write_bytes(walker.prev_x2, sizeof(walker.prev_x2)
            );
        }

        ok = ok && write_value(dp_bucket_count);
        for (auto it = dp_table.cbegin(); ok && it != dp_table.cend(); ++it) {
            uint64_t dp_x0 = it->first;
            uint64_t dp_count = static_cast<uint64_t>(it->second.size());
            ok = write_value(dp_x0) && write_value(dp_count);
            for (size_t i = 0; ok && i < it->second.size(); i++) {
                ok = write_dp_entry(it->second[i]);
            }
        }

        ok = ok && write_bytes(&checksum, sizeof(checksum), false);
        ok = ok && (::fsync(snapoint_fd) == 0);
        if (::close(snapoint_fd) != 0) ok = false;

        if (!ok) {
            ::unlink(temporary_file.c_str());
            return false;
        }
        if (::rename(temporary_file.c_str(), snapoint_file.c_str()) != 0) {
            ::unlink(temporary_file.c_str());
            return false;
        }
        return sync_snapoint_directory(snapoint_file);
    }

    struct stat snapoint_stat {};
    if (::stat(snapoint_file.c_str(), &snapoint_stat) != 0 || snapoint_stat.st_size < 64) return false;

    std::ifstream snapoint(snapoint_file.c_str(), std::ios::binary);
    if (!snapoint.good()) return false;

    uint64_t checksum = SNAPOINT_HASH_OFFSET;
    auto read_bytes = [&](void* data, size_t size, bool include_in_checksum = true) {
        snapoint.read(static_cast<char*>(data), size);
        if (!snapoint) return false;
        if (include_in_checksum) update_checksum(checksum, data, size);
        return true;
    };

    auto read_value = [&](auto& value) {
        return read_bytes(&value, sizeof(value));
    };

    auto read_string = [&](std::string& value, uint32_t max_size) {
        uint32_t string_size = 0;
        if (!read_value(string_size) || string_size > max_size) return false;
        value.assign(string_size, '\0');
        return string_size == 0 || read_bytes(&value[0], string_size);
    };

    auto read_uint256 = [&](uint256_t& value) {
        return read_bytes(value.limbs, sizeof(value.limbs));
    };

    auto read_jacobian = [&](ECPointJacobian& point) {
        int32_t infinity = 0;
        if (!read_bytes(point.X, sizeof(point.X)) ||
            !read_bytes(point.Y, sizeof(point.Y)) ||
            !read_bytes(point.Z, sizeof(point.Z)) ||
            !read_value(infinity)) {
            return false;
        }
        point.infinity = infinity;
        return true;
    };

    auto read_dp_entry = [&](DPEntry& entry) {
        return read_uint256(entry.a) && read_uint256(entry.b) && read_value(entry.x);
    };

    char magic[8] = {};
    uint32_t version = 0;
    std::string saved_target;
    int32_t saved_key_range = 0;
    int32_t saved_walkers = 0;
    int32_t saved_dp_bits = 0;
    int32_t snapoint_window_size = 0;
    uint32_t saved_n_steps = 0;
    uint64_t saved_total_iters = 0;
    uint64_t saved_total_cycles = 0;
    uint64_t walkers_count = 0;

    if (!read_bytes(magic, sizeof(magic)) ||
        memcmp(magic, SNAPOINT_CONSTANTS, sizeof(SNAPOINT_CONSTANTS)) != 0 ||
        !read_value(version) ||
        version != 1 ||
        !read_string(saved_target, 128) ||
        !read_value(saved_key_range) ||
        !read_value(saved_walkers) ||
        !read_value(saved_dp_bits) ||
        !read_value(snapoint_window_size) ||
        !read_value(saved_n_steps) ||
        !read_value(saved_total_iters) ||
        !read_value(saved_total_cycles) ||
        !read_value(walkers_count)) {
        return false;
    }

    if (saved_target != target_pubkey_hex) {
	return false;
    }

    std::vector<WalkState> loaded_walkers(static_cast<size_t>(walkers_count));
    for (size_t i = 0; i < loaded_walkers.size(); i++) {
        WalkState walker{};
        if (!read_value(walker.walk_id) ||
            !read_uint256(walker.a) ||
            !read_uint256(walker.b) ||
            !read_jacobian(walker.R) ||
            !read_value(walker.snapshot_steps) ||
            !read_bytes(walker.snapshot_x, sizeof(walker.snapshot_x)) ||
            !read_bytes(walker.prev_x1, sizeof(walker.prev_x1)) ||
            !read_bytes(walker.prev_x2, sizeof(walker.prev_x2))) {
            return false;
        }
        if (walker.walk_id != i) return false;
        walker.buffers = nullptr;
        loaded_walkers[i] = walker;
    }

    uint64_t dp_bucket_count = 0;
    if (!read_value(dp_bucket_count) || dp_bucket_count > static_cast<uint64_t>(snapoint_stat.st_size) / sizeof(uint64_t)) return false;

    DPTable loaded_dp_table;
    for (uint64_t bucket_idx = 0; bucket_idx < dp_bucket_count; bucket_idx++) {
        uint64_t dp_x0 = 0;
        uint64_t dp_count = 0;
        if (!read_value(dp_x0) || !read_value(dp_count)) return false;
        if (dp_count > static_cast<uint64_t>(snapoint_stat.st_size) / sizeof(DPEntry)) return false;

        std::vector<DPEntry> dp_entries;
        dp_entries.reserve(static_cast<size_t>(dp_count));
        for (uint64_t entry_idx = 0; entry_idx < dp_count; entry_idx++) {
            DPEntry entry;
            if (!read_dp_entry(entry)) return false;
            dp_entries.push_back(entry);
        }
        loaded_dp_table.emplace(dp_x0, std::move(dp_entries));
    }

    uint64_t expected_checksum = 0;
    char extra = 0;
    if (!read_bytes(&expected_checksum, sizeof(expected_checksum), false)) return false;
    snapoint.read(&extra, 1);
    if (!snapoint.eof() || expected_checksum != checksum) return false;

    walkers_state = std::move(loaded_walkers);
    dp_table.swap(loaded_dp_table);
    total_iters = saved_total_iters;
    total_cycles = saved_total_cycles;
    key_range = saved_key_range;
    walkers   = saved_walkers;
    dp_bits   = saved_dp_bits;
    return true;
}

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

auto cores = std::thread::hardware_concurrency();
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
        throw std::invalid_argument("The hexadecimal string must have n pairs of characters!");
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
            std::cout << ORANGE << "WRNG: " << e.what() << RESET << std::endl;
            continue;
        }
        catch(const std::out_of_range& e) {
            std::cout << ORANGE << "WRNG: " << e.what() << RESET << std::endl;
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
        uint64_t z2[4], tmp_mont[4];

        modMulMontP(z2, z_inv, z_inv);
        modMulMontP(tmp_mont, jac_in[i].X, z2);
        fromMontgomeryP(aff_out[i].x, tmp_mont);

        aff_out[i].infinity = 0;
    }
}

uint256_t prho(std::string target_pubkey_hex, int key_range, int WALKERS, int DP_BITS, const std::string& snapoint_path, int snaptime_sec) {
    std::atomic<bool> search_in_progress(true);
    std::atomic<int> loaded_walkers{0};
    std::atomic<unsigned long long> total_iters{0};
    std::atomic<unsigned long long> total_cycles{0};
    auto start_time = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time);
    std::tm start_tm{};
    localtime_r(&start_time_t, &start_tm);

    uint256_t k{};
    uint256_t min_scalar{}, max_scalar{};
    {
        int limb = (key_range - 1) / 64;
        int bit = (key_range - 1) % 64;
        min_scalar.limbs[limb] = 1ULL << bit;
        for (int i = 0; i < limb; i++) max_scalar.limbs[i] = ~0ULL;
        max_scalar.limbs[limb] = (bit == 63) ? ~0ULL : (1ULL << (bit + 1)) - 1;
    }

    auto target_pubkey = hex_to_bytes(target_pubkey_hex);
    ECPointAffine target_affine{};
    ECPointJacobian target_affine_jac{};
    decompressPublicKey(&target_affine, target_pubkey.data());
    affineToJacobian(&target_affine_jac, &target_affine);
    initPreCompH(&target_affine_jac, windowSize);

    const uint32_t N_STEPS = 2048;
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

    DPTable dp_table;
    std::vector<WalkState> walkers_state(WALKERS);
    std::vector<uint64_t> sharedScalarStepsG((1ULL << windowSize) * 4);
    std::vector<uint64_t> sharedScalarStepsH((1ULL << windowSize) * 4);
    initScalarSteps(sharedScalarStepsG.data(), windowSize);
    initScalarSteps(sharedScalarStepsH.data(), windowSize);

    unsigned long long restored_iters = 0;
    unsigned long long restored_cycles = 0;
    int dyn_key_range = key_range;
    int dyn_walkers = WALKERS;
    int dyn_dp_bits = DP_BITS;

    bool resumed_snapoint = snapoint_walkers_state(false, snapoint_path, target_pubkey_hex, dyn_key_range, dyn_walkers, dyn_dp_bits, windowSize, N_STEPS, walkers_state, dp_table, restored_iters, restored_cycles);

    if (resumed_snapoint) {
        if(WALKERS != dyn_walkers){
            std::cout << ORANGE << "[WRNG] The number of walkers has been overwritten by the file configuration: " << RESET << CYAN << target_pubkey_hex << ".saved\n"  << RESET;
        }

        if(DP_BITS != dyn_dp_bits){
            std::cout << ORANGE << "[WRNG] The number of DP's has been overwritten by the file configuration: " << RESET << CYAN << target_pubkey_hex << ".saved\n\n"  << RESET;
        }

        key_range = dyn_key_range;
	    WALKERS = dyn_walkers;
	    DP_BITS = dyn_dp_bits;
        total_iters.store(restored_iters, std::memory_order_relaxed);
        total_cycles.store(restored_cycles, std::memory_order_relaxed);
    }

    std::cout << CYAN << "Started at: " << RESET << PINK << std::put_time(&start_tm, "%H:%M:%S") << RESET << std::endl;
    std::cout << CYAN << "Threads: " << RESET << PINK << cores << RESET << std::endl;
    std::cout << CYAN << "Walkers: " << RESET << PINK << WALKERS << RESET << std::endl;
    std::cout << CYAN << "DP Bits: " << RESET << PINK << DP_BITS << RESET << std::endl;
    std::cout << CYAN << "Key Range: " << RESET << PINK << (key_range) << RESET << std::endl;
    std::cout << CYAN << "Min Range: " << RESET << gradient_zeros(uint256_to_hex(min_scalar), DARK_PINK, PINK) << std::endl;
    std::cout << CYAN << "Max Range: " << RESET << gradient_zeros(uint256_to_hex(max_scalar), DARK_PINK, PINK) << std::endl;
    std::cout << BLUE << "---------------------------------------------------------------------------" << RESET;
    std::cout << "\n\n\n\n";

    std::string header = "\033[96m[!] Loading Walkers... \033[0m";

    if (!resumed_snapoint)
    {
        unsigned int load_threads = std::min<unsigned int>(cores, WALKERS);

        std::vector<std::thread> load_workers;
        load_workers.reserve(load_threads);
        std::atomic<int> loaded_count{0};

        auto load_worker = [&](int start, int end)
        {
            for (int i = start; i < end; i++)
            {
                walkers_state[i].rng.seed(std::random_device{}() ^ (uint64_t)i);
                walkers_state[i].buffers = nullptr;

                if (i % 2 == 0) {
                    walkers_state[i].a = rng_mersenne_twister(min_scalar, max_scalar, key_range, walkers_state[i].rng);
                    walkers_state[i].b = uint256_t{};
                }
                else {
                    walkers_state[i].a = rng_mersenne_twister(uint256_t{0}, stepSize, key_range / 2, walkers_state[i].rng);
                    walkers_state[i].b = uint256_t{};
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
                }
                else {
                    pointAddJacobian(&walkers_state[i].R, &Ra, &target_affine_jac);
                }

                /*
                int current = loaded_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (current % 32 == 0 || current == WALKERS) {
                    static std::mutex loading_mutex;
                    std::lock_guard<std::mutex> lock(loading_mutex);
                    loading_bar(current, WALKERS, header);
                }
            */
            }
        };

        int chunk = WALKERS / load_threads;
        for (unsigned int t = 0; t < load_threads; t++)
        {
            int start = t * chunk;
            int end = (t == load_threads - 1) ? WALKERS : start + chunk;
            load_workers.emplace_back(load_worker, start, end);
        }

        for (auto& th : load_workers)
        {
            if (th.joinable()) { th.join(); }
        }
    }

    for (auto& walker : walkers_state) {
        if (walker.buffers == nullptr) walker.buffers = new Buffers();
        walker.buffers->scalarStepsG = sharedScalarStepsG.data();
        walker.buffers->scalarStepsH = sharedScalarStepsH.data();
    }

    std::atomic<unsigned long long> snapoint_saves{0};
    std::atomic<unsigned long long> snapoint_errors{0};
    if (!resumed_snapoint && !snapoint_path.empty() && snaptime_sec > 0) {
        unsigned long long saved_iters = total_iters.load(std::memory_order_relaxed);
        unsigned long long saved_cycles = total_cycles.load(std::memory_order_relaxed);
        bool saved = snapoint_walkers_state(true, snapoint_path, target_pubkey_hex, key_range, WALKERS, DP_BITS, windowSize, N_STEPS, walkers_state, dp_table, saved_iters, saved_cycles);
        if (saved) snapoint_saves.fetch_add(1, std::memory_order_relaxed);
        else snapoint_errors.fetch_add(1, std::memory_order_relaxed);
    }

    auto last_print = std::chrono::steady_clock::now();
    std::mutex snapoint_mutex;
    std::condition_variable snapoint_cv;
    bool snapoint_requested = false;
    unsigned int snapoint_paused = 0;
    unsigned int snapoint_thread_count = 0;

    auto snapoint_pause = [&]() {
        if (snapoint_path.empty() || snaptime_sec <= 0) return;
        std::unique_lock<std::mutex> lock(snapoint_mutex);
        if (!snapoint_requested) return;
        snapoint_paused++;
        snapoint_cv.notify_all();
        snapoint_cv.wait(lock, [&]() {
            return !snapoint_requested || !search_in_progress.load(std::memory_order_acquire);
        });
        snapoint_paused--;
        snapoint_cv.notify_all();
    };

    auto save_coordinated_snapoint = [&]() {
        if (snapoint_path.empty() || snaptime_sec <= 0 || snapoint_thread_count == 0) return false;
        {
            std::unique_lock<std::mutex> lock(snapoint_mutex);
            snapoint_requested = true;
            snapoint_cv.notify_all();
            snapoint_cv.wait(lock, [&]() {
                return snapoint_paused >= snapoint_thread_count || !search_in_progress.load(std::memory_order_acquire);
            });
            if (!search_in_progress.load(std::memory_order_acquire)) {
                snapoint_requested = false;
                lock.unlock();
                snapoint_cv.notify_all();
                return false;
            }
        }

        unsigned long long saved_iters = total_iters.load(std::memory_order_relaxed);
        unsigned long long saved_cycles = total_cycles.load(std::memory_order_relaxed);
        bool ok = snapoint_walkers_state(true, snapoint_path, target_pubkey_hex, key_range, WALKERS, DP_BITS, windowSize, N_STEPS, walkers_state, dp_table, saved_iters, saved_cycles);

        if(snaptime_sec > 0){
            if (ok) snapoint_saves.fetch_add(1, std::memory_order_relaxed);
            else snapoint_errors.fetch_add(1, std::memory_order_relaxed);
        }

        {
            if(snaptime_sec <= 0) return false;
            std::unique_lock<std::mutex> lock(snapoint_mutex);
            snapoint_requested = false;
            lock.unlock();
            snapoint_cv.notify_all();
            lock.lock();
            snapoint_cv.wait(lock, [&]() {
                return snapoint_paused == 0;
            });
        }
        snapoint_cv.notify_all();
        return ok;
    };

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
                snapoint_pause();
                if (!search_in_progress.load(std::memory_order_acquire)) break;
                total_iters.fetch_add(local_count, std::memory_order_relaxed);
                for (int i = 0; i < local_count; i++) {
                    WalkState* w = &walkers_state[id_start + i];
                    memcpy(w->prev_x2, w->prev_x1, 32);
                    memcpy(w->prev_x1, aff_batch[i].x, 32);
                    uint32_t step_idx = get_step_idx(aff_batch[i].x, N_STEPS);
                    pointAddJacobian(&w->R, &w->R, &localStepTable[step_idx].point);
                    scalarAdd(w->a.limbs, w->a.limbs, localStepTable[step_idx].a.limbs);
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
                                if (entry.x[1] == aff_batch[i].x[1] && entry.x[2] == aff_batch[i].x[2] && entry.x[3] == aff_batch[i].x[3]) {
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
                                std::memcpy(new_entry.x, aff_batch[i].x, sizeof(new_entry.x));
                                new_entry.a = w->a;
                                new_entry.b = w->b;
                                dps.push_back(new_entry);
                            }
                        }, [&](auto bucket) {
                            DPEntry entry;
                            std::memcpy(entry.x, aff_batch[i].x, sizeof(entry.x));
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
                                inv_db = modinv(db, N);
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
                long double x = (k * k) / (2.0L * M);
                long double d = x;
                long double prob = (1.0L - expl(-d)) * 100.0L;
                std::string healthWalking = total_cycles.load() == 0 ? " - \033[92mGood\033[0m" : " - \033[91mBad\033[0m";
                std::string snapointStatus = snapoint_path.empty() ? "Off" : (resumed_snapoint ? "Restored/" : "") + std::to_string(snapoint_saves.load(std::memory_order_relaxed));
                if (snapoint_errors.load(std::memory_order_relaxed) > 0) snapointStatus += " Err:" + std::to_string(snapoint_errors.load(std::memory_order_relaxed));
                std::cout << CYAN << "\033[3A\r" << "\033[2KTotal Ops/10s: " << RESET << GREEN << total_iters.load() << RESET << "\n" << CYAN << "\033[2KSelf-Collision Cycles: " << RESET << GREEN << total_cycles.load() << healthWalking << RESET << "\n" << CYAN << "\033[2KCollision Probability: " << RESET << GREEN << std::fixed << std::setprecision(8) << (prob) << "...%" << RESET << CYAN << " | Snapoints: " << RESET << PINK << snapointStatus << RESET << "\n" << std::flush;
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
    snapoint_thread_count = cores;

    for (unsigned int t = 0; t < cores; t++) {
        int start = t * chunk;
        int end = (t == cores - 1) ? WALKERS : start + chunk;
        threads.emplace_back(worker, start, end);
    }

    std::thread snapoint_thread;
    if (!snapoint_path.empty() && snaptime_sec > 0) {
        snapoint_thread = std::thread([&]() {
            while (search_in_progress.load(std::memory_order_acquire)) {
                int waited_ms = 0;
                int interval_ms = snaptime_sec * 1000;
                while (waited_ms < interval_ms && search_in_progress.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    waited_ms += 200;
                }
                if (!search_in_progress.load(std::memory_order_acquire)) break;
                save_coordinated_snapoint();
            }
        });
    }

    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }

    search_in_progress.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(snapoint_mutex);
        snapoint_requested = false;
    }
    snapoint_cv.notify_all();
    if(snapoint_thread.joinable()) snapoint_thread.join();
    if(progress_thread.joinable()) progress_thread.join();

    { ops = total_iters.load(); sqrtM = powl(2.0L, key_range / 2.0L); kFactor = (long double)ops / sqrtM; }

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
        std::cout << ORANGE << "[INFO!] " << RESET << "Key successfully saved in " << RESET << BLUE << "DISCRETE_LOGS_SOLVED" << RESET << std::endl;
    } else {
        std::cerr << RED << "[ERROR!] We were unable to open the file to save the key!" << RESET << std::endl;
    }
}

std::string EncodeBase58(const std::vector<unsigned char>& data) {
    std::vector<int> digits;
    for (unsigned char byte : data) {
        int carry = byte;
        for (size_t j = 0; j < digits.size(); ++j) {
            carry += digits[j] * 256;
            digits[j] = carry % 58;
            carry /= 58;
        }
        while (carry > 0) {
            digits.push_back(carry % 58);
            carry /= 58;
        }
    }
    std::string result;
    for (unsigned char byte : data) { if (byte == 0) result += '1'; else break; }
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) { result += BASE_58[*it]; }
    return result;
}

std::vector<unsigned char> DoubleSHA256(const std::vector<unsigned char>& data) {
    std::vector<unsigned char> hash1(SHA256_DIGEST_LENGTH);
    std::vector<unsigned char> hash2(SHA256_DIGEST_LENGTH);
    SHA256(data.data(), data.size(), hash1.data());
    SHA256(hash1.data(), hash1.size(), hash2.data());
    return hash2;
}

std::string EncodeBase58Check(const std::vector<unsigned char>& payload) {
    std::vector<unsigned char> hash = DoubleSHA256(payload);
    std::vector<unsigned char> dataToEncode = payload;
    dataToEncode.insert(dataToEncode.end(), hash.begin(), hash.begin() + 4);
    return EncodeBase58(dataToEncode);
}

std::string HexToWif(const std::string& hexKey) {
    std::vector<unsigned char> keyBytes = hex_to_bytes(hexKey);
    std::vector<unsigned char> payload;
    payload.push_back(0x80);
    payload.insert(payload.end(), keyBytes.begin(), keyBytes.end());
    payload.push_back(0x01);
    return EncodeBase58Check(payload);
}

int main(int argc, char* argv[]) {
    std::string pub_key_hex;
    int key_range;
    int walkers;
    int dp = -1;
    std::string snapoint_path;
    int snaptime_sec = 20;

    if (argc == 1) {
        std::cout << "The Parameters Cannot Be Empty!" << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--pubkey" && i + 1 < argc) {
            pub_key_hex = argv[++i];
        } else if (arg == "--keyrange" && i + 1 < argc) {
            key_range = std::stoi(argv[++i]);
        } else if (arg == "--walkers" && i + 1 < argc) {
            walkers = std::stoi(argv[++i]);
        } else if (arg == "--dp" && i + 1 < argc) {
            dp = std::stoi(argv[++i]);
        } else if (arg == "--snaptime" && i + 1 < argc) {
            snaptime_sec = std::stoi(argv[++i]);
        } else if (arg == "--t" && i + 1 < argc) {
            cores = std::stoi(argv[++i]);
        }
    }

    if (pub_key_hex.length() != 66) {
        std::cerr << RED << "[ERROR] Invalid Public Key Length." << RESET << std::endl;
        return 1;
    }

    if (dp <= 0 || dp > static_cast<int>(sizeof(int32_t) * CHAR_BIT)) {
        std::cerr << ORANGE << "[INFO] " << RESET << GREEN << "Setting DP automatically..." << RESET << std::endl;
        dp = std::max<int>(1, std::min<int>(key_range >> 2, static_cast<int>(sizeof(int32_t) * CHAR_BIT)));
    }

    if (snapoint_path.empty()) {
        snapoint_path = pub_key_hex + ".saved";
    }

    init_secp256k1(key_range);

    constexpr int TOTAL_RUNS = 1000;

    long double sum_kfactor = 0.0L;
    std::vector<long double> k_values;
    k_values.reserve(TOTAL_RUNS);

    for (int run = 0; run < TOTAL_RUNS; run++) {

        std::cout << CYAN 
                  << "\n[RUN " << (run + 1) << "/" << TOTAL_RUNS << "]"
                  << RESET << std::endl;

        uint256_t found_key = prho(
            pub_key_hex,
            key_range,
            walkers,
            dp,
            snapoint_path,
            snaptime_sec
        );

        long double current_k = kFactor;

        k_values.push_back(current_k);
        sum_kfactor += current_k;

        std::cout << GREEN 
                  << "[Collision " << (run + 1) << "] "
                  << RESET
                  << "K-Factor: "
                  << PINK
                  << std::fixed 
                  << std::setprecision(8)
                  << (double)current_k
                  << RESET
                  << std::endl;
    }

    long double average_k = sum_kfactor / TOTAL_RUNS;

    std::cout << "\n"
              << BLUE << "---------------------------------------------------------------------------"
              << RESET << std::endl;

    std::cout << CYAN 
              << "[RESULT] Average K-Factor (" 
              << TOTAL_RUNS 
              << " runs): "
              << RESET
              << PINK
              << std::fixed
              << std::setprecision(8)
              << (double)average_k
              << RESET
              << std::endl;

    std::cout << BLUE << "---------------------------------------------------------------------------"
              << RESET << std::endl;


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

/*
int main(int argc, char* argv[]) {
    std::string pub_key_hex;
    int key_range;
    int walkers;
    int dp = -1;
    std::string snapoint_path;

    if (argc == 1) {
        std::cout << "The Parameters Cannot Be Empty!" << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--pubkey" && i + 1 < argc) {
            pub_key_hex = argv[++i];
        } else if (arg == "--keyrange" && i + 1 < argc) {
            key_range = std::stoi(argv[++i]);
        } else if (arg == "--walkers" && i + 1 < argc) {
            walkers = std::stoi(argv[++i]);
        } else if (arg == "--dp" && i + 1 < argc) {
            dp = std::stoi(argv[++i]);
        }
          else if (arg == "--snaptime" && i + 1 < argc) {
            snaptime_sec = std::stoi(argv[++i]);
        }
          else if (arg == "--t" && i + 1 < argc) {
            cores = std::stoi(argv[++i]);
        }
        else {
            std::cout << BLUE << "---------------------------------------------------------------------------" << RESET << std::endl;
            std::cerr << ORANGE << "[USAGE]: " << RESET << GREEN << argv[0] << RESET << " --? <?> --? <?> --? <?>\n"
            << GREEN << "*" << RESET << " --pubkey        => Compressed Public Key        <hex> ("<< RED << "*" << RESET << "Required)\n"
            << GREEN << "*" << RESET << " --keyrange      => Key Range Bits               <int> ("<< RED << "*" << RESET << "Required)\n"
            << GREEN << "*" << RESET << " --walkers       => Number Of Walkers            <int> ("<< RED << "*" << RESET << "Required)\n"
            << GREEN << "*" << RESET << " --dp            => Distinguished Points Bits    <int> ("<< GREEN << "*" << RESET << "Optional)\n"
            << GREEN << "*" << RESET << " --snaptime      => Snapoints Seconds            <int> 0 disables periodic saves ("<< GREEN << "*" << RESET << "Optional)\n"
            << GREEN << "*" << RESET << " --t             => Work Threads                 <int> ("<< GREEN << "*" << RESET << "Optional)\n";
            std::cout << BLUE << "---------------------------------------------------------------------------" << RESET << std::endl;
            return 1;
        }
    }

    if (pub_key_hex.length() != 66) {
        std::cerr << RED << "[ERROR] The Compressed Public Key Must Be Exactly 66 Characters Long, Prefix 02/03 + 64 Hex." << RESET << std::endl;
        std::cerr << "Current Length: " << pub_key_hex.length() << std::endl;
        return 1;
    }

    std::string prefix = pub_key_hex.substr(0, 2);
    if (prefix != "02" && prefix != "03") {
        std::cerr << RED << "[ERROR] Unusual Compressed Key Prefix: " << prefix <<", Expected: 02/03." << RESET << std::endl;
        std::cerr << "Prefix Entered: " << prefix << std::endl;
        return 1;
    }

    if (key_range < 1 || key_range > 256) {
        std::cerr << RED << "[ERROR] Key Range Outside Permitted Limits (1 - 256)." << RESET << std::endl;
        std::cerr << "Value Entered: " << key_range << std::endl;
        return 1;
    }

    std::cout << BLUE << "---------------------------------------------------------------------------" << RESET << std::endl;
    std::cout << ORANGE << "[INFO] " << RESET << CYAN << "Add a Star to Support this Project ;)\n" << RESET;
    if (dp <= 0 || dp > static_cast<int>(sizeof(int32_t) * CHAR_BIT)) {
        std::cerr << ORANGE << "[INFO] " << RESET << GREEN << "Setting DP automatically..." << RESET << std::endl;
        dp = std::max<int>(1, std::min<int>(key_range >> 2, static_cast<int>(sizeof(int32_t) * CHAR_BIT)));
    }

    if (snapoint_path.empty()) {
        snapoint_path = pub_key_hex + ".saved";
    }

    std::cout << ORANGE << "[INFO] " << RESET << GREEN << "Press 'Ctrl Z' to Quit\n" << RESET;
    std::cout << ORANGE << "[INFO] " << RESET << GREEN << "Auto Window-Size for secp256k1: " << RESET << PINK << windowSize << RESET << std::endl;
    std::cout << ORANGE << "[INFO] " << RESET << GREEN << "For DP: " << PINK << dp << RESET << GREEN << " the rarity is \033[35m1\033[0m \033[92min " << RESET << PINK << (1ULL << dp) << RESET << GREEN << " points" << RESET << std::endl;
    std::cout << ORANGE << "[INFO] " << RESET << GREEN << "Snapoints File: " << RESET << CYAN << snapoint_path << RESET << std::endl;
    std::cout << ORANGE << "[INFO] " << RESET << GREEN << "Snaptime Interval: " << RESET << PINK << snaptime_sec << RESET << GREEN << "s" << RESET << std::endl;
    uint64_t max_throughput = cores * 512ULL;
    if (walkers != max_throughput) {
        std::cout << ORANGE << "[WRNG] For its " << PINK << cores << RESET << ORANGE << " Cores, Maximum Throughput Reached At: ~" << RESET << PINK << max_throughput << RESET << "." << std::endl;
    }
    std::cout << BLUE << "---------------------------------------------------------------------------" << RESET << std::endl;

    init_secp256k1(key_range);
    uint256_t found_key = prho(pub_key_hex, key_range, walkers, dp, snapoint_path, snaptime_sec);

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
    std::cout << ORANGE << "[WIF Key]: " << CYAN << HexToWif(uint256_to_hex(found_key)) << RESET << std::endl;
    std::cout << GREEN << "[Private Key]: " << RESET << gradient_zeros(uint256_to_hex(found_key), DARK_PINK, PINK) << std::endl;

    double key_val = 0;
    for (int i = 0; i < 4; i++) {
        key_val += (double)found_key.limbs[i] * std::pow(2.0, i * 64);
    }

    double range_start = std::pow(2.0, key_range - 1);
    double range_end = std::pow(2.0, key_range);
    double relative_pos = (key_val - range_start) / (range_end - range_start);
    double percentage = relative_pos * 100.0;

    std::string healthK = kFactor < 2 ? " - \033[92mStatistics Luck!\033[0m" : " - \033[91mStatistical Bad Luck!\033[0m";
    std::cout << CYAN << "[% Of The Range]: " << RESET << PINK << std::fixed << std::setprecision(2) << percentage << "%" << RESET << std::endl;
    std::cout << CYAN << "[K-Factor] : " << RESET << PINK << std::fixed << std::setprecision(4) << (double)kFactor << healthK << RESET << std::endl;
    std::cout << CYAN << "[Total OPS]: " << RESET << PINK << ops << RESET << std::endl;

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
*/
