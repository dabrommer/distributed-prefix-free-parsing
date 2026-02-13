#pragma once

#include <cmath>
#include <cstdint>
#include <string>

struct rabin_karp {
    int            w_size = 0;
    int*           window;
    int            alphabet_size = 256;
    uint64_t       hash          = 0;
    uint64_t const prime         = 27162335252586509;
    uint64_t       num_chars     = 0;
    uint64_t       alphabet_pot;


    // Compute (a * b) % m without overflow
    uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t m) {
        uint64_t res = 0;
        a %= m;
        while (b > 0) {
            if (b % 2 == 1) {
                res = (res + a) % m;
            }
            a = (a * 2) % m;
            b /= 2;
        }
        return res;
    }

    // Compute (base^exp) % mod using modular exponentiation
    uint64_t int_pow(uint64_t base, uint64_t exp, uint64_t mod) {
        uint64_t res = 1;
        base %= mod;
        while (exp > 0) {
            if (exp % 2 == 1) {
                res = mul_mod(res, base, mod);
            }
            base = mul_mod(base, base, mod);
            exp /= 2;
        }
        return res;
    }

    explicit rabin_karp(int window_size) : w_size(window_size) {
        alphabet_pot = int_pow(alphabet_size, window_size - 1, prime) % prime;
        window       = new int[w_size];
        reset();
    }

    rabin_karp() = default;

    void reset() {
        for (int i = 0; i < w_size; ++i) {
            window[i] = 0;
        }
        hash      = 0;
        num_chars = 0;
    }

    uint64_t add_char(unsigned char c) {
        uint64_t k = num_chars % w_size;
        hash += (prime - (window[k] * alphabet_pot) % prime);
        hash      = (alphabet_size * hash + c) % prime;
        window[k] = c;
        ++num_chars;
        return hash;
    }

    uint64_t add_char_fingerprint(unsigned char c) {
        hash = (256 * hash + c) % prime;
        return hash;
    }

    template <typename str>
    uint64_t kr_print(str const& phrase) {
        uint64_t loc_hash = 0;
        for (auto c: phrase) {
            loc_hash = (256 * loc_hash + c) % prime; //  add char k
        }
        return loc_hash;
    }

    void print_window() {
        for (int i = 0; i < w_size; ++i) {
            std::cout << (char)(window[i]) << " ";
        }
        std::cout << std::endl;
    }
};
