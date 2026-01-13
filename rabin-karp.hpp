#pragma once

#include <string>
#include <cstdint>
#include <cmath>

struct rabin_karp {


    int w_size;
    int* window;
    int alphabet_size = 256;
    uint64_t hash;
    uint64_t prime = 1999999973;
    uint64_t num_chars;
    int alphabet_pot;

    int int_pow(int x, int p) {
        if (p == 0) return 1;
        if (p == 1) return x;

        int tmp = int_pow(x, p/2);
        if (p%2 == 0) {
            return tmp * tmp;
        }
        else {
            return x * tmp * tmp;
        }
    }

    rabin_karp(int window_size) : w_size(window_size) {
        alphabet_pot = int_pow(alphabet_size, window_size - 1) % prime;
        window = new int[w_size];
        reset();
    }

    void reset() {
        for (int i = 0; i < w_size; ++i) {
            window[i] = 0;
        }
        hash = 0;
        num_chars = 0;
    }

    uint64_t add_char(char c) {
        int k = num_chars % w_size;
        hash += (prime - (window[k] * alphabet_pot) % prime);
        hash = (alphabet_size * hash + c) % prime;
        window[k] = c;
        ++num_chars;
        return hash;
    }

    template<typename str>
    uint64_t kr_print(str const& phrase) {
        uint64_t loc_hash = 0;
        const uint64_t loc_prime = 3355443229;     // next prime(2**31+2**30+2**27)
        //const uint64_t prime = 1999999973; //27162335252586509; // next prime (2**54 + 2**53 + 2**47 + 2**13)
        for(auto c : phrase) {
             //assert(c>=0 && c< 256);
            loc_hash = (256*loc_hash + c) % loc_prime;    //  add char k
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


