#pragma once

#include <cstdint>

struct Pair {
    uint64_t hash;
    int      rank;

    Pair(uint64_t key, int value) : hash(key), rank(value) {}
    Pair() = default;
};