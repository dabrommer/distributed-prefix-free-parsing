#pragma once
#include <cstdlib>
#include <iostream>
#include <string>


template<typename HashType>
struct hash {

};

struct rabin_hash : public hash {

};

struct buz_hash : public hash  {


};

struct adler_32_hash : public hash  {


};

hash hash_factory(std::string const& hash_name) {
    if (hash_name == "rabin") {
        return rabin_hash{};
    } else if (hash_name == "buz") {
        return buz_hash{};
    } else if (hash_name == "adler32") {
        return adler_32_hash{};
    } else {
        std::cerr << "Hash " << hash_name << " does not exist!" << std::endl;
        exit(1);
    }
}