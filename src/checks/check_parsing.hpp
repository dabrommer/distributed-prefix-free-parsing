#pragma once

#include <iostream>
#include <vector>

#include "kamping/communicator.hpp"
#include "util/cli_parser.hpp"

inline bool check_parsing(std::vector<int> const& ranks, Params const& params, std::vector<unsigned char> const& phrases, kamping::Communicator<>& comm, unsigned char DELIMITER) {


    std::ifstream file(params.input_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open input file '" << params.input_path << "' for checking\n";
        return false;
    }
    std::vector<char> file_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<std::string> all_phrases;
    std::string curr_str;
    for (const auto& c : phrases) {
        if (c == DELIMITER) {
            all_phrases.push_back(curr_str);
            curr_str.clear();
        } else {
            curr_str.push_back(static_cast<char>(c));
        }
    }

    std::size_t matches = 0;
    std::size_t missmatches = 0;
    bool correct = true;
    std::size_t index = 0;
    bool first_phrase = true;

    for (int rank : ranks) {
        if (rank < 0 || static_cast<std::size_t>(rank) >= all_phrases.size()) {
            std::cerr << "check_parsing: invalid rank " << rank << " (phrases size=" << all_phrases.size() << ")\n";
            return false;
        }

        std::string expected_phrase = all_phrases[static_cast<std::size_t>(rank)];

        if (first_phrase) {
            if (!expected_phrase.empty()) {
                expected_phrase.erase(0, 1);
            }
            first_phrase = false;
        }

        std::ptrdiff_t comp_len = static_cast<std::ptrdiff_t>(expected_phrase.size()) - static_cast<std::ptrdiff_t>(params.window_size);
        if (comp_len <= 0) {
            // Nothing to compare for this phrase
            continue;
        }

        if (index + static_cast<std::size_t>(comp_len) > file_data.size()) {
            std::cerr << "check_parsing: file buffer too small for reconstructed phrases (needed " << (index + comp_len)
                      << ", have " << file_data.size() << ")\n";
            return false;
        }

        for (std::ptrdiff_t k = 0; k < comp_len; ++k) {
            char expected_c = expected_phrase[static_cast<std::size_t>(k)];
            char file_c = file_data[index];
            if (file_c == expected_c) {
                ++matches;
            } else {
                ++missmatches;
                correct = false;
            }
            ++index;
        }
    }

    std::cout << "Total matches: " << matches << ", total mismatches: " << missmatches << "\n";
    return correct;
}
