#pragma once

#include <iostream>
#include <vector>

#include "kamping/communicator.hpp"
#include "util/cli_parser.hpp"
#include "util/pair.hpp"

bool check_sort_unique(std::vector<unsigned char> const& to_check, const unsigned char DELIMITER) {
    std::string prev;
    std::string curr;

    for (auto const c: to_check) {
        curr.push_back(c);
        if (c == DELIMITER) {
            int check = curr.compare(prev);
            if (check < 0) {
                std::cout << "Sorting check failed: '" << curr << "' is less than previous phrase '" << prev << "'\n";
                return false;
            }
            if (check == 0) {
                std::cout << "Duplicate phrase found: '" << curr << "'\n";
                return false;
            }
            prev = curr;
            curr = "";
        }
    }
    return true;
}


inline bool check_parsing(std::vector<uint32_t> const& ranks, Params const& params, std::vector<unsigned char> const& phrases, kamping::Communicator<>& comm, const unsigned char DELIMITER) {

    auto ranks_complete = comm.gatherv(kamping::send_buf(ranks));
    auto phrases_complete = comm.gatherv(kamping::send_buf(phrases));

    if (!comm.is_root()) {
        return true;
    }


    bool sorted = check_sort_unique(phrases_complete, DELIMITER);
    if (!sorted) {
        std::cout << "check_parsing: phrases are not sorted or contain duplicates\n";
        return false;
    }

    std::ifstream file(params.input_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open input file '" << params.input_path << "' for checking\n";
        return false;
    }
    std::vector<char> file_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<std::string> all_phrases;
    std::string curr_str;
    for (const auto& c : phrases_complete) {
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

    for (int rank : ranks_complete) {
        // Rank is 1-based
        if (rank < 1 || static_cast<std::size_t>(rank - 1) >= all_phrases.size()) {
            std::cerr << "check_parsing: invalid rank " << rank << " (phrases size=" << all_phrases.size() << ")\n";
            return false;
        }

        std::string expected_phrase = all_phrases[static_cast<std::size_t>(rank - 1)];

        // Erase leading # from the frist phrase
        if (first_phrase) {
            expected_phrase.erase(0, 1);
            first_phrase = false;
        }

        auto comp_len = expected_phrase.size() - params.window_size;

        if (index + comp_len > file_data.size()) {
            std::cout << "check_parsing: file buffer too small for reconstructed phrases (needed " << (index + comp_len)
                      << ", have " << file_data.size() << ")\n";
            return false;
        }

        for (size_t k = 0; k < comp_len; ++k) {
            char expected_c = expected_phrase[k];
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


bool check_duplicates(std::vector<Pair> const& hash_pairs) {
    for (size_t i = 1; i < hash_pairs.size(); ++i) {
        if (hash_pairs[i].hash == hash_pairs[i - 1].hash) {
            std::cout << "Duplicate hash found at indices " << (i - 1) << " and " << i << ": hash=" << hash_pairs[i].hash
                      << ", ranks=" << hash_pairs[i - 1].rank << " and " << hash_pairs[i].rank << "\n";
            return false;
        }
        if (hash_pairs[i].rank != hash_pairs[i - 1].rank + 1) {
            std::cout << "Non-consecutive ranks found at indices " << (i - 1) << " and " << i << ": hash=" << hash_pairs[i].hash
                      << ", ranks=" << hash_pairs[i - 1].rank << " and " << hash_pairs[i].rank << "\n";
            return false;
        }
    }
    return true;
}

bool check_sort(std::vector<unsigned char> const& to_check, const unsigned char DELIMITER) {
    std::string prev;
    std::string curr;

    for (auto const c: to_check) {
        curr.push_back(c);
        if (c == DELIMITER) {
            int check = curr.compare(prev);
            if (check < 0) {
                std::cout << "Sorting check failed: '" << curr << "' is less than previous phrase '" << prev << "'\n";
                return false;
            }
            prev = curr;
            curr = "";
        }
    }
    return true;
}

bool check_sort_unique_global(std::vector<Pair> const& to_check, kamping::Communicator<>& comm) {
    auto pairs = comm.gatherv(kamping::send_buf(to_check));
    if (!comm.is_root()) {
        return true;
    }

    uint64_t prev_hash = 0;

    for (auto const pair: pairs) {
        if (prev_hash > pair.hash) {
            std::cout << "Global sorting check failed: hash " << pair.hash << " is less than previous hash " << prev_hash
                      << ", rank=" << pair.rank << "\n";
            return false;
        }
        if (prev_hash == pair.hash) {
            std::cout << "Global sorting check failed: duplicate hash " << pair.hash << " found with ranks " << pair.rank
                      << " and " << (pair.rank - 1) << "\n";
            return false;
        }
        prev_hash = pair.hash;
    }
    return true;
}

