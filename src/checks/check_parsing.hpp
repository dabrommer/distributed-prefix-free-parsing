#pragma once

#include <iostream>
#include <vector>

#include "kamping/communicator.hpp"
#include "util/cli_parser.hpp"
#include "util/pair.hpp"
#include "util/logger.hpp"

using namespace logs;

bool check_sort_unique(std::vector<unsigned char> const& to_check, const unsigned char DELIMITER) {
    std::string prev;
    std::string curr;

    for (auto const c: to_check) {
        curr.push_back(c);
        if (c == DELIMITER) {
            int check = curr.compare(prev);
            if (check < 0) {
                logger::print_on_root("Sorting check failed: '{}' is less than previous phrase '{}'", curr, prev);
                return false;
            }
            if (check == 0) {
                logger::print_on_root("Duplicate phrase found: '{}'", curr);
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
        logger::print_on_root("check_parsing: phrases are not sorted or contain duplicates");
        return false;
    }

    std::ifstream file(params.input_path, std::ios::binary);
    if (!file) {
        logger::print_on_root("Failed to open input file '{}' for checking", params.input_path);
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
            logger::print_on_root("check_parsing: invalid rank {} (phrases size={})", rank, all_phrases.size());
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
            logger::print_on_root("check_parsing: file buffer too small for reconstructed phrases (needed {}, have {})", (index + comp_len), file_data.size());
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

    logger::print_on_root("Total matches: {}, total mismatches: {}", matches, missmatches);
    return correct;
}


bool check_duplicates(std::vector<Pair> const& hash_pairs) {
    for (size_t i = 1; i < hash_pairs.size(); ++i) {
        if (hash_pairs[i].hash == hash_pairs[i - 1].hash) {
            logger::print_on_root("Duplicate hash found at indices {} and {}: hash={}, ranks={} and {}", (i - 1), i, hash_pairs[i].hash, hash_pairs[i - 1].rank, hash_pairs[i].rank);
            return false;
        }
        if (hash_pairs[i].rank != hash_pairs[i - 1].rank + 1) {
            logger::print_on_root("Non-consecutive ranks found at indices {} and {}: hash={}, ranks={} and {}", (i - 1), i, hash_pairs[i].hash, hash_pairs[i - 1].rank, hash_pairs[i].rank);
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
                logger::print_on_root("Sorting check failed: '{}' is less than previous phrase '{}'", curr, prev);
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
            logger::print_on_root("Global sorting check failed: hash {} is less than previous hash {}, rank={}", pair.hash, prev_hash, pair.rank);
            return false;
        }
        if (prev_hash == pair.hash) {
            logger::print_on_root("Global sorting check failed: duplicate hash {} found with ranks {} and {}", pair.hash, pair.rank, (pair.rank - 1));
            return false;
        }
        prev_hash = pair.hash;
    }
    return true;
}

