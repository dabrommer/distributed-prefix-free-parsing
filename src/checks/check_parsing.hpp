#pragma once

#include <iostream>
#include <vector>

#include "kamping/communicator.hpp"
#include "util/cli_parser.hpp"

bool check_parsing(std::vector<int> const& ranks, Params const& params, std::vector<unsigned char> const& dict, kamping::Communicator<>& comm, char DELIMITER) {

    std::ifstream file(params.input_path, std::ios::binary);
    std::vector<char> file_data(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
    );

    std::vector<std::string> phrases;
    std::string current_phrase;
    for (auto c : dict) {
        if (c == DELIMITER) {
            phrases.push_back(current_phrase);
            current_phrase.clear();
        } else {
            current_phrase.push_back(c);
        }
    }

    bool correct = true;
    int index = 0;
    for (int rank : ranks) {

        std::string& expected_phrase = phrases[rank];
        if (rank == 0) {
            expected_phrase.erase(0,1); // Remove leading $
        }
        int          size            = expected_phrase.size() - params.window_size;
        for (int j = index; j < index + size; ++j) {
            auto expected = file_data[j];
            auto is = expected_phrase[j - index];
            correct = file_data[j] == expected_phrase[j - index];
        }
        index += size;
    }
    return correct;
}

