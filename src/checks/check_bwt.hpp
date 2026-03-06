#pragma once

#include <fstream>

#include "kamping/collectives/allgather.hpp"

using namespace kamping;

inline bool check_bwt(std::string& input_data, kamping::Communicator<>& comm, std::vector<unsigned char>& bwt_local, std::string& bwt_exec_path)
{
    // todo distribute this check
    auto bwt = comm.allgatherv(send_buf(bwt_local));
    bool correct = true;

    if (comm.is_root()) {
        std::string command = bwt_exec_path + " " + input_data;
        std::system(command.c_str());


        std::ifstream file(input_data + ".bwt", std::ios::binary | std::ios::ate);
        if (!file)
        {
            std::cout << "Could not open BWT file for checking: " << input_data + ".bwt" << std::endl;
            return false;
        }

        auto correct_bwt_size = file.tellg();
        if (bwt.size() != correct_bwt_size)
        {
            std::cout << "BWT size mismatch in PE " << comm.rank_signed() << ": expected " << correct_bwt_size << ", got " << bwt.size() << std::endl;
            return false;
        }

        file.seekg(0, std::ios::beg);



        int count = 0;
        int missmatches = 0;


        char correct_char;
        while (file.get(correct_char) && count < bwt.size()) {
            if (bwt[count] == correct_char) {
                //continue;
            }
            else {
                std::cout << "BWT mismatch in PE " << comm.rank_signed() << " at position " << count << ": expected " << correct_char << ", got " << bwt[count] << std::endl;
                correct = false;
                ++missmatches;
                if (missmatches >= 10)
                {
                    std::cout << "Too many mismatches, stopping check" << std::endl;
                    return correct;
                }
            }
            ++count;
        }
    }
    return correct;


}