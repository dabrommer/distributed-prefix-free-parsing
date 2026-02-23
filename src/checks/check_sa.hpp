#pragma once

#include <vector>
#include <cstdint>
#include "kamping/communicator.hpp"
#include "kamping/collectives/gather.hpp"

#include "DDCX_lib/dss_interface.hpp"

using namespace kamping;

template<typename SAType, typename T>
bool check_sa(std::vector<SAType>& sa, T& data, Communicator<>& comm) {

    // Gather SA and data on root
    auto sa_complete = comm.gatherv(send_buf(sa));
    auto data_complete = comm.gatherv(send_buf(data));
    if (!comm.is_root()) {
        return true;
    }
    // Check SA on root
    int index = 0;
    uint64_t prev_sa_index;
    if constexpr (std::is_same_v<SAType, dsss::UIntPair<uint8_t>>) {
        prev_sa_index = sa_complete.front().u64();
    }
    else {
        prev_sa_index = sa_complete.front();
    }
    for (auto& sa_value: sa_complete) {
        std::string prev_string;
        std::string curr_string;
        uint64_t curr_sa_index;
        if constexpr (std::is_same_v<SAType, dsss::UIntPair<uint8_t>>) {
            curr_sa_index = sa_value.u64();
        }
        else {
            curr_sa_index = sa_value;
        }
        for (auto [prev, curr] = std::tuple{prev_sa_index, curr_sa_index}; prev < data_complete.size() && curr < data_complete.size(); ++prev, ++curr)  {
            prev_string.push_back(static_cast<char>(data_complete[prev]));
            curr_string.push_back(static_cast<char>(data_complete[curr]));
            if (data_complete[prev] == data_complete[curr]) {
                continue;
            } else if (data_complete[prev] < data_complete[curr]) {
                break;
            }
            else {
                return false;
            }

        }
        ++index;
        prev_sa_index = curr_sa_index;

    }

    return true;

}