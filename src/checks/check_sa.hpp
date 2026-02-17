#pragma once

#include <vector>
#include <cstdint>
#include "kamping/communicator.hpp"
#include "kamping/collectives/gather.hpp"

#include "DDCX_lib/dss_interface.hpp"

using namespace kamping;

bool check_sa(std::vector<dsss::UIntPair<uint8_t>>& sa, std::vector<uint32_t>& data, Communicator<>& comm) {

    // Gather SA and data on root
    auto sa_complete = comm.gatherv(send_buf(sa));
    auto data_complete = comm.gatherv(send_buf(data));
    if (!comm.is_root()) {
        return true;
    }
    // Check SA on root
    int index = 0;
    uint64_t prev_sa_index = sa_complete.front().u64();
    for (auto const& pair: sa_complete) {
        auto curr_sa_index = pair.u64();
        for (auto [prev, curr] = std::tuple{prev_sa_index, curr_sa_index}; prev < data_complete.size() && curr < data_complete.size(); ++prev, ++curr)  {

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