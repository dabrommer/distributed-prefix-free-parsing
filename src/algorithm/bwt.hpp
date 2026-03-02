#pragma once

#include <fstream>
#include <iostream>
#include <ranges>
#include <vector>

#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/printer.hpp>

#include "DDCX_lib/dss_interface.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/alltoall.hpp"
#include "kamping/communicator.hpp"
#include "util/dcx_caller.hpp"

using namespace kamping;

size_t get_pe_from_index(uint idx, size_t chunk_size, size_t left_over, int comm_size) {
    // uint == -1
    if (idx == -1) {
        return comm_size - 1;
    }
    if (idx < left_over * (chunk_size + 1)) {
        // first m bigger intervals
        return idx / (chunk_size + 1);
    } else {
        // last n-m smaller intervals
        return left_over + ((idx - left_over * (chunk_size + 1)) / chunk_size);
    }
}

template <typename char_type>
std::vector<uint32_t> compute_bwt(std::vector<char_type>& parse, Communicator<>& comm) {
    auto arg_strings = get_dcx_args();

    auto        sa_argc = static_cast<int32_t>(arg_strings.size());
    char const* sa_argv[sa_argc];

    for (int i = 0; i < sa_argc; ++i) {
        sa_argv[i] = arg_strings[i].c_str();
    }

    auto& timer = kamping::measurements::timer();

    timer.synchronize_and_start("Compute SA timer");
    auto& sa = get_sa(parse, comm, sa_argc, sa_argv);
    comm.barrier();
    timer.stop();

    timer.synchronize_and_start("BWT timer");

    size_t sa_size    = sa.size();
    auto   total_size = comm.allreduce_single(send_buf(parse.size()), kamping::op(kamping::ops::plus<>()));

    std::vector<std::vector<uint64_t>> requests(comm.size(), std::vector<uint64_t>());
    std::vector<char_type>             bwt;
    bwt.reserve(sa_size);

    auto offset     = 0;
    auto comm_size  = comm.size();
    auto chunk_size = total_size / comm.size();
    auto left_over  = total_size % comm.size();

    if (comm.rank_signed() < left_over)
        offset = comm.rank_signed() * (chunk_size + 1);
    else
        offset = left_over * (chunk_size + 1) + (comm.rank_signed() - left_over) * chunk_size;

    for (auto v: sa) {
        auto index = v.u64() - 1;

        auto max_value = offset + sa_size;

        if (index >= offset && index < max_value) {
            // index in local string
            continue;
        } else {
            // index on other pe
            auto pe = get_pe_from_index(index, chunk_size, left_over, comm_size);
            requests[pe].push_back(index);
        }
    }

    std::vector<int> send_counts;
    for (auto& x: requests) {
        send_counts.push_back(x.size());
    }

    auto             flatReqView = requests | std::ranges::views::join;
    std::vector<int> flatReq(flatReqView.begin(), flatReqView.end());

    auto [recv_requests, recv_counts] =
        comm.alltoallv(kamping::send_buf(flatReq), kamping::send_counts(send_counts), kamping::recv_counts_out());

    std::vector<std::vector<char_type>> response(comm.size(), std::vector<char_type>());

    int curr_pe_num = 0;
    int curr_pe     = 0;
    for (int curr: recv_requests) {
        if (curr_pe_num >= recv_counts[curr_pe]) {
            ++curr_pe;
            curr_pe_num = 0;
        }
        char_type c;
        if (curr == -1) {
            c = parse.back();
        } else {
            c = parse[curr - offset];
        }
        response[curr_pe].push_back(c);
    }

    auto                   flatRespView = response | std::ranges::views::join;
    std::vector<char_type> flatResp(flatRespView.begin(), flatRespView.end());

    auto [out_response, response_counts] =
        comm.alltoallv(kamping::send_buf(flatResp), kamping::send_counts(recv_counts), kamping::recv_counts_out());

    int                 rank = comm.rank_signed();
    std::vector<size_t> per_pe_counters(comm.size(), 0);

    for (auto idx: sa) {
        uint index = idx.u64();
        auto pe    = get_pe_from_index(index - 1, chunk_size, left_over, comm_size);

        if (pe == rank) {
            if (index == 0) {
                bwt.push_back(parse.back());
                continue;
            }
            bwt.push_back(parse[index - 1 - offset]);
        } else {
            size_t resp_offset = 0;
            for (int i = 0; i < pe; ++i) {
                resp_offset += response_counts[i];
            }
            auto pe_offset = per_pe_counters[pe];
            bwt.push_back(out_response[resp_offset + pe_offset]);
            per_pe_counters[pe]++;
        }
    }
    comm.barrier();
    timer.stop();
    if (comm.rank_signed() == 0) {
        std::cout << "BWT result size: " << bwt.size() << std::endl;
    }
    timer.aggregate_and_print(kamping::measurements::SimpleJsonPrinter<>(std::cout));

    return std::move(bwt);
}
