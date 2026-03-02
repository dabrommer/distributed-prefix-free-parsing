#pragma once

#include <numeric>
#include <vector>

#include "kamping/collectives/allgather.hpp"
#include "kamping/collectives/exscan.hpp"
#include "kamping/collectives/reduce.hpp"
#include "kamping/communicator.hpp"
#include "kamping/kassert/kassert.hpp"

using namespace kamping;

template <typename DataType>
MPI_Datatype get_big_type(size_t const size) {
    size_t mpi_max_int = std::numeric_limits<int>::max();
    size_t num_blocks  = size / mpi_max_int;
    size_t remainder   = size % mpi_max_int;

    MPI_Datatype result_type, block_type, blocks_type;
    MPI_Type_contiguous(mpi_max_int, kamping::mpi_datatype<DataType>(), &block_type);
    MPI_Type_contiguous(num_blocks, block_type, &blocks_type);

    if (remainder > 0) {
        MPI_Datatype leftover_type;
        MPI_Type_contiguous(remainder, kamping::mpi_datatype<DataType>(), &leftover_type);

        MPI_Aint lb, extent;
        MPI_Type_get_extent(kamping::mpi_datatype<DataType>(), &lb, &extent);
        MPI_Aint     displ       = num_blocks * mpi_max_int * extent;
        MPI_Aint     displs[2]   = {0, displ};
        std::int32_t blocklen[2] = {1, 1};
        MPI_Datatype mpitypes[2] = {blocks_type, leftover_type};
        MPI_Type_create_struct(2, blocklen, displs, mpitypes, &result_type);
        MPI_Type_commit(&result_type);
        MPI_Type_free(&leftover_type);
        MPI_Type_free(&blocks_type);
    } else {
        result_type = blocks_type;
        MPI_Type_commit(&result_type);
    }
    return result_type;
}

template <typename SendBuf>
auto alltoallv_direct(
    SendBuf&& send_buf, std::span<int64_t> send_counts, std::span<int64_t> recv_counts, Communicator<>& comm
) {
    using DataType = std::remove_reference_t<SendBuf>::value_type;
    std::vector<size_t> send_displs(comm.size()), recv_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), size_t{0});
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), size_t{0});

    auto const               recv_total = recv_displs.back() + recv_counts.back();
    std::vector<DataType>    receive_data(recv_total);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * comm.size());

    for (int i = 0; i < comm.size_signed(); ++i) {
        int source = (comm.rank_signed() + (comm.size_signed() - i)) % comm.size_signed();
        if (recv_counts[source] > 0) {
            auto receive_type = get_big_type<DataType>(recv_counts[source]);
            MPI_Irecv(
                receive_data.data() + recv_displs[source],
                1,
                receive_type,
                source,
                44227,
                comm.mpi_communicator(),
                &requests.emplace_back(MPI_REQUEST_NULL)
            );
        }
    }

    for (int i = 0; i < comm.size_signed(); ++i) {
        int target = (comm.rank_signed() + i) % comm.size_signed();
        if (send_counts[target] > 0) {
            auto send_type = get_big_type<DataType>(send_counts[target]);
            MPI_Issend(
                send_buf.data() + send_displs[target],
                1,
                send_type,
                target,
                44227,
                comm.mpi_communicator(),
                &requests.emplace_back(MPI_REQUEST_NULL)
            );
        }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    return receive_data;
}

template <typename SendBuf>
auto alltoallv_native(
    SendBuf&& send_buffer, std::span<int64_t> send_counts, std::span<int64_t> recv_counts, Communicator<>& comm
) {
    std::vector<int> send_counts_int{send_counts.begin(), send_counts.end()};
    std::vector<int> recv_counts_int{recv_counts.begin(), recv_counts.end()};

    return comm.alltoallv(
        kamping::send_buf(send_buffer),
        kamping::send_counts(send_counts_int),
        kamping::recv_counts(recv_counts_int)
    );
}

template <typename T, typename Operation>
T all_reduce(T& local_data, Operation operation, Communicator<>& comm) {
    // reduce returns result only on root process
    auto combined = comm.reduce(send_buf(local_data), op(operation, ops::commutative));
    T    combined_local;
    if (comm.rank() == 0) {
        combined_local = combined.front();
    }
    comm.bcast_single(send_recv_buf(combined_local));
    return combined_local;
}

template <typename T>
T all_reduce_sum(T local_data, kamping::Communicator<>& comm) {
    return all_reduce(local_data, kamping::ops::plus<>(), comm);
}

template <typename SendBuf>
auto alltoallv_combined(
    SendBuf&& send_buffer, std::span<int64_t> send_counts, std::span<int64_t> recv_counts, kamping::Communicator<>& comm
) {
    int64_t const send_total = std::accumulate(send_counts.begin(), send_counts.end(), int64_t{0});
    int64_t const recv_total = std::accumulate(recv_counts.begin(), recv_counts.end(), int64_t{0});
    int64_t const local_max  = std::max<int64_t>(send_total, recv_total);
    int64_t const global_max = comm.allreduce_single(kamping::send_buf(local_max), kamping::op(kamping::ops::max<>{}));

    if (global_max < std::numeric_limits<int>::max()) {
        return alltoallv_native(send_buffer, send_counts, recv_counts, comm);
    } else {
        return alltoallv_direct(send_buffer, send_counts, recv_counts, comm);
    }
}

template <typename SendBuf>
auto alltoallv_combined(SendBuf&& send_buffer, std::span<int64_t> send_counts, Communicator<>& comm) {
    auto recv_counts = comm.alltoall(send_buf(send_counts));
    return alltoallv_combined(std::forward<SendBuf>(send_buffer), send_counts, recv_counts, comm);
}

template <typename DataType>
std::vector<DataType>
distribute_data_custom(std::vector<DataType>& local_data, int64_t local_target_size, kamping::Communicator<>& comm) {
    int64_t num_processes = comm.size();
    int64_t local_size    = local_data.size();

    bool check = all_reduce_sum(local_size, comm) == all_reduce_sum(local_target_size, comm);
    if (!check) {
        std::cout << "total and target size don't match" << std::endl;
    }

    std::vector<int64_t> target_sizes = comm.allgather(kamping::send_buf(local_target_size));
    std::vector<int64_t> preceding_target_size(num_processes);
    std::exclusive_scan(target_sizes.begin(), target_sizes.end(), preceding_target_size.begin(), int64_t(0));

    int64_t local_data_size = local_data.size();
    int64_t preceding_size  = comm.exscan(kamping::send_buf(local_size), kamping::op(kamping::ops::plus<>{}))[0];

    std::vector<int64_t> send_cnts(num_processes, 0);
    for (int64_t cur_rank = 0; cur_rank < num_processes - 1 && local_data_size > 0; cur_rank++) {
        int64_t to_send     = std::max(int64_t(0), preceding_target_size[cur_rank + 1] - preceding_size);
        to_send             = std::min(to_send, local_data_size);
        send_cnts[cur_rank] = to_send;
        local_data_size -= to_send;
        preceding_size += to_send;
    }
    send_cnts.back() += local_data_size;

    std::vector<DataType> result = alltoallv_combined(local_data, send_cnts, comm);
    return result;
}

template <typename DataType>
std::vector<DataType> redistribute_block_balanced(std::vector<DataType>& local_data, kamping::Communicator<>& comm) {
    auto total_size = comm.allreduce_single(send_buf(local_data.size()), kamping::op(kamping::ops::plus<>()));
    auto slice_size = total_size / comm.size();
    auto res_size   = total_size % comm.size();

    auto local_target_size = slice_size + (comm.rank() < res_size ? 1 : 0);

    return distribute_data_custom(local_data, local_target_size, comm);
}
