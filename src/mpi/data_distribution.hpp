#pragma once
#include <string>

#include <sys/stat.h>

#include "kamping/communicator.hpp"

using namespace kamping;

bool file_exists(std::string const& input_path) {
    struct stat buffer;
    return (stat(input_path.c_str(), &buffer) == 0);
}

std::vector<char> open_file(std::string const& input_path, int window_size, Communicator<>& comm) {
    if (!file_exists(input_path)) {
        if (comm.rank() == 0) {
            std::cerr << "File " << input_path << " does not exist!" << std::endl;
        }
        exit(1);
    }

    MPI_File input_file;
    MPI_File_open(comm.mpi_communicator(), input_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);

    MPI_Offset file_size;
    MPI_File_get_size(input_file, &file_size);

    int comm_size = static_cast<int>(comm.size());
    int comm_rank = static_cast<int>(comm.rank_signed());

    MPI_Offset local_slice_size = file_size / comm_size;
    MPI_Offset larger_slices    = file_size % comm_size;

    MPI_Offset offset;
    if (comm.rank() < larger_slices) {
        ++local_slice_size;
        offset = local_slice_size * comm_rank;
    } else {
        offset = larger_slices * (local_slice_size + 1);
        offset += (comm_rank - larger_slices) * local_slice_size;
    }

    MPI_File_seek(input_file, offset, MPI_SEEK_SET);

    int read_count;
    if (comm.rank() == comm.size() - 1) {
        read_count = static_cast<int>(local_slice_size);
    } else {
        read_count = static_cast<int>(local_slice_size + window_size - 1);
    }

    std::vector<char> data(read_count);
    MPI_File_read(input_file, data.data(), read_count, MPI_CHAR, MPI_STATUS_IGNORE);

    return data;
}
