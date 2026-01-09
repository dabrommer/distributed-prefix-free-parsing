#include "DataDistribution.hpp"
#include "Parser.hpp"
#include "spdlog/spdlog.h"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/p2p/sendrecv.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/measurements/timer.hpp"
#include <kamping/measurements/printer.hpp>

#include <iostream>
#include <ranges>

#include "rabin-karp.hpp"

std::vector<int> compute_splitters(std::vector<char>& data, Communicator<>& comm, Params params) {
  rabin_karp rk(params.window_size);
  std::vector<int> splits;
  for (int i = 0; i < data.size(); i++) {
    if (rk.add_char(data[i]) % params.p_mod == 0) {
      splits.push_back(i - params.window_size);
    }
  }
  return splits;
}

std::set<std::string> compute_dict(std::vector<int> const& splits, std::vector<char> const& data, Params params, Communicator<>& comm) {
  std::set<std::string> dict;
  for (int i = 0; i < splits.size() - 1; i++) {
    std::string phrase(data.begin() + splits[i], data.begin() + splits[i + 1] + params.window_size);
    dict.insert(phrase);
  }

  if (comm.size() == 1) {
    return dict;
  }

  int last_split = splits.back();
  std::vector<char> phrase_to_send(data.begin() + params.window_size, data.begin() + last_split + params.window_size);
  size_t prev_pe = comm.rank_shifted_cyclic(-1);

  // Rank n only needs to send to n - 1
  if (comm.rank_signed() == comm.size_signed() - 1) {
    comm.send(send_buf(phrase_to_send), destination(prev_pe));
    return dict;
  }

  std::vector<char> phrase;
  size_t next_pe = comm.rank_shifted_cyclic(1);
  std::string last_phrase(data.begin() + last_split, data.end());


  // Rank 0 only needs to receive
  if (comm.rank_signed() == 0) {
    comm.recv(recv_buf(phrase));
  }
  // All other ranks i send to i - 1 and receive from i + 1
  else {
    comm.sendrecv<char>(send_buf(phrase_to_send), destination(prev_pe), recv_buf<BufferResizePolicy::resize_to_fit>(phrase), source(next_pe));
  }

  std::string phrase_str(phrase.begin(), phrase.end());
  last_phrase += phrase_str;
  dict.insert(phrase_str);

  return dict;
}


int main(int argc, char const* argv[]) {
  kamping::Environment env;
  kamping::Communicator comm;

  Params params = read_parameters(argc, argv);

  auto& timer = kamping::measurements::timer();

  timer.synchronize_and_start("Distribute data");

  // Distribute Data
  auto data = open_file(params.input_path, params.window_size, comm);

  timer.stop();



  timer.synchronize_and_start("Compute splitters");
  auto splits = compute_splitters(data, comm, params);
  timer.stop();

  auto total_splits = comm.allreduce_single(send_buf(splits.size()), kamping::op(kamping::ops::plus<>()));

  spdlog::info("PE {} has {}% of the splits ({})", comm.rank(), (100.0 * splits.size()) / total_splits, splits.size());


  // Run hashes on each PE
  timer.synchronize_and_start("Compute dict");
  auto dict = compute_dict(splits, data, params, comm);
  timer.stop();
  spdlog::info("PE {} has dictionary size {}, which is {} less then no of splits", comm.rank(), dict.size(), splits.size() - dict.size());
  // Unify Results

  // Check Results


  timer.aggregate_and_print(kamping::measurements::SimpleJsonPrinter<>(std::cout));
}
