#include "DataDistribution.hpp"
#include "Parser.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/p2p/sendrecv.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/measurements/timer.hpp"
#include <kamping/measurements/printer.hpp>

#include <iostream>
#include <ranges>
#include <span>
#include <print>
#include <vector>



#include "rabin-karp.hpp"

template class std::vector<std::vector<char>> ;

struct Parse {
    std::vector<std::vector<char>> dict;
    std::vector<uint64_t> hashes;
};

struct phrase_vector {
    std::vector<char> phrases;
    std::vector<size_t> offsets;

    void insert_phrase(std::vector<char> const& phrase) {
        offsets.push_back(phrases.size());
        phrases.insert(phrases.end(), phrase.begin(), phrase.end());
    }

    explicit phrase_vector(std::vector<std::vector<char>> const& dict_set) {
        offsets.reserve(dict_set.size());
        for (auto c_vec : dict_set) {
            offsets.push_back(c_vec.size());
            phrases.insert(phrases.end(), c_vec.begin(), c_vec.end());
        }

    }
};

std::vector<int> compute_splitters(std::vector<char>& data, Communicator<>& comm, Params const& params) {
  rabin_karp rk(params.window_size);
  std::vector<int> splits;
  for (int i = 0; i < data.size(); i++) {
    uint64_t hash = rk.add_char(data[i]);

    // Make sure the window is filled
    if (i < params.window_size) {
        continue;
    }

    if (hash % params.p_mod == 0) {
      splits.push_back(i - params.window_size);
    }
  }
  return splits;
}

void update_parse(std::vector<std::vector<char>>& dict, rabin_karp& rk, std::vector<uint64_t>& hashes, const std::vector<char>& phrase) {
  uint64_t hash = rk.kr_print(phrase);
  hashes.push_back(hash);
  dict.push_back(phrase);
}

Parse compute_dict(std::vector<int> const& splits, std::vector<char> const& data, const Params& params, Communicator<>& comm) {
  std::vector<std::vector<char>> dict;
  rabin_karp kr(params.window_size);
  std::vector<uint64_t> hashes;
  hashes.reserve(splits.size());
  // Prepend $ to the first phrase
  if (comm.rank_signed() == 0) {
      std::vector<char> first_phrase;
      first_phrase.push_back('0');
      first_phrase.insert(first_phrase.end(), data.begin(), data.begin() + splits[0] + params.window_size);
      update_parse(dict, kr, hashes, first_phrase);
  }

  // Extract all phrases except the last one
  for (int i = 0; i < splits.size() - 1; i++) {
    auto begin = data.begin() + splits[i];
    auto end = data.begin() + splits[i + 1] + params.window_size;
    update_parse(dict, kr, hashes, std::vector<char>(begin, end));
  }

  // Debug for only one PE
  if (comm.size() == 1) {
    // Append params.window_size many '0' to the last phrase
    auto begin = data.begin() + splits.back();
    std::vector<char> last_phrase(begin, data.end());
    for (int i = 0; i < params.window_size; i++) {
        last_phrase.push_back('0');
    }
    update_parse(dict, kr, hashes, last_phrase);
    return Parse{dict, hashes};
  }

  // Send the chars before the first splitter to the previous PE
  int first_split = splits.front();
  std::vector<char> phrase_to_send(data.begin() + params.window_size, data.begin() + first_split + params.window_size);
  size_t prev_pe = comm.rank_shifted_cyclic(-1);

  // There is a dummy send recv between PE n and PE 0 to prevent UB caused by the implicit sendrecv done by kamping::sendrecv
  // Rank n only needs to send to n - 1, but its last phrase needs window_size many '0' appended
  if (comm.rank_signed() == comm.size_signed() - 1) {
    auto begin = data.begin() + splits.back();
    std::vector<char> last_phrase(begin, data.end());
    for (int i = 0; i < params.window_size; i++) {
        last_phrase.push_back('0');
    }
    // Only send needed
    comm.sendrecv<char>(send_buf(phrase_to_send), destination(prev_pe), source(0));
    update_parse(dict, kr, hashes, last_phrase);
    return Parse{dict, hashes};
  }

  int last_split = splits.back();
  std::vector<char> phrase;
  size_t next_pe = comm.rank_shifted_cyclic(1);
  std::vector<char> last_phrase(data.begin() + last_split, data.end());

  // Rank 0 only needs to receive
  if (comm.rank_signed() == 0) {
    comm.sendrecv(send_buf(ignore<char>), destination(prev_pe), source(next_pe), recv_buf<BufferResizePolicy::resize_to_fit>(phrase));
  }
  // All other ranks i send to i - 1 and receive from i + 1
  else {
    comm.sendrecv(send_buf(phrase_to_send), destination(prev_pe), recv_buf<BufferResizePolicy::resize_to_fit>(phrase), source(next_pe));
  }

  last_phrase.insert(last_phrase.end(), phrase.begin(), phrase.end());
  update_parse(dict, kr, hashes, last_phrase);

  return Parse{dict, hashes};
}

std::vector<std::string> sort_dict(std::vector<std::vector<char>> const& dict, Communicator<>& comm) {
    // Convert vector to phrase_vector
   phrase_vector pv(dict);
   std::vector<char> phrases;
   std::vector<size_t> offsets;
   comm.gatherv(send_buf(pv.phrases), root(0), recv_buf<BufferResizePolicy::resize_to_fit>(phrases));
   comm.gatherv(send_buf(pv.offsets), root(0), recv_buf<BufferResizePolicy::resize_to_fit>(offsets));
   if (comm.rank_signed() == 0) {
       std::vector<std::string> flat_dict;
       int c = 0;
       for (size_t size : offsets) {
           flat_dict.emplace_back(phrases.data() + c, size);
           c += static_cast<int>(size);
       }

       std::sort(flat_dict.begin(), flat_dict.end());
       return flat_dict;
   }

   return {};
}

std::vector<int> parse_ranks(Parse& parse, std::vector<std::string> const& sorted_dict, Communicator<>& comm, int window_size) {
    rabin_karp kr(window_size);
    std::vector<int> ranks;
    std::map<uint64_t, int> hashed_ranks;
    // request hashes from rank 0
    if (comm.rank_signed() == 0) {
        int rank = 0;
        for (const auto& str : sorted_dict) {
            hashed_ranks.insert({kr.kr_print(str), rank});
            ++rank;
        }
        for (auto h : parse.hashes) {
            ranks.push_back(hashed_ranks.at(h));
        }

        if (comm.size() == 1) {
            return ranks;
        }
    }
    auto result = comm.gatherv(send_buf(parse.hashes), root(0), recv_counts_out());
    auto recv_counts = result.extract_recv_counts();
    if (comm.rank_signed() == 0) {
        auto pe_hashes = result.get_recv_buffer();
        int offset = 0;
        int rank = 0;
        for (auto count : recv_counts) {
            if (rank == 0) {
                ++rank;
                continue;
            }
            std::vector<int> pe_ranks;
            for (int i = 0; i < count; ++i) {
                pe_ranks.push_back(hashed_ranks.at(pe_hashes[offset + i]));
            }
            comm.send(send_buf(pe_ranks), destination(rank));
            offset += count;
            ++rank;
        }
    }
    else {
        comm.recv(recv_buf<BufferResizePolicy::resize_to_fit>(ranks), source(0));
        return ranks;
    }
    return ranks;

}

std::vector<unsigned char> sort_dict(std::vector<std::vector<char>>& dict, Communicator<>& comm) {

   std::vector<unsigned char> phrases;
   // todo: Estimate avg length of phrases based on window_size and p
   phrases.reserve(dict.size() * 15 * 2);
   for (auto const& c_vec : dict) {
       phrases.insert(phrases.end(), c_vec.begin(), c_vec.end());
       phrases.push_back(DELIMITER);
   }

   dict.clear();
   auto result = run_sorter(phrases, comm);
   return result;
}

std::pair<std::map<uint64_t, int>, int> remove_duplicates(std::vector<unsigned char>& phrases, Communicator<>& comm) {
    uint64_t  first_hash = 0;
    bool first_hash_set = false;
    uint64_t prev_hash = 0;
    uint64_t hash = 0;
    int rank = 0;
    std::map<uint64_t, int> sorted_hashes;
    rabin_karp rk{};

    for (int i = 0; i < phrases.size(); ++i) {
        // End of phrase
        if (phrases[i] == DELIMITER) {
            if (!first_hash_set) {
                first_hash = hash;
                first_hash_set = true;
                prev_hash = hash;
                sorted_hashes.insert({hash, rank});
                ++rank;
            }
            else if (hash != prev_hash) {
                sorted_hashes.insert({hash, rank});
                prev_hash = hash;
                ++rank;
            }
            rk.reset();
        }
        else {
            hash = rk.add_char_fingerprint(phrases[i]);
        }
    }

    // Exchange last hashes to remove duplicates across PEs
    // Each PE sends its last hash to the next PE and compares it to its first hash
    auto next_pe = comm.rank_shifted_cyclic(1);
    auto prev_pe = comm.rank_shifted_cyclic(-1);
    auto res = comm.sendrecv<uint64_t>(send_buf(hash), destination(next_pe), source(prev_pe));
    if (res.front() == first_hash) {
        sorted_hashes.erase(first_hash);
    }

    // Compute offset for global ranks
    int size =  sorted_hashes.size();
    auto offset = comm.exscan_single(send_buf(size), op(kamping::ops::plus<>()));

    return {sorted_hashes, offset};
}

std::vector<char> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    std::vector<char> buffer(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
    );
    buffer.insert(buffer.begin(), '0');
    return buffer;
}

void printSizeHistogram(const std::vector<std::vector<char>>& data, Communicator<>& comm)
{
    std::map<std::size_t, std::size_t> counts;

    // Count sizes
    for (const auto& inner : data) {
        ++counts[inner.size()];
    }

    // Print JSON
    std::string out("{\n");

    for (auto it = counts.begin(); it != counts.end(); ++it) {
        out += ("  \"" + std::to_string(it->first) + "\": " + std::to_string(it->second));
        if (std::next(it) != counts.end())
            out += ",";
    }

    out += "}\n";

    std::vector<char> out_vec(out.begin(), out.end());

    auto outs = comm.gatherv(send_buf(out_vec), recv_counts_out());

    auto offsets = outs.extract_recv_counts();
    auto strings = outs.get_recv_buffer();
    int rank = 0;
    for (auto o : offsets) {
        std::string result(strings.begin(), strings.begin() + o);
        strings.erase(strings.begin(), strings.begin() + o);
        std::print("PE {}: \n", rank);
        std::print("{}\n", result);
        ++rank;
    }

}

void printOnRoot(std::string const& to_print, Communicator<>& comm)
{
    std::vector<char> out_vec(to_print.begin(), to_print.end());

    auto outs = comm.gatherv(send_buf(out_vec), recv_counts_out());

    auto offsets = outs.extract_recv_counts();
    auto strings = outs.get_recv_buffer();
    int rank = 0;
    for (auto o : offsets) {
        std::string result(strings.begin(), strings.begin() + o);
        strings.erase(strings.begin(), strings.begin() + o);
        std::print("{}\n", result);
        ++rank;
    }
}


int main(int argc, char const* argv[]) {
  kamping::Environment env;
  kamping::Communicator comm;



  Params params = read_parameters(argc, argv);

  std::string out_name = params.input_path+ "_n_" + std::to_string(comm.size()) + "_w_" + std::to_string(params.window_size) + "_p_" + std::to_string(params.p_mod) + ".txt";
  FILE* fp = std::freopen(out_name.c_str(), "w", stdout); // redirect stdout
  if (!fp) return 1;

  auto& timer = kamping::measurements::timer();

  timer.synchronize_and_start("Distribute data");

  // Distribute Data
  auto data = open_file(params.input_path, params.window_size, comm);

  timer.stop();

  
  timer.synchronize_and_start("Compute splitters");
  auto splits = compute_splitters(data, comm, params);
  timer.stop();

  auto total_splits = comm.allreduce_single(send_buf(splits.size()), kamping::op(kamping::ops::plus<>()));

  //std::print("PE {} has {:.2f} % of the splits ({}) \n", comm.rank(), (100.0 * splits.size()) / total_splits, splits.size());
  std::string to_print = std::format("PE {} has {:.2f} % of the splits ({}) \n", comm.rank(), (100.0 * splits.size()) / total_splits, splits.size());
  printOnRoot(to_print, comm);

  // Run hashes on each PE
  timer.synchronize_and_start("Compute dict");
  auto parse = compute_dict(splits, data, params, comm);
  timer.stop();

  //if (comm.rank_signed() == 0) {
      printSizeHistogram(parse.dict, comm);
  //}
  auto dict_size = comm.allreduce_single(send_buf(parse.dict.size()), kamping::op(kamping::ops::plus<>()));

  //std::print("PE {} has dictionary size {} \n", comm.rank(), parse.dict.size());
  if (comm.rank_signed() == 0) {
      std::print("Dict size is {:.2f}% of the total input size \n", (100.0 * dict_size * params.window_size) / data.size());
      std::print("Total dictionary size is {} phrases \n", dict_size);
  }


  // parse contains vector of phrases and their hashes (locally)
  // Sort phrases globally
  timer.synchronize_and_start("Global sort");
  auto sorted_dict = sort_dict(parse.dict, comm);
  timer.stop();

  // remove duplicates
  timer.synchronize_and_start("Remove duplicates");
  if (comm.rank_signed() == 0) {
      int prev = sorted_dict.size();
      auto ret = std::ranges::unique(sorted_dict.begin(), sorted_dict.end());
      sorted_dict.erase(ret.begin(), ret.end());
      int after = sorted_dict.size();
      std::print("Removed {} duplicates \n", prev - after);

  }
  timer.stop();

  // Swap hashes with lexicographical rank
  timer.synchronize_and_start("Exchange hashes with ranks");

  std::vector<int> lex_ranks = parse_ranks(parse, sorted_dict, comm, params.window_size);

  timer.stop();

  // Check the resulting parse
  if (comm.rank_signed() == 0) {
      std::vector<char> input = read_file(params.input_path);
      int pos = 0;
      int misses = 0;
      int rank_index = 0;
      for (auto rank : lex_ranks) {
          std::string cur = sorted_dict[rank];
          if (pos != 0) {
              cur.erase(0, params.window_size);
          }
          for (char & c : cur) {
              if (pos >= input.size()) {
                  break;
              }
              if (input[pos] != c) {
                  ++misses;
              }
              ++pos;
          }
          ++rank_index;
      }
  }

  std::ranges::sort(lex_ranks);
  auto unique_end = std::ranges::unique(lex_ranks).begin();
  int dups = lex_ranks.size() - std::distance(lex_ranks.begin(), unique_end);
  //std::print("PE {} parsing contains {} duplicates, the parsing has size {} \n", comm.rank_signed(), dups, lex_ranks.size());


  timer.aggregate_and_print(kamping::measurements::SimpleJsonPrinter<>(std::cout));
  std::fclose(fp);
}
