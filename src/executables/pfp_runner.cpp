#include "../../external/scalable-distributed-string-sorting/src/executables/sort_caller.cpp"
#include <iostream>
#include <ranges>
#include <vector>

#include "AmsSort/AmsSort.hpp"
#include "checks/check_parsing.hpp"
#include "hash/rabin-karp.hpp"
#include "kamping/collectives/allreduce.hpp"
#include "kamping/collectives/gather.hpp"
#include "kamping/measurements/printer.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/p2p/recv.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/p2p/sendrecv.hpp"
#include "mpi/data_distribution.hpp"
#include "util/cli_parser.hpp"
#include "util/logger.hpp"

constexpr unsigned char DELIMITER = 0;
constexpr unsigned char DOLLAR    = 1;

struct Parse {
    std::vector<unsigned char> dict;
    std::vector<uint64_t>      hashes;
};

struct Pair {
    uint64_t hash;
    int      rank;

    Pair(uint64_t key, int value) : hash(key), rank(value) {}
    Pair() = default;
};

struct phrase_vector {
    std::vector<char>   phrases;
    std::vector<size_t> offsets;

    void insert_phrase(std::vector<char> const& phrase) {
        offsets.push_back(phrases.size());
        phrases.insert(phrases.end(), phrase.begin(), phrase.end());
    }

    explicit phrase_vector(std::vector<std::vector<char>> const& dict_set) {
        offsets.reserve(dict_set.size());
        for (auto c_vec: dict_set) {
            offsets.push_back(c_vec.size());
            phrases.insert(phrases.end(), c_vec.begin(), c_vec.end());
        }
    }
};

std::vector<int> compute_splitters(std::vector<char>& data, Communicator<>& comm, Params const& params) {
    rabin_karp       rk(params.window_size);
    std::vector<int> splits;
    for (size_t i = 0; i < data.size(); i++) {
        uint64_t hash = rk.add_char(data[i]);

        if (data[i] == DOLLAR || data[i] == DELIMITER) {
            std::cout << "Found char " << data[i] << " in data at position " << i << " on PE " << comm.rank_signed() << "\n";
        }

        // Make sure the window is filled
        if (i < params.window_size) {
            continue;
        }

        if (hash % params.p_mod == 0) {
            splits.push_back(static_cast<int>(i) - params.window_size);
        }
    }
    return splits;
}

void update_parse(
    std::vector<unsigned char>&       dict,
    rabin_karp&                       rk,
    std::vector<uint64_t>&            hashes,
    std::vector<unsigned char> const& phrase
) {
    uint64_t const hash = rk.kr_print(phrase);
    hashes.push_back(hash);
    dict.insert(dict.end(), phrase.begin(), phrase.end());
    dict.push_back(DELIMITER);
}

Parse compute_dict(
    std::vector<int> const& splits, std::vector<char> const& data, Params const& params, Communicator<>& comm
) {
    std::vector<unsigned char> dict;
    rabin_karp                 kr(params.window_size);
    std::vector<uint64_t>      hashes;
    hashes.reserve(splits.size());

    // Prepend $ to the first phrase
    if (comm.rank_signed() == 0) {
        std::vector<unsigned char> first_phrase;
        first_phrase.push_back(DOLLAR);
        first_phrase.insert(first_phrase.end(), data.begin(), data.begin() + splits[0] + params.window_size);
        update_parse(dict, kr, hashes, first_phrase);

    }

    // Extract all phrases except the last one (safe loop)
    for (size_t i = 0; i + 1 < splits.size(); ++i) {
        auto begin = data.begin() + splits[i];
        auto end   = data.begin() + splits[i + 1] + params.window_size;
        update_parse(dict, kr, hashes, std::vector<unsigned char>(begin, end));
    }

    // Debug for only one PE
    if (comm.size() == 1) {
        // Append params.window_size many $ to the last phrase
        auto                       begin = data.begin() + splits.back();
        std::vector<unsigned char> last_phrase(begin, data.end());
        for (int i = 0; i < params.window_size; i++) {
            last_phrase.push_back(DOLLAR);
        }
        update_parse(dict, kr, hashes, last_phrase);
        return Parse{dict, hashes};
    }

    // Send the chars before the first splitter to the previous PE
    int                        first_split = splits.front();
    std::vector<unsigned char> phrase_to_send(
        data.begin() + params.window_size,
        data.begin() + first_split + params.window_size
    );
    size_t prev_pe = comm.rank_shifted_cyclic(-1);

    // There is a dummy send recv between PE n and PE 0 to prevent UB caused by the implicit sendrecv done by
    // kamping::sendrecv Rank n only needs to send to n - 1, but its last phrase needs window_size many $ appended
    if (comm.rank_signed() == comm.size_signed() - 1) {
        auto                       begin = data.begin() + splits.back();
        std::vector<unsigned char> last_phrase(begin, data.end());
        for (int i = 0; i < params.window_size; i++) {
            last_phrase.push_back(DOLLAR);
        }
        // Only send needed
        comm.sendrecv<unsigned char>(send_buf(phrase_to_send), destination(prev_pe), source(0));
        update_parse(dict, kr, hashes, last_phrase);
        return Parse{dict, hashes};
    }

    int                        last_split = splits.back();
    std::vector<char>          phrase;
    size_t                     next_pe = comm.rank_shifted_cyclic(1);
    std::vector<unsigned char> last_phrase(data.begin() + last_split, data.end());

    // Rank 0 only needs to receive
    if (comm.rank_signed() == 0) {
        comm.sendrecv(
            send_buf(ignore<unsigned char>),
            destination(prev_pe),
            source(next_pe),
            recv_buf<BufferResizePolicy::resize_to_fit>(phrase)
        );
    }
    // All other ranks i send to i - 1 and receive from i + 1
    else {
        comm.sendrecv(
            send_buf(phrase_to_send),
            destination(prev_pe),
            recv_buf<BufferResizePolicy::resize_to_fit>(phrase),
            source(next_pe)
        );
    }

    last_phrase.insert(last_phrase.end(), phrase.begin(), phrase.end());
    update_parse(dict, kr, hashes, last_phrase);

    return Parse{dict, hashes};
}

bool check_sort(std::vector<unsigned char> const& to_check) {
    std::string prev;
    std::string curr;

    for (auto const c: to_check) {
        curr.push_back(c);
        if (c == DELIMITER) {
            int check = curr.compare(prev);
            if (check < 0) {
                return false;
            }
            prev = curr;
            curr = "";
        }
    }
    return true;
}

std::vector<unsigned char> sort_dict(std::vector<unsigned char>& dict, Communicator<>& comm) {
    auto result = run_sorter(dict, comm);
    return result;
}

std::pair<std::vector<Pair>, size_t> remove_duplicates(std::vector<unsigned char>& phrases, Communicator<>& comm, int window_size) {
    uint64_t          first_hash     = 0;
    bool              first_hash_set = false;
    uint64_t          prev_hash      = 0;
    uint64_t          hash           = 0;
    int               rank           = 0;
    std::vector<Pair> sorted_hashes;
    rabin_karp        rk{window_size};
    std::vector<unsigned char> curr_phrase;
    std::vector<unsigned char> unique_phrases;

    for (const unsigned char c: phrases) {
        // End of phrase
        if (c == DELIMITER) {
            // Compute hash for the full phrase
            hash = rk.kr_print(curr_phrase);

            if (!first_hash_set) {
                first_hash     = hash;
                first_hash_set = true;
                prev_hash      = hash;
                sorted_hashes.emplace_back(hash, rank);
                ++rank;
                unique_phrases.insert(
                    unique_phrases.end(),
                    curr_phrase.begin(),
                    curr_phrase.end()
                );
                unique_phrases.push_back(DELIMITER);
            } else if (hash != prev_hash) {
                sorted_hashes.emplace_back(hash, rank);
                prev_hash = hash;
                ++rank;
                unique_phrases.insert(
                    unique_phrases.end(),
                    curr_phrase.begin(),
                    curr_phrase.end()
                );
                unique_phrases.push_back(DELIMITER);
            }
            curr_phrase.clear();
            rk.reset();
            hash = 0;
        } else {
            curr_phrase.push_back(c);
        }
    }

    phrases.swap(unique_phrases);

    // Exchange last hashes to remove duplicates across PEs
    // Each PE sends its last hash to the next PE and compares it to its first hash
    if (comm.size() > 1) {
        auto next_pe = comm.rank_shifted_cyclic(1);
        auto prev_pe = comm.rank_shifted_cyclic(-1);
        auto res     = comm.sendrecv<uint64_t>(send_buf(prev_hash), destination(next_pe), source(prev_pe));
        if (res.front() == first_hash) {
            sorted_hashes.erase(sorted_hashes.begin());
        }
    }

    // Compute offset for global ranks
    auto  size   = sorted_hashes.size();
    auto offset = comm.exscan_single(send_buf(size), op(kamping::ops::plus<>()));

    return {sorted_hashes, offset};
}

MPI_Datatype create_pair_type() {
    MPI_Datatype pair_type;

    int blocklengths[2] = {1, 1};

    MPI_Datatype types[2] = {MPI_UINT64_T, MPI_INT};
    MPI_Aint     displacements[2];
    displacements[0] = offsetof(Pair, hash);
    displacements[1] = offsetof(Pair, rank);

    MPI_Type_create_struct(2, blocklengths, displacements, types, &pair_type);
    MPI_Type_commit(&pair_type);

    return pair_type;
}

std::pair<std::unordered_map<uint64_t, int>, uint64_t>
sort_hashes(std::vector<Pair>& hash_vec, int offset, Communicator<>& comm) {
    int const kway      = 64;
    auto      pair_comp = [](Pair const& a, Pair const& b) {
        return a.hash < b.hash;
    };
    std::random_device rd;
    std::mt19937_64    gen(rd());

    Ams::sort(create_pair_type(), hash_vec, kway, gen, comm.mpi_communicator(), pair_comp);

    std::unordered_map<uint64_t, int> map;
    for (auto const& p: hash_vec) {
        map.insert({p.hash, p.rank});
    }
    return {map, hash_vec.back().hash};
}


std::vector<int> exchange_hashes(
    std::unordered_map<uint64_t, int>& hash_vec,
    std::vector<uint64_t> const&       hashes,
    uint64_t                           last_hash,
    Communicator<> const&              comm
) {
    auto                               border_hashes = comm.allgather(send_buf(last_hash));
    std::vector<std::vector<uint64_t>> hashes_to_request(comm.size());
    std::vector<int>                   pe_order;
    for (auto const& h: hashes) {
        auto const it = std::ranges::lower_bound(border_hashes, h);
        auto const dist = std::distance(border_hashes.begin(), it);
        // distance returns a signed type; cast to size_t and clamp to last peer if it == border_hashes.size()
        size_t pe = static_cast<size_t>(dist);
        if (pe >= border_hashes.size()) {
            pe = border_hashes.size() - 1;
        }
        hashes_to_request[pe].push_back(h);
        pe_order.push_back(static_cast<int>(pe));
    }

    // Compute size_v for alltoallv
    std::vector<int> size_v;
    for (auto const& vec: hashes_to_request) {
        size_v.push_back(vec.size());
    }
    // Flatten the request vector
    auto                  flat_requests = hashes_to_request | std::views::join;
    std::vector<uint64_t> flat_hashes(flat_requests.begin(), flat_requests.end());

    auto requests = comm.alltoallv(send_buf(flat_hashes), send_counts(size_v), recv_counts_out());

    auto const recv_counts = requests.extract_recv_counts();
    auto const recv_buf    = requests.get_recv_buffer();


    int              rank  = 0;
    int              count = 0;
    std::vector<int> responses;
    // Build responses in the same order as recv_buf
    responses.reserve(recv_buf.size());
    for (auto const& hash: recv_buf) {
        auto it = hash_vec.find(hash);
        if (it != hash_vec.end()) {
            responses.push_back(it->second);
        } else {
            // Unexpected: requested hash not found in the local hash map
            responses.push_back(-1);
        }
    }

    // The number of responses to send back to each rank is exactly the counts we received from them.
    std::vector<int> response_size_v = recv_counts;
    // Send back the responses
    auto             result = comm.alltoallv(send_buf(responses), send_counts(response_size_v));

    // Compute starting offsets into the flattened requests vector for each peer
    std::vector<int> offsets(comm.size());
    int              off = 0;
    for (size_t peer = 0; peer < hashes_to_request.size(); ++peer) {
        offsets[peer] = off;
        off += static_cast<int>(hashes_to_request[peer].size());
    }
    std::vector<int> final_ranks;
    final_ranks.reserve(pe_order.size());
    for (auto i: pe_order) {
        int curr_rank = result[offsets[i]];
        final_ranks.push_back(curr_rank);
        offsets[i]++;
    }

    return final_ranks;
}

int main(int argc, char const* argv[]) {
    kamping::Environment  env;
    kamping::Communicator comm;

    Params params = read_parameters(argc, argv);

    std::string out_name = params.input_path + "_n_" + std::to_string(comm.size()) + "_w_"
                           + std::to_string(params.window_size) + "_p_" + std::to_string(params.p_mod) + ".txt";
    logs::printer printer{};

    auto& timer = kamping::measurements::timer();

    // Distribute Data
    timer.synchronize_and_start("Distribute data");
    auto data = open_file(params.input_path, params.window_size, comm);
    timer.stop();

    // Compute positions of splitters
    timer.synchronize_and_start("Compute splitters");
    auto splits = compute_splitters(data, comm, params);
    timer.stop();

    auto total_splits = comm.allreduce_single(send_buf(splits.size()), kamping::op(kamping::ops::plus<>()));

    std::string to_print =
        std::format("{:.2f} % of the splits ({}) \n", (100.0 * splits.size()) / total_splits, splits.size());
    printer.print_all_on_root(to_print, comm);
    comm.barrier();
    // Compute phrases
    timer.synchronize_and_start("Compute dict");
    auto parse = compute_dict(splits, data, params, comm);
    timer.stop();

    // printer.log_phrase_size(parse.dict, comm, DELIMITER);

    auto dict_size = comm.allreduce_single(send_buf(parse.hashes.size()), kamping::op(kamping::ops::plus<>()));

    printer.print_on_root(
        std::format(
            "Dict size is {:.2f}% of the total input size \n",
            (100.0 * dict_size * params.window_size) / data.size()
        ),
        comm
    );
    printer.print_on_root(std::format("Total dictionary size is {} phrases \n", dict_size), comm);

    // Sort phrases globally
    timer.synchronize_and_start("Global sort");
    auto sorted_dict = sort_dict(parse.dict, comm);
    timer.stop();

    bool check = check_sort(sorted_dict);
    printer.print_on_root(std::to_string(check), comm);

    // Remove duplicates and hash sorted phrases
    timer.synchronize_and_start("Remove duplicates");
    auto [phrase_map, offset] = remove_duplicates(sorted_dict, comm, params.window_size);
    timer.stop();


    // Globally sort hashes
    timer.synchronize_and_start("Sort hashes");
    auto [hashes, last_hash] = sort_hashes(phrase_map, offset, comm);
    timer.stop();

    // Exchange hashes
    timer.synchronize_and_start("Exchange hashes");
    auto final_ranks = exchange_hashes(hashes, parse.hashes, last_hash, comm);
    timer.stop();

    if (comm.size() == 1 && comm.is_root()) {
        check = check_parsing(final_ranks, params, sorted_dict, comm, DELIMITER);
        printer.print_on_root(std::format("Parsing is correct: {} \n", check), comm);
    }

    timer.aggregate_and_print(kamping::measurements::SimpleJsonPrinter<>(std::cout));
}
