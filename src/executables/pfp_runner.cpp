#include "../../external/psac/src/sa_lcp_caller.cpp"
#include "../../external/scalable-distributed-string-sorting/src/executables/sort_caller.cpp"
#include <vector>

#include "AmsSort/AmsSort.hpp"
#include "algorithm/bwt.hpp"
#include "algorithm/pfp.hpp"
#include "checks/check_parsing.hpp"
#include "checks/check_sa.hpp"
#include "kamping/measurements/printer.hpp"
#include "kamping/measurements/timer.hpp"
#include "mpi/data_distribution.hpp"
#include "mpi/mpi_utils.hpp"
#include "mpi/request_response.hpp"
#include "util/cli_parser.hpp"
#include "util/logger.hpp"


using namespace logs;

std::vector<uint32_t> compute_occ(std::vector<uint32_t>& parse, Communicator<>& comm, size_t max_rank) {
    std::vector<uint32_t> rank_counts(max_rank, 0);
    for (auto r: parse) {
        // final_ranks is 1-based, rank_counts is 0-based
        rank_counts[r - 1]++;
    }

    comm.allreduce_inplace(send_recv_buf(rank_counts), op(kamping::ops::plus<>()));
    return rank_counts;
}

std::vector<uint32_t> compute_phrase_prefix_sum(std::vector<unsigned char>& dict, Communicator<>& comm) {

    std::vector<uint32_t> prefix_sum;
    uint32_t              sum = 0;

    for (auto c: dict) {
        ++sum;
        if (c == DELIMITER) {
            prefix_sum.push_back(sum);
        }
    }

    auto local_size = static_cast<uint32_t>(dict.size());
    auto offset = comm.exscan_single(send_buf(local_size), op(kamping::ops::plus<>{}));

    for (auto& value : prefix_sum) {
        value += offset;
    }

    auto global_prefix_sum = comm.allgatherv(send_buf(prefix_sum));
    return global_prefix_sum;
}

std::vector<uint32_t> compute_phrase_mapping(std::vector<uint32_t>& phrase_sa, std::vector<uint32_t>& phrase_prefix_sum) {
    std::vector<uint32_t> mapping(phrase_sa.size(), 0);

    for (size_t i = 0; i < phrase_sa.size(); ++i) {
        uint32_t sa_value = phrase_sa[i];
        // Binary search in prefix sum to find the corresponding phrase
        auto      it       = std::ranges::upper_bound(phrase_prefix_sum, sa_value);
        size_t    index    = std::distance(phrase_prefix_sum.begin(), it);
        mapping[i] = static_cast<uint32_t>(index);
    }
    return mapping;
}

std::pair<uint32_t, bool> get_phrase(uint64_t suffix, std::vector<uint32_t>& phrase_prefix_sum, int window_size) {
    auto it = std::ranges::lower_bound(phrase_prefix_sum, suffix);
    if (*it == suffix) {
        // The suffix is in the prefix sum -> it is the first char of a phrase
        return {static_cast<uint32_t>(std::distance(phrase_prefix_sum.begin(), it)), true};
    }
    if (*(it + 1) - suffix < window_size + 1) {
        // The suffix is less then w + 1 positions before a prefix sum -> it is the delimiter or it is smaller then the
        // window size todo make sure we can use 0 (phrase 0 can only be the first phrase for which we have a special
        // case anyways)
        return {0, false};
    }
    return {static_cast<uint32_t>(std::distance(phrase_prefix_sum.begin(), it)), false};
}

uint32_t phrase_suffix_len(uint32_t suffix, uint32_t offset, std::vector<unsigned char>& phrases) {
    for (auto i = suffix - offset; i < phrases.size(); ++i) {
        if (phrases[i] == DELIMITER) {
            return i - (suffix - offset);
        }
    }
    // this only happens for the last phrase
    return phrases.size() + offset - suffix;
}

// Compute the phrase lens for all given suffixes
std::vector<uint32_t> suffix_to_phrase_lens(std::vector<uint64_t>& suffixes, std::vector<uint32_t>& phrase_borders, Communicator<>& comm, std::vector<unsigned char>& phrases, uint64_t pe_offset) {
    std::vector<std::vector<uint64_t>> requests(phrase_borders.size());
    std::vector<int> pe_order(suffixes.size());
    for (auto suffix: suffixes) {
        auto it = std::ranges::lower_bound(phrase_borders, suffix);
        size_t index = std::distance(phrase_borders.begin(), it);
        if (index == comm.rank_signed()) {

        }
        else {
            requests[index].push_back(suffix);
        }
    }


    return request_response(requests, [&](uint64_t suffix) -> uint32_t {
        return phrase_suffix_len(suffix, pe_offset, phrases);
    }, pe_order, comm );
}

std::vector<unsigned char> get_prev_chars(std::vector<unsigned char>& phrases, std::vector<std::pair<uint32_t, uint32_t>>& prev_chars, std::vector<uint32_t>& phrase_prerfix_sum, Communicator<>& comm) {
    std::vector<std::vector<uint32_t>> requests(phrase_prerfix_sum.size());
    std::vector<int> pe_order;
    for (auto& [phrase_id, suffix]: prev_chars) {
        auto it = std::ranges::lower_bound(phrase_prerfix_sum, phrase_id);
        size_t pe = std::distance(phrase_prerfix_sum.begin(), it);
        // it is the offset for this pe
        requests[pe].push_back(suffix - *it);
        pe_order.push_back(pe);
    }

    return request_response(requests, [&](uint32_t suffix) -> unsigned char {
        return phrases[suffix - 1];
    }, pe_order, comm);

}

int main(int argc, char const* argv[]) {
    kamping::Environment  env;
    kamping::Communicator comm;

    Params params = read_parameters(argc, argv);

    std::string out_name = params.input_path + "_n_" + std::to_string(comm.size()) + "_w_"
                           + std::to_string(params.window_size) + "_p_" + std::to_string(params.p_mod) + ".txt";
    logger::set_comm(comm, out_name);

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
    logger::print_all_on_root("Found {} splitters, total splitters: {}", splits.size(), total_splits);

    comm.barrier();
    // Compute phrases
    timer.synchronize_and_start("Compute dict");
    auto parse = compute_dict(splits, data, params, comm);
    timer.stop();

    // Cleanup
    splits.clear();
    data.clear();

    // Sort phrases globally
    timer.synchronize_and_start("Global sort");
    auto sorted_dict = sort_dict(parse.dict, comm);
    timer.stop();

    bool check = check_sort(sorted_dict, DELIMITER);
    logger::print_all_on_root("Phrase sorting is correct: {}", check);

    // Remove duplicates and hash sorted phrases
    timer.synchronize_and_start("Remove duplicates");
    auto phrase_map = remove_duplicates(sorted_dict, comm, params.window_size);
    timer.stop();

    check = check_sort_unique(sorted_dict, DELIMITER);
    logger::print_all_on_root("Duplicates removal sorting is correct: {}", check);

    bool dup_check = check_duplicates(phrase_map);
    logger::print_all_on_root("Duplicates removal is correct: {}", dup_check);

    // Globally sort hashes
    timer.synchronize_and_start("Sort hashes");
    auto [hashes, last_hash] = sort_hashes(phrase_map, comm);
    timer.stop();

    // Cleanup
    phrase_map.clear();

    // Exchange hashes
    timer.synchronize_and_start("Exchange hashes");
    auto final_ranks = exchange_hashes(hashes, parse.hashes, last_hash, comm);
    timer.stop();

    check = check_parsing(final_ranks, params, sorted_dict, comm, DELIMITER);
    logger::print_all_on_root("Parsing is correct: {}", check);

    int local_max_rank = std::ranges::max(final_ranks);
    int global_max_rank = comm.allreduce_single(send_buf(local_max_rank), kamping::op(kamping::ops::max<>()));

    if (params.verbose) {
        logger::print_on_root("Max Rank: {}", global_max_rank);
        logger::print_rank_distribution(final_ranks);
    }

    // Compute BWT of P
    timer.synchronize_and_start("Redistribute parse");
    // todo bwt needs redistribution for correct sa_index <-> pe computation, this can be fixed
    auto ranks_out = redistribute_block_balanced(final_ranks, comm);
    timer.stop();

    timer.synchronize_and_start("Compute BWT of P");
    auto values = compute_bwt(ranks_out, comm);
    timer.stop();

    timer.synchronize_and_start("Compute rank counts");
    auto phrase_occ = compute_occ(final_ranks, comm, global_max_rank);
    timer.stop();

    timer.synchronize_and_start("Compute prefix sums");
    auto phrase_prefixes = compute_phrase_prefix_sum(sorted_dict, comm);
    timer.stop();

    timer.synchronize_and_start("Compute phrase mapping");
    auto phrase_mapping = compute_phrase_mapping(values, phrase_prefixes);
    timer.stop();

    timer.synchronize_and_start("Compute SA and LCP of D");

    // Redistribute phrases and get phrases in a std::string which is needed by psac
    timer.synchronize_and_start("Prepare computation of the LCP and SA of D");
    auto out                                        = redistribute_block_balanced(sorted_dict, comm);
    auto [sorted_dict_string, last_char_per_phrase] = convert_dict_to_string(out, params.window_size);
    timer.stop();

    timer.synchronize_and_start("Compute LCP and SA of D");
    auto sa = sa_builder();
    sa.construct_sa_lcp(comm.mpi_communicator(), sorted_dict_string);

    auto& sa_result  = sa.get_sa();
    auto& lcp_result = sa.get_lcp();
    timer.stop();

    std::vector<unsigned char> dict;
    for (auto c: sorted_dict_string) {
        dict.push_back(static_cast<unsigned char>(c));
    }

    bool t = check_sa(sa_result, dict, comm);
    logger::print_on_root("SA check for D: {}", t);

    timer.synchronize_and_start("Compute phrase suffix lengths");
    auto phrase_offsets = comm.exscan_single(send_buf(out.size()), op(kamping::ops::plus<>()));
    auto suffix_lens = suffix_to_phrase_lens(sa_result, phrase_prefixes, comm, out, phrase_offsets);
    timer.stop();

    // Main loop to construct the BWT
    std::vector<unsigned char> bwt;
    bwt.reserve(sa_result.size());

    // This vector indicates how many times a char has to be inserted at which position. (position, size)
    std::vector<std::pair<size_t, size_t>> prev_chars_pos;
    // This vector stores the prev chars needed to be requested (phrase_id, suffix)
    std::vector<std::pair<uint32_t, uint32_t>> prev_chars;
    size_t bwt_pos = 0;
    for (int sa_index = 0; sa_index < sa_result.size(); ++sa_index) {
        auto suffix = sa_result[sa_index];
        // Get the phrase the suffix belongs to
        auto suffix_phrase = get_phrase(suffix, phrase_mapping, params.window_size);


        if (suffix_phrase.first == 0) {
            // Suffix is in the delimiter or smaller than the window size
            continue;
        }
        if (suffix_phrase.second) {
            // Suffix is the first char of a phrase
            // Store chars and their starting pos in the bwt
            // todo add elements to first_chars
            // auto pre_chars = get_chars_before_phrase(suffix_phrase, dict, comm);
            //bwt_pos += pre_chars.size();
            continue;
        }
        else {
            // Suffix is in the middle of a phrase
            auto phrase_index = suffix_phrase.first;
            auto suffix_len = suffix_lens[sa_index];

            if (lcp_result[sa_index + 1] < suffix_len) {
                // The phrase suffix is larger than the lcp value, so the first char in the bwt is the char before the suffix
                // store the prev char that is needed
                prev_chars.emplace_back(phrase_index, suffix);
                // store its position and needed length in the bwt
                prev_chars_pos.emplace_back(bwt_pos, phrase_occ[phrase_index - 1]);
                bwt_pos += phrase_occ[phrase_index - 1];
                // todo check if the phrases is on this PE, in this case the char can be included directly

            }
            else {
                // The phrase suffix is larger the order of the suffixes can not be decided by the phrase alone
                // we need to take the parse into account
                //
            }
        }

    }


    // Request the prev chars and fill the bwt
    auto p_chars = get_prev_chars(out, prev_chars, phrase_prefixes, comm);
    int index = 0;
    for (auto [pos, size] : prev_chars_pos) {
        auto c = p_chars[index];
        for (size_t i = 0; i < size; ++i) {
            bwt[pos + i] = c;
        }
        index++;
    }

    timer.aggregate_and_print(kamping::measurements::SimpleJsonPrinter<>(logger::get_ostream()));

    return 0;
}
