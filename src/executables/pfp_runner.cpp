#include "../../external/psac/src/sa_lcp_caller.cpp"
#include "../../external/scalable-distributed-string-sorting/src/executables/sort_caller.cpp"
#include <vector>

#include "AmsSort/AmsSort.hpp"
#include "algorithm/bwt.hpp"
#include "algorithm/pfp.hpp"
#include "checks/check_bwt.hpp"
#include "checks/check_parsing.hpp"
#include "checks/check_sa.hpp"
#include "kamping/measurements/printer.hpp"
#include "kamping/measurements/timer.hpp"
#include "kamping/collectives/scan.hpp"
#include "kamping/p2p/send.hpp"
#include "kamping/p2p/recv.hpp"
#include "mpi/data_distribution.hpp"
#include "mpi/mpi_utils.hpp"
#include "mpi/request_response.hpp"
#include "util/cli_parser.hpp"
#include "util/logger.hpp"


using namespace logs;

std::vector<uint32_t> compute_occ(std::vector<uint32_t>& parse, Communicator<>& comm, size_t max_rank) {
    std::vector<uint32_t> rank_counts(max_rank, 0);
    for (auto r: parse) {
        // parse is 1-based, rank_counts is 0-based
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

std::vector<uint32_t> compute_phrase_mapping(std::vector<uint64_t>& phrase_sa, std::vector<uint32_t>& phrase_prefix_sum) {
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
    std::vector<int> pe_order;
    pe_order.reserve(suffixes.size());
    for (auto suffix: suffixes) {
        auto it = std::ranges::lower_bound(phrase_borders, suffix);
        size_t index = std::distance(phrase_borders.begin(), it);
        pe_order.push_back(static_cast<int>(index));
        requests[index].push_back(suffix);

    }


    return request_response(requests, [&](uint64_t suffix) -> uint32_t {
        return phrase_suffix_len(suffix, pe_offset, phrases);
    }, pe_order, comm );
}

std::vector<unsigned char> get_prev_chars(std::vector<unsigned char>& phrases, std::vector<std::pair<uint32_t, uint64_t>>& prev_chars, std::vector<uint32_t>& phrase_prerfix_sum, Communicator<>& comm) {
    std::vector<std::vector<uint32_t>> requests(comm.size());
    std::vector<int> pe_order;
    for (auto& [phrase_id, suffix]: prev_chars) {
        // find the PE which has suffix - 1 in it local phrases
        auto it = std::ranges::upper_bound(phrase_prerfix_sum, suffix - 1);
        size_t pe = std::distance(phrase_prerfix_sum.begin(), it);
        // offset is the value of the previous prefix sum value, 0 in case of PE 0
        auto offset = pe == 0 ? 0 : *(it - 1);
        requests[pe].push_back(suffix - offset);
        pe_order.push_back(pe);
    }

    // todo is this 0 case possible?
    return request_response(requests, [&](uint32_t suffix) -> unsigned char {
        return suffix == 0 ? phrases.back() : phrases[suffix - 1];
    }, pe_order, comm);

}

// computes and returned the order of the phrase ids for the given hard case
std::vector<uint32_t>
solve_hard_case(std::span<std::pair<uint64_t, uint32_t>> hard_chars, std::vector<std::vector<uint32_t>>& inverted_list) {
    using Pair = std::pair<uint32_t, uint32_t>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<>> min_heap;


    for (auto& [sa_index, phrase_id] : hard_chars) {
        for (auto pos : inverted_list[phrase_id - 1]) {
            min_heap.emplace(pos, phrase_id);
        }
    }

    std::vector<uint32_t> phrase_id_order;
    phrase_id_order.reserve(min_heap.size());

    while (!min_heap.empty()) {
        auto [pos, phrase_id] = min_heap.top();
        min_heap.pop();
        phrase_id_order.push_back(phrase_id);
    }

    return phrase_id_order;
}

std::vector<std::vector<uint32_t>> compute_inverted_list(
    std::vector<uint32_t> const&      parse_bwt,
    uint64_t                          max_rank,
    Communicator<>&                   comm
) {

    std::vector<std::vector<uint32_t>> inverted_list(max_rank, std::vector<uint32_t>());
    for (size_t i = 0; i < parse_bwt.size(); ++i) {
        auto phrase_id = parse_bwt[i];
        inverted_list[phrase_id - 1].push_back(i);
    }

    for (int phrase_id = 0; phrase_id < max_rank; ++phrase_id) {
        auto global_list         = comm.allgatherv(send_buf(inverted_list[phrase_id]));
        inverted_list[phrase_id] = std::move(global_list);
    }

    return inverted_list;
}
std::vector<unsigned char> permutate_last_chars(const std::vector<uint32_t>& parse_bwt, const std::vector<unsigned char>& last_char_per_phrase, const Communicator<std::vector>& comm) {
    // todo this is not efficient
    auto all_last_chars = comm.allgatherv(send_buf(last_char_per_phrase));

    std::vector<unsigned char> permutated_last_chars(parse_bwt.size());
    for (size_t i = 0; i < parse_bwt.size(); ++i) {
        auto phrase_id = parse_bwt[i];
        permutated_last_chars[i] = all_last_chars[phrase_id - 1];
    }
    // todo this can stay local when the main loop is fixed
    return comm.allgatherv(send_buf(permutated_last_chars));//permutated_last_chars;

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

    auto local_max_rank = std::ranges::max(final_ranks);
    auto global_max_rank = static_cast<uint64_t>(comm.allreduce_single(send_buf(local_max_rank), kamping::op(kamping::ops::max<>())));

    if (params.verbose) {
        logger::print_on_root("Max Rank: {}", global_max_rank);
        logger::print_rank_distribution(final_ranks);
    }

    // Redistribute phrases and get phrases in a std::string which is needed by psac
    timer.synchronize_and_start("Prepare computation of the LCP and SA of D");
    auto out                                        = redistribute_block_balanced(sorted_dict, comm);
    auto [sorted_dict_string, last_char_per_phrase] = convert_dict_to_string(out, params.window_size, comm);
    timer.stop();

    char x = out.front();

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

    timer.synchronize_and_start("Compute rank counts");
    auto phrase_occ = compute_occ(final_ranks, comm, global_max_rank);
    timer.stop();

    timer.synchronize_and_start("Compute prefix sums");
    auto phrase_prefixes = compute_phrase_prefix_sum(sorted_dict, comm);
    timer.stop();

    timer.synchronize_and_start("Compute phrase mapping");
    auto phrase_mapping = compute_phrase_mapping(sa_result, phrase_prefixes);
    timer.stop();

    // Compute BWT of P
    timer.synchronize_and_start("Redistribute parse");
    // todo bwt needs redistribution for correct sa_index <-> pe computation, this can be fixed
    auto ranks_out = redistribute_block_balanced(final_ranks, comm);
    timer.stop();

    timer.synchronize_and_start("Compute BWT of P");
    auto parse_bwt = compute_bwt(ranks_out, comm);
    timer.stop();

    std::cout << "Passed BWT of P" << std::endl;

    timer.synchronize_and_start("Compute inverted list of the BWT of P");
    auto inverted_list = compute_inverted_list(parse_bwt, global_max_rank, comm);
    timer.stop();

    std::cout << "Passed inverted list of BWT of P" << std::endl;

    /*bool t = check_sa(sa_result, dict, comm);
    logger::print_on_root("SA check for D: {}", t);*/

    timer.synchronize_and_start("Compute phrase suffix lengths");
    // Get the offset for each PEs phrases , PE 0 = 0, PE 1 = |PE(0).phrases|, ...
    auto phrase_offsets = comm.exscan_single(send_buf(out.size()), op(kamping::ops::plus<>()), values_on_rank_0(0));
    // Get the prefix sum of the sizes:
    auto local_phrase_prefix_sum = comm.scan_single(send_buf(static_cast<uint32_t>(out.size())), op(kamping::ops::plus<>()));
    auto global_phrase_prefix_sum = comm.allgather(send_buf(local_phrase_prefix_sum));
    auto suffix_lens = suffix_to_phrase_lens(sa_result, global_phrase_prefix_sum, comm, out, phrase_offsets);
    timer.stop();

    std::cout << "Passed phrase suffix lengths" << std::endl;

    timer.synchronize_and_start("Permutate the W array");
    auto w_array = permutate_last_chars(parse_bwt, last_char_per_phrase, comm);
    timer.stop();


    // Main loop to construct the BWT
    // todo sa_result.size() * 10 should be an upper bound, is there a better way to handle this?
    std::vector<unsigned char> bwt(sa_result.size() * 10);
    bwt.reserve(sa_result.size());

    timer.synchronize_and_start("Main BWT loop");

    std::cout << "Reached Main Loop" << std::endl;
    // todo this does not scale
    auto global_parse = comm.allgatherv(send_buf(ranks_out));

    // This vector indicates how many times a char has to be inserted at which position. (position, size)
    std::vector<std::pair<size_t, size_t>> prev_chars_pos;
    // This vector stores the prev chars needed to be requested (phrase_id, suffix)
    std::vector<std::pair<uint32_t, uint64_t>> prev_chars;


    // Store which suffix index in which phrase is a hard char (sa index, phrase_id)
    std::vector<std::pair<uint64_t, uint32_t>> hard_chars;
    // Store the index at which the hard chars should be inserted in the bwt
    std::vector<uint32_t> hard_chars_idx;
    // Store how many hard cases compete for the same position in the bwt
    // eg hard_chars_len[i] = 3, the i-th hard case has 3 suffixes competing
    std::vector<uint32_t> hard_chars_len;

    size_t bwt_pos = 0;
    // The last
    if (comm.rank_signed() == 0) {
        bwt_pos = 1; // The first char in the bwt is always the end of text symbol, so we can start at position 1
        auto first_char = comm.recv<unsigned char>(source(comm.size() - 1));
        bwt[0] = first_char.front();
    }
    else if (comm.rank_signed() == comm.size() - 1) {
        // todo what is the first char of the bwt? (what is the last char of the input data?)
        // This is just the last char of the lex largest phrase
        auto last_char = out[out.size() - params.window_size];
        comm.send(send_buf(last_char), destination(0));
    }
    // ignore the last index for now
    // PE 0 skips the first suffix, because it will be 0, all other PEs start with the first suffix
    for (int sa_index = comm.rank_signed() == 0 ? 1 : 0; sa_index < sa_result.size() - 1; ++sa_index) {
        auto suffix = sa_result[sa_index];
        // Get the phrase the suffix belongs to
        auto suffix_phrase = get_phrase(suffix, phrase_prefixes, params.window_size);


        if (suffix_phrase.first == 0) {
            // Suffix is in the delimiter or smaller than the window size
            continue;
        }
        if (suffix_phrase.second) {
            // Suffix is the first char of a phrase
            // The inverted list stores the positions of the preceding phrases in correct order.
            // The actual chars are retrieved from the permutated W array
            auto& inverted_list_index = inverted_list[suffix_phrase.first];
                for (auto pos: inverted_list_index) {
                    // todo w_array is distributed, need to request these as well
                    bwt.push_back(w_array[pos]);
                    bwt_pos++;
                }
            continue;
        }
        else {
            // Suffix is in the middle of a phrase
            auto phrase_index = suffix_phrase.first;
            auto suffix_len = suffix_lens[sa_index];

            if (lcp_result[sa_index + 1] < suffix_lens[sa_index]) {
                // The phrase suffix is larger than the lcp value, so the first char in the bwt is the char before the suffix
                // store the prev char that is needed
                prev_chars.emplace_back(phrase_index, suffix);
                // store its position and needed length in the bwt
                // todo it could be the case that this stretches accross the bwt border
                prev_chars_pos.emplace_back(bwt_pos, phrase_occ[phrase_index - 1]);
                bwt_pos += phrase_occ[phrase_index - 1];
                // todo check if the phrases is on this PE, in this case the char can be included directly

            }
            else {
                hard_chars_len.push_back(1);
                hard_chars_idx.push_back(bwt_pos);
                hard_chars.emplace_back(suffix, suffix_phrase.first);
                while (sa_index < sa_result.size() - 1 && lcp_result[sa_index + 1] > suffix_lens[sa_index]) {
                    // Loop to find all next suffixes that belong to this hard case, they all compete for the same position in the bwt
                    bwt_pos += phrase_occ[phrase_index - 1];

                    // get the next suffix
                    sa_index++;
                    suffix = sa_result[sa_index];

                    // Add the next suffix to the hard case vectors

                    suffix_phrase = get_phrase(suffix, phrase_prefixes, params.window_size);
                    // we dont have to check for suffixes that are the delimiter or smaller than the window size
                    // because in this case the lcp value would be smaller than the suffix len
                    hard_chars.emplace_back(suffix, suffix_phrase.first);
                    hard_chars_len.back()++;
                }

            }
        }
    }

    timer.stop();

    std::cout << "Finished loop, start prev char exchange" << std::endl;

    timer.synchronize_and_start("Retrieving easy prev chars");
    // Request the prev chars and fill the bwt
    auto p_chars = get_prev_chars(out, prev_chars, global_phrase_prefix_sum, comm);
    int index = 0;
    for (auto& [pos, size] : prev_chars_pos) {
        auto c = p_chars[index];
        for (size_t i = 0; i < size; ++i) {
            if (pos + i >= bwt.size()) {
                std::cout << "Error: position " << pos + i << " is out of bounds for the bwt of size " << bwt.size() << std::endl;
            }
            bwt[pos + i] = c;
        }
        index++;
    }
    timer.stop();

    std::cout << "Finished prev char exchange, start solving hard cases" << std::endl;

    timer.synchronize_and_start("Solve hard cases");

    // Solve the hard cases
    std::vector<std::pair<uint32_t, uint64_t>> hard_chars_requests;
    for (uint32_t current_hard_case = 0; auto len : hard_chars_len) {

        // Get the sa indices that compete for the same position in the bwt
        std::span curr_hard_chars(hard_chars.begin() + current_hard_case, hard_chars.begin() + current_hard_case + len);
        // This orders the phrase_ids of the current hard case
        auto phrase_id_order = solve_hard_case(curr_hard_chars, inverted_list);
        // We need to map the phrase_ids back to the sa indices to create the requests
        std::map<uint32_t, uint32_t> phrase_to_sa_index;
        for (auto& [sa_index, phrase_id] : curr_hard_chars) {
            phrase_to_sa_index[phrase_id] = sa_index;
        }
        for (auto phrase_id : phrase_id_order) {
            hard_chars_requests.emplace_back(phrase_id, phrase_to_sa_index[phrase_id]);
        }
        current_hard_case += len;
    }

    auto hard_prev_chars = get_prev_chars(out, hard_chars_requests, global_phrase_prefix_sum, comm);

    int char_pos = 0;
    for (size_t hard_index = 0; hard_index < hard_chars_idx.size(); ++hard_index) {
        for (size_t hard_count = 0; hard_count < hard_chars_len[hard_index]; ++hard_count) {
            bwt[hard_chars_idx[hard_index] + hard_count] = hard_prev_chars[char_pos];
            ++char_pos;
        }
    }

    timer.stop();
    timer.aggregate_and_print(kamping::measurements::SimpleJsonPrinter<>(logger::get_ostream()));

    bool bwt_correct = check_bwt(params.input_path, comm, bwt, params.bwt_check_path);
    logger::print_on_root("BWT check is: {}", bwt_correct);

    return 0;
}
