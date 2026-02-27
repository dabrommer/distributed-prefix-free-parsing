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
#include "util/cli_parser.hpp"
#include "util/logger.hpp"

using namespace logs;

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

    int max_rank = std::max(final_ranks.front(), final_ranks.back());

    if (params.verbose) {
        logger::print_on_root("Max Rank: {}", max_rank);
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

    bool sa_correct = check_sa(values, final_ranks, comm);
    logger::print_on_root("SA check : {}", sa_correct);

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

    timer.aggregate_and_print(kamping::measurements::SimpleJsonPrinter<>(logger::get_ostream()));

    return 0;
}
