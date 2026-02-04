#pragma once

#include <print>

#include <kamping/communicator.hpp>

namespace logs {

struct printer {

    printer() : out(&std::cout) {}  ;


    explicit printer(const std::string& filename)
        : file_stream(filename), out(&file_stream)
    {}

    ~printer() {

    }

    std::ostream& get_ostream()
    {
        return file_stream;

    }


    void log_phrase_size(std::vector<unsigned char> const& data, Communicator<> const& comm, char const delimiter) {
        std::map<std::size_t, std::size_t> counts;

        size_t current_size = 0;
        for (auto const c: data) {
            if (c != delimiter) {
                ++current_size;
            } else {
                ++counts[current_size];
                current_size = 0;
            }
        }

        print_map_to_json(counts, comm);
    }

    void print_all_on_root(std::string const& to_print, Communicator<> const& comm) {
        std::vector<char> out_vec(to_print.begin(), to_print.end());

        auto outs = comm.gatherv(send_buf(out_vec), recv_counts_out());

        if (comm.is_root()) {
            auto offsets = outs.extract_recv_counts();
            auto strings = outs.get_recv_buffer();
            int  rank    = 0;
            for (auto o: offsets) {
                std::string result(strings.begin(), strings.begin() + o);
                strings.erase(strings.begin(), strings.begin() + o);
                std::print(*out, "PE [{}]: {}\n", rank, result);
                ++rank;
            }
        }
    }

    void print_on_root(std::string const& to_print, Communicator<> const& comm) {
        if (comm.is_root()) {
            std::print(*out, "PE [{}]: ", comm.rank());
            std::print(*out, "{} \n", to_print);
        }
    }

    template<typename key, typename value>
    void print_map_to_json(std::map<key, value>& values, const Communicator<>& comm) {

        std::string output("{\n");
        for (auto it = values.begin(); it != values.end(); ++it) {
            output += ("  \"" + std::to_string(it->first) + "\": " + std::to_string(it->second));
            if (std::next(it) != values.end())
                output += ",";
        }
        output += "}\n";

        std::vector<char> out_vec(output.begin(), output.end());

        auto outs = comm.gatherv(send_buf(out_vec), recv_counts_out());

        // print results on root
        if (comm.is_root()) {
            auto const offsets = outs.extract_recv_counts();
            auto       strings = outs.get_recv_buffer();
            int        rank    = 0;
            for (auto const o: offsets) {
                std::string result(strings.begin(), strings.begin() + o);
                strings.erase(strings.begin(), strings.begin() + o);
                std::print(*out, "PE [{}]: \n", rank);
                std::print(*out, "{}\n", result);
                ++rank;
            }
        }
    }

    void print_rank_distribution(const std::vector<int>& ranks, const Communicator<>& comm) {
        if (!comm.is_root()) {
            return;
        }
        std::map<int, int> distribution;
        for (auto r: ranks) {
            distribution[r]++;
        }

        print_map_to_json(distribution, comm);

    }

private:
    std::string filename = "log.txt";
    std::ofstream file_stream;
    std::ostream* out;
};

} // namespace logs