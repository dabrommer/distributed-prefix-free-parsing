#pragma once

#include <print>

#include <kamping/communicator.hpp>

namespace logs {

struct printer {
    std::FILE* file = stdout;

    printer() = default;
    explicit printer(std::string const& filename) {
        file = std::fopen(filename.c_str(), "w");
    }

    ~printer() {
        if (file && file != stdout && file != stderr) {
            std::fclose(file);
        }
    }

    std::ostream get_ostream() {
        return std::ostream(file);
    }
    void
    log_phrase_size(std::vector<unsigned char> const& data, Communicator<> const& comm, char const delimiter) const {
        std::map<std::size_t, std::size_t> counts;

        size_t current_size = 0;
        for (auto const c: data) {
            if (c != 0) {
                ++current_size;
            } else {
                ++counts[current_size];
                current_size = 0;
            }
        }

        std::string out("{\n");
        for (auto it = counts.begin(); it != counts.end(); ++it) {
            out += ("  \"" + std::to_string(it->first) + "\": " + std::to_string(it->second));
            if (std::next(it) != counts.end())
                out += ",";
        }
        out += "}\n";

        std::vector<char> out_vec(out.begin(), out.end());

        auto outs = comm.gatherv(send_buf(out_vec), recv_counts_out());

        // print results on root
        if (comm.is_root()) {
            auto const offsets = outs.extract_recv_counts();
            auto       strings = outs.get_recv_buffer();
            int        rank    = 0;
            for (auto const o: offsets) {
                std::string result(strings.begin(), strings.begin() + o);
                strings.erase(strings.begin(), strings.begin() + o);
                std::print(file, "PE [{}]: \n", rank);
                std::print(file, "{}\n", result);
                ++rank;
            }
        }
    }

    void print_all_on_root(std::string const& to_print, Communicator<> const& comm) const {
        std::vector<char> out_vec(to_print.begin(), to_print.end());

        auto outs = comm.gatherv(send_buf(out_vec), recv_counts_out());

        if (comm.is_root()) {
            auto offsets = outs.extract_recv_counts();
            auto strings = outs.get_recv_buffer();
            int  rank    = 0;
            for (auto o: offsets) {
                std::string result(strings.begin(), strings.begin() + o);
                strings.erase(strings.begin(), strings.begin() + o);
                std::print(file, "PE [{}]: {}\n", rank, result);
                ++rank;
            }
        }
    }

    void print_on_root(std::string const& to_print, Communicator<> const& comm) const {
        if (comm.is_root()) {
            std::print(file, "PE [{}]: ", comm.rank());
            std::print(file, "{} \n", to_print);
        }
    }
};

} // namespace logs