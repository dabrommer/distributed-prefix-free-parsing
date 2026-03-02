#pragma once

#include <fstream>
#include <map>
#include <print>

#include <kamping/communicator.hpp>

namespace logs {

class logger {
public:
    // Initialize the logger with a communicator reference
    static void set_comm(Communicator<> const& comm) {
        comm_ptr = &comm;
        out      = &std::cout;
    }

    // Initialize the logger with a communicator and output file
    static void set_comm(Communicator<> const& comm, std::string const& filename) {
        comm_ptr = &comm;
        if (file_stream.is_open()) {
            file_stream.close();
        }
        file_stream.open(filename);
        out = &file_stream;
    }

    // Get the output stream being used by the logger
    static std::ostream& get_ostream() {
        if (out == nullptr) {
            return std::cout;
        }
        return *out;
    }

    // Log phrase sizes with delimiter
    template <typename char_type>
    static void log_phrase_size(std::vector<char_type> const& data, char const delimiter) {
        if (!check_initialized())
            return;
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

        print_map_to_json(counts);
    }

    // Print message from all ranks on root
    static void print_all_on_root(std::string const& to_print) {
        if (!check_initialized())
            return;
        std::vector<char> out_vec(to_print.begin(), to_print.end());

        auto outs = comm_ptr->gatherv(send_buf(out_vec), recv_counts_out());

        if (comm_ptr->is_root()) {
            auto offsets = outs.extract_recv_counts();
            auto strings = outs.get_recv_buffer();
            int  rank    = 0;
            for (auto o: offsets) {
                std::string result(strings.begin(), strings.begin() + o);
                strings.erase(strings.begin(), strings.begin() + o);
                std::print(*out, "PE [{}]: {}\n", rank, result);
                ++rank;
            }
            out->flush();
        }
    }

    // Print formatted message from all ranks on root
    template <typename... Args>
    static void print_all_on_root(std::format_string<Args...> fmt, Args&&... args) {
        std::string formatted = std::format(fmt, std::forward<Args>(args)...);
        print_all_on_root(formatted);
    }

    // Print message only on root rank
    static void print_on_root(std::string const& to_print) {
        if (!check_initialized())
            return;
        if (comm_ptr->is_root()) {
            std::print(*out, "PE [{}]: {}\n", comm_ptr->rank(), to_print);
            out->flush();
        }
    }

    // Print formatted message only on root rank
    template <typename... Args>
    static void print_on_root(std::format_string<Args...> fmt, Args&&... args) {
        std::string formatted = std::format(fmt, std::forward<Args>(args)...);
        print_on_root(formatted);
    }

    // Print map as JSON from all ranks
    template <typename key, typename value>
    static void print_map_to_json(std::map<key, value>& values) {
        if (!check_initialized())
            return;

        std::string output("{\n");
        for (auto it = values.begin(); it != values.end(); ++it) {
            output += ("  \"" + std::to_string(it->first) + "\": " + std::to_string(it->second));
            if (std::next(it) != values.end())
                output += ",";
        }
        output += "}\n";

        std::vector<char> out_vec(output.begin(), output.end());

        auto outs = comm_ptr->gatherv(send_buf(out_vec), recv_counts_out());

        // print results on root
        if (comm_ptr->is_root()) {
            auto const offsets = outs.extract_recv_counts();
            auto       strings = outs.get_recv_buffer();
            int        rank    = 0;
            for (auto const o: offsets) {
                std::string result(strings.begin(), strings.begin() + o);
                strings.erase(strings.begin(), strings.begin() + o);
                std::print(*out, "PE [{}]: {}\n", rank, result);
                ++rank;
            }
            out->flush();
        }
    }

    // Print rank distribution
    static void print_rank_distribution(std::vector<uint32_t> const& ranks) {
        if (!check_initialized())
            return;
        if (!comm_ptr->is_root()) {
            return;
        }
        std::map<int, int> distribution;
        for (auto r: ranks) {
            distribution[r]++;
        }

        print_map_to_json(distribution);
    }

private:
    static bool check_initialized() {
        if (comm_ptr == nullptr) {
            std::print(std::cerr, "Logger not initialized, logging disabled. Call logger::set_comm() first.\n");
            return false;
        }
        return true;
    }

    static inline Communicator<> const* comm_ptr = nullptr;
    static inline std::ofstream         file_stream;
    static inline std::ostream*         out = &std::cout;
};

} // namespace logs