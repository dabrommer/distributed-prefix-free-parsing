#pragma once

#include <functional>
#include <ranges>
#include <vector>

#include "kamping/communicator.hpp"
#include "kamping/collectives/alltoall.hpp"

using namespace kamping;

// Reorders the result given by request_response according to the given pe_order
template <typename ResponseType>
std::vector<ResponseType> get_responses_in_order(
    std::vector<ResponseType> const& responses,
    std::vector<int> const&          response_counts,
    std::vector<int> const&          pe_order
) {
    // Compute offsets for each PE
    std::vector<size_t> offsets(response_counts.size());
    size_t offset = 0;
    for (size_t i = 0; i < response_counts.size(); ++i) {
        offsets[i] = offset;
        offset += response_counts[i];
    }

    std::vector<size_t> consumed(response_counts.size(), 0);

    // Build result by iterating through pe_order
    std::vector<ResponseType> result;
    result.reserve(pe_order.size());

    for (int i = 0; i < responses.size(); ++i) {
        int pe = pe_order[i];
        // Get the next response from this PE
        size_t pe_offset = offsets[pe] + consumed[pe];
        result.push_back(responses[pe_offset]);
        ++consumed[pe];
    }

    return result;
}

// Given a vector of requests for each PE and a processing function this uses alltoallv to exchange requests and responses between PEs.
// the requests are processed according to the ProcessFunc
template <typename RequestType, typename ProcessFunc>
auto request_response(
    std::vector<std::vector<RequestType>> const& requests,
    ProcessFunc&&                                 process_request,
    Communicator<>&                               comm
) -> std::pair<std::vector<std::invoke_result_t<ProcessFunc, RequestType>>, std::vector<int>> {
    using ResponseType = std::invoke_result_t<ProcessFunc, RequestType>;
    // Flatten requests and compute send counts
    std::vector<int> send_counts;
    send_counts.reserve(requests.size());
    for (auto const& req: requests) {
        send_counts.push_back(static_cast<int>(req.size()));
    }

    auto                      flat_req_view = requests | std::ranges::views::join;
    std::vector<RequestType>  flat_requests(flat_req_view.begin(), flat_req_view.end());

    // Exchange requests via alltoallv
    auto [recv_requests, recv_counts] = comm.alltoallv(
        kamping::send_buf(flat_requests),
        kamping::send_counts(send_counts),
        kamping::recv_counts_out()
    );

    std::vector<std::vector<ResponseType>> responses(comm.size());

    int curr_pe     = 0;
    int curr_pe_idx = 0;

    for (auto const& request: recv_requests) {
        // Move to next PE if we've processed all requests for current PE
        while (curr_pe_idx >= recv_counts[curr_pe] && curr_pe < comm.size() - 1) {
            ++curr_pe;
            curr_pe_idx = 0;
        }
        if (curr_pe >= comm.size()) {
            // This should not happen if recv_counts are correct
            throw std::runtime_error("Received more requests than expected");
        }
        responses[curr_pe].push_back(process_request(request));
        ++curr_pe_idx;
    }

    // Flatten responses
    auto                      flat_resp_view = responses | std::ranges::views::join;
    std::vector<ResponseType> flat_responses(flat_resp_view.begin(), flat_resp_view.end());

    // Exchange responses via alltoallv
    auto [out_responses, response_counts] = comm.alltoallv(
        kamping::send_buf(flat_responses),
        kamping::send_counts(recv_counts),
        kamping::recv_counts_out()
    );

    return {std::move(out_responses), std::move(response_counts)};
}

// Overload which returns the responses in the given pe_order
template <typename RequestType, typename ProcessFunc>
auto request_response(
    std::vector<std::vector<RequestType>> const& requests,
    ProcessFunc&&                                 process_request,
    std::vector<int> const&                      pe_order,
    Communicator<>&                               comm
) {
    using ResponseType = std::invoke_result_t<ProcessFunc, RequestType>;

    // Call the base request_response function
    auto [responses, response_counts] = request_response(requests, std::forward<ProcessFunc>(process_request), comm);

    // Reorder responses according to pe_order
    return get_responses_in_order<ResponseType>(responses, response_counts, pe_order);
}







