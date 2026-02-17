#pragma once

#include <span>
#include <sstream>
#include <vector>

std::vector<std::string> get_dcx_args() {
    // Without --use-char-packing-samples --use-char-packing-merging
    std::string dcx_args = "--atomic-sorter=ams --discarding-threshold=0.7 --ams-levels=2 --splitter-sampling=random --splitter-sorting=central --use-random-sampling-splitters --num-samples-splitters=20000 --buckets-sample-phase=16,16 --buckets-merging-phase=64,64,16 --use-binary-search-for-splitters --use-randomized-chunks --avg-chunks-pe=10000 --buckets-phase3=1 --samples-buckets-phase3=10000 --rearrange-buckets-balanced --use-compressed-buckets --pack-extra-words=0";


    std::vector<std::string> arg_strings;
    std::stringstream ss(dcx_args);
    std::string segment;
    while (ss >> segment) {
        arg_strings.push_back(segment);
    }

    return arg_strings;
}