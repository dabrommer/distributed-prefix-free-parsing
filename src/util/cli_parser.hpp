#pragma once
#include "CLI/App.hpp"
#include "util/CLI_mpi.hpp"

#include <string>

struct Params {

  std::string input_path;
  int window_size = 10;
  std::string hash;
  int p_mod = 100;
};

Params read_parameters(int argc, char const* argv[]) {
  CLI::App app{"Prefix Free Parsing"};

  Params params;
  app.add_option("--input", params.input_path, "Path to input file");
  app.add_option("--w", params.window_size, "Window size");
  app.add_option("--hash", params.hash, "Hash function to use");
  app.add_option("--p", params.p_mod, "Modulo for hash splitters");

  CLI11_PARSE_MPI(app, argc, argv);

  return params;
}