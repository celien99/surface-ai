#pragma once

#include <cstddef>
#include <string>

struct CliArgs {
    std::string image_dir;
    std::string output_dir = "/tmp/surface-ai/results/";
    std::string coreset_path;           // --coreset: load pre-built coreset for detection
    std::string dataset_path;            // --dataset: YAML manifest for coreset building
    std::string coreset_output_path;     // --coreset-output: where to save the built coreset
    std::string coreset_algo = "greedy"; // --coreset-algo: greedy | uniform
    std::size_t coreset_max_samples = 10000; // --coreset-max-samples N
    bool headless = false;
    bool cpu_mode = false;              // --cpu: force CPU embedder (no GPU)
};

auto ParseArgs(int argc, char* argv[]) -> CliArgs;
