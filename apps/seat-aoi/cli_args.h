#pragma once

#include <cstddef>
#include <string>

struct CliArgs {
    std::string image_dir;
    std::string output_dir = "/tmp/surface-ai/results/";
    std::string coreset_path;           // --coreset: load pre-built coreset for detection
    std::string coreset_manifest_path;  // --coreset-manifest: multi-position YAML registry
    std::string dataset_path;            // --dataset: YAML manifest for coreset building
    std::string coreset_output_path;     // --coreset-output: where to save the built coreset
    std::string coreset_algo = "greedy"; // --coreset-algo: greedy | uniform
    std::string review_dir;              // --review-dir: path to JSON results for GUI review
    std::size_t coreset_max_samples = 10000; // --coreset-max-samples N
    bool headless = false;
    bool train_mode = false;             // --train or --coreset-output: build coreset mode
    bool cpu_mode = false;              // --cpu: force CPU embedder (no GPU)
};

auto ParseArgs(int argc, char* argv[]) -> CliArgs;
