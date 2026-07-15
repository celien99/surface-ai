#include "cli_args.h"

#include <string_view>

auto ParseArgs(int argc, char* argv[]) -> CliArgs {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--image-dir" && i + 1 < argc) {
            args.image_dir = argv[++i];
            args.headless = true;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
        } else if (arg == "--headless") {
            args.headless = true;
        } else if (arg == "--coreset" && i + 1 < argc) {
            args.coreset_path = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            args.dataset_path = argv[++i];
        } else if (arg == "--coreset-output" && i + 1 < argc) {
            args.coreset_output_path = argv[++i];
        } else if (arg == "--coreset-algo" && i + 1 < argc) {
            args.coreset_algo = argv[++i];
        } else if (arg == "--coreset-max-samples" && i + 1 < argc) {
            args.coreset_max_samples = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--cpu") {
            args.cpu_mode = true;
        }
    }
    if (!args.dataset_path.empty() && args.coreset_output_path.empty()) {
        args.coreset_output_path = "resources/coreset.bin";
    }
    return args;
}
