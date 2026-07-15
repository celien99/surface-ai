#include <iostream>

#include "app_builder.h"
#include "cli_args.h"
#include "coreset_builder.h"
#include "gui_runner.h"
#include "headless_runner.h"

auto main(int argc, char* argv[]) -> int {
    auto cli = ParseArgs(argc, argv);

    // --dataset mode (without --coreset-output): build coreset and exit
    if (!cli.dataset_path.empty() && cli.coreset_output_path.empty()) {
        return BuildCoreset(cli);
    }

    // Assemble the full application
    auto app_result = AssembleApplication(cli);
    if (!app_result) {
        std::cerr << "Application assembly failed: "
                  << app_result.error().message << "\n";
        return 1;
    }
    auto app = std::move(*app_result);

    // Headless batch mode
    if (!cli.image_dir.empty() || !cli.dataset_path.empty()) {
        return RunHeadless(cli, app);
    }

    // GUI mode
    return RunGui(argc, argv, app);
}
