#include <iostream>

#include "app_builder.h"
#include "cli_args.h"
#include "coreset_builder.h"
#include "gui_runner.h"
#include "headless_runner.h"

auto main(int argc, char* argv[]) -> int {
    auto cli = ParseArgs(argc, argv);

    // ── Train mode: build coreset and exit ──
    if (cli.train_mode) {
        return BuildCoreset(cli);
    }

    // ── Review mode: GUI without Pipeline / Camera / GPU ──
    // Does NOT call AssembleApplication — review mode reads JSON+images from
    // disk and only uses ViewModels. No model files or CUDA required.
    if (!cli.review_dir.empty()) {
        AssembledApp app;  // empty shell, review mode doesn't touch it
        return RunGui(argc, argv, app, cli);
    }

    // ── Headless batch mode ──
    if (!cli.image_dir.empty() || !cli.dataset_path.empty()) {
        auto app_result = AssembleApplication(cli);
        if (!app_result) {
            std::cerr << "Application assembly failed: "
                      << app_result.error().message << "\n";
            return 1;
        }
        auto app = std::move(*app_result);
        return RunHeadless(cli, app);
    }

    // ── GUI live mode (FakeCamera) ──
    auto app_result = AssembleApplication(cli);
    if (!app_result) {
        std::cerr << "Application assembly failed: "
                  << app_result.error().message << "\n";
        return 1;
    }
    auto app = std::move(*app_result);
    return RunGui(argc, argv, app, cli);
}
