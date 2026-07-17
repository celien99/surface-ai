#pragma once

struct AssembledApp;
struct CliArgs;

int RunGui(int argc, char* argv[], AssembledApp& app, const CliArgs& cli);
