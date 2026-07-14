#include <gtest/gtest.h>
#include <QGuiApplication>

#include "sai/visualization/config_viewmodel.h"

TEST(ConfigViewModelTest, LoadConfigReadsFile) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::ConfigViewModel vm;
    // Loading a non-existent file should produce empty string (not crash)
    vm.LoadConfig("nonexistent.yaml");
    EXPECT_TRUE(vm.pipelineYaml().isEmpty());
}

TEST(ConfigViewModelTest, ValidateYamlValidInput) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::ConfigViewModel vm;
    vm.ValidateYaml("key: value\nnested:\n  sub: 42");
    EXPECT_TRUE(vm.validationErrors().isEmpty());
}

TEST(ConfigViewModelTest, ValidateYamlEmptyInput) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::ConfigViewModel vm;
    vm.ValidateYaml("");
    EXPECT_FALSE(vm.validationErrors().isEmpty());
}

TEST(ConfigViewModelTest, ValidateYamlInvalidSyntax) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::ConfigViewModel vm;
    vm.ValidateYaml("invalid: [unclosed\n  - list");
    EXPECT_FALSE(vm.validationErrors().isEmpty());
}

TEST(ConfigViewModelTest, ApplyRuleChangesWithoutEngine) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::ConfigViewModel vm;
    vm.ApplyRuleChanges("rules: []");
    EXPECT_TRUE(vm.reloadStatus().startsWith("Error"));
}

TEST(ConfigViewModelTest, ApplyTreeChangesWithoutReasoner) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::ConfigViewModel vm;
    vm.ApplyTreeChanges("tree: {}");
    EXPECT_TRUE(vm.reloadStatus().startsWith("Error"));
}

TEST(ConfigViewModelTest, NeedsRestartDefaultsFalse) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::ConfigViewModel vm;
    EXPECT_FALSE(vm.needsRestart());
}

TEST(ConfigViewModelTest, SignalEmittedOnPropertyChange) {
    int argc = 0;
    QGuiApplication app(argc, nullptr);

    sai::visualization::ConfigViewModel vm;

    int pipeline_changed = 0;
    QObject::connect(&vm, &sai::visualization::ConfigViewModel::pipelineYamlChanged,
                     [&]() { pipeline_changed++; });

    vm.setPipelineYaml("test: yaml content");
    EXPECT_EQ(pipeline_changed, 1);
    EXPECT_EQ(vm.pipelineYaml(), "test: yaml content");
}
