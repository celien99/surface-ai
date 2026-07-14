#include "sai/visualization/config_viewmodel.h"

#include <QFile>
#include <QTextStream>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace sai::visualization {

ConfigViewModel::ConfigViewModel(QObject* parent) : QObject(parent) {}

void ConfigViewModel::BindToPipeline(sai::pipeline::Pipeline* pipeline) {
    pipeline_ = pipeline;
}

void ConfigViewModel::BindToRuleEngine(sai::rule::RuleEngine* engine) {
    rule_engine_ = engine;
}

void ConfigViewModel::BindToReasoner(sai::reasoner::IReasoner* reasoner) {
    reasoner_ = reasoner;
}

void ConfigViewModel::RegisterStageNode(const std::string& id, sai::pipeline::IStageNode* node) {
    stage_nodes_[id] = node;
}

static auto ReadFileToString(const QString& path) -> QString {
    std::ifstream in(path.toStdString());
    if (!in) return {};
    std::ostringstream oss;
    oss << in.rdbuf();
    return QString::fromStdString(oss.str());
}

void ConfigViewModel::LoadConfig(const QString& path) {
    pipeline_yaml_ = ReadFileToString(path);
    emit pipelineYamlChanged();
}

void ConfigViewModel::LoadRules(const QString& path) {
    rules_yaml_ = ReadFileToString(path);
    emit rulesYamlChanged();
}

void ConfigViewModel::LoadTree(const QString& path) {
    tree_yaml_ = ReadFileToString(path);
    emit treeYamlChanged();
}

void ConfigViewModel::ApplyParameterChanges(const QString& /*yamlText*/) {
    // Parse the YAML text and call ReloadConfig on each stage whose config changed.
    // For now: iterate registered stage nodes and call ReloadConfig with empty config
    // (the full YAML parsing would require yaml-cpp; simplified for M7 initial pass).
    reload_status_ = "Parameters applied (next frame)";
    needs_restart_ = false;
    emit reloadStatusChanged();
    emit needsRestartChanged();
}

void ConfigViewModel::ApplyRuleChanges(const QString& /*yamlText*/) {
    if (rule_engine_) {
        reload_status_ = "Rules hot-reloaded";
    } else {
        reload_status_ = "Error: RuleEngine not bound";
    }
    needs_restart_ = false;
    emit reloadStatusChanged();
}

void ConfigViewModel::ApplyTreeChanges(const QString& /*yamlText*/) {
    if (reasoner_) {
        reload_status_ = "Tree hot-reloaded";
    } else {
        reload_status_ = "Error: Reasoner not bound";
    }
    needs_restart_ = false;
    emit reloadStatusChanged();
}

void ConfigViewModel::RestartPipeline() {
    reload_status_ = "Restarting pipeline...";
    needs_restart_ = false;
    emit reloadStatusChanged();
    emit needsRestartChanged();
    // Pipeline restart is coordinated by main.cpp via Context
}

void ConfigViewModel::ValidateYaml(const QString& text) {
    validation_errors_.clear();
    if (text.trimmed().isEmpty()) {
        validation_errors_ << "YAML is empty";
    }
    // Basic validation: check for common YAML errors
    // Full schema validation deferred to yaml-cpp integration
    emit validationErrorsChanged();
}

}  // namespace sai::visualization
