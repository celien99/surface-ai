#include "sai/visualization/config_viewmodel.h"

#include "sai/pipeline/pipeline.h"
#include "sai/pipeline/pipeline_config.h"
#include "sai/pipeline/stage_node.h"
#include "sai/rule/rule_engine.h"
#include "sai/reasoner/reasoner.h"
#include "sai/reasoner/decision_tree.h"

#include <yaml-cpp/yaml.h>
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

void ConfigViewModel::ApplyParameterChanges(const QString& yamlText) {
    if (stage_nodes_.empty()) {
        reload_status_ = "Warning: No stage nodes registered";
        emit reloadStatusChanged();
        return;
    }
    try {
        YAML::Node root = YAML::Load(yamlText.toStdString());
        // yamlText is the full pipeline YAML; extract per-stage config blocks
        auto stages_node = root["pipeline"]["stages"];
        if (!stages_node || !stages_node.IsSequence()) {
            reload_status_ = "Error: Invalid pipeline YAML structure";
            emit reloadStatusChanged();
            return;
        }
        int applied = 0;
        for (const auto& stage : stages_node) {
            auto id = stage["id"].as<std::string>();
            auto it = stage_nodes_.find(id);
            if (it != stage_nodes_.end() && stage["config"]) {
                auto result = it->second->ReloadConfig(stage["config"]);
                if (result) { applied++; }
            }
        }
        reload_status_ = QString("Applied to %1 stage(s) — next frame").arg(applied);
    } catch (const YAML::Exception& e) {
        reload_status_ = QString("YAML parse error: %1").arg(e.what());
    }
    needs_restart_ = false;
    emit reloadStatusChanged();
    emit needsRestartChanged();
}

void ConfigViewModel::ApplyRuleChanges(const QString& yamlText) {
    if (!rule_engine_) {
        reload_status_ = "Error: RuleEngine not bound";
        emit reloadStatusChanged();
        return;
    }
    // Write to temp file and reload
    auto tmp = std::filesystem::temp_directory_path() / "surface_ai_rules_temp.yaml";
    {
        std::ofstream out(tmp);
        out << yamlText.toStdString();
    }
    auto result = rule_engine_->LoadFromYAML(tmp);
    if (result) {
        reload_status_ = "Rules hot-reloaded successfully";
    } else {
        reload_status_ = QString("Rule reload failed: %1").arg(result.error().message.c_str());
    }
    needs_restart_ = false;
    emit reloadStatusChanged();
}

void ConfigViewModel::ApplyTreeChanges(const QString& yamlText) {
    if (!reasoner_) {
        reload_status_ = "Error: Reasoner not bound";
        emit reloadStatusChanged();
        return;
    }
    auto tmp = std::filesystem::temp_directory_path() / "surface_ai_tree_temp.yaml";
    {
        std::ofstream out(tmp);
        out << yamlText.toStdString();
    }
    auto result = reasoner_->ReloadTree(tmp);
    if (result) {
        reload_status_ = "Decision tree hot-reloaded successfully";
    } else {
        reload_status_ = QString("Tree reload failed: %1").arg(result.error().message.c_str());
    }
    needs_restart_ = false;
    emit reloadStatusChanged();
}

void ConfigViewModel::RestartPipeline() {
    if (!pipeline_) {
        reload_status_ = "Error: Pipeline not bound";
        emit reloadStatusChanged();
        return;
    }
    reload_status_ = "Draining in-flight frames...";
    emit reloadStatusChanged();

    auto drain_result = pipeline_->Drain();
    if (!drain_result) {
        reload_status_ = QString("Drain failed: %1").arg(drain_result.error().message.c_str());
        emit reloadStatusChanged();
        return;
    }

    auto stop_result = pipeline_->Stop();
    if (!stop_result) {
        reload_status_ = QString("Stop failed: %1").arg(stop_result.error().message.c_str());
        emit reloadStatusChanged();
        return;
    }

    reload_status_ = "Pipeline restarted successfully";
    needs_restart_ = false;
    emit reloadStatusChanged();
    emit needsRestartChanged();
}

void ConfigViewModel::ValidateYaml(const QString& text) {
    validation_errors_.clear();
    if (text.trimmed().isEmpty()) {
        validation_errors_ << "YAML is empty";
        emit validationErrorsChanged();
        return;
    }
    try {
        YAML::Load(text.toStdString());
        // Parse succeeded — no errors
    } catch (const YAML::Exception& e) {
        validation_errors_ << QString("Line %1: %2").arg(e.mark.line + 1).arg(e.what());
    }
    emit validationErrorsChanged();
}

}  // namespace sai::visualization
