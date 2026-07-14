#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <map>
#include <string>

namespace sai::pipeline {
class Pipeline;
class IStageNode;
}  // namespace sai::pipeline

namespace sai::rule {
class RuleEngine;
}  // namespace sai::rule

namespace sai::reasoner {
class IReasoner;
}  // namespace sai::reasoner

namespace sai::visualization {

class ConfigViewModel : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString pipelineYaml READ pipelineYaml WRITE setPipelineYaml NOTIFY pipelineYamlChanged)
    Q_PROPERTY(QString rulesYaml READ rulesYaml WRITE setRulesYaml NOTIFY rulesYamlChanged)
    Q_PROPERTY(QString treeYaml READ treeYaml WRITE setTreeYaml NOTIFY treeYamlChanged)
    Q_PROPERTY(QString reloadStatus READ reloadStatus NOTIFY reloadStatusChanged)
    Q_PROPERTY(bool needsRestart READ needsRestart NOTIFY needsRestartChanged)
    Q_PROPERTY(QStringList validationErrors READ validationErrors NOTIFY validationErrorsChanged)
public:
    explicit ConfigViewModel(QObject* parent = nullptr);

    void BindToPipeline(sai::pipeline::Pipeline* pipeline);
    void BindToRuleEngine(sai::rule::RuleEngine* engine);
    void BindToReasoner(sai::reasoner::IReasoner* reasoner);

    // Store stage nodes for ReloadConfig access
    void RegisterStageNode(const std::string& id, sai::pipeline::IStageNode* node);

    auto pipelineYaml() const -> QString { return pipeline_yaml_; }
    auto rulesYaml() const -> QString { return rules_yaml_; }
    auto treeYaml() const -> QString { return tree_yaml_; }
    auto reloadStatus() const -> QString { return reload_status_; }
    auto needsRestart() const -> bool { return needs_restart_; }
    auto validationErrors() const -> QStringList { return validation_errors_; }

    void setPipelineYaml(const QString& yaml) { pipeline_yaml_ = yaml; emit pipelineYamlChanged(); }
    void setRulesYaml(const QString& yaml) { rules_yaml_ = yaml; emit rulesYamlChanged(); }
    void setTreeYaml(const QString& yaml) { tree_yaml_ = yaml; emit treeYamlChanged(); }

    Q_INVOKABLE void LoadConfig(const QString& path);
    Q_INVOKABLE void LoadRules(const QString& path);
    Q_INVOKABLE void LoadTree(const QString& path);
    Q_INVOKABLE void ApplyParameterChanges(const QString& yamlText);
    Q_INVOKABLE void ApplyRuleChanges(const QString& yamlText);
    Q_INVOKABLE void ApplyTreeChanges(const QString& yamlText);
    Q_INVOKABLE void RestartPipeline();
    Q_INVOKABLE void ValidateYaml(const QString& text);

signals:
    void pipelineYamlChanged();
    void rulesYamlChanged();
    void treeYamlChanged();
    void reloadStatusChanged();
    void needsRestartChanged();
    void validationErrorsChanged();

private:
    sai::pipeline::Pipeline* pipeline_{nullptr};
    sai::rule::RuleEngine* rule_engine_{nullptr};
    sai::reasoner::IReasoner* reasoner_{nullptr};
    std::map<std::string, sai::pipeline::IStageNode*> stage_nodes_;

    QString pipeline_yaml_;
    QString rules_yaml_;
    QString tree_yaml_;
    QString reload_status_{"Ready"};
    bool needs_restart_{false};
    QStringList validation_errors_;
};

}  // namespace sai::visualization
