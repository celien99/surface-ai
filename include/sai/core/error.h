#pragma once

#include <cstdint>
#include <source_location>
#include <string>

#include <tl/expected.hpp>

namespace sai {

enum class ErrorCode : std::uint32_t {
    Core_Unknown = 0,
    Core_ConstructionFailed,
    Core_TypeAlreadyRegistered,
    Core_TypeNotFound,
    Lifecycle_RegisterAfterAssembly,
    Memory_ArenaExhausted,
    Memory_RequestExceedsSlabSize,
    Memory_PoolExhausted,
    Plugin_VersionIncompatible,
    Plugin_CapabilityUnsupported,
    Plugin_LicenseInvalid,
    Plugin_CircularDependency,
    Runtime_QueueFull,
    Runtime_Cancelled,
    Runtime_NodeNotFound,
    Infra_LogSinkInitFailed,
    Infra_ConfigFileNotFound,       // ConfigStore::Load 指定路径不存在
    Infra_ConfigParseError,         // yaml-cpp 解析阶段失败（YAML 语法错误）
    Infra_ConfigValidationFailed,   // ConfigSchema::Validate 未通过
    Infra_ConfigKeyNotFound,        // ConfigStore::Get<T> 查询的键不存在
    Infra_ConfigKeyTypeMismatch,    // 键存在但无法转换为请求的 T
    Device_ConnectionFailed,
    Device_NotConnected,
    Device_AcquisitionInProgress,
    Image_UnsupportedPixelFormat,
    Image_DimensionMismatch,
    Image_PreprocessFailed,
    Io_ExportPathCreateFailed,
    Io_SerializationFailed,
    Io_ImportFileNotFound,
    Io_ImportParseFailed,
    Inference_EngineLoadFailed,
    Inference_EngineExecutionFailed,
    Inference_InvalidBinding,
    Inference_ReloadFailed,
    Inference_ModelConfigMismatch,
    Embedding_NotGpuImage,
    Embedding_DimensionMismatch,
    Embedding_InvalidPatchParameters,
    Detection_FeatureBankLoadFailed,
    Detection_InvalidPatchGrid,
    // Knowledge (M4)
    Knowledge_DbOpenFailed,
    Knowledge_SchemaMigrationFailed,
    Knowledge_NodeNotFound,
    Knowledge_EdgeNotFound,
    Knowledge_InvalidRelationship,
    Knowledge_SnapshotNotFound,
    Knowledge_SnapshotRestoreFailed,
    // Retrieval (M4)
    Retrieval_DimensionMismatch,  // reserved: query dim vs FeatureBank dim (validated by FAISS internally)
    Retrieval_EmptyIndex,
    Retrieval_FusionConfigInvalid,
    // Rule (M5)
    Rule_ParseError,
    Rule_InvalidPath,
    Rule_TypeMismatch,
    Rule_CyclicOverride,
    Rule_ReloadFailed,
    // Reasoner (M5)
    Reasoner_TreeLoadFailed,
    Reasoner_InvalidTree,
    Reasoner_ScoreComputationFailed,
    // Pipeline & Scheduler (M6)
    Pipeline_InvalidConfig,
    Pipeline_StageTypeMismatch,
    Pipeline_StageInitFailed,
    Pipeline_InvalidState,
    Pipeline_QueueFull,
    Scheduler_PoolNotFound,
    Scheduler_QueueCreateFailed,

    // Visualization (M7)
    Visualization_FrameBufferFull,
    Visualization_ConfigReloadFailed,
    Visualization_PipelineRestartFailed,

    // CoresetEvolution (online self-evolution)
    Detection_CoresetEvolution_UpdateFailed,
    Detection_CoresetEvolution_Degraded,
    Detection_CoresetEvolution_FullRebuildFailed,
    Detection_CoresetEvolution_ProfileLoadFailed,

    // Tuning (Bayesian auto-tuning of decision parameters)
    Tuning_SpaceEmpty,
    Tuning_ConstraintViolated,
    Tuning_ObjectiveEvalFailed,
    Tuning_RollbackTriggered,
    Tuning_ParameterApplyFailed,
};

struct ErrorInfo {
    ErrorCode code;
    std::string message;
    std::source_location source_location;
};

template <typename T>
using Result = tl::expected<T, ErrorInfo>;

}  // namespace sai
