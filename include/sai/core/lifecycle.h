#pragma once

namespace sai {

// Destroyed is the conceptual terminal state used to make the lifecycle
// diagram complete. Context::state_ never holds this value at runtime —
// there is no post-destruction state to query — so no API sets it explicitly.
enum class LifecycleState {
    Created,
    Initialized,
    Running,
    Stopped,
    Destroyed,
};

}  // namespace sai
