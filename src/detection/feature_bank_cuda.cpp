// feature_bank_cuda.cpp — FeatureBank FAISS GPU 后端（CUDA 门控）
// 仅在 CUDAToolkit_FOUND 时由 CMake 编译。
// 当前为占位（Task 8 可移植路径优先）；后续 Task 与 feature_bank.cpp 合并 LoadFromFile，
// 并包装为 faiss::gpu::StandardGpuResources → faiss::gpu::index_cpu_to_gpu。
