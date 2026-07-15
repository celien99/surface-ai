# Surface AI Framework — Production Docker Image
# Requires nvidia-container-toolkit for GPU access.
#
# Build:
#   docker build -t surface-ai:latest .
#
# Run (detection):
#   docker run --runtime=nvidia -v ./samples:/data/samples:ro \
#     -v ./results:/data/results surface-ai:latest \
#     detect --image-dir /data/samples/ --coreset /app/resources/coreset.bin \
#     --output-dir /data/results/
#
# Run (training):
#   docker run --runtime=nvidia -v ./normal_samples:/data/normal:ro \
#     -v ./coresets:/data/coresets surface-ai:latest \
#     train --image-dir /data/normal/ --coreset-output /data/coresets/coreset.bin

FROM nvidia/cuda:12.4-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libaravis-0.8-0 \
    libopen62541-1 \
    libfaiss0 \
    libspdlog1 \
    libyaml-cpp0.7 \
    libsqlite3-0 \
    qt6-base-dev \
    libgl1-mesa-glx \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

# Copy application binary and resources
COPY build/default/apps/seat-aoi/seat_aoi /app/seat_aoi
COPY resources/ /app/resources/
COPY apps/seat-aoi/resources/ /app/resources/

WORKDIR /app

# Default entrypoint: show usage
ENTRYPOINT ["./seat_aoi"]
CMD ["--help"]
