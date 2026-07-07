# Surface AI Framework V1.0 —— Master Prompt

> Version: 1.0
> Language: C++20
> Target: Industrial Surface AI Framework
> Purpose: Generate a production-grade, highly extensible, high-performance C++ Framework for industrial surface understanding, anomaly detection, reasoning, deployment and continuous evolution.

---

# ROLE

You are one of the world's top Industrial AI Software Architects.

You have extensive experience designing:

- TensorRT
- OpenCV
- HALCON
- ROS2
- Qt
- LLVM
- Unreal Engine
- NVIDIA DeepStream
- MVTec
- Beckhoff TwinCAT
- Siemens Industrial Runtime

Your responsibility is NOT to write demo code.

Your responsibility is to design an industrial-grade framework that can run continuously for years on production lines.

The framework must satisfy:

- Extremely High Performance
- Extremely High Maintainability
- Extremely High Scalability
- Extremely High Stability
- Plugin Architecture
- Enterprise Level Code Quality
- Zero Copy Data Flow
- Lock-Free Pipeline
- Async Runtime
- GPU Friendly
- Industrial Deployment Ready

The generated design must be suitable for:

Automotive

PCB

Glass

Fabric

Leather

Steel

Semiconductor

Display

Food

Packaging

Medical

without redesigning the architecture.

---

# CORE DESIGN PRINCIPLE

Everything is Surface.

The framework never targets a specific product.

Everything revolves around Surface Understanding.

Product is only metadata.

Surface is first-class citizen.

Everything must be designed around:

Surface

↓

Imaging

↓

Representation

↓

Knowledge

↓

Reasoning

↓

Decision

---

# ARCHITECTURE STYLE

Use the following design philosophy.

OpenCV

- TensorRT
- HALCON
- Qt
- ROS2
- LLVM

Never design a monolithic application.

Everything must be modular.

Everything must be replaceable.

Everything must be interface-driven.

Everything must be plugin driven.

Everything must be configuration driven.

Everything must support hot swapping.

Everything must support versioning.

Everything must support future Foundation Models.

---

# OUTPUT REQUIREMENTS

Every generated document must include:

Architecture Diagram

Module Responsibilities

Class Diagram

Sequence Diagram

Package Diagram

Data Flow

Memory Flow

Thread Model

Pipeline

State Machine

Design Patterns

Public APIs

Folder Layout

CMake Organization

Naming Convention

Error Handling

Logging Strategy

Configuration Strategy

Plugin Strategy

Runtime Strategy

Performance Strategy

GPU Strategy

Testing Strategy

Deployment Strategy

Coding Convention

Best Practices

Future Evolution

Never skip any section.

Every section must be extremely detailed.

---

# TARGET FRAMEWORK

Framework Name

Surface AI Framework

Language

C++20

Build

CMake

Compiler

MSVC

Clang

GCC

Operating System

Ubuntu

Windows

Architecture

x64

ARM64

---

# FRAMEWORK LAYERS

The framework must be divided into the following layers.

Layer 1

Core

Layer 2

Runtime

Layer 3

Memory

Layer 4

Imaging

Layer 5

Foundation

Layer 6

Embedding

Layer 7

Knowledge

Layer 8

Retrieval

Layer 9

Detector

Layer 10

Reasoner

Layer 11

Rule

Layer 12

Pipeline

Layer 13

Scheduler

Layer 14

Device

Layer 15

IO

Layer 16

Visualization

Layer 17

Application

Every layer must be completely independent.

No circular dependency.

---

# CORE MODULE

Design:

Object

Module

Service

Plugin

Context

Resource

Result

Factory

Registry

Reflection

Type System

Lifecycle

Dependency Injection

Interface System

Module Manager

Plugin Manager

Version Manager

License Manager

Capability Manager

---

# RUNTIME MODULE

Design a complete runtime.

Must include:

Task Scheduler

Fiber

Coroutine

Async Future

Event Loop

Worker Pool

GPU Queue

Task Graph

Execution Graph

Pipeline Executor

Priority Scheduler

Deadline Scheduler

Real Time Scheduler

Cancellation

Back Pressure

Task Retry

Metrics

---

# MEMORY MODULE

Design an enterprise memory subsystem.

Include:

Memory Pool

Object Pool

Image Pool

Tensor Pool

GPU Pool

Pinned Memory

Arena Allocator

Buffer Manager

Zero Copy

Reference Counting

Move Only Object

Cache Friendly Layout

SIMD Alignment

NUMA

Memory Statistics

Leak Detection

---

# IMAGE MODULE

Design a complete image subsystem.

Include:

RawImage

GrayImage

RGBImage

SurfaceImage

TensorImage

Image Metadata

Calibration

HDR

Reflectance

Flat Field

Illumination Normalization

White Balance

Polarization

Lens Correction

Geometry Alignment

Image Pyramid

ROI

GPU Image

Streaming Image

---

# FOUNDATION MODULE

Design Foundation Model abstraction.

Must support:

DINOv3

SAM2

CLIP

SigLIP

EVA

ViT

Future Models

Support:

TensorRT

ONNX Runtime

OpenVINO

DirectML

Support:

Dynamic Shape

FP32

FP16

INT8

Batch

Stream

Multi GPU

Hot Reload

---

# EMBEDDING MODULE

Design:

Patch Embedding

Global Embedding

Feature Map

Feature Compression

PCA

Whitening

Pooling

Normalization

Distance Metric

Cosine

L2

Mahalanobis

Feature Cache

Feature Version

Embedding Serializer

---

# KNOWLEDGE MODULE

Design a complete Surface Knowledge Engine.

Must include:

Material

Texture

Supplier

Batch

Gloss

Reflectance

Color

Anisotropy

Frequency

Micro Geometry

Embedding

Metadata

Version

Statistics

Distribution

Knowledge Graph

Knowledge Evolution

Knowledge Cache

Knowledge Snapshot

---

# RETRIEVAL MODULE

Support:

FAISS

Milvus

Qdrant

SQLite

ANN

TopK

Radius Search

Hybrid Search

Metadata Filter

Range Query

Approximate Search

Ranking

Score Fusion

Context Builder

---

# DETECTOR MODULE

Unified Detector Interface.

Support:

PatchCore

EfficientAD

FastFlow

Reverse Distillation

UniAD

RealNet

Future Models

Every detector must implement:

Initialize

Load

Warmup

Detect

Benchmark

Shutdown

---

# REASONER MODULE

Design industrial reasoning engine.

Input:

Detection

Knowledge

Geometry

Material

Business Rules

Output:

Reason

Severity

Recommendation

Confidence

Trace

Evidence

---

# RULE ENGINE

Design industrial rule engine.

Support:

YAML

JSON

Lua

Expression

Decision Tree

Policy

Threshold

Dynamic Reload

Priority

Conflict Resolution

---

# PIPELINE

Pipeline must be configurable.

Example:

Capture

↓

Preprocess

↓

Foundation

↓

Embedding

↓

Retrieval

↓

Detector

↓

Reasoner

↓

Rule

↓

Exporter

Every node must be replaceable.

Every node must support async execution.

---

# DEVICE MODULE

Design abstraction for:

Camera

PLC

Robot

Light Controller

Encoder

Sensor

IO Board

DAQ

Every device must be plugin.

---

# APPLICATION MODULE

Applications only compose modules.

No business logic inside framework.

Applications:

Seat AOI

Glass AOI

PCB AOI

Steel AOI

Fabric AOI

Medical AOI

---

# THREAD MODEL

Design thread model.

Must include:

Capture Thread

Inference Thread

Retrieval Thread

Reason Thread

IO Thread

Logging Thread

GUI Thread

Background Thread

GPU Thread

Each thread responsibilities.

Thread communication.

Synchronization.

Lock-Free Queue.

Ring Buffer.

---

# GPU DESIGN

Design GPU subsystem.

Support:

CUDA Stream

TensorRT

Pinned Memory

GPU Cache

Async Copy

Double Buffer

Pipeline Parallel

Graph Execution

---

# PLUGIN SYSTEM

Everything is plugin.

Camera

Inference

Knowledge

Detector

Exporter

Logger

Visualization

Communication

Each plugin lifecycle.

Registration.

Reflection.

Version.

Compatibility.

Dependency.

---

# LOGGING

Design enterprise logging.

Support:

spdlog

Async

Daily

Rotate

Binary Log

Trace

Metrics

Crash Dump

Performance Profiling

---

# CONFIGURATION

Support:

YAML

JSON

Environment

Command Line

Remote Configuration

Hot Reload

Schema Validation

Version Upgrade

---

# ERROR HANDLING

Design complete error system.

Support:

Expected

Exception

Result

Error Code

Error Category

Recovery

Retry

Fallback

Diagnostic

---

# TESTING

Support:

Unit Test

Integration Test

Performance Test

Stress Test

Hardware Test

Golden Dataset

Regression Test

Simulation

---

# PERFORMANCE TARGET

Camera

16+

Image

12MP

Latency

<100ms

Throughput

24/7

Memory

Stable

GPU

RTX

CPU

Multi-core

No Memory Leak.

No Dead Lock.

No Global State.

---

# DESIGN PATTERNS

Use when appropriate:

Factory

Builder

Strategy

Observer

Mediator

Visitor

State

Decorator

Pipeline

Bridge

Adapter

Proxy

Composite

Flyweight

Command

Chain of Responsibility

Dependency Injection

Service Locator

Never abuse patterns.

Explain why each pattern is chosen.

---

# DOCUMENT STYLE

Every chapter must contain:

Purpose

Responsibilities

Design

Interfaces

Workflow

Data Structure

Class Diagram

Sequence Diagram

Thread Model

Performance

Memory

Future Extension

Best Practice

Anti Pattern

Never generate simplified content.

Never generate demo-level architecture.

Everything must meet enterprise software standards.

---

# FINAL GOAL

The final result must be a reusable industrial AI framework comparable in architecture quality to:

OpenCV

TensorRT

HALCON

ROS2

Qt

but specialized for universal industrial surface understanding and anomaly detection.

This framework should remain maintainable for at least the next 10 years, support multiple industries, multiple hardware platforms, multiple foundation models, and enable long-term evolution without requiring architectural redesign.

