// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// CUDA graphs: CaptureGraph, LaunchCapturedGraph, DestroyCapturedGraph and IsStreamCapturing live in
// CudaBackend.Graph.cs.
public sealed partial class CudaBackend : IGpuGraphCapture
{
    bool IGpuGraphCapture.SupportsGraphCapture => IsAvailable;

    bool IGpuGraphCapture.ContextIsLive => ContextIsLive;
}