// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// Add then activation in one pass (AddRelu / AddSigmoid / AddGelu, the residual join of a ResNet block), compiled as its
// own library on first use: the same arithmetic as the element-wise add followed by the activation library's relu,
// sigmoid and gelu, without writing and re-reading the sum.
public sealed partial class MetalBackend
{
    private const string AddActivationLibraryName = "AddActivation";

    internal const string AddActivationSource = @"
#include <metal_stdlib>
using namespace metal;

constant float ADD_ACT_SQRT_2_OVER_PI = 0.7978845608028654f;
constant float ADD_ACT_GELU_COEF = 0.044715f;

kernel void add_relu(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // relu's contract: a NaN sum stays NaN (tested on the bits, which fast math cannot fold), -0 becomes +0.
    if (gid >= size) return;
    float s = A[gid] + B[gid];
    C[gid] = ((as_type<uint>(s) & 0x7fffffffu) > 0x7f800000u) ? s : (s > 0.0f ? s : 0.0f);
}

kernel void add_sigmoid(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = A[gid] + B[gid];
    // The activation library's numerically stable form.
    if (x >= 0.0f) {
        C[gid] = 1.0f / (1.0f + exp(-x));
    } else {
        float exp_x = exp(x);
        C[gid] = exp_x / (1.0f + exp_x);
    }
}

kernel void add_gelu(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    float x = A[gid] + B[gid];
    float inner = ADD_ACT_SQRT_2_OVER_PI * (x + ADD_ACT_GELU_COEF * x * x * x);
    C[gid] = 0.5f * x * (1.0f + tanh(inner));
}
";

    private IntPtr _addActivationLibrary;

    public void AddRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => LaunchAddActivation("add_relu", A, B, C, size);

    public void AddSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => LaunchAddActivation("add_sigmoid", A, B, C, size);

    public void AddGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size) => LaunchAddActivation("add_gelu", A, B, C, size);

    private void LaunchAddActivation(string kernelName, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int size)
    {
        ThrowIfDisposed();
        if (size <= 0) return;
        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer || C is not MetalGpuBuffer cBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        if (_addActivationLibrary == IntPtr.Zero)
        {
            lock (_tunedSlotsLock)
            {
                if (_addActivationLibrary == IntPtr.Zero)
                    _addActivationLibrary = _shaderLibrary.CompileLibrary(AddActivationLibraryName, AddActivationSource);
            }
        }

        var pipeline = GetPipeline(AddActivationLibraryName, _addActivationLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(size);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(aBuffer, 0);
        encoder.SetBuffer(bBuffer, 1);
        encoder.SetBuffer(cBuffer, 2);
        encoder.SetBytes((uint)size, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
