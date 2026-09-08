// Copyright (c) AiDotNet. All rights reserved.
// Hardware-independent capability-contract tests for HIP half-precision GEMM.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class HipHalfPrecisionCapabilityContractTests
{
    [Fact]
    public void MatMulBackwardFp16Fused_DirectForwardFallbackWithoutHipBlas_IsRejectedBeforeDispatch()
    {
        var backend = CreateDirectFallbackOnlyBackend();
        IGpuHalfPrecisionBackend halfPrecision = backend;
        using var buffer = new MockGpuBuffer(new float[1]);

        Assert.True(halfPrecision.SupportsHgemm);
        Assert.False(halfPrecision.SupportsFp16FusedBackward);

        var exception = Assert.Throws<NotSupportedException>(() =>
            halfPrecision.MatMulBackwardFp16Fused(
                buffer, buffer, buffer, buffer, buffer,
                m: 1, n: 1, k: 1, gradOutHalf: false));

        Assert.Contains("requires a compatible hipBLAS installation", exception.Message, StringComparison.Ordinal);
    }

    private static HipBackend CreateDirectFallbackOnlyBackend()
    {
#pragma warning disable SYSLIB0050 // Deliberately bypass hardware initialization to test capability dispatch.
        var backend = (HipBackend)FormatterServices.GetUninitializedObject(typeof(HipBackend));
#pragma warning restore SYSLIB0050
        var kernelCache = new Dictionary<string, IntPtr>(StringComparer.Ordinal)
        {
            ["convert_fp32_to_fp16"] = new IntPtr(1),
            ["convert_fp16_to_fp32"] = new IntPtr(2)
        };

        SetRequiredField(backend, "_kernelCache", kernelCache);
        SetRequiredField(backend, "_scalarGemmF32", new IntPtr(3));
        return backend;
    }

    private static void SetRequiredField(HipBackend backend, string fieldName, object value)
    {
        var field = typeof(HipBackend).GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"Required HipBackend field '{fieldName}' was not found.");
        field.SetValue(backend, value);
    }
}
