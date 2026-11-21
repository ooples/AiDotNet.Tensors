// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// The shortest double vector whose reduction goes to the device. A vector's values live on the host, so the device
    /// pays an upload per call; below this length the upload costs more than the CPU reduction.
    /// </summary>
    /// <remarks>
    /// Measured on an OpenCL device with fp64 against this library's CPU engine (best of 20, milliseconds):
    /// <code>
    /// elements   cpu sum   upload + device sum   device sum, data resident
    ///     65,536     0.44          0.49                 0.16
    ///  1,048,576     5.60          4.08                 0.27
    /// 16,777,216    88.4          48.4                  1.54
    /// </code>
    /// The upload is the whole cost: with the data already on the device the reduction is 20x faster at a million
    /// elements. The threshold is where the upload-paying path first wins, and it is what device-resident double
    /// tensors will remove.
    /// </remarks>
    internal const int Float64ReductionMinLength = 1 << 20;

    private bool TryGetFloat64Reductions(int length, out IDirectGpuBackend backend, out IGpuFloat64Reductions reductions)
    {
        reductions = null!;
        if (length < Float64ReductionMinLength || !TryGetBackend(out backend))
        {
            backend = null!;
            return false;
        }

        if (backend is IGpuFloat64Reductions candidate && candidate.SupportsFloat64Reductions
            && backend.SupportsElementType(GpuElementType.Float64))
        {
            reductions = candidate;
            return true;
        }

        return false;
    }

    private bool TrySumFloat64(Vector<double> v, out double sum)
    {
        sum = 0.0;
        if (!TryGetFloat64Reductions(v.Length, out var backend, out var reductions)) return false;
        try
        {
            using var buffer = backend.AllocateBuffer(v.Length, GpuElementType.Float64);
            backend.UploadBuffer<double>(buffer, v.AsSpan());
            sum = reductions.SumFloat64(buffer, v.Length);
            return true;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("SumFloat64", capabilityGap);
            return false;
        }
    }

    private bool TryDotFloat64(Vector<double> a, Vector<double> b, out double dot)
    {
        dot = 0.0;
        // A length mismatch is the CPU engine's error to report.
        if (a.Length != b.Length || !TryGetFloat64Reductions(a.Length, out var backend, out var reductions)) return false;
        try
        {
            using var left = backend.AllocateBuffer(a.Length, GpuElementType.Float64);
            backend.UploadBuffer<double>(left, a.AsSpan());
            if (ReferenceEquals(a, b))
            {
                dot = reductions.DotFloat64(left, left, a.Length);
                return true;
            }

            using var right = backend.AllocateBuffer(b.Length, GpuElementType.Float64);
            backend.UploadBuffer<double>(right, b.AsSpan());
            dot = reductions.DotFloat64(left, right, a.Length);
            return true;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("DotFloat64", capabilityGap);
            return false;
        }
    }
}
