// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// The total gradient norm of a device-side <see cref="GpuOptimizer.TryClipGradNorm"/>, held on the device. Like the
/// tensor <c>torch.nn.utils.clip_grad_norm_</c> returns, reading it is what synchronizes: the clip itself never does.
/// </summary>
public sealed class DeviceGradientNorm : IDisposable
{
    private readonly IMultiTensorKernels _kernels;
    private IGpuBuffer? _sumOfSquares;
    private IGpuBuffer? _scale;
    private double? _value;

    internal DeviceGradientNorm(IMultiTensorKernels kernels, IGpuBuffer sumOfSquares, IGpuBuffer? scale)
    {
        _kernels = kernels;
        _sumOfSquares = sumOfSquares;
        _scale = scale;
    }

    /// <summary>The sum of squares of every gradient element (one device-to-host read on first access).</summary>
    public double TotalNormSquared
    {
        get
        {
            if (_value is { } v) return v;
            var buffer = _sumOfSquares ?? throw new ObjectDisposedException(nameof(DeviceGradientNorm));
            double read = _kernels.ReadSumOfSquares(buffer);
            _value = read;
            return read;
        }
    }

    /// <summary>The total L2 norm.</summary>
    public double TotalNorm => Math.Sqrt(TotalNormSquared);

    /// <summary>False when a gradient element was NaN or infinite (the gradients were then left unscaled).</summary>
    public bool IsFinite
    {
        get
        {
            double sq = TotalNormSquared;
            return !double.IsNaN(sq) && !double.IsInfinity(sq);
        }
    }

    /// <summary>Releases the device scratch buffers.</summary>
    public void Dispose()
    {
        _sumOfSquares?.Dispose();
        _sumOfSquares = null;
        _scale?.Dispose();
        _scale = null;
    }
}
