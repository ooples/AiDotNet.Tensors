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
    private readonly int[]? _gradientIndex;
    private IGpuBuffer? _sumOfSquares;
    private IGpuBuffer? _scale;
    private uint[]? _record;

    internal DeviceGradientNorm(IMultiTensorKernels kernels, IGpuBuffer record, IGpuBuffer? scale, int[]? gradientIndex = null)
    {
        _kernels = kernels;
        _sumOfSquares = record;
        _scale = scale;
        _gradientIndex = gradientIndex;
    }

    // The whole record in one device-to-host read, on first access.
    private uint[] Record
    {
        get
        {
            if (_record is { } r) return r;
            var buffer = _sumOfSquares ?? throw new ObjectDisposedException(nameof(DeviceGradientNorm));
            if (_kernels is not IDirectGpuBackend backend)
                throw new InvalidOperationException("The norm's kernels are not a GPU backend.");
            return _record = MultiTensorArgs.RecordFromSlots(backend.DownloadBuffer(buffer));
        }
    }

    /// <summary>The sum of squares of every gradient element, correctly rounded to double from the exact
    /// (<see cref="DeterministicGradientNorm"/>) sum: the same bits on every device. NaN when an element was NaN,
    /// +Inf when one was infinite.</summary>
    public double TotalNormSquared
    {
        get
        {
            var r = Record;
            return BitConverter.Int64BitsToDouble((long)(((ulong)r[DeterministicGradientNorm.WordSumSquaresHi] << 32)
                | r[DeterministicGradientNorm.WordSumSquaresLo]));
        }
    }

    /// <summary>The total L2 norm, in double.</summary>
    public double TotalNorm => Math.Sqrt(TotalNormSquared);

    /// <summary>The total L2 norm as fp32, correctly rounded from the exact sum (+Inf past float range): the value
    /// <c>torch.nn.utils.clip_grad_norm_</c> returns for fp32 gradients, without its overflow.</summary>
    public float Norm => DeterministicGradientNorm.UInt32BitsToSingle(Record[DeterministicGradientNorm.WordNorm]);

    /// <summary>The factor the gradients were multiplied by, min(1, maxNorm / (norm + 1e-6)); 1 when the call only
    /// measured the norm.</summary>
    public float ClipCoefficient => _scale is null ? 1f
        : DeterministicGradientNorm.UInt32BitsToSingle(Record[DeterministicGradientNorm.WordScale]);

    /// <summary>False when a gradient element was NaN or infinite.</summary>
    public bool IsFinite => Record[DeterministicGradientNorm.WordNonFinite] == 0;

    /// <summary>NaN elements across every gradient (saturating at <see cref="uint.MaxValue"/>).</summary>
    public long NanCount => Record[DeterministicGradientNorm.WordNanCount];

    /// <summary>Infinite elements across every gradient (saturating at <see cref="uint.MaxValue"/>).</summary>
    public long InfinityCount => Record[DeterministicGradientNorm.WordInfCount];

    /// <summary>The index, in the list passed in, of the first gradient holding a NaN or infinite element; -1 when
    /// every element was finite.</summary>
    public int FirstNonFiniteGradient
    {
        get
        {
            uint t = Record[DeterministicGradientNorm.WordFirstNonFiniteTensor];
            if (t == uint.MaxValue) return -1;
            return _gradientIndex is { } map ? map[(int)t] : (int)t;
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
