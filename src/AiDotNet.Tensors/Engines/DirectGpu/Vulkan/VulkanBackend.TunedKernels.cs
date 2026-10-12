using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// Tuned-kernel registry dispatch on Vulkan (#1115): softmax and its backward, LayerNorm forward / input gradient /
// affine gradients and RMSNorm's gamma gradient run through TunedKernelSlots whose reference is the established
// shader, with the generated VulkanTunedRowKernels variants as the alternatives (see OpenClBackend.TunedKernels).
public sealed unsafe partial class VulkanBackend
{
    private const double TunedRowTolerance = 1e-4;

    private readonly object _tunedSlotsLock = new object();
    private readonly object _tuneMeasureLock = new object();
    private TunedKernelSlot<TunedSoftmaxArgs>? _softmaxSlot;
    private TunedKernelSlot<TunedSoftmaxBackwardArgs>? _softmaxBackwardSlot;
    private TunedKernelSlot<TunedLayerNormArgs>? _layerNormSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormBackwardSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _layerNormGradParamsSlot;
    private TunedKernelSlot<TunedNormBackwardArgs>? _rmsNormGradGammaSlot;
    private string? _tunedDeviceKey;

    /// <summary>The registry device key, e.g. <c>vulkan:AMD Radeon RX 5500 XT</c>.</summary>
    internal string TunedDeviceKey => _tunedDeviceKey ??= "vulkan:" + _device.DeviceName;

    private static TunedShape RowShape(int rows, int n) => TunedShape.Of2(TunedKernelDType.Float32, rows, n);

    // Outside batch mode every dispatch submits and waits, so host time over `repetitions` runs is the device time
    // plus submission overhead, the same for every candidate compared. A batch would only record, so it is not measured.
    internal double MeasureTunedKernelMilliseconds(Action run, int repetitions)
    {
        lock (_tuneMeasureLock)
        {
            var watch = Stopwatch.StartNew();
            for (int i = 0; i < repetitions; i++) run();
            return watch.Elapsed.TotalMilliseconds;
        }
    }

    private bool CanMeasureTunedKernels => IsAvailable && IsGlslCompilerAvailable && !_batchMode;

    private BackendTunedKernelHarness<TArgs> Harness<TArgs>(Func<TArgs, (IGpuBuffer Buffer, int Count)[]> outputs,
        Func<TArgs, bool> aliased) where TArgs : struct
        => new BackendTunedKernelHarness<TArgs>(this, TunedRowTolerance, () => CanMeasureTunedKernels, MeasureTunedKernelMilliseconds,
            outputs, aliased);

    private static int RowThreads(int rows, int lanes) => checked(rows * lanes);

    private static uint[] Push(params int[] values)
    {
        var words = new uint[values.Length];
        for (int i = 0; i < values.Length; i++) words[i] = unchecked((uint)values[i]);
        return words;
    }

    // A generated variant applies only where the GLSL compiler can build it.
    private bool TunedVariantsAvailable => IsGlslCompilerAvailable;

    // ---------------------------------------------------------------- softmax

    internal TunedKernelSlot<TunedSoftmaxArgs> SoftmaxSlot
    {
        get
        {
            if (_softmaxSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_softmaxSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<TunedSoftmaxArgs>>
                    {
                        new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("vulkan.softmax.thread_per_row", TunedKernelOrigin.Builtin, true,
                            _ => true, (in TunedSoftmaxArgs a) => SoftmaxReference(a.Input, a.Output, a.Rows, a.N)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.Softmax(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxArgs>("generated.softmax.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && TunedVariantsAvailable,
                            (in TunedSoftmaxArgs a) => GlslNaryOp(source, new[] { a.Input, a.Output }, RowThreads(a.Rows, l), Push(a.Rows, a.N))));
                    }
                    _softmaxSlot = new TunedKernelSlot<TunedSoftmaxArgs>(TunedKernelOp.Softmax, TunedDeviceKey,
                        Harness<TunedSoftmaxArgs>(a => new[] { (a.Output, a.Rows * a.N) }, a => ReferenceEquals(a.Input, a.Output)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _softmaxSlot;
            }
        }
    }

    internal TunedKernelSlot<TunedSoftmaxBackwardArgs> SoftmaxBackwardSlot
    {
        get
        {
            if (_softmaxBackwardSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_softmaxBackwardSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<TunedSoftmaxBackwardArgs>>
                    {
                        new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("vulkan.softmax_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true,
                            (in TunedSoftmaxBackwardArgs a) => SoftmaxBackwardReference(a.GradOutput, a.Output, a.GradInput, a.Rows, a.N)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.SoftmaxBackward(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedSoftmaxBackwardArgs>("generated.softmax_backward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && TunedVariantsAvailable,
                            (in TunedSoftmaxBackwardArgs a) => GlslNaryOp(source, new[] { a.GradOutput, a.Output, a.GradInput },
                                RowThreads(a.Rows, l), Push(a.Rows, a.N))));
                    }
                    _softmaxBackwardSlot = new TunedKernelSlot<TunedSoftmaxBackwardArgs>(TunedKernelOp.SoftmaxBackward, TunedDeviceKey,
                        Harness<TunedSoftmaxBackwardArgs>(a => new[] { (a.GradInput, a.Rows * a.N) },
                            a => ReferenceEquals(a.GradInput, a.GradOutput) || ReferenceEquals(a.GradInput, a.Output)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _softmaxBackwardSlot;
            }
        }
    }

    // ---------------------------------------------------------------- LayerNorm

    internal TunedKernelSlot<TunedLayerNormArgs> LayerNormSlot
    {
        get
        {
            if (_layerNormSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_layerNormSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<TunedLayerNormArgs>>
                    {
                        new DelegateTunedKernelCandidate<TunedLayerNormArgs>("vulkan.layernorm_forward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedLayerNormArgs a) => LayerNormReference(a)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.LayerNorm(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedLayerNormArgs>("generated.layernorm_forward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && TunedVariantsAvailable,
                            (in TunedLayerNormArgs a) => GlslNaryOp(source,
                                new[] { a.Input, a.Output, a.Gamma, a.Beta, a.SaveMean, a.SaveInvVar }, RowThreads(a.Rows, l),
                                new[] { unchecked((uint)a.Rows), unchecked((uint)a.N), FloatBits(a.Epsilon) })));
                    }
                    _layerNormSlot = new TunedKernelSlot<TunedLayerNormArgs>(TunedKernelOp.LayerNorm, TunedDeviceKey,
                        Harness<TunedLayerNormArgs>(
                            a => new[] { (a.Output, a.Rows * a.N), (a.SaveMean, a.Rows), (a.SaveInvVar, a.Rows) },
                            a => ReferenceEquals(a.Output, a.Input)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _layerNormSlot;
            }
        }
    }

    internal TunedKernelSlot<TunedNormBackwardArgs> LayerNormBackwardSlot
    {
        get
        {
            if (_layerNormBackwardSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_layerNormBackwardSlot is null)
                {
                    var candidates = new List<ITunedKernelCandidate<TunedNormBackwardArgs>>
                    {
                        new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("vulkan.layernorm_backward.thread_per_row",
                            TunedKernelOrigin.Builtin, true, _ => true, (in TunedNormBackwardArgs a) => LayerNormBackwardInputReference(a)),
                    };
                    foreach (int lanes in VulkanTunedRowKernels.RowLanes)
                    {
                        int l = lanes;
                        string source = VulkanTunedRowKernels.LayerNormBackward(l);
                        candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>("generated.layernorm_backward.lanes" + l,
                            TunedKernelOrigin.Generated, true, shape => shape[1] <= 64 * l && TunedVariantsAvailable,
                            (in TunedNormBackwardArgs a) => GlslNaryOp(source,
                                new[] { a.GradOutput, a.Input, Required(a.Gamma), a.Stat0, Required(a.Stat1), a.Out0 },
                                RowThreads(a.Rows, l), Push(a.Rows, a.N))));
                    }
                    _layerNormBackwardSlot = new TunedKernelSlot<TunedNormBackwardArgs>(TunedKernelOp.LayerNormBackward, TunedDeviceKey,
                        Harness<TunedNormBackwardArgs>(a => new[] { (a.Out0, a.Rows * a.N) },
                            a => ReferenceEquals(a.Out0, a.GradOutput) || ReferenceEquals(a.Out0, a.Input)),
                        () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return _layerNormBackwardSlot;
            }
        }
    }

    internal TunedKernelSlot<TunedNormBackwardArgs> LayerNormGradParametersSlot
    {
        get
        {
            if (_layerNormGradParamsSlot is { } s) return s;
            lock (_tunedSlotsLock)
                return _layerNormGradParamsSlot ??= CreateColumnGradSlot(TunedKernelOp.LayerNormGradParameters,
                    "vulkan.layernorm_grad_params.thread_per_column", rms: false);
        }
    }

    internal TunedKernelSlot<TunedNormBackwardArgs> RmsNormGradGammaSlot
    {
        get
        {
            if (_rmsNormGradGammaSlot is { } s) return s;
            lock (_tunedSlotsLock)
                return _rmsNormGradGammaSlot ??= CreateColumnGradSlot(TunedKernelOp.RmsNormGradGamma,
                    "vulkan.rmsnorm_grad_gamma.thread_per_column", rms: true);
        }
    }

    private TunedKernelSlot<TunedNormBackwardArgs> CreateColumnGradSlot(TunedKernelOp op, string referenceId, bool rms)
    {
        var candidates = new List<ITunedKernelCandidate<TunedNormBackwardArgs>>
        {
            new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(referenceId, TunedKernelOrigin.Builtin, true, _ => true,
                (in TunedNormBackwardArgs a) => ColumnGradReference(rms, a)),
        };
        foreach (int rowLanes in VulkanTunedRowKernels.ColumnRowLanes)
        {
            int r = rowLanes;
            string source = rms ? VulkanTunedRowKernels.RmsNormGradGamma(r) : VulkanTunedRowKernels.LayerNormGradParams(r);
            candidates.Add(new DelegateTunedKernelCandidate<TunedNormBackwardArgs>(
                (rms ? "generated.rmsnorm_grad_gamma.rows" : "generated.layernorm_grad_params.rows") + r,
                TunedKernelOrigin.Generated, true, _ => TunedVariantsAvailable,
                (in TunedNormBackwardArgs a) =>
                {
                    // One workgroup of 32 * R invocations per 32-column tile.
                    int threads = checked(((a.N + 31) / 32) * 32 * r);
                    var buffers = rms
                        ? new[] { a.GradOutput, a.Input, a.Stat0, a.Out0 }
                        : new[] { a.GradOutput, a.Input, a.Stat0, Required(a.Stat1), a.Out0, Required(a.Out1) };
                    GlslNaryOp(source, buffers, threads, Push(a.Rows, a.N), 32 * r);
                }));
        }
        return new TunedKernelSlot<TunedNormBackwardArgs>(op, TunedDeviceKey,
            Harness<TunedNormBackwardArgs>(
                a => rms ? new[] { (a.Out0, a.N) } : new[] { (a.Out0, a.N), (Required(a.Out1), a.N) },
                a => ReferenceEquals(a.Out0, a.GradOutput) || ReferenceEquals(a.Out0, a.Input) ||
                     (a.Out1 is { } o1 && (ReferenceEquals(o1, a.GradOutput) || ReferenceEquals(o1, a.Input)))),
            () => GpuDeterminism.IsActive, candidates.ToArray());
    }

    private static IGpuBuffer Required(IGpuBuffer? buffer)
        => buffer ?? throw new InvalidOperationException("A required tuned-kernel buffer argument is missing.");

    // ---------------------------------------------------------------- reference launches (the established shaders)

    private void SoftmaxReference(IGpuBuffer input, IGpuBuffer output, int rows, int n)
        => GlslUnaryOp(VulkanGlslKernels.SoftmaxRows, input, output, rows, new uint[] { (uint)rows, (uint)n }, 2 * sizeof(uint));

    private void SoftmaxBackwardReference(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer gradInput, int rows, int n)
        => GlslBinaryOp(VulkanGlslKernels.SoftmaxBackward, gradOutput, output, gradInput, rows,
            new uint[] { (uint)rows, (uint)n }, 2 * sizeof(uint));

    private void LayerNormReference(in TunedLayerNormArgs a)
    {
        using var runningMeanDummy = AllocateBuffer(1);
        using var runningVarianceDummy = AllocateBuffer(1);
        DispatchNormalizationForward(
            new[] { a.Input, a.Output, a.Gamma, a.Beta, runningMeanDummy, runningVarianceDummy, a.SaveMean, a.SaveInvVar }, a.Rows,
            new[] { 1u, (uint)a.Rows, (uint)a.N, 1u, 1u, FloatBits(a.Epsilon), 0u, 1u });
    }

    // NormalizationBackward (mode 1 LayerNorm, operation 0) writes only the input gradient; its gamma / beta bindings
    // are unused there and take dummies.
    private void LayerNormBackwardInputReference(in TunedNormBackwardArgs a)
    {
        using var gradGammaDummy = AllocateBuffer(1);
        using var gradBetaDummy = AllocateBuffer(1);
        GlslNaryOp(VulkanGlslKernels.NormalizationBackward,
            new[] { a.GradOutput, a.Input, Required(a.Gamma), a.Stat0, Required(a.Stat1), a.Out0, gradGammaDummy, gradBetaDummy },
            a.Rows, new[] { 1u, 0u, (uint)a.Rows, (uint)a.N, 1u });
    }

    // NormalizationBackward operation 1: the affine gradients, one invocation per column. LayerNorm (mode 1) writes
    // gradGamma and gradBeta; RMSNorm (mode 3) writes gradGamma from the saved RMS in the scale binding.
    private void ColumnGradReference(bool rms, in TunedNormBackwardArgs a)
    {
        if (rms)
        {
            using var savedMeanDummy = AllocateBuffer(Math.Max(1, a.Rows));
            using var gammaDummy = AllocateBuffer(Math.Max(1, a.N));
            using var gradInputDummy = AllocateBuffer(1);
            using var gradBetaDummy = AllocateBuffer(Math.Max(1, a.N));
            DispatchNormalizationBackward(
                new[] { a.GradOutput, a.Input, gammaDummy, savedMeanDummy, a.Stat0, gradInputDummy, a.Out0, gradBetaDummy }, a.N,
                new[] { 3u, 1u, (uint)a.Rows, (uint)a.N, 1u });
            return;
        }
        using var gammaUnused = AllocateBuffer(Math.Max(1, a.N));
        using var gradInputUnused = AllocateBuffer(1);
        DispatchNormalizationBackward(
            new[] { a.GradOutput, a.Input, gammaUnused, a.Stat0, Required(a.Stat1), gradInputUnused, a.Out0, Required(a.Out1) }, a.N,
            new[] { 1u, 1u, (uint)a.Rows, (uint)a.N, 1u });
    }
}