// Copyright (c) AiDotNet. All rights reserved.
// Predict a kernel's bottleneck from its index maps, without a GPU.
//
// This exists because we optimised the wrong thing for a long time. The release gate
// was static machine-code metrics -- SASS instruction count, LDG count, registers,
// spills. Those said vectorised loads were a 24.7% improvement; wall clock moved 3.7%.
// The gate measured what was easy to measure, not what decides whether a kernel is
// competitive.
//
// Profiling eventually showed the dense convolution pinned at 89.99% of L1 request
// throughput while issuing 158x more load instructions than cuDNN for identical
// arithmetic. That was discoverable from the specification alone: the index maps say
// how many loads each output costs, and the machine says how many it can retire. This
// model does that arithmetic up front, so a kernel that cannot win is identified
// BEFORE anyone writes a lowering for it -- and it costs nothing per kernel, which is
// what matters when there are hundreds of them.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>Which hardware resource a kernel is predicted to saturate first.</summary>
public enum CodegenLimiter
{
    /// <summary>Issuing global load instructions -- the L1/LSU request path.</summary>
    LoadIssue,

    /// <summary>Moving unique bytes to and from device memory.</summary>
    DramBandwidth,

    /// <summary>Fused multiply-add throughput.</summary>
    Compute
}

/// <summary>
/// Machine constants for one device. Deliberately few, and each one is a published
/// or directly measurable figure rather than a fitted fudge factor.
/// </summary>
public sealed class CodegenMachineModel
{
    /// <summary>Creates a machine model.</summary>
    public CodegenMachineModel(
        string name, int multiprocessors, double clockHz,
        double loadInstructionsPerSmPerCycle, double fmaLanesPerSm, double dramBytesPerSecond,
        double tensorCoreMacsPerSmPerCycle = 0)
    {
        TensorCoreMacsPerSmPerCycle = tensorCoreMacsPerSmPerCycle;
        Name = name;
        Multiprocessors = multiprocessors;
        ClockHz = clockHz;
        LoadInstructionsPerSmPerCycle = loadInstructionsPerSmPerCycle;
        FmaLanesPerSm = fmaLanesPerSm;
        DramBytesPerSecond = dramBytesPerSecond;
    }

    /// <summary>Device name.</summary>
    public string Name { get; }

    /// <summary>Streaming multiprocessor count.</summary>
    public int Multiprocessors { get; }

    /// <summary>Core clock in Hz.</summary>
    public double ClockHz { get; }

    /// <summary>
    /// Warp-level global load instructions one SM can retire per cycle.
    /// </summary>
    /// <remarks>
    /// Calibrated, not guessed. The dense 3x3 kernel executed 4,017,216 warp-level
    /// global loads in 126.85 us at 1770 MHz on 68 SMs, while Nsight Compute reported
    /// l1tex throughput at 89.99% of peak. That gives
    /// (4.017e6 / 126.85e-6) / (68 * 1.77e9) / 0.8999 = 0.293 instructions per SM-cycle.
    /// </remarks>
    public double LoadInstructionsPerSmPerCycle { get; }

    /// <summary>FP32 FMA lanes per SM.</summary>
    public double FmaLanesPerSm { get; }

    /// <summary>Peak device-memory bandwidth in bytes per second.</summary>
    public double DramBytesPerSecond { get; }

    /// <summary>Warp-level load instructions per second across the whole device.</summary>
    public double LoadInstructionsPerSecond =>
        LoadInstructionsPerSmPerCycle * Multiprocessors * ClockHz;

    /// <summary>Fused multiply-adds per second across the whole device.</summary>
    public double MacsPerSecond => FmaLanesPerSm * Multiprocessors * ClockHz;

    /// <summary>
    /// Tensor-core multiply-accumulates per SM per cycle, or zero when the model has none.
    /// </summary>
    /// <remarks>
    /// WITHOUT THIS THE MODEL CANNOT SCORE A TENSOR-CORE KERNEL AT ALL. FmaLanesPerSm counts
    /// the fp32 pipe: 128 lanes at 68 SMs and 1.77 GHz is 30.8 TFLOP/s, and a measured
    /// tensor-core matmul on the same device reaches 61.3. Scored against the fp32 rate such a
    /// kernel reads as 180%+ of "peak", which is not a headroom figure -- it is a category
    /// error, and it silently reports finished work on kernels with half their performance
    /// still on the table.
    /// </remarks>
    public double TensorCoreMacsPerSmPerCycle { get; }

    /// <summary>Tensor-core multiply-accumulates per second.</summary>
    public double TensorCoreMacsPerSecond =>
        TensorCoreMacsPerSmPerCycle * Multiprocessors * ClockHz;

    /// <summary>True when this model can bound a tensor-core kernel.</summary>
    public bool HasTensorCores => TensorCoreMacsPerSmPerCycle > 0;

    /// <summary>
    /// The arithmetic rate for a kernel, picking the tensor-core pipe when the kernel uses it.
    /// </summary>
    public double MacsPerSecondFor(bool usesTensorCores) =>
        usesTensorCores && HasTensorCores ? TensorCoreMacsPerSecond : MacsPerSecond;

    /// <summary>
    /// Coefficient of the occupancy penalty <c>1 + c / blocksPerSm</c>.
    /// </summary>
    /// <remarks>
    /// THE ONE FITTED CONSTANT IN THIS MODEL, and therefore its weakest part. Everything
    /// else is a published figure or derived from a counter; this is fit to three
    /// measured points, and it is recorded as fitted so nobody mistakes it for physics:
    ///
    ///   blocks/SM  measured penalty   1 + 0.5/perSm
    ///     0.72          1.71              1.69
    ///     1.44          1.46              1.35
    ///    92.2           1.17              1.005
    ///
    /// The first shape tried -- wave quantisation plus a latency-hiding floor -- was
    /// derived from first principles and predicted a 2.78x penalty where 1.46x was
    /// measured, turning dense 3x3 from 0.41x optimistic into 1.90x pessimistic. The
    /// derivation assumed a block that does not fill an SM wastes the rest of it, which
    /// is wrong: at 1.44 blocks per SM there are still ~11 resident warps, enough to
    /// hide a good deal of latency.
    /// </remarks>
    public double OccupancyPenaltyCoefficient { get; init; } = 0.5;

    /// <summary>RTX 3080 (GA102, 68 SMs) with clocks locked at 1770 MHz.</summary>
    public static CodegenMachineModel Rtx3080Locked { get; } = new(
        name: "RTX 3080 (GA102, clocks locked 1770 MHz)",
        multiprocessors: 68,
        clockHz: 1.77e9,
        loadInstructionsPerSmPerCycle: 0.293,
        fmaLanesPerSm: 128,
        dramBytesPerSecond: 760e9,

        // 256 fp16 MACs per SM per cycle with an fp32 accumulator: 256 * 68 * 1.77e9 = 30.8
        // TMAC/s = 61.6 TFLOP/s. CONFIRMED by measurement rather than taken from the
        // datasheet -- the mma ceiling probe, which runs the same instruction mix with the
        // fragment loads hoisted out of the K loop, reaches 61.3 TFLOP/s, or 99.5% of it.
        tensorCoreMacsPerSmPerCycle: 256);
}

/// <summary>What the model predicts for one kernel.</summary>
public sealed class CodegenPerformancePrediction
{
    internal CodegenPerformancePrediction(
        string kernel, long outputs, long macs, long uniqueBytes,
        long warpLoadInstructions, double loadsPerMac,
        double loadIssueMicroseconds, double dramMicroseconds, double computeMicroseconds)
    {
        Kernel = kernel;
        Outputs = outputs;
        Macs = macs;
        UniqueBytes = uniqueBytes;
        WarpLoadInstructions = warpLoadInstructions;
        LoadsPerMac = loadsPerMac;
        LoadIssueMicroseconds = loadIssueMicroseconds;
        DramMicroseconds = dramMicroseconds;
        ComputeMicroseconds = computeMicroseconds;
    }

    /// <summary>Kernel name.</summary>
    public string Kernel { get; }

    /// <summary>Output elements produced.</summary>
    public long Outputs { get; }

    /// <summary>Fused multiply-adds required by the specification.</summary>
    public long Macs { get; }

    /// <summary>Distinct bytes the kernel touches, counting each tensor once.</summary>
    public long UniqueBytes { get; }

    /// <summary>Warp-level global load instructions the lowering will execute.</summary>
    public long WarpLoadInstructions { get; }

    /// <summary>
    /// Thread-level loads per MAC. The headline number: a tiled implementation needs
    /// roughly 0.03, and anything near or above 1.0 cannot be compute-bound.
    /// </summary>
    public double LoadsPerMac { get; }

    /// <summary>Microseconds if load issue is the only constraint.</summary>
    public double LoadIssueMicroseconds { get; }

    /// <summary>Microseconds if device-memory bandwidth is the only constraint.</summary>
    public double DramMicroseconds { get; }

    /// <summary>Microseconds if FMA throughput is the only constraint.</summary>
    public double ComputeMicroseconds { get; }

    /// <summary>The resource predicted to bind first.</summary>
    public CodegenLimiter Limiter =>
        LoadIssueMicroseconds >= DramMicroseconds && LoadIssueMicroseconds >= ComputeMicroseconds
            ? CodegenLimiter.LoadIssue
            : DramMicroseconds >= ComputeMicroseconds
                ? CodegenLimiter.DramBandwidth
                : CodegenLimiter.Compute;

    /// <summary>
    /// Multiplier applied for imperfect occupancy: wave quantisation plus the loss of
    /// latency hiding when too few blocks are resident. 1.0 means occupancy is not a
    /// factor.
    /// </summary>
    public double OccupancyPenalty { get; internal set; } = 1.0;

    /// <summary>Blocks the launch will use.</summary>
    public long Blocks { get; internal set; }

    /// <summary>Predicted runtime: the slowest constraint, scaled by occupancy.</summary>
    public double PredictedMicroseconds =>
        Math.Max(LoadIssueMicroseconds, Math.Max(DramMicroseconds, ComputeMicroseconds))
        * OccupancyPenalty;

    /// <summary>
    /// How much faster the kernel could run if load issue stopped binding. 1.0 means
    /// load issue is not the constraint and cutting loads buys nothing.
    /// </summary>
    public double HeadroomIfLoadsWereFree =>
        PredictedMicroseconds / Math.Max(1e-9, Math.Max(DramMicroseconds, ComputeMicroseconds));
}

/// <summary>Predicts a kernel's bottleneck from its specification.</summary>
public static class CodegenPerformanceModel
{
    private const int WarpSize = 32;
    private const int BytesPerElement = 4;

    /// <summary>
    /// Predicts the bottleneck for <paramref name="spec"/> under a lowering that
    /// executes <paramref name="dynamicLoadsPerThread"/> global loads per thread across
    /// <paramref name="threads"/> threads.
    /// </summary>
    /// <remarks>
    /// The load count comes from the emitter rather than being re-derived here.
    /// Re-deriving it would duplicate every lowering decision (coarsening, operand
    /// sharing, vectorisation, strip-mining) and the copy would drift out of step with
    /// the emitter -- which is precisely the class of bug the index-map IR exists to
    /// prevent. Emission is pure string building, so asking the emitter costs nothing
    /// and needs no device.
    /// </remarks>
    public static CodegenPerformancePrediction Predict(
        CodegenKernelSpec spec, long threads, long dynamicLoadsPerThread,
        CodegenMachineModel? machine = null, int blockThreads = 256)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        if (threads <= 0) throw new ArgumentOutOfRangeException(nameof(threads));
        machine ??= CodegenMachineModel.Rtx3080Locked;

        long outputs = spec.Output.ElementCount;
        long macs = outputs * Math.Max(1, spec.Space.ReductionTripCount);

        long uniqueBytes = spec.Output.ElementCount * BytesPerElement;
        for (int i = 0; i < spec.Inputs.Count; i++)
            uniqueBytes += spec.Inputs[i].ElementCount * BytesPerElement;

        // A warp issues one instruction for its 32 threads, so instruction count is
        // thread-load-count scaled by threads and divided by the warp width.
        long threadLoads = dynamicLoadsPerThread * threads;
        long warpLoads = (threadLoads + WarpSize - 1) / WarpSize;

        double loadsPerMac = macs > 0 ? (double)threadLoads / macs : 0.0;

        double loadIssueUs = warpLoads / machine.LoadInstructionsPerSecond * 1e6;
        double dramUs = uniqueBytes / machine.DramBytesPerSecond * 1e6;
        double computeUs = macs / machine.MacsPerSecond * 1e6;

        var prediction = new CodegenPerformancePrediction(
            spec.Name, outputs, macs, uniqueBytes, warpLoads, loadsPerMac,
            loadIssueUs, dramUs, computeUs);

        long blocks = (threads + blockThreads - 1) / blockThreads;
        prediction.Blocks = blocks;
        prediction.OccupancyPenalty = OccupancyPenaltyFor(blocks, machine);
        return prediction;
    }

    /// <summary>
    /// How much slower imperfect occupancy makes a kernel, as a multiplier.
    /// </summary>
    /// <remarks>
    /// Two effects, and the larger one governs.
    ///
    /// WAVE QUANTISATION. Blocks are dispatched in waves across the SMs. 98 blocks on 68
    /// SMs is 1.44 waves, but the hardware runs 2, so 28% of the second wave's capacity
    /// is idle and the kernel takes 2/1.44 = 1.39x what perfect packing would cost. This
    /// shrinks as the wave count grows and is negligible past a few waves.
    ///
    /// LATENCY HIDING. Below roughly four resident blocks per SM there are not enough
    /// warps to cover memory latency, and the kernel stalls regardless of how few loads
    /// it issues.
    ///
    /// This is the term whose absence made the model 0.68x optimistic on dense 3x3 after
    /// reuse tiling cut its loads: the tiling worked, L1 pressure fell from 89.99% to
    /// 53.4%, and the kernel stopped being load-bound -- so the model kept predicting a
    /// load-bound time for a kernel that no longer was one.
    /// </remarks>
    private static double OccupancyPenaltyFor(long blocks, CodegenMachineModel machine)
    {
        if (blocks <= 0) return 1.0;

        double perSm = blocks / (double)machine.Multiprocessors;
        return 1.0 + machine.OccupancyPenaltyCoefficient / Math.Max(perSm, 0.05);
    }

    /// <summary>
    /// For each operand, the iteration axes it does NOT depend on. Those are exactly
    /// the axes along which a load can be reused, and therefore the axes worth tiling.
    /// </summary>
    /// <remarks>
    /// Stated as data rather than prose because it is the actionable half of the
    /// diagnosis. The dense convolution's input is independent of the output-channel
    /// axis, which is where its measured 64x load redundancy comes from.
    /// </remarks>
    public static IReadOnlyDictionary<string, IReadOnlyList<string>> ReuseAxes(CodegenKernelSpec spec)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));

        var axes = spec.Space.Axes;
        var result = new Dictionary<string, IReadOnlyList<string>>(StringComparer.Ordinal);

        foreach (int inputIndex in spec.ProductInputs)
        {
            var binding = spec.Inputs[inputIndex];
            var used = new HashSet<int>();
            for (int d = 0; d < binding.Map.Count; d++)
                foreach (var term in binding.Map[d].Terms)
                    used.Add(term.Axis);

            var free = new List<string>();
            for (int a = 0; a < axes.Count; a++)
                if (!used.Contains(a) && !axes[a].IsReduction)
                    free.Add(axes[a].Name);

            result[binding.Name] = free;
        }
        return result;
    }

    /// <summary>Human-readable one-kernel report.</summary>
    public static string Describe(CodegenPerformancePrediction p, CodegenKernelSpec spec)
    {
        if (p is null) throw new ArgumentNullException(nameof(p));

        var sb = new StringBuilder();
        var ic = CultureInfo.InvariantCulture;
        sb.Append(p.Kernel).Append('\n');
        sb.Append("  outputs ").Append(p.Outputs.ToString("N0", ic))
          .Append("  MACs ").Append(p.Macs.ToString("N0", ic))
          .Append("  unique ").Append((p.UniqueBytes / 1024.0 / 1024.0).ToString("F2", ic)).Append(" MiB\n");
        sb.Append("  warp loads ").Append(p.WarpLoadInstructions.ToString("N0", ic))
          .Append("   loads/MAC ").Append(p.LoadsPerMac.ToString("F3", ic)).Append('\n');
        sb.Append("  load-issue ").Append(p.LoadIssueMicroseconds.ToString("F1", ic))
          .Append(" us | dram ").Append(p.DramMicroseconds.ToString("F1", ic))
          .Append(" us | compute ").Append(p.ComputeMicroseconds.ToString("F1", ic)).Append(" us\n");
        sb.Append("  PREDICTED ").Append(p.Limiter.ToString().ToUpperInvariant())
          .Append(" at ").Append(p.PredictedMicroseconds.ToString("F1", ic))
          .Append(" us; headroom if loads were free ")
          .Append(p.HeadroomIfLoadsWereFree.ToString("F2", ic)).Append("x\n");

        if (spec != null)
        {
            foreach (var pair in ReuseAxes(spec))
                sb.Append("  reuse ").Append(pair.Key).Append(" is invariant in {")
                  .Append(string.Join(", ", pair.Value)).Append("}\n");
        }
        return sb.ToString();
    }
}
