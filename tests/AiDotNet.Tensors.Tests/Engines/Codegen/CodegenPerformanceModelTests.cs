// Copyright (c) AiDotNet. All rights reserved.
// Validates the static bottleneck model against measured hardware.
//
// A performance model nobody checks is a decoration. These tests pin it to numbers
// measured on an idle RTX 3080 with clocks locked at 1770 MHz and true fp32 on both
// sides, so the model fails loudly when it drifts away from reality rather than
// quietly misdirecting the next optimisation.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Codegen;

public class CodegenPerformanceModelTests
{
    /// <summary>Measured microseconds, locked-clock true-fp32 bake-off.</summary>
    private static readonly Dictionary<string, double> Measured = new(StringComparer.Ordinal)
    {
        ["depthwise_conv2d_3x3_bias_relu"] = 72.9,
        ["depthwise_conv2d_3x3"] = 72.5,
        ["depthwise_conv2d_3x3_bwd_data"] = 72.9,
        ["conv2d_1x1_bias_relu"] = 29.2,
        ["conv2d_1x1_bwd_data"] = 32.9,
        ["conv2d_3x3_bias_relu"] = 62.3,
        ["conv2d_3x3_bwd_data"] = 85.0,
        ["maxpool2d_2x2"] = 157.6,
        ["conv_transpose2d_3x3_stride2"] = 100.7,
    };

    /// <summary>
    /// Kernels whose measured access pattern is well coalesced and whose occupancy is
    /// not register-limited. The model is expected to be accurate on these.
    /// </summary>
    private static readonly HashSet<string> WellBehaved = new(StringComparer.Ordinal)
    {
        "depthwise_conv2d_3x3_bias_relu",
        "depthwise_conv2d_3x3",
        "depthwise_conv2d_3x3_bwd_data",
        "maxpool2d_2x2",
    };

    private static CodegenPerformancePrediction PredictFor(CodegenCatalogEntry entry)
    {
        var emitter = new PtxAffineEmitter();
        emitter.Emit(entry.Bench, 8, 6);
        long threads = entry.Bench.Space.TotalThreads / Math.Max(1, emitter.CoarsenedLanes);
        return CodegenPerformanceModel.Predict(
            entry.Bench, threads, emitter.DynamicLoadsPerThread, CodegenMachineModel.Rtx3080Locked);
    }

    /// <summary>
    /// The limiter is the actionable output: it says WHICH lever to pull. Check it
    /// against the kernels where hardware evidence exists.
    ///
    /// dense 3x3: Nsight Compute measured l1tex at 89.99% of peak and DRAM at 2.41%.
    /// depthwise and maxpool: measured at 93-108% of the DRAM bandwidth roofline.
    /// </summary>
    [Theory]
    [InlineData("conv2d_3x3_bias_relu", CodegenLimiter.LoadIssue)]
    [InlineData("conv2d_3x3_bwd_data", CodegenLimiter.LoadIssue)]
    [InlineData("depthwise_conv2d_3x3", CodegenLimiter.DramBandwidth)]
    [InlineData("depthwise_conv2d_3x3_bias_relu", CodegenLimiter.DramBandwidth)]
    [InlineData("depthwise_conv2d_3x3_bwd_data", CodegenLimiter.DramBandwidth)]
    [InlineData("maxpool2d_2x2", CodegenLimiter.DramBandwidth)]
    public void PredictedLimiter_MatchesHardwareEvidence(string kernel, CodegenLimiter expected)
    {
        var entry = CodegenKernelCatalog.Find(kernel);
        Assert.NotNull(entry);
        Assert.Equal(expected, PredictFor(entry!).Limiter);
    }

    /// <summary>
    /// On kernels whose limiter the model actually carries, predicted runtime must be
    /// within 20% of measured. Dense 3x3 sits at 0.92x and its backward at 0.99x, which
    /// only became true once the occupancy term was added -- before it they were 0.41x
    /// and 0.40x, because reuse tiling had moved them out of the load-bound regime.
    /// </summary>
    [Fact]
    public void PredictedTime_IsAccurateOnWellBehavedKernels()
    {
        foreach (var entry in CodegenKernelCatalog.All.Where(e => WellBehaved.Contains(e.Name)))
        {
            var p = PredictFor(entry);
            double measured = Measured[entry.Name];
            double ratio = p.PredictedMicroseconds / measured;
            Assert.True(ratio > 0.80 && ratio < 1.20,
                entry.Name + ": predicted " + p.PredictedMicroseconds.ToString("F1") +
                " us vs measured " + measured.ToString("F1") + " us (" + ratio.ToString("F2") +
                "x). The model has drifted from hardware.");
        }
    }

    /// <summary>
    /// The model is OPTIMISTIC where it ignores two effects it does not yet model:
    /// sector efficiency when a warp's accesses straddle rows, and occupancy loss from
    /// register pressure. Pinning that keeps it an honest lower bound rather than a
    /// claim, and the day one of these becomes accurate the test says so.
    /// </summary>
    [Theory]
    [InlineData("conv2d_1x1_bias_relu")]        // ow=28 is not a warp multiple: warps straddle rows
    [InlineData("conv2d_1x1_bwd_data")]
    [InlineData("conv_transpose2d_3x3_stride2")] // 77 registers, occupancy limited
    // Shared-memory staging moved these two OUT of the regime the model describes: it
    // credits the per-thread global loads that staging removes, but carries no term for
    // the shared reads that replace them or for the two barriers per strip-mine step.
    // Measured, staging took dense 3x3 from 68.1 us to 62.3; the model predicts 35.6.
    [InlineData("conv2d_3x3_bias_relu")]
    [InlineData("conv2d_3x3_bwd_data")]
    // The three that remain optimistic map EXACTLY onto the two limiters the model
    // does not carry, as measured by the limiter gate: both 1x1 kernels are L2-bound
    // (66% and 54%) and conv_transpose is SM-bound (82.5%). The model has LoadIssue,
    // DRAM and Compute terms only. That correspondence is the useful part -- the gate
    // says which term to add next.
    public void ModelIsOptimisticWhereItIgnoresCoalescingAndOccupancy(string kernel)
    {
        var entry = CodegenKernelCatalog.Find(kernel);
        Assert.NotNull(entry);

        var p = PredictFor(entry!);
        double ratio = p.PredictedMicroseconds / Measured[kernel];
        Assert.True(ratio < 1.0,
            kernel + ": the model is documented as optimistic here but predicted " +
            ratio.ToString("F2") + "x. If it is now accurate, promote it to the " +
            "well-behaved set and record why.");
    }

    /// <summary>
    /// The reuse analysis must find the axis that caused the measured 64x input-load
    /// redundancy in dense convolution: the input is independent of the output-channel
    /// axis. This is the input to automatic tile selection.
    /// </summary>
    [Fact]
    public void ReuseAnalysis_FindsTheOutputChannelAxisForDenseConvolution()
    {
        var entry = CodegenKernelCatalog.Find("conv2d_3x3_bias_relu");
        Assert.NotNull(entry);

        var reuse = CodegenPerformanceModel.ReuseAxes(entry!.Bench);
        Assert.Contains("k", reuse["input"]);
        Assert.Contains("ow", reuse["weights"]);
        Assert.Contains("oh", reuse["weights"]);
    }

    /// <summary>
    /// Loads per MAC is the headline diagnostic, and 1.0 is the wall that tiling only
    /// the contiguous axis can never break: with a single tiled axis the ratio is
    /// (Tw+1)/Tw. Dense convolution measured 1.251 and lost to cuDNN by 4.5x. Reuse
    /// analysis added the output-channel axis, and it must stay below the wall.
    /// </summary>
    [Fact]
    public void ReuseTiling_BreaksTheOneLoadPerMacWall()
    {
        foreach (string kernel in new[] { "conv2d_3x3_bias_relu", "conv2d_3x3_bwd_data" })
        {
            var p = PredictFor(CodegenKernelCatalog.Find(kernel)!);
            Assert.True(p.LoadsPerMac < 0.6,
                kernel + " issues " + p.LoadsPerMac.ToString("F3") + " loads/MAC. Above " +
                "1.0 it cannot reach the compute roofline, and only a second tiled axis " +
                "gets below it -- so this regressing means tile selection stopped " +
                "finding the reuse axis.");
        }
    }

    /// <summary>
    /// Tile selection must DERIVE the output-channel axis for dense convolution rather
    /// than being told, and must decline it where it does not pay. The depthwise family
    /// is at the DRAM roofline, where fewer loads cannot help because the bytes are
    /// fixed: taking a second axis there measured 73 us -> 100 us.
    /// </summary>
    [Fact]
    public void TileSelection_TakesTheReuseAxisOnlyWhereItPays()
    {
        var dense = new PtxAffineEmitter();
        dense.Emit(CodegenKernelCatalog.Find("conv2d_3x3_bias_relu")!.Bench, 8, 6);
        Assert.Contains("k x", dense.TileDescription, StringComparison.Ordinal);

        var depthwise = new PtxAffineEmitter();
        depthwise.Emit(CodegenKernelCatalog.Find("depthwise_conv2d_3x3")!.Bench, 8, 6);
        Assert.Equal(4, depthwise.CoarsenedLanes);
    }

    /// <summary>Coarsening is a factor, not an on/off switch.</summary>
    [Fact]
    public void Coarsening_BoundsTheContiguousTileFactor()
    {
        var spec = CodegenKernelCatalog.Find("depthwise_conv2d_3x3")!.Bench;
        var two = new PtxAffineEmitter { Coarsening = 2 };
        two.Emit(spec, 8, 6);
        var four = new PtxAffineEmitter { Coarsening = 4 };
        four.Emit(spec, 8, 6);

        Assert.Contains("ow x2", two.TileDescription, StringComparison.Ordinal);
        Assert.Contains("ow x4", four.TileDescription, StringComparison.Ordinal);
        Assert.NotEqual(two.TileDescription, four.TileDescription);
    }
}
