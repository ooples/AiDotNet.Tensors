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
        ["depthwise_conv2d_3x3_bias_relu"] = 73.0,
        ["depthwise_conv2d_3x3"] = 72.6,
        ["depthwise_conv2d_3x3_bwd_data"] = 72.3,
        ["conv2d_1x1_bias_relu"] = 45.8,
        ["conv2d_1x1_bwd_data"] = 47.5,
        ["conv2d_3x3_bias_relu"] = 125.9,
        ["conv2d_3x3_bwd_data"] = 127.9,
        ["maxpool2d_2x2"] = 156.2,
        ["conv_transpose2d_3x3_stride2"] = 99.2,
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
        "conv2d_3x3_bias_relu",
        "conv2d_3x3_bwd_data",
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
    /// On well-coalesced, non-register-limited kernels the predicted runtime must be
    /// within 20% of measured. The two dense 3x3 kernels -- the ones the model exists
    /// to diagnose -- land at 1.02x and 1.00x.
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
    [InlineData("conv_transpose2d_3x3_stride2")] // 78 registers, occupancy limited
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
    /// Loads per MAC is the headline diagnostic. A kernel at or above 1.0 cannot be
    /// compute-bound, and every dense-convolution loss sits there.
    /// </summary>
    [Fact]
    public void LoadsPerMac_FlagsTheKernelsThatCannotBeComputeBound()
    {
        var dense = PredictFor(CodegenKernelCatalog.Find("conv2d_3x3_bias_relu")!);
        Assert.True(dense.LoadsPerMac > 1.0,
            "dense 3x3 issues " + dense.LoadsPerMac.ToString("F3") + " loads/MAC; " +
            "above 1.0 it cannot reach the compute roofline.");

        // And the model must agree that removing the load constraint is worth a lot.
        Assert.True(dense.HeadroomIfLoadsWereFree > 3.0,
            "dense 3x3 headroom if loads were free is only " +
            dense.HeadroomIfLoadsWereFree.ToString("F2") + "x.");
    }
}
