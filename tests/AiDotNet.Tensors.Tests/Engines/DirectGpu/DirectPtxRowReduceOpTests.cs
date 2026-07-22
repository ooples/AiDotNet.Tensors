using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the operator-parameterized row-reduction family
/// (issue #843): mean, max, min, and sum-of-squares. The emitter and
/// shape-domain assertions run without a GPU; driver correctness is skipped
/// unless an admitted SM86 device is present. Every specialization stays
/// disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
public class DirectPtxRowReduceOpTests
{
    [Theory]
    [InlineData((int)DirectPtxRowReduceOp.Mean, "aidotnet_fused_row_mean_f32", "op=mean")]
    [InlineData((int)DirectPtxRowReduceOp.Max, "aidotnet_fused_row_max_f32", "op=max")]
    [InlineData((int)DirectPtxRowReduceOp.Min, "aidotnet_fused_row_min_f32", "op=min")]
    [InlineData((int)DirectPtxRowReduceOp.SumOfSquares, "aidotnet_fused_row_sumsq_f32", "op=sumsq")]
    public void Emitter_GivesEachOperatorItsOwnModuleAndIsPointerOnly(
        int opOrdinal, string entryPoint, string tag)
    {
        var op = (DirectPtxRowReduceOp)opOrdinal;
        string ptx = PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, op, 1024, 128);
        Assert.Equal(entryPoint, PtxFusedRowReduceOpF32Kernel.EntryPointFor(op));
        Assert.Contains($".visible .entry {entryPoint}(", ptx);
        Assert.Contains(tag, ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        // 128 columns / 32 lanes = one v4 load per lane.
        Assert.Equal(1, Count(ptx, "ld.global.ca.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32"));
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("div.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void Emitter_UsesTheCorrectIdentityAndCombinePerOperator()
    {
        string mean = PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, DirectPtxRowReduceOp.Mean, 1024, 128);
        string max = PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, DirectPtxRowReduceOp.Max, 1024, 128);
        string min = PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, DirectPtxRowReduceOp.Min, 1024, 128);
        string sumsq = PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, DirectPtxRowReduceOp.SumOfSquares, 1024, 128);

        // Identities: zero for the additive folds, the infinities for max/min so a
        // row of all-infinities still reduces exactly.
        Assert.Contains("mov.f32 %f4, 0f00000000;", mean);
        Assert.Contains("mov.f32 %f4, 0fFF800000;", max);   // -inf
        Assert.Contains("mov.f32 %f4, 0f7F800000;", min);   // +inf
        Assert.Contains("mov.f32 %f4, 0f00000000;", sumsq);

        // Lane fold: one combine per loaded value (%f0..%f3). Match the operand
        // index explicitly - a bare "%f" prefix would also catch the warp-fold
        // combines, which read the shuffle scratch register %f5.
        for (int i = 0; i < 4; i++)
        {
            Assert.Contains($"add.rn.f32 %f4, %f4, %f{i};", mean);
            Assert.Contains($"max.f32 %f4, %f4, %f{i};", max);
            Assert.Contains($"min.f32 %f4, %f4, %f{i};", min);
            // Sum-of-squares squares and accumulates in one rounding, not two.
            Assert.Contains($"fma.rn.f32 %f4, %f{i}, %f{i}, %f4;", sumsq);
        }
        Assert.Equal(4, Count(sumsq, "fma.rn.f32 %f4,"));
        Assert.DoesNotContain("mul.rn.f32 %f4, %f0, %f0", sumsq);

        // Warp fold: five butterfly steps, each combining the shuffle scratch.
        Assert.Equal(5, Count(mean, "add.rn.f32 %f4, %f4, %f5;"));
        Assert.Equal(5, Count(max, "max.f32 %f4, %f4, %f5;"));
        Assert.Equal(5, Count(min, "min.f32 %f4, %f4, %f5;"));
        Assert.Equal(5, Count(sumsq, "add.rn.f32 %f4, %f4, %f5;"));
    }

    [Fact]
    public void Emitter_BakesTheMeanReciprocalSoThereIsNoDivide()
    {
        // 1/128 is exactly representable, so the literal is exact.
        string ptx = PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, DirectPtxRowReduceOp.Mean, 1024, 128);
        uint bits = MathCompat.SingleToUInt32Bits(1f / 128f);
        Assert.Contains($"mul.rn.f32 %f4, %f4, 0f{bits:X8};", ptx);
        // Only mean finalizes; the others must not scale at all.
        foreach (var op in new[] { DirectPtxRowReduceOp.Max, DirectPtxRowReduceOp.Min, DirectPtxRowReduceOp.SumOfSquares })
            Assert.DoesNotContain("mul.rn.f32 %f4, %f4, 0f",
                PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, op, 1024, 128));
    }

    [Fact]
    public void Emitter_ShufflesThroughB32RegistersNotFloatRegisters()
    {
        foreach (DirectPtxRowReduceOp op in
                 (DirectPtxRowReduceOp[])Enum.GetValues(typeof(DirectPtxRowReduceOp)))
        {
            string ptx = PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, op, 1024, 128);
            // shfl is a bit-manipulation instruction: its operands must be .b32.
            Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32 %r6, %r5,"));
            Assert.DoesNotContain("shfl.sync.bfly.b32 %f", ptx);
        }
    }

    [Fact]
    public void ShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedRowReduceOpF32Kernel.IsSupportedShape(1024, 128));
        Assert.True(PtxFusedRowReduceOpF32Kernel.IsSupportedShape(256, 1024));
        Assert.False(PtxFusedRowReduceOpF32Kernel.IsSupportedShape(1000, 128));
        Assert.False(PtxFusedRowReduceOpF32Kernel.IsSupportedShape(1024, 96));
        Assert.False(PtxFusedRowReduceOpF32Kernel.IsPromotedShape(1024, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, DirectPtxRowReduceOp.Mean, 1000, 128));
        // 2048 columns would need a v8 load, which does not exist.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedRowReduceOpF32Kernel.EmitPtx(8, 6, DirectPtxRowReduceOp.Mean, 1024, 2048));
    }

    [SkippableTheory]
    [InlineData((int)DirectPtxRowReduceOp.Mean)]
    [InlineData((int)DirectPtxRowReduceOp.Max)]
    [InlineData((int)DirectPtxRowReduceOp.Min)]
    [InlineData((int)DirectPtxRowReduceOp.SumOfSquares)]
    public void DriverOnly_MatchesTheDoubleOracleAndHasZeroLocalBytes(int opOrdinal)
    {
        var op = (DirectPtxRowReduceOp)opOrdinal;
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Requires an NVIDIA CUDA driver and GPU.");
        using var runtime = new DirectPtxRuntime();
        Skip.IfNot(DirectPtxArchitecture.HasValidatedRowReduction(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor),
            "The checked-in row-reduction specializations are admitted only on SM86.");
        const int rows = 256, columns = 128;
        using var kernel = new PtxFusedRowReduceOpF32Kernel(runtime, op, rows, columns);
        Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
        Assert.Equal(0, kernel.Audit.Function.StaticSharedBytes);

        var random = RandomHelper.CreateSeededRandom(20260722 + (int)op);
        float[] input = Enumerable.Range(0, rows * columns)
            .Select(_ => (float)((random.NextDouble() * 2.0 - 1.0) * 8.0)).ToArray();

        using var inputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[0].RequiredBytes);
        using var outputBuffer = runtime.AllocateBytes(kernel.Blueprint.Tensors[1].RequiredBytes);
        inputBuffer.Upload<float>(input);
        kernel.Launch(
            DirectPtxTensorView.CreateOwned(inputBuffer, kernel.Blueprint.Tensors[0]),
            DirectPtxTensorView.CreateOwned(outputBuffer, kernel.Blueprint.Tensors[1]));
        runtime.Synchronize();

        var actual = new float[rows];
        outputBuffer.Download<float>(actual);
        for (int r = 0; r < rows; r++)
        {
            double expected = CpuOracle(op, input, r, columns);
            // Max and Min only select an element, so they are bit-exact; the
            // additive folds reassociate across lanes and need a tolerance.
            if (op is DirectPtxRowReduceOp.Max or DirectPtxRowReduceOp.Min)
                Assert.Equal((float)expected, actual[r]);
            else
                Assert.True(Math.Abs(actual[r] - expected) <= 1e-4 * Math.Max(1.0, Math.Abs(expected)),
                    $"row {r}: actual {actual[r]:G9}, expected {expected:G9}");
        }
    }

    private static double CpuOracle(DirectPtxRowReduceOp op, float[] input, int row, int columns)
    {
        int offset = row * columns;
        switch (op)
        {
            case DirectPtxRowReduceOp.Mean:
            {
                double sum = 0;
                for (int c = 0; c < columns; c++) sum += input[offset + c];
                return sum / columns;
            }
            case DirectPtxRowReduceOp.SumOfSquares:
            {
                double sum = 0;
                for (int c = 0; c < columns; c++) sum += (double)input[offset + c] * input[offset + c];
                return sum;
            }
            case DirectPtxRowReduceOp.Max:
            {
                double best = double.NegativeInfinity;
                for (int c = 0; c < columns; c++) best = Math.Max(best, input[offset + c]);
                return best;
            }
            case DirectPtxRowReduceOp.Min:
            {
                double best = double.PositiveInfinity;
                for (int c = 0; c < columns; c++) best = Math.Min(best, input[offset + c]);
                return best;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
