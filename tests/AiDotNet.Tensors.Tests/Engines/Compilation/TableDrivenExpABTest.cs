using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B test: Table-driven exp vs current Estrin polynomial exp vs MathF.Exp.
/// Measures both performance AND accuracy.
/// </summary>
public class TableDrivenExpABTest
{
    private readonly ITestOutputHelper _output;
    public TableDrivenExpABTest(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(100_000)]
    [InlineData(1_000_000)]
    public unsafe void Exp_TableDriven_vs_Estrin_vs_Scalar(int length)
    {
        var rng = new Random(42);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 20 - 10);

        var outputTable = new float[length];
        var outputEstrin = new float[length];
        var outputScalar = new float[length];
        int warmup = 5, iters = 30;

        var hIn = GCHandle.Alloc(input, GCHandleType.Pinned);
        var hOutTable = GCHandle.Alloc(outputTable, GCHandleType.Pinned);
        var hOutEstrin = GCHandle.Alloc(outputEstrin, GCHandleType.Pinned);

        float* pIn = (float*)hIn.AddrOfPinnedObject();
        float* pOutTable = (float*)hOutTable.AddrOfPinnedObject();
        float* pOutEstrin = (float*)hOutEstrin.AddrOfPinnedObject();

        // Path A: Table-driven exp (NEW)
        double tableMs = Measure(() =>
        {
            TableDrivenExp.ExpArray(pIn, pOutTable, length);
        }, warmup, iters);

        // Path B: Current Estrin polynomial exp
        double estrinMs = Measure(() =>
        {
            SimdKernels.ExpUnsafe(pIn, pOutEstrin, length);
        }, warmup, iters);

        // Path C: Scalar MathF.Exp (reference for accuracy)
        double scalarMs = Measure(() =>
        {
            for (int i = 0; i < length; i++)
                outputScalar[i] = MathF.Exp(input[i]);
        }, warmup, iters);

        hIn.Free();
        hOutTable.Free();
        hOutEstrin.Free();

        // Accuracy check: compare table-driven vs scalar (ground truth)
        float maxErrTable = 0f, maxErrEstrin = 0f;
        double sumErrTable = 0, sumErrEstrin = 0;
        for (int i = 0; i < length; i++)
        {
            float exact = outputScalar[i];
            if (exact == 0f || float.IsInfinity(exact)) continue;

            float errTable = MathF.Abs(outputTable[i] - exact) / exact;
            float errEstrin = MathF.Abs(outputEstrin[i] - exact) / exact;
            maxErrTable = MathF.Max(maxErrTable, errTable);
            maxErrEstrin = MathF.Max(maxErrEstrin, errEstrin);
            sumErrTable += errTable;
            sumErrEstrin += errEstrin;
        }
        double avgErrTable = sumErrTable / length;
        double avgErrEstrin = sumErrEstrin / length;

        _output.WriteLine($"exp({length:N0} elements):");
        _output.WriteLine($"  Table-driven: {tableMs:F3}ms   (max err: {maxErrTable:E2}, avg err: {avgErrTable:E2})");
        _output.WriteLine($"  Estrin poly:  {estrinMs:F3}ms   (max err: {maxErrEstrin:E2}, avg err: {avgErrEstrin:E2})");
        _output.WriteLine($"  Scalar:       {scalarMs:F3}ms   (reference)");
        _output.WriteLine($"");
        _output.WriteLine($"  Table vs Estrin speedup: {estrinMs / tableMs:F2}x");
        _output.WriteLine($"  Table vs Scalar speedup: {scalarMs / tableMs:F2}x");
        _output.WriteLine($"  PyTorch exp BDN: ~{(length == 1_000_000 ? "0.654" : "0.069")}ms");

        // Accuracy threshold: must be within float32 precision (~1.2e-7)
        Assert.True(maxErrTable < 2e-6f,
            $"Table-driven exp max relative error {maxErrTable:E3} exceeds 2e-6 threshold");
    }

    private static double Measure(Action action, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++) action();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }
}
