using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class LogSoftmaxABTest
{
    private readonly ITestOutputHelper _output;
    public LogSoftmaxABTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void LogSoftmax_BreakdownByPath()
    {
        int rows = 256, cols = 256, length = rows * cols;
        var rng = new Random(42);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 6 - 3);
        var output = new float[length];
        int warmup = 5, iters = 100;

        var hIn = GCHandle.Alloc(input, GCHandleType.Pinned);
        var hOut = GCHandle.Alloc(output, GCHandleType.Pinned);

        // Path A: FusedKernels.LogSoftmaxUnsafe per row
        double fusedMs = Measure(() =>
        {
            float* pIn = (float*)hIn.AddrOfPinnedObject();
            float* pOut = (float*)hOut.AddrOfPinnedObject();
            for (int r = 0; r < rows; r++)
                FusedKernels.LogSoftmaxUnsafe(pIn + r * cols, pOut + r * cols, cols);
        }, warmup, iters);

        // Path B: Scalar log-sum-exp per row
        double scalarMs = Measure(() =>
        {
            float* pIn = (float*)hIn.AddrOfPinnedObject();
            float* pOut = (float*)hOut.AddrOfPinnedObject();
            for (int r = 0; r < rows; r++)
            {
                float* row = pIn + r * cols;
                float* oRow = pOut + r * cols;
                float maxVal = row[0];
                for (int c = 1; c < cols; c++)
                    if (row[c] > maxVal) maxVal = row[c];
                float logSumExp = 0f;
                for (int c = 0; c < cols; c++)
                    logSumExp += MathF.Exp(row[c] - maxVal);
                logSumExp = MathF.Log(logSumExp);
                for (int c = 0; c < cols; c++)
                    oRow[c] = row[c] - maxVal - logSumExp;
            }
        }, warmup, iters);

        hIn.Free();
        hOut.Free();

        _output.WriteLine($"LogSoftmax [256x256] breakdown:");
        _output.WriteLine($"  FusedKernels.LogSoftmaxUnsafe: {fusedMs:F3}ms");
        _output.WriteLine($"  Scalar log-sum-exp:            {scalarMs:F3}ms");
        _output.WriteLine($"  PyTorch BDN: 0.099ms");
        _output.WriteLine($"  Our best kernel: {Math.Min(fusedMs, scalarMs):F3}ms = {Math.Min(fusedMs, scalarMs) / 0.099:F1}x vs PyTorch");
    }

    [Fact]
    public void LogSoftmax_Compiled_CurrentPerformance()
    {
        var engine = new CpuEngine();
        var input = CreateRandom(new[] { 256, 256 }, 42);
        int warmup = 5, iters = 100;

        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.TensorLogSoftmax(input, -1);
            plan = scope.CompileInference<float>();
        }
        double compiledMs = Measure(() => plan.Execute(), warmup, iters);
        plan.Dispose();

        _output.WriteLine($"LogSoftmax compiled: {compiledMs:F3}ms (PyTorch: 0.099ms, ratio: {compiledMs / 0.099:F2}x)");
    }

    private static double Measure(Action action, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++) action();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int length = 1;
        for (int i = 0; i < shape.Length; i++) length *= shape[i];
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 6 - 3);
        return new Tensor<float>(data, shape);
    }
}
