using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class SigmoidABTest
{
    private readonly ITestOutputHelper _output;
    public SigmoidABTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Sigmoid_1M_PathBreakdown()
    {
        int length = 1_000_000;
        var rng = new Random(42);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 10 - 5);
        var output = new float[length];
        int warmup = 3, iters = 20;

        var hIn = GCHandle.Alloc(input, GCHandleType.Pinned);
        var hOut = GCHandle.Alloc(output, GCHandleType.Pinned);

        // Path A: SimdKernels.SigmoidUnsafe
        double simdMs = Measure(() =>
        {
            SimdKernels.SigmoidUnsafe(
                (float*)hIn.AddrOfPinnedObject(),
                (float*)hOut.AddrOfPinnedObject(), length);
        }, warmup, iters);

        // Path B: Raw multiply baseline (memory throughput)
        double rawMs = Measure(() =>
        {
            SimdKernels.VectorMultiplyUnsafe(
                (float*)hIn.AddrOfPinnedObject(),
                (float*)hIn.AddrOfPinnedObject(),
                (float*)hOut.AddrOfPinnedObject(), length);
        }, warmup, iters);

        // Path C: Compiled
        var engine = new CpuEngine();
        var tensorIn = new Tensor<float>(input, new[] { length });
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.Sigmoid(tensorIn);
            plan = scope.CompileInference<float>();
        }
        double compiledMs = Measure(() => plan.Execute(), warmup, iters);
        plan.Dispose();

        hIn.Free();
        hOut.Free();

        _output.WriteLine($"Sigmoid 1M elements:");
        _output.WriteLine($"  SIMD kernel:    {simdMs:F3}ms");
        _output.WriteLine($"  Raw multiply:   {rawMs:F3}ms (memory throughput baseline)");
        _output.WriteLine($"  Compiled plan:  {compiledMs:F3}ms");
        _output.WriteLine($"  VML available:  {AiDotNet.Tensors.Helpers.VmlProvider.IsInitialized}");
        _output.WriteLine($"");
        _output.WriteLine($"  Sigmoid overhead vs raw: {simdMs / rawMs:F1}x (exp cost)");
        _output.WriteLine($"  PyTorch BDN: 0.488ms");
        _output.WriteLine($"  Our compiled: {compiledMs:F3}ms = {compiledMs / 0.488:F2}x vs PyTorch");
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
