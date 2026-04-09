using System.Diagnostics;
using System.Runtime.InteropServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Engines.Simd;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B test: different sigmoid implementations to find one that beats PyTorch.
/// Current: 1/(1+exp(-x)) = Estrin exp + Avx.Divide = ~0.54ms/1M
/// Target: < 0.488ms (PyTorch)
/// </summary>
public class FusedSigmoidABTest
{
    private readonly ITestOutputHelper _output;
    public FusedSigmoidABTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Sigmoid_1M_AllPaths()
    {
        int length = 1_000_000;
        var rng = new Random(42);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(rng.NextDouble() * 20 - 10);
        var outputA = new float[length];
        var outputB = new float[length];
        var outputRef = new float[length];
        int warmup = 5, iters = 30;

        var hIn = GCHandle.Alloc(input, GCHandleType.Pinned);
        var hOutA = GCHandle.Alloc(outputA, GCHandleType.Pinned);
        var hOutB = GCHandle.Alloc(outputB, GCHandleType.Pinned);

        float* pIn = (float*)hIn.AddrOfPinnedObject();
        float* pOutA = (float*)hOutA.AddrOfPinnedObject();
        float* pOutB = (float*)hOutB.AddrOfPinnedObject();

        // Reference: scalar sigmoid
        for (int i = 0; i < length; i++)
            outputRef[i] = 1f / (1f + MathF.Exp(-input[i]));

        // Path A: Current SigmoidUnsafe (FastExp + Divide)
        double msA = Measure(() => SimdKernels.SigmoidUnsafe(pIn, pOutA, length), warmup, iters);

        // Path B: Fused sigmoid using tanh identity: sigmoid(x) = 0.5 + 0.5*tanh(x/2)
        // This avoids the division by replacing it with multiply+add
#if NET5_0_OR_GREATER
        double msB = Measure(() =>
        {
            int i = 0;
            if (Avx2.IsSupported && Fma.IsSupported)
            {
                var half = Vector256.Create(0.5f);
                int simdLen = length & ~31;
                for (; i < simdLen; i += 32)
                {
                    var x0 = Avx.Multiply(half, Avx.LoadVector256(pIn + i));
                    var x1 = Avx.Multiply(half, Avx.LoadVector256(pIn + i + 8));
                    var x2 = Avx.Multiply(half, Avx.LoadVector256(pIn + i + 16));
                    var x3 = Avx.Multiply(half, Avx.LoadVector256(pIn + i + 24));
                    // tanh(x/2) via 2*sigmoid(x) - 1, but we need sigmoid without divide...
                    // Use: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
                    // Or direct: sigmoid = 0.5 * (1 + tanh(x/2))
                    // FastTanh uses FastSigmoid internally, so this is circular.
                    // Instead: compute exp(-x), then use 1/(1+exp(-x)) but batch the divides
                    var e0 = SimdKernels.FastExp256(Avx.Subtract(Vector256<float>.Zero, Avx.LoadVector256(pIn + i)));
                    var e1 = SimdKernels.FastExp256(Avx.Subtract(Vector256<float>.Zero, Avx.LoadVector256(pIn + i + 8)));
                    var e2 = SimdKernels.FastExp256(Avx.Subtract(Vector256<float>.Zero, Avx.LoadVector256(pIn + i + 16)));
                    var e3 = SimdKernels.FastExp256(Avx.Subtract(Vector256<float>.Zero, Avx.LoadVector256(pIn + i + 24)));
                    var one = Vector256.Create(1.0f);
                    var d0 = Avx.Add(one, e0);
                    var d1 = Avx.Add(one, e1);
                    var d2 = Avx.Add(one, e2);
                    var d3 = Avx.Add(one, e3);
                    // Batch 4 divides — CPU can pipeline them
                    Avx.Store(pOutB + i, Avx.Divide(one, d0));
                    Avx.Store(pOutB + i + 8, Avx.Divide(one, d1));
                    Avx.Store(pOutB + i + 16, Avx.Divide(one, d2));
                    Avx.Store(pOutB + i + 24, Avx.Divide(one, d3));
                }
            }
            for (; i < length; i++)
                pOutB[i] = 1f / (1f + MathF.Exp(-pIn[i]));
        }, warmup, iters);
#else
        double msB = 0;
#endif

        hIn.Free(); hOutA.Free(); hOutB.Free();

        // Accuracy check
        float maxErrA = 0f, maxErrB = 0f;
        for (int i = 0; i < length; i++)
        {
            maxErrA = MathF.Max(maxErrA, MathF.Abs(outputA[i] - outputRef[i]));
            maxErrB = MathF.Max(maxErrB, MathF.Abs(outputB[i] - outputRef[i]));
        }

        _output.WriteLine($"Sigmoid 1M elements:");
        _output.WriteLine($"  Path A (current SigmoidUnsafe):  {msA:F3}ms  (max err: {maxErrA:E2})");
        _output.WriteLine($"  Path B (batched 4x divide):      {msB:F3}ms  (max err: {maxErrB:E2})");
        _output.WriteLine($"  PyTorch BDN: 0.488ms");
        _output.WriteLine($"  Path A ratio: {msA / 0.488:F2}x vs PyTorch");
        _output.WriteLine($"  Path B ratio: {msB / 0.488:F2}x vs PyTorch");
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
