using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Phase (CNN conv-inference): locates the Phase-0 CNN gap (bs1 ~7.6 ms vs
/// torch.compile ~0.42 ms = 18×). The parity benchmark measured the WHOLE
/// AiDotNet CNN <c>Predict()</c> (Conv→ReLU→Pool→Conv→ReLU→Pool→Flatten→Linear)
/// through the high-level layer stack. This micro-bench isolates the Tensors
/// kernel cost — bare <see cref="CpuEngine.FusedConv2D{T}"/> and raw
/// <c>Conv2D</c> on the exact CNN conv shapes — to decide WHICH component is
/// slow: the Tensors conv kernel (fixable here) or the AiDotNet per-layer
/// dispatch (fixable with a fused conv-stem primitive, like MlpForward).
/// Env-gated (AIDOTNET_RUN_JIT_PERF=1); CI-safe no-op otherwise.
/// </summary>
public class ConvInferenceBreakdownBench
{
    private readonly ITestOutputHelper _output;
    public ConvInferenceBreakdownBench(ITestOutputHelper output) => _output = output;

    [Fact]
    public void CnnConvShapes_BareKernel_Breakdown()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_JIT_PERF") != "1") return;

        var engine = new CpuEngine();
#if NET5_0_OR_GREATER
        // System.Runtime.Intrinsics.X86 and OneDnnProvider are .NET-Core-only; net471
        // lacks both (CS0234 / CS0103), so this richer diagnostic is net5+ only.
        _output.WriteLine($"AVX2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} cores={Environment.ProcessorCount} " +
            $"HasRawSgemm={BlasProvider.HasRawSgemm} OneDnn={OneDnnProvider.IsAvailable}");
#else
        _output.WriteLine($"cores={Environment.ProcessorCount} HasRawSgemm={BlasProvider.HasRawSgemm}");
#endif

        // The two CNN conv layers from benchmarks/AiDotNet.PyTorchParity:
        //   Conv2d(1,16,3,pad=1) on 28×28  → [N,16,28,28]
        //   Conv2d(16,32,3,pad=1) on 14×14 → [N,32,14,14]  (after a 2× maxpool)
        foreach (int bs in new[] { 1, 8, 32 })
        {
            BenchConv(engine, bs, inC: 1, outC: 16, h: 28, w: 28, label: "conv1 1->16 @28x28");
            BenchConv(engine, bs, inC: 16, outC: 32, h: 14, w: 14, label: "conv2 16->32 @14x14");
        }
    }

    private void BenchConv(CpuEngine engine, int bs, int inC, int outC, int h, int w, string label)
    {
        var input = MakeTensor(new[] { bs, inC, h, w }, 11);
        var kernel = MakeTensor(new[] { outC, inC, 3, 3 }, 22);
        var bias = MakeTensor(new[] { outC }, 33);

        // Full FusedConv2D (conv + bias + ReLU fused) — what an inference layer should call.
        Func<Tensor<float>> fused = () =>
            engine.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
        // Raw Conv2D only (no epilogue) — isolates the conv kernel from bias/act.
        Func<Tensor<float>> raw = () =>
            engine.Conv2D(input, kernel, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });

        // im2col + cached-B GEMM + fused epilogue, reusing scratch (the plan's pattern).
        //   im2col[K, outHW]  (K = inC*9),  out[outC, outHW] = kernel[outC,K] @ im2col[K,outHW]
        int K = inC * 9;
        int colW = h * w;                     // per-batch im2col width (pad1,s1 → outH=h,outW=w)
        var im2col = new float[K * colW];     // reused per-batch im2col (rebuilt every batch)
        var kernelArr = (float[])(object)kernel.GetDataArray();
        var biasArr = (float[])(object)bias.GetDataArray();
        var gemmOut = new float[bs * outC * colW];
        var inputSliceSize = inC * h * w;
        Func<Tensor<float>> gemmConv = () =>
        {
            var inSpan = input.AsSpan();
            // Non-caching managed GEMM (im2col is input-dependent — caching is
            // semantically wrong; SgemmWithCachedB would serve a stale pack).
            //   C[outC, colW] = kernel[outC, K] @ im2col[K, colW]   (NCHW output)
            for (int b = 0; b < bs; b++)
            {
                Im2ColHelper.Im2Col(inSpan.Slice(b * inputSliceSize, inputSliceSize), im2col.AsSpan(0, K * colW),
                    1, inC, h, w, 3, 3, 1, 1, 1, 1, 1, 1);
                AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(
                    kernelArr.AsSpan(0, outC * K),
                    im2col.AsSpan(0, K * colW),
                    gemmOut.AsSpan(b * outC * colW, outC * colW),
                    outC, K, colW);
            }
            CpuFusedOperations.ApplyBiasActivationNCHWInPlace(
                gemmOut, biasArr, bs, outC, h, w, FusedActivationType.ReLU);
            return null!;
        };

        // Correctness: GEMM-conv output must match the direct FusedConv2D output.
        gemmConv();
        var refOut = engine.FusedConv2D(input, kernel, bias, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU).ToArray();
        double maxDiff = 0;
        for (int i = 0; i < refOut.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(refOut[i] - gemmOut[i]));

        double fusedMs = Bench(fused, out long fusedBytes);
        double rawMs = Bench(raw, out long rawBytes);
        double gemmMs = BenchVoid(gemmConv);

        _output.WriteLine(
            $"[bs={bs,3}] {label,-22}  FusedConv2D {fusedMs * 1000:F1}us ({fusedBytes} B/call)   " +
            $"rawConv2D {rawMs * 1000:F1}us   " +
            $"im2col+Sgemm+epilogue {gemmMs * 1000:F1}us   (maxDiff {maxDiff:E2})");
    }

    private static double BenchVoid(Func<Tensor<float>> f)
    {
        for (int i = 0; i < 20; i++) { var _ = f(); }
        double best = double.MaxValue;
        for (int r = 0; r < 200; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var _ = f();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    private static double Bench(Func<Tensor<float>> f, out long bytesPerCall)
    {
        for (int i = 0; i < 20; i++) { var _ = f(); }
        long before = AllocatedBytes();
        const int reps = 200;
        double best = double.MaxValue;
        for (int r = 0; r < reps; r++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var _ = f();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        long after = AllocatedBytes();
        bytesPerCall = (after - before) / reps;
        return best;
    }

    // GC.GetAllocatedBytesForCurrentThread() is .NET Core 3.0+ only; on net471 the
    // build broke (CS0117). Fall back to a coarse whole-heap proxy there so this
    // diagnostic bench compiles and runs on both target frameworks.
    private static long AllocatedBytes()
    {
#if NET5_0_OR_GREATER
        return GC.GetAllocatedBytesForCurrentThread();
#else
        return GC.GetTotalMemory(forceFullCollection: false);
#endif
    }

    private static Tensor<float> MakeTensor(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }
}
