// SCRATCH (not committed): times GEMM entry points on a narrow-N product.
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Parity;

public class ZzNarrowGemmScratchBench
{
    private readonly ITestOutputHelper _o;
    public ZzNarrowGemmScratchBench(ITestOutputHelper o) => _o = o;

    [Fact]
    public void Time()
    {
        if (Environment.GetEnvironmentVariable("ZZ_ALLOC") == "1")
        {
            var sw0 = Stopwatch.StartNew();
            for (int i = 0; i < 3_000_000; i++) AiDotNet.Tensors.Helpers.TensorAllocator.RentUninitialized<float>(new[] { 128, 10 });
            _o.WriteLine($"ZZ alloc loop {sw0.Elapsed.TotalMilliseconds * 1000 / 3_000_000:F2} us");
            return;
        }
        if (Environment.GetEnvironmentVariable("ZZ_NARROW") != "1" && Environment.GetEnvironmentVariable("GITHUB_ACTIONS") != "true") return;
        int m = 128, k = 256, n = 10;
        var rng = new Random(1);
        var a = new float[m * k]; var b = new float[k * n]; var c = new float[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)rng.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)rng.NextDouble();
        var engine = new CpuEngine();
        var ta = new Tensor<float>(a, new[] { m, k }); var tb = new Tensor<float>(b, new[] { k, n });
        void T(string name, Action f)
        {
            for (int i = 0; i < 2000; i++) f();
            var times = new double[301];
            for (int r = 0; r < times.Length; r++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < 20; i++) f();
                times[r] = sw.Elapsed.TotalMilliseconds * 1000 / 20;
            }
            Array.Sort(times);
            _o.WriteLine($"ZZ {name,-28} {times[times.Length / 2]:F2} us");
            Console.WriteLine($"ZZ {name,-28} {times[times.Length / 2]:F2} us");
        }
        T("TensorMatMul", () => engine.TensorMatMul(ta, tb));
        var mi = typeof(CpuEngine).GetMethod("TensorMatMul2D", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.MakeGenericMethod(typeof(float));
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>();
        var d2 = (Func<Tensor<float>, Tensor<float>, AiDotNet.Tensors.Interfaces.INumericOperations<float>, Tensor<float>>)Delegate.CreateDelegate(
            typeof(Func<Tensor<float>, Tensor<float>, AiDotNet.Tensors.Interfaces.INumericOperations<float>, Tensor<float>>), engine, mi);
        T("TensorMatMul2D", () => d2(ta, tb, numOps));
        T("TensorMatMul+Dispose", () => engine.TensorMatMul(ta, tb).Dispose());
        T("RentUninit+Dispose", () => AiDotNet.Tensors.Helpers.TensorAllocator.RentUninitialized<float>(new[] { m, n }).Dispose());
        T("SimdGemm.Sgemm", () => SimdGemm.Sgemm(a, b, c, m, k, n));
        T("SimdGemm.SgemmSequential", () => SimdGemm.SgemmSequential(a, b, c, m, k, n));
        T("RentOrAllocate", () => AiDotNet.Tensors.Helpers.AutoTensorCache.RentOrAllocate<float>(new[] { m, n }));
        T("AllocUninit1280", () => GC.AllocateUninitializedArray<float>(m * n));
        T("FromMemory", () => Tensor<float>.FromMemory(new Memory<float>(GC.AllocateUninitializedArray<float>(m * n)), new[] { m, n }));
        T("TensorAllocator.RentUninit", () => AiDotNet.Tensors.Helpers.TensorAllocator.RentUninitialized<float>(new[] { m, n }));
        T("FrozenTryGetHandle", () => AiDotNet.Tensors.Engines.BlasManaged.FrozenWeightRegistry.TryGetHandle(tb));
        T("AsSpanUnmarked2", () => { ta.AsSpanUnmarked(); tb.AsSpanUnmarked(); });
        T("BlasManaged.GemmDA", () => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k, new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float> { PackingMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode.DisableAutotune }));
        System.Threading.Thread.Sleep(500);
        T("TensorMatMul(again)", () => engine.TensorMatMul(ta, tb));
        T("TensorMatMul2D(again)", () => d2(ta, tb, numOps));
        T("RentUninit(again)", () => AiDotNet.Tensors.Helpers.TensorAllocator.RentUninitialized<float>(new[] { m, n }));
        T("OpScope", () => { using var sc = AiDotNet.Tensors.Engines.Profiling.Profiler.OpScope("TensorMatMul"); });
        T("GraphMode.IsActive", () => { _ = AiDotNet.Tensors.Engines.Compilation.GraphMode.IsActive; });
        T("AutoTracer.TryGet", () => AiDotNet.Tensors.Engines.Compilation.AutoTracer.TryGetCompiledPlan<float>("TensorMatMul", new[] { m, n }));
        T("AutoTracer.RecordOp", () => { var r0 = ta; AiDotNet.Tensors.Engines.Compilation.AutoTracer.RecordOp("TensorMatMul", ta, eng => eng.TensorMatMul(r0, tb)); });
        T("RecordBinary", () => AiDotNet.Tensors.Engines.Autodiff.DifferentiableOps.RecordBinary("TensorMatMul", ta, ta, tb, AiDotNet.Tensors.Engines.Autodiff.BackwardFunctions<float>.MatMulBackward));
        T("IsContiguous2", () => { _ = ta.IsContiguous; _ = tb.IsContiguous; });
        T("GetNumOps", () => AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<float>());
        T("TensorMatMul(3rd)", () => engine.TensorMatMul(ta, tb));
        T("Sgemm(narrow512 on)", () => SimdGemm.Sgemm(a, k, false, b, n, false, c, m, k, n));
        SimdGemm.s_narrow512 = false;
        T("Sgemm(narrow512 off)", () => SimdGemm.Sgemm(a, k, false, b, n, false, c, m, k, n));
        T("TensorMatMul(narrow512 off)", () => engine.TensorMatMul(ta, tb));
        SimdGemm.s_narrow512 = true;
        T("TensorMatMul(4th)", () => engine.TensorMatMul(ta, tb));
        T("BlasManaged.Gemm", () => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k));
    }
}
