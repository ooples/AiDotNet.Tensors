#if NETCOREAPP
using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1549: under an FP16 or BF16 <see cref="AutocastScope"/>, 2D convolution runs in that precision on a backend with
/// native 16-bit convolution math (cuDNN on CUDA): forward, input gradient and weight gradient, eagerly and through the
/// tape. Each result is checked against an FP32 CPU convolution of the operands rounded to the autocast precision, so
/// the tolerance only has to cover accumulation order and the 16-bit rounding of the output; the dispatch counters
/// prove the 16-bit route ran, and the off-paths (no scope, an FP32-only scope, the kill switch) prove it did not.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class AutocastConvTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public AutocastConvTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        AiDotNetEngine.Current = _prior;
        DirectGpuTensorEngine.AutocastConvOverride = null;
        DirectGpuTensorEngine.AutocastConvInjectedFault = null;
    }

    private static DirectGpuTensorEngine RequireCudnnEngine(PrecisionMode precision = PrecisionMode.Float16)
    {
        DirectGpuTensorEngine? engine = null;
        try
        {
            engine = new DirectGpuTensorEngine();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException or InvalidOperationException)
        {
            Skip.If(true, "GPU backend did not resolve: " + ex.Message);
        }
        if (!engine!.IsGpuAvailable
            || engine.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend
            || engine.GetBackend() is not AiDotNet.Tensors.Engines.Gpu.INativeReducedPrecisionConv { NativeReducedPrecisionMath: true }
            || engine.GetBackend()!.MixedPrecisionConv is not { } conv
            || (precision == PrecisionMode.BFloat16 && !conv.SupportsBFloat16Conv))
        {
            engine.Dispose();
            Skip.If(true, $"Needs CUDA with cuDNN tensor-core {precision} convolution.");
        }
        return engine;
    }

    private static Tensor<float> Random(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    // The operand as the 16-bit route sees it: FP16 rounds to nearest; BF16 keeps the top 16 bits, round to nearest even.
    private static float Round16(float x, PrecisionMode precision)
    {
        if (precision == PrecisionMode.Float16) return (float)(Half)x;
        int bits = BitConverter.SingleToInt32Bits(x);
        bits += 0x7FFF + ((bits >> 16) & 1);
        return BitConverter.Int32BitsToSingle(bits & unchecked((int)0xFFFF0000));
    }

    private static Tensor<float> Rounded(Tensor<float> t, PrecisionMode precision)
    {
        var r = new Tensor<float>(t._shape);
        for (int i = 0; i < t.Length; i++) r[i] = Round16(t[i], precision);
        return r;
    }

    // Relative to the reference's largest magnitude: FP16 output rounding is 2^-11, BF16 2^-8; plus accumulation order.
    private static void AssertClose(float[] reference, float[] actual, PrecisionMode precision, string what)
    {
        Assert.Equal(reference.Length, actual.Length);
        double scale = Math.Max(1e-6, reference.Max(v => Math.Abs((double)v)));
        double tolerance = (precision == PrecisionMode.Float16 ? 2e-3 : 1.6e-2) * scale;
        double worst = 0;
        int worstAt = -1;
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), $"{what}: non-finite at {i}");
            double d = Math.Abs(reference[i] - actual[i]);
            if (d > worst) { worst = d; worstAt = i; }
        }
        Assert.True(worst <= tolerance,
            $"{what}: max |gpu - ref| {worst:E3} at {worstAt} exceeds {tolerance:E3} (ref scale {scale:E3})");
    }

    public static TheoryData<int, int, int, int, int, int, int, int, int> Shapes => new()
    {
        // batch, inC, outC, size, kernel, stride, pad, dilation, (unused seed salt)
        { 2, 16, 32, 12, 3, 1, 1, 1, 1 },
        { 3, 8, 16, 11, 3, 2, 1, 1, 2 },   // stride 2, odd size
        { 2, 32, 16, 9, 1, 1, 0, 1, 3 },   // 1x1
        { 1, 16, 16, 13, 3, 1, 2, 2, 4 },  // dilation 2
        { 4, 3, 24, 16, 5, 2, 2, 1, 5 },   // 5x5, 3 input channels (a stem)
        { 2, 64, 64, 8, 3, 1, 1, 1, 6 },   // deeper channels, small plane
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Forward_RunsIn16Bit_AndMatchesTheRoundedReference(int batch, int inC, int outC, int size, int k,
        int stride, int pad, int dilation, int salt)
    {
        foreach (var precision in new[] { PrecisionMode.Float16, PrecisionMode.BFloat16 })
        {
            using var engine = RequireCudnnEngine(PrecisionMode.Float16);
            if (precision == PrecisionMode.BFloat16
                && engine.GetBackend()!.MixedPrecisionConv is not { SupportsBFloat16Conv: true })
                continue;
            var x = Random([batch, inC, size, size], 10 + salt);
            var w = Random([outC, inC, k, k], 20 + salt, 0.3f);
            var b = Random([outC], 30 + salt, 0.5f);
            var reference = new CpuEngine().FusedConv2D(Rounded(x, precision), Rounded(w, precision), b,
                stride, stride, pad, pad, dilation, dilation, FusedActivationType.ReLU).ToArray();

            var before = engine.AutocastConvDispatchCounts.Forward;
            float[] actual;
            using (new AutocastScope(precision))
                actual = engine.FusedConv2D(x, w, b, stride, stride, pad, pad, dilation, dilation,
                    FusedActivationType.ReLU).ToArray();

            Assert.Equal(before + 1, engine.AutocastConvDispatchCounts.Forward);
            AssertClose(reference, actual, precision, $"{precision} forward");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void InputAndWeightGradients_RunIn16Bit_AndMatchTheRoundedReference(int batch, int inC, int outC, int size,
        int k, int stride, int pad, int dilation, int salt)
    {
        foreach (var precision in new[] { PrecisionMode.Float16, PrecisionMode.BFloat16 })
        {
            using var engine = RequireCudnnEngine(PrecisionMode.Float16);
            if (precision == PrecisionMode.BFloat16
                && engine.GetBackend()!.MixedPrecisionConv is not { SupportsBFloat16Conv: true })
                continue;
            int outSize = (size + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
            var x = Random([batch, inC, size, size], 40 + salt);
            var w = Random([outC, inC, k, k], 50 + salt, 0.3f);
            var dy = Random([batch, outC, outSize, outSize], 60 + salt);
            int[] s = [stride, stride], p = [pad, pad], d = [dilation, dilation];
            IEngine cpu = new CpuEngine();
            var refDx = cpu.Conv2DBackwardInput(Rounded(dy, precision), Rounded(w, precision), x._shape, s, p, d).ToArray();
            var refDw = cpu.Conv2DBackwardKernel(Rounded(dy, precision), Rounded(x, precision), w._shape, s, p, d).ToArray();

            var before = engine.AutocastConvDispatchCounts;
            float[] dx, dw;
            using (new AutocastScope(precision))
            {
                dx = ((IEngine)engine).Conv2DBackwardInput(dy, w, x._shape, s, p, d).ToArray();
                dw = ((IEngine)engine).Conv2DBackwardKernel(dy, x, w._shape, s, p, d).ToArray();
            }

            var after = engine.AutocastConvDispatchCounts;
            Assert.Equal(before.BackwardData + 1, after.BackwardData);
            Assert.Equal(before.BackwardFilter + 1, after.BackwardFilter);
            AssertClose(refDx, dx, precision, $"{precision} dX");
            AssertClose(refDw, dw, precision, $"{precision} dW");
        }
    }

    // Through the tape: y = relu(conv(x, w) + b), loss = sum(y^2). Every gradient the tape returns comes from the 16-bit
    // route, and agrees with an FP32 CPU tape over the rounded operands to 16-bit accuracy (the tape also rounds dY).
    [SkippableFact]
    public void TapeGradients_UnderFp16Autocast_UseThe16BitRoute_AndTrackTheCpuTape()
    {
        using var engine = RequireCudnnEngine();
        var x = Random([2, 8, 10, 10], 70);
        var w = Random([16, 8, 3, 3], 71, 0.3f);
        var b = Random([16], 72, 0.2f);

        Tensor<float> Loss(IEngine e, Tensor<float> xi, Tensor<float> wi, Tensor<float> bi)
        {
            var y = e.FusedConv2D(xi, wi, bi, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
            return e.ReduceSum(e.TensorMultiply(y, y), null);
        }

        var cpu = new CpuEngine();
        var rx = Rounded(x, PrecisionMode.Float16);
        var rw = Rounded(w, PrecisionMode.Float16);
        float[] refDx, refDw, refDb;
        AiDotNetEngine.Current = cpu;
        using (var tape = new GradientTape<float>())
        {
            var g = tape.ComputeGradients(Loss(cpu, rx, rw, b), sources: new[] { rx, rw, b });
            refDx = g[rx].ToArray(); refDw = g[rw].ToArray(); refDb = g[b].ToArray();
        }

        AiDotNetEngine.Current = engine;
        var before = engine.AutocastConvDispatchCounts;
        float[] dx, dw, db;
        using (new AutocastScope(PrecisionMode.Float16))
        using (var tape = new GradientTape<float>())
        {
            var g = tape.ComputeGradients(Loss(engine, x, w, b), sources: new[] { x, w, b });
            dx = g[x].ToArray(); dw = g[w].ToArray(); db = g[b].ToArray();
        }
        var after = engine.AutocastConvDispatchCounts;
        _output.WriteLine($"dispatches: forward {after.Forward - before.Forward}, dX {after.BackwardData - before.BackwardData}, dW {after.BackwardFilter - before.BackwardFilter}");
        Assert.True(after.Forward > before.Forward, "the forward did not take the 16-bit route");
        Assert.True(after.BackwardData > before.BackwardData, "dX did not take the 16-bit route");
        Assert.True(after.BackwardFilter > before.BackwardFilter, "dW did not take the 16-bit route");
        // dY is itself rounded to FP16 before the backward convolutions, so these compare looser than one op does.
        AssertCloseLoose(refDx, dx, "tape dX");
        AssertCloseLoose(refDw, dw, "tape dW");
        AssertCloseLoose(refDb, db, "tape db");
    }

    private static void AssertCloseLoose(float[] reference, float[] actual, string what)
    {
        double scale = Math.Max(1e-6, reference.Max(v => Math.Abs((double)v)));
        double worst = reference.Zip(actual, (r, a) => Math.Abs((double)r - a)).Max();
        Assert.True(worst <= 1e-2 * scale, $"{what}: max |gpu - ref| {worst:E3} exceeds {1e-2 * scale:E3}");
    }

    // Outside autocast, under an FP32 scope, and with the kill switch, the convolution keeps the FP32 routes: the
    // 16-bit counters do not move, and the result is the FP32 one (much closer to the unrounded reference).
    [SkippableFact]
    public void Fp32Paths_DoNotTakeThe16BitRoute()
    {
        using var engine = RequireCudnnEngine();
        var x = Random([2, 16, 12, 12], 80);
        var w = Random([32, 16, 3, 3], 81, 0.3f);
        var exact = new CpuEngine().FusedConv2D(x, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
        var start = engine.AutocastConvDispatchCounts.Forward;

        float[] plain = engine.FusedConv2D(x, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
        float[] fp32Scope;
        using (new AutocastScope(PrecisionMode.Float32))
            fp32Scope = engine.FusedConv2D(x, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
        float[] killed;
        DirectGpuTensorEngine.AutocastConvOverride = false;
        using (new AutocastScope(PrecisionMode.Float16))
            killed = engine.FusedConv2D(x, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();

        Assert.Equal(start, engine.AutocastConvDispatchCounts.Forward);
        double scale = exact.Max(v => Math.Abs((double)v));
        foreach (var (name, values) in new[] { ("no scope", plain), ("fp32 scope", fp32Scope), ("kill switch", killed) })
        {
            double worst = exact.Zip(values, (r, a) => Math.Abs((double)r - a)).Max();
            // FP32 (possibly TF32-free cuDNN or direct kernels) stays well inside FP16 rounding of the operands.
            Assert.True(worst <= 1e-4 * Math.Max(1, scale), $"{name}: FP32 conv drifted {worst:E3}");
        }
    }

    // The engine's Conv2DInto (the pre-allocated-output conv) takes the route too.
    [SkippableFact]
    public void Conv2DInto_UnderFp16Autocast_TakesThe16BitRoute()
    {
        using var engine = RequireCudnnEngine();
        var x = Random([2, 16, 10, 10], 90);
        var w = Random([24, 16, 3, 3], 91, 0.3f);
        var reference = new CpuEngine().FusedConv2D(Rounded(x, PrecisionMode.Float16), Rounded(w, PrecisionMode.Float16),
            null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
        var output = new Tensor<float>([2, 24, 10, 10]);
        var before = engine.AutocastConvDispatchCounts.Forward;
        using (new AutocastScope(PrecisionMode.Float16))
            ((IEngine)engine).Conv2DInto(output, x, w, 1, 1, 1);
        Assert.Equal(before + 1, engine.AutocastConvDispatchCounts.Forward);
        AssertClose(reference, output.ToArray(), PrecisionMode.Float16, "Conv2DInto");
    }

    // A 16-bit convolution cuDNN declines (an unsupported algorithm or data type for the shape, a failed workspace
    // allocation) runs as the FP32 GPU convolution: no exception reaches the caller, nothing moves to the host, and the
    // result is the FP32 one. The fault is injected where the cuDNN call would fail.
    [SkippableFact]
    public void A_declined_16bit_convolution_runs_the_fp32_gpu_convolution()
    {
        using var engine = RequireCudnnEngine();
        var x = Random([2, 8, 10, 10], 100);
        var w = Random([16, 8, 3, 3], 101, 0.3f);
        var dy = Random([2, 16, 10, 10], 102);
        int[] s = [1, 1], p = [1, 1], d = [1, 1];
        IEngine cpu = new CpuEngine();
        var exactY = cpu.Conv2D(x, w, s, p, d).ToArray();
        var exactDx = cpu.Conv2DBackwardInput(dy, w, x._shape, s, p, d).ToArray();
        var exactDw = cpu.Conv2DBackwardKernel(dy, x, w._shape, s, p, d).ToArray();
        var fallbacksBefore = GpuLaunchProbe.Fallbacks;

        var before = engine.AutocastConvDispatchCounts;
        float[] y, dx, dw;
        DirectGpuTensorEngine.AutocastConvInjectedFault = "convolution forward";
        try
        {
            using (new AutocastScope(PrecisionMode.Float16))
            {
                y = engine.FusedConv2D(x, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
                dx = ((IEngine)engine).Conv2DBackwardInput(dy, w, x._shape, s, p, d).ToArray();
                dw = ((IEngine)engine).Conv2DBackwardKernel(dy, x, w._shape, s, p, d).ToArray();
            }
        }
        finally
        {
            DirectGpuTensorEngine.AutocastConvInjectedFault = null;
        }

        Assert.Equal(before, engine.AutocastConvDispatchCounts);
        _output.WriteLine("fallbacks: " + string.Join("; ", GpuLaunchProbe.Fallbacks.Where(f => f.Contains("Conv"))));
        Assert.Contains(GpuLaunchProbe.Fallbacks, f => f.Contains("AutocastConv2DForward: InvalidOperationException"));
        Assert.Contains(GpuLaunchProbe.Fallbacks, f => f.Contains("AutocastConv2DBackwardData: InvalidOperationException"));
        Assert.Contains(GpuLaunchProbe.Fallbacks, f => f.Contains("AutocastConv2DBackwardFilter: InvalidOperationException"));
        // The ops themselves stayed on the device: no new host fallback was recorded for them (entries read "Nx op: reason").
        static string[] HostFallbacks(string[] entries) => entries.Where(f => f.Contains(" FusedConv2D:")
            || f.Contains(" Conv2DBackwardInput:") || f.Contains(" Conv2DBackwardKernel:")).OrderBy(f => f).ToArray();
        Assert.Equal(HostFallbacks(fallbacksBefore), HostFallbacks(GpuLaunchProbe.Fallbacks));
        foreach (var (name, exact, actual) in new[] { ("y", exactY, y), ("dX", exactDx, dx), ("dW", exactDw, dw) })
        {
            double scale = Math.Max(1, exact.Max(v => Math.Abs((double)v)));
            double worst = exact.Zip(actual, (r, a) => Math.Abs((double)r - a)).Max();
            Assert.True(worst <= 1e-4 * scale, $"{name}: the fallback is not the FP32 convolution (drift {worst:E3})");
        }
    }

    // An explicit PreserveInputType policy keeps convolutions in FP32 under autocast, as it does every other autocast op.
    [SkippableFact]
    public void PreserveInputTypePolicy_KeepsTheConvolutionInFp32()
    {
        using var engine = RequireCudnnEngine();
        var x = Random([2, 8, 10, 10], 110);
        var w = Random([16, 8, 3, 3], 111, 0.3f);
        var exact = new CpuEngine().FusedConv2D(x, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
        var before = engine.AutocastConvDispatchCounts;
        float[] y;
        using (new AutocastScope(PrecisionMode.Float16))
        using (new GpuExecutionPolicyScope(GpuExecutionPolicy.Preserve))
            y = engine.FusedConv2D(x, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
        Assert.Equal(before, engine.AutocastConvDispatchCounts);
        double worst = exact.Zip(y, (r, a) => Math.Abs((double)r - a)).Max();
        Assert.True(worst <= 1e-4 * Math.Max(1, exact.Max(v => Math.Abs((double)v))), $"drift {worst:E3}");
    }

    // The GPU-resident entry points (FusedConv2DGpu, Conv2DBackwardInputGpu, Conv2DBackwardKernelGpu) take the route too.
    [SkippableFact]
    public void GpuResidentEntryPoints_UnderFp16Autocast_TakeThe16BitRoute()
    {
        using var engine = RequireCudnnEngine();
        const PrecisionMode fp16 = PrecisionMode.Float16;
        var x = Random([2, 8, 10, 10], 120);
        var w = Random([16, 8, 3, 3], 121, 0.3f);
        var dy = Random([2, 16, 10, 10], 122);
        int[] s = [1, 1], p = [1, 1], d = [1, 1];
        IEngine cpu = new CpuEngine();
        var refY = cpu.Conv2D(Rounded(x, fp16), Rounded(w, fp16), s, p, d).ToArray();
        var refDx = cpu.Conv2DBackwardInput(Rounded(dy, fp16), Rounded(w, fp16), x._shape, s, p, d).ToArray();
        var refDw = cpu.Conv2DBackwardKernel(Rounded(dy, fp16), Rounded(x, fp16), w._shape, s, p, d).ToArray();

        var gx = engine.UploadToGpu(x, GpuTensorRole.Activation);
        var gdy = engine.UploadToGpu(dy, GpuTensorRole.Gradient);
        var before = engine.AutocastConvDispatchCounts;
        float[] y, dx, dw;
        using (new AutocastScope(fp16))
        {
            y = engine.FusedConv2DGpu(gx, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
            dx = engine.Conv2DBackwardInputGpu(gdy, w, x._shape, s, p, d).ToArray();
            dw = engine.Conv2DBackwardKernelGpu(gdy, gx, w._shape, s, p, d).ToArray();
        }
        var after = engine.AutocastConvDispatchCounts;
        Assert.Equal((before.Forward + 1, before.BackwardData + 1, before.BackwardFilter + 1), after);
        AssertClose(refY, y, fp16, "FusedConv2DGpu");
        AssertClose(refDx, dx, fp16, "Conv2DBackwardInputGpu");
        AssertClose(refDw, dw, fp16, "Conv2DBackwardKernelGpu");
    }

    // Conv2DInto inside a compiled step's resident scope: the 16-bit convolution writes the resident FP32 output (the
    // branch a captured step replays), and with the route off the plain FP32 convolution does.
    [SkippableFact]
    public void Conv2DInto_InAResidentStep_UnderFp16Autocast_WritesTheResidentOutputIn16Bit()
    {
        using var engine = RequireCudnnEngine();
        var x = Random([2, 16, 10, 10], 130);
        var w = Random([24, 16, 3, 3], 131, 0.3f);
        var rounded = new CpuEngine().FusedConv2D(Rounded(x, PrecisionMode.Float16), Rounded(w, PrecisionMode.Float16),
            null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();
        var exact = new CpuEngine().FusedConv2D(x, w, null, 1, 1, 1, 1, 1, 1, FusedActivationType.None).ToArray();

        float[] Run(bool routeOn, out long forwards)
        {
            var output = new Tensor<float>([2, 24, 10, 10]);
            var previousPool = DirectGpuTensorEngine.EnterScratchPool(engine.CreateScratchPool());
            engine.SuspendActivationEviction();
            long start = engine.AutocastConvDispatchCounts.Forward;
            DirectGpuTensorEngine.AutocastConvOverride = routeOn ? null : false;
            try
            {
                using (engine.EnterCompiledCapturePath())
                {
                    Assert.True(engine.ResidentStepActive);
                    engine.SetCurrentScratchAction(0);
                    using (new AutocastScope(PrecisionMode.Float16))
                        ((IEngine)engine).Conv2DInto(output, x, w, 1, 1, 1);
                    Assert.NotNull(output._gpuBuffer);   // bound resident, not downloaded
                }
            }
            finally
            {
                DirectGpuTensorEngine.AutocastConvOverride = null;
                engine.SetCurrentScratchAction(-1);
                DirectGpuTensorEngine.RestoreScratchPool(previousPool);
                engine.ResumeActivationEviction();
            }
            forwards = engine.AutocastConvDispatchCounts.Forward - start;
            var values = output.ToArray();
            engine.ClearActionScratchPool();
            return values;
        }

        var on = Run(routeOn: true, out long onForwards);
        var off = Run(routeOn: false, out long offForwards);
        Assert.Equal(1, onForwards);
        Assert.Equal(0, offForwards);
        AssertClose(rounded, on, PrecisionMode.Float16, "resident Conv2DInto");
        double worst = exact.Zip(off, (r, a) => Math.Abs((double)r - a)).Max();
        Assert.True(worst <= 1e-4 * Math.Max(1, exact.Max(v => Math.Abs((double)v))), $"FP32 resident drift {worst:E3}");
    }

    // A downsampling block (3x3 stride-2 conv and 1x1 stride-2 projection of the same activation) compiled under FP16
    // autocast: the FP32 shortcut-pair kernel is declined, each input gradient takes the 16-bit route, and the
    // gradients track the FP32 plan's to 16-bit accuracy.
    [SkippableFact]
    public void A_compiled_downsampling_block_under_fp16_autocast_declines_the_fp32_shortcut_pair()
    {
        using var engine = RequireCudnnEngine();
        const int batch = 2, inC = 4, c = 16, k = 24, size = 16, steps = 3;

        (float[][] Grads, long PairLaunches, (long, long, long) Dispatches) Train(bool autocast)
        {
            AiDotNetEngine.Current = engine;
            var x = Random([batch, inC, size, size], 140).Gpu();
            var w0 = Random([c, inC, 3, 3], 141, 0.3f).Gpu();
            var w3 = Random([k, c, 3, 3], 142, 0.1f).Gpu();
            var w1 = Random([k, c, 1, 1], 143, 0.2f).Gpu();
            using var scope16 = autocast ? new AutocastScope(PrecisionMode.Float16) : null;
            ICompiledTrainingPlan<float> plan;
            using (var scope = AiDotNet.Tensors.Engines.Compilation.GraphMode.Enable())
            {
                var h = engine.ReLU(engine.Conv2D(x, w0, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }));
                var main = engine.Conv2D(h, w3, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 });
                var shortcut = engine.Conv2D(h, w1, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 1, 1 });
                var sum = engine.TensorAdd(main, shortcut);
                engine.ReduceSum(engine.TensorMultiply(sum, sum), new[] { 0, 1, 2, 3 }, keepDims: false);
                plan = scope.CompileTraining(new[] { w0, w3, w1 });
            }
            long pairBefore = DirectGpuTensorEngine.ShortcutPairLaunches;
            var countsBefore = engine.AutocastConvDispatchCounts;
            float[][] grads = [];
            using (plan)
                for (int st = 0; st < steps; st++)
                {
                    plan.Step();
                    grads = [plan.Gradients[0].ToArray(), plan.Gradients[1].ToArray(), plan.Gradients[2].ToArray()];
                }
            var countsAfter = engine.AutocastConvDispatchCounts;
            return (grads, DirectGpuTensorEngine.ShortcutPairLaunches - pairBefore,
                (countsAfter.Forward - countsBefore.Forward, countsAfter.BackwardData - countsBefore.BackwardData,
                    countsAfter.BackwardFilter - countsBefore.BackwardFilter));
        }

        var fp32 = Train(autocast: false);
        var fp16 = Train(autocast: true);
        _output.WriteLine($"pair launches fp32 {fp32.PairLaunches}, fp16 {fp16.PairLaunches}; fp16 dispatches {fp16.Dispatches}");
        Assert.Equal(0, fp16.PairLaunches);
        var (forwards, dataGrads, filterGrads) = fp16.Dispatches;
        Assert.True(forwards >= 3, $"forward dispatches {forwards}");
        Assert.True(dataGrads >= 2, $"input-gradient dispatches {dataGrads}: the pair's two input gradients run in 16-bit");
        Assert.True(filterGrads >= 3, $"weight-gradient dispatches {filterGrads}");
        string[] names = ["dW0", "dW3", "dW1"];
        for (int i = 0; i < 3; i++)
            AssertCloseLoose(fp32.Grads[i], fp16.Grads[i], "after 3 steps " + names[i]);
    }
}
#endif
