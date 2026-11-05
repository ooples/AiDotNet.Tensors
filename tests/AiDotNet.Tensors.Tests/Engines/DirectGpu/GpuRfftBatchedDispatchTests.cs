using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Correctness cover for the batched GPU RFFT/IRFFT dispatch.
///
/// WHAT CHANGED: the GPU path used to run one backend.RFFT PER SIGNAL and then weave the split real/imag
/// results into IEngine's interleaved layout with TWO single-element device copies PER FREQUENCY BIN — at
/// numFreqs=129 and batchSize=4096 that is ~1.06 MILLION device-to-device copies for one call. It now issues
/// a fixed handful of launches: Fill/CopyRows to zero-pad, one BatchedFFT, CopyRows to keep the positive
/// frequencies, and one InterleaveComplex. BatchedFFT already existed and was unused; InterleaveComplex is
/// new and had to be added to every backend, since all SplitComplex* ops work on separate real/imag buffers.
///
/// WHY THIS TEST MATTERS: none of that had ever executed. The kernels compiled and the launch-count
/// reduction is arithmetic from the code, but a rewritten index-weaving path is exactly where an off-by-one
/// or a transposed row silently produces plausible-looking garbage. The CPU engine is the reference.
///
/// The GPU engine is constructed DIRECTLY rather than via auto-detect, which vetoes adoption.
/// </summary>
[Collection("DirectGpuSerial")]
public class GpuRfftBatchedDispatchTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public GpuRfftBatchedDispatchTests(ITestOutputHelper output) => _out = output;
    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static Tensor<float> Signal(int batch, int n, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>([batch, n]);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    /// <summary>
    /// Constructs the GPU engine and PROVES the GPU path will actually be taken.
    /// </summary>
    /// <remarks>
    /// Constructing the engine is not sufficient evidence. IEngine.RFFT silently falls back to the managed
    /// CpuEngine implementation when no backend resolves, so without this guard the whole suite passes
    /// while exercising nothing but CPU code. That happened on the first run here: every CPU-vs-GPU delta
    /// came back as EXACTLY 0.000E+000, which is impossible for two independent float32 FFT
    /// implementations and was the tell that the CUDA natives were missing from the test output directory.
    /// A vacuous pass is worse than a skip, because it reads as verification.
    /// </remarks>
    private static bool TryGpu(out DirectGpuTensorEngine? engine)
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (!candidate.IsGpuAvailable)
            {
                candidate.Dispose();
                engine = null;
                return false;
            }
            engine = candidate;
            return true;
        }
        catch (Exception)
        {
            engine = null;
            return false;
        }
    }

    /// <summary>
    /// The batched GPU path must reproduce the CPU reference across batch sizes and lengths, including a
    /// NON-power-of-two length (192) that forces the zero-pad-to-nFft branch, which is where the strided
    /// CopyRows padding could go wrong.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 128)]
    [InlineData(8, 128)]
    [InlineData(64, 192)]
    [InlineData(256, 256)]
    public void Gpu_rfft_matches_cpu(int batch, int n)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null,
            "GPU backend did not resolve, so this test would have silently exercised the CPU path. " +
            "Copy the CUDA natives (cublas64_12, cublasLt64_12, cudart64_12, nvrtc64_120_0, " +
            "nvrtc64_120_0.alt, nvrtc-builtins64_126) into the test output directory after every build.");

        using (gpu!)
        {
            var x = Signal(batch, n, seed: 11);

            AiDotNetEngine.Current = new CpuEngine();
            var expected = AiDotNetEngine.Current.RFFT(x);

            // MUST go through IEngine: DirectGpuTensorEngine implements RFFT as an EXPLICIT interface
            // implementation (Tensor<T> IEngine.RFFT<T>), which is unreachable from the concrete type.
            // Calling gpu.RFFT(x) silently resolves to the inherited CpuEngine.RFFT — that is precisely how
            // the first version of this test compared the CPU implementation against itself and "passed".
            AiDotNetEngine.Current = gpu;
            IEngine gpuEngine = gpu;
            var actual = gpuEngine.RFFT(x);

            Assert.Equal(expected.Length, actual.Length);

            double maxAbs = 0, maxRel = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                double diff = Math.Abs(expected[i] - actual[i]);
                maxAbs = Math.Max(maxAbs, diff);
                maxRel = Math.Max(maxRel, diff / Math.Max(1.0, Math.Abs(expected[i])));
            }

            _out.WriteLine($"batch={batch,4} n={n,4}  maxAbs={maxAbs:E3}  maxRel={maxRel:E3}");

            // Independent float32 implementations do not agree to the last bit over thousands of values.
            // Exact equality means IEngine.RFFT fell back to CpuEngine and this compared CPU against CPU.
            Assert.True(maxAbs > 0.0,
                "GPU and CPU results were BIT-IDENTICAL, which means the GPU path did not run and this " +
                "assertion compared the CPU implementation with itself.");

            // float32 transform of length up to 256: accumulated rounding, not a formula difference.
            Assert.True(maxRel <= 1e-4,
                $"GPU batched RFFT diverged from CPU at batch={batch} n={n}: maxRel={maxRel:E3}. " +
                "A gross mismatch here means the CopyRows padding, the positive-frequency slice, or the " +
                "InterleaveComplex weave has its indexing wrong — not a rounding issue.");
        }
    }

    /// <summary>
    /// The GPU path must stay usable WITH A TAPE ACTIVE, and its gradients must match the CPU reference.
    /// </summary>
    /// <remarks>
    /// RFFT/IRFFT previously bailed to the managed CPU implementation whenever IsTapeActive was true, so the
    /// GPU FFT kernels on all seven backends were unreachable during training — the one situation where the
    /// cost actually compounds. The bail is gone and the tape node is recorded on the GPU result instead
    /// (backward of RFFT is IRFFT and vice versa, exactly as CpuEngine does it).
    ///
    /// This asserts the gradient, not just that the forward ran: a tape node that records the wrong
    /// backward, or a deferred GPU result the tape cannot resolve, would still produce a correct forward.
    /// </remarks>
    [SkippableFact]
    public void Gpu_rfft_gradients_match_cpu_with_tape_active()
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null,
            "GPU backend did not resolve — see Gpu_rfft_matches_cpu for the natives this needs.");

        using (gpu!)
        {
            const int batch = 8, n = 128;
            var x = Signal(batch, n, seed: 17);

            float[] GradientOf(IEngine engine)
            {
                AiDotNetEngine.Current = engine;
                var input = new Tensor<float>(x.Shape.ToArray());
                for (int i = 0; i < x.Length; i++) input[i] = x[i];

                using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
                var spectrum = engine.RFFT(input);
                // Non-uniform weighting so the upstream gradient is not constant.
                var weight = new Tensor<float>(spectrum.Shape.ToArray());
                for (int i = 0; i < weight.Length; i++) weight[i] = 0.1f + 0.01f * (i % 7);
                var loss = engine.ReduceSum(engine.TensorMultiply(spectrum, weight), null);

                var grads = tape.ComputeGradients(loss, new[] { input });
                Assert.True(grads.TryGetValue(input, out var g) && g is not null,
                    "no gradient flowed back to the RFFT input — the tape node was not recorded");
                var flat = new float[g.Length];
                for (int i = 0; i < g.Length; i++) flat[i] = g[i];
                return flat;
            }

            var cpuGrad = GradientOf(new CpuEngine());
            var gpuGrad = GradientOf(gpu);

            Assert.Equal(cpuGrad.Length, gpuGrad.Length);

            double maxAbs = 0, maxRel = 0;
            for (int i = 0; i < cpuGrad.Length; i++)
            {
                double diff = Math.Abs(cpuGrad[i] - gpuGrad[i]);
                maxAbs = Math.Max(maxAbs, diff);
                maxRel = Math.Max(maxRel, diff / Math.Max(1.0, Math.Abs(cpuGrad[i])));
            }

            _out.WriteLine($"tape-active gradient  maxAbs={maxAbs:E3}  maxRel={maxRel:E3}");
            Assert.True(maxAbs > 0.0,
                "CPU and GPU gradients were BIT-IDENTICAL — the GPU path fell back to CPU despite the fix.");
            Assert.True(maxRel <= 1e-4, $"GPU gradient diverged from CPU: maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Round-trip through the rewritten paths: IRFFT(RFFT(x)) must recover x. This is the check that
    /// catches a de-interleave that mirrors the interleave's own mistake — both halves being consistently
    /// wrong would still pass a one-way comparison against a differently-laid-out reference.
    /// </summary>
    [SkippableTheory]
    [InlineData(8, 128)]
    [InlineData(64, 256)]
    public void Gpu_rfft_irfft_roundtrip_recovers_input(int batch, int n)
    {
        Skip.IfNot(TryGpu(out var gpu) && gpu is not null,
            "GPU backend did not resolve — see Gpu_rfft_matches_cpu for the natives this needs.");

        using (gpu!)
        {
            var x = Signal(batch, n, seed: 13);

            // Explicit interface implementation — see Gpu_rfft_matches_cpu.
            AiDotNetEngine.Current = gpu;
            IEngine gpuEngine = gpu;
            var spectrum = gpuEngine.RFFT(x);
            var recovered = gpuEngine.IRFFT(spectrum, n);

            Assert.Equal(x.Length, recovered.Length);

            double maxAbs = 0;
            for (int i = 0; i < x.Length; i++)
                maxAbs = Math.Max(maxAbs, Math.Abs(x[i] - recovered[i]));

            _out.WriteLine($"batch={batch,4} n={n,4}  roundtrip maxAbs={maxAbs:E3}");
            Assert.True(maxAbs <= 1e-3, $"round-trip lost the signal at batch={batch} n={n}: maxAbs={maxAbs:E3}");
        }
    }

    /// <summary>
    /// The IRFFT gather/sign tables and the adjoint's bin weights are cached per backend and shape. Interleaving batch
    /// sizes on ONE engine (3, 5, 3, 5) checks the cache key: a table reused across shapes would gather the wrong rows
    /// (or read past a smaller table), and a cached table that was disposed or overwritten would break the repeat calls.
    /// Forward and gradient are both compared with the CPU reference on every call.
    /// </summary>
    [SkippableFact]
    public void Gpu_irfft_cached_tables_match_cpu_across_interleaved_shapes()
    {
        if (!TryGpu(out var gpu) || gpu is null)
            throw new SkipException("GPU backend did not resolve — see Gpu_rfft_matches_cpu for the natives this needs.");

        using (gpu)
        {
            const int numFreqs = 65, n = 128;
            (float[] Out, float[] Grad) Run(IEngine engine, Tensor<float> spectrumIn)
            {
                AiDotNetEngine.Current = engine;
                var input = new Tensor<float>(spectrumIn.Shape.ToArray());
                for (int i = 0; i < input.Length; i++) input[i] = spectrumIn[i];
                using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
                var signal = engine.IRFFT(input, n);
                var weight = new Tensor<float>(signal.Shape.ToArray());
                for (int i = 0; i < weight.Length; i++) weight[i] = 0.1f + 0.01f * (i % 11);
                var loss = engine.ReduceSum(engine.TensorMultiply(signal, weight), null);
                var grads = tape.ComputeGradients(loss, new[] { input });
                if (!grads.TryGetValue(input, out var g) || g is null)
                    throw new Xunit.Sdk.XunitException("no gradient reached the IRFFT input");
                var o = new float[signal.Length];
                for (int i = 0; i < o.Length; i++) o[i] = signal[i];
                var gr = new float[g.Length];
                for (int i = 0; i < gr.Length; i++) gr[i] = g[i];
                return (o, gr);
            }

            static double MaxRel(float[] a, float[] b)
            {
                Assert.Equal(a.Length, b.Length);
                double m = 0;
                for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]) / Math.Max(1.0, Math.Abs(a[i])));
                return m;
            }

            int call = 0;
            foreach (int batch in new[] { 3, 5, 3, 5 })
            {
                var spectrum = Signal(batch, numFreqs * 2, seed: 100 + call++);
                var cpu = Run(new CpuEngine(), spectrum);
                var dev = Run(gpu, spectrum);
                double outRel = MaxRel(cpu.Out, dev.Out), gradRel = MaxRel(cpu.Grad, dev.Grad);
                _out.WriteLine($"call={call} batch={batch}  out maxRel={outRel:E3}  grad maxRel={gradRel:E3}");
                Assert.True(outRel <= 1e-4, $"cached IRFFT forward diverged at call {call}, batch={batch}: {outRel:E3}");
                Assert.True(gradRel <= 1e-4, $"cached IRFFT adjoint diverged at call {call}, batch={batch}: {gradRel:E3}");
            }
        }
    }

    /// <summary>
    /// Under a DeferredScope the IRFFT runs on the scope's recording backend. The cached tables must be keyed on the
    /// physical backend: keyed on the wrapper, every scope added its own entries (filling the bounded cache with tables
    /// for dead scopes), and engine disposal released them with an immediate Dispose because the wrapper has no queued
    /// release. Two scopes and an eager call of one shape must share the same two tables and still match the CPU.
    /// </summary>
    [SkippableFact]
    public void Gpu_irfft_cached_tables_are_shared_across_deferred_scopes()
    {
        if (!TryGpu(out var gpu) || gpu is null)
            throw new SkipException("GPU backend did not resolve — see Gpu_rfft_matches_cpu for the natives this needs.");

        using (gpu)
        {
            const int batch = 4, numFreqs = 33, n = 64;
            var spectrum = Signal(batch, numFreqs * 2, seed: 7);
            AiDotNetEngine.Current = new CpuEngine();
            var expected = AiDotNetEngine.Current.IRFFT(spectrum, n);
            AiDotNetEngine.Current = gpu;
            // Through IEngine: the GPU IRFFT is an explicit interface implementation.
            IEngine engine = gpu;

            void AssertMatchesCpu(Tensor<float> actual, string label)
            {
                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4 * Math.Max(1.0, Math.Abs(expected[i])),
                        $"{label}: [{i}] cpu {expected[i]} vs gpu {actual[i]}");
            }

            for (int s = 0; s < 2; s++)
            {
                // Trace-by-execution: pure deferral is only correct for the primitives RecordingGpuBackend records
                // (GpuExecutionOptions.ExecuteEagerlyWhileRecording), and IRFFT's gather/FFT are not among them.
                var scope = gpu.BeginDeferredScope(
                    new AiDotNet.Tensors.Engines.Gpu.GpuExecutionOptions { ExecuteEagerlyWhileRecording = true });
                if (scope is null) throw new SkipException("Backend has no deferred execution");
                using (scope)
                {
                    var r = engine.IRFFT(spectrum, n);
                    scope.Execute();
                    AssertMatchesCpu(r.Contiguous(), $"scope {s}");
                }
                _out.WriteLine($"after scope {s}: {gpu.CachedSpectrumConstantCount} cached tables");
            }
            AssertMatchesCpu(engine.IRFFT(spectrum, n).Contiguous(), "eager");

            // One gather index and one sign table for the single shape, whichever path built them.
            Assert.Equal(2, gpu.CachedSpectrumConstantCount);
        }
    }
}
