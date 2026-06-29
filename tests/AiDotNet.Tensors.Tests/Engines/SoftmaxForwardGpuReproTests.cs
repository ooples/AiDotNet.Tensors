using System;
using System.Globalization;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Engines.Gpu.Graph;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Repro for the "GPU softmax forward returns a non-distribution / throws" bug (hawkeye capstone).
/// Covers the two reachable GPU softmax-forward paths:
///   1. Eager fused-linear (<see cref="DirectGpuTensorEngine.FusedLinearGpu"/>) — what DenseLayer.ForwardGpu/Predict use.
///   2. Async fused GEMM+bias+activation (<c>IAsyncGpuBackend.FusedGemmBiasActivationAsync</c>) — the
///      deferred/graph-fusion path (KernelFusionPass folds GEMM→bias→Softmax into this single call).
/// Both must produce a valid per-row distribution (each row in [0,1], sums to 1) matching a CPU
/// reference. Run on native Windows so the OpenCL GPU engages.
/// </summary>
public class SoftmaxForwardGpuReproTests
{
    private readonly ITestOutputHelper _out;
    public SoftmaxForwardGpuReproTests(ITestOutputHelper outp) => _out = outp;

    private const int InF = 4, OutF = 4;

    private static (float[] a, float[] w, float[] b) MakeInputs(int batch)
    {
        var a = new float[batch * InF];
        for (int i = 0; i < a.Length; i++) a[i] = (float)Math.Sin(0.6 * (i + 1));
        var w = new float[InF * OutF];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(0.3 * Math.Cos(0.4 * (i + 1)));
        var b = new float[OutF];
        for (int j = 0; j < OutF; j++) b[j] = 0.05f * (j - 1);
        return (a, w, b);
    }

    // softmax(input @ W + b), per row — the correct reference.
    private static float[] CpuReference(float[] a, float[] w, float[] b, int batch)
    {
        var refOut = new float[batch * OutF];
        for (int r = 0; r < batch; r++)
        {
            var logits = new float[OutF];
            for (int j = 0; j < OutF; j++)
            {
                float acc = b[j];
                for (int i = 0; i < InF; i++) acc += a[r * InF + i] * w[i * OutF + j];
                logits[j] = acc;
            }
            float mx = logits.Max();
            float sum = 0f;
            for (int j = 0; j < OutF; j++) { logits[j] = (float)Math.Exp(logits[j] - mx); sum += logits[j]; }
            for (int j = 0; j < OutF; j++) refOut[r * OutF + j] = logits[j] / sum;
        }
        return refOut;
    }

    private void AssertValidDistribution(float[] gpu, float[] refOut, int batch)
    {
        string Fmt(float[] x, int r) =>
            "[" + string.Join(", ", Enumerable.Range(0, OutF).Select(j => x[r * OutF + j].ToString("F4", CultureInfo.InvariantCulture))) + "]";
        for (int r = 0; r < batch; r++)
        {
            float rowSum = Enumerable.Range(0, OutF).Sum(j => gpu[r * OutF + j]);
            _out.WriteLine($"row[{r}] GPU={Fmt(gpu, r)} sum={rowSum:F4}   CPU={Fmt(refOut, r)}");
        }
        for (int r = 0; r < batch; r++)
        {
            float rowSum = 0f;
            for (int j = 0; j < OutF; j++)
            {
                float v = gpu[r * OutF + j];
                Assert.False(float.IsNaN(v) || float.IsInfinity(v), $"row {r} col {j} is NaN/Inf: {v}");
                Assert.True(v >= -1e-5f && v <= 1f + 1e-5f, $"row {r} col {j} out of [0,1]: {v}");
                rowSum += v;
            }
            Assert.True(Math.Abs(rowSum - 1f) < 1e-3f, $"row {r} sum={rowSum} != 1");
            for (int j = 0; j < OutF; j++)
                Assert.True(Math.Abs(gpu[r * OutF + j] - refOut[r * OutF + j]) < 1e-3f,
                    $"row {r} col {j}: GPU={gpu[r * OutF + j]} CPU={refOut[r * OutF + j]}");
        }
    }

    /// <summary>Path 1 — eager fused linear, exactly what DenseLayer.ForwardGpu calls.</summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(2)]
    public void FusedLinearSoftmax_IsValidDistribution(int batch)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.SupportsGpu, "No GPU backend available (expected on WSL/CPU-only hosts).");
        _out.WriteLine($"Engine: {engine.Name}   batch={batch}");

        var (a, w, b) = MakeInputs(batch);
        var inputT = new Tensor<float>(a, new[] { batch, InF });
        var wT = new Tensor<float>(w, new[] { InF, OutF });
        var bT = new Tensor<float>(b, new[] { OutF });

        float[] gpu;
        try
        {
            using var resultT = engine.FusedLinearGpu(inputT, wT, bT, FusedActivationType.Softmax);
            gpu = resultT.ToArray();
        }
        catch (Exception ex)
        {
            _out.WriteLine($"FusedLinearGpu(Softmax) THREW: {ex.GetType().Name}: {ex.Message}");
            throw new Xunit.Sdk.XunitException($"BUG: FusedLinearGpu(Softmax) threw for batch={batch}: {ex.GetType().Name}: {ex.Message}");
        }
        AssertValidDistribution(gpu, CpuReference(a, w, b, batch), batch);
    }

    /// <summary>
    /// Path 2 — async fused GEMM+bias+Softmax. This is what KernelFusionPass folds a
    /// GEMM→bias→Softmax chain into and runs under streamed graph execution.
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(2)]
    public void FusedGemmBiasActivationAsyncSoftmax_IsValidDistribution(int batch)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.SupportsGpu, "No GPU backend available (expected on WSL/CPU-only hosts).");
        var backend = engine.GetBackend();
        Skip.IfNot(backend is IAsyncGpuBackend, "Backend is not async-capable.");
        var asyncBackend = (IAsyncGpuBackend)backend;
        _out.WriteLine($"Engine: {engine.Name}   batch={batch}");

        var (a, w, b) = MakeInputs(batch);
        // M=batch rows, K=inFeatures, N=outFeatures.
        var aBuf = backend.AllocateBuffer(a);
        var wBuf = backend.AllocateBuffer(w);
        var biasBuf = backend.AllocateBuffer(b);
        var outBuf = backend.AllocateBuffer(batch * OutF);
        var stream = asyncBackend.DefaultStream;

        float[] gpu;
        try
        {
            asyncBackend.FusedGemmBiasActivationAsync(aBuf, wBuf, biasBuf, outBuf,
                batch, OutF, InF, FusedActivationType.Softmax, stream);
            asyncBackend.SynchronizeStream(stream);
            gpu = new float[batch * OutF];
            backend.DownloadBuffer(outBuf, gpu);
        }
        catch (Exception ex)
        {
            _out.WriteLine($"FusedGemmBiasActivationAsync(Softmax) THREW: {ex.GetType().Name}: {ex.Message}");
            throw new Xunit.Sdk.XunitException($"BUG C: async fused GEMM+bias+Softmax threw for batch={batch}: {ex.GetType().Name}: {ex.Message}");
        }
        AssertValidDistribution(gpu, CpuReference(a, w, b, batch), batch);
    }

    /// <summary>
    /// Path 2b — async fused GEMM+bias+Softmax on a NON-DEFAULT stream, with the inputs uploaded
    /// asynchronously on that same stream. A backend that ignores the caller-supplied <c>stream</c>
    /// (running GemmBias/Softmax on its default stream while the uploads are still in flight on the
    /// compute stream) can read the operands before they land. Regression guard for the stream-ordering
    /// fix in <c>FusedGemmBiasActivationAsync</c> — the original code synchronized only AFTER the op and
    /// on the wrong stream, so it neither fenced prior <c>stream</c> work nor was visible to
    /// <c>SynchronizeStream(stream)</c>.
    /// </summary>
    [SkippableTheory]
    [InlineData(2)]
    public void FusedGemmBiasActivationAsyncSoftmax_HonorsNonDefaultStream(int batch)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.SupportsGpu, "No GPU backend available (expected on WSL/CPU-only hosts).");
        var backend = engine.GetBackend();
        Skip.IfNot(backend is IAsyncGpuBackend, "Backend is not async-capable.");
        var asyncBackend = (IAsyncGpuBackend)backend;
        _out.WriteLine($"Engine: {engine.Name}   batch={batch}  (non-default stream)");

        var (a, w, b) = MakeInputs(batch);
        var stream = asyncBackend.CreateStream(GpuStreamType.Compute);
        float[] gpu;
        try
        {
            // Allocate device buffers, then upload the inputs ASYNCHRONOUSLY on the compute stream so the
            // fused op MUST honor `stream` (drain it before reading) to see the correct operands.
            var aBuf = backend.AllocateBuffer(a.Length);
            var wBuf = backend.AllocateBuffer(w.Length);
            var biasBuf = backend.AllocateBuffer(b.Length);
            var outBuf = backend.AllocateBuffer(batch * OutF);
            asyncBackend.UploadBufferAsync(a, aBuf, stream);
            asyncBackend.UploadBufferAsync(w, wBuf, stream);
            asyncBackend.UploadBufferAsync(b, biasBuf, stream);

            asyncBackend.FusedGemmBiasActivationAsync(aBuf, wBuf, biasBuf, outBuf,
                batch, OutF, InF, FusedActivationType.Softmax, stream);
            asyncBackend.SynchronizeStream(stream);
            gpu = new float[batch * OutF];
            backend.DownloadBuffer(outBuf, gpu);
        }
        catch (Exception ex)
        {
            _out.WriteLine($"non-default-stream fused Softmax THREW: {ex.GetType().Name}: {ex.Message}");
            throw new Xunit.Sdk.XunitException($"BUG: async fused Softmax on a non-default stream threw for batch={batch}: {ex.GetType().Name}: {ex.Message}");
        }
        finally
        {
            (stream as IDisposable)?.Dispose();
        }
        AssertValidDistribution(gpu, CpuReference(a, w, b, batch), batch);
    }

    // GPU-resident operands for a [batch, InF] @ [InF, OutF] + bias linear layer, built via the graph.
    private static (Tensor<float> input, Tensor<float> weights, Tensor<float> bias, Tensor<float> output)
        MakeGpuOperands(int batch, float[] a, float[] w, float[] bv)
    {
        var input = new Tensor<float>(a, new[] { batch, InF }).Gpu();
        var weights = new Tensor<float>(w, new[] { InF, OutF }).Gpu();
        var bias = new Tensor<float>(bv, new[] { OutF }).Gpu();
        var output = new Tensor<float>(new[] { batch, OutF }).Gpu();
        return (input, weights, bias, output);
    }

    /// <summary>
    /// Path 3 — the DEFERRED graph path, ASYNC (fused) route. Builds the layer through
    /// <see cref="ExecutionGraphBuilderExtensions.AddLinearLayer"/> and runs it with a stream pool, so
    /// the work actually flows through the graph machinery (KernelFusionPass folds GEMM→bias→Softmax into
    /// a single <c>FusedKernelNode</c> whose action calls <c>FusedGemmBiasActivationAsync</c>) rather than
    /// a direct backend call. Reachable on async backends (CUDA/HIP/OpenCL).
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(2)]
    public void DeferredGraph_FusedLinearSoftmax_AsyncFusedRoute_IsValidDistribution(int batch)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.SupportsGpu, "No GPU backend available (expected on WSL/CPU-only hosts).");
        var backend = engine.GetBackend();
        Skip.IfNot(backend is IAsyncGpuBackend, "Async fused graph route requires an IAsyncGpuBackend (CUDA/HIP/OpenCL).");
        var asyncBackend = (IAsyncGpuBackend)backend;
        _out.WriteLine($"Engine: {engine.Name}   batch={batch}  (deferred graph, async fused route)");

        var (a, w, b) = MakeInputs(batch);
        var (input, weights, bias, output) = MakeGpuOperands(batch, a, w, b);
        float[] gpu;
        try
        {
            using var builder = new ExecutionGraphBuilder();
            builder.AddLinearLayer(input, weights, bias, output, batch, InF, OutF, FusedActivationType.Softmax, backend);
            using var graph = builder.Build();
            using var streamPool = new GpuStreamPool(asyncBackend);
            graph.Execute(backend, streamPool);   // streamed execution → fused node fires on a real stream
            gpu = new float[batch * OutF];
            backend.DownloadBuffer(output.Buffer, gpu);
        }
        finally
        {
            input.Dispose(); weights.Dispose(); bias.Dispose(); output.Dispose();
        }
        AssertValidDistribution(gpu, CpuReference(a, w, b, batch), batch);
    }

    /// <summary>
    /// Path 4 — the DEFERRED graph path, SEPARATE-NODE route (GEMM → bias-add → <c>ApplyActivation</c>).
    /// <see cref="ExecutionGraphBuilderExtensions.AddLinearLayer"/> only takes this route on NON-async
    /// backends (Vulkan / WebGPU / Metal); async backends fold into the fused node instead. So this is
    /// the test that exercises <c>ExecutionGraphBuilder.ApplyActivation</c> (the per-row, scratch-buffered
    /// Softmax). It is gated to non-async backends and SKIPS on CUDA/HIP/OpenCL — run it on a machine where
    /// the active DirectGpu backend is Vulkan/WebGPU/Metal (e.g. via AIDOTNET_DIRECTGPU_BACKENDS).
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(2)]
    public void DeferredGraph_FusedLinearSoftmax_SeparateNodeRoute_IsValidDistribution(int batch)
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.SupportsGpu, "No GPU backend available (expected on WSL/CPU-only hosts).");
        var backend = engine.GetBackend();
        Skip.If(backend is IAsyncGpuBackend,
            "Separate-node graph route only runs on non-async backends (Vulkan/WebGPU/Metal); async backends use the fused node.");
        _out.WriteLine($"Engine: {engine.Name}   batch={batch}  (deferred graph, separate-node route)");

        var (a, w, b) = MakeInputs(batch);
        var (input, weights, bias, output) = MakeGpuOperands(batch, a, w, b);
        float[] gpu;
        try
        {
            using var builder = new ExecutionGraphBuilder();
            builder.AddLinearLayer(input, weights, bias, output, batch, InF, OutF, FusedActivationType.Softmax, backend);
            using var graph = builder.Build();
            graph.Execute(backend);   // sync execution → GEMM, bias-add, then ApplyActivation(Softmax)
            gpu = new float[batch * OutF];
            backend.DownloadBuffer(output.Buffer, gpu);
        }
        finally
        {
            input.Dispose(); weights.Dispose(); bias.Dispose(); output.Dispose();
        }
        AssertValidDistribution(gpu, CpuReference(a, w, b, batch), batch);
    }
}
