using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Dedicated GPU-vs-CPU accuracy coverage for the conv/pool/attention kernels that the generic
/// auto-differential harness cannot drive (distinct shapes per tensor arg / interdependent geometry).
/// These kernels were previously hidden from the EveryGpuKernel coverage gate by a `public new` hide on
/// DirectGpuTensorEngine; once converted to `override` the gate (correctly) demands an accuracy check.
/// Each test runs the op on a real GPU engine and the CPU reference and asserts element-wise agreement
/// (TF32-tolerant). They no-op on CPU-only hosts. Shapes for backward ops are derived from a forward pass
/// so the geometry is always self-consistent.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class GpuConvKernelCoverageTests : IDisposable
{
    private readonly DirectGpuTensorEngine _gpu;
    private readonly CpuEngine _cpu = new();
    private readonly bool _ready;

    public GpuConvKernelCoverageTests()
    {
        // Only swallow the two exceptions that mean "no native GPU runtime on this host"
        // (PlatformNotSupportedException / DllNotFoundException). A real GPU setup or kernel/module
        // compilation regression MUST surface as a test failure, not get converted into an "unavailable"
        // skip that hides the regression this suite exists to catch. Mirrors DetectionGpuParityTests.
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _ready = _gpu.SupportsGpu;
        }
        catch (PlatformNotSupportedException) { _ready = false; }
        catch (DllNotFoundException) { _ready = false; }

        // Prove the GPU kernel ACTUALLY runs: make the conv/pool/attention catch blocks (here in the engine
        // AND inside the Metal/Vulkan backends) rethrow instead of silently falling back to the CPU reference.
        // Without this, a GPU kernel that threw would fall back to CPU and every GPU-vs-CPU assertion below
        // would compare CPU-vs-CPU and trivially pass (false green) — the exact gap that left issue #622
        // looking "fixed" without proof. The flag is [ThreadStatic], so set it only when a GPU is present and
        // always clear it in Dispose so it never leaks into other test collections on this thread.
        if (_ready) DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
    }

    public void Dispose()
    {
        DirectGpuTensorEngine.ThrowOnGpuKernelFallback = false;
        _gpu?.Dispose();
    }

    // Skip (visibly, via Xunit.SkipException) rather than silently `return` when
    // no GPU is present, so a failed/absent GPU setup shows as a skipped test
    // instead of a false green. Mirrors the DirectGpu suite's SkipIfUnavailable
    // convention (e.g. DetectionGpuParityTests).
    private void SkipIfUnavailable()
        => Skip.If(!_ready, "DirectGpu/GPU unavailable; skipping GPU-vs-CPU conv coverage.");

    private static Tensor<float> R(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        return new Tensor<float>(data, shape);
    }

    private static Tensor<float> Like(int seed, Tensor<float> t)
    {
        var rng = new Random(seed);
        var data = new float[t.Length];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() + 0.1);
        return new Tensor<float>(data, t.Shape.ToArray());
    }

    private static void AssertClose(Tensor<float> cpu, Tensor<float> gpu, string name, float rtol = 2e-2f, float atol = 3e-3f)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        var c = cpu.GetDataArray();
        var g = gpu.GetDataArray();
        for (int i = 0; i < c.Length; i++)
        {
            float diff = MathF.Abs(c[i] - g[i]);
            float tol = atol + rtol * MathF.Abs(c[i]);
            Assert.True(diff <= tol, $"{name}[{i}]: cpu={c[i]:G6} gpu={g[i]:G6} diff={diff:G4} > tol={tol:G4}");
        }
    }

    // ---- DepthwiseConv2D ----
    [SkippableFact]
    public void DepthwiseConv2D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var input = R(1, 1, 2, 5, 5);
        var kernel = R(2, 2, 1, 3, 3); // [inCh, mult, kH, kW]
        int[] stride = { 1, 1 }, pad = { 1, 1 };
        AssertClose(_cpu.DepthwiseConv2D(input, kernel, stride, pad),
                    _gpu.DepthwiseConv2D(input, kernel, stride, pad), "DepthwiseConv2D");
    }

    // ---- DeformableConv2D (forward + all four backward) ----
    private (Tensor<float> input, Tensor<float> kernel, Tensor<float> offset, Tensor<float> mask,
             int[] stride, int[] pad, int[] dil, Tensor<float> fwd, Tensor<float> grad) DeformSetup()
    {
        var input = R(10, 1, 2, 5, 5);
        var kernel = R(11, 3, 2, 3, 3);            // [outCh, inCh, kH, kW]
        var offset = R(12, 1, 2 * 3 * 3, 5, 5);    // [B, 2*kH*kW, outH, outW], small deformations
        var mask = R(13, 1, 3 * 3, 5, 5);          // [B, kH*kW, outH, outW]
        int[] stride = { 1, 1 }, pad = { 1, 1 }, dil = { 1, 1 };
        var fwd = _cpu.DeformableConv2D(input, kernel, offset, mask, stride, pad, dil);
        return (input, kernel, offset, mask, stride, pad, dil, fwd, Like(14, fwd));
    }

    [SkippableFact]
    public void DeformableConv2D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var s = DeformSetup();
        AssertClose(s.fwd, _gpu.DeformableConv2D(s.input, s.kernel, s.offset, s.mask, s.stride, s.pad, s.dil),
                    "DeformableConv2D");
    }

    [SkippableFact]
    public void DeformableConv2DBackwardInput_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var s = DeformSetup();
        int[] inShape = s.input.Shape.ToArray();
        AssertClose(
            _cpu.DeformableConv2DBackwardInput(s.grad, s.input, s.kernel, s.offset, s.mask, inShape, s.stride, s.pad, s.dil),
            _gpu.DeformableConv2DBackwardInput(s.grad, s.input, s.kernel, s.offset, s.mask, inShape, s.stride, s.pad, s.dil),
            "DeformableConv2DBackwardInput");
    }

    [SkippableFact]
    public void DeformableConv2DBackwardKernel_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var s = DeformSetup();
        int[] kShape = s.kernel.Shape.ToArray();
        AssertClose(
            _cpu.DeformableConv2DBackwardKernel(s.grad, s.input, s.offset, s.mask, kShape, s.stride, s.pad, s.dil),
            _gpu.DeformableConv2DBackwardKernel(s.grad, s.input, s.offset, s.mask, kShape, s.stride, s.pad, s.dil),
            "DeformableConv2DBackwardKernel");
    }

    [SkippableFact]
    public void DeformableConv2DBackwardOffset_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var s = DeformSetup();
        AssertClose(
            _cpu.DeformableConv2DBackwardOffset(s.grad, s.input, s.kernel, s.offset, s.mask, s.stride, s.pad, s.dil),
            _gpu.DeformableConv2DBackwardOffset(s.grad, s.input, s.kernel, s.offset, s.mask, s.stride, s.pad, s.dil),
            "DeformableConv2DBackwardOffset");
    }

    [SkippableFact]
    public void DeformableConv2DBackwardMask_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var s = DeformSetup();
        AssertClose(
            _cpu.DeformableConv2DBackwardMask(s.grad, s.input, s.kernel, s.offset, s.mask, s.stride, s.pad, s.dil),
            _gpu.DeformableConv2DBackwardMask(s.grad, s.input, s.kernel, s.offset, s.mask, s.stride, s.pad, s.dil),
            "DeformableConv2DBackwardMask");
    }

    // ---- FusedConv3D ----
    [SkippableFact]
    public void FusedConv3D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var input = R(20, 1, 1, 4, 4, 4);   // [B, Cin, D, H, W]
        var kernel = R(21, 2, 1, 2, 2, 2);  // [Cout, Cin, kD, kH, kW]
        AssertClose(
            _cpu.FusedConv3D(input, kernel, null, 1, 1, 1, 0, 0, 0, 1, 1, 1, FusedActivationType.None),
            _gpu.FusedConv3D(input, kernel, null, 1, 1, 1, 0, 0, 0, 1, 1, 1, FusedActivationType.None),
            "FusedConv3D");
    }

    // ---- FusedConvTranspose2D ----
    [SkippableFact]
    public void FusedConvTranspose2D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var input = R(30, 1, 1, 3, 3);   // [B, Cin, H, W]
        var kernel = R(31, 1, 2, 2, 2);  // [Cin, Cout, kH, kW]
        AssertClose(
            _cpu.FusedConvTranspose2D(input, kernel, null, 2, 2, 0, 0, 0, 0, FusedActivationType.None),
            _gpu.FusedConvTranspose2D(input, kernel, null, 2, 2, 0, 0, 0, 0, FusedActivationType.None),
            "FusedConvTranspose2D");
    }

    // ---- LocallyConnectedConv2D (forward + two backward) ----
    private (Tensor<float> input, Tensor<float> weights, int[] stride, Tensor<float> fwd, Tensor<float> grad) LocalSetup()
    {
        var input = R(40, 1, 1, 3, 3);
        var weights = R(41, 2, 2, 2, 1, 2, 2); // [outH, outW, outCh, inCh, kH, kW]
        int[] stride = { 1, 1 };
        var fwd = _cpu.LocallyConnectedConv2D(input, weights, null, stride);
        return (input, weights, stride, fwd, Like(42, fwd));
    }

    [SkippableFact]
    public void LocallyConnectedConv2D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var s = LocalSetup();
        AssertClose(s.fwd, _gpu.LocallyConnectedConv2D(s.input, s.weights, null, s.stride), "LocallyConnectedConv2D");
    }

    [SkippableFact]
    public void LocallyConnectedConv2DBackwardInput_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var s = LocalSetup();
        int[] inShape = s.input.Shape.ToArray();
        AssertClose(
            _cpu.LocallyConnectedConv2DBackwardInput(s.grad, s.weights, inShape, s.stride),
            _gpu.LocallyConnectedConv2DBackwardInput(s.grad, s.weights, inShape, s.stride),
            "LocallyConnectedConv2DBackwardInput");
    }

    [SkippableFact]
    public void LocallyConnectedConv2DBackwardWeights_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var s = LocalSetup();
        int[] wShape = s.weights.Shape.ToArray();
        AssertClose(
            _cpu.LocallyConnectedConv2DBackwardWeights(s.grad, s.input, wShape, s.stride),
            _gpu.LocallyConnectedConv2DBackwardWeights(s.grad, s.input, wShape, s.stride),
            "LocallyConnectedConv2DBackwardWeights");
    }

    // ---- FlashAttentionBackward ----
    [SkippableFact]
    public void FlashAttentionBackward_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        // [B, H, S, D]
        var q = R(50, 1, 2, 4, 8);
        var k = R(51, 1, 2, 4, 8);
        var v = R(52, 1, 2, 4, 8);
        double scale = 1.0 / Math.Sqrt(8);
        var output = _cpu.FlashAttention(q, k, v, scale, false, out var stats, null);
        var grad = Like(53, output);

        var dQc = _cpu.FlashAttentionBackward(grad, q, k, v, output, stats, scale, false,
            out var dKc, out var dVc, out var _unusedC, null);
        var dQg = _gpu.FlashAttentionBackward(grad, q, k, v, output, stats, scale, false,
            out var dKg, out var dVg, out var _unusedG, null);
        // The method returns gradQuery and yields gradKey/gradValue via out params.
        AssertClose(dQc, dQg, "FlashAttentionBackward.dQ");
        AssertClose(dKc, dKg, "FlashAttentionBackward.dK");
        AssertClose(dVc, dVg, "FlashAttentionBackward.dV");
    }

    // ---- Conv/pool family forward coverage (#646): these high-level engine ops dispatch to the backend
    // conv/pool kernels, so on a Metal/Vulkan runner they gate the new MSL/GLSL kernels (and on OpenCL they
    // confirm the shared math vs the CPU reference). ----
    [SkippableFact]
    public void Conv2D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var input = R(60, 1, 2, 6, 6);
        var kernel = R(61, 3, 2, 3, 3); // [outC, inC, kH, kW]
        int[] stride = { 1, 1 }, pad = { 1, 1 }, dil = { 1, 1 };
        AssertClose(_cpu.Conv2D(input, kernel, stride, pad, dil),
                    _gpu.Conv2D(input, kernel, stride, pad, dil), "Conv2D");
    }

    [SkippableFact]
    public void Conv3D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var input = R(62, 1, 2, 4, 4, 4);
        var kernel = R(63, 3, 2, 2, 2, 2); // [outC, inC, kD, kH, kW]
        int[] stride = { 1, 1, 1 }, pad = { 0, 0, 0 }, dil = { 1, 1, 1 };
        AssertClose(_cpu.Conv3D(input, kernel, stride, pad, dil),
                    _gpu.Conv3D(input, kernel, stride, pad, dil), "Conv3D");
    }

    [SkippableFact]
    public void GlobalAvgPool2D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var input = R(64, 2, 3, 5, 5);
        AssertClose(_cpu.GlobalAvgPool2D(input), _gpu.GlobalAvgPool2D(input), "GlobalAvgPool2D");
    }

    [SkippableFact]
    public void GlobalMaxPool2D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var input = R(65, 2, 3, 5, 5);
        AssertClose(_cpu.GlobalMaxPool2D(input), _gpu.GlobalMaxPool2D(input), "GlobalMaxPool2D");
    }

    [SkippableFact]
    public void MaxPool3D_Gpu_MatchesCpu()
    {
        SkipIfUnavailable();
        var input = R(66, 1, 2, 4, 4, 4);
        int[] pool = { 2, 2, 2 }, stride = { 2, 2, 2 }, pad = { 0, 0, 0 };
        AssertClose(_cpu.MaxPool3D(input, pool, stride, pad),
                    _gpu.MaxPool3D(input, pool, stride, pad), "MaxPool3D");
    }
}
