using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The single-GEMM transposed-convolution forward (<see cref="PortableConvTransposeGemm"/>, the kernels of
/// <see cref="ConvTransposeGemmKernelTemplate"/>) on each non-CUDA backend against a float64 reference: the DCGAN
/// generator's stem, middle stage and RGB head, an output padding, odd geometry and a batch past the column cap (chunked),
/// each within 1e-4 of sum|terms|, every output written, bit-reproducible across two runs. Both the route itself and the
/// backend's public ConvTranspose2D, which takes it first, are checked, and the bias form (IConvTranspose2DBiasBackend,
/// the bias added in the overlap-add) must equal the route's output plus the bias bit for bit. HIP's module is compiled as CUDA C++ under NVRTC
/// (the dialect has nothing HIP-specific), so it is checked on any CUDA box; CUDA's own route is covered by
/// ConvTranspose GPU-vs-CPU tests.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ConvTransposeGemmPortTests
{
    private readonly ITestOutputHelper _out;
    public ConvTransposeGemmPortTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void ColumnWorkspaceMustFitAtLeastOneSample()
    {
        Assert.Equal(0, PortableConvTransposeGemm.Chunk(1, 32, 512, 512, 32, 512, 512, 7, 7, 1, 1, 3, 3));
        Assert.Equal(8, PortableConvTransposeGemm.Chunk(10, 4, 32, 32, 64, 64, 64, 4, 4, 2, 2, 1, 1));
    }

    // batch, inC, inH, inW, outC, k, stride, pad, outputPad.
    public static IEnumerable<object[]> Cases() => new[]
    {
        new object[] { 4, 32, 1, 1, 16, 4, 1, 0, 0 },    // the generator stem: 1x1 -> 4x4
        new object[] { 3, 32, 4, 4, 16, 4, 2, 1, 0 },    // a middle stage: 4x4 -> 8x8
        new object[] { 2, 16, 8, 8, 3, 4, 2, 1, 0 },     // the RGB head: 8x8 -> 16x16, three channels
        new object[] { 2, 5, 5, 3, 7, 3, 2, 1, 1 },      // odd extents, an output padding
        new object[] { 1, 3, 4, 6, 2, 5, 3, 2, 0 },      // a kernel wider than twice the stride
        new object[] { 10, 4, 32, 32, 64, 4, 2, 1, 0 },  // P * L = 1M: the column cap runs it in chunks of eight
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void OpenCl_matches_the_float64_reference(int batch, int inC, int inH, int inW, int outC, int k, int s, int p,
        int op)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        if (backend.InitializationState == GpuBackendInitializationState.Failed)
            throw new InvalidOperationException("OpenCL initialization failed.", backend.InitializationException);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        Check(backend, backend.TryConvTranspose2DGemm, batch, inC, inH, inW, outC, k, s, p, op);
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Vulkan_matches_the_float64_reference(int batch, int inC, int inH, int inW, int outC, int k, int s, int p,
        int op)
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        Check(backend, backend.TryConvTranspose2DGemm, batch, inC, inH, inW, outC, k, s, p, op);
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Metal_matches_the_float64_reference(int batch, int inC, int inH, int inW, int outC, int k, int s, int p,
        int op)
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        Check(backend, backend.TryConvTranspose2DGemm, batch, inC, inH, inW, outC, k, s, p, op);
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Hip_matches_the_float64_reference(int batch, int inC, int inH, int inW, int outC, int k, int s, int p,
        int op)
    {
        using var backend = new HipBackend();
        if (backend.InitializationState == GpuBackendInitializationState.Failed)
            throw new InvalidOperationException("HIP initialization failed.", backend.InitializationException);
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        Check(backend, backend.TryConvTranspose2DGemm, batch, inC, inH, inW, outC, k, s, p, op);
    }

#if NET7_0_OR_GREATER
    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void WebGpu_matches_the_float64_reference(int batch, int inC, int inH, int inW, int outC, int k, int s, int p,
        int op)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Check(backend, backend.TryConvTranspose2DGemm, batch, inC, inH, inW, outC, k, s, p, op);
    }
#endif

    [SkippableFact]
    public void Hip_module_compiles_as_cuda_cxx()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.IsAvailable, "NVRTC is not available.");
        string source = ConvTransposeGemmKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip);
        IntPtr program = IntPtr.Zero;
        Assert.Equal(AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcResult.Success,
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, "convt_hip",
                0, IntPtr.Zero, IntPtr.Zero));
        try
        {
            var result = AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcCompileProgram(program, 1,
                new[] { "--gpu-architecture=compute_75" });
            Assert.True(result == AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcResult.Success,
                $"the HIP dialect did not compile ({result})");
        }
        finally
        {
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
    }

    [Fact]
    public void The_chunk_respects_the_column_cap_and_refuses_degenerate_shapes()
    {
        // P * L = 1024 * 1024: eight samples fill the 8M-float column buffer.
        Assert.Equal(8, PortableConvTransposeGemm.Chunk(10, 4, 32, 32, 64, 64, 64, 4, 4, 2, 2, 1, 1));
        Assert.Equal(3, PortableConvTransposeGemm.Chunk(3, 32, 4, 4, 16, 8, 8, 4, 4, 2, 2, 1, 1));
        Assert.Equal(0, PortableConvTransposeGemm.Chunk(0, 32, 4, 4, 16, 8, 8, 4, 4, 2, 2, 1, 1));
        Assert.Equal(0, PortableConvTransposeGemm.Chunk(2, 32, 4, 4, 16, 8, 8, 4, 4, 0, 2, 1, 1));
        Assert.Equal(0, PortableConvTransposeGemm.Chunk(2, 32, 4, 4, 16, 8, 8, 4, 4, 2, 2, -1, 1));
    }

    internal delegate bool Route(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output, int batch, int inChannels,
        int inHeight, int inWidth, int outChannels, int outHeight, int outWidth, int kernelH, int kernelW, int strideH,
        int strideW, int padH, int padW, IGpuBuffer? bias);

    private static float[] Rand(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static readonly ConcurrentDictionary<string, (float[] x, float[] w, double[] sum, double[] mag)> s_references = new();

    private void Check(IDirectGpuBackend backend, Route route, int batch, int inC, int inH, int inW, int outC, int k,
        int s, int p, int op)
    {
        int outH = (inH - 1) * s - 2 * p + k + op, outW = (inW - 1) * s - 2 * p + k + op;
        var r = s_references.GetOrAdd($"{batch} {inC} {inH} {inW} {outC} {k} {s} {p} {op}",
            _ => Reference(batch, inC, inH, inW, outC, k, s, p, outH, outW));
        using var x = backend.AllocateBuffer(r.x);
        using var w = backend.AllocateBuffer(r.w);
        var poison = Enumerable.Repeat(float.NaN, batch * outC * outH * outW).ToArray();
        using var y1 = backend.AllocateBuffer(poison);
        using var y2 = backend.AllocateBuffer(poison);
        using var y3 = backend.AllocateBuffer(poison);
        using var y4 = backend.AllocateBuffer(poison);
        var biasValues = Rand(outC, new Random(77 + outC));
        using var bias = backend.AllocateBuffer(biasValues);
        foreach (var y in new[] { y1, y2 })
            Assert.True(route(x, w, y, batch, inC, inH, inW, outC, outH, outW, k, k, s, s, p, p, null),
                "the transposed-convolution GEMM route refused a served shape");
        // The public entry point takes the route first.
        backend.ConvTranspose2D(x, w, y3, batch, inC, inH, inW, outC, outH, outW, k, k, s, s, p, p, op, op);
        var fusedBias = Assert.IsAssignableFrom<IConvTranspose2DBiasBackend>(backend);
        fusedBias.ConvTranspose2DBias(x, w, bias, y4, batch, inC, inH, inW, outC, outH, outW, k, k, s, s, p, p, op, op);
        backend.Synchronize();
        float[] got = backend.DownloadBuffer(y1), again = backend.DownloadBuffer(y2), viaPublic = backend.DownloadBuffer(y3);
        float[] biased = backend.DownloadBuffer(y4);
        for (int i = 0; i < biased.Length; i++)
        {
            float expected = got[i] + biasValues[i / (outH * outW) % outC];
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(expected) == TestHelpers.MathCompat.SingleToInt32Bits(biased[i]),
                $"y[{i}] with the bias: {biased[i]} vs the route's output plus the bias {expected}");
        }
        double worst = 0;
        for (int i = 0; i < got.Length; i++)
        {
            Assert.False(float.IsNaN(got[i]), $"y[{i}] was never written");
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(got[i]) == TestHelpers.MathCompat.SingleToInt32Bits(again[i]),
                $"y[{i}] differs between two identical runs");
            Assert.True(TestHelpers.MathCompat.SingleToInt32Bits(got[i]) == TestHelpers.MathCompat.SingleToInt32Bits(viaPublic[i]),
                $"y[{i}]: ConvTranspose2D {viaPublic[i]} vs the route {got[i]}");
            double rel = Math.Abs(got[i] - r.sum[i]) / Math.Max(1.0, r.mag[i]);
            worst = Math.Max(worst, rel);
            Assert.True(rel <= 1e-4, $"y[{i}] = {got[i]} vs {r.sum[i]} (|terms| {r.mag[i]})");
        }
        _out.WriteLine($"b{batch} {inC}->{outC} {inH}x{inW} k{k} s{s} p{p} op{op}: max err / sum|terms| = {worst:E2}");
    }

    // y[b, co, oh, ow] = sum over ci, kh, kw with oh = ih * s - p + kh, ow = iw * s - p + kw of x[b, ci, ih, iw] *
    // w[ci, co, kh, kw] (the scatter form, independent of the gather the kernels use).
    private static (float[], float[], double[], double[]) Reference(int batch, int inC, int inH, int inW, int outC, int k,
        int s, int p, int outH, int outW)
    {
        var rng = new Random(9157 + batch * 31 + inC * 7 + outC * 3 + k + s + p);
        var x = Rand(batch * inC * inH * inW, rng);
        var w = Rand(inC * outC * k * k, rng);
        var sum = new double[batch * outC * outH * outW];
        var mag = new double[sum.Length];
        for (int b = 0; b < batch; b++)
            for (int ci = 0; ci < inC; ci++)
                for (int ih = 0; ih < inH; ih++)
                    for (int iw = 0; iw < inW; iw++)
                    {
                        double xv = x[((b * inC + ci) * inH + ih) * inW + iw];
                        for (int co = 0; co < outC; co++)
                            for (int kh = 0; kh < k; kh++)
                            {
                                int oh = ih * s - p + kh;
                                if (oh < 0 || oh >= outH) continue;
                                for (int kw = 0; kw < k; kw++)
                                {
                                    int ow = iw * s - p + kw;
                                    if (ow < 0 || ow >= outW) continue;
                                    double t = xv * w[((ci * outC + co) * k + kh) * k + kw];
                                    int o = ((b * outC + co) * outH + oh) * outW + ow;
                                    sum[o] += t;
                                    mag[o] += Math.Abs(t);
                                }
                            }
                    }
        return (x, w, sum, mag);
    }
}
