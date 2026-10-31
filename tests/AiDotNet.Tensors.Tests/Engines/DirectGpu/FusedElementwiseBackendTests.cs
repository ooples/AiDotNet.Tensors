using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The members that used to be compositions of other backend members (TENS0008), each against a host reference: AddRelu,
/// AddSigmoid and AddGelu; Fma with its output aliasing the addend -- the in-place accumulate the old
/// Multiply-into-D-then-Add composition got wrong (2AB instead of AB + C); the split-complex element-wise family; the
/// batched dot product; and the CSR SpMM with bias. Runs on every backend this
/// machine has; each one absent is skipped by name.
/// </summary>
public abstract class FusedElementwiseBackendTestsBase
{
    private readonly string _name;

    protected FusedElementwiseBackendTestsBase(string name) => _name = name;
    private const int Length = 4099;          // a float4 body and a scalar tail
    private const double RelativeTolerance = 2e-6;
    private const double AbsoluteTolerance = 1e-6;

    private static float[] Values(int salt) => Values(salt, Length);

    private static float[] Values(int salt, int length)
    {
        var rng = RandomHelper.CreateSeededRandom(1638 + salt);
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 8 - 4);
        return data;
    }

    private static void AssertClose(double expected, float actual, int index, string op)
        => Assert.True(Math.Abs(expected - actual) <= AbsoluteTolerance + RelativeTolerance * Math.Abs(expected),
            $"{op}[{index}]: expected {expected}, got {actual}");

    [SkippableFact]
    public void AddActivations_EqualTheActivatedSum()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var a = Values(1);
        var b = Values(2);
        using var bufA = backend.AllocateBuffer(a);
        using var bufB = backend.AllocateBuffer(b);
        using var bufC = backend.AllocateBuffer(Length);

        var cases = new (string Op, Action Run, Func<double, double> Reference)[]
        {
            ("AddRelu", () => backend.AddRelu(bufA, bufB, bufC, Length), x => x > 0 ? x : 0),
            ("AddSigmoid", () => backend.AddSigmoid(bufA, bufB, bufC, Length), x => 1 / (1 + Math.Exp(-x))),
            ("AddGelu", () => backend.AddGelu(bufA, bufB, bufC, Length),
                x => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)))),
        };
        foreach (var (op, run, reference) in cases)
        {
            run();
            var c = backend.DownloadBuffer(bufC);
            for (int i = 0; i < Length; i++) AssertClose(reference((double)(a[i] + b[i])), c[i], i, op);
        }
    }

    [SkippableFact]
    public void Fma_IntoItsAddend_IsAProductPlusTheAddend()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var a = Values(3);
        var b = Values(4);
        var c = Values(5);
        using var bufA = backend.AllocateBuffer(a);
        using var bufB = backend.AllocateBuffer(b);
        using var bufC = backend.AllocateBuffer(c);

        backend.Fma(bufA, bufB, bufC, bufC, Length);   // D is C

        var d = backend.DownloadBuffer(bufC);
        for (int i = 0; i < Length; i++) AssertClose((double)a[i] * b[i] + c[i], d[i], i, "Fma");
    }

    [SkippableFact]
    public void SplitComplexFamily_EqualsTheComplexArithmetic()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        var aR = Values(6); var aI = Values(7); var bR = Values(8); var bI = Values(9);
        using var bufAR = backend.AllocateBuffer(aR);
        using var bufAI = backend.AllocateBuffer(aI);
        using var bufBR = backend.AllocateBuffer(bR);
        using var bufBI = backend.AllocateBuffer(bI);
        using var oR = backend.AllocateBuffer(Length);
        using var oI = backend.AllocateBuffer(Length);

        void Check(string op, Func<int, (double Re, double Im)> reference)
        {
            var re = backend.DownloadBuffer(oR);
            var im = backend.DownloadBuffer(oI);
            for (int i = 0; i < Length; i++)
            {
                var (er, ei) = reference(i);
                AssertClose(er, re[i], i, op + ".re");
                AssertClose(ei, im[i], i, op + ".im");
            }
        }

        backend.SplitComplexMultiply(bufAR, bufAI, bufBR, bufBI, oR, oI, Length);
        Check("Multiply", i => ((double)aR[i] * bR[i] - (double)aI[i] * bI[i], (double)aR[i] * bI[i] + (double)aI[i] * bR[i]));
        backend.SplitComplexAdd(bufAR, bufAI, bufBR, bufBI, oR, oI, Length);
        Check("Add", i => ((double)aR[i] + bR[i], (double)aI[i] + bI[i]));
        backend.SplitComplexCrossSpectral(bufAR, bufAI, bufBR, bufBI, oR, oI, Length);
        Check("CrossSpectral", i => ((double)aR[i] * bR[i] + (double)aI[i] * bI[i], (double)aI[i] * bR[i] - (double)aR[i] * bI[i]));
        backend.SplitComplexConjugate(bufAR, bufAI, oR, oI, Length);
        Check("Conjugate", i => (aR[i], -aI[i]));
        backend.SplitComplexScale(bufAR, bufAI, oR, oI, -1.75f, Length);
        Check("Scale", i => (aR[i] * -1.75, aI[i] * -1.75));

        backend.SplitComplexMagnitudeSquared(bufAR, bufAI, oR, Length);
        var mag = backend.DownloadBuffer(oR);
        for (int i = 0; i < Length; i++) AssertClose((double)aR[i] * aR[i] + (double)aI[i] * aI[i], mag[i], i, "MagnitudeSquared");
    }

    [SkippableFact]
    public void BatchedDotProduct_EqualsEachRowsDot()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        const int rows = 5, width = 1027;   // more columns than a 256-thread group, and not a multiple of it
        var a = Values(10, rows * width); var b = Values(11, rows * width);
        using var bufA = backend.AllocateBuffer(a);
        using var bufB = backend.AllocateBuffer(b);
        using var result = backend.AllocateBuffer(rows);

        backend.BatchedDotProduct(bufA, bufB, result, rows, width);

        var r = backend.DownloadBuffer(result);
        for (int row = 0; row < rows; row++)
        {
            double expected = 0;
            for (int i = 0; i < width; i++) expected += (double)a[row * width + i] * b[row * width + i];
            Assert.True(Math.Abs(expected - r[row]) <= 1e-3 * (1 + Math.Abs(expected)), $"row {row}: expected {expected}, got {r[row]}");
        }
    }

    [SkippableFact]
    public void CsrSpMMBias_AddsTheColumnBiasToTheProduct()
    {
        using var lease = Create(_name, out var backend);
        Skip.If(backend is null, _name + " is not available on this machine.");
        // A [3 x 4] in CSR: row 0 {0: 2, 3: -1}, row 1 empty, row 2 {1: 0.5, 2: 4}; B [4 x 5]; bias [5].
        const int m = 3, k = 4, n = 5;
        var values = new[] { 2f, -1f, 0.5f, 4f };
        var columns = new[] { 0, 3, 1, 2 };
        var rowPointers = new[] { 0, 2, 2, 4 };
        var dense = Values(12);
        var bias = new[] { 0.25f, -0.5f, 1f, 2f, -3f };
        using var bufValues = backend.AllocateBuffer(values);
        using var bufColumns = backend.AllocateIntBuffer(columns);
        using var bufRows = backend.AllocateIntBuffer(rowPointers);
        using var bufDense = backend.AllocateBuffer(dense);
        using var bufBias = backend.AllocateBuffer(bias);
        using var output = backend.AllocateBuffer(m * n);

        backend.CsrSpMMBias(bufValues, bufColumns, bufRows, bufDense, bufBias, output, m, k, n, values.Length);

        var c = backend.DownloadBuffer(output);
        for (int row = 0; row < m; row++)
            for (int col = 0; col < n; col++)
            {
                double expected = bias[col];
                for (int e = rowPointers[row]; e < rowPointers[row + 1]; e++) expected += (double)values[e] * dense[columns[e] * n + col];
                AssertClose(expected, c[row * n + col], row * n + col, "CsrSpMMBias");
            }
    }

    // The backend, or null when unavailable; the returned disposable disposes a backend the test created.
    private static IDisposable? Create(string name, out IDirectGpuBackend? backend)
    {
        backend = null;
        try
        {
            IDirectGpuBackend candidate = name switch
            {
                "cuda" => new CudaBackend(),
                "hip" => new HipBackend(),
                "opencl" => new OpenClBackend(),
                "vulkan" => VulkanBackend.Instance,   // the process-wide instance; never disposed here
                "metal" => new MetalBackend(),
#if NET7_0_OR_GREATER
                "webgpu" => DawnHostedWebGpu.BackendOrNull()
                            ?? throw new PlatformNotSupportedException("Dawn is not set up (AIDOTNET_WEBGPU_NODE_DIR)."),
#endif
                _ => throw new PlatformNotSupportedException(name),
            };
            if (candidate is VulkanBackend vulkan && !vulkan.IsAvailable) vulkan.Initialize();
            bool shared = name is "vulkan" or "webgpu";
            if (!candidate.IsAvailable)
            {
                if (!shared) (candidate as IDisposable)?.Dispose();
                return null;
            }
            backend = candidate;
            return shared ? null : candidate as IDisposable;
        }
        catch (Exception ex) when (ex is not Xunit.Sdk.XunitException)
        {
            return null;   // no driver for this backend on this machine
        }
    }
}

// One class per backend, so each backend's CI lane selects its own (the Metal lane runs FullyQualifiedName~Metal).
[Collection("DirectGpuSerial")]
public sealed class CudaFusedElementwiseTests : FusedElementwiseBackendTestsBase { public CudaFusedElementwiseTests() : base("cuda") { } }

[Collection("DirectGpuSerial")]
public sealed class HipFusedElementwiseTests : FusedElementwiseBackendTestsBase { public HipFusedElementwiseTests() : base("hip") { } }

[Collection("DirectGpuSerial")]
public sealed class OpenClFusedElementwiseTests : FusedElementwiseBackendTestsBase { public OpenClFusedElementwiseTests() : base("opencl") { } }

[Collection("DirectGpuSerial")]
public sealed class VulkanFusedElementwiseTests : FusedElementwiseBackendTestsBase { public VulkanFusedElementwiseTests() : base("vulkan") { } }

[Collection("DirectGpuSerial")]
public sealed class MetalFusedElementwiseTests : FusedElementwiseBackendTestsBase { public MetalFusedElementwiseTests() : base("metal") { } }

[Collection("DirectGpuSerial")]
public sealed class WebGpuFusedElementwiseTests : FusedElementwiseBackendTestsBase { public WebGpuFusedElementwiseTests() : base("webgpu") { } }
