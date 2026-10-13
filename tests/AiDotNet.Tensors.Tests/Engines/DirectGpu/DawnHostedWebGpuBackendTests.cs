#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.WebGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// One real <see cref="WebGpuBackend"/> for the test process, executing on Dawn through <see cref="DawnWebGpuHost"/>.
/// The backend and its device are process-wide singletons underneath, so this is one too. It is not thread-safe:
/// every test class that uses it belongs to the <c>DirectGpuSerial</c> collection.
/// </summary>
internal static class DawnHostedWebGpu
{
    private static readonly object Gate = new();
    private static bool s_tried;
    private static WebGpuBackend? s_backend;
    private static DawnWebGpuHost? s_host;
    private static string s_reason = "";

    public static DawnWebGpuHost? Host { get { lock (Gate) return s_host; } }

    /// <summary>The backend, or a skip when Dawn is not set up (AIDOTNET_WEBGPU_NODE_DIR) or has no adapter.</summary>
    public static WebGpuBackend BackendOrSkip()
    {
        lock (Gate)
        {
            if (!s_tried)
            {
                s_tried = true;
                Start();
            }
            Skip.If(s_backend is null, s_reason);
            return s_backend!;
        }
    }

    private static void Start()
    {
        var host = DawnWebGpuHost.TryCreate();
        if (host is null)
        {
            s_reason = "Set AIDOTNET_WEBGPU_NODE_DIR to a directory with node_modules/webgpu (npm install webgpu).";
            return;
        }
        WebGpuNativeBindings.Host = host;
        var backend = new WebGpuBackend();
        if (!backend.InitializeAsync().GetAwaiter().GetResult())
        {
            WebGpuNativeBindings.Host = null;
            host.Dispose();
            s_reason = "Dawn has no WebGPU adapter on this machine.";
            return;
        }
        s_host = host;
        s_backend = backend;
        AppDomain.CurrentDomain.ProcessExit += (_, _) => host.Dispose();
    }
}

/// <summary>
/// #1117 groundwork: the real WebGPU backend running on Dawn. These are the first tests that execute
/// <see cref="WebGpuBackend"/>'s own code and the shipped <c>webgpu-tensors.js</c> at all; until now every WebGPU
/// test either skipped ("WebGPU is not available") or ran a kernel's WGSL through a separate launcher.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DawnHostedWebGpuBackendTests
{
    [SkippableFact]
    public void Initializes_ThroughTheShippedModule_AndReportsTheDevice()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        Assert.True(backend.IsAvailable);
        Assert.False(string.IsNullOrWhiteSpace(WebGpuDevice.Instance.AdapterInfo));
        // The limits come from the module's getDeviceLimitsJson export. The binding used to import a name the
        // module does not export, so in a browser this call could not have resolved.
        var limits = backend.Limits;
        Assert.NotNull(limits);
        Assert.True(limits!.MaxComputeInvocationsPerWorkgroup >= 256, $"MaxComputeInvocationsPerWorkgroup = {limits.MaxComputeInvocationsPerWorkgroup}");
        Assert.True(limits.MaxStorageBufferBindingSize >= 128 * 1024 * 1024, $"MaxStorageBufferBindingSize = {limits.MaxStorageBufferBindingSize}");
        var calls = DawnHostedWebGpu.Host!.Calls;
        Assert.True(calls.ContainsKey("initializeWebGpu"));
        Assert.True(calls.ContainsKey("getDeviceLimitsJson"));
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(1000)]
    public void FloatBuffers_RoundTrip_Exactly_IncludingNonFiniteValues(int n)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        var rng = new Random(3 + n);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)((rng.NextDouble() - 0.5) * 1e6);
        float[] special = { float.NaN, float.PositiveInfinity, float.NegativeInfinity, -0f, float.Epsilon, float.MaxValue, 1e-30f, 0.1f };
        for (int i = 0; i < Math.Min(n, special.Length); i++) data[n - 1 - i] = special[i];

        using var buffer = backend.AllocateBuffer(data);
        float[] back = backend.DownloadBuffer(buffer);
        Assert.Equal(n, back.Length);
        for (int i = 0; i < n; i++)
            Assert.True(BitConverter.SingleToInt32Bits(data[i]) == BitConverter.SingleToInt32Bits(back[i])
                    || (float.IsNaN(data[i]) && float.IsNaN(back[i])),
                $"[{i}] wrote {data[i]:R}, read {back[i]:R}");
    }

    [SkippableFact]
    public void ASmallRead_AfterALargeOne_ReturnsOnlyWhatWasAskedFor()
    {
        // The module pools its staging buffers and hands back any that is large enough, so the read after a large one
        // maps a buffer bigger than the request.
        var backend = DawnHostedWebGpu.BackendOrSkip();
        var large = new float[4096];
        for (int i = 0; i < large.Length; i++) large[i] = i;
        float[] small = { 10f, 20f, 30f };
        using var dLarge = backend.AllocateBuffer(large);
        using var dSmall = backend.AllocateBuffer(small);
        Assert.Equal(large, backend.DownloadBuffer(dLarge));
        Assert.Equal(small, backend.DownloadBuffer(dSmall));
        var destination = new float[3];
        backend.DownloadBuffer(dSmall, destination);
        Assert.Equal(small, destination);
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(6)]
    [InlineData(1001)]
    public void ByteBuffers_RoundTrip_AtLengthsThatAreNotWholeWords(int n)
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        var data = new byte[n];
        new Random(7 + n).NextBytes(data);
        data[0] = 0xFF;
        using var buffer = backend.AllocateByteBuffer(n);
        backend.UploadByteBuffer(buffer, data);
        Assert.Equal(data, backend.DownloadByteBuffer(buffer, n));
    }

    [SkippableFact]
    public void ElementwiseAndRowKernels_MatchTheCpu()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        const int rows = 7, cols = 33, n = rows * cols;
        var rng = new Random(11);
        float[] a = new float[n], b = new float[n];
        for (int i = 0; i < n; i++) { a[i] = (float)(rng.NextDouble() * 4 - 2); b[i] = (float)(rng.NextDouble() * 4 - 2); }
        using var da = backend.AllocateBuffer(a);
        using var db = backend.AllocateBuffer(b);
        using var dc = backend.AllocateBuffer(n);

        backend.Fill(dc, float.NaN, n);
        backend.Add(da, db, dc, n);
        float[] sum = backend.DownloadBuffer(dc);
        for (int i = 0; i < n; i++) Assert.Equal(a[i] + b[i], sum[i]);

        backend.Fill(dc, float.NaN, n);
        backend.Softmax(da, dc, rows, cols);
        float[] soft = backend.DownloadBuffer(dc);
        for (int r = 0; r < rows; r++)
        {
            double max = double.NegativeInfinity, total = 0;
            for (int c = 0; c < cols; c++) max = Math.Max(max, a[r * cols + c]);
            for (int c = 0; c < cols; c++) total += Math.Exp(a[r * cols + c] - max);
            for (int c = 0; c < cols; c++)
            {
                double expected = Math.Exp(a[r * cols + c] - max) / total;
                Assert.True(Math.Abs(expected - soft[r * cols + c]) <= 2e-6, $"softmax[{r},{c}]: expected {expected}, got {soft[r * cols + c]}");
            }
        }
    }
}
#endif
