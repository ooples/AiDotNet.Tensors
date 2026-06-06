using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Phase-4 measurement (Tensors #555, docs/fp16-activation-storage-design.md): the resident-activation
/// memory win. Builds an L-layer matmul stack and sums the actual activation-BUFFER bytes both ways —
/// FP32 (default) vs FP16 activation storage (AIDOTNET_FP16_ACTIVATIONS path). The FP16 stack stores
/// every layer's matmul activation as Tensor&lt;Half&gt; (2 bytes) instead of FP32 (4 bytes), so the
/// total activation footprint is exactly halved. This guards that the emission scales across ALL layers
/// (not just the first) — the property the ~50% resident win depends on. When run on a CUDA backend it
/// also reports real device VRAM via nvidia-smi as an on-hardware datapoint.
/// </summary>
public class Fp16ActivationMemoryTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private readonly ITestOutputHelper _out;
    public Fp16ActivationMemoryTests(ITestOutputHelper o) => _out = o;

    private static Tensor<float> Rand(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var d = new float[rows * cols];
        for (int i = 0; i < d.Length; i++) d[i] = (float)(rng.NextDouble() * 0.1);
        return new Tensor<float>(d, new[] { rows, cols });
    }

    // Build an L-layer square matmul stack; return the per-layer activation tensors (object-typed).
    private List<object> BuildStack(int B, int d, int L, bool fp16Activations)
    {
        var acts = new List<object>();
        var scope = new LazyTensorScope(null);
        var prevOverride = MixedPrecisionEmit.TestOverrideEnabled;
        AutocastScope? ac = fp16Activations ? new AutocastScope(PrecisionMode.Float16) : null;
        if (fp16Activations) MixedPrecisionEmit.TestOverrideEnabled = true;
        try
        {
            GraphMode.SetCurrent(scope);
            var h = Rand(B, d, 1);
            for (int l = 0; l < L; l++)
            {
                var W = Rand(d, d, 100 + l);
                var y = _engine.TensorMatMul(h, W); // FP16 path emits Half activation
                if (fp16Activations && y.LazySource is CrossTypeLazyNode<Half, float> up)
                    acts.Add(up.Input);   // the Tensor<Half> activation
                else
                    acts.Add(y);          // the Tensor<float> activation
                h = y;
            }
            GraphMode.SetCurrent(null);
            _ = h.ToArray(); // realize the whole stack
        }
        finally
        {
            GraphMode.SetCurrent(null);
            MixedPrecisionEmit.TestOverrideEnabled = prevOverride;
            ac?.Dispose();
        }
        return acts;
    }

    private static long ActivationBytes(List<object> acts)
    {
        long bytes = 0;
        foreach (var a in acts)
        {
            switch (a)
            {
                case Tensor<float> tf: bytes += (long)tf.Length * sizeof(float); break;
                case Tensor<Half> th: bytes += (long)th.Length * 2; break;
            }
        }
        return bytes;
    }

    private long? GpuUsedMiB()
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=memory.used --format=csv,noheader,nounits")
            { RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true };
            using var p = Process.Start(psi);
            if (p is null) return null;
            string outp = p.StandardOutput.ReadLine() ?? "";
            p.WaitForExit(5000);
            return long.TryParse(outp.Trim(), out var v) ? v : (long?)null;
        }
        catch { return null; }
    }

    [Fact]
    public void Fp16ActivationStorage_Halves_ActivationFootprint_AcrossAllLayers()
    {
        const int B = 256, d = 512, L = 8;

        var fp32Acts = BuildStack(B, d, L, fp16Activations: false);
        long? vramFp32 = GpuUsedMiB();
        long fp32Bytes = ActivationBytes(fp32Acts);

        var fp16Acts = BuildStack(B, d, L, fp16Activations: true);
        long? vramFp16 = GpuUsedMiB();
        long fp16Bytes = ActivationBytes(fp16Acts);

        _out.WriteLine($"engine={_engine.GetType().Name}");
        _out.WriteLine($"stack B={B} d={d} L={L}");
        _out.WriteLine($"FP32 activation bytes = {fp32Bytes:N0} ({fp32Bytes / (1024.0 * 1024):F2} MB)");
        _out.WriteLine($"FP16 activation bytes = {fp16Bytes:N0} ({fp16Bytes / (1024.0 * 1024):F2} MB)");
        _out.WriteLine($"ratio FP16/FP32 = {(double)fp16Bytes / fp32Bytes:F3}");
        if (vramFp32 is not null && vramFp16 is not null)
            _out.WriteLine($"device VRAM used: FP32-pass={vramFp32} MiB, FP16-pass={vramFp16} MiB");

        // Every layer's activation must be FP16 in the FP16 path (the win scales across the whole model).
        foreach (var a in fp16Acts) Assert.IsType<Tensor<Half>>(a);
        // Exactly half the activation footprint.
        Assert.Equal(L, fp16Acts.Count);
        Assert.Equal(fp32Bytes / 2, fp16Bytes);
    }
}
