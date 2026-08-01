using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Isolates which stage of the N-D GPU FFT diverges from the CPU.
/// </summary>
/// <remarks>
/// <para>
/// The evidence that motivates this: NativeComplexFFT2D, NativeComplexFFTComplex, NativeComplexIFFT
/// and NativeComplexIFFT2DReal all PASS the GPU/CPU parity sweep, while NativeComplexFFTND and
/// NativeComplexIFFTNDReal fail on shape [128,128] — the same shape the 2-D variant handles
/// correctly. Same data, same maths, different code path.
/// </para>
/// <para>
/// The N-D path is the only one that calls <c>backend.Permute</c>: ExecuteResidentFftAxes moves a
/// non-final axis to the end, runs the batched transform, then permutes back. The 2-D path uses
/// dedicated row/column kernels and never permutes. So the discriminating experiment is the axis
/// argument, not the shape — transforming ONLY the last axis skips the permute entirely.
/// </para>
/// <para>
/// These are reported rather than asserted, because the point is to attribute the failure, not to
/// add another red test to a suite that already has five.
/// </para>
/// </remarks>
[Collection("DirectGpuSerial")]
public class FftNdAxisIsolationProbe : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _cpu = new();
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _available;

    public FftNdAxisIsolationProbe(ITestOutputHelper o)
    {
        _out = o;
        try { _gpu = new DirectGpuTensorEngine(); _available = _gpu.IsGpuAvailable; }
        catch { _available = false; }
    }

    public void Dispose() { _gpu?.Dispose(); GC.SuppressFinalize(this); }

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private double MaxAbsErr(Tensor<Complex<float>> a, Tensor<Complex<float>> b)
    {
        double worst = 0;
        for (int i = 0; i < a.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(a[i].Real - b[i].Real));
            worst = Math.Max(worst, Math.Abs(a[i].Imaginary - b[i].Imaginary));
        }
        return worst;
    }

    [SkippableFact]
    public void WhichAxisArgumentDiverges()
    {
        Skip.If(!_available, "GPU backend not available");

        foreach (var shape in new[] { new[] { 8, 8 }, new[] { 16, 32 }, new[] { 128, 128 } })
        {
            var x = Rand(shape, 5);
            _out.WriteLine($"shape [{string.Join(",", shape)}]");

            foreach (var (label, axes) in new (string, int[])[]
                     {
                         ("axes=[1]  last axis only, NO permute", [1]),
                         ("axes=[0]  first axis only, PERMUTES ", [0]),
                         ("axes=[0,1] both                     ", [0, 1]),
                     })
            {
                double err;
                try
                {
                    var cpu = _cpu.NativeComplexFFTND(x, axes);
                    var gpu = _gpu!.NativeComplexFFTND(x, axes);
                    err = MaxAbsErr(cpu, gpu);
                }
                catch (Exception ex)
                {
                    _out.WriteLine($"    {label} -> threw {ex.GetType().Name}: {ex.Message}");
                    continue;
                }

                _out.WriteLine($"    {label} -> max_abs_err {err:E3}{(err > 1e-2 ? "   <== DIVERGES" : "")}");
            }
        }
    }

    /// <summary>
    /// Sweeps the batch count with the transform length held fixed.
    /// </summary>
    /// <remarks>
    /// BatchedFFT is the only FFT entry point the N-D path uses; FFT2D drives its own row/column
    /// kernels and the 1-D entry points drive the single-signal kernel, which is why all of those
    /// pass. Holding n fixed at 8 and walking batch from 1 upward separates "the transform is
    /// wrong" from "the batching is wrong" — the batched kernels index by a flattened
    /// get_global_id(0), so a dispatch that does not span batch*n leaves later signals untouched.
    /// </remarks>
    [SkippableFact]
    public void WhichBatchCountDiverges()
    {
        Skip.If(!_available, "GPU backend not available");

        _out.WriteLine("transform length n=8 held fixed, batch swept");
        foreach (int batch in new[] { 1, 2, 3, 4, 8, 16 })
        {
            // Rank-1 for batch 1 so the N-D path sees exactly one signal.
            int[] shape = batch == 1 ? [8] : [batch, 8];
            int[] axes = batch == 1 ? [0] : [1];

            var x = Rand(shape, 9);
            try
            {
                var cpu = _cpu.NativeComplexFFTND(x, axes);
                var gpu = _gpu!.NativeComplexFFTND(x, axes);
                double err = MaxAbsErr(cpu, gpu);
                _out.WriteLine($"  batch={batch,-3} -> max_abs_err {err:E3}{(err > 1e-2 ? "   <== DIVERGES" : "   ok")}");
            }
            catch (Exception ex)
            {
                _out.WriteLine($"  batch={batch,-3} -> threw {ex.GetType().Name}: {ex.Message}");
            }
        }

        // BatchedFFT derives the bit-reversal width as (int)MathHelper.Log2(n), which is
        // Math.Log(x)/Math.Log(2). If that lands a hair under an integer, the cast truncates and the
        // bit-reversal permutes the wrong number of bits — leaving n=2 (no swaps needed) correct and
        // everything larger scrambled. The butterfly stage loop does not use it, so this would hit
        // bit-reversal alone.
        _out.WriteLine("bit-reversal width: (int)(Math.Log(n)/Math.Log(2)) vs exact log2");
        foreach (int n in new[] { 2, 4, 8, 16, 32, 64, 128 })
        {
            double raw = Math.Log(n) / Math.Log(2);
            int truncated = (int)raw;
            int exact = System.Numerics.BitOperations.Log2((uint)n);
            _out.WriteLine($"  n={n,-4} raw={raw:R}  (int)={truncated}  exact={exact}"
                + (truncated != exact ? "   <== TRUNCATES LOW" : ""));
        }

        // Is the GPU output simply a PERMUTATION of the correct spectrum? n=2 passing is exactly the
        // case where bit-reversal is the identity, which is the signature of the reversal being
        // applied on the wrong side of the butterfly (decimation-in-time reversal paired with a
        // decimation-in-frequency butterfly, or the reverse).
        _out.WriteLine("is GPU output a bit-reversed permutation of CPU output?");
        foreach (int n in new[] { 4, 8, 16 })
        {
            var x1 = Rand([1, n], 17);
            var cpu1 = _cpu.NativeComplexFFTND(x1, [1]);
            var gpu1 = _gpu!.NativeComplexFFTND(x1, [1]);

            int bits = System.Numerics.BitOperations.Log2((uint)n);
            static int Rev(int v, int bits)
            {
                int r = 0;
                for (int k = 0; k < bits; k++) { r = (r << 1) | (v & 1); v >>= 1; }
                return r;
            }

            double direct = 0, permuted = 0;
            for (int k = 0; k < n; k++)
            {
                direct = Math.Max(direct, Math.Abs(cpu1[k].Real - gpu1[k].Real));
                permuted = Math.Max(permuted, Math.Abs(cpu1[Rev(k, bits)].Real - gpu1[k].Real));
            }
            _out.WriteLine($"  n={n,-3} direct={direct:E3}  bitReversed={permuted:E3}"
                + (permuted < 1e-3 ? "   <== GPU IS BIT-REVERSED" : ""));
        }

        // Which side is actually wrong? The parity test assumes the CPU is truth. Compare BOTH
        // against a direct O(n^2) DFT, which has no algorithmic structure to get wrong.
        _out.WriteLine("both engines vs a direct O(n^2) DFT reference");
        foreach (int n in new[] { 4, 8, 16 })
        {
            var x1 = Rand([1, n], 19);
            var cpu1 = _cpu.NativeComplexFFTND(x1, [1]);
            var gpu1 = _gpu!.NativeComplexFFTND(x1, [1]);

            double cpuErr = 0, gpuErr = 0;
            for (int k = 0; k < n; k++)
            {
                double re = 0, im = 0;
                for (int t = 0; t < n; t++)
                {
                    double ang = -2.0 * Math.PI * k * t / n;
                    re += x1[0, t] * Math.Cos(ang);
                    im += x1[0, t] * Math.Sin(ang);
                }
                cpuErr = Math.Max(cpuErr, Math.Max(Math.Abs(cpu1[k].Real - re), Math.Abs(cpu1[k].Imaginary - im)));
                gpuErr = Math.Max(gpuErr, Math.Max(Math.Abs(gpu1[k].Real - re), Math.Abs(gpu1[k].Imaginary - im)));
            }
            _out.WriteLine($"  n={n,-3} cpu_vs_dft={cpuErr:E3}  gpu_vs_dft={gpuErr:E3}"
                + (cpuErr > 1e-3 && gpuErr < 1e-3 ? "   <== CPU IS THE WRONG ONE" : "")
                + (gpuErr > 1e-3 && cpuErr < 1e-3 ? "   <== GPU IS THE WRONG ONE" : "")
                + (gpuErr > 1e-3 && cpuErr > 1e-3 ? "   <== BOTH WRONG" : ""));
        }

        _out.WriteLine("batch=4 held fixed, transform length swept");
        foreach (int n in new[] { 2, 4, 8, 16, 32 })
        {
            var x = Rand([4, n], 11);
            try
            {
                var cpu = _cpu.NativeComplexFFTND(x, [1]);
                var gpu = _gpu!.NativeComplexFFTND(x, [1]);
                double err = MaxAbsErr(cpu, gpu);
                _out.WriteLine($"  n={n,-3} -> max_abs_err {err:E3}{(err > 1e-2 ? "   <== DIVERGES" : "   ok")}");
            }
            catch (Exception ex)
            {
                _out.WriteLine($"  n={n,-3} -> threw {ex.GetType().Name}: {ex.Message}");
            }
        }
    }
}
