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
}
