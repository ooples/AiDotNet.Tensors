using System;
using System.Buffers;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

// A convolution's spatial workers share this immutable pack until the outer region has joined.
// No parameter cache is retained: scratch is bounded by the kernel, never the waveform or batch.
internal sealed class Conv1DColumnPanelKernel : IDisposable
{
    internal const int Kc = 256;
    internal const int MaxSharedBytes = 16 * 1024 * 1024;
    private const int Mr = 6;
    private const int Nr = 16;
    private readonly float[] _kernel;
    private readonly int _offset, _m, _k, _alignedM, _panelStride;
    private float[]? _packed;

    internal int RequestedPackedFloats { get; }
    internal int WorkerScratchFloats { get; }

    private Conv1DColumnPanelKernel(float[] kernel, int offset, int m, int k, int columns, int requested)
    {
        _kernel = kernel;
        _offset = offset;
        _m = m;
        _k = k;
        _alignedM = m / Mr * Mr;
        _panelStride = checked(_alignedM * Kc);
        RequestedPackedFloats = requested;
        WorkerScratchFloats = checked(columns * Kc);
        _packed = CpuStorageAllocationDiagnostics.RentWorkspace<float>(requested, "workspace.conv1d_shared_kernel");
        try
        {
            for (int pc = 0, panel = 0; pc < k; pc += Kc, panel++)
            {
                int count = Math.Min(Kc, k - pc);
                Avx2Pack.PackA<float>(kernel.AsSpan(offset + pc), k, false,
                    _packed.AsSpan(panel * _panelStride, _alignedM * count), _alignedM, count, Mr);
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    internal static Conv1DColumnPanelKernel? TryCreate(float[] kernel, int offset, int m, int k, int columns)
    {
        // The existing macro kernel consumes exactly the AVX2 6x16 pack. Other ISAs retain the general GEMM.
        var tile = BlasManaged.GetEvolutionTuningTile<float>(PackingMode.ForcePackBoth, BlasProvider.IsDeterministicMode);
        if (tile.Mr != Mr || tile.Nr != Nr || m < Mr || k <= 0 || columns < Nr
            || !MachineKernelGemm.IsFp32MacroAvailable)
            return null;
        long requested = (long)(m / Mr * Mr) * Kc * ((k + (long)Kc - 1) / Kc);
        if (requested <= 0 || requested > MaxSharedBytes / sizeof(float))
            return null;
        return new Conv1DColumnPanelKernel(kernel, offset, m, k, columns, (int)requested);
    }

    internal unsafe void Multiply(ReadOnlySpan<float> b, int n, Span<float> output, int ldc, Span<float> scratch)
    {
        var packed = _packed ?? throw new ObjectDisposedException(nameof(Conv1DColumnPanelKernel));
        int fullN = n / Nr * Nr;
        // The macro kernel accumulates. Clear only the logical output columns, preserving neighbouring panels.
        for (int row = 0; row < _m; row++)
            output.Slice(row * ldc, n).Clear();
        if (fullN > 0)
        {
            fixed (float* aPointer = packed)
            fixed (float* bPointer = scratch)
            fixed (float* cPointer = output)
            {
                for (int pc = 0, panel = 0; pc < _k; pc += Kc, panel++)
                {
                    int count = Math.Min(Kc, _k - pc);
                    Avx2Pack.PackB<float>(b.Slice(pc * n), n, false,
                        scratch.Slice(0, count * fullN), fullN, count, Nr);
                    MachineKernelGemm.RunMacroPanelFp32(aPointer + panel * _panelStride, bPointer, cPointer,
                        ldc, count, fullN / Nr, _alignedM / Mr, checked(count * Mr * sizeof(float)));
                }
            }
        }

        // Keep the canonical streaming arithmetic for edge rows/columns. Every destination is disjoint,
        // and the inner operations stay serial because the convolution already owns the worker budget.
        var options = new BlasOptions<float> { NumThreads = 1, PackingMode = PackingMode.ForcePackBoth };
        if (_alignedM != _m)
            StreamingStrategy.Run<float>(_kernel.AsSpan(_offset + _alignedM * _k), _k, false,
                b, n, false, output.Slice(_alignedM * ldc), ldc, _m - _alignedM, n, _k, in options);
        if (fullN != n)
            StreamingStrategy.Run<float>(_kernel.AsSpan(_offset), _k, false,
                b.Slice(fullN), n, false, output.Slice(fullN), ldc, _alignedM, n - fullN, _k, in options);
    }

    public void Dispose()
    {
        var packed = Interlocked.Exchange(ref _packed, null);
        if (packed is not null)
            ArrayPool<float>.Shared.Return(packed);
    }
}
