using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Kernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The arguments of the fused-attention kernels (<see cref="CudaFusedAttentionKernels"/>), laid out exactly as the
/// device struct <c>FaParams</c>: element strides of strided [B, H, S, D] operands (unit-stride last axis), the softmax
/// scale, the soft-cap (0: none), causal masking as <c>j &lt;= i + CausalOffset</c>, and an optional int32 mask read
/// through its own (broadcast) strides.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct FusedAttentionParams
{
    public int B, H, Sq, Sk;
    public long Qb, Qh, Qr, Kb, Kh, Kr, Vb, Vh, Vr, Ob, Oh, Or;
    public float Scale, Softcap;
    public int Causal, CausalOffset;
    public int HasMask;
    private int _pad;   // the device struct aligns the following long longs to 8
    public long Mb, Mh, Mi, Mj;
}

/// <summary>A backend with the #1141 fused attention kernels.</summary>
internal interface IFusedAttentionBackend
{
    /// <summary>True when the kernels are compiled for this head dimension (query, key and value width).</summary>
    bool SupportsFusedAttention(int headDim);

    /// <summary>o and lse (one log-sum-exp per (b, h, query row); -inf for a row with no allowed key).</summary>
    void FusedAttentionForward(IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer o, IGpuBuffer lse, IGpuBuffer? mask,
        in FusedAttentionParams p, int headDim);

    /// <summary>dQ, dK and dV (written, not accumulated) in the layouts of q, k and v; dO in o's layout. A dQ whose
    /// layout is not dense (q read in place from a packed projection) needs <paramref name="dQSpan"/>: the buffer it lies
    /// in, zeroed whole where the kernels accumulate dQ, its other elements being written by the kernels or left zero.</summary>
    void FusedAttentionBackward(IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer o, IGpuBuffer dO, IGpuBuffer lse,
        IGpuBuffer? mask, IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV, in FusedAttentionParams p, int headDim,
        IGpuBuffer? dQSpan = null);
}

public sealed partial class CudaBackend : IFusedAttentionBackend
{
    private IntPtr _fusedAttentionModule;

    // Query rows per forward block: FA_BR in CudaFusedAttentionKernels (4 threads each).
    private const int FaForwardRows = 32;
    private string? _fusedAttentionCompileError;

    // Compiled with the rest of the backend; a failure leaves the capability off rather than failing startup.
    private void CompileFusedAttentionModule(int device)
    {
        try
        {
            _fusedAttentionModule = CompileKernelModule(device, CudaFusedAttentionKernels.GetSource(), "fused_attention_kernels",
                CudaFusedAttentionKernels.GetKernelNames());
            _fusedAttentionCompileError = null;
        }
        catch (OutOfMemoryException)
        {
            throw;
        }
        catch (Exception ex)
        {
            _fusedAttentionModule = IntPtr.Zero;
            _fusedAttentionCompileError = ex.Message;
            System.Diagnostics.Trace.TraceWarning($"CUDA fused attention kernels are unavailable: {ex.Message}");
        }
    }

    bool IFusedAttentionBackend.SupportsFusedAttention(int headDim)
        => _fusedAttentionModule != IntPtr.Zero && (headDim == 16 || headDim == 32 || headDim == 64);

    private IntPtr FusedAttentionKernel(string name)
    {
        if (_fusedAttentionModule == IntPtr.Zero || !_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException(
                $"CUDA fused attention kernel {name} is unavailable{(_fusedAttentionCompileError is null ? "" : ": " + _fusedAttentionCompileError)}.");
        return kernel;
    }

    private static void CheckFusedAttentionGrid(in FusedAttentionParams p)
    {
        if ((long)p.B * p.H > 65535)
            throw new NotSupportedException($"fused attention: batch * heads = {(long)p.B * p.H} exceeds the 65535 grid rows.");
    }

    unsafe void IFusedAttentionBackend.FusedAttentionForward(IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer o,
        IGpuBuffer lse, IGpuBuffer? mask, in FusedAttentionParams p, int headDim)
    {
        if (p.Sq <= 0 || p.B * p.H <= 0) return;
        CheckFusedAttentionGrid(p);
        IntPtr kernel = FusedAttentionKernel($"fa_forward_d{headDim}");
        using var _ = PushContext();
        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle, po = o.Handle, pl = lse.Handle, pm = mask?.Handle ?? IntPtr.Zero;
        var args0 = p;
        void** args = stackalloc void*[7];
        args[0] = &pq; args[1] = &pk; args[2] = &pv; args[3] = &po; args[4] = &pl; args[5] = &pm; args[6] = &args0;
        LaunchKernel2D(kernel, (uint)((p.Sq + FaForwardRows - 1) / FaForwardRows), (uint)(p.B * p.H), (uint)(FaForwardRows * 4), 1, args);
    }

    unsafe void IFusedAttentionBackend.FusedAttentionBackward(IGpuBuffer q, IGpuBuffer k, IGpuBuffer v, IGpuBuffer o,
        IGpuBuffer dO, IGpuBuffer lse, IGpuBuffer? mask, IGpuBuffer dQ, IGpuBuffer dK, IGpuBuffer dV,
        in FusedAttentionParams p, int headDim, IGpuBuffer? dQSpan)
    {
        if (p.B * p.H <= 0) return;
        CheckFusedAttentionGrid(p);
        // dQ's dense extent is what the atomic path zeroes; a strided dQ must come with the buffer it lies in.
        long dQElements = (long)p.B * p.H * p.Sq * headDim;
        long dQExtent = p.Sq <= 0 ? 0 : (p.B - 1) * p.Qb + (p.H - 1) * p.Qh + (p.Sq - 1) * p.Qr + headDim;
        if (dQSpan is null && dQExtent != dQElements)
            throw new NotSupportedException("fused attention backward: a strided dQ needs the buffer it lies in (dQSpan).");
        IntPtr deltaKernel = FusedAttentionKernel("fa_backward_delta");
        IntPtr kvKernel = FusedAttentionKernel($"fa_backward_kv_d{headDim}");
        IntPtr qKernel = FusedAttentionKernel($"fa_backward_q_d{headDim}");
        int rows = p.B * p.H * p.Sq;
        using var delta = AllocateStreamOrderedScratchBuffer(Math.Max(1, rows));
        using var _ = PushContext();
        var args0 = p;
        IntPtr pq = q.Handle, pk = k.Handle, pv = v.Handle, po = o.Handle, pg = dO.Handle, pl = lse.Handle;
        IntPtr pd = delta.Handle, pm = mask?.Handle ?? IntPtr.Zero;
        IntPtr pdq = dQ.Handle, pdk = dK.Handle, pdv = dV.Handle;
        int d = headDim;
        if (rows > 0)
        {
            // A row per D/4 threads (one float4 each) when D/4 is a power of two up to 32 and every row of O and dO
            // starts 16-byte aligned; else a thread per row.
            int quads = headDim / 4;
            bool aligned = ((po.ToInt64() | pg.ToInt64()) & 15) == 0 && p.Ob % 4 == 0 && p.Oh % 4 == 0 && p.Or % 4 == 0;
            int lanes = headDim % 4 == 0 && quads > 1 && quads <= 32 && (quads & (quads - 1)) == 0 && aligned ? quads : 1;
            long threads = (long)rows * lanes;
            void** dargs = stackalloc void*[6];
            dargs[0] = &pg; dargs[1] = &po; dargs[2] = &pd; dargs[3] = &args0; dargs[4] = &d; dargs[5] = &lanes;
            LaunchKernel(deltaKernel, (uint)((threads + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, dargs);
        }
        // Default: one fused pass with atomic dQ (as torch's default backward). DeterministicReductions keeps the
        // two-pass kernels, whose every sum has a fixed order (torch.use_deterministic_algorithms).
        if (!AiDotNet.Tensors.Helpers.CpuParallelSettings.DeterministicReductions)
        {
            // With no query rows the fused kernel still runs: its query loop is empty and it writes zero dK / dV.
            if (dQSpan is not null)
                MemsetBuffer(dQSpan, 0, dQSpan.SizeInBytes);
            else if (p.Sq > 0)
                MemsetBuffer(dQ, 0, dQElements * sizeof(float));   // dQ covers q's elements exactly
            if (p.Sk <= 0) return;
            IntPtr fusedKernel = FusedAttentionKernel($"fa_backward_fused_d{headDim}");
            void** fargs = stackalloc void*[11];
            fargs[0] = &pq; fargs[1] = &pk; fargs[2] = &pv; fargs[3] = &pg; fargs[4] = &pl; fargs[5] = &pd;
            fargs[6] = &pdq; fargs[7] = &pdk; fargs[8] = &pdv; fargs[9] = &pm; fargs[10] = &args0;
            LaunchKernel2D(fusedKernel, (uint)((p.Sk + 63) / 64), (uint)(p.B * p.H), 128, 1, fargs);   // FA_FKB = 64 keys x 2
            return;
        }
        if (p.Sk > 0)
        {
            void** kargs = stackalloc void*[10];
            kargs[0] = &pq; kargs[1] = &pk; kargs[2] = &pv; kargs[3] = &pg; kargs[4] = &pl; kargs[5] = &pd;
            kargs[6] = &pdk; kargs[7] = &pdv; kargs[8] = &pm; kargs[9] = &args0;
            LaunchKernel2D(kvKernel, (uint)((p.Sk + 127) / 128), (uint)(p.B * p.H), 256, 1, kargs);   // FA_KB = 128 keys x 2
        }
        if (p.Sq > 0)
        {
            void** qargs = stackalloc void*[9];
            qargs[0] = &pq; qargs[1] = &pk; qargs[2] = &pv; qargs[3] = &pg; qargs[4] = &pl; qargs[5] = &pd;
            qargs[6] = &pdq; qargs[7] = &pm; qargs[8] = &args0;
            LaunchKernel2D(qKernel, (uint)((p.Sq + 127) / 128), (uint)(p.B * p.H), 256, 1, qargs);   // FA_QB = 128 rows x 2
        }
    }
}
