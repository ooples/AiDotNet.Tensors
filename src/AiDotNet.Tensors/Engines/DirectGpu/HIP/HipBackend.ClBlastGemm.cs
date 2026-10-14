using System;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// CLBlast's indirect GEMM on HIP (#1266): the kernel OpenCL runs on this GPU, built from the same source with the
// configuration CLBlast tuned for the device, packed and launched the same way.
public sealed partial class HipBackend
{
    private readonly object _clblastLock = new object();
    private IntPtr _clblastIndirectModule;
    private IntPtr _clblastPackingModule;
    private GemmConfig _clblastIndirectConfig;
    private ClBlastPadParameters _clblastPadParams;
    private ClBlastPadTransposeParameters _clblastPadTransposeParams;
    private int _clblastMinIndirectSize;
    private bool _clblastIndirectTried;
    private bool _clblastIndirectReady;
    private long _clblastIndirectGemmRuns;
    // Grow-only packing buffers: per-call allocate / free cost more than the GEMM itself at 448^3.
    private IGpuBuffer? _clblastScratchA, _clblastScratchB, _clblastScratchC;

    /// <summary>The CLBlast database key for this device: the one OpenCL uses for the same GPU.</summary>
    internal ClBlastDeviceInfo ClBlastDevice => ClBlastDeviceInfo.ForAmdGpu(_deviceProps.GcnArchName ?? _architectureTarget, DeviceName);

    /// <summary>How many GEMMs the indirect path has run on this backend (lets tests prove the route was taken).</summary>
    internal long ClBlastIndirectGemmRuns => Interlocked.Read(ref _clblastIndirectGemmRuns);

    /// <summary>Whether Gemm goes to hipBLAS first (and so never reaches the CLBlast kernels).</summary>
    internal bool IsHipBlasAvailable => _hipblasAvailable && _hipblasHandle != IntPtr.Zero;

    /// <summary>
    /// Builds the indirect GEMM (clblast_xgemm) and its packing kernels once, with the database configuration unless
    /// <paramref name="config"/> is given.
    /// </summary>
    internal bool EnsureClBlastIndirectGemm(GemmConfig? config = null)
    {
        lock (_clblastLock)
        {
            if (config is null && _clblastIndirectTried) return _clblastIndirectReady;
            _clblastIndirectTried = true;
            _clblastIndirectReady = false;
            var device = ClBlastDevice;
            GemmConfig c;
            if (config is GemmConfig given) c = given;
            else if (!ClBlastXgemmDatabase.TryGetConfig(device, out c)) return false;
            if (!IsAvailable || !HipClBlastGemmKernels.Supports(c)) return false;

            if (_clblastPackingModule == IntPtr.Zero)
            {
                _clblastPadParams = ClBlastPadDatabase.GetParameters(device);
                _clblastPadTransposeParams = ClBlastPadTransposeDatabase.GetParameters(device);
                var packing = HipClBlastGemmKernels.BuildPackingSource(_clblastPadParams, _clblastPadTransposeParams,
                    ClBlastCopyDatabase.GetParameters(device), ClBlastTransposeDatabase.GetParameters(device));
                if (CompileKernelModule(packing, "clblast_packing", ref _clblastPackingModule,
                    HipClBlastGemmKernels.PackingKernelNames, useFastMath: false) is not null) return false;
            }
            if (_clblastIndirectModule != IntPtr.Zero)
            {
                HipNativeBindings.hipModuleUnload(_clblastIndirectModule);
                _clblastIndirectModule = IntPtr.Zero;
            }
            if (CompileKernelModule(HipClBlastGemmKernels.BuildIndirectSource(c), "clblast_indirect", ref _clblastIndirectModule,
                new[] { "clblast_xgemm" }, useFastMath: false) is not null) return false;
            _clblastIndirectConfig = c;
            _clblastMinIndirectSize = ClBlastGemmRoutineDatabase.GetXgemmMinIndirectSize(device);
            _clblastIndirectReady = _kernelCache.ContainsKey("clblast_xgemm") && _kernelCache.ContainsKey("CopyPadMatrix") &&
                _kernelCache.ContainsKey("CopyMatrix") && _kernelCache.ContainsKey("TransposePadMatrix");
            return _clblastIndirectReady;
        }
    }

    /// <summary>
    /// The shapes Gemm sends through the indirect path: M or N at least CLBlast's MinIndirectSize for the device (where
    /// OpenCL switches to it), and tile padding adding at most half again the work. Measured on an RX 5500 XT, a side
    /// padded from 8 to the 64-wide tile (8x512x512: 22 vs 48 GF) or from 16 (16x1024x1024: 125 vs 218 GF) loses to the
    /// HIP kernels, while unpadded skinny shapes win (4096x64x4096: 727 vs 406 GF).
    /// </summary>
    private bool ShouldUseClBlastIndirectGemm(int M, int N, int K)
    {
        if (M <= 1 || N <= 1 || !EnsureClBlastIndirectGemm() || _clblastMinIndirectSize <= 0) return false;
        if (M < _clblastMinIndirectSize && N < _clblastMinIndirectSize) return false;
        var c = _clblastIndirectConfig;
        double padded = (double)CeilTo(N, c.TileM) * CeilTo(M, c.TileN) * CeilTo(K, c.TileK * (c.KReg > 0 ? c.KReg : 1));
        return padded <= 1.5 * M * N * K;
    }

    /// <summary>
    /// Row-major C = alpha * A[M x K] * B[K x N] + beta * C through CLBlast's indirect path, as OpenClBackend runs it
    /// (TryExecuteClBlastBaselineGemm's row-major swap): B is padded into the kernel's A slot, A is transposed into its
    /// B slot, the GEMM runs on tile-multiple sizes, and C is copied out of a padded buffer when it needs one.
    /// </summary>
    internal unsafe bool TryExecuteClBlastIndirectGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha = 1f, float beta = 0f)
    {
        if (M <= 0 || N <= 0 || K <= 0 || !EnsureClBlastIndirectGemm()) return false;
        var c = _clblastIndirectConfig;
        int kUnit = c.TileK * (c.KReg > 0 ? c.KReg : 1);
        int mCeiled = CeilTo(N, c.TileM), nCeiled = CeilTo(M, c.TileN), kCeiled = CeilTo(K, kUnit);
        long aSize = (long)mCeiled * kCeiled, bSize = (long)nCeiled * kCeiled, cSize = (long)mCeiled * nCeiled;
        if (aSize > int.MaxValue || bSize > int.MaxValue || cSize > int.MaxValue) return false;
        bool aNeedsPad = N != mCeiled || K != kCeiled;
        bool cNeedsPad = N != mCeiled || M != nCeiled;

        // One stream and one set of scratch buffers: the lock keeps calls from interleaving their packing launches.
        lock (_clblastLock)
        {
            IGpuBuffer aBuf = B, cBuf = C;
            if (aNeedsPad)
            {
                var aTemp = ClBlastScratch(ref _clblastScratchA, aSize);
                ClBlastPack("CopyPadMatrix", B, aTemp, N, K, N, mCeiled, kCeiled, mCeiled);
                aBuf = aTemp;
            }
            // The kernel reads its B operand as bgm[k * ld + n], so A goes in physically transposed, padded or not.
            var bTemp = ClBlastScratch(ref _clblastScratchB, bSize);
            ClBlastPack("TransposePadMatrix", A, bTemp, K, M, K, nCeiled, kCeiled, nCeiled);
            if (cNeedsPad)
            {
                var cTemp = ClBlastScratch(ref _clblastScratchC, cSize);
                if (beta != 0f) ClBlastPack("CopyPadMatrix", C, cTemp, N, M, N, mCeiled, nCeiled, mCeiled);
                cBuf = cTemp;
            }

            IntPtr kernel = _kernelCache["clblast_xgemm"];
            int zero = 0;
            IntPtr pa = aBuf.Handle, pb = bTemp.Handle, pc = cBuf.Handle;
            void** args = stackalloc void*[10];
            args[0] = &mCeiled; args[1] = &nCeiled; args[2] = &kCeiled; args[3] = &alpha; args[4] = &beta;
            args[5] = &pa; args[6] = &pb; args[7] = &pc; args[8] = &zero; args[9] = &zero;
            LaunchKernel2D(kernel, (uint)(mCeiled / c.TileM), (uint)(nCeiled / c.TileN), (uint)c.ThreadTileM, (uint)c.ThreadTileN, args);

            if (cNeedsPad) ClBlastPack("CopyMatrix", cBuf, C, mCeiled, nCeiled, mCeiled, N, M, N);
            Interlocked.Increment(ref _clblastIndirectGemmRuns);
            return true;
        }
    }

    private IGpuBuffer ClBlastScratch(ref IGpuBuffer? buffer, long size)
    {
        if (buffer is not null && buffer.Size >= size) return buffer;
        // hipFree waits for the device, so the old buffer outlives every launch already queued against it.
        buffer?.Dispose();
        buffer = AllocateBufferUninitialized((int)size);
        return buffer;
    }

    private static int CeilTo(int value, int multiple) => (value + multiple - 1) / multiple * multiple;

    /// <summary>One CLBlast pad / transpose launch with OpenClBackend's argument list and geometry.</summary>
    private unsafe void ClBlastPack(string name, IGpuBuffer src, IGpuBuffer dst, int srcOne, int srcTwo, int srcLd,
        int destOne, int destTwo, int destLd)
    {
        IntPtr kernel = _kernelCache[name];
        int offset = 0, flag = 0;
        float one = 1f;
        IntPtr ps = src.Handle, pd = dst.Handle;
        void** args = stackalloc void*[14];
        args[0] = &srcOne; args[1] = &srcTwo; args[2] = &srcLd; args[3] = &offset; args[4] = &ps;
        args[5] = &destOne; args[6] = &destTwo; args[7] = &destLd; args[8] = &offset; args[9] = &pd;
        args[10] = &one; args[11] = &flag; args[12] = &flag; args[13] = &flag;
        uint blockX, blockY, gridX, gridY;
        if (name == "TransposePadMatrix")
        {
            int tile = _clblastPadTransposeParams.Tile, wpt = _clblastPadTransposeParams.WorkPerThread;
            blockX = blockY = (uint)tile;
            gridX = (uint)((destOne + wpt * tile - 1) / (wpt * tile));
            gridY = (uint)((destTwo + wpt * tile - 1) / (wpt * tile));
        }
        else
        {
            var p = _clblastPadParams;
            blockX = (uint)p.DimX;
            blockY = (uint)p.DimY;
            gridX = (uint)((destOne + p.WorkPerThreadX * p.DimX - 1) / (p.WorkPerThreadX * p.DimX));
            gridY = (uint)((destTwo + p.WorkPerThreadY * p.DimY - 1) / (p.WorkPerThreadY * p.DimY));
        }
        LaunchKernel2D(kernel, gridX, gridY, blockX, blockY, args);
    }

    private void DisposeClBlastGemm()
    {
        lock (_clblastLock)
        {
            if (_clblastIndirectModule != IntPtr.Zero) { HipNativeBindings.hipModuleUnload(_clblastIndirectModule); _clblastIndirectModule = IntPtr.Zero; }
            if (_clblastPackingModule != IntPtr.Zero) { HipNativeBindings.hipModuleUnload(_clblastPackingModule); _clblastPackingModule = IntPtr.Zero; }
            _clblastIndirectReady = false;
            _clblastScratchA?.Dispose(); _clblastScratchA = null;
            _clblastScratchB?.Dispose(); _clblastScratchB = null;
            _clblastScratchC?.Dispose(); _clblastScratchC = null;
        }
    }
}