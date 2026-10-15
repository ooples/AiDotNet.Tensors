using System;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// CLBlast's indirect GEMM on HIP (#1266): the kernel OpenCL runs on this GPU, built from the same source with the
// configuration CLBlast tuned for the device, packed and launched the same way. Evolution-tuned configurations
// (HipBackend.EvolutionTuning.cs) build as further variants beside it and are deployed per shape.
public sealed partial class HipBackend
{
    /// <summary>One compiled clblast_xgemm configuration: its own module, under its own kernel name.</summary>
    private sealed class ClBlastXgemmVariant
    {
        internal ClBlastXgemmVariant(GemmConfig config, IntPtr module, IntPtr kernel)
        {
            Config = config;
            Module = module;
            Kernel = kernel;
        }

        internal GemmConfig Config { get; }
        internal IntPtr Module { get; }
        internal IntPtr Kernel { get; }
    }

    private readonly object _clblastLock = new object();
    private IntPtr _clblastPackingModule;
    private ClBlastXgemmVariant? _clblastDefault;
    private readonly Dictionary<OpenClGemmConfiguration, ClBlastXgemmVariant> _clblastVariants = new();
    private readonly Dictionary<(int M, int N, int K), ClBlastXgemmVariant> _clblastDeployments = new();
    private ClBlastPadParameters _clblastPadParams;
    private ClBlastPadTransposeParameters _clblastPadTransposeParams;
    private int _clblastMinIndirectSize;
    private bool _clblastIndirectTried;
    private long _clblastIndirectGemmRuns;
    // Grow-only packing buffers: per-call allocate / free cost more than the GEMM itself at 448^3.
    private IGpuBuffer? _clblastScratchA, _clblastScratchB, _clblastScratchC;

    /// <summary>The CLBlast database key for this device: the one OpenCL uses for the same GPU.</summary>
    internal ClBlastDeviceInfo ClBlastDevice => ClBlastDeviceInfo.ForAmdGpu(_deviceProps.GcnArchName ?? _architectureTarget, DeviceName);

    /// <summary>Whether this backend has tried to build the CLBlast modules (lets tests prove small GEMMs never do).</summary>
    internal bool ClBlastModulesAttempted
    {
        get
        {
            lock (_clblastLock) return _clblastIndirectTried;
        }
    }

    /// <summary>How many GEMMs the indirect path has run on this backend (lets tests prove the route was taken).</summary>
    internal long ClBlastIndirectGemmRuns => Interlocked.Read(ref _clblastIndirectGemmRuns);

    /// <summary>Whether Gemm goes to hipBLAS first (and so never reaches the CLBlast kernels).</summary>
    internal bool IsHipBlasAvailable => HipBlasReady;

    /// <summary>CLBlast's MinIndirectSize for this device (0 when the database has none), looked up once without compiling.</summary>
    private int ClBlastMinIndirectSize
    {
        get
        {
            int cached = Volatile.Read(ref _clblastMinIndirectSize);
            if (cached != 0) return cached;
            int looked = ClBlastGemmRoutineDatabase.GetXgemmMinIndirectSize(ClBlastDevice);
            cached = looked > 0 ? looked : -1;
            Volatile.Write(ref _clblastMinIndirectSize, cached);
            return cached;
        }
    }

    /// <summary>The configuration the database gives this device, once the indirect path is built.</summary>
    internal GemmConfig? ClBlastDefaultConfig
    {
        get
        {
            lock (_clblastLock) return _clblastDefault?.Config;
        }
    }

    /// <summary>
    /// Builds the packing kernels and the database configuration's clblast_xgemm once. Returns whether the indirect
    /// path is usable on this device.
    /// </summary>
    internal bool EnsureClBlastIndirectGemm()
    {
        lock (_clblastLock)
        {
            if (_clblastIndirectTried) return _clblastDefault is not null;
            _clblastIndirectTried = true;
            var device = ClBlastDevice;
            if (!IsAvailable || !ClBlastXgemmDatabase.TryGetConfig(device, out GemmConfig config) || !EnsureClBlastPacking(device))
                return false;
            _clblastDefault = BuildClBlastVariant(config, HipClBlastGemmKernels.IndirectKernelName);
            return _clblastDefault is not null;
        }
    }

    /// <summary>
    /// The compiled variant for <paramref name="config"/>, building it (and the packing kernels) on first use. Null when
    /// HIP cannot build that configuration.
    /// </summary>
    private ClBlastXgemmVariant? GetOrBuildClBlastVariant(GemmConfig config)
    {
        lock (_clblastLock)
        {
            if (!EnsureClBlastIndirectGemm()) return null;
            // Keyed on the fields the kernel reads, so genomes that differ only elsewhere share one module.
            var key = HipClBlastGemmVariation.Canonical(OpenClGemmConfiguration.FromGemmConfig(config));
            if (_clblastVariants.TryGetValue(key, out var existing)) return existing;
            var variant = BuildClBlastVariant(config, HipClBlastGemmKernels.IndirectKernelName + "_v" + _clblastVariants.Count.ToString(System.Globalization.CultureInfo.InvariantCulture));
            if (variant is not null) _clblastVariants[key] = variant;
            return variant;
        }
    }

    /// <summary>Whether this backend can build <paramref name="config"/> as a clblast_xgemm variant.</summary>
    internal bool TryPrepareClBlastGemm(GemmConfig config) => HipClBlastGemmKernels.Supports(config) && GetOrBuildClBlastVariant(config) is not null;

    /// <summary>Sends shape (M, N, K) to <paramref name="config"/>'s variant from now on, building it if needed.</summary>
    internal bool DeployClBlastGemm(int M, int N, int K, GemmConfig config)
    {
        var variant = HipClBlastGemmKernels.Supports(config) ? GetOrBuildClBlastVariant(config) : null;
        if (variant is null) return false;
        // The exact configuration asked for, on the module its kernel-equivalent genome already built (modules belong to
        // _clblastVariants and are unloaded from there).
        lock (_clblastLock) _clblastDeployments[(M, N, K)] = new ClBlastXgemmVariant(config, variant.Module, variant.Kernel);
        return true;
    }

    /// <summary>The configuration shape (M, N, K) runs with: its deployment, or the database's.</summary>
    internal GemmConfig? ClBlastConfigFor(int M, int N, int K) => ClBlastVariantFor(M, N, K)?.Config;

    private ClBlastXgemmVariant? ClBlastVariantFor(int M, int N, int K)
    {
        if (!EnsureClBlastIndirectGemm()) return null;
        lock (_clblastLock)
            return _clblastDeployments.TryGetValue((M, N, K), out var deployed) ? deployed : _clblastDefault;
    }

    private bool EnsureClBlastPacking(ClBlastDeviceInfo device)
    {
        if (_clblastPackingModule != IntPtr.Zero) return true;
        _clblastPadParams = ClBlastPadDatabase.GetParameters(device);
        _clblastPadTransposeParams = ClBlastPadTransposeDatabase.GetParameters(device);
        var packing = HipClBlastGemmKernels.BuildPackingSource(_clblastPadParams, _clblastPadTransposeParams,
            ClBlastCopyDatabase.GetParameters(device), ClBlastTransposeDatabase.GetParameters(device));
        return CompileKernelModule(packing, "clblast_packing", ref _clblastPackingModule,
            HipClBlastGemmKernels.PackingKernelNames, useFastMath: false) is null;
    }

    private ClBlastXgemmVariant? BuildClBlastVariant(GemmConfig config, string kernelName)
    {
        if (!HipClBlastGemmKernels.Supports(config)) return null;
        IntPtr module = IntPtr.Zero;
        if (CompileKernelModule(HipClBlastGemmKernels.BuildIndirectSource(config, kernelName), kernelName, ref module,
            new[] { kernelName }, useFastMath: false) is not null || !_kernelCache.TryGetValue(kernelName, out IntPtr kernel))
        {
            if (module != IntPtr.Zero) HipNativeBindings.hipModuleUnload(module);
            return null;
        }
        return new ClBlastXgemmVariant(config, module, kernel);
    }

    /// <summary>
    /// The shapes Gemm sends through the indirect path: M or N at least <see cref="HipIndirectMinSize"/>, and tile
    /// padding adding at most half again the work. Measured on an RX 5500 XT, a side padded from 8 to the 64-wide tile
    /// (8x512x512: 22 vs 48 GF) or from 16 (16x1024x1024: 125 vs 218 GF) loses to the HIP kernels, while unpadded skinny
    /// shapes win (4096x64x4096: 727 vs 406 GF).
    /// </summary>
    // #1266: CLBlast's MinIndirectSize (448 for this device) is where its OpenCL switches from the direct to the indirect
    // kernel. On HIP the alternative is not CLBlast's direct kernel but ours, and the indirect kernel beat it at every
    // size measured below 448 on an RX 5500 XT (ms, ours vs indirect, bit-identical results): 64^3 0.066 vs 0.019, 128^3
    // 0.063 vs 0.028, 256^3 0.137 vs 0.048, 384^3 0.298 vs 0.080, 447^3 0.396 vs 0.122, 128x4096x128 0.126 vs 0.063,
    // 4096x128x256 0.272 vs 0.202, 200x500x300 0.155 vs 0.071.
    private const int HipIndirectMinSize = 64;

    private bool ShouldUseClBlastIndirectGemm(int M, int N, int K)
    {
        if (M <= 1 || N <= 1) return false;
        // Checked before ClBlastVariantFor, so the smallest GEMMs (including the first a process runs) never compile the
        // CLBlast modules.
        int minIndirect = ClBlastMinIndirectSize > 0 ? Math.Min(ClBlastMinIndirectSize, HipIndirectMinSize) : 0;
        if (minIndirect <= 0 || (M < minIndirect && N < minIndirect)) return false;
        var variant = ClBlastVariantFor(M, N, K);
        if (variant is null) return false;
        var c = variant.Config;
        double padded = (double)CeilTo(N, c.TileM) * CeilTo(M, c.TileN) * CeilTo(K, c.TileK * (c.KReg > 0 ? c.KReg : 1));
        return padded <= 1.5 * M * N * K;
    }

    /// <summary>
    /// Row-major C = alpha * A[M x K] * B[K x N] + beta * C through CLBlast's indirect path with the shape's deployed
    /// configuration (or the database's), as OpenClBackend runs it.
    /// </summary>
    internal bool TryExecuteClBlastIndirectGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha = 1f, float beta = 0f)
    {
        if (M <= 0 || N <= 0 || K <= 0) return false;
        var variant = ClBlastVariantFor(M, N, K);
        return variant is not null && ExecuteClBlastIndirectGemm(variant, A, B, C, M, N, K, alpha, beta);
    }

    /// <summary>
    /// Runs <paramref name="config"/>'s variant (building it if needed) whatever is deployed for the shape: the tuner's
    /// measurement path. Returns false when HIP cannot build that configuration.
    /// </summary>
    internal bool TryExecuteClBlastIndirectGemm(GemmConfig config, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha = 1f, float beta = 0f)
    {
        if (M <= 0 || N <= 0 || K <= 0 || !HipClBlastGemmKernels.Supports(config)) return false;
        var variant = GetOrBuildClBlastVariant(config);
        return variant is not null && ExecuteClBlastIndirectGemm(variant, A, B, C, M, N, K, alpha, beta);
    }

    /// <summary>
    /// The row-major swap of OpenClBackend's TryExecuteClBlastBaselineGemm: B is padded into the kernel's A slot, A is
    /// transposed into its B slot, the GEMM runs on tile-multiple sizes, and C is copied out of a padded buffer when it
    /// needs one.
    /// </summary>
    private unsafe bool ExecuteClBlastIndirectGemm(ClBlastXgemmVariant variant, IGpuBuffer A, IGpuBuffer B, IGpuBuffer C,
        int M, int N, int K, float alpha, float beta)
    {
        var c = variant.Config;
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

            int zero = 0;
            IntPtr pa = aBuf.Handle, pb = bTemp.Handle, pc = cBuf.Handle;
            void** args = stackalloc void*[10];
            args[0] = &mCeiled; args[1] = &nCeiled; args[2] = &kCeiled; args[3] = &alpha; args[4] = &beta;
            args[5] = &pa; args[6] = &pb; args[7] = &pc; args[8] = &zero; args[9] = &zero;
            LaunchKernel2D(variant.Kernel, (uint)(mCeiled / c.TileM), (uint)(nCeiled / c.TileN), (uint)c.ThreadTileM, (uint)c.ThreadTileN, args);

            if (cNeedsPad) ClBlastPack("CopyMatrix", cBuf, C, mCeiled, nCeiled, mCeiled, N, M, N);
            Interlocked.Increment(ref _clblastIndirectGemmRuns);
            return true;
        }
    }
    /// <summary>Returns <paramref name="buffer"/> when it holds <paramref name="size"/> floats, otherwise replaces it with a larger one.</summary>
    private IGpuBuffer ClBlastScratch(ref IGpuBuffer? buffer, long size)
    {
        if (buffer is not null && buffer.Size >= size) return buffer;
        // Dispose hands the buffer back to the shared pool without synchronizing, and another caller could rent it while
        // launches queued against it are still running; drain them first. This only happens when a shape outgrows it.
        if (buffer is not null)
        {
            HipNativeBindings.CheckError(HipNativeBindings.hipStreamSynchronize(_stream), "hipStreamSynchronize");
            buffer.Dispose();
        }
        buffer = AllocateBufferUninitialized((int)size);
        return buffer;
    }

    /// <summary><paramref name="value"/> rounded up to a multiple of <paramref name="multiple"/>.</summary>
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

    /// <summary>Unloads the CLBlast modules and releases the scratch buffers; called from the backend's Dispose.</summary>
    private void DisposeClBlastGemm()
    {
        lock (_clblastLock)
        {
            foreach (var variant in _clblastVariants.Values) HipNativeBindings.hipModuleUnload(variant.Module);
            _clblastVariants.Clear();
            _clblastDeployments.Clear();
            if (_clblastDefault is not null) HipNativeBindings.hipModuleUnload(_clblastDefault.Module);
            _clblastDefault = null;
            if (_clblastPackingModule != IntPtr.Zero) { HipNativeBindings.hipModuleUnload(_clblastPackingModule); _clblastPackingModule = IntPtr.Zero; }
            _clblastScratchA?.Dispose(); _clblastScratchA = null;
            _clblastScratchB?.Dispose(); _clblastScratchB = null;
            _clblastScratchC?.Dispose(); _clblastScratchC = null;
        }
    }
}