using System;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// The CLBlast direct GEMM on HIP (#1266): the kernel OpenCL runs on this GPU, built from the same source with the
// parameters CLBlast tuned for the device, launched with the same geometry.
public sealed partial class HipBackend
{
    private readonly object _clblastLock = new object();
    private IntPtr _clblastDirectModule;
    private ClBlastXgemmDirectParameters _clblastDirectParams;
    private bool _clblastDirectTried;
    private bool _clblastDirectReady;

    /// <summary>The CLBlast database key for this device: the one OpenCL uses for the same GPU.</summary>
    internal ClBlastDeviceInfo ClBlastDevice => ClBlastDeviceInfo.ForAmdGpu(_architectureTarget, DeviceName);

    /// <summary>Builds the direct GEMM module once, with the database parameters unless <paramref name="parameters"/> is given.</summary>
    internal bool EnsureClBlastDirectGemm(ClBlastXgemmDirectParameters? parameters = null)
    {
        lock (_clblastLock)
        {
            if (parameters is null && _clblastDirectTried) return _clblastDirectReady;
            var p = parameters ?? ClBlastXgemmDirectDatabase.GetParameters(ClBlastDevice);
            if (!IsAvailable || !HipClBlastGemmKernels.Supports(p))
            {
                _clblastDirectTried = true;
                _clblastDirectReady = false;
                return false;
            }
            if (_clblastDirectModule != IntPtr.Zero)
            {
                HipNativeBindings.hipModuleUnload(_clblastDirectModule);
                _clblastDirectModule = IntPtr.Zero;
            }
            var failure = CompileKernelModule(HipClBlastGemmKernels.BuildDirectSource(p), "clblast_direct", ref _clblastDirectModule,
                HipClBlastGemmKernels.DirectKernelNames(), useFastMath: false);
            _clblastDirectTried = true;
            _clblastDirectReady = failure is null && _kernelCache.ContainsKey("XgemmDirectTN");
            _clblastDirectParams = p;
            return _clblastDirectReady;
        }
    }

    /// <summary>
    /// Row-major C = alpha * A[M x K] * B[K x N] + beta * C through XgemmDirectTN with C transposed, exactly as
    /// OpenClBackend runs it. Returns false when the module is not available.
    /// </summary>
    internal unsafe bool TryExecuteClBlastDirectGemm(IGpuBuffer A, IGpuBuffer B, IGpuBuffer C, int M, int N, int K,
        float alpha = 1f, float beta = 0f)
    {
        if (M <= 0 || N <= 0 || K <= 0 || !EnsureClBlastDirectGemm()) return false;
        var p = _clblastDirectParams;
        IntPtr kernel = _kernelCache["XgemmDirectTN"];
        int aOffset = 0, bOffset = 0, cOffset = 0, aLd = K, bLd = N, cLd = N, cTranspose = 1, aConj = 0, bConj = 0;
        IntPtr pa = A.Handle, pb = B.Handle, pc = C.Handle;
        void** args = stackalloc void*[17];
        args[0] = &M; args[1] = &N; args[2] = &K; args[3] = &alpha; args[4] = &beta;
        args[5] = &pa; args[6] = &aOffset; args[7] = &aLd;
        args[8] = &pb; args[9] = &bOffset; args[10] = &bLd;
        args[11] = &pc; args[12] = &cOffset; args[13] = &cLd;
        args[14] = &cTranspose; args[15] = &aConj; args[16] = &bConj;
        uint gridX = (uint)((M + p.Wgd - 1) / p.Wgd), gridY = (uint)((N + p.Wgd - 1) / p.Wgd);
        LaunchKernel2D(kernel, gridX, gridY, (uint)p.MdimCd, (uint)p.NdimCd, args);
        return true;
    }

    private void DisposeClBlastGemm()
    {
        lock (_clblastLock)
        {
            if (_clblastDirectModule != IntPtr.Zero)
            {
                HipNativeBindings.hipModuleUnload(_clblastDirectModule);
                _clblastDirectModule = IntPtr.Zero;
            }
            _clblastDirectReady = false;
        }
    }
}