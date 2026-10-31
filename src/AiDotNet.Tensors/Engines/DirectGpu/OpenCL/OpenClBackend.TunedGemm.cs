// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // GEMM through the tuned-kernel registry, as CudaBackend.DeclaredGemm: the reference is the established route (CLBlast
    // above its work threshold, the built-in kernels below), the candidate the register-blocked split-K GEMM. Neither wins
    // everywhere on gfx1012: split-K took 0.38-0.95x CLBlast's time on ViT's and the MLPs' products (2080 x 192 x 192:
    // 91 against 109 us; 256 x 25088 x 64: 401 against 1043 us) and 1.16-1.37x on the large ones (1024^3, 32 x 4096 x 4096,
    // the big im2col products), so each shape keeps whichever measured faster. An unmeasured shape keeps the reference.
    public sealed partial class OpenClBackend
    {
        // Split-K sums K in a different order than the reference: a few ulps of the output scale.
        private const double GemmTolerance = 1e-4;

        private TunedKernelSlot<TunedGemmArgs>? _gemmSlot;

        // Set while the slot's reference runs Gemm, so that call takes the established route instead of the slot again.
        [ThreadStatic]
        private static bool t_inGemmReference;

        internal TunedKernelSlot<TunedGemmArgs> GemmSlot
        {
            get
            {
                if (_gemmSlot is { } s) return s;
                lock (_tunedSlotsLock)
                {
                    if (_gemmSlot is null)
                    {
                        var candidates = new ITunedKernelCandidate<TunedGemmArgs>[]
                        {
                            new DelegateTunedKernelCandidate<TunedGemmArgs>("opencl.gemm.entry", TunedKernelOrigin.Builtin, true,
                                _ => true, (in TunedGemmArgs a) => GemmReference(a)),
                            new DelegateTunedKernelCandidate<TunedGemmArgs>("opencl.gemm.rb_splitk", TunedKernelOrigin.Builtin, true,
                                _ => HasTunedKernel(Kernels.BatchedGemmKernels.Rb64SplitK)
                                    && HasTunedKernel(Kernels.BatchedGemmKernels.SplitKReduce),
                                (in TunedGemmArgs a) =>
                                {
                                    RbSplitKGemm(a.A, a.B, a.C, a.M, a.N, a.K);
                                    RecordGemmDispatch(OpenClGemmDispatchPath.BuiltInKernel);
                                }),
                        };
                        _gemmSlot = new TunedKernelSlot<TunedGemmArgs>(TunedKernelOp.Gemm, TunedDeviceKey,
                            new BackendTunedKernelHarness<TunedGemmArgs>(this, GemmTolerance, () => IsAvailable,
                                MeasureTunedKernelMilliseconds, a => new[] { (a.C, a.OutputCount) },
                                a => a.C.Handle == a.A.Handle || a.C.Handle == a.B.Handle),
                            () => GpuDeterminism.IsActive, candidates);
                    }
                    return _gemmSlot;
                }
            }
        }

        private void GemmReference(in TunedGemmArgs a) => RunGemmReference(this, a);

        // Static: the flag is per thread, not per backend.
        private static void RunGemmReference(OpenClBackend backend, in TunedGemmArgs a)
        {
            bool prior = t_inGemmReference;
            t_inGemmReference = true;
            try
            {
                backend.Gemm(a.A, a.B, a.C, a.M, a.N, a.K);
            }
            finally
            {
                t_inGemmReference = prior;
            }
        }

        // True when the slot ran the product (C = A * B at alpha 1, beta 0); false to take the established route.
        private bool TryTunedGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k, float alpha, float beta)
        {
            if (t_inGemmReference || alpha is not 1f || beta is not 0f || m <= 0 || n <= 0 || k <= 0
                || (long)m * n > int.MaxValue)
                return false;
            var args = new TunedGemmArgs(a, b, c, m, n, k);
            return GemmSlot.TryExecute(args.Shape(), args);
        }
    }
}