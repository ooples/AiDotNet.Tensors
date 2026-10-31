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
                                _ => HasTunedKernel(Kernels.BatchedGemmKernels.Rb64SplitK, Rb64GroupSide * Rb64GroupSide)
                                    && HasTunedKernel(Kernels.BatchedGemmKernels.SplitKReduce),
                                (in TunedGemmArgs a) =>
                                {
                                    RbSplitKGemm(a.A, a.B, a.C, a.M, a.N, a.K);
                                    RecordGemmDispatch(OpenClGemmDispatchPath.BuiltInKernel);
                                }),
                            // A linear layer over a small batch: a read of B (the weights) at bandwidth.
                            new DelegateTunedKernelCandidate<TunedGemmArgs>("opencl.gemm.skinny", TunedKernelOrigin.Builtin, true,
                                shape => shape[0] <= SkinnyGemmMaxRows && shape[1] % SkinnyVectorWidth == 0
                                    && HasTunedKernel(Kernels.BatchedGemmKernels.SkinnySplitK, SkinnyGroupSize)
                                    && HasTunedKernel(Kernels.BatchedGemmKernels.SplitKReduce),
                                (in TunedGemmArgs a) =>
                                {
                                    SkinnyGemm(a.A, a.B, a.C, a.M, a.N, a.K);
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

        // gemm_skinny_splitk's layout: a work item reads SkinnyVectorWidth columns (one float4) for SkinnyRowTile rows,
        // SkinnyVectorsPerGroup float4 columns per SkinnyGroupSize-item group, and the group's remaining items split K
        // into lanes (SkinnyMaxRowTiles row tiles at most, so M <= SkinnyGemmMaxRows). K is split until the grid covers
        // about SkinnyGroupsPerComputeUnit groups per compute unit, each slice at least SkinnyMinSlice deep and a multiple
        // of SkinnyChunkAlign.
        private const int SkinnyRowTile = 8;
        private const int SkinnyMaxRowTiles = 4;
        private const int SkinnyGemmMaxRows = SkinnyRowTile * SkinnyMaxRowTiles;
        private const int SkinnyVectorWidth = 4;
        private const int SkinnyVectorsPerGroup = 64;
        private const int SkinnyGroupSize = 256;
        private const int SkinnyGroupsPerComputeUnit = 8;
        private const int SkinnyMinSlice = 256;
        private const int SkinnyChunkAlign = 64;

        // C[M x N] = A[M x K] * B[K x N] through gemm_skinny_splitk (M <= SkinnyGemmMaxRows, N a multiple of
        // SkinnyVectorWidth), the planes summed by gemm_splitk_reduce.
        internal void SkinnyGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int n, int k)
        {
            int rowTiles = (m + SkinnyRowTile - 1) / SkinnyRowTile, kLanes = SkinnyMaxRowTiles / rowTiles;
            int columnBlocks = (n / SkinnyVectorWidth + SkinnyVectorsPerGroup - 1) / SkinnyVectorsPerGroup;
            int splits = SplitKFactor(columnBlocks, k, SkinnyMinSlice, SkinnyGroupsPerComputeUnit, SplitComputeUnits);
            int kChunk = RoundUpToMultiple((k + splits - 1) / splits, SkinnyChunkAlign);
            splits = (k + kChunk - 1) / kChunk;
            int planes = splits * kLanes;
            using IGpuBuffer? partial = planes > 1 ? AllocateBuffer(checked(planes * m * n)) : null;
            var gemm = _kernelCache[Kernels.BatchedGemmKernels.SkinnySplitK];
            gemm.SetArg(0, BufferHandle(a)); gemm.SetArg(1, BufferHandle(b)); gemm.SetArg(2, BufferHandle(partial ?? c));
            gemm.SetArg(3, m); gemm.SetArg(4, n); gemm.SetArg(5, k); gemm.SetArg(6, kChunk); gemm.SetArg(7, rowTiles);
            gemm.Execute2D(columnBlocks * SkinnyGroupSize, splits, SkinnyGroupSize, 1);
            if (partial is null) return;
            var reduce = _kernelCache[Kernels.BatchedGemmKernels.SplitKReduce];
            reduce.SetArg(0, BufferHandle(partial)); reduce.SetArg(1, BufferHandle(c));
            reduce.SetArg(2, m * n); reduce.SetArg(3, planes);
            reduce.SetArg(4, BufferHandle(c)); reduce.SetArg(5, 0); reduce.SetArg(6, n);
            reduce.Execute1D(RoundUpToMultiple(m * n, ReduceGroupSize), ReduceGroupSize);
        }

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