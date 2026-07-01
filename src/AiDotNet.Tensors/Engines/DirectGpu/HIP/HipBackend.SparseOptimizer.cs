// Copyright (c) AiDotNet. All rights reserved.
// HIP backend - native sparse optimizer scatter updates.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    private static void EnsureSparseArgs(IGpuBuffer? param, IGpuBuffer? sparseIndices, IGpuBuffer? sparseValues, int nnz)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (sparseIndices is null) throw new ArgumentNullException(nameof(sparseIndices));
        if (sparseValues is null) throw new ArgumentNullException(nameof(sparseValues));
        if (nnz < 0) throw new ArgumentOutOfRangeException(nameof(nnz));
    }

    private IntPtr GetSparseOptimizerKernel(string name)
    {
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {name}");
        return kernel;
    }

    public unsafe void SparseSgdUpdate(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_sgd_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[7];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &learningRate; args[4] = &weightDecay; args[5] = &nnz; args[6] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_sgd_momentum_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pVel = velocity.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[9];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pVel;
        args[4] = &learningRate; args[5] = &momentum; args[6] = &weightDecay; args[7] = &nnz; args[8] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_adam_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pM = m.Handle, pVv = v.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[13];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pM; args[4] = &pVv;
        args[5] = &learningRate; args[6] = &beta1; args[7] = &beta2; args[8] = &epsilon; args[9] = &weightDecay; args[10] = &step; args[11] = &nnz; args[12] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_adamw_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pM = m.Handle, pVv = v.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[13];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pM; args[4] = &pVv;
        args[5] = &learningRate; args[6] = &beta1; args[7] = &beta2; args[8] = &epsilon; args[9] = &weightDecay; args[10] = &step; args[11] = &nnz; args[12] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float rho, float epsilon, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (squaredAvg is null) throw new ArgumentNullException(nameof(squaredAvg));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_rmsprop_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pSq = squaredAvg.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[10];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pSq;
        args[4] = &learningRate; args[5] = &rho; args[6] = &epsilon; args[7] = &weightDecay; args[8] = &nnz; args[9] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float epsilon, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (accumulatedGrad is null) throw new ArgumentNullException(nameof(accumulatedGrad));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_adagrad_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pA = accumulatedGrad.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[9];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pA;
        args[4] = &learningRate; args[5] = &epsilon; args[6] = &weightDecay; args[7] = &nnz; args[8] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (velocity is null) throw new ArgumentNullException(nameof(velocity));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_nag_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pVel = velocity.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[9];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pVel;
        args[4] = &learningRate; args[5] = &momentum; args[6] = &weightDecay; args[7] = &nnz; args[8] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float rho, float epsilon, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (accumGrad is null) throw new ArgumentNullException(nameof(accumGrad));
        if (accumUpdate is null) throw new ArgumentNullException(nameof(accumUpdate));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_adadelta_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pAg = accumGrad.Handle, pAu = accumUpdate.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[10];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pAg; args[4] = &pAu;
        args[5] = &rho; args[6] = &epsilon; args[7] = &weightDecay; args[8] = &nnz; args[9] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (vMax is null) throw new ArgumentNullException(nameof(vMax));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_amsgrad_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pM = m.Handle, pVv = v.Handle, pVm = vMax.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[14];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pM; args[4] = &pVv; args[5] = &pVm;
        args[6] = &learningRate; args[7] = &beta1; args[8] = &beta2; args[9] = &epsilon; args[10] = &weightDecay; args[11] = &step; args[12] = &nnz; args[13] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (u is null) throw new ArgumentNullException(nameof(u));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_adamax_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pM = m.Handle, pU = u.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[13];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pM; args[4] = &pU;
        args[5] = &learningRate; args[6] = &beta1; args[7] = &beta2; args[8] = &epsilon; args[9] = &weightDecay; args[10] = &step; args[11] = &nnz; args[12] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float weightDecay)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_lion_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pM = m.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[10];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pM;
        args[4] = &learningRate; args[5] = &beta1; args[6] = &beta2; args[7] = &weightDecay; args[8] = &nnz; args[9] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (m is null) throw new ArgumentNullException(nameof(m));
        if (v is null) throw new ArgumentNullException(nameof(v));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_nadam_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pM = m.Handle, pVv = v.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[13];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pM; args[4] = &pVv;
        args[5] = &learningRate; args[6] = &beta1; args[7] = &beta2; args[8] = &epsilon; args[9] = &weightDecay; args[10] = &step; args[11] = &nnz; args[12] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Reg, float l2Reg, float beta)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (z is null) throw new ArgumentNullException(nameof(z));
        if (n is null) throw new ArgumentNullException(nameof(n));
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_ftrl_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle, pZ = z.Handle, pN = n.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[11];
        args[0] = &pP; args[1] = &pI; args[2] = &pV; args[3] = &pZ; args[4] = &pN;
        args[5] = &learningRate; args[6] = &l1Reg; args[7] = &l2Reg; args[8] = &beta; args[9] = &nnz; args[10] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void SparseProximalL1Update(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Strength)
    {
        EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
        if (nnz == 0) return;
        var kernel = GetSparseOptimizerKernel("sparse_proximal_l1_update");
        uint grid = (uint)((nnz + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr pP = param.Handle, pI = sparseIndices.Handle, pV = sparseValues.Handle;
        int paramSize = param.Size;
        void** args = stackalloc void*[7];
        args[0] = &pP; args[1] = &pI; args[2] = &pV;
        args[3] = &learningRate; args[4] = &l1Strength; args[5] = &nnz; args[6] = &paramSize;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }
}
