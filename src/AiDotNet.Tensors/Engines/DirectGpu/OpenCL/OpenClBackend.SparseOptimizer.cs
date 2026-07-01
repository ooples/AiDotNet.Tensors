// Copyright (c) AiDotNet. All rights reserved.
// OpenCL backend - native sparse optimizer scatter updates.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    public sealed partial class OpenClBackend
    {
        private static void EnsureSparseArgs(IGpuBuffer? param, IGpuBuffer? sparseIndices, IGpuBuffer? sparseValues, int nnz)
        {
            if (param is null) throw new ArgumentNullException(nameof(param));
            if (sparseIndices is null) throw new ArgumentNullException(nameof(sparseIndices));
            if (sparseValues is null) throw new ArgumentNullException(nameof(sparseValues));
            if (nnz < 0) throw new ArgumentOutOfRangeException(nameof(nnz));
        }

        public void SparseSgdUpdate(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float weightDecay)
        {
            var k = BeginSparseKernel("sparse_sgd_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, weightDecay);
            SetSparseSizes(k, ref arg, nnz, param.Size);
            ExecuteSparse(k, nnz);
        }

        public void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        {
            var k = BeginSparseKernel("sparse_sgd_momentum_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, velocity);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, momentum);
            k.SetArg(arg++, weightDecay);
            SetSparseSizes(k, ref arg, nnz, param.Size);
            ExecuteSparse(k, nnz);
        }

        public void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        {
            EnsureBiasCorrectedSparseArgs(m, v, epsilon, step);
            var k = BeginSparseKernel("sparse_adam_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, m);
            SetBuffer(k, ref arg, v);
            SetAdamScalars(k, ref arg, nnz, param.Size, learningRate, beta1, beta2, epsilon, weightDecay, step);
            ExecuteSparse(k, nnz);
        }

        public void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        {
            EnsureBiasCorrectedSparseArgs(m, v, epsilon, step);
            var k = BeginSparseKernel("sparse_adamw_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, m);
            SetBuffer(k, ref arg, v);
            SetAdamScalars(k, ref arg, nnz, param.Size, learningRate, beta1, beta2, epsilon, weightDecay, step);
            ExecuteSparse(k, nnz);
        }

        public void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float rho, float epsilon, float weightDecay)
        {
            EnsureEpsilon(epsilon);
            var k = BeginSparseKernel("sparse_rmsprop_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, squaredAvg);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, rho);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            SetSparseSizes(k, ref arg, nnz, param.Size);
            ExecuteSparse(k, nnz);
        }

        public void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float epsilon, float weightDecay)
        {
            EnsureEpsilon(epsilon);
            var k = BeginSparseKernel("sparse_adagrad_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, accumulatedGrad);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            SetSparseSizes(k, ref arg, nnz, param.Size);
            ExecuteSparse(k, nnz);
        }

        public void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        {
            var k = BeginSparseKernel("sparse_nag_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, velocity);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, momentum);
            k.SetArg(arg++, weightDecay);
            SetSparseSizes(k, ref arg, nnz, param.Size);
            ExecuteSparse(k, nnz);
        }

        public void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float rho, float epsilon, float weightDecay)
        {
            EnsureEpsilon(epsilon);
            var k = BeginSparseKernel("sparse_adadelta_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, accumGrad);
            SetBuffer(k, ref arg, accumUpdate);
            k.SetArg(arg++, rho);
            k.SetArg(arg++, epsilon);
            k.SetArg(arg++, weightDecay);
            SetSparseSizes(k, ref arg, nnz, param.Size);
            ExecuteSparse(k, nnz);
        }

        public void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        {
            EnsureBiasCorrectedSparseArgs(m, v, epsilon, step);
            if (vMax is null) throw new ArgumentNullException(nameof(vMax));
            var k = BeginSparseKernel("sparse_amsgrad_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, m);
            SetBuffer(k, ref arg, v);
            SetBuffer(k, ref arg, vMax);
            SetAdamScalars(k, ref arg, nnz, param.Size, learningRate, beta1, beta2, epsilon, weightDecay, step);
            ExecuteSparse(k, nnz);
        }

        public void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        {
            EnsureBiasCorrectedSparseArgs(m, u, epsilon, step);
            var k = BeginSparseKernel("sparse_adamax_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, m);
            SetBuffer(k, ref arg, u);
            SetAdamScalars(k, ref arg, nnz, param.Size, learningRate, beta1, beta2, epsilon, weightDecay, step);
            ExecuteSparse(k, nnz);
        }

        public void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float weightDecay)
        {
            var k = BeginSparseKernel("sparse_lion_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, m);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, beta1);
            k.SetArg(arg++, beta2);
            k.SetArg(arg++, weightDecay);
            SetSparseSizes(k, ref arg, nnz, param.Size);
            ExecuteSparse(k, nnz);
        }

        public void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        {
            EnsureBiasCorrectedSparseArgs(m, v, epsilon, step);
            var k = BeginSparseKernel("sparse_nadam_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, m);
            SetBuffer(k, ref arg, v);
            SetAdamScalars(k, ref arg, nnz, param.Size, learningRate, beta1, beta2, epsilon, weightDecay, step);
            ExecuteSparse(k, nnz);
        }

        public void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Reg, float l2Reg, float beta)
        {
            var k = BeginSparseKernel("sparse_ftrl_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            SetBuffer(k, ref arg, z);
            SetBuffer(k, ref arg, n);
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, l1Reg);
            k.SetArg(arg++, l2Reg);
            k.SetArg(arg++, beta);
            SetSparseSizes(k, ref arg, nnz, param.Size);
            ExecuteSparse(k, nnz);
        }

        public void SparseProximalL1Update(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Strength)
        {
            var k = BeginSparseKernel("sparse_proximal_l1_update", param, sparseIndices, sparseValues, nnz, out uint arg);
            if (k is null) return;
            k.SetArg(arg++, learningRate);
            k.SetArg(arg++, l1Strength);
            SetSparseSizes(k, ref arg, nnz, param.Size);
            ExecuteSparse(k, nnz);
        }

        private DirectOpenClKernel? BeginSparseKernel(string kernelName, IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, out uint arg)
        {
            EnsureSparseArgs(param, sparseIndices, sparseValues, nnz);
            arg = 0;
            if (nnz == 0) return null;
            var kernel = _kernelCache[kernelName];
            SetBuffer(kernel, ref arg, param);
            SetBuffer(kernel, ref arg, sparseIndices);
            SetBuffer(kernel, ref arg, sparseValues);
            return kernel;
        }

        private static void SetBuffer(DirectOpenClKernel kernel, ref uint arg, IGpuBuffer buffer)
        {
            if (buffer is null) throw new ArgumentNullException(nameof(buffer));
            kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)buffer).Buffer.Handle);
        }

        private static void SetAdamScalars(DirectOpenClKernel kernel, ref uint arg, int nnz, int paramSize, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        {
            kernel.SetArg(arg++, learningRate);
            kernel.SetArg(arg++, beta1);
            kernel.SetArg(arg++, beta2);
            kernel.SetArg(arg++, epsilon);
            kernel.SetArg(arg++, weightDecay);
            kernel.SetArg(arg++, step);
            SetSparseSizes(kernel, ref arg, nnz, paramSize);
        }

        private static void SetSparseSizes(DirectOpenClKernel kernel, ref uint arg, int nnz, int paramSize)
        {
            kernel.SetArg(arg++, nnz);
            kernel.SetArg(arg++, paramSize);
        }

        private static void ExecuteSparse(DirectOpenClKernel kernel, int nnz)
            => kernel.Execute1D(nnz, Math.Min(256, nnz));

        private static void EnsureBiasCorrectedSparseArgs(IGpuBuffer state1, IGpuBuffer state2, float epsilon, int step)
        {
            if (state1 is null) throw new ArgumentNullException(nameof(state1));
            if (state2 is null) throw new ArgumentNullException(nameof(state2));
            if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));
            EnsureEpsilon(epsilon);
        }

        private static void EnsureEpsilon(float epsilon)
        {
            if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        }
    }
}
