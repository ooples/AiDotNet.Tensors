// Copyright (c) AiDotNet. All rights reserved.
// Metal backend - sparse optimizer scatter updates.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    public void SparseSgdUpdate(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float weightDecay)
        => ApplySparse(param, sparseIndices, sparseValues, (p, i, v) =>
            SparseOptimizerReference.SparseSgdUpdate(p, i, v, nnz, learningRate, weightDecay));

    public void SparseSgdMomentumUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => ApplySparse1(param, velocity, sparseIndices, sparseValues, (p, s, i, v) =>
            SparseOptimizerReference.SparseSgdMomentumUpdate(p, s, i, v, nnz, learningRate, momentum, weightDecay));

    public void SparseAdamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => ApplySparse2(param, m, v, sparseIndices, sparseValues, (p, s1, s2, i, vals) =>
            SparseOptimizerReference.SparseAdamUpdate(p, s1, s2, i, vals, nnz, learningRate, beta1, beta2, epsilon, weightDecay, step));

    public void SparseAdamWUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => ApplySparse2(param, m, v, sparseIndices, sparseValues, (p, s1, s2, i, vals) =>
            SparseOptimizerReference.SparseAdamWUpdate(p, s1, s2, i, vals, nnz, learningRate, beta1, beta2, epsilon, weightDecay, step));

    public void SparseRmspropUpdate(IGpuBuffer param, IGpuBuffer squaredAvg, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float rho, float epsilon, float weightDecay)
        => ApplySparse1(param, squaredAvg, sparseIndices, sparseValues, (p, s, i, v) =>
            SparseOptimizerReference.SparseRmspropUpdate(p, s, i, v, nnz, learningRate, rho, epsilon, weightDecay));

    public void SparseAdagradUpdate(IGpuBuffer param, IGpuBuffer accumulatedGrad, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float epsilon, float weightDecay)
        => ApplySparse1(param, accumulatedGrad, sparseIndices, sparseValues, (p, s, i, v) =>
            SparseOptimizerReference.SparseAdagradUpdate(p, s, i, v, nnz, learningRate, epsilon, weightDecay));

    public void SparseNagUpdate(IGpuBuffer param, IGpuBuffer velocity, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float momentum, float weightDecay)
        => ApplySparse1(param, velocity, sparseIndices, sparseValues, (p, s, i, v) =>
            SparseOptimizerReference.SparseNagUpdate(p, s, i, v, nnz, learningRate, momentum, weightDecay));

    public void SparseAdadeltaUpdate(IGpuBuffer param, IGpuBuffer accumGrad, IGpuBuffer accumUpdate, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float rho, float epsilon, float weightDecay)
        => ApplySparse2(param, accumGrad, accumUpdate, sparseIndices, sparseValues, (p, s1, s2, i, v) =>
            SparseOptimizerReference.SparseAdadeltaUpdate(p, s1, s2, i, v, nnz, rho, epsilon, weightDecay));

    public void SparseAmsgradUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer vMax, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => ApplySparse3(param, m, v, vMax, sparseIndices, sparseValues, (p, s1, s2, s3, i, vals) =>
            SparseOptimizerReference.SparseAmsgradUpdate(p, s1, s2, s3, i, vals, nnz, learningRate, beta1, beta2, epsilon, weightDecay, step));

    public void SparseAdamaxUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer u, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => ApplySparse2(param, m, u, sparseIndices, sparseValues, (p, s1, s2, i, vals) =>
            SparseOptimizerReference.SparseAdamaxUpdate(p, s1, s2, i, vals, nnz, learningRate, beta1, beta2, epsilon, weightDecay, step));

    public void SparseLionUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float weightDecay)
        => ApplySparse1(param, m, sparseIndices, sparseValues, (p, s, i, v) =>
            SparseOptimizerReference.SparseLionUpdate(p, s, i, v, nnz, learningRate, beta1, beta2, weightDecay));

    public void SparseNadamUpdate(IGpuBuffer param, IGpuBuffer m, IGpuBuffer v, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, int step)
        => ApplySparse2(param, m, v, sparseIndices, sparseValues, (p, s1, s2, i, vals) =>
            SparseOptimizerReference.SparseNadamUpdate(p, s1, s2, i, vals, nnz, learningRate, beta1, beta2, epsilon, weightDecay, step));

    public void SparseFtrlUpdate(IGpuBuffer param, IGpuBuffer z, IGpuBuffer n, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Reg, float l2Reg, float beta)
        => ApplySparse2(param, z, n, sparseIndices, sparseValues, (p, s1, s2, i, v) =>
            SparseOptimizerReference.SparseFtrlUpdate(p, s1, s2, i, v, nnz, learningRate, l1Reg, l2Reg, beta));

    public void SparseProximalL1Update(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, int nnz, float learningRate, float l1Strength)
        => ApplySparse(param, sparseIndices, sparseValues, (p, i, v) =>
            SparseOptimizerReference.SparseProximalL1Update(p, i, v, nnz, learningRate, l1Strength));

    private void ApplySparse(IGpuBuffer param, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, Action<float[], float[], float[]> update)
    {
        ThrowIfDisposed();
        var p = DownloadBuffer(param);
        var i = DownloadBuffer(sparseIndices);
        var v = DownloadBuffer(sparseValues);
        update(p, i, v);
        UploadToBuffer(param, p);
    }

    private void ApplySparse1(IGpuBuffer param, IGpuBuffer state, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, Action<float[], float[], float[], float[]> update)
    {
        ThrowIfDisposed();
        var p = DownloadBuffer(param);
        var s = DownloadBuffer(state);
        var i = DownloadBuffer(sparseIndices);
        var v = DownloadBuffer(sparseValues);
        update(p, s, i, v);
        UploadToBuffer(param, p);
        UploadToBuffer(state, s);
    }

    private void ApplySparse2(IGpuBuffer param, IGpuBuffer state1, IGpuBuffer state2, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, Action<float[], float[], float[], float[], float[]> update)
    {
        ThrowIfDisposed();
        var p = DownloadBuffer(param);
        var s1 = DownloadBuffer(state1);
        var s2 = DownloadBuffer(state2);
        var i = DownloadBuffer(sparseIndices);
        var v = DownloadBuffer(sparseValues);
        update(p, s1, s2, i, v);
        UploadToBuffer(param, p);
        UploadToBuffer(state1, s1);
        UploadToBuffer(state2, s2);
    }

    private void ApplySparse3(IGpuBuffer param, IGpuBuffer state1, IGpuBuffer state2, IGpuBuffer state3, IGpuBuffer sparseIndices, IGpuBuffer sparseValues, Action<float[], float[], float[], float[], float[], float[]> update)
    {
        ThrowIfDisposed();
        var p = DownloadBuffer(param);
        var s1 = DownloadBuffer(state1);
        var s2 = DownloadBuffer(state2);
        var s3 = DownloadBuffer(state3);
        var i = DownloadBuffer(sparseIndices);
        var v = DownloadBuffer(sparseValues);
        update(p, s1, s2, s3, i, v);
        UploadToBuffer(param, p);
        UploadToBuffer(state1, s1);
        UploadToBuffer(state2, s2);
        UploadToBuffer(state3, s3);
    }
}
