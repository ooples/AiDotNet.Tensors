// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    public Tensor<float> FusedLoRAForward(
        Tensor<float> input,
        Tensor<float> baseOutput,
        Tensor<float> loraA,
        Tensor<float> loraB,
        float scaling)
    {
        ValidateLoRA(input, baseOutput, loraA, loraB, out int batch, out int inFeat, out int rank, out int outFeat);

        if (TryGetBackend(out var backend) && backend is IIssue301FusedBackend fusedBackend)
        {
            try
            {
                var gi = UploadTensorRaw(backend, input);
                var gb = UploadTensorRaw(backend, baseOutput);
                var ga = UploadTensorRaw(backend, loraA);
                var glb = UploadTensorRaw(backend, loraB);
                var go = backend.AllocateBuffer(batch * outFeat);
                fusedBackend.FusedLoRAForward(gi, gb, ga, glb, go, batch, inFeat, rank, outFeat, scaling);
                return DeferTensorResult<float>(backend, go, batch * outFeat, new[] { batch, outFeat });
            }
            catch
            {
                // Fall through to the CPU fused helper; correctness beats GPU availability.
            }
        }

        var output = new Tensor<float>(new[] { batch, outFeat });
        CpuFusedOperations.FusedLoRAForward(input, baseOutput, loraA, loraB, scaling, output);
        return output;
    }

    public Tensor<float> FusedDDIMStep(
        Tensor<float> xT,
        Tensor<float> epsilonTheta,
        float alphaBarT,
        float alphaBarTMinus1)
    {
        ValidateDDIM(xT, epsilonTheta, alphaBarT, alphaBarTMinus1);

        if (TryGetBackend(out var backend) && backend is IIssue301FusedBackend fusedBackend)
        {
            try
            {
                var gx = UploadTensorRaw(backend, xT);
                var ge = UploadTensorRaw(backend, epsilonTheta);
                var go = backend.AllocateBuffer(xT.Length);
                fusedBackend.FusedDDIMStep(gx, ge, go, xT.Length, alphaBarT, alphaBarTMinus1);
                return DeferTensorResult<float>(backend, go, xT.Length, xT.Shape.ToArray());
            }
            catch
            {
                // Fall through to CPU fused helper.
            }
        }

        var output = new Tensor<float>(xT.Shape.ToArray());
        CpuFusedOperations.FusedDDIMStep(xT, epsilonTheta, alphaBarT, alphaBarTMinus1, output);
        return output;
    }

    public Tensor<float> FusedSparseLinear(
        Tensor<float> input,
        int[] sparseRowOffsets,
        int[] sparseColIndices,
        Tensor<float> sparseValues,
        Tensor<float>? bias,
        FusedActivationType activation)
    {
        ValidateSparseLinear(input, sparseRowOffsets, sparseColIndices, sparseValues, bias,
            out int batch, out int inFeat, out int outFeat, out int nnz);

        if (IsGpuSparseActivationSupported(activation)
            && TryGetBackend(out var backend)
            && backend is IIssue301FusedBackend fusedBackend)
        {
            try
            {
                var gi = UploadTensorRaw(backend, input);
                var gv = UploadTensorRaw(backend, sparseValues);
                using var csr = new OwnedBuffer(backend.AllocateIntBuffer(PackCsr(sparseRowOffsets, sparseColIndices, nnz)), true);
                using var zeroBias = bias is null
                    ? new OwnedBuffer(backend.AllocateBuffer(new[] { 0f }), true)
                    : default;
                var gb = bias is null ? zeroBias.Buffer : UploadTensorRaw(backend, bias);
                var go = backend.AllocateBuffer(batch * outFeat);

                fusedBackend.FusedSparseLinear(
                    gi, csr.Buffer, gv, gb, go,
                    batch, inFeat, outFeat, nnz, bias is null ? 0 : 1, (int)activation);

                return DeferTensorResult<float>(backend, go, batch * outFeat, new[] { batch, outFeat });
            }
            catch
            {
                // Fall through to CPU fused helper.
            }
        }

        var output = new Tensor<float>(new[] { batch, outFeat });
        CpuFusedOperations.FusedSparseLinear(
            input, sparseRowOffsets, sparseColIndices, sparseValues, bias, activation, output);
        return output;
    }

    private static void ValidateLoRA(
        Tensor<float> input,
        Tensor<float> baseOutput,
        Tensor<float> loraA,
        Tensor<float> loraB,
        out int batch,
        out int inFeat,
        out int rank,
        out int outFeat)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (baseOutput is null) throw new ArgumentNullException(nameof(baseOutput));
        if (loraA is null) throw new ArgumentNullException(nameof(loraA));
        if (loraB is null) throw new ArgumentNullException(nameof(loraB));
        if (input.Rank != 2) throw new ArgumentException("input must be rank-2 [batch, in].", nameof(input));
        if (loraA.Rank != 2) throw new ArgumentException("loraA must be rank-2 [in, rank].", nameof(loraA));
        if (loraB.Rank != 2) throw new ArgumentException("loraB must be rank-2 [rank, out].", nameof(loraB));
        if (baseOutput.Rank != 2) throw new ArgumentException("baseOutput must be rank-2 [batch, out].", nameof(baseOutput));

        batch = input.Shape[0];
        inFeat = input.Shape[1];
        rank = loraA.Shape[1];
        outFeat = loraB.Shape[1];

        if (loraA.Shape[0] != inFeat)
            throw new ArgumentException("loraA inner dim must equal input feature dim.", nameof(loraA));
        if (loraB.Shape[0] != rank)
            throw new ArgumentException("loraB inner dim must equal LoRA rank.", nameof(loraB));
        if (baseOutput.Shape[0] != batch || baseOutput.Shape[1] != outFeat)
            throw new ArgumentException("baseOutput shape must be [batch, out].", nameof(baseOutput));
    }

    private static void ValidateDDIM(
        Tensor<float> xT,
        Tensor<float> epsilonTheta,
        float alphaBarT,
        float alphaBarTMinus1)
    {
        if (xT is null) throw new ArgumentNullException(nameof(xT));
        if (epsilonTheta is null) throw new ArgumentNullException(nameof(epsilonTheta));
        if (epsilonTheta.Length != xT.Length)
            throw new ArgumentException("epsilonTheta length must equal xT length.", nameof(epsilonTheta));
        if (!(alphaBarT > 0f && alphaBarT <= 1f))
            throw new ArgumentOutOfRangeException(nameof(alphaBarT), "alphaBarT must be in (0, 1].");
        if (!(alphaBarTMinus1 >= 0f && alphaBarTMinus1 <= 1f))
            throw new ArgumentOutOfRangeException(nameof(alphaBarTMinus1), "alphaBarTMinus1 must be in [0, 1].");
    }

    private static void ValidateSparseLinear(
        Tensor<float> input,
        int[] sparseRowOffsets,
        int[] sparseColIndices,
        Tensor<float> sparseValues,
        Tensor<float>? bias,
        out int batch,
        out int inFeat,
        out int outFeat,
        out int nnz)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (sparseRowOffsets is null) throw new ArgumentNullException(nameof(sparseRowOffsets));
        if (sparseColIndices is null) throw new ArgumentNullException(nameof(sparseColIndices));
        if (sparseValues is null) throw new ArgumentNullException(nameof(sparseValues));
        if (input.Rank != 2) throw new ArgumentException("input must be rank-2 [batch, in].", nameof(input));
        if (sparseRowOffsets.Length < 2)
            throw new ArgumentException("sparseRowOffsets must contain at least one output row plus the terminal nnz.", nameof(sparseRowOffsets));

        batch = input.Shape[0];
        inFeat = input.Shape[1];
        outFeat = sparseRowOffsets.Length - 1;
        if (sparseRowOffsets[0] != 0)
            throw new ArgumentException("sparseRowOffsets[0] must be 0.", nameof(sparseRowOffsets));
        for (int i = 1; i < sparseRowOffsets.Length; i++)
        {
            if (sparseRowOffsets[i] < sparseRowOffsets[i - 1])
                throw new ArgumentException("sparseRowOffsets must be monotonically non-decreasing.", nameof(sparseRowOffsets));
        }

        nnz = sparseRowOffsets[outFeat];
        if (sparseColIndices.Length < nnz)
            throw new ArgumentException("sparseColIndices length must be at least nnz.", nameof(sparseColIndices));
        if (sparseValues.Length < nnz)
            throw new ArgumentException("sparseValues length must be at least nnz.", nameof(sparseValues));
        if (bias is not null && (bias.Rank != 1 || bias.Length != outFeat))
            throw new ArgumentException("bias length must equal sparse output feature count.", nameof(bias));
        for (int i = 0; i < nnz; i++)
        {
            int col = sparseColIndices[i];
            if ((uint)col >= (uint)inFeat)
                throw new ArgumentException("sparseColIndices contains an out-of-range input feature index.", nameof(sparseColIndices));
        }
    }

    private static int[] PackCsr(int[] rowOffsets, int[] colIndices, int nnz)
    {
        var packed = new int[rowOffsets.Length + nnz];
        Array.Copy(rowOffsets, packed, rowOffsets.Length);
        Array.Copy(colIndices, 0, packed, rowOffsets.Length, nnz);
        return packed;
    }

    private static bool IsGpuSparseActivationSupported(FusedActivationType activation)
        => activation is FusedActivationType.None
            or FusedActivationType.ReLU
            or FusedActivationType.GELU
            or FusedActivationType.Sigmoid
            or FusedActivationType.Tanh
            or FusedActivationType.LeakyReLU
            or FusedActivationType.Swish;
}
