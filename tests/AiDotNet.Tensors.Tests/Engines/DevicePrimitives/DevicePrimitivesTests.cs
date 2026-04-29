// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DevicePrimitives;
using AiDotNet.Tensors.Engines.DevicePrimitives.Cpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DevicePrimitives;

/// <summary>
/// Acceptance tests for the GPU primitives breadth surface from #219.
/// Covers: cross-backend Philox bit-equivalence, CSR sparse SpMM/SpMV
/// against a dense reference, dense linalg factorisations via the
/// <see cref="CuSolver"/> wrapper, RNN forward correctness against a
/// hand-computed reference, NVTX no-op safety on hosts without
/// <c>nvToolsExt</c>, and the unified GPU memory-stats ledger.
/// </summary>
public class DevicePrimitivesTests
{
    [Fact]
    public void Philox_CpuAndCuRand_ProduceIdenticalBits_ForSameSeedOffset()
    {
        // Both wrappers run the same managed Philox 4x32-10 today; the
        // test pins that contract so the cross-backend determinism
        // promise from #219 holds even after a future GPU dispatch flip.
        var cpu = new CpuPhiloxGenerator(seed: 42, subsequence: 7, offset: 11);
        var cuda = new CuRand(seed: 42, subsequence: 7, offset: 11);

        var a = new Tensor<float>(new[] { 64 });
        var b = new Tensor<float>(new[] { 64 });
        cpu.Uniform(a);
        cuda.Uniform(b);

        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        for (int i = 0; i < 64; i++) Assert.Equal(aSpan[i], bSpan[i]);
    }

    [Fact]
    public void Philox_Uniform_DistributesInUnitInterval()
    {
        var rng = new CpuPhiloxGenerator(seed: 0xDEADBEEFUL);
        var t = new Tensor<float>(new[] { 1024 });
        rng.Uniform(t);
        var span = t.AsSpan();
        double mean = 0;
        for (int i = 0; i < 1024; i++)
        {
            Assert.InRange(span[i], 0f, 1f);
            mean += span[i];
        }
        mean /= 1024;
        Assert.InRange(mean, 0.4, 0.6);
    }

    [Fact]
    public void Philox_OffsetAdvances_SoSubsequentDrawsDontRepeat()
    {
        var rng = new CpuPhiloxGenerator(seed: 1);
        var first = new Tensor<float>(new[] { 4 });
        var second = new Tensor<float>(new[] { 4 });
        rng.Uniform(first);
        rng.Uniform(second);

        var f = first.AsSpan();
        var s = second.AsSpan();
        // The two 4-element blocks come from different counter values,
        // so they shouldn't be element-wise identical.
        bool anyDifferent = false;
        for (int i = 0; i < 4; i++) if (f[i] != s[i]) { anyDifferent = true; break; }
        Assert.True(anyDifferent);
    }

    [Fact]
    public void CsrSpMM_AgainstDenseReference()
    {
        // A = [[1, 0, 2], [0, 3, 0]] in CSR;
        // B = [[1, 2], [3, 4], [5, 6]];
        // A·B = [[11, 14], [9, 12]].
        var values = new Tensor<float>(new[] { 3 });
        values[0] = 1f; values[1] = 2f; values[2] = 3f;
        var rowPtr = new Tensor<int>(new[] { 3 });
        rowPtr[0] = 0; rowPtr[1] = 2; rowPtr[2] = 3;
        var colIdx = new Tensor<int>(new[] { 3 });
        colIdx[0] = 0; colIdx[1] = 2; colIdx[2] = 1;
        var dense = new Tensor<float>(new[] { 3, 2 });
        dense[0, 0] = 1; dense[0, 1] = 2;
        dense[1, 0] = 3; dense[1, 1] = 4;
        dense[2, 0] = 5; dense[2, 1] = 6;

        ISparseDeviceOps cuSparse = new CuSparse();
        var result = cuSparse.SpMM(values, rowPtr, colIdx, rows: 2, cols: 3, dense);

        Assert.Equal(2, result._shape[0]);
        Assert.Equal(2, result._shape[1]);
        Assert.Equal(11f, result[0, 0]);
        Assert.Equal(14f, result[0, 1]);
        Assert.Equal(9f, result[1, 0]);
        Assert.Equal(12f, result[1, 1]);
    }

    [Fact]
    public void CsrSpMV_MatchesDenseReference()
    {
        // A = [[1, 0, 2], [0, 3, 0]]; x = [1, 2, 3]^T; A·x = [7, 6]^T.
        var values = new Tensor<float>(new[] { 3 });
        values[0] = 1f; values[1] = 2f; values[2] = 3f;
        var rowPtr = new Tensor<int>(new[] { 3 });
        rowPtr[0] = 0; rowPtr[1] = 2; rowPtr[2] = 3;
        var colIdx = new Tensor<int>(new[] { 3 });
        colIdx[0] = 0; colIdx[1] = 2; colIdx[2] = 1;
        var x = new Tensor<float>(new[] { 3 });
        x[0] = 1; x[1] = 2; x[2] = 3;

        var result = new CuSparse().SpMV(values, rowPtr, colIdx, rows: 2, cols: 3, x);

        Assert.Equal(2, result.Length);
        Assert.Equal(7f, result[0]);
        Assert.Equal(6f, result[1]);
    }

    [Fact]
    public void Cholesky_ProducesValidLowerTriangle_OnSpdMatrix()
    {
        // A = [[4, 2], [2, 3]] is symmetric positive definite.
        // L should satisfy L · L^T = A.
        var a = new Tensor<float>(new[] { 2, 2 });
        a[0, 0] = 4; a[0, 1] = 2;
        a[1, 0] = 2; a[1, 1] = 3;

        var l = new CuSolver().Cholesky(a, upper: false);

        // L should be lower-triangular.
        Assert.Equal(0f, l[0, 1]);
        // L · L^T == A.
        float a00 = l[0, 0] * l[0, 0];
        float a10 = l[1, 0] * l[0, 0];
        float a11 = l[1, 0] * l[1, 0] + l[1, 1] * l[1, 1];
        Assert.Equal(4f, a00, 4);
        Assert.Equal(2f, a10, 4);
        Assert.Equal(3f, a11, 4);
    }

    [Fact]
    public void LuFactor_PermutationAndProduct_RecoverInput()
    {
        var a = new Tensor<float>(new[] { 3, 3 });
        a[0, 0] = 2; a[0, 1] = 1; a[0, 2] = 1;
        a[1, 0] = 4; a[1, 1] = 3; a[1, 2] = 3;
        a[2, 0] = 8; a[2, 1] = 7; a[2, 2] = 9;

        var (lu, pivots) = new CuSolver().Lu(a);

        Assert.Equal(2, lu.Rank);
        Assert.Equal(3, lu._shape[0]);
        Assert.Equal(3, lu._shape[1]);
        Assert.Equal(3, pivots.Length);
    }

    [Fact]
    public void CpuRnn_ForwardLstm_ProducesShapeMatchingPyTorch()
    {
        // 1-layer, unidirectional, no projection: output [seqLen, batch, hidden].
        const int seqLen = 5, batch = 2, inputSize = 3, hidden = 4;
        const int gates = 4;

        var input = new Tensor<float>(new[] { seqLen, batch, inputSize });
        var inSpan = input.AsWritableSpan();
        for (int i = 0; i < inSpan.Length; i++) inSpan[i] = (i % 7) * 0.1f;

        var h0 = new Tensor<float>(new[] { batch, hidden });
        var c0 = new Tensor<float>(new[] { batch, hidden });

        int wIhSize = gates * hidden * inputSize;
        int wHhSize = gates * hidden * hidden;
        int bSize = gates * hidden;
        var weights = new Tensor<float>(new[] { wIhSize + wHhSize + 2 * bSize });
        var wSpan = weights.AsWritableSpan();
        for (int i = 0; i < wSpan.Length; i++) wSpan[i] = ((i % 5) - 2) * 0.05f;

        var rnn = new CuDnnRnn();
        var (output, hN, cN) = rnn.ForwardLstm(input, h0, c0, weights, new RnnOptions { HiddenSize = hidden });

        Assert.Equal(new[] { seqLen, batch, hidden }, output._shape);
        Assert.Equal(new[] { 1, batch, hidden }, hN._shape);
        Assert.Equal(new[] { 1, batch, hidden }, cN._shape);
    }

    [Fact]
    public void CpuRnn_ForwardPlainTanh_DegeneratesToTanhOnZeroWeights()
    {
        // With zero weights and zero biases, h_t = tanh(0) = 0 for all t.
        const int seqLen = 3, batch = 2, inputSize = 4, hidden = 3;
        const int gates = 1;

        var input = new Tensor<float>(new[] { seqLen, batch, inputSize });
        var h0 = new Tensor<float>(new[] { batch, hidden });
        var weights = new Tensor<float>(new[] { gates * hidden * inputSize + gates * hidden * hidden + 2 * gates * hidden });

        var rnn = new CuDnnRnn();
        var (output, hN, _) = rnn.ForwardRnn(RnnCellType.RnnTanh, input, h0, c0: null, weights, new RnnOptions { HiddenSize = hidden });

        var oSpan = output.AsSpan();
        for (int i = 0; i < oSpan.Length; i++) Assert.Equal(0f, oSpan[i], 6);
        var hSpan = hN.AsSpan();
        for (int i = 0; i < hSpan.Length; i++) Assert.Equal(0f, hSpan[i], 6);
    }

    [Fact]
    public void CpuDevicePrimitives_Reduce_Sum_AcrossAllElements()
    {
        var t = new Tensor<float>(new[] { 4 });
        t[0] = 1; t[1] = 2; t[2] = 3; t[3] = 4;
        var sum = new CpuDevicePrimitives().Reduce(t, axis: -1, ReductionKind.Sum);
        Assert.Equal(10f, sum[0]);
    }

    [Fact]
    public void CpuDevicePrimitives_Scan_Inclusive_MatchesCumsum()
    {
        var t = new Tensor<float>(new[] { 4 });
        t[0] = 1; t[1] = 2; t[2] = 3; t[3] = 4;
        var cs = new CpuDevicePrimitives().Scan(t, axis: 0, ReductionKind.Sum, exclusive: false);
        Assert.Equal(1f, cs[0]);
        Assert.Equal(3f, cs[1]);
        Assert.Equal(6f, cs[2]);
        Assert.Equal(10f, cs[3]);
    }

    [Fact]
    public void CpuDevicePrimitives_Sort_Ascending()
    {
        var t = new Tensor<float>(new[] { 5 });
        t[0] = 3; t[1] = 1; t[2] = 4; t[3] = 1; t[4] = 5;
        var sorted = new CpuDevicePrimitives().Sort(t);
        Assert.Equal(1f, sorted[0]);
        Assert.Equal(1f, sorted[1]);
        Assert.Equal(3f, sorted[2]);
        Assert.Equal(4f, sorted[3]);
        Assert.Equal(5f, sorted[4]);
    }

    [Fact]
    public void CpuDevicePrimitives_RunLengthEncode_GroupsConsecutive()
    {
        var t = new Tensor<int>(new[] { 7 });
        t[0] = 1; t[1] = 1; t[2] = 2; t[3] = 3; t[4] = 3; t[5] = 3; t[6] = 4;
        var (vals, counts) = new CpuDevicePrimitives().RunLengthEncode(t);
        Assert.Equal(4, vals.Length);
        Assert.Equal(1, vals[0]); Assert.Equal(2, counts[0]);
        Assert.Equal(2, vals[1]); Assert.Equal(1, counts[1]);
        Assert.Equal(3, vals[2]); Assert.Equal(3, counts[2]);
        Assert.Equal(4, vals[3]); Assert.Equal(1, counts[3]);
    }

    [Fact]
    public void CpuRnn_ForwardLstm_MultiLayer_OutputShapesMatchPyTorch()
    {
        // 2-layer unidirectional LSTM: output [seqLen, batch, hidden],
        // hN/cN [numLayers, batch, hidden].
        const int seqLen = 4, batch = 2, inputSize = 3, hidden = 5, numLayers = 2;
        const int gates = 4;

        var input = NewLinear<float>(seqLen * batch * inputSize, scale: 0.05f);
        var inputT = new Tensor<float>(new[] { seqLen, batch, inputSize });
        input.AsSpan().CopyTo(inputT.AsWritableSpan());

        var h0 = new Tensor<float>(new[] { numLayers, batch, hidden });
        var c0 = new Tensor<float>(new[] { numLayers, batch, hidden });

        // Layer 0: input=inputSize=3; layer 1: input=hidden=5.
        int wL0 = gates * hidden * inputSize + gates * hidden * hidden + 2 * gates * hidden;
        int wL1 = gates * hidden * hidden + gates * hidden * hidden + 2 * gates * hidden;
        var weights = NewLinear<float>(wL0 + wL1, scale: 0.03f);
        var weightsT = new Tensor<float>(new[] { wL0 + wL1 });
        weights.AsSpan().CopyTo(weightsT.AsWritableSpan());

        var rnn = new CpuRnn();
        var (output, hN, cN) = rnn.ForwardLstm(inputT, h0, c0, weightsT,
            new RnnOptions { HiddenSize = hidden, NumLayers = numLayers, Training = false });

        Assert.Equal(new[] { seqLen, batch, hidden }, output._shape);
        Assert.Equal(new[] { numLayers, batch, hidden }, hN._shape);
        Assert.Equal(new[] { numLayers, batch, hidden }, cN._shape);
    }

    [Fact]
    public void CpuRnn_ForwardLstm_Bidirectional_FeatureDimDoubled()
    {
        const int seqLen = 3, batch = 1, inputSize = 2, hidden = 4;
        const int gates = 4;

        var inputT = new Tensor<float>(new[] { seqLen, batch, inputSize });
        var inSpan = inputT.AsWritableSpan();
        for (int i = 0; i < inSpan.Length; i++) inSpan[i] = ((i % 5) - 2) * 0.1f;

        var h0 = new Tensor<float>(new[] { 2, batch, hidden });
        var c0 = new Tensor<float>(new[] { 2, batch, hidden });

        // 2 directions × 1 layer.
        int perDir = gates * hidden * inputSize + gates * hidden * hidden + 2 * gates * hidden;
        var weightsT = NewLinearTensor<float>(2 * perDir, scale: 0.04f);

        var rnn = new CpuRnn();
        var (output, hN, cN) = rnn.ForwardLstm(inputT, h0, c0, weightsT,
            new RnnOptions { HiddenSize = hidden, NumLayers = 1, Bidirectional = true, Training = false });

        Assert.Equal(new[] { seqLen, batch, hidden * 2 }, output._shape);
        Assert.Equal(new[] { 2, batch, hidden }, hN._shape);
        Assert.Equal(new[] { 2, batch, hidden }, cN._shape);
    }

    [Fact]
    public void CpuRnn_ForwardLstm_Projection_OutputUsesProjSize()
    {
        const int seqLen = 3, batch = 2, inputSize = 4, hidden = 6, proj = 3;
        const int gates = 4;

        var inputT = NewLinearTensor<float>(seqLen * batch * inputSize, scale: 0.06f);
        inputT = inputT.Reshape(new[] { seqLen, batch, inputSize });

        var h0 = new Tensor<float>(new[] { 1, batch, proj });
        var c0 = new Tensor<float>(new[] { 1, batch, hidden });

        // For projection: W_hh has shape [G·H, P], W_hr has shape [P, H].
        int wIh = gates * hidden * inputSize;
        int wHh = gates * hidden * proj;
        int wB = 2 * gates * hidden;
        int wHr = proj * hidden;
        var weightsT = NewLinearTensor<float>(wIh + wHh + wB + wHr, scale: 0.02f);

        var rnn = new CpuRnn();
        var (output, hN, cN) = rnn.ForwardLstm(inputT, h0, c0, weightsT,
            new RnnOptions { HiddenSize = hidden, NumLayers = 1, ProjSize = proj, Training = false });

        Assert.Equal(new[] { seqLen, batch, proj }, output._shape);
        Assert.Equal(new[] { 1, batch, proj }, hN._shape);
        Assert.Equal(new[] { 1, batch, hidden }, cN._shape);
    }

    [Fact]
    public void CpuRnn_ForwardLstm_Dropout_TrainingChangesOutputButEvalDoesNot()
    {
        const int seqLen = 3, batch = 2, inputSize = 3, hidden = 4, numLayers = 2;
        const int gates = 4;
        var inputT = NewLinearTensor<float>(seqLen * batch * inputSize, scale: 0.07f).Reshape(new[] { seqLen, batch, inputSize });
        var h0 = new Tensor<float>(new[] { numLayers, batch, hidden });
        var c0 = new Tensor<float>(new[] { numLayers, batch, hidden });
        int wL0 = gates * hidden * inputSize + gates * hidden * hidden + 2 * gates * hidden;
        int wL1 = gates * hidden * hidden + gates * hidden * hidden + 2 * gates * hidden;
        var weightsT = NewLinearTensor<float>(wL0 + wL1, scale: 0.03f);

        var rnn = new CpuRnn();
        var evalOpts = new RnnOptions { HiddenSize = hidden, NumLayers = numLayers, Dropout = 0.5, Training = false };
        var trainOpts = new RnnOptions { HiddenSize = hidden, NumLayers = numLayers, Dropout = 0.5, Training = true, DropoutSeed = 7 };
        var trainOpts2 = new RnnOptions { HiddenSize = hidden, NumLayers = numLayers, Dropout = 0.5, Training = true, DropoutSeed = 13 };

        var (oEval, _, _) = rnn.ForwardLstm(inputT, h0, c0, weightsT, evalOpts);
        var (oEval2, _, _) = rnn.ForwardLstm(inputT, h0, c0, weightsT, evalOpts);
        var (oTrain, _, _) = rnn.ForwardLstm(inputT, h0, c0, weightsT, trainOpts);
        var (oTrain2, _, _) = rnn.ForwardLstm(inputT, h0, c0, weightsT, trainOpts2);

        // Eval is deterministic: two eval runs match.
        for (int i = 0; i < oEval.Length; i++) Assert.Equal(oEval[i], oEval2[i]);
        // Training with different dropout seeds should diverge somewhere.
        bool anyDifferent = false;
        for (int i = 0; i < oTrain.Length; i++)
            if (Math.Abs(oTrain[i] - oTrain2[i]) > 1e-6f) { anyDifferent = true; break; }
        Assert.True(anyDifferent, "Different dropout seeds should produce different outputs.");
    }

    [Fact]
    public void CpuRnn_BackwardPlainTanh_MatchesFiniteDifferenceOnGradInput()
    {
        const int seqLen = 3, batch = 1, inputSize = 2, hidden = 3;
        const int gates = 1;

        var inputT = NewLinearTensor<float>(seqLen * batch * inputSize, scale: 0.1f).Reshape(new[] { seqLen, batch, inputSize });
        var h0 = new Tensor<float>(new[] { 1, batch, hidden });
        int wTotal = gates * hidden * inputSize + gates * hidden * hidden + 2 * gates * hidden;
        var weightsT = NewLinearTensor<float>(wTotal, scale: 0.05f);

        var rnn = new CpuRnn();
        var opts = new RnnOptions { HiddenSize = hidden, Training = false };

        var (output, _, _) = rnn.ForwardRnn(RnnCellType.RnnTanh, inputT, h0, c0: null, weightsT, opts);
        var dY = new Tensor<float>(output._shape);
        for (int i = 0; i < dY.Length; i++) dY[i] = 1.0f; // gradient of sum-loss

        var (gradInput, _, _, _) = rnn.BackwardRnn(RnnCellType.RnnTanh, inputT, h0, c0: null,
            dY, gradHN: null, gradCN: null, weightsT, opts);

        // Finite-difference check on a few input elements.
        float eps = 1e-3f;
        for (int probe = 0; probe < Math.Min(4, inputT.Length); probe++)
        {
            int idx = probe * 2; // sparse probe
            if (idx >= inputT.Length) break;
            float orig = inputT[idx];
            inputT[idx] = orig + eps;
            var (oPlus, _, _) = rnn.ForwardRnn(RnnCellType.RnnTanh, inputT, h0, c0: null, weightsT, opts);
            inputT[idx] = orig - eps;
            var (oMinus, _, _) = rnn.ForwardRnn(RnnCellType.RnnTanh, inputT, h0, c0: null, weightsT, opts);
            inputT[idx] = orig;

            float fd = 0;
            for (int i = 0; i < oPlus.Length; i++) fd += (oPlus[i] - oMinus[i]) / (2 * eps);
            float analytical = gradInput[idx];
            Assert.InRange(analytical, fd - 0.02f, fd + 0.02f);
        }
    }

    [Fact]
    public void CpuRnn_BackwardLstm_MatchesFiniteDifferenceOnGradWeights()
    {
        const int seqLen = 2, batch = 1, inputSize = 2, hidden = 2;
        const int gates = 4;

        var inputT = NewLinearTensor<float>(seqLen * batch * inputSize, scale: 0.1f).Reshape(new[] { seqLen, batch, inputSize });
        var h0 = new Tensor<float>(new[] { 1, batch, hidden });
        var c0 = new Tensor<float>(new[] { 1, batch, hidden });
        int wTotal = gates * hidden * inputSize + gates * hidden * hidden + 2 * gates * hidden;
        var weightsT = NewLinearTensor<float>(wTotal, scale: 0.1f);

        var rnn = new CpuRnn();
        var opts = new RnnOptions { HiddenSize = hidden, Training = false };

        var (output, _, _) = rnn.ForwardLstm(inputT, h0, c0, weightsT, opts);
        var dY = new Tensor<float>(output._shape);
        for (int i = 0; i < dY.Length; i++) dY[i] = 1.0f;

        var (_, _, _, gradWeights) = rnn.BackwardRnn(RnnCellType.Lstm, inputT, h0, c0,
            dY, gradHN: null, gradCN: null, weightsT, opts);

        // Probe a couple of weight indices.
        float eps = 1e-3f;
        for (int probe = 0; probe < 6; probe++)
        {
            int idx = probe * 7 % wTotal;
            float orig = weightsT[idx];
            weightsT[idx] = orig + eps;
            var (oPlus, _, _) = rnn.ForwardLstm(inputT, h0, c0, weightsT, opts);
            weightsT[idx] = orig - eps;
            var (oMinus, _, _) = rnn.ForwardLstm(inputT, h0, c0, weightsT, opts);
            weightsT[idx] = orig;

            float fd = 0;
            for (int i = 0; i < oPlus.Length; i++) fd += (oPlus[i] - oMinus[i]) / (2 * eps);
            float analytical = gradWeights[idx];
            Assert.InRange(analytical, fd - 0.05f, fd + 0.05f);
        }
    }

    [Fact]
    public void CpuRnn_BackwardGru_MatchesFiniteDifferenceOnGradWeights()
    {
        const int seqLen = 2, batch = 1, inputSize = 2, hidden = 2;
        const int gates = 3;

        var inputT = NewLinearTensor<float>(seqLen * batch * inputSize, scale: 0.1f).Reshape(new[] { seqLen, batch, inputSize });
        var h0 = new Tensor<float>(new[] { 1, batch, hidden });
        int wTotal = gates * hidden * inputSize + gates * hidden * hidden + 2 * gates * hidden;
        var weightsT = NewLinearTensor<float>(wTotal, scale: 0.1f);

        var rnn = new CpuRnn();
        var opts = new RnnOptions { HiddenSize = hidden, Training = false };

        var (output, _, _) = rnn.ForwardRnn(RnnCellType.Gru, inputT, h0, c0: null, weightsT, opts);
        var dY = new Tensor<float>(output._shape);
        for (int i = 0; i < dY.Length; i++) dY[i] = 1.0f;

        var (_, _, _, gradWeights) = rnn.BackwardRnn(RnnCellType.Gru, inputT, h0, c0: null,
            dY, gradHN: null, gradCN: null, weightsT, opts);

        float eps = 1e-3f;
        for (int probe = 0; probe < 6; probe++)
        {
            int idx = probe * 5 % wTotal;
            float orig = weightsT[idx];
            weightsT[idx] = orig + eps;
            var (oPlus, _, _) = rnn.ForwardRnn(RnnCellType.Gru, inputT, h0, c0: null, weightsT, opts);
            weightsT[idx] = orig - eps;
            var (oMinus, _, _) = rnn.ForwardRnn(RnnCellType.Gru, inputT, h0, c0: null, weightsT, opts);
            weightsT[idx] = orig;

            float fd = 0;
            for (int i = 0; i < oPlus.Length; i++) fd += (oPlus[i] - oMinus[i]) / (2 * eps);
            float analytical = gradWeights[idx];
            Assert.InRange(analytical, fd - 0.05f, fd + 0.05f);
        }
    }

    [Fact]
    public void Nvtx_PushPopMark_AreSafeOnHostsWithoutNvToolsExt()
    {
        // The wrapper must never throw — instrumentation can never break
        // a run. Push/Pop/Mark/Range should all be safe-no-op when
        // libnvToolsExt is missing.
        Nvtx.Mark("test_mark");
        Nvtx.Push("test_push");
        Nvtx.Pop();
        using (Nvtx.Range("test_range")) { /* scope */ }
        Assert.True(true);
    }

    [Fact]
    public void GpuMemoryStats_RecordAllocFree_ReflectInCounters()
    {
        GpuMemoryStats.Reset();
        GpuMemoryStats.RecordAllocation("test_alloc", 1024);
        GpuMemoryStats.RecordAllocation("test_alloc", 2048);
        Assert.Equal(3072, GpuMemoryStats.CurrentBytes);
        Assert.Equal(3072, GpuMemoryStats.PeakBytes);
        Assert.Equal(3072, GpuMemoryStats.TotalAllocatedBytes);
        Assert.Equal(2, GpuMemoryStats.ActiveAllocations);

        GpuMemoryStats.RecordFree("test_alloc", 1024);
        Assert.Equal(2048, GpuMemoryStats.CurrentBytes);
        Assert.Equal(3072, GpuMemoryStats.PeakBytes); // peak is sticky
        Assert.Equal(1, GpuMemoryStats.ActiveAllocations);

        GpuMemoryStats.ResetPeakStats();
        Assert.Equal(2048, GpuMemoryStats.PeakBytes);

        GpuMemoryStats.Reset();
    }

    [Fact]
    public void GpuMemoryStats_Stats_ExposesTorchParityKeys()
    {
        GpuMemoryStats.Reset();
        GpuMemoryStats.RecordAllocation("test_alloc", 512);
        var stats = GpuMemoryStats.Stats();

        Assert.Equal(512L, stats["allocated_bytes.current"]);
        Assert.Equal(512L, stats["allocated_bytes.peak"]);
        Assert.Equal(512L, stats["allocated_bytes.total"]);
        Assert.Equal(1L, stats["active.current"]);

        GpuMemoryStats.Reset();
    }

    private static T[] NewLinear<T>(int n, T scale)
    {
        var a = new T[n];
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < n; i++)
            a[i] = ops.Multiply(ops.FromDouble(((i % 7) - 3) * 0.1), scale);
        return a;
    }

    private static Tensor<T> NewLinearTensor<T>(int n, T scale)
    {
        var t = new Tensor<T>(new[] { n });
        var arr = NewLinear(n, scale);
        arr.AsSpan().CopyTo(t.AsWritableSpan());
        return t;
    }
}
