// Copyright (c) AiDotNet. All rights reserved.
// CPU-vs-GPU op-parity scaffold (Tensors #775). OpInfo registry.
// Phase 1 seeds the ViT-path ops implicated by #775 (patch-embed conv, matmul, softmax,
// LayerNorm, GELU/Sigmoid) to localize the CPU/GPU divergence; expands to the full IEngine
// surface from here. Tolerances: exact ops at 0 ULP; accumulation ops at a fp32 relative bound
// (matched to the 1e-3 the existing GPU parity tests use); elementwise transcendentals at a few ULP.
#if !NETFRAMEWORK

using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

public static class OpParityRegistry
{
    /// <summary>Every registered parity case (the ViT-path localization set + the broad
    /// elementwise/reduction batch). Tests and the coverage audit run over this.</summary>
    public static IEnumerable<OpCase> All() => ViTPath().Concat(Elementwise()).Concat(Elementwise2())
        .Concat(BinaryScalarShape()).Concat(ReduceNormPool()).Concat(BackwardMatmul())
        .Concat(ConvIndexLoss()).Concat(MoreMathShape()).Concat(GatedMisc()).Concat(PadDistDiag())
        .Concat(IndexComplexAudio()).Concat(NativeAudioBox()).Concat(ScatterSoftmaxMisc());

    // Scatter-reduce, top-k, more activations/losses, positional encoding, einsum.
    public static IEnumerable<OpCase> ScatterSoftmaxMisc()
    {
        var src = OpInput.Rand(1300, new[] { 4, 8 });
        var sidx = new int[32]; for (int k = 0; k < 32; k++) sidx[k] = k % 6;
        Tensor<int> Idx() => new Tensor<int>((int[])sidx.Clone(), new[] { 4, 8 });
        yield return new OpCase("ScatterMean[4,8->6,8]", "index", e => { var y = e.ScatterMean(src.F(), Idx(), out _, 0, 6); return y; }, e => { var y = e.ScatterMean(src.D(), Idx(), out _, 0, 6); return y; }, ParityTol.Ulp(16, 1e-6), opMethod: "ScatterMean");
        // ScatterMax left pending: it fills unmapped output positions with -inf (empty-group max
        // sentinel), which the harness's finiteness check rejects; needs a spec that guarantees full
        // output coverage under the exact scatter-dim semantics.
        yield return new OpCase("ScatterSoftmax[4,8]", "index", e => e.ScatterSoftmax(src.F(), Idx(), 0, 6), e => e.ScatterSoftmax(src.D(), Idx(), 0, 6), ParityTol.Accum(1e-3), opMethod: "ScatterSoftmax");

        yield return new OpCase("TensorTopK[4,8;k3,ax1]", "reduction", e => { var y = e.TensorTopK(src.F(), 3, 1, out _); return y; }, e => { var y = e.TensorTopK(src.D(), 3, 1, out _); return y; }, ParityTol.Exact, opMethod: "TensorTopK");

        var a = OpInput.Rand(1310, new[] { 4, 8 }, -4.0, 4.0);
        yield return U("TensorSELU", "activation", (e, t) => e.TensorSELU(t), (e, t) => e.TensorSELU(t), ParityTol.Ulp(64, 1e-6), a);
        yield return new OpCase("TensorPReLU[4,8]", "activation", e => e.TensorPReLU(a.F(), OpInput.Rand(1311, new[] { 8 }, 0.1, 0.5).F()), e => e.TensorPReLU(a.D(), OpInput.Rand(1311, new[] { 8 }, 0.1, 0.5).D()), ParityTol.Ulp(4, 1e-6), opMethod: "TensorPReLU");
        yield return new OpCase("TensorRReLU[4,8;eval]", "activation", e => e.TensorRReLU(a.F(), 0.125, 0.333, false), e => e.TensorRReLU(a.D(), 0.125, 0.333, false), ParityTol.Ulp(4, 1e-6), opMethod: "TensorRReLU");
        yield return new OpCase("TensorSquash[4,8]", "activation", e => e.TensorSquash(OpInput.Rand(1312, new[] { 4, 8 }).F(), -1), e => e.TensorSquash(OpInput.Rand(1312, new[] { 4, 8 }).D(), -1), ParityTol.Accum(1e-3), opMethod: "TensorSquash");
        yield return new OpCase("SphericalSoftmax[4,8]", "activation", e => e.SphericalSoftmax(OpInput.Rand(1313, new[] { 4, 8 }, -3.0, 3.0).F(), -1), e => e.SphericalSoftmax(OpInput.Rand(1313, new[] { 4, 8 }, -3.0, 3.0).D(), -1), ParityTol.Accum(1e-3), opMethod: "SphericalSoftmax");

        var pred = OpInput.Rand(1320, new[] { 4, 8 }, 0.05, 0.95);
        var tgt = OpInput.Rand(1321, new[] { 4, 8 }, 0.0, 1.0);
        yield return new OpCase("TensorBinaryCrossEntropy[4,8]", "loss", e => e.TensorBinaryCrossEntropy(pred.F(), tgt.F(), 1e-7f), e => e.TensorBinaryCrossEntropy(pred.D(), tgt.D(), 1e-7), ParityTol.Accum(1e-3), opMethod: "TensorBinaryCrossEntropy");
        yield return B("TensorCosineSimilarityLoss", "loss", (e, u, v) => e.TensorCosineSimilarityLoss(u, v), (e, u, v) => e.TensorCosineSimilarityLoss(u, v), ParityTol.Accum(1e-3), OpInput.Rand(1322, new[] { 4, 8 }), OpInput.Rand(1323, new[] { 4, 8 }));

        yield return new OpCase("PositionalEncoding[4,3;f4]", "misc", e => e.PositionalEncoding(OpInput.Rand(1330, new[] { 4, 3 }).F(), 4), e => e.PositionalEncoding(OpInput.Rand(1330, new[] { 4, 3 }).D(), 4), ParityTol.Ulp(64, 1e-6), opMethod: "PositionalEncoding");
        yield return new OpCase("TensorEinsum[ij,jk->ik]", "matmul", e => e.TensorEinsum("ij,jk->ik", OpInput.Rand(1340, new[] { 4, 8 }).F(), OpInput.Rand(1341, new[] { 8, 6 }).F()), e => e.TensorEinsum("ij,jk->ik", OpInput.Rand(1340, new[] { 4, 8 }).D(), OpInput.Rand(1341, new[] { 8, 6 }).D()), ParityTol.Accum(1e-3), opMethod: "TensorEinsum");
    }

    // Native math, audio features, and bounding-box ops.
    public static IEnumerable<OpCase> NativeAudioBox()
    {
        var s = new[] { 4, 8 };
        var x = OpInput.Rand(1200, s, -4.0, 4.0);
        yield return U("NativeExp", "arithmetic", (e, t) => e.NativeExp(t), (e, t) => e.NativeExp(t), ParityTol.Ulp(64, 1e-6), x);
        yield return U("NativeTanh", "arithmetic", (e, t) => e.NativeTanh(t), (e, t) => e.NativeTanh(t), ParityTol.Ulp(16, 1e-6), x);
        yield return B("NativeAtan2", "arithmetic", (e, u, v) => e.NativeAtan2(u, v), (e, u, v) => e.NativeAtan2(u, v), ParityTol.Ulp(64, 1e-6), x, OpInput.Rand(1201, s));
        yield return U("NativeNormalizeRows", "norm", (e, t) => e.NativeNormalizeRows(t, false), (e, t) => e.NativeNormalizeRows(t, false), ParityTol.Accum(1e-3), OpInput.Rand(1202, s));

        yield return U("AmplitudeToDB", "audio", (e, t) => e.AmplitudeToDB(t, 1e-10f, null), (e, t) => e.AmplitudeToDB(t, 1e-10f, null), ParityTol.Ulp(256, 1e-4), OpInput.RandPositive(1210, s, 0.1, 4.0));
        yield return U("ComputeDeltas", "audio", (e, t) => e.ComputeDeltas(t, 5), (e, t) => e.ComputeDeltas(t, 5), ParityTol.Accum(1e-3), OpInput.Rand(1211, new[] { 4, 16 }));
        yield return new OpCase("CreateWindow[hann;16]", "audio", e => e.CreateWindow<float>("hann", 16), e => e.CreateWindow<double>("hann", 16), ParityTol.Ulp(64, 1e-6), opMethod: "CreateWindow");

        // Bounding boxes [x1,y1,x2,y2] with positive extent.
        var boxA = OpInput.From(new double[] { 0, 0, 2, 2, 1, 1, 3, 4, 0, 1, 5, 3 }, new[] { 3, 4 });
        var boxB = OpInput.From(new double[] { 0, 0, 2, 3, 2, 2, 4, 5 }, new[] { 2, 4 });
        yield return new OpCase("BoxArea[3,4]", "box", e => e.BoxArea(boxA.F()), e => e.BoxArea(boxA.D()), ParityTol.Ulp(4, 1e-6), opMethod: "BoxArea");
        yield return new OpCase("BoxIou[3,4x2,4]", "box", e => e.BoxIou(boxA.F(), boxB.F()), e => e.BoxIou(boxA.D(), boxB.D()), ParityTol.Accum(1e-3), opMethod: "BoxIou");
        yield return new OpCase("GeneralizedBoxIou[3,4x2,4]", "box", e => e.GeneralizedBoxIou(boxA.F(), boxB.F()), e => e.GeneralizedBoxIou(boxA.D(), boxB.D()), ParityTol.Accum(1e-3), opMethod: "GeneralizedBoxIou");

        // Distances / kernels.
        yield return B("PairwiseDistance", "reduction", (e, u, v) => e.PairwiseDistance(u, v), (e, u, v) => e.PairwiseDistance(u, v), ParityTol.Accum(1e-3), OpInput.Rand(1220, s), OpInput.Rand(1221, s));
        yield return B("PairwiseDistanceSquared", "reduction", (e, u, v) => e.PairwiseDistanceSquared(u, v), (e, u, v) => e.PairwiseDistanceSquared(u, v), ParityTol.Accum(1e-3), OpInput.Rand(1222, s), OpInput.Rand(1223, s));
        yield return new OpCase("RBFKernel[4,8;c3]", "kernel", e => e.RBFKernel(OpInput.Rand(1230, s).F(), OpInput.Rand(1231, new[] { 3, 8 }).F(), OpInput.Rand(1232, new[] { 3 }, 0.5, 1.5).F()), e => e.RBFKernel(OpInput.Rand(1230, s).D(), OpInput.Rand(1231, new[] { 3, 8 }).D(), OpInput.Rand(1232, new[] { 3 }, 0.5, 1.5).D()), ParityTol.Accum(1e-3), opMethod: "RBFKernel");
    }

    // Gather/scatter (int-index), complex-magnitude, histogram, and audio resample.
    public static IEnumerable<OpCase> IndexComplexAudio()
    {
        // torch.gather: out[i][j] = source[indices[i][j]][j]; indices same shape as output.
        var gidx = new int[32]; for (int k = 0; k < 32; k++) gidx[k] = k % 4;
        yield return new OpCase("TensorGather[4,8;ax0]", "index",
            e => e.TensorGather(OpInput.Rand(1100, new[] { 4, 8 }).F(), new Tensor<int>((int[])gidx.Clone(), new[] { 4, 8 }), 0),
            e => e.TensorGather(OpInput.Rand(1100, new[] { 4, 8 }).D(), new Tensor<int>((int[])gidx.Clone(), new[] { 4, 8 }), 0), ParityTol.Exact, opMethod: "TensorGather");

        // ScatterAdd: order-independent sum into a [6,8] output.
        var sidx = new int[32]; for (int k = 0; k < 32; k++) sidx[k] = k % 6;
        yield return new OpCase("ScatterAdd[4,8->6,8;d0]", "index",
            e => e.ScatterAdd(OpInput.Rand(1101, new[] { 4, 8 }).F(), new Tensor<int>((int[])sidx.Clone(), new[] { 4, 8 }), 0, 6),
            e => e.ScatterAdd(OpInput.Rand(1101, new[] { 4, 8 }).D(), new Tensor<int>((int[])sidx.Clone(), new[] { 4, 8 }), 0, 6), ParityTol.Ulp(8, 1e-6), opMethod: "ScatterAdd");

        // Complex magnitude squared from separate real/imag tensors.
        var re = OpInput.Rand(1110, new[] { 4, 8 });
        var im = OpInput.Rand(1111, new[] { 4, 8 });
        yield return new OpCase("ComplexMagnitudeSquared[4,8]", "complex",
            e => e.ComplexMagnitudeSquared(re.F(), im.F()), e => e.ComplexMagnitudeSquared(re.D(), im.D()), ParityTol.Ulp(4, 1e-6), opMethod: "ComplexMagnitudeSquared");

        // Histogram counts (integer-valued → exact).
        yield return new OpCase("TensorHistc[4,8;b5]", "reduction",
            e => e.TensorHistc(OpInput.Rand(1120, new[] { 4, 8 }).F(), 5, -1f, 1f),
            e => e.TensorHistc(OpInput.Rand(1120, new[] { 4, 8 }).D(), 5, -1.0, 1.0), ParityTol.Exact, opMethod: "TensorHistc");

        // Audio resample (accumulation).
        yield return new OpCase("Resample[1,16;16->8]", "audio",
            e => e.Resample(OpInput.Rand(1130, new[] { 1, 16 }).F(), 16, 8),
            e => e.Resample(OpInput.Rand(1130, new[] { 1, 16 }).D(), 16, 8), ParityTol.Accum(1e-3), opMethod: "Resample");
    }

    // Padding, unfold, distances, diagonal/linalg, and reduction-backward ops.
    public static IEnumerable<OpCase> PadDistDiag()
    {
        var img = OpInput.Rand(1000, new[] { 1, 2, 4, 4 });
        yield return new OpCase("Pad[1,2,4,4;1111]", "shape", e => e.Pad(img.F(), 1, 1, 1, 1, 0f), e => e.Pad(img.D(), 1, 1, 1, 1, 0.0), ParityTol.Exact, opMethod: "Pad");
        var m = OpInput.Rand(1001, new[] { 4, 8 });
        yield return new OpCase("TensorConstantPad[4,8;1111]", "shape", e => e.TensorConstantPad(m.F(), new[] { 1, 1, 1, 1 }, 0f), e => e.TensorConstantPad(m.D(), new[] { 1, 1, 1, 1 }, 0.0), ParityTol.Exact, opMethod: "TensorConstantPad");
        yield return new OpCase("TensorUnfold[4,8;d1,4,2]", "shape", e => e.TensorUnfold(m.F(), 1, 4, 2), e => e.TensorUnfold(m.D(), 1, 4, 2), ParityTol.Exact, opMethod: "TensorUnfold");
        // FOUND (quarantined): GPU Unfold (im2col) emits its columns in a DIFFERENT element order than
        // CPU/oracle (values are a permutation — CPU matches oracle at 0 ULP, GPU is far off) — a
        // layout/convention mismatch in the GPU im2col kernel. Tracked for a separate GPU-side fix.
        yield return new OpCase("Unfold[1,3,8,8;k3,3]", "shape",
            e => e.Unfold(OpInput.Rand(1002, new[] { 1, 3, 8, 8 }).F(), new[] { 3, 3 }, new[] { 1, 1 }, new[] { 0, 0 }),
            e => e.Unfold(OpInput.Rand(1002, new[] { 1, 3, 8, 8 }).D(), new[] { 3, 3 }, new[] { 1, 1 }, new[] { 0, 0 }), ParityTol.Exact, opMethod: "Unfold")
        { KnownDivergence = "GPU Unfold/im2col emits columns in a different element order than the CPU convention." };

        var r = OpInput.Rand(1010, new[] { 4, 32 });
        yield return new OpCase("ReduceLogVariance[4,32;ax1]", "reduction", e => e.ReduceLogVariance(r.F(), new[] { 1 }, false, 1e-8), e => e.ReduceLogVariance(r.D(), new[] { 1 }, false, 1e-8), ParityTol.Accum(1e-3), opMethod: "ReduceLogVariance");
        yield return new OpCase("ReduceMeanBackward[4,1->4,8]", "reduction",
            e => e.ReduceMeanBackward(OpInput.Rand(1011, new[] { 4, 1 }).F(), new[] { 4, 8 }, new[] { 1 }),
            e => e.ReduceMeanBackward(OpInput.Rand(1011, new[] { 4, 1 }).D(), new[] { 4, 8 }, new[] { 1 }), ParityTol.Ulp(4, 1e-6), opMethod: "ReduceMeanBackward");
        yield return new OpCase("TensorLogCumSumExp[4,32;ax1]", "reduction", e => e.TensorLogCumSumExp(r.F(), 1), e => e.TensorLogCumSumExp(r.D(), 1), ParityTol.Accum(1e-3), opMethod: "TensorLogCumSumExp");
        yield return new OpCase("TensorCumMin[4,32;ax1]", "reduction", e => e.TensorCumMin(r.F(), 1), e => e.TensorCumMin(r.D(), 1), ParityTol.Exact, opMethod: "TensorCumMin");

        yield return new OpCase("TensorPDist[4,8]", "reduction", e => e.TensorPDist(OpInput.Rand(1020, new[] { 4, 8 }).F(), 2.0), e => e.TensorPDist(OpInput.Rand(1020, new[] { 4, 8 }).D(), 2.0), ParityTol.Accum(1e-3), opMethod: "TensorPDist");
        yield return new OpCase("TensorCDist[3,8x4,8]", "reduction", e => e.TensorCDist(OpInput.Rand(1021, new[] { 3, 8 }).F(), OpInput.Rand(1022, new[] { 4, 8 }).F(), 2.0), e => e.TensorCDist(OpInput.Rand(1021, new[] { 3, 8 }).D(), OpInput.Rand(1022, new[] { 4, 8 }).D(), 2.0), ParityTol.Accum(1e-3), opMethod: "TensorCDist");

        var sq = OpInput.Rand(1030, new[] { 4, 4 });
        yield return new OpCase("TensorDiagonal[4,4]", "linalg", e => e.TensorDiagonal(sq.F()), e => e.TensorDiagonal(sq.D()), ParityTol.Exact, opMethod: "TensorDiagonal");
        yield return new OpCase("TensorDiagEmbed[4]", "linalg", e => e.TensorDiagEmbed(OpInput.Rand(1031, new[] { 4 }).F(), 0), e => e.TensorDiagEmbed(OpInput.Rand(1031, new[] { 4 }).D(), 0), ParityTol.Exact, opMethod: "TensorDiagEmbed");
        yield return new OpCase("TensorBlockDiag[2x2,2]", "linalg", e => e.TensorBlockDiag(new[] { OpInput.Rand(1032, new[] { 2, 2 }).F(), OpInput.Rand(1033, new[] { 2, 2 }).F() }), e => e.TensorBlockDiag(new[] { OpInput.Rand(1032, new[] { 2, 2 }).D(), OpInput.Rand(1033, new[] { 2, 2 }).D() }), ParityTol.Exact, opMethod: "TensorBlockDiag");
        yield return new OpCase("TensorCross[4,3]", "linalg", e => e.TensorCross(OpInput.Rand(1034, new[] { 4, 3 }).F(), OpInput.Rand(1035, new[] { 4, 3 }).F(), -1), e => e.TensorCross(OpInput.Rand(1034, new[] { 4, 3 }).D(), OpInput.Rand(1035, new[] { 4, 3 }).D(), -1), ParityTol.Ulp(8, 1e-6), opMethod: "TensorCross");
        yield return new OpCase("TensorMultiDot[4,8.8,6.6,5]", "matmul", e => e.TensorMultiDot(new[] { OpInput.Rand(1036, new[] { 4, 8 }).F(), OpInput.Rand(1037, new[] { 8, 6 }).F(), OpInput.Rand(1038, new[] { 6, 5 }).F() }), e => e.TensorMultiDot(new[] { OpInput.Rand(1036, new[] { 4, 8 }).D(), OpInput.Rand(1037, new[] { 8, 6 }).D(), OpInput.Rand(1038, new[] { 6, 5 }).D() }), ParityTol.Accum(1e-3), opMethod: "TensorMultiDot");
        yield return new OpCase("TensorNormalize[4,8;ax1]", "norm", e => e.TensorNormalize(m.F(), 1, 1e-8f), e => e.TensorNormalize(m.D(), 1, 1e-8), ParityTol.Accum(1e-3), opMethod: "TensorNormalize");
    }

    // Gated activations, softmax variants, one-hot/eye, upsample/pixelshuffle, cosine-sim, stopgrad.
    public static IEnumerable<OpCase> GatedMisc()
    {
        var glu = OpInput.Rand(900, new[] { 4, 16 });
        yield return new OpCase("GLU[4,16]", "activation", e => e.GLU(glu.F(), -1), e => e.GLU(glu.D(), -1), ParityTol.Ulp(16, 1e-6), opMethod: "GLU");
        yield return new OpCase("GeGLU[4,16]", "activation", e => e.GeGLU(glu.F(), -1), e => e.GeGLU(glu.D(), -1), ParityTol.Ulp(256, 2e-5), opMethod: "GeGLU");
        yield return new OpCase("SwiGLU[4,16]", "activation", e => e.SwiGLU(glu.F(), -1), e => e.SwiGLU(glu.D(), -1), ParityTol.Ulp(64, 1e-6), opMethod: "SwiGLU");
        yield return new OpCase("ReGLU[4,16]", "activation", e => e.ReGLU(glu.F(), -1), e => e.ReGLU(glu.D(), -1), ParityTol.Ulp(2, 1e-6), opMethod: "ReGLU");

        var sm = OpInput.Rand(901, new[] { 4, 16 }, -4.0, 4.0);
        yield return new OpCase("Sparsemax[4,16]", "activation", e => e.Sparsemax(sm.F(), -1), e => e.Sparsemax(sm.D(), -1), ParityTol.Accum(1e-3), opMethod: "Sparsemax");
        yield return new OpCase("TaylorSoftmax[4,16]", "activation", e => e.TaylorSoftmax(sm.F(), 2, -1), e => e.TaylorSoftmax(sm.D(), 2, -1), ParityTol.Accum(1e-3), opMethod: "TaylorSoftmax");

        yield return new OpCase("TensorOneHot[idx4;d5]", "index",
            e => e.TensorOneHot<float>(new Tensor<int>(new[] { 0, 3, 1, 4 }, new[] { 4 }), 5),
            e => e.TensorOneHot<double>(new Tensor<int>(new[] { 0, 3, 1, 4 }, new[] { 4 }), 5), ParityTol.Exact, opMethod: "TensorOneHot");
        yield return new OpCase("TensorEye[5]", "shape", e => e.TensorEye<float>(5), e => e.TensorEye<double>(5), ParityTol.Exact, opMethod: "TensorEye");

        var img = OpInput.Rand(902, new[] { 1, 2, 4, 4 });
        yield return new OpCase("Upsample[1,2,4,4;2x2]", "shape", e => e.Upsample(img.F(), 2, 2), e => e.Upsample(img.D(), 2, 2), ParityTol.Exact, opMethod: "Upsample");
        var ps = OpInput.Rand(903, new[] { 1, 4, 4, 4 });
        yield return new OpCase("PixelShuffle[1,4,4,4;2]", "shape", e => e.PixelShuffle(ps.F(), 2), e => e.PixelShuffle(ps.D(), 2), ParityTol.Exact, opMethod: "PixelShuffle");

        var cx = OpInput.Rand(904, new[] { 4, 8 });
        var cy = OpInput.Rand(905, new[] { 4, 8 });
        yield return new OpCase("TensorCosineSimilarity[4,8]", "reduction", e => e.TensorCosineSimilarity(cx.F(), cy.F(), -1, 1e-8), e => e.TensorCosineSimilarity(cx.D(), cy.D(), -1, 1e-8), ParityTol.Accum(1e-3), opMethod: "TensorCosineSimilarity");
        yield return U("StopGradient", "shape", (e, t) => e.StopGradient(t), (e, t) => e.StopGradient(t), ParityTol.Exact, OpInput.Rand(906, new[] { 4, 64 }));
    }

    // More binary/scalar math, unary special functions, shape movement, and cumulative ops.
    public static IEnumerable<OpCase> MoreMathShape()
    {
        var s = new[] { 4, 64 };
        var a = OpInput.Rand(800, s);
        var b = OpInput.Rand(801, s);
        var ypos = OpInput.Rand(802, s, 0.3, 4.0);
        var bnz = OpInput.Rand(803, s, 0.5, 2.0);

        yield return B("TensorXlogy", "arithmetic", (e, u, v) => e.TensorXlogy(u, v), (e, u, v) => e.TensorXlogy(u, v), ParityTol.Ulp(64, 1e-6), a, ypos);
        yield return B("TensorFloatPower", "arithmetic", (e, u, v) => e.TensorFloatPower(u, v), (e, u, v) => e.TensorFloatPower(u, v), ParityTol.Ulp(256, 1e-5), ypos, OpInput.Rand(804, s, 0.5, 2.5));
        yield return B("TensorFmod", "arithmetic", (e, u, v) => e.TensorFmod(u, v), (e, u, v) => e.TensorFmod(u, v), ParityTol.Ulp(8, 1e-6), a, bnz);
        yield return B("TensorRemainder", "arithmetic", (e, u, v) => e.TensorRemainder(u, v), (e, u, v) => e.TensorRemainder(u, v), ParityTol.Ulp(8, 1e-6), a, bnz);

        yield return new OpCase("TensorLerp[4,64]", "arithmetic", e => e.TensorLerp(a.F(), b.F(), 0.3f), e => e.TensorLerp(a.D(), b.D(), 0.3), ParityTol.Ulp(4, 1e-6), opMethod: "TensorLerp");
        yield return new OpCase("TensorClip[4,64]", "arithmetic", e => e.TensorClip(a.F(), -0.5f, 0.5f), e => e.TensorClip(a.D(), -0.5, 0.5), ParityTol.Exact, opMethod: "TensorClip");
        yield return new OpCase("TensorThreshold[4,64]", "arithmetic", e => e.TensorThreshold(a.F(), 0.0f, 0.0f), e => e.TensorThreshold(a.D(), 0.0, 0.0), ParityTol.Exact, opMethod: "TensorThreshold");

        yield return U("TensorErfinv", "arithmetic", (e, t) => e.TensorErfinv(t), (e, t) => e.TensorErfinv(t), ParityTol.Ulp(256, 1e-5), OpInput.Rand(805, s, -0.9, 0.9));
        yield return U("TensorI0", "arithmetic", (e, t) => e.TensorI0(t), (e, t) => e.TensorI0(t), ParityTol.Ulp(256, 1e-5), OpInput.Rand(806, s, -2.0, 2.0));

        // Shape / movement — bit-exact.
        var m = OpInput.Rand(810, new[] { 4, 8 });
        var sq = OpInput.Rand(811, new[] { 4, 1, 8 });
        var diag = OpInput.Rand(812, new[] { 4 });
        yield return new OpCase("TensorSqueeze[4,1,8;ax1]", "shape", e => e.TensorSqueeze(sq.F(), 1), e => e.TensorSqueeze(sq.D(), 1), ParityTol.Exact, opMethod: "TensorSqueeze");
        yield return new OpCase("TensorNarrow[4,8;d1,1,4]", "shape", e => e.TensorNarrow(m.F(), 1, 1, 4), e => e.TensorNarrow(m.D(), 1, 1, 4), ParityTol.Exact, opMethod: "TensorNarrow");
        yield return new OpCase("TensorRepeatInterleave[4,8;2,d0]", "shape", e => e.TensorRepeatInterleave(m.F(), 2, 0), e => e.TensorRepeatInterleave(m.D(), 2, 0), ParityTol.Exact, opMethod: "TensorRepeatInterleave");
        yield return new OpCase("TensorTile[4,8;2,1]", "shape", e => e.TensorTile(m.F(), new[] { 2, 1 }), e => e.TensorTile(m.D(), new[] { 2, 1 }), ParityTol.Exact, opMethod: "TensorTile");
        yield return new OpCase("TensorRot90[4,8]", "shape", e => e.TensorRot90(m.F(), 1, new[] { 0, 1 }), e => e.TensorRot90(m.D(), 1, new[] { 0, 1 }), ParityTol.Exact, opMethod: "TensorRot90");
        yield return new OpCase("TensorDiag[4]", "shape", e => e.TensorDiag(diag.F()), e => e.TensorDiag(diag.D()), ParityTol.Exact, opMethod: "TensorDiag");
        yield return new OpCase("TensorFliplr[4,8]", "shape", e => e.TensorFliplr(m.F()), e => e.TensorFliplr(m.D()), ParityTol.Exact, opMethod: "TensorFliplr");
        yield return new OpCase("TensorFlipud[4,8]", "shape", e => e.TensorFlipud(m.F()), e => e.TensorFlipud(m.D()), ParityTol.Exact, opMethod: "TensorFlipud");
        yield return new OpCase("TensorRepeatElements[4,8;2,ax0]", "shape", e => e.TensorRepeatElements(m.F(), 2, 0), e => e.TensorRepeatElements(m.D(), 2, 0), ParityTol.Exact, opMethod: "TensorRepeatElements");

        // Cumulative selection.
        var rr = OpInput.Rand(820, new[] { 4, 32 });
        yield return new OpCase("TensorCumMax[4,32;ax1]", "reduction", e => e.TensorCumMax(rr.F(), 1), e => e.TensorCumMax(rr.D(), 1), ParityTol.Exact, opMethod: "TensorCumMax");
    }

    // Conv variants, index/embedding ops, losses (incl. the #775 BCE-with-logits), concat/stack.
    public static IEnumerable<OpCase> ConvIndexLoss()
    {
        // Conv variants.
        yield return new OpCase("Conv1D[1,3,16;k8,3,3]", "conv",
            e => e.Conv1D(OpInput.Rand(700, new[] { 1, 3, 16 }).F(), OpInput.Rand(701, new[] { 8, 3, 3 }).F(), 1, 0, 1),
            e => e.Conv1D(OpInput.Rand(700, new[] { 1, 3, 16 }).D(), OpInput.Rand(701, new[] { 8, 3, 3 }).D(), 1, 0, 1), ParityTol.Accum(1e-3), opMethod: "Conv1D");
        yield return new OpCase("Conv3D[1,2,4,4,4;k3,2,2,2,2]", "conv",
            e => e.Conv3D(OpInput.Rand(702, new[] { 1, 2, 4, 4, 4 }).F(), OpInput.Rand(703, new[] { 3, 2, 2, 2, 2 }).F(), 1, 0, 1),
            e => e.Conv3D(OpInput.Rand(702, new[] { 1, 2, 4, 4, 4 }).D(), OpInput.Rand(703, new[] { 3, 2, 2, 2, 2 }).D(), 1, 0, 1), ParityTol.Accum(1e-3), opMethod: "Conv3D");
        yield return new OpCase("DepthwiseConv2D[1,4,8,8;k4,1,3,3]", "conv",
            e => e.DepthwiseConv2D(OpInput.Rand(704, new[] { 1, 4, 8, 8 }).F(), OpInput.Rand(705, new[] { 4, 1, 3, 3 }).F(), new[] { 1, 1 }, new[] { 1, 1 }),
            e => e.DepthwiseConv2D(OpInput.Rand(704, new[] { 1, 4, 8, 8 }).D(), OpInput.Rand(705, new[] { 4, 1, 3, 3 }).D(), new[] { 1, 1 }, new[] { 1, 1 }), ParityTol.Accum(1e-3), opMethod: "DepthwiseConv2D");

        // Index / embedding (Tensor<int> indices are identical across float/double runs).
        yield return new OpCase("Embedding[idx4;table10,8]", "index",
            e => e.Embedding(new Tensor<int>(new[] { 1, 3, 0, 5 }, new[] { 4 }), OpInput.Rand(710, new[] { 10, 8 }).F()),
            e => e.Embedding(new Tensor<int>(new[] { 1, 3, 0, 5 }, new[] { 4 }), OpInput.Rand(710, new[] { 10, 8 }).D()), ParityTol.Exact, opMethod: "Embedding");
        yield return new OpCase("TensorIndexSelect[6,8;ax0]", "index",
            e => e.TensorIndexSelect(OpInput.Rand(711, new[] { 6, 8 }).F(), new Tensor<int>(new[] { 0, 2, 5, 1 }, new[] { 4 }), 0),
            e => e.TensorIndexSelect(OpInput.Rand(711, new[] { 6, 8 }).D(), new Tensor<int>(new[] { 0, 2, 5, 1 }, new[] { 4 }), 0), ParityTol.Exact, opMethod: "TensorIndexSelect");

        // Losses (reductions → relative tol). BCEWithLogits is the #775 loss itself.
        var pred = OpInput.Rand(720, new[] { 4, 64 });
        var tgt01 = OpInput.Rand(721, new[] { 4, 64 }, 0.0, 1.0);
        yield return new OpCase("TensorMSELoss[4,64]", "loss", e => e.TensorMSELoss(pred.F(), tgt01.F()), e => e.TensorMSELoss(pred.D(), tgt01.D()), ParityTol.Accum(1e-3), opMethod: "TensorMSELoss");
        yield return new OpCase("TensorL1Loss[4,64]", "loss", e => e.TensorL1Loss(pred.F(), tgt01.F()), e => e.TensorL1Loss(pred.D(), tgt01.D()), ParityTol.Accum(1e-3), opMethod: "TensorL1Loss");
        yield return new OpCase("TensorHuberLoss[4,64]", "loss", e => e.TensorHuberLoss(pred.F(), tgt01.F(), 1.0), e => e.TensorHuberLoss(pred.D(), tgt01.D(), 1.0), ParityTol.Accum(1e-3), opMethod: "TensorHuberLoss");
        yield return new OpCase("TensorBCEWithLogitsLoss[4,64]", "loss", e => e.TensorBCEWithLogitsLoss(pred.F(), tgt01.F()), e => e.TensorBCEWithLogitsLoss(pred.D(), tgt01.D()), ParityTol.Accum(1e-3), opMethod: "TensorBCEWithLogitsLoss");

        // Concat / stack — pure movement, bit-exact.
        var c1 = OpInput.Rand(730, new[] { 4, 8 }); var c2 = OpInput.Rand(731, new[] { 4, 8 });
        yield return new OpCase("TensorConcatenate[2x4,8;ax0]", "shape",
            e => e.TensorConcatenate(new[] { c1.F(), c2.F() }, 0), e => e.TensorConcatenate(new[] { c1.D(), c2.D() }, 0), ParityTol.Exact, opMethod: "TensorConcatenate");
        yield return new OpCase("TensorStack[2x4,8;ax0]", "shape",
            e => e.TensorStack(new[] { c1.F(), c2.F() }, 0), e => e.TensorStack(new[] { c1.D(), c2.D() }, 0), ParityTol.Exact, opMethod: "TensorStack");
        yield return new OpCase("Concat[2x4,8;ax1]", "shape",
            e => e.Concat(new[] { c1.F(), c2.F() }, 1), e => e.Concat(new[] { c1.D(), c2.D() }, 1), ParityTol.Exact, opMethod: "Concat");
    }

    // Activation backward/derivative ops and the matmul/linear family.
    public static IEnumerable<OpCase> BackwardMatmul()
    {
        var s = new[] { 4, 64 };
        var go = OpInput.Rand(600, s);
        var inp = OpInput.Rand(601, s, -4.0, 4.0);

        yield return new OpCase("MishBackward[4,64]", "activation-bwd", e => e.MishBackward(go.F(), inp.F()), e => e.MishBackward(go.D(), inp.D()), ParityTol.Ulp(64, 1e-6), opMethod: "MishBackward");
        yield return new OpCase("SwishBackward[4,64]", "activation-bwd", e => e.SwishBackward(go.F(), inp.F()), e => e.SwishBackward(go.D(), inp.D()), ParityTol.Ulp(64, 1e-6), opMethod: "SwishBackward");
        yield return new OpCase("SoftplusBackward[4,64]", "activation-bwd", e => e.SoftplusBackward(go.F(), inp.F()), e => e.SoftplusBackward(go.D(), inp.D()), ParityTol.Ulp(64, 1e-6), opMethod: "SoftplusBackward");
        yield return new OpCase("SeluBackward[4,64]", "activation-bwd", e => e.SeluBackward(go.F(), inp.F()), e => e.SeluBackward(go.D(), inp.D()), ParityTol.Ulp(64, 1e-6), opMethod: "SeluBackward");
        yield return new OpCase("HardswishBackward[4,64]", "activation-bwd", e => e.HardswishBackward(go.F(), inp.F()), e => e.HardswishBackward(go.D(), inp.D()), ParityTol.Ulp(16, 1e-6), opMethod: "HardswishBackward");
        yield return new OpCase("HardsigmoidBackward[4,64]", "activation-bwd", e => e.HardsigmoidBackward(go.F(), inp.F()), e => e.HardsigmoidBackward(go.D(), inp.D()), ParityTol.Ulp(16, 1e-6), opMethod: "HardsigmoidBackward");
        // #775: CPU GeluBackward now evaluates its derivative via the accurate Padé sigmoid (same
        // fix family as the forward), so it matches the GPU builtin-tanh derivative tightly.
        yield return new OpCase("GeluBackward[4,64]", "activation-bwd", e => e.GeluBackward(go.F(), inp.F()), e => e.GeluBackward(go.D(), inp.D()), ParityTol.Accum(1e-3), opMethod: "GeluBackward");
        yield return new OpCase("Relu6Backward[4,64]", "activation-bwd", e => e.Relu6Backward(go.F(), inp.F()), e => e.Relu6Backward(go.D(), inp.D()), ParityTol.Exact, opMethod: "Relu6Backward");
        yield return new OpCase("LeakyReluBackward[4,64]", "activation-bwd", e => e.LeakyReluBackward(go.F(), inp.F(), 0.1), e => e.LeakyReluBackward(go.D(), inp.D(), 0.1), ParityTol.Ulp(4, 1e-6), opMethod: "LeakyReluBackward");
        yield return new OpCase("EluBackward[4,64]", "activation-bwd",
            e => e.EluBackward(go.F(), inp.F(), e.ELU(inp.F(), 1.0), 1.0), e => e.EluBackward(go.D(), inp.D(), e.ELU(inp.D(), 1.0), 1.0), ParityTol.Ulp(64, 1e-6), opMethod: "EluBackward");

        yield return new OpCase("ReLUDerivative[4,64]", "activation-bwd", e => e.ReLUDerivative(inp.F()), e => e.ReLUDerivative(inp.D()), ParityTol.Exact, opMethod: "ReLUDerivative");
        yield return new OpCase("SigmoidDerivative[4,64]", "activation-bwd", e => e.SigmoidDerivative(OpInput.Rand(602, s, 0.1, 0.9).F()), e => e.SigmoidDerivative(OpInput.Rand(602, s, 0.1, 0.9).D()), ParityTol.Ulp(8, 1e-6), opMethod: "SigmoidDerivative");
        yield return new OpCase("TanhDerivative[4,64]", "activation-bwd", e => e.TanhDerivative(OpInput.Rand(603, s, -0.9, 0.9).F()), e => e.TanhDerivative(OpInput.Rand(603, s, -0.9, 0.9).D()), ParityTol.Ulp(8, 1e-6), opMethod: "TanhDerivative");

        // Matmul / linear family.
        var ba = OpInput.Rand(610, new[] { 2, 4, 8 });
        var bb = OpInput.Rand(611, new[] { 2, 8, 6 });
        yield return new OpCase("BatchMatMul[2,4,8x2,8,6]", "matmul", e => e.BatchMatMul(ba.F(), bb.F()), e => e.BatchMatMul(ba.D(), bb.D()), ParityTol.Accum(1e-3), opMethod: "BatchMatMul");
        yield return new OpCase("TensorBatchMatMul[2,4,8x2,8,6]", "matmul", e => e.TensorBatchMatMul(ba.F(), bb.F()), e => e.TensorBatchMatMul(ba.D(), bb.D()), ParityTol.Accum(1e-3), opMethod: "TensorBatchMatMul");
        var ta = OpInput.Rand(612, new[] { 4, 8 });
        var tbt = OpInput.Rand(613, new[] { 6, 8 });
        yield return new OpCase("TensorMatMulTransposed[4,8x6,8]", "matmul", e => e.TensorMatMulTransposed(ta.F(), tbt.F()), e => e.TensorMatMulTransposed(ta.D(), tbt.D()), ParityTol.Accum(1e-3), opMethod: "TensorMatMulTransposed");
        var mmIn = OpInput.Rand(614, new[] { 4, 6 });
        var mmA = OpInput.Rand(615, new[] { 4, 8 });
        var mmB = OpInput.Rand(616, new[] { 8, 6 });
        yield return new OpCase("TensorAddMM[4,6;4,8x8,6]", "matmul", e => e.TensorAddMM(mmIn.F(), mmA.F(), mmB.F()), e => e.TensorAddMM(mmIn.D(), mmA.D(), mmB.D()), ParityTol.Accum(1e-3), opMethod: "TensorAddMM");
        yield return new OpCase("TensorDot[4,8.8,6]", "matmul", e => e.TensorDot(mmA.F(), mmB.F(), new[] { 1 }, new[] { 0 }), e => e.TensorDot(mmA.D(), mmB.D(), new[] { 1 }, new[] { 0 }), ParityTol.Accum(1e-3), opMethod: "TensorDot");
        yield return new OpCase("TensorOuter[4x6]", "matmul", e => e.TensorOuter(OpInput.Rand(617, new[] { 4 }).F(), OpInput.Rand(618, new[] { 6 }).F()), e => e.TensorOuter(OpInput.Rand(617, new[] { 4 }).D(), OpInput.Rand(618, new[] { 6 }).D()), ParityTol.Ulp(2, 1e-6), opMethod: "TensorOuter");
        yield return new OpCase("TensorKron[2,3x2,2]", "matmul", e => e.TensorKron(OpInput.Rand(619, new[] { 2, 3 }).F(), OpInput.Rand(620, new[] { 2, 2 }).F()), e => e.TensorKron(OpInput.Rand(619, new[] { 2, 3 }).D(), OpInput.Rand(620, new[] { 2, 2 }).D()), ParityTol.Ulp(2, 1e-6), opMethod: "TensorKron");

        // Variadic reductions over a small tensor list.
        var m1 = OpInput.Rand(630, s); var m2 = OpInput.Rand(631, s); var m3 = OpInput.Rand(632, s);
        yield return new OpCase("TensorAddMany[3x4,64]", "arithmetic", e => e.TensorAddMany(m1.F(), m2.F(), m3.F()), e => e.TensorAddMany(m1.D(), m2.D(), m3.D()), ParityTol.Ulp(4, 1e-6), opMethod: "TensorAddMany");
        var n1 = OpInput.Rand(633, s, 0.8, 1.2); var n2 = OpInput.Rand(634, s, 0.8, 1.2); var n3 = OpInput.Rand(635, s, 0.8, 1.2);
        yield return new OpCase("TensorMultiplyMany[3x4,64]", "arithmetic", e => e.TensorMultiplyMany(n1.F(), n2.F(), n3.F()), e => e.TensorMultiplyMany(n1.D(), n2.D(), n3.D()), ParityTol.Ulp(8, 1e-6), opMethod: "TensorMultiplyMany");
    }

    // Reductions / cumulative, comparison, norm family, and pooling.
    public static IEnumerable<OpCase> ReduceNormPool()
    {
        // Reductions & cumulative.
        var r = OpInput.Rand(500, new[] { 4, 32 });
        yield return new OpCase("TensorCumSum[4,32;ax1]", "reduction", e => e.TensorCumSum(r.F(), 1), e => e.TensorCumSum(r.D(), 1), ParityTol.Accum(1e-3), opMethod: "TensorCumSum");
        yield return new OpCase("TensorCumProd[4,32;ax1]", "reduction", e => e.TensorCumProd(OpInput.Rand(501, new[] { 4, 32 }, 0.8, 1.2).F(), 1), e => e.TensorCumProd(OpInput.Rand(501, new[] { 4, 32 }, 0.8, 1.2).D(), 1), ParityTol.Accum(1e-3), opMethod: "TensorCumProd");
        yield return new OpCase("TensorNorm[4,32;ax1]", "reduction", e => e.TensorNorm(r.F(), 1, false), e => e.TensorNorm(r.D(), 1, false), ParityTol.Accum(1e-3), opMethod: "TensorNorm");
        yield return new OpCase("TensorStd[4,32]", "reduction", e => e.TensorStd(r.F()), e => e.TensorStd(r.D()), ParityTol.Accum(1e-3), opMethod: "TensorStd");
        yield return new OpCase("TensorVar[4,32]", "reduction", e => e.TensorVar(r.F()), e => e.TensorVar(r.D()), ParityTol.Accum(1e-3), opMethod: "TensorVar");
        yield return new OpCase("ReduceStd[4,32;ax1]", "reduction", e => e.ReduceStd(r.F(), new[] { 1 }, false), e => e.ReduceStd(r.D(), new[] { 1 }, false), ParityTol.Accum(1e-3), opMethod: "ReduceStd");
        yield return new OpCase("ReduceVariance[4,32;ax1]", "reduction", e => e.ReduceVariance(r.F(), new[] { 1 }, false), e => e.ReduceVariance(r.D(), new[] { 1 }, false), ParityTol.Accum(1e-3), opMethod: "ReduceVariance");
        yield return new OpCase("TensorLogSumExp[4,32;ax1]", "reduction", e => e.TensorLogSumExp(r.F(), 1, false), e => e.TensorLogSumExp(r.D(), 1, false), ParityTol.Accum(1e-3), opMethod: "TensorLogSumExp");

        // Comparison / selection — 0/1 or exact-select, bit-exact.
        var a = OpInput.Rand(510, new[] { 4, 64 });
        var b = OpInput.Rand(511, new[] { 4, 64 });
        yield return B("TensorGreaterThan", "comparison", (e, u, v) => e.TensorGreaterThan(u, v), (e, u, v) => e.TensorGreaterThan(u, v), ParityTol.Exact, a, b);
        yield return B("TensorLessThan", "comparison", (e, u, v) => e.TensorLessThan(u, v), (e, u, v) => e.TensorLessThan(u, v), ParityTol.Exact, a, b);
        yield return new OpCase("TensorNotEquals[4,64]", "comparison", e => e.TensorNotEquals(a.F(), 0.5f), e => e.TensorNotEquals(a.D(), 0.5), ParityTol.Exact, opMethod: "TensorNotEquals");
        {
            // Where(cond, x, y): cond is a 0/1 tensor.
            var condData = new double[256];
            var rng = new Random(512);
            for (int i = 0; i < condData.Length; i++) condData[i] = rng.NextDouble() < 0.5 ? 0.0 : 1.0;
            var cond = OpInput.From(condData, new[] { 4, 64 });
            yield return new OpCase("TensorWhere[4,64]", "comparison",
                e => e.TensorWhere(cond.F(), a.F(), b.F()), e => e.TensorWhere(cond.D(), a.D(), b.D()), ParityTol.Exact, opMethod: "TensorWhere");
        }

        // Norm family.
        var nx = OpInput.Rand(520, new[] { 4, 64 });
        var g = OpInput.Rand(521, new[] { 64 }, 0.5, 1.5);
        var be = OpInput.Rand(522, new[] { 64 }, -0.2, 0.2);
        yield return new OpCase("TensorLayerNorm[4,64]", "norm", e => e.TensorLayerNorm(nx.F(), g.F(), be.F(), 1e-5), e => e.TensorLayerNorm(nx.D(), g.D(), be.D(), 1e-5), ParityTol.Accum(1e-3), opMethod: "TensorLayerNorm");
        yield return new OpCase("RMSNorm[4,64]", "norm", e => { var y = e.RMSNorm(nx.F(), g.F(), 1e-5, out _); return y; }, e => { var y = e.RMSNorm(nx.D(), g.D(), 1e-5, out _); return y; }, ParityTol.Accum(1e-3), opMethod: "RMSNorm");
        {
            var gx = OpInput.Rand(523, new[] { 2, 8, 4, 4 });
            var gg = OpInput.Rand(524, new[] { 8 }, 0.5, 1.5);
            var gb = OpInput.Rand(525, new[] { 8 }, -0.2, 0.2);
            yield return new OpCase("GroupNorm[2,8,4,4;g2]", "norm", e => { var y = e.GroupNorm(gx.F(), 2, gg.F(), gb.F(), 1e-5, out _, out _); return y; }, e => { var y = e.GroupNorm(gx.D(), 2, gg.D(), gb.D(), 1e-5, out _, out _); return y; }, ParityTol.Accum(1e-3), opMethod: "GroupNorm");
            yield return new OpCase("InstanceNorm[2,8,4,4]", "norm", e => { var y = e.InstanceNorm(gx.F(), gg.F(), gb.F(), 1e-5, out _, out _); return y; }, e => { var y = e.InstanceNorm(gx.D(), gg.D(), gb.D(), 1e-5, out _, out _); return y; }, ParityTol.Accum(1e-3), opMethod: "InstanceNorm");
        }

        // Pooling (NCHW).
        var p = OpInput.Rand(530, new[] { 1, 4, 8, 8 });
        yield return new OpCase("MaxPool2D[1,4,8,8;k2]", "pool", e => e.MaxPool2D(p.F(), 2), e => e.MaxPool2D(p.D(), 2), ParityTol.Exact, opMethod: "MaxPool2D");
        yield return new OpCase("AvgPool2D[1,4,8,8;k2]", "pool", e => e.AvgPool2D(p.F(), 2), e => e.AvgPool2D(p.D(), 2), ParityTol.Accum(1e-3), opMethod: "AvgPool2D");
        yield return new OpCase("GlobalAvgPool2D[1,4,8,8]", "pool", e => e.GlobalAvgPool2D(p.F()), e => e.GlobalAvgPool2D(p.D()), ParityTol.Accum(1e-3), opMethod: "GlobalAvgPool2D");
        yield return new OpCase("GlobalMaxPool2D[1,4,8,8]", "pool", e => e.GlobalMaxPool2D(p.F()), e => e.GlobalMaxPool2D(p.D()), ParityTol.Exact, opMethod: "GlobalMaxPool2D");
    }

    // Binary elementwise, scalar-arg, and shape ops (the last are pure movement → bit-exact).
    public static IEnumerable<OpCase> BinaryScalarShape()
    {
        var s = new[] { 4, 64 };
        var a = OpInput.Rand(400, s);
        var b = OpInput.Rand(401, s);
        var bpos = OpInput.Rand(402, s, 0.5, 3.0);
        var brow = OpInput.Rand(403, new[] { 1, 64 });

        // Binary (two same-shape tensors).
        yield return B("TensorMax", "arithmetic", (e, u, v) => e.TensorMax(u, v), (e, u, v) => e.TensorMax(u, v), ParityTol.Exact, a, b);
        yield return B("TensorMin", "arithmetic", (e, u, v) => e.TensorMin(u, v), (e, u, v) => e.TensorMin(u, v), ParityTol.Exact, a, b);
        yield return B("TensorHypot", "arithmetic", (e, u, v) => e.TensorHypot(u, v), (e, u, v) => e.TensorHypot(u, v), ParityTol.Ulp(16, 1e-6), a, b);
        yield return B("TensorCopysign", "arithmetic", (e, u, v) => e.TensorCopysign(u, v), (e, u, v) => e.TensorCopysign(u, v), ParityTol.Exact, a, b);
        yield return B("TensorLogAddExp", "arithmetic", (e, u, v) => e.TensorLogAddExp(u, v), (e, u, v) => e.TensorLogAddExp(u, v), ParityTol.Ulp(64, 1e-6), a, b);

        // Broadcast binary (row broadcast [1,64] over [4,64]).
        yield return B("TensorBroadcastAdd", "arithmetic", (e, u, v) => e.TensorBroadcastAdd(u, v), (e, u, v) => e.TensorBroadcastAdd(u, v), ParityTol.Exact, a, brow);
        yield return B("TensorBroadcastSubtract", "arithmetic", (e, u, v) => e.TensorBroadcastSubtract(u, v), (e, u, v) => e.TensorBroadcastSubtract(u, v), ParityTol.Exact, a, brow);
        yield return B("TensorBroadcastMultiply", "arithmetic", (e, u, v) => e.TensorBroadcastMultiply(u, v), (e, u, v) => e.TensorBroadcastMultiply(u, v), ParityTol.Ulp(2, 1e-6), a, brow);
        yield return B("TensorBroadcastDivide", "arithmetic", (e, u, v) => e.TensorBroadcastDivide(u, v), (e, u, v) => e.TensorBroadcastDivide(u, v), ParityTol.Ulp(4, 1e-6), a, OpInput.Rand(404, new[] { 1, 64 }, 0.5, 3.0));

        // Scalar-arg ops.
        yield return new OpCase("TensorAddScalar[4,64]", "arithmetic", e => e.TensorAddScalar(a.F(), 1.5f), e => e.TensorAddScalar(a.D(), 1.5), ParityTol.Exact, opMethod: "TensorAddScalar");
        yield return new OpCase("TensorSubtractScalar[4,64]", "arithmetic", e => e.TensorSubtractScalar(a.F(), 1.5f), e => e.TensorSubtractScalar(a.D(), 1.5), ParityTol.Exact, opMethod: "TensorSubtractScalar");
        yield return new OpCase("TensorMultiplyScalar[4,64]", "arithmetic", e => e.TensorMultiplyScalar(a.F(), 1.5f), e => e.TensorMultiplyScalar(a.D(), 1.5), ParityTol.Ulp(2, 1e-6), opMethod: "TensorMultiplyScalar");
        yield return new OpCase("TensorDivideScalar[4,64]", "arithmetic", e => e.TensorDivideScalar(a.F(), 1.5f), e => e.TensorDivideScalar(a.D(), 1.5), ParityTol.Ulp(4, 1e-6), opMethod: "TensorDivideScalar");
        yield return new OpCase("TensorClampMin[4,64]", "arithmetic", e => e.TensorClampMin(a.F(), -0.5f), e => e.TensorClampMin(a.D(), -0.5), ParityTol.Exact, opMethod: "TensorClampMin");
        yield return new OpCase("TensorClampMax[4,64]", "arithmetic", e => e.TensorClampMax(a.F(), 0.5f), e => e.TensorClampMax(a.D(), 0.5), ParityTol.Exact, opMethod: "TensorClampMax");
        yield return new OpCase("TensorClamp[4,64]", "arithmetic", e => e.TensorClamp(a.F(), -0.5f, 0.5f), e => e.TensorClamp(a.D(), -0.5, 0.5), ParityTol.Exact, opMethod: "TensorClamp");
        yield return new OpCase("TensorPow[4,64]", "arithmetic", e => e.TensorPow(bpos.F(), 2.0f), e => e.TensorPow(bpos.D(), 2.0), ParityTol.Ulp(64, 1e-6), opMethod: "TensorPow");

        // Shape / movement ops — bit-exact (no arithmetic).
        var m = OpInput.Rand(410, new[] { 4, 8 });
        yield return new OpCase("TensorFlatten[4,8]", "shape", e => e.TensorFlatten(m.F()), e => e.TensorFlatten(m.D()), ParityTol.Exact, opMethod: "TensorFlatten");
        yield return new OpCase("TensorTranspose[4,8]", "shape", e => e.TensorTranspose(m.F()), e => e.TensorTranspose(m.D()), ParityTol.Exact, opMethod: "TensorTranspose");
        yield return new OpCase("TensorFlip[4,8;ax0]", "shape", e => e.TensorFlip(m.F(), new[] { 0 }), e => e.TensorFlip(m.D(), new[] { 0 }), ParityTol.Exact, opMethod: "TensorFlip");
        yield return new OpCase("TensorTril[4,8]", "shape", e => e.TensorTril(m.F(), 0), e => e.TensorTril(m.D(), 0), ParityTol.Exact, opMethod: "TensorTril");
        yield return new OpCase("TensorTriu[4,8]", "shape", e => e.TensorTriu(m.F(), 0), e => e.TensorTriu(m.D(), 0), ParityTol.Exact, opMethod: "TensorTriu");
        yield return new OpCase("TensorRoll[4,8]", "shape", e => e.TensorRoll(m.F(), new[] { 1 }, new[] { 0 }), e => e.TensorRoll(m.D(), new[] { 1 }, new[] { 0 }), ParityTol.Exact, opMethod: "TensorRoll");
        yield return new OpCase("TensorMoveDim[4,8]", "shape", e => e.TensorMoveDim(m.F(), 0, 1), e => e.TensorMoveDim(m.D(), 0, 1), ParityTol.Exact, opMethod: "TensorMoveDim");
        yield return new OpCase("TensorSwapAxes[4,8]", "shape", e => e.TensorSwapAxes(m.F(), 0, 1), e => e.TensorSwapAxes(m.D(), 0, 1), ParityTol.Exact, opMethod: "TensorSwapAxes");
        yield return new OpCase("TensorExpandDims[4,8;ax0]", "shape", e => e.TensorExpandDims(m.F(), 0), e => e.TensorExpandDims(m.D(), 0), ParityTol.Exact, opMethod: "TensorExpandDims");
        yield return new OpCase("TensorBroadcastTo[1,64->4,64]", "shape", e => e.TensorBroadcastTo(brow.F(), new[] { 4, 64 }), e => e.TensorBroadcastTo(brow.D(), new[] { 4, 64 }), ParityTol.Exact, opMethod: "TensorBroadcastTo");
    }

    // ---- batch helpers for the uniform op families -------------------------------------------
    private static string Dims(OpInput i) => string.Join(",", i.Shape);

    /// <summary>Unary elementwise op: Op(tensor).</summary>
    private static OpCase U(
        string method, string cat, System.Func<IEngine, Tensor<float>, Tensor<float>> f,
        System.Func<IEngine, Tensor<double>, Tensor<double>> d, ParityTol tol, OpInput x)
        => new OpCase($"{method}[{Dims(x)}]", cat, e => f(e, x.F()), e => d(e, x.D()), tol, opMethod: method);

    /// <summary>Binary elementwise op: Op(a, b).</summary>
    private static OpCase B(
        string method, string cat, System.Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>> f,
        System.Func<IEngine, Tensor<double>, Tensor<double>, Tensor<double>> d, ParityTol tol, OpInput a, OpInput b)
        => new OpCase($"{method}[{Dims(a)}]", cat, e => f(e, a.F(), b.F()), e => d(e, a.D(), b.D()), tol, opMethod: method);

    public static IEnumerable<OpCase> Elementwise()
    {
        var s = new[] { 4, 64 };

        // Bounded-range activations (transcendental → a handful of ULP).
        var act = OpInput.Rand(200, s, -6.0, 6.0);
        yield return U("Tanh", "activation", (e, t) => e.Tanh(t), (e, t) => e.Tanh(t), ParityTol.Ulp(16), act);
        yield return U("ReLU", "activation", (e, t) => e.ReLU(t), (e, t) => e.ReLU(t), ParityTol.Exact, act);
        yield return U("Mish", "activation", (e, t) => e.Mish(t), (e, t) => e.Mish(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("Swish", "activation", (e, t) => e.Swish(t), (e, t) => e.Swish(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("Softplus", "activation", (e, t) => e.Softplus(t), (e, t) => e.Softplus(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("HardSwish", "activation", (e, t) => e.HardSwish(t), (e, t) => e.HardSwish(t), ParityTol.Ulp(16, 1e-6), act);
        yield return U("ELU", "activation", (e, t) => e.ELU(t, 1.0), (e, t) => e.ELU(t, 1.0), ParityTol.Ulp(64, 1e-6), act);
        yield return new OpCase($"LeakyReLU[{Dims(act)}]", "activation",
            e => e.LeakyReLU(act.F(), 0.1f), e => e.LeakyReLU(act.D(), 0.1), ParityTol.Ulp(4, 1e-6), opMethod: "LeakyReLU");

        // Elementwise math.
        yield return U("TensorNegate", "arithmetic", (e, t) => e.TensorNegate(t), (e, t) => e.TensorNegate(t), ParityTol.Exact, act);
        yield return U("TensorAbs", "arithmetic", (e, t) => e.TensorAbs(t), (e, t) => e.TensorAbs(t), ParityTol.Exact, act);
        yield return U("TensorSquare", "arithmetic", (e, t) => e.TensorSquare(t), (e, t) => e.TensorSquare(t), ParityTol.Ulp(2, 1e-6), act);
        yield return U("TensorSin", "arithmetic", (e, t) => e.TensorSin(t), (e, t) => e.TensorSin(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("TensorCos", "arithmetic", (e, t) => e.TensorCos(t), (e, t) => e.TensorCos(t), ParityTol.Ulp(64, 1e-6), act);
        yield return U("TensorExp", "arithmetic", (e, t) => e.TensorExp(t), (e, t) => e.TensorExp(t), ParityTol.Ulp(64, 1e-6),
            OpInput.Rand(201, s, -4.0, 4.0));
        yield return U("TensorLog", "arithmetic", (e, t) => e.TensorLog(t), (e, t) => e.TensorLog(t), ParityTol.Ulp(64, 1e-6),
            OpInput.RandPositive(202, s, 0.1, 8.0));
        yield return U("TensorSqrt", "arithmetic", (e, t) => e.TensorSqrt(t), (e, t) => e.TensorSqrt(t), ParityTol.Ulp(8, 1e-6),
            OpInput.RandPositive(203, s, 0.01, 8.0));

        // Binary elementwise.
        var a = OpInput.Rand(210, s);
        var b = OpInput.Rand(211, s);
        var bnz = OpInput.Rand(212, s, 0.5, 2.0); // non-zero divisor
        yield return B("TensorSubtract", "arithmetic", (e, x, y) => e.TensorSubtract(x, y), (e, x, y) => e.TensorSubtract(x, y), ParityTol.Exact, a, b);
        yield return B("TensorMultiply", "arithmetic", (e, x, y) => e.TensorMultiply(x, y), (e, x, y) => e.TensorMultiply(x, y), ParityTol.Ulp(2, 1e-6), a, b);
        yield return B("TensorDivide", "arithmetic", (e, x, y) => e.TensorDivide(x, y), (e, x, y) => e.TensorDivide(x, y), ParityTol.Ulp(4, 1e-6), a, bnz);

        // Reductions (summation order differs → relative bound; Max is selection → tight).
        var red = OpInput.Rand(220, new[] { 4, 32 });
        yield return new OpCase("ReduceSum[4,32]", "reduction",
            e => e.ReduceSum(red.F(), null, false), e => e.ReduceSum(red.D(), null, false), ParityTol.Accum(1e-3), opMethod: "ReduceSum");
        yield return new OpCase("ReduceMean[4,32;ax1]", "reduction",
            e => e.ReduceMean(red.F(), new[] { 1 }, false), e => e.ReduceMean(red.D(), new[] { 1 }, false), ParityTol.Accum(1e-3), opMethod: "ReduceMean");
        yield return new OpCase("ReduceMax[4,32;ax1]", "reduction",
            e => e.ReduceMax(red.F(), new[] { 1 }, false, out _), e => e.ReduceMax(red.D(), new[] { 1 }, false, out _), ParityTol.Ulp(0, 1e-6), opMethod: "ReduceMax");
    }

    // Second elementwise batch: the broad Tensor* unary math/activation family + softmax variants.
    public static IEnumerable<OpCase> Elementwise2()
    {
        var s = new[] { 4, 64 };
        var x = OpInput.Rand(300, s, -4.0, 4.0);
        var pos = OpInput.RandPositive(301, s, 0.2, 6.0);

        // Rounding / sign — exact (integer-valued results, identical scalar math).
        yield return U("TensorFloor", "arithmetic", (e, t) => e.TensorFloor(t), (e, t) => e.TensorFloor(t), ParityTol.Exact, x);
        yield return U("TensorCeiling", "arithmetic", (e, t) => e.TensorCeiling(t), (e, t) => e.TensorCeiling(t), ParityTol.Exact, x);
        yield return U("TensorRound", "arithmetic", (e, t) => e.TensorRound(t), (e, t) => e.TensorRound(t), ParityTol.Exact, x);
        yield return U("TensorSign", "arithmetic", (e, t) => e.TensorSign(t), (e, t) => e.TensorSign(t), ParityTol.Exact, x);
        // #775: was quarantined — the GPU frac_kernel under-wrote its output (scalar kernel launched
        // with size/4 work items by ExecuteActivation, which assumes float4) leaving 3/4 uninitialized
        // → nondeterministic garbage. Fixed by vectorizing frac_kernel to float4; now bit-clean.
        yield return U("TensorFrac", "arithmetic", (e, t) => e.TensorFrac(t), (e, t) => e.TensorFrac(t), ParityTol.Ulp(4, 1e-6), x);
        yield return U("TensorReciprocal", "arithmetic", (e, t) => e.TensorReciprocal(t), (e, t) => e.TensorReciprocal(t), ParityTol.Ulp(8, 1e-6), pos);

        // Transcendental unary math.
        yield return U("TensorCosh", "arithmetic", (e, t) => e.TensorCosh(t), (e, t) => e.TensorCosh(t), ParityTol.Ulp(64, 1e-6), x);
        yield return U("TensorSinh", "arithmetic", (e, t) => e.TensorSinh(t), (e, t) => e.TensorSinh(t), ParityTol.Ulp(64, 1e-6), x);
        yield return U("TensorErfc", "arithmetic", (e, t) => e.TensorErfc(t), (e, t) => e.TensorErfc(t), ParityTol.Ulp(256, 1e-5), x);
        yield return U("TensorLgamma", "arithmetic", (e, t) => e.TensorLgamma(t), (e, t) => e.TensorLgamma(t), ParityTol.Ulp(256, 1e-5), pos);
        yield return U("TensorDigamma", "arithmetic", (e, t) => e.TensorDigamma(t), (e, t) => e.TensorDigamma(t), ParityTol.Ulp(256, 1e-5), pos);

        // Activation family (Tensor* variants).
        yield return U("TensorTanh", "activation", (e, t) => e.TensorTanh(t), (e, t) => e.TensorTanh(t), ParityTol.Ulp(16, 1e-6), x);
        yield return U("TensorSigmoid", "activation", (e, t) => e.TensorSigmoid(t), (e, t) => e.TensorSigmoid(t), ParityTol.Ulp(16, 1e-6), x);
        yield return U("TensorGELU", "activation", (e, t) => e.TensorGELU(t), (e, t) => e.TensorGELU(t), ParityTol.Ulp(256, 2e-5), x);
        yield return U("TensorMish", "activation", (e, t) => e.TensorMish(t), (e, t) => e.TensorMish(t), ParityTol.Ulp(64, 1e-6), x);
        yield return U("TensorReLU", "activation", (e, t) => e.TensorReLU(t), (e, t) => e.TensorReLU(t), ParityTol.Exact, x);
        yield return U("TensorReLU6", "activation", (e, t) => e.TensorReLU6(t), (e, t) => e.TensorReLU6(t), ParityTol.Exact, x);
        yield return U("TensorSiLU", "activation", (e, t) => e.TensorSiLU(t), (e, t) => e.TensorSiLU(t), ParityTol.Ulp(64, 1e-6), x);
        yield return U("TensorHardSigmoid", "activation", (e, t) => e.TensorHardSigmoid(t), (e, t) => e.TensorHardSigmoid(t), ParityTol.Ulp(8, 1e-6), x);
        yield return U("TensorHardSwish", "activation", (e, t) => e.TensorHardSwish(t), (e, t) => e.TensorHardSwish(t), ParityTol.Ulp(8, 1e-6), x);

        // Softmax family (accumulation over an axis).
        var sm = OpInput.Rand(302, s, -4.0, 4.0);
        yield return new OpCase("TensorSoftmax[4,64]", "activation",
            e => e.TensorSoftmax(sm.F(), -1), e => e.TensorSoftmax(sm.D(), -1), ParityTol.Accum(1e-3), opMethod: "TensorSoftmax");
        yield return new OpCase("TensorLogSoftmax[4,64]", "activation",
            e => e.TensorLogSoftmax(sm.F(), -1), e => e.TensorLogSoftmax(sm.D(), -1), ParityTol.Accum(1e-3), opMethod: "TensorLogSoftmax");
        yield return new OpCase("TensorSoftmaxRows[4,64]", "activation",
            e => e.TensorSoftmaxRows(sm.F()), e => e.TensorSoftmaxRows(sm.D()), ParityTol.Accum(1e-3), opMethod: "TensorSoftmaxRows");
    }

    public static IEnumerable<OpCase> ViTPath()
    {
        // --- Elementwise, must be bit-identical (identical scalar math, no accumulation) ---
        {
            var a = OpInput.Rand(1, new[] { 2, 64 });
            var b = OpInput.Rand(2, new[] { 2, 64 });
            yield return new OpCase("Add[2,64]", "arithmetic",
                e => e.TensorAdd(a.F(), b.F()), e => e.TensorAdd(a.D(), b.D()), ParityTol.Exact, opMethod: "TensorAdd");
        }
        {
            var a = OpInput.Rand(3, new[] { 2, 64 });
            yield return new OpCase("Reshape[2,64->128]", "shape",
                e => e.Reshape(a.F(), new[] { 128 }), e => e.Reshape(a.D(), new[] { 128 }), ParityTol.Exact, opMethod: "Reshape");
        }

        // --- Accumulation ops (summation order differs across engines → relative bound) ---
        {
            var a = OpInput.Rand(4, new[] { 8, 32 });
            var b = OpInput.Rand(5, new[] { 32, 16 });
            yield return new OpCase("MatMul[8x32x16]", "matmul",
                e => e.TensorMatMul(a.F(), b.F()), e => e.TensorMatMul(a.D(), b.D()), ParityTol.Accum(1e-3), opMethod: "TensorMatMul");
        }
        {
            var a = OpInput.Rand(6, new[] { 4, 16 }, -4.0, 4.0);
            yield return new OpCase("Softmax[4,16]", "activation",
                e => e.Softmax(a.F(), -1), e => e.Softmax(a.D(), -1), ParityTol.Accum(1e-3));
        }
        {
            var x = OpInput.Rand(9, new[] { 4, 64 });
            var g = OpInput.Rand(10, new[] { 64 }, 0.5, 1.5);
            var b = OpInput.Rand(11, new[] { 64 }, -0.2, 0.2);
            yield return new OpCase("LayerNorm[4,64]", "norm",
                e => { var r = e.LayerNorm(x.F(), g.F(), b.F(), 1e-5, out _, out _); return r; },
                e => { var r = e.LayerNorm(x.D(), g.D(), b.D(), 1e-5, out _, out _); return r; },
                ParityTol.Accum(1e-3));
        }
        {
            // Patch-embed proxy: [1,3,16,16] conv with a 4×4 stride-4 kernel → [1,8,4,4].
            var x = OpInput.Rand(12, new[] { 1, 3, 16, 16 });
            var k = OpInput.Rand(13, new[] { 8, 3, 4, 4 });
            yield return new OpCase("Conv2D[1,3,16,16;k8x3x4x4;s4]", "conv",
                e => e.Conv2D(x.F(), k.F(), 4, 0, 1), e => e.Conv2D(x.D(), k.D(), 4, 0, 1), ParityTol.Accum(1e-3), opMethod: "Conv2D");
        }

        // --- Elementwise transcendentals (a few ULP of per-element rounding) ---
        {
            // NOTE (#775): the CPU and GPU GELU disagree by up to ~7e-6 in the near-zero flat
            // region (GPU flushes small outputs to 0 vs CPU's erf-exact tiny value; the oracle
            // shows GPU drifts further from truth). That is negligible in isolation but is exactly
            // the kind of small ViT forward delta BCE-with-logits amplifies — hence the wider
            // near-zero floor here, WITH the report capturing the ULP/oracle drift for follow-up.
            var a = OpInput.Rand(7, new[] { 4, 64 }, -6.0, 6.0);
            yield return new OpCase("GELU[4,64]", "activation",
                e => e.GELU(a.F()), e => e.GELU(a.D()), ParityTol.Ulp(256, 2e-5));
        }
        {
            // GELUInto is the COMPILED / GraphMode training-path GELU kernel (the one #775's
            // diverging CpuEngine training runs). Deliberately includes near-zero inputs where the
            // old 2·sigmoid(2z)−1 cancellation corrupted the result. Now unified onto the stable
            // x·sigmoid kernel, so it must match the eager GELU + oracle tightly.
            var a = OpInput.Rand(7, new[] { 4, 64 }, -6.0, 6.0);
            yield return new OpCase("GELUInto[4,64]", "activation",
                e => { var dst = new Tensor<float>((int[])a.Shape.Clone()); e.GELUInto(dst, a.F()); return dst; },
                e => { var dst = new Tensor<double>((int[])a.Shape.Clone()); e.GELUInto(dst, a.D()); return dst; },
                ParityTol.Ulp(256, 2e-5), opMethod: "GELU"); // credits GELU; GELUInto is a void Into-variant
        }
        {
            var a = OpInput.Rand(8, new[] { 4, 64 }, -8.0, 8.0);
            yield return new OpCase("Sigmoid[4,64]", "activation",
                e => e.Sigmoid(a.F()), e => e.Sigmoid(a.D()), ParityTol.Ulp(64, 1e-6));
        }

        // --- Backward ops are first-class IEngine ops; their CPU/GPU parity IS backward parity ---
        {
            var go = OpInput.Rand(20, new[] { 4, 64 });
            var input = OpInput.Rand(21, new[] { 4, 64 });
            yield return new OpCase("ReluBackward[4,64]", "activation-bwd",
                e => e.ReluBackward(go.F(), input.F()), e => e.ReluBackward(go.D(), input.D()), ParityTol.Ulp(4, 1e-6));
        }
        {
            var go = OpInput.Rand(22, new[] { 4, 64 });
            var outp = OpInput.Rand(23, new[] { 4, 64 }, 0.05, 0.95);
            yield return new OpCase("SigmoidBackward[4,64]", "activation-bwd",
                e => e.SigmoidBackward(go.F(), outp.F()), e => e.SigmoidBackward(go.D(), outp.D()), ParityTol.Ulp(16, 1e-6));
        }
        {
            var go = OpInput.Rand(24, new[] { 4, 64 });
            var outp = OpInput.Rand(25, new[] { 4, 64 }, -0.95, 0.95);
            yield return new OpCase("TanhBackward[4,64]", "activation-bwd",
                e => e.TanhBackward(go.F(), outp.F()), e => e.TanhBackward(go.D(), outp.D()), ParityTol.Ulp(16, 1e-6));
        }
        {
            // Softmax backward needs a valid softmax output (rows sum to 1) as its second arg.
            var go = OpInput.Rand(26, new[] { 4, 16 });
            var logits = OpInput.Rand(27, new[] { 4, 16 }, -4.0, 4.0);
            yield return new OpCase("SoftmaxBackward[4,16]", "activation-bwd",
                e => e.SoftmaxBackward(go.F(), e.Softmax(logits.F(), -1), -1),
                e => e.SoftmaxBackward(go.D(), e.Softmax(logits.D(), -1), -1),
                ParityTol.Accum(1e-3));
        }
    }
}
#endif
