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
        .Concat(IndexComplexAudio()).Concat(NativeAudioBox()).Concat(ScatterSoftmaxMisc()).Concat(ConvPoolLinear())
        .Concat(SpecialAttnNorm()).Concat(SlicePoolTake()).Concat(GridConvBwdLoss()).Concat(SliceScatterMisc())
        .Concat(NormConvBackward()).Concat(StackIndexEmbed()).Concat(FusedLinAffine()).Concat(GluCropSoftmaxBwd())
        .Concat(MoreBackward()).Concat(AudioFftSplat()).Concat(GeometryNerf()).Concat(FusedRoiLoss())
        .Concat(Conv3DBoxIou()).Concat(SortConvInterp()).Concat(AttentionFused())
        .Concat(ScalarShapePad()).Concat(ComplexReal()).Concat(TensorMathBatch())
        .Concat(FusedConvMlp()).Concat(GatherScatterPool()).Concat(SdpaScatterUnique())
        .Concat(RecurrentScans()).Concat(MoreScansComplex()).Concat(AudioSpectral());

    // FFT-based audio features (MFCC / wideband / mel-spectrogram).
    public static IEnumerable<OpCase> AudioSpectral()
    {
        var wave = OpInput.Rand(3800, new[] { 2, 512 }, -1, 1);
        yield return new OpCase("NativeMfccFeatures[2,512;seg4;mfcc13]", "audio",
            e => e.NativeMfccFeatures(wave.F(), 4, 13, 256), e => e.NativeMfccFeatures(wave.D(), 4, 13, 256),
            ParityTol.Accum(2e-3), opMethod: "NativeMfccFeatures");
        // FOUND (quarantined): GPU wideband-feature FFT kernel diverges massively (CPU 3 ULP vs
        // oracle, GPU ~48M ULP / 2.24 abs).
        yield return new OpCase("NativeWidebandFeatures[2,512;seg4;bins20]", "audio",
            e => e.NativeWidebandFeatures(wave.F(), 4, 20), e => e.NativeWidebandFeatures(wave.D(), 4, 20),
            ParityTol.Accum(2e-3), opMethod: "NativeWidebandFeatures")
        { KnownDivergence = "GPU wideband-feature FFT kernel diverges ~2.24 abs; CPU matches oracle." };

        // Mel-spectrogram of a [1,128] signal, nFft 16, hop 8, 8 mels.
        // FOUND (quarantined): GPU emits a different frame count (120 vs CPU/oracle 136) -> STFT
        // framing/centering mismatch on the GPU path.
        var mwave = OpInput.Rand(3810, new[] { 1, 128 }, -1, 1);
        var window = OpInput.RandPositive(3811, new[] { 16 }, 0.2, 1.0);
        yield return new OpCase("MelSpectrogram[1,128;nfft16;mels8]", "audio",
            e => e.MelSpectrogram(mwave.F(), 16000, 16, 8, 8, 0f, 8000f, window.F(), true),
            e => e.MelSpectrogram(mwave.D(), 16000, 16, 8, 8, 0.0, 8000.0, window.D(), true),
            ParityTol.Accum(2e-3), opMethod: "MelSpectrogram")
        { KnownDivergence = "GPU mel-spectrogram emits a different frame count (STFT framing mismatch)." };
    }

    // More linear-attention scans (RWKV-7, xLSTM, gated-delta-net) + scatter + interleaved complex.
    public static IEnumerable<OpCase> MoreScansComplex()
    {
        const int B = 2, S = 4, D = 6, H = 2;
        var seqShape = new[] { B, S, D };
        var gShape = new[] { B, S, H };

        var r7 = OpInput.Rand(3700, seqShape);
        var k7 = OpInput.Rand(3701, seqShape);
        var v7 = OpInput.Rand(3702, seqShape);
        var a7 = OpInput.Rand(3703, seqShape);
        var b7 = OpInput.Rand(3704, seqShape);
        yield return new OpCase("Rwkv7SequenceForward[2,4,6;h2]", "scan",
            e => e.Rwkv7SequenceForward(r7.F(), k7.F(), v7.F(), a7.F(), b7.F(), H),
            e => e.Rwkv7SequenceForward(r7.D(), k7.D(), v7.D(), a7.D(), b7.D(), H),
            ParityTol.Accum(1e-3), opMethod: "Rwkv7SequenceForward");

        var xq = OpInput.Rand(3710, seqShape);
        var xk = OpInput.Rand(3711, seqShape);
        var xv = OpInput.Rand(3712, seqShape);
        var xi = OpInput.Rand(3713, gShape, 0.0, 2.0);
        var xf = OpInput.Rand(3714, gShape, 0.0, 1.0);
        var xo = OpInput.Rand(3715, gShape, 0.0, 1.0);
        yield return new OpCase("XLstmScanForward[2,4,6;h2]", "scan",
            e => e.XLstmScanForward(xq.F(), xk.F(), xv.F(), xi.F(), xf.F(), xo.F(), H),
            e => e.XLstmScanForward(xq.D(), xk.D(), xv.D(), xi.D(), xf.D(), xo.D(), H),
            ParityTol.Accum(1e-3), opMethod: "XLstmScanForward");

        var dq = OpInput.Rand(3720, seqShape);
        var dk = OpInput.Rand(3721, seqShape);
        var dv = OpInput.Rand(3722, seqShape);
        var dalpha = OpInput.Rand(3723, gShape, 0.0, 1.0);
        var dbeta = OpInput.Rand(3724, gShape, 0.0, 1.0);
        yield return new OpCase("GatedDeltaNetScanForward[2,4,6;h2]", "scan",
            e => e.GatedDeltaNetScanForward(dq.F(), dk.F(), dv.F(), dalpha.F(), dbeta.F(), H),
            e => e.GatedDeltaNetScanForward(dq.D(), dk.D(), dv.D(), dalpha.D(), dbeta.D(), H),
            ParityTol.Accum(1e-3), opMethod: "GatedDeltaNetScanForward");

        // Scatter: flat 1-D indices applied per outer slice; values shaped [outer, L, inner].
        // Distinct permutation over axis-1 (length 6) -> deterministic, values [4,6].
        var sd = OpInput.Rand(3730, new[] { 4, 6 });
        var sv = OpInput.Rand(3731, new[] { 4, 6 });
        var sIdx = new int[] { 0, 1, 2, 3, 4, 5 };
        yield return new OpCase("Scatter[4,6;idx6;axis1]", "index",
            e => e.Scatter(sd.F(), new Tensor<int>((int[])sIdx.Clone(), new[] { 6 }), sv.F(), 1),
            e => e.Scatter(sd.D(), new Tensor<int>((int[])sIdx.Clone(), new[] { 6 }), sv.D(), 1),
            ParityTol.Exact, opMethod: "Scatter");

        // Interleaved-complex magnitude/multiply ([...,re,im] pairs in a real tensor).
        var ca = OpInput.Rand(3740, new[] { 4, 6 });
        var cb = OpInput.Rand(3741, new[] { 4, 6 });
        yield return new OpCase("TensorComplexMagnitude[4,6]", "complex",
            e => e.TensorComplexMagnitude(ca.F()), e => e.TensorComplexMagnitude(ca.D()),
            ParityTol.Accum(1e-3), opMethod: "TensorComplexMagnitude");
        yield return new OpCase("TensorComplexMultiply[4,6]", "complex",
            e => e.TensorComplexMultiply(ca.F(), cb.F()), e => e.TensorComplexMultiply(ca.D(), cb.D()),
            ParityTol.Ulp(8), opMethod: "TensorComplexMultiply");
    }

    // Linear-attention / SSM / RNN sequence-scan forwards (batch=2, seq=4, dim=6).
    public static IEnumerable<OpCase> RecurrentScans()
    {
        const int B = 2, S = 4, D = 6, ST = 3, H = 2;
        var seqShape = new[] { B, S, D };

        // Mamba selective scan: x/delta [B,S,D], aLog [D,ST], b/c [B,S,ST], d [D].
        var mx = OpInput.Rand(3600, seqShape);
        var mdelta = OpInput.RandPositive(3601, seqShape, 0.1, 1.0);
        var maLog = OpInput.Rand(3602, new[] { D, ST });
        var mb = OpInput.Rand(3603, new[] { B, S, ST });
        var mc = OpInput.Rand(3604, new[] { B, S, ST });
        var md = OpInput.Rand(3605, new[] { D });
        // FOUND (GPU-unsafe): the OpenCL mamba_selective_scan_forward kernel allocates two private
        // float[256] arrays per work-item; on this device the launch errors and poisons the OpenCL
        // command queue so a LATER unrelated GPU op crashes the host process. Passes in isolation,
        // crashes the full run. Skipped before GPU execution until the kernel's private-memory use
        // is bounded to the actual stateDim.
        yield return new OpCase("MambaSelectiveScanForward[2,4,6]", "scan",
            e => e.MambaSelectiveScanForward(mx.F(), mdelta.F(), maLog.F(), mb.F(), mc.F(), md.F()),
            e => e.MambaSelectiveScanForward(mx.D(), mdelta.D(), maLog.D(), mb.D(), mc.D(), md.D()),
            ParityTol.Accum(1e-3), opMethod: "MambaSelectiveScanForward")
        { GpuUnsafe = true, KnownDivergence = "OpenCL mamba kernel over-allocates private memory -> command-queue error crashes a later GPU op." };
        // Mamba2 SSD: delta [B,S,H], aLog/D per-head [H], shared B/C [B,S,ST].
        var m2delta = OpInput.RandPositive(3606, new[] { B, S, H }, 0.1, 1.0);
        var m2aLog = OpInput.Rand(3607, new[] { H });
        var m2d = OpInput.Rand(3608, new[] { H });
        yield return new OpCase("Mamba2SsdScanForward[2,4,6;h2]", "scan",
            e => e.Mamba2SsdScanForward(mx.F(), m2delta.F(), m2aLog.F(), mb.F(), mc.F(), m2d.F(), H),
            e => e.Mamba2SsdScanForward(mx.D(), m2delta.D(), m2aLog.D(), mb.D(), mc.D(), m2d.D(), H),
            ParityTol.Accum(1e-3), opMethod: "Mamba2SsdScanForward");

        // GLA: q/k/v [B,S,D], gate [B,S,H] in (0,1).
        var gq = OpInput.Rand(3610, seqShape);
        var gk = OpInput.Rand(3611, seqShape);
        var gv = OpInput.Rand(3612, seqShape);
        var gg = OpInput.Rand(3613, new[] { B, S, H }, 0.0, 1.0);
        yield return new OpCase("GlaScanForward[2,4,6;h2]", "scan",
            e => e.GlaScanForward(gq.F(), gk.F(), gv.F(), gg.F(), H),
            e => e.GlaScanForward(gq.D(), gk.D(), gv.D(), gg.D(), H),
            ParityTol.Accum(1e-3), opMethod: "GlaScanForward");

        // RWKV-4 WKV: r/k/v [B,S,D], timeDecay/timeFirst [D].
        var rr = OpInput.Rand(3620, seqShape);
        var rk = OpInput.Rand(3621, seqShape);
        var rv = OpInput.Rand(3622, seqShape);
        var td = OpInput.Rand(3623, new[] { D }, 0.1, 0.9);
        var tf = OpInput.Rand(3624, new[] { D }, 0.1, 0.9);
        yield return new OpCase("Rwkv4WkvForward[2,4,6]", "scan",
            e => e.Rwkv4WkvForward(rr.F(), rk.F(), rv.F(), td.F(), tf.F()),
            e => e.Rwkv4WkvForward(rr.D(), rk.D(), rv.D(), td.D(), tf.D()),
            ParityTol.Accum(1e-3), opMethod: "Rwkv4WkvForward");

        // RG-LRU: value [B,S,D], recGate/inpGate [B,S,D] in (0,1), decay [D] in (0,1).
        var lv = OpInput.Rand(3630, seqShape);
        var lrg = OpInput.Rand(3631, seqShape, 0.0, 1.0);
        var lig = OpInput.Rand(3632, seqShape, 0.0, 1.0);
        var ldec = OpInput.Rand(3633, new[] { D }, 0.5, 0.95);
        yield return new OpCase("RgLruScanForward[2,4,6]", "scan",
            e => e.RgLruScanForward(lv.F(), lrg.F(), lig.F(), ldec.F()),
            e => e.RgLruScanForward(lv.D(), lrg.D(), lig.D(), ldec.D()),
            ParityTol.Accum(1e-3), opMethod: "RgLruScanForward");

        // LSTM sequence: input [B,S,inF=8], wIh [4*hid,inF], wHh [4*hid,hid], hid=4.
        var lstmIn = OpInput.Rand(3640, new[] { B, S, 8 });
        var wIh = OpInput.Rand(3641, new[] { 16, 8 });
        var wHh = OpInput.Rand(3642, new[] { 16, 4 });
        yield return new OpCase("LstmSequenceForward[2,4,8;hid4]", "scan",
            e => e.LstmSequenceForward(lstmIn.F(), null, null, wIh.F(), wHh.F(), null, null, false),
            e => e.LstmSequenceForward(lstmIn.D(), null, null, wIh.D(), wHh.D(), null, null, false),
            ParityTol.Accum(1e-3), opMethod: "LstmSequenceForward");
    }

    // Scaled-dot-product attention, scatter-max (all outputs covered), unique.
    public static IEnumerable<OpCase> SdpaScatterUnique()
    {
        // SDPA expects 4D q/k/v [batch, heads, seq, d_k].
        var q = OpInput.Rand(3500, new[] { 1, 2, 4, 8 });
        var k = OpInput.Rand(3501, new[] { 1, 2, 4, 8 });
        var val = OpInput.Rand(3502, new[] { 1, 2, 4, 8 });
        yield return new OpCase("ScaledDotProductAttention[4,8]", "attention",
            e => e.ScaledDotProductAttention(q.F(), k.F(), val.F(), null, null, out _),
            e => e.ScaledDotProductAttention(q.D(), k.D(), val.D(), null, null, out _),
            ParityTol.Accum(1e-3), opMethod: "ScaledDotProductAttention");

        // ScatterMax: every output index covered (0..3) so no empty-group -inf; max is deterministic.
        var smSrc = OpInput.Rand(3510, new[] { 6 });
        var smIdx = new int[] { 0, 1, 2, 3, 0, 1 };
        yield return new OpCase("ScatterMax[6->4]", "index",
            e => e.ScatterMax(smSrc.F(), new Tensor<int>((int[])smIdx.Clone(), new[] { 6 }), out _, 0, 4),
            e => e.ScatterMax(smSrc.D(), new Tensor<int>((int[])smIdx.Clone(), new[] { 6 }), out _, 0, 4),
            ParityTol.Exact, opMethod: "ScatterMax");

        // Unique of a strictly-distinct input (sorted) -> output equals sorted input, same length.
        var uniq = OpInput.Rand(3520, new[] { 12 });
        yield return new OpCase("TensorUnique[12;sorted]", "index",
            e => e.TensorUnique(uniq.F(), true), e => e.TensorUnique(uniq.D(), true),
            ParityTol.Exact, opMethod: "TensorUnique");
    }

    // Gather/scatter with explicit int indices + max-pool-with-indices.
    public static IEnumerable<OpCase> GatherScatterPool()
    {
        var src = OpInput.Rand(3400, new[] { 4, 6 });
        // Per-row distinct column indices (axis 1) so scatter is order-independent/deterministic.
        var gIdx = new int[] { 0, 2, 4, 1, 3, 5, 0, 2, 5, 1, 3, 4 };
        yield return new OpCase("Gather[4,6;idx4,3;axis1]", "index",
            e => e.Gather(src.F(), new Tensor<int>((int[])gIdx.Clone(), new[] { 4, 3 }), 1),
            e => e.Gather(src.D(), new Tensor<int>((int[])gIdx.Clone(), new[] { 4, 3 }), 1),
            ParityTol.Exact, opMethod: "Gather");

        var dest = OpInput.Rand(3410, new[] { 4, 6 });
        var upd = OpInput.Rand(3411, new[] { 4, 3 });
        yield return new OpCase("TensorScatter[4,6;idx4,3;axis1]", "index",
            e => e.TensorScatter(dest.F(), new Tensor<int>((int[])gIdx.Clone(), new[] { 4, 3 }), upd.F(), 1),
            e => e.TensorScatter(dest.D(), new Tensor<int>((int[])gIdx.Clone(), new[] { 4, 3 }), upd.D(), 1),
            ParityTol.Exact, opMethod: "TensorScatter");
        // index_add semantics: 1-D indices select rows (axis 0); distinct rows -> deterministic.
        var addUpd = OpInput.Rand(3412, new[] { 3, 6 });
        var addIdx = new int[] { 0, 2, 3 };
        yield return new OpCase("TensorScatterAdd[4,6;idx3;axis0]", "index",
            e => e.TensorScatterAdd(dest.F(), new Tensor<int>((int[])addIdx.Clone(), new[] { 3 }), addUpd.F(), 0),
            e => e.TensorScatterAdd(dest.D(), new Tensor<int>((int[])addIdx.Clone(), new[] { 3 }), addUpd.D(), 0),
            ParityTol.Ulp(4), opMethod: "TensorScatterAdd");

        var selIdx = new int[] { 0, 2, 3 };
        yield return new OpCase("TensorIndexSelectDiff[4,6;idx3;axis0]", "index",
            e => e.TensorIndexSelectDiff(src.F(), new Tensor<int>((int[])selIdx.Clone(), new[] { 3 }), 0),
            e => e.TensorIndexSelectDiff(src.D(), new Tensor<int>((int[])selIdx.Clone(), new[] { 3 }), 0),
            ParityTol.Exact, opMethod: "TensorIndexSelectDiff");

        var poolIn = OpInput.Rand(3420, new[] { 1, 2, 8, 8 });
        yield return new OpCase("MaxPool2DWithIndices[1,2,8,8;2x2]", "pool",
            e => e.MaxPool2DWithIndices(poolIn.F(), new[] { 2, 2 }, new[] { 2, 2 }, out _),
            e => e.MaxPool2DWithIndices(poolIn.D(), new[] { 2, 2 }, new[] { 2, 2 }, out _),
            ParityTol.Exact, opMethod: "MaxPool2DWithIndices");
    }

    // Fused conv2d/3d/transpose, convT3d backward, mel filterbank, MLP forward.
    public static IEnumerable<OpCase> FusedConvMlp()
    {
        // FusedConv2D: input [1,2,8,8], kernel [3,2,3,3], bias [3], stride1 pad1 -> [1,3,8,8].
        var c2In = OpInput.Rand(3300, new[] { 1, 2, 8, 8 });
        var c2K = OpInput.Rand(3301, new[] { 3, 2, 3, 3 });
        var c2B = OpInput.Rand(3302, new[] { 3 });
        yield return new OpCase("FusedConv2D[1,2,8,8;k3,2,3,3]", "conv",
            e => e.FusedConv2D(c2In.F(), c2K.F(), c2B.F(), 1, 1, 1, 1, 1, 1, FusedActivationType.None),
            e => e.FusedConv2D(c2In.D(), c2K.D(), c2B.D(), 1, 1, 1, 1, 1, 1, FusedActivationType.None),
            ParityTol.Accum(1e-3), opMethod: "FusedConv2D");

        // FusedConv3D: input [1,2,4,4,4], kernel [3,2,2,2,2], bias [3], stride1 pad0 -> [1,3,3,3,3].
        var c3In = OpInput.Rand(3310, new[] { 1, 2, 4, 4, 4 });
        var c3K = OpInput.Rand(3311, new[] { 3, 2, 2, 2, 2 });
        var c3B = OpInput.Rand(3312, new[] { 3 });
        yield return new OpCase("FusedConv3D[1,2,4,4,4;k3,2,2,2,2]", "conv",
            e => e.FusedConv3D(c3In.F(), c3K.F(), c3B.F(), 1, 1, 1, 0, 0, 0, 1, 1, 1, FusedActivationType.None),
            e => e.FusedConv3D(c3In.D(), c3K.D(), c3B.D(), 1, 1, 1, 0, 0, 0, 1, 1, 1, FusedActivationType.None),
            ParityTol.Accum(1e-3), opMethod: "FusedConv3D");

        // FusedConvTranspose2D: input [1,2,4,4], kernel [2,3,2,2], stride2 -> [1,3,8,8].
        var ctIn = OpInput.Rand(3320, new[] { 1, 2, 4, 4 });
        var ctK = OpInput.Rand(3321, new[] { 2, 3, 2, 2 });
        var ctB = OpInput.Rand(3322, new[] { 3 });
        yield return new OpCase("FusedConvTranspose2D[1,2,4,4;k2,3,2,2]", "conv",
            e => e.FusedConvTranspose2D(ctIn.F(), ctK.F(), ctB.F(), 2, 2, 0, 0, 0, 0, FusedActivationType.None),
            e => e.FusedConvTranspose2D(ctIn.D(), ctK.D(), ctB.D(), 2, 2, 0, 0, 0, 0, FusedActivationType.None),
            ParityTol.Accum(1e-3), opMethod: "FusedConvTranspose2D");

        // ConvTranspose3D backward: convT3d input [1,2,2,2,2] kernel [2,3,2,2,2] stride2 -> [1,3,4,4,4].
        var t3Go = OpInput.Rand(3330, new[] { 1, 3, 4, 4, 4 });
        var t3In = OpInput.Rand(3331, new[] { 1, 2, 2, 2, 2 });
        var t3K = OpInput.Rand(3332, new[] { 2, 3, 2, 2, 2 });
        yield return new OpCase("ConvTranspose3DBackwardInput[go1,3,4,4,4;k2,3,2,2,2]", "conv",
            e => e.ConvTranspose3DBackwardInput(t3Go.F(), t3K.F(), new[] { 1, 2, 2, 2, 2 }, new[] { 2, 2, 2 }, new[] { 0, 0, 0 }),
            e => e.ConvTranspose3DBackwardInput(t3Go.D(), t3K.D(), new[] { 1, 2, 2, 2, 2 }, new[] { 2, 2, 2 }, new[] { 0, 0, 0 }),
            ParityTol.Accum(1e-3), opMethod: "ConvTranspose3DBackwardInput");
        yield return new OpCase("ConvTranspose3DBackwardKernel[go1,3,4,4,4;in1,2,2,2,2]", "conv",
            e => e.ConvTranspose3DBackwardKernel(t3Go.F(), t3In.F(), new[] { 2, 3, 2, 2, 2 }, new[] { 2, 2, 2 }, new[] { 0, 0, 0 }),
            e => e.ConvTranspose3DBackwardKernel(t3Go.D(), t3In.D(), new[] { 2, 3, 2, 2, 2 }, new[] { 2, 2, 2 }, new[] { 0, 0, 0 }),
            ParityTol.Accum(1e-3), opMethod: "ConvTranspose3DBackwardKernel");

        // Mel filterbank (no tensor input) — deterministic triangular filters.
        yield return new OpCase("CreateMelFilterbank[8,16]", "audio",
            e => e.CreateMelFilterbank<float>(8, 16, 16000, 0f, 8000f),
            e => e.CreateMelFilterbank<double>(8, 16, 16000, 0.0, 8000.0),
            ParityTol.Accum(1e-3), opMethod: "CreateMelFilterbank");

        // MLP forward: input [4,8] -> [8,16] -> [16,6], no bias, None activations.
        var mlpIn = OpInput.Rand(3340, new[] { 4, 8 });
        var w1 = OpInput.Rand(3341, new[] { 8, 16 });
        var w2 = OpInput.Rand(3342, new[] { 16, 6 });
        yield return new OpCase("MlpForward[4,8;8,16;16,6]", "matmul",
            e => e.MlpForward(mlpIn.F(), new[] { w1.F(), w2.F() }, new Tensor<float>?[] { null, null }, FusedActivationType.None, FusedActivationType.None),
            e => e.MlpForward(mlpIn.D(), new[] { w1.D(), w2.D() }, new Tensor<double>?[] { null, null }, FusedActivationType.None, FusedActivationType.None),
            ParityTol.Accum(1e-3), opMethod: "MlpForward");
    }

    // Assorted Tensor* math: leaky-relu, power, inner/outer, expand, CIoU loss, upsample, tri-mask, zeta.
    public static IEnumerable<OpCase> TensorMathBatch()
    {
        var a = OpInput.Rand(3200, new[] { 4, 6 });
        var b = OpInput.Rand(3201, new[] { 4, 6 });
        yield return new OpCase("TensorLeakyReLU[4,6;0.01]", "activation",
            e => e.TensorLeakyReLU(a.F(), 0.01f), e => e.TensorLeakyReLU(a.D(), 0.01),
            ParityTol.Ulp(2), opMethod: "TensorLeakyReLU");
        yield return new OpCase("TensorPower[4,6;^2]", "elementwise",
            e => e.TensorPower(a.F(), 2f), e => e.TensorPower(a.D(), 2.0),
            ParityTol.Ulp(8), opMethod: "TensorPower");
        yield return new OpCase("TensorInner[4,6;4,6]", "matmul",
            e => e.TensorInner(a.F(), b.F()), e => e.TensorInner(a.D(), b.D()),
            ParityTol.Accum(1e-3), opMethod: "TensorInner");

        var u = OpInput.Rand(3210, new[] { 4 });
        var v = OpInput.Rand(3211, new[] { 6 });
        yield return new OpCase("TensorOuterProduct[4;6]", "matmul",
            e => e.TensorOuterProduct(u.F(), v.F()), e => e.TensorOuterProduct(u.D(), v.D()),
            ParityTol.Ulp(4), opMethod: "TensorOuterProduct");

        var row = OpInput.Rand(3220, new[] { 1, 6 });
        var like = OpInput.Rand(3221, new[] { 4, 6 });
        yield return new OpCase("TensorExpandAs[1,6->4,6]", "shape",
            e => e.TensorExpandAs(row.F(), like.F()), e => e.TensorExpandAs(row.D(), like.D()),
            ParityTol.Exact, opMethod: "TensorExpandAs");

        // CIoU loss between predicted/target boxes [3,4] in XYXY.
        var pb = OpInput.From(new double[] { 0, 0, 2, 2, 1, 1, 3, 4, 0, 1, 4, 3 }, new[] { 3, 4 });
        var tb = OpInput.From(new double[] { 0, 0, 2, 3, 1, 0, 3, 3, 0, 0, 3, 3 }, new[] { 3, 4 });
        yield return new OpCase("TensorCIoULoss[3,4;3,4]", "loss",
            e => e.TensorCIoULoss(pb.F(), tb.F()), e => e.TensorCIoULoss(pb.D(), tb.D()),
            ParityTol.Accum(1e-3), opMethod: "TensorCIoULoss");

        // FOUND (quarantined): GPU TensorUpsampleBilinear diverges badly (CPU 26 ULP vs oracle, GPU
        // ~0.45 abs off) — GPU bilinear-upsample kernel bug (note plain Interpolate bilinear passes,
        // so it's this specific kernel).
        var upIn = OpInput.Rand(3230, new[] { 1, 2, 4, 4 });
        yield return new OpCase("TensorUpsampleBilinear[1,2,4,4->8,8]", "resize",
            e => e.TensorUpsampleBilinear(upIn.F(), new[] { 8, 8 }), e => e.TensorUpsampleBilinear(upIn.D(), new[] { 8, 8 }),
            ParityTol.Accum(1e-3), opMethod: "TensorUpsampleBilinear")
        { KnownDivergence = "GPU bilinear-upsample kernel diverges ~0.45 abs; CPU matches oracle." };

        yield return new OpCase("TensorTriangularMask[4;lower]", "shape",
            e => e.TensorTriangularMask<float>(4, false, 0), e => e.TensorTriangularMask<double>(4, false, 0),
            ParityTol.Exact, opMethod: "TensorTriangularMask");

        // Hurwitz zeta zeta(x,q), x>1, q>0.
        var zx = OpInput.Rand(3240, new[] { 3, 4 }, 2.0, 4.0);
        var zq = OpInput.Rand(3241, new[] { 3, 4 }, 0.5, 3.0);
        yield return new OpCase("TensorZeta[3,4]", "special",
            e => e.TensorZeta(zx.F(), zq.F()), e => e.TensorZeta(zx.D(), zq.D()),
            ParityTol.Accum(1e-3), opMethod: "TensorZeta");
    }

    // Complex-input ops that return a REAL tensor (magnitude/phase family).
    public static IEnumerable<OpCase> ComplexReal()
    {
        var re = OpInput.Rand(3100, new[] { 4, 6 });
        var im = OpInput.Rand(3101, new[] { 4, 6 });
        yield return new OpCase("NativeComplexMagnitude[4,6]", "complex",
            e => e.NativeComplexMagnitude(re.CF(im)), e => e.NativeComplexMagnitude(re.CD(im)),
            ParityTol.Accum(1e-3), opMethod: "NativeComplexMagnitude");
        yield return new OpCase("NativeComplexMagnitudeSquared[4,6]", "complex",
            e => e.NativeComplexMagnitudeSquared(re.CF(im)), e => e.NativeComplexMagnitudeSquared(re.CD(im)),
            ParityTol.Ulp(8), opMethod: "NativeComplexMagnitudeSquared");
        yield return new OpCase("NativeComplexPhase[4,6]", "complex",
            e => e.NativeComplexPhase(re.CF(im)), e => e.NativeComplexPhase(re.CD(im)),
            ParityTol.Accum(1e-3), opMethod: "NativeComplexPhase");
    }

    // Scalar-minus-tensor, add-scaled, at-least-Nd shape promotion, N-d pad.
    public static IEnumerable<OpCase> ScalarShapePad()
    {
        var a = OpInput.Rand(3000, new[] { 4, 6 });
        var b = OpInput.Rand(3001, new[] { 4, 6 });
        yield return new OpCase("ScalarMinusTensor[2-;4,6]", "elementwise",
            e => e.ScalarMinusTensor(2f, a.F()), e => e.ScalarMinusTensor(2.0, a.D()),
            ParityTol.Exact, opMethod: "ScalarMinusTensor");
        yield return new OpCase("TensorAddScaled[4,6;2,3]", "elementwise",
            e => e.TensorAddScaled(a.F(), b.F(), 2f, 3f), e => e.TensorAddScaled(a.D(), b.D(), 2.0, 3.0),
            ParityTol.Ulp(4), opMethod: "TensorAddScaled");

        var v = OpInput.Rand(3010, new[] { 4 });
        yield return new OpCase("TensorAtLeast1D[4]", "shape",
            e => e.TensorAtLeast1D(v.F()), e => e.TensorAtLeast1D(v.D()), ParityTol.Exact, opMethod: "TensorAtLeast1D");
        yield return new OpCase("TensorAtLeast2D[4]", "shape",
            e => e.TensorAtLeast2D(v.F()), e => e.TensorAtLeast2D(v.D()), ParityTol.Exact, opMethod: "TensorAtLeast2D");
        yield return new OpCase("TensorAtLeast3D[4]", "shape",
            e => e.TensorAtLeast3D(v.F()), e => e.TensorAtLeast3D(v.D()), ParityTol.Exact, opMethod: "TensorAtLeast3D");

        var padIn = OpInput.Rand(3020, new[] { 2, 4 });
        yield return new OpCase("PadNd[2,4;1,1,1,1;constant]", "pad",
            e => e.PadNd(padIn.F(), new[] { 1, 1, 1, 1 }, PadMode.Constant, 0f),
            e => e.PadNd(padIn.D(), new[] { 1, 1, 1, 1 }, PadMode.Constant, 0.0),
            ParityTol.Exact, opMethod: "PadNd");
        yield return new OpCase("PadNd[2,4;1,1,1,1;reflect]", "pad",
            e => e.PadNd(padIn.F(), new[] { 1, 1, 1, 1 }, PadMode.Reflect, 0f),
            e => e.PadNd(padIn.D(), new[] { 1, 1, 1, 1 }, PadMode.Reflect, 0.0),
            ParityTol.Exact, opMethod: "PadNd");
    }

    // Attention forward paths + fused batchnorm (eval) + mu-law encoding.
    public static IEnumerable<OpCase> AttentionFused()
    {
        // FlashAttention: q/k/v [batch, heads, seq, headDim] = [1,2,4,8].
        var fq = OpInput.Rand(2900, new[] { 1, 2, 4, 8 });
        var fk = OpInput.Rand(2901, new[] { 1, 2, 4, 8 });
        var fv = OpInput.Rand(2902, new[] { 1, 2, 4, 8 });
        yield return new OpCase("FlashAttention[1,2,4,8]", "attention",
            e => e.FlashAttention(fq.F(), fk.F(), fv.F(), null, false, out _, null),
            e => e.FlashAttention(fq.D(), fk.D(), fv.D(), null, false, out _, null),
            ParityTol.Accum(1e-3), opMethod: "FlashAttention");

        // GroupedQueryAttention: Q [1,4,4,8], K/V [1,2,4,8], 2 queries per KV head.
        var gq = OpInput.Rand(2910, new[] { 1, 4, 4, 8 });
        var gk = OpInput.Rand(2911, new[] { 1, 2, 4, 8 });
        var gv = OpInput.Rand(2912, new[] { 1, 2, 4, 8 });
        yield return new OpCase("GroupedQueryAttention[q1,4,4,8;kv1,2,4,8]", "attention",
            e => e.GroupedQueryAttention(gq.F(), gk.F(), gv.F(), 2, null, false, out _),
            e => e.GroupedQueryAttention(gq.D(), gk.D(), gv.D(), 2, null, false, out _),
            ParityTol.Accum(1e-3), opMethod: "GroupedQueryAttention");

        // MultiHeadAttentionForward: input [B,S,dModel] = [2,4,8], projections [8,8], 2 heads.
        var mhaIn = OpInput.Rand(2920, new[] { 2, 4, 8 });
        var qw = OpInput.Rand(2921, new[] { 8, 8 });
        var kw = OpInput.Rand(2922, new[] { 8, 8 });
        var vw = OpInput.Rand(2923, new[] { 8, 8 });
        var ow = OpInput.Rand(2924, new[] { 8, 8 });
        yield return new OpCase("MultiHeadAttentionForward[2,4,8;h2]", "attention",
            e => e.MultiHeadAttentionForward(mhaIn.F(), qw.F(), kw.F(), vw.F(), ow.F(), 2, null),
            e => e.MultiHeadAttentionForward(mhaIn.D(), qw.D(), kw.D(), vw.D(), ow.D(), 2, null),
            ParityTol.Accum(1e-3), opMethod: "MultiHeadAttentionForward");

        // FusedBatchNorm in eval mode (training=false) uses running stats -> deterministic identity-ish.
        var bnIn = OpInput.Rand(2930, new[] { 2, 4, 4, 4 });
        var gamma = OpInput.Rand(2931, new[] { 4 });
        var beta = OpInput.Rand(2932, new[] { 4 });
        var rMean = OpInput.Rand(2933, new[] { 4 });
        var rVar = OpInput.RandPositive(2934, new[] { 4 });
        yield return new OpCase("FusedBatchNorm[2,4,4,4;eval]", "norm",
            e => e.FusedBatchNorm(bnIn.F(), gamma.F(), beta.F(), rMean.F(), rVar.F(), 1e-5, 0.1, false, FusedActivationType.None, out _, out _),
            e => e.FusedBatchNorm(bnIn.D(), gamma.D(), beta.D(), rMean.D(), rVar.D(), 1e-5, 0.1, false, FusedActivationType.None, out _, out _),
            ParityTol.Accum(1e-3), opMethod: "FusedBatchNorm");
    }

    // 1D/transpose conv backward kernels, interpolate, dropout(eval).
    public static IEnumerable<OpCase> SortConvInterp()
    {
        // Conv1D backward kernel: input [1,2,8], kernel [3,2,3] -> out [1,3,6].
        var go1d = OpInput.Rand(2810, new[] { 1, 3, 6 });
        var in1d = OpInput.Rand(2811, new[] { 1, 2, 8 });
        yield return new OpCase("Conv1DBackwardKernel[go1,3,6;in1,2,8]", "conv",
            e => e.Conv1DBackwardKernel(go1d.F(), in1d.F(), new[] { 3, 2, 3 }, 1, 0, 1),
            e => e.Conv1DBackwardKernel(go1d.D(), in1d.D(), new[] { 3, 2, 3 }, 1, 0, 1),
            ParityTol.Accum(1e-3), opMethod: "Conv1DBackwardKernel");

        // Depthwise conv1d backward kernel: input [1,4,8], kernelShape [4,1,3] -> out [1,4,6].
        var dwGo1 = OpInput.Rand(2820, new[] { 1, 4, 6 });
        var dwIn1 = OpInput.Rand(2821, new[] { 1, 4, 8 });
        yield return new OpCase("DepthwiseConv1DBackwardKernel[go1,4,6;in1,4,8]", "conv",
            e => e.DepthwiseConv1DBackwardKernel(dwGo1.F(), dwIn1.F(), new[] { 4, 1, 3 }, 1, 0),
            e => e.DepthwiseConv1DBackwardKernel(dwGo1.D(), dwIn1.D(), new[] { 4, 1, 3 }, 1, 0),
            ParityTol.Accum(1e-3), opMethod: "DepthwiseConv1DBackwardKernel");

        // ConvTranspose2D backward kernel: convT input [1,2,4,4] kernel [2,3,2,2] stride2 -> [1,3,8,8].
        var ct2Go = OpInput.Rand(2830, new[] { 1, 3, 8, 8 });
        var ct2In = OpInput.Rand(2831, new[] { 1, 2, 4, 4 });
        yield return new OpCase("ConvTranspose2DBackwardKernel[go1,3,8,8;in1,2,4,4]", "conv",
            e => e.ConvTranspose2DBackwardKernel(ct2Go.F(), ct2In.F(), new[] { 2, 3, 2, 2 }, new[] { 2, 2 }, new[] { 0, 0 }),
            e => e.ConvTranspose2DBackwardKernel(ct2Go.D(), ct2In.D(), new[] { 2, 3, 2, 2 }, new[] { 2, 2 }, new[] { 0, 0 }),
            ParityTol.Accum(1e-3), opMethod: "ConvTranspose2DBackwardKernel");

        // Interpolate [1,2,4,4] -> [8,8] bilinear.
        var interpIn = OpInput.Rand(2840, new[] { 1, 2, 4, 4 });
        yield return new OpCase("Interpolate[1,2,4,4->8,8;bilinear]", "resize",
            e => e.Interpolate(interpIn.F(), new[] { 8, 8 }, InterpolateMode.Bilinear, false),
            e => e.Interpolate(interpIn.D(), new[] { 8, 8 }, InterpolateMode.Bilinear, false),
            ParityTol.Accum(1e-3), opMethod: "Interpolate");
        yield return new OpCase("InterpolateByScale[1,2,4,4;x2;nearest]", "resize",
            e => e.InterpolateByScale(interpIn.F(), new double[] { 2, 2 }, InterpolateMode.Nearest, false),
            e => e.InterpolateByScale(interpIn.D(), new double[] { 2, 2 }, InterpolateMode.Nearest, false),
            ParityTol.Exact, opMethod: "InterpolateByScale");

        // Dropout in eval mode (training=false) is a deterministic identity.
        var dropIn = OpInput.Rand(2850, new[] { 4, 6 });
        yield return new OpCase("Dropout[4,6;eval]", "regularize",
            e => e.Dropout(dropIn.F(), 0.5, false, out _),
            e => e.Dropout(dropIn.D(), 0.5, false, out _),
            ParityTol.Exact, opMethod: "Dropout");
    }

    // 3D conv backward kernels/inputs, ConvTranspose3D, depthwise-conv backward, box-IoU + convert.
    public static IEnumerable<OpCase> Conv3DBoxIou()
    {
        // Conv3D shapes: input [1,2,4,4,4], kernel [3,2,2,2,2] -> output [1,3,3,3,3].
        var go3d = OpInput.Rand(2700, new[] { 1, 3, 3, 3, 3 });
        var in3d = OpInput.Rand(2701, new[] { 1, 2, 4, 4, 4 });
        var k3d = OpInput.Rand(2702, new[] { 3, 2, 2, 2, 2 });
        yield return new OpCase("Conv3DBackwardKernel[go1,3,3,3,3;in1,2,4,4,4]", "conv",
            e => e.Conv3DBackwardKernel(go3d.F(), in3d.F(), new[] { 3, 2, 2, 2, 2 }, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }, new[] { 1, 1, 1 }),
            e => e.Conv3DBackwardKernel(go3d.D(), in3d.D(), new[] { 3, 2, 2, 2, 2 }, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }, new[] { 1, 1, 1 }),
            ParityTol.Accum(1e-3), opMethod: "Conv3DBackwardKernel");
        yield return new OpCase("Conv3DBackwardInput[go1,3,3,3,3;k3,2,2,2,2]", "conv",
            e => e.Conv3DBackwardInput(go3d.F(), k3d.F(), new[] { 1, 2, 4, 4, 4 }, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }, new[] { 1, 1, 1 }),
            e => e.Conv3DBackwardInput(go3d.D(), k3d.D(), new[] { 1, 2, 4, 4, 4 }, new[] { 1, 1, 1 }, new[] { 0, 0, 0 }, new[] { 1, 1, 1 }),
            ParityTol.Accum(1e-3), opMethod: "Conv3DBackwardInput");

        // ConvTranspose3D: input [1,2,2,2,2], kernel [2,3,2,2,2] (inC,outC,kD,kH,kW), stride 2.
        var ctIn = OpInput.Rand(2710, new[] { 1, 2, 2, 2, 2 });
        var ctK = OpInput.Rand(2711, new[] { 2, 3, 2, 2, 2 });
        yield return new OpCase("ConvTranspose3D[1,2,2,2,2;k2,3,2,2,2]", "conv",
            e => e.ConvTranspose3D(ctIn.F(), ctK.F(), new[] { 2, 2, 2 }, new[] { 0, 0, 0 }, new[] { 0, 0, 0 }),
            e => e.ConvTranspose3D(ctIn.D(), ctK.D(), new[] { 2, 2, 2 }, new[] { 0, 0, 0 }, new[] { 0, 0, 0 }),
            ParityTol.Accum(1e-3), opMethod: "ConvTranspose3D");

        // Depthwise conv2d backward kernel: gradOutput/input [1,4,8,8], kernelShape [4,1,3,3].
        var dwGo = OpInput.Rand(2720, new[] { 1, 4, 6, 6 });
        var dwIn = OpInput.Rand(2721, new[] { 1, 4, 8, 8 });
        yield return new OpCase("DepthwiseConv2DBackwardKernel[go1,4,6,6;in1,4,8,8]", "conv",
            e => e.DepthwiseConv2DBackwardKernel(dwGo.F(), dwIn.F(), new[] { 4, 1, 3, 3 }, new[] { 1, 1 }, new[] { 0, 0 }),
            e => e.DepthwiseConv2DBackwardKernel(dwGo.D(), dwIn.D(), new[] { 4, 1, 3, 3 }, new[] { 1, 1 }, new[] { 0, 0 }),
            ParityTol.Accum(1e-3), opMethod: "DepthwiseConv2DBackwardKernel");

        // Box-IoU variants: boxesA [3,4], boxesB [2,4] in XYXY.
        var ba = OpInput.From(new double[] { 0, 0, 2, 2, 1, 1, 3, 4, 0, 1, 4, 3 }, new[] { 3, 4 });
        var bb = OpInput.From(new double[] { 0, 0, 2, 3, 1, 0, 3, 3 }, new[] { 2, 4 });
        yield return new OpCase("CompleteBoxIou[3,4;2,4]", "box",
            e => e.CompleteBoxIou(ba.F(), bb.F()), e => e.CompleteBoxIou(ba.D(), bb.D()),
            ParityTol.Accum(1e-3), opMethod: "CompleteBoxIou");
        yield return new OpCase("DistanceBoxIou[3,4;2,4]", "box",
            e => e.DistanceBoxIou(ba.F(), bb.F()), e => e.DistanceBoxIou(ba.D(), bb.D()),
            ParityTol.Accum(1e-3), opMethod: "DistanceBoxIou");
        yield return new OpCase("BoxConvert[3,4;XYXY->XYWH]", "box",
            e => e.BoxConvert(ba.F(), BoxFormat.XYXY, BoxFormat.XYWH),
            e => e.BoxConvert(ba.D(), BoxFormat.XYXY, BoxFormat.XYWH),
            ParityTol.Accum(1e-4), opMethod: "BoxConvert");
    }

    // Fused-linear (activation enum), RoI align/pool, NLL/KL losses.
    public static IEnumerable<OpCase> FusedRoiLoss()
    {
        var lin = OpInput.Rand(2600, new[] { 4, 8 });
        var w = OpInput.Rand(2601, new[] { 8, 6 });
        var bias = OpInput.Rand(2602, new[] { 6 });
        yield return new OpCase("FusedLinear[4,8;w8,6;None]", "matmul",
            e => e.FusedLinear(lin.F(), w.F(), bias.F(), FusedActivationType.None, null),
            e => e.FusedLinear(lin.D(), w.D(), bias.D(), FusedActivationType.None, null), ParityTol.Accum(1e-3), opMethod: "FusedLinear");

        var img = OpInput.Rand(2610, new[] { 1, 2, 8, 8 });
        var boxes = OpInput.From(new double[] { 0, 0, 0, 4, 4, 0, 1, 1, 6, 6, 0, 2, 0, 8, 5 }, new[] { 3, 5 });
        yield return new OpCase("RoIAlign[1,2,8,8;b3;o2x2]", "pool",
            e => e.RoIAlign(img.F(), boxes.F(), 2, 2, 1f, 2, false), e => e.RoIAlign(img.D(), boxes.D(), 2, 2, 1f, 2, false), ParityTol.Accum(1e-3), opMethod: "RoIAlign");
        yield return new OpCase("RoIPool[1,2,8,8;b3;o2x2]", "pool",
            e => e.RoIPool(img.F(), boxes.F(), 2, 2, 1f), e => e.RoIPool(img.D(), boxes.D(), 2, 2, 1f), ParityTol.Exact, opMethod: "RoIPool");

        var lp = OpInput.Rand(2620, new[] { 4, 8 }, -3.0, 3.0);
        var tg = OpInput.Rand(2621, new[] { 4, 8 }, 0.0, 1.0);
        // FOUND (quarantined): GPU TensorNLLLoss returns 0 where CPU/oracle give the loss — GPU
        // loss-kernel divergence.
        yield return new OpCase("TensorNLLLoss[4,8]", "loss", e => e.TensorNLLLoss(e.TensorLogSoftmax(lp.F(), -1), tg.F()), e => e.TensorNLLLoss(e.TensorLogSoftmax(lp.D(), -1), tg.D()), ParityTol.Accum(1e-3), opMethod: "TensorNLLLoss")
        { KnownDivergence = "GPU TensorNLLLoss returns 0 (loss-kernel divergence)." };
        yield return new OpCase("TensorKLDivLoss[4,8]", "loss", e => e.TensorKLDivLoss(e.TensorLogSoftmax(lp.F(), -1), e.TensorSoftmax(tg.F(), -1)), e => e.TensorKLDivLoss(e.TensorLogSoftmax(lp.D(), -1), e.TensorSoftmax(tg.D(), -1)), ParityTol.Accum(1e-3), opMethod: "TensorKLDivLoss");
    }

    // Geometry / NeRF: affine-grid, volume-render, spherical harmonics, upsample backward.
    public static IEnumerable<OpCase> GeometryNerf()
    {
        yield return new OpCase("AffineGrid[2,2,3->4,4]", "geometry",
            e => e.AffineGrid(OpInput.Rand(2500, new[] { 2, 2, 3 }, -1.0, 1.0).F(), 4, 4),
            e => e.AffineGrid(OpInput.Rand(2500, new[] { 2, 2, 3 }, -1.0, 1.0).D(), 4, 4), ParityTol.Accum(1e-3), opMethod: "AffineGrid");
        yield return new OpCase("AffineGrid3D[2,3,4->2,2,2]", "geometry",
            e => e.AffineGrid3D(OpInput.Rand(2501, new[] { 2, 3, 4 }, -1.0, 1.0).F(), 2, 2, 2, false),
            e => e.AffineGrid3D(OpInput.Rand(2501, new[] { 2, 3, 4 }, -1.0, 1.0).D(), 2, 2, 2, false), ParityTol.Accum(1e-3), opMethod: "AffineGrid3D");
        yield return new OpCase("VolumeRendering[2,4,3]", "geometry",
            e => e.VolumeRendering(OpInput.Rand(2502, new[] { 2, 4, 3 }, 0.0, 1.0).F(), OpInput.RandPositive(2503, new[] { 2, 4 }, 0.1, 2.0).F(), OpInput.RandPositive(2504, new[] { 2, 4 }, 0.1, 1.0).F()),
            e => e.VolumeRendering(OpInput.Rand(2502, new[] { 2, 4, 3 }, 0.0, 1.0).D(), OpInput.RandPositive(2503, new[] { 2, 4 }, 0.1, 2.0).D(), OpInput.RandPositive(2504, new[] { 2, 4 }, 0.1, 1.0).D()), ParityTol.Accum(1e-3), opMethod: "VolumeRendering");
        // EvaluateSphericalHarmonics left pending: its SH-coefficient tensor layout for a given degree
        // needs the exact convention (a naive [N,(deg+1)^2*3] shape indexed out of bounds).

        yield return new OpCase("PixelShuffleBackward[1,1,8,8->1,4,4,4]", "shape",
            e => e.PixelShuffleBackward(OpInput.Rand(2510, new[] { 1, 1, 8, 8 }).F(), new[] { 1, 4, 4, 4 }, 2),
            e => e.PixelShuffleBackward(OpInput.Rand(2510, new[] { 1, 1, 8, 8 }).D(), new[] { 1, 4, 4, 4 }, 2), ParityTol.Exact, opMethod: "PixelShuffleBackward");
        yield return new OpCase("UpsampleBackward[1,2,8,8->1,2,4,4]", "shape",
            e => e.UpsampleBackward(OpInput.Rand(2511, new[] { 1, 2, 8, 8 }).F(), new[] { 1, 2, 4, 4 }, 2, 2),
            e => e.UpsampleBackward(OpInput.Rand(2511, new[] { 1, 2, 8, 8 }).D(), new[] { 1, 2, 4, 4 }, 2, 2), ParityTol.Accum(1e-3), opMethod: "UpsampleBackward");
        yield return new OpCase("Upsample3DBackward[1,2,4,4,4->1,2,2,2,2]", "shape",
            e => e.Upsample3DBackward(OpInput.Rand(2512, new[] { 1, 2, 4, 4, 4 }).F(), new[] { 1, 2, 2, 2, 2 }, 2, 2, 2),
            e => e.Upsample3DBackward(OpInput.Rand(2512, new[] { 1, 2, 4, 4, 4 }).D(), new[] { 1, 2, 2, 2, 2 }, 2, 2, 2), ParityTol.Accum(1e-3), opMethod: "Upsample3DBackward");
    }

    // Audio (mu-law), FFT, gaussian-splat covariance, gumbel-softmax backward.
    public static IEnumerable<OpCase> AudioFftSplat()
    {
        var q = new int[8]; for (int i = 0; i < 8; i++) q[i] = i * 30 + 5;
        yield return new OpCase("MuLawDecoding[idx8;q256]", "audio",
            e => e.MuLawDecoding<float>(new Tensor<int>((int[])q.Clone(), new[] { 8 }), 256),
            e => e.MuLawDecoding<double>(new Tensor<int>((int[])q.Clone(), new[] { 8 }), 256), ParityTol.Ulp(16, 1e-6), opMethod: "MuLawDecoding");
        yield return new OpCase("RFFT[16]", "fft", e => e.RFFT(OpInput.Rand(2400, new[] { 16 }).F()), e => e.RFFT(OpInput.Rand(2400, new[] { 16 }).D()), ParityTol.Accum(1e-3), opMethod: "RFFT");
        yield return new OpCase("Spectrogram[1,64;n16h8w16]", "audio", e => e.Spectrogram(OpInput.Rand(2401, new[] { 1, 64 }).F(), 16, 8, 16, null), e => e.Spectrogram(OpInput.Rand(2401, new[] { 1, 64 }).D(), 16, 8, 16, null), ParityTol.Accum(1e-3), opMethod: "Spectrogram");

        var lg = OpInput.Rand(2410, new[] { 4, 8 }, -3.0, 3.0);
        var gg = OpInput.Rand(2411, new[] { 4, 8 });
        yield return new OpCase("GumbelSoftmaxBackward[4,8]", "activation-bwd", e => e.GumbelSoftmaxBackward(gg.F(), e.TensorSoftmax(lg.F(), -1), 1.0, -1), e => e.GumbelSoftmaxBackward(gg.D(), e.TensorSoftmax(lg.D(), -1), 1.0, -1), ParityTol.Accum(1e-3), opMethod: "GumbelSoftmaxBackward");

        yield return new OpCase("ComputeGaussianCovariance[3,4;3,3]", "geometry",
            e => e.ComputeGaussianCovariance(OpInput.Rand(2420, new[] { 3, 4 }, -1.0, 1.0).F(), OpInput.RandPositive(2421, new[] { 3, 3 }, 0.2, 1.5).F()),
            e => e.ComputeGaussianCovariance(OpInput.Rand(2420, new[] { 3, 4 }, -1.0, 1.0).D(), OpInput.RandPositive(2421, new[] { 3, 3 }, 0.2, 1.5).D()), ParityTol.Accum(1e-3), opMethod: "ComputeGaussianCovariance");
    }

    // More activation/conv/embedding backward ops.
    public static IEnumerable<OpCase> MoreBackward()
    {
        var s = new[] { 4, 8 };
        var go = OpInput.Rand(2300, s);
        var inp = OpInput.Rand(2301, s, -3.0, 3.0);
        yield return new OpCase("ThresholdBackward[4,8]", "activation-bwd", e => e.ThresholdBackward(go.F(), inp.F(), 0.0), e => e.ThresholdBackward(go.D(), inp.D(), 0.0), ParityTol.Exact, opMethod: "ThresholdBackward");
        yield return new OpCase("ReciprocalBackward[4,8]", "activation-bwd", e => e.ReciprocalBackward(go.F(), OpInput.Rand(2302, s, 0.5, 3.0).F()), e => e.ReciprocalBackward(go.D(), OpInput.Rand(2302, s, 0.5, 3.0).D()), ParityTol.Ulp(8, 1e-6), opMethod: "ReciprocalBackward");
        {
            var maskD = new double[32]; var rng = new Random(2303); for (int i = 0; i < 32; i++) maskD[i] = rng.NextDouble() < 0.5 ? 0.0 : 2.0;
            var mask = OpInput.From(maskD, s);
            yield return new OpCase("DropoutBackward[4,8]", "activation-bwd", e => e.DropoutBackward(go.F(), mask.F(), 0.5), e => e.DropoutBackward(go.D(), mask.D(), 0.5), ParityTol.Ulp(4, 1e-6), opMethod: "DropoutBackward");
        }
        // FOUND (quarantined): GPU CrossEntropyBackward returns 0 where CPU/oracle give the gradient
        // — part of the GPU backward-kernel divergence cluster.
        {
            var cep = OpInput.Rand(2304, s, -3.0, 3.0); var cet = OpInput.Rand(2305, s, 0.0, 1.0);
            yield return new OpCase("CrossEntropyBackward[4,8]", "loss-bwd",
                e => e.CrossEntropyBackward(e.TensorSoftmax(cep.F(), -1), cet.F()), e => e.CrossEntropyBackward(e.TensorSoftmax(cep.D(), -1), cet.D()), ParityTol.Accum(1e-3), opMethod: "CrossEntropyBackward")
            { KnownDivergence = "GPU CrossEntropyBackward returns 0 (backward-kernel bug cluster)." };
        }

        yield return new OpCase("Conv1DBackwardInput[1,8,14->1,3,16]", "conv",
            e => e.Conv1DBackwardInput(OpInput.Rand(2310, new[] { 1, 8, 14 }).F(), OpInput.Rand(2311, new[] { 8, 3, 3 }).F(), new[] { 1, 3, 16 }, 1, 0, 1),
            e => e.Conv1DBackwardInput(OpInput.Rand(2310, new[] { 1, 8, 14 }).D(), OpInput.Rand(2311, new[] { 8, 3, 3 }).D(), new[] { 1, 3, 16 }, 1, 0, 1), ParityTol.Accum(1e-3), opMethod: "Conv1DBackwardInput");
        yield return new OpCase("ConvTranspose2DBackwardInput[1,8,8,8->1,4,4,4]", "conv",
            e => e.ConvTranspose2DBackwardInput(OpInput.Rand(2312, new[] { 1, 8, 8, 8 }).F(), OpInput.Rand(2313, new[] { 4, 8, 2, 2 }).F(), new[] { 1, 4, 4, 4 }, new[] { 2, 2 }, new[] { 0, 0 }),
            e => e.ConvTranspose2DBackwardInput(OpInput.Rand(2312, new[] { 1, 8, 8, 8 }).D(), OpInput.Rand(2313, new[] { 4, 8, 2, 2 }).D(), new[] { 1, 4, 4, 4 }, new[] { 2, 2 }, new[] { 0, 0 }), ParityTol.Accum(1e-3), opMethod: "ConvTranspose2DBackwardInput");
        yield return new OpCase("DepthwiseConv1DBackwardInput[1,4,14->1,4,16]", "conv",
            e => e.DepthwiseConv1DBackwardInput(OpInput.Rand(2314, new[] { 1, 4, 14 }).F(), OpInput.Rand(2315, new[] { 4, 1, 3 }).F(), new[] { 1, 4, 16 }, 1, 0),
            e => e.DepthwiseConv1DBackwardInput(OpInput.Rand(2314, new[] { 1, 4, 14 }).D(), OpInput.Rand(2315, new[] { 4, 1, 3 }).D(), new[] { 1, 4, 16 }, 1, 0), ParityTol.Accum(1e-3), opMethod: "DepthwiseConv1DBackwardInput");

        yield return new OpCase("EmbeddingBackward[go4,8;v10]", "index",
            e => e.EmbeddingBackward(OpInput.Rand(2320, new[] { 4, 8 }).F(), new Tensor<int>(new[] { 1, 3, 0, 5 }, new[] { 4 }), 10, 8),
            e => e.EmbeddingBackward(OpInput.Rand(2320, new[] { 4, 8 }).D(), new Tensor<int>(new[] { 1, 3, 0, 5 }, new[] { 4 }), 10, 8), ParityTol.Ulp(8, 1e-6), opMethod: "EmbeddingBackward");
    }

    // GLU-variant backward (validates the gating-half fix), crop/pad backward, softmax-variant backward.
    public static IEnumerable<OpCase> GluCropSoftmaxBwd()
    {
        var inp = OpInput.Rand(2200, new[] { 4, 16 }, -3.0, 3.0);
        var go = OpInput.Rand(2201, new[] { 4, 8 });
        yield return new OpCase("GLUBackward[4,16]", "activation-bwd", e => e.GLUBackward(go.F(), inp.F(), -1), e => e.GLUBackward(go.D(), inp.D(), -1), ParityTol.Accum(1e-3), opMethod: "GLUBackward");
        yield return new OpCase("GeGLUBackward[4,16]", "activation-bwd", e => e.GeGLUBackward(go.F(), inp.F(), -1), e => e.GeGLUBackward(go.D(), inp.D(), -1), ParityTol.Accum(2e-3), opMethod: "GeGLUBackward");
        yield return new OpCase("SwiGLUBackward[4,16]", "activation-bwd", e => e.SwiGLUBackward(go.F(), inp.F(), -1), e => e.SwiGLUBackward(go.D(), inp.D(), -1), ParityTol.Accum(1e-3), opMethod: "SwiGLUBackward");
        yield return new OpCase("ReGLUBackward[4,16]", "activation-bwd", e => e.ReGLUBackward(go.F(), inp.F(), -1), e => e.ReGLUBackward(go.D(), inp.D(), -1), ParityTol.Ulp(4, 1e-6), opMethod: "ReGLUBackward");

        yield return new OpCase("CropBackward[1,2,4,4->1,2,8,8]", "shape",
            e => e.CropBackward(OpInput.Rand(2210, new[] { 1, 2, 4, 4 }).F(), new[] { 1, 2, 8, 8 }, 1, 1),
            e => e.CropBackward(OpInput.Rand(2210, new[] { 1, 2, 4, 4 }).D(), new[] { 1, 2, 8, 8 }, 1, 1), ParityTol.Exact, opMethod: "CropBackward");
        yield return new OpCase("PadBackward[1,2,10,10->1,2,8,8]", "shape",
            e => e.PadBackward(OpInput.Rand(2211, new[] { 1, 2, 10, 10 }).F(), 1, 1, new[] { 1, 2, 8, 8 }),
            e => e.PadBackward(OpInput.Rand(2211, new[] { 1, 2, 10, 10 }).D(), 1, 1, new[] { 1, 2, 8, 8 }), ParityTol.Exact, opMethod: "PadBackward");

        var lg = OpInput.Rand(2220, new[] { 4, 8 }, -3.0, 3.0);
        var sgo = OpInput.Rand(2221, new[] { 4, 8 });
        yield return new OpCase("SparsemaxBackward[4,8]", "activation-bwd", e => e.SparsemaxBackward(sgo.F(), e.Sparsemax(lg.F(), -1), -1), e => e.SparsemaxBackward(sgo.D(), e.Sparsemax(lg.D(), -1), -1), ParityTol.Accum(1e-3), opMethod: "SparsemaxBackward");
    }

    // Fused-linear activation variants + affine batchnorm.
    public static IEnumerable<OpCase> FusedLinAffine()
    {
        var lin = OpInput.Rand(2100, new[] { 4, 8 });
        var w = OpInput.Rand(2101, new[] { 8, 6 });
        var bias = OpInput.Rand(2102, new[] { 6 });
        yield return new OpCase("FusedLinearSigmoid[4,8;w8,6]", "matmul", e => e.FusedLinearSigmoid(lin.F(), w.F(), bias.F()), e => e.FusedLinearSigmoid(lin.D(), w.D(), bias.D()), ParityTol.Accum(1e-3), opMethod: "FusedLinearSigmoid");
        yield return new OpCase("FusedLinearTanh[4,8;w8,6]", "matmul", e => e.FusedLinearTanh(lin.F(), w.F(), bias.F()), e => e.FusedLinearTanh(lin.D(), w.D(), bias.D()), ParityTol.Accum(1e-3), opMethod: "FusedLinearTanh");
        yield return new OpCase("FusedLinearSwish[4,8;w8,6]", "matmul", e => e.FusedLinearSwish(lin.F(), w.F(), bias.F()), e => e.FusedLinearSwish(lin.D(), w.D(), bias.D()), ParityTol.Accum(1e-3), opMethod: "FusedLinearSwish");

        var bx = OpInput.Rand(2110, new[] { 2, 4, 4, 4 });
        var bg = OpInput.Rand(2111, new[] { 4 }, 0.5, 1.5);
        var bb = OpInput.Rand(2112, new[] { 4 }, -0.2, 0.2);
        var bm = OpInput.Rand(2113, new[] { 4 }, -0.5, 0.5);
        var bv = OpInput.Rand(2114, new[] { 4 }, 0.5, 1.5);
        yield return new OpCase("BatchNormAffine[2,4,4,4]", "norm", e => e.BatchNormAffine(bx.F(), bg.F(), bb.F(), bm.F(), bv.F(), 1e-5), e => e.BatchNormAffine(bx.D(), bg.D(), bb.D(), bm.D(), bv.D(), 1e-5), ParityTol.Accum(1e-3), opMethod: "BatchNormAffine");
    }

    // Stack variants, index-copy/fill/put, embedding-from-float, cartesian-prod, softmax-backward.
    public static IEnumerable<OpCase> StackIndexEmbed()
    {
        var c1 = OpInput.Rand(2000, new[] { 4, 8 });
        var c2 = OpInput.Rand(2001, new[] { 4, 8 });
        Tensor<float>[] Fs() => new[] { c1.F(), c2.F() };
        Tensor<double>[] Ds() => new[] { c1.D(), c2.D() };
        yield return new OpCase("TensorVStack[2x4,8]", "shape", e => e.TensorVStack(Fs()), e => e.TensorVStack(Ds()), ParityTol.Exact, opMethod: "TensorVStack");
        yield return new OpCase("TensorHStack[2x4,8]", "shape", e => e.TensorHStack(Fs()), e => e.TensorHStack(Ds()), ParityTol.Exact, opMethod: "TensorHStack");
        yield return new OpCase("TensorColumnStack[2x4,8]", "shape", e => e.TensorColumnStack(Fs()), e => e.TensorColumnStack(Ds()), ParityTol.Exact, opMethod: "TensorColumnStack");
        yield return new OpCase("TensorRowStack[2x4,8]", "shape", e => e.TensorRowStack(Fs()), e => e.TensorRowStack(Ds()), ParityTol.Exact, opMethod: "TensorRowStack");
        yield return new OpCase("TensorDStack[2x4,8]", "shape", e => e.TensorDStack(Fs()), e => e.TensorDStack(Ds()), ParityTol.Exact, opMethod: "TensorDStack");

        var m = OpInput.Rand(2010, new[] { 4, 8 });
        yield return new OpCase("TensorPut[4,8;idx3]", "index",
            e => e.TensorPut(m.F(), new Tensor<int>(new[] { 0, 10, 25 }, new[] { 3 }), OpInput.Rand(2011, new[] { 3 }).F()),
            e => e.TensorPut(m.D(), new Tensor<int>(new[] { 0, 10, 25 }, new[] { 3 }), OpInput.Rand(2011, new[] { 3 }).D()), ParityTol.Exact, opMethod: "TensorPut");
        yield return new OpCase("TensorIndexCopy[4,8;ax0]", "index",
            e => e.TensorIndexCopy(m.F(), 0, new Tensor<int>(new[] { 0, 2 }, new[] { 2 }), OpInput.Rand(2012, new[] { 2, 8 }).F()),
            e => e.TensorIndexCopy(m.D(), 0, new Tensor<int>(new[] { 0, 2 }, new[] { 2 }), OpInput.Rand(2012, new[] { 2, 8 }).D()), ParityTol.Exact, opMethod: "TensorIndexCopy");
        yield return new OpCase("TensorIndexFill[4,8;ax0]", "index",
            e => e.TensorIndexFill(m.F(), 0, new Tensor<int>(new[] { 1, 3 }, new[] { 2 }), 0.5f),
            e => e.TensorIndexFill(m.D(), 0, new Tensor<int>(new[] { 1, 3 }, new[] { 2 }), 0.5), ParityTol.Exact, opMethod: "TensorIndexFill");

        yield return new OpCase("TensorEmbeddingLookupFromFloatIndices[10,8;idx4]", "index",
            e => e.TensorEmbeddingLookupFromFloatIndices(OpInput.Rand(2020, new[] { 10, 8 }).F(), OpInput.From(new double[] { 1, 3, 0, 5 }, new[] { 4 }).F()),
            e => e.TensorEmbeddingLookupFromFloatIndices(OpInput.Rand(2020, new[] { 10, 8 }).D(), OpInput.From(new double[] { 1, 3, 0, 5 }, new[] { 4 }).D()), ParityTol.Exact, opMethod: "TensorEmbeddingLookupFromFloatIndices");
        yield return new OpCase("TensorCartesianProd[2.3]", "shape",
            e => e.TensorCartesianProd(new[] { OpInput.Rand(2021, new[] { 2 }).F(), OpInput.Rand(2022, new[] { 3 }).F() }),
            e => e.TensorCartesianProd(new[] { OpInput.Rand(2021, new[] { 2 }).D(), OpInput.Rand(2022, new[] { 3 }).D() }), ParityTol.Exact, opMethod: "TensorCartesianProd");

        var logits = OpInput.Rand(2030, new[] { 4, 8 }, -4.0, 4.0);
        var sgo = OpInput.Rand(2031, new[] { 4, 8 });
        // FOUND (quarantined): GPU TensorSoftmaxBackward diverges strongly from CPU/oracle (maxRel
        // ~1.3; CPU matches the oracle) — part of the GPU backward-kernel divergence cluster with the
        // norm backwards. (The plain SoftmaxBackward op passes; this Tensor* variant does not.)
        yield return new OpCase("TensorSoftmaxBackward[4,8]", "activation-bwd",
            e => e.TensorSoftmaxBackward(e.TensorSoftmax(logits.F(), -1), sgo.F(), -1),
            e => e.TensorSoftmaxBackward(e.TensorSoftmax(logits.D(), -1), sgo.D(), -1), ParityTol.Accum(1e-3), opMethod: "TensorSoftmaxBackward")
        { KnownDivergence = "GPU TensorSoftmaxBackward diverges strongly from CPU/oracle (backward-kernel bug)." };
    }

    // Norm-family backward (forward run inline for saved stats), conv/pool backward, global-max, mse/ce bwd.
    public static IEnumerable<OpCase> NormConvBackward()
    {
        var x = OpInput.Rand(1900, new[] { 4, 64 });
        var g = OpInput.Rand(1901, new[] { 64 }, 0.5, 1.5);
        var beta = OpInput.Rand(1902, new[] { 64 }, -0.2, 0.2);
        var go = OpInput.Rand(1903, new[] { 4, 64 });
        yield return new OpCase("LayerNormBackward[4,64]", "norm-bwd",
            e => { e.LayerNorm(x.F(), g.F(), beta.F(), 1e-5, out var mn, out var vr); return e.LayerNormBackward(go.F(), x.F(), g.F(), mn, vr, 1e-5, out _, out _); },
            e => { e.LayerNorm(x.D(), g.D(), beta.D(), 1e-5, out var mn, out var vr); return e.LayerNormBackward(go.D(), x.D(), g.D(), mn, vr, 1e-5, out _, out _); }, ParityTol.Accum(2e-3), opMethod: "LayerNormBackward")
            { KnownDivergence = "GPU norm backward diverges strongly from CPU/oracle (norm-backward kernel bug/convention); CPU matches the double oracle." };
        yield return new OpCase("RMSNormBackward[4,64]", "norm-bwd",
            e => { e.RMSNorm(x.F(), g.F(), 1e-5, out var rms); return e.RMSNormBackward(go.F(), x.F(), g.F(), rms, 1e-5, out _); },
            e => { e.RMSNorm(x.D(), g.D(), 1e-5, out var rms); return e.RMSNormBackward(go.D(), x.D(), g.D(), rms, 1e-5, out _); }, ParityTol.Accum(2e-3), opMethod: "RMSNormBackward");

        var gx = OpInput.Rand(1910, new[] { 2, 8, 4, 4 });
        var gg = OpInput.Rand(1911, new[] { 8 }, 0.5, 1.5);
        var gb = OpInput.Rand(1912, new[] { 8 }, -0.2, 0.2);
        var ggo = OpInput.Rand(1913, new[] { 2, 8, 4, 4 });
        yield return new OpCase("GroupNormBackward[2,8,4,4;g2]", "norm-bwd",
            e => { e.GroupNorm(gx.F(), 2, gg.F(), gb.F(), 1e-5, out var mn, out var vr); return e.GroupNormBackward(ggo.F(), gx.F(), 2, gg.F(), mn, vr, 1e-5, out _, out _); },
            e => { e.GroupNorm(gx.D(), 2, gg.D(), gb.D(), 1e-5, out var mn, out var vr); return e.GroupNormBackward(ggo.D(), gx.D(), 2, gg.D(), mn, vr, 1e-5, out _, out _); }, ParityTol.Accum(2e-3), opMethod: "GroupNormBackward")
            { KnownDivergence = "GPU norm backward diverges strongly from CPU/oracle (norm-backward kernel bug/convention); CPU matches the double oracle." };
        yield return new OpCase("InstanceNormBackward[2,8,4,4]", "norm-bwd",
            e => { e.InstanceNorm(gx.F(), gg.F(), gb.F(), 1e-5, out var mn, out var vr); return e.InstanceNormBackward(ggo.F(), gx.F(), gg.F(), mn, vr, 1e-5, out _, out _); },
            e => { e.InstanceNorm(gx.D(), gg.D(), gb.D(), 1e-5, out var mn, out var vr); return e.InstanceNormBackward(ggo.D(), gx.D(), gg.D(), mn, vr, 1e-5, out _, out _); }, ParityTol.Accum(2e-3), opMethod: "InstanceNormBackward")
            { KnownDivergence = "GPU norm backward diverges strongly from CPU/oracle (norm-backward kernel bug/convention); CPU matches the double oracle." };
        yield return new OpCase("BatchNormBackward[2,8,4,4]", "norm-bwd",
            e => { e.BatchNorm(gx.F(), gg.F(), gb.F(), 1e-5, out var mn, out var vr); return e.BatchNormBackward(ggo.F(), gx.F(), gg.F(), mn, vr, 1e-5, out _, out _); },
            e => { e.BatchNorm(gx.D(), gg.D(), gb.D(), 1e-5, out var mn, out var vr); return e.BatchNormBackward(ggo.D(), gx.D(), gg.D(), mn, vr, 1e-5, out _, out _); }, ParityTol.Accum(2e-3), opMethod: "BatchNormBackward")
            { KnownDivergence = "GPU norm backward diverges strongly from CPU/oracle (norm-backward kernel bug/convention); CPU matches the double oracle." };

        // Conv / pool backward.
        yield return new OpCase("Conv2DBackwardKernel[go1,4,6,6;in1,3,8,8]", "conv",
            e => e.Conv2DBackwardKernel(OpInput.Rand(1920, new[] { 1, 4, 6, 6 }).F(), OpInput.Rand(1921, new[] { 1, 3, 8, 8 }).F(), new[] { 4, 3, 3, 3 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }),
            e => e.Conv2DBackwardKernel(OpInput.Rand(1920, new[] { 1, 4, 6, 6 }).D(), OpInput.Rand(1921, new[] { 1, 3, 8, 8 }).D(), new[] { 4, 3, 3, 3 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }), ParityTol.Accum(1e-3), opMethod: "Conv2DBackwardKernel");
        yield return new OpCase("DepthwiseConv2DBackwardInput[1,4,8,8]", "conv",
            e => e.DepthwiseConv2DBackwardInput(OpInput.Rand(1922, new[] { 1, 4, 8, 8 }).F(), OpInput.Rand(1923, new[] { 4, 1, 3, 3 }).F(), new[] { 1, 4, 8, 8 }, new[] { 1, 1 }, new[] { 1, 1 }),
            e => e.DepthwiseConv2DBackwardInput(OpInput.Rand(1922, new[] { 1, 4, 8, 8 }).D(), OpInput.Rand(1923, new[] { 4, 1, 3, 3 }).D(), new[] { 1, 4, 8, 8 }, new[] { 1, 1 }, new[] { 1, 1 }), ParityTol.Accum(1e-3), opMethod: "DepthwiseConv2DBackwardInput");
        yield return new OpCase("AvgPool3DBackward[1,2,2,2,2->1,2,4,4,4]", "pool",
            e => e.AvgPool3DBackward(OpInput.Rand(1924, new[] { 1, 2, 2, 2, 2 }).F(), new[] { 1, 2, 4, 4, 4 }, new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, new[] { 0, 0, 0 }),
            e => e.AvgPool3DBackward(OpInput.Rand(1924, new[] { 1, 2, 2, 2, 2 }).D(), new[] { 1, 2, 4, 4, 4 }, new[] { 2, 2, 2 }, new[] { 2, 2, 2 }, new[] { 0, 0, 0 }), ParityTol.Accum(1e-3), opMethod: "AvgPool3DBackward");

        yield return new OpCase("GlobalMaxPool2D[1,2,8,8]", "pool", e => e.GlobalMaxPool2D(OpInput.Rand(1930, new[] { 1, 2, 8, 8 }).F()), e => e.GlobalMaxPool2D(OpInput.Rand(1930, new[] { 1, 2, 8, 8 }).D()), ParityTol.Exact, opMethod: "GlobalMaxPool2D");
        var pr = OpInput.Rand(1931, new[] { 4, 8 }); var tg = OpInput.Rand(1932, new[] { 4, 8 });
        yield return B("MseBackward", "loss-bwd", (e, u, v) => e.MseBackward(u, v), (e, u, v) => e.MseBackward(u, v), ParityTol.Ulp(8, 1e-6), pr, tg);
    }

    // Slice-scatter, clamp-tensor, nan-to-num, squash-backward.
    public static IEnumerable<OpCase> SliceScatterMisc()
    {
        var m = OpInput.Rand(1800, new[] { 4, 8 });
        yield return new OpCase("TensorClampTensor[4,8]", "arithmetic",
            e => e.TensorClampTensor(m.F(), OpInput.Rand(1801, new[] { 4, 8 }, -0.6, -0.4).F(), OpInput.Rand(1802, new[] { 4, 8 }, 0.4, 0.6).F()),
            e => e.TensorClampTensor(m.D(), OpInput.Rand(1801, new[] { 4, 8 }, -0.6, -0.4).D(), OpInput.Rand(1802, new[] { 4, 8 }, 0.4, 0.6).D()), ParityTol.Exact, opMethod: "TensorClampTensor");
        yield return U("TensorNanToNum", "arithmetic", (e, t) => e.TensorNanToNum(t, null, null, null), (e, t) => e.TensorNanToNum(t, null, null, null), ParityTol.Exact, OpInput.Rand(1803, new[] { 4, 8 }));

        yield return new OpCase("TensorSetSlice[4,8;<-2,4@1,2]", "shape",
            e => e.TensorSetSlice(m.F(), OpInput.Rand(1804, new[] { 2, 4 }).F(), new[] { 1, 2 }),
            e => e.TensorSetSlice(m.D(), OpInput.Rand(1804, new[] { 2, 4 }).D(), new[] { 1, 2 }), ParityTol.Exact, opMethod: "TensorSetSlice");
        yield return new OpCase("TensorSliceScatter[4,8;d1,2,4]", "shape",
            e => e.TensorSliceScatter(m.F(), OpInput.Rand(1805, new[] { 4, 4 }).F(), 1, 2, 4),
            e => e.TensorSliceScatter(m.D(), OpInput.Rand(1805, new[] { 4, 4 }).D(), 1, 2, 4), ParityTol.Exact, opMethod: "TensorSliceScatter");
        yield return new OpCase("TensorSelectScatter[4,8;d0,i1]", "shape",
            e => e.TensorSelectScatter(m.F(), OpInput.Rand(1806, new[] { 8 }).F(), 0, 1),
            e => e.TensorSelectScatter(m.D(), OpInput.Rand(1806, new[] { 8 }).D(), 0, 1), ParityTol.Exact, opMethod: "TensorSelectScatter");

        var sqi = OpInput.Rand(1807, new[] { 4, 8 });
        yield return new OpCase("TensorSquashBackward[4,8]", "activation-bwd",
            e => e.TensorSquashBackward(OpInput.Rand(1808, new[] { 4, 8 }).F(), sqi.F(), e.TensorSquash(sqi.F(), -1), -1),
            e => e.TensorSquashBackward(OpInput.Rand(1808, new[] { 4, 8 }).D(), sqi.D(), e.TensorSquash(sqi.D(), -1), -1), ParityTol.Accum(1e-3), opMethod: "TensorSquashBackward");
    }

    // Grid-sample, upsample3d/crop, depthwise-1d, conv/pool backward, IoU + CE losses.
    public static IEnumerable<OpCase> GridConvBwdLoss()
    {
        // FOUND (quarantined): CPU and GPU GridSample produce DIFFERENT OUTPUT SHAPES for the same
        // input+grid (CPU 64 elements vs GPU 32) — a hard shape/convention mismatch. Tracked.
        yield return new OpCase("GridSample[1,2,4,4;g1,4,4,2]", "sample",
            e => e.GridSample(OpInput.Rand(1700, new[] { 1, 2, 4, 4 }).F(), OpInput.Rand(1701, new[] { 1, 4, 4, 2 }, -1.0, 1.0).F()),
            e => e.GridSample(OpInput.Rand(1700, new[] { 1, 2, 4, 4 }).D(), OpInput.Rand(1701, new[] { 1, 4, 4, 2 }, -1.0, 1.0).D()), ParityTol.Accum(1e-3), opMethod: "GridSample")
        { KnownDivergence = "CPU and GPU GridSample return different output shapes (64 vs 32 elements)." };
        yield return new OpCase("Upsample3D[1,2,2,2,2;2x2x2]", "shape",
            e => e.Upsample3D(OpInput.Rand(1702, new[] { 1, 2, 2, 2, 2 }).F(), 2, 2, 2),
            e => e.Upsample3D(OpInput.Rand(1702, new[] { 1, 2, 2, 2, 2 }).D(), 2, 2, 2), ParityTol.Exact, opMethod: "Upsample3D");
        yield return new OpCase("Crop[1,2,8,8;1,1,4,4]", "shape",
            e => e.Crop(OpInput.Rand(1703, new[] { 1, 2, 8, 8 }).F(), 1, 1, 4, 4),
            e => e.Crop(OpInput.Rand(1703, new[] { 1, 2, 8, 8 }).D(), 1, 1, 4, 4), ParityTol.Exact, opMethod: "Crop");
        yield return new OpCase("DepthwiseConv1D[1,4,16;k4,1,3]", "conv",
            e => e.DepthwiseConv1D(OpInput.Rand(1704, new[] { 1, 4, 16 }).F(), OpInput.Rand(1705, new[] { 4, 1, 3 }).F(), 1, 0),
            e => e.DepthwiseConv1D(OpInput.Rand(1704, new[] { 1, 4, 16 }).D(), OpInput.Rand(1705, new[] { 4, 1, 3 }).D(), 1, 0), ParityTol.Accum(1e-3), opMethod: "DepthwiseConv1D");

        yield return new OpCase("Conv2DBackwardInput[1,4,6,6->1,3,8,8]", "conv",
            e => e.Conv2DBackwardInput(OpInput.Rand(1710, new[] { 1, 4, 6, 6 }).F(), OpInput.Rand(1711, new[] { 4, 3, 3, 3 }).F(), new[] { 1, 3, 8, 8 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }),
            e => e.Conv2DBackwardInput(OpInput.Rand(1710, new[] { 1, 4, 6, 6 }).D(), OpInput.Rand(1711, new[] { 4, 3, 3, 3 }).D(), new[] { 1, 3, 8, 8 }, new[] { 1, 1 }, new[] { 0, 0 }, new[] { 1, 1 }), ParityTol.Accum(1e-3), opMethod: "Conv2DBackwardInput");
        yield return new OpCase("AvgPool2DBackward[1,2,4,4->1,2,8,8]", "pool",
            e => e.AvgPool2DBackward(OpInput.Rand(1712, new[] { 1, 2, 4, 4 }).F(), new[] { 1, 2, 8, 8 }, new[] { 2, 2 }, new[] { 2, 2 }),
            e => e.AvgPool2DBackward(OpInput.Rand(1712, new[] { 1, 2, 4, 4 }).D(), new[] { 1, 2, 8, 8 }, new[] { 2, 2 }, new[] { 2, 2 }), ParityTol.Accum(1e-3), opMethod: "AvgPool2DBackward");

        // IoU-family box losses (predicted vs target boxes [N,4]).
        var pb = OpInput.From(new double[] { 0, 0, 2, 2, 1, 1, 3, 4, 0, 1, 4, 3 }, new[] { 3, 4 });
        var tb = OpInput.From(new double[] { 0, 0, 2, 3, 1, 0, 3, 3, 1, 1, 5, 4 }, new[] { 3, 4 });
        yield return new OpCase("TensorIoULoss[3,4]", "loss", e => e.TensorIoULoss(pb.F(), tb.F()), e => e.TensorIoULoss(pb.D(), tb.D()), ParityTol.Accum(1e-3), opMethod: "TensorIoULoss");
        yield return new OpCase("TensorGIoULoss[3,4]", "loss", e => e.TensorGIoULoss(pb.F(), tb.F()), e => e.TensorGIoULoss(pb.D(), tb.D()), ParityTol.Accum(1e-3), opMethod: "TensorGIoULoss");
        yield return new OpCase("TensorDIoULoss[3,4]", "loss", e => e.TensorDIoULoss(pb.F(), tb.F()), e => e.TensorDIoULoss(pb.D(), tb.D()), ParityTol.Accum(1e-3), opMethod: "TensorDIoULoss");
        var logits = OpInput.Rand(1720, new[] { 4, 8 }, -3.0, 3.0);
        var ceTgt = OpInput.Rand(1721, new[] { 4, 8 }, 0.0, 1.0);
        yield return new OpCase("TensorCrossEntropyLoss[4,8]", "loss", e => e.TensorCrossEntropyLoss(logits.F(), ceTgt.F()), e => e.TensorCrossEntropyLoss(logits.D(), ceTgt.D()), ParityTol.Accum(1e-3), opMethod: "TensorCrossEntropyLoss");
    }

    // 1D pools, permute/slice/take, index-add, Tensor* pool/conv variants.
    public static IEnumerable<OpCase> SlicePoolTake()
    {
        var seq = OpInput.Rand(1600, new[] { 1, 4, 16 });
        yield return new OpCase("TensorAvgPool1D[1,4,16;k2s2]", "pool", e => e.TensorAvgPool1D(seq.F(), 2, 2), e => e.TensorAvgPool1D(seq.D(), 2, 2), ParityTol.Accum(1e-3), opMethod: "TensorAvgPool1D");
        yield return new OpCase("TensorMaxPool1D[1,4,16;k2s2]", "pool", e => e.TensorMaxPool1D(seq.F(), 2, 2), e => e.TensorMaxPool1D(seq.D(), 2, 2), ParityTol.Exact, opMethod: "TensorMaxPool1D");

        var t3 = OpInput.Rand(1601, new[] { 2, 3, 4 });
        yield return new OpCase("TensorPermute[2,3,4;201]", "shape", e => e.TensorPermute(t3.F(), new[] { 2, 0, 1 }), e => e.TensorPermute(t3.D(), new[] { 2, 0, 1 }), ParityTol.Exact, opMethod: "TensorPermute");
        var m = OpInput.Rand(1602, new[] { 4, 8 });
        yield return new OpCase("TensorSlice[4,8;s1,2;l2,4]", "shape", e => e.TensorSlice(m.F(), new[] { 1, 2 }, new[] { 2, 4 }), e => e.TensorSlice(m.D(), new[] { 1, 2 }, new[] { 2, 4 }), ParityTol.Exact, opMethod: "TensorSlice");
        yield return new OpCase("TensorSliceAxis[4,8;ax0,i1]", "shape", e => e.TensorSliceAxis(m.F(), 0, 1), e => e.TensorSliceAxis(m.D(), 0, 1), ParityTol.Exact, opMethod: "TensorSliceAxis");

        yield return new OpCase("TensorTake[4,8;idx5]", "index",
            e => e.TensorTake(m.F(), new Tensor<int>(new[] { 0, 7, 15, 20, 31 }, new[] { 5 })),
            e => e.TensorTake(m.D(), new Tensor<int>(new[] { 0, 7, 15, 20, 31 }, new[] { 5 })), ParityTol.Exact, opMethod: "TensorTake");
        {
            var tad = new int[32]; for (int i = 0; i < 32; i++) tad[i] = i % 8;
            yield return new OpCase("TensorTakeAlongDim[4,8;d1]", "index",
                e => e.TensorTakeAlongDim(m.F(), new Tensor<int>((int[])tad.Clone(), new[] { 4, 8 }), 1),
                e => e.TensorTakeAlongDim(m.D(), new Tensor<int>((int[])tad.Clone(), new[] { 4, 8 }), 1), ParityTol.Exact, opMethod: "TensorTakeAlongDim");
        }
        yield return new OpCase("TensorIndexAdd[4,8;ax0]", "index",
            e => e.TensorIndexAdd(m.F(), 0, new Tensor<int>(new[] { 0, 2 }, new[] { 2 }), OpInput.Rand(1603, new[] { 2, 8 }).F()),
            e => e.TensorIndexAdd(m.D(), 0, new Tensor<int>(new[] { 0, 2 }, new[] { 2 }), OpInput.Rand(1603, new[] { 2, 8 }).D()), ParityTol.Ulp(4, 1e-6), opMethod: "TensorIndexAdd");

        var pool = OpInput.Rand(1610, new[] { 1, 2, 8, 8 });
        yield return new OpCase("TensorMaxPool2D[1,2,8,8;k2]", "pool", e => e.TensorMaxPool2D(pool.F(), 2), e => e.TensorMaxPool2D(pool.D(), 2), ParityTol.Exact, opMethod: "TensorMaxPool2D");
        yield return new OpCase("TensorAvgPool2D[1,2,8,8;k2]", "pool", e => e.TensorAvgPool2D(pool.F(), 2), e => e.TensorAvgPool2D(pool.D(), 2), ParityTol.Accum(1e-3), opMethod: "TensorAvgPool2D");
        yield return new OpCase("TensorConv2D[1,3,8,8;k4,3,3,3]", "conv",
            e => e.TensorConv2D(OpInput.Rand(1611, new[] { 1, 3, 8, 8 }).F(), OpInput.Rand(1612, new[] { 4, 3, 3, 3 }).F(), 1, 0, 1),
            e => e.TensorConv2D(OpInput.Rand(1611, new[] { 1, 3, 8, 8 }).D(), OpInput.Rand(1612, new[] { 4, 3, 3, 3 }).D(), 1, 0, 1), ParityTol.Accum(1e-3), opMethod: "TensorConv2D");
    }

    // Special functions, attention, batchnorm.
    public static IEnumerable<OpCase> SpecialAttnNorm()
    {
        var s = new[] { 4, 8 };
        yield return U("TensorI1", "arithmetic", (e, t) => e.TensorI1(t), (e, t) => e.TensorI1(t), ParityTol.Ulp(256, 1e-5), OpInput.Rand(1500, s, -2.0, 2.0));
        yield return U("TensorI0e", "arithmetic", (e, t) => e.TensorI0e(t), (e, t) => e.TensorI0e(t), ParityTol.Ulp(256, 1e-5), OpInput.Rand(1501, s, -2.0, 2.0));
        yield return U("TensorI1e", "arithmetic", (e, t) => e.TensorI1e(t), (e, t) => e.TensorI1e(t), ParityTol.Ulp(256, 1e-5), OpInput.Rand(1502, s, -2.0, 2.0));
        yield return B("TensorLogAddExp2", "arithmetic", (e, u, v) => e.TensorLogAddExp2(u, v), (e, u, v) => e.TensorLogAddExp2(u, v), ParityTol.Ulp(64, 1e-6), OpInput.Rand(1503, s), OpInput.Rand(1504, s));
        yield return B("TensorNextAfter", "arithmetic", (e, u, v) => e.TensorNextAfter(u, v), (e, u, v) => e.TensorNextAfter(u, v), ParityTol.Ulp(2, 1e-30), OpInput.Rand(1505, s), OpInput.Rand(1506, s));
        yield return B("TensorXlog1py", "arithmetic", (e, u, v) => e.TensorXlog1py(u, v), (e, u, v) => e.TensorXlog1py(u, v), ParityTol.Ulp(64, 1e-6), OpInput.Rand(1507, s), OpInput.RandPositive(1508, s, 0.1, 4.0));
        yield return new OpCase("TensorPolygamma[n1;4,8]", "arithmetic", e => e.TensorPolygamma(1, OpInput.RandPositive(1509, s, 0.5, 4.0).F()), e => e.TensorPolygamma(1, OpInput.RandPositive(1509, s, 0.5, 4.0).D()), ParityTol.Ulp(512, 1e-4), opMethod: "TensorPolygamma");
        {
            var xe = new int[32]; for (int k = 0; k < 32; k++) xe[k] = (k % 5) - 2;
            yield return new OpCase("TensorLdexp[4,8]", "arithmetic",
                e => e.TensorLdexp(OpInput.Rand(1510, s).F(), new Tensor<int>((int[])xe.Clone(), s)),
                e => e.TensorLdexp(OpInput.Rand(1510, s).D(), new Tensor<int>((int[])xe.Clone(), s)), ParityTol.Ulp(2, 1e-6), opMethod: "TensorLdexp");
        }

        // Scaled dot-product attention (Q,K,V same shape).
        var qq = OpInput.Rand(1520, new[] { 4, 8 });
        var kk = OpInput.Rand(1521, new[] { 4, 8 });
        var vv = OpInput.Rand(1522, new[] { 4, 8 });
        yield return new OpCase("TensorScaledDotProductAttention[4,8]", "attention", e => e.TensorScaledDotProductAttention(qq.F(), kk.F(), vv.F()), e => e.TensorScaledDotProductAttention(qq.D(), kk.D(), vv.D()), ParityTol.Accum(1e-3), opMethod: "TensorScaledDotProductAttention");

        // BatchNorm (train stats) + inference (given stats).
        var bx = OpInput.Rand(1530, new[] { 2, 4, 4, 4 });
        var bg = OpInput.Rand(1531, new[] { 4 }, 0.5, 1.5);
        var bb = OpInput.Rand(1532, new[] { 4 }, -0.2, 0.2);
        yield return new OpCase("BatchNorm[2,4,4,4]", "norm", e => { var y = e.BatchNorm(bx.F(), bg.F(), bb.F(), 1e-5, out _, out _); return y; }, e => { var y = e.BatchNorm(bx.D(), bg.D(), bb.D(), 1e-5, out _, out _); return y; }, ParityTol.Accum(1e-3), opMethod: "BatchNorm");
        var bm = OpInput.Rand(1533, new[] { 4 }, -0.5, 0.5);
        var bvv = OpInput.Rand(1534, new[] { 4 }, 0.5, 1.5);
        yield return new OpCase("BatchNormInference[2,4,4,4]", "norm", e => e.BatchNormInference(bx.F(), bg.F(), bb.F(), bm.F(), bvv.F(), 1e-5), e => e.BatchNormInference(bx.D(), bg.D(), bb.D(), bm.D(), bvv.D(), 1e-5), ParityTol.Accum(1e-3), opMethod: "BatchNormInference");
    }

    // Conv-transpose, fused-linear, adaptive/3D pooling, batch-outer, linspace.
    public static IEnumerable<OpCase> ConvPoolLinear()
    {
        yield return new OpCase("ConvTranspose2D[1,4,4,4;k4,8,2,2;s2]", "conv",
            e => e.ConvTranspose2D(OpInput.Rand(1400, new[] { 1, 4, 4, 4 }).F(), OpInput.Rand(1401, new[] { 4, 8, 2, 2 }).F(), new[] { 2, 2 }, new[] { 0, 0 }, new[] { 0, 0 }),
            e => e.ConvTranspose2D(OpInput.Rand(1400, new[] { 1, 4, 4, 4 }).D(), OpInput.Rand(1401, new[] { 4, 8, 2, 2 }).D(), new[] { 2, 2 }, new[] { 0, 0 }, new[] { 0, 0 }), ParityTol.Accum(1e-3), opMethod: "ConvTranspose2D");

        var lin = OpInput.Rand(1410, new[] { 4, 8 });
        var w = OpInput.Rand(1411, new[] { 8, 6 }); // [in, out] — FusedLinear does input·weight
        var bias = OpInput.Rand(1412, new[] { 6 });
        yield return new OpCase("FusedLinearReLU[4,8;w8,6]", "matmul", e => e.FusedLinearReLU(lin.F(), w.F(), bias.F()), e => e.FusedLinearReLU(lin.D(), w.D(), bias.D()), ParityTol.Accum(1e-3), opMethod: "FusedLinearReLU");
        yield return new OpCase("FusedLinearGELU[4,8;w8,6]", "matmul", e => e.FusedLinearGELU(lin.F(), w.F(), bias.F()), e => e.FusedLinearGELU(lin.D(), w.D(), bias.D()), ParityTol.Accum(1e-3), opMethod: "FusedLinearGELU");

        var pool = OpInput.Rand(1420, new[] { 1, 2, 8, 8 });
        yield return new OpCase("AdaptiveAvgPool2D[1,2,8,8->4,4]", "pool", e => e.AdaptiveAvgPool2D(pool.F(), 4, 4), e => e.AdaptiveAvgPool2D(pool.D(), 4, 4), ParityTol.Accum(1e-3), opMethod: "AdaptiveAvgPool2D");
        yield return new OpCase("TensorAdaptiveMaxPool2D[1,2,8,8->4,4]", "pool", e => e.TensorAdaptiveMaxPool2D(pool.F(), new[] { 4, 4 }), e => e.TensorAdaptiveMaxPool2D(pool.D(), new[] { 4, 4 }), ParityTol.Exact, opMethod: "TensorAdaptiveMaxPool2D");
        var pool3 = OpInput.Rand(1421, new[] { 1, 2, 4, 4, 4 });
        yield return new OpCase("AvgPool3D[1,2,4,4,4;k2]", "pool", e => e.AvgPool3D(pool3.F(), 2), e => e.AvgPool3D(pool3.D(), 2), ParityTol.Accum(1e-3), opMethod: "AvgPool3D");
        yield return new OpCase("MaxPool3D[1,2,4,4,4;k2]", "pool", e => e.MaxPool3D(pool3.F(), 2), e => e.MaxPool3D(pool3.D(), 2), ParityTol.Exact, opMethod: "MaxPool3D");

        yield return B("TensorBatchOuterProduct", "matmul", (e, u, v) => e.TensorBatchOuterProduct(u, v), (e, u, v) => e.TensorBatchOuterProduct(u, v), ParityTol.Ulp(2, 1e-6), OpInput.Rand(1430, new[] { 4, 3 }), OpInput.Rand(1431, new[] { 4, 5 }));
        yield return new OpCase("TensorLinspace[0,1,8]", "misc", e => e.TensorLinspace<float>(0f, 1f, 8), e => e.TensorLinspace<double>(0.0, 1.0, 8), ParityTol.Ulp(4, 1e-6), opMethod: "TensorLinspace");
    }

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

        // FOUND (quarantined): GPU MishBackward is NONDETERMINISTIC run-to-run (racy/uninitialized
        // kernel, like the frac bug) — a real GPU bug the determinism check catches intermittently.
        yield return new OpCase("MishBackward[4,64]", "activation-bwd", e => e.MishBackward(go.F(), inp.F()), e => e.MishBackward(go.D(), inp.D()), ParityTol.Ulp(64, 1e-6), opMethod: "MishBackward")
        { KnownDivergence = "GPU MishBackward is nondeterministic run-to-run (racy/uninitialized backward kernel)." };
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
