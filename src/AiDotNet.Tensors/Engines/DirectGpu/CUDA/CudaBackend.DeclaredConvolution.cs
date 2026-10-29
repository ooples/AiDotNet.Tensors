// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Convolutions on declared programs (#890 phase 4b). A convolution's spec -- forward, or the data or weight gradient
// derived from it by CodegenAdjoint -- is lowered as a tiled implicit GEMM (PtxAffineEmitter.EmitImplicitGemm), on its
// own or split along a reduction axis (CodegenSplitReduction.TryPlanChunkedForTiles) for small outputs with long
// reductions. Each schedule is a candidate in the tuned convolution slots, so it serves a shape class only where the
// slot measured it correct and fastest; the established routes stay the reference.
//
// A call allocates nothing: the geometry and schedule are a value key, and the program (modules, scratch, argument
// slots) is built once per key through the same path as the other declared families.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>Which convolution operator a declared program computes.</summary>
internal enum DeclaredConvPass : byte
{
    Forward,
    BackwardData,
    BackwardWeights,
}

/// <summary>
/// A convolution schedule: the implicit GEMM's output tile (M rows -- what only the first operand reads -- by N
/// columns), or tile 0 for the affine emitter's own lowering (a depthwise convolution's kh*kw-long reduction has no
/// GEMM worth tiling); above 1,
/// the factor a reduction axis is split by into a partial pass and a combine; and whether a strided data gradient is
/// split by output parity first (CodegenParitySplit), so no pass visits a tap its exact-division window lacks.
/// </summary>
internal readonly record struct DeclaredConvSchedule(int TileM, int TileN, int SplitFactor, bool Parity = false)
{
    /// <summary>The candidate-name suffix: "m128n64", "split4.m128n64" or "parity.m64n64".</summary>
    internal string Suffix => (Parity ? "parity." : "") + (SplitFactor > 1 ? "split" + SplitFactor + "." : "") +
        (TileM == 0 ? "affine" : "m" + TileM + "n" + TileN);
}

/// <summary>
/// A declared convolution program: one square-window geometry and channel grouping (Groups = InChannels is a depthwise
/// convolution), the fused bias + ReLU or not, a schedule.
/// </summary>
internal readonly record struct DeclaredConvKey(
    DeclaredConvPass Pass, int Batch, int InChannels, int OutChannels, int Height, int Width,
    int Kernel, int Stride, int Pad, int Dilation, bool BiasRelu, DeclaredConvSchedule Schedule, int Groups = 1);

public sealed partial class CudaBackend
{
    // The schedules registered per slot, each the fastest declared one for some measured shape (sm_75,
    // --kernel-routes --family conv): every candidate costs online tuning time, so one that never wins is not
    // registered. Forward and data gradients are large-output GEMMs, where a split only adds the combine's traffic;
    // the weight gradient's output is filters-sized and its reduction runs over every position, which is what the
    // split is for -- the deeper the smaller the output (1x1, 64 -> 64 channels: unsplit 1406 us, split 14 173 us).
    private static readonly DeclaredConvSchedule[] s_declaredConvLargeOutputSchedules =
    {
        new(64, 64, 0), new(128, 64, 0),
    };

    // A strided data gradient also tries the parity split: at stride 2 three quarters of a 3x3 window's taps are
    // absent for any one input position, and the unsplit GEMM gathers zeros for them.
    private static readonly DeclaredConvSchedule[] s_declaredConvDataSchedules =
    {
        new(64, 64, 0), new(128, 64, 0), new(64, 64, 0, Parity: true),
    };

    private static readonly DeclaredConvSchedule[] s_declaredConvWeightSchedules =
    {
        new(128, 64, 4), new(64, 64, 4), new(64, 64, 7), new(64, 64, 14),
    };

    /// <summary>The schedules registered as candidates for a pass (the route tooling times each).</summary>
    internal static IReadOnlyList<DeclaredConvSchedule> DeclaredConvSchedules(DeclaredConvPass pass) => pass switch
    {
        DeclaredConvPass.Forward => s_declaredConvLargeOutputSchedules,
        DeclaredConvPass.BackwardData => s_declaredConvDataSchedules,
        _ => s_declaredConvWeightSchedules,
    };

    private static readonly string[] s_convForwardNames = { "input", "weights", "output" };
    private static readonly string[] s_convForwardBiasReluNames = { "input", "weights", "bias", "output" };
    private static readonly string[] s_convBackwardDataNames = { "dOut", "weights", "dinput" };
    private static readonly string[] s_convBackwardWeightNames = { "dOut", "input", "dweights" };

    // Lowered programs by key; a null entry records a key that declined, so it is not lowered again.
    private readonly Dictionary<DeclaredConvKey, DeclaredProgram?> _declaredConvPrograms = new();
    private readonly Dictionary<DeclaredConvKey, string> _declaredConvDeclines = new();

    /// <summary>
    /// The spec of a convolution pass, or null when the geometry is outside what the declared builder takes: one
    /// window size, stride, padding and dilation for both axes, and the caller's output extent.
    /// </summary>
    internal static CodegenKernelSpec? DeclaredConvSpec(DeclaredConvPass pass, int batch, int inChannels,
        int outChannels, int height, int width, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int outHeight, int outWidth, bool biasRelu, int groups = 1)
    {
        if (groups < 1 || inChannels % groups != 0 || outChannels % groups != 0) return null;
        if (!DeclaredConvGeometryFits(batch, inChannels, outChannels, height, width, kernelH, kernelW, strideH, strideW,
                padH, padW, dilationH, dilationW, outHeight, outWidth))
            return null;
        var forward = CodegenKernelCatalog.Conv2D(batch, inChannels, outChannels, height, width, kernelH, strideH, padH,
            dilationH, groups, epilogue: pass == DeclaredConvPass.Forward && biasRelu);
        return pass switch
        {
            DeclaredConvPass.Forward => forward,
            DeclaredConvPass.BackwardData => CodegenAdjoint.BackwardData(forward, 0),
            _ => CodegenAdjoint.BackwardWeights(forward, 1),
        };
    }

    // The geometry checks of DeclaredConvSpec without building the spec, so a call can test them without allocating.
    private static bool DeclaredConvGeometryFits(int batch, int inChannels, int outChannels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        int outHeight, int outWidth)
    {
        if (kernelH != kernelW || strideH != strideW || padH != padW || dilationH != dilationW) return false;
        if (batch <= 0 || inChannels <= 0 || outChannels <= 0 || kernelH <= 0 || strideH <= 0 || dilationH <= 0 ||
            padH < 0)
            return false;
        int span = dilationH * (kernelH - 1) + 1;
        int expectedH = (height + 2 * padH - span) / strideH + 1, expectedW = (width + 2 * padH - span) / strideH + 1;
        if (height + 2 * padH < span || width + 2 * padH < span || expectedH != outHeight || expectedW != outWidth)
            return false;
        // Every tensor is addressed with 32-bit offsets.
        return (long)batch * inChannels * height * width <= int.MaxValue &&
            (long)batch * outChannels * outHeight * outWidth <= int.MaxValue;
    }

    // Whether a schedule can lower a slot's shape class (the slot asks before it measures a candidate).
    private bool DeclaredConvAdmits(DeclaredConvPass pass, TunedShape s, DeclaredConvSchedule schedule)
    {
        if (!DeclaredKernelsEnabled) return false;
        var spec = DeclaredConvSpec(pass, s[0], s[1], s[4], s[2], s[3], s[5], s[6], s[7], s[8], s[9], s[10], s[11],
            s[12], s[13], s[14], biasRelu: false);
        return spec is not null && DeclaredConvSpecAdmits(spec, schedule);
    }

    /// <summary>Runs a forward convolution as a declared program; false when the shape declines.</summary>
    internal bool TryDeclaredConv2DForward(in CudaConv2DArgs a, DeclaredConvSchedule schedule)
    {
        bool biasRelu = a.Bias is not null && a.Activation == FusedActivationType.ReLU;
        if (!DeclaredConvGeometryFits(a.Batch, a.InChannels, a.OutChannels, a.InHeight, a.InWidth, a.KernelH, a.KernelW,
                a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW, a.OutHeight, a.OutWidth))
            return false;
        var key = new DeclaredConvKey(DeclaredConvPass.Forward, a.Batch, a.InChannels, a.OutChannels, a.InHeight,
            a.InWidth, a.KernelH, a.StrideH, a.PadH, a.DilationH, biasRelu, schedule);
        bool ran = a.Bias is { } bias && biasRelu
            ? TryDeclaredConv(key, s_convForwardBiasReluNames, new DeclaredBindings(a.Input, a.Kernel, bias, a.Output))
            : TryDeclaredConv(key, s_convForwardNames, new DeclaredBindings(a.Input, a.Kernel, a.Output));
        // Any other epilogue runs after the convolution, as on the routes whose kernels do not take it.
        if (ran && !biasRelu) RunConvEpiloguePasses(a);
        return ran;
    }

    /// <summary>Runs a data or weight gradient as a declared program; false when the shape declines.</summary>
    internal bool TryDeclaredConv2DBackward(in TunedConv2DBackwardArgs a, DeclaredConvSchedule schedule)
    {
        var pass = a.WeightGradient ? DeclaredConvPass.BackwardWeights : DeclaredConvPass.BackwardData;
        if (!DeclaredConvGeometryFits(a.Batch, a.InChannels, a.OutChannels, a.InHeight, a.InWidth, a.KernelH, a.KernelW,
                a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW, a.OutHeight, a.OutWidth))
            return false;
        var key = new DeclaredConvKey(pass, a.Batch, a.InChannels, a.OutChannels, a.InHeight, a.InWidth, a.KernelH,
            a.StrideH, a.PadH, a.DilationH, BiasRelu: false, schedule);
        // Weight gradient: First = input, Second = gradOutput. Data gradient: First = gradOutput, Second = kernel.
        return a.WeightGradient
            ? TryDeclaredConv(key, s_convBackwardWeightNames, new DeclaredBindings(a.Second, a.First, a.Output))
            : TryDeclaredConv(key, s_convBackwardDataNames, new DeclaredBindings(a.First, a.Second, a.Output));
    }

    private bool TryDeclaredConv(in DeclaredConvKey key, string[] names, in DeclaredBindings bound)
    {
        if (!DeclaredKernelsEnabled) return false;
        DeclaredProgram? program;
        lock (_directPtxLock)
        {
            if (!_declaredConvPrograms.TryGetValue(key, out program))
            {
                if (IsStreamCapturing()) return false;
                program = BuildDeclaredConv(key, names);
                _declaredConvPrograms[key] = program;
            }
        }
        if (program is null || !AlignedFor(program, names, bound)) return false;
        RunDeclaredProgram(program, names, bound);
        return true;
    }

    // The program a key lowers: the pass's spec, or its parity split; null with the reason when there is none.
    private static CodegenProgram? DeclaredConvSource(in DeclaredConvKey key, out string? reason)
    {
        int outH = (key.Height + 2 * key.Pad - key.Dilation * (key.Kernel - 1) - 1) / key.Stride + 1;
        int outW = (key.Width + 2 * key.Pad - key.Dilation * (key.Kernel - 1) - 1) / key.Stride + 1;
        var spec = DeclaredConvSpec(key.Pass, key.Batch, key.InChannels, key.OutChannels, key.Height, key.Width,
            key.Kernel, key.Kernel, key.Stride, key.Stride, key.Pad, key.Pad, key.Dilation, key.Dilation, outH, outW,
            key.BiasRelu, key.Groups);
        reason = null;
        if (spec is null)
        {
            reason = "the geometry is outside the declared convolution builder";
            return null;
        }
        if (!key.Schedule.Parity) return new CodegenProgram(new[] { spec }, Array.Empty<long>(), spec.Name);
        var parity = CodegenParitySplit.TrySplit(spec, out string splitReason);
        if (parity is null) reason = splitReason;
        return parity;
    }

    private DeclaredProgram? BuildDeclaredConv(DeclaredConvKey key, string[] names)
    {
        if (DeclaredConvSource(key, out string? reason) is not { } source)
        {
            _declaredConvDeclines[key] = reason ?? "no program";
            return null;
        }
        var schedule = key.Schedule;
        var program = BuildDeclaredProgram(source, key.ToString(), names,
            pass => LowerDeclaredConv(pass, schedule, minBlocks: 2), out string? decline);
        // The two-blocks-a-SM bound caps registers at 128; a ragged shape's guards can need more, and the JIT then
        // spills, which the direct-kernel contract rejects. Unbounded, it fits one block a SM instead.
        if (program is null && decline is not null && decline.IndexOf("local bytes", StringComparison.Ordinal) >= 0)
            program = BuildDeclaredProgram(source, key.ToString(), names,
                pass => LowerDeclaredConv(pass, schedule, minBlocks: 0), out decline);
        if (decline is not null) _declaredConvDeclines[key] = decline;
        return program;
    }

    // The schedule's lowering of the convolution pass; NotSupportedException declines the key.
    private PtxPassLowering LowerDeclaredConv(CodegenKernelSpec pass, DeclaredConvSchedule schedule, int minBlocks,
        bool vectorCombine = false)
    {
        string label = (schedule.TileM == 0 ? "affine " : "implicit-gemm ") + schedule.Suffix;
        if (schedule.SplitFactor <= 1)
            return new PtxPassLowering(new[] { EmitConvKernel(pass, schedule, minBlocks) }, null, label);
        var split = CodegenSplitReduction.TryPlanChunkedForTiles(pass, schedule.SplitFactor)
            ?? throw new NotSupportedException(pass.Name + " has no reduction axis " + schedule.SplitFactor + " divides.");
        var combine = new PtxAffineEmitter { VectorTileReductions = vectorCombine };
        string combinePtx = combine.Emit(split.Combine, _ccMajor, _ccMinor);
        return new PtxPassLowering(new[]
        {
            EmitConvKernel(split.Partial, schedule, minBlocks),
            new PtxPassKernel(combinePtx, split.Combine.Name, combine.LaunchBlocks, (uint)combine.LaunchBlockX,
                (uint)combine.LaunchBlockY),
        }, split, label);
    }

    // The schedule's kernel for one pass: the affine emitter's own lowering at tile 0, else the implicit GEMM.
    private PtxPassKernel EmitConvKernel(CodegenKernelSpec spec, DeclaredConvSchedule schedule, int minBlocks)
    {
        if (schedule.TileM != 0) return EmitImplicitGemmKernel(spec, schedule, minBlocks);
        var emitter = new PtxAffineEmitter();
        string ptx = emitter.Emit(spec, _ccMajor, _ccMinor);
        return new PtxPassKernel(ptx, spec.Name, emitter.LaunchBlocks, (uint)emitter.LaunchBlockX,
            (uint)emitter.LaunchBlockY);
    }

    // Whether a schedule can lower a convolution pass's spec (decided without compiling anything).
    private static bool DeclaredConvSpecAdmits(CodegenKernelSpec spec, DeclaredConvSchedule schedule)
    {
        bool Plans(CodegenKernelSpec s) => schedule.TileM == 0 || PtxAffineEmitter.PlanImplicitGemm(s, out _) is not null;
        if (!Plans(spec)) return false;
        if (schedule.Parity)
        {
            if (CodegenParitySplit.TrySplit(spec, out _) is not { } parity) return false;
            foreach (var parityPass in parity.Passes)
                if (!Plans(parityPass)) return false;
        }
        if (schedule.SplitFactor <= 1) return true;
        return CodegenSplitReduction.TryPlanChunkedForTiles(spec, schedule.SplitFactor) is { } split && Plans(split.Partial);
    }

    private PtxPassKernel EmitImplicitGemmKernel(CodegenKernelSpec spec, DeclaredConvSchedule schedule, int minBlocks)
    {
        var emitter = new PtxAffineEmitter
        {
            ImplicitGemm = true, ImplicitGemmTile = schedule.TileN, ImplicitGemmTileM = schedule.TileM,
            ImplicitGemmMinBlocksPerMultiprocessor = minBlocks,
        };
        string ptx = emitter.Emit(spec, _ccMajor, _ccMinor);
        // The emitter falls back to its affine lowering when the implicit GEMM does not apply; that is not this
        // candidate, so the key declines rather than run under its name.
        if (!emitter.TileDescription.StartsWith("implicit-gemm", StringComparison.Ordinal))
            throw new NotSupportedException(spec.Name + " has no implicit-GEMM lowering.");
        return new PtxPassKernel(ptx, spec.Name, emitter.LaunchBlocks, (uint)emitter.LaunchBlockX,
            (uint)emitter.LaunchBlockY);
    }

    /// <summary>
    /// A convolution pass as a resident-program stage: the program and kernels the routed call runs for
    /// <paramref name="key"/>, bound to <paramref name="first"/>, <paramref name="second"/> and
    /// <paramref name="output"/> in the pass's roles (forward: input, weights, output; data gradient: dOut, weights,
    /// dInput; weight gradient: dOut, input, dWeights).
    /// </summary>
    internal DeclaredResidentStage ConvStage(DeclaredConvKey key, IGpuBuffer first, IGpuBuffer second, IGpuBuffer output)
    {
        if (key.BiasRelu) throw new ArgumentException("A bias + ReLU stage takes four buffers.", nameof(key));
        var source = DeclaredConvSource(key, out string? reason)
            ?? throw new NotSupportedException("Convolution " + key + " has no declared program: " + reason);
        string[] names = key.Pass switch
        {
            DeclaredConvPass.Forward => s_convForwardNames,
            DeclaredConvPass.BackwardData => s_convBackwardDataNames,
            _ => s_convBackwardWeightNames,
        };
        var schedule = key.Schedule;
        // The dimensions are for diagnostics; the Name (the whole convolution key) is what makes it unique.
        var declaredKey = new DeclaredKey(DeclaredFamily.Convolution, key.Batch, key.InChannels, key.OutChannels,
            key.Height, 0, 0, key.ToString(), key.Width, key.Kernel);
        return new DeclaredResidentStage(declaredKey, _ => source, names, new DeclaredBindings(first, second, output),
            pass => LowerDeclaredConv(pass, schedule, minBlocks: 2));
    }

    /// <summary>A depthwise or transposed convolution as a resident stage, in its entry point's buffer order.</summary>
    internal DeclaredResidentStage ConvVariantStage(in CudaConvVariantArgs a, DeclaredConvSchedule schedule)
    {
        var e = ConvVariantEquivalent(a.Variant, a.Shape())
            ?? throw new NotSupportedException(a.Variant + " needs a square window, stride and padding.");
        var key = new DeclaredConvKey(e.Pass, e.Batch, e.InChannels, e.OutChannels, e.Height, e.Width, e.Kernel,
            e.Stride, e.Pad, 1, BiasRelu: false, schedule, e.Groups);
        return ConvStage(key, a.A, a.B, a.Output);
    }

    /// <summary>Why a declared convolution key declined, or null when it lowered (or was never asked for).</summary>
    internal string? DeclaredConvDeclineReason(DeclaredConvKey key)
    {
        lock (_directPtxLock) return _declaredConvDeclines.TryGetValue(key, out string? reason) ? reason : null;
    }

    private void DisposeDeclaredConvolutions()
    {
        foreach (var program in _declaredConvPrograms.Values) program?.Dispose();
        _declaredConvPrograms.Clear();
    }

    // Candidates for the tuned slots: one per schedule, registered only while the declared route is on for the process
    // (persisted decisions are keyed by the registered set). A key that declines at run time -- a capture before it was
    // built, a driver rejection -- falls back to the given route so the call still completes.
    private void AddDeclaredConvForwardCandidates(List<ITunedKernelCandidate<CudaConv2DArgs>> candidates)
    {
        if (!s_declaredKernels) return;
        foreach (var schedule in s_declaredConvLargeOutputSchedules)
        {
            var sc = schedule;
            candidates.Add(new CudaTunedKernelCandidate<CudaConv2DArgs>("ptx.conv2d.implicit_gemm." + sc.Suffix,
                TunedKernelOrigin.Generated, true, shape => DeclaredConvAdmits(DeclaredConvPass.Forward, shape, sc),
                (in CudaConv2DArgs a) =>
                {
                    if (TryDeclaredConv2DForward(a, sc)) return;
                    LaunchConv2DGeneric(a.Input, a.Kernel, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
                        a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW,
                        a.PadH, a.PadW, a.DilationH, a.DilationW);
                    RunConvEpiloguePasses(a);
                }));
        }
    }

    private void AddDeclaredConvBackwardCandidates(List<ITunedKernelCandidate<TunedConv2DBackwardArgs>> candidates,
        bool weightGradient)
    {
        if (!s_declaredKernels) return;
        var pass = weightGradient ? DeclaredConvPass.BackwardWeights : DeclaredConvPass.BackwardData;
        string prefix = weightGradient ? "ptx.conv2d_dw.implicit_gemm." : "ptx.conv2d_dx.implicit_gemm.";
        foreach (var schedule in weightGradient ? s_declaredConvWeightSchedules : s_declaredConvDataSchedules)
        {
            var sc = schedule;
            candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>(prefix + sc.Suffix,
                TunedKernelOrigin.Generated, true, shape => DeclaredConvAdmits(pass, shape, sc),
                (in TunedConv2DBackwardArgs a) =>
                {
                    if (TryDeclaredConv2DBackward(a, sc)) return;
                    if (a.WeightGradient) LaunchDwDirect(a);
                    else LaunchDxDirect(a);
                }));
        }
    }
}
