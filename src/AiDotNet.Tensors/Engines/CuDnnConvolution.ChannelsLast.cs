// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Convolutions whose activations are channels-last (NHWC): every pass with a format per tensor, so a compiled plan that
// keeps its activations NHWC runs cuDNN's tensor-core kernels on them directly, without the NCHW <-> NHWC passes cuDNN
// runs around them for NCHW tensors.
using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines;

public static partial class CuDnnNative
{
    /// <summary>One result of the backward-data and backward-filter algorithm queries (cudnnConvolutionBwd*AlgoPerf_t:
    /// the same 48-byte layout as <see cref="CudnnConvolutionFwdAlgoPerf"/>, the algorithm as an int).</summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct CudnnConvolutionAlgoPerf
    {
        public int Algo;
        public CudnnStatus Status;
        public float Time;
        public UIntPtr Memory;
        public int Determinism;
        public CudnnMathType MathType;
        private int _reserved0, _reserved1, _reserved2;
    }

    [DllImport(CudnnLibrary, EntryPoint = "cudnnFindConvolutionForwardAlgorithmEx")]
    internal static extern CudnnStatus cudnnFindConvolutionForwardAlgorithmEx(IntPtr handle, IntPtr xDesc, IntPtr x,
        IntPtr wDesc, IntPtr w, IntPtr convDesc, IntPtr yDesc, IntPtr y, int requestedAlgoCount, out int returnedAlgoCount,
        IntPtr perfResults, IntPtr workSpace, ulong workSpaceSizeInBytes);

    [DllImport(CudnnLibrary, EntryPoint = "cudnnFindConvolutionBackwardDataAlgorithmEx")]
    internal static extern CudnnStatus cudnnFindConvolutionBackwardDataAlgorithmEx(IntPtr handle, IntPtr wDesc, IntPtr w,
        IntPtr dyDesc, IntPtr dy, IntPtr convDesc, IntPtr dxDesc, IntPtr dx, int requestedAlgoCount,
        out int returnedAlgoCount, IntPtr perfResults, IntPtr workSpace, ulong workSpaceSizeInBytes);

    [DllImport(CudnnLibrary, EntryPoint = "cudnnFindConvolutionBackwardFilterAlgorithmEx")]
    internal static extern CudnnStatus cudnnFindConvolutionBackwardFilterAlgorithmEx(IntPtr handle, IntPtr xDesc, IntPtr x,
        IntPtr dyDesc, IntPtr dy, IntPtr convDesc, IntPtr dwDesc, IntPtr dw, int requestedAlgoCount,
        out int returnedAlgoCount, IntPtr perfResults, IntPtr workSpace, ulong workSpaceSizeInBytes);
}

public sealed partial class CuDnnConvolution
{
    // A cached search that found no algorithm for the formats.
    private const int UnsupportedFormats = int.MinValue;

    /// <summary>
    /// The workspace an algorithm found for a channels-last pass may use. 256 MB: at 64 MB the BF16 tensor-core forward
    /// of a DCGAN layer (64 -> 128 channels, 32x32, batch 64, 4x4 stride 2) could not run in the search (it asks 75.5 MB)
    /// and cuDNN's generic engine won at 0.80 ms against the tensor-core kernel's 0.03-0.3 ms. The search scratch is
    /// freed after the search; the shared workspace grows only to what the chosen algorithm asks, and among algorithms
    /// within 3% of the fastest the one asking least is taken.
    /// </summary>
    internal const ulong ChannelsLastMaxWorkspace = 256UL << 20;

    /// <summary>
    /// One channels-last pass ready to launch: its descriptors, the measured algorithm and the workspace it asks. Kept
    /// per shape because building them is the pass's host cost: four descriptors set and a workspace query per call
    /// (cuDNN 9 answers the legacy query through its heuristics) left the GPU idle 0.2-1.8 ms around every eager
    /// convolution of a BF16 DCGAN generator, several times the convolution itself.
    /// </summary>
    private sealed class ChannelsLastPlan : IDisposable
    {
        internal readonly CuDnnTensorDescriptor Input = new();
        internal readonly CuDnnTensorDescriptor Output = new();
        internal readonly CuDnnFilterDescriptor Filter = new();
        internal readonly CuDnnConvolutionDescriptor Convolution = new();
        internal int Algorithm = UnsupportedFormats;
        internal ulong Workspace;

        public void Dispose()
        {
            Convolution.Dispose();
            Filter.Dispose();
            Output.Dispose();
            Input.Dispose();
        }
    }

    // The plan per (pass, geometry, formats, math, determinism); an unsupported shape keeps a plan whose algorithm says so.
    private readonly ConcurrentDictionary<(int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int),
        ChannelsLastPlan> _channelsLastPlans = new();

    private void DisposeChannelsLastPlans()
    {
        foreach (var plan in _channelsLastPlans.Values) plan.Dispose();
        _channelsLastPlans.Clear();
    }

    /// <summary>Algorithm searches run for channels-last passes (tests).</summary>
    internal static long ChannelsLastSearches;

    /// <summary>
    /// One convolution pass with a format per tensor. <paramref name="pass"/> 0 is the forward (a = x, b = w, c = y),
    /// 1 the backward data (a = w, b = dy, c = dx), 2 the backward filter (a = x, b = dy, c = dw). The input-side tensor
    /// (x or dx) is channels-last when <paramref name="inputNhwc"/> is set, the output side (y or dy) when
    /// <paramref name="outputNhwc"/> is; the filter is [k, c, r, s], or [k, r, s, c] when <paramref name="filterNhwc"/>
    /// is set (cuDNN computes no FP32 filter gradient of channels-last activations into a [k, c, r, s] one). The algorithm is cuDNN's fastest for the
    /// pass measured on its first call (cudnnFind*Ex, as PyTorch's benchmark mode), within
    /// <see cref="ChannelsLastMaxWorkspace"/> and, when <paramref name="deterministic"/>, one without atomics; during a
    /// stream capture, which cannot measure, the first call takes cuDNN's implicit-GEMM default. False -- nothing
    /// launched -- when cuDNN supports no algorithm for the pass in these formats.
    /// </summary>
    internal bool ConvolutionChannelsLast(int pass, IntPtr a, IntPtr b, IntPtr c, int n, int channels, int h, int w,
        int k, int filterH, int filterW, int outputHeight, int outputWidth, int padH, int padW, int strideH, int strideW,
        int dilationH, int dilationW, bool inputNhwc, bool outputNhwc, CuDnnNative.CudnnMathType mathType,
        bool deterministic, bool capturing, bool filterNhwc = false,
        CuDnnNative.CudnnDataType dataType = CuDnnNative.CudnnDataType.Float)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuDnnConvolution));
        if (pass is < 0 or > 2) throw new ArgumentOutOfRangeException(nameof(pass));
        var key = (pass, n, channels, h, w, k, filterH, filterW, padH, padW, strideH, strideW,
            dilationH * 64 + dilationW,
            (inputNhwc ? 1 : 0) | (outputNhwc ? 2 : 0) | (deterministic ? 4 : 0) | (filterNhwc ? 8 : 0) | ((int)dataType << 8),
            (int)mathType, outputHeight * 65536 + outputWidth);
        ChannelsLastPlan? transient = null;
        try
        {
            if (!_channelsLastPlans.TryGetValue(key, out var plan))
            {
                plan = BuildChannelsLastPlan(n, channels, h, w, k, filterH, filterW, outputHeight, outputWidth, padH,
                    padW, strideH, strideW, dilationH, dilationW, inputNhwc, outputNhwc, filterNhwc, mathType, dataType);
                // Inside a capture no search can run: the pass takes a fixed algorithm, for this call only.
                int algorithm = capturing ? -1 : FindChannelsLast(pass, a, b, c, plan.Input.Handle, plan.Output.Handle,
                    plan.Filter.Handle, plan.Convolution.Handle, mathType, deterministic);
                if (algorithm < 0 && !capturing)
                {
                    // The search measured every algorithm and none runs these formats: remembered, so the caller's
                    // other formats are tried at once on later calls.
                    plan.Algorithm = UnsupportedFormats;
                    plan.Dispose();
                    _channelsLastPlans[key] = plan;
                    return false;
                }
                plan.Algorithm = algorithm >= 0 ? algorithm
                    : pass == 0 ? (int)CuDnnNative.CudnnConvolutionFwdAlgo.ImplicitPrecompGemm
                    : (int)CuDnnNative.CudnnConvolutionBwdDataAlgo.Algo1;   // Algo1 for both backward passes
                if (!TryQueryChannelsLastWorkspace(pass, plan)) { plan.Dispose(); return false; }
                if (capturing) transient = plan;
                else if (!_channelsLastPlans.TryAdd(key, plan)) { plan.Dispose(); plan = _channelsLastPlans[key]; }
            }
            if (plan.Algorithm == UnsupportedFormats) return false;
            return LaunchChannelsLastPlan(pass, plan, a, b, c, capturing);
        }
        finally
        {
            transient?.Dispose();
        }
    }

    private ChannelsLastPlan BuildChannelsLastPlan(int n, int channels, int h, int w, int k, int filterH,
        int filterW, int outputHeight, int outputWidth, int padH, int padW, int strideH, int strideW, int dilationH,
        int dilationW, bool inputNhwc, bool outputNhwc, bool filterNhwc, CuDnnNative.CudnnMathType mathType,
        CuDnnNative.CudnnDataType dataType)
    {
        // The operands' element type; accumulation is FP32 whatever it is (the convolution descriptor's compute type).
        var plan = new ChannelsLastPlan();
        try
        {
            plan.Input.Set4D(inputNhwc ? CuDnnNative.CudnnTensorFormat.NHWC : CuDnnNative.CudnnTensorFormat.NCHW,
                dataType, n, channels, h, w);
            plan.Output.Set4D(outputNhwc ? CuDnnNative.CudnnTensorFormat.NHWC : CuDnnNative.CudnnTensorFormat.NCHW,
                dataType, n, k, outputHeight, outputWidth);
            plan.Filter.Set4D(dataType, k, channels, filterH, filterW,
                filterNhwc ? CuDnnNative.CudnnTensorFormat.NHWC : CuDnnNative.CudnnTensorFormat.NCHW);
            SetConvolution(plan.Convolution, padH, padW, strideH, strideW, dilationH, dilationW, mathType);
            return plan;
        }
        catch
        {
            plan.Dispose();
            throw;
        }
    }

    private bool TryQueryChannelsLastWorkspace(int pass, ChannelsLastPlan plan)
    {
        IntPtr x = plan.Input.Handle, y = plan.Output.Handle, wd = plan.Filter.Handle, cd = plan.Convolution.Handle;
        ulong workspaceSize;
        CuDnnNative.CudnnStatus status = pass switch
        {
            0 => CuDnnNative.cudnnGetConvolutionForwardWorkspaceSize(_context.Handle, x, wd, cd, y,
                (CuDnnNative.CudnnConvolutionFwdAlgo)plan.Algorithm, out workspaceSize),
            1 => CuDnnNative.cudnnGetConvolutionBackwardDataWorkspaceSize(_context.Handle, wd, y, cd, x,
                (CuDnnNative.CudnnConvolutionBwdDataAlgo)plan.Algorithm, out workspaceSize),
            _ => CuDnnNative.cudnnGetConvolutionBackwardFilterWorkspaceSize(_context.Handle, x, y, cd, wd,
                (CuDnnNative.CudnnConvolutionBwdFilterAlgo)plan.Algorithm, out workspaceSize),
        };
        plan.Workspace = workspaceSize;
        return status == CuDnnNative.CudnnStatus.Success;
    }

    private bool LaunchChannelsLastPlan(int pass, ChannelsLastPlan plan, IntPtr a, IntPtr b, IntPtr c, bool capturing)
    {
        IntPtr x = plan.Input.Handle, y = plan.Output.Handle, wd = plan.Filter.Handle, cd = plan.Convolution.Handle;
        int algorithm = plan.Algorithm;
        ulong workspaceSize = plan.Workspace;
        CuDnnNative.CudnnStatus status;

        lock (_context.StreamBindingLock)
        lock (_workspaceLock)
        {
            if (workspaceSize > 0)
            {
                if (_workspaceSize < workspaceSize && capturing) return false;   // a capture cannot allocate
                EnsureWorkspaceUnlocked(workspaceSize);
            }
            IntPtr workspace = workspaceSize > 0 ? _workspace : IntPtr.Zero;
            float alpha = 1f, beta = 0f;
            status = pass switch
            {
                0 => CuDnnNative.cudnnConvolutionForward(_context.Handle, ref alpha, x, a, wd, b, cd,
                    (CuDnnNative.CudnnConvolutionFwdAlgo)algorithm, workspace, workspaceSize, ref beta, y, c),
                1 => CuDnnNative.cudnnConvolutionBackwardData(_context.Handle, ref alpha, wd, a, y, b, cd,
                    (CuDnnNative.CudnnConvolutionBwdDataAlgo)algorithm, workspace, workspaceSize, ref beta, x, c),
                _ => CuDnnNative.cudnnConvolutionBackwardFilter(_context.Handle, ref alpha, x, a, y, b, cd,
                    (CuDnnNative.CudnnConvolutionBwdFilterAlgo)algorithm, workspace, workspaceSize, ref beta, wd, c),
            };
            CuDnnContext.CheckStatus(status, "ConvolutionChannelsLast pass " + pass);
            DirectGpu.GpuLaunchProbe.OnLaunch();
        }
        return true;
    }

    // cudnnFind*Ex over the real buffers (it writes the pass's output, which the launch after it overwrites) with a
    // workspace of its own, freed after: the fastest supported algorithm within the bound and the determinism asked for,
    // preferring among those within 3% of it the one needing the least workspace. -1 when none qualifies.
    private int FindChannelsLast(int pass, IntPtr a, IntPtr b, IntPtr c, IntPtr x, IntPtr y, IntPtr wd, IntPtr cd,
        CuDnnNative.CudnnMathType mathType, bool deterministic)
    {
        const int requested = 16;
        int size = Marshal.SizeOf<CuDnnNative.CudnnConvolutionAlgoPerf>();
        IntPtr results = Marshal.AllocHGlobal(size * requested);
        IntPtr scratch = IntPtr.Zero;
        try
        {
            ulong scratchSize = ChannelsLastMaxWorkspace;
            if (CuBlasNative.cuMemAlloc(out scratch, scratchSize) != CudaResult.Success)
            {
                scratch = IntPtr.Zero;
                scratchSize = 0;
            }
            CuDnnNative.CudnnStatus status;
            int returned;
            lock (_context.StreamBindingLock)
            {
                status = pass switch
                {
                    0 => CuDnnNative.cudnnFindConvolutionForwardAlgorithmEx(_context.Handle, x, a, wd, b, cd, y, c,
                        requested, out returned, results, scratch, scratchSize),
                    1 => CuDnnNative.cudnnFindConvolutionBackwardDataAlgorithmEx(_context.Handle, wd, a, y, b, cd, x, c,
                        requested, out returned, results, scratch, scratchSize),
                    _ => CuDnnNative.cudnnFindConvolutionBackwardFilterAlgorithmEx(_context.Handle, x, a, y, b, cd, wd, c,
                        requested, out returned, results, scratch, scratchSize),
                };
            }
            System.Threading.Interlocked.Increment(ref ChannelsLastSearches);
            if (status != CuDnnNative.CudnnStatus.Success) return -1;
            bool tensorOpAllowed = mathType is CuDnnNative.CudnnMathType.TensorOp
                or CuDnnNative.CudnnMathType.TensorOpAllowConversion;
            int best = -1;
            float bestTime = float.MaxValue;
            ulong bestMemory = 0;
            var candidates = new System.Collections.Generic.List<CuDnnNative.CudnnConvolutionAlgoPerf>();
            for (int i = 0; i < Math.Min(returned, requested); i++)
            {
                var perf = Marshal.PtrToStructure<CuDnnNative.CudnnConvolutionAlgoPerf>(results + i * size);
                if (perf.Status != CuDnnNative.CudnnStatus.Success || perf.Time < 0) continue;
                if ((ulong)perf.Memory > ChannelsLastMaxWorkspace) continue;
                if (deterministic && perf.Determinism == 0) continue;
                bool tensorOp = perf.MathType is CuDnnNative.CudnnMathType.TensorOp
                    or CuDnnNative.CudnnMathType.TensorOpAllowConversion;
                if (tensorOp && !tensorOpAllowed) continue;
                candidates.Add(perf);
                if (perf.Time < bestTime) bestTime = perf.Time;
            }
            foreach (var perf in candidates)
            {
                if (perf.Time > bestTime * 1.03f) continue;
                if (best < 0 || (ulong)perf.Memory < bestMemory) { best = perf.Algo; bestMemory = (ulong)perf.Memory; }
            }
            return best;
        }
        catch (EntryPointNotFoundException)
        {
            return -1;
        }
        finally
        {
            Marshal.FreeHGlobal(results);
            if (scratch != IntPtr.Zero) CuBlasNative.cuMemFree(scratch);
        }
    }
}
