// Copyright (c) AiDotNet. All rights reserved.
// #1125: the PyTorch-parity kernels in FP32 (TorchParityF32Kernels), translated to MSL and compiled with strict math.
// The Dawn parity suites run the same dialect source (as WGSL) against the CPU references.
namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : ITorchParityElementwiseBackend
{
    private const string TorchParityLibName = "TorchParityF32";
    private readonly object _torchParityLibraryLock = new object();
    private IntPtr _torchParityLibrary;

    private void LaunchTorchParity(string name, int threads, params object[] args)
    {
        if (threads <= 0) return;
        ThrowIfDisposed();
        if (_torchParityLibrary == IntPtr.Zero)
        {
            lock (_torchParityLibraryLock)
            {
                if (_torchParityLibrary == IntPtr.Zero)
                    _torchParityLibrary = _shaderLibrary.CompileLibrary(
                        TorchParityLibName, MetalBatchExecutionKernels.TorchParitySource, requireStrictFloatingPoint: true);
            }
        }
        var pipe = GetPipeline(TorchParityLibName, _torchParityLibrary, name);
        using var enc = _commandQueue.CreateScopedComputeEncoder();
        enc.SetPipelineState(pipe.Handle);
        BindBatchArguments(in enc, name, args);
        var (groups, perGroup) = pipe.Calculate1DDispatch(threads);
        enc.DispatchThreadgroups(groups, perGroup);
    }


    void ITorchParityElementwiseBackend.TorchParityUnary(TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output, int n, float p0, float p1)
        => PortableTorchParity.Unary(LaunchTorchParity, op, input, output, n, p0, p1);
    void ITorchParityElementwiseBackend.TorchParityBinary(TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
        => PortableTorchParity.Binary(LaunchTorchParity, op, a, b, output, n);
    void ITorchParityElementwiseBackend.TorchParityPolynomial(int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree, IGpuBuffer output, int n)
        => PortableTorchParity.Polynomial(LaunchTorchParity, kind, shifted, x, degree, output, n);
    void ITorchParityElementwiseBackend.TorchParityRange(IGpuBuffer output, int n, double start, double span, double denominator)
        => PortableTorchParity.Range(LaunchTorchParity, output, n, start, span, denominator);
    void ITorchParityElementwiseBackend.TorchParityLogspace(IGpuBuffer output, int n, double start, double span, double denominator, double logBase, bool single)
        => PortableTorchParity.Logspace(LaunchTorchParity, output, n, start, span, denominator, logBase, single);
    void ITorchParityElementwiseBackend.TorchParityWindow(IGpuBuffer output, int n, int kind, int m, double alpha, double beta)
        => PortableTorchParity.Window(LaunchTorchParity, output, n, kind, m, alpha, beta);
    void ITorchParityElementwiseBackend.TorchParityTriIndices(IGpuBuffer output, int row, int col, int offset, bool lower, int total)
        => PortableTorchParity.TriIndices(LaunchTorchParity, output, row, col, offset, lower, total);
    void ITorchParityElementwiseBackend.TorchParityDiagflat(IGpuBuffer input, IGpuBuffer output, int n, int size, int offset)
        => PortableTorchParity.Diagflat(LaunchTorchParity, input, output, n, size, offset);
    void ITorchParityElementwiseBackend.TorchParityDiagonalScatter(IGpuBuffer src, IGpuBuffer output, int count, int[] otherExtents, int[] otherStrides, int stride1, int stride2, int start1, int start2, int diagonal)
        => PortableTorchParity.DiagonalScatter(this, LaunchTorchParity, src, output, count, otherExtents, otherStrides, stride1, stride2, start1, start2, diagonal);
    void ITorchParityElementwiseBackend.TorchParityQuantileRows(IGpuBuffer sorted, IGpuBuffer output, int rows, int length, double q, int mode, bool ignoreNan)
        => PortableTorchParity.QuantileRows(LaunchTorchParity, sorted, output, rows, length, q, mode, ignoreNan);
    void ITorchParityElementwiseBackend.TorchParityFlagReset(IGpuBuffer flag)
        => PortableTorchParity.FlagReset(LaunchTorchParity, flag);
    void ITorchParityElementwiseBackend.TorchParitySample(IGpuBuffer output, int n, int kind, double a, double b, ulong seed)
        => PortableTorchParity.Sample(LaunchTorchParity, output, n, kind, a, b, seed);
    void ITorchParityElementwiseBackend.TorchParityBernoulli(IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
        => PortableTorchParity.Bernoulli(this, LaunchTorchParity, probabilities, output, n, seed, flag);
    void ITorchParityElementwiseBackend.TorchParityPoisson(IGpuBuffer rates, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
        => PortableTorchParity.Poisson(this, LaunchTorchParity, rates, output, n, seed, flag);
    void ITorchParityElementwiseBackend.TorchParityBinomial(IGpuBuffer counts, IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
        => PortableTorchParity.Binomial(this, LaunchTorchParity, counts, probabilities, output, n, seed, flag);
    void ITorchParityElementwiseBackend.TorchParityMultinomial(IGpuBuffer weights, IGpuBuffer scratch, IGpuBuffer output, int rows, int categories, int numSamples, bool replacement, ulong seed, IGpuBuffer flag)
        => PortableTorchParity.Multinomial(this, LaunchTorchParity, weights, scratch, output, rows, categories, numSamples, replacement, seed, flag);
    void ITorchParityElementwiseBackend.TorchParityDropoutAffine(IGpuBuffer scale, IGpuBuffer offset, int n, int block, double p, double a, double sat, double b, bool alpha, ulong seed)
        => PortableTorchParity.DropoutAffine(LaunchTorchParity, scale, offset, n, block, p, a, sat, b, alpha, seed);
    void ITorchParityElementwiseBackend.TorchParityRrelu(IGpuBuffer input, IGpuBuffer output, int n, double lower, double upper, bool training, ulong seed)
        => PortableTorchParity.Rrelu(LaunchTorchParity, input, output, n, lower, upper, training, seed);
    void ITorchParityElementwiseBackend.TorchParityRandomKeys(IGpuBuffer output, int n, ulong seed)
        => PortableTorchParity.RandomKeys(LaunchTorchParity, output, n, seed);
    void ITorchParityElementwiseBackend.TorchParityIntBinary(int op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, IGpuBuffer flag)
        => PortableTorchParity.IntBinary(this, LaunchTorchParity, op, a, b, output, n, flag);
    void ITorchParityElementwiseBackend.TorchParityIntNot(IGpuBuffer input, IGpuBuffer output, int n)
        => PortableTorchParity.IntNot(LaunchTorchParity, input, output, n);
    void ITorchParityElementwiseBackend.TorchParityInt64Binary(int op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int typeCode, IGpuBuffer flag)
        => PortableTorchParity.Int64Binary(this, LaunchTorchParity, op, a, b, output, n, typeCode, flag);
    void ITorchParityElementwiseBackend.TorchParityInt64Not(IGpuBuffer input, IGpuBuffer output, int n, int typeCode)
        => PortableTorchParity.Int64Not(LaunchTorchParity, input, output, n, typeCode);
    void ITorchParityElementwiseBackend.TorchParityInclusiveScan(IGpuBuffer input, IGpuBuffer output, int n)
        => PortableTorchParity.InclusiveScan(this, LaunchTorchParity, input, output, n);
    void ITorchParityElementwiseBackend.TorchParityNonzeroStaticScatter(IGpuBuffer input, IGpuBuffer positions, IGpuBuffer output, int n, int size, int[] shape)
        => PortableTorchParity.NonzeroStaticScatter(this, LaunchTorchParity, input, positions, output, n, size, shape);
    void ITorchParityElementwiseBackend.TorchParityUnravelAxis(IGpuBuffer indices, IGpuBuffer output, int n, long total, long stride, long extent, bool validate, IGpuBuffer flag)
        => PortableTorchParity.UnravelAxis(this, LaunchTorchParity, indices, output, n, total, stride, extent, validate, flag);
    void ITorchParityElementwiseBackend.TorchParitySliceBitonicStep(IGpuBuffer slices, IGpuBuffer order, int rowLen, int k, int j, int length, int sliceCount)
        => PortableTorchParity.SliceBitonicStep(LaunchTorchParity, slices, order, rowLen, k, j, length, sliceCount);
    void ITorchParityElementwiseBackend.TorchParitySliceFirstFlags(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, int n, int length)
        => PortableTorchParity.SliceFirstFlags(LaunchTorchParity, slices, order, flags, n, length);
    void ITorchParityElementwiseBackend.TorchParitySliceGatherUnique(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, IGpuBuffer positions, IGpuBuffer output, int n, int length, int outputRows)
        => PortableTorchParity.SliceGatherUnique(LaunchTorchParity, slices, order, flags, positions, output, n, length, outputRows);
    void ITorchParityElementwiseBackend.TorchParityGridSample3D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output, int batches, int channels, int depth, int height, int width, int points, int mode, int padding, bool alignCorners)
        => PortableTorchParity.GridSample3D(LaunchTorchParity, input, grid, output, batches, channels, depth, height, width, points, mode, padding, alignCorners);
    void ITorchParityElementwiseBackend.TorchParityGridSample3DBackward(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer outputGrad, IGpuBuffer inputGrad, IGpuBuffer gridGrad, int batches, int channels, int depth, int height, int width, int points, int mode, int padding, bool alignCorners)
        => PortableTorchParity.GridSample3DBackward(LaunchTorchParity, input, grid, outputGrad, inputGrad, gridGrad, batches, channels, depth, height, width, points, mode, padding, alignCorners);
}