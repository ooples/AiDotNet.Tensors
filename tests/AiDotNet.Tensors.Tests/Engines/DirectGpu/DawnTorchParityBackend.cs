#if NET7_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The OpenCL backend with its PyTorch-parity kernels replaced, member by member, by the FP32 kernels Metal and WebGPU
/// run (<see cref="TorchParityF32Kernels"/>), executed as their generated WGSL on Dawn. The parity device suites run on
/// an engine over it, so the WebGPU / Metal kernels meet the same CPU references as every other backend.
/// </summary>
internal sealed class DawnTorchParityBackend : DelegatingGpuBackend, ITorchParityElementwiseBackend
{
    private readonly PortableGenericFft.Launcher _launch;

    public DawnTorchParityBackend(OpenClBackend host, DawnWgslLauncher dawn) : base(host)
    {
        _launch = dawn.For(host);
    }

    public void TorchParityUnary(TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output, int n, float p0, float p1)
        => PortableTorchParity.Unary(_launch, op, input, output, n, p0, p1);
    public void TorchParityBinary(TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n)
        => PortableTorchParity.Binary(_launch, op, a, b, output, n);
    public void TorchParityPolynomial(int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree, IGpuBuffer output, int n)
        => PortableTorchParity.Polynomial(_launch, kind, shifted, x, degree, output, n);
    public void TorchParityRange(IGpuBuffer output, int n, double start, double span, double denominator)
        => PortableTorchParity.Range(_launch, output, n, start, span, denominator);
    public void TorchParityLogspace(IGpuBuffer output, int n, double start, double span, double denominator, double logBase, bool single)
        => PortableTorchParity.Logspace(_launch, output, n, start, span, denominator, logBase, single);
    public void TorchParityWindow(IGpuBuffer output, int n, int kind, int m, double alpha, double beta)
        => PortableTorchParity.Window(_launch, output, n, kind, m, alpha, beta);
    public void TorchParityTriIndices(IGpuBuffer output, int row, int col, int offset, bool lower, int total)
        => PortableTorchParity.TriIndices(_launch, output, row, col, offset, lower, total);
    public void TorchParityDiagflat(IGpuBuffer input, IGpuBuffer output, int n, int size, int offset)
        => PortableTorchParity.Diagflat(_launch, input, output, n, size, offset);
    public void TorchParityDiagonalScatter(IGpuBuffer src, IGpuBuffer output, int count, int[] otherExtents, int[] otherStrides, int stride1, int stride2, int start1, int start2, int diagonal)
        => PortableTorchParity.DiagonalScatter(Inner, _launch, src, output, count, otherExtents, otherStrides, stride1, stride2, start1, start2, diagonal);
    public void TorchParityQuantileRows(IGpuBuffer sorted, IGpuBuffer output, int rows, int length, double q, int mode, bool ignoreNan)
        => PortableTorchParity.QuantileRows(_launch, sorted, output, rows, length, q, mode, ignoreNan);
    public void TorchParityFlagReset(IGpuBuffer flag)
        => PortableTorchParity.FlagReset(_launch, flag);
    public void TorchParitySample(IGpuBuffer output, int n, int kind, double a, double b, ulong seed)
        => PortableTorchParity.Sample(_launch, output, n, kind, a, b, seed);
    public void TorchParityBernoulli(IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
        => PortableTorchParity.Bernoulli(Inner, _launch, probabilities, output, n, seed, flag);
    public void TorchParityPoisson(IGpuBuffer rates, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
        => PortableTorchParity.Poisson(Inner, _launch, rates, output, n, seed, flag);
    public void TorchParityBinomial(IGpuBuffer counts, IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag)
        => PortableTorchParity.Binomial(Inner, _launch, counts, probabilities, output, n, seed, flag);
    public void TorchParityMultinomial(IGpuBuffer weights, IGpuBuffer scratch, IGpuBuffer output, int rows, int categories, int numSamples, bool replacement, ulong seed, IGpuBuffer flag)
        => PortableTorchParity.Multinomial(Inner, _launch, weights, scratch, output, rows, categories, numSamples, replacement, seed, flag);
    public void TorchParityDropoutAffine(IGpuBuffer scale, IGpuBuffer offset, int n, int block, double p, double a, double sat, double b, bool alpha, ulong seed)
        => PortableTorchParity.DropoutAffine(_launch, scale, offset, n, block, p, a, sat, b, alpha, seed);
    public void TorchParityRrelu(IGpuBuffer input, IGpuBuffer output, int n, double lower, double upper, bool training, ulong seed)
        => PortableTorchParity.Rrelu(_launch, input, output, n, lower, upper, training, seed);
    public void TorchParityRandomKeys(IGpuBuffer output, int n, ulong seed)
        => PortableTorchParity.RandomKeys(_launch, output, n, seed);
    public void TorchParityIntBinary(int op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, IGpuBuffer flag)
        => PortableTorchParity.IntBinary(Inner, _launch, op, a, b, output, n, flag);
    public void TorchParityIntNot(IGpuBuffer input, IGpuBuffer output, int n)
        => PortableTorchParity.IntNot(_launch, input, output, n);
    public void TorchParityInt64Binary(int op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, int typeCode, IGpuBuffer flag)
        => PortableTorchParity.Int64Binary(Inner, _launch, op, a, b, output, n, typeCode, flag);
    public void TorchParityInt64Not(IGpuBuffer input, IGpuBuffer output, int n, int typeCode)
        => PortableTorchParity.Int64Not(_launch, input, output, n, typeCode);
    public void TorchParityInclusiveScan(IGpuBuffer input, IGpuBuffer output, int n)
        => PortableTorchParity.InclusiveScan(Inner, _launch, input, output, n);
    public void TorchParityNonzeroStaticScatter(IGpuBuffer input, IGpuBuffer positions, IGpuBuffer output, int n, int size, int[] shape)
        => PortableTorchParity.NonzeroStaticScatter(Inner, _launch, input, positions, output, n, size, shape);
    public void TorchParityUnravelAxis(IGpuBuffer indices, IGpuBuffer output, int n, long total, long stride, long extent, bool validate, IGpuBuffer flag)
        => PortableTorchParity.UnravelAxis(Inner, _launch, indices, output, n, total, stride, extent, validate, flag);
    public void TorchParitySliceBitonicStep(IGpuBuffer slices, IGpuBuffer order, int rowLen, int k, int j, int length, int sliceCount)
        => PortableTorchParity.SliceBitonicStep(_launch, slices, order, rowLen, k, j, length, sliceCount);
    public void TorchParitySliceFirstFlags(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, int n, int length)
        => PortableTorchParity.SliceFirstFlags(_launch, slices, order, flags, n, length);
    public void TorchParitySliceGatherUnique(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, IGpuBuffer positions, IGpuBuffer output, int n, int length, int outputRows)
        => PortableTorchParity.SliceGatherUnique(_launch, slices, order, flags, positions, output, n, length, outputRows);
    public void TorchParityGridSample3D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output, int batches, int channels, int depth, int height, int width, int points, int mode, int padding, bool alignCorners)
        => PortableTorchParity.GridSample3D(_launch, input, grid, output, batches, channels, depth, height, width, points, mode, padding, alignCorners);
    public void TorchParityGridSample3DBackward(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer outputGrad, IGpuBuffer inputGrad, IGpuBuffer gridGrad, int batches, int channels, int depth, int height, int width, int points, int mode, int padding, bool alignCorners)
        => PortableTorchParity.GridSample3DBackward(_launch, input, grid, outputGrad, inputGrad, gridGrad, batches, channels, depth, height, width, points, mode, padding, alignCorners);
}

/// <summary>An engine over <see cref="DawnTorchParityBackend"/>; unavailable without OpenCL or Dawn.</summary>
public sealed class DawnTorchParityFixture : IDirectGpuTensorEngineFixture, IDisposable
{
    private readonly OpenClBackend? _host;
    private readonly DawnWgslLauncher? _dawn;

    public DirectGpuTensorEngine? Engine { get; }
    public bool IsAvailable => Engine?.IsGpuAvailable == true;

    public DawnTorchParityFixture()
    {
        var dawn = DawnWgslLauncher.TryCreate(TorchParityModules.Value);
        if (dawn is null) return;
        OpenClBackend? host = null;
        try
        {
            host = new OpenClBackend();
            if (!host.IsAvailable)
            {
                // An unusable fixture must not keep the node process alive until xUnit disposes it.
                host.Dispose();
                dawn.Dispose();
                return;
            }
            Engine = new DirectGpuTensorEngine(new DirectGpuEngine(new DawnTorchParityBackend(host, dawn)));
        }
        catch
        {
            // A constructor that throws is never disposed by xUnit: release the node process and the host here.
            host?.Dispose();
            dawn.Dispose();
            throw;
        }
        _host = host;
        _dawn = dawn;
    }

    internal static readonly Lazy<System.Collections.Generic.IReadOnlyDictionary<string, string>> TorchParityModules = new(() =>
        AiDotNet.Tensors.Engines.DirectGpu.WebGpu.CudaDialectToWgsl.Translate(TorchParityF32Kernels.GetSource()));

    public void Dispose()
    {
        Engine?.Dispose();
        _host?.Dispose();
        _dawn?.Dispose();
    }
}
#endif