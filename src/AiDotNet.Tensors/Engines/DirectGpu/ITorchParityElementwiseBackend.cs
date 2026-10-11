namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Unary PyTorch-parity elementwise ops with a device kernel (#1112).</summary>
internal enum TorchParityUnaryOp
{
    Celu, Hardtanh, LogSigmoid, Softsign, Logit, Sinc, Erf,
    BesselJ0, BesselJ1, BesselY0, BesselY1, ModifiedBesselI0, ModifiedBesselI1, Erfcx, Ndtr, Ndtri, Entr, Signbit, IsPosInf, IsNegInf,
    LogNdtr, ModifiedBesselK0, ModifiedBesselK1, ScaledModifiedBesselK0, ScaledModifiedBesselK1, SphericalBesselJ0, AiryAi, Mvlgamma,
    Angle, NonZero,
}

/// <summary>Binary (same-shape) PyTorch-parity elementwise ops with a device kernel (#1112).</summary>
internal enum TorchParityBinaryOp { Fmax, Fmin, Heaviside, GreaterEqual, LessEqual, Igamma, Igammac, FloorDivide }

/// <summary>
/// Backends with device kernels for the PyTorch-parity elementwise ops that otherwise compute on the host. Optional:
/// a backend without it keeps the recorded host fallback (<c>GpuLaunchProbe.OnFallback</c>).
/// </summary>
/// <remarks>
/// Implemented by the CUDA backend. Not yet implemented, each tracked by its own follow-up issue: HIP (#1124),
/// Metal (#1125), OpenCL (#1126), Vulkan (#1127) and WebGPU (#1128). Until then those backends compute these ops on
/// the host.
/// </remarks>
internal interface ITorchParityElementwiseBackend
{
    /// <summary><paramref name="output"/>[i] = op(<paramref name="input"/>[i]; p0, p1) for i in [0, n).</summary>
    void TorchParityUnary(TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output, int n, float p0, float p1);

    /// <summary><paramref name="output"/>[i] = op(<paramref name="a"/>[i], <paramref name="b"/>[i]) for i in [0, n).</summary>
    void TorchParityBinary(TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n);

    /// <summary>
    /// <paramref name="output"/>[i] = P_degree[i](<paramref name="x"/>[i]) for the orthogonal-polynomial family
    /// <paramref name="kind"/> (0 Chebyshev T, 1 U, 2 V, 3 W, 4 Hermite H, 5 Hermite He, 6 Laguerre L, 7 Legendre P),
    /// optionally shifted (x -> 2x - 1).
    /// </summary>
    void TorchParityPolynomial(int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree, IGpuBuffer output, int n);

    /// <summary><paramref name="output"/>[i] = start + i * span / denominator (arange / range pass the step and 1).</summary>
    void TorchParityRange(IGpuBuffer output, int n, double start, double span, double denominator);

    /// <summary><paramref name="output"/>[i] = logBase ^ (single ? start : start + i * span / denominator).</summary>
    void TorchParityLogspace(IGpuBuffer output, int n, double start, double span, double denominator, double logBase, bool single);

    /// <summary>
    /// A window over t = i / (m - 1): kind 0 Hann, 1 Hamming (alpha, beta), 2 Blackman, 3 Bartlett, 4 Kaiser (beta in
    /// <paramref name="alpha"/>); n = 1 gives [1].
    /// </summary>
    void TorchParityWindow(IGpuBuffer output, int n, int kind, int m, double alpha, double beta);

    /// <summary>The [2, total] lower (j - i &lt;= offset) or upper (j - i &gt;= offset) triangle indices of a row x col grid.</summary>
    void TorchParityTriIndices(IGpuBuffer output, int row, int col, int offset, bool lower, int total);

    /// <summary>A size x size matrix holding the n values of <paramref name="input"/> on diagonal <paramref name="offset"/>.</summary>
    void TorchParityDiagflat(IGpuBuffer input, IGpuBuffer output, int n, int size, int offset);

    /// <summary>
    /// Writes <paramref name="src"/> (torch.diagonal's layout) onto the diagonal of <paramref name="output"/>, which already
    /// holds the input. The other axes (at most 8) are given by extent and stride; the diagonal by its two strides and
    /// starting indices.
    /// </summary>
    void TorchParityDiagonalScatter(IGpuBuffer src, IGpuBuffer output, int count, int[] otherExtents, int[] otherStrides,
        int stride1, int stride2, int start1, int start2, int diagonal);

    /// <summary>
    /// The q-quantile of each of <paramref name="rows"/> sorted rows (ascending, NaN last); mode 0 linear, 1 lower,
    /// 2 higher, 3 nearest, 4 midpoint.
    /// </summary>
    void TorchParityQuantileRows(IGpuBuffer sorted, IGpuBuffer output, int rows, int length, double q, int mode, bool ignoreNan);

    // ---- random sampling (Philox; see the kernels). A flag is a 2-float (64-bit) buffer: reset it before a validating
    // launch; afterwards it holds (index << 3 | code) of the first invalid element, or all ones. ----

    /// <summary>Sets a 64-bit validation flag to "no error".</summary>
    void TorchParityFlagReset(IGpuBuffer flag);

    /// <summary>
    /// n draws of distribution <paramref name="kind"/>: 0 uniform(a, b), 1 cauchy(median a, sigma b), 2 exponential(rate a),
    /// 3 geometric(p a), 4 log-normal(mean a, std b), 5 standard normal, 6 randint(low a, span b).
    /// </summary>
    void TorchParitySample(IGpuBuffer output, int n, int kind, double a, double b, ulong seed);

    /// <summary>Bernoulli draws; flag code 1 for p outside [0, 1].</summary>
    void TorchParityBernoulli(IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag);

    /// <summary>Poisson draws; flag code 1 for a negative or NaN rate.</summary>
    void TorchParityPoisson(IGpuBuffer rates, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag);

    /// <summary>Binomial draws; flag code 1 for a count that is not a whole number, 2 for count &lt; 0 or p outside [0, 1].</summary>
    void TorchParityBinomial(IGpuBuffer counts, IGpuBuffer probabilities, IGpuBuffer output, int n, ulong seed, IGpuBuffer flag);

    /// <summary>
    /// numSamples category draws per row of weights (rows x categories; scratch is as large); flag code 1 for an invalid
    /// weight, 2 for a row with no positive weight left.
    /// </summary>
    void TorchParityMultinomial(IGpuBuffer weights, IGpuBuffer scratch, IGpuBuffer output, int rows, int categories, int numSamples,
        bool replacement, ulong seed, IGpuBuffer flag);

    /// <summary>
    /// Per-element scale and offset of a dropout keep mask drawn once per block: alpha dropout (a * keep,
    /// a * sat * (1 - keep) + b) or channel dropout (keep * a, 0).
    /// </summary>
    void TorchParityDropoutAffine(IGpuBuffer scale, IGpuBuffer offset, int n, int block, double p, double a, double sat, double b,
        bool alpha, ulong seed);

    /// <summary>rrelu with the eval slope or (training) a uniform slope in [lower, upper] per element.</summary>
    void TorchParityRrelu(IGpuBuffer input, IGpuBuffer output, int n, double lower, double upper, bool training, ulong seed);

    /// <summary>n uniform sort keys for a random permutation.</summary>
    void TorchParityRandomKeys(IGpuBuffer output, int n, ulong seed);

    /// <summary>
    /// An int32 binary op on raw int32 buffers, computed in 64 bits and wrapped: 0 and, 1 or, 2 xor, 3 left shift,
    /// 4 right shift, 5 gcd, 6 lcm, 7 floor divide (flag code 1 for a zero divisor).
    /// </summary>
    void TorchParityIntBinary(int op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n, IGpuBuffer flag);

    /// <summary>Bitwise not of a raw int32 buffer.</summary>
    void TorchParityIntNot(IGpuBuffer input, IGpuBuffer output, int n);

    /// <summary>Inclusive prefix sum of n floats (exact for integer counts below 2^24).</summary>
    void TorchParityInclusiveScan(IGpuBuffer input, IGpuBuffer output, int n);

    /// <summary>
    /// nonzero_static's scatter: each non-zero element whose inclusive-scan position is within <paramref name="size"/>
    /// writes its coordinates (shape at most 8 axes) to that row of the [size, rank] output.
    /// </summary>
    void TorchParityNonzeroStaticScatter(IGpuBuffer input, IGpuBuffer positions, IGpuBuffer output, int n, int size, int[] shape);

    /// <summary>One unravel_index output axis; <paramref name="validate"/> flags (code 1) indices outside [-total, total).</summary>
    void TorchParityUnravelAxis(IGpuBuffer indices, IGpuBuffer output, int n, long total, long stride, long extent, bool validate, IGpuBuffer flag);

    /// <summary>
    /// One bitonic step over slice indices (padding = -1) ordered by the CPU's unique_dim slice comparison;
    /// <paramref name="sliceCount"/> real slices of <paramref name="length"/> elements.
    /// </summary>
    void TorchParitySliceBitonicStep(IGpuBuffer slices, IGpuBuffer order, int rowLen, int k, int j, int length, int sliceCount);

    /// <summary>1 where a sorted slice differs from the previous one (the first always), else 0.</summary>
    void TorchParitySliceFirstFlags(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, int n, int length);

    /// <summary>Gathers each first-of-a-run sorted slice into row (position - 1) of the output.</summary>
    void TorchParitySliceGatherUnique(IGpuBuffer slices, IGpuBuffer order, IGpuBuffer flags, IGpuBuffer positions, IGpuBuffer output, int n, int length,
        int outputRows);

    /// <summary>grid_sample 3-D (mode 0 bilinear / 1 nearest; padding 0 zeros / 1 border / 2 reflection).</summary>
    void TorchParityGridSample3D(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output, int batches, int channels, int depth, int height,
        int width, int points, int mode, int padding, bool alignCorners);

    /// <summary>
    /// grid_sample 3-D adjoints: accumulates into <paramref name="inputGrad"/> (zeroed by the caller) and writes
    /// <paramref name="gridGrad"/> ([N, D', H', W', 3]).
    /// </summary>
    void TorchParityGridSample3DBackward(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer outputGrad, IGpuBuffer inputGrad, IGpuBuffer gridGrad,
        int batches, int channels, int depth, int height, int width, int points, int mode, int padding, bool alignCorners);
}
