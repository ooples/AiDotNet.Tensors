using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Provides vectorized implementations of Instant-NGP operations.
/// These operations are shared between CPU and GPU engines.
/// </summary>
public static class InstantNGPOperations
{
    #region Multiresolution Hash Encoding

    // tiny-cuda-nn's coherent_prime_hash (include/tiny-cuda-nn/common_device.h) for three dimensions, the factors of
    // Müller et al. 2022, eq. 4. The x factor is 1, so neighbouring x vertices share a cache line.
    internal const uint HashPrimeY = 2654435761u;
    internal const uint HashPrimeZ = 805459861u;

    // tiny-cuda-nn grid_index's MAX_BASES[3]: the largest vertex count per axis whose cube still fits in uint32. A
    // level with more vertices than this is always hashed.
    internal const int MaxDenseResolution = 0x659;

    // The largest grid scale a level may have, so that every grid coordinate is an exact float and fits in an int.
    internal const float MaxGridScale = 16777216f;

    // Positions are clamped to [0, PositionClampMax] before scaling, on every backend, and NaN maps to 0. tiny-cuda-nn's
    // kernel_grid does not clamp at all (encodings/grid.h passes positions_in straight to pos_fract), so the clamp is the
    // closed unit cube: identical to tiny-cuda-nn on its whole domain [0, 1], and well defined outside it.
    internal const float PositionClampMax = 1f;

    // The relative distance within which N_min * b^l counts as the integer the paper's real arithmetic gives.
    private const double PaperIntegerTolerance = 1e-9;

    /// <summary>One level's grid: <c>pos = x * Scale + PositionOffset</c>, with <c>GridResolution</c> vertices per axis.</summary>
    internal readonly struct HashGridLevel
    {
        internal HashGridLevel(float scale, float positionOffset, int gridResolution)
        {
            Scale = scale;
            PositionOffset = positionOffset;
            GridResolution = gridResolution;
        }

        internal float Scale { get; }

        internal float PositionOffset { get; }

        internal int GridResolution { get; }
    }

    /// <summary>
    /// Performs the multiresolution hash encoding of Instant-NGP (Müller et al. 2022), matching tiny-cuda-nn's
    /// GridEncoding with a hash grid, three input dimensions and linear interpolation.
    /// </summary>
    /// <param name="positions">Positions <c>[N, 3]</c> in the unit cube; each coordinate is clamped to [0, 1], NaN to 0.</param>
    /// <param name="hashTables">One table <c>[T_l, featuresPerLevel]</c> per level, coarsest first.</param>
    /// <param name="baseResolution">N_min, the coarsest level's resolution (tiny-cuda-nn's <c>base_resolution</c>).</param>
    /// <param name="perLevelScale">b, the growth factor between levels (tiny-cuda-nn's <c>per_level_scale</c>), at least 1.</param>
    /// <param name="featuresPerLevel">F, the features stored per table entry.</param>
    /// <param name="alignment">How each level's grid is placed; tiny-cuda-nn's by default.</param>
    /// <returns>The encoding <c>[N, levels * featuresPerLevel]</c>, level-major in each row.</returns>
    /// <remarks>
    /// <para>
    /// For level l the position is scaled to the grid (see <see cref="HashGridAlignment"/>), floored to a cell, and the
    /// eight corner features are blended with trilinear weights from the fractional part. A corner is looked up densely
    /// (<c>x + y*R + z*R*R</c>, R the level's vertices per axis) when <c>R^3</c> is at most the table size, and through the
    /// coherent prime hash <c>x ^ y*2654435761 ^ z*805459861</c> otherwise, both in wrapping uint32 arithmetic and taken
    /// modulo the table size, as tiny-cuda-nn's <c>grid_index</c> does.
    /// </para>
    /// <para>
    /// The grid arithmetic is fp32, as in tiny-cuda-nn and on every GPU backend, so each backend selects the same cells.
    /// The interpolation is accumulated in double and keeps double precision for a double table.
    /// </para>
    /// </remarks>
    public static Tensor<T> MultiresolutionHashEncoding<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int baseResolution,
        double perLevelScale,
        int featuresPerLevel,
        HashGridAlignment alignment = HashGridAlignment.TinyCudaNn)
    {
        ValidateHashEncodingArguments(positions, hashTables, featuresPerLevel);
        var levels = ComputeHashGridLevels(hashTables.Length, baseResolution, perLevelScale, alignment);

        var numOps = MathHelper.GetNumericOperations<T>();
        int numPoints = positions._shape[0];
        int numLevels = hashTables.Length;
        int totalFeatures = checked(numLevels * featuresPerLevel);

        float[] coordinates = ReadPositions(positions, numOps);
        var tables = new double[numLevels][];
        var tableSizes = new int[numLevels];
        for (int level = 0; level < numLevels; level++)
        {
            tables[level] = ReadAsDouble(hashTables[level], numOps);
            tableSizes[level] = hashTables[level]._shape[0];
        }

        var features = new double[checked(numPoints * totalFeatures)];
        AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, numPoints, (long)numPoints * totalFeatures * 8, n =>
        {
            Span<uint> corners = stackalloc uint[8];
            Span<double> weights = stackalloc double[8];
            for (int level = 0; level < numLevels; level++)
            {
                ComputeHashGridCorners(coordinates, n, levels[level], tableSizes[level], corners, weights);
                var table = tables[level];
                int featureBase = n * totalFeatures + level * featuresPerLevel;
                for (int f = 0; f < featuresPerLevel; f++)
                {
                    double value = 0.0;
                    for (int c = 0; c < 8; c++)
                        value += weights[c] * table[(int)corners[c] * featuresPerLevel + f];
                    features[featureBase + f] = value;
                }
            }
        });

        return new Tensor<T>(FromDouble(features, numOps), new[] { numPoints, totalFeatures });
    }

    /// <summary>
    /// Computes the gradient of <see cref="MultiresolutionHashEncoding{T}"/> with respect to each hash table.
    /// </summary>
    /// <remarks>
    /// The encoding is linear in the table entries, so each entry receives the sum, over every point and corner that
    /// looked it up, of the output gradient times that corner's trilinear weight; colliding corners add up. Each level is
    /// summed over the points in order, so the result is deterministic. The positions receive no gradient.
    /// </remarks>
    public static Tensor<T>[] MultiresolutionHashEncodingBackward<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int baseResolution,
        double perLevelScale,
        int featuresPerLevel,
        Tensor<T> outputGradient,
        HashGridAlignment alignment = HashGridAlignment.TinyCudaNn)
    {
        ValidateHashEncodingArguments(positions, hashTables, featuresPerLevel);
        if (outputGradient is null) throw new ArgumentNullException(nameof(outputGradient));
        var levels = ComputeHashGridLevels(hashTables.Length, baseResolution, perLevelScale, alignment);

        var numOps = MathHelper.GetNumericOperations<T>();
        int numPoints = positions._shape[0];
        int numLevels = hashTables.Length;
        int totalFeatures = checked(numLevels * featuresPerLevel);
        if (outputGradient.Length != checked(numPoints * totalFeatures))
            throw new ArgumentException(
                "Output gradient must have N * levels * featuresPerLevel elements.", nameof(outputGradient));

        float[] coordinates = ReadPositions(positions, numOps);
        double[] gradient = ReadAsDouble(outputGradient, numOps);
        var tableSizes = new int[numLevels];
        for (int level = 0; level < numLevels; level++)
            tableSizes[level] = hashTables[level]._shape[0];

        // One level per task: no two tasks write the same table, so no locks, and each sum runs in point order.
        var levelGradients = new double[numLevels][];
        AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, numLevels, (long)numPoints * totalFeatures * 8, level =>
        {
            var accumulated = new double[checked(tableSizes[level] * featuresPerLevel)];
            Span<uint> corners = stackalloc uint[8];
            Span<double> weights = stackalloc double[8];
            for (int n = 0; n < numPoints; n++)
            {
                ComputeHashGridCorners(coordinates, n, levels[level], tableSizes[level], corners, weights);
                int gradientBase = n * totalFeatures + level * featuresPerLevel;
                for (int f = 0; f < featuresPerLevel; f++)
                {
                    double g = gradient[gradientBase + f];
                    if (g == 0.0) continue;
                    for (int c = 0; c < 8; c++)
                        accumulated[(int)corners[c] * featuresPerLevel + f] += g * weights[c];
                }
            }

            levelGradients[level] = accumulated;
        });

        var hashGradients = new Tensor<T>[numLevels];
        for (int level = 0; level < numLevels; level++)
            hashGradients[level] = new Tensor<T>(
                FromDouble(levelGradients[level], numOps), new[] { tableSizes[level], featuresPerLevel });
        return hashGradients;
    }

    /// <summary>Validates the arguments shared by the hash encoding and its backward, on every engine.</summary>
    internal static void ValidateHashEncodingArguments<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int featuresPerLevel)
    {
        if (positions is null) throw new ArgumentNullException(nameof(positions));
        if (hashTables is null) throw new ArgumentNullException(nameof(hashTables));
        if (positions._shape.Length != 2 || positions._shape[1] != 3)
            throw new ArgumentException("Positions must be [N, 3].", nameof(positions));
        if (featuresPerLevel <= 0)
            throw new ArgumentOutOfRangeException(nameof(featuresPerLevel), "Features per level must be positive.");
        for (int level = 0; level < hashTables.Length; level++)
        {
            var table = hashTables[level];
            if (table is null)
                throw new ArgumentException($"Hash table {level} is null.", nameof(hashTables));
            if (table._shape.Length == 0 || table._shape[0] <= 0 ||
                table.Length < (long)table._shape[0] * featuresPerLevel)
                throw new ArgumentException(
                    $"Hash table {level} must be [tableSize, featuresPerLevel] with a positive tableSize.",
                    nameof(hashTables));
        }
    }

    /// <summary>The grid of every level, coarsest first (see <see cref="HashGridAlignment"/>).</summary>
    internal static HashGridLevel[] ComputeHashGridLevels(
        int levelCount, int baseResolution, double perLevelScale, HashGridAlignment alignment)
    {
        if (baseResolution < 1)
            throw new ArgumentOutOfRangeException(nameof(baseResolution), "The base resolution must be at least 1.");
        if (double.IsNaN(perLevelScale) || double.IsInfinity(perLevelScale) || perLevelScale < 1.0)
            throw new ArgumentOutOfRangeException(nameof(perLevelScale), "The per-level scale must be finite and at least 1.");
        var levels = new HashGridLevel[levelCount];
        for (int level = 0; level < levelCount; level++)
            levels[level] = ComputeHashGridLevel(level, baseResolution, perLevelScale, alignment);
        return levels;
    }

    /// <summary>One level's grid (see <see cref="HashGridAlignment"/>).</summary>
    internal static HashGridLevel ComputeHashGridLevel(
        int level, int baseResolution, double perLevelScale, HashGridAlignment alignment)
    {
        float scale;
        float positionOffset;
        switch (alignment)
        {
            case HashGridAlignment.TinyCudaNn:
            {
                // grid_scale: exp2f(level * log2_per_level_scale) * base_resolution - 1.0f, all fp32, with the
                // multiply-subtract contracted to an fma as nvcc does by default. GridEncodingTemplated stores
                // per_level_scale as a float (m_per_level_scale) and passes std::log2 of that float, so b is rounded
                // to fp32 before the logarithm.
                float perLevelScaleF = (float)perLevelScale;
                float log2PerLevelScale = (float)(Math.Log(perLevelScaleF) / Math.Log(2.0));
                float exponent = (float)level * log2PerLevelScale;
                float levelScale = (float)Math.Pow(2.0, exponent);
                scale = FusedMultiplyAdd(levelScale, (float)baseResolution, -1f);
                positionOffset = 0.5f;
                break;
            }
            case HashGridAlignment.Paper:
            {
                // Eq. 2, N_l = floor(N_min * b^l), is real arithmetic: with b from eq. 3, N_min * b^l is an integer at
                // some levels (N_{L-1} = N_max), and double rounding can put it just below that integer. Within
                // rounding error of an integer it is that integer; otherwise it is floored.
                double cells = baseResolution * Math.Pow(perLevelScale, level);
                double nearest = Math.Round(cells);
                scale = (float)(Math.Abs(cells - nearest) <= PaperIntegerTolerance * nearest ? nearest : Math.Floor(cells));
                positionOffset = 0f;
                break;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(alignment), alignment, "Unknown hash-grid alignment.");
        }

        if (!(scale >= 0f) || scale > MaxGridScale)
            throw new ArgumentOutOfRangeException(
                nameof(perLevelScale),
                $"Level {level} has grid scale {scale}, outside [0, {MaxGridScale}]; lower the base resolution, " +
                "the per-level scale or the number of levels.");

        // grid_resolution: (uint32_t)ceilf(scale) + 1 vertices per axis.
        int gridResolution = (int)Math.Ceiling(scale) + 1;
        return new HashGridLevel(scale, positionOffset, gridResolution);
    }

    /// <summary>
    /// Whether a level is indexed densely: tiny-cuda-nn's <c>grid_index</c> hashes when <c>hashmap_size &lt; stride</c>,
    /// with <c>stride = R^3</c> for <c>R &lt;= MAX_BASES[3]</c> and <c>0xFFFFFFFF</c> beyond it.
    /// </summary>
    internal static bool UsesDenseHashGridIndex(int gridResolution, int tableSize)
    {
        unchecked
        {
            uint resolution = (uint)gridResolution;
            uint stride = gridResolution <= MaxDenseResolution ? resolution * resolution * resolution : uint.MaxValue;
            return (uint)tableSize >= stride;
        }
    }

    /// <summary>
    /// tiny-cuda-nn's <c>grid_index</c> for three dimensions: the dense index when the level's whole grid fits the table,
    /// the coherent prime hash otherwise, in wrapping uint32 arithmetic and modulo the table size.
    /// </summary>
    internal static uint HashGridIndex(uint x, uint y, uint z, int gridResolution, int tableSize)
    {
        unchecked
        {
            uint resolution = (uint)gridResolution;
            uint index = !UsesDenseHashGridIndex(gridResolution, tableSize)
                ? x ^ (y * HashPrimeY) ^ (z * HashPrimeZ)
                : x + y * resolution + z * resolution * resolution;
            return index % (uint)tableSize;
        }
    }

    /// <summary>
    /// tiny-cuda-nn's <c>pos_fract</c> for one coordinate: <c>pos = fma(scale, x, offset)</c>, the cell
    /// <c>floor(pos)</c> and the fraction <c>pos - floor(pos)</c>, all fp32.
    /// </summary>
    internal static void HashGridAxis(float coordinate, HashGridLevel level, out uint cell, out float fraction)
    {
        float x = !(coordinate > 0f) ? 0f : (coordinate >= PositionClampMax ? PositionClampMax : coordinate);
        float position = FusedMultiplyAdd(level.Scale, x, level.PositionOffset);
        float floor = (float)Math.Floor(position);
        cell = (uint)(int)floor;
        fraction = (float)(position - floor);
    }

    /// <summary>
    /// <c>fmaf(a, b, c)</c>: <c>a * b + c</c> rounded once to fp32, as tiny-cuda-nn, the NVRTC kernels and PTX
    /// <c>fma.rn.f32</c> compute it. Rounding <c>a * b + c</c> to double first and then to float is not enough: the
    /// double sum is inexact once <c>a * b</c> is small next to <c>c</c>, and can land on a float tie the exact value is not on.
    /// </summary>
    internal static float FusedMultiplyAdd(float a, float b, float c)
    {
#if NETCOREAPP3_0_OR_GREATER
        return MathF.FusedMultiplyAdd(a, b, c);
#else
        double product = (double)a * b;   // exact: two 24-bit significands
        double sum = product + c;
        // TwoSum (Knuth): sum + error == product + c exactly.
        double virtualC = sum - product;
        double error = (product - (sum - virtualC)) + (c - virtualC);
        double rounded = (float)sum;
        if (error != 0.0 && rounded != sum)
        {
            // Rounding sum to float is wrong only when sum is exactly halfway between two floats; the error then says
            // which side the exact value is on. The other float is 2*sum - rounded, exact in double for a midpoint.
            double other = 2.0 * sum - rounded;
            if ((double)(float)other == other)
                return (float)((error > 0.0) == (other > rounded) ? other : rounded);
        }

        return (float)rounded;
#endif
    }

    /// <summary>
    /// The eight corners of a point's cell on one level, as table entries, and their trilinear weights. Corner c takes
    /// the upper vertex on axis d when bit d of c is set, the order of tiny-cuda-nn's interpolation loop.
    /// </summary>
    internal static void ComputeHashGridCorners(
        float[] coordinates, int point, HashGridLevel level, int tableSize, Span<uint> corners, Span<double> weights)
    {
        HashGridAxis(coordinates[point * 3], level, out uint gx, out float fx);
        HashGridAxis(coordinates[point * 3 + 1], level, out uint gy, out float fy);
        HashGridAxis(coordinates[point * 3 + 2], level, out uint gz, out float fz);
        for (int c = 0; c < 8; c++)
        {
            bool upperX = (c & 1) != 0, upperY = (c & 2) != 0, upperZ = (c & 4) != 0;
            weights[c] = (upperX ? fx : 1.0 - fx) * (upperY ? fy : 1.0 - fy) * (upperZ ? fz : 1.0 - fz);
            corners[c] = HashGridIndex(
                upperX ? gx + 1u : gx, upperY ? gy + 1u : gy, upperZ ? gz + 1u : gz,
                level.GridResolution, tableSize);
        }
    }

    // The positions as fp32, the precision of the grid arithmetic. float and double are converted directly, other
    // element types through the vectorized float conversion, so no per-element INumericOperations call is made.
    private static float[] ReadPositions<T>(Tensor<T> positions, INumericOperations<T> numOps)
    {
        T[] data = positions.ToArray();
        if (data is float[] floats) return floats;
        if (data is double[] doubles)
        {
            var narrowed = new float[doubles.Length];
            for (int i = 0; i < doubles.Length; i++) narrowed[i] = (float)doubles[i];
            return narrowed;
        }

        return ToFloat(data, numOps);
    }

    // A tensor's values as double: a double tensor keeps its precision, other element types go through the
    // vectorized float conversion (exact for float, Half and BFloat16).
    private static double[] ReadAsDouble<T>(Tensor<T> tensor, INumericOperations<T> numOps)
    {
        T[] data = tensor.ToArray();
        if (data is double[] doubles) return doubles;
        float[] floats = data is float[] direct ? direct : ToFloat(data, numOps);
        var widened = new double[floats.Length];
        for (int i = 0; i < floats.Length; i++) widened[i] = floats[i];
        return widened;
    }

    private static float[] ToFloat<T>(T[] data, INumericOperations<T> numOps)
    {
        var result = new float[data.Length];
        numOps.ToFloatSpan(new ReadOnlySpan<T>(data), new Span<float>(result));
        return result;
    }

    private static T[] FromDouble<T>(double[] values, INumericOperations<T> numOps)
    {
        if (values is T[] same) return same;
        var floats = new float[values.Length];
        for (int i = 0; i < values.Length; i++) floats[i] = (float)values[i];
        if (floats is T[] direct) return direct;
        var result = new T[values.Length];
        numOps.FromFloatSpan(new ReadOnlySpan<float>(floats), new Span<T>(result));
        return result;
    }

    #endregion

    #region Occupancy Grid

    /// <summary>
    /// Updates occupancy grid for efficient ray sampling.
    /// </summary>
    public static Tensor<T> UpdateOccupancyGrid<T>(
        Tensor<T> occupancyGrid,
        Tensor<T> densities,
        Tensor<T> positions,
        int gridSize,
        T threshold,
        T decayFactor)
    {
        if (occupancyGrid == null) throw new ArgumentNullException(nameof(occupancyGrid));
        if (densities == null) throw new ArgumentNullException(nameof(densities));
        if (positions == null) throw new ArgumentNullException(nameof(positions));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numSamples = positions._shape[0];
        double thresholdVal = numOps.ToDouble(threshold);
        double decayVal = numOps.ToDouble(decayFactor);

        // Create new grid as copy
        var newGrid = new T[gridSize * gridSize * gridSize];
        Array.Copy(occupancyGrid.ToArray(), newGrid, newGrid.Length);

        // Apply decay
        for (int i = 0; i < newGrid.Length; i++)
        {
            newGrid[i] = numOps.FromDouble(numOps.ToDouble(newGrid[i]) * decayVal);
        }

        // Use striped locking to reduce contention while avoiding memory overhead of per-cell locks
        // Number of stripes is a power of 2 for efficient modulo operation via bitwise AND
        const int NumStripes = 256;
        var stripeLocks = new object[NumStripes];
        for (int i = 0; i < NumStripes; i++)
        {
            stripeLocks[i] = new object();
        }

        AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, numSamples, numSamples, i =>
        {
            double px = numOps.ToDouble(positions.GetFlat(i * 3));
            double py = numOps.ToDouble(positions.GetFlat(i * 3 + 1));
            double pz = numOps.ToDouble(positions.GetFlat(i * 3 + 2));
            double density = numOps.ToDouble(densities.GetFlat(i));

            // Clamp to [0, 1]
            px = Clamp01(px);
            py = Clamp01(py);
            pz = Clamp01(pz);

            // Map to grid cell
            int gx = Math.Min((int)(px * gridSize), gridSize - 1);
            int gy = Math.Min((int)(py * gridSize), gridSize - 1);
            int gz = Math.Min((int)(pz * gridSize), gridSize - 1);

            int gridIdx = (gx * gridSize + gy) * gridSize + gz;

            // Use striped lock based on grid index to reduce contention
            int stripeIdx = gridIdx & (NumStripes - 1);
            lock (stripeLocks[stripeIdx])
            {
                double current = numOps.ToDouble(newGrid[gridIdx]);
                double alpha = 1.0 - Math.Exp(-density);
                newGrid[gridIdx] = numOps.FromDouble(Math.Max(current, alpha));
            }
        });

        return new Tensor<T>(newGrid, [gridSize, gridSize, gridSize]);
    }

    /// <summary>
    /// Samples rays while skipping empty space using occupancy grid.
    /// </summary>
    public static (Tensor<T> positions, Tensor<T> directions, Tensor<bool> validMask, Tensor<T> tValues)
        SampleRaysWithOccupancy<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        Tensor<uint> occupancyBitfield,
        int gridSize,
        Vector<T> sceneBoundsMin,
        Vector<T> sceneBoundsMax,
        T nearBound,
        T farBound,
        int maxSamples)
    {
        if (rayOrigins == null) throw new ArgumentNullException(nameof(rayOrigins));
        if (rayDirections == null) throw new ArgumentNullException(nameof(rayDirections));
        if (occupancyBitfield == null) throw new ArgumentNullException(nameof(occupancyBitfield));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numRays = rayOrigins._shape[0];

        double near = numOps.ToDouble(nearBound);
        double far = numOps.ToDouble(farBound);

        double minX = numOps.ToDouble(sceneBoundsMin[0]);
        double minY = numOps.ToDouble(sceneBoundsMin[1]);
        double minZ = numOps.ToDouble(sceneBoundsMin[2]);
        double maxX = numOps.ToDouble(sceneBoundsMax[0]);
        double maxY = numOps.ToDouble(sceneBoundsMax[1]);
        double maxZ = numOps.ToDouble(sceneBoundsMax[2]);

        double invSizeX = 1.0 / Math.Max(1e-10, maxX - minX);
        double invSizeY = 1.0 / Math.Max(1e-10, maxY - minY);
        double invSizeZ = 1.0 / Math.Max(1e-10, maxZ - minZ);

        int totalSamples = numRays * maxSamples;
        var positions = new T[totalSamples * 3];
        var directions = new T[totalSamples * 3];
        var validMask = new bool[totalSamples];
        var tValues = new T[numRays * maxSamples];

        int totalCells = gridSize * gridSize * gridSize;

        AiDotNet.Tensors.Helpers.CpuParallelSettings.ParallelForOrSerial(0, numRays, numRays, r =>
        {
            double ox = numOps.ToDouble(rayOrigins.GetFlat(r * 3));
            double oy = numOps.ToDouble(rayOrigins.GetFlat(r * 3 + 1));
            double oz = numOps.ToDouble(rayOrigins.GetFlat(r * 3 + 2));
            double dx = numOps.ToDouble(rayDirections.GetFlat(r * 3));
            double dy = numOps.ToDouble(rayDirections.GetFlat(r * 3 + 1));
            double dz = numOps.ToDouble(rayDirections.GetFlat(r * 3 + 2));

            // Compute ray-scene intersection
            double tMin = near;
            double tMax = far;

            if (!ComputeRayBounds(ox, oy, oz, dx, dy, dz, minX, maxX, minY, maxY, minZ, maxZ, ref tMin, ref tMax))
            {
                // Ray doesn't intersect scene
                return;
            }

            double step = (tMax - tMin) / maxSamples;
            int baseSampleIdx = r * maxSamples;

            for (int s = 0; s < maxSamples; s++)
            {
                double t = tMin + step * (s + 0.5);
                double px = ox + t * dx;
                double py = oy + t * dy;
                double pz = oz + t * dz;

                // Normalize to [0, 1]
                double nx = (px - minX) * invSizeX;
                double ny = (py - minY) * invSizeY;
                double nz = (pz - minZ) * invSizeZ;

                nx = Clamp01(nx);
                ny = Clamp01(ny);
                nz = Clamp01(nz);

                // Map to grid cell
                int gx = Math.Min((int)(nx * gridSize), gridSize - 1);
                int gy = Math.Min((int)(ny * gridSize), gridSize - 1);
                int gz = Math.Min((int)(nz * gridSize), gridSize - 1);

                int gridIdx = (gx * gridSize + gy) * gridSize + gz;

                // Check occupancy bitfield
                bool isOccupied = gridIdx < totalCells && IsBitSet(occupancyBitfield, gridIdx);

                int sampleIdx = baseSampleIdx + s;
                int posIdx = sampleIdx * 3;

                positions[posIdx] = numOps.FromDouble(px);
                positions[posIdx + 1] = numOps.FromDouble(py);
                positions[posIdx + 2] = numOps.FromDouble(pz);
                directions[posIdx] = rayDirections.GetFlat(r * 3);
                directions[posIdx + 1] = rayDirections.GetFlat(r * 3 + 1);
                directions[posIdx + 2] = rayDirections.GetFlat(r * 3 + 2);
                validMask[sampleIdx] = isOccupied;
                tValues[sampleIdx] = numOps.FromDouble(t);
            }
        });

        return (
            new Tensor<T>(positions, [totalSamples, 3]),
            new Tensor<T>(directions, [totalSamples, 3]),
            new Tensor<bool>(validMask, [totalSamples]),
            new Tensor<T>(tValues, [numRays, maxSamples])
        );
    }

    #endregion

    #region Helper Methods

    private static bool IsBitSet(Tensor<uint> bitfield, int index)
    {
        int word = index >> 5;  // index / 32
        int bit = index & 31;   // index % 32
        if (word >= bitfield.Length) return false;
        return (bitfield.GetFlat(word) & (1u << bit)) != 0;
    }

    private static bool ComputeRayBounds(
        double ox, double oy, double oz,
        double dx, double dy, double dz,
        double minX, double maxX,
        double minY, double maxY,
        double minZ, double maxZ,
        ref double tMin, ref double tMax)
    {
        if (!IntersectAxis(ox, dx, minX, maxX, ref tMin, ref tMax)) return false;
        if (!IntersectAxis(oy, dy, minY, maxY, ref tMin, ref tMax)) return false;
        if (!IntersectAxis(oz, dz, minZ, maxZ, ref tMin, ref tMax)) return false;
        return tMax >= tMin;
    }

    private static bool IntersectAxis(
        double o, double d, double minBound, double maxBound,
        ref double tMin, ref double tMax)
    {
        const double eps = 1e-8;
        if (Math.Abs(d) < eps)
        {
            return o >= minBound && o <= maxBound;
        }

        double inv = 1.0 / d;
        double t1 = (minBound - o) * inv;
        double t2 = (maxBound - o) * inv;

        if (t1 > t2)
        {
            (t1, t2) = (t2, t1);
        }

        tMin = Math.Max(tMin, t1);
        tMax = Math.Min(tMax, t2);
        return tMax >= tMin;
    }

    private static double Clamp01(double value)
    {
        if (value <= 0.0) return 0.0;
        if (value >= 1.0 - 1e-6) return 1.0 - 1e-6;
        return value;
    }

    #endregion
}
