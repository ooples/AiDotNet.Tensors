using AiDotNet.Tensors.Groups;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Execution engine for mathematical operations.
/// Implementations can target CPU, GPU, or other accelerators.
/// </summary>
/// <remarks>
/// <para>
/// The IEngine interface provides a pluggable execution model for AiDotNet.
/// By swapping implementations, users can transparently accelerate computations
/// on different hardware without changing their code.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "compute backend".
///
/// - CpuEngine: Runs operations on your CPU using standard C# code
/// - GpuEngine: Runs operations on your GPU for massive speedups
/// - Future: TPU, distributed computing, etc.
///
/// Your code stays the same - just swap the engine to change where it runs!
/// </para>
/// </remarks>
public interface IEngine
{
    /// <summary>
    /// Gets the name of this engine.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets whether this engine supports GPU acceleration.
    /// </summary>
    bool SupportsGpu { get; }

    /// <summary>
    /// Gets the DirectGpu engine for high-performance fused GPU operations.
    /// Returns null if DirectGpu is not available.
    /// </summary>
    /// <remarks>
    /// <para>
    /// DirectGpu provides 10-100x faster operations than standard GPU paths by using
    /// custom optimized kernels with fused operations (e.g., GEMM+Bias+Activation in one pass).
    /// </para>
    /// <para>
    /// Works on ALL .NET versions including .NET Framework 4.6.2 via pure P/Invoke.
    /// </para>
    /// </remarks>
    DirectGpu.DirectGpuEngine? DirectGpu { get; }

    #region Vector Operations

    /// <summary>
    /// Adds two vectors element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector containing the element-wise sum.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    Vector<T> Add<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Subtracts vector b from vector a element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector containing the element-wise difference.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    Vector<T> Subtract<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Multiplies two vectors element-wise (Hadamard product).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector containing the element-wise product.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    Vector<T> Multiply<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Multiplies a vector by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to multiply.</param>
    /// <param name="scalar">The scalar value.</param>
    /// <returns>A new vector with all elements multiplied by the scalar.</returns>
    Vector<T> Multiply<T>(Vector<T> vector, T scalar);

    /// <summary>
    /// Divides vector a by vector b element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The numerator vector.</param>
    /// <param name="b">The denominator vector.</param>
    /// <returns>A new vector containing the element-wise quotient.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <exception cref="DivideByZeroException">Thrown when any element of b is zero.</exception>
    Vector<T> Divide<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Divides a vector by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to divide.</param>
    /// <param name="scalar">The scalar divisor.</param>
    /// <returns>A new vector with all elements divided by the scalar.</returns>
    /// <exception cref="DivideByZeroException">Thrown when scalar is zero.</exception>
    Vector<T> Divide<T>(Vector<T> vector, T scalar);

    /// <summary>
    /// Computes the square root of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the square roots.</returns>
    Vector<T> Sqrt<T>(Vector<T> vector);

    /// <summary>
    /// Raises each element of the vector to the specified power.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <param name="exponent">The exponent to raise elements to.</param>
    /// <returns>A new vector with elements raised to the power.</returns>
    Vector<T> Power<T>(Vector<T> vector, T exponent);

    /// <summary>
    /// Computes the element-wise maximum of two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector where each element is max(a[i], b[i]).</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for AdaMax optimizer.</para>
    /// </remarks>
    Vector<T> Max<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the element-wise minimum of two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector where each element is min(a[i], b[i]).</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for various optimizers.</para>
    /// </remarks>
    Vector<T> Min<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the absolute value of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the absolute values.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for AdaMax and other optimizers.</para>
    /// </remarks>
    Vector<T> Abs<T>(Vector<T> vector);

    /// <summary>
    /// Computes the exponential (e^x) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the exponentials.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for natural gradient optimizers.</para>
    /// </remarks>
    Vector<T> Exp<T>(Vector<T> vector);

    /// <summary>
    /// Computes 2^x (base-2 exponential) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the base-2 exponentials.</returns>
    /// <remarks>
    /// Used in information theory (entropy calculations), binary computations, and scientific applications
    /// that work with powers of 2 (common in signal processing and computer graphics).
    /// </remarks>
    Vector<T> Exp2<T>(Vector<T> vector);

    /// <summary>
    /// Computes 10^x (base-10 exponential) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the base-10 exponentials.</returns>
    /// <remarks>
    /// Used in scientific and engineering applications, decibel calculations (dB), and pH computations.
    /// Common in physics, chemistry, and signal processing where base-10 is the standard scale.
    /// </remarks>
    Vector<T> Exp10<T>(Vector<T> vector);

    /// <summary>
    /// Computes the natural logarithm of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the logarithms.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for natural gradient optimizers.</para>
    /// <para>
    /// <b>Note:</b> For elements <= 0, the behavior is:
    /// - Zero input produces NegativeInfinity
    /// - Negative input produces NaN
    /// - No exception is thrown (silent NaN propagation)
    /// </para>
    /// </remarks>
    Vector<T> Log<T>(Vector<T> vector);

    /// <summary>
    /// Computes the base-2 logarithm of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the base-2 logarithms.</returns>
    /// <remarks>
    /// Used in information theory (entropy, bits), binary tree computations, and quantization.
    /// </remarks>
    Vector<T> Log2<T>(Vector<T> vector);

    /// <summary>
    /// Computes exp(x) - 1 for each element with higher precision for small values.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing exp(x) - 1 for each element.</returns>
    /// <remarks>
    /// More accurate than Exp(x) - 1 for values near zero. Used in loss functions and probability computations.
    /// </remarks>
    Vector<T> ExpM1<T>(Vector<T> vector);

    /// <summary>
    /// Computes log(1 + x) for each element with higher precision for small values.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing log(1 + x) for each element.</returns>
    /// <remarks>
    /// More accurate than Log(1 + x) for values near zero. Used in probability and loss computations.
    /// </remarks>
    Vector<T> Log1P<T>(Vector<T> vector);

    /// <summary>
    /// Computes the sign (-1, 0, or +1) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing the signs.</returns>
    /// <remarks>
    /// <para><b>Phase B: US-GPU-015</b> - Required for Lion optimizer.</para>
    /// </remarks>
    Vector<T> Sign<T>(Vector<T> vector);

    /// <summary>
    /// Negates each element of the vector (computes -x).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with each element negated.</returns>
    /// <remarks>
    /// Used for gradient reversal, adversarial training, and sign flipping operations.
    /// </remarks>
    Vector<T> Negate<T>(Vector<T> vector);

    /// <summary>
    /// Gathers elements from a vector at regular stride intervals.
    /// Extracts source[offset], source[offset + stride], source[offset + 2*stride], ...
    /// </summary>
    /// <param name="source">The source vector to gather from.</param>
    /// <param name="offset">The starting index in the source vector.</param>
    /// <param name="stride">The step between consecutive gathered elements. Must be positive.</param>
    /// <param name="count">The number of elements to gather. If -1, gathers as many as fit.</param>
    /// <returns>A new vector containing the gathered elements.</returns>
    Vector<T> StridedGather<T>(Vector<T> source, int offset, int stride, int count = -1);

    /// <summary>
    /// Scatters elements from a source vector into a destination vector at regular stride intervals.
    /// Sets destination[offset + i*stride] = source[i] for i = 0..source.Length-1.
    /// </summary>
    /// <param name="destination">The destination vector to scatter into (modified in-place).</param>
    /// <param name="source">The source vector containing values to scatter.</param>
    /// <param name="offset">The starting index in the destination vector.</param>
    /// <param name="stride">The step between consecutive scattered elements. Must be positive.</param>
    void StridedScatter<T>(Vector<T> destination, Vector<T> source, int offset, int stride);

    #endregion

    #region Reduction Operations

    /// <summary>
    /// Computes the sum of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The sum of all elements.</returns>
    /// <remarks>
    /// <para>
    /// Reduction operation that sums all elements: result = v[0] + v[1] + ... + v[n-1].
    /// Critical for computing totals, norms, and other aggregate statistics.
    /// CPU implementation uses parallel reduction for large vectors.
    /// GPU implementation uses warp-level reduction primitives for maximum efficiency.
    /// </para>
    /// </remarks>
    T Sum<T>(Vector<T> vector);

    /// <summary>
    /// Computes the dot product (inner product) of two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The dot product of the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Computes result = sum(a[i] * b[i]) for all i.
    /// Fundamental operation in linear algebra used for:
    /// - Computing similarities and distances
    /// - Matrix-vector products (each row dot product with vector)
    /// - Neural network forward/backward passes
    /// - ARIMA/time series predictions
    /// </para>
    /// <para>
    /// CPU implementation uses SIMD and parallel reduction.
    /// GPU implementation uses warp-level primitives for maximum throughput.
    /// This is one of the most performance-critical operations in deep learning.
    /// </para>
    /// </remarks>
    T DotProduct<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the dot product of vector a with a strided window of vector b.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="a">The first vector (contiguous).</param>
    /// <param name="b">The source vector to read from with stride.</param>
    /// <param name="bOffset">Starting index in b.</param>
    /// <param name="bStride">Step between elements in b (e.g., -1 for reverse).</param>
    /// <returns>The dot product sum(a[i] * b[bOffset + i * bStride]).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like a normal dot product, but reads elements from b
    /// at non-contiguous positions. A stride of -1 reads backwards (useful for AR/MA models).
    /// A stride of 2 reads every other element. The length is determined by a.Length.</para>
    /// <para><b>Out-of-range semantics:</b> If bOffset + i * bStride falls outside [0, b.Length),
    /// that element contributes 0 to the sum (boundary clamping). This is intentional for
    /// time series AR/MA models where the lag window may extend before the start of the series.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when a or b is null.</exception>
    T DotProduct<T>(Vector<T> a, Vector<T> b, int bOffset, int bStride);

    /// <summary>
    /// Computes the mean (average) of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The mean of all elements.</returns>
    /// <remarks>
    /// <para>
    /// Computes result = sum(v[i]) / length.
    /// Equivalent to Sum(vector) divided by vector length, but may use optimized implementations.
    /// Used extensively in statistics, normalization, and time series analysis.
    /// </para>
    /// </remarks>
    T Mean<T>(Vector<T> vector);

    /// <summary>
    /// Applies the softmax function to convert a vector of values into a probability distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector of logits.</param>
    /// <returns>A new vector where elements sum to 1 and represent probabilities.</returns>
    /// <remarks>
    /// <para>
    /// Softmax converts arbitrary real values into a probability distribution using:
    /// softmax(x)[i] = exp(x[i]) / sum(exp(x[j])) for all j
    /// </para>
    /// <para>
    /// For numerical stability, this is computed as:
    /// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
    /// </para>
    /// <para><b>Common Uses:</b></para>
    /// <list type="bullet">
    /// <item><description>Final layer of classification networks (converts scores to probabilities)</description></item>
    /// <item><description>Attention mechanisms (weighs how much focus to give each element)</description></item>
    /// <item><description>Mixture-of-Experts routing (determines which experts to use)</description></item>
    /// <item><description>Reinforcement learning policy networks</description></item>
    /// </list>
    /// <para>
    /// GPU implementation provides 10-50x speedup for large vectors.
    /// Uses hardware-accelerated exp() and reduction operations.
    /// </para>
    /// </remarks>
    Vector<T> Softmax<T>(Vector<T> vector);

    /// <summary>
    /// Computes the cosine similarity between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A value between -1 and 1 representing the cosine of the angle between the vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Cosine similarity measures the cosine of the angle between two non-zero vectors:
    /// cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
    /// </para>
    /// <para><b>Return Values:</b></para>
    /// <list type="bullet">
    /// <item><description>1.0: Vectors point in exactly the same direction (identical)</description></item>
    /// <item><description>0.0: Vectors are orthogonal (perpendicular, no similarity)</description></item>
    /// <item><description>-1.0: Vectors point in opposite directions</description></item>
    /// </list>
    /// <para><b>Common Uses:</b></para>
    /// <list type="bullet">
    /// <item><description>Text similarity in NLP (document/sentence embeddings)</description></item>
    /// <item><description>Recommendation systems (user/item similarity)</description></item>
    /// <item><description>Image similarity (feature vector comparison)</description></item>
    /// <item><description>Attention mechanisms in transformers</description></item>
    /// </list>
    /// <para>
    /// GPU implementation provides 20-100x speedup by parallelizing dot product and norm computations.
    /// Returns zero if either vector has zero magnitude to avoid division by zero.
    /// </para>
    /// </remarks>
    T CosineSimilarity<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the product of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The product of all elements.</returns>
    /// <remarks>
    /// Used for geometric mean computation and product aggregation in statistics.
    /// </remarks>
    T Product<T>(Vector<T> vector);

    /// <summary>
    /// Computes the standard deviation of all elements in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The standard deviation of all elements.</returns>
    /// <remarks>
    /// <para>
    /// Standard deviation measures the spread of values: sqrt(variance).
    /// Essential for batch normalization, layer normalization, and outlier detection.
    /// </para>
    /// </remarks>
    T StdDev<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Euclidean norm (L2 norm) of the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>The L2 norm: sqrt(sum(x[i]ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²)).</returns>
    /// <remarks>
    /// <para>
    /// L2 norm is the Euclidean length of the vector.
    /// Critical for:
    /// - Gradient clipping (clip by norm)
    /// - L2 regularization
    /// - Vector normalization (unit vectors)
    /// - Distance metrics
    /// </para>
    /// </remarks>
    T Norm<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Euclidean distance between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The Euclidean distance: sqrt(sum((a[i] - b[i])ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â²)).</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Euclidean distance is the straight-line distance between two points.
    /// Used extensively in:
    /// - k-Nearest Neighbors (k-NN)
    /// - Clustering algorithms (k-means, DBSCAN)
    /// - Metric learning
    /// - Similarity search
    /// </para>
    /// </remarks>
    T Distance<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the element-wise minimum magnitude between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector where each element is the value with minimum absolute value.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// Used for magnitude-based comparisons and weight pruning strategies.
    /// </remarks>
    Vector<T> MinMagnitude<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Computes the element-wise maximum magnitude between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>A new vector where each element is the value with maximum absolute value.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// Used for gradient clipping by magnitude and weight analysis.
    /// </remarks>
    Vector<T> MaxMagnitude<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Creates a vector filled with a constant value.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="length">The length of the vector to create.</param>
    /// <param name="value">The value to fill all elements with.</param>
    /// <returns>A new vector with all elements set to the specified value.</returns>
    Vector<T> Fill<T>(int length, T value);

    /// <summary>
    /// Creates a vector filled with zeros.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="length">The length of the vector to create.</param>
    /// <returns>A new vector with all elements set to zero.</returns>
    Vector<T> FillZero<T>(int length);

    /// <summary>
    /// Generates a dropout mask where each element is either zero or a scale value.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="length">The length of the mask vector to create.</param>
    /// <param name="dropoutRate">Probability of dropping each element (0 to 1).</param>
    /// <param name="scale">Scale value for kept elements.</param>
    /// <param name="seed">Random seed for reproducibility (optional).</param>
    /// <returns>A new vector containing the dropout mask.</returns>
    Vector<T> GenerateDropoutMask<T>(int length, T dropoutRate, T scale, int? seed = null);

    /// <summary>
    /// Copies elements from a vector to a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="source">The source vector.</param>
    /// <param name="destination">The destination tensor.</param>
    void CopyVectorToTensor<T>(Vector<T> source, Tensor<T> destination);
    /// <summary>
    /// Generates Gaussian random noise using the Box-Muller transform.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="length">The length of the noise vector to create.</param>
    /// <param name="mean">The mean of the Gaussian distribution.</param>
    /// <param name="standardDeviation">The standard deviation of the Gaussian distribution.</param>
    /// <param name="seed">Random seed for reproducibility (optional).</param>
    /// <returns>A new vector containing Gaussian random noise.</returns>
    Vector<T> GenerateGaussianNoise<T>(int length, T mean, T standardDeviation, int? seed = null);

    #endregion

    #region Specialized Operations

    /// <summary>
    /// Clamps each element of a vector to the specified range [min, max].
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    /// <returns>A new vector with each element clamped to [min, max].</returns>
    /// <remarks>
    /// <para>
    /// Clamp ensures all values are within bounds: result[i] = max(min, min(max, vector[i])).
    /// Critical for:
    /// - Gradient clipping (prevent exploding gradients)
    /// - Activation function bounds (e.g., ReLU6)
    /// - Numerical stability
    /// - Value range enforcement
    /// </para>
    /// </remarks>
    Vector<T> Clamp<T>(Vector<T> vector, T min, T max);

    /// <summary>
    /// Performs linear interpolation between two vectors.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vectors.</typeparam>
    /// <param name="a">The start vector.</param>
    /// <param name="b">The end vector.</param>
    /// <param name="t">The interpolation weight (0 to 1).</param>
    /// <returns>A new vector with interpolated values: a + t * (b - a).</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Linear interpolation blends two vectors based on weight t.
    /// Used for:
    /// - Exponential moving average (EMA) in optimizers
    /// - Model weight interpolation
    /// - Smooth transitions between states
    /// - Temporal blending
    /// </para>
    /// </remarks>
    Vector<T> Lerp<T>(Vector<T> a, Vector<T> b, T t);

    /// <summary>
    /// Computes the reciprocal (1/x) of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing 1/x for each element.</returns>
    /// <remarks>
    /// <para>
    /// Reciprocal is used for:
    /// - Division optimization (multiply by reciprocal)
    /// - Normalization operations
    /// - Inverse scaling
    /// </para>
    /// </remarks>
    Vector<T> Reciprocal<T>(Vector<T> vector);

    /// <summary>
    /// Computes the reciprocal square root (1/sqrt(x)) of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing 1/sqrt(x) for each element.</returns>
    /// <remarks>
    /// <para>
    /// Reciprocal square root is CRITICAL for normalization efficiency.
    /// Essential for:
    /// - Layer normalization: x / sqrt(variance + epsilon)
    /// - Batch normalization: (x - mean) / sqrt(variance + epsilon)
    /// - RMS normalization (RMSNorm used in LLaMA, GPT-NeoX)
    /// - Fast inverse square root (Quake III algorithm)
    /// </para>
    /// <para>
    /// GPU/SIMD implementations provide hardware-accelerated rsqrt instruction,
    /// which is significantly faster than computing sqrt followed by division.
    /// </para>
    /// </remarks>
    Vector<T> ReciprocalSqrt<T>(Vector<T> vector);

    #endregion

    #region Trigonometric Operations

    /// <summary>
    /// Computes the sine of each element (in radians).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (angles in radians).</param>
    /// <returns>A new vector containing sine values.</returns>
    /// <remarks>
    /// <para>
    /// Sine is used extensively in:
    /// - Positional encodings for transformers (sin/cos for position embedding)
    /// - Signal processing and wave functions
    /// - Fourier transforms
    /// - Periodic activation functions
    /// </para>
    /// </remarks>
    Vector<T> Sin<T>(Vector<T> vector);

    /// <summary>
    /// Computes the cosine of each element (in radians).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (angles in radians).</param>
    /// <returns>A new vector containing cosine values.</returns>
    /// <remarks>
    /// <para>
    /// Cosine is used extensively in:
    /// - Positional encodings for transformers (sin/cos for position embedding)
    /// - Cosine annealing learning rate schedules
    /// - Attention mechanisms
    /// - Signal processing
    /// </para>
    /// </remarks>
    Vector<T> Cos<T>(Vector<T> vector);

    /// <summary>
    /// Computes both sine and cosine of each element simultaneously (in radians).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (angles in radians).</param>
    /// <param name="sinResult">Output vector containing sine values.</param>
    /// <param name="cosResult">Output vector containing cosine values.</param>
    /// <remarks>
    /// <para>
    /// SinCos computes both sin and cos simultaneously, which is more efficient
    /// than calling Sin() and Cos() separately.
    /// Critical for:
    /// - Positional encodings in transformers (need both sin and cos)
    /// - Complex number operations
    /// - Rotary position embeddings (RoPE)
    /// </para>
    /// <para>
    /// Hardware implementations can compute both with ~1.5x the cost of a single
    /// sin/cos operation, rather than 2x.
    /// </para>
    /// </remarks>
    void SinCos<T>(Vector<T> vector, out Vector<T> sinResult, out Vector<T> cosResult);

    /// <summary>
    /// Computes the sine of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write sine values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles or 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 8-12x speedup on AVX-512 hardware compared to scalar Math.Sin loop.
    /// </para>
    /// </remarks>
    void Sin(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the sine of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write sine values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles or 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 8-12x speedup on AVX-512 hardware compared to scalar Math.Sin loop.
    /// </para>
    /// </remarks>
    void Sin(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the cosine of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write cosine values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles or 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 8-12x speedup on AVX-512 hardware compared to scalar Math.Cos loop.
    /// </para>
    /// </remarks>
    void Cos(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the cosine of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write cosine values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles or 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 8-12x speedup on AVX-512 hardware compared to scalar Math.Cos loop.
    /// </para>
    /// </remarks>
    void Cos(System.ReadOnlySpan<double> x, System.Span<double> destination);

    #endregion

    #region Exponential Operations

    /// <summary>
    /// Computes e raised to the power of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (exponents).</param>
    /// <param name="destination">The destination span to write exp values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 4-12x speedup on AVX-512 hardware compared to scalar Math.Exp loop.
    /// </para>
    /// </remarks>
    void Exp(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes e raised to the power of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (exponents).</param>
    /// <param name="destination">The destination span to write exp values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 4-12x speedup on AVX-512 hardware compared to scalar Math.Exp loop.
    /// </para>
    /// </remarks>
    void Exp(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the natural logarithm of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be positive).</param>
    /// <param name="destination">The destination span to write log values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 16 floats simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 4-12x speedup on AVX-512 hardware compared to scalar Math.Log loop.
    /// </para>
    /// </remarks>
    void Log(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the natural logarithm of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be positive).</param>
    /// <param name="destination">The destination span to write log values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// <para>
    /// On .NET 8.0 with AVX-512 hardware, processes 8 doubles simultaneously.
    /// On .NET Framework 4.6.2/4.7.1, uses scalar operations only.
    /// </para>
    /// <para>
    /// Performance: 4-12x speedup on AVX-512 hardware compared to scalar Math.Log loop.
    /// </para>
    /// </remarks>
    void Log(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes exp(x) - 1 for each element in a span with numerical stability using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write exp(x) - 1 values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// More accurate than Exp(x) - 1 for values near zero. Used in loss functions and probability computations.
    /// On .NET 5.0+, uses Math.ExpM1 for improved numerical stability.
    /// </para>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// </remarks>
    void ExpM1(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes exp(x) - 1 for each element in a span with numerical stability using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write exp(x) - 1 values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// More accurate than Exp(x) - 1 for values near zero. Used in loss functions and probability computations.
    /// On .NET 5.0+, uses Math.ExpM1 for improved numerical stability.
    /// </para>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// </remarks>
    void ExpM1(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes log(1 + x) for each element in a span with numerical stability using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write log(1 + x) values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// More accurate than Log(1 + x) for values near zero. Used in probability and loss computations.
    /// On .NET 5.0+, uses Math.Log1P for improved numerical stability.
    /// </para>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// </remarks>
    void Log1P(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes log(1 + x) for each element in a span with numerical stability using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write log(1 + x) values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// More accurate than Log(1 + x) for values near zero. Used in probability and loss computations.
    /// On .NET 5.0+, uses Math.Log1P for improved numerical stability.
    /// </para>
    /// <para>
    /// Span-based overload for maximum performance when working with raw memory.
    /// Automatically dispatches to AVX-512, AVX2, SSE, or scalar based on hardware support.
    /// </para>
    /// </remarks>
    void Log1P(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the tangent of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write tangent values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Tan(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the tangent of each element in a span (in radians) using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (angles in radians).</param>
    /// <param name="destination">The destination span to write tangent values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Tan(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse sine of each element in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (values in range [-1, 1]).</param>
    /// <returns>A new vector containing arcsin values in range [-Ãâ‚¬/2, Ãâ‚¬/2].</returns>
    /// <remarks>
    /// <para>
    /// Inverse sine (arcsin) is the inverse of the sine function.
    /// Input domain: [-1, 1]
    /// Output range: [-Ãâ‚¬/2, Ãâ‚¬/2] radians
    /// </para>
    /// </remarks>
    Vector<T> Asin<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse sine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (values in range [-1, 1]).</param>
    /// <param name="destination">The destination span to write arcsin values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Asin(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse sine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (values in range [-1, 1]).</param>
    /// <param name="destination">The destination span to write arcsin values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Asin(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse cosine of each element in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector (values in range [-1, 1]).</param>
    /// <returns>A new vector containing arccos values in range [0, Ãâ‚¬].</returns>
    /// <remarks>
    /// <para>
    /// Inverse cosine (arccos) is the inverse of the cosine function.
    /// Input domain: [-1, 1]
    /// Output range: [0, Ãâ‚¬] radians
    /// </para>
    /// </remarks>
    Vector<T> Acos<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse cosine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (values in range [-1, 1]).</param>
    /// <param name="destination">The destination span to write arccos values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Acos(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse cosine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (values in range [-1, 1]).</param>
    /// <param name="destination">The destination span to write arccos values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Acos(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse tangent of each element in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing arctan values in range (-Ãâ‚¬/2, Ãâ‚¬/2).</returns>
    /// <remarks>
    /// <para>
    /// Inverse tangent (arctan) is the inverse of the tangent function.
    /// Input domain: (-Inf, Inf)
    /// Output range: (-Ãâ‚¬/2, Ãâ‚¬/2) radians
    /// </para>
    /// </remarks>
    Vector<T> Atan<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse tangent of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write arctan values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Atan(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse tangent of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write arctan values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Atan(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the square root of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be non-negative).</param>
    /// <param name="destination">The destination span to write sqrt values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Sqrt(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the square root of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be non-negative).</param>
    /// <param name="destination">The destination span to write sqrt values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Sqrt(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the absolute value of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write absolute values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Abs(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the absolute value of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write absolute values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Abs(System.ReadOnlySpan<double> x, System.Span<double> destination);

    #endregion

    #region Hyperbolic Operations

    /// <summary>
    /// Computes the hyperbolic sine of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing sinh values.</returns>
    /// <remarks>
    /// Hyperbolic sine: sinh(x) = (e^x - e^-x) / 2.
    /// Used in some activation functions and mathematical transformations.
    /// </remarks>
    Vector<T> Sinh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the hyperbolic cosine of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing cosh values.</returns>
    /// <remarks>
    /// Hyperbolic cosine: cosh(x) = (e^x + e^-x) / 2.
    /// Component of tanh gradient and used in hyperbolic geometry.
    /// </remarks>
    Vector<T> Cosh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse hyperbolic sine of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing asinh values.</returns>
    /// <remarks>
    /// Inverse hyperbolic sine: asinh(x) = log(x + sqrt(x^2 + 1)).
    /// Also called the hyperbolic area sine function. Domain: (-inf, inf), Range: (-inf, inf).
    /// </remarks>
    Vector<T> Asinh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse hyperbolic cosine of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing acosh values.</returns>
    /// <remarks>
    /// Inverse hyperbolic cosine: acosh(x) = log(x + sqrt(x^2 - 1)).
    /// Also called the hyperbolic area cosine function. Domain: [1, inf), Range: [0, inf).
    /// </remarks>
    Vector<T> Acosh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the inverse hyperbolic tangent of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing atanh values.</returns>
    /// <remarks>
    /// Inverse hyperbolic tangent: atanh(x) = 0.5 * log((1 + x) / (1 - x)).
    /// Also called the hyperbolic area tangent function. Domain: (-1, 1), Range: (-inf, inf).
    /// </remarks>
    Vector<T> Atanh<T>(Vector<T> vector);

    void Sinh(System.ReadOnlySpan<float> x, System.Span<float> destination);
    void Sinh(System.ReadOnlySpan<double> x, System.Span<double> destination);
    void Cosh(System.ReadOnlySpan<float> x, System.Span<float> destination);
    void Cosh(System.ReadOnlySpan<double> x, System.Span<double> destination);
    void Tanh(System.ReadOnlySpan<float> x, System.Span<float> destination);
    void Tanh(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse hyperbolic sine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write asinh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Asinh(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse hyperbolic sine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span.</param>
    /// <param name="destination">The destination span to write asinh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Asinh(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse hyperbolic cosine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be >= 1).</param>
    /// <param name="destination">The destination span to write acosh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Acosh(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse hyperbolic cosine of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be >= 1).</param>
    /// <param name="destination">The destination span to write acosh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Acosh(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the inverse hyperbolic tangent of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be in range (-1, 1)).</param>
    /// <param name="destination">The destination span to write atanh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Atanh(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the inverse hyperbolic tangent of each element in a span using SIMD acceleration.
    /// </summary>
    /// <param name="x">The input span (must be in range (-1, 1)).</param>
    /// <param name="destination">The destination span to write atanh values.</param>
    /// <exception cref="ArgumentException">Thrown when spans have different lengths.</exception>
    void Atanh(System.ReadOnlySpan<double> x, System.Span<double> destination);

    #endregion

    #region Additional Mathematical Operations

    /// <summary>
    /// Computes the reciprocal (1/x) of each element.
    /// </summary>
    void Reciprocal(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the reciprocal (1/x) of each element.
    /// </summary>
    void Reciprocal(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the cube root of each element.
    /// </summary>
    void Cbrt(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the cube root of each element.
    /// </summary>
    void Cbrt(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the base-2 logarithm of each element.
    /// </summary>
    void Log2(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the base-2 logarithm of each element.
    /// </summary>
    void Log2(System.ReadOnlySpan<double> x, System.Span<double> destination);

    /// <summary>
    /// Computes the base-10 logarithm of each element.
    /// </summary>
    void Log10(System.ReadOnlySpan<float> x, System.Span<float> destination);

    /// <summary>
    /// Computes the base-10 logarithm of each element.
    /// </summary>
    void Log10(System.ReadOnlySpan<double> x, System.Span<double> destination);

    #endregion

    #region Rounding Operations

    /// <summary>
    /// Rounds each element to the nearest integer.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with each element rounded to nearest integer.</returns>
    /// <remarks>
    /// <para>
    /// Rounding is used for:
    /// - Quantization (neural network compression)
    /// - Discretization for inference
    /// - Integer conversion
    /// </para>
    /// </remarks>
    Vector<T> Round<T>(Vector<T> vector);

    /// <summary>
    /// Computes the floor (round down) of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with each element rounded down to integer.</returns>
    /// <remarks>
    /// Floor rounds toward negative infinity. Used for integer conversion and binning operations.
    /// </remarks>
    Vector<T> Floor<T>(Vector<T> vector);

    /// <summary>
    /// Computes the ceiling (round up) of each element.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with each element rounded up to integer.</returns>
    /// <remarks>
    /// Ceiling rounds toward positive infinity. Used for integer conversion and binning operations.
    /// </remarks>
    Vector<T> Ceiling<T>(Vector<T> vector);

    /// <summary>
    /// Truncates each element toward zero (removes fractional part).
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with fractional parts removed.</returns>
    /// <remarks>
    /// Truncate rounds toward zero. Used for integer conversion and quantization schemes.
    /// </remarks>
    Vector<T> Truncate<T>(Vector<T> vector);

    #endregion

    #region Activation Functions

    /// <summary>
    /// Computes the hyperbolic tangent of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing tanh values between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Tanh activation function: tanh(x) = (e^x - e^-x) / (e^x + e^-x).
    /// Commonly used in hidden layers of neural networks.
    /// CPU implementation uses TensorPrimitives for SIMD optimization (3-6ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â speedup for float).
    /// GPU implementation uses DirectGpu backends.
    /// </para>
    /// </remarks>
    Vector<T> Tanh<T>(Vector<T> vector);

    /// <summary>
    /// Computes the sigmoid function of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector containing sigmoid values between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Sigmoid activation function: ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚ÂÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢(x) = 1 / (1 + e^-x).
    /// Commonly used for binary classification and gate functions in LSTMs/GRUs.
    /// CPU implementation uses TensorPrimitives for SIMD optimization (3-6ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â speedup for float).
    /// GPU implementation uses DirectGpu backends.
    /// </para>
    /// </remarks>
    Vector<T> Sigmoid<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Rectified Linear Unit (ReLU) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector where each element is max(0, x).</returns>
    /// <remarks>
    /// <para>
    /// ReLU activation function: ReLU(x) = max(0, x).
    /// Most commonly used activation in modern deep learning.
    /// CPU implementation uses TensorPrimitives for SIMD optimization.
    /// GPU implementation uses DirectGpu backends.
    /// </para>
    /// </remarks>
    Vector<T> ReLU<T>(Vector<T> vector);

    /// <summary>
    /// Computes the hyperbolic tangent of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A new tensor containing tanh values between -1 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Tensor version of Tanh for multi-dimensional data.
    /// CPU implementation uses TensorPrimitives for SIMD optimization.
    /// GPU implementation uses DirectGpu backends.
    /// </para>
    /// </remarks>
    Tensor<T> Tanh<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the sigmoid function of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A new tensor containing sigmoid values between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// Tensor version of Sigmoid for multi-dimensional data.
    /// CPU implementation uses TensorPrimitives for SIMD optimization.
    /// GPU implementation uses DirectGpu backends.
    /// </para>
    /// </remarks>
    Tensor<T> Sigmoid<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the ReLU of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the tensor.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A new tensor where each element is max(0, x).</returns>
    /// <remarks>
    /// <para>
    /// Tensor version of ReLU for multi-dimensional data.
    /// CPU implementation uses TensorPrimitives for SIMD optimization.
    /// GPU implementation uses DirectGpu backends.
    /// </para>
    /// </remarks>
    Tensor<T> ReLU<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the GELU (Gaussian Error Linear Unit) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with GELU activation applied.</returns>
    /// <remarks>
    /// <para>
    /// GELU activation: x * ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦(x) where ÃƒÆ’Ã†â€™Ãƒâ€¦Ã‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ is the standard Gaussian cumulative distribution.
    /// Commonly used in transformers (BERT, GPT) and modern architectures.
    /// Approximation: 0.5 * x * (1 + tanh(ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã¢â‚¬Â¹ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡(2/ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬) * (x + 0.044715 * xÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³)))
    /// </para>
    /// </remarks>
    Vector<T> GELU<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Mish activation of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with Mish activation applied.</returns>
    /// <remarks>
    /// <para>
    /// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    /// Smooth, self-regularizing activation function with better performance than ReLU in some tasks.
    /// </para>
    /// </remarks>
    Vector<T> Mish<T>(Vector<T> vector);

    /// <summary>
    /// Computes the Swish/SiLU activation of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <returns>A new vector with Swish activation applied.</returns>
    /// <remarks>
    /// <para>
    /// Swish/SiLU activation: x * sigmoid(x) = x / (1 + exp(-x)).
    /// Used in EfficientNet and other modern architectures. Self-gated activation.
    /// </para>
    /// </remarks>
    Vector<T> Swish<T>(Vector<T> vector);

    /// <summary>
    /// Computes the ELU (Exponential Linear Unit) of each element in the vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector.</typeparam>
    /// <param name="vector">The input vector.</param>
    /// <param name="alpha">Scale factor for negative values (default 1.0).</param>
    /// <returns>A new vector with ELU activation applied.</returns>
    /// <remarks>
    /// <para>
    /// ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise.
    /// Helps with vanishing gradient problem and can produce negative outputs.
    /// </para>
    /// </remarks>
    Vector<T> ELU<T>(Vector<T> vector, double alpha = 1.0);

    /// <summary>
    /// Computes the GELU of each element in the tensor.
    /// </summary>
    Tensor<T> GELU<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the Mish activation of each element in the tensor.
    /// </summary>
    Tensor<T> Mish<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the Swish/SiLU activation of each element in the tensor.
    /// </summary>
    Tensor<T> Swish<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the Leaky ReLU of each element in the tensor: f(x) = max(alpha * x, x).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="alpha">The slope for negative values (typically 0.01).</param>
    /// <returns>A new tensor with Leaky ReLU applied element-wise.</returns>
    Tensor<T> LeakyReLU<T>(Tensor<T> tensor, T alpha);

    /// <summary>
    /// Computes the ELU of each element in the tensor.
    /// </summary>
    Tensor<T> ELU<T>(Tensor<T> tensor, double alpha = 1.0);

    /// <summary>
    /// Applies the Gated Linear Unit (GLU) activation: GLU(a, b) = a * sigmoid(b).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor with shape [..., 2*dim] where the last dimension will be split in half.</param>
    /// <param name="dim">The dimension to split (default: -1, last dimension).</param>
    /// <returns>Output tensor with shape [..., dim].</returns>
    /// <remarks>
    /// <para>
    /// GLU splits the input along the specified dimension into two halves (a and b),
    /// then computes: output = a * sigmoid(b)
    /// </para>
    /// <para>
    /// Introduced in "Language Modeling with Gated Convolutional Networks" (Dauphin et al., 2017).
    /// </para>
    /// </remarks>
    Tensor<T> GLU<T>(Tensor<T> input, int dim = -1);

    /// <summary>
    /// Computes the backward pass for GLU.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer.</param>
    /// <param name="input">Original input tensor from forward pass.</param>
    /// <param name="dim">The dimension that was split.</param>
    /// <returns>Gradient with respect to the input.</returns>
    Tensor<T> GLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim = -1);

    /// <summary>
    /// Applies GeGLU activation: GeGLU(a, b) = a * GELU(b).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor with shape [..., 2*dim].</param>
    /// <param name="dim">The dimension to split (default: -1, last dimension).</param>
    /// <returns>Output tensor with shape [..., dim].</returns>
    /// <remarks>
    /// <para>
    /// GeGLU splits the input and applies: output = a * GELU(b)
    /// </para>
    /// <para>
    /// Used in PaLM, Gemma, and other modern transformers.
    /// Provides better gradient flow than standard GLU.
    /// </para>
    /// <para>
    /// Reference: "GLU Variants Improve Transformer" (Shazeer, 2020).
    /// </para>
    /// </remarks>
    Tensor<T> GeGLU<T>(Tensor<T> input, int dim = -1);

    /// <summary>
    /// Computes the backward pass for GeGLU.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer.</param>
    /// <param name="input">Original input tensor from forward pass.</param>
    /// <param name="dim">The dimension that was split.</param>
    /// <returns>Gradient with respect to the input.</returns>
    Tensor<T> GeGLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim = -1);

    /// <summary>
    /// Applies SwiGLU activation: SwiGLU(a, b) = a * Swish(b) = a * (b * sigmoid(b)).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor with shape [..., 2*dim].</param>
    /// <param name="dim">The dimension to split (default: -1, last dimension).</param>
    /// <returns>Output tensor with shape [..., dim].</returns>
    /// <remarks>
    /// <para>
    /// SwiGLU splits the input and applies: output = a * Swish(b) = a * b * sigmoid(b)
    /// </para>
    /// <para>
    /// Used in LLaMA, LLaMA-2, Mistral, and other modern LLMs.
    /// Generally outperforms GELU and ReLU in transformer feed-forward networks.
    /// </para>
    /// <para>
    /// Reference: "GLU Variants Improve Transformer" (Shazeer, 2020).
    /// </para>
    /// </remarks>
    Tensor<T> SwiGLU<T>(Tensor<T> input, int dim = -1);

    /// <summary>
    /// Computes the backward pass for SwiGLU.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer.</param>
    /// <param name="input">Original input tensor from forward pass.</param>
    /// <param name="dim">The dimension that was split.</param>
    /// <returns>Gradient with respect to the input.</returns>
    Tensor<T> SwiGLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim = -1);

    /// <summary>
    /// Applies ReGLU activation: ReGLU(a, b) = a * ReLU(b).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor with shape [..., 2*dim].</param>
    /// <param name="dim">The dimension to split (default: -1, last dimension).</param>
    /// <returns>Output tensor with shape [..., dim].</returns>
    /// <remarks>
    /// <para>
    /// ReGLU splits the input and applies: output = a * ReLU(b)
    /// </para>
    /// <para>
    /// Simpler than GeGLU/SwiGLU but can still outperform standard activations.
    /// </para>
    /// </remarks>
    Tensor<T> ReGLU<T>(Tensor<T> input, int dim = -1);

    /// <summary>
    /// Computes the backward pass for ReGLU.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer.</param>
    /// <param name="input">Original input tensor from forward pass.</param>
    /// <param name="dim">The dimension that was split.</param>
    /// <returns>Gradient with respect to the input.</returns>
    Tensor<T> ReGLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim = -1);

    /// <summary>
    /// Computes the Softplus activation: softplus(x) = log(1 + exp(x)).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor with Softplus applied element-wise.</returns>
    Tensor<T> Softplus<T>(Tensor<T> input);

    /// <summary>
    /// Computes the HardSwish activation: x * min(max(x + 3, 0), 6) / 6.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor with HardSwish applied element-wise.</returns>
    Tensor<T> HardSwish<T>(Tensor<T> input);

    /// <summary>
    /// Computes the backward pass for ReLU.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input);

    /// <summary>
    /// Computes the backward pass for Sigmoid.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="output">The sigmoid output from forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> SigmoidBackward<T>(Tensor<T> gradOutput, Tensor<T> output);

    /// <summary>
    /// Computes the backward pass for Tanh.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="output">The tanh output from forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> TanhBackward<T>(Tensor<T> gradOutput, Tensor<T> output);

    /// <summary>
    /// Computes the backward pass for GELU.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> GeluBackward<T>(Tensor<T> gradOutput, Tensor<T> input);

    /// <summary>
    /// Computes the backward pass for LeakyReLU.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="negativeSlope">The slope for negative values.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> LeakyReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input, double negativeSlope);

    /// <summary>Backward pass for Swish/SiLU activation.</summary>
    Tensor<T> SwishBackward<T>(Tensor<T> gradOutput, Tensor<T> input);

    /// <summary>Backward pass for Mish activation.</summary>
    Tensor<T> MishBackward<T>(Tensor<T> gradOutput, Tensor<T> input);

    /// <summary>Backward pass for Softplus activation.</summary>
    Tensor<T> SoftplusBackward<T>(Tensor<T> gradOutput, Tensor<T> input);

    /// <summary>Backward pass for HardSwish activation.</summary>
    Tensor<T> HardswishBackward<T>(Tensor<T> gradOutput, Tensor<T> input);

    /// <summary>Backward pass for SELU activation.</summary>
    Tensor<T> SeluBackward<T>(Tensor<T> gradOutput, Tensor<T> input);

    /// <summary>Backward pass for HardSigmoid activation.</summary>
    Tensor<T> HardsigmoidBackward<T>(Tensor<T> gradOutput, Tensor<T> input);

    /// <summary>Backward pass for ReLU6 activation.</summary>
    Tensor<T> Relu6Backward<T>(Tensor<T> gradOutput, Tensor<T> input);

    /// <summary>Backward pass for ELU activation.</summary>
    Tensor<T> EluBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, double alpha);

    /// <summary>Backward pass for Threshold activation. Uses double threshold to match forward pass precision.</summary>
    Tensor<T> ThresholdBackward<T>(Tensor<T> gradOutput, Tensor<T> input, double threshold);

    /// <summary>Backward pass for Reciprocal.</summary>
    Tensor<T> ReciprocalBackward<T>(Tensor<T> gradOutput, Tensor<T> output);

    /// <summary>Backward pass for PReLU: returns (inputGrad, alphaGrad).</summary>
    (Tensor<T> inputGrad, Tensor<T> alphaGrad) PReLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> alpha);

    /// <summary>Backward pass for global variance reduction. For axis-specific backward, see ReduceVarianceBackward.
    /// This method handles the common case of global variance used by autograd.</summary>
    Tensor<T> VarBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, int[] axes);

    /// <summary>Backward pass for Std reduction.</summary>
    Tensor<T> StdBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, Tensor<T> std, int[] axes);

    #endregion

    #region Matrix Operations (Phase B: Epic 2)

    /// <summary>
    /// Performs matrix-matrix multiplication (GEMM: General Matrix Multiply).
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="a">The first matrix (M x K).</param>
    /// <param name="b">The second matrix (K x N).</param>
    /// <returns>The product matrix (M x N).</returns>
    /// <exception cref="ArgumentException">Thrown when matrix dimensions are incompatible.</exception>
    /// <remarks>
    /// <para><b>US-GPU-007: GEMM</b></para>
    /// <para>
    /// Matrix multiplication is O(nÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³) - highly computationally intensive.
    /// GPU acceleration provides 100-1000x speedup for large matrices.
    /// Essential for dense neural network layers.
    /// </para>
    /// </remarks>
    Matrix<T> MatrixMultiply<T>(Matrix<T> a, Matrix<T> b);

    /// <summary>
    /// Performs matrix-vector multiplication (GEMV).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="matrix">The matrix (M x N).</param>
    /// <param name="vector">The vector (N elements).</param>
    /// <returns>The result vector (M elements).</returns>
    /// <exception cref="ArgumentException">Thrown when dimensions are incompatible.</exception>
    /// <remarks>
    /// <para><b>US-GPU-008: GEMV</b></para>
    /// <para>
    /// Computes result[i] = sum(matrix[i, j] * vector[j]) for all i.
    /// Critical for neural network inference.
    /// </para>
    /// </remarks>
    Vector<T> MatrixVectorMultiply<T>(Matrix<T> matrix, Vector<T> vector);

    /// <summary>
    /// Transposes a matrix (rows become columns).
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The input matrix (M x N).</param>
    /// <returns>The transposed matrix (N x M).</returns>
    /// <remarks>
    /// <para><b>US-GPU-009: Matrix Transpose</b></para>
    /// <para>
    /// Required for backpropagation in neural networks.
    /// GPU implementation uses shared memory for coalesced access.
    /// </para>
    /// </remarks>
    Matrix<T> MatrixTranspose<T>(Matrix<T> matrix);

    /// <summary>
    /// Adds two matrices element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="a">The first matrix.</param>
    /// <param name="b">The second matrix.</param>
    /// <returns>A new matrix containing the element-wise sum.</returns>
    /// <exception cref="ArgumentException">Thrown when matrix dimensions don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-010: Matrix Element-Wise Operations</b></para>
    /// </remarks>
    Matrix<T> MatrixAdd<T>(Matrix<T> a, Matrix<T> b);

    /// <summary>
    /// Multiplies a matrix by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="matrix">The matrix to multiply.</param>
    /// <param name="scalar">The scalar value.</param>
    /// <returns>A new matrix with all elements multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>US-GPU-010: Matrix Element-Wise Operations</b></para>
    /// </remarks>
    Matrix<T> MatrixMultiplyScalar<T>(Matrix<T> matrix, T scalar);

    /// <summary>
    /// Subtracts matrix b from matrix a element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="a">The first matrix.</param>
    /// <param name="b">The second matrix.</param>
    /// <returns>A new matrix containing the element-wise difference (a - b).</returns>
    /// <exception cref="ArgumentException">Thrown when matrix dimensions don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-010: Matrix Element-Wise Operations</b></para>
    /// </remarks>
    Matrix<T> MatrixSubtract<T>(Matrix<T> a, Matrix<T> b);

    /// <summary>
    /// Computes the sum of squared elements of a matrix (used for Frobenius norm computation).
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The input matrix.</param>
    /// <returns>The sum of all squared elements: sum_{i,j} matrix[i,j]^2</returns>
    /// <remarks>
    /// <para><b>US-GPU-010: Matrix Element-Wise Operations</b></para>
    /// <para>
    /// This is used to compute the squared Frobenius norm: ||A||_F^2 = sum_{i,j} A_{ij}^2
    /// To get the actual Frobenius norm, take sqrt of the result.
    /// </para>
    /// </remarks>
    T MatrixSumOfSquares<T>(Matrix<T> matrix);

    /// <summary>
    /// Swaps two columns in a matrix in-place using vectorized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The matrix to modify.</param>
    /// <param name="col1">The first column index.</param>
    /// <param name="col2">The second column index.</param>
    /// <remarks>
    /// GPU-accelerated column swapping for matrix decompositions.
    /// </remarks>
    void SwapColumns<T>(Matrix<T> matrix, int col1, int col2);

    /// <summary>
    /// Swaps two rows in a matrix in-place using vectorized operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The matrix to modify.</param>
    /// <param name="row1">The first row index.</param>
    /// <param name="row2">The second row index.</param>
    /// <remarks>
    /// GPU-accelerated row swapping for matrix decompositions.
    /// </remarks>
    void SwapRows<T>(Matrix<T> matrix, int row1, int row2);

    /// <summary>
    /// Computes the outer product of two vectors: result[i,j] = a[i] * b[j].
    /// </summary>
    /// <typeparam name="T">The numeric type of vector elements.</typeparam>
    /// <param name="a">The first vector (length M).</param>
    /// <param name="b">The second vector (length N).</param>
    /// <returns>An MÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂN matrix containing the outer product.</returns>
    /// <remarks>
    /// GPU-accelerated outer product for SVD and other decompositions.
    /// </remarks>
    Matrix<T> OuterProduct<T>(Vector<T> a, Vector<T> b);

    /// <summary>
    /// Extracts a column from a matrix as a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="columnIndex">The column index to extract.</param>
    /// <returns>A vector containing the column values.</returns>
    /// <remarks>
    /// GPU-accelerated column extraction.
    /// </remarks>
    Vector<T> GetColumn<T>(Matrix<T> matrix, int columnIndex);

    /// <summary>
    /// Extracts a row from a matrix as a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The source matrix.</param>
    /// <param name="rowIndex">The row index to extract.</param>
    /// <returns>A vector containing the row values.</returns>
    /// <remarks>
    /// GPU-accelerated row extraction.
    /// </remarks>
    Vector<T> GetRow<T>(Matrix<T> matrix, int rowIndex);

    /// <summary>
    /// Sets a column in a matrix from a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The target matrix.</param>
    /// <param name="columnIndex">The column index to set.</param>
    /// <param name="values">The vector of values to set.</param>
    /// <remarks>
    /// GPU-accelerated column setting.
    /// </remarks>
    void SetColumn<T>(Matrix<T> matrix, int columnIndex, Vector<T> values);

    /// <summary>
    /// Sets a row in a matrix from a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of matrix elements.</typeparam>
    /// <param name="matrix">The target matrix.</param>
    /// <param name="rowIndex">The row index to set.</param>
    /// <param name="values">The vector of values to set.</param>
    /// <remarks>
    /// GPU-accelerated row setting.
    /// </remarks>
    void SetRow<T>(Matrix<T> matrix, int rowIndex, Vector<T> values);

    #endregion

    #region Tensor Operations (Phase B: Epic 3)

    /// <summary>
    /// Reshapes a tensor to a new shape with the same total number of elements.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor to reshape.</param>
    /// <param name="newShape">The new shape dimensions.</param>
    /// <returns>A new tensor with the specified shape and the same data.</returns>
    /// <exception cref="ArgumentException">Thrown when the total number of elements doesn't match.</exception>
    Tensor<T> Reshape<T>(Tensor<T> tensor, int[] newShape);

    /// <summary>
    /// Performs batched matrix multiplication on 3D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor [B, M, K] - B batches of MÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂK matrices.</param>
    /// <param name="b">The second tensor [B, K, N] - B batches of KÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂN matrices.</param>
    /// <returns>The result tensor [B, M, N] - B batches of MÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂN matrices.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor dimensions are incompatible.</exception>
    /// <remarks>
    /// <para><b>US-GPU-013: BatchMatMul</b></para>
    /// <para>
    /// Batched matrix multiplication performs C[i] = A[i] @ B[i] for all i in the batch.
    /// Critical for transformer models and attention mechanisms where multiple matrices
    /// must be multiplied in parallel.
    /// </para>
    /// <para>
    /// Input shapes:
    /// - a: [B, M, K] where B = batch size, M = rows, K = inner dimension
    /// - b: [B, K, N] where N = columns
    /// Output: [B, M, N]
    /// </para>
    /// <para>
    /// GPU acceleration provides 50-500x speedup by processing all batches in parallel.
    /// </para>
    /// </remarks>
    Tensor<T> BatchMatMul<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise sum.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// <para>
    /// Performs result[i] = a[i] + b[i] for all elements.
    /// Both tensors must have identical shapes.
    /// GPU acceleration provides significant speedup for large tensors.
    /// </para>
    /// </remarks>
    Tensor<T> TensorAdd<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Adds two tensors with broadcasting support, following NumPy/PyTorch broadcasting rules.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor. Can have different shape if broadcastable.</param>
    /// <returns>A new tensor containing the element-wise sum with broadcasting.</returns>
    /// <exception cref="ArgumentException">Thrown when shapes are not broadcastable.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations with Broadcasting</b></para>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be added together by automatically
    /// expanding dimensions of size 1 to match the other tensor. This is essential for operations
    /// like adding per-channel bias in convolutional layers.
    /// </para>
    /// <para>
    /// For example, adding shapes [batch, channels, H, W] + [1, channels, 1, 1] broadcasts
    /// the bias across batch and spatial dimensions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBroadcastAdd<T>(Tensor<T> a, Tensor<T> b);

    #region In-Place and Into Operations (Zero-Allocation)

    /// <summary>
    /// Adds tensor b into tensor a in-place: a[i] += b[i]. Zero allocation.
    /// </summary>
    void TensorAddInPlace<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Adds two tensors into a pre-allocated destination: dest[i] = a[i] + b[i]. Zero allocation.
    /// </summary>
    void TensorAddInto<T>(Tensor<T> destination, Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Multiplies tensor b into tensor a element-wise in-place: a[i] *= b[i]. Zero allocation.
    /// </summary>
    void TensorMultiplyInPlace<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Multiplies two tensors element-wise into a pre-allocated destination: dest[i] = a[i] * b[i]. Zero allocation.
    /// </summary>
    void TensorMultiplyInto<T>(Tensor<T> destination, Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Subtracts tensor b from tensor a in-place: a[i] -= b[i]. Zero allocation.
    /// </summary>
    void TensorSubtractInPlace<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Subtracts two tensors into a pre-allocated destination: dest[i] = a[i] - b[i]. Zero allocation.
    /// </summary>
    void TensorSubtractInto<T>(Tensor<T> destination, Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Multiplies all elements of tensor a by a scalar in-place: a[i] *= scalar. Zero allocation.
    /// </summary>
    void TensorMultiplyScalarInPlace<T>(Tensor<T> a, T scalar);

    /// <summary>
    /// Multiplies all elements of a tensor by a scalar into a pre-allocated destination: dest[i] = a[i] * scalar. Zero allocation.
    /// </summary>
    void TensorMultiplyScalarInto<T>(Tensor<T> destination, Tensor<T> a, T scalar);

    /// <summary>
    /// Adds tensor b to tensor a in-place with broadcasting: a += broadcast(b). Zero allocation.
    /// Essential for bias addition in Conv/Dense layers without allocating a new output tensor.
    /// </summary>
    void TensorBroadcastAddInPlace<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Applies sigmoid activation in-place: tensor[i] = 1 / (1 + exp(-tensor[i])). Zero allocation.
    /// </summary>
    void SigmoidInPlace<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes sigmoid into a pre-allocated destination: dest[i] = sigmoid(input[i]). Zero allocation.
    /// </summary>
    void SigmoidInto<T>(Tensor<T> destination, Tensor<T> input);

    /// <summary>
    /// Applies ReLU activation in-place: tensor[i] = max(0, tensor[i]). Zero allocation.
    /// </summary>
    void ReLUInPlace<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes ReLU into a pre-allocated destination: dest[i] = max(0, input[i]). Zero allocation.
    /// </summary>
    void ReLUInto<T>(Tensor<T> destination, Tensor<T> input);

    /// <summary>
    /// Computes Group Normalization into a pre-allocated output tensor.
    /// The main output avoids allocation since it's pre-allocated by the caller.
    /// The mean and variance stats are small tensors [batch, numGroups] allocated by the callee
    /// (needed for backward pass; negligible size compared to the feature map output).
    /// </summary>
    void GroupNormInto<T>(Tensor<T> output, Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance);

    /// <summary>
    /// Fused GroupNorm + Swish/SiLU: output = swish(GroupNorm(input)).
    /// Eliminates one intermediate tensor per DiffusionResBlock.
    /// </summary>
    void GroupNormSwishInto<T>(Tensor<T> output, Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon);

    /// <summary>
    /// Fused Add + GroupNorm: output = GroupNorm(a + b).
    /// For residual connections followed by normalization.
    /// </summary>
    void AddGroupNormInto<T>(Tensor<T> output, Tensor<T> a, Tensor<T> b, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon);

    /// <summary>In-place Swish/SiLU activation.</summary>
    void SwishInPlace<T>(Tensor<T> tensor);

    /// <summary>Swish/SiLU into destination.</summary>
    void SwishInto<T>(Tensor<T> destination, Tensor<T> input);

    /// <summary>In-place GELU activation.</summary>
    void GELUInPlace<T>(Tensor<T> tensor);

    /// <summary>GELU into destination.</summary>
    void GELUInto<T>(Tensor<T> destination, Tensor<T> input);

    /// <summary>In-place Tanh activation.</summary>
    void TanhInPlace<T>(Tensor<T> tensor);

    /// <summary>Tanh into destination.</summary>
    void TanhInto<T>(Tensor<T> destination, Tensor<T> input);

    /// <summary>In-place Mish activation.</summary>
    void MishInPlace<T>(Tensor<T> tensor);

    /// <summary>Mish into destination.</summary>
    void MishInto<T>(Tensor<T> destination, Tensor<T> input);

    /// <summary>In-place LeakyReLU activation.</summary>
    void LeakyReLUInPlace<T>(Tensor<T> tensor, T alpha);

    /// <summary>LeakyReLU into destination.</summary>
    void LeakyReLUInto<T>(Tensor<T> destination, Tensor<T> input, T alpha);

    /// <summary>Matrix multiply into pre-allocated destination.</summary>
    void MatMulInto<T>(Tensor<T> destination, Tensor<T> a, Tensor<T> b);

    /// <summary>Concatenate tensors along axis into pre-allocated destination.</summary>
    void ConcatInto<T>(Tensor<T> destination, Tensor<T>[] tensors, int axis);

    /// <summary>Transpose tensor into pre-allocated destination.</summary>
    void TransposeInto<T>(Tensor<T> destination, Tensor<T> input, int[] axes);

    /// <summary>Softmax into pre-allocated destination. Zero allocation.</summary>
    void SoftmaxInto<T>(Tensor<T> destination, Tensor<T> input, int axis);

    /// <summary>LogSoftmax into pre-allocated destination. Zero allocation.</summary>
    void LogSoftmaxInto<T>(Tensor<T> destination, Tensor<T> input, int axis);

    #endregion

    /// <summary>
    /// Subtracts two tensors element-wise with NumPy-style broadcasting.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor to subtract (will be broadcast to match a if needed).</param>
    /// <returns>A new tensor containing the element-wise difference with broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be subtracted by automatically
    /// expanding the smaller tensor. This is commonly used in operations like normalizing
    /// by subtracting a mean: [batch, features] - [1, features] broadcasts the mean across the batch.
    /// </para>
    /// <para>
    /// For example, subtracting shapes [batch, channels, H, W] - [1, channels, 1, 1] broadcasts
    /// the bias subtraction across batch and spatial dimensions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBroadcastSubtract<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Divides two tensors element-wise with NumPy-style broadcasting.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The dividend tensor.</param>
    /// <param name="b">The divisor tensor (will be broadcast to match a if needed).</param>
    /// <returns>A new tensor containing the element-wise quotient with broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be divided by automatically
    /// expanding the smaller tensor. This is commonly used in normalization operations
    /// like dividing by a sum: [batch, features] / [batch, 1] broadcasts the divisor across features.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBroadcastDivide<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Multiplies two tensors element-wise with NumPy-style broadcasting.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor (will be broadcast to match a if needed).</param>
    /// <returns>A new tensor containing the element-wise product with broadcasting.</returns>
    /// <remarks>
    /// <para>
    /// Broadcasting allows tensors of different shapes to be multiplied together by automatically
    /// expanding the smaller tensor. For example, [B,H,W,C] * [B,1,1,C] broadcasts the [B,1,1,C]
    /// tensor across the spatial dimensions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBroadcastMultiply<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Adds multiple tensors element-wise in a single optimized operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">The tensors to add together.</param>
    /// <returns>A new tensor containing the element-wise sum of all inputs.</returns>
    /// <exception cref="ArgumentException">Thrown when fewer than 2 tensors provided or shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// <para>
    /// Like PyTorch's torch.stack + torch.sum pattern, this avoids intermediate allocations
    /// by computing all additions in a single pass. Essential for residual networks and
    /// skip connections that combine multiple feature maps.
    /// </para>
    /// <para>
    /// Performance: O(n*elements) where n is number of tensors, but with single output allocation
    /// instead of n-1 intermediate allocations from chained binary additions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorAddMany<T>(params Tensor<T>[] tensors);

    /// <summary>
    /// Subtracts tensor b from tensor a element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise difference.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// </remarks>
    Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Multiplies two tensors element-wise (Hadamard product).
    /// When shapes match exactly, performs direct element-wise multiplication.
    /// When shapes differ but are broadcast-compatible (NumPy/PyTorch rules),
    /// broadcasts the smaller tensor to match.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A new tensor containing the element-wise product.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes are not broadcast-compatible.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// </remarks>
    Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Multiplies multiple tensors element-wise in a single optimized operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">The tensors to multiply together.</param>
    /// <returns>A new tensor containing the element-wise product of all inputs.</returns>
    /// <exception cref="ArgumentException">Thrown when fewer than 2 tensors provided or shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// <para>
    /// Like PyTorch's torch.stack + torch.prod pattern, this avoids intermediate allocations
    /// by computing all multiplications in a single pass. Useful for gating mechanisms
    /// and attention computations that combine multiple masks or weights.
    /// </para>
    /// <para>
    /// Performance: O(n*elements) where n is number of tensors, but with single output allocation
    /// instead of n-1 intermediate allocations from chained binary multiplications.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMultiplyMany<T>(params Tensor<T>[] tensors);

    /// <summary>
    /// Multiplies a tensor by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor to multiply.</param>
    /// <param name="scalar">The scalar value.</param>
    /// <returns>A new tensor with all elements multiplied by the scalar.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// </remarks>
    Tensor<T> TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar);

    /// <summary>
    /// Divides tensor a by tensor b element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The numerator tensor.</param>
    /// <param name="b">The denominator tensor.</param>
    /// <returns>A new tensor containing the element-wise quotient.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <exception cref="DivideByZeroException">Thrown when any element of b is zero.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Element-Wise Operations</b></para>
    /// </remarks>
    Tensor<T> TensorDivide<T>(Tensor<T> a, Tensor<T> b);

    #region Tensor Comparison Operations

    /// <summary>
    /// Compares each element of a tensor to a scalar value for equality.
    /// Returns a tensor where each element is 1 if equal, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor of the same shape with 1 where equal, 0 where not equal.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// <para>
    /// Like PyTorch's torch.eq(), this enables vectorized comparison operations.
    /// Essential for masking operations in neural networks.
    /// </para>
    /// </remarks>
    Tensor<T> TensorEquals<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Compares two tensors element-wise for equality.
    /// Returns a tensor where each element is 1 if equal, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor of the same shape with 1 where equal, 0 where not equal.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorEquals<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Compares each element of a tensor to a scalar value for inequality.
    /// Returns a tensor where each element is 1 if not equal, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor of the same shape with 1 where not equal, 0 where equal.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// <para>
    /// Like PyTorch's torch.ne(), this enables vectorized inequality comparison.
    /// Essential for masking layers where we need to identify non-padding values.
    /// </para>
    /// </remarks>
    Tensor<T> TensorNotEquals<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Compares two tensors element-wise for inequality.
    /// Returns a tensor where each element is 1 if not equal, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor of the same shape with 1 where not equal, 0 where equal.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorNotEquals<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Compares each element of tensor a to corresponding element of tensor b for greater than.
    /// Returns a tensor where each element is 1 if a > b, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor of the same shape with 1 where a > b, 0 otherwise.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorGreaterThan<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Compares each element of a tensor to a scalar value for greater than.
    /// Returns a tensor where each element is 1 if element > value, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor of the same shape with 1 where greater, 0 otherwise.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorGreaterThan<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Compares each element of tensor a to corresponding element of tensor b for less than.
    /// Returns a tensor where each element is 1 if a &lt; b, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor of the same shape with 1 where a &lt; b, 0 otherwise.</returns>
    /// <exception cref="ArgumentException">Thrown when tensor shapes don't match.</exception>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorLessThan<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Compares each element of a tensor to a scalar value for less than.
    /// Returns a tensor where each element is 1 if element &lt; value, 0 otherwise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor of the same shape with 1 where less, 0 otherwise.</returns>
    /// <remarks>
    /// <para><b>US-GPU-014: Tensor Comparison Operations</b></para>
    /// </remarks>
    Tensor<T> TensorLessThan<T>(Tensor<T> tensor, T value);

    #endregion

    #region Tensor Element-wise Math Operations

    /// <summary>
    /// Computes the element-wise natural logarithm of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with the natural logarithm of each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Computes log(x) for each element. Used in:
    /// - Cross-entropy loss calculation
    /// - Log-probability computations
    /// - Attention entropy regularization
    /// </para>
    /// </remarks>
    Tensor<T> TensorLog<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise exponential of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with exp(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in softmax computation, probability distributions, and exponential scaling.
    /// </para>
    /// </remarks>
    Tensor<T> TensorExp<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise square root of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with sqrt(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in normalization layers, RMSProp/Adam optimizers, and standard deviation calculations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSqrt<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise absolute value of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with abs(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in L1 regularization, MAE loss, and gradient clipping.
    /// </para>
    /// </remarks>
    Tensor<T> TensorAbs<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise negation of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with -x for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// </remarks>
    Tensor<T> TensorNegate<T>(Tensor<T> tensor);

    /// <summary>
    /// Returns a tensor with the same values but severed from the gradient tape.
    /// No backward function is recorded, so gradients do not flow through this operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor to detach from the computation graph.</param>
    /// <returns>A new tensor with identical values but no tape recording.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When training GANs, you sometimes need to pass data through
    /// one network (like the discriminator) without updating its weights. StopGradient creates
    /// a copy of the tensor that the gradient tape can't "see through" — gradients stop here.</para>
    ///
    /// <para>This is equivalent to PyTorch's <c>.detach()</c> or TensorFlow's <c>tf.stop_gradient()</c>.</para>
    ///
    /// <para><b>Example — GAN discriminator training:</b></para>
    /// <code>
    /// // Generator forward
    /// var fakeImages = generator.ForwardForTraining(noise);
    ///
    /// // Detach so discriminator training does NOT update generator weights
    /// var detachedFakes = engine.StopGradient(fakeImages);
    /// var discScore = discriminator.ForwardForTraining(detachedFakes);
    /// // Gradients flow to discriminator only, NOT back through generator
    /// </code>
    /// </remarks>
    Tensor<T> StopGradient<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes Intersection over Union (IoU) loss for bounding box regression.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predicted">Predicted boxes [N, 4] in (x1, y1, x2, y2) format.</param>
    /// <param name="target">Target boxes [N, 4] in (x1, y1, x2, y2) format.</param>
    /// <returns>IoU loss per box [N] where loss = 1 - IoU. Tape-differentiable.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> IoU measures how much two bounding boxes overlap.
    /// An IoU of 1 means perfect overlap, 0 means no overlap. The loss is 1 - IoU
    /// so that the optimizer minimizes it toward zero (perfect overlap).</para>
    /// <para>Reference: Standard IoU metric used in object detection (PASCAL VOC, COCO).</para>
    /// </remarks>
    Tensor<T> TensorIoULoss<T>(Tensor<T> predicted, Tensor<T> target);

    /// <summary>
    /// Computes Generalized Intersection over Union (GIoU) loss.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predicted">Predicted boxes [N, 4] in (x1, y1, x2, y2) format.</param>
    /// <param name="target">Target boxes [N, 4] in (x1, y1, x2, y2) format.</param>
    /// <returns>GIoU loss per box [N] where loss = 1 - GIoU. Tape-differentiable.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> GIoU improves on IoU by also penalizing predictions that
    /// are far away from the target, even when they don't overlap at all. Regular IoU gives
    /// zero gradient when boxes don't overlap, making learning impossible. GIoU fixes this.</para>
    /// <para>Reference: Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019.</para>
    /// </remarks>
    Tensor<T> TensorGIoULoss<T>(Tensor<T> predicted, Tensor<T> target);

    /// <summary>
    /// Computes Distance Intersection over Union (DIoU) loss.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predicted">Predicted boxes [N, 4] in (x1, y1, x2, y2) format.</param>
    /// <param name="target">Target boxes [N, 4] in (x1, y1, x2, y2) format.</param>
    /// <returns>DIoU loss per box [N] where loss = 1 - DIoU. Tape-differentiable.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> DIoU adds a penalty based on the distance between the
    /// centers of predicted and target boxes. This gives a stronger gradient signal than GIoU,
    /// especially when boxes overlap partially but centers are misaligned.</para>
    /// <para>Reference: Zheng et al., "Distance-IoU Loss", AAAI 2020.</para>
    /// </remarks>
    Tensor<T> TensorDIoULoss<T>(Tensor<T> predicted, Tensor<T> target);

    /// <summary>
    /// Computes Complete Intersection over Union (CIoU) loss.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predicted">Predicted boxes [N, 4] in (x1, y1, x2, y2) format.</param>
    /// <param name="target">Target boxes [N, 4] in (x1, y1, x2, y2) format.</param>
    /// <returns>CIoU loss per box [N] where loss = 1 - CIoU. Tape-differentiable.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> CIoU considers three factors simultaneously:
    /// overlap (IoU), center distance (DIoU penalty), and aspect ratio consistency.
    /// This makes it the most comprehensive box regression loss available.</para>
    /// <para>Reference: Zheng et al., "Distance-IoU Loss", AAAI 2020.</para>
    /// </remarks>
    Tensor<T> TensorCIoULoss<T>(Tensor<T> predicted, Tensor<T> target);

    /// <summary>
    /// Fused MatMul + Bias Add + ReLU forward pass.
    /// Records a single tape entry with fused backward instead of three separate entries.
    /// Saves 2 tape entries; GPU backends also save kernel launch overhead.
    /// </summary>
    Tensor<T> FusedLinearReLU<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias);

    /// <summary>Fused MatMul + Bias Add + Sigmoid forward pass.</summary>
    Tensor<T> FusedLinearSigmoid<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias);

    /// <summary>Fused MatMul + Bias Add + Tanh forward pass.</summary>
    Tensor<T> FusedLinearTanh<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias);

    /// <summary>Fused MatMul + Bias Add + GELU forward pass.</summary>
    Tensor<T> FusedLinearGELU<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias);

    /// <summary>Fused MatMul + Bias Add + Swish/SiLU forward pass.</summary>
    Tensor<T> FusedLinearSwish<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias);

    /// <summary>
    /// Computes the element-wise power of a tensor raised to a scalar exponent.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor (base values).</param>
    /// <param name="exponent">The scalar exponent to raise each element to.</param>
    /// <returns>A tensor with pow(x, exponent) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in attention sharpening (Neural Turing Machines), gamma correction,
    /// polynomial features, and various normalization operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorPower<T>(Tensor<T> tensor, T exponent);

    /// <summary>
    /// Computes the element-wise power of a tensor raised to another tensor (element-wise exponents).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="bases">The input tensor of base values.</param>
    /// <param name="exponents">The tensor of exponents (must have same shape as bases).</param>
    /// <returns>A tensor with pow(bases[i], exponents[i]) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used for element-wise power operations where exponents vary per element.
    /// </para>
    /// </remarks>
    Tensor<T> TensorPower<T>(Tensor<T> bases, Tensor<T> exponents);

    /// <summary>
    /// Computes the element-wise floor of a tensor (largest integer less than or equal to each element).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with floor(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in hash encoding for 3D AI (NeRF, Gaussian Splatting), grid-based calculations,
    /// and index computation. Essential for converting continuous coordinates to discrete grid indices.
    /// </para>
    /// </remarks>
    Tensor<T> TensorFloor<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise ceiling of a tensor (smallest integer greater than or equal to each element).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with ceiling(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in ceil mode pooling, index calculations, and grid-based spatial computations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorCeiling<T>(Tensor<T> tensor);

    /// <summary>Element-wise round to nearest integer using MidpointRounding.ToEven (banker's rounding). GPU backends use hardware rounding which matches this mode.</summary>
    Tensor<T> TensorRound<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise fractional part of a tensor (x - floor(x)).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>A tensor with frac(x) = x - floor(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Essential for hash encoding in neural radiance fields (NeRF) and Instant-NGP.
    /// The fractional part is used to interpolate between discrete grid corners for
    /// smooth, differentiable spatial encoding. Also used in periodic functions and texture mapping.
    /// </para>
    /// </remarks>
    Tensor<T> TensorFrac<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise sine of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor with angles in radians.</param>
    /// <returns>A tensor with sin(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Essential for positional encoding in transformers and neural radiance fields.
    /// Positional encoding uses sin(position * frequency) to create smooth,
    /// periodic spatial features that help models understand relative positions.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSin<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the element-wise cosine of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor with angles in radians.</param>
    /// <returns>A tensor with cos(x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Essential for positional encoding in transformers and neural radiance fields.
    /// Positional encoding uses cos(position * frequency) alongside sin to create
    /// unique, differentiable position representations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorCos<T>(Tensor<T> tensor);

    /// <summary>
    /// Performs trilinear interpolation on a 3D grid.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="grid">The 3D feature grid of shape [D, H, W, C] where D=depth, H=height, W=width, C=channels.</param>
    /// <param name="positions">The 3D coordinates to sample at, shape [N, 3] where each row is (z, y, x) in range [0, D-1], [0, H-1], [0, W-1].</param>
    /// <returns>Interpolated values of shape [N, C].</returns>
    /// <remarks>
    /// <para><b>US-GPU-017: 3D Spatial Operations</b></para>
    /// <para>
    /// Essential for hash encoding in neural radiance fields (NeRF) and Instant-NGP.
    /// Trilinear interpolation samples from a discrete 3D grid using continuous coordinates,
    /// computing weighted averages of the 8 surrounding voxel corners. The fractional part
    /// of each coordinate determines the interpolation weights.
    /// 
    /// Formula for weights at position (z, y, x):
    /// - fz, fy, fx = fractional parts of z, y, x
    /// - Corners weighted by (1-fz)*(1-fy)*(1-fx), fz*(1-fy)*(1-fx), etc.
    /// </para>
    /// </remarks>
    Tensor<T> TensorTrilinearInterpolate<T>(Tensor<T> grid, Tensor<T> positions);

    /// <summary>
    /// Computes the backward pass for trilinear interpolation, returning gradients for the grid.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the upstream layer [numPositions, channels].</param>
    /// <param name="grid">The original 3D grid [depth, height, width, channels].</param>
    /// <param name="positions">The positions at which interpolation was performed [numPositions, 3].</param>
    /// <returns>The gradient with respect to the grid [depth, height, width, channels].</returns>
    /// <remarks>
    /// <para><b>US-GPU-017: 3D Spatial Operations</b></para>
    /// <para>
    /// Essential for training neural radiance fields (NeRF) and Instant-NGP with backpropagation.
    /// The backward pass scatters gradients to the 8 surrounding voxel corners using the same
    /// trilinear interpolation weights computed during the forward pass.
    /// </para>
    /// </remarks>
    Tensor<T> TensorTrilinearInterpolateBackward<T>(Tensor<T> gradOutput, Tensor<T> grid, Tensor<T> positions);

    /// <summary>
    /// Computes the element-wise power of a tensor raised to a scalar exponent.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor (base).</param>
    /// <param name="exponent">The scalar exponent.</param>
    /// <returns>A tensor with pow(x, exponent) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in polynomial features, learning rate scheduling, and custom activations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorPow<T>(Tensor<T> tensor, T exponent);

    /// <summary>
    /// Computes the element-wise maximum of two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor with max(a[i], b[i]) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in ReLU activation, gradient clipping, and element-wise maximum operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMax<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Computes the element-wise maximum of a tensor and a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor with max(x, value) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in ReLU activation (max(0, x)), clamping lower bounds, and preventing log(0).
    /// </para>
    /// </remarks>
    Tensor<T> TensorMax<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Computes the element-wise minimum of two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    /// <returns>A tensor with min(a[i], b[i]) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// </remarks>
    Tensor<T> TensorMin<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Computes the element-wise minimum of a tensor and a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="value">The scalar value to compare against.</param>
    /// <returns>A tensor with min(x, value) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in clamping upper bounds and gradient clipping.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMin<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Clamps tensor values to a specified range.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="min">The minimum value (lower bound).</param>
    /// <param name="max">The maximum value (upper bound).</param>
    /// <returns>A tensor with values clamped to [min, max].</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Equivalent to min(max(x, min), max). Used for gradient clipping and value normalization.
    /// </para>
    /// </remarks>
    Tensor<T> TensorClamp<T>(Tensor<T> tensor, T min, T max);

    /// <summary>
    /// Computes the sum of all elements in a tensor (full reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar sum of all elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Performs full reduction to a scalar. For axis-wise reduction, use ReduceSum.
    /// </para>
    /// </remarks>
    T TensorSum<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the sum along specified axes (axis reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axes">The axes along which to sum. Null or empty for full reduction.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <returns>The reduced tensor.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Used in batch/layer normalization, attention weight computation, and loss calculation.
    /// </para>
    /// </remarks>
    Tensor<T> ReduceSum<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false);

    /// <summary>
    /// Computes the maximum value of all elements in a tensor (full reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar maximum of all elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Performs full reduction to find the maximum value. Used in:
    /// - Attention weight analysis (max weight indicates peakiness)
    /// - Gradient clipping (finding max gradient magnitude)
    /// - Numerical stability (finding scale factors)
    /// - Normalization (max-normalization)
    /// </para>
    /// </remarks>
    T TensorMaxValue<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the minimum value of all elements in a tensor (full reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar minimum of all elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Performs full reduction to find the minimum value. Used in:
    /// - Range normalization (min-max scaling)
    /// - Gradient analysis
    /// - Numerical stability checks
    /// </para>
    /// </remarks>
    T TensorMinValue<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the mean (average) of all elements in a tensor (full reduction).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar mean of all elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Performs full reduction to compute mean. Used in:
    /// - Layer normalization (computing mean for centering)
    /// - Batch statistics
    /// - Loss averaging
    /// </para>
    /// </remarks>
    T TensorMean<T>(Tensor<T> tensor);

    #endregion

    /// <summary>
    /// Performs 2D max pooling on a 4D tensor (batch, channels, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The size of the pooling window (e.g., 2 for 2x2 pooling).</param>
    /// <param name="stride">The stride of the pooling window. If 0, defaults to poolSize.</param>
    /// <param name="padding">The amount of zero-padding to add to the input.</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input is not a 4D tensor.</exception>
    /// <remarks>
    /// <para><b>US-GPU-012: MaxPool2D</b></para>
    /// <para>
    /// Max pooling downsamples the spatial dimensions by taking the maximum value
    /// in each pooling window. Commonly used in CNNs for:
    /// - Reducing spatial dimensions
    /// - Providing translation invariance
    /// - Reducing computation in deeper layers
    /// </para>
    /// <para>
    /// Output dimensions:
    /// output_height = floor((height + 2*padding - poolSize) / stride) + 1
    /// output_width = floor((width + 2*padding - poolSize) / stride) + 1
    /// </para>
    /// <para>
    /// GPU acceleration provides 20-100x speedup for large feature maps.
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Performs 2D average pooling on a 4D tensor (batch, channels, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The size of the pooling window (e.g., 2 for 2x2 pooling).</param>
    /// <param name="stride">The stride of the pooling window. If 0, defaults to poolSize.</param>
    /// <param name="padding">The amount of zero-padding to add to the input.</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input is not a 4D tensor.</exception>
    /// <remarks>
    /// <para><b>US-GPU-012: AvgPool2D</b></para>
    /// <para>
    /// Average pooling downsamples the spatial dimensions by taking the average value
    /// in each pooling window. Often used as an alternative to max pooling for:
    /// - Smoother downsampling
    /// - Preserving more spatial information
    /// - Global average pooling before final classification layer
    /// </para>
    /// <para>
    /// GPU acceleration provides 20-100x speedup for large feature maps.
    /// </para>
    /// </remarks>
    Tensor<T> AvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Performs 2D convolution on a 4D input tensor using a 4D kernel.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride of the convolution. Defaults to 1.</param>
    /// <param name="padding">The amount of zero-padding to add to the input. Defaults to 0.</param>
    /// <param name="dilation">The spacing between kernel elements. Defaults to 1.</param>
    /// <returns>The convolved tensor [batch, out_channels, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input or kernel dimensions are invalid.</exception>
    /// <remarks>
    /// <para><b>US-GPU-011: Conv2D</b></para>
    /// <para>
    /// 2D convolution is the core operation in convolutional neural networks (CNNs).
    /// It applies learned filters to detect features like edges, textures, and patterns.
    /// Critical for:
    /// - Image classification (ResNet, VGG, etc.)
    /// - Object detection (YOLO, Faster R-CNN)
    /// - Semantic segmentation (U-Net, DeepLab)
    /// - Style transfer and image generation
    /// </para>
    /// <para>
    /// Output dimensions:
    /// output_height = floor((height + 2*padding - dilation*(kernel_height-1) - 1) / stride) + 1
    /// output_width = floor((width + 2*padding - dilation*(kernel_width-1) - 1) / stride) + 1
    /// </para>
    /// <para>
    /// GPU acceleration provides 50-500x speedup for typical CNN layers.
    /// This is the most computationally expensive operation in deep learning.
    /// </para>
    /// </remarks>
    Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1);

    /// <summary>
    /// Performs 1D convolution on a 3D tensor (batch, channels, length).
    /// Implemented by reshaping to Conv2D with height=1.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, length].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_length].</param>
    /// <param name="stride">The stride of the convolution.</param>
    /// <param name="padding">The amount of zero-padding to add.</param>
    /// <param name="dilation">The dilation spacing between kernel elements.</param>
    /// <returns>The convolved tensor [batch, out_channels, output_length].</returns>
    Tensor<T> Conv1D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1);

    /// <summary>
    /// Computes gradient w.r.t. input for 1D convolution backward pass.
    /// </summary>
    Tensor<T> Conv1DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int stride, int padding, int dilation);

    /// <summary>
    /// Computes gradient w.r.t. kernel for 1D convolution backward pass.
    /// </summary>
    Tensor<T> Conv1DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int stride, int padding, int dilation);

    /// <summary>
    /// Performs 2D convolution with asymmetric stride, padding, and dilation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideH, strideW] of the convolution.</param>
    /// <param name="padding">The padding [padH, padW] to add to the input.</param>
    /// <param name="dilation">The dilation [dilationH, dilationW] spacing between kernel elements.</param>
    /// <returns>The convolved tensor [batch, out_channels, output_height, output_width].</returns>
    Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Performs 2D convolution into a pre-allocated output tensor (zero-allocation forward pass).
    /// OVERWRITES the output tensor completely — previous contents are discarded.
    /// The output tensor must have correct shape [batch, out_channels, output_height, output_width].
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="output">Pre-allocated output tensor with correct shape.</param>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride of the convolution.</param>
    /// <param name="padding">The padding to add to the input.</param>
    /// <param name="dilation">The dilation spacing between kernel elements.</param>
    void Conv2DInto<T>(Tensor<T> output, Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1);

    /// <summary>
    /// Computes the gradient of Conv2D with respect to the input tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="kernel">The convolution kernel used in forward pass.</param>
    /// <param name="inputShape">The shape of the original input tensor.</param>
    /// <param name="stride">The stride [strideH, strideW] used in forward pass.</param>
    /// <param name="padding">The padding [padH, padW] used in forward pass.</param>
    /// <param name="dilation">The dilation [dilationH, dilationW] used in forward pass.</param>
    /// <returns>The gradient with respect to the input tensor.</returns>
    Tensor<T> Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Computes the gradient of Conv2D with respect to the kernel (weights).
    /// </summary>
    Tensor<T> Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Performs a Locally Connected 2D convolution.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="weights">The weights tensor [out_h, out_w, out_channels, in_channels, kernel_h, kernel_w].</param>
    /// <param name="bias">Optional bias tensor [out_channels].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <returns>The convolved tensor [batch, out_channels, output_height, output_width].</returns>
    Tensor<T> LocallyConnectedConv2D<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, int[] stride);

    /// <summary>
    /// Computes the gradient of LocallyConnectedConv2D with respect to the input tensor.
    /// </summary>
    Tensor<T> LocallyConnectedConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> weights, int[] inputShape, int[] stride);

    /// <summary>
    /// Computes the gradient of LocallyConnectedConv2D with respect to the weights.
    /// </summary>
    Tensor<T> LocallyConnectedConv2DBackwardWeights<T>(Tensor<T> gradOutput, Tensor<T> input, int[] weightsShape, int[] stride);

    /// <summary>
    /// Computes the gradient of LocallyConnectedConv2D with respect to the bias.
    /// </summary>
    Tensor<T> LocallyConnectedConv2DBackwardBias<T>(Tensor<T> gradOutput);

    /// <summary>
    /// Transposes a 2D tensor (matrix represented as tensor).
    /// </summary>
    Tensor<T> TensorTranspose<T>(Tensor<T> tensor);

    /// <summary>
    /// Performs matrix multiplication supporting tensors of any rank (PyTorch-style batched matmul).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor. Can be 2D [M, N] or higher rank [..., M, N].</param>
    /// <param name="b">The second tensor. Can be 2D [N, P] or higher rank [..., N, P].</param>
    /// <returns>The result tensor with appropriately broadcasted batch dimensions.</returns>
    /// <remarks>
    /// <para>
    /// This follows PyTorch's torch.matmul semantics for batched matrix multiplication:
    /// </para>
    /// <para><b>Supported combinations:</b></para>
    /// <list type="bullet">
    ///   <item>2D x 2D: [M, N] @ [N, P] = [M, P] (standard matrix multiplication)</item>
    ///   <item>3D x 2D: [B, M, N] @ [N, P] = [B, M, P] (batch matmul, weights broadcasted)</item>
    ///   <item>ND x 2D: [..., M, N] @ [N, P] = [..., M, P] (any batch dims, weights broadcasted)</item>
    ///   <item>3D x 3D: [B, M, N] @ [B, N, P] = [B, M, P] (batched matrix multiplication)</item>
    /// </list>
    /// <para><b>For Transformers:</b></para>
    /// <para>
    /// Input [batch, seq, features] @ weights [features, output] = [batch, seq, output]
    /// </para>
    /// </remarks>
    Tensor<T> TensorMatMul<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Performs 2D max pooling with asymmetric pool size and stride, returning max indices for backpropagation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The pool size [poolH, poolW].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <param name="maxIndices">Output: indices of max elements for backpropagation.</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices);

    /// <summary>
    /// Computes the gradient of MaxPool2D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the output.</param>
    /// <param name="maxIndices">The max indices from forward pass.</param>
    /// <param name="inputShape">The shape of the original input.</param>
    /// <param name="poolSize">The pool size used in forward pass.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride);

    /// <summary>
    /// Performs 2D average pooling with asymmetric pool size and stride.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">The pool size [poolH, poolW].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <returns>The pooled tensor [batch, channels, output_height, output_width].</returns>
    Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride);

    /// <summary>
    /// Computes the gradient of AvgPool2D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the output.</param>
    /// <param name="inputShape">The shape of the original input.</param>
    /// <param name="poolSize">The pool size used in forward pass.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> AvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride);

    /// <summary>
    /// Performs depthwise 2D convolution where each input channel is convolved independently.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The kernel tensor [in_channels, multiplier, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <param name="padding">The padding [padH, padW].</param>
    /// <returns>The convolved tensor [batch, in_channels * multiplier, output_height, output_width].</returns>
    Tensor<T> DepthwiseConv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of DepthwiseConv2D with respect to the input.
    /// </summary>
    Tensor<T> DepthwiseConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of DepthwiseConv2D with respect to the kernel.
    /// </summary>
    Tensor<T> DepthwiseConv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding);

    /// <summary>
    /// Performs 2D transposed convolution (deconvolution) for upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The kernel tensor [in_channels, out_channels, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideH, strideW].</param>
    /// <param name="padding">The padding [padH, padW].</param>
    /// <param name="outputPadding">Output padding for size adjustment [outPadH, outPadW].</param>
    /// <returns>The upsampled tensor.</returns>
    Tensor<T> ConvTranspose2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding);

    /// <summary>
    /// Computes the gradient of ConvTranspose2D with respect to the input.
    /// </summary>
    Tensor<T> ConvTranspose2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of ConvTranspose2D with respect to the kernel.
    /// </summary>
    Tensor<T> ConvTranspose2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding);

    /// <summary>
    /// Performs deformable 2D convolution (DCNv2) with learned spatial offsets and modulation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="offset">
    /// The learned offset tensor [batch, 2*kernel_h*kernel_w*deformGroups, out_h, out_w].
    /// Each kernel position has (dy, dx) offsets for bilinear sampling. When using multiple
    /// deformable groups, the channel dimension is multiplied by deformGroups.
    /// </param>
    /// <param name="mask">
    /// Optional modulation mask [batch, kernel_h*kernel_w*deformGroups, out_h, out_w].
    /// Values in [0,1] modulate each kernel position. deformGroups is inferred from the
    /// offset/mask channel counts. Null uses no modulation (DCNv1).
    /// </param>
    /// <param name="stride">The stride [strideH, strideW] of the convolution.</param>
    /// <param name="padding">The padding [padH, padW] to add to the input.</param>
    /// <param name="dilation">The dilation [dilationH, dilationW] spacing between kernel elements.</param>
    /// <returns>The convolved tensor [batch, out_channels, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when tensor dimensions are invalid.</exception>
    /// <remarks>
    /// <para><b>US-GPU-DCN: Deformable Convolution v2</b></para>
    /// <para>
    /// Deformable convolution learns spatial deformations to adapt the sampling grid.
    /// Unlike standard convolution which samples on a fixed grid, deformable convolution
    /// adds learned 2D offsets to each sampling position, enabling the network to:
    /// - Focus on relevant parts of objects regardless of their shape
    /// - Handle geometric transformations (rotation, scale, deformation)
    /// - Adapt receptive fields to object structure
    /// </para>
    /// <para><b>DCNv2 Features (this implementation):</b></para>
    /// <list type="bullet">
    ///   <item>Per-sample offsets learned via backpropagation</item>
    ///   <item>Modulation mask to weight each sampling position (attention mechanism)</item>
    ///   <item>Bilinear interpolation for sub-pixel sampling</item>
    /// </list>
    /// <para><b>Applications:</b></para>
    /// <list type="bullet">
    ///   <item>Object detection (DETR, deformable attention)</item>
    ///   <item>Video super-resolution (aligning features across frames)</item>
    ///   <item>Optical flow estimation (SpyNet, PWC-Net)</item>
    ///   <item>Semantic segmentation with geometric adaptation</item>
    /// </list>
    /// <para>
    /// GPU acceleration provides 20-100x speedup due to parallel bilinear sampling.
    /// </para>
    /// </remarks>
    Tensor<T> DeformableConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of DeformableConv2D with respect to the input tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="kernel">The convolution kernel from forward pass.</param>
    /// <param name="offset">The offset tensor from forward pass.</param>
    /// <param name="mask">The modulation mask from forward pass (null for DCNv1).</param>
    /// <param name="inputShape">The shape of the original input tensor.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <param name="dilation">The dilation used in forward pass.</param>
    /// <returns>The gradient with respect to the input tensor.</returns>
    Tensor<T> DeformableConv2DBackwardInput<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] inputShape,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of DeformableConv2D with respect to the kernel (weights).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="offset">The offset tensor from forward pass.</param>
    /// <param name="mask">The modulation mask from forward pass (null for DCNv1).</param>
    /// <param name="kernelShape">The shape of the kernel tensor.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <param name="dilation">The dilation used in forward pass.</param>
    /// <returns>The gradient with respect to the kernel tensor.</returns>
    Tensor<T> DeformableConv2DBackwardKernel<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of DeformableConv2D with respect to the offset tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="kernel">The convolution kernel from forward pass.</param>
    /// <param name="offset">The offset tensor from forward pass.</param>
    /// <param name="mask">The modulation mask from forward pass (null for DCNv1).</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <param name="dilation">The dilation used in forward pass.</param>
    /// <returns>The gradient with respect to the offset tensor.</returns>
    /// <remarks>
    /// <para>
    /// This gradient enables learning of the spatial deformations.
    /// The gradient flows through the bilinear interpolation operation.
    /// </para>
    /// </remarks>
    Tensor<T> DeformableConv2DBackwardOffset<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of DeformableConv2D with respect to the modulation mask.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="kernel">The convolution kernel from forward pass.</param>
    /// <param name="offset">The offset tensor from forward pass.</param>
    /// <param name="mask">The modulation mask from forward pass.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <param name="dilation">The dilation used in forward pass.</param>
    /// <returns>The gradient with respect to the modulation mask tensor.</returns>
    /// <remarks>
    /// <para>
    /// This gradient enables learning of per-position attention weights (DCNv2).
    /// Returns zero tensor if mask was null in forward pass.
    /// </para>
    /// </remarks>
    Tensor<T> DeformableConv2DBackwardMask<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation);

    /// <summary>
    /// Computes the gradient of GridSample with respect to the input (NHWC format).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, outH, outW, channels].</param>
    /// <param name="grid">The sampling grid from forward pass [batch, outH, outW, 2].</param>
    /// <param name="inputShape">The shape of the original input [batch, height, width, channels].</param>
    /// <returns>The gradient with respect to the input tensor [batch, height, width, channels].</returns>
    Tensor<T> GridSampleBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> grid, int[] inputShape);

    /// <summary>
    /// Computes the gradient of GridSample with respect to the grid (NHWC format).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, outH, outW, channels].</param>
    /// <param name="input">The original input tensor [batch, height, width, channels].</param>
    /// <param name="grid">The sampling grid from forward pass [batch, outH, outW, 2].</param>
    /// <returns>The gradient with respect to the grid tensor [batch, outH, outW, 2].</returns>
    Tensor<T> GridSampleBackwardGrid<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> grid);

    #endregion

    #region 3D Convolution and Pooling Operations

    /// <summary>
    /// Performs 3D convolution on a 5D tensor for volumetric data processing.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, depth, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride of the convolution (applied equally to all spatial dimensions).</param>
    /// <param name="padding">The zero-padding to add to all sides (applied equally to all spatial dimensions).</param>
    /// <param name="dilation">The spacing between kernel elements (applied equally to all spatial dimensions).</param>
    /// <returns>The convolved tensor [batch, out_channels, output_depth, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input or kernel dimensions are invalid.</exception>
    /// <remarks>
    /// <para><b>US-GPU-030: Conv3D</b></para>
    /// <para>
    /// 3D convolution extends 2D convolution to volumetric data by applying learnable filters
    /// across depth, height, and width dimensions. This is essential for:
    /// - Voxel-based 3D object recognition (ModelNet, ShapeNet)
    /// - Medical imaging analysis (CT scans, MRI volumes)
    /// - Video understanding (treating time as the third spatial dimension)
    /// - Point cloud processing after voxelization
    /// </para>
    /// <para>
    /// Output dimensions:
    /// output_depth = floor((depth + 2*padding - dilation*(kernel_depth-1) - 1) / stride) + 1
    /// output_height = floor((height + 2*padding - dilation*(kernel_height-1) - 1) / stride) + 1
    /// output_width = floor((width + 2*padding - dilation*(kernel_width-1) - 1) / stride) + 1
    /// </para>
    /// <para>
    /// GPU acceleration provides 100-1000x speedup for typical 3D CNN layers due to the
    /// cubic growth in computation compared to 2D convolutions.
    /// </para>
    /// <para><b>For Beginners:</b> Think of 3D convolution as sliding a small 3D cube (the kernel)
    /// through a larger 3D volume (the input), computing dot products at each position.
    /// This allows the network to learn 3D patterns like surfaces, edges, and volumetric shapes.
    /// </para>
    /// </remarks>
    Tensor<T> Conv3D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1);

    /// <summary>
    /// Performs 3D convolution with asymmetric stride, padding, and dilation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, depth, height, width].</param>
    /// <param name="kernel">The convolution kernel [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW] of the convolution.</param>
    /// <param name="padding">The padding [padD, padH, padW] to add to the input.</param>
    /// <param name="dilation">The dilation [dilationD, dilationH, dilationW] spacing between kernel elements.</param>
    /// <returns>The convolved tensor [batch, out_channels, output_depth, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input or kernel dimensions are invalid or stride/padding/dilation arrays have incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This overload allows different stride, padding, and dilation values for each spatial dimension,
    /// providing more flexibility for architectures that need asymmetric operations.
    /// </para>
    /// </remarks>
    Tensor<T> Conv3D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Computes the gradient of Conv3D with respect to the input tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, out_channels, out_depth, out_height, out_width].</param>
    /// <param name="kernel">The convolution kernel used in forward pass [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="inputShape">The shape of the original input tensor [batch, in_channels, depth, height, width].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW] used in forward pass.</param>
    /// <param name="padding">The padding [padD, padH, padW] used in forward pass.</param>
    /// <param name="dilation">The dilation [dilationD, dilationH, dilationW] used in forward pass.</param>
    /// <returns>The gradient with respect to the input tensor [batch, in_channels, depth, height, width].</returns>
    /// <remarks>
    /// <para>
    /// This is used during backpropagation to compute how the loss changes with respect to the input.
    /// The operation is mathematically a transposed convolution (deconvolution) of the gradient
    /// with the kernel rotated 180 degrees in each spatial dimension.
    /// </para>
    /// </remarks>
    Tensor<T> Conv3DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Computes the gradient of Conv3D with respect to the kernel (weights).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output [batch, out_channels, out_depth, out_height, out_width].</param>
    /// <param name="input">The original input tensor from forward pass [batch, in_channels, depth, height, width].</param>
    /// <param name="kernelShape">The shape of the kernel [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW] used in forward pass.</param>
    /// <param name="padding">The padding [padD, padH, padW] used in forward pass.</param>
    /// <param name="dilation">The dilation [dilationD, dilationH, dilationW] used in forward pass.</param>
    /// <returns>The gradient with respect to the kernel [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].</returns>
    /// <remarks>
    /// <para>
    /// This is used during backpropagation to compute how the loss changes with respect to the weights.
    /// The operation is mathematically a convolution between the input and the output gradient.
    /// </para>
    /// </remarks>
    Tensor<T> Conv3DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation);

    /// <summary>
    /// Performs 3D max pooling on a 5D tensor (batch, channels, depth, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The size of the pooling window (applied equally to all spatial dimensions).</param>
    /// <param name="stride">The stride of the pooling window. If 0, defaults to poolSize.</param>
    /// <param name="padding">The amount of zero-padding to add to the input.</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input is not a 5D tensor.</exception>
    /// <remarks>
    /// <para><b>US-GPU-031: MaxPool3D</b></para>
    /// <para>
    /// Max pooling downsamples the spatial dimensions by taking the maximum value
    /// in each 3D pooling window. Commonly used in 3D CNNs for:
    /// - Reducing spatial dimensions of volumetric data
    /// - Providing translation invariance in 3D
    /// - Reducing computation in deeper layers
    /// </para>
    /// <para>
    /// Output dimensions:
    /// output_d = floor((depth + 2*padding - poolSize) / stride) + 1
    /// output_h = floor((height + 2*padding - poolSize) / stride) + 1
    /// output_w = floor((width + 2*padding - poolSize) / stride) + 1
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool3D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Performs 3D max pooling with asymmetric pool size, stride, and padding.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="padding">The padding [padD, padH, padW].</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    Tensor<T> MaxPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding);

    /// <summary>
    /// Performs 3D max pooling and returns the indices of the maximum values.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="maxIndices">Output parameter containing the 3D coordinates (d, h, w) of maximum values for each output position.</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    /// <remarks>
    /// <para>
    /// The maxIndices array stores the 3D coordinates (depth, height, width) within the input tensor
    /// where the maximum value was found for each output position. This is essential for the backward
    /// pass to route gradients correctly.
    /// Shape of maxIndices: [batch, channels, output_depth, output_height, output_width, 3]
    /// where the last dimension contains [max_d_index, max_h_index, max_w_index].
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool3DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,,] maxIndices);

    /// <summary>
    /// Computes the gradient of MaxPool3D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="maxIndices">The indices of maximum values from the forward pass.</param>
    /// <param name="inputShape">The shape of the original input tensor.</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// During backpropagation, gradients are routed only to the positions that had the maximum
    /// values in the forward pass (as indicated by maxIndices). All other positions receive zero gradient.
    /// </para>
    /// </remarks>
    Tensor<T> MaxPool3DBackward<T>(Tensor<T> gradOutput, int[,,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride);

    /// <summary>
    /// Performs 3D average pooling on a 5D tensor (batch, channels, depth, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The size of the pooling window (applied equally to all spatial dimensions).</param>
    /// <param name="stride">The stride of the pooling window. If 0, defaults to poolSize.</param>
    /// <param name="padding">The amount of zero-padding to add to the input.</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    /// <exception cref="ArgumentException">Thrown when input is not a 5D tensor.</exception>
    /// <remarks>
    /// <para><b>US-GPU-032: AvgPool3D</b></para>
    /// <para>
    /// Average pooling downsamples the spatial dimensions by computing the average value
    /// in each 3D pooling window. Compared to max pooling:
    /// - Smoother gradients during backpropagation
    /// - Better for preserving overall magnitude information
    /// - Often used in later layers or for global pooling
    /// </para>
    /// </remarks>
    Tensor<T> AvgPool3D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Performs 3D average pooling with asymmetric pool size, stride, and padding.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, channels, depth, height, width].</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="padding">The padding [padD, padH, padW].</param>
    /// <returns>The pooled tensor [batch, channels, output_depth, output_height, output_width].</returns>
    Tensor<T> AvgPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of AvgPool3D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient flowing back from the output.</param>
    /// <param name="inputShape">The shape of the original input tensor.</param>
    /// <param name="poolSize">The pooling window size [poolD, poolH, poolW].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="padding">The padding [padD, padH, padW].</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// During backpropagation, each gradient value from the output is divided equally among
    /// all the input positions that contributed to that output (i.e., divided by the pool volume).
    /// </para>
    /// </remarks>
    Tensor<T> AvgPool3DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride, int[] padding);

    /// <summary>
    /// Performs 3D nearest-neighbor upsampling on a 5D tensor (batch, channels, depth, height, width).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, depth, height, width].</param>
    /// <param name="scaleD">The depth scaling factor.</param>
    /// <param name="scaleH">The height scaling factor.</param>
    /// <param name="scaleW">The width scaling factor.</param>
    /// <returns>The upsampled tensor with shape [batch, channels, depth*scaleD, height*scaleH, width*scaleW].</returns>
    /// <remarks>
    /// <para><b>US-GPU-035: Upsample3D</b></para>
    /// <para>
    /// 3D upsampling increases the spatial dimensions of volumetric data by replicating values.
    /// This is essential for decoder paths in encoder-decoder architectures like 3D U-Net.
    /// </para>
    /// </remarks>
    Tensor<T> Upsample3D<T>(Tensor<T> input, int scaleD, int scaleH, int scaleW);

    /// <summary>
    /// Computes the backward pass for 3D upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer [batch, channels, out_depth, out_height, out_width].</param>
    /// <param name="inputShape">The original input shape [batch, channels, depth, height, width].</param>
    /// <param name="scaleD">The depth scaling factor used in forward pass.</param>
    /// <param name="scaleH">The height scaling factor used in forward pass.</param>
    /// <param name="scaleW">The width scaling factor used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// During backpropagation, gradients are accumulated from all output positions that were
    /// derived from each input position (i.e., summed over the scaling block).
    /// </para>
    /// </remarks>
    Tensor<T> Upsample3DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleD, int scaleH, int scaleW);

    /// <summary>
    /// Performs 3D transposed convolution (deconvolution) for learned upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor [batch, in_channels, depth, height, width].</param>
    /// <param name="kernel">The kernel tensor [in_channels, out_channels, kernel_depth, kernel_height, kernel_width].</param>
    /// <param name="stride">The stride [strideD, strideH, strideW].</param>
    /// <param name="padding">The padding [padD, padH, padW].</param>
    /// <param name="outputPadding">Output padding for size adjustment [outPadD, outPadH, outPadW].</param>
    /// <returns>The upsampled tensor.</returns>
    /// <remarks>
    /// <para><b>US-GPU-036: ConvTranspose3D</b></para>
    /// <para>
    /// Transposed 3D convolution learns upsampling filters, providing more flexibility than
    /// nearest-neighbor upsampling. Used in decoder paths of 3D U-Net and similar architectures.
    /// </para>
    /// </remarks>
    Tensor<T> ConvTranspose3D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding);

    /// <summary>
    /// Computes the gradient of ConvTranspose3D with respect to the input.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="kernel">The kernel tensor used in forward pass.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ConvTranspose3DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding);

    /// <summary>
    /// Computes the gradient of ConvTranspose3D with respect to the kernel.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor from forward pass.</param>
    /// <param name="kernelShape">The shape of the kernel.</param>
    /// <param name="stride">The stride used in forward pass.</param>
    /// <param name="padding">The padding used in forward pass.</param>
    /// <returns>The gradient with respect to the kernel.</returns>
    Tensor<T> ConvTranspose3DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding);

    #endregion

    #region Normalization and Activation Operations

    /// <summary>
    /// Applies softmax activation along the specified axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axis">The axis along which to apply softmax. Default is -1 (last axis).</param>
    /// <returns>A tensor where values along the axis sum to 1.</returns>
    Tensor<T> Softmax<T>(Tensor<T> input, int axis = -1);

    /// <summary>
    /// Computes the backward pass for softmax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="output">The output from the forward softmax pass.</param>
    /// <param name="axis">The axis along which softmax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> SoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1);

    /// <summary>
    /// Applies Gumbel-Softmax activation to produce differentiable categorical samples.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor of logits.</param>
    /// <param name="temperature">Temperature parameter controlling the softness. Must be positive.</param>
    /// <param name="hard">If true, uses straight-through estimator for discrete outputs.</param>
    /// <param name="axis">The axis along which to apply Gumbel-Softmax. Default is -1 (last axis).</param>
    /// <returns>A tensor with Gumbel-Softmax applied.</returns>
    /// <remarks>
    /// <para>
    /// Gumbel-Softmax provides a differentiable approximation to categorical sampling.
    /// As temperature approaches 0, outputs approach one-hot categorical samples.
    /// When hard=true, uses straight-through estimator for discrete outputs with gradient pass-through.
    /// </para>
    /// </remarks>
    Tensor<T> GumbelSoftmax<T>(Tensor<T> input, double temperature = 1.0, bool hard = false, int axis = -1);

    /// <summary>
    /// Computes the backward pass for Gumbel-Softmax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="output">The output from the forward Gumbel-Softmax pass.</param>
    /// <param name="temperature">Temperature parameter used in forward pass.</param>
    /// <param name="axis">The axis along which Gumbel-Softmax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> GumbelSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, double temperature, int axis = -1);

    /// <summary>
    /// Applies Taylor-Softmax activation using polynomial approximation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="order">The order of Taylor expansion. Default is 2.</param>
    /// <param name="axis">The axis along which to apply Taylor-Softmax. Default is -1 (last axis).</param>
    /// <returns>A tensor with Taylor-Softmax applied.</returns>
    /// <remarks>
    /// <para>
    /// TaylorSoftmax uses Taylor series approximation of exp(x):
    /// exp(x) Ã¢â€°Ë† 1 + x + xÃ‚Â²/2! + xÃ‚Â³/3! + ... + xÃ¢ÂÂ¿/n!
    /// Then normalizes like standard softmax.
    /// More computationally efficient than standard softmax for some hardware.
    /// </para>
    /// </remarks>
    Tensor<T> TaylorSoftmax<T>(Tensor<T> input, int order = 2, int axis = -1);

    /// <summary>
    /// Computes the backward pass for Taylor-Softmax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="output">The output from the forward Taylor-Softmax pass.</param>
    /// <param name="order">The order of Taylor expansion used in forward pass.</param>
    /// <param name="axis">The axis along which Taylor-Softmax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> TaylorSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int order, int axis = -1);

    /// <summary>
    /// Applies Sparsemax activation to produce sparse probability distributions.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axis">The axis along which to apply Sparsemax. Default is -1 (last axis).</param>
    /// <returns>A tensor with Sparsemax applied.</returns>
    /// <remarks>
    /// <para>
    /// Sparsemax produces sparse probability distributions where some outputs are exactly zero.
    /// Unlike softmax which always gives positive probabilities to all classes, sparsemax
    /// can assign exactly zero to low-scoring classes.
    /// </para>
    /// </remarks>
    Tensor<T> Sparsemax<T>(Tensor<T> input, int axis = -1);

    /// <summary>
    /// Computes the backward pass for Sparsemax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="output">The output from the forward Sparsemax pass (used to determine support set).</param>
    /// <param name="axis">The axis along which Sparsemax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> SparsemaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1);

    /// <summary>
    /// Applies Spherical-Softmax activation (L2-normalized softmax).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axis">The axis along which to apply Spherical-Softmax. Default is -1 (last axis).</param>
    /// <returns>A tensor with Spherical-Softmax applied.</returns>
    /// <remarks>
    /// <para>
    /// SphericalSoftmax = softmax(x / ||x||Ã¢â€šâ€š)
    /// First L2-normalizes the input, then applies softmax.
    /// This improves numerical stability for inputs with varying magnitudes.
    /// </para>
    /// </remarks>
    Tensor<T> SphericalSoftmax<T>(Tensor<T> input, int axis = -1);

    /// <summary>
    /// Computes the backward pass for Spherical-Softmax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="output">The output from the forward Spherical-Softmax pass.</param>
    /// <param name="axis">The axis along which Spherical-Softmax was applied.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> SphericalSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis = -1);

    #endregion

    #region Attention Operations

    /// <summary>
    /// Computes scaled dot-product attention as defined in "Attention Is All You Need" (Vaswani et al., 2017).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="query">Query tensor with shape [batch, heads, seq_q, d_k].</param>
    /// <param name="key">Key tensor with shape [batch, heads, seq_k, d_k].</param>
    /// <param name="value">Value tensor with shape [batch, heads, seq_k, d_v].</param>
    /// <param name="mask">Optional attention mask with shape broadcastable to [batch, heads, seq_q, seq_k].
    /// True values indicate positions to attend to, false values are masked out (set to -infinity before softmax).</param>
    /// <param name="scale">Optional scaling factor. If null, uses 1/sqrt(d_k) as per the paper.</param>
    /// <param name="attentionWeights">Output: the attention weights after softmax with shape [batch, heads, seq_q, seq_k].
    /// Useful for visualization and backward pass.</param>
    /// <returns>The attention output with shape [batch, heads, seq_q, d_v].</returns>
    /// <remarks>
    /// <para>
    /// Implements the formula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    /// </para>
    /// <para><b>Usage in multi-head attention:</b></para>
    /// <list type="bullet">
    /// <item><description>Split input into heads: [batch, seq, embed] -> [batch, heads, seq, head_dim]</description></item>
    /// <item><description>Apply linear projections to get Q, K, V</description></item>
    /// <item><description>Call ScaledDotProductAttention</description></item>
    /// <item><description>Concatenate heads and apply output projection</description></item>
    /// </list>
    /// <para><b>Mask conventions:</b></para>
    /// <list type="bullet">
    /// <item><description>Causal mask: Lower triangular matrix (for autoregressive models)</description></item>
    /// <item><description>Padding mask: False for padded positions</description></item>
    /// <item><description>Combined: Element-wise AND of causal and padding masks</description></item>
    /// </list>
    /// </remarks>
    Tensor<T> ScaledDotProductAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<bool>? mask,
        double? scale,
        out Tensor<T> attentionWeights);

    /// <summary>
    /// Computes the backward pass for scaled dot-product attention.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer with shape [batch, heads, seq_q, d_v].</param>
    /// <param name="query">Original query tensor from forward pass.</param>
    /// <param name="key">Original key tensor from forward pass.</param>
    /// <param name="value">Original value tensor from forward pass.</param>
    /// <param name="attentionWeights">Attention weights from forward pass (after softmax).</param>
    /// <param name="scale">The scale factor used in forward pass.</param>
    /// <param name="gradQuery">Output: gradient with respect to query.</param>
    /// <param name="gradKey">Output: gradient with respect to key.</param>
    /// <param name="gradValue">Output: gradient with respect to value.</param>
    /// <returns>The gradient with respect to the output (same as gradOutput for chaining).</returns>
    /// <remarks>
    /// <para>
    /// Gradient computation follows the chain rule through:
    /// 1. V @ attention_weights^T for gradValue
    /// 2. Softmax backward for attention score gradients
    /// 3. Q @ grad_scores^T / scale for gradKey
    /// 4. grad_scores @ K / scale for gradQuery
    /// </para>
    /// </remarks>
    Tensor<T> ScaledDotProductAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> attentionWeights,
        double scale,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue);

    /// <summary>
    /// Computes memory-efficient attention using the FlashAttention algorithm.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="query">Query tensor with shape [batch, heads, seq_q, head_dim].</param>
    /// <param name="key">Key tensor with shape [batch, heads, seq_kv, head_dim].</param>
    /// <param name="value">Value tensor with shape [batch, heads, seq_kv, head_dim].</param>
    /// <param name="scale">Optional scale factor. If null, uses 1/sqrt(head_dim).</param>
    /// <param name="isCausal">If true, applies causal masking (lower triangular).</param>
    /// <param name="softmaxStats">Output: log-sum-exp statistics for backward pass [batch, heads, seq_q].</param>
    /// <param name="attentionBias">Optional additive attention bias with shape [batch, heads, seq_q, seq_kv] or [heads, seq_q, seq_kv].
    /// Added to attention scores after QK^T * scale and before softmax. Used for ALiBi, relative position biases, etc.
    /// When null, no bias is applied (backward compatible).</param>
    /// <returns>The attention output with shape [batch, heads, seq_q, head_dim].</returns>
    /// <remarks>
    /// <para><b>FlashAttention Algorithm (Dao et al., 2022):</b></para>
    /// <para>
    /// FlashAttention is a memory-efficient and IO-aware attention algorithm that:
    /// - Uses tiling to compute attention in SRAM without materializing the full attention matrix
    /// - Reduces memory usage from O(N²) to O(N) where N is sequence length
    /// - Provides significant speedup on long sequences by minimizing HBM (GPU memory) accesses
    /// </para>
    /// <para><b>Key Benefits:</b></para>
    /// <list type="bullet">
    /// <item><description>O(N) memory vs O(N²) for standard attention</description></item>
    /// <item><description>2-4x faster than standard attention for long sequences</description></item>
    /// <item><description>Enables training on much longer sequences (16K-64K tokens)</description></item>
    /// <item><description>Numerically equivalent to standard attention (uses online softmax)</description></item>
    /// </list>
    /// <para><b>Algorithm Overview:</b></para>
    /// <list type="number">
    /// <item><description>Tile Q, K, V into blocks that fit in SRAM</description></item>
    /// <item><description>For each Q block, iterate over K,V blocks computing partial attention</description></item>
    /// <item><description>Use online softmax to accumulate results without full materialization</description></item>
    /// <item><description>Store softmax statistics (log-sum-exp) for backward pass</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> Standard attention requires storing a huge attention matrix
    /// (N×N for sequence length N). For a 4096-token sequence, that's 16 million values per head!
    /// FlashAttention cleverly avoids this by computing attention in small chunks, dramatically
    /// reducing memory usage while still getting the exact same results.</para>
    /// </remarks>
    Tensor<T> FlashAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        double? scale,
        bool isCausal,
        out Tensor<T> softmaxStats,
        Tensor<T>? attentionBias = null);

    /// <summary>
    /// Computes the backward pass for FlashAttention.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer with shape [batch, heads, seq_q, head_dim].</param>
    /// <param name="query">Original query tensor from forward pass.</param>
    /// <param name="key">Original key tensor from forward pass.</param>
    /// <param name="value">Original value tensor from forward pass.</param>
    /// <param name="output">The output from the forward pass.</param>
    /// <param name="softmaxStats">The log-sum-exp statistics from forward pass.</param>
    /// <param name="scale">The scale factor used in forward pass.</param>
    /// <param name="isCausal">Whether causal masking was used in forward pass.</param>
    /// <param name="gradQuery">Output: gradient with respect to query.</param>
    /// <param name="gradKey">Output: gradient with respect to key.</param>
    /// <param name="gradValue">Output: gradient with respect to value.</param>
    /// <param name="attentionBias">Optional additive attention bias used in the forward pass.
    /// Must match the same bias passed to FlashAttention. The bias is used to reproduce the forward
    /// attention scores during recomputation; no gradient is produced for the bias tensor itself.</param>
    /// <returns>The gradient with respect to the output (same as gradOutput for chaining).</returns>
    /// <remarks>
    /// <para>
    /// The backward pass also uses tiling and recomputes attention weights from Q, K, and
    /// the stored softmax statistics, maintaining the O(N) memory complexity.
    /// </para>
    /// </remarks>
    Tensor<T> FlashAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> output,
        Tensor<T> softmaxStats,
        double scale,
        bool isCausal,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue,
        Tensor<T>? attentionBias = null);

    /// <summary>
    /// Computes Grouped Query Attention (GQA) for efficient inference.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="query">Query tensor with shape [batch, num_q_heads, seq, head_dim].</param>
    /// <param name="key">Key tensor with shape [batch, num_kv_heads, seq, head_dim].</param>
    /// <param name="value">Value tensor with shape [batch, num_kv_heads, seq, head_dim].</param>
    /// <param name="numQueriesPerKV">Number of query heads per key-value head (num_q_heads / num_kv_heads).</param>
    /// <param name="scale">Optional scale factor. If null, uses 1/sqrt(head_dim).</param>
    /// <param name="isCausal">If true, applies causal masking.</param>
    /// <param name="attentionWeights">Output: attention weights [batch, num_q_heads, seq_q, seq_kv].</param>
    /// <returns>The attention output with shape [batch, num_q_heads, seq, head_dim].</returns>
    /// <remarks>
    /// <para><b>Grouped Query Attention (Ainslie et al., 2023):</b></para>
    /// <para>
    /// GQA is a variant of multi-head attention that groups multiple query heads to share
    /// the same key-value head. This reduces memory bandwidth and KV-cache size during inference
    /// while maintaining most of the model quality.
    /// </para>
    /// <para><b>Key Benefits:</b></para>
    /// <list type="bullet">
    /// <item><description>Reduces KV-cache memory by num_q_heads/num_kv_heads factor</description></item>
    /// <item><description>Faster inference due to reduced memory bandwidth</description></item>
    /// <item><description>Interpolates between MHA (all unique) and MQA (all shared)</description></item>
    /// <item><description>Used in LLaMA 2, Mistral, and other modern LLMs</description></item>
    /// </list>
    /// <para><b>Example:</b> With 32 query heads and 8 KV heads, each KV head is shared by
    /// 4 query heads, reducing KV-cache size by 4x compared to standard MHA.</para>
    /// <para><b>For Beginners:</b> In standard attention, each query has its own key-value pair.
    /// GQA is more efficient because multiple queries share the same key-value pair. It's like
    /// having 32 students (queries) but only 8 teachers (key-value pairs) - each teacher handles
    /// 4 students, using less resources while still providing good education.</para>
    /// </remarks>
    Tensor<T> GroupedQueryAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        int numQueriesPerKV,
        double? scale,
        bool isCausal,
        out Tensor<T> attentionWeights);

    /// <summary>
    /// Computes the backward pass for Grouped Query Attention.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer with shape [batch, num_q_heads, seq, head_dim].</param>
    /// <param name="query">Original query tensor from forward pass.</param>
    /// <param name="key">Original key tensor from forward pass.</param>
    /// <param name="value">Original value tensor from forward pass.</param>
    /// <param name="attentionWeights">Attention weights from forward pass.</param>
    /// <param name="numQueriesPerKV">Number of query heads per key-value head.</param>
    /// <param name="scale">The scale factor used in forward pass.</param>
    /// <param name="gradQuery">Output: gradient with respect to query [batch, num_q_heads, seq, head_dim].</param>
    /// <param name="gradKey">Output: gradient with respect to key [batch, num_kv_heads, seq, head_dim].</param>
    /// <param name="gradValue">Output: gradient with respect to value [batch, num_kv_heads, seq, head_dim].</param>
    /// <returns>The gradient with respect to the output (same as gradOutput for chaining).</returns>
    /// <remarks>
    /// <para>
    /// The gradients for K and V are accumulated across all query heads that share them.
    /// This is the reverse of the forward pass where K and V are broadcast to multiple query heads.
    /// </para>
    /// </remarks>
    Tensor<T> GroupedQueryAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> attentionWeights,
        int numQueriesPerKV,
        double scale,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue);

    /// <summary>
    /// Computes Graph Attention Network (GAT) style attention over graph nodes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="nodeFeatures">Node feature tensor with shape [batch, nodes, features].</param>
    /// <param name="edgeSourceIndices">Source node indices for each edge with shape [num_edges].</param>
    /// <param name="edgeTargetIndices">Target node indices for each edge with shape [num_edges].</param>
    /// <param name="attentionWeightSource">Attention weight vector for source nodes [features].</param>
    /// <param name="attentionWeightTarget">Attention weight vector for target nodes [features].</param>
    /// <param name="leakyReluAlpha">Negative slope for LeakyReLU activation (typically 0.2).</param>
    /// <param name="attentionCoeffs">Output: attention coefficients for each edge [batch, num_edges].</param>
    /// <returns>The aggregated node features with shape [batch, nodes, features].</returns>
    /// <remarks>
    /// <para>
    /// Implements the GAT attention mechanism from Veličković et al.:
    /// α_ij = softmax_j(LeakyReLU(a_src^T @ h_i + a_tgt^T @ h_j))
    /// h'_i = Σ_j α_ij @ h_j
    /// </para>
    /// <para><b>Key differences from scaled dot-product attention:</b></para>
    /// <list type="bullet">
    /// <item><description>Uses additive attention with learnable vectors instead of dot-product</description></item>
    /// <item><description>Applies LeakyReLU before softmax for non-linearity</description></item>
    /// <item><description>Operates on graph structure defined by edge indices</description></item>
    /// <item><description>Softmax is computed per-node over its neighbors only</description></item>
    /// </list>
    /// </remarks>
    Tensor<T> GraphAttention<T>(
        Tensor<T> nodeFeatures,
        Tensor<int> edgeSourceIndices,
        Tensor<int> edgeTargetIndices,
        Tensor<T> attentionWeightSource,
        Tensor<T> attentionWeightTarget,
        double leakyReluAlpha,
        out Tensor<T> attentionCoeffs);

    /// <summary>
    /// Computes the backward pass for Graph Attention Network attention.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer with shape [batch, nodes, features].</param>
    /// <param name="nodeFeatures">Original node features from forward pass.</param>
    /// <param name="edgeSourceIndices">Source node indices for each edge.</param>
    /// <param name="edgeTargetIndices">Target node indices for each edge.</param>
    /// <param name="attentionWeightSource">Attention weight vector for source nodes.</param>
    /// <param name="attentionWeightTarget">Attention weight vector for target nodes.</param>
    /// <param name="attentionCoeffs">Attention coefficients from forward pass.</param>
    /// <param name="leakyReluAlpha">Negative slope for LeakyReLU.</param>
    /// <param name="gradNodeFeatures">Output: gradient with respect to node features.</param>
    /// <param name="gradAttentionWeightSource">Output: gradient with respect to source attention weights.</param>
    /// <param name="gradAttentionWeightTarget">Output: gradient with respect to target attention weights.</param>
    /// <returns>The gradient with respect to the output (same as gradOutput for chaining).</returns>
    Tensor<T> GraphAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> nodeFeatures,
        Tensor<int> edgeSourceIndices,
        Tensor<int> edgeTargetIndices,
        Tensor<T> attentionWeightSource,
        Tensor<T> attentionWeightTarget,
        Tensor<T> attentionCoeffs,
        double leakyReluAlpha,
        out Tensor<T> gradNodeFeatures,
        out Tensor<T> gradAttentionWeightSource,
        out Tensor<T> gradAttentionWeightTarget);

    /// <summary>
    /// Computes multi-head Graph Attention with concatenation or averaging of heads.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="nodeFeatures">Node feature tensor with shape [batch, nodes, features].</param>
    /// <param name="edgeSourceIndices">Source node indices for each edge.</param>
    /// <param name="edgeTargetIndices">Target node indices for each edge.</param>
    /// <param name="headWeights">Weight tensor for all heads [num_heads, features, head_dim].</param>
    /// <param name="attentionWeightsSource">Attention weights for source [num_heads, head_dim].</param>
    /// <param name="attentionWeightsTarget">Attention weights for target [num_heads, head_dim].</param>
    /// <param name="leakyReluAlpha">Negative slope for LeakyReLU.</param>
    /// <param name="concatenate">If true, concatenate heads; if false, average them.</param>
    /// <param name="attentionCoeffs">Output: attention coefficients [batch, num_heads, num_edges].</param>
    /// <returns>Output features [batch, nodes, num_heads * head_dim] if concat, else [batch, nodes, head_dim].</returns>
    Tensor<T> MultiHeadGraphAttention<T>(
        Tensor<T> nodeFeatures,
        Tensor<int> edgeSourceIndices,
        Tensor<int> edgeTargetIndices,
        Tensor<T> headWeights,
        Tensor<T> attentionWeightsSource,
        Tensor<T> attentionWeightsTarget,
        double leakyReluAlpha,
        bool concatenate,
        out Tensor<T> attentionCoeffs);

    #endregion

    #region FFT and Signal Processing

    /// <summary>
    /// Computes the 1D Fast Fourier Transform of a real-valued signal.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Real-valued input tensor with shape [..., n] where n should be a power of 2.</param>
    /// <returns>Complex output tensor with shape [..., n/2 + 1] containing positive frequency components.</returns>
    /// <remarks>
    /// <para><b>Real FFT (RFFT):</b></para>
    /// <para>
    /// For real-valued inputs, the FFT output has conjugate symmetry, meaning the negative frequency
    /// components are redundant. RFFT exploits this by only computing and returning the positive
    /// frequencies (0 to Nyquist), reducing computation and memory by approximately half.
    /// </para>
    /// <para><b>Output Format:</b></para>
    /// <para>
    /// Returns interleaved real/imaginary pairs: [re0, im0, re1, im1, ..., re(n/2), im(n/2)].
    /// The output length is (n/2 + 1) * 2 = n + 2 elements total.
    /// </para>
    /// <para><b>For Beginners:</b> The FFT converts a signal from the time domain (amplitude over time)
    /// to the frequency domain (amplitude at each frequency). This is essential for audio processing,
    /// where you might want to analyze which frequencies are present in a sound, remove noise at
    /// specific frequencies, or compress audio data.</para>
    /// <para><b>Example:</b> A 440 Hz sine wave (musical note A) will show a peak at the 440 Hz
    /// frequency bin in the FFT output.</para>
    /// </remarks>
    Tensor<T> RFFT<T>(Tensor<T> input);

    /// <summary>
    /// Computes the inverse 1D FFT, converting from frequency domain back to real-valued time domain.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Complex input tensor with shape [..., n/2 + 1] (interleaved real/imaginary).</param>
    /// <param name="outputLength">The desired output length (original signal length before RFFT).</param>
    /// <returns>Real-valued output tensor with shape [..., outputLength].</returns>
    /// <remarks>
    /// <para>
    /// Reconstructs the original real-valued signal from its frequency representation.
    /// The outputLength parameter is needed because the original length cannot always be
    /// determined from the RFFT output (odd vs even input lengths).
    /// </para>
    /// <para><b>For Beginners:</b> This reverses the FFT operation. If you modified frequencies
    /// (like removing noise or applying effects), IRFFT converts back to a playable audio signal.</para>
    /// </remarks>
    Tensor<T> IRFFT<T>(Tensor<T> input, int outputLength);

    /// <summary>
    /// Computes the 1D complex-to-complex Fast Fourier Transform.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="inputReal">Real part of input with shape [..., n].</param>
    /// <param name="inputImag">Imaginary part of input with shape [..., n].</param>
    /// <param name="outputReal">Output: Real part of FFT result with shape [..., n].</param>
    /// <param name="outputImag">Output: Imaginary part of FFT result with shape [..., n].</param>
    /// <remarks>
    /// <para>
    /// Full complex FFT that handles both real and imaginary input components.
    /// Unlike RFFT, this returns all n frequency bins (including negative frequencies).
    /// </para>
    /// <para><b>When to use:</b></para>
    /// <list type="bullet">
    /// <item><description>When input is already complex (e.g., after previous FFT operations)</description></item>
    /// <item><description>When you need both positive and negative frequency components</description></item>
    /// <item><description>For 2D FFT building blocks</description></item>
    /// </list>
    /// </remarks>
    void FFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag);

    /// <summary>
    /// Computes the inverse 1D complex-to-complex Fast Fourier Transform.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="inputReal">Real part of frequency domain input with shape [..., n].</param>
    /// <param name="inputImag">Imaginary part of frequency domain input with shape [..., n].</param>
    /// <param name="outputReal">Output: Real part of time domain result with shape [..., n].</param>
    /// <param name="outputImag">Output: Imaginary part of time domain result with shape [..., n].</param>
    /// <remarks>
    /// <para>
    /// Inverse of the complex FFT. Converts frequency domain representation back to time domain.
    /// Applies 1/n normalization to ensure round-trip FFT->IFFT recovers the original signal.
    /// </para>
    /// </remarks>
    void IFFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag);

    /// <summary>
    /// Computes the 2D Fast Fourier Transform for image and spectrogram processing.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="inputReal">Real part of input with shape [..., height, width].</param>
    /// <param name="inputImag">Imaginary part of input with shape [..., height, width].</param>
    /// <param name="outputReal">Output: Real part of 2D FFT result.</param>
    /// <param name="outputImag">Output: Imaginary part of 2D FFT result.</param>
    /// <remarks>
    /// <para>
    /// 2D FFT is computed as 1D FFT along each axis sequentially (separable property).
    /// Used extensively in image processing and for analyzing 2D patterns.
    /// </para>
    /// <para><b>Applications:</b></para>
    /// <list type="bullet">
    /// <item><description>Image filtering (blur, sharpen, edge detection)</description></item>
    /// <item><description>Image compression (JPEG uses DCT, related to FFT)</description></item>
    /// <item><description>Spectrogram analysis (time-frequency representations)</description></item>
    /// <item><description>Convolution via frequency domain multiplication</description></item>
    /// </list>
    /// </remarks>
    void FFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag);

    /// <summary>
    /// Computes the inverse 2D Fast Fourier Transform.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="inputReal">Real part of frequency domain input.</param>
    /// <param name="inputImag">Imaginary part of frequency domain input.</param>
    /// <param name="outputReal">Output: Real part of spatial domain result.</param>
    /// <param name="outputImag">Output: Imaginary part of spatial domain result.</param>
    void IFFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag);

    /// <summary>
    /// Computes the Short-Time Fourier Transform (STFT) for time-frequency analysis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Audio signal with shape [batch, samples] or [samples].</param>
    /// <param name="nFft">FFT size (window size). Must be power of 2. Default: 2048.</param>
    /// <param name="hopLength">Number of samples between frames. Default: nFft/4.</param>
    /// <param name="window">Window function tensor of length nFft (e.g., Hann window).</param>
    /// <param name="center">If true, pad signal by nFft/2 on each side. Default: true.</param>
    /// <param name="magnitudeOut">Output: Magnitude spectrogram [..., numFrames, nFft/2+1].</param>
    /// <param name="phaseOut">Output: Phase spectrogram [..., numFrames, nFft/2+1].</param>
    /// <remarks>
    /// <para><b>Short-Time Fourier Transform (STFT):</b></para>
    /// <para>
    /// STFT analyzes how the frequency content of a signal changes over time by computing
    /// the FFT on overlapping windows of the signal. The result is a 2D spectrogram showing
    /// frequency (vertical) vs time (horizontal).
    /// </para>
    /// <para><b>Parameters explained:</b></para>
    /// <list type="bullet">
    /// <item><description><b>nFft</b>: Larger = better frequency resolution, worse time resolution</description></item>
    /// <item><description><b>hopLength</b>: Smaller = more frames, smoother time evolution</description></item>
    /// <item><description><b>window</b>: Reduces spectral leakage (Hann window is standard for audio)</description></item>
    /// <item><description><b>center</b>: Ensures first/last frames are centered on signal edges</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> Think of STFT as taking "snapshots" of frequencies at different
    /// points in time. A piano note will show its fundamental frequency appearing when the key is
    /// pressed and fading as the note decays.</para>
    /// </remarks>
    void STFT<T>(
        Tensor<T> input,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        out Tensor<T> magnitudeOut,
        out Tensor<T> phaseOut);

    /// <summary>
    /// Computes the inverse Short-Time Fourier Transform to reconstruct audio from spectrogram.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="magnitude">Magnitude spectrogram [..., numFrames, numFreqs].</param>
    /// <param name="phase">Phase spectrogram [..., numFrames, numFreqs].</param>
    /// <param name="nFft">FFT size used in forward STFT.</param>
    /// <param name="hopLength">Hop length used in forward STFT.</param>
    /// <param name="window">Window function tensor (same as forward STFT).</param>
    /// <param name="center">Whether centering was used in forward STFT.</param>
    /// <param name="length">Optional: exact output length. If null, computed from spectrogram shape.</param>
    /// <returns>Reconstructed audio signal [..., samples].</returns>
    /// <remarks>
    /// <para>
    /// Reconstructs the time-domain signal from magnitude and phase spectrograms using
    /// overlap-add synthesis. For perfect reconstruction, use the same parameters as forward STFT.
    /// </para>
    /// <para><b>Note:</b> If only magnitude is available (no phase), use Griffin-Lim algorithm
    /// for iterative phase estimation.</para>
    /// </remarks>
    Tensor<T> ISTFT<T>(
        Tensor<T> magnitude,
        Tensor<T> phase,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        int? length = null);

    /// <summary>
    /// Computes the Mel spectrogram from an audio signal.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Audio signal with shape [batch, samples] or [samples].</param>
    /// <param name="sampleRate">Audio sample rate in Hz (e.g., 22050, 44100).</param>
    /// <param name="nFft">FFT size. Default: 2048.</param>
    /// <param name="hopLength">Hop length between frames. Default: 512.</param>
    /// <param name="nMels">Number of Mel frequency bands. Default: 128.</param>
    /// <param name="fMin">Minimum frequency in Hz. Default: 0.</param>
    /// <param name="fMax">Maximum frequency in Hz. Default: sampleRate/2 (Nyquist).</param>
    /// <param name="window">Window function tensor.</param>
    /// <param name="powerToDb">If true, convert to decibel scale. Default: true.</param>
    /// <returns>Mel spectrogram [..., numFrames, nMels].</returns>
    /// <remarks>
    /// <para><b>Mel Spectrogram:</b></para>
    /// <para>
    /// The Mel scale approximates human auditory perception, where we perceive pitch differences
    /// logarithmically (an octave sounds like the same "distance" regardless of absolute frequency).
    /// Mel spectrograms are the standard input representation for speech and music ML models.
    /// </para>
    /// <para><b>Common configurations:</b></para>
    /// <list type="bullet">
    /// <item><description><b>Speech:</b> nMels=80, nFft=1024, hopLength=256</description></item>
    /// <item><description><b>Music:</b> nMels=128, nFft=2048, hopLength=512</description></item>
    /// <item><description><b>Riffusion:</b> nMels=512, nFft=2048, hopLength=512</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> A Mel spectrogram is like a regular spectrogram but with frequency
    /// bands spaced according to how humans hear. Low frequencies get more resolution (we're sensitive
    /// to small pitch changes there), while high frequencies are grouped together.</para>
    /// </remarks>
    Tensor<T> MelSpectrogram<T>(
        Tensor<T> input,
        int sampleRate,
        int nFft,
        int hopLength,
        int nMels,
        T fMin,
        T fMax,
        Tensor<T> window,
        bool powerToDb = true);

    /// <summary>
    /// Reconstructs audio from a magnitude spectrogram using the Griffin-Lim algorithm.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="magnitude">Magnitude spectrogram [..., numFrames, numFreqs].</param>
    /// <param name="nFft">FFT size.</param>
    /// <param name="hopLength">Hop length between frames.</param>
    /// <param name="window">Window function tensor.</param>
    /// <param name="iterations">Number of Griffin-Lim iterations. Default: 60.</param>
    /// <param name="momentum">Momentum for faster convergence (0-0.99). Default: 0.99.</param>
    /// <param name="length">Optional: exact output length.</param>
    /// <returns>Reconstructed audio signal [..., samples].</returns>
    /// <remarks>
    /// <para><b>Griffin-Lim Algorithm:</b></para>
    /// <para>
    /// When only magnitude information is available (phase is unknown), Griffin-Lim iteratively
    /// estimates the phase by repeatedly applying STFT/ISTFT and enforcing magnitude consistency.
    /// </para>
    /// <para><b>Algorithm steps:</b></para>
    /// <list type="number">
    /// <item><description>Initialize with random phase</description></item>
    /// <item><description>Reconstruct signal: ISTFT(magnitude, estimated_phase)</description></item>
    /// <item><description>Compute new STFT of reconstructed signal</description></item>
    /// <item><description>Extract phase from new STFT, keep original magnitude</description></item>
    /// <item><description>Repeat steps 2-4 for specified iterations</description></item>
    /// </list>
    /// <para><b>Quality vs Speed:</b></para>
    /// <list type="bullet">
    /// <item><description>30 iterations: Acceptable quality, some artifacts</description></item>
    /// <item><description>60 iterations: Good quality for most applications</description></item>
    /// <item><description>100+ iterations: Diminishing returns</description></item>
    /// </list>
    /// <para><b>Momentum:</b> Setting momentum > 0 accelerates convergence by using velocity
    /// from previous iterations. 0.99 is recommended; set to 0 for original algorithm.</para>
    /// </remarks>
    Tensor<T> GriffinLim<T>(
        Tensor<T> magnitude,
        int nFft,
        int hopLength,
        Tensor<T> window,
        int iterations = 60,
        double momentum = 0.99,
        int? length = null);

    /// <summary>
    /// Creates a Mel filterbank matrix for converting power spectrograms to Mel scale.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="nMels">Number of Mel bands.</param>
    /// <param name="nFft">FFT size (filterbank will have nFft/2+1 frequency bins).</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="fMin">Minimum frequency in Hz.</param>
    /// <param name="fMax">Maximum frequency in Hz.</param>
    /// <returns>Mel filterbank matrix [nMels, nFft/2+1].</returns>
    /// <remarks>
    /// <para>
    /// Creates triangular filters spaced on the Mel scale. Each filter covers a range of
    /// frequencies, with overlap between adjacent filters. The filterbank is applied by
    /// matrix multiplication: melSpec = filterbank @ powerSpec.
    /// </para>
    /// <para><b>Mel scale formula:</b> mel = 2595 * log10(1 + hz/700)</para>
    /// </remarks>
    Tensor<T> CreateMelFilterbank<T>(int nMels, int nFft, int sampleRate, T fMin, T fMax);

    /// <summary>
    /// Creates a window function tensor for STFT analysis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="windowType">Type of window: "hann", "hamming", "blackman", "bartlett".</param>
    /// <param name="length">Window length (typically equals nFft).</param>
    /// <returns>Window function tensor of shape [length].</returns>
    /// <remarks>
    /// <para><b>Window functions</b> reduce spectral leakage in FFT analysis:</para>
    /// <list type="bullet">
    /// <item><description><b>Hann (Hanning):</b> Most common for audio. Good frequency resolution.</description></item>
    /// <item><description><b>Hamming:</b> Slightly better sidelobe suppression than Hann.</description></item>
    /// <item><description><b>Blackman:</b> Best sidelobe suppression, wider main lobe.</description></item>
    /// <item><description><b>Bartlett:</b> Triangular window, simple but less optimal.</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> Without windowing, abrupt signal edges cause artifacts in the FFT.
    /// Windows taper the signal smoothly to zero at the edges, producing cleaner frequency analysis.</para>
    /// </remarks>
    Tensor<T> CreateWindow<T>(string windowType, int length);

    #endregion

    #region Fused Operations

    /// <summary>
    /// Performs a fused linear transformation: output = activation(input @ weights + bias).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor with shape [..., inputFeatures].</param>
    /// <param name="weights">Weight matrix with shape [inputFeatures, outputFeatures].</param>
    /// <param name="bias">Optional bias vector with shape [outputFeatures].</param>
    /// <param name="activation">Activation function to apply.</param>
    /// <returns>Output tensor with shape [..., outputFeatures].</returns>
    Tensor<T> FusedLinear<T>(
        Tensor<T> input,
        Tensor<T> weights,
        Tensor<T>? bias,
        FusedActivationType activation);

    /// <summary>
    /// Computes the backward pass for fused linear transformation.
    /// </summary>
    Tensor<T> FusedLinearBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> weights,
        Tensor<T> preActivation,
        FusedActivationType activation,
        out Tensor<T> gradWeights,
        out Tensor<T>? gradBias);

    /// <summary>
    /// Performs a fused 2D convolution: output = activation(conv2d(input, kernel) + bias).
    /// </summary>
    Tensor<T> FusedConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int dilationH, int dilationW,
        FusedActivationType activation);

    /// <summary>
    /// Performs a fused 3D convolution: output = activation(conv3d(input, kernel) + bias).
    /// Combines convolution, bias addition, and activation into a single optimized operation.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, channels, depth, height, width].</param>
    /// <param name="kernel">Convolution kernels with shape [outChannels, inChannels, kD, kH, kW].</param>
    /// <param name="bias">Optional bias vector with shape [outChannels].</param>
    /// <param name="strideD">Stride in depth dimension.</param>
    /// <param name="strideH">Stride in height dimension.</param>
    /// <param name="strideW">Stride in width dimension.</param>
    /// <param name="padD">Padding in depth dimension.</param>
    /// <param name="padH">Padding in height dimension.</param>
    /// <param name="padW">Padding in width dimension.</param>
    /// <param name="dilationD">Dilation in depth dimension.</param>
    /// <param name="dilationH">Dilation in height dimension.</param>
    /// <param name="dilationW">Dilation in width dimension.</param>
    /// <param name="activation">Activation function to apply.</param>
    /// <returns>Output tensor with shape [batch, outChannels, outD, outH, outW].</returns>
    Tensor<T> FusedConv3D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW,
        FusedActivationType activation);

    /// <summary>
    /// Performs a fused transposed 2D convolution: output = activation(convTranspose2d(input, kernel) + bias).
    /// Combines transposed convolution, bias addition, and activation into a single optimized operation.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, channels, height, width].</param>
    /// <param name="kernel">Convolution kernels with shape [inChannels, outChannels, kH, kW].</param>
    /// <param name="bias">Optional bias vector with shape [outChannels].</param>
    /// <param name="strideH">Stride in height dimension.</param>
    /// <param name="strideW">Stride in width dimension.</param>
    /// <param name="padH">Padding in height dimension.</param>
    /// <param name="padW">Padding in width dimension.</param>
    /// <param name="outputPadH">Output padding in height dimension.</param>
    /// <param name="outputPadW">Output padding in width dimension.</param>
    /// <param name="activation">Activation function to apply.</param>
    /// <returns>Output tensor with upsampled spatial dimensions.</returns>
    Tensor<T> FusedConvTranspose2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int outputPadH, int outputPadW,
        FusedActivationType activation);

    /// <summary>
    /// Performs fused batch normalization with activation.
    /// </summary>
    Tensor<T> FusedBatchNorm<T>(
        Tensor<T> input,
        Tensor<T> gamma,
        Tensor<T> beta,
        Tensor<T> runningMean,
        Tensor<T> runningVar,
        double epsilon,
        double momentum,
        bool training,
        FusedActivationType activation,
        out Tensor<T> saveMean,
        out Tensor<T> saveVar);

    #endregion

    #region Persistent Tensor Management

    /// <summary>
    /// Registers a tensor as persistent for GPU memory optimization.
    /// </summary>
    void RegisterPersistentTensor<T>(Tensor<T> tensor, PersistentTensorRole role);

    /// <summary>
    /// Unregisters a previously registered persistent tensor.
    /// </summary>
    void UnregisterPersistentTensor<T>(Tensor<T> tensor);

    /// <summary>
    /// Notifies the engine that a persistent tensor's contents have been updated.
    /// </summary>
    void InvalidatePersistentTensor<T>(Tensor<T> tensor);

    #endregion

    #region Normalization Operations

    /// <summary>
    /// Applies batch normalization to a 2D tensor [batch, features].
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, features].</param>
    /// <param name="gamma">Scale parameter with shape [features].</param>
    /// <param name="beta">Shift parameter with shape [features].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="mean">Output: computed mean with shape [features].</param>
    /// <param name="variance">Output: computed variance with shape [features].</param>
    /// <returns>The normalized tensor.</returns>
    Tensor<T> BatchNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance);

    /// <summary>
    /// Computes the backward pass for batch normalization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="gamma">Scale parameter.</param>
    /// <param name="mean">The mean computed during forward pass.</param>
    /// <param name="variance">The variance computed during forward pass.</param>
    /// <param name="epsilon">Small constant used during forward pass.</param>
    /// <param name="gradGamma">Output: gradient with respect to gamma.</param>
    /// <param name="gradBeta">Output: gradient with respect to beta.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> BatchNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta);

    /// <summary>
    /// Applies layer normalization to a tensor of any rank, normalizing over the last N dimensions.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [d0, d1, ..., dn] of any rank.</param>
    /// <param name="gamma">Scale parameter with shape matching the last N dimensions to normalize over.
    /// For example, if input is [batch, seq, embed] and gamma is [embed], normalizes over the last dimension.
    /// If gamma is [seq, embed], normalizes over the last two dimensions.</param>
    /// <param name="beta">Shift parameter with the same shape as gamma.</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="mean">Output: computed mean with shape [d0, d1, ..., d(n-N)] where N is the number of normalized dimensions.</param>
    /// <param name="variance">Output: computed variance with the same shape as mean.</param>
    /// <returns>The normalized tensor with the same shape as input.</returns>
    /// <remarks>
    /// <para>
    /// This follows the industry standard (PyTorch/TensorFlow) behavior for layer normalization:
    /// - Supports tensors of any rank (2D, 3D, 4D, etc.)
    /// - Normalizes over the last N dimensions as determined by gamma's shape
    /// - Each position in the preceding dimensions is normalized independently
    /// </para>
    /// <para><b>Examples:</b>
    /// - Input [32, 64] with gamma [64]: normalizes each of 32 samples over 64 features
    /// - Input [2, 10, 64] with gamma [64]: normalizes each of 20 positions over 64 features
    /// - Input [2, 10, 64] with gamma [10, 64]: normalizes each of 2 batches over 640 features
    /// </para>
    /// </remarks>
    Tensor<T> LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance);

    /// <summary>
    /// Computes the backward pass for layer normalization on tensors of any rank.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer (same shape as forward output).</param>
    /// <param name="input">The original input tensor of any rank.</param>
    /// <param name="gamma">Scale parameter with shape matching normalized dimensions.</param>
    /// <param name="mean">The mean computed during forward pass.</param>
    /// <param name="variance">The variance computed during forward pass.</param>
    /// <param name="epsilon">Small constant used during forward pass.</param>
    /// <param name="gradGamma">Output: gradient with respect to gamma (same shape as gamma).</param>
    /// <param name="gradBeta">Output: gradient with respect to beta (same shape as beta).</param>
    /// <returns>The gradient with respect to the input (same shape as input).</returns>
    /// <remarks>
    /// This backward pass supports the same any-rank tensor semantics as the forward pass.
    /// </remarks>
    Tensor<T> LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta);

    /// <summary>
    /// Applies Root Mean Square Layer Normalization (RMSNorm) as introduced in "Root Mean Square Layer Normalization" (Zhang &amp; Sennrich, 2019).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, seq, features] or any rank.</param>
    /// <param name="gamma">Scale parameter (gain) with shape matching the last dimension(s) to normalize over.</param>
    /// <param name="epsilon">Small constant for numerical stability (typically 1e-6 or 1e-8).</param>
    /// <param name="rms">Output: the computed root mean square values for backward pass.</param>
    /// <returns>The normalized tensor with the same shape as input.</returns>
    /// <remarks>
    /// <para>
    /// RMSNorm is computationally simpler than LayerNorm as it omits the mean centering step:
    /// <code>
    /// RMSNorm(x) = x / RMS(x) * gamma
    /// where RMS(x) = sqrt(mean(x^2) + epsilon)
    /// </code>
    /// </para>
    /// <para><b>Advantages over LayerNorm:</b></para>
    /// <list type="bullet">
    /// <item><description>~7-64% faster than LayerNorm (no mean computation/subtraction)</description></item>
    /// <item><description>Simpler backward pass (fewer operations)</description></item>
    /// <item><description>Used in LLaMA, T5, and other modern transformers</description></item>
    /// <item><description>Empirically shows comparable or better performance</description></item>
    /// </list>
    /// <para><b>Key differences from LayerNorm:</b></para>
    /// <list type="bullet">
    /// <item><description>No mean subtraction (no re-centering)</description></item>
    /// <item><description>No beta/bias parameter (only gamma/gain)</description></item>
    /// <item><description>Normalizes by RMS instead of standard deviation</description></item>
    /// </list>
    /// </remarks>
    Tensor<T> RMSNorm<T>(Tensor<T> input, Tensor<T> gamma, double epsilon, out Tensor<T> rms);

    /// <summary>
    /// Computes the backward pass for RMSNorm.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="gamma">Scale parameter (gain).</param>
    /// <param name="rms">The RMS values computed during forward pass.</param>
    /// <param name="epsilon">Small constant used during forward pass.</param>
    /// <param name="gradGamma">Output: gradient with respect to gamma.</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// The gradient computation is simpler than LayerNorm due to no mean centering.
    /// </para>
    /// </remarks>
    Tensor<T> RMSNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> rms, double epsilon, out Tensor<T> gradGamma);

    /// <summary>
    /// Applies group normalization to a tensor with shape [batch, channels, ...spatial].
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width] or [batch, channels].</param>
    /// <param name="numGroups">The number of groups to divide channels into.</param>
    /// <param name="gamma">Scale parameter with shape [channels].</param>
    /// <param name="beta">Shift parameter with shape [channels].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="mean">Output: computed mean per group with shape [batch, numGroups].</param>
    /// <param name="variance">Output: computed variance per group with shape [batch, numGroups].</param>
    /// <returns>The normalized tensor with the same shape as input.</returns>
    Tensor<T> GroupNorm<T>(Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance);

    /// <summary>
    /// Computes the backward pass for group normalization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="numGroups">The number of groups used during forward pass.</param>
    /// <param name="gamma">Scale parameter.</param>
    /// <param name="mean">The mean computed during forward pass.</param>
    /// <param name="variance">The variance computed during forward pass.</param>
    /// <param name="epsilon">Small constant used during forward pass.</param>
    /// <param name="gradGamma">Output: gradient with respect to gamma.</param>
    /// <param name="gradBeta">Output: gradient with respect to beta.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> GroupNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta);

    /// <summary>
    /// Applies Instance Normalization to the input tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, spatial...].</param>
    /// <param name="gamma">Scale parameter with shape [channels].</param>
    /// <param name="beta">Shift parameter with shape [channels].</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="mean">Output: computed mean per instance with shape [batch, channels].</param>
    /// <param name="variance">Output: computed variance per instance with shape [batch, channels].</param>
    /// <returns>The normalized tensor with the same shape as input.</returns>
    /// <remarks>
    /// Instance normalization normalizes each channel of each sample independently.
    /// Commonly used in style transfer and generative models.
    /// </remarks>
    Tensor<T> InstanceNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance);

    /// <summary>
    /// Computes the backward pass for instance normalization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="gamma">Scale parameter.</param>
    /// <param name="mean">The mean computed during forward pass.</param>
    /// <param name="variance">The variance computed during forward pass.</param>
    /// <param name="epsilon">Small constant used during forward pass.</param>
    /// <param name="gradGamma">Output: gradient with respect to gamma.</param>
    /// <param name="gradBeta">Output: gradient with respect to beta.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> InstanceNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta);

    #endregion

    #region Dropout Operations

    /// <summary>
    /// Applies dropout during training, scaling retained values.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="dropoutRate">The probability of dropping each element (0 to 1).</param>
    /// <param name="training">If true, applies dropout; if false, returns input unchanged.</param>
    /// <param name="mask">Output: the binary mask used for dropout (for backward pass).</param>
    /// <returns>The tensor with dropout applied during training.</returns>
    Tensor<T> Dropout<T>(Tensor<T> input, double dropoutRate, bool training, out Tensor<T> mask);

    /// <summary>
    /// Computes the backward pass for dropout.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="mask">The mask from forward pass.</param>
    /// <param name="dropoutRate">The dropout rate used during forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> DropoutBackward<T>(Tensor<T> gradOutput, Tensor<T> mask, double dropoutRate);

    #endregion

    #region Embedding Operations

    /// <summary>
    /// Performs embedding lookup from an embedding table.
    /// </summary>
    /// <typeparam name="T">The numeric type of embedding values.</typeparam>
    /// <param name="indices">The indices to look up.</param>
    /// <param name="embeddingTable">The embedding table with shape [vocabSize, embeddingDim].</param>
    /// <returns>The embeddings for the given indices with shape [indices.Shape..., embeddingDim].</returns>
    Tensor<T> Embedding<T>(Tensor<int> indices, Tensor<T> embeddingTable);

    /// <summary>
    /// Computes the backward pass for embedding lookup.
    /// </summary>
    /// <typeparam name="T">The numeric type of embedding values.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="indices">The indices used during forward pass.</param>
    /// <param name="vocabSize">The size of the vocabulary.</param>
    /// <param name="embeddingDim">The dimension of each embedding vector.</param>
    /// <returns>The gradient with respect to the embedding table.</returns>
    Tensor<T> EmbeddingBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, int vocabSize, int embeddingDim);

    #endregion

    #region Loss Function Operations

    /// <summary>
    /// Computes cross-entropy loss for classification tasks.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predictions">The predicted logits with shape [batch, numClasses].</param>
    /// <param name="targets">The target class indices or one-hot encoded targets.</param>
    /// <returns>The scalar loss value.</returns>
    T CrossEntropyLoss<T>(Tensor<T> predictions, Tensor<T> targets);

    /// <summary>
    /// Computes the gradient of cross-entropy loss.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predictions">The predicted logits.</param>
    /// <param name="targets">The target values.</param>
    /// <returns>The gradient with respect to predictions.</returns>
    Tensor<T> CrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets);

    /// <summary>
    /// Computes mean squared error loss.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predictions">The predicted values.</param>
    /// <param name="targets">The target values.</param>
    /// <returns>The scalar loss value.</returns>
    T MseLoss<T>(Tensor<T> predictions, Tensor<T> targets);

    /// <summary>
    /// Computes the gradient of mean squared error loss.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predictions">The predicted values.</param>
    /// <param name="targets">The target values.</param>
    /// <returns>The gradient with respect to predictions.</returns>
    Tensor<T> MseBackward<T>(Tensor<T> predictions, Tensor<T> targets);

    #endregion

    #region Global Pooling Operations

    /// <summary>
    /// Performs global average pooling over spatial dimensions.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <returns>The pooled tensor with shape [batch, channels, 1, 1].</returns>
    Tensor<T> GlobalAvgPool2D<T>(Tensor<T> input);

    /// <summary>
    /// Performs global max pooling over spatial dimensions.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <returns>The pooled tensor with shape [batch, channels, 1, 1].</returns>
    Tensor<T> GlobalMaxPool2D<T>(Tensor<T> input);

    /// <summary>
    /// Performs adaptive average pooling to a target output size.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <param name="outputHeight">The target output height.</param>
    /// <param name="outputWidth">The target output width.</param>
    /// <returns>The pooled tensor with shape [batch, channels, outputHeight, outputWidth].</returns>
    Tensor<T> AdaptiveAvgPool2D<T>(Tensor<T> input, int outputHeight, int outputWidth);

    #endregion

    #region Tensor Reduction Operations

    /// <summary>
    /// Computes the maximum value along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axes">The axes along which to compute the maximum.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <param name="maxIndices">Output: indices of maximum values for backward pass.</param>
    /// <returns>The tensor containing maximum values.</returns>
    Tensor<T> ReduceMax<T>(Tensor<T> input, int[] axes, bool keepDims, out int[] maxIndices);

    /// <summary>
    /// Computes the backward pass for reduce max.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="maxIndices">The indices of maximum values from forward pass.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ReduceMaxBackward<T>(Tensor<T> gradOutput, int[] maxIndices, int[] inputShape);

    /// <summary>
    /// Computes the mean along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axes">The axes along which to compute the mean.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <returns>The tensor containing mean values.</returns>
    Tensor<T> ReduceMean<T>(Tensor<T> input, int[] axes, bool keepDims);

    /// <summary>
    /// Computes the backward pass for reduce mean.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <param name="axes">The axes that were reduced.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ReduceMeanBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] axes);

    /// <summary>
    /// Computes the variance of tensor elements along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axes">The axes along which to compute the variance.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <returns>The tensor containing variance values.</returns>
    Tensor<T> ReduceVariance<T>(Tensor<T> input, int[] axes, bool keepDims);

    /// <summary>
    /// Computes the backward pass for reduce variance.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="mean">The mean values computed during forward pass.</param>
    /// <param name="axes">The axes that were reduced.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ReduceVarianceBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, int[] axes);

    /// <summary>
    /// Computes the natural logarithm of variance of tensor elements along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="axes">The axes along which to compute the log variance.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions with size 1.</param>
    /// <param name="epsilon">Small value for numerical stability (prevents log(0)).</param>
    /// <returns>The tensor containing log variance values.</returns>
    Tensor<T> ReduceLogVariance<T>(Tensor<T> input, int[] axes, bool keepDims, double epsilon = 1e-8);

    /// <summary>
    /// Computes the backward pass for reduce log variance.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="input">The original input tensor.</param>
    /// <param name="mean">The mean values computed during forward pass.</param>
    /// <param name="variance">The variance values computed during forward pass.</param>
    /// <param name="axes">The axes that were reduced.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> ReduceLogVarianceBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, Tensor<T> variance, int[] axes);

    #endregion

    #region Scatter Operations (Graph Neural Networks)

    /// <summary>
    /// Performs scatter-add operation: aggregates source values into output by adding at indices.
    /// Essential for Graph Neural Network message passing and sparse tensor operations.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="source">Source tensor with values to scatter.</param>
    /// <param name="indices">Index tensor specifying where to scatter values along the given dimension.
    /// Shape must be broadcastable to source shape.</param>
    /// <param name="dim">The dimension along which to scatter (default: 0).</param>
    /// <param name="outputSize">Size of the output along the scatter dimension.
    /// If null, uses max(indices) + 1.</param>
    /// <returns>Output tensor with scattered (summed) values.</returns>
    /// <remarks>
    /// <para>
    /// For each index i in the index tensor:
    /// <code>output[indices[i], ...] += source[i, ...]</code>
    /// </para>
    /// <para><b>Common GNN uses:</b></para>
    /// <list type="bullet">
    /// <item><description>Aggregating neighbor messages to nodes</description></item>
    /// <item><description>Edge-to-node aggregation in MessagePassingLayer</description></item>
    /// <item><description>Pooling operations in graph classification</description></item>
    /// </list>
    /// </remarks>
    Tensor<T> ScatterAdd<T>(Tensor<T> source, Tensor<int> indices, int dim = 0, int? outputSize = null);

    /// <summary>
    /// Computes the backward pass for scatter-add.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer.</param>
    /// <param name="indices">The indices used in forward pass.</param>
    /// <param name="sourceShape">The original source tensor shape.</param>
    /// <param name="dim">The dimension along which scatter was performed.</param>
    /// <returns>Gradient with respect to the source tensor.</returns>
    Tensor<T> ScatterAddBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, int[] sourceShape, int dim = 0);

    /// <summary>
    /// Performs scatter-mean operation: computes mean of scattered values at each index.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="source">Source tensor with values to scatter.</param>
    /// <param name="indices">Index tensor specifying where to scatter values.</param>
    /// <param name="dim">The dimension along which to scatter (default: 0).</param>
    /// <param name="outputSize">Size of the output along the scatter dimension.</param>
    /// <param name="counts">Output: count of elements at each index (for backward pass).</param>
    /// <returns>Output tensor with mean values at each scattered position.</returns>
    /// <remarks>
    /// <para>
    /// Computes: output[i] = sum(source[indices == i]) / count(indices == i)
    /// </para>
    /// <para>
    /// Useful for mean aggregation in GNNs where node degree varies.
    /// </para>
    /// </remarks>
    Tensor<T> ScatterMean<T>(Tensor<T> source, Tensor<int> indices, out Tensor<int>? counts, int dim = 0, int? outputSize = null);

    /// <summary>
    /// Computes the backward pass for scatter-mean.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer.</param>
    /// <param name="indices">The indices used in forward pass.</param>
    /// <param name="counts">The counts computed during forward pass.</param>
    /// <param name="sourceShape">The original source tensor shape.</param>
    /// <param name="dim">The dimension along which scatter was performed.</param>
    /// <returns>Gradient with respect to the source tensor.</returns>
    Tensor<T> ScatterMeanBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, Tensor<int> counts, int[] sourceShape, int dim = 0);

    /// <summary>
    /// Performs scatter-max operation: takes maximum of scattered values at each index.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="source">Source tensor with values to scatter.</param>
    /// <param name="indices">Index tensor specifying where to scatter values.</param>
    /// <param name="dim">The dimension along which to scatter (default: 0).</param>
    /// <param name="outputSize">Size of the output along the scatter dimension.</param>
    /// <param name="argmax">Output: indices of maximum values (for backward pass).</param>
    /// <returns>Output tensor with maximum values at each scattered position.</returns>
    /// <remarks>
    /// <para>
    /// Computes: output[i] = max(source[indices == i])
    /// </para>
    /// <para>
    /// Useful for max-pooling aggregation in GNNs.
    /// </para>
    /// </remarks>
    Tensor<T> ScatterMax<T>(Tensor<T> source, Tensor<int> indices, out Tensor<int>? argmax, int dim = 0, int? outputSize = null);

    /// <summary>
    /// Computes the backward pass for scatter-max.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer.</param>
    /// <param name="argmax">The argmax indices from forward pass.</param>
    /// <param name="sourceShape">The original source tensor shape.</param>
    /// <param name="dim">The dimension along which scatter was performed.</param>
    /// <returns>Gradient with respect to the source tensor.</returns>
    Tensor<T> ScatterMaxBackward<T>(Tensor<T> gradOutput, Tensor<int> argmax, int[] sourceShape, int dim = 0);

    /// <summary>
    /// Performs scatter-softmax operation: applies softmax over scattered groups.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="source">Source tensor with values to scatter.</param>
    /// <param name="indices">Index tensor specifying group membership.</param>
    /// <param name="dim">The dimension along which to scatter (default: 0).</param>
    /// <param name="outputSize">Size of the output along the scatter dimension.</param>
    /// <returns>Output tensor with softmax applied within each scatter group.</returns>
    /// <remarks>
    /// <para>
    /// For each unique index i, computes softmax over all source values with that index:
    /// <code>output[j] = exp(source[j]) / sum(exp(source[indices == indices[j]]))</code>
    /// </para>
    /// <para>
    /// Essential for attention mechanisms in Graph Attention Networks (GAT).
    /// </para>
    /// </remarks>
    Tensor<T> ScatterSoftmax<T>(Tensor<T> source, Tensor<int> indices, int dim = 0, int? outputSize = null);

    /// <summary>
    /// Computes the backward pass for scatter-softmax.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">Gradient from the next layer.</param>
    /// <param name="output">The output from forward scatter-softmax (softmax values).</param>
    /// <param name="indices">The indices used in forward pass.</param>
    /// <param name="dim">The dimension along which scatter was performed.</param>
    /// <returns>Gradient with respect to the source tensor.</returns>
    Tensor<T> ScatterSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, Tensor<int> indices, int dim = 0);

    #endregion

    #region Spatial Operations

    /// <summary>
    /// Performs nearest-neighbor upsampling on a tensor of any rank (at least 2D).
    /// The last two dimensions are treated as height and width for upsampling.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with at least 2 dimensions, where the last two are height and width.
    /// Supports: 2D [H, W], 3D [C, H, W], 4D [B, C, H, W], 5D+.</param>
    /// <param name="scaleH">The height scaling factor.</param>
    /// <param name="scaleW">The width scaling factor.</param>
    /// <returns>The upsampled tensor with scaled height and width dimensions.</returns>
    Tensor<T> Upsample<T>(Tensor<T> input, int scaleH, int scaleW);

    /// <summary>
    /// Computes the backward pass for upsampling on a tensor of any rank (at least 2D).
    /// The last two dimensions are treated as height and width.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="inputShape">The original input shape (any rank, at least 2D).</param>
    /// <param name="scaleH">The height scaling factor used in forward pass.</param>
    /// <param name="scaleW">The width scaling factor used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> UpsampleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleH, int scaleW);

    /// <summary>
    /// Performs pixel shuffle (depth-to-space) operation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <param name="upscaleFactor">The factor to upscale spatial dimensions.</param>
    /// <returns>The rearranged tensor with increased spatial dimensions.</returns>
    Tensor<T> PixelShuffle<T>(Tensor<T> input, int upscaleFactor);

    /// <summary>
    /// Computes the backward pass for pixel shuffle.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <param name="upscaleFactor">The upscale factor used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> PixelShuffleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int upscaleFactor);

    /// <summary>
    /// Generates a normalized affine grid for spatial transformer sampling.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="theta">Affine matrices of shape [batch, 2, 3].</param>
    /// <param name="outputHeight">Target grid height.</param>
    /// <param name="outputWidth">Target grid width.</param>
    /// <returns>Grid tensor of shape [batch, outputHeight, outputWidth, 2] in [-1, 1] normalized coords.</returns>
    /// <remarks>
    /// <para>
    /// <b>IMPORTANT: Layout Note</b> - This method and <see cref="GridSample{T}"/> use NHWC layout
    /// [batch, height, width, channels/coords], which differs from Conv2D, MaxPool2D, and other
    /// spatial operations that use NCHW layout [batch, channels, height, width].
    /// </para>
    /// <para>
    /// When using these methods with NCHW tensors, you must transpose:
    /// <code>
    /// // NCHW to NHWC before GridSample
    /// var inputNHWC = input.Transpose([0, 2, 3, 1]);
    /// var output = engine.GridSample(inputNHWC, grid);
    /// // NHWC to NCHW after GridSample
    /// var outputNCHW = output.Transpose([0, 3, 1, 2]);
    /// </code>
    /// </para>
    /// </remarks>
    Tensor<T> AffineGrid<T>(Tensor<T> theta, int outputHeight, int outputWidth);

    /// <summary>
    /// Samples an input tensor using a normalized grid with bilinear interpolation.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="input">Input tensor [batch, height, width, channels] (NHWC format).</param>
    /// <param name="grid">Sampling grid [batch, outH, outW, 2] with coords in [-1, 1].</param>
    /// <returns>Sampled output tensor [batch, outH, outW, channels] (NHWC format).</returns>
    /// <remarks>
    /// <para>
    /// <b>IMPORTANT: Layout Note</b> - This method uses NHWC layout [batch, height, width, channels],
    /// which differs from Conv2D, MaxPool2D, and other spatial operations that use NCHW layout
    /// [batch, channels, height, width]. Ensure inputs are transposed appropriately.
    /// </para>
    /// <para>
    /// The grid coordinates are normalized to [-1, 1] range where (-1, -1) is the top-left corner
    /// and (1, 1) is the bottom-right corner of the input tensor.
    /// </para>
    /// </remarks>
    Tensor<T> GridSample<T>(Tensor<T> input, Tensor<T> grid);

    /// <summary>
    /// Extracts sliding local blocks (patches) from a batched input tensor.
    /// Similar to PyTorch's torch.nn.functional.unfold (im2col).
    /// Input shape: [batch, channels, height, width]
    /// Output shape: [batch, channels * kernelH * kernelW, L] where L = number of patches.
    /// </summary>
    Tensor<T> Unfold<T>(Tensor<T> input, int[] kernelSize, int[] stride, int[] padding);

    /// <summary>
    /// Combines an array of sliding local blocks back into a full tensor.
    /// Inverse of <see cref="Unfold{T}"/>. Similar to PyTorch's torch.nn.functional.fold (col2im).
    /// Input shape: [batch, channels * kernelH * kernelW, L]
    /// Output shape: [batch, channels, outputH, outputW]
    /// </summary>
    Tensor<T> Fold<T>(Tensor<T> input, int[] outputSize, int[] kernelSize, int[] stride, int[] padding);

    /// <summary>
    /// Performs complex-valued matrix multiplication using split real/imaginary tensors.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="aReal">Real part of left matrix [M, K].</param>
    /// <param name="aImag">Imag part of left matrix [M, K].</param>
    /// <param name="bReal">Real part of right matrix [K, N].</param>
    /// <param name="bImag">Imag part of right matrix [K, N].</param>
    /// <returns>(real, imag) tuple representing the product [M, N].</returns>
    (Tensor<T> real, Tensor<T> imag) ComplexMatMul<T>(Tensor<T> aReal, Tensor<T> aImag, Tensor<T> bReal, Tensor<T> bImag);

    /// <summary>
    /// Computes magnitude squared of a complex tensor given real/imag split.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="real">Real part tensor.</param>
    /// <param name="imag">Imag part tensor.</param>
    /// <returns>Tensor of magnitudes squared.</returns>
    Tensor<T> ComplexMagnitudeSquared<T>(Tensor<T> real, Tensor<T> imag);

    /// <summary>
    /// Normalizes a complex tensor (real/imag split) so that sum(|z|^2) = 1.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="real">Real part tensor.</param>
    /// <param name="imag">Imag part tensor.</param>
    /// <returns>(real, imag) tuple of normalized complex tensor.</returns>
    (Tensor<T> real, Tensor<T> imag) ComplexNormalize<T>(Tensor<T> real, Tensor<T> imag);

    /// <summary>
    /// Crops a region from a 4D tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, channels, height, width].</param>
    /// <param name="top">The top offset for cropping.</param>
    /// <param name="left">The left offset for cropping.</param>
    /// <param name="height">The height of the cropped region.</param>
    /// <param name="width">The width of the cropped region.</param>
    /// <returns>The cropped tensor.</returns>
    Tensor<T> Crop<T>(Tensor<T> input, int top, int left, int height, int width);

    /// <summary>
    /// Computes the backward pass for crop.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <param name="top">The top offset used in forward pass.</param>
    /// <param name="left">The left offset used in forward pass.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> CropBackward<T>(Tensor<T> gradOutput, int[] inputShape, int top, int left);

    /// <summary>
    /// Pads a 2D tensor with specified values.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor.</param>
    /// <param name="padTop">Padding for top edge.</param>
    /// <param name="padBottom">Padding for bottom edge.</param>
    /// <param name="padLeft">Padding for left edge.</param>
    /// <param name="padRight">Padding for right edge.</param>
    /// <param name="padValue">The value to use for padding.</param>
    /// <returns>The padded tensor.</returns>
    Tensor<T> Pad<T>(Tensor<T> input, int padTop, int padBottom, int padLeft, int padRight, T padValue);

    /// <summary>
    /// Computes the backward pass for padding.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer.</param>
    /// <param name="padTop">Padding used for top edge.</param>
    /// <param name="padLeft">Padding used for left edge.</param>
    /// <param name="inputShape">The original input shape.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> PadBackward<T>(Tensor<T> gradOutput, int padTop, int padLeft, int[] inputShape);

    /// <summary>
    /// Concatenates tensors along a specified axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">The list of tensors to concatenate.</param>
    /// <param name="axis">The axis along which to concatenate.</param>
    /// <returns>The concatenated tensor.</returns>
    Tensor<T> Concat<T>(IReadOnlyList<Tensor<T>> tensors, int axis);

    /// <summary>
    /// Computes the sum of squares of all elements in a tensor (L2 norm squared).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <returns>The scalar sum of squared elements.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Element-wise Math Operations</b></para>
    /// <para>
    /// Computes ÃŽÂ£(x_iÃ‚Â²) for all elements in the tensor. Used in:
    /// - L2 regularization loss computation
    /// - Frobenius norm calculation (sqrt of sum of squares)
    /// - Gradient magnitude computation
    /// - Weight decay penalties
    /// </para>
    /// <para>
    /// GPU acceleration provides significant speedup for large tensors.
    /// </para>
    /// </remarks>
    T TensorSumOfSquares<T>(Tensor<T> tensor);

    /// <summary>
    /// Performs embedding lookup - gathers rows from an embedding table based on indices.
    /// </summary>
    /// <typeparam name="TValue">The numeric type of embedding values.</typeparam>
    /// <typeparam name="TIndex">The integer type for indices (must be unmanaged).</typeparam>
    /// <param name="embeddings">The embedding table tensor [vocab_size, embedding_dim].</param>
    /// <param name="indices">The indices tensor containing token IDs.</param>
    /// <returns>The gathered embeddings with shape [*indices.shape, embedding_dim].</returns>
    /// <remarks>
    /// <para><b>US-GPU-017: Embedding Operations</b></para>
    /// <para>
    /// Embedding lookup is a fundamental operation for NLP and sequence models:
    /// - Word/token embeddings in language models
    /// - Item embeddings in recommendation systems
    /// - Categorical feature embeddings
    /// </para>
    /// <para>
    /// For each index i in indices, retrieves embeddings[i, :] and places it in the output.
    /// GPU acceleration provides significant speedup for large vocabularies and batch sizes.
    /// </para>
    /// </remarks>
    Tensor<TValue> TensorEmbeddingLookup<TValue, TIndex>(Tensor<TValue> embeddings, Tensor<TIndex> indices)
        where TIndex : unmanaged;

    /// <summary>
    /// Performs embedding lookup backward pass - scatters gradients back to embedding table.
    /// </summary>
    /// <typeparam name="TValue">The numeric type of gradient and embedding values.</typeparam>
    /// <typeparam name="TIndex">The integer type for indices (must be unmanaged).</typeparam>
    /// <param name="gradOutput">The gradient from the next layer [*indices.shape, embedding_dim].</param>
    /// <param name="indices">The indices tensor containing token IDs.</param>
    /// <param name="vocabSize">The vocabulary size (number of rows in embedding table).</param>
    /// <param name="embeddingDim">The embedding dimension.</param>
    /// <returns>The gradient for the embedding table [vocab_size, embedding_dim].</returns>
    /// <remarks>
    /// <para><b>US-GPU-017: Embedding Operations</b></para>
    /// <para>
    /// Computes the gradient for embedding parameters by accumulating gradients for each index.
    /// For each index i, adds gradOutput[position] to embeddingGrad[i, :].
    /// Handles duplicate indices by accumulating their gradients.
    /// </para>
    /// </remarks>
    Tensor<TValue> TensorEmbeddingLookupBackward<TValue, TIndex>(Tensor<TValue> gradOutput, Tensor<TIndex> indices, int vocabSize, int embeddingDim)
        where TIndex : unmanaged;

    /// <summary>
    /// Computes the Radial Basis Function (RBF) kernel between input samples and centers.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The input tensor with shape [batch, features].</param>
    /// <param name="centers">The RBF center positions with shape [numCenters, features].</param>
    /// <param name="epsilons">The epsilon values (1/(2*widthÃ‚Â²)) for each center with shape [numCenters].</param>
    /// <returns>The RBF kernel output with shape [batch, numCenters], computing exp(-epsilon * ||x - center||Ã‚Â²).</returns>
    /// <remarks>
    /// <para>
    /// Computes Gaussian RBF: K(x, c) = exp(-epsilon * ||x - c||Ã‚Â²) where:
    /// - x is an input sample
    /// - c is a center
    /// - epsilon = 1/(2*widthÃ‚Â²) controls the spread
    /// </para>
    /// <para><b>For Beginners:</b> RBF kernels measure similarity between points.
    /// Points close to a center produce values near 1, distant points produce values near 0.
    /// </para>
    /// </remarks>
    Tensor<T> RBFKernel<T>(Tensor<T> input, Tensor<T> centers, Tensor<T> epsilons);

    /// <summary>
    /// Computes the backward pass for the RBF kernel.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient from the next layer with shape [batch, numCenters].</param>
    /// <param name="input">The original input tensor with shape [batch, features].</param>
    /// <param name="centers">The RBF center positions with shape [numCenters, features].</param>
    /// <param name="epsilons">The epsilon values with shape [numCenters].</param>
    /// <param name="output">The output from the forward pass with shape [batch, numCenters].</param>
    /// <returns>A tuple containing gradients for (input, centers, epsilons).</returns>
    (Tensor<T> gradInput, Tensor<T> gradCenters, Tensor<T> gradEpsilons) RBFKernelBackward<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> centers, Tensor<T> epsilons, Tensor<T> output);

    #endregion

    #region Tensor Shape Operations

    /// <summary>
    /// Repeats each element of a tensor along the specified axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor to repeat.</param>
    /// <param name="repeats">The number of times to repeat each element.</param>
    /// <param name="axis">The axis along which to repeat. Default is 0.</param>
    /// <returns>A tensor with elements repeated along the specified axis.</returns>
    /// <remarks>
    /// <para>
    /// This operation is similar to numpy.repeat(). For a 1D tensor [a, b, c] with repeats=2:
    /// Result: [a, a, b, b, c, c]
    /// </para>
    /// <para><b>For Beginners:</b> This is useful for creating masks or expanding data
    /// where each element needs to be duplicated multiple times.
    /// </para>
    /// </remarks>
    Tensor<T> TensorRepeatElements<T>(Tensor<T> tensor, int repeats, int axis = 0);

    /// <summary>
    /// Tiles (repeats) a tensor along each axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor to tile.</param>
    /// <param name="multiples">The number of times to tile along each axis.</param>
    /// <returns>A tensor that is the input tiled according to multiples.</returns>
    /// <remarks>
    /// <para>
    /// This operation is similar to numpy.tile(). For a tensor [a, b] with multiples=[3]:
    /// Result: [a, b, a, b, a, b]
    /// </para>
    /// <para>
    /// For a 2D tensor [[1, 2], [3, 4]] with multiples=[2, 3]:
    /// Result: [[1,2,1,2,1,2], [3,4,3,4,3,4], [1,2,1,2,1,2], [3,4,3,4,3,4]]
    /// </para>
    /// </remarks>
    Tensor<T> TensorTile<T>(Tensor<T> tensor, int[] multiples);

    /// <summary>
    /// Extracts a slice from a tensor along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor to slice.</param>
    /// <param name="start">The starting indices for each axis.</param>
    /// <param name="length">The length to extract along each axis.</param>
    /// <returns>A tensor containing the sliced portion.</returns>
    /// <remarks>
    /// <para>
    /// This operation extracts a contiguous region from the tensor.
    /// For a 1D tensor [a, b, c, d, e] with start=[1] and length=[3]:
    /// Result: [b, c, d]
    /// </para>
    /// </remarks>
    Tensor<T> TensorSlice<T>(Tensor<T> tensor, int[] start, int[] length);

    /// <summary>
    /// Sets a slice of a tensor to values from another tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="destination">The tensor to modify (in-place or returns new tensor).</param>
    /// <param name="source">The tensor containing values to set.</param>
    /// <param name="start">The starting indices where to place the source tensor.</param>
    /// <returns>A tensor with the slice set to the source values.</returns>
    /// <remarks>
    /// <para>
    /// This operation sets values in a region of the destination tensor.
    /// Useful for building tensors piece by piece without manual loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSetSlice<T>(Tensor<T> destination, Tensor<T> source, int[] start);

    /// <summary>
    /// Creates a tensor by selecting elements based on a condition mask.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="condition">A boolean-like tensor where non-zero means true.</param>
    /// <param name="x">Values to select where condition is true.</param>
    /// <param name="y">Values to select where condition is false.</param>
    /// <returns>A tensor with elements from x where condition is true, else from y.</returns>
    /// <remarks>
    /// <para>
    /// This operation is similar to numpy.where() or torch.where().
    /// Result[i] = condition[i] != 0 ? x[i] : y[i]
    /// </para>
    /// <para>
    /// <b>Note:</b> Prefer the overload that accepts <c>Tensor&lt;bool&gt;</c> for explicit boolean conditions.
    /// This overload treats any non-zero value as true, which may lead to unexpected behavior with floating-point types.
    /// </para>
    /// </remarks>
    Tensor<T> TensorWhere<T>(Tensor<T> condition, Tensor<T> x, Tensor<T> y);

    #endregion

    #region Loop Elimination Operations

    /// <summary>
    /// Copies data from a source tensor to a destination tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="source">The source tensor to copy from.</param>
    /// <param name="destination">The destination tensor to copy to.</param>
    /// <remarks>
    /// <para>
    /// This operation performs an in-place copy of tensor data.
    /// Both tensors must have the same total number of elements.
    /// </para>
    /// </remarks>
    void TensorCopy<T>(Tensor<T> source, Tensor<T> destination);

    /// <summary>
    /// Fills a tensor with a constant value.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to fill.</param>
    /// <param name="value">The value to fill with.</param>
    /// <remarks>
    /// <para>
    /// This operation sets all elements of the tensor to the specified value.
    /// Useful for initialization without manual loops.
    /// </para>
    /// </remarks>
    void TensorFill<T>(Tensor<T> tensor, T value);

    /// <summary>
    /// Computes the outer product of two tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor (typically a vector [N]).</param>
    /// <param name="b">The second tensor (typically a vector [M]).</param>
    /// <returns>A tensor containing the outer product [N, M].</returns>
    /// <remarks>
    /// <para>
    /// For vectors a[N] and b[M], produces a matrix [N, M] where result[i,j] = a[i] * b[j].
    /// This is useful for computing weight gradients: dW = x^T * dy.
    /// </para>
    /// </remarks>
    Tensor<T> TensorOuterProduct<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Computes batched outer products for all items in a batch.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">The first tensor [batch, N].</param>
    /// <param name="b">The second tensor [batch, M].</param>
    /// <returns>A tensor containing the batched outer products [batch, N, M].</returns>
    /// <remarks>
    /// <para>
    /// For each batch item, computes result[b, i, j] = a[b, i] * b[b, j].
    /// Useful for batched gradient computations without explicit loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBatchOuterProduct<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Permutes the dimensions of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to permute.</param>
    /// <param name="axes">The new order of dimensions (e.g., [0, 2, 1] swaps last two dims).</param>
    /// <returns>A tensor with permuted dimensions.</returns>
    /// <remarks>
    /// <para>
    /// This generalizes transpose to arbitrary dimension reordering.
    /// Similar to numpy.transpose() or torch.permute().
    /// </para>
    /// </remarks>
    Tensor<T> TensorPermute<T>(Tensor<T> tensor, int[] axes);

    /// <summary>
    /// Expands dimensions by inserting a new axis of size 1.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to expand.</param>
    /// <param name="axis">The position where to insert the new axis.</param>
    /// <returns>A tensor with an additional dimension of size 1.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.expand_dims() or torch.unsqueeze().
    /// Useful for broadcasting operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorExpandDims<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Removes singleton dimensions from a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to squeeze.</param>
    /// <param name="axis">The axis to remove (must have size 1). Use -1 to remove all singleton dims.</param>
    /// <returns>A tensor with the specified singleton dimension removed.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.squeeze() or torch.squeeze().
    /// Removes dimensions of size 1.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSqueeze<T>(Tensor<T> tensor, int axis = -1);

    /// <summary>
    /// Performs scatter-add: adds values to specific indices in a destination tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="destination">The destination tensor to add values to.</param>
    /// <param name="indices">The indices where to add values (integer tensor).</param>
    /// <param name="updates">The values to add at the specified indices.</param>
    /// <param name="axis">The axis along which to scatter.</param>
    /// <returns>A tensor with values added at the specified indices.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.scatter_add() or tf.tensor_scatter_nd_add().
    /// Useful for sparse gradient accumulation in embeddings.
    /// </para>
    /// </remarks>
    Tensor<T> TensorScatterAdd<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> updates, int axis = 0);

    /// <summary>
    /// Gathers values from a tensor along an axis using indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="source">The source tensor to gather from.</param>
    /// <param name="indices">The indices specifying which values to gather (integer tensor).</param>
    /// <param name="axis">The axis along which to gather.</param>
    /// <returns>A tensor containing the gathered values.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.gather() or tf.gather().
    /// Useful for index-based lookups without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorGather<T>(Tensor<T> source, Tensor<int> indices, int axis = 0);

    /// <summary>
    /// Computes a cumulative sum along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to compute the cumulative sum.</param>
    /// <returns>A tensor where each element is the sum of all previous elements along the axis.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.cumsum() or torch.cumsum().
    /// Useful for CRF forward-backward and other sequence operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorCumSum<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Computes a log-sum-exp reduction along an axis (numerically stable).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to compute logsumexp.</param>
    /// <param name="keepDims">Whether to keep the reduced dimension.</param>
    /// <returns>A tensor with the log-sum-exp values.</returns>
    /// <remarks>
    /// <para>
    /// Computes log(sum(exp(x))) in a numerically stable way.
    /// Essential for CRF partition functions and attention mechanisms.
    /// </para>
    /// </remarks>
    Tensor<T> TensorLogSumExp<T>(Tensor<T> tensor, int axis, bool keepDims = false);

    /// <summary>
    /// Generates a tensor filled with random values from a uniform distribution [0, 1).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the output tensor.</param>
    /// <returns>A tensor filled with random uniform values.</returns>
    /// <remarks>
    /// <para>
    /// Useful for weight initialization and dropout masks without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorRandomUniform<T>(int[] shape);

    /// <summary>
    /// Generates a tensor filled with random values from a normal distribution.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the output tensor.</param>
    /// <param name="mean">The mean of the distribution.</param>
    /// <param name="stddev">The standard deviation of the distribution.</param>
    /// <returns>A tensor filled with random normal values.</returns>
    /// <remarks>
    /// <para>
    /// Useful for Xavier/He initialization without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorRandomNormal<T>(int[] shape, T mean, T stddev);

    /// <summary>
    /// Generates a tensor filled with random values from a uniform distribution within a specified range.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the output tensor.</param>
    /// <param name="min">The minimum value (inclusive).</param>
    /// <param name="max">The maximum value (exclusive).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A tensor filled with random values in [min, max).</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Random Operations</b></para>
    /// <para>
    /// Used for weight initialization with specific ranges (e.g., Xavier uniform [-limit, limit]),
    /// embedding initialization, and data augmentation. More flexible than TensorRandomUniform
    /// which only generates values in [0, 1).
    /// </para>
    /// </remarks>
    Tensor<T> TensorRandomUniformRange<T>(int[] shape, T min, T max, int? seed = null);

    /// <summary>
    /// Generates a dropout mask tensor where each element is either zero or a scale value.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shape">The shape of the output tensor.</param>
    /// <param name="dropoutRate">Probability of dropping each element (0 to 1).</param>
    /// <param name="scale">The scale factor for non-dropped elements (typically 1/(1-dropoutRate)).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>A tensor containing the dropout mask.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Dropout Operations</b></para>
    /// <para>
    /// Used in dropout layers during training. Elements are randomly set to zero with probability
    /// dropoutRate, and remaining elements are scaled to maintain expected values.
    /// The mask can be multiplied element-wise with activations: output = input * mask.
    /// </para>
    /// </remarks>
    Tensor<T> TensorDropoutMask<T>(int[] shape, T dropoutRate, T scale, int? seed = null);

    /// <summary>
    /// Subtracts a tensor from a scalar value element-wise (scalar - tensor).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="scalar">The scalar value to subtract from.</param>
    /// <param name="tensor">The tensor to subtract.</param>
    /// <returns>A tensor with (scalar - x) for each element.</returns>
    /// <remarks>
    /// <para><b>US-GPU-016: Tensor Arithmetic Operations</b></para>
    /// <para>
    /// Used for computing (1 - p) in probability calculations, BCE loss gradients,
    /// and other operations where the subtraction order matters.
    /// More efficient than TensorNegate(TensorSubtractScalar(tensor, scalar)).
    /// </para>
    /// </remarks>
    Tensor<T> ScalarMinusTensor<T>(T scalar, Tensor<T> tensor);

    /// <summary>
    /// Creates an identity matrix as a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="size">The size of the square identity matrix.</param>
    /// <returns>A tensor [size, size] with 1s on diagonal and 0s elsewhere.</returns>
    /// <remarks>
    /// <para>
    /// Useful for initializing transformation matrices and attention masks.
    /// </para>
    /// </remarks>
    Tensor<T> TensorEye<T>(int size);

    /// <summary>
    /// Creates a diagonal tensor from a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="diagonal">The values to place on the diagonal.</param>
    /// <returns>A tensor with the diagonal values and zeros elsewhere.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.diag() or torch.diag().
    /// </para>
    /// </remarks>
    Tensor<T> TensorDiag<T>(Tensor<T> diagonal);

    /// <summary>
    /// Extracts the diagonal from a matrix tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input matrix tensor.</param>
    /// <returns>A 1D tensor containing the diagonal elements.</returns>
    /// <remarks>
    /// <para>
    /// Extracts diagonal[i] = tensor[i, i] without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorDiagonal<T>(Tensor<T> tensor);

    /// <summary>
    /// Applies einsum (Einstein summation) notation for flexible tensor contractions.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="subscripts">The einsum subscript notation (e.g., "ij,jk->ik" for matmul).</param>
    /// <param name="tensors">The input tensors.</param>
    /// <returns>The result of the einsum operation.</returns>
    /// <remarks>
    /// <para>
    /// Einsum is a powerful notation for expressing tensor operations.
    /// Common patterns:
    /// - "ij,jk->ik": matrix multiplication
    /// - "bij,bjk->bik": batched matrix multiplication
    /// - "bijk,bkl->bijl": batched tensor contraction
    /// - "bi,bj->bij": batched outer product
    /// </para>
    /// </remarks>
    Tensor<T> TensorEinsum<T>(string subscripts, params Tensor<T>[] tensors);

    /// <summary>
    /// Adds a scalar to all elements of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="scalar">The scalar to add.</param>
    /// <returns>A tensor with the scalar added to all elements.</returns>
    Tensor<T> TensorAddScalar<T>(Tensor<T> tensor, T scalar);

    /// <summary>
    /// Subtracts a scalar from all elements of a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="scalar">The scalar to subtract.</param>
    /// <returns>A tensor with the scalar subtracted from all elements.</returns>
    Tensor<T> TensorSubtractScalar<T>(Tensor<T> tensor, T scalar);

    /// <summary>
    /// Divides all elements of a tensor by a scalar.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="scalar">The scalar divisor.</param>
    /// <returns>A tensor with all elements divided by the scalar.</returns>
    Tensor<T> TensorDivideScalar<T>(Tensor<T> tensor, T scalar);

    /// <summary>
    /// Applies the hyperbolic tangent derivative element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tanhOutput">The output of tanh (not the input).</param>
    /// <returns>A tensor containing 1 - tanh(x)^2 for each element.</returns>
    /// <remarks>
    /// <para>
    /// Given y = tanh(x), the derivative is 1 - y^2.
    /// This takes the tanh output directly to avoid recomputation.
    /// </para>
    /// </remarks>
    Tensor<T> TanhDerivative<T>(Tensor<T> tanhOutput);

    /// <summary>
    /// Applies the sigmoid derivative element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="sigmoidOutput">The output of sigmoid (not the input).</param>
    /// <returns>A tensor containing sigmoid(x) * (1 - sigmoid(x)) for each element.</returns>
    /// <remarks>
    /// <para>
    /// Given y = sigmoid(x), the derivative is y * (1 - y).
    /// This takes the sigmoid output directly to avoid recomputation.
    /// </para>
    /// </remarks>
    Tensor<T> SigmoidDerivative<T>(Tensor<T> sigmoidOutput);

    /// <summary>
    /// Applies the ReLU derivative element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">The original input to ReLU.</param>
    /// <returns>A tensor containing 1 where input > 0, else 0.</returns>
    Tensor<T> ReLUDerivative<T>(Tensor<T> input);

    /// <summary>
    /// Creates a triangular mask tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="size">The size of the square mask.</param>
    /// <param name="upper">If true, creates upper triangular; otherwise lower triangular.</param>
    /// <param name="diagonal">Offset from the main diagonal (0 = include diagonal).</param>
    /// <returns>A tensor with 1s in the triangular region and 0s elsewhere.</returns>
    /// <remarks>
    /// <para>
    /// Useful for causal attention masks without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorTriangularMask<T>(int size, bool upper = false, int diagonal = 0);

    /// <summary>
    /// Applies element-wise squash function for capsule networks.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to compute the norm.</param>
    /// <returns>A tensor with vectors squashed to have magnitude in [0, 1).</returns>
    /// <remarks>
    /// <para>
    /// Implements squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)
    /// Used in capsule networks to ensure output vectors have bounded magnitude.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSquash<T>(Tensor<T> tensor, int axis = -1);

    /// <summary>
    /// Computes the backward pass for squash function.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="gradOutput">The gradient of the loss with respect to squash output.</param>
    /// <param name="input">The original input to squash.</param>
    /// <param name="output">The output of squash.</param>
    /// <param name="axis">The axis along which squash was computed.</param>
    /// <returns>The gradient with respect to the input.</returns>
    Tensor<T> TensorSquashBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis = -1);

    /// <summary>
    /// Computes the L2 norm along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to compute the norm.</param>
    /// <param name="keepDims">Whether to keep the reduced dimension.</param>
    /// <returns>A tensor containing the L2 norms.</returns>
    Tensor<T> TensorNorm<T>(Tensor<T> tensor, int axis, bool keepDims = false);

    /// <summary>
    /// Normalizes vectors along an axis to unit length.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to normalize.</param>
    /// <param name="epsilon">Small value to prevent division by zero.</param>
    /// <returns>A tensor with vectors normalized to unit length.</returns>
    Tensor<T> TensorNormalize<T>(Tensor<T> tensor, int axis, T epsilon);

    /// <summary>
    /// Clips tensor values to a range. This is an alias for <see cref="TensorClamp{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="minValue">The minimum value.</param>
    /// <param name="maxValue">The maximum value.</param>
    /// <returns>A tensor with values clipped to [minValue, maxValue].</returns>
    /// <remarks>
    /// <para>
    /// This method provides the same functionality as <see cref="TensorClamp{T}"/>.
    /// Both "clip" and "clamp" are common names for the same operation (min(max(x, min), max)).
    /// </para>
    /// </remarks>
    Tensor<T> TensorClip<T>(Tensor<T> tensor, T minValue, T maxValue);

    /// <summary>
    /// Creates a tensor by concatenating tensors along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">The tensors to concatenate.</param>
    /// <param name="axis">The axis along which to concatenate.</param>
    /// <returns>A tensor containing the concatenated tensors.</returns>
    Tensor<T> TensorConcatenate<T>(Tensor<T>[] tensors, int axis = 0);

    /// <summary>
    /// Splits a tensor into multiple tensors along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to split.</param>
    /// <param name="numSplits">The number of equal splits.</param>
    /// <param name="axis">The axis along which to split.</param>
    /// <returns>An array of tensors.</returns>
    Tensor<T>[] TensorSplit<T>(Tensor<T> tensor, int numSplits, int axis = 0);

    /// <summary>
    /// Creates a one-hot encoded tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of the output tensor elements.</typeparam>
    /// <param name="indices">The indices tensor (must be integer values).</param>
    /// <param name="depth">The number of classes (size of one-hot dimension).</param>
    /// <returns>A tensor with one-hot encoding.</returns>
    /// <remarks>
    /// <para>
    /// For indices [0, 2, 1] and depth 3:
    /// [[1,0,0], [0,0,1], [0,1,0]]
    /// </para>
    /// <para>
    /// Note: This is a breaking API change. Indices must now be Tensor&lt;int&gt; for type safety.
    /// </para>
    /// </remarks>
    Tensor<T> TensorOneHot<T>(Tensor<int> indices, int depth);

    /// <summary>
    /// Computes argmax along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of input tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to find the maximum index.</param>
    /// <returns>A tensor containing the integer indices of maximum values.</returns>
    /// <remarks>
    /// <para>
    /// Note: This is a breaking API change. Return type is now Tensor&lt;int&gt; for type safety.
    /// </para>
    /// </remarks>
    Tensor<int> TensorArgMax<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Computes argmin along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of input tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to find the minimum index.</param>
    /// <returns>A tensor containing the integer indices of minimum values.</returns>
    /// <remarks>
    /// <para>
    /// Note: This is a breaking API change. Return type is now Tensor&lt;int&gt; for type safety.
    /// </para>
    /// </remarks>
    Tensor<int> TensorArgMin<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Computes binary cross-entropy loss element-wise.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predictions">The predicted probabilities (0-1).</param>
    /// <param name="targets">The target values (0 or 1).</param>
    /// <param name="epsilon">Small value for numerical stability.</param>
    /// <returns>A tensor containing the loss for each element.</returns>
    Tensor<T> TensorBinaryCrossEntropy<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon);

    /// <summary>
    /// Computes the backward pass for binary cross-entropy.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="predictions">The predicted probabilities.</param>
    /// <param name="targets">The target values.</param>
    /// <param name="epsilon">Small value for numerical stability.</param>
    /// <returns>The gradient with respect to predictions.</returns>
    Tensor<T> TensorBinaryCrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon);

    /// <summary>
    /// Creates coordinate meshgrids from 1D coordinate arrays.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="x">1D array of x coordinates [width].</param>
    /// <param name="y">1D array of y coordinates [height].</param>
    /// <returns>A tuple of (X, Y) grids, each with shape [height, width].</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.meshgrid() or torch.meshgrid().
    /// Creates 2D coordinate grids from 1D coordinate arrays.
    /// Useful for spatial transformer networks and coordinate-based operations.
    /// </para>
    /// </remarks>
    (Tensor<T> X, Tensor<T> Y) TensorMeshgrid<T>(Tensor<T> x, Tensor<T> y);

    /// <summary>
    /// Extracts a slice along a specific axis from a 3D tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor [dim0, dim1, dim2].</param>
    /// <param name="axis">The axis to slice (0, 1, or 2).</param>
    /// <param name="index">The index to extract along the axis.</param>
    /// <returns>A 2D tensor with the specified slice.</returns>
    /// <remarks>
    /// <para>
    /// For a 3D tensor [H, W, C], slicing along axis 2 with index i gives [H, W] at channel i.
    /// This is useful for extracting channels from multi-channel tensors without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSliceAxis<T>(Tensor<T> tensor, int axis, int index);

    /// <summary>
    /// Creates a tensor filled with values from a linear range.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="start">The starting value of the range.</param>
    /// <param name="end">The ending value of the range (exclusive).</param>
    /// <param name="count">The number of values in the range.</param>
    /// <returns>A 1D tensor with linearly spaced values from start to end.</returns>
    /// <remarks>
    /// <para>
    /// Similar to numpy.linspace() or torch.linspace().
    /// Useful for creating coordinate ranges without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorLinspace<T>(T start, T end, int count);

    /// <summary>
    /// Batched matrix multiplication for 3D tensors. This is an alias for <see cref="BatchMatMul{T}"/>.
    /// Computes batched matrix multiply: result[b] = a[b] @ b[b] for each batch.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">First tensor of shape [batch, M, K].</param>
    /// <param name="b">Second tensor of shape [batch, K, N] or [K, N] for broadcasting.</param>
    /// <returns>Result tensor of shape [batch, M, N].</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.bmm() or np.matmul() with 3D tensors.
    /// Essential for RNN/LSTM/GRU vectorization where we compute all timesteps at once.
    /// If b is 2D [K, N], it broadcasts across the batch dimension.
    /// </para>
    /// <para>
    /// This method provides the same functionality as <see cref="BatchMatMul{T}"/>.
    /// The "Tensor" prefix variant exists for API consistency with other tensor operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorBatchMatMul<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Sets a slice of a tensor along a specific axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="destination">The destination tensor to modify.</param>
    /// <param name="source">The source tensor to copy from.</param>
    /// <param name="axis">The axis along which to set the slice.</param>
    /// <param name="index">The index at which to set the slice.</param>
    /// <remarks>
    /// <para>
    /// Inverse of TensorSliceAxis. Sets destination[..., index, ...] = source along the specified axis.
    /// Essential for building output tensors without loops.
    /// </para>
    /// </remarks>
    void TensorSetSliceAxis<T>(Tensor<T> destination, Tensor<T> source, int axis, int index);

    /// <summary>
    /// Applies softmax along a specified axis. This is an alias for <see cref="Softmax{T}(Tensor{T}, int)"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to apply softmax.</param>
    /// <returns>A tensor with softmax applied along the specified axis.</returns>
    /// <remarks>
    /// <para>
    /// Computes softmax(x)_i = exp(x_i) / sum(exp(x_j)) along the specified axis.
    /// Numerically stable implementation subtracts max before exp.
    /// </para>
    /// <para>
    /// This method provides the same functionality as <see cref="Softmax{T}(Tensor{T}, int)"/>.
    /// The "Tensor" prefix variant exists for API consistency with other tensor operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSoftmax<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Computes the backward pass for softmax. This is an alias for <see cref="SoftmaxBackward{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="softmaxOutput">The output from the forward softmax pass.</param>
    /// <param name="outputGradient">The gradient flowing back.</param>
    /// <param name="axis">The axis along which softmax was applied.</param>
    /// <returns>The gradient with respect to the softmax input.</returns>
    /// <remarks>
    /// <para>
    /// This method provides the same functionality as <see cref="SoftmaxBackward{T}"/>.
    /// The "Tensor" prefix variant exists for API consistency with other tensor operations.
    /// </para>
    /// </remarks>
    Tensor<T> TensorSoftmaxBackward<T>(Tensor<T> softmaxOutput, Tensor<T> outputGradient, int axis);

    /// <summary>
    /// Computes log-softmax along a specified axis (more numerically stable than log(softmax(x))).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="axis">The axis along which to apply log-softmax.</param>
    /// <returns>A tensor with log-softmax applied along the specified axis.</returns>
    Tensor<T> TensorLogSoftmax<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Top-K selection along an axis, returning both values and indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="k">Number of top elements to select.</param>
    /// <param name="axis">The axis along which to select.</param>
    /// <param name="indices">Output tensor containing indices of top-k elements.</param>
    /// <returns>A tensor containing the top-k values.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.topk(). Essential for MoE gating without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorTopK<T>(Tensor<T> tensor, int k, int axis, out Tensor<int> indices);

    /// <summary>
    /// Scatter operation: sets values at specified indices along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="destination">The destination tensor (modified in place or returns new tensor).</param>
    /// <param name="indices">Integer indices where to scatter values.</param>
    /// <param name="source">Values to scatter.</param>
    /// <param name="axis">The axis along which to scatter.</param>
    /// <returns>The tensor with scattered values.</returns>
    Tensor<T> TensorScatter<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> source, int axis);

    /// <summary>
    /// Index select: gathers slices from a tensor along an axis using integer indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The source tensor.</param>
    /// <param name="indices">Integer indices to select.</param>
    /// <param name="axis">The axis along which to select.</param>
    /// <returns>Selected slices concatenated along the axis.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.index_select(). Essential for embedding lookups and expert selection.
    /// </para>
    /// </remarks>
    Tensor<T> TensorIndexSelect<T>(Tensor<T> tensor, Tensor<int> indices, int axis);

    /// <summary>
    /// Stacks multiple tensors along a new axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensors">Array of tensors to stack (must have same shape).</param>
    /// <param name="axis">The axis along which to stack.</param>
    /// <returns>A new tensor with an additional dimension.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.stack() or np.stack().
    /// Essential for building batch tensors without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorStack<T>(Tensor<T>[] tensors, int axis);

    /// <summary>
    /// Unstacks a tensor along an axis into multiple tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The tensor to unstack.</param>
    /// <param name="axis">The axis along which to unstack.</param>
    /// <returns>An array of tensors.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.unbind() or tf.unstack().
    /// Inverse of TensorStack.
    /// </para>
    /// </remarks>
    Tensor<T>[] TensorUnstack<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Applies a function element-wise to a tensor (vectorized map).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="func">The function to apply to each element.</param>
    /// <returns>A tensor with the function applied to each element.</returns>
    /// <remarks>
    /// <para>
    /// This enables custom element-wise operations without explicit loops.
    /// The implementation should be parallelized internally.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMap<T>(Tensor<T> tensor, Func<T, T> func);

    /// <summary>
    /// Masked fill: fills tensor elements with a value where mask is true.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">The input tensor.</param>
    /// <param name="mask">Boolean mask tensor (must have the same shape as input).</param>
    /// <param name="value">The value to fill where mask is true.</param>
    /// <returns>A tensor with masked positions filled.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.masked_fill(). Essential for attention masking.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMaskedFill<T>(Tensor<T> tensor, Tensor<bool> mask, T value);

    /// <summary>
    /// Masked fill using a <see cref="Bit"/> mask tensor. A Bit.True entry fills the corresponding
    /// position in the output with <paramref name="value"/>; Bit.False leaves it unchanged.
    /// </summary>
    Tensor<T> TensorMaskedFill<T>(Tensor<T> tensor, Tensor<Bit> mask, T value);

    /// <summary>
    /// Masked select: returns a 1D tensor containing elements of
    /// <paramref name="tensor"/> at positions where <paramref name="mask"/> is
    /// <see cref="Bit.True"/>. Mirrors <c>torch.masked_select</c>.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Source tensor.</param>
    /// <param name="mask">Bit-packed mask of the same shape as <paramref name="tensor"/>.</param>
    /// <returns>A 1-D tensor of length <c>count(mask == Bit.True)</c>.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: picks the elements of <paramref name="tensor"/> whose
    /// corresponding <paramref name="mask"/> position is set, and lays them
    /// out as a flat vector. Commonly used to extract the "valid" part of a
    /// padded sequence or the non-zero entries of a sparse activation.
    /// </para>
    /// <para>Backward: gradient is scattered back to the original shape —
    /// non-masked positions receive zero, masked positions receive the
    /// incoming flat gradient.</para>
    /// <para>
    /// We use <see cref="Bit"/> (bit-packed) rather than <c>bool</c> so the
    /// mask tensor costs 1 bit per position instead of a full byte — a
    /// noticeable win on attention masks and long sequences. PyTorch stores
    /// masks as full <c>bool</c> tensors.
    /// </para>
    /// </remarks>
    Tensor<T> TensorMaskedSelect<T>(Tensor<T> tensor, Tensor<Bit> mask);

    /// <summary>Rolls elements along the given axes with wrap-around (torch.roll).</summary>
    Tensor<T> TensorRoll<T>(Tensor<T> tensor, int[] shifts, int[] axes);

    /// <summary>Reverses the tensor along the given axes (torch.flip).</summary>
    Tensor<T> TensorFlip<T>(Tensor<T> tensor, int[] axes);

    /// <summary>Repeats each element along <paramref name="dim"/> <paramref name="repeats"/> times (torch.repeat_interleave).</summary>
    Tensor<T> TensorRepeatInterleave<T>(Tensor<T> tensor, int repeats, int dim);

    /// <summary>Flip along the last axis (torch.fliplr).</summary>
    Tensor<T> TensorFliplr<T>(Tensor<T> tensor);

    /// <summary>Flip along the first axis (torch.flipud).</summary>
    Tensor<T> TensorFlipud<T>(Tensor<T> tensor);

    /// <summary>
    /// Rotate 90° in the plane spanned by two axes (torch.rot90).
    /// <paramref name="k"/> counts 90° turns (can be negative).
    /// </summary>
    Tensor<T> TensorRot90<T>(Tensor<T> tensor, int k = 1, int[]? axes = null);

    /// <summary>Swap two dimensions (torch.swapaxes / numpy.swapaxes).</summary>
    Tensor<T> TensorSwapAxes<T>(Tensor<T> tensor, int axis1, int axis2);

    /// <summary>
    /// Move a single dimension from <paramref name="source"/> to
    /// <paramref name="destination"/> (torch.movedim).
    /// </summary>
    Tensor<T> TensorMoveDim<T>(Tensor<T> tensor, int source, int destination);

    /// <summary>Promote to rank ≥1 (torch.atleast_1d).</summary>
    Tensor<T> TensorAtLeast1D<T>(Tensor<T> tensor);

    /// <summary>
    /// General tensor contraction along arbitrary axes (torch.tensordot).
    /// Internally builds an einsum equation and dispatches to TensorEinsum,
    /// so it inherits the greedy path optimizer and fast-path routing.
    /// </summary>
    Tensor<T> TensorDot<T>(Tensor<T> a, Tensor<T> b, int[] axesA, int[] axesB);

    /// <summary>
    /// Fused matmul + bias add with default α = β = 1: <c>A · B + input</c>
    /// (torch.addmm).
    /// </summary>
    Tensor<T> TensorAddMM<T>(Tensor<T> input, Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Fused matmul + bias add with explicit scalars: <c>α · (A · B) + β · input</c>.
    /// </summary>
    Tensor<T> TensorAddMM<T>(Tensor<T> input, Tensor<T> a, Tensor<T> b, T alpha, T beta);

    /// <summary>1-D vector inner product Σ a[i]·b[i] (torch.linalg.vecdot).</summary>
    T TensorVecDot<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Sum of the main diagonal of a 2-D tensor (torch.trace).</summary>
    T TensorTrace<T>(Tensor<T> tensor);

    /// <summary>
    /// Place the last-dim values on the diagonal of a new (rank+1) tensor
    /// (torch.diag_embed). <paramref name="offset"/> shifts the diagonal
    /// (positive = super-diagonal, negative = sub-diagonal). Output's last
    /// two dims both have length <c>L + |offset|</c>, where <c>L</c> is the
    /// input's last-dim size.
    /// </summary>
    Tensor<T> TensorDiagEmbed<T>(Tensor<T> tensor, int offset = 0);

    /// <summary>
    /// Vector cross product a × b along <paramref name="dim"/> (torch.cross /
    /// torch.linalg.cross). The named dim must have size 3.
    /// </summary>
    Tensor<T> TensorCross<T>(Tensor<T> a, Tensor<T> b, int dim = -1);

    /// <summary>
    /// Build coordinate grids from 1-D input tensors (torch.meshgrid).
    /// <paramref name="indexing"/> is "ij" (default, matrix-style) or "xy"
    /// (Cartesian, swaps the first two output axes).
    /// </summary>
    Tensor<T>[] TensorMeshgrid<T>(Tensor<T>[] tensors, string indexing = "ij");

    /// <summary>
    /// Cartesian product of N 1-D tensors (torch.cartesian_prod). Output
    /// shape is [∏ lengths, N]; each row is one combination.
    /// </summary>
    Tensor<T> TensorCartesianProd<T>(Tensor<T>[] tensors);

    /// <summary>
    /// Kronecker product of 2-D matrices (torch.kron). Result shape is
    /// <c>(m·p) × (n·q)</c> for inputs <c>A ∈ ℝ^{m×n}, B ∈ ℝ^{p×q}</c>.
    /// Rank-1 inputs are promoted to <c>(1, len)</c>.
    /// </summary>
    Tensor<T> TensorKron<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Inner product contracting the last axis of both operands (torch.inner).
    /// Output shape is <c>a.shape[:-1] + b.shape[:-1]</c>.
    /// </summary>
    Tensor<T> TensorInner<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Pairwise p-norm distances over the N rows of a [N, D] matrix
    /// (torch.pdist). Output is a 1-D tensor of length N·(N−1)/2 ordered
    /// (0,1), (0,2), …, (N−2, N−1).
    /// </summary>
    Tensor<T> TensorPDist<T>(Tensor<T> input, double p = 2.0);

    /// <summary>
    /// Cosine similarity along <paramref name="dim"/> with an
    /// epsilon-clamped denominator (torch.nn.functional.cosine_similarity).
    /// </summary>
    Tensor<T> TensorCosineSimilarity<T>(Tensor<T> x1, Tensor<T> x2, int dim = -1, double eps = 1e-8);

    /// <summary>
    /// Cross pairwise distance: output[i, j] = ‖x1[i] − x2[j]‖_p
    /// (torch.cdist).
    /// </summary>
    Tensor<T> TensorCDist<T>(Tensor<T> x1, Tensor<T> x2, double p = 2.0);

    /// <summary>Promote to rank ≥2 (torch.atleast_2d).</summary>
    Tensor<T> TensorAtLeast2D<T>(Tensor<T> tensor);

    /// <summary>Promote to rank ≥3 (torch.atleast_3d).</summary>
    Tensor<T> TensorAtLeast3D<T>(Tensor<T> tensor);

    /// <summary>Horizontal stack (torch.hstack): concat along axis 0 for 1D, axis 1 for ≥2D.</summary>
    Tensor<T> TensorHStack<T>(Tensor<T>[] tensors);

    /// <summary>Vertical stack (torch.vstack): 1D tensors become rows; then concat along axis 0.</summary>
    Tensor<T> TensorVStack<T>(Tensor<T>[] tensors);

    /// <summary>Depth stack (torch.dstack): promotes to ≥3D and concats along axis 2.</summary>
    Tensor<T> TensorDStack<T>(Tensor<T>[] tensors);

    /// <summary>Column stack (torch.column_stack): 1D tensors become columns; ≥2D concat along axis 1.</summary>
    Tensor<T> TensorColumnStack<T>(Tensor<T>[] tensors);

    /// <summary>Row stack (torch.row_stack), alias for vstack.</summary>
    Tensor<T> TensorRowStack<T>(Tensor<T>[] tensors);

    /// <summary>Horizontal split (torch.hsplit).</summary>
    Tensor<T>[] TensorHSplit<T>(Tensor<T> tensor, int sections);

    /// <summary>Vertical split (torch.vsplit).</summary>
    Tensor<T>[] TensorVSplit<T>(Tensor<T> tensor, int sections);

    /// <summary>Depth split (torch.dsplit).</summary>
    Tensor<T>[] TensorDSplit<T>(Tensor<T> tensor, int sections);

    /// <summary>Broadcast <paramref name="tensor"/> to <paramref name="shape"/> (torch.broadcast_to).</summary>
    Tensor<T> TensorBroadcastTo<T>(Tensor<T> tensor, int[] shape);

    /// <summary>
    /// Pick elements from the flattened tensor at the positions specified by
    /// <paramref name="indices"/> (torch.take). Output shape matches
    /// <paramref name="indices"/>.
    /// </summary>
    Tensor<T> TensorTake<T>(Tensor<T> tensor, Tensor<int> indices);

    /// <summary>
    /// Gather along <paramref name="dim"/>: for every non-<paramref name="dim"/>
    /// position, index into the source via the matching position in
    /// <paramref name="indices"/> (torch.take_along_dim).
    /// </summary>
    Tensor<T> TensorTakeAlongDim<T>(Tensor<T> tensor, Tensor<int> indices, int dim);

    /// <summary>Cumulative product along <paramref name="axis"/> (torch.cumprod).</summary>
    Tensor<T> TensorCumProd<T>(Tensor<T> tensor, int axis);

    /// <summary>Cumulative max along <paramref name="axis"/> (torch.cummax — values only here).</summary>
    Tensor<T> TensorCumMax<T>(Tensor<T> tensor, int axis);

    /// <summary>Cumulative min along <paramref name="axis"/> (torch.cummin — values only here).</summary>
    Tensor<T> TensorCumMin<T>(Tensor<T> tensor, int axis);

    /// <summary>Cumulative log-sum-exp along <paramref name="axis"/> (torch.logcumsumexp).</summary>
    Tensor<T> TensorLogCumSumExp<T>(Tensor<T> tensor, int axis);

    /// <summary>
    /// Element-wise closeness check: |a − b| ≤ atol + rtol · |b|, with optional
    /// NaN-equal semantics. Returns a bit-packed mask to save memory.
    /// </summary>
    Tensor<Bit> TensorIsClose<T>(Tensor<T> a, Tensor<T> b, T rtol, T atol, bool equalNan = false);

    /// <summary>True if every element is close (torch.allclose).</summary>
    bool TensorAllClose<T>(Tensor<T> a, Tensor<T> b, T rtol, T atol, bool equalNan = false);

    /// <summary>Element-wise test for set membership (torch.isin).</summary>
    Tensor<Bit> TensorIsIn<T>(Tensor<T> elements, Tensor<T> testElements, bool invert = false);

    /// <summary>Element-wise test for finite values — neither NaN nor ±∞ (torch.isfinite).</summary>
    Tensor<Bit> TensorIsFinite<T>(Tensor<T> tensor);

    /// <summary>
    /// Replace NaN / +∞ / −∞ with finite substitutes (torch.nan_to_num).
    /// Null parameters fall back to the PyTorch defaults: NaN → 0,
    /// +∞ → dtype-max, −∞ → dtype-min (approximated here via double extremes
    /// then saturated by the numeric ops).
    /// </summary>
    Tensor<T> TensorNanToNum<T>(Tensor<T> tensor, double? nan = null, double? posinf = null, double? neginf = null);

    /// <summary>Element-wise NaN test (torch.isnan).</summary>
    Tensor<Bit> TensorIsNan<T>(Tensor<T> tensor);

    /// <summary>Element-wise ±∞ test (torch.isinf).</summary>
    Tensor<Bit> TensorIsInf<T>(Tensor<T> tensor);

    /// <summary>
    /// Sub-byte packed Gather over a byte-buffered packed tensor
    /// (int1 / int2 / int4 / NF4 / FP4 — storage rides inside
    /// <see cref="Tensor{Byte}"/>). Gathers rows directly in the packed
    /// domain — no dequantisation needed. PyTorch forces dequant → gather →
    /// requant for the same workload. The gather axis must not be the
    /// last axis when <paramref name="valuesPerByte"/> &gt; 1 (crosses the
    /// packing boundary).
    /// </summary>
    /// <param name="valuesPerByte">1 (plain byte), 2 (int4/NF4/FP4), 4 (int2), 8 (int1/BitNet).</param>
    Tensor<byte> TensorGatherPacked(
        Tensor<byte> packed, Tensor<int> indices, int axis, int valuesPerByte);

    /// <summary>
    /// Sub-byte packed Scatter — inverse of <see cref="TensorGatherPacked"/>.
    /// </summary>
    Tensor<byte> TensorScatterPacked(
        Tensor<byte> packed, Tensor<int> indices, Tensor<byte> source, int axis, int valuesPerByte);


    /// <summary>Element-wise logical AND on bit-packed masks (torch.logical_and).</summary>
    Tensor<Bit> TensorLogicalAnd(Tensor<Bit> a, Tensor<Bit> b);

    /// <summary>Element-wise logical OR on bit-packed masks (torch.logical_or).</summary>
    Tensor<Bit> TensorLogicalOr(Tensor<Bit> a, Tensor<Bit> b);

    /// <summary>Element-wise logical XOR on bit-packed masks (torch.logical_xor).</summary>
    Tensor<Bit> TensorLogicalXor(Tensor<Bit> a, Tensor<Bit> b);

    /// <summary>Element-wise logical NOT on a bit-packed mask (torch.logical_not).</summary>
    Tensor<Bit> TensorLogicalNot(Tensor<Bit> a);

    /// <summary>
    /// Upper-triangular fill: keep elements where (col − row) ≥ <paramref name="diagonal"/>;
    /// zero everything else. Works on the last two dims; batch-preserving
    /// (torch.triu).
    /// </summary>
    Tensor<T> TensorTriu<T>(Tensor<T> tensor, int diagonal = 0);

    /// <summary>
    /// Lower-triangular fill: keep elements where (col − row) ≤ <paramref name="diagonal"/>;
    /// zero everything else (torch.tril).
    /// </summary>
    Tensor<T> TensorTril<T>(Tensor<T> tensor, int diagonal = 0);

    /// <summary>
    /// Return coordinates of nonzero elements as a [N, rank] int tensor
    /// (torch.nonzero). Ordered row-major over the input.
    /// </summary>
    Tensor<int> TensorNonzero<T>(Tensor<T> tensor);

    /// <summary>Count nonzero elements in the flattened tensor (torch.count_nonzero).</summary>
    int TensorCountNonzero<T>(Tensor<T> tensor);

    /// <summary>Element-wise <c>max(x, min)</c> (torch.clamp_min).</summary>
    Tensor<T> TensorClampMin<T>(Tensor<T> tensor, T min);

    /// <summary>Element-wise <c>min(x, max)</c> (torch.clamp_max).</summary>
    Tensor<T> TensorClampMax<T>(Tensor<T> tensor, T max);

    /// <summary>
    /// Single-pass element-wise min + max (torch.aminmax). Faster than calling
    /// ReduceMin and ReduceMax separately because it visits memory once.
    /// </summary>
    (T Min, T Max) TensorAminmax<T>(Tensor<T> tensor);

    /// <summary>
    /// Element-wise <c>clamp</c> with tensor-valued bounds (torch.clamp with
    /// tensor min/max). Either bound may be <c>null</c>. Current v1 requires
    /// exact-shape bounds; a broadcasting overload is a follow-up.
    /// </summary>
    Tensor<T> TensorClampTensor<T>(Tensor<T> tensor, Tensor<T>? min, Tensor<T>? max);

    /// <summary>
    /// Write a (rank-1 smaller) slice into <paramref name="tensor"/> at a
    /// single axis position (torch.select_scatter).
    /// </summary>
    Tensor<T> TensorSelectScatter<T>(Tensor<T> tensor, Tensor<T> source, int dim, int index);

    /// <summary>Element-wise sqrt(a² + b²) without under/overflow (torch.hypot).</summary>
    Tensor<T> TensorHypot<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Copies the sign of <paramref name="b"/> onto the magnitude of <paramref name="a"/> (torch.copysign).</summary>
    Tensor<T> TensorCopysign<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>IEEE truncation-toward-zero remainder; result has the sign of <paramref name="a"/> (torch.fmod).</summary>
    Tensor<T> TensorFmod<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Python-style modulo; result has the sign of <paramref name="b"/> (torch.remainder).</summary>
    Tensor<T> TensorRemainder<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Element-wise power computed in floating point (torch.float_power).</summary>
    Tensor<T> TensorFloatPower<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Numerically-stable <c>log(exp(a) + exp(b))</c> (torch.logaddexp). Uses the
    /// max-shift trick so no overflow for large inputs.
    /// </summary>
    Tensor<T> TensorLogAddExp<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Same as LogAddExp but in base-2 (torch.logaddexp2).</summary>
    Tensor<T> TensorLogAddExp2<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Element-wise <c>x · 2^exp</c> (torch.ldexp). <paramref name="exp"/> must
    /// be integer and broadcast to the same shape as <paramref name="x"/>.
    /// </summary>
    Tensor<T> TensorLdexp<T>(Tensor<T> x, Tensor<int> exp);

    /// <summary>
    /// Element-wise next representable floating-point value after
    /// <paramref name="a"/> toward <paramref name="b"/> (torch.nextafter).
    /// </summary>
    Tensor<T> TensorNextAfter<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Flat-indexed scatter (torch.put). Writes <paramref name="source"/>[i]
    /// into <paramref name="tensor"/> at flat position <paramref name="indices"/>[i].
    /// The inverse of <see cref="TensorTake{T}"/>.
    /// </summary>
    Tensor<T> TensorPut<T>(Tensor<T> tensor, Tensor<int> indices, Tensor<T> source);

    /// <summary>Complementary error function 1 − erf(x) (torch.special.erfc).</summary>
    Tensor<T> TensorErfc<T>(Tensor<T> tensor);

    /// <summary>x · log(y), with 0·log(y) = 0 by convention (torch.special.xlogy).</summary>
    Tensor<T> TensorXlogy<T>(Tensor<T> x, Tensor<T> y);

    /// <summary>x · log(1 + y), with 0·log(…) = 0 by convention (torch.special.xlog1py).</summary>
    Tensor<T> TensorXlog1py<T>(Tensor<T> x, Tensor<T> y);

    /// <summary>Log of the absolute value of the gamma function (torch.special.gammaln / torch.lgamma).</summary>
    Tensor<T> TensorLgamma<T>(Tensor<T> tensor);

    /// <summary>
    /// Digamma function ψ(x) = Γ'(x)/Γ(x) (torch.special.digamma). Uses
    /// asymptotic series with recurrence shift; accuracy ~1e-5 on fp32 in
    /// the common range x ∈ [0.1, 100].
    /// </summary>
    Tensor<T> TensorDigamma<T>(Tensor<T> tensor);

    /// <summary>
    /// Inverse error function (torch.special.erfinv). Winitzki seed +
    /// 2 Newton iterations — ~7-digit accuracy.
    /// </summary>
    Tensor<T> TensorErfinv<T>(Tensor<T> tensor);

    /// <summary>
    /// Polygamma function ψ^{(n)}(x) (torch.special.polygamma). v1 supports
    /// n=0 (alias for digamma) and n=1 (trigamma). Higher orders throw
    /// NotImplementedException until the Hurwitz zeta path lands.
    /// </summary>
    Tensor<T> TensorPolygamma<T>(int n, Tensor<T> tensor);

    /// <summary>Modified Bessel function of the first kind, order 0 (torch.special.i0).</summary>
    Tensor<T> TensorI0<T>(Tensor<T> tensor);

    /// <summary>Modified Bessel function of the first kind, order 1 (torch.special.i1).</summary>
    Tensor<T> TensorI1<T>(Tensor<T> tensor);

    /// <summary>Exponentially-scaled I₀: e^(-|x|) · I₀(x) (torch.special.i0e). Safe for large x.</summary>
    Tensor<T> TensorI0e<T>(Tensor<T> tensor);

    /// <summary>Exponentially-scaled I₁: e^(-|x|) · I₁(x) (torch.special.i1e).</summary>
    Tensor<T> TensorI1e<T>(Tensor<T> tensor);

    /// <summary>
    /// Decompose floating-point values into mantissa ∈ [0.5, 1) and integer
    /// exponent such that x = mantissa · 2^exp (torch.frexp). Zero maps to
    /// (0, 0).
    /// </summary>
    (Tensor<T> Mantissa, Tensor<int> Exponent) TensorFrexp<T>(Tensor<T> tensor);

    /// <summary>
    /// Sort along an axis; returns both the sorted values and the permutation
    /// indices (torch.sort). Ascending by default, descending when
    /// <paramref name="descending"/> is true.
    /// </summary>
    (Tensor<T> Values, Tensor<int> Indices) TensorSort<T>(Tensor<T> input, int axis = -1, bool descending = false);

    /// <summary>
    /// Returns the indices that would sort the tensor along the given axis
    /// (torch.argsort). Ascending by default, descending when
    /// <paramref name="descending"/> is true.
    /// </summary>
    Tensor<int> TensorArgsort<T>(Tensor<T> input, int axis = -1, bool descending = false);

    /// <summary>
    /// Hurwitz zeta function ζ(x, q) = Σ_{k=0}^∞ 1 / (k + q)^x.
    /// When q is omitted (passed as 1), this reduces to the Riemann zeta.
    /// Matches <c>torch.special.zeta(x, q)</c>.
    /// </summary>
    Tensor<T> TensorZeta<T>(Tensor<T> x, Tensor<T> q);

    /// <summary>
    /// Sliding-window unfold (<c>torch.Tensor.unfold</c>). Slides a window
    /// of length <paramref name="size"/> along <paramref name="dim"/> with
    /// stride <paramref name="step"/>, replacing <c>shape[dim]</c> with
    /// <c>(shape[dim] - size) / step + 1</c> and appending a new trailing
    /// axis of length <paramref name="size"/>.
    /// </summary>
    Tensor<T> TensorUnfold<T>(Tensor<T> tensor, int dim, int size, int step);

    /// <summary>
    /// Returns the <paramref name="k"/>-th smallest value of the flattened
    /// tensor with its flat index. <paramref name="k"/> is 1-based
    /// (torch.kthvalue convention).
    /// </summary>
    (T Value, int Index) TensorKthvalue<T>(Tensor<T> input, int k);

    /// <summary>
    /// Median of the flattened tensor (torch.median). For even length returns
    /// the lower median, matching PyTorch.
    /// </summary>
    T TensorMedian<T>(Tensor<T> input);

    /// <summary>
    /// Unique values of the flattened tensor. When <paramref name="sorted"/>
    /// is true (default), the result is ascending; otherwise the input order
    /// is preserved (first occurrence).
    /// </summary>
    Tensor<T> TensorUnique<T>(Tensor<T> input, bool sorted = true);

    /// <summary>
    /// Branchless binary search. For each value in <paramref name="values"/>,
    /// returns the insertion index into the sorted 1-D <paramref name="sortedSequence"/>
    /// (torch.searchsorted).
    /// </summary>
    Tensor<int> TensorSearchSorted<T>(Tensor<T> sortedSequence, Tensor<T> values, bool right = false);

    /// <summary>
    /// Counts values falling into <paramref name="bins"/> equal-width bins on
    /// <c>[min, max]</c>. Values outside the range are dropped. Mirrors
    /// <c>torch.histc</c>.
    /// </summary>
    Tensor<int> TensorHistogram<T>(Tensor<T> input, int bins, T min, T max);

    /// <summary>
    /// N-dimensional histogram (torch.histogramdd). Input samples have shape
    /// [N, D]; bins, mins, maxs all have length D. Output shape = bins[0] ×
    /// … × bins[D-1] of int counts.
    /// </summary>
    Tensor<int> TensorHistogramDD<T>(Tensor<T> samples, int[] bins, T[] mins, T[] maxs);

    /// <summary>
    /// Count occurrences of each non-negative integer value (torch.bincount).
    /// Output length = max(max(input) + 1, minLength).
    /// </summary>
    Tensor<int> TensorBinCount(Tensor<int> input, int? minLength = null);

    /// <summary>
    /// Chain of matrix multiplications. Builds an einsum expression and
    /// dispatches to TensorEinsum, inheriting the greedy path optimiser —
    /// so a chain like (A · B · C · D) picks an efficient contraction order
    /// (torch.linalg.multi_dot).
    /// </summary>
    Tensor<T> TensorMultiDot<T>(Tensor<T>[] matrices);

    /// <summary>
    /// Median ignoring NaN values (torch.nanmedian). Returns NaN if every
    /// value is NaN. Lower-median convention on even counts.
    /// </summary>
    T TensorNanMedian<T>(Tensor<T> input);

    /// <summary>
    /// Most frequent value in the flattened tensor with its occurrence count
    /// (torch.mode — flattened here; per-axis variant can come later). Ties
    /// broken by smallest-value-wins.
    /// </summary>
    (T Value, int Count) TensorMode<T>(Tensor<T> input);

    /// <summary>
    /// Bucketize values into the bins defined by a sorted 1-D
    /// <paramref name="boundaries"/> tensor (torch.bucketize). Equivalent to
    /// searchsorted with swapped argument order; kept as its own API for
    /// familiarity.
    /// </summary>
    Tensor<int> TensorBucketize<T>(Tensor<T> input, Tensor<T> boundaries, bool right = false);

    /// <summary>
    /// Add values from <paramref name="source"/> into <paramref name="tensor"/>
    /// at positions specified by <paramref name="indices"/> along
    /// <paramref name="axis"/> (torch.index_add). Duplicate indices accumulate.
    /// </summary>
    Tensor<T> TensorIndexAdd<T>(Tensor<T> tensor, int axis, Tensor<int> indices, Tensor<T> source);

    /// <summary>
    /// Fill positions at <paramref name="indices"/> along <paramref name="axis"/>
    /// with <paramref name="value"/> (torch.index_fill).
    /// </summary>
    Tensor<T> TensorIndexFill<T>(Tensor<T> tensor, int axis, Tensor<int> indices, T value);

    /// <summary>
    /// Copy slices from <paramref name="source"/> into <paramref name="tensor"/>
    /// at positions specified by <paramref name="indices"/> along
    /// <paramref name="axis"/> (torch.index_copy). Overwrites instead of
    /// accumulates — duplicate indices keep the last written value.
    /// </summary>
    Tensor<T> TensorIndexCopy<T>(Tensor<T> tensor, int axis, Tensor<int> indices, Tensor<T> source);

    /// <summary>
    /// Full multi-axis scatter (torch.index_put). One 1-D index tensor per
    /// tensor axis; all index tensors must have the same length; source
    /// has that same length. With <paramref name="accumulate"/> = true,
    /// duplicate-index writes sum (matching torch's index_put_(accumulate=True)).
    /// </summary>
    Tensor<T> TensorIndexPut<T>(Tensor<T> tensor, Tensor<int>[] indices, Tensor<T> source, bool accumulate = false);

    /// <summary>Broadcast <paramref name="tensor"/> to the shape of <paramref name="other"/> (torch.expand_as).</summary>
    Tensor<T> TensorExpandAs<T>(Tensor<T> tensor, Tensor<T> other);

    /// <summary>
    /// Broadcast a set of tensors to a common shape (torch.broadcast_tensors).
    /// Returns one output per input, each with the broadcast shape.
    /// </summary>
    Tensor<T>[] TensorBroadcastTensors<T>(Tensor<T>[] tensors);

    /// <summary>
    /// Unique *consecutive* values — only collapses runs of repeated values
    /// (torch.unique_consecutive). Unlike Unique, does not change relative
    /// order or remove non-adjacent repeats.
    /// </summary>
    Tensor<T> TensorUniqueConsecutive<T>(Tensor<T> input);

    /// <summary>
    /// Build a block-diagonal matrix from 2-D matrices (torch.block_diag).
    /// Result shape = (Σ rows) × (Σ cols); off-diagonal blocks are zero.
    /// </summary>
    Tensor<T> TensorBlockDiag<T>(Tensor<T>[] matrices);

    /// <summary>
    /// Overwrite a contiguous slice of <paramref name="tensor"/> along
    /// <paramref name="dim"/> with <paramref name="source"/> (torch.slice_scatter).
    /// </summary>
    Tensor<T> TensorSliceScatter<T>(Tensor<T> tensor, Tensor<T> source, int dim, int start, int length);

    /// <summary>
    /// Scatter elements from <paramref name="source"/> into
    /// <paramref name="tensor"/> at positions where <paramref name="mask"/> is
    /// <see cref="Bit.True"/>, consuming the source in row-major order
    /// (torch.masked_scatter).
    /// </summary>
    Tensor<T> TensorMaskedScatter<T>(Tensor<T> tensor, Tensor<Bit> mask, Tensor<T> source);

    /// <summary>
    /// Scatter with a reduction at each target slot (torch.scatter_reduce).
    /// Supported modes: sum, prod, mean, amin, amax. When
    /// <paramref name="includeSelf"/> is false, target positions that any
    /// index touches are first reset to the reduction identity (0 for
    /// sum/mean, 1 for prod, ±∞ equivalents for amin/amax) before
    /// accumulating.
    /// </summary>
    Tensor<T> TensorScatterReduce<T>(
        Tensor<T> tensor, int dim, Tensor<int> indices, Tensor<T> source,
        ScatterReduceMode mode, bool includeSelf = true);

    /// <summary>
    /// Where operation: selects elements from two tensors based on a condition.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="condition">Boolean condition tensor.</param>
    /// <param name="x">Tensor to select from where condition is true.</param>
    /// <param name="y">Tensor to select from where condition is false.</param>
    /// <returns>A tensor with selected elements.</returns>
    /// <remarks>
    /// <para>
    /// Similar to torch.where() or np.where().
    /// Essential for conditional operations without loops.
    /// </para>
    /// </remarks>
    Tensor<T> TensorWhere<T>(Tensor<bool> condition, Tensor<T> x, Tensor<T> y);

    /// <summary>
    /// Where operation using a <see cref="Bit"/> condition tensor.
    /// Bit.True selects from <paramref name="x"/>; Bit.False selects from <paramref name="y"/>.
    /// </summary>
    Tensor<T> TensorWhere<T>(Tensor<Bit> condition, Tensor<T> x, Tensor<T> y);

    #endregion

    #region Neural Radiance Fields Operations

    /// <summary>
    /// Computes positional encoding for Neural Radiance Fields.
    /// Applies sin/cos frequency encoding: [sin(2^0*Ãâ‚¬*x), cos(2^0*Ãâ‚¬*x), ..., sin(2^(L-1)*Ãâ‚¬*x), cos(2^(L-1)*Ãâ‚¬*x)]
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="positions">Input positions tensor of shape [N, D] where D is typically 3 (x,y,z).</param>
    /// <param name="numFrequencies">Number of frequency levels (L in the paper, typically 10 for positions, 4 for directions).</param>
    /// <returns>Encoded tensor of shape [N, D * 2 * numFrequencies].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Positional encoding transforms low-dimensional coordinates into
    /// high-dimensional features that help neural networks learn high-frequency details.
    /// </para>
    /// <para>
    /// Without positional encoding, neural networks tend to learn smooth, blurry functions.
    /// The sin/cos encoding at multiple frequencies enables sharp, detailed reconstructions.
    /// </para>
    /// <para>
    /// Reference: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    /// by Mildenhall et al., ECCV 2020
    /// </para>
    /// </remarks>
    Tensor<T> PositionalEncoding<T>(Tensor<T> positions, int numFrequencies);

    /// <summary>
    /// Computes the backward pass for positional encoding.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="positions">Original input positions tensor of shape [N, D].</param>
    /// <param name="encodedGradient">Gradient of loss with respect to encoded output [N, D * 2 * numFrequencies].</param>
    /// <param name="numFrequencies">Number of frequency levels used in forward pass.</param>
    /// <returns>Gradient with respect to input positions [N, D].</returns>
    Tensor<T> PositionalEncodingBackward<T>(Tensor<T> positions, Tensor<T> encodedGradient, int numFrequencies);

    /// <summary>
    /// Performs volume rendering along rays using alpha compositing.
    /// Computes: C(r) = ÃŽÂ£ T_i * ÃŽÂ±_i * c_i where T_i = ÃŽÂ (1 - ÃŽÂ±_j) for j &lt; i
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rgbSamples">RGB color samples of shape [numRays, numSamples, 3].</param>
    /// <param name="densitySamples">Density/sigma samples of shape [numRays, numSamples, 1].</param>
    /// <param name="tValues">Distance values along rays of shape [numRays, numSamples].</param>
    /// <returns>Rendered colors of shape [numRays, 3].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Volume rendering accumulates color along a ray by considering
    /// how much light is blocked (transmittance) at each sample point.
    /// </para>
    /// <para>
    /// The alpha value at each point represents how much light is absorbed there.
    /// Transmittance tracks how much light reaches each point from the camera.
    /// Final color is the weighted sum of all sample colors.
    /// </para>
    /// </remarks>
    Tensor<T> VolumeRendering<T>(Tensor<T> rgbSamples, Tensor<T> densitySamples, Tensor<T> tValues);

    /// <summary>
    /// Computes the backward pass for volume rendering.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rgbSamples">RGB color samples from forward pass [numRays, numSamples, 3].</param>
    /// <param name="densitySamples">Density samples from forward pass [numRays, numSamples, 1].</param>
    /// <param name="tValues">Distance values from forward pass [numRays, numSamples].</param>
    /// <param name="outputGradient">Gradient of loss with respect to rendered colors [numRays, 3].</param>
    /// <param name="rgbGradient">Output: Gradient with respect to RGB samples.</param>
    /// <param name="densityGradient">Output: Gradient with respect to density samples.</param>
    void VolumeRenderingBackward<T>(
        Tensor<T> rgbSamples,
        Tensor<T> densitySamples,
        Tensor<T> tValues,
        Tensor<T> outputGradient,
        out Tensor<T> rgbGradient,
        out Tensor<T> densityGradient);

    /// <summary>
    /// Samples points uniformly along rays for volume rendering.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rayOrigins">Ray origin points of shape [numRays, 3].</param>
    /// <param name="rayDirections">Ray direction vectors of shape [numRays, 3] (should be normalized).</param>
    /// <param name="nearBound">Near clipping distance.</param>
    /// <param name="farBound">Far clipping distance.</param>
    /// <param name="numSamples">Number of samples per ray.</param>
    /// <param name="stratified">If true, adds jitter to sample positions for anti-aliasing.</param>
    /// <returns>Tuple of (sample positions [numRays * numSamples, 3], sample directions [numRays * numSamples, 3], t values [numRays, numSamples]).</returns>
    (Tensor<T> positions, Tensor<T> directions, Tensor<T> tValues) SampleRayPoints<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        T nearBound,
        T farBound,
        int numSamples,
        bool stratified = true);

    /// <summary>
    /// Performs importance sampling based on coarse network density predictions.
    /// Samples more points where density is high (hierarchical sampling).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tValuesCoarse">Coarse sample t values [numRays, numCoarseSamples].</param>
    /// <param name="weightsCoarse">Rendering weights from coarse samples [numRays, numCoarseSamples].</param>
    /// <param name="numFineSamples">Number of additional fine samples to generate.</param>
    /// <returns>Fine sample t values [numRays, numFineSamples] concentrated where density is high.</returns>
    Tensor<T> ImportanceSampling<T>(Tensor<T> tValuesCoarse, Tensor<T> weightsCoarse, int numFineSamples);

    /// <summary>
    /// Generates camera rays for each pixel in an image.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="cameraPosition">Camera position in world coordinates [3].</param>
    /// <param name="cameraRotation">Camera rotation matrix [3, 3] (camera-to-world).</param>
    /// <param name="imageWidth">Image width in pixels.</param>
    /// <param name="imageHeight">Image height in pixels.</param>
    /// <param name="focalLength">Camera focal length.</param>
    /// <returns>Tuple of (ray origins [H*W, 3], ray directions [H*W, 3]).</returns>
    (Tensor<T> origins, Tensor<T> directions) GenerateCameraRays<T>(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength);

    #endregion

    #region Gaussian Splatting Operations

    /// <summary>
    /// Projects 3D Gaussians to 2D screen space for rasterization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="means3D">Gaussian center positions [N, 3].</param>
    /// <param name="covariances3D">3D covariance matrices [N, 3, 3] or [N, 6] (upper triangular).</param>
    /// <param name="viewMatrix">World-to-camera transformation [4, 4].</param>
    /// <param name="projMatrix">Camera projection matrix [4, 4].</param>
    /// <param name="imageWidth">Target image width.</param>
    /// <param name="imageHeight">Target image height.</param>
    /// <param name="means2D">Output: Projected 2D means [N, 2].</param>
    /// <param name="covariances2D">Output: Projected 2D covariances [N, 3] (a, b, c for axÃ‚Â² + 2bxy + cyÃ‚Â²).</param>
    /// <param name="depths">Output: Depth values for sorting [N].</param>
    /// <param name="visible">Output: Visibility mask (in frustum and valid) [N].</param>
    void ProjectGaussians3DTo2D<T>(
        Tensor<T> means3D,
        Tensor<T> covariances3D,
        Matrix<T> viewMatrix,
        Matrix<T> projMatrix,
        int imageWidth,
        int imageHeight,
        out Tensor<T> means2D,
        out Tensor<T> covariances2D,
        out Tensor<T> depths,
        out Tensor<bool> visible);

    /// <summary>
    /// Rasterizes 2D Gaussians onto an image using alpha blending.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="means2D">Projected 2D Gaussian centers [N, 2].</param>
    /// <param name="covariances2D">2D covariance parameters [N, 3].</param>
    /// <param name="colors">Gaussian colors [N, C] where C is typically 3 (RGB) or more for SH.</param>
    /// <param name="opacities">Gaussian opacities [N].</param>
    /// <param name="depths">Depth values for sorting [N].</param>
    /// <param name="imageWidth">Output image width.</param>
    /// <param name="imageHeight">Output image height.</param>
    /// <param name="tileSize">Tile size for tiled rasterization (typically 16).</param>
    /// <returns>Rendered image [H, W, C].</returns>
    Tensor<T> RasterizeGaussians<T>(
        Tensor<T> means2D,
        Tensor<T> covariances2D,
        Tensor<T> colors,
        Tensor<T> opacities,
        Tensor<T> depths,
        int imageWidth,
        int imageHeight,
        int tileSize = 16);

    /// <summary>
    /// Computes the backward pass for Gaussian rasterization.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="means2D">Projected 2D Gaussian centers [N, 2].</param>
    /// <param name="covariances2D">2D covariance parameters [N, 3].</param>
    /// <param name="colors">Gaussian colors [N, C].</param>
    /// <param name="opacities">Gaussian opacities [N].</param>
    /// <param name="depths">Depth values [N].</param>
    /// <param name="imageWidth">Image width.</param>
    /// <param name="imageHeight">Image height.</param>
    /// <param name="outputGradient">Gradient of loss with respect to rendered image [H, W, C].</param>
    /// <param name="tileSize">Tile size used in forward pass.</param>
    /// <param name="means2DGrad">Output: Gradient with respect to 2D means.</param>
    /// <param name="covariances2DGrad">Output: Gradient with respect to 2D covariances.</param>
    /// <param name="colorsGrad">Output: Gradient with respect to colors.</param>
    /// <param name="opacitiesGrad">Output: Gradient with respect to opacities.</param>
    void RasterizeGaussiansBackward<T>(
        Tensor<T> means2D,
        Tensor<T> covariances2D,
        Tensor<T> colors,
        Tensor<T> opacities,
        Tensor<T> depths,
        int imageWidth,
        int imageHeight,
        Tensor<T> outputGradient,
        int tileSize,
        out Tensor<T> means2DGrad,
        out Tensor<T> covariances2DGrad,
        out Tensor<T> colorsGrad,
        out Tensor<T> opacitiesGrad);

    /// <summary>
    /// Evaluates spherical harmonics for view-dependent color in Gaussian Splatting.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shCoefficients">Spherical harmonics coefficients [N, (degree+1)Ã‚Â², C] where C=3 for RGB.</param>
    /// <param name="viewDirections">Normalized view directions [N, 3] or [1, 3] for broadcast.</param>
    /// <param name="degree">SH degree (0-3, where 0=constant, 3=full view dependence).</param>
    /// <returns>Evaluated colors [N, C].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spherical harmonics allow colors to change based on viewing angle,
    /// enabling realistic specular highlights and view-dependent effects.
    /// </para>
    /// <para>
    /// Degree 0: Constant color (1 coefficient per channel)
    /// Degree 1: Linear variation (4 coefficients per channel)
    /// Degree 2: Quadratic variation (9 coefficients per channel)
    /// Degree 3: Cubic variation (16 coefficients per channel)
    /// </para>
    /// </remarks>
    Tensor<T> EvaluateSphericalHarmonics<T>(Tensor<T> shCoefficients, Tensor<T> viewDirections, int degree);

    /// <summary>
    /// Computes the backward pass for spherical harmonics evaluation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="shCoefficients">SH coefficients from forward pass.</param>
    /// <param name="viewDirections">View directions from forward pass.</param>
    /// <param name="degree">SH degree.</param>
    /// <param name="outputGradient">Gradient with respect to evaluated colors.</param>
    /// <returns>Gradient with respect to SH coefficients.</returns>
    Tensor<T> EvaluateSphericalHarmonicsBackward<T>(
        Tensor<T> shCoefficients,
        Tensor<T> viewDirections,
        int degree,
        Tensor<T> outputGradient);

    /// <summary>
    /// Computes 3D covariance matrices from rotation quaternions and scale vectors.
    /// Covariance = R * S * S^T * R^T where R is rotation matrix from quaternion, S is diagonal scale.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rotations">Rotation quaternions [N, 4] (w, x, y, z).</param>
    /// <param name="scales">Scale vectors [N, 3].</param>
    /// <returns>Covariance matrices [N, 3, 3] or [N, 6] (upper triangular).</returns>
    Tensor<T> ComputeGaussianCovariance<T>(Tensor<T> rotations, Tensor<T> scales);

    /// <summary>
    /// Computes the backward pass for Gaussian covariance computation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rotations">Rotation quaternions from forward pass.</param>
    /// <param name="scales">Scale vectors from forward pass.</param>
    /// <param name="covarianceGradient">Gradient with respect to covariances.</param>
    /// <param name="rotationsGrad">Output: Gradient with respect to rotations.</param>
    /// <param name="scalesGrad">Output: Gradient with respect to scales.</param>
    void ComputeGaussianCovarianceBackward<T>(
        Tensor<T> rotations,
        Tensor<T> scales,
        Tensor<T> covarianceGradient,
        out Tensor<T> rotationsGrad,
        out Tensor<T> scalesGrad);

    #endregion

    #region Instant-NGP Operations

    /// <summary>
    /// Performs multiresolution hash encoding for Instant-NGP.
    /// Encodes 3D positions using a hierarchy of hash tables with trilinear interpolation.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="positions">Input positions [N, 3] normalized to [0, 1].</param>
    /// <param name="hashTables">List of hash tables, one per resolution level.</param>
    /// <param name="resolutions">Resolution at each level.</param>
    /// <param name="featuresPerLevel">Number of features stored per hash entry.</param>
    /// <returns>Encoded features [N, numLevels * featuresPerLevel].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hash encoding replaces expensive positional encoding with fast
    /// table lookups, enabling 100x faster training and 1000x faster rendering than NeRF.
    /// </para>
    /// <para>
    /// The key insight is that hash collisions are okay because:
    /// 1. Multiple positions mapping to the same entry often have similar features
    /// 2. The neural network learns to handle collisions during training
    /// 3. The speed benefit far outweighs the minor quality impact
    /// </para>
    /// <para>
    /// Reference: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
    /// by MÃƒÂ¼ller et al., ACM Transactions on Graphics 2022
    /// </para>
    /// </remarks>
    Tensor<T> MultiresolutionHashEncoding<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel);

    /// <summary>
    /// Computes the backward pass for multiresolution hash encoding.
    /// Accumulates gradients to the appropriate hash table entries.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="positions">Input positions from forward pass.</param>
    /// <param name="hashTables">Hash tables from forward pass.</param>
    /// <param name="resolutions">Resolutions from forward pass.</param>
    /// <param name="featuresPerLevel">Features per level from forward pass.</param>
    /// <param name="outputGradient">Gradient with respect to encoded features.</param>
    /// <returns>Gradients for each hash table.</returns>
    Tensor<T>[] MultiresolutionHashEncodingBackward<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel,
        Tensor<T> outputGradient);

    /// <summary>
    /// Updates occupancy grid for efficient ray sampling in Instant-NGP.
    /// Marks which voxels contain geometry based on density threshold.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="occupancyGrid">Current occupancy grid [gridSize, gridSize, gridSize].</param>
    /// <param name="densities">Sampled density values.</param>
    /// <param name="positions">Positions where densities were sampled.</param>
    /// <param name="gridSize">Size of the occupancy grid.</param>
    /// <param name="threshold">Density threshold for occupancy.</param>
    /// <param name="decayFactor">EMA decay for updating occupancy values.</param>
    /// <returns>Updated occupancy grid.</returns>
    Tensor<T> UpdateOccupancyGrid<T>(
        Tensor<T> occupancyGrid,
        Tensor<T> densities,
        Tensor<T> positions,
        int gridSize,
        T threshold,
        T decayFactor);

    /// <summary>
    /// Samples rays while skipping empty space using occupancy grid.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="rayOrigins">Ray origins [numRays, 3].</param>
    /// <param name="rayDirections">Ray directions [numRays, 3].</param>
    /// <param name="occupancyBitfield">Packed occupancy bits for fast lookup as a 1D tensor.</param>
    /// <param name="gridSize">Size of the occupancy grid.</param>
    /// <param name="sceneBoundsMin">Minimum scene bounds [3].</param>
    /// <param name="sceneBoundsMax">Maximum scene bounds [3].</param>
    /// <param name="nearBound">Near clipping distance.</param>
    /// <param name="farBound">Far clipping distance.</param>
    /// <param name="maxSamples">Maximum samples per ray.</param>
    /// <returns>Tuple of (sample positions, sample directions, valid mask, t values).</returns>
    (Tensor<T> positions, Tensor<T> directions, Tensor<bool> validMask, Tensor<T> tValues) SampleRaysWithOccupancy<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        Tensor<uint> occupancyBitfield,
        int gridSize,
        Vector<T> sceneBoundsMin,
        Vector<T> sceneBoundsMax,
        T nearBound,
        T farBound,
        int maxSamples);

    #endregion

    #region Mesh Convolution Operations

    /// <summary>
    /// Performs spiral convolution on mesh vertex features.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="vertexFeatures">Input vertex features [numVertices, inputChannels].</param>
    /// <param name="spiralIndices">Spiral neighbor indices [numVertices, spiralLength].</param>
    /// <param name="weights">Convolution weights [outputChannels, inputChannels * spiralLength].</param>
    /// <param name="biases">Bias values [outputChannels].</param>
    /// <returns>Output vertex features [numVertices, outputChannels].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Spiral convolution extends traditional convolution to irregular mesh surfaces.
    /// Unlike grid-based convolutions where neighbors are in fixed positions, mesh vertices have
    /// variable connectivity. Spiral convolution solves this by:
    /// 
    /// 1. Defining a consistent spiral ordering of neighbors around each vertex
    /// 2. Gathering features from neighbors in this canonical order
    /// 3. Applying learned weights to the ordered features
    /// 
    /// This creates translation-equivariant convolutions on arbitrary mesh topologies.
    /// </para>
    /// <para>
    /// <b>Mathematical Formulation:</b>
    /// For each vertex v with spiral neighbors S(v) = [nÃ¢â€šÂ, nÃ¢â€šâ€š, ..., nÃ¢â€šâ€“]:
    /// 
    /// gathered[v] = concat(features[nÃ¢â€šÂ], features[nÃ¢â€šâ€š], ..., features[nÃ¢â€šâ€“])
    /// output[v] = weights @ gathered[v] + bias
    /// 
    /// The spiral ordering ensures that the convolution is invariant to mesh parameterization.
    /// </para>
    /// <para>
    /// Reference: "Neural 3D Morphable Models: Spiral Convolutional Networks" by Bouritsas et al.
    /// Reference: "SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator" by Gong et al.
    /// </para>
    /// </remarks>
    Tensor<T> SpiralConv<T>(
        Tensor<T> vertexFeatures,
        Tensor<int> spiralIndices,
        Tensor<T> weights,
        Tensor<T> biases);

    /// <summary>
    /// Computes the backward pass for spiral convolution with respect to input features.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <param name="spiralIndices">Spiral neighbor indices [numVertices, spiralLength].</param>
    /// <param name="weights">Convolution weights [outputChannels, inputChannels * spiralLength].</param>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <returns>Gradient with respect to input features [numVertices, inputChannels].</returns>
    /// <remarks>
    /// <para>
    /// The backward pass scatters gradients back to the original vertex positions according
    /// to the spiral indices. This uses atomic scatter-add operations for correctness when
    /// multiple spiral paths reference the same vertex.
    /// </para>
    /// </remarks>
    Tensor<T> SpiralConvBackwardInput<T>(
        Tensor<T> outputGradient,
        Tensor<int> spiralIndices,
        Tensor<T> weights,
        int inputChannels);

    /// <summary>
    /// Computes the backward pass for spiral convolution with respect to weights.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <param name="vertexFeatures">Input vertex features from forward pass [numVertices, inputChannels].</param>
    /// <param name="spiralIndices">Spiral neighbor indices [numVertices, spiralLength].</param>
    /// <returns>Gradient with respect to weights [outputChannels, inputChannels * spiralLength].</returns>
    Tensor<T> SpiralConvBackwardWeights<T>(
        Tensor<T> outputGradient,
        Tensor<T> vertexFeatures,
        Tensor<int> spiralIndices);

    /// <summary>
    /// Computes the backward pass for spiral convolution with respect to biases.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <returns>Gradient with respect to biases [outputChannels].</returns>
    Tensor<T> SpiralConvBackwardBias<T>(Tensor<T> outputGradient);

    /// <summary>
    /// Performs diffusion convolution on mesh vertex features using the Laplacian operator.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="vertexFeatures">Input vertex features [numVertices, inputChannels].</param>
    /// <param name="laplacian">Mesh Laplacian matrix [numVertices, numVertices].</param>
    /// <param name="weights">Diffusion weights [outputChannels, inputChannels].</param>
    /// <param name="biases">Bias values [outputChannels].</param>
    /// <param name="diffusionTime">Diffusion time parameter controlling spatial extent.</param>
    /// <returns>Output vertex features [numVertices, outputChannels].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Diffusion convolution uses heat diffusion on the mesh surface
    /// to define the convolution kernel. Features spread across the mesh according to the
    /// heat equation, which respects the intrinsic geometry of the surface.
    /// 
    /// The diffusion time controls how far features propagate:
    /// - Small time: local features, fine detail
    /// - Large time: global features, coarse structure
    /// </para>
    /// <para>
    /// <b>Mathematical Formulation:</b>
    /// output = exp(-t * L) @ input @ weights + bias
    /// 
    /// Where L is the mesh Laplacian and t is the diffusion time.
    /// The matrix exponential is computed using eigendecomposition or Taylor series.
    /// </para>
    /// <para>
    /// Reference: "DiffusionNet: Discretization Agnostic Learning on Surfaces" by Sharp et al.
    /// </para>
    /// </remarks>
    Tensor<T> DiffusionConv<T>(
        Tensor<T> vertexFeatures,
        Tensor<T> laplacian,
        Tensor<T> weights,
        Tensor<T> biases,
        T diffusionTime);

    /// <summary>
    /// Computes the backward pass for diffusion convolution.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="outputGradient">Gradient with respect to output [numVertices, outputChannels].</param>
    /// <param name="vertexFeatures">Input vertex features from forward pass.</param>
    /// <param name="laplacian">Mesh Laplacian matrix from forward pass.</param>
    /// <param name="weights">Diffusion weights from forward pass.</param>
    /// <param name="diffusionTime">Diffusion time from forward pass.</param>
    /// <returns>Tuple of (input gradient, weight gradient, bias gradient).</returns>
    (Tensor<T> inputGrad, Tensor<T> weightGrad, Tensor<T> biasGrad) DiffusionConvBackward<T>(
        Tensor<T> outputGradient,
        Tensor<T> vertexFeatures,
        Tensor<T> laplacian,
        Tensor<T> weights,
        T diffusionTime);

    /// <summary>
    /// Computes the mesh Laplacian matrix from vertex positions and face indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="vertices">Vertex positions [numVertices, 3].</param>
    /// <param name="faces">Face indices [numFaces, 3] for triangular mesh.</param>
    /// <param name="laplacianType">Type of Laplacian operator to compute.</param>
    /// <returns>Laplacian matrix [numVertices, numVertices].</returns>
    /// <remarks>
    /// <para>
    /// <b>Laplacian Types:</b>
    /// - <see cref="LaplacianType.Uniform"/>: Simple adjacency-based, ignores geometry
    /// - <see cref="LaplacianType.Cotangent"/>: Geometry-aware, preserves angles
    /// - <see cref="LaplacianType.Normalized"/>: Cotangent normalized by vertex areas
    /// </para>
    /// </remarks>
    Tensor<T> ComputeMeshLaplacian<T>(
        Tensor<T> vertices,
        Tensor<int> faces,
        LaplacianType laplacianType = LaplacianType.Cotangent);

    /// <summary>
    /// Generates spiral indices for mesh vertices based on connectivity.
    /// </summary>
    /// <typeparam name="T">The numeric type for vertex positions.</typeparam>
    /// <param name="vertices">Vertex positions [numVertices, 3].</param>
    /// <param name="faces">Face indices [numFaces, 3].</param>
    /// <param name="spiralLength">Number of neighbors in each spiral.</param>
    /// <returns>Spiral indices [numVertices, spiralLength].</returns>
    /// <remarks>
    /// <para>
    /// The algorithm:
    /// 1. Build adjacency list from faces
    /// 2. For each vertex, find the initial reference direction
    /// 3. Sort neighbors by angle from reference in consistent winding order
    /// 4. Extend to spiral length by following the ring structure
    /// </para>
    /// </remarks>
    Tensor<int> GenerateSpiralIndices<T>(
        Tensor<T> vertices,
        Tensor<int> faces,
        int spiralLength);

    #endregion

    #region Advanced Vectorization Operations

    /// <summary>
    /// Computes pairwise squared Euclidean distances between two sets of points.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="x">First set of points [N, D] where N is number of points and D is dimensionality.</param>
    /// <param name="y">Second set of points [M, D] where M is number of points and D is dimensionality.</param>
    /// <returns>Distance matrix [N, M] where element [i,j] is squared distance between x[i] and y[j].</returns>
    /// <remarks>
    /// Uses the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y for efficiency.
    /// This avoids the O(N*M*D) explicit subtraction and enables GPU parallelization.
    /// </remarks>
    Tensor<T> PairwiseDistanceSquared<T>(Tensor<T> x, Tensor<T> y);

    /// <summary>
    /// Computes pairwise Euclidean distances between two sets of points.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="x">First set of points [N, D].</param>
    /// <param name="y">Second set of points [M, D].</param>
    /// <returns>Distance matrix [N, M] where element [i,j] is Euclidean distance between x[i] and y[j].</returns>
    Tensor<T> PairwiseDistance<T>(Tensor<T> x, Tensor<T> y);

    /// <summary>
    /// Returns the k largest or smallest elements along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor.</param>
    /// <param name="k">Number of top elements to return.</param>
    /// <param name="axis">Axis along which to find top-k. Default -1 means last axis.</param>
    /// <param name="largest">If true, return k largest elements; if false, return k smallest.</param>
    /// <returns>Tuple of (values, indices) where values contains the top-k elements and indices their positions.</returns>
    (Tensor<T> values, Tensor<int> indices) TopK<T>(Tensor<T> input, int k, int axis = -1, bool largest = true);

    /// <summary>
    /// Returns indices that would sort the tensor along an axis.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor.</param>
    /// <param name="axis">Axis along which to sort. Default -1 means last axis.</param>
    /// <param name="descending">If true, sort in descending order.</param>
    /// <returns>Tensor of indices that would sort the input.</returns>
    Tensor<int> ArgSort<T>(Tensor<T> input, int axis = -1, bool descending = false);

    /// <summary>
    /// Gathers elements from input tensor along an axis using indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Source tensor to gather from.</param>
    /// <param name="indices">Indices specifying which elements to gather.</param>
    /// <param name="axis">Axis along which to gather.</param>
    /// <returns>Tensor of gathered elements.</returns>
    Tensor<T> Gather<T>(Tensor<T> input, Tensor<int> indices, int axis);

    /// <summary>
    /// Scatters values into a new tensor at positions specified by indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor providing the shape and initial values.</param>
    /// <param name="indices">Indices where to scatter values.</param>
    /// <param name="values">Values to scatter.</param>
    /// <param name="axis">Axis along which to scatter.</param>
    /// <returns>New tensor with scattered values.</returns>
    Tensor<T> Scatter<T>(Tensor<T> input, Tensor<int> indices, Tensor<T> values, int axis);

    /// <summary>
    /// Scatters values by adding them to existing values at positions specified by indices.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor providing the shape and initial values.</param>
    /// <param name="indices">Indices where to add values.</param>
    /// <param name="values">Values to add at specified indices.</param>
    /// <param name="axis">Axis along which to scatter-add.</param>
    /// <returns>New tensor with values added at specified positions.</returns>
    Tensor<T> ScatterAdd<T>(Tensor<T> input, Tensor<int> indices, Tensor<T> values, int axis);

    /// <summary>
    /// Computes the hyperbolic cosine of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with hyperbolic cosine applied element-wise.</returns>
    Tensor<T> TensorCosh<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the hyperbolic sine of each element in the tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with hyperbolic sine applied element-wise.</returns>
    Tensor<T> TensorSinh<T>(Tensor<T> tensor);

    /// <summary>
    /// Computes the outer product of two 1D tensors.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">First 1D tensor of length N.</param>
    /// <param name="b">Second 1D tensor of length M.</param>
    /// <returns>2D tensor [N, M] where result[i,j] = a[i] * b[j].</returns>
    Tensor<T> TensorOuter<T>(Tensor<T> a, Tensor<T> b);

    #endregion

    #region Tensor-Level Activation Operations

    /// <summary>
    /// Applies element-wise sigmoid activation: 1 / (1 + exp(-x)).
    /// Alias for <see cref="Sigmoid{T}(Tensor{T})"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with sigmoid applied element-wise.</returns>
    Tensor<T> TensorSigmoid<T>(Tensor<T> tensor);

    /// <summary>
    /// Applies element-wise ReLU activation: max(0, x).
    /// Alias for <see cref="ReLU{T}(Tensor{T})"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with ReLU applied element-wise.</returns>
    Tensor<T> TensorReLU<T>(Tensor<T> tensor);

    /// <summary>
    /// Applies element-wise GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    /// Alias for <see cref="GELU{T}(Tensor{T})"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with GELU applied element-wise.</returns>
    Tensor<T> TensorGELU<T>(Tensor<T> tensor);

    /// <summary>
    /// Applies element-wise SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
    /// Also known as Swish. Default activation in modern diffusion models.
    /// Alias for <see cref="Swish{T}(Tensor{T})"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with SiLU applied element-wise.</returns>
    Tensor<T> TensorSiLU<T>(Tensor<T> tensor);

    /// <summary>
    /// Applies element-wise tanh activation.
    /// Alias for <see cref="Tanh{T}(Tensor{T})"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with tanh applied element-wise.</returns>
    Tensor<T> TensorTanh<T>(Tensor<T> tensor);

    /// <summary>
    /// Applies element-wise Leaky ReLU activation: x if x > 0, else alpha * x.
    /// Alias for <see cref="LeakyReLU{T}(Tensor{T}, T)"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <param name="alpha">Negative slope coefficient (commonly 0.01).</param>
    /// <returns>Tensor with Leaky ReLU applied element-wise.</returns>
    Tensor<T> TensorLeakyReLU<T>(Tensor<T> tensor, T alpha);

    /// <summary>
    /// Applies element-wise Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x)).
    /// Alias for <see cref="Mish{T}(Tensor{T})"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with Mish applied element-wise.</returns>
    Tensor<T> TensorMish<T>(Tensor<T> tensor);

    /// <summary>
    /// Applies element-wise HardSwish activation: x * relu6(x + 3) / 6.
    /// Alias for <see cref="HardSwish{T}(Tensor{T})"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Tensor with HardSwish applied element-wise.</returns>
    Tensor<T> TensorHardSwish<T>(Tensor<T> tensor);

    #endregion

    #region Tensor-Level Composite Operations

    /// <summary>
    /// Applies layer normalization on a tensor.
    /// Alias for <see cref="LayerNorm{T}"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor.</param>
    /// <param name="gamma">Scale parameter tensor.</param>
    /// <param name="beta">Shift parameter tensor.</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <returns>Normalized tensor.</returns>
    Tensor<T> TensorLayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon = 1e-5);

    /// <summary>
    /// Computes standard deviation along specified axes.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">Input tensor.</param>
    /// <param name="axes">Axes along which to compute standard deviation.</param>
    /// <param name="keepDims">Whether to keep reduced dimensions.</param>
    /// <returns>Tensor containing standard deviations along specified axes.</returns>
    Tensor<T> ReduceStd<T>(Tensor<T> input, int[] axes, bool keepDims);

    /// <summary>
    /// Linearly interpolates between two tensors element-wise: (1 - t) * a + t * b.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">Start tensor.</param>
    /// <param name="b">End tensor.</param>
    /// <param name="t">Interpolation factor (0 = a, 1 = b).</param>
    /// <returns>Interpolated tensor.</returns>
    Tensor<T> TensorLerp<T>(Tensor<T> a, Tensor<T> b, T t);

    /// <summary>
    /// Computes fused scaled addition of two tensors: scaleA * a + scaleB * b.
    /// Commonly used for noise mixing in diffusion models (alpha * signal + sigma * noise).
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="a">First tensor.</param>
    /// <param name="b">Second tensor.</param>
    /// <param name="scaleA">Scale factor for first tensor.</param>
    /// <param name="scaleB">Scale factor for second tensor.</param>
    /// <returns>Result tensor: scaleA * a + scaleB * b.</returns>
    Tensor<T> TensorAddScaled<T>(Tensor<T> a, Tensor<T> b, T scaleA, T scaleB);

    /// <summary>
    /// Alias for <see cref="MaxPool2D{T}(Tensor{T}, int, int, int)"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">4D input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">Pooling window size.</param>
    /// <param name="stride">Stride. If 0, defaults to poolSize.</param>
    /// <param name="padding">Zero-padding size.</param>
    /// <returns>Pooled tensor.</returns>
    Tensor<T> TensorMaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Alias for <see cref="AvgPool2D{T}(Tensor{T}, int, int, int)"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">4D input tensor [batch, channels, height, width].</param>
    /// <param name="poolSize">Pooling window size.</param>
    /// <param name="stride">Stride. If 0, defaults to poolSize.</param>
    /// <param name="padding">Zero-padding size.</param>
    /// <returns>Pooled tensor.</returns>
    Tensor<T> TensorAvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0);

    /// <summary>
    /// Alias for <see cref="Conv2D{T}(Tensor{T}, Tensor{T}, int, int, int)"/> with Tensor prefix for API consistency.
    /// </summary>
    /// <typeparam name="T">The numeric type of tensor elements.</typeparam>
    /// <param name="input">4D input tensor [batch, channels, height, width].</param>
    /// <param name="kernel">4D kernel tensor [outChannels, inChannels, kH, kW].</param>
    /// <param name="stride">Convolution stride.</param>
    /// <param name="padding">Zero-padding size.</param>
    /// <param name="dilation">Dilation factor.</param>
    /// <returns>Convolution result tensor.</returns>
    Tensor<T> TensorConv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1);

    #endregion

    // ──────────────────────────────────────────────────────────────
    // Differentiable loss functions (return scalar Tensor<T> for tape)
    // ──────────────────────────────────────────────────────────────

    /// <summary>MSE loss: mean((pred - target)^2). Returns scalar tensor.</summary>
    Tensor<T> TensorMSELoss<T>(Tensor<T> predictions, Tensor<T> targets);

    /// <summary>L1 loss: mean(|pred - target|). Returns scalar tensor.</summary>
    Tensor<T> TensorL1Loss<T>(Tensor<T> predictions, Tensor<T> targets);

    /// <summary>Huber loss: smooth L1. Returns scalar tensor.</summary>
    Tensor<T> TensorHuberLoss<T>(Tensor<T> predictions, Tensor<T> targets, double delta = 1.0);

    /// <summary>BCE with logits: sigmoid cross-entropy. Returns scalar tensor.</summary>
    Tensor<T> TensorBCEWithLogitsLoss<T>(Tensor<T> logits, Tensor<T> targets);

    /// <summary>Cross-entropy with softmax. Returns scalar tensor.</summary>
    Tensor<T> TensorCrossEntropyLoss<T>(Tensor<T> logits, Tensor<T> targets);

    /// <summary>NLL loss on log-probabilities. Returns scalar tensor.</summary>
    Tensor<T> TensorNLLLoss<T>(Tensor<T> logProbs, Tensor<T> targets);

    /// <summary>KL divergence loss. Returns scalar tensor.</summary>
    Tensor<T> TensorKLDivLoss<T>(Tensor<T> input, Tensor<T> target);

    /// <summary>Cosine similarity. Returns scalar tensor.</summary>
    Tensor<T> TensorCosineSimilarityLoss<T>(Tensor<T> a, Tensor<T> b);

    // ──────────────────────────────────────────────────────────────
    // Differentiable activations
    // ──────────────────────────────────────────────────────────────

    /// <summary>SELU activation.</summary>
    Tensor<T> TensorSELU<T>(Tensor<T> tensor);

    /// <summary>HardSigmoid: clamp(x/6 + 0.5, 0, 1).</summary>
    Tensor<T> TensorHardSigmoid<T>(Tensor<T> tensor);

    /// <summary>ReLU6: min(max(0, x), 6).</summary>
    Tensor<T> TensorReLU6<T>(Tensor<T> tensor);

    /// <summary>PReLU with learnable alpha.</summary>
    Tensor<T> TensorPReLU<T>(Tensor<T> tensor, Tensor<T> alpha);

    /// <summary>RReLU with random leaky slope.</summary>
    Tensor<T> TensorRReLU<T>(Tensor<T> tensor, double lower = 0.125, double upper = 0.333, bool training = true);

    /// <summary>Threshold: x > threshold ? x : value.</summary>
    Tensor<T> TensorThreshold<T>(Tensor<T> tensor, T threshold, T value);

    /// <summary>Element-wise reciprocal: 1/x.</summary>
    Tensor<T> TensorReciprocal<T>(Tensor<T> tensor);

    /// <summary>Element-wise sign: -1, 0, or 1.</summary>
    Tensor<T> TensorSign<T>(Tensor<T> tensor);

    // ──────────────────────────────────────────────────────────────
    // Differentiable shape/indexing/pooling ops
    // ──────────────────────────────────────────────────────────────

    /// <summary>Flatten tensor to 1D.</summary>
    Tensor<T> TensorFlatten<T>(Tensor<T> tensor);

    /// <summary>Narrow (slice along one axis).</summary>
    Tensor<T> TensorNarrow<T>(Tensor<T> tensor, int dim, int start, int length);

    /// <summary>Constant padding for N-dimensional tensors.</summary>
    Tensor<T> TensorConstantPad<T>(Tensor<T> tensor, int[] padding, T value);

    /// <summary>Upsample bilinear (4D NCHW).</summary>
    Tensor<T> TensorUpsampleBilinear<T>(Tensor<T> input, int[] outputSize);

    /// <summary>1D average pooling.</summary>
    Tensor<T> TensorAvgPool1D<T>(Tensor<T> input, int kernelSize, int stride);

    /// <summary>1D max pooling.</summary>
    Tensor<T> TensorMaxPool1D<T>(Tensor<T> input, int kernelSize, int stride);

    /// <summary>Full mean reduction returning scalar tensor for tape.</summary>
    Tensor<T> TensorMeanDiff<T>(Tensor<T> tensor);

    // ──────────────────────────────────────────────────────────────
    // Missing Phase 1 ops
    // ──────────────────────────────────────────────────────────────

    /// <summary>Stack tensors along a new axis.</summary>
    Tensor<T> TensorStackDiff<T>(Tensor<T>[] tensors, int axis = 0);

    /// <summary>Variance of all elements, returns scalar tensor.</summary>
    Tensor<T> TensorVar<T>(Tensor<T> tensor);

    /// <summary>Standard deviation of all elements, returns scalar tensor.</summary>
    Tensor<T> TensorStd<T>(Tensor<T> tensor);

    /// <summary>Element-wise square: x^2.</summary>
    Tensor<T> TensorSquare<T>(Tensor<T> tensor);

    /// <summary>LogSumExp: log(sum(exp(x))). Numerically stable.</summary>
    Tensor<T> TensorLogSumExp<T>(Tensor<T> tensor);

    /// <summary>L2 norm: sqrt(sum(x^2)).</summary>
    Tensor<T> TensorNorm<T>(Tensor<T> tensor);

    /// <summary>Adaptive max pool 2D.</summary>
    Tensor<T> TensorAdaptiveMaxPool2D<T>(Tensor<T> input, int[] outputSize);

    /// <summary>Where: select from x or y based on condition.</summary>
    Tensor<T> TensorWhere<T>(bool[] condition, Tensor<T> x, Tensor<T> y);

    /// <summary>MaskedFill: fill where mask is true.</summary>
    Tensor<T> TensorMaskedFill<T>(Tensor<T> tensor, bool[] mask, T value);

    /// <summary>Scaled dot-product attention: softmax(Q@K^T/sqrt(dk)) @ V.</summary>
    Tensor<T> TensorScaledDotProductAttention<T>(Tensor<T> query, Tensor<T> key, Tensor<T> value);

    /// <summary>IndexSelect with differentiable backward.</summary>
    Tensor<T> TensorIndexSelectDiff<T>(Tensor<T> source, Tensor<int> indices, int axis);

    #region Hyperbolic Manifold Operations

    /// <summary>
    /// Exponential map on the Poincaré ball: projects a tangent vector at a point onto the manifold.
    /// Formula: exp_x(v) = x ⊕_c tanh(√c · λ_x · ||v|| / 2) · (v / (√c · ||v||))
    /// where λ_x = 2 / (1 - c||x||²) is the conformal factor and c = |curvature|.
    /// The curvature parameter is the negative sectional curvature (e.g., -1.0 for standard hyperbolic space).
    /// Internally, c = |curvature| is used in formulas where √c appears.
    /// </summary>
    Vector<T> PoincareExpMap<T>(Vector<T> basePoint, Vector<T> tangentVector, T curvature);

    /// <summary>
    /// Logarithmic map on the Poincaré ball: computes the tangent vector from one point to another.
    /// Inverse of PoincareExpMap.
    /// </summary>
    Vector<T> PoincareLogMap<T>(Vector<T> basePoint, Vector<T> targetPoint, T curvature);

    /// <summary>
    /// Möbius addition: the hyperbolic analog of vector addition that stays inside the Poincaré ball.
    /// Formula: x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
    /// </summary>
    Vector<T> MobiusAdd<T>(Vector<T> x, Vector<T> y, T curvature);

    /// <summary>
    /// Geodesic distance between two points on the Poincaré ball.
    /// Formula: d(x,y) = (2/√c) * arctanh(√c * ||(-x) ⊕ y||)
    /// </summary>
    T PoincareDistance<T>(Vector<T> x, Vector<T> y, T curvature);

    /// <summary>
    /// Parallel transport: moves a tangent vector from one point to another along the geodesic.
    /// </summary>
    Vector<T> PoincareParallelTransport<T>(Vector<T> x, Vector<T> y, Vector<T> v, T curvature);

    /// <summary>
    /// Projects a point onto the Poincaré ball (clamps to valid region ||x|| &lt; 1/√c).
    /// </summary>
    Vector<T> PoincareProject<T>(Vector<T> point, T curvature, T epsilon);

    /// <summary>
    /// Exponential map on the hyperboloid model.
    /// Formula: exp_x(v) = cosh(√c · ||v||_L) · x + sinh(√c · ||v||_L) · (v / (√c · ||v||_L))
    /// where c = |curvature| and ||v||_L is the Lorentzian norm of v.
    /// </summary>
    Vector<T> HyperboloidExpMap<T>(Vector<T> basePoint, Vector<T> tangentVector, T curvature);

    /// <summary>
    /// Logarithmic map on the hyperboloid model.
    /// </summary>
    Vector<T> HyperboloidLogMap<T>(Vector<T> basePoint, Vector<T> targetPoint, T curvature);

    /// <summary>
    /// Geodesic distance on the hyperboloid using the Minkowski inner product.
    /// Formula: d(x,y) = (1/√c) * arcosh(-c * ⟨x,y⟩_L)
    /// </summary>
    T HyperboloidDistance<T>(Vector<T> x, Vector<T> y, T curvature);

    /// <summary>
    /// Projects a point onto the hyperboloid (ensures constraint: -x₀² + x₁² + ... + xₙ² = -1/c).
    /// </summary>
    Vector<T> HyperboloidProject<T>(Vector<T> point, T curvature);

    /// <summary>
    /// Converts a point from Poincaré ball model to hyperboloid model.
    /// </summary>
    Vector<T> PoincareToHyperboloid<T>(Vector<T> poincarePoint, T curvature);

    /// <summary>
    /// Converts a point from hyperboloid model to Poincaré ball model.
    /// </summary>
    Vector<T> HyperboloidToPoincare<T>(Vector<T> hyperboloidPoint, T curvature);

    /// <summary>
    /// Batched exponential map on the Poincaré ball.
    /// </summary>
    Matrix<T> PoincareExpMapBatch<T>(Matrix<T> basePoints, Matrix<T> tangentVectors, T curvature);

    /// <summary>
    /// Batched geodesic distance computation on the Poincaré ball.
    /// </summary>
    Vector<T> PoincareDistanceBatch<T>(Matrix<T> x, Matrix<T> y, T curvature);

    #endregion

    #region Advanced Algebra Operations

    /// <summary>Batch octonion-octonion multiplication (non-associative).</summary>
    Octonion<T>[] OctonionMultiplyBatch<T>(Octonion<T>[] left, Octonion<T>[] right);

    /// <summary>Batch octonion addition.</summary>
    Octonion<T>[] OctonionAddBatch<T>(Octonion<T>[] left, Octonion<T>[] right);

    /// <summary>Batch octonion conjugation.</summary>
    Octonion<T>[] OctonionConjugateBatch<T>(Octonion<T>[] octonions);

    /// <summary>Computes batch octonion norms.</summary>
    T[] OctonionNormBatch<T>(Octonion<T>[] octonions);

    /// <summary>Octonion-matrix multiplication for neural network layers.</summary>
    Octonion<T>[,] OctonionMatMul<T>(Octonion<T>[,] input, Octonion<T>[,] weight);

    /// <summary>
    /// Tensor-based octonion matrix multiplication. Input and weight tensors store octonion
    /// components in the last dimension (size 8). Output tensor has shape [batch, outputFeatures, 8].
    /// This is the unified Tensor path — no Octonion&lt;T&gt; object allocation needed.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, inputFeatures, 8].</param>
    /// <param name="weight">Weight tensor with shape [outputFeatures, inputFeatures, 8].</param>
    /// <returns>Output tensor with shape [batch, outputFeatures, 8].</returns>
    Tensor<T> OctonionMatMulTensor<T>(Tensor<T> input, Tensor<T> weight);

    /// <summary>
    /// Tensor-based octonion addition. Both tensors must have 8 as last dimension.
    /// </summary>
    Tensor<T> OctonionAddTensor<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>Batch geometric product of multivectors.</summary>
    Multivector<T>[] GeometricProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right);

    /// <summary>Batch wedge (outer) product of multivectors.</summary>
    Multivector<T>[] WedgeProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right);

    /// <summary>Batch inner product of multivectors.</summary>
    Multivector<T>[] InnerProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right);

    /// <summary>Batch multivector addition.</summary>
    Multivector<T>[] MultivectorAddBatch<T>(Multivector<T>[] left, Multivector<T>[] right);

    /// <summary>Batch multivector reverse operation.</summary>
    Multivector<T>[] MultivectorReverseBatch<T>(Multivector<T>[] multivectors);

    /// <summary>Batch grade projection - extracts components of a specific grade.</summary>
    Multivector<T>[] GradeProjectBatch<T>(Multivector<T>[] multivectors, int grade);

    /// <summary>Batch exponential map for SO(3) (rotation group).</summary>
    So3<T>[] So3ExpBatch<T>(So3Group<T> group, Vector<T>[] tangentVectors);

    /// <summary>Batch logarithm map for SO(3).</summary>
    Vector<T>[] So3LogBatch<T>(So3Group<T> group, So3<T>[] rotations);

    /// <summary>Batch group composition for SO(3).</summary>
    So3<T>[] So3ComposeBatch<T>(So3Group<T> group, So3<T>[] left, So3<T>[] right);

    /// <summary>Batch exponential map for SE(3) (rigid transformation group).</summary>
    Se3<T>[] Se3ExpBatch<T>(Se3Group<T> group, Vector<T>[] tangentVectors);

    /// <summary>Batch logarithm map for SE(3).</summary>
    Vector<T>[] Se3LogBatch<T>(Se3Group<T> group, Se3<T>[] transforms);

    /// <summary>Batch group composition for SE(3).</summary>
    Se3<T>[] Se3ComposeBatch<T>(Se3Group<T> group, Se3<T>[] left, Se3<T>[] right);

    /// <summary>Batch adjoint representation for SO(3).</summary>
    Matrix<T>[] So3AdjointBatch<T>(So3Group<T> group, So3<T>[] rotations);

    #endregion

    #region Complex Tensor Operations

    /// <summary>
    /// Element-wise complex multiplication on interleaved real/imaginary tensors.
    /// Input format: [re0, im0, re1, im1, ...] where length must be even.
    /// (a_re + a_im*i) * (b_re + b_im*i) = (a_re*b_re - a_im*b_im) + (a_re*b_im + a_im*b_re)*i
    /// </summary>
    Tensor<T> TensorComplexMultiply<T>(Tensor<T> a, Tensor<T> b);

    /// <summary>
    /// Complex conjugate on interleaved real/imaginary tensors.
    /// Negates imaginary components (odd indices): [re0, -im0, re1, -im1, ...]
    /// </summary>
    Tensor<T> TensorComplexConjugate<T>(Tensor<T> a);

    /// <summary>
    /// Complex magnitude on interleaved real/imaginary tensors.
    /// Computes sqrt(re^2 + im^2) per complex pair.
    /// Output length is half the input length.
    /// </summary>
    Tensor<T> TensorComplexMagnitude<T>(Tensor<T> a);

    // --- Native Complex<T> Tensor Operations ---
    // These operate on Tensor<Complex<T>> directly, avoiding interleaved format conversion overhead.
    // For maximum performance in spectral processing (FFT, filtering, HRE).
    //
    // CPU path: Operates on Tensor<Complex<T>> directly with no conversion.
    // GPU path: DirectGpuTensorEngine decomposes Tensor<Complex<T>> to split
    // real/imag float[] arrays, uploads to GPU, dispatches via
    // IDirectGpuBackend.SplitComplex* methods, downloads and recomposes.
    // FFT/IFFT dispatch through existing backend.FFT() with split buffers.

    /// <summary>
    /// Forward 1D FFT on a real-valued tensor, returning native Complex&lt;T&gt; tensor.
    /// Transforms along the last axis. Output has the same shape as input, with each element
    /// being a complex frequency bin.
    /// </summary>
    /// <param name="input">Real-valued input signal. Length along last axis must be a power of 2.</param>
    /// <returns>Complex-valued frequency domain tensor of same shape.</returns>
    /// <exception cref="ArgumentException">Thrown if input length is not a power of 2.</exception>
    Tensor<Complex<T>> NativeComplexFFT<T>(Tensor<T> input);

    /// <summary>
    /// Span-based forward 1D FFT on a real-valued signal.
    /// Writes complex frequency bins directly into the caller-provided output span,
    /// eliminating per-call wrapping overhead (virtual indexer, tensor allocation) for
    /// hot paths that call FFT tens of thousands of times.
    /// Internally dispatches to type-specialized float/double kernels via reinterpret;
    /// the generic signature is uniform with the rest of the engine API.
    /// </summary>
    /// <typeparam name="T">Element type (float or double recommended for fast path).</typeparam>
    /// <param name="input">Real-valued input samples. Length must be a power of 2.</param>
    /// <param name="output">Preallocated complex output buffer. Length must equal input length.</param>
    /// <exception cref="ArgumentException">Thrown if lengths mismatch or input is not a power of 2.</exception>
    void NativeComplexFFTSpan<T>(ReadOnlySpan<T> input, Span<Complex<T>> output);

    /// <summary>
    /// Span-based inverse 1D FFT (complex-to-complex) with 1/N normalization.
    /// </summary>
    /// <param name="input">Complex input spectrum. Length must be a power of 2.</param>
    /// <param name="output">Preallocated complex output buffer. Length must equal input length.</param>
    /// <exception cref="ArgumentException">Thrown if lengths mismatch or input is not a power of 2.</exception>
    void NativeComplexIFFTSpan<T>(ReadOnlySpan<Complex<T>> input, Span<Complex<T>> output);

    /// <summary>
    /// Span-based complex-to-complex forward 1D FFT.
    /// </summary>
    /// <param name="input">Complex input samples. Length must be a power of 2.</param>
    /// <param name="output">Preallocated complex output buffer. Length must equal input length.</param>
    /// <exception cref="ArgumentException">Thrown if lengths mismatch or input is not a power of 2.</exception>
    void NativeComplexFFTComplexSpan<T>(ReadOnlySpan<Complex<T>> input, Span<Complex<T>> output);

    /// <summary>
    /// Span-based inverse 1D FFT returning real output (Hermitian symmetry assumed).
    /// Most important variant for Hilbert/PAC/MFCC pipelines that need to return to time-domain real.
    /// Applies 1/N normalization, discards imaginary part.
    /// </summary>
    /// <param name="input">Complex input spectrum (should satisfy Hermitian symmetry for meaningful output). Length must be a power of 2.</param>
    /// <param name="output">Preallocated real output buffer. Length must equal input length.</param>
    /// <exception cref="ArgumentException">Thrown if lengths mismatch or input is not a power of 2.</exception>
    void NativeComplexIFFTRealSpan<T>(ReadOnlySpan<Complex<T>> input, Span<T> output);

    /// <summary>
    /// Fused analytic-signal kernel (Hilbert transform via FFT).
    /// Computes analytic signal z(t) = x(t) + i·H{x}(t) in one fused call:
    /// forward FFT → zero negative frequencies (and optionally zero outside [freqLow, freqHigh])
    /// → double positive bins → inverse FFT. Collapses what is otherwise 3 separate engine
    /// calls (FFT, bin masking, IFFT) into a single kernel.
    /// </summary>
    /// <param name="input">Real-valued time-domain input. Last axis length must be power of 2.</param>
    /// <param name="freqLow">Optional low-frequency cutoff in Hz (inclusive). Bins below are zeroed.</param>
    /// <param name="freqHigh">Optional high-frequency cutoff in Hz (exclusive). Bins at/above are zeroed.</param>
    /// <param name="sampleRate">Sample rate in Hz for interpreting freqLow/freqHigh. Must be positive and finite.
    /// Band-limiting is effectively disabled when the defaults <paramref name="freqLow"/>=0 and
    /// <paramref name="freqHigh"/>=<see cref="double.MaxValue"/> are used; sampleRate then has no effect on the output.</param>
    /// <returns>Complex-valued analytic signal tensor of same shape as input.</returns>
    /// <exception cref="ArgumentNullException">Thrown if input is null.</exception>
    /// <exception cref="ArgumentException">Thrown if sampleRate is non-positive/non-finite,
    /// if freqLow is negative or NaN, or if freqHigh &lt; freqLow.</exception>
    Tensor<Complex<T>> NativeAnalyticSignal<T>(Tensor<T> input, double freqLow = 0.0, double freqHigh = double.MaxValue, double sampleRate = 1.0);

    /// <summary>
    /// Per-row L2 normalization. For each row of a 2D tensor, divides by its L2 norm.
    /// Uses SIMD for the sum-of-squares accumulation and multiplication. Rows with zero norm
    /// are left as zeros (no division).
    /// </summary>
    /// <param name="input">2D input tensor [rows, cols].</param>
    /// <param name="inPlace">If true, writes back into the input buffer (no allocation). Default false.</param>
    /// <returns>2D output tensor of same shape with each row having unit L2 norm.</returns>
    Tensor<T> NativeNormalizeRows<T>(Tensor<T> input, bool inPlace = false);

    /// <summary>
    /// Element-wise hyperbolic tangent with SIMD acceleration on float/double.
    /// </summary>
    /// <remarks>
    /// <b>Non-differentiable:</b> this op is registered as non-differentiable in the autograd
    /// <c>OpRegistry</c> and does not record a backward. Use the differentiable equivalent
    /// <c>Tanh&lt;T&gt;</c> in training graphs; use <c>NativeTanh</c> only for inference or
    /// post-training pipelines (e.g. HRE spectral/audio ops) that need SIMD throughput.
    /// </remarks>
    Tensor<T> NativeTanh<T>(Tensor<T> input);

    /// <summary>
    /// Element-wise exponential (e^x) with SIMD acceleration on float/double.
    /// </summary>
    /// <remarks>
    /// <b>Non-differentiable:</b> this op is registered as non-differentiable in the autograd
    /// <c>OpRegistry</c> and does not record a backward. Use the differentiable equivalent
    /// <c>TensorExp&lt;T&gt;</c> in training graphs; use <c>NativeExp</c> only for inference or
    /// post-training pipelines that need SIMD throughput.
    /// </remarks>
    Tensor<T> NativeExp<T>(Tensor<T> input);

    /// <summary>
    /// Element-wise atan2(imag, real) with SIMD acceleration. Both tensors must have the same shape.
    /// </summary>
    /// <remarks>
    /// <b>Non-differentiable:</b> no backward is recorded. Intended for phase extraction in
    /// signal-processing pipelines, not for training graphs.
    /// </remarks>
    Tensor<T> NativeAtan2<T>(Tensor<T> imag, Tensor<T> real);

    /// <summary>
    /// Computes magnitude and phase of a complex tensor in a single pass.
    /// Returns magnitude; writes phase into the out parameter.
    /// </summary>
    /// <remarks>
    /// <b>Non-differentiable:</b> no backward is recorded. Intended for spectral feature
    /// extraction, not for training graphs.
    /// </remarks>
    Tensor<T> NativeMagnitudeAndPhase<T>(Tensor<Complex<T>> input, out Tensor<T> phase);

    /// <summary>
    /// Third-order bispectrum: B(f1, f2) = X(f1) · X(f2) · conj(X(f1+f2)).
    /// Input is a 1D complex spectrum of length N (must be a power of 2 for FFT-derived spectra).
    /// Output shape is [maxF1, maxF2]. Used for higher-order phase-coupling features.
    /// </summary>
    /// <param name="spectrum">1D complex spectrum.</param>
    /// <param name="maxF1">Maximum f1 frequency bin (exclusive).</param>
    /// <param name="maxF2">Maximum f2 frequency bin (exclusive).</param>
    Tensor<Complex<T>> NativeBispectrum<T>(Tensor<Complex<T>> spectrum, int maxF1, int maxF2);

    /// <summary>
    /// Fourth-order trispectrum: T(f1, f2, f3) = X(f1) · X(f2) · X(f3) · conj(X(f1+f2+f3)).
    /// </summary>
    Tensor<Complex<T>> NativeTrispectrum<T>(Tensor<Complex<T>> spectrum, int maxF1, int maxF2, int maxF3);

    /// <summary>
    /// Batched forward pass for resonant-cavity style operators.
    /// Applies a per-cavity transfer function (set of complex filter responses) to a batch of inputs.
    /// Input: [batch, N]. Filters: [numCavities, N] complex. Output: [batch, numCavities, N] real.
    /// Fuses FFT → per-cavity multiply → IFFT → nonlinear bounce (tanh) across all cavities in one call.
    /// </summary>
    /// <param name="input">Real-valued [batch, N] input waveforms.</param>
    /// <param name="cavityFilters">Complex [numCavities, N] frequency-domain filter responses.</param>
    /// <param name="numBounces">Number of recursive nonlinear bounces per cavity.</param>
    /// <returns>[batch, numCavities, N] real-valued output.</returns>
    Tensor<T> NativeBatchedCavityForward<T>(Tensor<T> input, Tensor<Complex<T>> cavityFilters, int numBounces);

    /// <summary>
    /// End-to-end MFCC feature extraction pipeline for a batch of waveforms.
    /// Fuses STFT → power spectrum → mel filterbank → log → DCT → per-segment pooling.
    /// </summary>
    /// <param name="waveforms">[batch, numSamples] or [numSamples] real waveforms.</param>
    /// <param name="numSegments">Number of non-overlapping time segments.</param>
    /// <param name="numMfcc">Number of MFCC coefficients per segment.</param>
    /// <param name="paddedDim">Padded dimension (power of 2) used for FFT.</param>
    /// <returns>[batch, numSegments * numMfcc] or [numSegments * numMfcc] feature tensor.</returns>
    Tensor<T> NativeMfccFeatures<T>(Tensor<T> waveforms, int numSegments, int numMfcc, int paddedDim);

    /// <summary>
    /// End-to-end wideband spectral feature extraction pipeline.
    /// Fuses segmentation → FFT → log-magnitude binning → pooling.
    /// </summary>
    Tensor<T> NativeWidebandFeatures<T>(Tensor<T> waveforms, int numSegments, int numBins);

    /// <summary>
    /// End-to-end phase-amplitude coupling (PAC) feature extraction pipeline.
    /// For each gamma band: analytic signal on theta → phase, analytic signal on gamma → amplitude,
    /// compute modulation index. Fuses Hilbert + envelope + PAC MI in one call.
    /// </summary>
    /// <param name="waveforms">[batch, numSamples] or [numSamples] real waveforms.</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="envelopeRate">Target envelope sample rate in Hz for decimation.</param>
    /// <param name="thetaLow">Theta band low-frequency (Hz).</param>
    /// <param name="thetaHigh">Theta band high-frequency (Hz).</param>
    /// <param name="gammaBands">Array of (lowHz, highHz) gamma band tuples.</param>
    /// <returns>[batch, gammaBands.Length] PAC modulation index tensor.</returns>
    Tensor<T> NativePacFeatures<T>(Tensor<T> waveforms, int sampleRate, int envelopeRate,
        double thetaLow, double thetaHigh, (double low, double high)[] gammaBands);

    /// <summary>
    /// Inverse 1D FFT from Complex&lt;T&gt; tensor, returning real-valued tensor.
    /// Extracts only the real component of the inverse transform. Use this when the original
    /// signal was real-valued (Hermitian symmetry assumed). Applies 1/N normalization.
    /// </summary>
    /// <param name="input">Complex-valued frequency domain tensor. Length must be a power of 2.</param>
    /// <returns>Real-valued time domain tensor of same shape.</returns>
    /// <exception cref="ArgumentException">Thrown if input length is not a power of 2.</exception>
    Tensor<T> NativeComplexIFFTReal<T>(Tensor<Complex<T>> input);

    /// <summary>
    /// Inverse 1D FFT from Complex&lt;T&gt; tensor, returning Complex&lt;T&gt; tensor.
    /// General complex-to-complex IFFT with 1/N normalization. Use this when the time-domain
    /// signal may have nonzero imaginary components (e.g., intermediate results in spectral
    /// filtering chains, or Hilbert transform output).
    /// </summary>
    /// <param name="input">Complex-valued frequency domain tensor. Length must be a power of 2.</param>
    /// <returns>Complex-valued time domain tensor of same shape.</returns>
    /// <exception cref="ArgumentException">Thrown if input length is not a power of 2.</exception>
    Tensor<Complex<T>> NativeComplexIFFT<T>(Tensor<Complex<T>> input);

    /// <summary>
    /// Forward 1D FFT on a complex-valued tensor, returning Complex&lt;T&gt; tensor.
    /// Complex-to-complex FFT — use when input already has nonzero imaginary components
    /// (e.g., intermediate results in spectral filtering chains).
    /// </summary>
    /// <param name="input">Complex-valued input tensor. Last axis length must be a power of 2.</param>
    /// <returns>Complex-valued frequency domain tensor of same shape.</returns>
    /// <exception cref="ArgumentException">Thrown if last axis length is not a power of 2.</exception>
    Tensor<Complex<T>> NativeComplexFFTComplex<T>(Tensor<Complex<T>> input);

    /// <summary>
    /// 2D FFT on the last two axes of a real tensor. Input shape [..., H, W] where H and W
    /// must be powers of 2. Applies row-wise FFT then column-wise FFT using the fast
    /// NativeComplexFFT batched path. Handles arbitrary leading batch dimensions (e.g.
    /// [B, C, H, W] for batched multi-channel vision models per Issue #137/#139).
    /// </summary>
    Tensor<Complex<T>> NativeComplexFFT2D<T>(Tensor<T> input);

    /// <summary>
    /// Inverse 2D FFT returning a real tensor. Input shape [..., H, W] complex.
    /// Applies column-wise IFFT then row-wise IFFT. Only the real part of the
    /// final result is returned — imaginary components are discarded. This is
    /// correct when the input spectrum represents a real-valued spatial signal
    /// (Hermitian symmetry), such as one produced by <see cref="NativeComplexFFT2D{T}"/>.
    /// For arbitrary complex spectra where the imaginary part is meaningful,
    /// use <see cref="NativeComplexIFFT{T}"/> instead.
    /// </summary>
    Tensor<T> NativeComplexIFFT2DReal<T>(Tensor<Complex<T>> input);

    /// <summary>
    /// N-D FFT over specified axes. Applies 1D FFT sequentially along each axis in the
    /// given order. Supports arbitrary tensor shapes and axis combinations, subsuming
    /// 1D (axes=[-1]) and 2D (axes=[-2,-1]) as special cases. Per Issue #135.
    /// </summary>
    /// <param name="input">Real-valued input tensor. Each transformed axis length must be a power of 2; non-transformed axes can have any length.</param>
    /// <param name="axes">Axes to transform. Negative indices count from the end.</param>
    Tensor<Complex<T>> NativeComplexFFTND<T>(Tensor<T> input, int[] axes);

    /// <summary>
    /// Inverse N-D FFT returning a real tensor. Applies 1D IFFT sequentially along each
    /// axis in reverse order. Only the real part of the final result is returned —
    /// imaginary components are discarded. Correct when the input was produced by
    /// <see cref="NativeComplexFFTND{T}"/> from a real-valued input.
    /// </summary>
    Tensor<T> NativeComplexIFFTNDReal<T>(Tensor<Complex<T>> input, int[] axes);

    /// <summary>
    /// Fused spectral filter: FFT2D(input) ⊙ filter → IFFT2DReal → output.
    /// Single engine call that eliminates 2 intermediate tensor allocations.
    /// Input shape: [H, W] or [..., H, W] (batches over leading dims).
    /// Filter shape: any rank ≥ 2 where last two dims match [H, W]. Leading dims
    /// are broadcast via modular indexing: [H,W] applies the same filter to every slice,
    /// [C,H,W] cycles per-channel across batches, [B,C,H,W] is a direct 1:1 match.
    /// The result is the real projection of the inverse transform — imaginary components
    /// from IFFT are discarded (correct when the original input was real-valued).
    /// </summary>
    /// <param name="input">Real-valued spatial input. Last two axes must be powers of 2.</param>
    /// <param name="filter">Complex-valued spectral filter. Last two dims must match input [H, W].
    /// Filter length must not exceed input length.</param>
    /// <returns>Real-valued filtered output of same shape as input (real projection of IFFT2D).</returns>
    Tensor<T> NativeSpectralFilter<T>(Tensor<T> input, Tensor<Complex<T>> filter);

    /// <summary>
    /// Batched spectral filter across samples and channels.
    /// Input: [B, C, H, W]. Filter: any rank ≥ 2 where last two dims match [H, W].
    /// Applies FFT2D → pointwise multiply → IFFT2DReal to every (b, c) slice in one call,
    /// replacing the O(B×C) dispatch loop that dominates vision-model training time.
    /// Filter broadcasting: [H,W] shared across all, [C,H,W] per-channel cycling across
    /// batches, [B,C,H,W] direct 1:1 match.
    /// The result is the real projection of each slice's inverse transform — imaginary
    /// components from IFFT are discarded (correct when the original input was real-valued).
    /// </summary>
    /// <param name="input">Real-valued 4D tensor [B, C, H, W]. H and W must be powers of 2.</param>
    /// <param name="filter">Complex-valued filter. Last two dims must match [H, W].
    /// Filter length must not exceed input length.</param>
    /// <returns>Real-valued output [B, C, H, W] (real projection of IFFT2D per slice).</returns>
    Tensor<T> NativeSpectralFilterBatch<T>(Tensor<T> input, Tensor<Complex<T>> filter);

    /// <summary>
    /// Selects the top-K elements by complex magnitude, zeroing all others.
    /// Used for spectral sparsity masking — retains the K strongest frequency components.
    /// </summary>
    /// <param name="input">Complex tensor to apply sparsity mask to.</param>
    /// <param name="k">Number of elements to retain.</param>
    /// <returns>Sparse complex tensor with only K non-zero elements (by magnitude).</returns>
    Tensor<Complex<T>> NativeComplexTopK<T>(Tensor<Complex<T>> input, int k);

    /// <summary>
    /// Applies softmax independently to each row of a 2D real tensor.
    /// Each row is normalized to sum to 1. For attention weight computation.
    /// </summary>
    /// <param name="input">2D tensor of shape [M, N].</param>
    /// <returns>2D tensor of same shape with softmax applied per row.</returns>
    /// <exception cref="ArgumentException">Thrown if input is not 2D.</exception>
    Tensor<T> TensorSoftmaxRows<T>(Tensor<T> input);

    /// <summary>
    /// Element-wise multiplication of two Complex&lt;T&gt; tensors (spectral filtering).
    /// (a.re*b.re - a.im*b.im) + i*(a.re*b.im + a.im*b.re) per element.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown if tensor lengths don't match.</exception>
    Tensor<Complex<T>> NativeComplexMultiply<T>(Tensor<Complex<T>> a, Tensor<Complex<T>> b);

    /// <summary>
    /// Element-wise conjugate of a Complex&lt;T&gt; tensor: (re, -im) per element.
    /// </summary>
    Tensor<Complex<T>> NativeComplexConjugate<T>(Tensor<Complex<T>> a);

    /// <summary>
    /// Extract magnitudes from Complex&lt;T&gt; tensor: sqrt(re^2 + im^2) per element.
    /// For performance-sensitive code where only ordering matters, use NativeComplexMagnitudeSquared instead.
    /// </summary>
    Tensor<T> NativeComplexMagnitude<T>(Tensor<Complex<T>> a);

    /// <summary>
    /// Extract squared magnitudes from Complex&lt;T&gt; tensor: re^2 + im^2 per element.
    /// Avoids the sqrt for better performance when magnitude ordering is sufficient.
    /// </summary>
    Tensor<T> NativeComplexMagnitudeSquared<T>(Tensor<Complex<T>> a);

    /// <summary>
    /// Extract phases from Complex&lt;T&gt; tensor: atan2(im, re) per element.
    /// Result is in radians, range [-pi, pi].
    /// </summary>
    Tensor<T> NativeComplexPhase<T>(Tensor<Complex<T>> a);

    /// <summary>
    /// Construct Complex&lt;T&gt; tensor from magnitude and phase tensors.
    /// result[i] = Complex(mag[i]*cos(phase[i]), mag[i]*sin(phase[i]))
    /// </summary>
    /// <exception cref="ArgumentException">Thrown if tensor lengths don't match.</exception>
    Tensor<Complex<T>> NativeComplexFromPolar<T>(Tensor<T> magnitudes, Tensor<T> phases);

    /// <summary>
    /// Scale all elements of a Complex&lt;T&gt; tensor by a real scalar.
    /// result[i] = Complex(a[i].re * scalar, a[i].im * scalar)
    /// </summary>
    Tensor<Complex<T>> NativeComplexScale<T>(Tensor<Complex<T>> a, T scalar);

    /// <summary>
    /// Cross-spectral density: X * conj(Y), fused for performance.
    /// Computes element-wise: result.re = x.re*y.re + x.im*y.im, result.im = x.im*y.re - x.re*y.im.
    /// Used in Hebbian learning and coherence analysis.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown if tensor lengths don't match.</exception>
    Tensor<Complex<T>> NativeComplexCrossSpectral<T>(Tensor<Complex<T>> x, Tensor<Complex<T>> y);

    /// <summary>
    /// Element-wise addition of two Complex&lt;T&gt; tensors.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown if tensor lengths don't match.</exception>
    Tensor<Complex<T>> NativeComplexAdd<T>(Tensor<Complex<T>> a, Tensor<Complex<T>> b);

    #endregion

    #region CTC Loss

    /// <summary>
    /// Connectionist Temporal Classification (CTC) loss.
    /// Computes per-batch CTC loss using the forward-backward algorithm over the label lattice.
    /// </summary>
    /// <param name="logProbs">Log probabilities tensor [T, N, C] where T=time, N=batch, C=classes.
    /// Must be log-softmax output (not raw logits).</param>
    /// <param name="targets">Target label indices (concatenated, no blanks) [sum(targetLengths)].</param>
    /// <param name="inputLengths">Length of each input sequence [N].</param>
    /// <param name="targetLengths">Length of each target sequence [N].</param>
    /// <param name="blank">Index of the blank label (default 0).</param>
    /// <returns>Per-batch CTC loss tensor [N].</returns>
    Tensor<T> TensorCTCLoss<T>(Tensor<T> logProbs, Tensor<int> targets, int[] inputLengths, int[] targetLengths, int blank = 0);

    #endregion
}

/// <summary>
/// Marker interface retained for backward compatibility. All tensor-level operations
/// are now part of <see cref="IEngine"/> directly.
/// </summary>
public interface ITensorLevelEngine : IEngine
{
}
