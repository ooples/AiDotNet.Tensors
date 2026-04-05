using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.CpuJit;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Groups;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Operators;
using static AiDotNet.Tensors.Helpers.CpuParallelSettings;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// CPU-based execution engine using INumericOperations for type-generic operations.
/// </summary>
/// <remarks>
/// <para>
/// CpuEngine provides the default execution backend for AiDotNet. It works with
/// any numeric type that implements INumericOperations{T}, including decimal,
/// BigInteger, and custom numeric types.
/// </para>
/// <para><b>For Beginners:</b> This is the standard, "always works" mode.
///
/// CpuEngine characteristics:
/// - Works with ANY numeric type (float, double, decimal, BigInteger, custom types)
/// - No special hardware required
/// - Good performance for small-to-medium datasets
/// - Single-threaded by default (can be parallelized in future versions)
///
/// When to use:
/// - You need decimal or high-precision arithmetic
/// - You don't have a GPU
/// - Your datasets are small (< 100K parameters)
/// - You're using custom numeric types
/// </para>
/// </remarks>
public class CpuEngine : ITensorLevelEngine
{
    /// <inheritdoc/>
    public string Name => "CPU Engine";

    /// <inheritdoc/>
    public bool SupportsGpu => false;

    /// <inheritdoc/>
    public DirectGpu.DirectGpuEngine? DirectGpu => Engine.DirectGpu;

    /// <inheritdoc/>
    public Vector<T> Add<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10x speedup for float)
        return TensorPrimitivesHelper<T>.Add(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Subtract<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10x speedup for float)
        return TensorPrimitivesHelper<T>.Subtract(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10x speedup for float)
        return TensorPrimitivesHelper<T>.Multiply(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Multiply<T>(Vector<T> vector, T scalar)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Create scalar vector and use SIMD-optimized multiplication
        var scalarVector = Vector<T>.CreateDefault(vector.Length, scalar);
        return TensorPrimitivesHelper<T>.Multiply(vector, scalarVector);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        // Check for division by zero before calling TensorPrimitivesHelper
        var numOps = MathHelper.GetNumericOperations<T>();
        var bArray = b.GetDataArray();
        for (int i = 0; i < bArray.Length; i++)
        {
            if (numOps.Equals(bArray[i], numOps.Zero))
            {
                throw new DivideByZeroException($"Division by zero at index {i}");
            }
        }

        // Use SIMD-optimized TensorPrimitivesHelper (5-10x speedup for float)
        return TensorPrimitivesHelper<T>.Divide(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Divide<T>(Vector<T> vector, T scalar)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Check for division by zero
        if (numOps.Equals(scalar, numOps.Zero))
        {
            throw new DivideByZeroException("Cannot divide by zero");
        }

        // Create scalar vector and use SIMD-optimized division
        var scalarVector = Vector<T>.CreateDefault(vector.Length, scalar);
        return TensorPrimitivesHelper<T>.Divide(vector, scalarVector);
    }

    /// <inheritdoc/>
    public Vector<T> Sqrt<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (5-10x speedup for float)
        return TensorPrimitivesHelper<T>.Sqrt(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Power<T>(Vector<T> vector, T exponent)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Power(vector[i], exponent);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Max<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.GreaterThan(a[i], b[i]) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Min<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector lengths must match. Got {a.Length} and {b.Length}");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            result[i] = numOps.LessThan(a[i], b[i]) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Abs<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Abs(vector[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Exp<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6x speedup for float)
        return TensorPrimitivesHelper<T>.Exp(vector);
    }

    /// <inheritdoc/>
    public Vector<T> Log<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6x speedup for float)
        return TensorPrimitivesHelper<T>.Log(vector);
    }

    /// <inheritdoc/>
    /// <inheritdoc/>
    public Vector<T> Exp2<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Exp2(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Exp2(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback: 2^x
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Pow(2.0, val));
        }
        return genericResult;
    }

    /// <inheritdoc/>
    public Vector<T> Exp10<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Exp10(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Exp10(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback: 10^x
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Pow(10.0, val));
        }
        return genericResult;
    }

    public Vector<T> Sign<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            // Sign returns -1, 0, or +1
            if (numOps.GreaterThan(vector[i], numOps.Zero))
            {
                result[i] = numOps.One;
            }
            else if (numOps.LessThan(vector[i], numOps.Zero))
            {
                result[i] = numOps.Negate(numOps.One);
            }
            else
            {
                result[i] = numOps.Zero;
            }
        }

        return result;
    }

    #region Reduction Operations

    /// <inheritdoc/>
    public unsafe T Sum<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Float fast path: bypass generic dispatch + Span overhead
        if (typeof(T) == typeof(float))
        {
            T[] arr = vector.GetDataArray();
            float[] fArr = Unsafe.As<T[], float[]>(ref arr);
            float result;
            fixed (float* ptr = fArr)
            {
                result = SimdKernels.SumUnsafe(ptr, vector.Length);
            }
            return Unsafe.As<float, T>(ref result);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;

        for (int i = 0; i < vector.Length; i++)
        {
            sum = numOps.Add(sum, vector[i]);
        }

        return sum;
    }

    /// <inheritdoc/>
    public T DotProduct<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vectors must have the same length for dot product. Got lengths {a.Length} and {b.Length}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        T result = numOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            result = numOps.Add(result, numOps.Multiply(a[i], b[i]));
        }

        return result;
    }

    /// <inheritdoc/>
    public T DotProduct<T>(Vector<T> a, Vector<T> b, int bOffset, int bStride)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (bStride == 0) throw new ArgumentException("bStride must be non-zero.", nameof(bStride));

        var numOps = MathHelper.GetNumericOperations<T>();
        T result = numOps.Zero;

        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        int len = a.Length;
        int bLen = b.Length;

        for (int i = 0; i < len; i++)
        {
            int bIdx = bOffset + i * bStride;
            // Out-of-range elements contribute 0 (boundary clamping for time series AR/MA)
            if (bIdx >= 0 && bIdx < bLen)
            {
                result = numOps.Add(result, numOps.Multiply(aSpan[i], bSpan[bIdx]));
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public unsafe T Mean<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Cannot compute mean of empty vector.");

        // Float fast path: bypass generic dispatch + Span overhead
        if (typeof(T) == typeof(float))
        {
            T[] arr = vector.GetDataArray();
            float[] fArr = Unsafe.As<T[], float[]>(ref arr);
            float result;
            fixed (float* ptr = fArr)
            {
                result = SimdKernels.SumUnsafe(ptr, vector.Length) / vector.Length;
            }
            return Unsafe.As<float, T>(ref result);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = Sum(vector);
        T length = numOps.FromDouble(vector.Length);
        return numOps.Divide(sum, length);
    }

    /// <inheritdoc/>
    public Vector<T> Softmax<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Cannot compute softmax of empty vector.");

        // Use SIMD-optimized TensorPrimitivesHelper (3-6x speedup for float)
        return TensorPrimitivesHelper<T>.Softmax(vector);
    }

    /// <inheritdoc/>
    public T CosineSimilarity<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length.");

        // Use SIMD-optimized TensorPrimitivesHelper (10-15x speedup for float)
        return TensorPrimitivesHelper<T>.CosineSimilarity(a, b);
    }

    /// <inheritdoc/>
    public Vector<T> Log2<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized TensorPrimitivesHelper (3-6x speedup for float)
        return TensorPrimitivesHelper<T>.Log2(vector);
    }

    /// <inheritdoc/>
    public Vector<T> ExpM1<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Subtract(numOps.Exp(vector[i]), numOps.One);
        }
        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Log1P<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Log(numOps.Add(vector[i], numOps.One));
        }
        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Negate<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Negate(vector[i]);
        }
        return result;
    }

    /// <inheritdoc/>
    public unsafe Vector<T> StridedGather<T>(Vector<T> source, int offset, int stride, int count = -1)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset), "Offset must be non-negative.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be positive.");

        if (count < 0)
        {
            count = offset < source.Length ? (source.Length - offset + stride - 1) / stride : 0;
        }

        if (count == 0) return new Vector<T>(0);

        int lastIndex = offset + (count - 1) * stride;
        if (lastIndex >= source.Length)
            throw new ArgumentOutOfRangeException(nameof(count),
                $"Strided gather would access index {lastIndex} but source length is {source.Length}.");

        var result = VectorAllocator.Rent<T>(count);

        // Float fast path: AVX2 VGATHERDPS (8 floats per instruction)
        // Pin via _memory to correctly handle sliced Vector.Wrap(array, offset, length)
        if (typeof(T) == typeof(float) && CpuParallelSettings.EnableSimd && CpuParallelSettings.EnableAvx2Gather)
        {
            if (source._cachedArray is float[] srcF && result._cachedArray is float[] dstF)
            {
                fixed (float* pSrc = srcF)
                fixed (float* pDst = dstF)
                    SimdKernels.StridedGatherFloat(pSrc, offset, stride, pDst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                    result[i] = source[offset + i * stride];
            }
            return result;
        }

        // Double fast path: AVX2 VGATHERQPD (4 doubles per instruction)
        if (typeof(T) == typeof(double) && CpuParallelSettings.EnableSimd && CpuParallelSettings.EnableAvx2Gather)
        {
            if (source._cachedArray is double[] srcD && result._cachedArray is double[] dstD)
            {
                fixed (double* pSrc = srcD)
                fixed (double* pDst = dstD)
                    SimdKernels.StridedGatherDouble(pSrc, offset, stride, pDst, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                    result[i] = source[offset + i * stride];
            }
            return result;
        }

        // Generic fallback: scalar loop
        for (int i = 0; i < count; i++)
        {
            result[i] = source[offset + i * stride];
        }

        return result;
    }

    /// <inheritdoc/>
    public unsafe void StridedScatter<T>(Vector<T> destination, Vector<T> source, int offset, int stride)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset), "Offset must be non-negative.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be positive.");

        int count = source.Length;
        int lastIndex = count > 0 ? offset + (count - 1) * stride : -1;
        if (lastIndex >= destination.Length)
            throw new ArgumentOutOfRangeException(nameof(source),
                $"Strided scatter would access index {lastIndex} but destination length is {destination.Length}.");

        // Float fast path: unrolled scatter (no hardware scatter on x86, AVX-512 only)
        // Pin via _memory to correctly handle sliced Vector.Wrap(array, offset, length)
        if (typeof(T) == typeof(float) && CpuParallelSettings.EnableSimd)
        {
            if (source._cachedArray is float[] srcF && destination._cachedArray is float[] dstF)
            {
                fixed (float* pSrc = srcF)
                fixed (float* pDst = dstF)
                    SimdKernels.StridedScatterFloat(pSrc, pDst, offset, stride, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                    destination[offset + i * stride] = source[i];
            }
            return;
        }

        // Double fast path: unrolled scatter
        if (typeof(T) == typeof(double) && CpuParallelSettings.EnableSimd)
        {
            if (source._cachedArray is double[] srcD && destination._cachedArray is double[] dstD)
            {
                fixed (double* pSrc = srcD)
                fixed (double* pDst = dstD)
                    SimdKernels.StridedScatterDouble(pSrc, pDst, offset, stride, count);
            }
            else
            {
                for (int i = 0; i < count; i++)
                    destination[offset + i * stride] = source[i];
            }
            return;
        }

        // Generic fallback
        for (int i = 0; i < count; i++)
        {
            destination[offset + i * stride] = source[i];
        }
    }

    /// <inheritdoc/>
    public T Product<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Cannot compute product of empty vector.");
        var numOps = MathHelper.GetNumericOperations<T>();
        T product = numOps.One;
        for (int i = 0; i < vector.Length; i++)
        {
            product = numOps.Multiply(product, vector[i]);
        }
        return product;
    }

    /// <inheritdoc/>
    public T StdDev<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (vector.Length == 0) throw new ArgumentException("Cannot compute standard deviation of empty vector.");
        var numOps = MathHelper.GetNumericOperations<T>();

        T mean = Mean(vector);
        T sumSquaredDiff = numOps.Zero;

        for (int i = 0; i < vector.Length; i++)
        {
            T diff = numOps.Subtract(vector[i], mean);
            sumSquaredDiff = numOps.Add(sumSquaredDiff, numOps.Square(diff));
        }

        T variance = numOps.Divide(sumSquaredDiff, numOps.FromDouble(vector.Length));
        return numOps.Sqrt(variance);
    }

    /// <inheritdoc/>
    public T Norm<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        T sumSquares = numOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares = numOps.Add(sumSquares, numOps.Square(vector[i]));
        }
        return numOps.Sqrt(sumSquares);
    }

    /// <inheritdoc/>
    public T Distance<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new ArgumentException("Vectors must have the same length.");

        var numOps = MathHelper.GetNumericOperations<T>();
        T sumSquaredDiff = numOps.Zero;

        for (int i = 0; i < a.Length; i++)
        {
            T diff = numOps.Subtract(a[i], b[i]);
            sumSquaredDiff = numOps.Add(sumSquaredDiff, numOps.Square(diff));
        }

        return numOps.Sqrt(sumSquaredDiff);
    }

    /// <inheritdoc/>
    public Vector<T> MinMagnitude<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new ArgumentException("Vectors must have the same length.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            T absA = numOps.Abs(a[i]);
            T absB = numOps.Abs(b[i]);
            result[i] = numOps.LessThan(absA, absB) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> MaxMagnitude<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new ArgumentException("Vectors must have the same length.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            T absA = numOps.Abs(a[i]);
            T absB = numOps.Abs(b[i]);
            result[i] = numOps.GreaterThan(absA, absB) ? a[i] : b[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Clamp<T>(Vector<T> vector, T min, T max)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            if (numOps.LessThan(vector[i], min))
                result[i] = min;
            else if (numOps.GreaterThan(vector[i], max))
                result[i] = max;
            else
                result[i] = vector[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Lerp<T>(Vector<T> a, Vector<T> b, T t)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new ArgumentException("Vectors must have the same length.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(a.Length);

        for (int i = 0; i < a.Length; i++)
        {
            T diff = numOps.Subtract(b[i], a[i]);
            T scaled = numOps.Multiply(t, diff);
            result[i] = numOps.Add(a[i], scaled);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Reciprocal<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Divide(numOps.One, vector[i]);
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> ReciprocalSqrt<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = numOps.Divide(numOps.One, numOps.Sqrt(vector[i]));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Sin<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // For float and double, use SIMD-accelerated operators
        if (typeof(T) == typeof(float))
        {
            T[] inputData = vector.GetDataArray();
            T[] outputData = new T[vector.Length];

            // Reinterpret T[] as float[] since we know T is float
            float[] floatInput = Unsafe.As<float[]>(inputData);
            float[] floatOutput = Unsafe.As<float[]>(outputData);

            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(
                floatInput.AsSpan(),
                floatOutput.AsSpan());

            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < outputData.Length; i++)
            {
                result[i] = outputData[i];
            }
            return result;
        }
        else if (typeof(T) == typeof(double))
        {
            T[] inputData = vector.GetDataArray();
            T[] outputData = new T[vector.Length];

            // Reinterpret T[] as double[] since we know T is double
            double[] doubleInput = Unsafe.As<double[]>(inputData);
            double[] doubleOutput = Unsafe.As<double[]>(outputData);

            TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(
                doubleInput.AsSpan(),
                doubleOutput.AsSpan());

            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < outputData.Length; i++)
            {
                result[i] = outputData[i];
            }
            return result;
        }
        else
        {
            // Fallback for other types using INumericOperations
            var numOps = MathHelper.GetNumericOperations<T>();
            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                double val = Convert.ToDouble(vector[i]);
                result[i] = numOps.FromDouble(Math.Sin(val));
            }
            return result;
        }
    }

    /// <inheritdoc/>
    public Vector<T> Cos<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));

        // For float and double, use SIMD-accelerated operators
        if (typeof(T) == typeof(float))
        {
            T[] inputData = vector.GetDataArray();
            T[] outputData = new T[vector.Length];

            // Reinterpret T[] as float[] since we know T is float
            float[] floatInput = Unsafe.As<float[]>(inputData);
            float[] floatOutput = Unsafe.As<float[]>(outputData);

            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(
                floatInput.AsSpan(),
                floatOutput.AsSpan());

            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < outputData.Length; i++)
            {
                result[i] = outputData[i];
            }
            return result;
        }
        else if (typeof(T) == typeof(double))
        {
            T[] inputData = vector.GetDataArray();
            T[] outputData = new T[vector.Length];

            // Reinterpret T[] as double[] since we know T is double
            double[] doubleInput = Unsafe.As<double[]>(inputData);
            double[] doubleOutput = Unsafe.As<double[]>(outputData);

            TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(
                doubleInput.AsSpan(),
                doubleOutput.AsSpan());

            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < outputData.Length; i++)
            {
                result[i] = outputData[i];
            }
            return result;
        }
        else
        {
            // Fallback for other types using INumericOperations
            var numOps = MathHelper.GetNumericOperations<T>();
            var result = new Vector<T>(vector.Length);
            for (int i = 0; i < vector.Length; i++)
            {
                double val = Convert.ToDouble(vector[i]);
                result[i] = numOps.FromDouble(Math.Cos(val));
            }
            return result;
        }
    }

    /// <inheritdoc/>
    public virtual void SinCos<T>(Vector<T> vector, out Vector<T> sinResult, out Vector<T> cosResult)
    {
        // For now, compute separately (can be optimized later with simultaneous computation)
        sinResult = Sin(vector);
        cosResult = Cos(vector);
    }

    /// <inheritdoc/>
    public virtual void Sin(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Sin(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Cos(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Cos(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CosOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<LogOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp2(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Exp2OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp2(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Exp2OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp10(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Exp10OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Exp10(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Exp10OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void ExpM1(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpM1OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void ExpM1(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ExpM1OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log1P(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log1POperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log1P(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log1POperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Tan(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Tan(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<TanOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Asin<T>(Vector<T> vector)
    {
        if (vector is null)
        {
            throw new ArgumentNullException(nameof(vector));
        }

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Asin(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Asin(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Asin(val));
        }
        return genericResult;
    }

    /// <inheritdoc/>
    public void Asin(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Asin(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Acos<T>(Vector<T> vector)
    {
        if (vector is null)
        {
            throw new ArgumentNullException(nameof(vector));
        }

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Acos(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Acos(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Acos(val));
        }
        return genericResult;
    }

    /// <inheritdoc/>
    public void Acos(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcosOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Acos(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcosOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Atan<T>(Vector<T> vector)
    {
        if (vector is null)
        {
            throw new ArgumentNullException(nameof(vector));
        }

        // For float and double, use optimized span operations
        if (typeof(T) == typeof(float) && vector is Vector<float> floatVec)
        {
            var result = new Vector<float>(floatVec.Length);
            Atan(floatVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }
        else if (typeof(T) == typeof(double) && vector is Vector<double> doubleVec)
        {
            var result = new Vector<double>(doubleVec.Length);
            Atan(doubleVec.AsSpan(), result.AsWritableSpan());
            if (result is Vector<T> typedResult)
            {
                return typedResult;
            }
        }

        // Generic fallback
        var numOps = MathHelper.GetNumericOperations<T>();
        var genericResult = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            genericResult[i] = numOps.FromDouble(Math.Atan(val));
        }
        return genericResult;
    }

    /// <inheritdoc/>
    public void Atan(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Atan(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Sqrt(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Sqrt(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SqrtOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Abs(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Abs(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AbsOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Sinh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Sinh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<SinhOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Cosh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Cosh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CoshOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public virtual void Tanh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Tanh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<TanhOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Asinh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Asinh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AsinhOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Acosh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcoshOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Acosh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AcoshOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Atanh(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanhOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Atanh(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<AtanhOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public virtual void Reciprocal(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ReciprocalOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Reciprocal(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<ReciprocalOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Cbrt(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CbrtOperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Cbrt(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<CbrtOperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log2(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log2OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log2(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log2OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public void Log10(ReadOnlySpan<float> x, Span<float> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log10OperatorFloat>(x, destination);
    }

    /// <inheritdoc/>
    public void Log10(ReadOnlySpan<double> x, Span<double> destination)
    {
        TensorPrimitivesCore.InvokeSpanIntoSpan<Log10OperatorDouble>(x, destination);
    }

    /// <inheritdoc/>
    public Vector<T> Sinh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // TensorPrimitives.Sinh not available - use Math.Sinh
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Sinh(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Cosh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // TensorPrimitives.Cosh not available - use Math.Cosh
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Cosh(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Asinh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // Use asinh(x) = log(x + sqrt(x^2 + 1))
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
#if NET5_0_OR_GREATER
            result[i] = numOps.FromDouble(Math.Asinh(val));
#else
            result[i] = numOps.FromDouble(Math.Log(val + Math.Sqrt(val * val + 1.0)));
#endif
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Acosh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // Use acosh(x) = log(x + sqrt(x^2 - 1))
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
#if NET5_0_OR_GREATER
            result[i] = numOps.FromDouble(Math.Acosh(val));
#else
            result[i] = numOps.FromDouble(Math.Log(val + Math.Sqrt(val * val - 1.0)));
#endif
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Atanh<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        // Use atanh(x) = 0.5 * log((1 + x) / (1 - x))
        for (int i = 0; i < vector.Length; i++)
        {
            double val = numOps.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(MathHelper.Atanh(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Round<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Round(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Floor<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Floor(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Ceiling<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Ceiling(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Truncate<T>(Vector<T> vector)
    {
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);

        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            result[i] = numOps.FromDouble(Math.Truncate(val));
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> Fill<T>(int length, T value)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = value;
        }
        return result;
    }

    /// <inheritdoc/>
    public Vector<T> FillZero<T>(int length)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        return new Vector<T>(length); // Vector constructor already initializes to zero
    }

    /// <inheritdoc/>
    public Vector<T> GenerateDropoutMask<T>(int length, T dropoutRate, T scale, int? seed = null)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var numOps = MathHelper.GetNumericOperations<T>();
        double dropoutRateDouble = Convert.ToDouble(dropoutRate);
        var mask = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            mask[i] = random.NextDouble() > dropoutRateDouble ? scale : numOps.Zero;
        }
        return mask;
    }

    /// <inheritdoc/>
    public void CopyVectorToTensor<T>(Vector<T> source, Tensor<T> destination)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (source.Length != destination.Length)
        {
            throw new ArgumentException(
                $"Vector length ({source.Length}) must equal tensor total elements ({destination.Length}).");
        }
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = source[i];
        }
    }
    /// <inheritdoc/>
    public Vector<T> GenerateGaussianNoise<T>(int length, T mean, T standardDeviation, int? seed = null)
    {
        if (length < 0) throw new ArgumentException("Length must be non-negative.", nameof(length));
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var numOps = MathHelper.GetNumericOperations<T>();
        var noise = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            // Box-Muller transform to generate Gaussian random numbers
            T u1 = numOps.FromDouble(random.NextDouble());
            T u2 = numOps.FromDouble(random.NextDouble());
            T z = numOps.Multiply(
                numOps.Sqrt(numOps.Multiply(numOps.FromDouble(-2.0), numOps.Log(u1))),
                numOps.FromDouble(Math.Cos(2.0 * Math.PI * Convert.ToDouble(u2))));
            noise[i] = numOps.Add(mean, numOps.Multiply(standardDeviation, z));
        }
        return noise;
    }

    #endregion

    #region Matrix Operations (Phase B: Epic 2)

    /// <inheritdoc/>
    public Matrix<T> MatrixMultiply<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First matrix is {a.Rows}x{a.Columns}, second is {b.Rows}x{b.Columns}. " +
                $"First matrix columns ({a.Columns}) must equal second matrix rows ({b.Rows}).");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int m = a.Rows;
        int k = a.Columns;
        int n = b.Columns;
        var result = new Matrix<T>(m, n);

        // Try BLAS/SimdGemm-accelerated path for float/double
        if (MatrixMultiplyHelper.TryGemm(a.AsMemory(), 0, b.AsMemory(), 0, result.AsWritableMemory(), 0, m, k, n))
        {
            return result;
        }

        // Fallback for non-float/double types: blocked multiplication using numOps
        MatrixMultiplyHelper.MultiplyBlocked(numOps, a.AsMemory(), b.AsMemory(), result.AsWritableMemory(), m, k, n, k, n, n);

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> MatrixVectorMultiply<T>(Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (vector == null) throw new ArgumentNullException(nameof(vector));
        if (matrix.Columns != vector.Length)
        {
            throw new ArgumentException(
                $"Matrix-vector dimensions incompatible. " +
                $"Matrix is {matrix.Rows}x{matrix.Columns}, vector has {vector.Length} elements. " +
                $"Matrix columns ({matrix.Columns}) must equal vector length ({vector.Length}).");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(matrix.Rows);

        for (int i = 0; i < matrix.Rows; i++)
        {
            T sum = numOps.Zero;
            for (int j = 0; j < matrix.Columns; j++)
            {
                sum = numOps.Add(sum, numOps.Multiply(matrix[i, j], vector[j]));
            }
            result[i] = sum;
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixTranspose<T>(Matrix<T> matrix)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        var result = new Matrix<T>(matrix.Columns, matrix.Rows);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[j, i] = matrix[i, j];
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixAdd<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Columns != b.Columns)
        {
            throw new ArgumentException(
                $"Matrix dimensions must match for addition. " +
                $"First matrix is {a.Rows}x{a.Columns}, second is {b.Rows}x{b.Columns}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(a.Rows, a.Columns);

        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                result[i, j] = numOps.Add(a[i, j], b[i, j]);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Matrix<T> MatrixMultiplyScalar<T>(Matrix<T> matrix, T scalar)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = numOps.Multiply(matrix[i, j], scalar);
            }
        }

        return result;
    }

    public Matrix<T> MatrixSubtract<T>(Matrix<T> a, Matrix<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            throw new ArgumentException("Matrix dimensions must match for subtraction");

        var result = new Matrix<T>(a.Rows, a.Columns);

        // VECTORIZED: Use existing Vector Subtract operation on each row
        for (int i = 0; i < a.Rows; i++)
        {
            var rowA = a.GetRow(i);
            var rowB = b.GetRow(i);
            var diffRow = Subtract(rowA, rowB); // Reuse vectorized Vector Subtract
            result.SetRow(i, diffRow);
        }

        return result;
    }

    public T MatrixSumOfSquares<T>(Matrix<T> matrix)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;

        // VECTORIZED: Use existing DotProduct operation on each row
        for (int i = 0; i < matrix.Rows; i++)
        {
            var row = matrix.GetRow(i);
            T rowSumSquares = DotProduct(row, row); // row ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· row = sum of squares for row
            sum = numOps.Add(sum, rowSumSquares);
        }

        return sum;
    }

    public void SwapColumns<T>(Matrix<T> matrix, int col1, int col2)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        // Direct element swap - no vectorization benefit for column swaps due to strided access
        for (int i = 0; i < matrix.Rows; i++)
        {
            T temp = matrix[i, col1];
            matrix[i, col1] = matrix[i, col2];
            matrix[i, col2] = temp;
        }
    }

    public void SwapRows<T>(Matrix<T> matrix, int row1, int row2)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));

        // Use vectorized operations for row swapping
        var tempRow1 = GetRow(matrix, row1);
        var tempRow2 = GetRow(matrix, row2);

        SetRow(matrix, row1, tempRow2);
        SetRow(matrix, row2, tempRow1);
    }

    public Matrix<T> OuterProduct<T>(Vector<T> a, Vector<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        var result = new Matrix<T>(a.Length, b.Length);
        var aArray = a.GetDataArray();
        var bArray = b.GetDataArray();

        // Use SIMD-optimized TensorPrimitives for float type
        if (typeof(T) == typeof(float) && bArray.Length >= 16)
        {
            var bFloat = (float[])(object)bArray;
            var aFloat = (float[])(object)aArray;

            for (int i = 0; i < aFloat.Length; i++)
            {
                var rowData = new float[bFloat.Length];
                // SIMD vectorized: multiply vector b by scalar a[i]
                Simd.SimdKernels.MultiplyScalar(bFloat.AsSpan(), aFloat[i], rowData.AsSpan());

                // Copy result to matrix
                for (int j = 0; j < bFloat.Length; j++)
                {
                    result[i, j] = (T)(object)rowData[j];
                }
            }
        }
        else
        {
            // Fallback using NumOps
            var numOps = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < aArray.Length; i++)
            {
                for (int j = 0; j < bArray.Length; j++)
                {
                    result[i, j] = numOps.Multiply(aArray[i], bArray[j]);
                }
            }
        }

        return result;
    }

    public Vector<T> GetColumn<T>(Matrix<T> matrix, int columnIndex)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (columnIndex < 0 || columnIndex >= matrix.Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex),
                $"Column index {columnIndex} is out of range. Valid range is 0 to {matrix.Columns - 1}.");

        // No vectorization benefit - column access is strided
        var result = new T[matrix.Rows];
        for (int i = 0; i < matrix.Rows; i++)
        {
            result[i] = matrix[i, columnIndex];
        }
        return new Vector<T>(result);
    }

    public Vector<T> GetRow<T>(Matrix<T> matrix, int rowIndex)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (rowIndex < 0 || rowIndex >= matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex),
                $"Row index {rowIndex} is out of range. Valid range is 0 to {matrix.Rows - 1}.");

        // Row access is contiguous - can use direct array copy
        var result = new T[matrix.Columns];
        for (int j = 0; j < matrix.Columns; j++)
        {
            result[j] = matrix[rowIndex, j];
        }
        return new Vector<T>(result);
    }

    public void SetColumn<T>(Matrix<T> matrix, int columnIndex, Vector<T> values)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (columnIndex < 0 || columnIndex >= matrix.Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex),
                $"Column index {columnIndex} is out of range. Valid range is 0 to {matrix.Columns - 1}.");
        if (values.Length != matrix.Rows)
            throw new ArgumentException(
                $"Values vector length ({values.Length}) must match matrix rows ({matrix.Rows}).",
                nameof(values));

        // No vectorization benefit - column access is strided
        var valuesArray = values.GetDataArray();
        for (int i = 0; i < matrix.Rows; i++)
        {
            matrix[i, columnIndex] = valuesArray[i];
        }
    }

    public void SetRow<T>(Matrix<T> matrix, int rowIndex, Vector<T> values)
    {
        if (matrix == null) throw new ArgumentNullException(nameof(matrix));
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (rowIndex < 0 || rowIndex >= matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex),
                $"Row index {rowIndex} is out of range. Valid range is 0 to {matrix.Rows - 1}.");
        if (values.Length != matrix.Columns)
            throw new ArgumentException(
                $"Values vector length ({values.Length}) must match matrix columns ({matrix.Columns}).",
                nameof(values));

        // Row access is contiguous - direct assignment
        var valuesArray = values.GetDataArray();
        for (int j = 0; j < matrix.Columns; j++)
        {
            matrix[rowIndex, j] = valuesArray[j];
        }
    }

    #endregion

    #region Tensor Operations (Phase B: Epic 3)

    /// <inheritdoc/>
    public virtual Tensor<T> Reshape<T>(Tensor<T> tensor, int[] newShape)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (newShape == null) throw new ArgumentNullException(nameof(newShape));

        var originalShape = tensor.Shape.ToArray();
        var result = tensor.Reshape(newShape);
        DifferentiableOps.RecordUnary("Reshape", result, tensor, BackwardFunctions<T>.ReshapeBackward, new object[] { originalShape });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> BatchMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        // For ND (N>=3) batch matmul, materialize non-contiguous views.
        // 2D stride-aware GEMM is handled below with transA/transB flags.
        if (a.Rank >= 3 && !a.IsContiguous) a = a.Contiguous();
        if (b.Rank >= 3 && !b.IsContiguous) b = b.Contiguous();

        // Industry-standard BatchMatMul supports any-rank tensors (2D, 3D, 4D, 5D, ...)
        // For ND tensors where N >= 2:
        // - First N-2 dimensions are batch dimensions (must match)
        // - Last 2 dimensions are matrix dimensions: [..., M, K] @ [..., K, N] = [..., M, N]
        if (a.Rank < 2 || b.Rank < 2)
        {
            throw new ArgumentException(
                $"BatchMatMul requires tensors with at least 2 dimensions. Got ranks {a.Rank} and {b.Rank}.");
        }

        if (a.Rank != b.Rank)
        {
            throw new ArgumentException(
                $"BatchMatMul requires tensors with matching ranks. Got ranks {a.Rank} and {b.Rank}.");
        }

        int rank = a.Rank;
        int m = a._shape[rank - 2];  // Second-to-last dimension
        int k = a._shape[rank - 1];  // Last dimension of a
        int k2 = b._shape[rank - 2]; // Second-to-last dimension of b
        int n = b._shape[rank - 1];  // Last dimension of b

        // Verify inner dimensions match for matrix multiplication
        if (k != k2)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible for multiplication. " +
                $"First tensor's last dimension is {k}, " +
                $"second tensor's second-to-last dimension is {k2}. " +
                $"Inner dimensions must match.");
        }

        // Verify batch dimensions match
        for (int i = 0; i < rank - 2; i++)
        {
            if (a._shape[i] != b._shape[i])
            {
                throw new ArgumentException(
                    $"Batch dimensions must match. Got {a._shape[i]} vs {b._shape[i]} at dimension {i}.");
            }
        }

        // Calculate total batch size (product of all batch dimensions)
        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++)
        {
            batchSize *= a._shape[i];
        }

        // Build output shape: [...batch dims..., m, n]
        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
        {
            outputShape[i] = a._shape[i];
        }
        outputShape[rank - 2] = m;
        outputShape[rank - 1] = n;

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(outputShape);

        // Handle 2D case (no batch dimensions)
        if (rank == 2)
        {
            // Stride-aware GEMM for float: detect transposed views and pass flags
            // SIMD GEMM: only for contiguous or simple-transpose operands
            if (typeof(T) == typeof(float) &&
                (a.IsContiguous || a.IsSimpleTranspose) &&
                (b.IsContiguous || b.IsSimpleTranspose))
            {
                bool transA = a.IsSimpleTranspose;
                bool transB = b.IsSimpleTranspose;
                int lda = transA ? a._strides[a.Rank - 1] : k;
                int ldb = transB ? b._strides[b.Rank - 1] : n;

                // Get raw storage arrays (NOT GetDataArray which copies for views)
                var aArr = a._storage.GetDataArray();
                var bArr = b._storage.GetDataArray();
                var rArr = result.GetDataArray();
                var aFloat = new ReadOnlySpan<float>((float[])(object)aArr).Slice(a._storageOffset);
                var bFloat = new ReadOnlySpan<float>((float[])(object)bArr).Slice(b._storageOffset);
                var cFloat = new Span<float>((float[])(object)rArr);

                Simd.SimdGemm.Sgemm(aFloat, lda, transA, bFloat, ldb, transB, cFloat, m, k, n);
                DifferentiableOps.RecordBinary("BatchMatMul", result, a, b, BackwardFunctions<T>.BatchMatMulBackward);
                return result;
            }

            // Try OneDNN/MKL BLAS for contiguous non-transposed
            if (a.IsContiguous && b.IsContiguous)
            {
                if (MatrixMultiplyHelper.TryGemm(a.Data, 0, b.Data, 0, result.Data, 0, m, k, n))
                {
                    DifferentiableOps.RecordBinary("BatchMatMul", result, a, b, BackwardFunctions<T>.BatchMatMulBackward);
                    return result;
                }
            }

            // Generic fallback: stride-aware scalar matmul (raw storage for stride access)
            var aDataArr = a._storage.GetDataArray();
            var bDataArr = b._storage.GetDataArray();
            var rDataArr = result.GetDataArray();
            int aOff = a._storageOffset, bOff = b._storageOffset;
            int aStride0 = a._strides[0], aStride1 = a._strides[1];
            int bStride0 = b._strides[0], bStride1 = b._strides[1];

            Parallel.For(0, m, i =>
            {
                for (int j = 0; j < n; j++)
                {
                    T sum = numOps.Zero;
                    for (int p = 0; p < k; p++)
                    {
                        T aVal = aDataArr[aOff + i * aStride0 + p * aStride1];
                        T bVal = bDataArr[bOff + p * bStride0 + j * bStride1];
                        sum = numOps.Add(sum, numOps.Multiply(aVal, bVal));
                    }
                    rDataArr[i * n + j] = sum;
                }
            });
            DifferentiableOps.RecordBinary("BatchMatMul", result, a, b,
                BackwardFunctions<T>.BatchMatMulBackward);
            return result;
        }

        var aData = a.GetDataArray();
        var bData = b.GetDataArray();
        var rData = result.GetDataArray();

        // Handle ND case with batch dimensions (N >= 3)
        int matrixSizeA = m * k;
        int matrixSizeB = k * n;
        int matrixSizeResult = m * n;

        // Fast path: for float, do sequential BLAS calls (avoids Parallel.For overhead for small batches)
        // For B <= 16, the Parallel.For thread pool overhead exceeds the parallelism benefit.
        if (typeof(T) == typeof(float) && batchSize <= 16)
        {
            for (int batch = 0; batch < batchSize; batch++)
            {
                int aOffset = batch * matrixSizeA;
                int bOffset = batch * matrixSizeB;
                int resultOffset = batch * matrixSizeResult;
                if (!MatrixMultiplyHelper.TryGemm(a.Data, aOffset, b.Data, bOffset, result.Data, resultOffset, m, k, n))
                {
                    // BLAS unavailable — fall through to parallel path
                    goto parallelFallback;
                }
            }
            goto batchDone;
        }

        parallelFallback:
        Parallel.For(0, batchSize, batch =>
        {
            int aOffset = batch * matrixSizeA;
            int bOffset = batch * matrixSizeB;
            int resultOffset = batch * matrixSizeResult;

            if (MatrixMultiplyHelper.TryGemm(a.Data, aOffset, b.Data, bOffset, result.Data, resultOffset, m, k, n))
            {
                return;
            }

            // Scalar fallback with direct array access
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    T sum = numOps.Zero;
                    for (int p = 0; p < k; p++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(aData[aOffset + i * k + p], bData[bOffset + p * n + j]));
                    }
                    rData[resultOffset + i * n + j] = sum;
                }
            }
        });

        batchDone:

        DifferentiableOps.RecordBinary("BatchMatMul", result, a, b,
            BackwardFunctions<T>.BatchMatMulBackward);
        return result;
    }

    /// <inheritdoc/>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public virtual unsafe Tensor<T> TensorAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        // RentUninitialized: kernel writes every element, no need to zero
        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        int length = a.Length;

        // Stride-aware: if either operand is non-contiguous, use strided iteration (zero-copy)
        // IMPORTANT: use _storage.GetDataArray() (raw backing array) not tensor.GetDataArray() (which copies for views)
        if (!a.IsContiguous || !b.IsContiguous)
        {
            var aRaw = a._storage.GetDataArray(); var bRaw = b._storage.GetDataArray(); var rArr = result.GetDataArray();
            var ops = MathHelper.GetNumericOperations<T>();
            if (a.IsContiguous) { int aOff = a._storageOffset; for (int i = 0; i < length; i++) rArr[i] = ops.Add(aRaw[aOff + i], bRaw[b.LogicalToStorageIndex(i)]); }
            else if (b.IsContiguous) { int bOff = b._storageOffset; for (int i = 0; i < length; i++) rArr[i] = ops.Add(aRaw[a.LogicalToStorageIndex(i)], bRaw[bOff + i]); }
            else { for (int i = 0; i < length; i++) rArr[i] = ops.Add(aRaw[a.LogicalToStorageIndex(i)], bRaw[b.LogicalToStorageIndex(i)]); }
        }
        else if (typeof(T) == typeof(float))
        {
            // Fast path for float tensors: bypass generic dispatch + Span bounds-checking
            // Use Memory<T>.Pin() directly — avoids GetDataArray() which can copy when segment != full array
            var aMem = AsFloatMemory(a.Data);
            var bMem = AsFloatMemory(b.Data);
            var rMem = AsFloatMemory(result.Data);
            using var pinA = aMem.Pin();
            using var pinB = bMem.Pin();
            using var pinR = rMem.Pin();
            float* pA = (float*)pinA.Pointer;
            float* pB = (float*)pinB.Pointer;
            float* pR = (float*)pinR.Pointer;

            // JIT-compiled kernels: size-specialized, 4x unrolled, NT stores
            if (CpuJitSelfTest.IsVerified && length >= 64)
            {
                JitBinaryDispatch(pA, pB, pR, length, JitBinaryOp.Add);
            }
            else
            {
                // Fallback: SimdKernels with parallel chunking for large arrays
                // Use PersistentParallelExecutor for near-zero dispatch overhead
                // (pre-spawned threads, no ThreadPool queuing, no closure allocation)
                int addChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 500_000));
                if (addChunks >= 2)
                {
                    int chunkSize = (length + addChunks - 1) / addChunks;
                    chunkSize = (chunkSize + 31) & ~31;
                    var pACap = pA; var pBCap = pB; var pRCap = pR;
                    int lenCap = length;

                    PersistentParallelExecutor.Instance.Execute(addChunks, chunk =>
                    {
                        int start = chunk * chunkSize;
                        int count = Math.Min(chunkSize, lenCap - start);
                        if (count > 0)
                        {
                            SimdKernels.VectorAddUnsafe(pACap + start, pBCap + start, pRCap + start, count);
                        }
                    });
                }
                else
                {
                    SimdKernels.VectorAddUnsafe(pA, pB, pR, length);
                }
            }
        }
        else if (typeof(T) == typeof(double))
        {
            var aMem = AsDoubleMemory(a.Data);
            var bMem = AsDoubleMemory(b.Data);
            var rMem = AsDoubleMemory(result.Data);
            using var pinA = aMem.Pin();
            using var pinB = bMem.Pin();
            using var pinR = rMem.Pin();
            double* pA = (double*)pinA.Pointer;
            double* pB = (double*)pinB.Pointer;
            double* pR = (double*)pinR.Pointer;
            // Parallel chunking for large double arrays (8 bytes/element = 2x bandwidth)
            int subChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 250_000));
            if (subChunks >= 2)
            {
                int chunkSize = (length + subChunks - 1) / subChunks;
                chunkSize = (chunkSize + 15) & ~15; // Align to AVX double boundary (4 doubles)
                IntPtr ipA = (IntPtr)pA, ipB = (IntPtr)pB, ipR = (IntPtr)pR;
                int totalLength = length;
                Parallel.For(0, subChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, totalLength - start);
                    if (count > 0)
                        SimdKernels.VectorAddUnsafe((double*)ipA + start, (double*)ipB + start, (double*)ipR + start, count);
                });
            }
            else
            {
                SimdKernels.VectorAddUnsafe(pA, pB, pR, length);
            }
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Add(a.AsSpan(), b.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordBinary("TensorAdd", result, a, b, BackwardFunctions<T>.AddBackward);
        return result;
    }

    /// <summary>
    /// Adds tensor b to tensor a in-place (a += b). Zero allocation.
    /// Uses parallel SIMD for large float tensors.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public virtual unsafe void TensorAddInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        // Save input before mutation when tape is active (for backward pass)
        Tensor<T>? savedA = null;
        var tape = GradientTape<T>.Current;
        if (tape is not null && tape.Options.RecordInPlace)
            savedA = a.Clone();

        // Increment version BEFORE mutation so prior tape entries detect the change
        a.IncrementVersion();

        // Stride-aware: in-place requires contiguous target; materialize source if needed
        if (!a.IsContiguous) throw new InvalidOperationException("In-place add requires contiguous target tensor.");
        if (!b.IsContiguous) b = b.Contiguous();

        int length = a.Length;

        // Fast path for float tensors: bypass generic dispatch + Span bounds-checking
        // Use Memory<T>.Pin() directly — avoids GetDataArray() which can copy (breaking in-place writes)
        if (typeof(T) == typeof(float))
        {
            var aMem = AsFloatMemory(a.Data);
            var bMem = AsFloatMemory(b.Data);
            using var pinA = aMem.Pin();
            using var pinB = bMem.Pin();
            float* pA = (float*)pinA.Pointer;
            float* pB = (float*)pinB.Pointer;

#if !NET471
            // Strategy 1: Try oneDNN for best performance (uses JIT-compiled native kernels)
            if (OneDnnProvider.IsAvailable)
            {
                if (OneDnnProvider.TryAdd(pB, pA, pA, length))
                {
                    if (savedA is not null) DifferentiableOps.RecordBinary("TensorAddInPlace", a, savedA, b, BackwardFunctions<T>.AddBackward);
                    return;
                }
            }
#endif

            // Strategy 2: JIT-compiled kernel (size-specialized, 4x unrolled, parallel for large)
            if (CpuJitSelfTest.IsVerified && length >= 64)
            {
                JitBinaryDispatch(pA, pB, pA, length, JitBinaryOp.Add);
                if (savedA is not null) DifferentiableOps.RecordBinary("TensorAddInPlace", a, savedA, b, BackwardFunctions<T>.AddBackward);
                return;
            }

            // Fallback: SimdKernels with parallel chunking for large arrays
            int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 2_000_000));
            if (numChunks >= 2)
            {
                int chunkSize = (length + numChunks - 1) / numChunks;
                chunkSize = (chunkSize + 31) & ~31;

                Parallel.For(0, numChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                    {
                        SimdKernels.VectorAddUnsafe(pB + start, pA + start, pA + start, count);
                    }
                });
            }
            else
            {
                SimdKernels.VectorAddUnsafe(pB, pA, pA, length);
            }
            if (savedA is not null) DifferentiableOps.RecordBinary("TensorAddInPlace", a, savedA, b, BackwardFunctions<T>.AddBackward);
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Add(a.AsSpan(), b.AsSpan(), a.AsWritableSpan());
        if (savedA is not null)
            DifferentiableOps.RecordBinary("TensorAddInPlace", a, savedA, b, BackwardFunctions<T>.AddBackward);
    }

    /// <summary>
    /// Adds tensors a and b, storing result in destination. Zero allocation.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public void TensorAddInto<T>(Tensor<T> destination, Tensor<T> a, Tensor<T> b)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!b.IsContiguous) b = b.Contiguous();
        if (!a.IsContiguous) a = a.Contiguous();
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!ShapesMatch(a._shape, b._shape) || !ShapesMatch(a._shape, destination._shape))
        {
            throw new ArgumentException("All tensor shapes must match.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Add(a.AsSpan(), b.AsSpan(), destination.AsWritableSpan());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBroadcastAdd<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        // Use optimized Tensor.BroadcastAdd which handles broadcasting logic
        var result = a.BroadcastAdd(b);
        DifferentiableOps.RecordBinary("TensorBroadcastAdd", result, a, b, BackwardFunctions<T>.BroadcastAddBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBroadcastSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();

        var result = a.BroadcastSubtract(b);
        DifferentiableOps.RecordBinary("TensorBroadcastSubtract", result, a, b, BackwardFunctions<T>.BroadcastSubtractBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBroadcastDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();

        var result = a.BroadcastDivide(b);
        DifferentiableOps.RecordBinary("TensorBroadcastDivide", result, a, b, BackwardFunctions<T>.BroadcastDivideBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBroadcastMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();

        var result = a.BroadcastMultiply(b);
        DifferentiableOps.RecordBinary("TensorBroadcastMultiply", result, a, b, BackwardFunctions<T>.BroadcastMultiplyBackward);
        return result;
    }

    /// <inheritdoc/>
    public void TensorBroadcastAddInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var aSpan = a.Data.Span;
        var bSpan = b.Data.Span;

        // Fast path: same shape — no broadcasting needed
        if (ShapesMatch(a._shape, b._shape))
        {
            for (int i = 0; i < aSpan.Length; i++)
                aSpan[i] = numOps.Add(aSpan[i], bSpan[i]);
            return;
        }

        // Conv bias pattern: a=[B,C,H,W], b=[1,C,1,1] — most common case
        if (a.Rank == 4 && b.Rank == 4 &&
            b._shape[0] == 1 && b._shape[2] == 1 && b._shape[3] == 1 &&
            a._shape[1] == b._shape[1])
        {
            int batch = a._shape[0];
            int channels = a._shape[1];
            int spatial = a._shape[2] * a._shape[3];

            for (int n = 0; n < batch; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    T biasVal = bSpan[c];
                    int offset = (n * channels + c) * spatial;
                    for (int s = 0; s < spatial; s++)
                    {
                        aSpan[offset + s] = numOps.Add(aSpan[offset + s], biasVal);
                    }
                }
            }
            return;
        }

        // Scalar broadcast: b has 1 element
        if (b.Length == 1)
        {
            T scalar = bSpan[0];
            for (int i = 0; i < aSpan.Length; i++)
                aSpan[i] = numOps.Add(aSpan[i], scalar);
            return;
        }

        // 1D bias along last dimension: a=[..., N], b=[N]
        if (b.Rank == 1 && a._shape[^1] == b._shape[0])
        {
            int lastDim = b._shape[0];
            int outerSize = a.Length / lastDim;
            for (int outer = 0; outer < outerSize; outer++)
            {
                int offset = outer * lastDim;
                for (int i = 0; i < lastDim; i++)
                    aSpan[offset + i] = numOps.Add(aSpan[offset + i], bSpan[i]);
            }
            return;
        }

        // General fallback: compute broadcast result and copy back.
        // This allocates a temporary — acceptable for rare arbitrary broadcast shapes.
        var result = TensorBroadcastAdd(a, b);
        result.Data.Span.CopyTo(aSpan);
    }

    /// <inheritdoc/>
    public void GroupNormInto<T>(Tensor<T> output, Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (!output.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        if (!gamma.IsContiguous) gamma = gamma.Contiguous();
        if (!beta.IsContiguous) beta = beta.Contiguous();
        // GroupNorm writes normalized values into pre-allocated output.
        // The mean/variance stats are small tensors [batch, numGroups] that the callee allocates.
        // The main output tensor avoids allocation since it's pre-allocated by the caller.
        var result = GroupNorm(input, numGroups, gamma, beta, epsilon, out mean, out variance);
        result.Data.Span.CopyTo(output.Data.Span);
    }

    /// <inheritdoc/>
    public void GroupNormSwishInto<T>(Tensor<T> output, Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon)
    {
        if (!output.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        if (!gamma.IsContiguous) gamma = gamma.Contiguous();
        if (!beta.IsContiguous) beta = beta.Contiguous();
        // Step 1: GroupNorm into output (single computation)
        GroupNormInto(output, input, numGroups, gamma, beta, epsilon, out _, out _);

        // Step 2: Swish in-place on the normalized output
        // swish(x) = x * sigmoid(x) — needs a temp buffer for sigmoid values
        SwishInPlace(output);
    }

    /// <inheritdoc/>
    public void AddGroupNormInto<T>(Tensor<T> output, Tensor<T> a, Tensor<T> b, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon)
    {
        if (!output.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        if (!gamma.IsContiguous) gamma = gamma.Contiguous();
        if (!beta.IsContiguous) beta = beta.Contiguous();
        // output = GroupNorm(a + b)
        // Step 1: Add a + b directly into output (zero alloc)
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Add(a.AsSpan(), b.AsSpan(), output.AsWritableSpan());
        // Step 2: GroupNorm in-place on output via GroupNormInto
        GroupNormInto(output, output, numGroups, gamma, beta, epsilon, out _, out _);
    }

    /// <inheritdoc/>
    public void SwishInPlace<T>(Tensor<T> tensor)
    {
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        // Swish uses ArrayPool internally for the sigmoid buffer (no GC allocation)
        numOps.Swish(tensor.AsSpan(), tensor.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void SwishInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Sigmoid(input.AsSpan(), destination.AsWritableSpan());
        numOps.Multiply(input.AsSpan(), destination.AsSpan(), destination.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void GELUInPlace<T>(Tensor<T> tensor)
    {
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.GELU(tensor.AsSpan(), tensor.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void GELUInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.GELU(input.AsSpan(), destination.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void TanhInPlace<T>(Tensor<T> tensor)
    {
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Tanh(tensor.AsSpan(), tensor.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void TanhInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Tanh(input.AsSpan(), destination.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void MishInPlace<T>(Tensor<T> tensor)
    {
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Mish(tensor.AsSpan(), tensor.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void MishInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Mish(input.AsSpan(), destination.AsWritableSpan());
    }

    /// <inheritdoc/>
    public virtual void LeakyReLUInPlace<T>(Tensor<T> tensor, T alpha)
    {
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.LeakyReLU(tensor.AsSpan(), alpha, tensor.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void LeakyReLUInto<T>(Tensor<T> destination, Tensor<T> input, T alpha)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.LeakyReLU(input.AsSpan(), alpha, destination.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void MatMulInto<T>(Tensor<T> destination, Tensor<T> a, Tensor<T> b)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!b.IsContiguous) b = b.Contiguous();
        if (!a.IsContiguous) a = a.Contiguous();
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");

        if (a.Rank == 2 && b.Rank == 2)
        {
            int m = a._shape[0];
            int n = a._shape[1];
            int p = b._shape[1];

            if (n != b._shape[0])
                throw new ArgumentException($"Matrix dimensions incompatible: [{m},{n}] x [{b._shape[0]},{p}]");

            // Try BLAS directly into destination memory (true zero-alloc)
            if (MatrixMultiplyHelper.TryGemm(a.Data, 0, b.Data, 0, destination.Data, 0, m, n, p))
                return;

            // Fallback: compute via numOps into destination spans
            var numOps = MathHelper.GetNumericOperations<T>();
            var aSpan = a.AsSpan();
            var bSpan = b.AsSpan();
            var dstSpan = destination.AsWritableSpan();
            dstSpan.Clear();

            for (int i = 0; i < m; i++)
            {
                for (int k = 0; k < n; k++)
                {
                    T aVal = aSpan[i * n + k];
                    for (int j = 0; j < p; j++)
                    {
                        dstSpan[i * p + j] = numOps.Add(dstSpan[i * p + j],
                            numOps.Multiply(aVal, bSpan[k * p + j]));
                    }
                }
            }
            return;
        }

        // For batched/ND cases, fall back to allocate-copy
        var result = TensorMatMul(a, b);
        result.Data.Span.CopyTo(destination.Data.Span);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Current implementation computes into a temporary tensor then copies to destination.
    /// </remarks>
    public void ConcatInto<T>(Tensor<T> destination, Tensor<T>[] tensors, int axis)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        var result = Concat(tensors, axis);
        result.Data.Span.CopyTo(destination.Data.Span);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Current implementation computes into a temporary tensor then copies to destination.
    /// </remarks>
    public void TransposeInto<T>(Tensor<T> destination, Tensor<T> input, int[] axes)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        if (axes == null || axes.Length == 0 ||
            (input.Rank == 2 && axes.Length == 2 && axes[0] == 1 && axes[1] == 0))
        {
            // Standard 2D transpose
            var result = TensorTranspose(input);
            result.Data.Span.CopyTo(destination.Data.Span);
            return;
        }

        // General ND permutation: map each destination coord back to source
        var srcShape = input._shape;
        var srcSpan = input.AsSpan();
        var dstSpan = destination.AsWritableSpan();
        int rank = srcShape.Length;

        var srcStrides = new int[rank];
        srcStrides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; d--)
            srcStrides[d] = srcStrides[d + 1] * srcShape[d + 1];

        var dstStrides = new int[rank];
        var dstShape = destination._shape;
        dstStrides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; d--)
            dstStrides[d] = dstStrides[d + 1] * dstShape[d + 1];

        int totalElements = destination.Length;
        var coords = new int[rank];

        for (int dstIdx = 0; dstIdx < totalElements; dstIdx++)
        {
            int remaining = dstIdx;
            for (int d = 0; d < rank; d++)
            {
                coords[d] = remaining / dstStrides[d];
                remaining %= dstStrides[d];
            }

            int srcIdx = 0;
            for (int d = 0; d < rank; d++)
                srcIdx += coords[d] * srcStrides[axes[d]];

            dstSpan[dstIdx] = srcSpan[srcIdx];
        }
    }

    /// <inheritdoc/>
    public unsafe void SoftmaxInto<T>(Tensor<T> destination, Tensor<T> input, int axis)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();

        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        int outerSize = 1, axisSize = input._shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input._shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= input._shape[i];

        if (typeof(T) == typeof(float) && innerSize == 1)
        {
            // Use Pin() for NativeMemory compatibility (no GetDataArray copy)
            using var pinIn = input.Data.Pin();
            using var pinOut = destination.Data.Pin();
            unsafe
            {
                SoftmaxFloatFastPtr((float*)pinIn.Pointer, (float*)pinOut.Pointer, outerSize, axisSize);
            }
            return;
        }

        // Fallback: compute and copy, return pooled tensor
        var result = Softmax(input, axis);
        try
        {
            result.Data.Span.CopyTo(destination.Data.Span);
        }
        finally
        {
            TensorAllocator.Return(result);
        }
    }

    /// <inheritdoc/>
    public void LogSoftmaxInto<T>(Tensor<T> destination, Tensor<T> input, int axis)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var result = TensorLogSoftmax(input, axis);
        try
        {
            result.Data.Span.CopyTo(destination.Data.Span);
        }
        finally
        {
            TensorAllocator.Return(result);
        }
    }

    /// <summary>Subtract into pre-allocated destination. Zero allocation.</summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public void TensorSubtractInto<T>(Tensor<T> destination, Tensor<T> a, Tensor<T> b)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Subtract(a.AsSpan(), b.AsSpan(), destination.AsWritableSpan());
    }

    /// <summary>Divide into pre-allocated destination. Zero allocation.</summary>
    public void TensorDivideInto<T>(Tensor<T> destination, Tensor<T> a, Tensor<T> b)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Divide(a.AsSpan(), b.AsSpan(), destination.AsWritableSpan());
    }

    /// <summary>Exp into pre-allocated destination. Zero allocation.</summary>
    public void TensorExpInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Exp(input.AsSpan(), destination.AsWritableSpan());
    }

    /// <summary>Log into pre-allocated destination. Zero allocation.</summary>
    public void TensorLogInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Log(input.AsSpan(), destination.AsWritableSpan());
    }

    /// <summary>Sqrt into pre-allocated destination. Zero allocation.</summary>
    public void TensorSqrtInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Sqrt(input.AsSpan(), destination.AsWritableSpan());
    }

    /// <summary>Abs into pre-allocated destination. Zero allocation.</summary>
    public void TensorAbsInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Abs(input.AsSpan(), destination.AsWritableSpan());
    }

    /// <inheritdoc/>
    public Tensor<T> TensorAddMany<T>(params Tensor<T>[] tensors)
    {
        if (tensors == null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length < 2)
            throw new ArgumentException("TensorAddMany requires at least 2 tensors.", nameof(tensors));

        // Validate all shapes match the first tensor
        var referenceShape = tensors[0]._shape;
        for (int t = 1; t < tensors.Length; t++)
        {
            if (!ShapesMatch(referenceShape, tensors[t]._shape))
            {
                throw new ArgumentException(
                    $"All tensor shapes must match. Tensor 0 has shape {FormatShape(referenceShape)}, " +
                    $"but tensor {t} has shape {FormatShape(tensors[t]._shape)}.");
            }
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int length = tensors[0].Length;

        // Chain pairwise additions using span-based numOps: result = t0 + t1 + t2 + ...
        // First: result = tensors[0] + tensors[1]
        var result = TensorAllocator.RentUninitialized<T>(referenceShape);
        numOps.Add(tensors[0].AsSpan(), tensors[1].AsSpan(), result.AsWritableSpan());

        // Accumulate remaining tensors into result
        for (int t = 2; t < tensors.Length; t++)
        {
            numOps.Add(result.AsSpan(), tensors[t].AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordIfActive("TensorAddMany", result, tensors, BackwardFunctions<T>.AddManyBackward);
        return result;
    }

    /// <inheritdoc/>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public virtual unsafe Tensor<T> TensorSubtract<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        int length = a.Length;

        // Stride-aware: strided iteration for non-contiguous views (zero-copy)
        if (!a.IsContiguous || !b.IsContiguous)
        {
            var aArr = a._storage.GetDataArray(); var bArr = b._storage.GetDataArray(); var rArr = result.GetDataArray();
            var ops = MathHelper.GetNumericOperations<T>();
            if (a.IsContiguous) { int aOff = a._storageOffset; for (int i = 0; i < length; i++) rArr[i] = ops.Subtract(aArr[aOff + i], bArr[b.LogicalToStorageIndex(i)]); }
            else if (b.IsContiguous) { int bOff = b._storageOffset; for (int i = 0; i < length; i++) rArr[i] = ops.Subtract(aArr[a.LogicalToStorageIndex(i)], bArr[bOff + i]); }
            else { for (int i = 0; i < length; i++) rArr[i] = ops.Subtract(aArr[a.LogicalToStorageIndex(i)], bArr[b.LogicalToStorageIndex(i)]); }
        }
        else if (typeof(T) == typeof(float))
        {
            // Fast path for float tensors
            var aMem = AsFloatMemory(a.Data);
            var bMem = AsFloatMemory(b.Data);
            var rMem = AsFloatMemory(result.Data);
            using var pinA = aMem.Pin();
            using var pinB = bMem.Pin();
            using var pinR = rMem.Pin();
            float* pA = (float*)pinA.Pointer;
            float* pB = (float*)pinB.Pointer;
            float* pR = (float*)pinR.Pointer;

            if (CpuJitSelfTest.IsVerified && length >= 64)
            {
                JitBinaryDispatch(pA, pB, pR, length, JitBinaryOp.Subtract);
            }
            else
            {
                int subChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 500_000));
                if (subChunks >= 2)
                {
                    int chunkSize = (length + subChunks - 1) / subChunks;
                    chunkSize = (chunkSize + 31) & ~31;

                    Parallel.For(0, subChunks, chunk =>
                    {
                        int start = chunk * chunkSize;
                        int count = Math.Min(chunkSize, length - start);
                        if (count > 0)
                        {
                            SimdKernels.VectorSubtractUnsafe(pA + start, pB + start, pR + start, count);
                        }
                    });
                }
                else
                {
                    SimdKernels.VectorSubtractUnsafe(pA, pB, pR, length);
                }
            }
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Subtract(a.AsSpan(), b.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordBinary("TensorSubtract", result, a, b, BackwardFunctions<T>.SubtractBackward);
        return result;
    }

    /// <inheritdoc/>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public virtual unsafe Tensor<T> TensorMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a._shape, b._shape))
        {
            // Shapes don't match — fall through to broadcasting (NumPy/PyTorch behavior)
            return TensorBroadcastMultiply(a, b);
        }

        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        int length = a.Length;

        // Stride-aware: strided iteration for non-contiguous views (zero-copy)
        if (!a.IsContiguous || !b.IsContiguous)
        {
            var aArr = a._storage.GetDataArray(); var bArr = b._storage.GetDataArray(); var rArr = result.GetDataArray();
            var ops = MathHelper.GetNumericOperations<T>();
            if (a.IsContiguous) { int aOff = a._storageOffset; for (int i = 0; i < length; i++) rArr[i] = ops.Multiply(aArr[aOff + i], bArr[b.LogicalToStorageIndex(i)]); }
            else if (b.IsContiguous) { int bOff = b._storageOffset; for (int i = 0; i < length; i++) rArr[i] = ops.Multiply(aArr[a.LogicalToStorageIndex(i)], bArr[bOff + i]); }
            else { for (int i = 0; i < length; i++) rArr[i] = ops.Multiply(aArr[a.LogicalToStorageIndex(i)], bArr[b.LogicalToStorageIndex(i)]); }
        }
        else if (typeof(T) == typeof(float))
        {
            var aMem = AsFloatMemory(a.Data);
            var bMem = AsFloatMemory(b.Data);
            var rMem = AsFloatMemory(result.Data);
            using var pinA = aMem.Pin();
            using var pinB = bMem.Pin();
            using var pinR = rMem.Pin();
            float* pA = (float*)pinA.Pointer;
            float* pB = (float*)pinB.Pointer;
            float* pR = (float*)pinR.Pointer;

            if (CpuJitSelfTest.IsVerified && length >= 64)
            {
                JitBinaryDispatch(pA, pB, pR, length, JitBinaryOp.Multiply);
            }
            else
            {
                int mulChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 500_000));
                if (mulChunks >= 2)
                {
                    int chunkSize = (length + mulChunks - 1) / mulChunks;
                    chunkSize = (chunkSize + 31) & ~31;

                    Parallel.For(0, mulChunks, chunk =>
                    {
                        int start = chunk * chunkSize;
                        int count = Math.Min(chunkSize, length - start);
                        if (count > 0)
                        {
                            SimdKernels.VectorMultiplyUnsafe(pA + start, pB + start, pR + start, count);
                        }
                    });
                }
                else
                {
                    SimdKernels.VectorMultiplyUnsafe(pA, pB, pR, length);
                }
            }
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Multiply(a.AsSpan(), b.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordBinary("TensorMultiply", result, a, b, BackwardFunctions<T>.MultiplyBackward);
        return result;
    }

    /// <summary>
    /// Multiplies tensor a by tensor b in-place (a *= b). Zero allocation.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public virtual unsafe void TensorMultiplyInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        Tensor<T>? savedA = null;
        var mulTape = GradientTape<T>.Current;
        if (mulTape is not null && mulTape.Options.RecordInPlace)
            savedA = a.Clone();

        a.IncrementVersion();

        if (!a.IsContiguous) throw new InvalidOperationException("In-place multiply requires contiguous target tensor.");
        if (!b.IsContiguous) b = b.Contiguous();

        int length = a.Length;

        // Fast path for float tensors: bypass generic dispatch + Span bounds-checking
        // Fast path for float tensors: bypass generic dispatch + Span bounds-checking
        // Use Memory<T>.Pin() directly — avoids GetDataArray() which can copy (breaking in-place writes)
        if (typeof(T) == typeof(float))
        {
            var aMem = AsFloatMemory(a.Data);
            var bMem = AsFloatMemory(b.Data);
            using var pinA = aMem.Pin();
            using var pinB = bMem.Pin();
            float* pA = (float*)pinA.Pointer;
            float* pB = (float*)pinB.Pointer;

#if !NET471
            // Strategy 1: Try oneDNN for best performance (uses JIT-compiled native kernels)
            if (OneDnnProvider.IsAvailable)
            {
                if (OneDnnProvider.TryMultiply(pB, pA, pA, length))
                {
                    if (savedA is not null) DifferentiableOps.RecordBinary("TensorMultiplyInPlace", a, savedA, b, BackwardFunctions<T>.MultiplyBackward);
                    return;
                }
            }
#endif

            if (CpuJitSelfTest.IsVerified && length >= 64)
            {
                JitBinaryDispatch(pA, pB, pA, length, JitBinaryOp.Multiply);
                if (savedA is not null) DifferentiableOps.RecordBinary("TensorMultiplyInPlace", a, savedA, b, BackwardFunctions<T>.MultiplyBackward);
                return;
            }

            int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 2_000_000));
            if (numChunks >= 2)
            {
                int chunkSize = (length + numChunks - 1) / numChunks;
                chunkSize = (chunkSize + 31) & ~31;

                Parallel.For(0, numChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                    {
                        SimdKernels.VectorMultiplyUnsafe(pB + start, pA + start, pA + start, count);
                    }
                });
            }
            else
            {
                SimdKernels.VectorMultiplyUnsafe(pB, pA, pA, length);
            }
            if (savedA is not null) DifferentiableOps.RecordBinary("TensorMultiplyInPlace", a, savedA, b, BackwardFunctions<T>.MultiplyBackward);
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Multiply(a.AsSpan(), b.AsSpan(), a.AsWritableSpan());
        if (savedA is not null)
            DifferentiableOps.RecordBinary("TensorMultiplyInPlace", a, savedA, b, BackwardFunctions<T>.MultiplyBackward);
    }

    /// <summary>
    /// Multiplies tensors a and b, storing result in destination. Zero allocation.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public void TensorMultiplyInto<T>(Tensor<T> destination, Tensor<T> a, Tensor<T> b)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!b.IsContiguous) b = b.Contiguous();
        if (!a.IsContiguous) a = a.Contiguous();
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!ShapesMatch(a._shape, b._shape) || !ShapesMatch(a._shape, destination._shape))
        {
            throw new ArgumentException("All tensor shapes must match.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Multiply(a.AsSpan(), b.AsSpan(), destination.AsWritableSpan());
    }

    /// <summary>
    /// Subtracts tensor b from tensor a in-place: a[i] -= b[i]. Zero allocation.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public virtual unsafe void TensorSubtractInPlace<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        Tensor<T>? savedASub = null;
        var subTape = GradientTape<T>.Current;
        if (subTape is not null && subTape.Options.RecordInPlace)
            savedASub = a.Clone();

        a.IncrementVersion();

        if (!a.IsContiguous) throw new InvalidOperationException("In-place subtract requires contiguous target tensor.");
        if (!b.IsContiguous) b = b.Contiguous();

        int length = a.Length;

        if (typeof(T) == typeof(float))
        {
            var aMem = AsFloatMemory(a.Data);
            var bMem = AsFloatMemory(b.Data);
            using var pinA = aMem.Pin();
            using var pinB = bMem.Pin();
            float* pA = (float*)pinA.Pointer;
            float* pB = (float*)pinB.Pointer;

            // No oneDNN subtract primitive — go straight to JIT/SIMD
            if (CpuJitSelfTest.IsVerified && length >= 64)
            {
                JitBinaryDispatch(pA, pB, pA, length, JitBinaryOp.Subtract);
                if (savedASub is not null) DifferentiableOps.RecordBinary("TensorSubtractInPlace", a, savedASub, b, BackwardFunctions<T>.SubtractBackward);
                return;
            }

            int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 2_000_000));
            if (numChunks >= 2)
            {
                int chunkSize = (length + numChunks - 1) / numChunks;
                chunkSize = (chunkSize + 31) & ~31;

                Parallel.For(0, numChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                    {
                        SimdKernels.VectorSubtractUnsafe(pA + start, pB + start, pA + start, count);
                    }
                });
            }
            else
            {
                SimdKernels.VectorSubtractUnsafe(pA, pB, pA, length);
            }
            if (savedASub is not null) DifferentiableOps.RecordBinary("TensorSubtractInPlace", a, savedASub, b, BackwardFunctions<T>.SubtractBackward);
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Subtract(a.AsSpan(), b.AsSpan(), a.AsWritableSpan());
        if (savedASub is not null)
            DifferentiableOps.RecordBinary("TensorSubtractInPlace", a, savedASub, b, BackwardFunctions<T>.SubtractBackward);
    }

    /// <summary>
    /// Multiplies all elements of tensor a by a scalar in-place: a[i] *= scalar. Zero allocation.
    /// Uses SIMD with parallel chunking for float, vectorized numOps for all types.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public unsafe void TensorMultiplyScalarInPlace<T>(Tensor<T> a, T scalar)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (!a.IsContiguous) throw new InvalidOperationException("In-place scalar multiply requires contiguous tensor.");

        int length = a.Length;

        // Float fast path: SIMD MultiplyScalar with parallel chunking
        if (typeof(T) == typeof(float))
        {
            float scalarF = (float)(object)scalar!;
            var aMem = AsFloatMemory(a.Data);
            using var pinA = aMem.Pin();
            float* pA = (float*)pinA.Pointer;

            int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 2_000_000));
            if (numChunks >= 2)
            {
                int chunkSize = (length + numChunks - 1) / numChunks;
                chunkSize = (chunkSize + 31) & ~31;
                Parallel.For(0, numChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                        SimdKernels.MultiplyScalarUnsafe(pA + start, scalarF, pA + start, count);
                });
            }
            else
            {
                SimdKernels.MultiplyScalarUnsafe(pA, scalarF, pA, length);
            }
            return;
        }

        // Double fast path: SIMD MultiplyScalar with parallel chunking
        if (typeof(T) == typeof(double))
        {
            double scalarD = (double)(object)scalar!;
            var aMem = AsDoubleMemory(a.Data);
            using var pinA = aMem.Pin();
            double* pA = (double*)pinA.Pointer;

            int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 2_000_000));
            if (numChunks >= 2)
            {
                int chunkSize = (length + numChunks - 1) / numChunks;
                chunkSize = (chunkSize + 31) & ~31;
                Parallel.For(0, numChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                        SimdKernels.MultiplyScalarUnsafe(pA + start, scalarD, pA + start, count);
                });
            }
            else
            {
                SimdKernels.MultiplyScalarUnsafe(pA, scalarD, pA, length);
            }
            return;
        }

        // Generic fallback: vectorized numOps
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.MultiplyScalar(a.AsSpan(), scalar, a.AsWritableSpan());
    }

    /// <summary>
    /// Multiplies all elements of a tensor by a scalar into a pre-allocated destination. Zero allocation.
    /// Uses SIMD with parallel chunking for float, vectorized numOps for all types.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public unsafe void TensorMultiplyScalarInto<T>(Tensor<T> destination, Tensor<T> a, T scalar)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!a.IsContiguous) a = a.Contiguous();
        if (destination.Length < a.Length)
            throw new ArgumentException($"Destination length ({destination.Length}) must be >= source length ({a.Length}).");

        int length = a.Length;

        // Float fast path: SIMD MultiplyScalar with parallel chunking
        if (typeof(T) == typeof(float))
        {
            float scalarF = (float)(object)scalar!;
            var aMem = AsFloatMemory(a.Data);
            var dMem = AsFloatMemory(destination.Data);
            using var pinA = aMem.Pin();
            using var pinD = dMem.Pin();
            float* pA = (float*)pinA.Pointer;
            float* pD = (float*)pinD.Pointer;

            int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 2_000_000));
            if (numChunks >= 2)
            {
                int chunkSize = (length + numChunks - 1) / numChunks;
                chunkSize = (chunkSize + 31) & ~31;
                Parallel.For(0, numChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                        SimdKernels.MultiplyScalarUnsafe(pA + start, scalarF, pD + start, count);
                });
            }
            else
            {
                SimdKernels.MultiplyScalarUnsafe(pA, scalarF, pD, length);
            }
            return;
        }

        // Double fast path: SIMD MultiplyScalar with parallel chunking
        if (typeof(T) == typeof(double))
        {
            double scalarD = (double)(object)scalar!;
            var aMem = AsDoubleMemory(a.Data);
            var dMem = AsDoubleMemory(destination.Data);
            using var pinA = aMem.Pin();
            using var pinD = dMem.Pin();
            double* pA = (double*)pinA.Pointer;
            double* pD = (double*)pinD.Pointer;

            int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 2_000_000));
            if (numChunks >= 2)
            {
                int chunkSize = (length + numChunks - 1) / numChunks;
                chunkSize = (chunkSize + 31) & ~31;
                Parallel.For(0, numChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                        SimdKernels.MultiplyScalarUnsafe(pA + start, scalarD, pD + start, count);
                });
            }
            else
            {
                SimdKernels.MultiplyScalarUnsafe(pA, scalarD, pD, length);
            }
            return;
        }

        // Generic fallback: vectorized numOps
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.MultiplyScalar(a.AsSpan(), scalar, destination.AsWritableSpan());
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMultiplyMany<T>(params Tensor<T>[] tensors)
    {
        if (tensors == null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length < 2)
            throw new ArgumentException("TensorMultiplyMany requires at least 2 tensors.", nameof(tensors));

        // Validate all shapes match the first tensor
        var referenceShape = tensors[0]._shape;
        for (int t = 1; t < tensors.Length; t++)
        {
            if (!ShapesMatch(referenceShape, tensors[t]._shape))
            {
                throw new ArgumentException(
                    $"All tensor shapes must match. Tensor 0 has shape {FormatShape(referenceShape)}, " +
                    $"but tensor {t} has shape {FormatShape(tensors[t]._shape)}.");
            }
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        // Chain pairwise multiplications: result = t0 * t1 * t2 * ...
        var result = TensorAllocator.RentUninitialized<T>(referenceShape);
        numOps.Multiply(tensors[0].AsSpan(), tensors[1].AsSpan(), result.AsWritableSpan());

        for (int t = 2; t < tensors.Length; t++)
        {
            numOps.Multiply(result.AsSpan(), tensors[t].AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordIfActive("TensorMultiplyMany", result, tensors, BackwardFunctions<T>.MultiplyManyBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMultiplyScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);

        if (tensor.IsContiguous)
        {
            numOps.MultiplyScalar(tensor.AsSpan(), scalar, result.AsWritableSpan());
        }
        else
        {
            var src = tensor._storage.GetDataArray();
            var dst = result.GetDataArray();
            for (int i = 0; i < tensor.Length; i++)
                dst[i] = numOps.Multiply(src[tensor.LogicalToStorageIndex(i)], scalar);
        }

        if (scalar is not null)
            DifferentiableOps.RecordUnary("TensorMultiplyScalar", result, tensor, BackwardFunctions<T>.MultiplyScalarBackward, new object[] { scalar });
        return result;
    }

    /// <inheritdoc/>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public virtual unsafe Tensor<T> TensorDivide<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        int length = a.Length;

        // Stride-aware: strided iteration for non-contiguous views (zero-copy)
        if (!a.IsContiguous || !b.IsContiguous)
        {
            var aArr = a._storage.GetDataArray(); var bArr = b._storage.GetDataArray(); var rArr = result.GetDataArray();
            var ops = MathHelper.GetNumericOperations<T>();
            if (a.IsContiguous) { int aOff = a._storageOffset; for (int i = 0; i < length; i++) rArr[i] = ops.Divide(aArr[aOff + i], bArr[b.LogicalToStorageIndex(i)]); }
            else if (b.IsContiguous) { int bOff = b._storageOffset; for (int i = 0; i < length; i++) rArr[i] = ops.Divide(aArr[a.LogicalToStorageIndex(i)], bArr[bOff + i]); }
            else { for (int i = 0; i < length; i++) rArr[i] = ops.Divide(aArr[a.LogicalToStorageIndex(i)], bArr[b.LogicalToStorageIndex(i)]); }
        }
        else if (typeof(T) == typeof(float))
        {
            var aMem = AsFloatMemory(a.Data);
            var bMem = AsFloatMemory(b.Data);
            var rMem = AsFloatMemory(result.Data);
            using var pinA = aMem.Pin();
            using var pinB = bMem.Pin();
            using var pinR = rMem.Pin();
            float* pA = (float*)pinA.Pointer;
            float* pB = (float*)pinB.Pointer;
            float* pR = (float*)pinR.Pointer;

            if (CpuJitSelfTest.IsVerified && length >= 64)
            {
                JitBinaryDispatch(pA, pB, pR, length, JitBinaryOp.Divide);
            }
            else
            {
                int subChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 500_000));
                if (subChunks >= 2)
                {
                    int chunkSize = (length + subChunks - 1) / subChunks;
                    chunkSize = (chunkSize + 31) & ~31;
                    Parallel.For(0, subChunks, chunk =>
                    {
                        int start = chunk * chunkSize;
                        int count = Math.Min(chunkSize, length - start);
                        if (count > 0)
                            SimdKernels.VectorDivideUnsafe(pA + start, pB + start, pR + start, count);
                    });
                }
                else
                {
                    SimdKernels.VectorDivideUnsafe(pA, pB, pR, length);
                }
            }
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Divide(a.AsSpan(), b.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordBinary("TensorDivide", result, a, b, BackwardFunctions<T>.DivideBackward);
        return result;
    }

    #region Tensor Comparison Operations

    /// <inheritdoc/>
    public Tensor<T> TensorEquals<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.Equals(src[i], value) ? numOps.One : numOps.Zero;

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorEquals<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        var srcA = a.AsSpan();
        var srcB = b.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < srcA.Length; i++)
            dest[i] = numOps.Equals(srcA[i], srcB[i]) ? numOps.One : numOps.Zero;

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorNotEquals<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = !numOps.Equals(src[i], value) ? numOps.One : numOps.Zero;

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorNotEquals<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        var srcA = a.AsSpan();
        var srcB = b.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < srcA.Length; i++)
            dest[i] = !numOps.Equals(srcA[i], srcB[i]) ? numOps.One : numOps.Zero;

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorGreaterThan<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        var srcA = a.AsSpan();
        var srcB = b.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < srcA.Length; i++)
            dest[i] = numOps.GreaterThan(srcA[i], srcB[i]) ? numOps.One : numOps.Zero;

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorGreaterThan<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.GreaterThan(src[i], value) ? numOps.One : numOps.Zero;

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorLessThan<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        var srcA = a.AsSpan();
        var srcB = b.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < srcA.Length; i++)
            dest[i] = numOps.LessThan(srcA[i], srcB[i]) ? numOps.One : numOps.Zero;

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorLessThan<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.LessThan(src[i], value) ? numOps.One : numOps.Zero;

        return result;
    }

    #endregion

    #region Tensor Element-wise Math Operations

    /// <inheritdoc/>
    public virtual unsafe Tensor<T> TensorLog<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        int length = tensor.Length;

        if (typeof(T) == typeof(float))
        {
            var iMem = AsFloatMemory(tensor.Data);
            var rMem = AsFloatMemory(result.Data);
            using var pinI = iMem.Pin();
            using var pinR = rMem.Pin();
            float* pI = (float*)pinI.Pointer;
            float* pR = (float*)pinR.Pointer;
            ParallelComputeBound(pI, pR, length, SimdKernels.LogUnsafe);
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Log(tensor.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordUnary("TensorLog", result, tensor, BackwardFunctions<T>.LogBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual unsafe Tensor<T> TensorExp<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        int length = tensor.Length;

        if (typeof(T) == typeof(float))
        {
            var iMem = AsFloatMemory(tensor.Data);
            var rMem = AsFloatMemory(result.Data);
            using var pinI = iMem.Pin();
            using var pinR = rMem.Pin();
            float* pI = (float*)pinI.Pointer;
            float* pR = (float*)pinR.Pointer;
            ParallelComputeBound(pI, pR, length, SimdKernels.ExpUnsafe);
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Exp(tensor.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordUnary("TensorExp", result, tensor, BackwardFunctions<T>.ExpBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual unsafe Tensor<T> TensorSqrt<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        int length = tensor.Length;

        if (typeof(T) == typeof(float))
        {
            var iMem = AsFloatMemory(tensor.Data);
            var rMem = AsFloatMemory(result.Data);
            using var pinI = iMem.Pin();
            using var pinR = rMem.Pin();
            float* pI = (float*)pinI.Pointer;
            float* pR = (float*)pinR.Pointer;
            ParallelComputeBound(pI, pR, length, SimdKernels.SqrtUnsafe);
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Sqrt(tensor.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordUnary("TensorSqrt", result, tensor, BackwardFunctions<T>.SqrtBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual unsafe Tensor<T> TensorAbs<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        int length = tensor.Length;

        if (typeof(T) == typeof(float))
        {
            var iMem = AsFloatMemory(tensor.Data);
            var rMem = AsFloatMemory(result.Data);
            using var pinI = iMem.Pin();
            using var pinR = rMem.Pin();
            float* pI = (float*)pinI.Pointer;
            float* pR = (float*)pinR.Pointer;
            ParallelComputeBound(pI, pR, length, SimdKernels.AbsUnsafe);
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Abs(tensor.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordUnary("TensorAbs", result, tensor, BackwardFunctions<T>.AbsBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorNegate<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        numOps.Negate(tensor.AsSpan(), result.AsWritableSpan());

        DifferentiableOps.RecordUnary("TensorNegate", result, tensor, BackwardFunctions<T>.NegateBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> StopGradient<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        // Copy data to a new tensor with no tape connection.
        // Intentionally does NOT call DifferentiableOps.Record — this is the whole point.
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        tensor.AsSpan().CopyTo(result.AsWritableSpan());
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPower<T>(Tensor<T> tensor, T exponent)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.Power(src[i], exponent);

        if (exponent is not null)
            DifferentiableOps.RecordUnary("TensorPower", result, tensor, BackwardFunctions<T>.PowerBackward, new object[] { exponent });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPower<T>(Tensor<T> bases, Tensor<T> exponents)
    {
        if (bases == null) throw new ArgumentNullException(nameof(bases));
        if (exponents == null) throw new ArgumentNullException(nameof(exponents));
        if (!bases.IsContiguous) bases = bases.Contiguous();
        if (!exponents.IsContiguous) exponents = exponents.Contiguous();
        if (!bases._shape.SequenceEqual(exponents._shape))
            throw new ArgumentException("Tensors must have the same shape for element-wise power.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(bases._shape);
        var srcB = bases.AsSpan();
        var srcE = exponents.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < srcB.Length; i++)
            dest[i] = numOps.Power(srcB[i], srcE[i]);

        DifferentiableOps.RecordBinary("TensorPowerTensor", result, bases, exponents,
            BackwardFunctions<T>.PowerTensorBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFloor<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.Floor(src[i]);

        DifferentiableOps.RecordUnary("Floor", result, tensor,
            BackwardFunctions<T>.SignBackward); // zero gradient: floor is piecewise constant
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCeiling<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.Ceiling(src[i]);

        DifferentiableOps.RecordUnary("Ceiling", result, tensor,
            BackwardFunctions<T>.SignBackward); // zero gradient: ceil is piecewise constant
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRound<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.FromDouble(Math.Round(numOps.ToDouble(src[i])));
        // Round uses straight-through estimator (same as PyTorch) for gradient
        DifferentiableOps.RecordUnary("Round", result, tensor,
            BackwardFunctions<T>.StraightThroughBackward);
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorFrac<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.Frac(src[i]);

        DifferentiableOps.RecordUnary("TensorFrac", result, tensor, BackwardFunctions<T>.FracBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSin<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        numOps.Sin(tensor.AsSpan(), result.AsWritableSpan());

        DifferentiableOps.RecordUnary("Sin", result, tensor,
            BackwardFunctions<T>.SinBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCos<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        numOps.Cos(tensor.AsSpan(), result.AsWritableSpan());

        DifferentiableOps.RecordUnary("Cos", result, tensor,
            BackwardFunctions<T>.CosBackward);
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorTrilinearInterpolate<T>(Tensor<T> grid, Tensor<T> positions)
    {
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (positions == null) throw new ArgumentNullException(nameof(positions));
        if (grid._shape.Length != 4)
            throw new ArgumentException("Grid must be 4D tensor of shape [D, H, W, C]", nameof(grid));
        if (positions._shape.Length != 2 || positions._shape[1] != 3)
            throw new ArgumentException("Positions must be 2D tensor of shape [N, 3]", nameof(positions));

        var numOps = MathHelper.GetNumericOperations<T>();
        int depth = grid._shape[0];
        int height = grid._shape[1];
        int width = grid._shape[2];
        int channels = grid._shape[3];
        int numPositions = positions._shape[0];

        var result = TensorAllocator.Rent<T>(new[] { numPositions, channels });

        Parallel.For(0, numPositions, n =>
        {
            // Get position (z, y, x)
            T pz = positions[n, 0];
            T py = positions[n, 1];
            T px = positions[n, 2];

            // Clamp to valid range and get integer and fractional parts
            double z = Math.Max(0, Math.Min(depth - 1.001, numOps.ToDouble(pz)));
            double y = Math.Max(0, Math.Min(height - 1.001, numOps.ToDouble(py)));
            double x = Math.Max(0, Math.Min(width - 1.001, numOps.ToDouble(px)));

            int z0 = (int)Math.Floor(z);
            int y0 = (int)Math.Floor(y);
            int x0 = (int)Math.Floor(x);
            int z1 = Math.Min(z0 + 1, depth - 1);
            int y1 = Math.Min(y0 + 1, height - 1);
            int x1 = Math.Min(x0 + 1, width - 1);

            double fz = z - z0;
            double fy = y - y0;
            double fx = x - x0;

            // Trilinear interpolation weights for 8 corners
            double w000 = (1 - fz) * (1 - fy) * (1 - fx);
            double w001 = (1 - fz) * (1 - fy) * fx;
            double w010 = (1 - fz) * fy * (1 - fx);
            double w011 = (1 - fz) * fy * fx;
            double w100 = fz * (1 - fy) * (1 - fx);
            double w101 = fz * (1 - fy) * fx;
            double w110 = fz * fy * (1 - fx);
            double w111 = fz * fy * fx;

            for (int c = 0; c < channels; c++)
            {
                // Get values at 8 corners
                double v000 = numOps.ToDouble(grid[z0, y0, x0, c]);
                double v001 = numOps.ToDouble(grid[z0, y0, x1, c]);
                double v010 = numOps.ToDouble(grid[z0, y1, x0, c]);
                double v011 = numOps.ToDouble(grid[z0, y1, x1, c]);
                double v100 = numOps.ToDouble(grid[z1, y0, x0, c]);
                double v101 = numOps.ToDouble(grid[z1, y0, x1, c]);
                double v110 = numOps.ToDouble(grid[z1, y1, x0, c]);
                double v111 = numOps.ToDouble(grid[z1, y1, x1, c]);

                // Weighted sum
                double interpolated = w000 * v000 + w001 * v001 + w010 * v010 + w011 * v011 +
                                     w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111;

                result[n, c] = numOps.FromDouble(interpolated);
            }
        });

        DifferentiableOps.RecordUnary("TensorTrilinearInterpolate", result, grid, BackwardFunctions<T>.TrilinearInterpolateBackward, new object[] { positions });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorTrilinearInterpolateBackward<T>(Tensor<T> gradOutput, Tensor<T> grid, Tensor<T> positions)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (positions == null) throw new ArgumentNullException(nameof(positions));
        if (grid._shape.Length != 4)
            throw new ArgumentException("Grid must be 4D tensor of shape [D, H, W, C]", nameof(grid));
        if (positions._shape.Length != 2 || positions._shape[1] != 3)
            throw new ArgumentException("Positions must be 2D tensor of shape [N, 3]", nameof(positions));

        var numOps = MathHelper.GetNumericOperations<T>();
        int depth = grid._shape[0];
        int height = grid._shape[1];
        int width = grid._shape[2];
        int channels = grid._shape[3];
        int numPositions = positions._shape[0];

        // Initialize gradient grid with zeros
        var gradGrid = TensorAllocator.RentUninitialized<T>(grid._shape);

        // Use thread-local gradients to avoid contention, then combine
        int numThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        var threadLocalGrads = new double[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            threadLocalGrads[t] = new double[depth * height * width * channels];
        }

        Parallel.For(0, numPositions, () => Thread.CurrentThread.ManagedThreadId % numThreads, (n, _, threadIndex) =>
        {
            var localGrad = threadLocalGrads[threadIndex];

            // Get position (z, y, x)
            T pz = positions[n, 0];
            T py = positions[n, 1];
            T px = positions[n, 2];

            // Clamp to valid range and get integer and fractional parts
            double z = Math.Max(0, Math.Min(depth - 1.001, numOps.ToDouble(pz)));
            double y = Math.Max(0, Math.Min(height - 1.001, numOps.ToDouble(py)));
            double x = Math.Max(0, Math.Min(width - 1.001, numOps.ToDouble(px)));

            int z0 = (int)Math.Floor(z);
            int y0 = (int)Math.Floor(y);
            int x0 = (int)Math.Floor(x);
            int z1 = Math.Min(z0 + 1, depth - 1);
            int y1 = Math.Min(y0 + 1, height - 1);
            int x1 = Math.Min(x0 + 1, width - 1);

            double fz = z - z0;
            double fy = y - y0;
            double fx = x - x0;

            // Trilinear interpolation weights for 8 corners (same as forward pass)
            double w000 = (1 - fz) * (1 - fy) * (1 - fx);
            double w001 = (1 - fz) * (1 - fy) * fx;
            double w010 = (1 - fz) * fy * (1 - fx);
            double w011 = (1 - fz) * fy * fx;
            double w100 = fz * (1 - fy) * (1 - fx);
            double w101 = fz * (1 - fy) * fx;
            double w110 = fz * fy * (1 - fx);
            double w111 = fz * fy * fx;

            for (int c = 0; c < channels; c++)
            {
                double grad = numOps.ToDouble(gradOutput[n, c]);

                // Scatter gradient to 8 corners weighted by interpolation weights
                int stride = height * width * channels;
                int idx000 = z0 * stride + y0 * width * channels + x0 * channels + c;
                int idx001 = z0 * stride + y0 * width * channels + x1 * channels + c;
                int idx010 = z0 * stride + y1 * width * channels + x0 * channels + c;
                int idx011 = z0 * stride + y1 * width * channels + x1 * channels + c;
                int idx100 = z1 * stride + y0 * width * channels + x0 * channels + c;
                int idx101 = z1 * stride + y0 * width * channels + x1 * channels + c;
                int idx110 = z1 * stride + y1 * width * channels + x0 * channels + c;
                int idx111 = z1 * stride + y1 * width * channels + x1 * channels + c;

                localGrad[idx000] += w000 * grad;
                localGrad[idx001] += w001 * grad;
                localGrad[idx010] += w010 * grad;
                localGrad[idx011] += w011 * grad;
                localGrad[idx100] += w100 * grad;
                localGrad[idx101] += w101 * grad;
                localGrad[idx110] += w110 * grad;
                localGrad[idx111] += w111 * grad;
            }

            return threadIndex;
        }, _ => { });

        // Combine thread-local gradients
        int totalElements = depth * height * width * channels;
        var gradGridData = gradGrid.GetDataArray();
        Parallel.For(0, totalElements, i =>
        {
            double sum = 0;
            for (int t = 0; t < numThreads; t++)
            {
                sum += threadLocalGrads[t][i];
            }
            gradGridData[i] = numOps.FromDouble(sum);
        });

        return gradGrid;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorPow<T>(Tensor<T> tensor, T exponent)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.Power(src[i], exponent);

        object boxedExponent = exponent is not null ? (object)exponent : throw new InvalidOperationException("Exponent cannot be null");
        DifferentiableOps.RecordUnary("TensorPow", result, tensor, BackwardFunctions<T>.TensorPowBackward, new object[] { boxedExponent });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMax<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        if (!ShapesMatch(a._shape, b._shape))
            throw new ArgumentException($"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        var srcA = a.AsSpan();
        var srcB = b.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < srcA.Length; i++)
        {
            var aVal = srcA[i];
            var bVal = srcB[i];
            dest[i] = numOps.GreaterThan(aVal, bVal) ? aVal : bVal;
        }

        DifferentiableOps.RecordBinary("TensorMax", result, a, b, BackwardFunctions<T>.MaxBackward);
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMax<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
        {
            var tVal = src[i];
            dest[i] = numOps.GreaterThan(tVal, value) ? tVal : value;
        }

        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMin<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        if (!ShapesMatch(a._shape, b._shape))
            throw new ArgumentException($"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        var srcA = a.AsSpan();
        var srcB = b.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < srcA.Length; i++)
        {
            var aVal = srcA[i];
            var bVal = srcB[i];
            dest[i] = numOps.LessThan(aVal, bVal) ? aVal : bVal;
        }

        DifferentiableOps.RecordBinary("TensorMin", result, a, b, BackwardFunctions<T>.MinBackward);
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMin<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
        {
            var tVal = src[i];
            dest[i] = numOps.LessThan(tVal, value) ? tVal : value;
        }

        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorClamp<T>(Tensor<T> tensor, T min, T max)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var src = tensor.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
        {
            var val = src[i];
            if (numOps.LessThan(val, min)) val = min;
            else if (numOps.GreaterThan(val, max)) val = max;
            dest[i] = val;
        }

        DifferentiableOps.RecordUnary("Clamp", result, tensor,
            BackwardFunctions<T>.ClampBackward,
            savedState: new object[] { numOps.ToDouble(min), numOps.ToDouble(max) });
        return result;
    }

    /// <inheritdoc/>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public unsafe T TensorSum<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Stride-aware: sum via stride indices for non-contiguous views
        if (!tensor.IsContiguous)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var srcArr = tensor._storage.GetDataArray();
            var idxBuf = new int[tensor.Length];
            tensor.FillStorageIndices(idxBuf);
            T sum = ops.Zero;
            for (int i = 0; i < tensor.Length; i++) sum = ops.Add(sum, srcArr[idxBuf[i]]);
            return sum;
        }

        // Float fast path: bypass generic dispatch + Span overhead
        if (typeof(T) == typeof(float))
        {
            T[] arr = tensor.GetDataArray();
            float[] fArr = Unsafe.As<T[], float[]>(ref arr);
            int length = tensor.Length;
            float result;

            // Compute numChunks purely from length so chunk boundaries (and therefore
            // FP summation order) are deterministic regardless of MaxDegreeOfParallelism.
            // Concurrency is limited separately by the LightweightParallel executor's
            // fixed thread pool; chunks exceeding the thread count are simply queued.
            int numChunks = length >= 200_000 ? Math.Max(2, length / 50_000) : 1;
            if (numChunks >= 2 && CpuParallelSettings.MaxDegreeOfParallelism > 1)
            {
                var handle = System.Runtime.InteropServices.GCHandle.Alloc(fArr, System.Runtime.InteropServices.GCHandleType.Pinned);
                try
                {
                    float* p = (float*)handle.AddrOfPinnedObject();
                    int chunkSize = (length + numChunks - 1) / numChunks;
                    chunkSize = (chunkSize + 31) & ~31; // Align to 32 floats for AVX
                    float[] partials = new float[numChunks];

                    CpuParallelSettings.LightweightParallel(numChunks, chunk =>
                    {
                        int start = chunk * chunkSize;
                        int count = Math.Min(chunkSize, length - start);
                        if (count > 0)
                        {
                            partials[chunk] = SimdKernels.SumUnsafe(p + start, count);
                        }
                    });

                    result = 0f;
                    for (int c = 0; c < numChunks; c++)
                        result += partials[c];
                }
                finally
                {
                    handle.Free();
                }
            }
            else
            {
                fixed (float* ptr = fArr)
                {
                    result = SimdKernels.SumUnsafe(ptr, length);
                }
            }
            return Unsafe.As<float, T>(ref result);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        // Use SIMD-optimized sum via IVectorizedOperations
        return numOps.Sum(tensor.AsSpan());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ReduceSum<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Stride-aware: for single-axis reduction on non-contiguous views, use direct stride math
        if (!tensor.IsContiguous && axes != null && axes.Length == 1)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            int axis = axes[0] < 0 ? tensor.Rank + axes[0] : axes[0];
            if (axis < 0 || axis >= tensor.Rank) throw new ArgumentOutOfRangeException(nameof(axes), $"Axis {axes[0]} out of range for rank {tensor.Rank}");
            var (outerSize, axisSize, innerSize) = tensor.GetReductionDims(axis);
            var outShape = new List<int>();
            for (int d = 0; d < tensor.Rank; d++)
            {
                if (d == axis) { if (keepDims) outShape.Add(1); }
                else outShape.Add(tensor._shape[d]);
            }
            var result = TensorAllocator.RentUninitialized<T>(outShape.ToArray());
            var rArr = result.GetDataArray();
            var srcArr = tensor._storage.GetDataArray();

            for (int o = 0; o < outerSize; o++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    T sum = numOps.Zero;
                    for (int a = 0; a < axisSize; a++)
                        sum = numOps.Add(sum, srcArr[tensor.ReductionStorageIndex(o, a, inner, axis)]);
                    rArr[o * innerSize + inner] = sum;
                }
            }
            DifferentiableOps.RecordUnary("ReduceSum", result, tensor, BackwardFunctions<T>.ReduceSumBackward,
                new object[] { new[] { axis }, keepDims });
            return result;
        }

        // For multi-axis or full reductions, materialize if needed
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        // Full reduction - sum all elements
        if (axes == null || axes.Length == 0)
        {
            // Full reduction: all axes
            var allAxes = new int[tensor.Rank];
            for (int ax = 0; ax < tensor.Rank; ax++) allAxes[ax] = ax;

            T sum = TensorSum(tensor);
            Tensor<T> fullResult;
            if (keepDims)
            {
                var shape = new int[tensor.Rank];
                for (int i = 0; i < tensor.Rank; i++) shape[i] = 1;
                fullResult = TensorAllocator.RentUninitialized<T>(shape);
                fullResult.GetDataArray()[0] = sum;
            }
            else
            {
                fullResult = TensorAllocator.Rent<T>([1], new Vector<T>([sum]));
            }
            DifferentiableOps.RecordUnary("ReduceSum", fullResult, tensor, BackwardFunctions<T>.ReduceSumBackward,
                new object[] { allAxes, keepDims });
            return fullResult;
        }

        // Validate and normalize axes consistently with other reducers
        var normalizedAxes = ValidateAndNormalizeAxes(axes, tensor.Rank);

        // Fast path: single-axis sum on 2D contiguous float tensor
        if (normalizedAxes.Length == 1 && tensor.Rank == 2 && tensor.IsContiguous && typeof(T) == typeof(float))
        {
            int axis = normalizedAxes[0];
            int rows = tensor._shape[0], cols = tensor._shape[1];
            var srcArr = (float[])(object)tensor.GetDataArray();

            if (axis == 0)
            {
                // Sum along rows → output [1, cols] or [cols]
                var outShape = keepDims ? new[] { 1, cols } : new[] { cols };
                var fastResult = TensorAllocator.RentUninitialized<T>(outShape);
                var rArr = (float[])(object)fastResult.GetDataArray();
                Array.Clear(rArr, 0, cols);
                for (int r = 0; r < rows; r++)
                {
                    int offset = r * cols;
                    for (int c = 0; c < cols; c++)
                        rArr[c] += srcArr[offset + c];
                }
                DifferentiableOps.RecordUnary("ReduceSum", fastResult, tensor, BackwardFunctions<T>.ReduceSumBackward,
                    new object[] { normalizedAxes.ToArray(), keepDims });
                return fastResult;
            }
            else if (axis == 1)
            {
                // Sum along cols → output [rows, 1] or [rows]
                var outShape = keepDims ? new[] { rows, 1 } : new[] { rows };
                var fastResult = TensorAllocator.RentUninitialized<T>(outShape);
                var rArr = (float[])(object)fastResult.GetDataArray();
                for (int r = 0; r < rows; r++)
                {
                    float sum = 0;
                    int offset = r * cols;
                    for (int c = 0; c < cols; c++)
                        sum += srcArr[offset + c];
                    rArr[r] = sum;
                }
                DifferentiableOps.RecordUnary("ReduceSum", fastResult, tensor, BackwardFunctions<T>.ReduceSumBackward,
                    new object[] { normalizedAxes.ToArray(), keepDims });
                return fastResult;
            }
        }

        // Calculate output shape
        var outputShape = new List<int>();
        for (int i = 0; i < tensor.Rank; i++)
        {
            if (normalizedAxes.Contains(i))
            {
                if (keepDims) outputShape.Add(1);
            }
            else
            {
                outputShape.Add(tensor._shape[i]);
            }
        }

        var result2 = TensorAllocator.RentUninitialized<T>(outputShape.ToArray());

        // Use tensor's built-in Sum which is already optimized
        var summed = tensor.Sum(normalizedAxes);

        Tensor<T> reduceSumResult;
        // Copy to result with correct shape
        if (keepDims && summed.Rank != result2.Rank)
        {
            Array.Copy(summed.GetDataArray(), result2.GetDataArray(), summed.Length);
            reduceSumResult = result2;
        }
        else
        {
            reduceSumResult = summed;
        }

        DifferentiableOps.RecordUnary("ReduceSum", reduceSumResult, tensor, BackwardFunctions<T>.ReduceSumBackward,
            new object[] { normalizedAxes.ToArray(), keepDims });
        return reduceSumResult;
    }

    /// <inheritdoc/>
    public unsafe T TensorMaxValue<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Length == 0) throw new ArgumentException("Cannot compute max of empty tensor.", nameof(tensor));

        if (typeof(T) == typeof(float) && tensor.GetFlattenedData() is float[] arr)
        {
            int length = tensor.Length;
            fixed (float* ptr = arr)
            {
                float result = ParallelReduceFloat(ptr, length, float.NegativeInfinity,
                    SimdKernels.MaxUnsafe, Math.Max);
                return Unsafe.As<float, T>(ref result);
            }
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Max(tensor.AsSpan());
    }

    /// <inheritdoc/>
    public unsafe T TensorMinValue<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (tensor.Length == 0) throw new ArgumentException("Cannot compute min of empty tensor.", nameof(tensor));

        if (typeof(T) == typeof(float) && tensor.GetFlattenedData() is float[] arr)
        {
            int length = tensor.Length;
            fixed (float* ptr = arr)
            {
                float result = ParallelReduceFloat(ptr, length, float.PositiveInfinity,
                    SimdKernels.MinUnsafe, Math.Min);
                return Unsafe.As<float, T>(ref result);
            }
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Min(tensor.AsSpan());
    }

    private unsafe delegate float UnsafeReductionKernel(float* data, int length);

    /// <summary>
    /// Parallel reduction for float arrays. Splits into chunks, reduces each chunk,
    /// then combines results. Used for Max, Min, Sum reductions on large arrays.
    /// </summary>
    private static unsafe float ParallelReduceFloat(float* data, int length, float identity,
        UnsafeReductionKernel kernel, Func<float, float, float> combine)
    {
        const int parallelThreshold = 256 * 1024;
        int maxThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        int chunks = Math.Min(maxThreads, Math.Max(1, length / parallelThreshold));

        if (chunks < 2)
            return kernel(data, length);

        int chunkSize = (length + chunks - 1) / chunks;
        chunkSize = (chunkSize + 31) & ~31; // Align to 32 elements
        float[] partials = new float[chunks];
        for (int i = 0; i < chunks; i++) partials[i] = identity;

        IntPtr pData = (IntPtr)data;
        int totalLength = length;

        Parallel.For(0, chunks, chunk =>
        {
            int start = chunk * chunkSize;
            int count = Math.Min(chunkSize, totalLength - start);
            if (count > 0)
                partials[chunk] = kernel((float*)pData + start, count);
        });

        float result = partials[0];
        for (int i = 1; i < chunks; i++)
            result = combine(result, partials[i]);
        return result;
    }

    /// <inheritdoc/>
    public unsafe T TensorMean<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Length == 0) throw new ArgumentException("Cannot compute mean of empty tensor.", nameof(tensor));

        // Route through TensorSum which has the parallel fast path
        T sum = TensorSum(tensor);
        if (typeof(T) == typeof(float))
        {
            float fSum = Unsafe.As<T, float>(ref sum);
            float result = fSum / tensor.Length;
            return Unsafe.As<float, T>(ref result);
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Divide(sum, numOps.FromDouble(tensor.Length));
    }

    #endregion

    /// <summary>
    /// Helper method to check if two shapes match.
    /// </summary>
    private bool ShapesMatch(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length)
            return false;

        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Helper method to format a shape for error messages.
    /// </summary>
    private string FormatShape(int[] shape)
    {
        return "[" + string.Join(", ", shape) + "]";
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float MaxPool3x3Padded(float* data, int h, int w, int ihStart, int iwStart)
    {
        float m = float.NegativeInfinity;
        int khStart = ihStart < 0 ? -ihStart : 0;
        int kwStart = iwStart < 0 ? -iwStart : 0;
        int khEnd = 3 < (h - ihStart) ? 3 : (h - ihStart);
        int kwEnd = 3 < (w - iwStart) ? 3 : (w - iwStart);

        for (int kh = khStart; kh < khEnd; kh++)
        {
            float* row = data + (ihStart + kh) * w + iwStart;
            for (int kw = kwStart; kw < kwEnd; kw++)
            {
                float v = row[kw];
                if (v > m) m = v;
            }
        }
        return m;
    }

    private static unsafe void MaxPool2DFloat3x3NoPad(float[] inArr, float[] outArr, int bc, int h, int w, int oH, int oW, int st)
    {
        fixed (float* pIn = inArr)
        fixed (float* pOut = outArr)
        {
            for (int idx = 0; idx < bc; idx++)
            {
                float* inBase = pIn + idx * h * w;
                float* outBase = pOut + idx * oH * oW;

                for (int oh = 0; oh < oH; oh++)
                {
                    int ihStart = oh * st;
                    float* r0 = inBase + ihStart * w;
                    float* r1 = r0 + w;
                    float* r2 = r1 + w;
                    float* dst = outBase + oh * oW;

                    for (int ow = 0; ow < oW; ow++)
                    {
                        int iw = ow * st;
                        float m = r0[iw];
                        float v = r0[iw + 1]; if (v > m) m = v;
                        v = r0[iw + 2]; if (v > m) m = v;
                        v = r1[iw]; if (v > m) m = v;
                        v = r1[iw + 1]; if (v > m) m = v;
                        v = r1[iw + 2]; if (v > m) m = v;
                        v = r2[iw]; if (v > m) m = v;
                        v = r2[iw + 1]; if (v > m) m = v;
                        v = r2[iw + 2]; if (v > m) m = v;
                        dst[ow] = m;
                    }
                }
            }
        }
    }

    private static unsafe void MaxPool2DFloat2x2NoPad(float[] inArr, float[] outArr, int bc, int h, int w, int oH, int oW, int st)
    {
        fixed (float* pIn = inArr)
        fixed (float* pOut = outArr)
        {
            for (int idx = 0; idx < bc; idx++)
            {
                float* inBase = pIn + idx * h * w;
                float* outBase = pOut + idx * oH * oW;

                for (int oh = 0; oh < oH; oh++)
                {
                    int ihStart = oh * st;
                    float* r0 = inBase + ihStart * w;
                    float* r1 = r0 + w;
                    float* dst = outBase + oh * oW;

                    for (int ow = 0; ow < oW; ow++)
                    {
                        int iw = ow * st;
                        float m = r0[iw];
                        float v = r0[iw + 1]; if (v > m) m = v;
                        v = r1[iw]; if (v > m) m = v;
                        v = r1[iw + 1]; if (v > m) m = v;
                        dst[ow] = m;
                    }
                }
            }
        }
    }

    private static unsafe void MaxPool2DFloat3x3Padded(float[] inArr, float[] outArr, int bc, int h, int w, int oH, int oW, int st, int pd)
    {
        // Compute interior bounds where all 9 kernel elements are valid
        int ohInteriorStart = (pd + st - 1) / st;
        int ohInteriorEnd = (h + pd - 3) / st + 1;
        int owInteriorStart = (pd + st - 1) / st;
        int owInteriorEnd = (w + pd - 3) / st + 1;
        if (ohInteriorEnd > oH) ohInteriorEnd = oH;
        if (owInteriorEnd > oW) owInteriorEnd = oW;

        fixed (float* pIn = inArr)
        fixed (float* pOut = outArr)
        {
            for (int idx = 0; idx < bc; idx++)
            {
                float* inBase = pIn + idx * h * w;
                float* outBase = pOut + idx * oH * oW;

                for (int oh = 0; oh < oH; oh++)
                {
                    float* dst = outBase + oh * oW;

                    if (oh >= ohInteriorStart && oh < ohInteriorEnd)
                    {
                        int ihStart = oh * st - pd;
                        float* r0 = inBase + ihStart * w;
                        float* r1 = r0 + w;
                        float* r2 = r1 + w;

                        // Left border columns
                        for (int ow = 0; ow < owInteriorStart && ow < oW; ow++)
                        {
                            dst[ow] = MaxPool3x3Padded(inBase, h, w, oh * st - pd, ow * st - pd);
                        }

                        // Interior columns: full unrolled 3x3
                        for (int ow = owInteriorStart; ow < owInteriorEnd; ow++)
                        {
                            int iw = ow * st - pd;
                            float m = r0[iw];
                            float v = r0[iw + 1]; if (v > m) m = v;
                            v = r0[iw + 2]; if (v > m) m = v;
                            v = r1[iw]; if (v > m) m = v;
                            v = r1[iw + 1]; if (v > m) m = v;
                            v = r1[iw + 2]; if (v > m) m = v;
                            v = r2[iw]; if (v > m) m = v;
                            v = r2[iw + 1]; if (v > m) m = v;
                            v = r2[iw + 2]; if (v > m) m = v;
                            dst[ow] = m;
                        }

                        // Right border columns
                        for (int ow = owInteriorEnd; ow < oW; ow++)
                        {
                            dst[ow] = MaxPool3x3Padded(inBase, h, w, oh * st - pd, ow * st - pd);
                        }
                    }
                    else
                    {
                        // Border rows
                        for (int ow = 0; ow < oW; ow++)
                        {
                            dst[ow] = MaxPool3x3Padded(inBase, h, w, oh * st - pd, ow * st - pd);
                        }
                    }
                }
            }
        }
    }

    private static unsafe void MaxPool2DFloatGeneric(float[] inArr, float[] outArr, int bc, int h, int w, int oH, int oW, int ps, int st, int pd)
    {
        fixed (float* pIn = inArr)
        fixed (float* pOut = outArr)
        {
            for (int idx = 0; idx < bc; idx++)
            {
                float* inBase = pIn + idx * h * w;
                float* outBase = pOut + idx * oH * oW;

                for (int oh = 0; oh < oH; oh++)
                {
                    float* dst = outBase + oh * oW;
                    for (int ow = 0; ow < oW; ow++)
                    {
                        float maxVal = float.NegativeInfinity;
                        int ihStart = oh * st - pd;
                        int iwStart = ow * st - pd;
                        int khStart = ihStart < 0 ? -ihStart : 0;
                        int kwStart = iwStart < 0 ? -iwStart : 0;
                        int khEnd = ps < (h - ihStart) ? ps : (h - ihStart);
                        int kwEnd = ps < (w - iwStart) ? ps : (w - iwStart);

                        for (int kh = khStart; kh < khEnd; kh++)
                        {
                            float* row = inBase + (ihStart + kh) * w + iwStart;
                            for (int kw = kwStart; kw < kwEnd; kw++)
                            {
                                float v = row[kw];
                                if (v > maxVal) maxVal = v;
                            }
                        }
                        dst[ow] = maxVal;
                    }
                }
            }
        }
    }

    public virtual Tensor<T> MaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();
        if (input.Rank != 4)
        {
            throw new ArgumentException($"MaxPool2D requires a 4D tensor [batch, channels, height, width]. Got rank {input.Rank}.");
        }
        if (poolSize <= 0) throw new ArgumentException("Pool size must be positive.");

        if (stride == 0) stride = poolSize; // Default stride equals pool size

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int channels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid pooling parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure poolSize={poolSize}, stride={stride}, padding={padding} are compatible with input size {height}x{width}.");
        }

        var outputShape = new[] { batch, channels, outputHeight, outputWidth };
        var result = TensorAllocator.RentUninitialized<T>(outputShape);

        // When tape is active, use WithIndices variant so backward can access max indices
        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var resultWithIdx = MaxPool2DWithIndices(input, new[] { poolSize, poolSize }, new[] { stride, stride }, out var maxIndices);
            DifferentiableOps.RecordUnary("MaxPool2D", resultWithIdx, input, BackwardFunctions<T>.MaxPool2DBackward,
                new object[] { maxIndices, new[] { poolSize, poolSize }, new[] { stride, stride } });
            return resultWithIdx;
        }

        // Float fast path: direct array access with specialized inner loops
        if (typeof(T) == typeof(float) && input.GetDataArray() is float[] inArr && result.GetDataArray() is float[] outArr)
        {
            int bc = batch * channels;
            int h = height, w = width, oH = outputHeight, oW = outputWidth;
            int ps = poolSize, st = stride, pd = padding;

            if (pd == 0 && ps == 3)
            {
                MaxPool2DFloat3x3NoPad(inArr, outArr, bc, h, w, oH, oW, st);
            }
            else if (pd == 0 && ps == 2)
            {
                MaxPool2DFloat2x2NoPad(inArr, outArr, bc, h, w, oH, oW, st);
            }
            else if (ps == 3)
            {
                MaxPool2DFloat3x3Padded(inArr, outArr, bc, h, w, oH, oW, st, pd);
            }
            else
            {
                MaxPool2DFloatGeneric(inArr, outArr, bc, h, w, oH, oW, ps, st, pd);
            }
            return result;
        }

        // Generic fallback
        var inputData = input.GetDataArray();
        var outputData = result.GetDataArray();

        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;
            int inputBaseOffset = (b * channels + c) * height * width;
            int outputBaseOffset = (b * channels + c) * outputHeight * outputWidth;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T maxValue = numOps.MinValue;

                    for (int kh = 0; kh < poolSize; kh++)
                    {
                        int ih = oh * stride + kh - padding;
                        if (ih < 0 || ih >= height) continue;

                        for (int kw = 0; kw < poolSize; kw++)
                        {
                            int iw = ow * stride + kw - padding;
                            if (iw < 0 || iw >= width) continue;

                            T value = inputData[inputBaseOffset + ih * width + iw];
                            if (numOps.GreaterThan(value, maxValue))
                                maxValue = value;
                        }
                    }

                    outputData[outputBaseOffset + oh * outputWidth + ow] = maxValue;
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> AvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();
        if (input.Rank != 4)
        {
            throw new ArgumentException($"AvgPool2D requires a 4D tensor [batch, channels, height, width]. Got rank {input.Rank}.");
        }
        if (poolSize <= 0) throw new ArgumentException("Pool size must be positive.");

        if (stride == 0) stride = poolSize; // Default stride equals pool size

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int channels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outputHeight = (height + 2 * padding - poolSize) / stride + 1;
        int outputWidth = (width + 2 * padding - poolSize) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid pooling parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure poolSize={poolSize}, stride={stride}, padding={padding} are compatible with input size {height}x{width}.");
        }

        var outputShape = new[] { batch, channels, outputHeight, outputWidth };
        var result = TensorAllocator.RentUninitialized<T>(outputShape);

        // Float fast path: direct array access, no virtual dispatch
        if (typeof(T) == typeof(float) && input.GetDataArray() is float[] inArr && result.GetDataArray() is float[] outArr)
        {
            int bc = batch * channels;
            int h = height, w = width, oH = outputHeight, oW = outputWidth;
            int ps = poolSize, st = stride, pd = padding;

            Parallel.For(0, bc, idx =>
            {
                int inputBase = idx * h * w;
                int outputBase = idx * oH * oW;

                for (int oh = 0; oh < oH; oh++)
                {
                    for (int ow = 0; ow < oW; ow++)
                    {
                        float sum = 0f;
                        int count = 0;

                        int ihStart = oh * st - pd;
                        int iwStart = ow * st - pd;
                        int khStart = ihStart < 0 ? -ihStart : 0;
                        int kwStart = iwStart < 0 ? -iwStart : 0;
                        int khEnd = Math.Min(ps, h - ihStart);
                        int kwEnd = Math.Min(ps, w - iwStart);

                        for (int kh = khStart; kh < khEnd; kh++)
                        {
                            int rowOff = inputBase + (ihStart + kh) * w + iwStart;
                            for (int kw = kwStart; kw < kwEnd; kw++)
                            {
                                sum += inArr[rowOff + kw];
                                count++;
                            }
                        }

                        outArr[outputBase + oh * oW + ow] = count > 0 ? sum / count : 0f;
                    }
                }
            });
            DifferentiableOps.RecordUnary("AvgPool2D", result, input, BackwardFunctions<T>.AvgPool2DBackward,
                new object[] { new[] { poolSize, poolSize }, new[] { stride, stride } });
            return result;
        }

        var inputData = input.GetDataArray();
        var outputData = result.GetDataArray();

        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;
            int inputBaseOffset = (b * channels + c) * height * width;
            int outputBaseOffset = (b * channels + c) * outputHeight * outputWidth;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T sum = numOps.Zero;
                    int count = 0;

                    for (int kh = 0; kh < poolSize; kh++)
                    {
                        int ih = oh * stride + kh - padding;
                        if (ih < 0 || ih >= height) continue;
                        for (int kw = 0; kw < poolSize; kw++)
                        {
                            int iw = ow * stride + kw - padding;
                            if (iw < 0 || iw >= width) continue;
                            sum = numOps.Add(sum, inputData[inputBaseOffset + ih * width + iw]);
                            count++;
                        }
                    }

                    outputData[outputBaseOffset + oh * outputWidth + ow] =
                        count > 0 ? numOps.Divide(sum, numOps.FromDouble(count)) : numOps.Zero;
                }
            }
        });

        DifferentiableOps.RecordUnary("AvgPool2D", result, input, BackwardFunctions<T>.AvgPool2DBackward,
            new object[] { new[] { poolSize, poolSize }, new[] { stride, stride } });
        return result;
    }

    /// <inheritdoc/>
    /// <summary>
    /// 1D convolution via reshape to Conv2D with height=1.
    /// Input: [batch, in_channels, length], Kernel: [out_channels, in_channels, kernel_length]
    /// </summary>
    public virtual Tensor<T> Conv1D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 3) throw new ArgumentException($"Conv1D requires 3D input [batch, channels, length]. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 3) throw new ArgumentException($"Conv1D requires 3D kernel [out_ch, in_ch, kernel_len]. Got rank {kernel.Rank}.", nameof(kernel));

        // Reshape to 4D: [B, C, 1, L] and [Cout, Cin, 1, K]
        var input4D = Reshape(input, new[] { input._shape[0], input._shape[1], 1, input._shape[2] });
        var kernel4D = Reshape(kernel, new[] { kernel._shape[0], kernel._shape[1], 1, kernel._shape[2] });

        var result4D = Conv2D(input4D, kernel4D, new[] { 1, stride }, new[] { 0, padding }, new[] { 1, dilation });

        // Squeeze height dimension: [B, Cout, 1, outL] -> [B, Cout, outL]
        var result = Reshape(result4D, new[] { result4D._shape[0], result4D._shape[1], result4D._shape[3] });

        DifferentiableOps.RecordBinary("Conv1D", result, input, kernel, BackwardFunctions<T>.Conv1DBackward,
            new object[] { stride, padding, dilation });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv1DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int stride, int padding, int dilation)
    {
        // Reshape to 4D and use Conv2DBackwardInput
        var grad4D = Reshape(gradOutput, new[] { gradOutput._shape[0], gradOutput._shape[1], 1, gradOutput._shape[2] });
        var kernel4D = Reshape(kernel, new[] { kernel._shape[0], kernel._shape[1], 1, kernel._shape[2] });
        var inputShape4D = new[] { inputShape[0], inputShape[1], 1, inputShape[2] };

        var result4D = Conv2DBackwardInput(grad4D, kernel4D, inputShape4D, new[] { 1, stride }, new[] { 0, padding }, new[] { 1, dilation });
        return Reshape(result4D, inputShape);
    }

    /// <inheritdoc/>
    public Tensor<T> Conv1DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int stride, int padding, int dilation)
    {
        // Reshape to 4D and use Conv2DBackwardKernel
        var grad4D = Reshape(gradOutput, new[] { gradOutput._shape[0], gradOutput._shape[1], 1, gradOutput._shape[2] });
        var input4D = Reshape(input, new[] { input._shape[0], input._shape[1], 1, input._shape[2] });
        var kernelShape4D = new[] { kernelShape[0], kernelShape[1], 1, kernelShape[2] };

        var result4D = Conv2DBackwardKernel(grad4D, input4D, kernelShape4D, new[] { 1, stride }, new[] { 0, padding }, new[] { 1, dilation });
        return Reshape(result4D, kernelShape);
    }

    public virtual Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (!input.IsContiguous) input = input.Contiguous();
        if (input.Rank != 4)
        {
            throw new ArgumentException($"Conv2D input requires a 4D tensor [batch, in_channels, height, width]. Got rank {input.Rank}.");
        }
        if (kernel.Rank != 4)
        {
            throw new ArgumentException($"Conv2D kernel requires a 4D tensor [out_channels, in_channels, kernel_height, kernel_width]. Got rank {kernel.Rank}.");
        }
        if (stride <= 0) throw new ArgumentException("Stride must be positive.");
        if (dilation <= 0) throw new ArgumentException("Dilation must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outChannels = kernel._shape[0];
        int kernelInChannels = kernel._shape[1];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        if (inChannels != kernelInChannels)
        {
            throw new ArgumentException(
                $"Input channels ({inChannels}) must match kernel input channels ({kernelInChannels}).");
        }

        int effectiveKernelHeight = dilation * (kernelHeight - 1) + 1;
        int effectiveKernelWidth = dilation * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padding - effectiveKernelHeight) / stride + 1;
        int outputWidth = (width + 2 * padding - effectiveKernelWidth) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid convolution parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure stride={stride}, padding={padding}, dilation={dilation} are compatible with input size {height}x{width} and kernel size {kernelHeight}x{kernelWidth}.");
        }

        var result = TensorAllocator.Rent<T>(new[] { batch, outChannels, outputHeight, outputWidth });

        // Use im2col + GEMM for float (significantly faster)
        if (typeof(T) == typeof(float))
        {
            Conv2DWithIm2ColFloat(
                input as Tensor<float> ?? throw new InvalidCastException(),
                kernel as Tensor<float> ?? throw new InvalidCastException(),
                result as Tensor<float> ?? throw new InvalidCastException(),
                batch, inChannels, height, width,
                outChannels, kernelHeight, kernelWidth,
                stride, padding, dilation,
                outputHeight, outputWidth);
            DifferentiableOps.RecordBinary("Conv2D", result, input, kernel,
                BackwardFunctions<T>.Conv2DBackward, new object[] { new[] { stride, stride }, new[] { padding, padding }, new[] { dilation, dilation } });
            return result;
        }

        // Use im2col + GEMM for double (same approach as float, with double BLAS)
        if (typeof(T) == typeof(double))
        {
            Conv2DWithIm2ColDouble(
                input as Tensor<double> ?? throw new InvalidCastException(),
                kernel as Tensor<double> ?? throw new InvalidCastException(),
                result as Tensor<double> ?? throw new InvalidCastException(),
                batch, inChannels, height, width,
                outChannels, kernelHeight, kernelWidth,
                stride, padding, dilation,
                outputHeight, outputWidth);
            DifferentiableOps.RecordBinary("Conv2D", result, input, kernel,
                BackwardFunctions<T>.Conv2DBackward, new object[] { new[] { stride, stride }, new[] { padding, padding }, new[] { dilation, dilation } });
            return result;
        }

        // Fallback for other types: naive implementation
        Conv2DNaive(input, kernel, result, numOps,
            batch, inChannels, height, width,
            outChannels, kernelHeight, kernelWidth,
            stride, padding, dilation,
            outputHeight, outputWidth);

        DifferentiableOps.RecordBinary("Conv2D", result, input, kernel,
            BackwardFunctions<T>.Conv2DBackward, new object[] { new[] { stride, stride }, new[] { padding, padding }, new[] { dilation, dilation } });
        return result;
    }

    /// <summary>
    /// Performs 2D convolution, storing result in pre-allocated output tensor. Zero allocation.
    /// </summary>
    /// <param name="output">Pre-allocated output tensor with correct shape [batch, out_channels, output_height, output_width].</param>
    /// <param name="input">Input tensor [batch, in_channels, height, width].</param>
    /// <param name="kernel">Kernel tensor [out_channels, in_channels, kernel_height, kernel_width].</param>
    /// <param name="stride">Stride for the convolution.</param>
    /// <param name="padding">Padding to add to input.</param>
    /// <param name="dilation">Dilation factor for the kernel.</param>
    public void Conv2DInto<T>(Tensor<T> output, Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1)
    {
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));

        // Cheap validation first — fail fast before any allocation
        if (input.Rank != 4)
            throw new ArgumentException($"Conv2D input requires a 4D tensor [batch, in_channels, height, width]. Got rank {input.Rank}.");
        if (kernel.Rank != 4)
            throw new ArgumentException($"Conv2D kernel requires a 4D tensor [out_channels, in_channels, kernel_height, kernel_width]. Got rank {kernel.Rank}.");
        if (output.Rank != 4)
            throw new ArgumentException($"Conv2D output requires a 4D tensor [batch, out_channels, output_height, output_width]. Got rank {output.Rank}.");
        if (stride <= 0) throw new ArgumentException("Stride must be positive.");
        if (dilation <= 0) throw new ArgumentException("Dilation must be positive.");

        // Auto-contiguous: Conv2D accesses .Data which requires contiguous memory.
        // Uses Contiguous() unconditionally to also handle offset views (non-zero _storageOffset
        // with row-major strides passes IsContiguous but still needs materialization for .Data).
        // Zero-copy when already contiguous with offset 0 (Contiguous() returns this).
        input = input.Contiguous();
        kernel = kernel.Contiguous();
        if (!output.IsContiguous)
            throw new InvalidOperationException("Output tensor must be contiguous for Conv2D.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outChannels = kernel._shape[0];
        int kernelInChannels = kernel._shape[1];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        if (inChannels != kernelInChannels)
        {
            throw new ArgumentException(
                $"Input channels ({inChannels}) must match kernel input channels ({kernelInChannels}).");
        }

        int effectiveKernelHeight = dilation * (kernelHeight - 1) + 1;
        int effectiveKernelWidth = dilation * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padding - effectiveKernelHeight) / stride + 1;
        int outputWidth = (width + 2 * padding - effectiveKernelWidth) / stride + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid convolution parameters. Output dimensions would be {outputHeight}x{outputWidth}. " +
                $"Ensure stride={stride}, padding={padding}, dilation={dilation} are compatible with input size {height}x{width} and kernel size {kernelHeight}x{kernelWidth}.");
        }

        // Validate output shape matches expected dimensions
        if (output._shape[0] != batch || output._shape[1] != outChannels ||
            output._shape[2] != outputHeight || output._shape[3] != outputWidth)
        {
            throw new ArgumentException(
                $"Output tensor shape [{string.Join(", ", output._shape)}] doesn't match expected shape [{batch}, {outChannels}, {outputHeight}, {outputWidth}].");
        }

        // Use im2col + GEMM for float (significantly faster)
        if (typeof(T) == typeof(float))
        {
            Conv2DWithIm2ColFloat(
                input as Tensor<float> ?? throw new InvalidCastException(),
                kernel as Tensor<float> ?? throw new InvalidCastException(),
                output as Tensor<float> ?? throw new InvalidCastException(),
                batch, inChannels, height, width,
                outChannels, kernelHeight, kernelWidth,
                stride, padding, dilation,
                outputHeight, outputWidth);
            return;
        }

        // Use im2col + GEMM for double
        if (typeof(T) == typeof(double))
        {
            Conv2DWithIm2ColDouble(
                input as Tensor<double> ?? throw new InvalidCastException(),
                kernel as Tensor<double> ?? throw new InvalidCastException(),
                output as Tensor<double> ?? throw new InvalidCastException(),
                batch, inChannels, height, width,
                outChannels, kernelHeight, kernelWidth,
                stride, padding, dilation,
                outputHeight, outputWidth);
            return;
        }

        // Fallback for other types: naive implementation
        Conv2DNaive(input, kernel, output, numOps,
            batch, inChannels, height, width,
            outChannels, kernelHeight, kernelWidth,
            stride, padding, dilation,
            outputHeight, outputWidth);
    }

    /// <summary>
    /// Optimized Conv2D using multiple strategies for float tensors.
    /// Tries in order: oneDNN, fused im2col-GEMM, Winograd, SIMD direct conv, im2col+GEMM fallback.
    /// BLAS-based approaches are preferred over SIMD direct conv for most sizes.
    /// </summary>
    private void Conv2DWithIm2ColFloat(
        Tensor<float> input,
        Tensor<float> kernel,
        Tensor<float> result,
        int batch,
        int inChannels,
        int height,
        int width,
        int outChannels,
        int kernelHeight,
        int kernelWidth,
        int stride,
        int padding,
        int dilation,
        int outputHeight,
        int outputWidth)
    {
#if !NET471
        // Strategy 1: Try oneDNN for best performance (uses optimized CPU kernels)
        if (TryConv2DOneDnn(input, kernel, result, batch, inChannels, height, width,
            outChannels, kernelHeight, kernelWidth, stride, padding, dilation, outputHeight, outputWidth))
        {
            return;
        }

        // Strategy 2: Try fused im2col-GEMM with cache tiling (BLAS-based, faster for most sizes)
        if (FusedConvHelper.ShouldUseFusedConv(kernelHeight, kernelWidth, stride, stride,
            outputHeight, outputWidth, inChannels, outChannels))
        {
            var inputSpan = input.Data.Span;
            var kernelSpan = kernel.Data.Span;
            var outputSpan = result.Data.Span;

            FusedConvHelper.Conv2DFused(
                inputSpan, kernelSpan, outputSpan,
                batch, inChannels, height, width,
                outChannels, kernelHeight, kernelWidth,
                stride, stride, padding, padding,
                dilation, dilation, outputHeight, outputWidth);
            return;
        }
#endif

        // Strategy 3: Try Winograd for large 3x3 convolutions with stride=1, dilation=1
        if (WinogradHelper.ShouldUseWinograd(kernelHeight, kernelWidth, stride, stride, dilation, dilation, outputHeight, outputWidth))
        {
            var inputSpan = input.Data.Span;
            var kernelSpan = kernel.Data.Span;
            var outputSpan = result.Data.Span;

            WinogradHelper.Conv2DWinograd(
                inputSpan, kernelSpan, outputSpan,
                batch, inChannels, height, width,
                outChannels, padding, padding);
            return;
        }

#if !NET471
        // Strategy 4: Try SIMD direct convolution for small 3x3 kernels with stride=1
        if (TryConv2DSimd(input, kernel, result, batch, inChannels, height, width,
            outChannels, kernelHeight, kernelWidth, stride, padding, dilation, outputHeight, outputWidth))
        {
            return;
        }
#endif

        // Strategy 5: Fallback to im2col + GEMM (works for all cases)
        Conv2DWithIm2ColGemm(input, kernel, result, batch, inChannels, height, width,
            outChannels, kernelHeight, kernelWidth, stride, padding, dilation, outputHeight, outputWidth);
    }

#if !NET471
    /// <summary>
    /// Try Conv2D using Intel oneDNN library.
    /// </summary>
    private unsafe bool TryConv2DOneDnn(
        Tensor<float> input, Tensor<float> kernel, Tensor<float> result,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelHeight, int kernelWidth,
        int stride, int padding, int dilation, int outputHeight, int outputWidth)
    {
        if (!OneDnnProvider.IsAvailable)
        {
            return false;
        }

        var inputSpan = input.Data.Span;
        var kernelSpan = kernel.Data.Span;
        var outputSpan = result.Data.Span;

        fixed (float* inputPtr = inputSpan)
        fixed (float* kernelPtr = kernelSpan)
        fixed (float* outputPtr = outputSpan)
        {
            return OneDnnProvider.TryConv2D(
                inputPtr, batch, inChannels, height, width,
                kernelPtr, outChannels, kernelHeight, kernelWidth,
                outputPtr, outputHeight, outputWidth,
                stride, stride, padding, padding, dilation, dilation);
        }
    }

    /// <summary>
    /// Try Conv2D using SIMD-optimized direct convolution.
    /// </summary>
    private unsafe bool TryConv2DSimd(
        Tensor<float> input, Tensor<float> kernel, Tensor<float> result,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelHeight, int kernelWidth,
        int stride, int padding, int dilation, int outputHeight, int outputWidth)
    {
        if (!SimdConvHelper.CanUseSimdConv(kernelHeight, kernelWidth, stride, stride))
        {
            return false;
        }

        var inputSpan = input.Data.Span;
        var kernelSpan = kernel.Data.Span;
        var outputSpan = result.Data.Span;

        fixed (float* inputPtr = inputSpan)
        fixed (float* kernelPtr = kernelSpan)
        fixed (float* outputPtr = outputSpan)
        {
            if (kernelHeight == 3 && kernelWidth == 3)
            {
                SimdConvHelper.Conv3x3Stride1(
                    inputPtr, kernelPtr, outputPtr,
                    batch, inChannels, height, width,
                    outChannels, padding, padding, dilation, dilation);
                return true;
            }
            else if (kernelHeight == 1 && kernelWidth == 1)
            {
                SimdConvHelper.Conv1x1(
                    inputPtr, kernelPtr, outputPtr,
                    batch, inChannels, height, width, outChannels);
                return true;
            }
        }

        return false;
    }
#endif

    /// <summary>
    /// Conv2D using im2col + GEMM (fallback implementation).
    /// </summary>
    private void Conv2DWithIm2ColGemm(
        Tensor<float> input, Tensor<float> kernel, Tensor<float> result,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelHeight, int kernelWidth,
        int stride, int padding, int dilation, int outputHeight, int outputWidth)
    {
        int colH = inChannels * kernelHeight * kernelWidth;
        int colW = outputHeight * outputWidth;
        // Allocate only one batch-slice worth of im2col buffer, not batch * colH * colW
        int sliceSize = colH * colW;

        var inputSpan = input.Data.Span;
        var kernelSpan = kernel.Data.Span;
        var outputSpan = result.Data.Span;
        int inputSliceSize = inChannels * height * width;

#if !NET471
        // Use native memory for im2col buffer to reduce GC pressure
        using var im2colBuffer = new NativeBuffer<float>(sliceSize);
        var im2colSpan = im2colBuffer.Span;
#else
        // Fallback to ArrayPool for .NET Framework
        var pool = System.Buffers.ArrayPool<float>.Shared;
        float[] im2colArray = pool.Rent(sliceSize);
        try
        {
        var im2colSpan = im2colArray.AsSpan(0, sliceSize);
#endif

        for (int b = 0; b < batch; b++)
        {
            // Step 1: im2col for this batch slice only
            Helpers.Im2ColHelper.Im2Col(
                inputSpan.Slice(b * inputSliceSize, inputSliceSize), im2colSpan,
                1, inChannels, height, width,
                kernelHeight, kernelWidth, stride, stride, padding, padding, dilation, dilation);

            // Step 2: GEMM for this batch
            int outputOffset = b * outChannels * colW;

            bool usedBlas = Helpers.BlasProvider.TryGemm(
                outChannels, colW, colH,
                kernelSpan.Slice(0, outChannels * colH),
                colH,
                im2colSpan.Slice(0, sliceSize),
                colW,
                outputSpan.Slice(outputOffset, outChannels * colW),
                colW);

            if (!usedBlas)
            {
                MultiplyMatrixBlockedFloat(
                    kernelSpan,
                    im2colSpan.Slice(0, sliceSize),
                    outputSpan.Slice(outputOffset, outChannels * colW),
                    outChannels, colH, colW);
            }
        }

#if NET471
        }
        finally
        {
            pool.Return(im2colArray);
        }
#endif
    }

    /// <summary>
    /// Performs Conv2D for double precision tensors using im2col + GEMM.
    /// Same approach as the float version but uses double BLAS routines.
    /// This gives 10-100x speedup over the naive 6-nested-loop implementation.
    /// </summary>
    private void Conv2DWithIm2ColDouble(
        Tensor<double> input, Tensor<double> kernel, Tensor<double> result,
        int batch, int inChannels, int height, int width,
        int outChannels, int kernelHeight, int kernelWidth,
        int stride, int padding, int dilation, int outputHeight, int outputWidth)
    {
        int colH = inChannels * kernelHeight * kernelWidth;
        int colW = outputHeight * outputWidth;
        // Allocate only one batch-slice worth of im2col buffer, not batch * colH * colW
        int sliceSize = colH * colW;

        var inputSpan = input.Data.Span;
        var kernelSpan = kernel.Data.Span;
        var outputSpan = result.Data.Span;
        int inputSliceSize = inChannels * height * width;

#if !NET471
        using var im2colBuffer = new NativeBuffer<double>(sliceSize);
        var im2colSpan = im2colBuffer.Span;
#else
        var pool = System.Buffers.ArrayPool<double>.Shared;
        double[] im2colArray = pool.Rent(sliceSize);
        try
        {
        var im2colSpan = im2colArray.AsSpan(0, sliceSize);
#endif

        for (int b = 0; b < batch; b++)
        {
            // Step 1: im2col for this batch slice only
            Helpers.Im2ColHelper.Im2Col(
                inputSpan.Slice(b * inputSliceSize, inputSliceSize), im2colSpan,
                1, inChannels, height, width,
                kernelHeight, kernelWidth, stride, stride, padding, padding, dilation, dilation);

            // Step 2: GEMM for this batch
            int outputOffset = b * outChannels * colW;

            bool usedBlas = Helpers.BlasProvider.TryGemm(
                outChannels, colW, colH,
                kernelSpan.Slice(0, outChannels * colH),
                colH,
                im2colSpan.Slice(0, sliceSize),
                colW,
                outputSpan.Slice(outputOffset, outChannels * colW),
                colW);

            if (!usedBlas)
            {
                Helpers.Im2ColHelper.MultiplyMatrixBlockedDouble(
                    kernelSpan.Slice(0, outChannels * colH),
                    im2colSpan.Slice(0, sliceSize),
                    outputSpan.Slice(outputOffset, outChannels * colW),
                    outChannels, colH, colW);
            }
        }

#if NET471
        }
        finally
        {
            pool.Return(im2colArray);
        }
#endif
    }

    /// <summary>
    /// High-performance matrix multiplication for float: C = A @ B.
    /// Delegates to SimdGemm which uses tiled BLIS-style architecture with FMA micro-kernel,
    /// panel packing, and cache-level blocking for maximum throughput.
    /// </summary>
    private static void MultiplyMatrixBlockedFloat(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n)
    {
        Simd.SimdGemm.Sgemm(a, b, c, m, k, n);
    }

    /// <summary>
    /// Naive Conv2D implementation for non-float types.
    /// </summary>
    private static void Conv2DNaive<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> result,
        INumericOperations<T> numOps,
        int batch,
        int inChannels,
        int height,
        int width,
        int outChannels,
        int kernelHeight,
        int kernelWidth,
        int stride,
        int padding,
        int dilation,
        int outputHeight,
        int outputWidth)
    {
        // Use parallel processing for batch and output channels
        bool useParallel = batch * outChannels >= 4 && CpuParallelSettings.MaxDegreeOfParallelism > 1;

        if (useParallel)
        {
            System.Threading.Tasks.Parallel.For(0, batch * outChannels, idx =>
            {
                int b = idx / outChannels;
                int oc = idx % outChannels;
                ProcessOutputChannel(input, kernel, result, numOps,
                    b, oc, inChannels, height, width,
                    kernelHeight, kernelWidth, stride, padding, dilation,
                    outputHeight, outputWidth);
            });
        }
        else
        {
            for (int b = 0; b < batch; b++)
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    ProcessOutputChannel(input, kernel, result, numOps,
                        b, oc, inChannels, height, width,
                        kernelHeight, kernelWidth, stride, padding, dilation,
                        outputHeight, outputWidth);
                }
            }
        }
    }

    private static void ProcessOutputChannel<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> result,
        INumericOperations<T> numOps,
        int b,
        int oc,
        int inChannels,
        int height,
        int width,
        int kernelHeight,
        int kernelWidth,
        int stride,
        int padding,
        int dilation,
        int outputHeight,
        int outputWidth)
    {
        for (int oh = 0; oh < outputHeight; oh++)
        {
            for (int ow = 0; ow < outputWidth; ow++)
            {
                T sum = numOps.Zero;

                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            int ih = oh * stride + kh * dilation - padding;
                            int iw = ow * stride + kw * dilation - padding;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                            {
                                T inputVal = input[b, ic, ih, iw];
                                T kernelVal = kernel[oc, ic, kh, kw];
                                sum = numOps.Add(sum, numOps.Multiply(inputVal, kernelVal));
                            }
                        }
                    }
                }

                result[b, oc, oh, ow] = sum;
            }
        }
    }

    #endregion

    #region Activation Functions

    public Vector<T> Tanh<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized Tanh (3-6x speedup for float)
        return TensorPrimitivesHelper<T>.Tanh(vector);
    }

    public Vector<T> Sigmoid<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        // Use SIMD-optimized Sigmoid (3-6x speedup for float)
        return TensorPrimitivesHelper<T>.Sigmoid(vector);
    }

    public Vector<T> ReLU<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(vector.Length);
        numOps.ReLU(vector.AsSpan(), result.AsWritableSpan());

        return result;
    }

    public unsafe Tensor<T> Tanh<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Stride-aware: strided scalar loop for small views, Contiguous()+SIMD for large
        if (!tensor.IsContiguous)
        {
            if (tensor.Length <= 4096)
            {
                var ops = MathHelper.GetNumericOperations<T>();
                var resultS = TensorAllocator.RentUninitialized<T>(tensor._shape);
                var srcRaw = tensor._storage.GetDataArray();
                var dstArr = resultS.GetDataArray();
                var idxBuf = new int[tensor.Length];
                tensor.FillStorageIndices(idxBuf);
                for (int i = 0; i < tensor.Length; i++)
                    { T v = srcRaw[idxBuf[i]]; T e2v = ops.Exp(ops.Multiply(ops.FromDouble(2.0), v)); dstArr[i] = ops.Divide(ops.Subtract(e2v, ops.One), ops.Add(e2v, ops.One)); }
                DifferentiableOps.RecordUnary("Tanh", resultS, tensor, BackwardFunctions<T>.TanhBackward);
                return resultS;
            }
            tensor = tensor.Contiguous();
        }

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);

        if (typeof(T) == typeof(float))
        {
            var srcMem = AsFloatMemory(tensor.Data);
            var dstMem = AsFloatMemory(result.Data);
            using var pinSrc = srcMem.Pin();
            using var pinDst = dstMem.Pin();
            float* pSrc = (float*)pinSrc.Pointer;
            float* pDst = (float*)pinDst.Pointer;
            ParallelComputeBound(pSrc, pDst, tensor.Length, SimdKernels.TanhUnsafe);
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Tanh(tensor.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordUnary("Tanh", result, tensor, BackwardFunctions<T>.TanhBackward);
        return result;
    }

    public virtual unsafe Tensor<T> Sigmoid<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Stride-aware: strided scalar loop for small views, Contiguous()+SIMD for large
        if (!tensor.IsContiguous)
        {
            if (tensor.Length <= 4096)
            {
                var ops = MathHelper.GetNumericOperations<T>();
                var resultS = TensorAllocator.RentUninitialized<T>(tensor._shape);
                var srcRaw = tensor._storage.GetDataArray();
                var dstArr = resultS.GetDataArray();
                var idxBuf = new int[tensor.Length];
                tensor.FillStorageIndices(idxBuf);
                for (int i = 0; i < tensor.Length; i++)
                    { T v = srcRaw[idxBuf[i]]; T negV = ops.Negate(v); dstArr[i] = ops.Divide(ops.One, ops.Add(ops.One, ops.Exp(negV))); }
                DifferentiableOps.RecordUnary("Sigmoid", resultS, tensor, BackwardFunctions<T>.SigmoidBackward);
                return resultS;
            }
            tensor = tensor.Contiguous();
        }

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        int length = tensor.Length;

        // Fast path for float tensors: bypass generic dispatch + Span bounds-checking
        // Use Memory<T>.Pin() directly — avoids GetDataArray() which can copy
        if (typeof(T) == typeof(float))
        {
            var srcMem = AsFloatMemory(tensor.Data);
            var dstMem = AsFloatMemory(result.Data);
            using var pinSrc = srcMem.Pin();
            using var pinDst = dstMem.Pin();
            float* pSrc = (float*)pinSrc.Pointer;
            float* pDst = (float*)pinDst.Pointer;

            // JIT-compiled sigmoid: constants baked in data section, 4x unrolled
            if (CpuJitSelfTest.IsVerified && length >= 64)
            {
                int sigChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 250_000));
                if (sigChunks >= 2)
                {
                    int chunkSize = (length + sigChunks - 1) / sigChunks;
                    chunkSize = (chunkSize + 31) & ~31;

                    Parallel.For(0, sigChunks, chunk =>
                    {
                        int start = chunk * chunkSize;
                        int count = Math.Min(chunkSize, length - start);
                        if (count > 0)
                        {
                            int jitCount = count & ~7;
                            if (jitCount > 0)
                            {
                                var kernel = CpuJitKernels.GetSigmoidKernel(jitCount);
                                kernel(pSrc + start, pDst + start, jitCount);
                            }
                            for (int ti = jitCount; ti < count; ti++)
                            {
                                pDst[start + ti] = 1.0f / (1.0f + MathF.Exp(-pSrc[start + ti]));
                            }
                        }
                    });
                }
                else
                {
                    // JIT kernel processes 8-wide SIMD chunks; handle tail with scalar fallback
                    int jitLen = length & ~7;
                    if (jitLen > 0)
                    {
                        var kernel = CpuJitKernels.GetSigmoidKernel(jitLen);
                        kernel(pSrc, pDst, jitLen);
                    }
                    for (int i = jitLen; i < length; i++)
                    {
                        pDst[i] = 1.0f / (1.0f + MathF.Exp(-pSrc[i]));
                    }
                }
                DifferentiableOps.RecordUnary("Sigmoid", result, tensor, BackwardFunctions<T>.SigmoidBackward);
                return result;
            }

            // SIMD fallback
            int fallbackChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 250_000));
            if (fallbackChunks >= 2)
            {
                int chunkSize = (length + fallbackChunks - 1) / fallbackChunks;
                chunkSize = (chunkSize + 31) & ~31;

                Parallel.For(0, fallbackChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                    {
                        SimdKernels.SigmoidUnsafe(pSrc + start, pDst + start, count);
                    }
                });
            }
            else
            {
                SimdKernels.SigmoidUnsafe(pSrc, pDst, length);
            }
            DifferentiableOps.RecordUnary("Sigmoid", result, tensor, BackwardFunctions<T>.SigmoidBackward);
            return result;
        }

        // Double fast path: pointer-based SIMD Sigmoid with polynomial approximation
        if (typeof(T) == typeof(double))
        {
            var srcMem = AsDoubleMemory(tensor.Data);
            var dstMem = AsDoubleMemory(result.Data);
            using var pinSrc = srcMem.Pin();
            using var pinDst = dstMem.Pin();
            double* pSrc = (double*)pinSrc.Pointer;
            double* pDst = (double*)pinDst.Pointer;

            // Parallel chunking for compute-bound sigmoid (lower threshold than bandwidth-bound)
            int sigChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 64_000));
            if (sigChunks >= 2)
            {
                int chunkSize = (length + sigChunks - 1) / sigChunks;
                chunkSize = (chunkSize + 15) & ~15;
                IntPtr ipSrc = (IntPtr)pSrc;
                IntPtr ipDst = (IntPtr)pDst;
                int len = length;

                Parallel.For(0, sigChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, len - start);
                    if (count > 0)
                    {
                        SimdKernels.SigmoidUnsafe((double*)ipSrc + start, (double*)ipDst + start, count);
                    }
                });
            }
            else
            {
                SimdKernels.SigmoidUnsafe(pSrc, pDst, length);
            }
            DifferentiableOps.RecordUnary("Sigmoid", result, tensor, BackwardFunctions<T>.SigmoidBackward);
            return result;
        }

        // Generic fallback
        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Sigmoid(tensor.AsSpan(), result.AsWritableSpan());

        DifferentiableOps.RecordUnary("Sigmoid", result, tensor, BackwardFunctions<T>.SigmoidBackward);
        return result;
    }

    /// <summary>
    /// Applies Sigmoid activation in-place: x = 1 / (1 + exp(-x)). Zero allocation.
    /// Uses oneDNN for float tensors when available, otherwise falls back to SIMD.
    /// </summary>
#if !NETFRAMEWORK
    public virtual unsafe void SigmoidInPlace<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        Tensor<T>? savedInput = null;
        var sigTape = GradientTape<T>.Current;
        if (sigTape is not null && sigTape.Options.RecordInPlace)
            savedInput = tensor.Clone();

        tensor.IncrementVersion();

        // Try oneDNN for float tensors
        if (typeof(T) == typeof(float) && OneDnnProvider.IsAvailable)
        {
            var floatMem = AsFloatMemory(tensor.Data);
            if (MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)floatMem, out var segment) &&
                segment.Array is not null && segment.Offset == 0)
            {
                fixed (float* ptr = segment.Array)
                {
                    if (OneDnnProvider.TrySigmoid(ptr, tensor.Length))
                    {
                        if (savedInput is not null) DifferentiableOps.RecordUnary("SigmoidInPlace", tensor, savedInput, BackwardFunctions<T>.SigmoidBackward);
                        return;
                    }
                }
            }
        }

        SigmoidParallel(tensor);
        if (savedInput is not null) DifferentiableOps.RecordUnary("SigmoidInPlace", tensor, savedInput, BackwardFunctions<T>.SigmoidBackward);
    }
#else
    public virtual void SigmoidInPlace<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        Tensor<T>? savedInput = null;
        var sigTape = GradientTape<T>.Current;
        if (sigTape is not null && sigTape.Options.RecordInPlace)
            savedInput = tensor.Clone();

        tensor.IncrementVersion();

        SigmoidParallel(tensor);
        if (savedInput is not null) DifferentiableOps.RecordUnary("SigmoidInPlace", tensor, savedInput, BackwardFunctions<T>.SigmoidBackward);
    }
#endif

    private unsafe void SigmoidParallel<T>(Tensor<T> tensor)
    {
        int length = tensor.Length;

        // Fast path for float tensors: bypass generic dispatch + Span bounds-checking
        // Use Memory<T>.Pin() directly — avoids GetDataArray() which can copy (breaking in-place writes)
        if (typeof(T) == typeof(float))
        {
            var mem = AsFloatMemory(tensor.Data);
            using var pin = mem.Pin();
            float* p = (float*)pin.Pointer;

            // JIT-compiled sigmoid in-place: constants baked in data section
            if (CpuJitSelfTest.IsVerified && length >= 64)
            {
                int jitChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 250_000));
                if (jitChunks >= 2)
                {
                    int chunkSize = (length + jitChunks - 1) / jitChunks;
                    chunkSize = (chunkSize + 31) & ~31;

                    Parallel.For(0, jitChunks, chunk =>
                    {
                        int start = chunk * chunkSize;
                        int count = Math.Min(chunkSize, length - start);
                        if (count > 0)
                        {
                            int jitCount = count & ~7;
                            if (jitCount > 0)
                            {
                                var kernel = CpuJitKernels.GetSigmoidKernel(jitCount);
                                kernel(p + start, p + start, jitCount);
                            }
                            for (int ti = jitCount; ti < count; ti++)
                            {
                                float* ptr = p + start + ti;
                                *ptr = 1.0f / (1.0f + MathF.Exp(-*ptr));
                            }
                        }
                    });
                }
                else
                {
                    int jitLen = length & ~7;
                    if (jitLen > 0)
                    {
                        var kernel = CpuJitKernels.GetSigmoidKernel(jitLen);
                        kernel(p, p, jitLen);
                    }
                    for (int i = jitLen; i < length; i++)
                    {
                        p[i] = 1.0f / (1.0f + MathF.Exp(-p[i]));
                    }
                }
                return;
            }

            // SIMD fallback
            int sigChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 250_000));
            if (sigChunks >= 2)
            {
                int chunkSize = (length + sigChunks - 1) / sigChunks;
                chunkSize = (chunkSize + 31) & ~31;

                Parallel.For(0, sigChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                    {
                        SimdKernels.SigmoidUnsafe(p + start, p + start, count);
                    }
                });
            }
            else
            {
                SimdKernels.SigmoidUnsafe(p, p, length);
            }
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Sigmoid(tensor.AsSpan(), tensor.AsWritableSpan());
    }

    /// <summary>
    /// Applies Sigmoid activation to input, storing in destination. Zero allocation.
    /// </summary>
    public void SigmoidInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!ShapesMatch(destination._shape, input._shape))
        {
            throw new ArgumentException("Tensor shapes must match.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.Sigmoid(input.AsSpan(), destination.AsWritableSpan());
    }

    public virtual unsafe Tensor<T> ReLU<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Stride-aware: strided scalar loop for small views, Contiguous()+SIMD for large
        if (!tensor.IsContiguous)
        {
            if (tensor.Length <= 4096)
            {
                var ops = MathHelper.GetNumericOperations<T>();
                var resultS = TensorAllocator.RentUninitialized<T>(tensor._shape);
                var srcRaw = tensor._storage.GetDataArray();
                var dstArr = resultS.GetDataArray();
                var idxBuf = new int[tensor.Length];
                tensor.FillStorageIndices(idxBuf);
                for (int i = 0; i < tensor.Length; i++)
                    { T v = srcRaw[idxBuf[i]]; dstArr[i] = ops.GreaterThan(v, ops.Zero) ? v : ops.Zero; }
                DifferentiableOps.RecordUnary("ReLU", resultS, tensor, BackwardFunctions<T>.ReLUBackward);
                return resultS;
            }
            tensor = tensor.Contiguous();
        }

        // RentUninitialized — ReLU writes every element, no need to zero
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        int length = tensor.Length;

        if (typeof(T) == typeof(float))
        {
            var srcArr = (float[])(object)tensor.GetDataArray();
            var dstArr = (float[])(object)result.GetDataArray();

            int reluChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 2_000_000));
            if (reluChunks >= 2)
            {
                // Parallel path: must use Pin for lambda capture
                var srcMem = AsFloatMemory(tensor.Data);
                var dstMem = AsFloatMemory(result.Data);
                using var pinSrc = srcMem.Pin();
                using var pinDst = dstMem.Pin();
                float* pSrcP = (float*)pinSrc.Pointer;
                float* pDstP = (float*)pinDst.Pointer;
                int chunkSize = (length + reluChunks - 1) / reluChunks;
                chunkSize = (chunkSize + 31) & ~31;
                Parallel.For(0, reluChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                        SimdKernels.ReLUUnsafe(pSrcP + start, pDstP + start, count);
                });
            }
            else
            {
                // Single-threaded: fixed avoids GCHandle allocation overhead
                fixed (float* pSrc = srcArr, pDst = dstArr)
                {
                    if (CpuJitSelfTest.IsVerified && length >= 64)
                        JitUnaryDispatch(pSrc, pDst, length);
                    else
                        SimdKernels.ReLUUnsafe(pSrc, pDst, length);
                }
            }
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.ReLU(tensor.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordUnary("ReLU", result, tensor, BackwardFunctions<T>.ReLUBackward);
        return result;
    }

    /// <summary>
    /// Applies ReLU activation in-place: x = max(0, x). Zero allocation.
    /// Uses parallel SIMD for large float tensors, oneDNN when available.
    /// </summary>
#if !NETFRAMEWORK
    public virtual unsafe void ReLUInPlace<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        Tensor<T>? savedInput = null;
        var reluTape = GradientTape<T>.Current;
        if (reluTape is not null && reluTape.Options.RecordInPlace)
            savedInput = tensor.Clone();

        tensor.IncrementVersion();

        // Try oneDNN for float tensors
        if (typeof(T) == typeof(float) && OneDnnProvider.IsAvailable)
        {
            var floatMem = AsFloatMemory(tensor.Data);
            if (MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)floatMem, out var segment) &&
                segment.Array is not null && segment.Offset == 0)
            {
                fixed (float* ptr = segment.Array)
                {
                    if (OneDnnProvider.TryReLU(ptr, tensor.Length))
                    {
                        if (savedInput is not null) DifferentiableOps.RecordUnary("ReLUInPlace", tensor, savedInput, BackwardFunctions<T>.ReLUBackward);
                        return;
                    }
                }
            }
        }

        ReLUParallel(tensor);
        if (savedInput is not null) DifferentiableOps.RecordUnary("ReLUInPlace", tensor, savedInput, BackwardFunctions<T>.ReLUBackward);
    }
#else
    public virtual void ReLUInPlace<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        Tensor<T>? savedInput = null;
        var reluTape = GradientTape<T>.Current;
        if (reluTape is not null && reluTape.Options.RecordInPlace)
            savedInput = tensor.Clone();

        tensor.IncrementVersion();

        ReLUParallel(tensor);
        if (savedInput is not null) DifferentiableOps.RecordUnary("ReLUInPlace", tensor, savedInput, BackwardFunctions<T>.ReLUBackward);
    }
#endif

    private unsafe void ReLUParallel<T>(Tensor<T> tensor)
    {
        int length = tensor.Length;

        // Fast path for float tensors: bypass generic dispatch + Span bounds-checking
        // Use Memory<T>.Pin() directly — avoids GetDataArray() which can copy (breaking in-place writes)
        if (typeof(T) == typeof(float))
        {
            var mem = AsFloatMemory(tensor.Data);
            using var pin = mem.Pin();
            float* p = (float*)pin.Pointer;

            // Bandwidth-bound: parallel only helps above ~2M elements
            int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 2_000_000));
            if (numChunks >= 2)
            {
                int chunkSize = (length + numChunks - 1) / numChunks;
                chunkSize = (chunkSize + 31) & ~31;

                Parallel.For(0, numChunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, length - start);
                    if (count > 0)
                    {
                        SimdKernels.ReLUUnsafe(p + start, p + start, count);
                    }
                });
            }
            else
            {
                SimdKernels.ReLUUnsafe(p, p, length);
            }
            return;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.ReLU(tensor.AsSpan(), tensor.AsWritableSpan());
    }

    /// <summary>
    /// Applies ReLU activation to input, storing in destination. Zero allocation.
    /// </summary>
    public void ReLUInto<T>(Tensor<T> destination, Tensor<T> input)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        if (!ShapesMatch(destination._shape, input._shape))
        {
            throw new ArgumentException("Tensor shapes must match.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        numOps.ReLU(input.AsSpan(), destination.AsWritableSpan());
    }

    public Vector<T> GELU<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return TensorPrimitivesHelper<T>.GELU(vector);
    }

    public Vector<T> Mish<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return TensorPrimitivesHelper<T>.Mish(vector);
    }

    public Vector<T> Swish<T>(Vector<T> vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return TensorPrimitivesHelper<T>.Swish(vector);
    }

    public Vector<T> ELU<T>(Vector<T> vector, double alpha = 1.0)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));

        return TensorPrimitivesHelper<T>.ELU(vector, alpha);
    }

    public virtual unsafe Tensor<T> GELU<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Stride-aware: strided scalar loop for small views, Contiguous()+SIMD for large
        if (!tensor.IsContiguous)
        {
            if (tensor.Length <= 4096)
            {
                var ops = MathHelper.GetNumericOperations<T>();
                var resultS = TensorAllocator.RentUninitialized<T>(tensor._shape);
                var srcRaw = tensor._storage.GetDataArray();
                var dstArr = resultS.GetDataArray();
                var idxBuf = new int[tensor.Length];
                tensor.FillStorageIndices(idxBuf);
                for (int i = 0; i < tensor.Length; i++)
                    { T v = srcRaw[idxBuf[i]]; T z = ops.Multiply(ops.FromDouble(0.7978845608), ops.Add(v, ops.Multiply(ops.FromDouble(0.044715), ops.Multiply(v, ops.Multiply(v, v))))); T e2z = ops.Exp(ops.Multiply(ops.FromDouble(2.0), z)); T th = ops.Divide(ops.Subtract(e2z, ops.One), ops.Add(e2z, ops.One)); T cdf = ops.Divide(ops.Add(ops.One, th), ops.FromDouble(2.0)); dstArr[i] = ops.Multiply(v, cdf); }
                DifferentiableOps.RecordUnary("GELU", resultS, tensor, BackwardFunctions<T>.GELUBackward);
                return resultS;
            }
            tensor = tensor.Contiguous();
        }

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);

        if (typeof(T) == typeof(float))
        {
            var srcMem = AsFloatMemory(tensor.Data);
            var dstMem = AsFloatMemory(result.Data);
            using var pinSrc = srcMem.Pin();
            using var pinDst = dstMem.Pin();
            float* pSrc = (float*)pinSrc.Pointer;
            float* pDst = (float*)pinDst.Pointer;
            ParallelComputeBound(pSrc, pDst, tensor.Length, SimdKernels.GELUUnsafe);
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.GELU(tensor.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordUnary("GELU", result, tensor, BackwardFunctions<T>.GELUBackward);
        return result;
    }

    public virtual unsafe Tensor<T> Mish<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous)
        {
            if (tensor.Length <= 4096)
            {
                var ops = MathHelper.GetNumericOperations<T>();
                var resultS = TensorAllocator.RentUninitialized<T>(tensor._shape);
                var srcRaw = tensor._storage.GetDataArray();
                var dstArr = resultS.GetDataArray();
                var idxBuf = new int[tensor.Length];
                tensor.FillStorageIndices(idxBuf);
                for (int i = 0; i < tensor.Length; i++)
                    { T v = srcRaw[idxBuf[i]]; T sp = ops.Log(ops.Add(ops.One, ops.Exp(v))); T e2sp = ops.Exp(ops.Multiply(ops.FromDouble(2.0), sp)); T th = ops.Divide(ops.Subtract(e2sp, ops.One), ops.Add(e2sp, ops.One)); dstArr[i] = ops.Multiply(v, th); }
                DifferentiableOps.RecordUnary("Mish", resultS, tensor, BackwardFunctions<T>.MishBackward);
                return resultS;
            }
            return Mish(tensor.Contiguous());
        }

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);

        if (typeof(T) == typeof(float))
        {
            var srcMem = AsFloatMemory(tensor.Data);
            var dstMem = AsFloatMemory(result.Data);
            using var pinSrc = srcMem.Pin();
            using var pinDst = dstMem.Pin();
            float* pSrc = (float*)pinSrc.Pointer;
            float* pDst = (float*)pinDst.Pointer;
            ParallelComputeBound(pSrc, pDst, tensor.Length, SimdKernels.MishUnsafe);
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.Mish(tensor.AsSpan(), result.AsWritableSpan());
        }

        DifferentiableOps.RecordUnary("Mish", result, tensor, BackwardFunctions<T>.MishBackward);
        return result;
    }

    public virtual Tensor<T> Swish<T>(Tensor<T> tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous)
        {
            if (tensor.Length <= 4096)
            {
                var ops2 = MathHelper.GetNumericOperations<T>();
                var r = TensorAllocator.RentUninitialized<T>(tensor._shape);
                var src = tensor._storage.GetDataArray(); var dst = r.GetDataArray();
                var idx = new int[tensor.Length]; tensor.FillStorageIndices(idx);
                for (int i = 0; i < tensor.Length; i++) { T v = src[idx[i]]; T negV = ops2.Negate(v); T sig = ops2.Divide(ops2.One, ops2.Add(ops2.One, ops2.Exp(negV))); dst[i] = ops2.Multiply(v, sig); }
                DifferentiableOps.RecordUnary("Swish", r, tensor, BackwardFunctions<T>.SwishBackward);
                return r;
            }
            tensor = tensor.Contiguous();
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);

        numOps.Swish(tensor.AsSpan(), result.AsWritableSpan());

        DifferentiableOps.RecordUnary("Swish", result, tensor, BackwardFunctions<T>.SwishBackward);
        return result;
    }

    public virtual Tensor<T> ELU<T>(Tensor<T> tensor, double alpha = 1.0)
    {
        if (!tensor.IsContiguous)
        {
            if (tensor.Length <= 4096)
            {
                var ops2 = MathHelper.GetNumericOperations<T>();
                var r = TensorAllocator.RentUninitialized<T>(tensor._shape);
                var src = tensor._storage.GetDataArray(); var dst = r.GetDataArray();
                var idx = new int[tensor.Length]; tensor.FillStorageIndices(idx);
                T alphaT = ops2.FromDouble(alpha);
                for (int i = 0; i < tensor.Length; i++)
                {
                    T v = src[idx[i]];
                    dst[i] = ops2.GreaterThan(v, ops2.Zero) ? v : ops2.Multiply(alphaT, ops2.Subtract(ops2.Exp(v), ops2.One));
                }
                DifferentiableOps.RecordUnary("ELU", r, tensor, BackwardFunctions<T>.ELUBackward, new object[] { alpha });
                return r;
            }
            tensor = tensor.Contiguous();
        }
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);

        numOps.ELU(tensor.AsSpan(), numOps.FromDouble(alpha), result.AsWritableSpan());

        DifferentiableOps.RecordUnary("ELU", result, tensor, BackwardFunctions<T>.ELUBackward, new object[] { alpha });
        return result;
    }

    /// <inheritdoc/>
    public virtual unsafe Tensor<T> LeakyReLU<T>(Tensor<T> tensor, T alpha)
    {
        if (!tensor.IsContiguous)
        {
            if (tensor.Length <= 4096)
            {
                var ops2 = MathHelper.GetNumericOperations<T>();
                var r = TensorAllocator.RentUninitialized<T>(tensor._shape);
                var src = tensor._storage.GetDataArray(); var dst = r.GetDataArray();
                var idx = new int[tensor.Length]; tensor.FillStorageIndices(idx);
                for (int i = 0; i < tensor.Length; i++)
                {
                    T v = src[idx[i]];
                    dst[i] = ops2.GreaterThan(v, ops2.Zero) ? v : ops2.Multiply(alpha, v);
                }
                DifferentiableOps.RecordUnary("LeakyReLU", r, tensor, BackwardFunctions<T>.LeakyReLUBackward, new object[] { MathHelper.GetNumericOperations<T>().ToDouble(alpha) });
                return r;
            }
            tensor = tensor.Contiguous();
        }
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var negSlope = MathHelper.GetNumericOperations<T>().ToDouble(alpha);

        if (typeof(T) == typeof(float))
        {
            float alphaF = System.Runtime.CompilerServices.Unsafe.As<T, float>(ref alpha);
            var srcMem = AsFloatMemory(tensor.Data);
            var dstMem = AsFloatMemory(result.Data);
            using var pinSrc = srcMem.Pin();
            using var pinDst = dstMem.Pin();
            float* pSrc = (float*)pinSrc.Pointer;
            float* pDst = (float*)pinDst.Pointer;
            int len = tensor.Length;
            const int parallelThreshold = 256 * 1024;
            int maxThreads = CpuParallelSettings.MaxDegreeOfParallelism;
            int chunks = Math.Min(maxThreads, Math.Max(1, len / parallelThreshold));
            if (chunks >= 2)
            {
                int chunkSize = (len + chunks - 1) / chunks;
                chunkSize = (chunkSize + 31) & ~31;
                IntPtr pIn = (IntPtr)pSrc;
                IntPtr pOut = (IntPtr)pDst;
                Parallel.For(0, chunks, chunk =>
                {
                    int start = chunk * chunkSize;
                    int count = Math.Min(chunkSize, len - start);
                    if (count > 0)
                        SimdKernels.LeakyReLUUnsafe((float*)pIn + start, (float*)pOut + start, count, alphaF);
                });
            }
            else
            {
                SimdKernels.LeakyReLUUnsafe(pSrc, pDst, len, alphaF);
            }
        }
        else
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            numOps.LeakyReLU(tensor.AsSpan(), alpha, result.AsWritableSpan());
        }

        DifferentiableOps.RecordUnary("LeakyReLU", result, tensor, BackwardFunctions<T>.LeakyReLUBackward, new object[] { negSlope });
        return result;
    }

    /// <summary>
    /// Gated Linear Unit: GLU(a, b) = a * sigmoid(b)
    /// Splits input in half along specified dimension and applies sigmoid gating.
    /// From "Language Modeling with Gated Convolutional Networks" (Dauphin et al., 2017)
    /// </summary>
    public Tensor<T> GLU<T>(Tensor<T> input, int dim = -1)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Handle negative dimension
        int actualDim = dim < 0 ? input.Rank + dim : dim;
        if (actualDim < 0 || actualDim >= input.Rank)
            throw new ArgumentException($"Invalid dimension {dim} for tensor with rank {input.Rank}");

        int dimSize = input._shape[actualDim];
        if (dimSize % 2 != 0)
            throw new ArgumentException($"Dimension {actualDim} must have even size for GLU, got {dimSize}");

        int halfSize = dimSize / 2;

        // Calculate output shape
        var outputShape = input.Shape.ToArray();
        outputShape[actualDim] = halfSize;

        var inputData = input.GetFlattenedData();
        var outputData = new T[input.Length / 2];

        // Calculate strides for the dimension
        int innerSize = 1;
        for (int i = actualDim + 1; i < input.Rank; i++)
            innerSize *= input._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= input._shape[i];

        // Apply GLU: output = a * sigmoid(b)
        Parallel.For(0, outerSize, outer =>
        {
            for (int h = 0; h < halfSize; h++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int aIdx = outer * dimSize * innerSize + h * innerSize + inner;
                    int bIdx = outer * dimSize * innerSize + (h + halfSize) * innerSize + inner;
                    int outIdx = outer * halfSize * innerSize + h * innerSize + inner;

                    T a = inputData[aIdx];
                    T b = inputData[bIdx];
                    T sigmoid_b = numOps.Divide(numOps.One, numOps.Add(numOps.One, numOps.Exp(numOps.Negate(b))));
                    outputData[outIdx] = numOps.Multiply(a, sigmoid_b);
                }
            }
        });

        var result = TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("GLU", result, input, BackwardFunctions<T>.GLUBackward, new object[] { actualDim });
        return result;
    }

    /// <summary>
    /// GLU backward pass.
    /// </summary>
    public Tensor<T> GLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim = -1)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? input.Rank + dim : dim;
        int dimSize = input._shape[actualDim];
        int halfSize = dimSize / 2;

        var inputData = input.GetFlattenedData();
        var gradOutData = gradOutput.GetFlattenedData();
        var gradInputData = new T[input.Length];

        int innerSize = 1;
        for (int i = actualDim + 1; i < input.Rank; i++)
            innerSize *= input._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= input._shape[i];

        // Gradient: d/da = sigmoid(b) * gradOut, d/db = a * sigmoid(b) * (1 - sigmoid(b)) * gradOut
        Parallel.For(0, outerSize, outer =>
        {
            for (int h = 0; h < halfSize; h++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int aIdx = outer * dimSize * innerSize + h * innerSize + inner;
                    int bIdx = outer * dimSize * innerSize + (h + halfSize) * innerSize + inner;
                    int outIdx = outer * halfSize * innerSize + h * innerSize + inner;

                    T a = inputData[aIdx];
                    T b = inputData[bIdx];
                    T gradOut = gradOutData[outIdx];

                    T sigmoid_b = numOps.Divide(numOps.One, numOps.Add(numOps.One, numOps.Exp(numOps.Negate(b))));
                    T one_minus_sigmoid = numOps.Subtract(numOps.One, sigmoid_b);

                    gradInputData[aIdx] = numOps.Multiply(sigmoid_b, gradOut);
                    gradInputData[bIdx] = numOps.Multiply(numOps.Multiply(numOps.Multiply(a, sigmoid_b), one_minus_sigmoid), gradOut);
                }
            }
        });

        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }

    /// <summary>
    /// GeGLU: Gaussian Error Gated Linear Unit.
    /// GeGLU(a, b) = a * GELU(b)
    /// From "GLU Variants Improve Transformer" (Shazeer, 2020)
    /// </summary>
    public Tensor<T> GeGLU<T>(Tensor<T> input, int dim = -1)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? input.Rank + dim : dim;
        if (actualDim < 0 || actualDim >= input.Rank)
            throw new ArgumentException($"Invalid dimension {dim} for tensor with rank {input.Rank}");

        int dimSize = input._shape[actualDim];
        if (dimSize % 2 != 0)
            throw new ArgumentException($"Dimension {actualDim} must have even size for GeGLU, got {dimSize}");

        int halfSize = dimSize / 2;

        var outputShape = input.Shape.ToArray();
        outputShape[actualDim] = halfSize;

        var inputData = input.GetFlattenedData();
        var outputData = new T[input.Length / 2];

        int innerSize = 1;
        for (int i = actualDim + 1; i < input.Rank; i++)
            innerSize *= input._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= input._shape[i];

        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double sqrtTwoOverPi = Math.Sqrt(2.0 / Math.PI);

        Parallel.For(0, outerSize, outer =>
        {
            for (int h = 0; h < halfSize; h++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int aIdx = outer * dimSize * innerSize + h * innerSize + inner;
                    int bIdx = outer * dimSize * innerSize + (h + halfSize) * innerSize + inner;
                    int outIdx = outer * halfSize * innerSize + h * innerSize + inner;

                    T a = inputData[aIdx];
                    T b = inputData[bIdx];

                    // GELU(b)
                    double bVal = numOps.ToDouble(b);
                    double gelu = 0.5 * bVal * (1.0 + Math.Tanh(sqrtTwoOverPi * (bVal + 0.044715 * bVal * bVal * bVal)));
                    outputData[outIdx] = numOps.Multiply(a, numOps.FromDouble(gelu));
                }
            }
        });

        return TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
    }

    /// <summary>
    /// GeGLU backward pass.
    /// </summary>
    public Tensor<T> GeGLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim = -1)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? input.Rank + dim : dim;
        int dimSize = input._shape[actualDim];
        int halfSize = dimSize / 2;

        var inputData = input.GetFlattenedData();
        var gradOutData = gradOutput.GetFlattenedData();
        var gradInputData = new T[input.Length];

        int innerSize = 1;
        for (int i = actualDim + 1; i < input.Rank; i++)
            innerSize *= input._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= input._shape[i];

        double sqrtTwoOverPi = Math.Sqrt(2.0 / Math.PI);

        Parallel.For(0, outerSize, outer =>
        {
            for (int h = 0; h < halfSize; h++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int aIdx = outer * dimSize * innerSize + h * innerSize + inner;
                    int bIdx = outer * dimSize * innerSize + (h + halfSize) * innerSize + inner;
                    int outIdx = outer * halfSize * innerSize + h * innerSize + inner;

                    T a = inputData[aIdx];
                    T b = inputData[bIdx];
                    T gradOut = gradOutData[outIdx];

                    double bVal = numOps.ToDouble(b);
                    double aVal = numOps.ToDouble(a);
                    double gradOutVal = numOps.ToDouble(gradOut);

                    // GELU and its derivative
                    double inner_val = sqrtTwoOverPi * (bVal + 0.044715 * bVal * bVal * bVal);
                    double tanh_val = Math.Tanh(inner_val);
                    double gelu = 0.5 * bVal * (1.0 + tanh_val);
                    double sech2 = 1.0 - tanh_val * tanh_val;
                    double gelu_deriv = 0.5 * (1.0 + tanh_val) + 0.5 * bVal * sech2 * sqrtTwoOverPi * (1.0 + 3.0 * 0.044715 * bVal * bVal);

                    gradInputData[aIdx] = numOps.FromDouble(gelu * gradOutVal);
                    gradInputData[bIdx] = numOps.FromDouble(aVal * gelu_deriv * gradOutVal);
                }
            }
        });

        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }

    /// <summary>
    /// SwiGLU: Swish-Gated Linear Unit.
    /// SwiGLU(a, b) = a * Swish(b) where Swish(x) = x * sigmoid(x)
    /// From "GLU Variants Improve Transformer" (Shazeer, 2020)
    /// Used in LLaMA, PaLM, and other modern transformers.
    /// </summary>
    public Tensor<T> SwiGLU<T>(Tensor<T> input, int dim = -1)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? input.Rank + dim : dim;
        if (actualDim < 0 || actualDim >= input.Rank)
            throw new ArgumentException($"Invalid dimension {dim} for tensor with rank {input.Rank}");

        int dimSize = input._shape[actualDim];
        if (dimSize % 2 != 0)
            throw new ArgumentException($"Dimension {actualDim} must have even size for SwiGLU, got {dimSize}");

        int halfSize = dimSize / 2;

        var outputShape = input.Shape.ToArray();
        outputShape[actualDim] = halfSize;

        var inputData = input.GetFlattenedData();
        var outputData = new T[input.Length / 2];

        int innerSize = 1;
        for (int i = actualDim + 1; i < input.Rank; i++)
            innerSize *= input._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= input._shape[i];

        // SwiGLU: a * (b * sigmoid(b))
        Parallel.For(0, outerSize, outer =>
        {
            for (int h = 0; h < halfSize; h++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int aIdx = outer * dimSize * innerSize + h * innerSize + inner;
                    int bIdx = outer * dimSize * innerSize + (h + halfSize) * innerSize + inner;
                    int outIdx = outer * halfSize * innerSize + h * innerSize + inner;

                    T a = inputData[aIdx];
                    T b = inputData[bIdx];

                    // Swish(b) = b * sigmoid(b)
                    T sigmoid_b = numOps.Divide(numOps.One, numOps.Add(numOps.One, numOps.Exp(numOps.Negate(b))));
                    T swish_b = numOps.Multiply(b, sigmoid_b);
                    outputData[outIdx] = numOps.Multiply(a, swish_b);
                }
            }
        });

        return TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
    }

    /// <summary>
    /// SwiGLU backward pass.
    /// </summary>
    public Tensor<T> SwiGLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim = -1)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? input.Rank + dim : dim;
        int dimSize = input._shape[actualDim];
        int halfSize = dimSize / 2;

        var inputData = input.GetFlattenedData();
        var gradOutData = gradOutput.GetFlattenedData();
        var gradInputData = new T[input.Length];

        int innerSize = 1;
        for (int i = actualDim + 1; i < input.Rank; i++)
            innerSize *= input._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= input._shape[i];

        // d/da = Swish(b) * gradOut
        // d/db = a * d_Swish(b) * gradOut where d_Swish(b) = sigmoid(b) + b * sigmoid(b) * (1 - sigmoid(b))
        Parallel.For(0, outerSize, outer =>
        {
            for (int h = 0; h < halfSize; h++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int aIdx = outer * dimSize * innerSize + h * innerSize + inner;
                    int bIdx = outer * dimSize * innerSize + (h + halfSize) * innerSize + inner;
                    int outIdx = outer * halfSize * innerSize + h * innerSize + inner;

                    T a = inputData[aIdx];
                    T b = inputData[bIdx];
                    T gradOut = gradOutData[outIdx];

                    T sigmoid_b = numOps.Divide(numOps.One, numOps.Add(numOps.One, numOps.Exp(numOps.Negate(b))));
                    T swish_b = numOps.Multiply(b, sigmoid_b);
                    T one_minus_sigmoid = numOps.Subtract(numOps.One, sigmoid_b);
                    T swish_deriv = numOps.Add(sigmoid_b, numOps.Multiply(numOps.Multiply(b, sigmoid_b), one_minus_sigmoid));

                    gradInputData[aIdx] = numOps.Multiply(swish_b, gradOut);
                    gradInputData[bIdx] = numOps.Multiply(numOps.Multiply(a, swish_deriv), gradOut);
                }
            }
        });

        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }

    /// <summary>
    /// ReGLU: ReLU-Gated Linear Unit.
    /// ReGLU(a, b) = a * ReLU(b)
    /// From "GLU Variants Improve Transformer" (Shazeer, 2020)
    /// </summary>
    public Tensor<T> ReGLU<T>(Tensor<T> input, int dim = -1)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? input.Rank + dim : dim;
        if (actualDim < 0 || actualDim >= input.Rank)
            throw new ArgumentException($"Invalid dimension {dim} for tensor with rank {input.Rank}");

        int dimSize = input._shape[actualDim];
        if (dimSize % 2 != 0)
            throw new ArgumentException($"Dimension {actualDim} must have even size for ReGLU, got {dimSize}");

        int halfSize = dimSize / 2;

        var outputShape = input.Shape.ToArray();
        outputShape[actualDim] = halfSize;

        var inputData = input.GetFlattenedData();
        var outputData = new T[input.Length / 2];

        int innerSize = 1;
        for (int i = actualDim + 1; i < input.Rank; i++)
            innerSize *= input._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= input._shape[i];

        // ReGLU: a * ReLU(b) = a * max(0, b)
        Parallel.For(0, outerSize, outer =>
        {
            for (int h = 0; h < halfSize; h++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int aIdx = outer * dimSize * innerSize + h * innerSize + inner;
                    int bIdx = outer * dimSize * innerSize + (h + halfSize) * innerSize + inner;
                    int outIdx = outer * halfSize * innerSize + h * innerSize + inner;

                    T a = inputData[aIdx];
                    T b = inputData[bIdx];

                    // ReLU(b) = max(0, b)
                    T relu_b = numOps.GreaterThan(b, numOps.Zero) ? b : numOps.Zero;
                    outputData[outIdx] = numOps.Multiply(a, relu_b);
                }
            }
        });

        return TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
    }

    /// <summary>
    /// ReGLU backward pass.
    /// </summary>
    public Tensor<T> ReGLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim = -1)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? input.Rank + dim : dim;
        int dimSize = input._shape[actualDim];
        int halfSize = dimSize / 2;

        var inputData = input.GetFlattenedData();
        var gradOutData = gradOutput.GetFlattenedData();
        var gradInputData = new T[input.Length];

        int innerSize = 1;
        for (int i = actualDim + 1; i < input.Rank; i++)
            innerSize *= input._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= input._shape[i];

        // d/da = ReLU(b) * gradOut
        // d/db = a * (b > 0 ? 1 : 0) * gradOut
        Parallel.For(0, outerSize, outer =>
        {
            for (int h = 0; h < halfSize; h++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int aIdx = outer * dimSize * innerSize + h * innerSize + inner;
                    int bIdx = outer * dimSize * innerSize + (h + halfSize) * innerSize + inner;
                    int outIdx = outer * halfSize * innerSize + h * innerSize + inner;

                    T a = inputData[aIdx];
                    T b = inputData[bIdx];
                    T gradOut = gradOutData[outIdx];

                    T relu_b = numOps.GreaterThan(b, numOps.Zero) ? b : numOps.Zero;
                    T relu_deriv = numOps.GreaterThan(b, numOps.Zero) ? numOps.One : numOps.Zero;

                    gradInputData[aIdx] = numOps.Multiply(relu_b, gradOut);
                    gradInputData[bIdx] = numOps.Multiply(numOps.Multiply(a, relu_deriv), gradOut);
                }
            }
        });

        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }


    #endregion

    #region Extended Tensor Operations

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTranspose<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank != 2)
            throw new ArgumentException($"TensorTranspose requires a 2D tensor. Got rank {tensor.Rank}.");

        int rows = tensor._shape[0];
        int cols = tensor._shape[1];
        var result = TensorAllocator.RentUninitialized<T>(new[] { cols, rows });

        var srcData = tensor.GetFlattenedData();
        var dstData = result.GetDataArray();

        // Cache-blocked transpose for better locality
        const int Block = 32;
        for (int ii = 0; ii < rows; ii += Block)
        {
            int iEnd = Math.Min(ii + Block, rows);
            for (int jj = 0; jj < cols; jj += Block)
            {
                int jEnd = Math.Min(jj + Block, cols);
                for (int i = ii; i < iEnd; i++)
                {
                    int srcRow = i * cols;
                    for (int j = jj; j < jEnd; j++)
                    {
                        dstData[j * rows + i] = srcData[srcRow + j];
                    }
                }
            }
        }

        DifferentiableOps.RecordUnary("TensorTranspose", result, tensor, BackwardFunctions<T>.TransposeBackward);
        return result;
    }

    /// <inheritdoc/>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public virtual Tensor<T> TensorMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank < 2) throw new ArgumentException($"TensorMatMul requires tensors of rank >= 2. Got rank {a.Rank} for first tensor.");
        if (b.Rank < 2) throw new ArgumentException($"TensorMatMul requires tensors of rank >= 2. Got rank {b.Rank} for second tensor.");

        // Materialize non-contiguous views so downstream paths can use .Data safely.
        // BatchMatMul already does this for rank >= 3; TensorMatMul2D did not.
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        Tensor<T> result;

        // Standard 2D x 2D case
        if (a.Rank == 2 && b.Rank == 2)
        {
            result = TensorMatMul2D(a, b, numOps);
        }
        // ND x 2D case: broadcast 2D weights over batch dimensions
        // This is the common transformer pattern: [batch, seq, features] @ [features, output]
        else if (b.Rank == 2)
        {
            result = TensorMatMulBatched(a, b, numOps);
        }
        // 3D x 3D or ND x ND: full batched matmul (batch dimensions must match)
        else if (a.Rank == b.Rank)
        {
            result = TensorMatMulFullBatched(a, b, numOps);
        }
        else
        {
            throw new ArgumentException($"Unsupported TensorMatMul combination: ranks {a.Rank} and {b.Rank}. Supported: 2Dx2D, NDx2D, NDxND (same rank).");
        }

        DifferentiableOps.RecordBinary("TensorMatMul", result, a, b, BackwardFunctions<T>.MatMulBackward);
        return result;
    }

    /// <summary>
    /// Standard 2D matrix multiplication: [M, N] @ [N, P] = [M, P]
    /// Uses BLAS when available for float/double, falls back to parallel loops otherwise.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    private Tensor<T> TensorMatMul2D<T>(Tensor<T> a, Tensor<T> b, INumericOperations<T> numOps)
    {
        int m = a._shape[0];
        int n = a._shape[1];
        int p = b._shape[1];

        if (n != b._shape[0])
            throw new ArgumentException($"Matrix dimensions incompatible: [{m},{n}] x [{b._shape[0]},{p}]");

        var result = TensorAllocator.Rent<T>(new[] { m, p });

        // Try BLAS-accelerated path for float/double tensors
        if (MatrixMultiplyHelper.TryGemm(a.Data, 0, b.Data, 0, result.Data, 0, m, n, p))
        {
            return result;
        }

        // Fallback to parallel loops for non-float/double or when BLAS unavailable
        var aData = a.GetDataArray();
        var bData = b.GetDataArray();
        var rData = result.GetDataArray();

        Parallel.For(0, m, i =>
        {
            for (int j = 0; j < p; j++)
            {
                T sum = numOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    sum = numOps.Add(sum, numOps.Multiply(aData[i * n + k], bData[k * p + j]));
                }
                rData[i * p + j] = sum;
            }
        });

        return result;
    }

    /// <summary>
    /// Batched matmul: [..., M, N] @ [N, P] = [..., M, P]
    /// Weights (b) are broadcasted over all batch dimensions of a.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    private Tensor<T> TensorMatMulBatched<T>(Tensor<T> a, Tensor<T> b, INumericOperations<T> numOps)
    {
        int aRank = a.Rank;
        int m = a._shape[aRank - 2];  // Second to last dim
        int n = a._shape[aRank - 1];  // Last dim
        int p = b._shape[1];          // Output features

        if (n != b._shape[0])
            throw new ArgumentException($"Matrix dimensions incompatible for batched matmul: last dim of a is {n}, first dim of b is {b._shape[0]}");

        // Calculate batch size (product of all dimensions except last 2)
        int batchSize = 1;
        var batchShape = new int[aRank - 2];
        for (int i = 0; i < aRank - 2; i++)
        {
            batchSize *= a._shape[i];
            batchShape[i] = a._shape[i];
        }

        // Output shape: [...batch dims..., m, p]
        var outputShape = new int[aRank];
        for (int i = 0; i < aRank - 2; i++)
        {
            outputShape[i] = a._shape[i];
        }
        outputShape[aRank - 2] = m;
        outputShape[aRank - 1] = p;

        var result = TensorAllocator.RentUninitialized<T>(outputShape);

        int matrixSizeA = m * n;
        int matrixSizeResult = m * p;

        var aData = a.GetDataArray();
        var bData = b.GetDataArray();
        var rData = result.GetDataArray();

        Parallel.For(0, batchSize, batch =>
        {
            int aOffset = batch * matrixSizeA;
            int resultOffset = batch * matrixSizeResult;

            // Try BLAS for each batch slice
            if (MatrixMultiplyHelper.TryGemm(a.Data, aOffset, b.Data, 0, result.Data, resultOffset, m, n, p))
            {
                return;
            }

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < n; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(aData[aOffset + i * n + k], bData[k * p + j]));
                    }
                    rData[resultOffset + i * p + j] = sum;
                }
            }
        });

        return result;
    }

    /// <summary>
    /// Full batched matmul: [..., M, N] @ [..., N, P] = [..., M, P]
    /// Batch dimensions must match.
    /// </summary>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    private Tensor<T> TensorMatMulFullBatched<T>(Tensor<T> a, Tensor<T> b, INumericOperations<T> numOps)
    {
        int rank = a.Rank;
        int m = a._shape[rank - 2];
        int n = a._shape[rank - 1];
        int p = b._shape[rank - 1];

        if (n != b._shape[rank - 2])
            throw new ArgumentException($"Matrix dimensions incompatible for batched matmul: [{m},{n}] x [{b._shape[rank - 2]},{p}]");

        // Verify batch dimensions match
        for (int i = 0; i < rank - 2; i++)
        {
            if (a._shape[i] != b._shape[i])
                throw new ArgumentException($"Batch dimensions must match. Got {a._shape[i]} vs {b._shape[i]} at dimension {i}");
        }

        // Calculate batch size
        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++)
        {
            batchSize *= a._shape[i];
        }

        // Output shape
        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
        {
            outputShape[i] = a._shape[i];
        }
        outputShape[rank - 2] = m;
        outputShape[rank - 1] = p;

        var result = TensorAllocator.RentUninitialized<T>(outputShape);

        int matrixSizeA = m * n;
        int matrixSizeB = n * p;
        int matrixSizeResult = m * p;

        var aData = a.GetDataArray();
        var bData = b.GetDataArray();
        var rData = result.GetDataArray();

        Parallel.For(0, batchSize, batch =>
        {
            int aOffset = batch * matrixSizeA;
            int bOffset = batch * matrixSizeB;
            int resultOffset = batch * matrixSizeResult;

            // Try BLAS for each batch slice
            if (MatrixMultiplyHelper.TryGemm(a.Data, aOffset, b.Data, bOffset, result.Data, resultOffset, m, n, p))
            {
                return;
            }

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < n; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(aData[aOffset + i * n + k], bData[bOffset + k * p + j]));
                    }
                    rData[resultOffset + i * p + j] = sum;
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (!input.IsContiguous) input = input.Contiguous();
        if (input.Rank != 4) throw new ArgumentException($"Conv2D requires 4D input tensor. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 4) throw new ArgumentException($"Conv2D requires 4D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements", nameof(padding));
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements", nameof(dilation));
        if (dilation[0] <= 0 || dilation[1] <= 0) throw new ArgumentException("Dilation elements must be positive", nameof(dilation));
        if (input._shape[1] != kernel._shape[1]) throw new ArgumentException($"Input channels ({input._shape[1]}) must match kernel in_channels ({kernel._shape[1]})");

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outChannels = kernel._shape[0];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        int effectiveKernelH = dilationH * (kernelHeight - 1) + 1;
        int effectiveKernelW = dilationW * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputWidth = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException($"Invalid output dimensions ({outputHeight}x{outputWidth}). Check kernel size, stride, padding, and dilation parameters.");

        var result = TensorAllocator.Rent<T>([batch, outChannels, outputHeight, outputWidth]);
        var inputData = input.GetDataArray();
        var kernelData = kernel.GetDataArray();
        var outputData = result.GetDataArray();

        Parallel.For(0, batch * outChannels, idx =>
        {
            int b = idx / outChannels;
            int oc = idx % outChannels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T sum = numOps.Zero;

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int ih = oh * strideH + kh * dilationH - padH;
                                int iw = ow * strideW + kw * dilationW - padW;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                                    sum = numOps.Add(sum, numOps.Multiply(inputData[inputIdx], kernelData[kernelIdx]));
                                }
                            }
                        }
                    }

                    int outputIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                    outputData[outputIdx] = sum;
                }
            }
        });

        DifferentiableOps.RecordBinary("Conv2D", result, input, kernel, BackwardFunctions<T>.Conv2DBackward,
            new object[] { stride, padding, dilation });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!kernel.IsContiguous) kernel = kernel.Contiguous();
        if (inputShape == null || inputShape.Length != 4) throw new ArgumentException("inputShape must be array of 4 elements [batch, inChannels, height, width]", nameof(inputShape));
        if (gradOutput.Rank != 4) throw new ArgumentException($"Conv2DBackwardInput requires 4D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (kernel.Rank != 4) throw new ArgumentException($"Conv2DBackwardInput requires 4D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements", nameof(padding));
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements", nameof(dilation));
        if (dilation[0] <= 0 || dilation[1] <= 0) throw new ArgumentException("Dilation elements must be positive", nameof(dilation));
        if (gradOutput._shape[0] != inputShape[0]) throw new ArgumentException($"gradOutput batch size ({gradOutput._shape[0]}) must match inputShape batch size ({inputShape[0]})");
        if (gradOutput._shape[1] != kernel._shape[0]) throw new ArgumentException($"gradOutput outChannels ({gradOutput._shape[1]}) must match kernel outChannels ({kernel._shape[0]})");
        if (inputShape[1] != kernel._shape[1]) throw new ArgumentException($"inputShape inChannels ({inputShape[1]}) must match kernel inChannels ({kernel._shape[1]})");

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int outChannels = kernel._shape[0];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];

        var gradInput = new T[batch * inChannels * height * width];
        var gradOutputData = gradOutput.GetDataArray();
        var kernelData = kernel.GetDataArray();

        Parallel.For(0, batch * inChannels, idx =>
        {
            int b = idx / inChannels;
            int ic = idx % inChannels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                        T gradVal = gradOutputData[gradOutIdx];

                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int ih = oh * strideH + kh * dilationH - padH;
                                int iw = ow * strideW + kw * dilationW - padW;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int gradInputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                                    // No lock needed - each (batch, inChannel) partition owns disjoint gradInput slices
                                    gradInput[gradInputIdx] = numOps.Add(gradInput[gradInputIdx], numOps.Multiply(gradVal, kernelData[kernelIdx]));
                                }
                            }
                        }
                    }
                }
            }
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInput));
    }

    /// <inheritdoc/>
    public Tensor<T> Conv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!input.IsContiguous) input = input.Contiguous();
        if (kernelShape == null || kernelShape.Length != 4) throw new ArgumentException("kernelShape must be array of 4 elements [outChannels, inChannels, kernelHeight, kernelWidth]", nameof(kernelShape));
        if (gradOutput.Rank != 4) throw new ArgumentException($"Conv2DBackwardKernel requires 4D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (input.Rank != 4) throw new ArgumentException($"Conv2DBackwardKernel requires 4D input tensor. Got rank {input.Rank}.", nameof(input));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements", nameof(padding));
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements", nameof(dilation));
        if (dilation[0] <= 0 || dilation[1] <= 0) throw new ArgumentException("Dilation elements must be positive", nameof(dilation));
        if (gradOutput._shape[0] != input._shape[0]) throw new ArgumentException($"gradOutput batch size ({gradOutput._shape[0]}) must match input batch size ({input._shape[0]})");
        if (gradOutput._shape[1] != kernelShape[0]) throw new ArgumentException($"gradOutput outChannels ({gradOutput._shape[1]}) must match kernelShape outChannels ({kernelShape[0]})");
        if (input._shape[1] != kernelShape[1]) throw new ArgumentException($"input inChannels ({input._shape[1]}) must match kernelShape inChannels ({kernelShape[1]})");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outChannels = kernelShape[0];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];

        var gradKernel = new T[outChannels * inChannels * kernelHeight * kernelWidth];
        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetDataArray();

        Parallel.For(0, outChannels * inChannels, idx =>
        {
            int oc = idx / inChannels;
            int ic = idx % inChannels;

            for (int kh = 0; kh < kernelHeight; kh++)
            {
                for (int kw = 0; kw < kernelWidth; kw++)
                {
                    T sum = numOps.Zero;

                    for (int b = 0; b < batch; b++)
                    {
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                int ih = oh * strideH + kh * dilationH - padH;
                                int iw = ow * strideW + kw * dilationW - padW;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                    int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    sum = numOps.Add(sum, numOps.Multiply(gradOutputData[gradOutIdx], inputData[inputIdx]));
                                }
                            }
                        }
                    }

                    int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                    gradKernel[kernelIdx] = sum;
                }
            }
        });

        return TensorAllocator.Rent<T>(kernelShape, new Vector<T>(gradKernel));
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,] maxIndices)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int channels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int poolH = poolSize[0], poolW = poolSize[1];
        int strideH = stride[0], strideW = stride[1];

        if (poolH > height || poolW > width)
            throw new ArgumentException($"Pool size ({poolH}x{poolW}) cannot exceed input spatial dimensions ({height}x{width})");

        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException($"Invalid output dimensions ({outputHeight}x{outputWidth}). Check pool size and stride.");

        var result = TensorAllocator.Rent<T>([batch, channels, outputHeight, outputWidth]);
        // Use local variable to avoid capturing out parameter in lambda
        var indices = new int[batch, channels, outputHeight, outputWidth, 2];

        var inputData = input.GetFlattenedData();
        var outputData = result.GetDataArray();

        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T maxVal = numOps.MinValue;
                    int maxH = 0, maxW = 0;

                    for (int kh = 0; kh < poolH; kh++)
                    {
                        for (int kw = 0; kw < poolW; kw++)
                        {
                            int ih = oh * strideH + kh;
                            int iw = ow * strideW + kw;

                            int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                            T val = inputData[inputIdx];

                            if (numOps.GreaterThan(val, maxVal))
                            {
                                maxVal = val;
                                maxH = ih;
                                maxW = iw;
                            }
                        }
                    }

                    int outputIdx = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                    outputData[outputIdx] = maxVal;
                    indices[b, c, oh, ow, 0] = maxH;
                    indices[b, c, oh, ow, 1] = maxW;
                }
            }
        });

        // Assign local variable to out parameter after parallel section
        maxIndices = indices;
        DifferentiableOps.RecordUnary("MaxPool2DWithIndices", result, input, BackwardFunctions<T>.MaxPool2DWithIndicesBackward, new object[] { maxIndices, poolSize, stride });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool2DBackward<T>(Tensor<T> gradOutput, int[,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];

        var result = TensorAllocator.RentUninitialized<T>(inputShape);
        var gradInputData = result.GetDataArray();
        var gradOutputData = gradOutput.GetFlattenedData();

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Note: Cannot parallelize across batch*channels because multiple output positions
        // may map to the same input position within a channel. Parallelize across batches only.
        Parallel.For(0, batch, b =>
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int maxH = maxIndices[b, c, oh, ow, 0];
                        int maxW = maxIndices[b, c, oh, ow, 1];

                        int gradOutIdx = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                        int gradInIdx = ((b * channels + c) * height + maxH) * width + maxW;

                        gradInputData[gradInIdx] = numOps.Add(gradInputData[gradInIdx], gradOutputData[gradOutIdx]);
                    }
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2D<T>(Tensor<T> input, int[] poolSize, int[] stride)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int channels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int poolH = poolSize[0], poolW = poolSize[1];
        int strideH = stride[0], strideW = stride[1];

        if (poolH > height || poolW > width)
            throw new ArgumentException($"Pool size ({poolH}x{poolW}) cannot exceed input spatial dimensions ({height}x{width})");

        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

        if (outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException($"Invalid output dimensions ({outputHeight}x{outputWidth}). Check pool size and stride.");

        var inputData = input.GetFlattenedData();
        var outputData = new T[batch * channels * outputHeight * outputWidth];
        T poolArea = numOps.FromDouble(poolH * poolW);

        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T sum = numOps.Zero;

                    for (int kh = 0; kh < poolH; kh++)
                    {
                        for (int kw = 0; kw < poolW; kw++)
                        {
                            int ih = oh * strideH + kh;
                            int iw = ow * strideW + kw;
                            int inputIdx = ((b * channels + c) * height + ih) * width + iw;
                            sum = numOps.Add(sum, inputData[inputIdx]);
                        }
                    }

                    int outputIdx = ((b * channels + c) * outputHeight + oh) * outputWidth + ow;
                    outputData[outputIdx] = numOps.Divide(sum, poolArea);
                }
            }
        });

        return TensorAllocator.Rent<T>([batch, channels, outputHeight, outputWidth], new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int poolH = poolSize[0], poolW = poolSize[1];
        int strideH = stride[0], strideW = stride[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];

        var result = TensorAllocator.RentUninitialized<T>(inputShape);
        var gradInputData = result.GetDataArray();
        var gradOutputData = gradOutput.GetFlattenedData();
        T poolArea = numOps.FromDouble(poolH * poolW);

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Parallelize across batches (within a batch*channel, output positions scatter to overlapping input regions)
        Parallel.For(0, batch, b =>
        {
            for (int c = 0; c < channels; c++)
            {
                int inputBaseOffset = (b * channels + c) * height * width;
                int outputBaseOffset = (b * channels + c) * outputHeight * outputWidth;

                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T grad = numOps.Divide(gradOutputData[outputBaseOffset + oh * outputWidth + ow], poolArea);

                        for (int kh = 0; kh < poolH; kh++)
                        {
                            int ih = oh * strideH + kh;
                            for (int kw = 0; kw < poolW; kw++)
                            {
                                int iw = ow * strideW + kw;
                                int gradInIdx = inputBaseOffset + ih * width + iw;
                                gradInputData[gradInIdx] = numOps.Add(gradInputData[gradInIdx], grad);
                            }
                        }
                    }
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int multiplier = kernel._shape[1];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];

        int outputHeight = (height + 2 * padH - kernelHeight) / strideH + 1;
        int outputWidth = (width + 2 * padW - kernelWidth) / strideW + 1;
        int outChannels = inChannels * multiplier;

        var inputData = input.GetFlattenedData();
        var kernelData = kernel.GetFlattenedData();
        var outputData = new T[batch * outChannels * outputHeight * outputWidth];

        Parallel.For(0, batch * outChannels, idx =>
        {
            int b = idx / outChannels;
            int oc = idx % outChannels;
            int ic = oc / multiplier;
            int m = oc % multiplier;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T sum = numOps.Zero;

                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            int ih = oh * strideH + kh - padH;
                            int iw = ow * strideW + kw - padW;

                            if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                            {
                                int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                int kernelIdx = ((ic * multiplier + m) * kernelHeight + kh) * kernelWidth + kw;
                                sum = numOps.Add(sum, numOps.Multiply(inputData[inputIdx], kernelData[kernelIdx]));
                            }
                        }
                    }

                    int outputIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                    outputData[outputIdx] = sum;
                }
            }
        });

        var dwConvResult = TensorAllocator.Rent<T>([batch, outChannels, outputHeight, outputWidth], new Vector<T>(outputData));
        DifferentiableOps.RecordBinary("DepthwiseConv2D", dwConvResult, input, kernel, BackwardFunctions<T>.DepthwiseConv2DBackward, new object[] { stride, padding });
        return dwConvResult;
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (!kernel.IsContiguous) kernel = kernel.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int multiplier = kernel._shape[1];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];
        int outChannels = inChannels * multiplier;

        var gradInput = new T[batch * inChannels * height * width];
        var gradOutputData = gradOutput.GetFlattenedData();
        var kernelData = kernel.GetFlattenedData();

        for (int i = 0; i < gradInput.Length; i++)
            gradInput[i] = numOps.Zero;

        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                int ic = oc / multiplier;
                int m = oc % multiplier;

                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                        T gradVal = gradOutputData[gradOutIdx];

                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int ih = oh * strideH + kh - padH;
                                int iw = ow * strideW + kw - padW;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    int gradInIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                    int kernelIdx = ((ic * multiplier + m) * kernelHeight + kh) * kernelWidth + kw;
                                    gradInput[gradInIdx] = numOps.Add(gradInput[gradInIdx], numOps.Multiply(gradVal, kernelData[kernelIdx]));
                                }
                            }
                        }
                    }
                }
            }
        }

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInput));
    }

    /// <inheritdoc/>
    public Tensor<T> DepthwiseConv2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int multiplier = kernelShape[1];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];

        var gradKernel = new T[inChannels * multiplier * kernelHeight * kernelWidth];
        var gradOutputData = gradOutput.GetFlattenedData();
        var inputData = input.GetFlattenedData();

        for (int i = 0; i < gradKernel.Length; i++)
            gradKernel[i] = numOps.Zero;

        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int m = 0; m < multiplier; m++)
            {
                int oc = ic * multiplier + m;

                for (int kh = 0; kh < kernelHeight; kh++)
                {
                    for (int kw = 0; kw < kernelWidth; kw++)
                    {
                        T sum = numOps.Zero;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int oh = 0; oh < outputHeight; oh++)
                            {
                                for (int ow = 0; ow < outputWidth; ow++)
                                {
                                    int ih = oh * strideH + kh - padH;
                                    int iw = ow * strideW + kw - padW;

                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    {
                                        int gradOutIdx = ((b * (inChannels * multiplier) + oc) * outputHeight + oh) * outputWidth + ow;
                                        int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                        sum = numOps.Add(sum, numOps.Multiply(gradOutputData[gradOutIdx], inputData[inputIdx]));
                                    }
                                }
                            }
                        }

                        int kernelIdx = ((ic * multiplier + m) * kernelHeight + kh) * kernelWidth + kw;
                        gradKernel[kernelIdx] = sum;
                    }
                }
            }
        }

        return TensorAllocator.Rent<T>(kernelShape, new Vector<T>(gradKernel));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ConvTranspose2D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (!input.IsContiguous) input = input.Contiguous();
        if (!kernel.IsContiguous) kernel = kernel.Contiguous();
        if (input.Rank != 4) throw new ArgumentException($"ConvTranspose2D requires 4D input tensor. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 4) throw new ArgumentException($"ConvTranspose2D requires 4D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements", nameof(padding));
        if (padding[0] < 0 || padding[1] < 0) throw new ArgumentException("Padding elements must be non-negative", nameof(padding));
        if (outputPadding == null || outputPadding.Length != 2) throw new ArgumentException("OutputPadding must be array of 2 elements", nameof(outputPadding));
        if (outputPadding[0] < 0 || outputPadding[1] < 0) throw new ArgumentException("OutputPadding elements must be non-negative", nameof(outputPadding));
        if (input._shape[1] != kernel._shape[0]) throw new ArgumentException($"Input inChannels ({input._shape[1]}) must match kernel inChannels ({kernel._shape[0]})");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outChannels = kernel._shape[1];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int outPadH = outputPadding[0], outPadW = outputPadding[1];

        int outputHeight = (height - 1) * strideH - 2 * padH + kernelHeight + outPadH;
        int outputWidth = (width - 1) * strideW - 2 * padW + kernelWidth + outPadW;

        var inputData = input.GetDataArray();
        var kernelData = kernel.GetDataArray();
        var outputData = new T[batch * outChannels * outputHeight * outputWidth];

        for (int i = 0; i < outputData.Length; i++)
            outputData[i] = numOps.Zero;

        // Use thread-local accumulation to avoid lock contention
        var lockObj = new object();
        Parallel.For(0, batch * inChannels,
            // Initialize thread-local storage
            () => new T[batch * outChannels * outputHeight * outputWidth],
            // Body
            (idx, state, localOutput) =>
            {
                int b = idx / inChannels;
                int ic = idx % inChannels;

                for (int ih = 0; ih < height; ih++)
                {
                    for (int iw = 0; iw < width; iw++)
                    {
                        int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                        T inputVal = inputData[inputIdx];

                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int oh = ih * strideH - padH + kh;
                                    int ow = iw * strideW - padW + kw;

                                    if (oh >= 0 && oh < outputHeight && ow >= 0 && ow < outputWidth)
                                    {
                                        int outputIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                        int kernelIdx = ((ic * outChannels + oc) * kernelHeight + kh) * kernelWidth + kw;
                                        localOutput[outputIdx] = numOps.Add(localOutput[outputIdx], numOps.Multiply(inputVal, kernelData[kernelIdx]));
                                    }
                                }
                            }
                        }
                    }
                }
                return localOutput;
            },
            // Merge thread-local results
            (localOutput) =>
            {
                lock (lockObj)
                {
                    for (int i = 0; i < outputData.Length; i++)
                    {
                        outputData[i] = numOps.Add(outputData[i], localOutput[i]);
                    }
                }
            });

        var convTransResult = TensorAllocator.Rent<T>([batch, outChannels, outputHeight, outputWidth], new Vector<T>(outputData));
        DifferentiableOps.RecordBinary("ConvTranspose2D", convTransResult, input, kernel,
            BackwardFunctions<T>.ConvTranspose2DBackward,
            savedState: new object[] { (int[])stride.Clone(), (int[])padding.Clone() });
        return convTransResult;
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        // ConvTranspose2D backward w.r.t. input is equivalent to Conv2D forward
        // Note: This implementation assumes unit dilation. For non-unit dilation, the gradient requires
        // more complex handling (e.g., dilated convolution with flipped kernel).
        var result = Conv2D(gradOutput, kernel, stride, padding, [1, 1]);

        // Validate that the result matches expected input shape
        if (result._shape[0] != inputShape[0] || result._shape[1] != inputShape[1] ||
            result._shape[2] != inputShape[2] || result._shape[3] != inputShape[3])
        {
            throw new InvalidOperationException(
                $"ConvTranspose2DBackwardInput result shape [{string.Join(",", result._shape)}] " +
                $"does not match expected inputShape [{string.Join(",", inputShape)}]. " +
                "This may occur with non-standard stride/padding configurations.");
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outChannels = kernelShape[1];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];

        var gradKernel = new T[inChannels * outChannels * kernelHeight * kernelWidth];
        var gradOutputData = gradOutput.GetFlattenedData();
        var inputData = input.GetFlattenedData();

        for (int i = 0; i < gradKernel.Length; i++)
            gradKernel[i] = numOps.Zero;

        for (int ic = 0; ic < inChannels; ic++)
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int kh = 0; kh < kernelHeight; kh++)
                {
                    for (int kw = 0; kw < kernelWidth; kw++)
                    {
                        T sum = numOps.Zero;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int ih = 0; ih < height; ih++)
                            {
                                for (int iw = 0; iw < width; iw++)
                                {
                                    int oh = ih * strideH - padH + kh;
                                    int ow = iw * strideW - padW + kw;

                                    if (oh >= 0 && oh < outputHeight && ow >= 0 && ow < outputWidth)
                                    {
                                        int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                        int inputIdx = ((b * inChannels + ic) * height + ih) * width + iw;
                                        sum = numOps.Add(sum, numOps.Multiply(gradOutputData[gradOutIdx], inputData[inputIdx]));
                                    }
                                }
                            }
                        }

                        int kernelIdx = ((ic * outChannels + oc) * kernelHeight + kh) * kernelWidth + kw;
                        gradKernel[kernelIdx] = sum;
                    }
                }
            }
        }

        return TensorAllocator.Rent<T>(kernelShape, new Vector<T>(gradKernel));
    }

    #region Deformable Convolution Operations

    /// <inheritdoc/>
    public Tensor<T> DeformableConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (offset == null) throw new ArgumentNullException(nameof(offset));
        if (input.Rank != 4) throw new ArgumentException($"DeformableConv2D requires 4D input tensor. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 4) throw new ArgumentException($"DeformableConv2D requires 4D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements.", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements.", nameof(padding));
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements.", nameof(dilation));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outChannels = kernel._shape[0];
        int kernelInChannels = kernel._shape[1];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        // Validate input/kernel channel compatibility
        if (inChannels != kernelInChannels)
            throw new ArgumentException($"Input channels ({inChannels}) must match kernel in_channels ({kernelInChannels}).", nameof(kernel));

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        // Validate stride and dilation are positive
        if (strideH <= 0 || strideW <= 0)
            throw new ArgumentException("Stride elements must be positive.", nameof(stride));
        if (dilationH <= 0 || dilationW <= 0)
            throw new ArgumentException("Dilation elements must be positive.", nameof(dilation));

        int effectiveKernelH = dilationH * (kernelHeight - 1) + 1;
        int effectiveKernelW = dilationW * (kernelWidth - 1) + 1;

        int outputHeight = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputWidth = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        // Validate output dimensions are positive
        if (outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException(
                $"Invalid deformable conv parameters: output size would be {outputHeight}x{outputWidth} " +
                $"for input {height}x{width}, kernel {kernelHeight}x{kernelWidth}, " +
                $"stride=({strideH},{strideW}), padding=({padH},{padW}), dilation=({dilationH},{dilationW}).");

        int numKernelPositions = kernelHeight * kernelWidth;

        // Validate offset shape: [batch, 2*kernel_h*kernel_w, out_h, out_w]
        if (offset.Rank != 4)
            throw new ArgumentException($"Offset tensor must be 4D. Got rank {offset.Rank}.", nameof(offset));
        if (offset._shape[0] != batch || offset._shape[1] != 2 * numKernelPositions ||
            offset._shape[2] != outputHeight || offset._shape[3] != outputWidth)
            throw new ArgumentException(
                $"Offset tensor must have shape [{batch}, {2 * numKernelPositions}, {outputHeight}, {outputWidth}]. " +
                $"Got [{string.Join(",", offset._shape)}].", nameof(offset));

        // Validate mask shape if provided: [batch, kernel_h*kernel_w, out_h, out_w]
        if (mask != null)
        {
            if (mask.Rank != 4)
                throw new ArgumentException($"Mask tensor must be 4D. Got rank {mask.Rank}.", nameof(mask));
            if (mask._shape[0] != batch || mask._shape[1] != numKernelPositions ||
                mask._shape[2] != outputHeight || mask._shape[3] != outputWidth)
                throw new ArgumentException(
                    $"Mask tensor must have shape [{batch}, {numKernelPositions}, {outputHeight}, {outputWidth}]. " +
                    $"Got [{string.Join(",", mask._shape)}].", nameof(mask));
        }

        var result = TensorAllocator.Rent<T>([batch, outChannels, outputHeight, outputWidth]);
        var inputData = input.GetDataArray();
        var kernelData = kernel.GetDataArray();
        var offsetData = offset.GetDataArray();
        var maskData = mask?.GetDataArray();
        var outputData = result.GetDataArray();

        // Parallel over batch * outChannels for maximum parallelism
        Parallel.For(0, batch * outChannels, idx =>
        {
            int b = idx / outChannels;
            int oc = idx % outChannels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    T sum = numOps.Zero;

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int kernelPosIdx = kh * kernelWidth + kw;

                                // Get offsets for this kernel position
                                int offsetYIdx = ((b * (2 * numKernelPositions) + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;
                                int offsetXIdx = ((b * (2 * numKernelPositions) + numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;

                                double offsetY = numOps.ToDouble(offsetData[offsetYIdx]);
                                double offsetX = numOps.ToDouble(offsetData[offsetXIdx]);

                                // Base sampling position + learned offset
                                double sampledH = oh * strideH - padH + kh * dilationH + offsetY;
                                double sampledW = ow * strideW - padW + kw * dilationW + offsetX;

                                // Bilinear interpolation
                                T sampledValue = BilinearSample(inputData, b, ic, inChannels, height, width, sampledH, sampledW, numOps);

                                // Apply modulation mask (DCNv2)
                                if (maskData != null)
                                {
                                    int maskIdx = ((b * numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;
                                    sampledValue = numOps.Multiply(sampledValue, maskData[maskIdx]);
                                }

                                // Multiply by kernel weight
                                int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                                sum = numOps.Add(sum, numOps.Multiply(sampledValue, kernelData[kernelIdx]));
                            }
                        }
                    }

                    int outputIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                    outputData[outputIdx] = sum;
                }
            }
        });

        return result;
    }

    /// <summary>
    /// Performs bilinear sampling at a fractional position.
    /// </summary>
    private static T BilinearSample<T>(
        T[] data,
        int batch,
        int channel,
        int totalChannels,
        int height,
        int width,
        double h,
        double w,
        INumericOperations<T> numOps)
    {
        // Out of bounds check
        if (h < -1 || h > height || w < -1 || w > width)
            return numOps.Zero;

        int h0 = (int)Math.Floor(h);
        int h1 = h0 + 1;
        int w0 = (int)Math.Floor(w);
        int w1 = w0 + 1;

        double lh = h - h0; // Lerp factor for height
        double lw = w - w0; // Lerp factor for width

        T v00 = GetPixelValue(data, batch, channel, totalChannels, height, width, h0, w0, numOps);
        T v01 = GetPixelValue(data, batch, channel, totalChannels, height, width, h0, w1, numOps);
        T v10 = GetPixelValue(data, batch, channel, totalChannels, height, width, h1, w0, numOps);
        T v11 = GetPixelValue(data, batch, channel, totalChannels, height, width, h1, w1, numOps);

        // Bilinear interpolation: (1-lh)*(1-lw)*v00 + (1-lh)*lw*v01 + lh*(1-lw)*v10 + lh*lw*v11
        double w00 = (1 - lh) * (1 - lw);
        double w01 = (1 - lh) * lw;
        double w10 = lh * (1 - lw);
        double w11 = lh * lw;

        T result = numOps.Multiply(v00, numOps.FromDouble(w00));
        result = numOps.Add(result, numOps.Multiply(v01, numOps.FromDouble(w01)));
        result = numOps.Add(result, numOps.Multiply(v10, numOps.FromDouble(w10)));
        result = numOps.Add(result, numOps.Multiply(v11, numOps.FromDouble(w11)));

        return result;
    }

    /// <summary>
    /// Gets a pixel value with boundary checking (returns zero for out-of-bounds).
    /// </summary>
    private static T GetPixelValue<T>(
        T[] data,
        int batch,
        int channel,
        int totalChannels,
        int height,
        int width,
        int h,
        int w,
        INumericOperations<T> numOps)
    {
        if (h < 0 || h >= height || w < 0 || w >= width)
            return numOps.Zero;

        int idx = ((batch * totalChannels + channel) * height + h) * width + w;
        return data[idx];
    }

    /// <inheritdoc/>
    public Tensor<T> DeformableConv2DBackwardInput<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] inputShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (offset == null) throw new ArgumentNullException(nameof(offset));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int outChannels = kernel._shape[0];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];
        int numKernelPositions = kernelHeight * kernelWidth;

        var gradInput = TensorAllocator.RentUninitialized<T>(inputShape);
        var gradInputData = gradInput.GetDataArray();
        var gradOutputData = gradOutput.GetDataArray();
        var kernelData = kernel.GetDataArray();
        var offsetData = offset.GetDataArray();
        var maskData = mask?.GetDataArray();

        // Use a lock array to avoid race conditions during atomic adds
        var locks = new object[batch * inChannels * height * width];
        for (int i = 0; i < locks.Length; i++) locks[i] = new object();

        Parallel.For(0, batch * outChannels, idx =>
        {
            int b = idx / outChannels;
            int oc = idx % outChannels;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                    T gradOutVal = gradOutputData[gradOutIdx];

                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                int kernelPosIdx = kh * kernelWidth + kw;
                                int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;

                                // Get offsets
                                int offsetYIdx = ((b * (2 * numKernelPositions) + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;
                                int offsetXIdx = ((b * (2 * numKernelPositions) + numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;

                                double offsetY = numOps.ToDouble(offsetData[offsetYIdx]);
                                double offsetX = numOps.ToDouble(offsetData[offsetXIdx]);

                                double sampledH = oh * strideH - padH + kh * dilationH + offsetY;
                                double sampledW = ow * strideW - padW + kw * dilationW + offsetX;

                                // Get modulation mask
                                T modulation = numOps.One;
                                if (maskData != null)
                                {
                                    int maskIdx = ((b * numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;
                                    modulation = maskData[maskIdx];
                                }

                                // Gradient multiplied by kernel weight and modulation
                                T gradVal = numOps.Multiply(numOps.Multiply(gradOutVal, kernelData[kernelIdx]), modulation);

                                // Distribute gradient to the 4 neighboring pixels using bilinear weights
                                BilinearBackwardInput(gradInputData, gradVal, b, ic, inChannels, height, width, sampledH, sampledW, numOps, locks);
                            }
                        }
                    }
                }
            }
        });

        return gradInput;
    }

    /// <summary>
    /// Distributes gradient to input using bilinear interpolation weights.
    /// </summary>
    private static void BilinearBackwardInput<T>(
        T[] gradData,
        T gradVal,
        int batch,
        int channel,
        int totalChannels,
        int height,
        int width,
        double h,
        double w,
        INumericOperations<T> numOps,
        object[] locks)
    {
        if (h <= -1 || h >= height || w <= -1 || w >= width)
            return;

        int h0 = (int)Math.Floor(h);
        int h1 = h0 + 1;
        int w0 = (int)Math.Floor(w);
        int w1 = w0 + 1;

        double lh = h - h0;
        double lw = w - w0;

        double w00 = (1 - lh) * (1 - lw);
        double w01 = (1 - lh) * lw;
        double w10 = lh * (1 - lw);
        double w11 = lh * lw;

        // Add gradient contributions to each of the 4 corners
        AddGradientToPixel(gradData, gradVal, batch, channel, totalChannels, height, width, h0, w0, w00, numOps, locks);
        AddGradientToPixel(gradData, gradVal, batch, channel, totalChannels, height, width, h0, w1, w01, numOps, locks);
        AddGradientToPixel(gradData, gradVal, batch, channel, totalChannels, height, width, h1, w0, w10, numOps, locks);
        AddGradientToPixel(gradData, gradVal, batch, channel, totalChannels, height, width, h1, w1, w11, numOps, locks);
    }

    /// <summary>
    /// Atomically adds gradient to a pixel position with boundary checking.
    /// </summary>
    private static void AddGradientToPixel<T>(
        T[] gradData,
        T gradVal,
        int batch,
        int channel,
        int totalChannels,
        int height,
        int width,
        int h,
        int w,
        double weight,
        INumericOperations<T> numOps,
        object[] locks)
    {
        if (h < 0 || h >= height || w < 0 || w >= width || Math.Abs(weight) < 1e-10)
            return;

        int idx = ((batch * totalChannels + channel) * height + h) * width + w;
        T contrib = numOps.Multiply(gradVal, numOps.FromDouble(weight));

        lock (locks[idx])
        {
            gradData[idx] = numOps.Add(gradData[idx], contrib);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> DeformableConv2DBackwardKernel<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] kernelShape,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (offset == null) throw new ArgumentNullException(nameof(offset));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        int outChannels = kernelShape[0];
        int kernelHeight = kernelShape[2];
        int kernelWidth = kernelShape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];
        int numKernelPositions = kernelHeight * kernelWidth;

        var gradKernel = TensorAllocator.RentUninitialized<T>(kernelShape);
        var gradKernelData = gradKernel.GetDataArray();
        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetDataArray();
        var offsetData = offset.GetDataArray();
        var maskData = mask?.GetDataArray();

        // Use locks for atomic operations on kernel gradients
        var locks = new object[outChannels * inChannels * kernelHeight * kernelWidth];
        for (int i = 0; i < locks.Length; i++) locks[i] = new object();

        Parallel.For(0, batch, b =>
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                        T gradOutVal = gradOutputData[gradOutIdx];

                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int kernelPosIdx = kh * kernelWidth + kw;
                                    int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;

                                    // Get offsets
                                    int offsetYIdx = ((b * (2 * numKernelPositions) + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;
                                    int offsetXIdx = ((b * (2 * numKernelPositions) + numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;

                                    double offsetY = numOps.ToDouble(offsetData[offsetYIdx]);
                                    double offsetX = numOps.ToDouble(offsetData[offsetXIdx]);

                                    double sampledH = oh * strideH - padH + kh * dilationH + offsetY;
                                    double sampledW = ow * strideW - padW + kw * dilationW + offsetX;

                                    // Sample input with bilinear interpolation
                                    T sampledValue = BilinearSample(inputData, b, ic, inChannels, height, width, sampledH, sampledW, numOps);

                                    // Apply modulation mask
                                    if (maskData != null)
                                    {
                                        int maskIdx = ((b * numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;
                                        sampledValue = numOps.Multiply(sampledValue, maskData[maskIdx]);
                                    }

                                    // gradKernel[oc, ic, kh, kw] += gradOutput * sampledValue
                                    T contrib = numOps.Multiply(gradOutVal, sampledValue);

                                    lock (locks[kernelIdx])
                                    {
                                        gradKernelData[kernelIdx] = numOps.Add(gradKernelData[kernelIdx], contrib);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        return gradKernel;
    }

    /// <inheritdoc/>
    public Tensor<T> DeformableConv2DBackwardOffset<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (offset == null) throw new ArgumentNullException(nameof(offset));

        // Validate tensor ranks (must be 4D for deformable conv)
        if (gradOutput.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardOffset requires 4D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (input.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardOffset requires 4D input tensor. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardOffset requires 4D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));
        if (offset.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardOffset requires 4D offset tensor. Got rank {offset.Rank}.", nameof(offset));
        if (mask != null && mask.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardOffset requires 4D mask tensor. Got rank {mask.Rank}.", nameof(mask));

        // Validate parameter arrays
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements.", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements.", nameof(padding));
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements.", nameof(dilation));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        // Validate batch consistency
        if (gradOutput._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: input has {batch}, gradOutput has {gradOutput._shape[0]}.", nameof(gradOutput));
        if (offset._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: input has {batch}, offset has {offset._shape[0]}.", nameof(offset));
        if (mask != null && mask._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: input has {batch}, mask has {mask._shape[0]}.", nameof(mask));

        int outChannels = kernel._shape[0];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];
        int numKernelPositions = kernelHeight * kernelWidth;

        var gradOffset = TensorAllocator.RentUninitialized<T>(offset._shape);
        var gradOffsetData = gradOffset.GetDataArray();
        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetDataArray();
        var kernelData = kernel.GetDataArray();
        var offsetData = offset.GetDataArray();
        var maskData = mask?.GetDataArray();

        Parallel.For(0, batch, b =>
        {
            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            int kernelPosIdx = kh * kernelWidth + kw;

                            // Get current offsets
                            int offsetYIdx = ((b * (2 * numKernelPositions) + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;
                            int offsetXIdx = ((b * (2 * numKernelPositions) + numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;

                            double offsetY = numOps.ToDouble(offsetData[offsetYIdx]);
                            double offsetX = numOps.ToDouble(offsetData[offsetXIdx]);

                            double sampledH = oh * strideH - padH + kh * dilationH + offsetY;
                            double sampledW = ow * strideW - padW + kw * dilationW + offsetX;

                            // Get modulation mask
                            T modulation = numOps.One;
                            if (maskData != null)
                            {
                                int maskIdx = ((b * numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;
                                modulation = maskData[maskIdx];
                            }

                            // Compute gradient w.r.t. offset by chain rule through bilinear interpolation
                            // d(output)/d(offsetY) = sum over (oc, ic) of gradOutput * kernel * d(bilinear)/d(h)
                            // d(output)/d(offsetX) = sum over (oc, ic) of gradOutput * kernel * d(bilinear)/d(w)

                            T gradOffsetY = numOps.Zero;
                            T gradOffsetX = numOps.Zero;

                            for (int oc = 0; oc < outChannels; oc++)
                            {
                                int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                T gradOutVal = gradOutputData[gradOutIdx];

                                for (int ic = 0; ic < inChannels; ic++)
                                {
                                    int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                                    T kernelWeight = kernelData[kernelIdx];

                                    // Compute d(bilinear)/d(h) and d(bilinear)/d(w)
                                    var (dH, dW) = BilinearGradient(inputData, b, ic, inChannels, height, width, sampledH, sampledW, numOps);

                                    // Chain rule: gradOutput * kernel * modulation * d(bilinear)/d(offset)
                                    T factor = numOps.Multiply(numOps.Multiply(gradOutVal, kernelWeight), modulation);
                                    gradOffsetY = numOps.Add(gradOffsetY, numOps.Multiply(factor, dH));
                                    gradOffsetX = numOps.Add(gradOffsetX, numOps.Multiply(factor, dW));
                                }
                            }

                            gradOffsetData[offsetYIdx] = gradOffsetY;
                            gradOffsetData[offsetXIdx] = gradOffsetX;
                        }
                    }
                }
            }
        });

        return gradOffset;
    }

    /// <summary>
    /// Computes the gradient of bilinear interpolation w.r.t. sampling position.
    /// </summary>
    private static (T gradH, T gradW) BilinearGradient<T>(
        T[] data,
        int batch,
        int channel,
        int totalChannels,
        int height,
        int width,
        double h,
        double w,
        INumericOperations<T> numOps)
    {
        // Out of bounds - no gradient
        if (h <= -1 || h >= height || w <= -1 || w >= width)
            return (numOps.Zero, numOps.Zero);

        int h0 = (int)Math.Floor(h);
        int h1 = h0 + 1;
        int w0 = (int)Math.Floor(w);
        int w1 = w0 + 1;

        double lh = h - h0;
        double lw = w - w0;

        T v00 = GetPixelValue(data, batch, channel, totalChannels, height, width, h0, w0, numOps);
        T v01 = GetPixelValue(data, batch, channel, totalChannels, height, width, h0, w1, numOps);
        T v10 = GetPixelValue(data, batch, channel, totalChannels, height, width, h1, w0, numOps);
        T v11 = GetPixelValue(data, batch, channel, totalChannels, height, width, h1, w1, numOps);

        // d/dh: derivative of bilinear w.r.t. h
        // output = (1-lh)*(1-lw)*v00 + (1-lh)*lw*v01 + lh*(1-lw)*v10 + lh*lw*v11
        // d/dlh = -(1-lw)*v00 - lw*v01 + (1-lw)*v10 + lw*v11
        //       = (1-lw)*(v10 - v00) + lw*(v11 - v01)
        T dH = numOps.Add(
            numOps.Multiply(numOps.FromDouble(1 - lw), numOps.Subtract(v10, v00)),
            numOps.Multiply(numOps.FromDouble(lw), numOps.Subtract(v11, v01)));

        // d/dw: derivative of bilinear w.r.t. w
        // d/dlw = -(1-lh)*v00 + (1-lh)*v01 - lh*v10 + lh*v11
        //       = (1-lh)*(v01 - v00) + lh*(v11 - v10)
        T dW = numOps.Add(
            numOps.Multiply(numOps.FromDouble(1 - lh), numOps.Subtract(v01, v00)),
            numOps.Multiply(numOps.FromDouble(lh), numOps.Subtract(v11, v10)));

        return (dH, dW);
    }

    /// <summary>
    /// Distributes gradient to input using bilinear interpolation weights (NHWC layout).
    /// </summary>
    private static void BilinearBackwardInputNHWC<T>(
        T[] gradData,
        T gradVal,
        int batch,
        int channel,
        int height,
        int width,
        int channels,
        double h,
        double w,
        INumericOperations<T> numOps,
        object[] locks)
    {
        if (h <= -1 || h >= height || w <= -1 || w >= width)
            return;

        int h0 = (int)Math.Floor(h);
        int h1 = h0 + 1;
        int w0 = (int)Math.Floor(w);
        int w1 = w0 + 1;

        double lh = h - h0;
        double lw = w - w0;

        double w00 = (1 - lh) * (1 - lw);
        double w01 = (1 - lh) * lw;
        double w10 = lh * (1 - lw);
        double w11 = lh * lw;

        // Add gradient contributions to each of the 4 corners (NHWC layout)
        AddGradientToPixelNHWC(gradData, gradVal, batch, channel, height, width, channels, h0, w0, w00, numOps, locks);
        AddGradientToPixelNHWC(gradData, gradVal, batch, channel, height, width, channels, h0, w1, w01, numOps, locks);
        AddGradientToPixelNHWC(gradData, gradVal, batch, channel, height, width, channels, h1, w0, w10, numOps, locks);
        AddGradientToPixelNHWC(gradData, gradVal, batch, channel, height, width, channels, h1, w1, w11, numOps, locks);
    }

    /// <summary>
    /// Atomically adds gradient to a pixel position with boundary checking (NHWC layout).
    /// </summary>
    private static void AddGradientToPixelNHWC<T>(
        T[] gradData,
        T gradVal,
        int batch,
        int channel,
        int height,
        int width,
        int channels,
        int h,
        int w,
        double weight,
        INumericOperations<T> numOps,
        object[] locks)
    {
        if (h < 0 || h >= height || w < 0 || w >= width || Math.Abs(weight) < 1e-10)
            return;

        // NHWC index: [batch, h, w, channel]
        int idx = ((batch * height + h) * width + w) * channels + channel;
        T contrib = numOps.Multiply(gradVal, numOps.FromDouble(weight));

        lock (locks[idx])
        {
            gradData[idx] = numOps.Add(gradData[idx], contrib);
        }
    }

    /// <summary>
    /// Gets a pixel value with boundary checking for NHWC layout (returns zero for out-of-bounds).
    /// </summary>
    private static T GetPixelValueNHWC<T>(
        T[] data,
        int batch,
        int channel,
        int height,
        int width,
        int channels,
        int h,
        int w,
        INumericOperations<T> numOps)
    {
        if (h < 0 || h >= height || w < 0 || w >= width)
            return numOps.Zero;

        // NHWC index: [batch, h, w, channel]
        int idx = ((batch * height + h) * width + w) * channels + channel;
        return data[idx];
    }

    /// <summary>
    /// Computes the gradient of bilinear interpolation w.r.t. sampling position (NHWC layout).
    /// </summary>
    private static (T gradH, T gradW) BilinearGradientNHWC<T>(
        T[] data,
        int batch,
        int channel,
        int height,
        int width,
        int channels,
        double h,
        double w,
        INumericOperations<T> numOps)
    {
        // Out of bounds - no gradient
        if (h <= -1 || h >= height || w <= -1 || w >= width)
            return (numOps.Zero, numOps.Zero);

        int h0 = (int)Math.Floor(h);
        int h1 = h0 + 1;
        int w0 = (int)Math.Floor(w);
        int w1 = w0 + 1;

        double lh = h - h0;
        double lw = w - w0;

        T v00 = GetPixelValueNHWC(data, batch, channel, height, width, channels, h0, w0, numOps);
        T v01 = GetPixelValueNHWC(data, batch, channel, height, width, channels, h0, w1, numOps);
        T v10 = GetPixelValueNHWC(data, batch, channel, height, width, channels, h1, w0, numOps);
        T v11 = GetPixelValueNHWC(data, batch, channel, height, width, channels, h1, w1, numOps);

        // d/dh: derivative of bilinear w.r.t. h
        T dH = numOps.Add(
            numOps.Multiply(numOps.FromDouble(1 - lw), numOps.Subtract(v10, v00)),
            numOps.Multiply(numOps.FromDouble(lw), numOps.Subtract(v11, v01)));

        // d/dw: derivative of bilinear w.r.t. w
        T dW = numOps.Add(
            numOps.Multiply(numOps.FromDouble(1 - lh), numOps.Subtract(v01, v00)),
            numOps.Multiply(numOps.FromDouble(lh), numOps.Subtract(v11, v10)));

        return (dH, dW);
    }

    /// <inheritdoc/>
    public Tensor<T> DeformableConv2DBackwardMask<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T> offset,
        Tensor<T>? mask,
        int[] stride,
        int[] padding,
        int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (offset == null) throw new ArgumentNullException(nameof(offset));

        // Validate tensor ranks (must be 4D for deformable conv)
        if (gradOutput.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardMask requires 4D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (input.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardMask requires 4D input tensor. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardMask requires 4D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));
        if (offset.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardMask requires 4D offset tensor. Got rank {offset.Rank}.", nameof(offset));
        if (mask != null && mask.Rank != 4) throw new ArgumentException($"DeformableConv2DBackwardMask requires 4D mask tensor. Got rank {mask.Rank}.", nameof(mask));

        // Validate parameter arrays
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements.", nameof(stride));
        if (padding == null || padding.Length != 2) throw new ArgumentException("Padding must be array of 2 elements.", nameof(padding));
        if (dilation == null || dilation.Length != 2) throw new ArgumentException("Dilation must be array of 2 elements.", nameof(dilation));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];

        // Validate batch consistency
        if (gradOutput._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: input has {batch}, gradOutput has {gradOutput._shape[0]}.", nameof(gradOutput));
        if (offset._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: input has {batch}, offset has {offset._shape[0]}.", nameof(offset));
        if (mask != null && mask._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: input has {batch}, mask has {mask._shape[0]}.", nameof(mask));

        int outChannels = kernel._shape[0];
        int kernelHeight = kernel._shape[2];
        int kernelWidth = kernel._shape[3];

        int strideH = stride[0], strideW = stride[1];
        int padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];

        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];
        int numKernelPositions = kernelHeight * kernelWidth;

        // Mask shape: [batch, kernel_h*kernel_w, out_h, out_w]
        var gradMask = TensorAllocator.Rent<T>([batch, numKernelPositions, outputHeight, outputWidth]);
        var gradMaskData = gradMask.GetDataArray();
        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetDataArray();
        var kernelData = kernel.GetDataArray();
        var offsetData = offset.GetDataArray();

        Parallel.For(0, batch, b =>
        {
            for (int oh = 0; oh < outputHeight; oh++)
            {
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    for (int kh = 0; kh < kernelHeight; kh++)
                    {
                        for (int kw = 0; kw < kernelWidth; kw++)
                        {
                            int kernelPosIdx = kh * kernelWidth + kw;
                            int maskIdx = ((b * numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;

                            // Get offsets
                            int offsetYIdx = ((b * (2 * numKernelPositions) + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;
                            int offsetXIdx = ((b * (2 * numKernelPositions) + numKernelPositions + kernelPosIdx) * outputHeight + oh) * outputWidth + ow;

                            double offsetY = numOps.ToDouble(offsetData[offsetYIdx]);
                            double offsetX = numOps.ToDouble(offsetData[offsetXIdx]);

                            double sampledH = oh * strideH - padH + kh * dilationH + offsetY;
                            double sampledW = ow * strideW - padW + kw * dilationW + offsetX;

                            // d(output)/d(mask[b,k,oh,ow]) = sum over (oc, ic) of gradOutput * kernel * sampledValue
                            T gradMaskVal = numOps.Zero;

                            for (int oc = 0; oc < outChannels; oc++)
                            {
                                int gradOutIdx = ((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow;
                                T gradOutVal = gradOutputData[gradOutIdx];

                                for (int ic = 0; ic < inChannels; ic++)
                                {
                                    int kernelIdx = ((oc * inChannels + ic) * kernelHeight + kh) * kernelWidth + kw;
                                    T kernelWeight = kernelData[kernelIdx];

                                    // Sample input
                                    T sampledValue = BilinearSample(inputData, b, ic, inChannels, height, width, sampledH, sampledW, numOps);

                                    gradMaskVal = numOps.Add(gradMaskVal, numOps.Multiply(numOps.Multiply(gradOutVal, kernelWeight), sampledValue));
                                }
                            }

                            gradMaskData[maskIdx] = gradMaskVal;
                        }
                    }
                }
            }
        });

        return gradMask;
    }

    #endregion

    #region Grid Sample Backward Operations

    /// <inheritdoc/>
    public Tensor<T> GridSampleBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> grid, int[] inputShape)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!grid.IsContiguous) grid = grid.Contiguous();
        if (inputShape == null || inputShape.Length != 4) throw new ArgumentException("inputShape must be array of 4 elements [batch, height, width, channels].", nameof(inputShape));

        // Validate tensor ranks for NHWC layout
        if (gradOutput.Rank != 4) throw new ArgumentException($"GridSampleBackwardInput requires 4D gradOutput tensor (NHWC). Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (grid.Rank != 4) throw new ArgumentException($"GridSampleBackwardInput requires 4D grid tensor [batch, outH, outW, 2]. Got rank {grid.Rank}.", nameof(grid));

        var numOps = MathHelper.GetNumericOperations<T>();

        // NHWC layout: [batch, height, width, channels]
        int batch = inputShape[0];
        int height = inputShape[1];
        int width = inputShape[2];
        int channels = inputShape[3];

        int outHeight = grid._shape[1];
        int outWidth = grid._shape[2];

        // Validate shape consistency
        if (grid._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: inputShape has {batch}, grid has {grid._shape[0]}.", nameof(grid));
        if (grid._shape[3] != 2) throw new ArgumentException($"Grid last dimension must be 2 (x,y coordinates). Got {grid._shape[3]}.", nameof(grid));
        if (gradOutput._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: inputShape has {batch}, gradOutput has {gradOutput._shape[0]}.", nameof(gradOutput));
        if (gradOutput._shape[1] != outHeight || gradOutput._shape[2] != outWidth)
            throw new ArgumentException($"gradOutput spatial dims [{gradOutput._shape[1]},{gradOutput._shape[2]}] must match grid spatial dims [{outHeight},{outWidth}].", nameof(gradOutput));
        if (gradOutput._shape[3] != channels) throw new ArgumentException($"Channel mismatch: inputShape has {channels}, gradOutput has {gradOutput._shape[3]}.", nameof(gradOutput));

        var gradInput = TensorAllocator.RentUninitialized<T>(inputShape);
        var gradInputData = gradInput.GetDataArray();
        var gradOutputData = gradOutput.GetDataArray();
        var gridData = grid.GetDataArray();

        // Use locks for atomic addition - NHWC layout
        var locks = new object[batch * height * width * channels];
        for (int i = 0; i < locks.Length; i++) locks[i] = new object();

        Parallel.For(0, batch, b =>
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    // Get normalized grid coordinates
                    int gridBaseIdx = ((b * outHeight + oh) * outWidth + ow) * 2;
                    double gx = numOps.ToDouble(gridData[gridBaseIdx]);
                    double gy = numOps.ToDouble(gridData[gridBaseIdx + 1]);

                    // Convert to pixel coordinates
                    double srcH = (gy + 1) / 2 * (height - 1);
                    double srcW = (gx + 1) / 2 * (width - 1);

                    // Distribute gradient for all channels using NHWC bilinear weights
                    for (int c = 0; c < channels; c++)
                    {
                        // NHWC gradOutput index: [b, oh, ow, c]
                        int gradOutIdx = ((b * outHeight + oh) * outWidth + ow) * channels + c;
                        T gradVal = gradOutputData[gradOutIdx];

                        BilinearBackwardInputNHWC(gradInputData, gradVal, b, c, height, width, channels, srcH, srcW, numOps, locks);
                    }
                }
            }
        });

        return gradInput;
    }

    /// <inheritdoc/>
    public Tensor<T> GridSampleBackwardGrid<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> grid)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (!grid.IsContiguous) grid = grid.Contiguous();
        if (!input.IsContiguous) input = input.Contiguous();
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();

        // Validate tensor ranks for NHWC layout
        if (gradOutput.Rank != 4) throw new ArgumentException($"GridSampleBackwardGrid requires 4D gradOutput tensor (NHWC). Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (input.Rank != 4) throw new ArgumentException($"GridSampleBackwardGrid requires 4D input tensor (NHWC). Got rank {input.Rank}.", nameof(input));
        if (grid.Rank != 4) throw new ArgumentException($"GridSampleBackwardGrid requires 4D grid tensor [batch, outH, outW, 2]. Got rank {grid.Rank}.", nameof(grid));

        var numOps = MathHelper.GetNumericOperations<T>();

        // NHWC layout: [batch, height, width, channels]
        int batch = input._shape[0];
        int height = input._shape[1];
        int width = input._shape[2];
        int channels = input._shape[3];

        int outHeight = grid._shape[1];
        int outWidth = grid._shape[2];

        // Validate shape consistency
        if (grid._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: input has {batch}, grid has {grid._shape[0]}.", nameof(grid));
        if (grid._shape[3] != 2) throw new ArgumentException($"Grid last dimension must be 2 (x,y coordinates). Got {grid._shape[3]}.", nameof(grid));
        if (gradOutput._shape[0] != batch) throw new ArgumentException($"Batch size mismatch: input has {batch}, gradOutput has {gradOutput._shape[0]}.", nameof(gradOutput));
        if (gradOutput._shape[1] != outHeight || gradOutput._shape[2] != outWidth)
            throw new ArgumentException($"gradOutput spatial dims [{gradOutput._shape[1]},{gradOutput._shape[2]}] must match grid spatial dims [{outHeight},{outWidth}].", nameof(gradOutput));
        if (gradOutput._shape[3] != channels) throw new ArgumentException($"Channel mismatch: input has {channels}, gradOutput has {gradOutput._shape[3]}.", nameof(gradOutput));

        var gradGrid = TensorAllocator.RentUninitialized<T>(grid._shape);
        var gradGridData = gradGrid.GetDataArray();
        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetDataArray();
        var gridData = grid.GetDataArray();

        Parallel.For(0, batch, b =>
        {
            for (int oh = 0; oh < outHeight; oh++)
            {
                for (int ow = 0; ow < outWidth; ow++)
                {
                    int gridBaseIdx = ((b * outHeight + oh) * outWidth + ow) * 2;
                    double gx = numOps.ToDouble(gridData[gridBaseIdx]);
                    double gy = numOps.ToDouble(gridData[gridBaseIdx + 1]);

                    // Convert to pixel coordinates
                    double srcH = (gy + 1) / 2 * (height - 1);
                    double srcW = (gx + 1) / 2 * (width - 1);

                    // Compute d(output)/d(gx) and d(output)/d(gy) by summing over all channels
                    T gradGx = numOps.Zero;
                    T gradGy = numOps.Zero;

                    for (int c = 0; c < channels; c++)
                    {
                        // NHWC gradOutput index: [b, oh, ow, c]
                        int gradOutIdx = ((b * outHeight + oh) * outWidth + ow) * channels + c;
                        T gradOutVal = gradOutputData[gradOutIdx];

                        // Get gradient of bilinear sampling w.r.t. h and w using NHWC layout
                        var (dH, dW) = BilinearGradientNHWC(inputData, b, c, height, width, channels, srcH, srcW, numOps);

                        // Chain rule: d/dgx = d/dw * dw/dgx where dw/dgx = (width-1)/2
                        // Chain rule: d/dgy = d/dh * dh/dgy where dh/dgy = (height-1)/2
                        T scaledDW = numOps.Multiply(dW, numOps.FromDouble((width - 1) / 2.0));
                        T scaledDH = numOps.Multiply(dH, numOps.FromDouble((height - 1) / 2.0));

                        gradGx = numOps.Add(gradGx, numOps.Multiply(gradOutVal, scaledDW));
                        gradGy = numOps.Add(gradGy, numOps.Multiply(gradOutVal, scaledDH));
                    }

                    gradGridData[gridBaseIdx] = gradGx;
                    gradGridData[gridBaseIdx + 1] = gradGy;
                }
            }
        });

        return gradGrid;
    }

    #endregion

    #region 3D Convolution and Pooling Operations

    /// <inheritdoc/>
    public virtual Tensor<T> Conv3D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1)
    {
        return Conv3D(input, kernel, [stride, stride, stride], [padding, padding, padding], [dilation, dilation, dilation]);
    }

    /// <inheritdoc/>
    public Tensor<T> Conv3D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] dilation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank != 5) throw new ArgumentException($"Conv3D requires 5D input tensor [batch, in_channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 5) throw new ArgumentException($"Conv3D requires 5D kernel tensor [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements [strideD, strideH, strideW].", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0) throw new ArgumentException("Stride elements must be positive.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements [padD, padH, padW].", nameof(padding));
        if (dilation == null || dilation.Length != 3) throw new ArgumentException("Dilation must be array of 3 elements [dilationD, dilationH, dilationW].", nameof(dilation));
        if (dilation[0] <= 0 || dilation[1] <= 0 || dilation[2] <= 0) throw new ArgumentException("Dilation elements must be positive.", nameof(dilation));
        if (input._shape[1] != kernel._shape[1]) throw new ArgumentException($"Input channels ({input._shape[1]}) must match kernel in_channels ({kernel._shape[1]}).");

        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];
        int dilationD = dilation[0], dilationH = dilation[1], dilationW = dilation[2];

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int depth = input._shape[2];
        int height = input._shape[3];
        int width = input._shape[4];

        int outChannels = kernel._shape[0];
        int kernelDepth = kernel._shape[2];
        int kernelHeight = kernel._shape[3];
        int kernelWidth = kernel._shape[4];

        int effectiveKernelD = dilationD * (kernelDepth - 1) + 1;
        int effectiveKernelH = dilationH * (kernelHeight - 1) + 1;
        int effectiveKernelW = dilationW * (kernelWidth - 1) + 1;

        int outputDepth = (depth + 2 * padD - effectiveKernelD) / strideD + 1;
        int outputHeight = (height + 2 * padH - effectiveKernelH) / strideH + 1;
        int outputWidth = (width + 2 * padW - effectiveKernelW) / strideW + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid output dimensions ({outputDepth}x{outputHeight}x{outputWidth}). " +
                $"Check kernel size, stride, padding, and dilation parameters for input size {depth}x{height}x{width}.");
        }

        var result = TensorAllocator.Rent<T>([batch, outChannels, outputDepth, outputHeight, outputWidth]);
        var inputData = input.GetDataArray();
        var kernelData = kernel.GetDataArray();
        var outputData = result.GetDataArray();

        // Parallel over batch * outChannels for maximum parallelism
        Parallel.For(0, batch * outChannels, idx =>
        {
            int b = idx / outChannels;
            int oc = idx % outChannels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;

                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kd = 0; kd < kernelDepth; kd++)
                            {
                                int id = od * strideD + kd * dilationD - padD;
                                if (id < 0 || id >= depth) continue;

                                for (int kh = 0; kh < kernelHeight; kh++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    if (ih < 0 || ih >= height) continue;

                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int iw = ow * strideW + kw * dilationW - padW;
                                        if (iw < 0 || iw >= width) continue;

                                        int inputIdx = (((b * inChannels + ic) * depth + id) * height + ih) * width + iw;
                                        int kernelIdx = (((oc * inChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw;
                                        sum = numOps.Add(sum, numOps.Multiply(inputData[inputIdx], kernelData[kernelIdx]));
                                    }
                                }
                            }
                        }

                        int outputIdx = (((b * outChannels + oc) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIdx] = sum;
                    }
                }
            }
        });

        DifferentiableOps.RecordBinary("Conv3D", result, input, kernel, BackwardFunctions<T>.Conv3DBackward,
            new object[] { stride, padding, dilation });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Conv3DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements [batch, in_channels, depth, height, width].", nameof(inputShape));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements.", nameof(padding));
        if (dilation == null || dilation.Length != 3) throw new ArgumentException("Dilation must be array of 3 elements.", nameof(dilation));

        // Rank and shape validation
        if (gradOutput.Rank != 5) throw new ArgumentException($"Conv3DBackwardInput requires 5D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (kernel.Rank != 5) throw new ArgumentException($"Conv3DBackwardInput requires 5D kernel tensor. Got rank {kernel.Rank}.", nameof(kernel));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int depth = inputShape[2];
        int height = inputShape[3];
        int width = inputShape[4];

        int outChannels = kernel._shape[0];
        int kernelDepth = kernel._shape[2];
        int kernelHeight = kernel._shape[3];
        int kernelWidth = kernel._shape[4];

        // Validate shape consistency
        if (gradOutput._shape[0] != batch)
            throw new ArgumentException($"gradOutput batch size ({gradOutput._shape[0]}) must match inputShape batch size ({batch}).", nameof(gradOutput));
        if (gradOutput._shape[1] != outChannels)
            throw new ArgumentException($"gradOutput outChannels ({gradOutput._shape[1]}) must match kernel out_channels ({outChannels}).", nameof(gradOutput));
        if (inputShape[1] != kernel._shape[1])
            throw new ArgumentException($"inputShape in_channels ({inputShape[1]}) must match kernel in_channels ({kernel._shape[1]}).", nameof(inputShape));

        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];
        int dilationD = dilation[0], dilationH = dilation[1], dilationW = dilation[2];

        int outputDepth = gradOutput._shape[2];
        int outputHeight = gradOutput._shape[3];
        int outputWidth = gradOutput._shape[4];

        var gradInputData = new T[batch * inChannels * depth * height * width];
        var gradOutputData = gradOutput.GetDataArray();
        var kernelData = kernel.GetDataArray();

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Parallel over (batch, inChannels) - each pair owns disjoint gradInput slices
        // so direct writes are race-free without thread-local buffers
        Parallel.For(0, batch * inChannels, idx =>
        {
            int b = idx / inChannels;
            int ic = idx % inChannels;

            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int od = 0; od < outputDepth; od++)
                {
                    for (int oh = 0; oh < outputHeight; oh++)
                    {
                        for (int ow = 0; ow < outputWidth; ow++)
                        {
                            int gradOutIdx = (((b * outChannels + oc) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                            T gradVal = gradOutputData[gradOutIdx];

                            for (int kd = 0; kd < kernelDepth; kd++)
                            {
                                int id = od * strideD + kd * dilationD - padD;
                                if (id < 0 || id >= depth) continue;

                                for (int kh = 0; kh < kernelHeight; kh++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    if (ih < 0 || ih >= height) continue;

                                    for (int kw = 0; kw < kernelWidth; kw++)
                                    {
                                        int iw = ow * strideW + kw * dilationW - padW;
                                        if (iw < 0 || iw >= width) continue;

                                        int inputIdx = (((b * inChannels + ic) * depth + id) * height + ih) * width + iw;
                                        int kernelIdx = (((oc * inChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw;
                                        gradInputData[inputIdx] = numOps.Add(gradInputData[inputIdx], numOps.Multiply(gradVal, kernelData[kernelIdx]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Conv3DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelShape == null || kernelShape.Length != 5) throw new ArgumentException("Kernel shape must be array of 5 elements [out_channels, in_channels, kernel_depth, kernel_height, kernel_width].", nameof(kernelShape));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements.", nameof(padding));
        if (dilation == null || dilation.Length != 3) throw new ArgumentException("Dilation must be array of 3 elements.", nameof(dilation));

        // Rank and shape validation
        if (gradOutput.Rank != 5) throw new ArgumentException($"Conv3DBackwardKernel requires 5D gradOutput tensor. Got rank {gradOutput.Rank}.", nameof(gradOutput));
        if (input.Rank != 5) throw new ArgumentException($"Conv3DBackwardKernel requires 5D input tensor. Got rank {input.Rank}.", nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int depth = input._shape[2];
        int height = input._shape[3];
        int width = input._shape[4];

        int outChannels = kernelShape[0];
        int kernelDepth = kernelShape[2];
        int kernelHeight = kernelShape[3];
        int kernelWidth = kernelShape[4];

        // Validate shape consistency
        if (gradOutput._shape[0] != batch)
            throw new ArgumentException($"gradOutput batch size ({gradOutput._shape[0]}) must match input batch size ({batch}).", nameof(gradOutput));
        if (gradOutput._shape[1] != outChannels)
            throw new ArgumentException($"gradOutput outChannels ({gradOutput._shape[1]}) must match kernelShape out_channels ({outChannels}).", nameof(gradOutput));
        if (input._shape[1] != kernelShape[1])
            throw new ArgumentException($"input in_channels ({input._shape[1]}) must match kernelShape in_channels ({kernelShape[1]}).", nameof(input));

        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];
        int dilationD = dilation[0], dilationH = dilation[1], dilationW = dilation[2];

        int outputDepth = gradOutput._shape[2];
        int outputHeight = gradOutput._shape[3];
        int outputWidth = gradOutput._shape[4];

        var gradKernelData = new T[outChannels * inChannels * kernelDepth * kernelHeight * kernelWidth];
        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetDataArray();

        // Initialize to zero
        for (int i = 0; i < gradKernelData.Length; i++)
            gradKernelData[i] = numOps.Zero;

        // Parallel over outChannels * inChannels for kernel gradient computation
        Parallel.For(0, outChannels * inChannels, idx =>
        {
            int oc = idx / inChannels;
            int ic = idx % inChannels;

            for (int kd = 0; kd < kernelDepth; kd++)
            {
                for (int kh = 0; kh < kernelHeight; kh++)
                {
                    for (int kw = 0; kw < kernelWidth; kw++)
                    {
                        T sum = numOps.Zero;

                        for (int b = 0; b < batch; b++)
                        {
                            for (int od = 0; od < outputDepth; od++)
                            {
                                int id = od * strideD + kd * dilationD - padD;
                                if (id < 0 || id >= depth) continue;

                                for (int oh = 0; oh < outputHeight; oh++)
                                {
                                    int ih = oh * strideH + kh * dilationH - padH;
                                    if (ih < 0 || ih >= height) continue;

                                    for (int ow = 0; ow < outputWidth; ow++)
                                    {
                                        int iw = ow * strideW + kw * dilationW - padW;
                                        if (iw < 0 || iw >= width) continue;

                                        int gradOutIdx = (((b * outChannels + oc) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                                        int inputIdx = (((b * inChannels + ic) * depth + id) * height + ih) * width + iw;
                                        sum = numOps.Add(sum, numOps.Multiply(gradOutputData[gradOutIdx], inputData[inputIdx]));
                                    }
                                }
                            }
                        }

                        int kernelIdx = (((oc * inChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth + kw;
                        gradKernelData[kernelIdx] = sum;
                    }
                }
            }
        });

        return TensorAllocator.Rent<T>(kernelShape, new Vector<T>(gradKernelData));
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool3D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (stride == 0) stride = poolSize;
        return MaxPool3D(input, [poolSize, poolSize, poolSize], [stride, stride, stride], [padding, padding, padding]);
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding)
    {
        // Use the core implementation directly - we don't need indices for simple forward
        return MaxPool3DCore(input, poolSize, stride, padding);
    }

    /// <summary>
    /// Core implementation of MaxPool3D without index tracking for simple forward pass.
    /// </summary>
    private Tensor<T> MaxPool3DCore<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 5) throw new ArgumentException($"MaxPool3D requires 5D input tensor [batch, channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (poolSize == null || poolSize.Length != 3) throw new ArgumentException("Pool size must be array of 3 elements [poolD, poolH, poolW].", nameof(poolSize));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements [strideD, strideH, strideW].", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements [padD, padH, padW].", nameof(padding));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int channels = input._shape[1];
        int depth = input._shape[2];
        int height = input._shape[3];
        int width = input._shape[4];

        int poolD = poolSize[0], poolH = poolSize[1], poolW = poolSize[2];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];

        int outputDepth = (depth + 2 * padD - poolD) / strideD + 1;
        int outputHeight = (height + 2 * padH - poolH) / strideH + 1;
        int outputWidth = (width + 2 * padW - poolW) / strideW + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid output dimensions ({outputDepth}x{outputHeight}x{outputWidth}). " +
                $"Check pool size, stride, and padding parameters for input size {depth}x{height}x{width}.");
        }

        var result = TensorAllocator.Rent<T>([batch, channels, outputDepth, outputHeight, outputWidth]);
        var inputData = input.GetDataArray();
        var outputData = result.GetDataArray();

        // Parallel over batch * channels
        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T maxVal = numOps.MinValue;
                        bool foundValid = false;

                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd - padD;
                            if (id < 0 || id >= depth) continue;

                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph - padH;
                                if (ih < 0 || ih >= height) continue;

                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw - padW;
                                    if (iw < 0 || iw >= width) continue;

                                    int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                                    T val = inputData[inputIdx];
                                    if (!foundValid || numOps.GreaterThan(val, maxVal))
                                    {
                                        maxVal = val;
                                        foundValid = true;
                                    }
                                }
                            }
                        }

                        int outputIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIdx] = foundValid ? maxVal : numOps.Zero;
                    }
                }
            }
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool3DWithIndices<T>(Tensor<T> input, int[] poolSize, int[] stride, out int[,,,,,] maxIndices)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 5) throw new ArgumentException($"MaxPool3D requires 5D input tensor [batch, channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (poolSize == null || poolSize.Length != 3) throw new ArgumentException("Pool size must be array of 3 elements [poolD, poolH, poolW].", nameof(poolSize));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements [strideD, strideH, strideW].", nameof(stride));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int channels = input._shape[1];
        int depth = input._shape[2];
        int height = input._shape[3];
        int width = input._shape[4];

        int poolD = poolSize[0], poolH = poolSize[1], poolW = poolSize[2];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];

        int outputDepth = (depth - poolD) / strideD + 1;
        int outputHeight = (height - poolH) / strideH + 1;
        int outputWidth = (width - poolW) / strideW + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid output dimensions ({outputDepth}x{outputHeight}x{outputWidth}). " +
                $"Check pool size and stride parameters for input size {depth}x{height}x{width}.");
        }

        var result = TensorAllocator.Rent<T>([batch, channels, outputDepth, outputHeight, outputWidth]);
        var inputData = input.GetFlattenedData();
        var outputData = result.GetDataArray();
        var localMaxIndices = new int[batch, channels, outputDepth, outputHeight, outputWidth, 3];

        // Parallel over batch * channels
        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T maxVal = numOps.MinValue;
                        int maxId = 0, maxIh = 0, maxIw = 0;
                        bool foundValid = false;

                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd;
                            if (id >= depth) continue;

                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph;
                                if (ih >= height) continue;

                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw;
                                    if (iw >= width) continue;

                                    int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                                    T val = inputData[inputIdx];
                                    if (!foundValid || numOps.GreaterThan(val, maxVal))
                                    {
                                        maxVal = val;
                                        maxId = id;
                                        maxIh = ih;
                                        maxIw = iw;
                                        foundValid = true;
                                    }
                                }
                            }
                        }

                        int outputIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIdx] = foundValid ? maxVal : numOps.Zero;
                        localMaxIndices[b, c, od, oh, ow, 0] = maxId;
                        localMaxIndices[b, c, od, oh, ow, 1] = maxIh;
                        localMaxIndices[b, c, od, oh, ow, 2] = maxIw;
                    }
                }
            }
        });

        maxIndices = localMaxIndices;
        DifferentiableOps.RecordUnary("MaxPool3DWithIndices", result, input, BackwardFunctions<T>.MaxPool3DBackward, new object[] { maxIndices, poolSize, stride });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> MaxPool3DBackward<T>(Tensor<T> gradOutput, int[,,,,,] maxIndices, int[] inputShape, int[] poolSize, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (maxIndices == null) throw new ArgumentNullException(nameof(maxIndices));
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements [batch, channels, depth, height, width].", nameof(inputShape));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int depth = inputShape[2];
        int height = inputShape[3];
        int width = inputShape[4];

        int outputDepth = gradOutput._shape[2];
        int outputHeight = gradOutput._shape[3];
        int outputWidth = gradOutput._shape[4];

        var gradInputData = new T[batch * channels * depth * height * width];
        var gradOutputData = gradOutput.GetFlattenedData();

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Parallel over batch * channels - each (b, c) pair writes to a unique slice of gradInputData
        // so no thread-local buffers are needed (avoiding racy thread-ID based indexing)
        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        int gradOutIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        T gradVal = gradOutputData[gradOutIdx];

                        int id = maxIndices[b, c, od, oh, ow, 0];
                        int ih = maxIndices[b, c, od, oh, ow, 1];
                        int iw = maxIndices[b, c, od, oh, ow, 2];

                        int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                        gradInputData[inputIdx] = numOps.Add(gradInputData[inputIdx], gradVal);
                    }
                }
            }
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool3D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        if (stride == 0) stride = poolSize;
        return AvgPool3D(input, [poolSize, poolSize, poolSize], [stride, stride, stride], [padding, padding, padding]);
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();
        if (input.Rank != 5) throw new ArgumentException($"AvgPool3D requires 5D input tensor [batch, channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (poolSize == null || poolSize.Length != 3) throw new ArgumentException("Pool size must be array of 3 elements [poolD, poolH, poolW].", nameof(poolSize));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements [strideD, strideH, strideW].", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements [padD, padH, padW].", nameof(padding));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int channels = input._shape[1];
        int depth = input._shape[2];
        int height = input._shape[3];
        int width = input._shape[4];

        int poolD = poolSize[0], poolH = poolSize[1], poolW = poolSize[2];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];

        int outputDepth = (depth + 2 * padD - poolD) / strideD + 1;
        int outputHeight = (height + 2 * padH - poolH) / strideH + 1;
        int outputWidth = (width + 2 * padW - poolW) / strideW + 1;

        if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentException(
                $"Invalid output dimensions ({outputDepth}x{outputHeight}x{outputWidth}). " +
                $"Check pool size, stride, and padding parameters for input size {depth}x{height}x{width}.");
        }

        var result = TensorAllocator.Rent<T>([batch, channels, outputDepth, outputHeight, outputWidth]);
        var inputData = input.GetDataArray();
        var outputData = result.GetDataArray();

        // Parallel over batch * channels
        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;
                        int count = 0;

                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd - padD;
                            if (id < 0 || id >= depth) continue;

                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph - padH;
                                if (ih < 0 || ih >= height) continue;

                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw - padW;
                                    if (iw < 0 || iw >= width) continue;

                                    int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                                    sum = numOps.Add(sum, inputData[inputIdx]);
                                    count++;
                                }
                            }
                        }

                        int outputIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        outputData[outputIdx] = count > 0 ? numOps.Divide(sum, numOps.FromDouble(count)) : numOps.Zero;
                    }
                }
            }
        });

        DifferentiableOps.RecordUnary("AvgPool3D", result, input, BackwardFunctions<T>.AvgPool3DBackward, new object[] { poolSize, stride, padding });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> AvgPool3DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements [batch, channels, depth, height, width].", nameof(inputShape));
        if (poolSize == null || poolSize.Length != 3) throw new ArgumentException("Pool size must be array of 3 elements.", nameof(poolSize));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements.", nameof(padding));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int depth = inputShape[2];
        int height = inputShape[3];
        int width = inputShape[4];

        int poolD = poolSize[0], poolH = poolSize[1], poolW = poolSize[2];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];

        int outputDepth = gradOutput._shape[2];
        int outputHeight = gradOutput._shape[3];
        int outputWidth = gradOutput._shape[4];

        var gradInputData = new T[batch * channels * depth * height * width];
        var gradOutputData = gradOutput.GetFlattenedData();

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        // Parallel over batch * channels - each (b, c) pair writes to a unique slice of gradInputData
        // so no thread-local buffers are needed (avoiding racy thread-ID based indexing)
        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;

            for (int od = 0; od < outputDepth; od++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        // Count valid positions in this pool window
                        int count = 0;
                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd - padD;
                            if (id < 0 || id >= depth) continue;
                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph - padH;
                                if (ih < 0 || ih >= height) continue;
                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw - padW;
                                    if (iw < 0 || iw >= width) continue;
                                    count++;
                                }
                            }
                        }

                        if (count == 0) continue;

                        int gradOutIdx = (((b * channels + c) * outputDepth + od) * outputHeight + oh) * outputWidth + ow;
                        T gradVal = numOps.Divide(gradOutputData[gradOutIdx], numOps.FromDouble(count));

                        // Distribute gradient equally to all contributing positions
                        for (int pd = 0; pd < poolD; pd++)
                        {
                            int id = od * strideD + pd - padD;
                            if (id < 0 || id >= depth) continue;

                            for (int ph = 0; ph < poolH; ph++)
                            {
                                int ih = oh * strideH + ph - padH;
                                if (ih < 0 || ih >= height) continue;

                                for (int pw = 0; pw < poolW; pw++)
                                {
                                    int iw = ow * strideW + pw - padW;
                                    if (iw < 0 || iw >= width) continue;

                                    int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                                    gradInputData[inputIdx] = numOps.Add(gradInputData[inputIdx], gradVal);
                                }
                            }
                        }
                    }
                }
            }
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Upsample3D<T>(Tensor<T> input, int scaleD, int scaleH, int scaleW)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 5) throw new ArgumentException($"Upsample3D requires 5D input tensor [batch, channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (scaleD <= 0 || scaleH <= 0 || scaleW <= 0) throw new ArgumentException("Scale factors must be positive.");

        int batch = input._shape[0];
        int channels = input._shape[1];
        int depth = input._shape[2];
        int height = input._shape[3];
        int width = input._shape[4];

        int outDepth = depth * scaleD;
        int outHeight = height * scaleH;
        int outWidth = width * scaleW;

        var outputData = new T[batch * channels * outDepth * outHeight * outWidth];
        var inputData = input.GetFlattenedData();

        // Use parallel processing over batch and channels
        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int od = 0; od < outDepth; od++)
            {
                int id = od / scaleD;
                for (int oh = 0; oh < outHeight; oh++)
                {
                    int ih = oh / scaleH;
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int iw = ow / scaleW;

                        int inputIdx = (((b * channels + c) * depth + id) * height + ih) * width + iw;
                        int outputIdx = (((b * channels + c) * outDepth + od) * outHeight + oh) * outWidth + ow;
                        outputData[outputIdx] = inputData[inputIdx];
                    }
                }
            }
        });

        var up3dResult = TensorAllocator.Rent<T>([batch, channels, outDepth, outHeight, outWidth], new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("Upsample3D", up3dResult, input, BackwardFunctions<T>.Upsample3DBackward, new object[] { scaleD, scaleH, scaleW });
        return up3dResult;
    }

    /// <inheritdoc/>
    public Tensor<T> Upsample3DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleD, int scaleH, int scaleW)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements [batch, channels, depth, height, width].", nameof(inputShape));
        if (scaleD <= 0 || scaleH <= 0 || scaleW <= 0) throw new ArgumentException("Scale factors must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int depth = inputShape[2];
        int height = inputShape[3];
        int width = inputShape[4];

        int outDepth = depth * scaleD;
        int outHeight = height * scaleH;
        int outWidth = width * scaleW;

        var gradInputData = new T[batch * channels * depth * height * width];
        var gradOutputData = gradOutput.GetFlattenedData();

        // Initialize gradient input to zero
        for (int i = 0; i < gradInputData.Length; i++)
        {
            gradInputData[i] = numOps.Zero;
        }

        // Each (b, c) iteration writes to a disjoint slice of gradInputData,
        // so no synchronization is needed - direct writes are thread-safe
        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            // Base offset for this (b, c) slice in gradInputData
            int inputBaseOffset = (b * channels + c) * depth * height * width;
            int outputBaseOffset = (b * channels + c) * outDepth * outHeight * outWidth;

            for (int od = 0; od < outDepth; od++)
            {
                int id = od / scaleD;
                for (int oh = 0; oh < outHeight; oh++)
                {
                    int ih = oh / scaleH;
                    for (int ow = 0; ow < outWidth; ow++)
                    {
                        int iw = ow / scaleW;

                        int inputIdx = inputBaseOffset + (id * height + ih) * width + iw;
                        int outputIdx = outputBaseOffset + (od * outHeight + oh) * outWidth + ow;
                        gradInputData[inputIdx] = numOps.Add(gradInputData[inputIdx], gradOutputData[outputIdx]);
                    }
                }
            }
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose3D<T>(Tensor<T> input, Tensor<T> kernel, int[] stride, int[] padding, int[] outputPadding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (!input.IsContiguous) input = input.Contiguous();
        if (!kernel.IsContiguous) kernel = kernel.Contiguous();
        if (input.Rank != 5) throw new ArgumentException($"ConvTranspose3D input requires 5D tensor [batch, in_channels, depth, height, width]. Got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 5) throw new ArgumentException($"ConvTranspose3D kernel requires 5D tensor [in_channels, out_channels, kD, kH, kW]. Got rank {kernel.Rank}.", nameof(kernel));
        if (stride == null || stride.Length != 3) throw new ArgumentException("Stride must be array of 3 elements.", nameof(stride));
        if (padding == null || padding.Length != 3) throw new ArgumentException("Padding must be array of 3 elements.", nameof(padding));
        if (outputPadding == null || outputPadding.Length != 3) throw new ArgumentException("Output padding must be array of 3 elements.", nameof(outputPadding));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int inDepth = input._shape[2];
        int inHeight = input._shape[3];
        int inWidth = input._shape[4];

        int kernelInChannels = kernel._shape[0];
        int outChannels = kernel._shape[1];
        int kD = kernel._shape[2];
        int kH = kernel._shape[3];
        int kW = kernel._shape[4];

        if (inChannels != kernelInChannels)
            throw new ArgumentException($"Kernel's input channels ({kernelInChannels}) must match input tensor's channels ({inChannels}).");

        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];
        int outPadD = outputPadding[0], outPadH = outputPadding[1], outPadW = outputPadding[2];

        // Calculate output dimensions for transposed convolution
        int outDepth = (inDepth - 1) * strideD - 2 * padD + kD + outPadD;
        int outHeight = (inHeight - 1) * strideH - 2 * padH + kH + outPadH;
        int outWidth = (inWidth - 1) * strideW - 2 * padW + kW + outPadW;

        var outputData = new T[batch * outChannels * outDepth * outHeight * outWidth];
        var inputData = input.GetDataArray();
        var kernelData = kernel.GetDataArray();
        var mergeLock = new object();

        // Use Parallel.For with true per-task local accumulators to avoid race conditions
        Parallel.For(
            0, batch * inChannels,
            () => new T[outputData.Length], // localInit: create per-task buffer
            (bic, state, localOutput) =>
            {
                int b = bic / inChannels;
                int ic = bic % inChannels;

                for (int id = 0; id < inDepth; id++)
                {
                    for (int ih = 0; ih < inHeight; ih++)
                    {
                        for (int iw = 0; iw < inWidth; iw++)
                        {
                            int inputIdx = (((b * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw;
                            T inputVal = inputData[inputIdx];

                            for (int oc = 0; oc < outChannels; oc++)
                            {
                                for (int kd = 0; kd < kD; kd++)
                                {
                                    int od = id * strideD - padD + kd;
                                    if (od < 0 || od >= outDepth) continue;

                                    for (int kh = 0; kh < kH; kh++)
                                    {
                                        int oh = ih * strideH - padH + kh;
                                        if (oh < 0 || oh >= outHeight) continue;

                                        for (int kw = 0; kw < kW; kw++)
                                        {
                                            int ow = iw * strideW - padW + kw;
                                            if (ow < 0 || ow >= outWidth) continue;

                                            int kernelIdx = (((ic * outChannels + oc) * kD + kd) * kH + kh) * kW + kw;
                                            int outputIdx = (((b * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow;

                                            localOutput[outputIdx] = numOps.Add(localOutput[outputIdx],
                                                numOps.Multiply(inputVal, kernelData[kernelIdx]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                return localOutput;
            },
            localOutput =>
            {
                // localFinally: merge per-task results under lock
                lock (mergeLock)
                {
                    for (int i = 0; i < outputData.Length; i++)
                    {
                        outputData[i] = numOps.Add(outputData[i], localOutput[i]);
                    }
                }
            });

        var ct3dResult = TensorAllocator.Rent<T>([batch, outChannels, outDepth, outHeight, outWidth], new Vector<T>(outputData));
        DifferentiableOps.RecordBinary("ConvTranspose3D", ct3dResult, input, kernel, BackwardFunctions<T>.ConvTranspose3DBackward, new object[] { stride, padding });
        return ct3dResult;
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose3DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!kernel.IsContiguous) kernel = kernel.Contiguous();
        if (inputShape == null || inputShape.Length != 5) throw new ArgumentException("Input shape must be array of 5 elements.", nameof(inputShape));

        // The backward pass for transposed convolution input is equivalent to a regular Conv3D
        // with the kernel applied in the normal direction
        return Conv3D(gradOutput, kernel, stride, padding, [1, 1, 1]);
    }

    /// <inheritdoc/>
    public Tensor<T> ConvTranspose3DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelShape == null || kernelShape.Length != 5) throw new ArgumentException("Kernel shape must be array of 5 elements.", nameof(kernelShape));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int inDepth = input._shape[2];
        int inHeight = input._shape[3];
        int inWidth = input._shape[4];

        int outChannels = gradOutput._shape[1];
        int outDepth = gradOutput._shape[2];
        int outHeight = gradOutput._shape[3];
        int outWidth = gradOutput._shape[4];

        int kD = kernelShape[2], kH = kernelShape[3], kW = kernelShape[4];
        int strideD = stride[0], strideH = stride[1], strideW = stride[2];
        int padD = padding[0], padH = padding[1], padW = padding[2];

        var gradKernelData = new T[inChannels * outChannels * kD * kH * kW];
        var inputData = input.GetFlattenedData();
        var gradOutputData = gradOutput.GetFlattenedData();

        // Use Parallel.For with localInit/localFinally for thread-safe accumulation
        var mergeLock = new object();

        Parallel.For(
            0, batch * inChannels,
            () => new T[gradKernelData.Length], // localInit: create per-task buffer
            (bic, state, localGradKernel) =>
            {
                int b = bic / inChannels;
                int ic = bic % inChannels;

                for (int id = 0; id < inDepth; id++)
                {
                    for (int ih = 0; ih < inHeight; ih++)
                    {
                        for (int iw = 0; iw < inWidth; iw++)
                        {
                            int inputIdx = (((b * inChannels + ic) * inDepth + id) * inHeight + ih) * inWidth + iw;
                            T inputVal = inputData[inputIdx];

                            for (int oc = 0; oc < outChannels; oc++)
                            {
                                for (int kd = 0; kd < kD; kd++)
                                {
                                    int od = id * strideD - padD + kd;
                                    if (od < 0 || od >= outDepth) continue;

                                    for (int kh = 0; kh < kH; kh++)
                                    {
                                        int oh = ih * strideH - padH + kh;
                                        if (oh < 0 || oh >= outHeight) continue;

                                        for (int kw = 0; kw < kW; kw++)
                                        {
                                            int ow = iw * strideW - padW + kw;
                                            if (ow < 0 || ow >= outWidth) continue;

                                            int gradOutputIdx = (((b * outChannels + oc) * outDepth + od) * outHeight + oh) * outWidth + ow;
                                            int kernelIdx = (((ic * outChannels + oc) * kD + kd) * kH + kh) * kW + kw;

                                            localGradKernel[kernelIdx] = numOps.Add(localGradKernel[kernelIdx],
                                                numOps.Multiply(inputVal, gradOutputData[gradOutputIdx]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return localGradKernel;
            },
            localGradKernel =>
            {
                // localFinally: merge per-task results under lock
                lock (mergeLock)
                {
                    for (int i = 0; i < gradKernelData.Length; i++)
                    {
                        gradKernelData[i] = numOps.Add(gradKernelData[i], localGradKernel[i]);
                    }
                }
            });

        return TensorAllocator.Rent<T>(kernelShape, new Vector<T>(gradKernelData));
    }

    #endregion

    /// <inheritdoc/>
    public Tensor<T> LocallyConnectedConv2D<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, int[] stride)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (!input.IsContiguous) input = input.Contiguous();
        if (!weights.IsContiguous) weights = weights.Contiguous();
        if (input.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2D input requires a 4D tensor [batch, in_channels, height, width]. Got rank {input.Rank}.");
        // weights shape: [output_height, output_width, out_channels, in_channels, kernel_height, kernel_width]
        if (weights.Rank != 6) throw new ArgumentException($"LocallyConnectedConv2D weights require a 6D tensor. Got rank {weights.Rank}.");
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive", nameof(stride));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int inputHeight = input._shape[2];
        int inputWidth = input._shape[3];

        int outputHeight = weights._shape[0];
        int outputWidth = weights._shape[1];
        int outChannels = weights._shape[2];
        int kernelInChannels = weights._shape[3]; // in_channels in the weights tensor definition
        int kernelHeight = weights._shape[4];
        int kernelWidth = weights._shape[5];

        int strideH = stride[0], strideW = stride[1];

        // Validate kernel in_channels matches input in_channels
        if (kernelInChannels != inChannels)
            throw new ArgumentException($"Weight's input channels ({kernelInChannels}) must match input tensor's in_channels ({inChannels}).");

        // Ensure output shape derived from input/kernel/stride matches weights
        int expectedOutputH = (inputHeight - kernelHeight) / strideH + 1;
        int expectedOutputW = (inputWidth - kernelWidth) / strideW + 1;

        if (outputHeight != expectedOutputH || outputWidth != expectedOutputW)
            throw new ArgumentException($"Calculated output dimensions ({expectedOutputH}x{expectedOutputW}) do not match weights dimensions ({outputHeight}x{outputWidth}). Check input, kernel, and stride parameters.");

        var result = TensorAllocator.Rent<T>(new[] { batch, outChannels, outputHeight, outputWidth });
        var inputData = input.GetDataArray();
        var weightsData = weights.GetDataArray();
        var biasData = bias?.GetDataArray();

        Parallel.For(0, batch, b => // Process each batch independently
        {
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        T sum = numOps.Zero;

                        for (int ic = 0; ic < inChannels; ic++)
                        {
                            for (int kh = 0; kh < kernelHeight; kh++)
                            {
                                for (int kw = 0; kw < kernelWidth; kw++)
                                {
                                    int ih = oh * strideH + kh;
                                    int iw = ow * strideW + kw;

                                    // Check bounds (LocallyConnected typically doesn't use padding in direct impl, but can be added)
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                    {
                                        T inputVal = inputData[((b * inChannels + ic) * inputHeight + ih) * inputWidth + iw];
                                        // weightsData index: [oh, ow, oc, ic, kh, kw]
                                        T weightVal = weightsData[((((oh * outputWidth + ow) * outChannels + oc) * kernelInChannels + ic) * kernelHeight + kh) * kernelWidth + kw];
                                        sum = numOps.Add(sum, numOps.Multiply(inputVal, weightVal));
                                    }
                                }
                            }
                        }

                        // Add bias if provided
                        if (biasData != null)
                        {
                            sum = numOps.Add(sum, biasData[oc]);
                        }

                        result[b, oc, oh, ow] = sum;
                    }
                }
            }
        });

        DifferentiableOps.RecordBinary("LocallyConnectedConv2D", result, input, weights, BackwardFunctions<T>.LocallyConnectedConv2DBackward, new object[] { stride });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> LocallyConnectedConv2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> weights, int[] inputShape, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (inputShape == null || inputShape.Length != 4) throw new ArgumentException("inputShape must be array of 4 elements [batch, inChannels, height, width]", nameof(inputShape));
        if (gradOutput.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2DBackwardInput gradOutput requires 4D tensor. Got rank {gradOutput.Rank}.");
        if (weights.Rank != 6) throw new ArgumentException($"LocallyConnectedConv2DBackwardInput weights require a 6D tensor. Got rank {weights.Rank}.");
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be an array of 2 elements", nameof(stride));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];

        int outputHeight = weights._shape[0];
        int outputWidth = weights._shape[1];
        int outChannels = weights._shape[2];
        int kernelInChannels = weights._shape[3];
        int kernelHeight = weights._shape[4];
        int kernelWidth = weights._shape[5];

        int strideH = stride[0], strideW = stride[1];

        var finalGradInput = new T[batch * inChannels * inputHeight * inputWidth];
        var gradOutputData = gradOutput.GetFlattenedData();
        var weightsData = weights.GetFlattenedData();

        Parallel.For(0, batch * inChannels * inputHeight * inputWidth, idx =>
        {
            int b = idx / (inChannels * inputHeight * inputWidth);
            int ic = (idx / (inputHeight * inputWidth)) % inChannels;
            int ih = (idx / inputWidth) % inputHeight;
            int iw = idx % inputWidth;

            T sumGrad = numOps.Zero;
            // Iterate over all possible output positions this input pixel contributes to
            for (int oh_candidate = 0; oh_candidate < outputHeight; oh_candidate++)
            {
                for (int ow_candidate = 0; ow_candidate < outputWidth; ow_candidate++)
                {
                    // Check if input pixel (ih, iw) is covered by kernel at (oh_candidate, ow_candidate)
                    // (ih, iw) should be in the kernel window
                    int kh_relative = ih - oh_candidate * strideH;
                    int kw_relative = iw - ow_candidate * strideW;

                    if (kh_relative >= 0 && kh_relative < kernelHeight && kw_relative >= 0 && kw_relative < kernelWidth)
                    {
                        for (int oc = 0; oc < outChannels; oc++)
                        {
                            T gradOutVal = gradOutputData[((b * outChannels + oc) * outputHeight + oh_candidate) * outputWidth + ow_candidate];
                            T weightVal = weightsData[((((oh_candidate * outputWidth + ow_candidate) * outChannels + oc) * kernelInChannels + ic) * kernelHeight + kh_relative) * kernelWidth + kw_relative];
                            sumGrad = numOps.Add(sumGrad, numOps.Multiply(gradOutVal, weightVal));
                        }
                    }
                }
            }
            finalGradInput[idx] = sumGrad;
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(finalGradInput));
    }

    /// <inheritdoc/>
    public Tensor<T> LocallyConnectedConv2DBackwardWeights<T>(Tensor<T> gradOutput, Tensor<T> input, int[] weightsShape, int[] stride)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (weightsShape == null || weightsShape.Length != 6) throw new ArgumentException("weightsShape must be array of 6 elements", nameof(weightsShape));
        if (gradOutput.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2DBackwardWeights gradOutput requires 4D tensor. Got rank {gradOutput.Rank}.");
        if (input.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2DBackwardWeights input requires 4D tensor. Got rank {input.Rank}.");
        if (stride == null || stride.Length != 2) throw new ArgumentException("Stride must be array of 2 elements", nameof(stride));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = input._shape[0];
        int inChannels = input._shape[1];
        int inputHeight = input._shape[2];
        int inputWidth = input._shape[3];

        int outputHeight = weightsShape[0];
        int outputWidth = weightsShape[1];
        int outChannels = weightsShape[2];
        int kernelInChannels = weightsShape[3];
        int kernelHeight = weightsShape[4];
        int kernelWidth = weightsShape[5];

        int strideH = stride[0], strideW = stride[1];

        var gradWeights = new T[weightsShape.Aggregate(1, (acc, val) => acc * val)];
        var gradOutputData = gradOutput.GetFlattenedData();
        var inputData = input.GetFlattenedData();

        for (int i = 0; i < gradWeights.Length; i++)
            gradWeights[i] = numOps.Zero;

        Parallel.For(0, weightsShape.Aggregate(1, (acc, val) => acc * val), idx => // Iterate over all weight elements
        {
            // Deconstruct idx to weights indices (oh, ow, oc, ic, kh, kw)
            int flatIdx = idx;
            int kw_w = kernelWidth;
            int kh_w = kernelHeight;
            int ic_w = kernelInChannels; // This is the 3rd dim in weights (index 3)
            int oc_w = outChannels;
            int ow_w = outputWidth;

            int kw = flatIdx % kw_w; flatIdx /= kw_w;
            int kh = flatIdx % kh_w; flatIdx /= kh_w;
            int ic = flatIdx % ic_w; flatIdx /= ic_w;
            int oc = flatIdx % oc_w; flatIdx /= oc_w;
            int ow = flatIdx % ow_w; flatIdx /= ow_w;
            int oh = flatIdx;

            T sum = numOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                int ih = oh * strideH + kh;
                int iw = ow * strideW + kw;

                // Check bounds for input
                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                {
                    T gradOutVal = gradOutputData[((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow];
                    T inputVal = inputData[((b * inChannels + ic) * inputHeight + ih) * inputWidth + iw];
                    sum = numOps.Add(sum, numOps.Multiply(gradOutVal, inputVal));
                }
            }
            gradWeights[idx] = sum;
        });

        return TensorAllocator.Rent<T>(weightsShape, new Vector<T>(gradWeights));
    }

    /// <inheritdoc/>
    public Tensor<T> LocallyConnectedConv2DBackwardBias<T>(Tensor<T> gradOutput)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradOutput.Rank != 4) throw new ArgumentException($"LocallyConnectedConv2DBackwardBias gradOutput requires 4D tensor. Got rank {gradOutput.Rank}.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = gradOutput._shape[0];
        int outChannels = gradOutput._shape[1];
        int outputHeight = gradOutput._shape[2];
        int outputWidth = gradOutput._shape[3];

        var gradBias = new T[outChannels]; // Bias gradient is 1D [outChannels]
        var gradOutputData = gradOutput.GetFlattenedData();

        for (int i = 0; i < gradBias.Length; i++)
            gradBias[i] = numOps.Zero;

        Parallel.For(0, outChannels, oc => // Iterate over output channels
        {
            T sum = numOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                for (int oh = 0; oh < outputHeight; oh++)
                {
                    for (int ow = 0; ow < outputWidth; ow++)
                    {
                        sum = numOps.Add(sum, gradOutputData[((b * outChannels + oc) * outputHeight + oh) * outputWidth + ow]);
                    }
                }
            }
            gradBias[oc] = sum;
        });

        return TensorAllocator.Rent<T>(new[] { outChannels }, new Vector<T>(gradBias));
    }

    #endregion

    #region Normalization and Activation Operations

    /// <inheritdoc/>
    public virtual Tensor<T> Softmax<T>(Tensor<T> input, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {rank} dimensions");

        // Compute outer and inner sizes
        int outerSize = 1, axisSize = input._shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input._shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= input._shape[i];

        // Fast SIMD path for float when softmax is on the last axis (innerSize==1)
        if (typeof(T) == typeof(float) && innerSize == 1)
        {
            // Use Pin() for NativeMemory compatibility (no GetDataArray copy)
            var result = TensorAllocator.RentUninitialized<T>(input._shape);
            using var pinIn = input.Data.Pin();
            using var pinOut = result.Data.Pin();
            unsafe
            {
                SoftmaxFloatFastPtr((float*)pinIn.Pointer, (float*)pinOut.Pointer, outerSize, axisSize);
            }
            DifferentiableOps.RecordUnary("Softmax", result, input, BackwardFunctions<T>.SoftmaxBackward, new object[] { axis });
            return result;
        }

        // Generic scalar fallback for non-float types or non-last-axis
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.GetFlattenedData();
        var result2 = TensorAllocator.RentUninitialized<T>(input._shape);
        var outputDataGeneric = result2.GetDataArray();

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            T maxVal = numOps.MinValue;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                if (numOps.GreaterThan(inputData[flatIdx], maxVal))
                    maxVal = inputData[flatIdx];
            }

            T sumExp = numOps.Zero;
            var expVals = new T[axisSize];
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                expVals[i] = numOps.Exp(numOps.Subtract(inputData[flatIdx], maxVal));
                sumExp = numOps.Add(sumExp, expVals[i]);
            }

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                outputDataGeneric[flatIdx] = numOps.Divide(expVals[i], sumExp);
            }
        });

        DifferentiableOps.RecordUnary("Softmax", result2, input, BackwardFunctions<T>.SoftmaxBackward, new object[] { axis });
        return result2;
    }

    /// <summary>
    /// Fast SIMD softmax for float arrays with parallel row processing.
    /// Each row is independent: max → subtract → exp → sum → divide.
    /// Parallelized across rows for large tensors.
    /// </summary>
    /// <summary>
    /// SIMD-optimized GroupNorm for float. Fuses mean+variance computation
    /// and parallelizes across batch*groups using PersistentParallelExecutor.
    /// </summary>
    private unsafe void GroupNormFloatPtr(
        float* inputData, float* outputData,
        float* gammaData, float* betaData,
        int batch, int channels, int spatialSize, int numGroups, int channelsPerGroup,
        float eps, out float[] meanArr, out float[] varArr)
    {
        // Use arrays directly (Span can't be captured in lambdas)

        int groupSize = channelsPerGroup * spatialSize;
        meanArr = new float[batch * numGroups];
        varArr = new float[batch * numGroups];

        int totalGroups = batch * numGroups;
        var meanLocal = meanArr;
        var varLocal = varArr;

        // Parallelize across batch*groups — each group is independent
        PersistentParallelExecutor.Instance.Execute(
            Math.Min(totalGroups, CpuParallelSettings.MaxDegreeOfParallelism),
            chunk =>
            {
                int chunkSize = (totalGroups + Math.Min(totalGroups, CpuParallelSettings.MaxDegreeOfParallelism) - 1)
                    / Math.Min(totalGroups, CpuParallelSettings.MaxDegreeOfParallelism);
                int startIdx = chunk * chunkSize;
                int endIdx = Math.Min(startIdx + chunkSize, totalGroups);

                for (int idx = startIdx; idx < endIdx; idx++)
                {
                    int b = idx / numGroups;
                    int g = idx % numGroups;
                    int startChannel = g * channelsPerGroup;
                    int batchOffset = b * channels * spatialSize;

                    // Mean computation (pointer-based for NativeMemory compat)
                    float sum = 0f;
                    for (int c = 0; c < channelsPerGroup; c++)
                    {
                        int chanOff = batchOffset + (startChannel + c) * spatialSize;
                        for (int s = 0; s < spatialSize; s++)
                            sum += inputData[chanOff + s];
                    }
                    float groupMean = sum / groupSize;
                    meanLocal[idx] = groupMean;

                    // Variance computation (pointer-based)
                    float sumSq = 0f;
                    for (int c = 0; c < channelsPerGroup; c++)
                    {
                        int chanOff = batchOffset + (startChannel + c) * spatialSize;
                        for (int s = 0; s < spatialSize; s++)
                        {
                            float diff = inputData[chanOff + s] - groupMean;
                            sumSq += diff * diff;
                        }
                    }
                    float groupVar = sumSq / groupSize;
                    varLocal[idx] = groupVar;

                    // Normalize and apply scale/shift
                    float invStd = 1f / MathF.Sqrt(groupVar + eps);
                    for (int c = 0; c < channelsPerGroup; c++)
                    {
                        int channel = startChannel + c;
                        int chanOff = batchOffset + channel * spatialSize;
                        float g_val = gammaData[channel];
                        float b_val = betaData[channel];
                        for (int s = 0; s < spatialSize; s++)
                        {
                            outputData[chanOff + s] = (inputData[chanOff + s] - groupMean) * invStd * g_val + b_val;
                        }
                    }
                }
            });
    }

    /// <summary>SIMD-accelerated sum of float array segment.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SimdSumFloat(float[] data, int offset, int count)
    {
        float sum = 0f;
        int i = 0;
#if NET5_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx.IsSupported && count >= 32)
        {
            var vsum0 = System.Runtime.Intrinsics.Vector256<float>.Zero;
            var vsum1 = vsum0; var vsum2 = vsum0; var vsum3 = vsum0;
            int simdLen = count & ~31;
            for (; i < simdLen; i += 32)
            {
                vsum0 = System.Runtime.Intrinsics.X86.Avx.Add(vsum0, SimdKernels.ReadVector256(data.AsSpan(), offset + i));
                vsum1 = System.Runtime.Intrinsics.X86.Avx.Add(vsum1, SimdKernels.ReadVector256(data.AsSpan(), offset + i + 8));
                vsum2 = System.Runtime.Intrinsics.X86.Avx.Add(vsum2, SimdKernels.ReadVector256(data.AsSpan(), offset + i + 16));
                vsum3 = System.Runtime.Intrinsics.X86.Avx.Add(vsum3, SimdKernels.ReadVector256(data.AsSpan(), offset + i + 24));
            }
            vsum0 = System.Runtime.Intrinsics.X86.Avx.Add(
                System.Runtime.Intrinsics.X86.Avx.Add(vsum0, vsum1),
                System.Runtime.Intrinsics.X86.Avx.Add(vsum2, vsum3));
            sum = SimdKernels.HorizontalSum(vsum0);
        }
#endif
        for (; i < count; i++)
            sum += data[offset + i];
        return sum;
    }

    /// <summary>SIMD-accelerated sum of squared differences: sum((x-mean)^2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SimdSumSquaredDiffFloat(float[] data, int offset, int count, float mean)
    {
        float sumSq = 0f;
        int i = 0;
#if NET5_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported && count >= 32)
        {
            var vmean = System.Runtime.Intrinsics.Vector256.Create(mean);
            var vsum0 = System.Runtime.Intrinsics.Vector256<float>.Zero;
            var vsum1 = vsum0; var vsum2 = vsum0; var vsum3 = vsum0;
            int simdLen = count & ~31;
            for (; i < simdLen; i += 32)
            {
                var d0 = System.Runtime.Intrinsics.X86.Avx.Subtract(SimdKernels.ReadVector256(data.AsSpan(), offset + i), vmean);
                var d1 = System.Runtime.Intrinsics.X86.Avx.Subtract(SimdKernels.ReadVector256(data.AsSpan(), offset + i + 8), vmean);
                var d2 = System.Runtime.Intrinsics.X86.Avx.Subtract(SimdKernels.ReadVector256(data.AsSpan(), offset + i + 16), vmean);
                var d3 = System.Runtime.Intrinsics.X86.Avx.Subtract(SimdKernels.ReadVector256(data.AsSpan(), offset + i + 24), vmean);
                vsum0 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d0, d0, vsum0);
                vsum1 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d1, d1, vsum1);
                vsum2 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d2, d2, vsum2);
                vsum3 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d3, d3, vsum3);
            }
            vsum0 = System.Runtime.Intrinsics.X86.Avx.Add(
                System.Runtime.Intrinsics.X86.Avx.Add(vsum0, vsum1),
                System.Runtime.Intrinsics.X86.Avx.Add(vsum2, vsum3));
            sumSq = SimdKernels.HorizontalSum(vsum0);
        }
#endif
        for (; i < count; i++)
        {
            float diff = data[offset + i] - mean;
            sumSq += diff * diff;
        }
        return sumSq;
    }

    /// <summary>Pointer-based softmax — works with both managed arrays and NativeMemory.</summary>
    private static unsafe void SoftmaxFloatFastPtr(float* pIn, float* pOut, int outerSize, int axisSize)
    {
        {
#if NET5_0_OR_GREATER
            // oneDNN path: fused SVML-accelerated softmax — processes all rows in one call.
            // This matches TorchSharp's libtorch which uses the same MKLDNN backend.
            if (OneDnnProvider.TrySoftmax(pIn, pOut, outerSize, axisSize))
                return;
#endif

            int maxThreads = CpuParallelSettings.MaxDegreeOfParallelism;
            bool useParallel = outerSize >= 4 && outerSize * axisSize >= 32768;

            if (useParallel)
            {
                // Use PersistentParallelExecutor for near-zero dispatch overhead
                // Each worker processes a chunk of consecutive rows for cache locality
                int numChunks = Math.Min(maxThreads, outerSize);
                int rowsPerChunk = (outerSize + numChunks - 1) / numChunks;
                float* pInCap = pIn;
                float* pOutCap = pOut;
                int axisSz = axisSize;
                int outerSz = outerSize;

                PersistentParallelExecutor.Instance.Execute(numChunks, chunk =>
                {
                    int startRow = chunk * rowsPerChunk;
                    int endRow = Math.Min(startRow + rowsPerChunk, outerSz);
                    for (int row = startRow; row < endRow; row++)
                    {
                        SimdKernels.SoftmaxRowUnsafe(pInCap + row * axisSz, pOutCap + row * axisSz, axisSz);
                    }
                });
            }
            else
            {
                for (int row = 0; row < outerSize; row++)
                    SimdKernels.SoftmaxRowUnsafe(pIn + row * axisSize, pOut + row * axisSize, axisSize);
            }
        }
    }

    /// <inheritdoc/>
    public Tensor<T> SoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (output == null) throw new ArgumentNullException(nameof(output));

        int rank = output.Rank;
        if (axis < 0) axis = rank + axis;

        int outerSize = 1, axisSize = output._shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= output._shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= output._shape[i];

#if NET5_0_OR_GREATER
        // Float SIMD fast path for last-axis softmax backward (innerSize == 1)
        if (typeof(T) == typeof(float) && innerSize == 1
            && gradOutput.GetFlattenedData() is float[] gOutF
            && output.GetFlattenedData() is float[] outF)
        {
            var gInF = new float[outF.Length];
            SoftmaxBackwardFloat(gOutF, outF, gInF, outerSize, axisSize);
            return (Tensor<T>)(object)TensorAllocator.Rent<T>(output._shape, new Vector<T>((T[])(object)gInF));
        }
#endif

        // Generic fallback
        var numOps = MathHelper.GetNumericOperations<T>();
        var gradOutputData = gradOutput.GetFlattenedData();
        var outputData = output.GetFlattenedData();
        var gradInputData = new T[outputData.Length];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            T dotProduct = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                dotProduct = numOps.Add(dotProduct, numOps.Multiply(gradOutputData[flatIdx], outputData[flatIdx]));
            }

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                gradInputData[flatIdx] = numOps.Multiply(outputData[flatIdx], numOps.Subtract(gradOutputData[flatIdx], dotProduct));
            }
        });

        return TensorAllocator.Rent<T>(output._shape, new Vector<T>(gradInputData));
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// SIMD-optimized softmax backward for float with contiguous axis (innerSize == 1).
    /// Per row: dotProduct = sum(gradOut * output), gradIn = output * (gradOut - dotProduct).
    /// </summary>
    private static unsafe void SoftmaxBackwardFloat(float[] gradOut, float[] output, float[] gradIn,
        int outerSize, int axisSize)
    {
        bool useParallel = outerSize >= 4 && outerSize * axisSize >= 32768;

        if (useParallel)
        {
            Parallel.For(0, outerSize, row =>
            {
                SoftmaxBackwardFloatRow(gradOut, output, gradIn, row, axisSize);
            });
        }
        else
        {
            for (int row = 0; row < outerSize; row++)
                SoftmaxBackwardFloatRow(gradOut, output, gradIn, row, axisSize);
        }
    }

    private static unsafe void SoftmaxBackwardFloatRow(float[] gradOut, float[] output, float[] gradIn,
        int row, int axisSize)
    {
        int baseIdx = row * axisSize;
        fixed (float* pGO = gradOut, pO = output, pGI = gradIn)
        {
            float* go = pGO + baseIdx;
            float* o = pO + baseIdx;
            float* gi = pGI + baseIdx;

            // Step 1: dot product = sum(gradOut[i] * output[i])
            float dot = 0f;
            int i = 0;
            if (System.Runtime.Intrinsics.X86.Fma.IsSupported && axisSize >= 32)
            {
                var vdot0 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                var vdot1 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                var vdot2 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                var vdot3 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                int simdLen = axisSize & ~31;
                for (; i < simdLen; i += 32)
                {
                    vdot0 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256(go + i),
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256(o + i), vdot0);
                    vdot1 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256(go + i + 8),
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256(o + i + 8), vdot1);
                    vdot2 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256(go + i + 16),
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256(o + i + 16), vdot2);
                    vdot3 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256(go + i + 24),
                        System.Runtime.Intrinsics.X86.Avx.LoadVector256(o + i + 24), vdot3);
                }
                vdot0 = System.Runtime.Intrinsics.X86.Avx.Add(
                    System.Runtime.Intrinsics.X86.Avx.Add(vdot0, vdot1),
                    System.Runtime.Intrinsics.X86.Avx.Add(vdot2, vdot3));
                dot = SimdKernels.HorizontalSum(vdot0);
            }
            for (; i < axisSize; i++)
                dot += go[i] * o[i];

            // Step 2: gradIn[i] = output[i] * (gradOut[i] - dot)
            i = 0;
            if (System.Runtime.Intrinsics.X86.Avx.IsSupported && axisSize >= 32)
            {
                var vdot = System.Runtime.Intrinsics.Vector256.Create(dot);
                int simdLen = axisSize & ~31;
                for (; i < simdLen; i += 32)
                {
                    var go0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(go + i);
                    var go1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(go + i + 8);
                    var go2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(go + i + 16);
                    var go3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(go + i + 24);
                    var o0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(o + i);
                    var o1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(o + i + 8);
                    var o2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(o + i + 16);
                    var o3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(o + i + 24);
                    System.Runtime.Intrinsics.X86.Avx.Store(gi + i,
                        System.Runtime.Intrinsics.X86.Avx.Multiply(o0, System.Runtime.Intrinsics.X86.Avx.Subtract(go0, vdot)));
                    System.Runtime.Intrinsics.X86.Avx.Store(gi + i + 8,
                        System.Runtime.Intrinsics.X86.Avx.Multiply(o1, System.Runtime.Intrinsics.X86.Avx.Subtract(go1, vdot)));
                    System.Runtime.Intrinsics.X86.Avx.Store(gi + i + 16,
                        System.Runtime.Intrinsics.X86.Avx.Multiply(o2, System.Runtime.Intrinsics.X86.Avx.Subtract(go2, vdot)));
                    System.Runtime.Intrinsics.X86.Avx.Store(gi + i + 24,
                        System.Runtime.Intrinsics.X86.Avx.Multiply(o3, System.Runtime.Intrinsics.X86.Avx.Subtract(go3, vdot)));
                }
            }
            for (; i < axisSize; i++)
                gi[i] = o[i] * (go[i] - dot);
        }
    }
#endif

    /// <inheritdoc/>
    public Tensor<T> GumbelSoftmax<T>(Tensor<T> input, double temperature = 1.0, bool hard = false, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), temperature, "Temperature must be positive.");
        if (double.IsNaN(temperature) || double.IsInfinity(temperature))
            throw new ArgumentOutOfRangeException(nameof(temperature), temperature, "Temperature must be a finite number.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.GetFlattenedData();
        var shape = input._shape;
        const double eps = 1e-10;

        // Add Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
        var random = RandomHelper.ThreadSafeRandom;
        var perturbedData = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            var u = random.NextDouble();
            u = Math.Max(u, eps);
            u = Math.Min(u, 1 - eps);
            var gumbel = numOps.FromDouble(-Math.Log(-Math.Log(u)));
            var val = numOps.Add(inputData[i], gumbel);
            perturbedData[i] = numOps.Divide(val, numOps.FromDouble(temperature));
        }

        // Apply softmax
        var perturbedTensor = TensorAllocator.Rent<T>(shape, new Vector<T>(perturbedData));
        var softResult = Softmax(perturbedTensor, axis);

        if (!hard)
            return softResult;

        // Hard mode: create one-hot and use straight-through estimator
        var softData = softResult.GetDataArray();
        var hardData = new T[softData.Length];
        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Find argmax
            int maxIdx = 0;
            T maxVal = softData[(outer * axisSize) * innerSize + inner];
            for (int i = 1; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                if (numOps.GreaterThan(softData[flatIdx], maxVal))
                {
                    maxVal = softData[flatIdx];
                    maxIdx = i;
                }
            }

            // Create one-hot
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                hardData[flatIdx] = i == maxIdx ? numOps.One : numOps.Zero;
            }
        });

        return TensorAllocator.Rent<T>(shape, new Vector<T>(hardData));
    }

    /// <inheritdoc/>
    public Tensor<T> GumbelSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, double temperature, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!output.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), temperature, "Temperature must be positive.");

        // Gradient flows through softmax, scaled by 1/temperature
        var softmaxGrad = SoftmaxBackward(gradOutput, output, axis);
        var numOps = MathHelper.GetNumericOperations<T>();
        var gradData = softmaxGrad.GetDataArray();
        var scale = numOps.FromDouble(1.0 / temperature);

        for (int i = 0; i < gradOutput.Length; i++)
        {
            gradData[i] = numOps.Multiply(gradData[i], scale);
        }

        return TensorAllocator.Rent<T>(output._shape, new Vector<T>(gradData));
    }

    /// <inheritdoc/>
    public Tensor<T> TaylorSoftmax<T>(Tensor<T> input, int order = 2, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (order < 1)
            throw new ArgumentOutOfRangeException(nameof(order), order, "Order must be at least 1.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.GetFlattenedData();
        var shape = input._shape;
        var outputData = new T[input.Length];

        // Precompute factorials
        var factorials = new double[order + 1];
        factorials[0] = 1;
        for (int i = 1; i <= order; i++)
            factorials[i] = factorials[i - 1] * i;

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Find max for numerical stability (similar to standard softmax)
            var maxVal = inputData[(outer * axisSize) * innerSize + inner];
            for (int i = 1; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                if (numOps.GreaterThan(inputData[flatIdx], maxVal))
                    maxVal = inputData[flatIdx];
            }

            // Compute Taylor approximation of exp for each position along axis
            var expApprox = new T[axisSize];
            T sumExp = numOps.Zero;

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                // Subtract max for numerical stability
                var x = numOps.Subtract(inputData[flatIdx], maxVal);

                // Taylor: 1 + x + x^2/2! + x^3/3! + ...
                var taylorExp = numOps.One;
                var xPower = numOps.One;
                for (int n = 1; n <= order; n++)
                {
                    xPower = numOps.Multiply(xPower, x);
                    taylorExp = numOps.Add(taylorExp, numOps.Divide(xPower, numOps.FromDouble(factorials[n])));
                }

                // Ensure non-negative for numerical stability
                if (numOps.LessThan(taylorExp, numOps.Zero))
                    taylorExp = numOps.FromDouble(1e-10);

                expApprox[i] = taylorExp;
                sumExp = numOps.Add(sumExp, taylorExp);
            }

            // Guard against zero sum (shouldn't happen with proper max subtraction, but just in case)
            if (numOps.Equals(sumExp, numOps.Zero))
                sumExp = numOps.FromDouble(1e-10);

            // Normalize
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                outputData[flatIdx] = numOps.Divide(expApprox[i], sumExp);
            }
        });

        var taylorResult = TensorAllocator.Rent<T>(shape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("TaylorSoftmax", taylorResult, input, BackwardFunctions<T>.TaylorSoftmaxBackward, new object[] { order, axis });
        return taylorResult;
    }

    /// <inheritdoc/>
    public Tensor<T> TaylorSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int order, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (!output.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = output.Rank;
        if (axis < 0) axis = rank + axis;

        var gradOutputData = gradOutput.GetFlattenedData();
        var inputData = input.GetFlattenedData();
        var outputData = output.GetDataArray();
        var shape = output._shape;
        var gradInputData = new T[outputData.Length];

        // Precompute factorials for derivative
        var factorials = new double[order + 1];
        factorials[0] = 1;
        for (int i = 1; i <= order; i++)
            factorials[i] = factorials[i - 1] * i;

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Compute g(x) and g'(x) for each position
            var gValues = new T[axisSize];
            var gPrimeValues = new T[axisSize];
            T sumG = numOps.Zero;

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var x = inputData[flatIdx];

                // g(x) = Taylor approximation
                var g = numOps.One;
                var xPower = numOps.One;
                for (int n = 1; n <= order; n++)
                {
                    xPower = numOps.Multiply(xPower, x);
                    g = numOps.Add(g, numOps.Divide(xPower, numOps.FromDouble(factorials[n])));
                }

                // g'(x) = derivative of Taylor = 1 + x + x^2/2! + ... (shifted)
                var gPrime = numOps.One;
                xPower = numOps.One;
                for (int n = 1; n < order; n++)
                {
                    xPower = numOps.Multiply(xPower, x);
                    gPrime = numOps.Add(gPrime, numOps.Divide(xPower, numOps.FromDouble(factorials[n])));
                }

                gValues[i] = g;
                gPrimeValues[i] = gPrime;
                sumG = numOps.Add(sumG, g);
            }

            // Compute gradient using chain rule: grad = softmaxGrad * g'(x) / g(x)
            T dotProduct = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                dotProduct = numOps.Add(dotProduct, numOps.Multiply(gradOutputData[flatIdx], outputData[flatIdx]));
            }

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var softmaxGrad = numOps.Multiply(outputData[flatIdx], numOps.Subtract(gradOutputData[flatIdx], dotProduct));
                var gPrimeOverG = numOps.Divide(gPrimeValues[i], gValues[i]);
                gradInputData[flatIdx] = numOps.Multiply(softmaxGrad, gPrimeOverG);
            }
        });

        return TensorAllocator.Rent<T>(shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Sparsemax<T>(Tensor<T> input, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.GetFlattenedData();
        var shape = input._shape;
        var outputData = new T[input.Length];

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Extract values along axis and sort by value (descending)
            var indexed = new List<(T value, int idx)>();
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                indexed.Add((inputData[flatIdx], i));
            }

            indexed.Sort((a, b) =>
            {
                if (numOps.GreaterThan(a.value, b.value)) return -1;
                if (numOps.LessThan(a.value, b.value)) return 1;
                return 0;
            });

            // Find threshold tau using the sparsemax algorithm
            T cumSum = numOps.Zero;
            int k = 0;
            T threshold = numOps.Zero;

            for (int i = 0; i < axisSize; i++)
            {
                cumSum = numOps.Add(cumSum, indexed[i].value);
                // Check if z[i] > (cumSum - 1) / (i + 1)
                var kPlusOne = numOps.FromDouble(i + 1);
                var testThreshold = numOps.Divide(numOps.Subtract(cumSum, numOps.One), kPlusOne);
                if (numOps.GreaterThan(indexed[i].value, testThreshold))
                {
                    k = i + 1;
                    threshold = testThreshold;
                }
            }

            // Compute output: max(0, z - tau)
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var val = numOps.Subtract(inputData[flatIdx], threshold);
                outputData[flatIdx] = numOps.GreaterThan(val, numOps.Zero) ? val : numOps.Zero;
            }
        });

        var sparsemaxResult = TensorAllocator.Rent<T>(shape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("Sparsemax", sparsemaxResult, input, BackwardFunctions<T>.SparsemaxBackward, new object[] { axis });
        return sparsemaxResult;
    }

    /// <inheritdoc/>
    public Tensor<T> SparsemaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (!output.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = output.Rank;
        if (axis < 0) axis = rank + axis;

        var gradOutputData = gradOutput.GetFlattenedData();
        var outputData = output.GetDataArray();
        var shape = output._shape;
        var gradInputData = new T[outputData.Length];

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Find support set (non-zero outputs) and compute mean of gradients in support
            T sumGradSupport = numOps.Zero;
            int supportSize = 0;

            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                if (numOps.GreaterThan(outputData[flatIdx], numOps.Zero))
                {
                    sumGradSupport = numOps.Add(sumGradSupport, gradOutputData[flatIdx]);
                    supportSize++;
                }
            }

            T meanGradSupport = supportSize > 0
                ? numOps.Divide(sumGradSupport, numOps.FromDouble(supportSize))
                : numOps.Zero;

            // Gradient: grad_input = grad_output - mean(grad_output[support]) for support, 0 otherwise
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                gradInputData[flatIdx] = numOps.GreaterThan(outputData[flatIdx], numOps.Zero)
                    ? numOps.Subtract(gradOutputData[flatIdx], meanGradSupport)
                    : numOps.Zero;
            }
        });

        return TensorAllocator.Rent<T>(shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> SphericalSoftmax<T>(Tensor<T> input, int axis = -1)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.GetFlattenedData();
        var shape = input._shape;
        var normalizedData = new T[input.Length];

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Compute L2 norm along axis
            T sumSquares = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var val = inputData[flatIdx];
                sumSquares = numOps.Add(sumSquares, numOps.Multiply(val, val));
            }
            var norm = numOps.Sqrt(sumSquares);

            // Avoid division by zero
            if (numOps.Equals(norm, numOps.Zero))
                norm = numOps.FromDouble(1e-10);

            // Normalize by L2 norm
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                normalizedData[flatIdx] = numOps.Divide(inputData[flatIdx], norm);
            }
        });

        // Apply softmax to normalized data
        var normalizedTensor = TensorAllocator.Rent<T>(shape, new Vector<T>(normalizedData));
        return Softmax(normalizedTensor, axis);
    }

    /// <inheritdoc/>
    public Tensor<T> SphericalSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (!output.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (!input.IsContiguous) input = input.Contiguous();
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = input.Rank;
        if (axis < 0) axis = rank + axis;

        var inputData = input.GetFlattenedData();
        var shape = input._shape;

        int outerSize = 1, axisSize = shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= shape[i];

        // First compute the normalized input
        var normalizedData = new T[input.Length];
        var norms = new T[outerSize * innerSize];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Compute L2 norm
            T sumSquares = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var val = inputData[flatIdx];
                sumSquares = numOps.Add(sumSquares, numOps.Multiply(val, val));
            }
            var norm = numOps.Sqrt(sumSquares);
            if (numOps.Equals(norm, numOps.Zero))
                norm = numOps.FromDouble(1e-10);
            norms[idx] = norm;

            // Normalize
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                normalizedData[flatIdx] = numOps.Divide(inputData[flatIdx], norm);
            }
        });

        // Get softmax gradient with respect to normalized input
        var normalizedTensor = TensorAllocator.Rent<T>(shape, new Vector<T>(normalizedData));
        var softmaxGrad = SoftmaxBackward(gradOutput, output, axis);
        var softmaxGradData = softmaxGrad.GetDataArray();

        // Chain rule through L2 normalization
        var gradInputData = new T[input.Length];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;
            var norm = norms[idx];
            var normCubed = numOps.Multiply(norm, numOps.Multiply(norm, norm));

            // Compute dot product of x and grad_normalized
            T dotProduct = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                dotProduct = numOps.Add(dotProduct, numOps.Multiply(inputData[flatIdx], softmaxGradData[flatIdx]));
            }

            // grad_x = (grad_normalized - normalized * dot_product) / norm
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                var term = numOps.Multiply(normalizedData[flatIdx], dotProduct);
                gradInputData[flatIdx] = numOps.Divide(numOps.Subtract(softmaxGradData[flatIdx], term), norm);
            }
        });

        return TensorAllocator.Rent<T>(shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> BatchNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gamma == null) throw new ArgumentNullException(nameof(gamma));
        if (beta == null) throw new ArgumentNullException(nameof(beta));
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        // Handle 1D [features] - treat as single sample [1, features]
        bool was1D = input._shape.Length == 1;
        Tensor<T> workingInput = was1D ? input.Reshape([1, input._shape[0]]) : input;

        // Handle 2D [batch, features], 3D [channels, height, width], and 4D [batch, channels, height, width] tensors
        if (workingInput._shape.Length == 4)
        {
            var result4D = BatchNorm4D(workingInput, gamma, beta, eps, numOps, out mean, out variance);
            var r4 = was1D ? result4D.Reshape([result4D._shape[1]]) : result4D;
            DifferentiableOps.RecordIfActive("BatchNorm", r4, new[] { input, gamma, beta }, BackwardFunctions<T>.BatchNormBackward, new object[] { mean, variance, epsilon });
            return r4;
        }

        if (workingInput._shape.Length == 3)
        {
            var result3D = BatchNorm3D(workingInput, gamma, beta, eps, numOps, out mean, out variance);
            var r3 = was1D ? result3D.Reshape([result3D.Length]) : result3D;
            DifferentiableOps.RecordIfActive("BatchNorm", r3, new[] { input, gamma, beta }, BackwardFunctions<T>.BatchNormBackward, new object[] { mean, variance, epsilon });
            return r3;
        }

        // 2D case: [batch, features]
        int batch = workingInput._shape[0];
        int features = workingInput._shape[1];

        var inputData = workingInput.GetDataArray();
        var gammaData = gamma.GetDataArray();
        var betaData = beta.GetDataArray();

        var meanData = new T[features];
        var varData = new T[features];
        var outputData = new T[batch * features];

        // Compute mean per feature
        for (int f = 0; f < features; f++)
        {
            T sum = numOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                sum = numOps.Add(sum, inputData[b * features + f]);
            }
            meanData[f] = numOps.Divide(sum, numOps.FromDouble(batch));
        }

        // Compute variance per feature
        for (int f = 0; f < features; f++)
        {
            T sumSq = numOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                T diff = numOps.Subtract(inputData[b * features + f], meanData[f]);
                sumSq = numOps.Add(sumSq, numOps.Multiply(diff, diff));
            }
            varData[f] = numOps.Divide(sumSq, numOps.FromDouble(batch));
        }

        // Normalize and scale
        Parallel.For(0, batch, b =>
        {
            for (int f = 0; f < features; f++)
            {
                T normalized = numOps.Divide(
                    numOps.Subtract(inputData[b * features + f], meanData[f]),
                    numOps.Sqrt(numOps.Add(varData[f], eps)));
                outputData[b * features + f] = numOps.Add(numOps.Multiply(gammaData[f], normalized), betaData[f]);
            }
        });

        mean = TensorAllocator.Rent<T>([features], new Vector<T>(meanData));
        variance = TensorAllocator.Rent<T>([features], new Vector<T>(varData));

        // Return with original shape (restore 1D if input was 1D)
        var result = TensorAllocator.Rent<T>(workingInput._shape, new Vector<T>(outputData));
        var bnResult = was1D ? result.Reshape([features]) : result;
        DifferentiableOps.RecordIfActive("BatchNorm", bnResult, new[] { input, gamma, beta }, BackwardFunctions<T>.BatchNormBackward, new object[] { mean, variance, epsilon });
        return bnResult;
    }

    /// <summary>
    /// Batch normalization for 3D tensors [channels, height, width].
    /// Normalizes per channel across height and width dimensions.
    /// </summary>
    private Tensor<T> BatchNorm3D<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, T eps, INumericOperations<T> numOps, out Tensor<T> mean, out Tensor<T> variance)
    {
        int channels = input._shape[0];
        int height = input._shape[1];
        int width = input._shape[2];
        int spatialSize = height * width;

        var inputData = input.GetDataArray();
        var gammaData = gamma.GetDataArray();
        var betaData = beta.GetDataArray();

        var meanData = new T[channels];
        var varData = new T[channels];
        var outputData = new T[input.Length];

        // Compute mean per channel (across height, width)
        Parallel.For(0, channels, c =>
        {
            T sum = numOps.Zero;
            int channelOffset = c * spatialSize;
            for (int s = 0; s < spatialSize; s++)
            {
                sum = numOps.Add(sum, inputData[channelOffset + s]);
            }
            meanData[c] = numOps.Divide(sum, numOps.FromDouble(spatialSize));
        });

        // Compute variance per channel
        Parallel.For(0, channels, c =>
        {
            T sumSq = numOps.Zero;
            T channelMean = meanData[c];
            int channelOffset = c * spatialSize;
            for (int s = 0; s < spatialSize; s++)
            {
                T diff = numOps.Subtract(inputData[channelOffset + s], channelMean);
                sumSq = numOps.Add(sumSq, numOps.Multiply(diff, diff));
            }
            varData[c] = numOps.Divide(sumSq, numOps.FromDouble(spatialSize));
        });

        // Normalize and scale
        Parallel.For(0, channels, c =>
        {
            T channelMean = meanData[c];
            T channelStdInv = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[c], eps)));
            T gammaVal = gammaData[c];
            T betaVal = betaData[c];
            int channelOffset = c * spatialSize;

            for (int s = 0; s < spatialSize; s++)
            {
                T normalized = numOps.Multiply(numOps.Subtract(inputData[channelOffset + s], channelMean), channelStdInv);
                outputData[channelOffset + s] = numOps.Add(numOps.Multiply(gammaVal, normalized), betaVal);
            }
        });

        mean = TensorAllocator.Rent<T>([channels], new Vector<T>(meanData));
        variance = TensorAllocator.Rent<T>([channels], new Vector<T>(varData));
        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(outputData));
    }


    /// <summary>
    /// Batch normalization for 4D tensors [batch, channels, height, width].
    /// Normalizes per channel across batch, height, and width dimensions.
    /// </summary>
    private Tensor<T> BatchNorm4D<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, T eps, INumericOperations<T> numOps, out Tensor<T> mean, out Tensor<T> variance)
    {
        int batch = input._shape[0];
        int channels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];
        int spatialSize = height * width;
        int elementsPerChannel = batch * spatialSize;

        var inputData = input.GetDataArray();
        var gammaData = gamma.GetDataArray();
        var betaData = beta.GetDataArray();

        // Float fast path with SIMD
        if (inputData is float[] inF && gammaData is float[] gamF && betaData is float[] betF)
        {
            float epsF = numOps.ToDouble(eps) is double d ? (float)d : 1e-5f;
            var meanF = new float[channels];
            var varF = new float[channels];
#if NET5_0_OR_GREATER
            var outF = GC.AllocateUninitializedArray<float>(inF.Length);
#else
            var outF = new float[inF.Length];
#endif
            BatchNorm4DFloat(inF, gamF, betF, epsF, batch, channels, spatialSize, meanF, varF, outF);
            mean = (Tensor<T>)(object)TensorAllocator.Rent<T>(new[] { channels }, (Vector<T>)(object)Vector<float>.FromMemory(meanF));
            variance = (Tensor<T>)(object)TensorAllocator.Rent<T>(new[] { channels }, (Vector<T>)(object)Vector<float>.FromMemory(varF));
            return (Tensor<T>)(object)TensorAllocator.Rent<T>(input._shape, (Vector<T>)(object)Vector<float>.FromMemory(outF));
        }

        var meanData = new T[channels];
        var varData = new T[channels];
        var outputData = new T[input.Length];

        for (int c = 0; c < channels; c++)
        {
            // Mean
            T sum = numOps.Zero;
            for (int n = 0; n < batch; n++)
            {
                int offset = n * channels * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                    sum = numOps.Add(sum, inputData[offset + s]);
            }
            meanData[c] = numOps.Divide(sum, numOps.FromDouble(elementsPerChannel));

            // Variance
            T sumSq = numOps.Zero;
            for (int n = 0; n < batch; n++)
            {
                int offset = n * channels * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    T diff = numOps.Subtract(inputData[offset + s], meanData[c]);
                    sumSq = numOps.Add(sumSq, numOps.Multiply(diff, diff));
                }
            }
            varData[c] = numOps.Divide(sumSq, numOps.FromDouble(elementsPerChannel));

            // Normalize
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[c], eps)));
            for (int n = 0; n < batch; n++)
            {
                int offset = n * channels * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = offset + s;
                    outputData[idx] = numOps.Add(numOps.Multiply(gammaData[c], numOps.Multiply(numOps.Subtract(inputData[idx], meanData[c]), invStd)), betaData[c]);
                }
            }
        }

        mean = TensorAllocator.Rent<T>(new[] { channels }, new Vector<T>(meanData));
        variance = TensorAllocator.Rent<T>(new[] { channels }, new Vector<T>(varData));
        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(outputData));
    }

    private static unsafe void BatchNorm4DFloat(float[] input, float[] gamma, float[] beta, float eps,
        int batch, int channels, int spatialSize, float[] meanOut, float[] varOut, float[] output)
    {
        int elementsPerChannel = batch * spatialSize;
        float invCount = 1f / elementsPerChannel;

        // Parallelize across channels — each channel's mean/var/normalize is independent
        Parallel.For(0, channels, c =>
        {
            BatchNorm4DFloatChannel(input, gamma, beta, eps, batch, channels, spatialSize,
                invCount, c, meanOut, varOut, output);
        });
    }

    private static unsafe void BatchNorm4DFloatChannel(float[] input, float[] gamma, float[] beta, float eps,
        int batch, int channels, int spatialSize, float invCount, int c,
        float[] meanOut, float[] varOut, float[] output)
    {
        {
#if NET5_0_OR_GREATER
            if (System.Runtime.Intrinsics.X86.Fma.IsSupported && spatialSize >= 32)
            {
                // Pin once for all 3 passes — avoids per-pass pinning overhead
                fixed (float* inp = input, outp = output)
                {
                    // Pass 1: Mean with 4x unrolled SIMD accumulation
                    float sum = 0f;
                    var vsum0 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                    var vsum1 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                    var vsum2 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                    var vsum3 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                    for (int n = 0; n < batch; n++)
                    {
                        int offset = n * channels * spatialSize + c * spatialSize;
                        float* ptr = inp + offset;
                        int s = 0;
                        int simdLen = spatialSize & ~31;
                        for (; s < simdLen; s += 32)
                        {
                            vsum0 = System.Runtime.Intrinsics.X86.Avx.Add(vsum0, System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s));
                            vsum1 = System.Runtime.Intrinsics.X86.Avx.Add(vsum1, System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s + 8));
                            vsum2 = System.Runtime.Intrinsics.X86.Avx.Add(vsum2, System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s + 16));
                            vsum3 = System.Runtime.Intrinsics.X86.Avx.Add(vsum3, System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s + 24));
                        }
                        for (; s < spatialSize; s++)
                            sum += ptr[s];
                    }
                    vsum0 = System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.Add(vsum0, vsum1), System.Runtime.Intrinsics.X86.Avx.Add(vsum2, vsum3));
                    sum += SimdKernels.HorizontalSum(vsum0);

                    float channelMean = sum * invCount;
                    meanOut[c] = channelMean;

                    // Pass 2: Variance with 4x unrolled FMA
                    float sumSq = 0f;
                    var vmean = System.Runtime.Intrinsics.Vector256.Create(channelMean);
                    var vsq0 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                    var vsq1 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                    var vsq2 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                    var vsq3 = System.Runtime.Intrinsics.Vector256<float>.Zero;
                    for (int n = 0; n < batch; n++)
                    {
                        int offset = n * channels * spatialSize + c * spatialSize;
                        float* ptr = inp + offset;
                        int s = 0;
                        int simdLen = spatialSize & ~31;
                        for (; s < simdLen; s += 32)
                        {
                            var d0 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s), vmean);
                            var d1 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s + 8), vmean);
                            var d2 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s + 16), vmean);
                            var d3 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s + 24), vmean);
                            vsq0 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d0, d0, vsq0);
                            vsq1 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d1, d1, vsq1);
                            vsq2 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d2, d2, vsq2);
                            vsq3 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d3, d3, vsq3);
                        }
                        for (; s < spatialSize; s++)
                        {
                            float diff = ptr[s] - channelMean;
                            sumSq += diff * diff;
                        }
                    }
                    vsq0 = System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.Add(vsq0, vsq1), System.Runtime.Intrinsics.X86.Avx.Add(vsq2, vsq3));
                    sumSq += SimdKernels.HorizontalSum(vsq0);

                    float channelVar = sumSq * invCount;
                    varOut[c] = channelVar;
                    float invStd = 1f / MathF.Sqrt(channelVar + eps);
                    float g = gamma[c];
                    float b = beta[c];

                    // Pass 3: Normalize with 4x unrolled FMA: output = gamma * (x - mean) * invStd + beta
                    var vscale = System.Runtime.Intrinsics.Vector256.Create(g * invStd);
                    var vbias = System.Runtime.Intrinsics.Vector256.Create(b);
                    for (int n = 0; n < batch; n++)
                    {
                        int offset = n * channels * spatialSize + c * spatialSize;
                        float* pi = inp + offset;
                        float* po = outp + offset;
                        int s = 0;
                        int simdLen = spatialSize & ~31;
                        for (; s < simdLen; s += 32)
                        {
                            var d0 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(pi + s), vmean);
                            var d1 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(pi + s + 8), vmean);
                            var d2 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(pi + s + 16), vmean);
                            var d3 = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(pi + s + 24), vmean);
                            System.Runtime.Intrinsics.X86.Avx.Store(po + s, System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d0, vscale, vbias));
                            System.Runtime.Intrinsics.X86.Avx.Store(po + s + 8, System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d1, vscale, vbias));
                            System.Runtime.Intrinsics.X86.Avx.Store(po + s + 16, System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d2, vscale, vbias));
                            System.Runtime.Intrinsics.X86.Avx.Store(po + s + 24, System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(d3, vscale, vbias));
                        }
                        for (; s < spatialSize; s++)
                            outp[offset + s] = g * (inp[offset + s] - channelMean) * invStd + b;
                    }
                }
                return;
            }
            else if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && spatialSize >= 8)
            {
                fixed (float* inp = input, outp = output)
                {
                    // Pass 1: Mean
                    float sum = 0f;
                    for (int n = 0; n < batch; n++)
                    {
                        int offset = n * channels * spatialSize + c * spatialSize;
                        float* ptr = inp + offset;
                        var vsum = System.Runtime.Intrinsics.Vector256<float>.Zero;
                        int s = 0;
                        int simdLen = spatialSize & ~7;
                        for (; s < simdLen; s += 8)
                            vsum = System.Runtime.Intrinsics.X86.Avx.Add(vsum, System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s));
                        sum += SimdKernels.HorizontalSum(vsum);
                        for (; s < spatialSize; s++)
                            sum += ptr[s];
                    }
                    float channelMean = sum * invCount;
                    meanOut[c] = channelMean;

                    // Pass 2: Variance
                    float sumSq = 0f;
                    var vmean = System.Runtime.Intrinsics.Vector256.Create(channelMean);
                    for (int n = 0; n < batch; n++)
                    {
                        int offset = n * channels * spatialSize + c * spatialSize;
                        float* ptr = inp + offset;
                        var vsumSq = System.Runtime.Intrinsics.Vector256<float>.Zero;
                        int s = 0;
                        int simdLen = spatialSize & ~7;
                        for (; s < simdLen; s += 8)
                        {
                            var diff = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(ptr + s), vmean);
                            vsumSq = System.Runtime.Intrinsics.X86.Avx.Add(vsumSq, System.Runtime.Intrinsics.X86.Avx.Multiply(diff, diff));
                        }
                        sumSq += SimdKernels.HorizontalSum(vsumSq);
                        for (; s < spatialSize; s++)
                        {
                            float diff = ptr[s] - channelMean;
                            sumSq += diff * diff;
                        }
                    }
                    float channelVar = sumSq * invCount;
                    varOut[c] = channelVar;
                    float invStd = 1f / MathF.Sqrt(channelVar + eps);
                    float g = gamma[c];
                    float b = beta[c];

                    // Pass 3: Normalize
                    var vscale = System.Runtime.Intrinsics.Vector256.Create(g * invStd);
                    var vbias = System.Runtime.Intrinsics.Vector256.Create(b);
                    for (int n = 0; n < batch; n++)
                    {
                        int offset = n * channels * spatialSize + c * spatialSize;
                        float* pi = inp + offset;
                        float* po = outp + offset;
                        int s = 0;
                        int simdLen = spatialSize & ~7;
                        for (; s < simdLen; s += 8)
                        {
                            var diff = System.Runtime.Intrinsics.X86.Avx.Subtract(System.Runtime.Intrinsics.X86.Avx.LoadVector256(pi + s), vmean);
                            System.Runtime.Intrinsics.X86.Avx.Store(po + s, System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.Multiply(diff, vscale), vbias));
                        }
                        for (; s < spatialSize; s++)
                            outp[offset + s] = g * (inp[offset + s] - channelMean) * invStd + b;
                    }
                }
                return;
            }
#endif
            // Scalar fallback
            {
                // Pass 1: Mean
                float sum = 0f;
                for (int n = 0; n < batch; n++)
                {
                    int offset = n * channels * spatialSize + c * spatialSize;
                    for (int s = 0; s < spatialSize; s++)
                        sum += input[offset + s];
                }
                float channelMean = sum * invCount;
                meanOut[c] = channelMean;

                // Pass 2: Variance
                float sumSq = 0f;
                for (int n = 0; n < batch; n++)
                {
                    int offset = n * channels * spatialSize + c * spatialSize;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        float diff = input[offset + s] - channelMean;
                        sumSq += diff * diff;
                    }
                }
                float channelVar = sumSq * invCount;
                varOut[c] = channelVar;
#if NET5_0_OR_GREATER
                float invStd = 1f / MathF.Sqrt(channelVar + eps);
#else
                float invStd = 1f / (float)Math.Sqrt(channelVar + eps);
#endif
                float g = gamma[c];
                float b = beta[c];

                // Pass 3: Normalize
                for (int n = 0; n < batch; n++)
                {
                    int offset = n * channels * spatialSize + c * spatialSize;
                    for (int s = 0; s < spatialSize; s++)
                        output[offset + s] = g * (input[offset + s] - channelMean) * invStd + b;
                }
            }
        }
    }

    /// <inheritdoc/>
    public Tensor<T> BatchNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        // Handle both 2D [batch, features] and 4D [batch, channels, height, width] tensors
        if (input._shape.Length == 4)
        {
            return BatchNormBackward4D(gradOutput, input, gamma, mean, variance, eps, numOps, out gradGamma, out gradBeta);
        }

        // 2D case: [batch, features]
        int batch = input._shape[0];
        int features = input._shape[1];
        T batchT = numOps.FromDouble(batch);

        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetFlattenedData();
        var gammaData = gamma.GetDataArray();
        var meanData = mean.GetDataArray();
        var varData = variance.GetDataArray();

        var gradGammaData = new T[features];
        var gradBetaData = new T[features];
        var gradInputData = new T[batch * features];

        // Compute gradGamma and gradBeta
        for (int f = 0; f < features; f++)
        {
            T gGamma = numOps.Zero;
            T gBeta = numOps.Zero;
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[f], eps)));

            for (int b = 0; b < batch; b++)
            {
                int idx = b * features + f;
                T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[f]), invStd);
                gGamma = numOps.Add(gGamma, numOps.Multiply(gradOutputData[idx], normalized));
                gBeta = numOps.Add(gBeta, gradOutputData[idx]);
            }

            gradGammaData[f] = gGamma;
            gradBetaData[f] = gBeta;
        }

        // Compute gradInput
        // Standard batch norm backward formula:
        // dx = (gamma / sqrt(var + eps) / N) * (N * dy - sum(dy) - (x - mean) / (var + eps) * sum(dy * (x - mean)))
        // All terms must be scaled by gamma for correctness
        Parallel.For(0, features, f =>
        {
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[f], eps)));
            T gamma = gammaData[f];
            T sumGrad = numOps.Zero;
            T sumGradX = numOps.Zero;

            // Accumulate sums over batch dimension
            for (int b = 0; b < batch; b++)
            {
                int idx = b * features + f;
                sumGrad = numOps.Add(sumGrad, gradOutputData[idx]);
                sumGradX = numOps.Add(sumGradX, numOps.Multiply(gradOutputData[idx], numOps.Subtract(inputData[idx], meanData[f])));
            }

            // Apply gamma scaling to accumulated sums
            T gammaSumGrad = numOps.Multiply(gamma, sumGrad);
            T gammaSumGradX = numOps.Multiply(gamma, sumGradX);

            for (int b = 0; b < batch; b++)
            {
                int idx = b * features + f;
                T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[f]), invStd);
                T gradNorm = numOps.Multiply(gamma, gradOutputData[idx]);
                T term1 = numOps.Multiply(batchT, gradNorm);
                T term2 = gammaSumGrad;
                T term3 = numOps.Multiply(normalized, numOps.Multiply(invStd, gammaSumGradX));
                gradInputData[idx] = numOps.Multiply(numOps.Divide(invStd, batchT), numOps.Subtract(numOps.Subtract(term1, term2), term3));
            }
        });

        gradGamma = TensorAllocator.Rent<T>([features], new Vector<T>(gradGammaData));
        gradBeta = TensorAllocator.Rent<T>([features], new Vector<T>(gradBetaData));
        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }


    /// <summary>
    /// Backward pass for 4D batch normalization [batch, channels, height, width].
    /// </summary>
    private Tensor<T> BatchNormBackward4D<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, T eps, INumericOperations<T> numOps, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        int batch = input._shape[0];
        int channels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];
        int spatialSize = height * width;
        int elementsPerChannel = batch * spatialSize;
        T elementsT = numOps.FromDouble(elementsPerChannel);

        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetDataArray();
        var gammaData = gamma.GetDataArray();
        var meanData = mean.GetDataArray();
        var varData = variance.GetDataArray();

        // Float fast path: direct float arithmetic avoids numOps virtual dispatch (4-8x faster)
        if (typeof(T) == typeof(float)
            && gradOutputData is float[] goF && inputData is float[] inF
            && gammaData is float[] gaF && meanData is float[] meF && varData is float[] vaF)
        {
            float epsF = (float)numOps.ToDouble(eps);
            var ggF = new float[channels];
            var gbF = new float[channels];
            var giF = new float[input.Length];
            float elemF = elementsPerChannel;

            Parallel.For(0, channels, c =>
            {
                float invStd = 1f / MathF.Sqrt(vaF[c] + epsF);
                float mean_c = meF[c];
                float gGamma = 0, gBeta = 0, sumGrad = 0, sumGradX = 0;

                for (int n = 0; n < batch; n++)
                {
                    int baseIdx = (n * channels + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = baseIdx + s;
                        float diff = inF[idx] - mean_c;
                        gGamma += goF[idx] * diff * invStd;
                        gBeta += goF[idx];
                        sumGrad += goF[idx];
                        sumGradX += goF[idx] * diff;
                    }
                }

                ggF[c] = gGamma;
                gbF[c] = gBeta;
                float gamma_c = gaF[c];
                float gammaSumGrad = gamma_c * sumGrad;
                float gammaSumGradX = gamma_c * sumGradX;

                for (int n = 0; n < batch; n++)
                {
                    int baseIdx2 = (n * channels + c) * spatialSize;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = baseIdx2 + s;
                        float normalized = (inF[idx] - mean_c) * invStd;
                        float gradNorm = gamma_c * goF[idx];
                        giF[idx] = invStd / elemF * (elemF * gradNorm - gammaSumGrad - normalized * invStd * gammaSumGradX);
                    }
                }
            });

            gradGamma = TensorAllocator.Rent<T>([channels], (Vector<T>)(object)new Vector<float>(ggF));
            gradBeta = TensorAllocator.Rent<T>([channels], (Vector<T>)(object)new Vector<float>(gbF));
            return TensorAllocator.Rent<T>(input._shape, (Vector<T>)(object)new Vector<float>(giF));
        }

        var gradGammaData = new T[channels];
        var gradBetaData = new T[channels];
        var gradInputData = new T[input.Length];

        // Compute gradGamma and gradBeta per channel (generic fallback)
        Parallel.For(0, channels, c =>
        {
            T gGamma = numOps.Zero;
            T gBeta = numOps.Zero;
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[c], eps)));
            T channelMean = meanData[c];

            for (int n = 0; n < batch; n++)
            {
                int batchOffset = n * channels * spatialSize;
                int channelOffset = c * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = batchOffset + channelOffset + s;
                    T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], channelMean), invStd);
                    gGamma = numOps.Add(gGamma, numOps.Multiply(gradOutputData[idx], normalized));
                    gBeta = numOps.Add(gBeta, gradOutputData[idx]);
                }
            }

            gradGammaData[c] = gGamma;
            gradBetaData[c] = gBeta;
        });

        // Compute gradInput
        Parallel.For(0, channels, c =>
        {
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[c], eps)));
            T gammaC = gammaData[c];
            T channelMean = meanData[c];
            T sumGrad = numOps.Zero;
            T sumGradX = numOps.Zero;

            // Accumulate sums over batch and spatial dimensions
            for (int n = 0; n < batch; n++)
            {
                int batchOffset = n * channels * spatialSize;
                int channelOffset = c * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = batchOffset + channelOffset + s;
                    sumGrad = numOps.Add(sumGrad, gradOutputData[idx]);
                    sumGradX = numOps.Add(sumGradX, numOps.Multiply(gradOutputData[idx], numOps.Subtract(inputData[idx], channelMean)));
                }
            }

            // Apply gamma scaling to accumulated sums
            T gammaSumGrad = numOps.Multiply(gammaC, sumGrad);
            T gammaSumGradX = numOps.Multiply(gammaC, sumGradX);

            for (int n = 0; n < batch; n++)
            {
                int batchOffset = n * channels * spatialSize;
                int channelOffset = c * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = batchOffset + channelOffset + s;
                    T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], channelMean), invStd);
                    T gradNorm = numOps.Multiply(gammaC, gradOutputData[idx]);
                    T term1 = numOps.Multiply(elementsT, gradNorm);
                    T term2 = gammaSumGrad;
                    T term3 = numOps.Multiply(normalized, numOps.Multiply(invStd, gammaSumGradX));
                    gradInputData[idx] = numOps.Multiply(numOps.Divide(invStd, elementsT), numOps.Subtract(numOps.Subtract(term1, term2), term3));
                }
            }
        });

        gradGamma = TensorAllocator.Rent<T>([channels], new Vector<T>(gradGammaData));
        gradBeta = TensorAllocator.Rent<T>([channels], new Vector<T>(gradBetaData));
        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> LayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gamma == null) throw new ArgumentNullException(nameof(gamma));
        if (beta == null) throw new ArgumentNullException(nameof(beta));

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        // Determine normalized dimensions from gamma shape
        // gamma.Shape defines which trailing dimensions to normalize over
        int normalizedDims = gamma._shape.Length;
        int inputRank = input._shape.Length;

        if (normalizedDims > inputRank)
        {
            throw new ArgumentException($"Gamma shape ({string.Join(", ", gamma._shape)}) has more dimensions than input shape ({string.Join(", ", input._shape)})");
        }

        // Verify that gamma shape matches the last N dimensions of input
        for (int i = 0; i < normalizedDims; i++)
        {
            int inputDimIdx = inputRank - normalizedDims + i;
            if (gamma._shape[i] != input._shape[inputDimIdx])
            {
                throw new ArgumentException($"Gamma shape ({string.Join(", ", gamma._shape)}) does not match the last {normalizedDims} dimensions of input shape ({string.Join(", ", input._shape)})");
            }
        }

        // Calculate batch size (product of non-normalized dimensions)
        // and feature size (product of normalized dimensions)
        int batchSize = 1;
        int batchDims = inputRank - normalizedDims;
        var batchShape = new int[Math.Max(1, batchDims)];

        for (int i = 0; i < batchDims; i++)
        {
            batchSize *= input._shape[i];
            batchShape[i] = input._shape[i];
        }

        // Handle case where all dimensions are normalized (batchDims = 0)
        if (batchDims == 0)
        {
            batchSize = 1;
            batchShape = new int[] { 1 };
        }

        int featureSize = gamma.Length; // product of normalized dimensions

        var inputData = input.GetDataArray();
        var gammaData = gamma.GetDataArray();
        var betaData = beta.GetDataArray();

        var meanData = new T[batchSize];
        var varData = new T[batchSize];
        var outputData = new T[batchSize * featureSize];

        // Compute mean, variance, normalize and scale - all fused per batch position
        Parallel.For(0, batchSize, b =>
        {
            int offset = b * featureSize;
            T featureSizeT = numOps.FromDouble(featureSize);

            // Mean
            T sum = numOps.Zero;
            for (int f = 0; f < featureSize; f++)
                sum = numOps.Add(sum, inputData[offset + f]);
            T m = numOps.Divide(sum, featureSizeT);
            meanData[b] = m;

            // Variance
            T sumSq = numOps.Zero;
            for (int f = 0; f < featureSize; f++)
            {
                T diff = numOps.Subtract(inputData[offset + f], m);
                sumSq = numOps.Add(sumSq, numOps.Multiply(diff, diff));
            }
            T v = numOps.Divide(sumSq, featureSizeT);
            varData[b] = v;

            // Normalize and scale
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(v, eps)));
            for (int f = 0; f < featureSize; f++)
            {
                T normalized = numOps.Multiply(numOps.Subtract(inputData[offset + f], m), invStd);
                outputData[offset + f] = numOps.Add(numOps.Multiply(gammaData[f], normalized), betaData[f]);
            }
        });

        // Create mean and variance tensors with batch shape
        mean = TensorAllocator.Rent<T>(batchShape, new Vector<T>(meanData));
        variance = TensorAllocator.Rent<T>(batchShape, new Vector<T>(varData));
        var lnResult = TensorAllocator.Rent<T>(input._shape, new Vector<T>(outputData));
        DifferentiableOps.RecordIfActive("LayerNorm", lnResult, new[] { input, gamma, beta },
            BackwardFunctions<T>.LayerNormBackward, new object[] { mean, variance, epsilon });
        return lnResult;
    }

    /// <inheritdoc/>
    public Tensor<T> LayerNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        // Determine normalized dimensions from gamma shape (same as forward pass)
        int normalizedDims = gamma._shape.Length;
        int inputRank = input._shape.Length;
        int batchDims = inputRank - normalizedDims;

        // Calculate batch size and feature size
        int batchSize = 1;
        for (int i = 0; i < batchDims; i++)
        {
            batchSize *= input._shape[i];
        }
        if (batchDims == 0) batchSize = 1;

        int featureSize = gamma.Length;
        T featureSizeT = numOps.FromDouble(featureSize);

        var gradOutputData = gradOutput.GetFlattenedData();
        var inputData = input.GetFlattenedData();
        var gammaData = gamma.GetFlattenedData();
        var meanData = mean.GetDataArray();
        var varData = variance.GetDataArray();

        var gradGammaData = new T[featureSize];
        var gradBetaData = new T[featureSize];
        var gradInputData = new T[batchSize * featureSize];
        // CLR zeros new T[] already — no explicit clearing needed

        // Compute gradGamma and gradBeta
        for (int b = 0; b < batchSize; b++)
        {
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[b], eps)));
            for (int f = 0; f < featureSize; f++)
            {
                int idx = b * featureSize + f;
                T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[b]), invStd);
                gradGammaData[f] = numOps.Add(gradGammaData[f], numOps.Multiply(gradOutputData[idx], normalized));
                gradBetaData[f] = numOps.Add(gradBetaData[f], gradOutputData[idx]);
            }
        }

        // Compute gradInput
        Parallel.For(0, batchSize, b =>
        {
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(varData[b], eps)));
            T sumGrad = numOps.Zero;
            T sumGradX = numOps.Zero;

            for (int f = 0; f < featureSize; f++)
            {
                int idx = b * featureSize + f;
                T scaledGrad = numOps.Multiply(gammaData[f], gradOutputData[idx]);
                sumGrad = numOps.Add(sumGrad, scaledGrad);
                sumGradX = numOps.Add(sumGradX, numOps.Multiply(scaledGrad, numOps.Subtract(inputData[idx], meanData[b])));
            }

            for (int f = 0; f < featureSize; f++)
            {
                int idx = b * featureSize + f;
                T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], meanData[b]), invStd);
                T gradNorm = numOps.Multiply(gammaData[f], gradOutputData[idx]);
                T term1 = numOps.Multiply(featureSizeT, gradNorm);
                T term2 = sumGrad;
                T term3 = numOps.Multiply(normalized, numOps.Multiply(invStd, sumGradX));
                gradInputData[idx] = numOps.Multiply(numOps.Divide(invStd, featureSizeT), numOps.Subtract(numOps.Subtract(term1, term2), term3));
            }
        });

        gradGamma = TensorAllocator.Rent<T>(gamma._shape, new Vector<T>(gradGammaData));
        gradBeta = TensorAllocator.Rent<T>(gamma._shape, new Vector<T>(gradBetaData));
        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> GroupNorm<T>(Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gamma == null) throw new ArgumentNullException(nameof(gamma));
        if (beta == null) throw new ArgumentNullException(nameof(beta));
        if (!input.IsContiguous) input = input.Contiguous();
        if (numGroups <= 0) throw new ArgumentOutOfRangeException(nameof(numGroups), "Number of groups must be positive.");

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        // Input shape: [batch, channels, ...spatial]
        int batch = input._shape[0];
        int channels = input._shape[1];

        if (channels % numGroups != 0)
        {
            throw new ArgumentException($"Number of channels ({channels}) must be divisible by number of groups ({numGroups}).");
        }

        int channelsPerGroup = channels / numGroups;

        // Compute spatial size (product of all dimensions after batch and channels)
        int spatialSize = 1;
        for (int i = 2; i < input._shape.Length; i++)
        {
            spatialSize *= input._shape[i];
        }

        int groupSize = channelsPerGroup * spatialSize;  // Elements per group

        // Fast float path: Pin() for NativeMemory compatibility
        if (typeof(T) == typeof(float))
        {
            var result = TensorAllocator.RentUninitialized<T>(input._shape);
            // Pin all memory to get float* pointers (works with both managed and NativeMemory)
            using var pinIn = input.Data.Pin();
            using var pinOut = result.Data.Pin();
            using var pinGamma = gamma.Data.Pin();
            using var pinBeta = beta.Data.Pin();
            unsafe
            {
                GroupNormFloatPtr(
                    (float*)pinIn.Pointer, (float*)pinOut.Pointer,
                    (float*)pinGamma.Pointer, (float*)pinBeta.Pointer,
                    batch, channels, spatialSize, numGroups, channelsPerGroup,
                    System.Runtime.CompilerServices.Unsafe.As<T, float>(ref eps),
                    out var meanArr, out var varArr);
                mean = TensorAllocator.Rent<T>(new[] { batch, numGroups },
                    new Vector<T>((T[])(object)meanArr));
                variance = TensorAllocator.Rent<T>(new[] { batch, numGroups },
                    new Vector<T>((T[])(object)varArr));
            }
            DifferentiableOps.RecordIfActive("GroupNorm", result, new[] { input, gamma, beta },
                BackwardFunctions<T>.GroupNormBackward, new object[] { numGroups, mean, variance, epsilon });
            return result;
        }

        var inputData = input.GetDataArray();
        var gammaData = gamma.GetDataArray();
        var betaData = beta.GetDataArray();

        // Mean and variance are computed per batch per group
        var meanData = new T[batch * numGroups];
        var varData = new T[batch * numGroups];
        // Allocate output via TensorAllocator to benefit from pooling.
        // Write directly into the tensor's backing array — avoid the Rent().GetDataArray()
        // round-trip which can return an oversized pooled array.
        var output = TensorAllocator.RentUninitialized<T>(input._shape);
        var outputData = output.GetDataArray();

        // Fused mean + variance + normalize per batch*group
        Parallel.For(0, batch * numGroups, idx =>
        {
            int b = idx / numGroups;
            int g = idx % numGroups;
            int startChannel = g * channelsPerGroup;
            int batchOffset = b * (channels * spatialSize);
            T groupSizeT = numOps.FromDouble(groupSize);

            // Compute mean
            T sum = numOps.Zero;
            for (int c = 0; c < channelsPerGroup; c++)
            {
                int chanOffset = batchOffset + (startChannel + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                    sum = numOps.Add(sum, inputData[chanOffset + s]);
            }
            T groupMean = numOps.Divide(sum, groupSizeT);
            meanData[b * numGroups + g] = groupMean;

            // Compute variance
            T sumSq = numOps.Zero;
            for (int c = 0; c < channelsPerGroup; c++)
            {
                int chanOffset = batchOffset + (startChannel + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    T diff = numOps.Subtract(inputData[chanOffset + s], groupMean);
                    sumSq = numOps.Add(sumSq, numOps.Multiply(diff, diff));
                }
            }
            T groupVar = numOps.Divide(sumSq, groupSizeT);
            varData[b * numGroups + g] = groupVar;

            // Normalize and apply scale/shift
            T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(groupVar, eps)));
            for (int c = 0; c < channelsPerGroup; c++)
            {
                int channel = startChannel + c;
                int chanOffset = batchOffset + channel * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    T normalized = numOps.Multiply(numOps.Subtract(inputData[chanOffset + s], groupMean), invStd);
                    outputData[chanOffset + s] = numOps.Add(numOps.Multiply(gammaData[channel], normalized), betaData[channel]);
                }
            }
        });

        mean = TensorAllocator.Rent<T>([batch, numGroups], Vector<T>.WrapMemory(meanData));
        variance = TensorAllocator.Rent<T>([batch, numGroups], Vector<T>.WrapMemory(varData));
        DifferentiableOps.RecordIfActive("GroupNorm", output, new[] { input, gamma, beta },
            BackwardFunctions<T>.GroupNormBackward, new object[] { numGroups, mean, variance, epsilon });
        return output;
    }

    /// <inheritdoc/>
    public Tensor<T> GroupNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int numGroups, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gamma == null) throw new ArgumentNullException(nameof(gamma));
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (variance == null) throw new ArgumentNullException(nameof(variance));

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);

        int batch = input._shape[0];
        int channels = input._shape[1];
        int channelsPerGroup = channels / numGroups;

        // Compute spatial size
        int spatialSize = 1;
        for (int i = 2; i < input._shape.Length; i++)
        {
            spatialSize *= input._shape[i];
        }

        int groupSize = channelsPerGroup * spatialSize;
        T groupSizeT = numOps.FromDouble(groupSize);

        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetFlattenedData();
        var gammaData = gamma.GetDataArray();
        var meanData = mean.GetDataArray();
        var varData = variance.GetDataArray();

        var gradGammaData = new T[channels];
        var gradBetaData = new T[channels];
        var gradInputData = new T[input.Length];

        // Initialize gradGamma and gradBeta to zero
        for (int c = 0; c < channels; c++)
        {
            gradGammaData[c] = numOps.Zero;
            gradBetaData[c] = numOps.Zero;
        }

        // Compute gradGamma and gradBeta (sum across batch and spatial)
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int g = c / channelsPerGroup;
                T groupMean = meanData[b * numGroups + g];
                T groupVar = varData[b * numGroups + g];
                T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(groupVar, eps)));

                for (int s = 0; s < spatialSize; s++)
                {
                    int idx = b * (channels * spatialSize) + c * spatialSize + s;
                    T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], groupMean), invStd);
                    gradGammaData[c] = numOps.Add(gradGammaData[c], numOps.Multiply(gradOutputData[idx], normalized));
                    gradBetaData[c] = numOps.Add(gradBetaData[c], gradOutputData[idx]);
                }
            }
        }

        // Compute gradInput using the group norm backward formula
        Parallel.For(0, batch, b =>
        {
            for (int g = 0; g < numGroups; g++)
            {
                T groupMean = meanData[b * numGroups + g];
                T groupVar = varData[b * numGroups + g];
                T invStd = numOps.Divide(numOps.One, numOps.Sqrt(numOps.Add(groupVar, eps)));

                // Compute sum of scaled gradients and sum of scaled gradients times normalized values for this group
                T sumGrad = numOps.Zero;
                T sumGradNorm = numOps.Zero;

                int startChannel = g * channelsPerGroup;
                for (int c = 0; c < channelsPerGroup; c++)
                {
                    int channel = startChannel + c;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = b * (channels * spatialSize) + channel * spatialSize + s;
                        T scaledGrad = numOps.Multiply(gammaData[channel], gradOutputData[idx]);
                        T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], groupMean), invStd);
                        sumGrad = numOps.Add(sumGrad, scaledGrad);
                        sumGradNorm = numOps.Add(sumGradNorm, numOps.Multiply(scaledGrad, normalized));
                    }
                }

                // Compute gradient for each element in this group
                for (int c = 0; c < channelsPerGroup; c++)
                {
                    int channel = startChannel + c;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int idx = b * (channels * spatialSize) + channel * spatialSize + s;
                        T normalized = numOps.Multiply(numOps.Subtract(inputData[idx], groupMean), invStd);
                        T gradNorm = numOps.Multiply(gammaData[channel], gradOutputData[idx]);
                        T term1 = numOps.Multiply(groupSizeT, gradNorm);
                        T term2 = sumGrad;
                        T term3 = numOps.Multiply(normalized, sumGradNorm);
                        gradInputData[idx] = numOps.Multiply(numOps.Divide(invStd, groupSizeT), numOps.Subtract(numOps.Subtract(term1, term2), term3));
                    }
                }
            }
        });

        gradGamma = TensorAllocator.Rent<T>([channels], new Vector<T>(gradGammaData));
        gradBeta = TensorAllocator.Rent<T>([channels], new Vector<T>(gradBetaData));
        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }


    /// <inheritdoc/>
    public virtual Tensor<T> RMSNorm<T>(Tensor<T> input, Tensor<T> gamma, double epsilon, out Tensor<T> rms)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gamma == null) throw new ArgumentNullException(nameof(gamma));
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);
        int normalizedDims = gamma._shape.Length;
        int inputRank = input._shape.Length;

        if (normalizedDims > inputRank)
            throw new ArgumentException("Gamma shape has more dimensions than input shape");

        for (int i = 0; i < normalizedDims; i++)
        {
            int inputDimIdx = inputRank - normalizedDims + i;
            if (gamma._shape[i] != input._shape[inputDimIdx])
                throw new ArgumentException("Gamma shape does not match input trailing dimensions");
        }

        int batchSize = 1;
        int batchDims = inputRank - normalizedDims;
        var batchShape = new int[Math.Max(1, batchDims)];
        for (int i = 0; i < batchDims; i++)
        {
            batchSize *= input._shape[i];
            batchShape[i] = input._shape[i];
        }
        if (batchDims == 0) { batchSize = 1; batchShape = [1]; }

        int featureSize = gamma.Length;
        var inputData = input.GetDataArray();
        var gammaData = gamma.GetDataArray();
        var rmsData = new T[batchSize];
        var outputData = new T[batchSize * featureSize];

        Parallel.For(0, batchSize, b =>
        {
            int offset = b * featureSize;
            T featureSizeT = numOps.FromDouble(featureSize);

            // RMS
            T sumSq = numOps.Zero;
            for (int f = 0; f < featureSize; f++)
            {
                T val = inputData[offset + f];
                sumSq = numOps.Add(sumSq, numOps.Multiply(val, val));
            }
            T rmsVal = numOps.Sqrt(numOps.Add(numOps.Divide(sumSq, featureSizeT), eps));
            rmsData[b] = rmsVal;

            // Normalize and scale
            T invRms = numOps.Divide(numOps.One, rmsVal);
            for (int f = 0; f < featureSize; f++)
                outputData[offset + f] = numOps.Multiply(gammaData[f], numOps.Multiply(inputData[offset + f], invRms));
        });

        rms = TensorAllocator.Rent<T>(batchShape, new Vector<T>(rmsData));
        var rmsResult = TensorAllocator.Rent<T>(input._shape, new Vector<T>(outputData));
        DifferentiableOps.RecordIfActive("RMSNorm", rmsResult, new[] { input, gamma },
            BackwardFunctions<T>.RMSNormBackward, new object[] { rms, epsilon });
        return rmsResult;
    }

    /// <inheritdoc/>
    public Tensor<T> RMSNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> rms, double epsilon, out Tensor<T> gradGamma)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gamma == null) throw new ArgumentNullException(nameof(gamma));
        if (rms == null) throw new ArgumentNullException(nameof(rms));

        var numOps = MathHelper.GetNumericOperations<T>();
        int normalizedDims = gamma._shape.Length;
        int inputRank = input._shape.Length;
        int batchDims = inputRank - normalizedDims;
        int batchSize = 1;
        for (int i = 0; i < batchDims; i++) batchSize *= input._shape[i];
        if (batchDims == 0) batchSize = 1;

        int featureSize = gamma.Length;
        var gradOutputData = gradOutput.GetDataArray();
        var inputData = input.GetFlattenedData();
        var gammaData = gamma.GetDataArray();
        var rmsData = rms.GetDataArray();
        var gradGammaData = new T[featureSize];
        var gradInputData = new T[input.Length];
        for (int f = 0; f < featureSize; f++) gradGammaData[f] = numOps.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            T invRms = numOps.Divide(numOps.One, rmsData[b]);
            for (int f = 0; f < featureSize; f++)
            {
                T normalized = numOps.Multiply(inputData[b * featureSize + f], invRms);
                gradGammaData[f] = numOps.Add(gradGammaData[f], numOps.Multiply(gradOutputData[b * featureSize + f], normalized));
            }
        }

        Parallel.For(0, batchSize, b =>
        {
            T invRms = numOps.Divide(numOps.One, rmsData[b]);
            T sumCorrection = numOps.Zero;
            for (int f = 0; f < featureSize; f++)
            {
                T scaledGrad = numOps.Multiply(gammaData[f], gradOutputData[b * featureSize + f]);
                T normalized = numOps.Multiply(inputData[b * featureSize + f], invRms);
                sumCorrection = numOps.Add(sumCorrection, numOps.Multiply(scaledGrad, normalized));
            }
            T meanCorrection = numOps.Divide(sumCorrection, numOps.FromDouble(featureSize));
            for (int f = 0; f < featureSize; f++)
            {
                T scaledGrad = numOps.Multiply(gammaData[f], gradOutputData[b * featureSize + f]);
                T normalized = numOps.Multiply(inputData[b * featureSize + f], invRms);
                gradInputData[b * featureSize + f] = numOps.Multiply(invRms, numOps.Subtract(scaledGrad, numOps.Multiply(normalized, meanCorrection)));
            }
        });

        gradGamma = TensorAllocator.Rent<T>(gamma._shape, new Vector<T>(gradGammaData));
        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }

    #endregion

    #region Attention Operations

    /// <summary>
    /// Computes scaled dot-product attention.
    /// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    /// From "Attention Is All You Need" (Vaswani et al., 2017)
    /// </summary>
    public Tensor<T> ScaledDotProductAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<bool>? mask,
        double? scale,
        out Tensor<T> attentionWeights)
    {
        if (query == null)
            throw new ArgumentNullException(nameof(query));
        if (key == null)
            throw new ArgumentNullException(nameof(key));
        if (value == null)
            throw new ArgumentNullException(nameof(value));

        // Stride-aware: attention QKV often come from reshape+transpose views
        if (!value.IsContiguous) value = value.Contiguous();

        // Expected shapes: [batch, heads, seq, d_k] for Q, K and [batch, heads, seq, d_v] for V
        if (query.Rank != 4 || key.Rank != 4 || value.Rank != 4)
            throw new ArgumentException("Query, Key, and Value must be 4D tensors [batch, heads, seq, d_k/d_v]");

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = query._shape[0];
        int heads = query._shape[1];
        int seqQ = query._shape[2];
        int d_k = query._shape[3];
        int seqK = key._shape[2];
        int d_v = value._shape[3];

        // Validate shapes
        if (key._shape[0] != batch || key._shape[1] != heads || key._shape[3] != d_k)
            throw new ArgumentException("Key shape mismatch with Query");
        if (value._shape[0] != batch || value._shape[1] != heads || value._shape[2] != seqK)
            throw new ArgumentException("Value shape mismatch with Key");

        // Compute scale factor
        double scaleVal = scale ?? (1.0 / Math.Sqrt(d_k));
        T scaleFactor = numOps.FromDouble(scaleVal);

        // Compute attention scores: Q @ K^T -> [batch, heads, seqQ, seqK]
        var scoresData = new T[batch * heads * seqQ * seqK];
        var queryData = query.GetFlattenedData();
        var keyData = key.GetFlattenedData();

        Parallel.For(0, batch * heads, bh =>
        {
            int qOffset = bh * seqQ * d_k;
            int kOffset = bh * seqK * d_k;
            int sOffset = bh * seqQ * seqK;

            for (int i = 0; i < seqQ; i++)
            {
                for (int j = 0; j < seqK; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < d_k; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(queryData[qOffset + i * d_k + k], keyData[kOffset + j * d_k + k]));
                    }
                    scoresData[sOffset + i * seqK + j] = numOps.Multiply(sum, scaleFactor);
                }
            }
        });

        // Apply mask (if provided) and softmax
        var weightsData = new T[batch * heads * seqQ * seqK];
        T negInf = numOps.FromDouble(double.NegativeInfinity);

        Parallel.For(0, batch * heads, bh =>
        {
            int b = bh / heads;
            int h = bh % heads;
            int offset = (b * heads + h) * seqQ * seqK;

            for (int i = 0; i < seqQ; i++)
            {
                // Apply mask and find max for numerical stability
                T maxVal = negInf;
                for (int j = 0; j < seqK; j++)
                {
                    int idx = offset + i * seqK + j;
                    if (mask != null && !mask[b, h, i, j])
                    {
                        scoresData[idx] = negInf;
                    }
                    if (numOps.GreaterThan(scoresData[idx], maxVal))
                    {
                        maxVal = scoresData[idx];
                    }
                }

                // Compute exp and sum
                T sumExp = numOps.Zero;
                for (int j = 0; j < seqK; j++)
                {
                    int idx = offset + i * seqK + j;
                    T expVal = numOps.Exp(numOps.Subtract(scoresData[idx], maxVal));
                    weightsData[idx] = expVal;
                    sumExp = numOps.Add(sumExp, expVal);
                }

                // Normalize
                for (int j = 0; j < seqK; j++)
                {
                    int idx = offset + i * seqK + j;
                    weightsData[idx] = numOps.Divide(weightsData[idx], sumExp);
                }
            }
        });

        attentionWeights = TensorAllocator.Rent<T>([batch, heads, seqQ, seqK], new Vector<T>(weightsData));

        // Compute output: weights @ V -> [batch, heads, seqQ, d_v]
        var outputData = new T[batch * heads * seqQ * d_v];
        var valueData = value.GetDataArray();

        Parallel.For(0, batch * heads, bh =>
        {
            int b = bh / heads;
            int h = bh % heads;
            int wOffset = (b * heads + h) * seqQ * seqK;
            int vOffset = (b * heads + h) * seqK * d_v;
            int oOffset = (b * heads + h) * seqQ * d_v;

            for (int i = 0; i < seqQ; i++)
            {
                for (int j = 0; j < d_v; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < seqK; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(weightsData[wOffset + i * seqK + k], valueData[vOffset + k * d_v + j]));
                    }
                    outputData[oOffset + i * d_v + j] = sum;
                }
            }
        });

        return TensorAllocator.Rent<T>([batch, heads, seqQ, d_v], new Vector<T>(outputData));
    }

    /// <summary>
    /// Computes the backward pass for scaled dot-product attention.
    /// </summary>
    public Tensor<T> ScaledDotProductAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> attentionWeights,
        double scale,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (query == null)
            throw new ArgumentNullException(nameof(query));
        if (key == null)
            throw new ArgumentNullException(nameof(key));
        if (value == null)
            throw new ArgumentNullException(nameof(value));
        if (attentionWeights == null)
            throw new ArgumentNullException(nameof(attentionWeights));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = query._shape[0];
        int heads = query._shape[1];
        int seqQ = query._shape[2];
        int d_k = query._shape[3];
        int seqK = key._shape[2];
        int d_v = value._shape[3];

        T scaleFactor = numOps.FromDouble(scale);

        var gradOutData = gradOutput.GetFlattenedData();
        var queryData = query.GetFlattenedData();
        var keyData = key.GetFlattenedData();
        var valueData = value.GetFlattenedData();
        var weightsData = attentionWeights.GetDataArray();

        var gradVData = new T[batch * heads * seqK * d_v];
        var gradQData = new T[batch * heads * seqQ * d_k];
        var gradKData = new T[batch * heads * seqK * d_k];

        Parallel.For(0, batch * heads, bh =>
        {
            int b = bh / heads;
            int h = bh % heads;
            int wOffset = (b * heads + h) * seqQ * seqK;
            int gOffset = (b * heads + h) * seqQ * d_v;
            int vOffset = (b * heads + h) * seqK * d_v;
            int qOffset = (b * heads + h) * seqQ * d_k;
            int kOffset = (b * heads + h) * seqK * d_k;

            // Gradient w.r.t. V: weights^T @ gradOutput
            for (int i = 0; i < seqK; i++)
            {
                for (int j = 0; j < d_v; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < seqQ; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(weightsData[wOffset + k * seqK + i], gradOutData[gOffset + k * d_v + j]));
                    }
                    gradVData[vOffset + i * d_v + j] = sum;
                }
            }

            // Gradient w.r.t. attention weights: gradOutput @ V^T
            var gradWeights = new T[seqQ * seqK];
            for (int i = 0; i < seqQ; i++)
            {
                for (int j = 0; j < seqK; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < d_v; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(gradOutData[gOffset + i * d_v + k], valueData[vOffset + j * d_v + k]));
                    }
                    gradWeights[i * seqK + j] = sum;
                }
            }

            // Softmax backward: gradScores = weights * (gradWeights - sum(weights * gradWeights))
            var gradScores = new T[seqQ * seqK];
            for (int i = 0; i < seqQ; i++)
            {
                T dotProduct = numOps.Zero;
                for (int j = 0; j < seqK; j++)
                {
                    dotProduct = numOps.Add(dotProduct, numOps.Multiply(weightsData[wOffset + i * seqK + j], gradWeights[i * seqK + j]));
                }
                for (int j = 0; j < seqK; j++)
                {
                    T w = weightsData[wOffset + i * seqK + j];
                    gradScores[i * seqK + j] = numOps.Multiply(w, numOps.Subtract(gradWeights[i * seqK + j], dotProduct));
                }
            }

            // Gradient w.r.t. Q: gradScores @ K * scale
            for (int i = 0; i < seqQ; i++)
            {
                for (int j = 0; j < d_k; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < seqK; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(gradScores[i * seqK + k], keyData[kOffset + k * d_k + j]));
                    }
                    gradQData[qOffset + i * d_k + j] = numOps.Multiply(sum, scaleFactor);
                }
            }

            // Gradient w.r.t. K: gradScores^T @ Q * scale
            for (int i = 0; i < seqK; i++)
            {
                for (int j = 0; j < d_k; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < seqQ; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(gradScores[k * seqK + i], queryData[qOffset + k * d_k + j]));
                    }
                    gradKData[kOffset + i * d_k + j] = numOps.Multiply(sum, scaleFactor);
                }
            }
        });

        gradQuery = TensorAllocator.Rent<T>(query._shape, new Vector<T>(gradQData));
        gradKey = TensorAllocator.Rent<T>(key._shape, new Vector<T>(gradKData));
        gradValue = TensorAllocator.Rent<T>(value._shape, new Vector<T>(gradVData));

        return gradOutput;
    }

    /// <summary>
    /// Computes memory-efficient attention using the FlashAttention algorithm.
    /// </summary>
    /// <remarks>
    /// This is a CPU implementation of FlashAttention that uses tiling and online softmax
    /// to achieve O(N) memory complexity instead of O(N²) for standard attention.
    /// While the CPU version doesn't benefit as much from the memory hierarchy optimizations
    /// as GPU, it still provides memory savings for long sequences.
    /// </remarks>
    public Tensor<T> FlashAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        double? scale,
        bool isCausal,
        out Tensor<T> softmaxStats,
        Tensor<T>? attentionBias = null)
    {
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (key == null) throw new ArgumentNullException(nameof(key));
        if (value == null) throw new ArgumentNullException(nameof(value));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = query._shape[0];
        int heads = query._shape[1];
        int seqQ = query._shape[2];
        int headDim = query._shape[3];
        int seqK = key._shape[2];

        // Extract and validate bias data if provided
        T[]? biasData = null;
        bool hasBias = false;
        bool biasBroadcastBatch = false;
        if (attentionBias is not null)
        {
            int biasRank = attentionBias._shape.Length;
            if (biasRank == 4)
            {
                if (attentionBias._shape[0] != batch || attentionBias._shape[1] != heads ||
                    attentionBias._shape[2] != seqQ || attentionBias._shape[3] != seqK)
                    throw new ArgumentException(
                        $"4D attention bias shape [{string.Join(",", attentionBias._shape)}] must match [batch={batch}, heads={heads}, seqQ={seqQ}, seqK={seqK}].",
                        nameof(attentionBias));
                biasBroadcastBatch = false;
            }
            else if (biasRank == 3)
            {
                if (attentionBias._shape[0] != heads || attentionBias._shape[1] != seqQ || attentionBias._shape[2] != seqK)
                    throw new ArgumentException(
                        $"3D attention bias shape [{string.Join(",", attentionBias._shape)}] must match [heads={heads}, seqQ={seqQ}, seqK={seqK}].",
                        nameof(attentionBias));
                biasBroadcastBatch = true;
            }
            else
            {
                throw new ArgumentException(
                    $"Attention bias must be rank 3 [heads, seqQ, seqK] or rank 4 [batch, heads, seqQ, seqK], got rank {biasRank}.",
                    nameof(attentionBias));
            }
            biasData = attentionBias.GetDataArray();
            hasBias = true;
        }

        // Compute scale if not provided
        double scaleValue = scale ?? 1.0 / Math.Sqrt(headDim);
        T scaleFactor = numOps.FromDouble(scaleValue);

        // Block sizes for tiling (tuned for CPU cache)
        const int BLOCK_Q = 64;
        const int BLOCK_KV = 64;

        var queryData = query.GetDataArray();
        var keyData = key.GetDataArray();
        var valueData = value.GetDataArray();

        // Output and statistics
        var outputData = new T[batch * heads * seqQ * headDim];
        var statsData = new T[batch * heads * seqQ]; // log-sum-exp statistics

        // Initialize output to zero and stats to negative infinity
        T negInf = numOps.FromDouble(double.NegativeInfinity);
        numOps.Fill(outputData.AsSpan(), numOps.Zero);
        numOps.Fill(statsData.AsSpan(), negInf);

        Parallel.For(0, batch * heads, bh =>
        {
            int b = bh / heads;
            int h = bh % heads;
            int qOffset = (b * heads + h) * seqQ * headDim;
            int kOffset = (b * heads + h) * seqK * headDim;
            int vOffset = (b * heads + h) * seqK * headDim;
            int oOffset = (b * heads + h) * seqQ * headDim;
            int sOffset = (b * heads + h) * seqQ;

            // Local accumulators per query position
            var rowMax = new T[seqQ];
            var rowSum = new T[seqQ];
            numOps.Fill(rowMax.AsSpan(), negInf);
            numOps.Fill(rowSum.AsSpan(), numOps.Zero);

            // Process KV blocks
            for (int kvBlockStart = 0; kvBlockStart < seqK; kvBlockStart += BLOCK_KV)
            {
                int kvBlockEnd = Math.Min(kvBlockStart + BLOCK_KV, seqK);

                // Process Q blocks
                for (int qBlockStart = 0; qBlockStart < seqQ; qBlockStart += BLOCK_Q)
                {
                    int qBlockEnd = Math.Min(qBlockStart + BLOCK_Q, seqQ);

                    // Compute attention scores for this tile
                    for (int qi = qBlockStart; qi < qBlockEnd; qi++)
                    {
                        // Causal: skip if all keys in block are after query
                        if (isCausal && kvBlockStart > qi)
                            continue;

                        T localMax = rowMax[qi];
                        T localSum = rowSum[qi];

                        // Temporary storage for this row's scores in the block
                        var blockScores = new T[kvBlockEnd - kvBlockStart];
                        var blockMaxScore = negInf;

                        // Compute scores: Q @ K^T * scale
                        for (int ki = kvBlockStart; ki < kvBlockEnd; ki++)
                        {
                            // Causal mask
                            if (isCausal && ki > qi)
                            {
                                blockScores[ki - kvBlockStart] = negInf;
                                continue;
                            }

                            T score = numOps.Zero;
                            for (int d = 0; d < headDim; d++)
                            {
                                score = numOps.Add(score, numOps.Multiply(
                                    queryData[qOffset + qi * headDim + d],
                                    keyData[kOffset + ki * headDim + d]));
                            }
                            score = numOps.Multiply(score, scaleFactor);

                            // Add attention bias if provided (applied after QK^T * scale, before softmax)
                            if (hasBias && biasData is not null)
                            {
                                int biasIdx = biasBroadcastBatch
                                    ? (h * seqQ * seqK + qi * seqK + ki)
                                    : (b * heads * seqQ * seqK + h * seqQ * seqK + qi * seqK + ki);
                                score = numOps.Add(score, biasData[biasIdx]);
                            }

                            blockScores[ki - kvBlockStart] = score;

                            if (numOps.ToDouble(score) > numOps.ToDouble(blockMaxScore))
                                blockMaxScore = score;
                        }

                        // Online softmax: compute new max
                        T newMax = numOps.ToDouble(blockMaxScore) > numOps.ToDouble(localMax) ? blockMaxScore : localMax;

                        // Rescale previous accumulator
                        T scale1 = numOps.Exp(numOps.Subtract(localMax, newMax));
                        T newSum = numOps.Multiply(localSum, scale1);

                        // Add new contributions
                        for (int ki = kvBlockStart; ki < kvBlockEnd; ki++)
                        {
                            if (isCausal && ki > qi)
                                continue;

                            T expScore = numOps.Exp(numOps.Subtract(blockScores[ki - kvBlockStart], newMax));
                            newSum = numOps.Add(newSum, expScore);

                            // Update output: O = O * scale1 + exp(score - newMax) * V
                            for (int d = 0; d < headDim; d++)
                            {
                                T vVal = valueData[vOffset + ki * headDim + d];
                                T oVal = outputData[oOffset + qi * headDim + d];

                                // First time rescaling for this block
                                if (ki == kvBlockStart)
                                    oVal = numOps.Multiply(oVal, scale1);

                                outputData[oOffset + qi * headDim + d] = numOps.Add(oVal, numOps.Multiply(expScore, vVal));
                            }
                        }

                        rowMax[qi] = newMax;
                        rowSum[qi] = newSum;
                    }
                }
            }

            // Final normalization: O = O / rowSum
            for (int qi = 0; qi < seqQ; qi++)
            {
                T invSum = numOps.Divide(numOps.One, rowSum[qi]);
                for (int d = 0; d < headDim; d++)
                {
                    outputData[oOffset + qi * headDim + d] = numOps.Multiply(
                        outputData[oOffset + qi * headDim + d], invSum);
                }
                // Store log-sum-exp: logsumexp = max + log(sum)
                statsData[sOffset + qi] = numOps.Add(rowMax[qi], numOps.Log(rowSum[qi]));
            }
        });

        softmaxStats = TensorAllocator.Rent<T>([batch, heads, seqQ], new Vector<T>(statsData));
        return TensorAllocator.Rent<T>([batch, heads, seqQ, headDim], new Vector<T>(outputData));
    }

    /// <summary>
    /// Computes the backward pass for FlashAttention.
    /// </summary>
    public Tensor<T> FlashAttentionBackward<T>(
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
        Tensor<T>? attentionBias = null)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (key == null) throw new ArgumentNullException(nameof(key));
        if (value == null) throw new ArgumentNullException(nameof(value));
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (softmaxStats == null) throw new ArgumentNullException(nameof(softmaxStats));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = query._shape[0];
        int heads = query._shape[1];
        int seqQ = query._shape[2];
        int headDim = query._shape[3];
        int seqK = key._shape[2];

        T scaleFactor = numOps.FromDouble(scale);

        const int BLOCK_Q = 64;
        const int BLOCK_KV = 64;

        var queryData = query.GetDataArray();
        var keyData = key.GetDataArray();
        var valueData = value.GetDataArray();
        var outputData = output.GetDataArray();
        var gradOutData = gradOutput.GetDataArray();
        var statsData = softmaxStats.GetDataArray();

        var gradQData = new T[batch * heads * seqQ * headDim];
        var gradKData = new T[batch * heads * seqK * headDim];
        var gradVData = new T[batch * heads * seqK * headDim];

        // Extract and validate bias data if provided (same validation as forward pass)
        T[]? biasData = null;
        bool hasBias = false;
        bool biasBroadcastBatch = false;
        if (attentionBias is not null)
        {
            int biasRank = attentionBias._shape.Length;
            if (biasRank == 4)
            {
                if (attentionBias._shape[0] != batch || attentionBias._shape[1] != heads ||
                    attentionBias._shape[2] != seqQ || attentionBias._shape[3] != seqK)
                    throw new ArgumentException(
                        $"4D attention bias shape [{string.Join(",", attentionBias._shape)}] must match [batch={batch}, heads={heads}, seqQ={seqQ}, seqK={seqK}].",
                        nameof(attentionBias));
                biasBroadcastBatch = false;
            }
            else if (biasRank == 3)
            {
                if (attentionBias._shape[0] != heads || attentionBias._shape[1] != seqQ || attentionBias._shape[2] != seqK)
                    throw new ArgumentException(
                        $"3D attention bias shape [{string.Join(",", attentionBias._shape)}] must match [heads={heads}, seqQ={seqQ}, seqK={seqK}].",
                        nameof(attentionBias));
                biasBroadcastBatch = true;
            }
            else
            {
                throw new ArgumentException(
                    $"Attention bias must be rank 3 [heads, seqQ, seqK] or rank 4 [batch, heads, seqQ, seqK], got rank {biasRank}.",
                    nameof(attentionBias));
            }
            biasData = attentionBias.GetDataArray();
            hasBias = true;
        }

        T negInf = numOps.FromDouble(double.NegativeInfinity);

        Parallel.For(0, batch * heads, bh =>
        {
            int b = bh / heads;
            int h = bh % heads;
            int qOffset = (b * heads + h) * seqQ * headDim;
            int kOffset = (b * heads + h) * seqK * headDim;
            int vOffset = (b * heads + h) * seqK * headDim;
            int oOffset = (b * heads + h) * seqQ * headDim;
            int sOffset = (b * heads + h) * seqQ;

            // Process in blocks (similar to forward, but recomputing attention)
            for (int kvBlockStart = 0; kvBlockStart < seqK; kvBlockStart += BLOCK_KV)
            {
                int kvBlockEnd = Math.Min(kvBlockStart + BLOCK_KV, seqK);

                for (int qBlockStart = 0; qBlockStart < seqQ; qBlockStart += BLOCK_Q)
                {
                    int qBlockEnd = Math.Min(qBlockStart + BLOCK_Q, seqQ);

                    for (int qi = qBlockStart; qi < qBlockEnd; qi++)
                    {
                        if (isCausal && kvBlockStart > qi)
                            continue;

                        T logsumexp = statsData[sOffset + qi];

                        // Recompute attention weights for this row segment
                        for (int ki = kvBlockStart; ki < kvBlockEnd; ki++)
                        {
                            if (isCausal && ki > qi)
                                continue;

                            // Recompute score
                            T score = numOps.Zero;
                            for (int d = 0; d < headDim; d++)
                            {
                                score = numOps.Add(score, numOps.Multiply(
                                    queryData[qOffset + qi * headDim + d],
                                    keyData[kOffset + ki * headDim + d]));
                            }
                            score = numOps.Multiply(score, scaleFactor);

                            // Add attention bias if provided (must match forward pass)
                            if (hasBias && biasData is not null)
                            {
                                int biasIdx = biasBroadcastBatch
                                    ? (h * seqQ * seqK + qi * seqK + ki)
                                    : (b * heads * seqQ * seqK + h * seqQ * seqK + qi * seqK + ki);
                                score = numOps.Add(score, biasData[biasIdx]);
                            }

                            // Recompute attention weight: exp(score - logsumexp)
                            T attnWeight = numOps.Exp(numOps.Subtract(score, logsumexp));

                            // Gradient w.r.t. V: attnWeight * gradOutput
                            for (int d = 0; d < headDim; d++)
                            {
                                T gradO = gradOutData[oOffset + qi * headDim + d];
                                int vIdx = vOffset + ki * headDim + d;
                                lock (gradVData) // Thread safety for accumulation
                                {
                                    gradVData[vIdx] = numOps.Add(gradVData[vIdx], numOps.Multiply(attnWeight, gradO));
                                }
                            }

                            // Compute dS = attnWeight * (dO @ V - sum(attnWeight * dO @ V))
                            // First compute dO @ v for this position
                            T doV = numOps.Zero;
                            for (int d = 0; d < headDim; d++)
                            {
                                doV = numOps.Add(doV, numOps.Multiply(
                                    gradOutData[oOffset + qi * headDim + d],
                                    valueData[vOffset + ki * headDim + d]));
                            }

                            // Compute dO @ O for the full row (dot product with output)
                            T doO = numOps.Zero;
                            for (int d = 0; d < headDim; d++)
                            {
                                doO = numOps.Add(doO, numOps.Multiply(
                                    gradOutData[oOffset + qi * headDim + d],
                                    outputData[oOffset + qi * headDim + d]));
                            }

                            // dS = attnWeight * (doV - doO)
                            T dS = numOps.Multiply(attnWeight, numOps.Subtract(doV, doO));
                            dS = numOps.Multiply(dS, scaleFactor);

                            // Gradient w.r.t. Q: dS * K
                            for (int d = 0; d < headDim; d++)
                            {
                                int qIdx = qOffset + qi * headDim + d;
                                gradQData[qIdx] = numOps.Add(gradQData[qIdx],
                                    numOps.Multiply(dS, keyData[kOffset + ki * headDim + d]));
                            }

                            // Gradient w.r.t. K: dS * Q
                            for (int d = 0; d < headDim; d++)
                            {
                                int kIdx = kOffset + ki * headDim + d;
                                lock (gradKData) // Thread safety for accumulation
                                {
                                    gradKData[kIdx] = numOps.Add(gradKData[kIdx],
                                        numOps.Multiply(dS, queryData[qOffset + qi * headDim + d]));
                                }
                            }
                        }
                    }
                }
            }
        });

        gradQuery = TensorAllocator.Rent<T>(query._shape, new Vector<T>(gradQData));
        gradKey = TensorAllocator.Rent<T>(key._shape, new Vector<T>(gradKData));
        gradValue = TensorAllocator.Rent<T>(value._shape, new Vector<T>(gradVData));

        return gradOutput;
    }

    /// <summary>
    /// Computes Grouped Query Attention (GQA) for efficient inference.
    /// </summary>
    /// <remarks>
    /// GQA allows multiple query heads to share the same key-value head,
    /// reducing memory bandwidth and KV-cache size during inference.
    /// </remarks>
    public Tensor<T> GroupedQueryAttention<T>(
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        int numQueriesPerKV,
        double? scale,
        bool isCausal,
        out Tensor<T> attentionWeights)
    {
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (key == null) throw new ArgumentNullException(nameof(key));
        if (value == null) throw new ArgumentNullException(nameof(value));
        if (numQueriesPerKV <= 0) throw new ArgumentOutOfRangeException(nameof(numQueriesPerKV));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = query._shape[0];
        int numQHeads = query._shape[1];
        int seqQ = query._shape[2];
        int headDim = query._shape[3];
        int numKVHeads = key._shape[1];
        int seqK = key._shape[2];

        if (numQHeads != numKVHeads * numQueriesPerKV)
            throw new ArgumentException($"Query heads ({numQHeads}) must equal KV heads ({numKVHeads}) * numQueriesPerKV ({numQueriesPerKV})");

        // Compute scale
        double scaleValue = scale ?? 1.0 / Math.Sqrt(headDim);
        T scaleFactor = numOps.FromDouble(scaleValue);
        T negInf = numOps.FromDouble(double.NegativeInfinity);

        var queryData = query.GetDataArray();
        var keyData = key.GetDataArray();
        var valueData = value.GetDataArray();

        var outputData = new T[batch * numQHeads * seqQ * headDim];
        var weightsData = new T[batch * numQHeads * seqQ * seqK];

        Parallel.For(0, batch * numQHeads, bqh =>
        {
            int b = bqh / numQHeads;
            int qh = bqh % numQHeads;
            int kvh = qh / numQueriesPerKV; // Which KV head this query head uses

            int qOffset = (b * numQHeads + qh) * seqQ * headDim;
            int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
            int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
            int oOffset = (b * numQHeads + qh) * seqQ * headDim;
            int wOffset = (b * numQHeads + qh) * seqQ * seqK;

            for (int qi = 0; qi < seqQ; qi++)
            {
                // Compute attention scores: Q @ K^T * scale
                var scores = new T[seqK];
                T maxScore = negInf;

                for (int ki = 0; ki < seqK; ki++)
                {
                    // Causal mask
                    if (isCausal && ki > qi)
                    {
                        scores[ki] = negInf;
                        continue;
                    }

                    T score = numOps.Zero;
                    for (int d = 0; d < headDim; d++)
                    {
                        score = numOps.Add(score, numOps.Multiply(
                            queryData[qOffset + qi * headDim + d],
                            keyData[kOffset + ki * headDim + d]));
                    }
                    score = numOps.Multiply(score, scaleFactor);
                    scores[ki] = score;

                    if (numOps.ToDouble(score) > numOps.ToDouble(maxScore))
                        maxScore = score;
                }

                // Softmax
                T sumExp = numOps.Zero;
                for (int ki = 0; ki < seqK; ki++)
                {
                    if (isCausal && ki > qi)
                    {
                        weightsData[wOffset + qi * seqK + ki] = numOps.Zero;
                        continue;
                    }
                    T expScore = numOps.Exp(numOps.Subtract(scores[ki], maxScore));
                    weightsData[wOffset + qi * seqK + ki] = expScore;
                    sumExp = numOps.Add(sumExp, expScore);
                }

                // Normalize
                T invSum = numOps.Divide(numOps.One, sumExp);
                for (int ki = 0; ki < seqK; ki++)
                {
                    weightsData[wOffset + qi * seqK + ki] = numOps.Multiply(
                        weightsData[wOffset + qi * seqK + ki], invSum);
                }

                // Compute output: weights @ V
                for (int d = 0; d < headDim; d++)
                {
                    T val = numOps.Zero;
                    for (int ki = 0; ki < seqK; ki++)
                    {
                        val = numOps.Add(val, numOps.Multiply(
                            weightsData[wOffset + qi * seqK + ki],
                            valueData[vOffset + ki * headDim + d]));
                    }
                    outputData[oOffset + qi * headDim + d] = val;
                }
            }
        });

        attentionWeights = TensorAllocator.Rent<T>([batch, numQHeads, seqQ, seqK], new Vector<T>(weightsData));
        return TensorAllocator.Rent<T>([batch, numQHeads, seqQ, headDim], new Vector<T>(outputData));
    }

    /// <summary>
    /// Computes the backward pass for Grouped Query Attention.
    /// </summary>
    public Tensor<T> GroupedQueryAttentionBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        Tensor<T> attentionWeights,
        int numQueriesPerKV,
        double scale,
        out Tensor<T> gradQuery,
        out Tensor<T> gradKey,
        out Tensor<T> gradValue)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (key == null) throw new ArgumentNullException(nameof(key));
        if (value == null) throw new ArgumentNullException(nameof(value));
        if (attentionWeights == null) throw new ArgumentNullException(nameof(attentionWeights));

        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = query._shape[0];
        int numQHeads = query._shape[1];
        int seqQ = query._shape[2];
        int headDim = query._shape[3];
        int numKVHeads = key._shape[1];
        int seqK = key._shape[2];

        T scaleFactor = numOps.FromDouble(scale);

        var queryData = query.GetDataArray();
        var keyData = key.GetDataArray();
        var valueData = value.GetDataArray();
        var weightsData = attentionWeights.GetDataArray();
        var gradOutData = gradOutput.GetDataArray();

        var gradQData = new T[batch * numQHeads * seqQ * headDim];
        var gradKData = new T[batch * numKVHeads * seqK * headDim];
        var gradVData = new T[batch * numKVHeads * seqK * headDim];

        // Lock objects for thread-safe accumulation of gradK and gradV
        var gradKLocks = new object[batch * numKVHeads * seqK];
        var gradVLocks = new object[batch * numKVHeads * seqK];
        for (int i = 0; i < gradKLocks.Length; i++)
        {
            gradKLocks[i] = new object();
            gradVLocks[i] = new object();
        }

        Parallel.For(0, batch * numQHeads, bqh =>
        {
            int b = bqh / numQHeads;
            int qh = bqh % numQHeads;
            int kvh = qh / numQueriesPerKV;

            int qOffset = (b * numQHeads + qh) * seqQ * headDim;
            int kOffset = (b * numKVHeads + kvh) * seqK * headDim;
            int vOffset = (b * numKVHeads + kvh) * seqK * headDim;
            int gOffset = (b * numQHeads + qh) * seqQ * headDim;
            int wOffset = (b * numQHeads + qh) * seqQ * seqK;

            for (int qi = 0; qi < seqQ; qi++)
            {
                // Gradient w.r.t. V: weights^T @ gradOutput
                for (int ki = 0; ki < seqK; ki++)
                {
                    T weight = weightsData[wOffset + qi * seqK + ki];
                    int lockIdx = (b * numKVHeads + kvh) * seqK + ki;

                    for (int d = 0; d < headDim; d++)
                    {
                        T gradV = numOps.Multiply(weight, gradOutData[gOffset + qi * headDim + d]);
                        lock (gradVLocks[lockIdx])
                        {
                            gradVData[vOffset + ki * headDim + d] = numOps.Add(
                                gradVData[vOffset + ki * headDim + d], gradV);
                        }
                    }
                }

                // Gradient w.r.t. attention weights: gradOutput @ V^T
                var gradWeights = new T[seqK];
                for (int ki = 0; ki < seqK; ki++)
                {
                    T sum = numOps.Zero;
                    for (int d = 0; d < headDim; d++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(
                            gradOutData[gOffset + qi * headDim + d],
                            valueData[vOffset + ki * headDim + d]));
                    }
                    gradWeights[ki] = sum;
                }

                // Softmax backward: gradScores = weights * (gradWeights - sum(weights * gradWeights))
                T dotProduct = numOps.Zero;
                for (int ki = 0; ki < seqK; ki++)
                {
                    dotProduct = numOps.Add(dotProduct, numOps.Multiply(
                        weightsData[wOffset + qi * seqK + ki], gradWeights[ki]));
                }

                var gradScores = new T[seqK];
                for (int ki = 0; ki < seqK; ki++)
                {
                    T w = weightsData[wOffset + qi * seqK + ki];
                    gradScores[ki] = numOps.Multiply(w, numOps.Subtract(gradWeights[ki], dotProduct));
                    gradScores[ki] = numOps.Multiply(gradScores[ki], scaleFactor);
                }

                // Gradient w.r.t. Q: gradScores @ K
                for (int d = 0; d < headDim; d++)
                {
                    T sum = numOps.Zero;
                    for (int ki = 0; ki < seqK; ki++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(gradScores[ki], keyData[kOffset + ki * headDim + d]));
                    }
                    gradQData[qOffset + qi * headDim + d] = sum;
                }

                // Gradient w.r.t. K: gradScores^T @ Q (accumulated across query heads sharing this KV)
                for (int ki = 0; ki < seqK; ki++)
                {
                    int lockIdx = (b * numKVHeads + kvh) * seqK + ki;
                    for (int d = 0; d < headDim; d++)
                    {
                        T gradK = numOps.Multiply(gradScores[ki], queryData[qOffset + qi * headDim + d]);
                        lock (gradKLocks[lockIdx])
                        {
                            gradKData[kOffset + ki * headDim + d] = numOps.Add(
                                gradKData[kOffset + ki * headDim + d], gradK);
                        }
                    }
                }
            }
        });

        gradQuery = TensorAllocator.Rent<T>(query._shape, new Vector<T>(gradQData));
        gradKey = TensorAllocator.Rent<T>(key._shape, new Vector<T>(gradKData));
        gradValue = TensorAllocator.Rent<T>(value._shape, new Vector<T>(gradVData));

        return gradOutput;
    }

    /// <summary>
    /// Computes Graph Attention Network (GAT) style attention over graph nodes.
    /// </summary>
    public Tensor<T> GraphAttention<T>(
        Tensor<T> nodeFeatures,
        Tensor<int> edgeSourceIndices,
        Tensor<int> edgeTargetIndices,
        Tensor<T> attentionWeightSource,
        Tensor<T> attentionWeightTarget,
        double leakyReluAlpha,
        out Tensor<T> attentionCoeffs)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        int batchSize = nodeFeatures._shape[0];
        int numNodes = nodeFeatures._shape[1];
        int features = nodeFeatures._shape[2];
        int numEdges = edgeSourceIndices._shape[0];

        var nodeData = nodeFeatures.AsSpan();
        var srcIndices = edgeSourceIndices.AsSpan();
        var tgtIndices = edgeTargetIndices.AsSpan();
        var attnSrcData = attentionWeightSource.AsSpan();
        var attnTgtData = attentionWeightTarget.AsSpan();

        // Output tensors
        var outputData = new T[batchSize * numNodes * features];
        var coeffsData = new T[batchSize * numEdges];

        T alpha = numOps.FromDouble(leakyReluAlpha);

        for (int b = 0; b < batchSize; b++)
        {
            int batchOffset = b * numNodes * features;

            // Step 1: Compute attention scores for each edge
            var edgeScores = new T[numEdges];
            for (int e = 0; e < numEdges; e++)
            {
                int src = srcIndices[e];
                int tgt = tgtIndices[e];

                // Compute a_src^T @ h_src + a_tgt^T @ h_tgt
                T score = numOps.Zero;
                for (int f = 0; f < features; f++)
                {
                    T srcFeature = nodeData[batchOffset + src * features + f];
                    T tgtFeature = nodeData[batchOffset + tgt * features + f];
                    score = numOps.Add(score, numOps.Multiply(attnSrcData[f], srcFeature));
                    score = numOps.Add(score, numOps.Multiply(attnTgtData[f], tgtFeature));
                }

                // Apply LeakyReLU
                if (numOps.ToDouble(score) < 0)
                {
                    score = numOps.Multiply(alpha, score);
                }
                edgeScores[e] = score;
            }

            // Step 2: Compute softmax per target node (normalize over incoming edges)
            // First, group edges by target node
            var targetEdges = new System.Collections.Generic.Dictionary<int, System.Collections.Generic.List<int>>();
            for (int e = 0; e < numEdges; e++)
            {
                int tgt = tgtIndices[e];
                if (!targetEdges.ContainsKey(tgt))
                    targetEdges[tgt] = new System.Collections.Generic.List<int>();
                targetEdges[tgt].Add(e);
            }

            // Softmax per target node
            var attentionWeights = new T[numEdges];
            foreach (var kvp in targetEdges)
            {
                var edges = kvp.Value;

                // Find max for numerical stability
                double maxScore = double.MinValue;
                foreach (int e in edges)
                {
                    double s = numOps.ToDouble(edgeScores[e]);
                    if (s > maxScore) maxScore = s;
                }

                // Compute exp and sum
                double expSum = 0.0;
                var expScores = new double[edges.Count];
                for (int i = 0; i < edges.Count; i++)
                {
                    expScores[i] = Math.Exp(numOps.ToDouble(edgeScores[edges[i]]) - maxScore);
                    expSum += expScores[i];
                }

                // Normalize
                for (int i = 0; i < edges.Count; i++)
                {
                    attentionWeights[edges[i]] = numOps.FromDouble(expScores[i] / expSum);
                }
            }

            // Store attention coefficients
            for (int e = 0; e < numEdges; e++)
            {
                coeffsData[b * numEdges + e] = attentionWeights[e];
            }

            // Step 3: Aggregate features using attention weights
            // h'_i = sum_j alpha_ij * h_j (where j are neighbors of i)
            for (int e = 0; e < numEdges; e++)
            {
                int src = srcIndices[e];
                int tgt = tgtIndices[e];
                T weight = attentionWeights[e];

                for (int f = 0; f < features; f++)
                {
                    T srcFeature = nodeData[batchOffset + src * features + f];
                    T contribution = numOps.Multiply(weight, srcFeature);
                    outputData[batchOffset + tgt * features + f] = numOps.Add(
                        outputData[batchOffset + tgt * features + f], contribution);
                }
            }
        }

        attentionCoeffs = TensorAllocator.Rent<T>(new[] { batchSize, numEdges }, new Vector<T>(coeffsData));
        return TensorAllocator.Rent<T>(nodeFeatures._shape, new Vector<T>(outputData));
    }

    /// <summary>
    /// Computes the backward pass for Graph Attention Network attention.
    /// </summary>
    public Tensor<T> GraphAttentionBackward<T>(
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
        out Tensor<T> gradAttentionWeightTarget)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        int batchSize = nodeFeatures._shape[0];
        int numNodes = nodeFeatures._shape[1];
        int features = nodeFeatures._shape[2];
        int numEdges = edgeSourceIndices._shape[0];

        var nodeData = nodeFeatures.AsSpan();
        var gradOutData = gradOutput.AsSpan();
        var srcIndices = edgeSourceIndices.AsSpan();
        var tgtIndices = edgeTargetIndices.AsSpan();
        var attnSrcData = attentionWeightSource.AsSpan();
        var attnTgtData = attentionWeightTarget.AsSpan();
        var coeffsData = attentionCoeffs.AsSpan();

        T alpha = numOps.FromDouble(leakyReluAlpha);

        var gradNodeData = new T[batchSize * numNodes * features];
        var gradAttnSrc = new T[features];
        var gradAttnTgt = new T[features];

        for (int b = 0; b < batchSize; b++)
        {
            int batchOffset = b * numNodes * features;
            int coeffsOffset = b * numEdges;

            // Group edges by target for softmax backward
            var targetEdges = new System.Collections.Generic.Dictionary<int, System.Collections.Generic.List<int>>();
            for (int e = 0; e < numEdges; e++)
            {
                int tgt = tgtIndices[e];
                if (!targetEdges.ContainsKey(tgt))
                    targetEdges[tgt] = new System.Collections.Generic.List<int>();
                targetEdges[tgt].Add(e);
            }

            // Compute gradients for each edge
            var gradCoeffs = new T[numEdges];

            // Gradient w.r.t. attention coefficients from aggregation
            // dL/d_alpha_ij = dL/d_h'_tgt @ h_src
            for (int e = 0; e < numEdges; e++)
            {
                int src = srcIndices[e];
                int tgt = tgtIndices[e];

                T gradCoeff = numOps.Zero;
                for (int f = 0; f < features; f++)
                {
                    T gradOutTgt = gradOutData[batchOffset + tgt * features + f];
                    T srcFeature = nodeData[batchOffset + src * features + f];
                    gradCoeff = numOps.Add(gradCoeff, numOps.Multiply(gradOutTgt, srcFeature));
                }
                gradCoeffs[e] = gradCoeff;
            }

            // Gradient w.r.t. node features from aggregation
            // dL/d_h_src += alpha_ij * dL/d_h'_tgt
            for (int e = 0; e < numEdges; e++)
            {
                int src = srcIndices[e];
                int tgt = tgtIndices[e];
                T coeff = coeffsData[coeffsOffset + e];

                for (int f = 0; f < features; f++)
                {
                    T gradOutTgt = gradOutData[batchOffset + tgt * features + f];
                    T contribution = numOps.Multiply(coeff, gradOutTgt);
                    gradNodeData[batchOffset + src * features + f] = numOps.Add(
                        gradNodeData[batchOffset + src * features + f], contribution);
                }
            }

            // Backprop through softmax per target node
            foreach (var kvp in targetEdges)
            {
                var edges = kvp.Value;

                // Softmax backward: dL/d_score_i = sum_j (alpha_i * (delta_ij - alpha_j)) * dL/d_alpha_j
                for (int i = 0; i < edges.Count; i++)
                {
                    int ei = edges[i];
                    T alphaI = coeffsData[coeffsOffset + ei];
                    T gradScore = numOps.Zero;

                    for (int j = 0; j < edges.Count; j++)
                    {
                        int ej = edges[j];
                        T alphaJ = coeffsData[coeffsOffset + ej];
                        T gradAlphaJ = gradCoeffs[ej];

                        T jacobian = i == j
                            ? numOps.Multiply(alphaI, numOps.Subtract(numOps.One, alphaJ))
                            : numOps.Negate(numOps.Multiply(alphaI, alphaJ));

                        gradScore = numOps.Add(gradScore, numOps.Multiply(jacobian, gradAlphaJ));
                    }

                    // Backprop through LeakyReLU
                    // Recompute score to check sign
                    int src = srcIndices[ei];
                    int tgt = tgtIndices[ei];
                    T score = numOps.Zero;
                    for (int f = 0; f < features; f++)
                    {
                        T srcFeature = nodeData[batchOffset + src * features + f];
                        T tgtFeature = nodeData[batchOffset + tgt * features + f];
                        score = numOps.Add(score, numOps.Multiply(attnSrcData[f], srcFeature));
                        score = numOps.Add(score, numOps.Multiply(attnTgtData[f], tgtFeature));
                    }

                    T leakyGrad = numOps.ToDouble(score) < 0 ? alpha : numOps.One;
                    gradScore = numOps.Multiply(gradScore, leakyGrad);

                    // Gradient w.r.t. attention weights
                    // dL/d_a_src += dL/d_score * h_src
                    // dL/d_a_tgt += dL/d_score * h_tgt
                    for (int f = 0; f < features; f++)
                    {
                        T srcFeature = nodeData[batchOffset + src * features + f];
                        T tgtFeature = nodeData[batchOffset + tgt * features + f];

                        gradAttnSrc[f] = numOps.Add(gradAttnSrc[f], numOps.Multiply(gradScore, srcFeature));
                        gradAttnTgt[f] = numOps.Add(gradAttnTgt[f], numOps.Multiply(gradScore, tgtFeature));

                        // Gradient w.r.t. node features through attention
                        T gradSrcFromAttn = numOps.Multiply(gradScore, attnSrcData[f]);
                        T gradTgtFromAttn = numOps.Multiply(gradScore, attnTgtData[f]);

                        gradNodeData[batchOffset + src * features + f] = numOps.Add(
                            gradNodeData[batchOffset + src * features + f], gradSrcFromAttn);
                        gradNodeData[batchOffset + tgt * features + f] = numOps.Add(
                            gradNodeData[batchOffset + tgt * features + f], gradTgtFromAttn);
                    }
                }
            }
        }

        gradNodeFeatures = TensorAllocator.Rent<T>(nodeFeatures._shape, new Vector<T>(gradNodeData));
        gradAttentionWeightSource = TensorAllocator.Rent<T>(attentionWeightSource._shape, new Vector<T>(gradAttnSrc));
        gradAttentionWeightTarget = TensorAllocator.Rent<T>(attentionWeightTarget._shape, new Vector<T>(gradAttnTgt));

        return gradOutput;
    }

    /// <summary>
    /// Computes multi-head Graph Attention with concatenation or averaging of heads.
    /// </summary>
    public Tensor<T> MultiHeadGraphAttention<T>(
        Tensor<T> nodeFeatures,
        Tensor<int> edgeSourceIndices,
        Tensor<int> edgeTargetIndices,
        Tensor<T> headWeights,
        Tensor<T> attentionWeightsSource,
        Tensor<T> attentionWeightsTarget,
        double leakyReluAlpha,
        bool concatenate,
        out Tensor<T> attentionCoeffs)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        int batchSize = nodeFeatures._shape[0];
        int numNodes = nodeFeatures._shape[1];
        int inFeatures = nodeFeatures._shape[2];
        int numHeads = headWeights._shape[0];
        int headDim = headWeights._shape[2];
        int numEdges = edgeSourceIndices._shape[0];

        var nodeData = nodeFeatures.AsSpan();
        var weightData = headWeights.AsSpan();
        var attnSrcData = attentionWeightsSource.AsSpan();
        var attnTgtData = attentionWeightsTarget.AsSpan();

        int outFeatures = concatenate ? numHeads * headDim : headDim;
        var outputData = new T[batchSize * numNodes * outFeatures];
        var allCoeffsData = new T[batchSize * numHeads * numEdges];

        for (int h = 0; h < numHeads; h++)
        {
            // Transform node features for this head: [batch, nodes, in_features] @ W_h -> [batch, nodes, head_dim]
            var transformedData = new T[batchSize * numNodes * headDim];

            for (int b = 0; b < batchSize; b++)
            {
                for (int n = 0; n < numNodes; n++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        T sum = numOps.Zero;
                        for (int f = 0; f < inFeatures; f++)
                        {
                            int nodeIdx = b * numNodes * inFeatures + n * inFeatures + f;
                            int weightIdx = h * inFeatures * headDim + f * headDim + d;
                            sum = numOps.Add(sum, numOps.Multiply(nodeData[nodeIdx], weightData[weightIdx]));
                        }
                        transformedData[b * numNodes * headDim + n * headDim + d] = sum;
                    }
                }
            }

            // Extract attention weights for this head
            var headAttnSrc = new T[headDim];
            var headAttnTgt = new T[headDim];
            for (int d = 0; d < headDim; d++)
            {
                headAttnSrc[d] = attnSrcData[h * headDim + d];
                headAttnTgt[d] = attnTgtData[h * headDim + d];
            }

            // Create transformed tensor for this head
            var transformedTensor = TensorAllocator.Rent<T>(new[] { batchSize, numNodes, headDim }, new Vector<T>(transformedData));
            var headAttnSrcTensor = TensorAllocator.Rent<T>(new[] { headDim }, new Vector<T>(headAttnSrc));
            var headAttnTgtTensor = TensorAllocator.Rent<T>(new[] { headDim }, new Vector<T>(headAttnTgt));

            // Apply single-head GAT
            var headOutput = GraphAttention(
                transformedTensor,
                edgeSourceIndices,
                edgeTargetIndices,
                headAttnSrcTensor,
                headAttnTgtTensor,
                leakyReluAlpha,
                out var headCoeffs);

            var headOutputData = headOutput.AsSpan();
            var headCoeffsData = headCoeffs.AsSpan();

            // Store coefficients
            for (int b = 0; b < batchSize; b++)
            {
                for (int e = 0; e < numEdges; e++)
                {
                    allCoeffsData[b * numHeads * numEdges + h * numEdges + e] = headCoeffsData[b * numEdges + e];
                }
            }

            // Combine head outputs
            if (concatenate)
            {
                // Concatenate: place head output at appropriate offset
                for (int b = 0; b < batchSize; b++)
                {
                    for (int n = 0; n < numNodes; n++)
                    {
                        for (int d = 0; d < headDim; d++)
                        {
                            int srcIdx = b * numNodes * headDim + n * headDim + d;
                            int dstIdx = b * numNodes * outFeatures + n * outFeatures + h * headDim + d;
                            outputData[dstIdx] = headOutputData[srcIdx];
                        }
                    }
                }
            }
            else
            {
                // Average: accumulate and divide later
                for (int b = 0; b < batchSize; b++)
                {
                    for (int n = 0; n < numNodes; n++)
                    {
                        for (int d = 0; d < headDim; d++)
                        {
                            int idx = b * numNodes * headDim + n * headDim + d;
                            outputData[idx] = numOps.Add(outputData[idx], headOutputData[idx]);
                        }
                    }
                }
            }
        }

        // If averaging, divide by number of heads
        if (!concatenate)
        {
            T divisor = numOps.FromDouble(numHeads);
            for (int i = 0; i < outputData.Length; i++)
            {
                outputData[i] = numOps.Divide(outputData[i], divisor);
            }
        }

        attentionCoeffs = TensorAllocator.Rent<T>(new[] { batchSize, numHeads, numEdges }, new Vector<T>(allCoeffsData));
        return TensorAllocator.Rent<T>(new[] { batchSize, numNodes, outFeatures }, new Vector<T>(outputData));
    }


    #endregion

    #region Scatter Operations (Graph Neural Networks)

    /// <summary>
    /// Scatter add: Aggregates source values at indices using addition.
    /// Used in GNN message passing for sum aggregation.
    /// </summary>
    public Tensor<T> ScatterAdd<T>(Tensor<T> source, Tensor<int> indices, int dim = 0, int? outputSize = null)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Handle negative dimension
        int actualDim = dim < 0 ? source.Rank + dim : dim;
        if (actualDim < 0 || actualDim >= source.Rank)
            throw new ArgumentException($"Invalid dimension {dim} for tensor with rank {source.Rank}");

        // Determine output size along scatter dimension
        var indicesData = indices.GetFlattenedData();
        int maxIndex = 0;
        for (int i = 0; i < indicesData.Length; i++)
        {
            if (indicesData[i] > maxIndex) maxIndex = indicesData[i];
        }
        int outDimSize = outputSize ?? (maxIndex + 1);

        // Build output shape
        var outputShape = source.Shape.ToArray();
        outputShape[actualDim] = outDimSize;

        var sourceData = source.GetFlattenedData();
        var outputData = new T[outputShape.Aggregate(1, (a, b) => a * b)];

        // Initialize to zero
        for (int i = 0; i < outputData.Length; i++)
            outputData[i] = numOps.Zero;

        // Calculate strides
        int innerSize = 1;
        for (int i = actualDim + 1; i < source.Rank; i++)
            innerSize *= source._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= source._shape[i];

        int srcDimSize = source._shape[actualDim];

        // Scatter add
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int d = 0; d < srcDimSize; d++)
            {
                int targetIdx = indicesData[d % indicesData.Length];
                if (targetIdx < 0 || targetIdx >= outDimSize) continue;

                for (int inner = 0; inner < innerSize; inner++)
                {
                    int srcIdx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int dstIdx = outer * outDimSize * innerSize + targetIdx * innerSize + inner;
                    outputData[dstIdx] = numOps.Add(outputData[dstIdx], sourceData[srcIdx]);
                }
            }
        }

        var scatterAddResult = TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("ScatterAdd", scatterAddResult, source, BackwardFunctions<T>.ScatterAddBackward, new object[] { indices, actualDim });
        return scatterAddResult;
    }

    /// <summary>
    /// Backward pass for scatter add.
    /// </summary>
    public Tensor<T> ScatterAddBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, int[] sourceShape, int dim = 0)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));
        if (!indices.IsContiguous) indices = indices.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? sourceShape.Length + dim : dim;

        var gradOutData = gradOutput.GetFlattenedData();
        var indicesData = indices.GetFlattenedData();
        var gradInputData = new T[sourceShape.Aggregate(1, (a, b) => a * b)];

        int innerSize = 1;
        for (int i = actualDim + 1; i < sourceShape.Length; i++)
            innerSize *= sourceShape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= sourceShape[i];

        int srcDimSize = sourceShape[actualDim];
        int outDimSize = gradOutput._shape[actualDim];

        // Gather gradients back
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int d = 0; d < srcDimSize; d++)
            {
                int targetIdx = indicesData[d % indicesData.Length];
                if (targetIdx < 0 || targetIdx >= outDimSize) continue;

                for (int inner = 0; inner < innerSize; inner++)
                {
                    int srcIdx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int dstIdx = outer * outDimSize * innerSize + targetIdx * innerSize + inner;
                    gradInputData[srcIdx] = gradOutData[dstIdx];
                }
            }
        }

        return TensorAllocator.Rent<T>(sourceShape, new Vector<T>(gradInputData));
    }

    /// <summary>
    /// Scatter mean: Aggregates source values at indices using mean.
    /// </summary>
    public virtual Tensor<T> ScatterMean<T>(Tensor<T> source, Tensor<int> indices, out Tensor<int>? counts, int dim = 0, int? outputSize = null)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? source.Rank + dim : dim;

        var indicesData = indices.GetFlattenedData();
        int maxIndex = 0;
        for (int i = 0; i < indicesData.Length; i++)
        {
            if (indicesData[i] > maxIndex) maxIndex = indicesData[i];
        }
        int outDimSize = outputSize ?? (maxIndex + 1);

        var outputShape = source.Shape.ToArray();
        outputShape[actualDim] = outDimSize;

        var sourceData = source.GetFlattenedData();
        var outputData = new T[outputShape.Aggregate(1, (a, b) => a * b)];
        var countData = new int[outDimSize];

        // Initialize
        for (int i = 0; i < outputData.Length; i++)
            outputData[i] = numOps.Zero;

        int innerSize = 1;
        for (int i = actualDim + 1; i < source.Rank; i++)
            innerSize *= source._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= source._shape[i];

        int srcDimSize = source._shape[actualDim];

        // Sum and count
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int d = 0; d < srcDimSize; d++)
            {
                int targetIdx = indicesData[d % indicesData.Length];
                if (targetIdx < 0 || targetIdx >= outDimSize) continue;

                if (outer == 0) countData[targetIdx]++;

                for (int inner = 0; inner < innerSize; inner++)
                {
                    int srcIdx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int dstIdx = outer * outDimSize * innerSize + targetIdx * innerSize + inner;
                    outputData[dstIdx] = numOps.Add(outputData[dstIdx], sourceData[srcIdx]);
                }
            }
        }

        // Divide by counts
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int d = 0; d < outDimSize; d++)
            {
                if (countData[d] > 0)
                {
                    T divisor = numOps.FromDouble(countData[d]);
                    for (int inner = 0; inner < innerSize; inner++)
                    {
                        int idx = outer * outDimSize * innerSize + d * innerSize + inner;
                        outputData[idx] = numOps.Divide(outputData[idx], divisor);
                    }
                }
            }
        }

        counts = new Tensor<int>([outDimSize], new Vector<int>(countData));
        var scatterMeanResult = TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("ScatterMean", scatterMeanResult, source, BackwardFunctions<T>.ScatterMeanBackward, new object[] { indices, counts });
        return scatterMeanResult;
    }

    /// <summary>
    /// Backward pass for scatter mean.
    /// </summary>
    public Tensor<T> ScatterMeanBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, Tensor<int> counts, int[] sourceShape, int dim = 0)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));
        if (counts == null)
            throw new ArgumentNullException(nameof(counts));
        if (!indices.IsContiguous) indices = indices.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? sourceShape.Length + dim : dim;

        var gradOutData = gradOutput.GetFlattenedData();
        var indicesData = indices.GetFlattenedData();
        var countsData = counts.GetFlattenedData();
        var gradInputData = new T[sourceShape.Aggregate(1, (a, b) => a * b)];

        int innerSize = 1;
        for (int i = actualDim + 1; i < sourceShape.Length; i++)
            innerSize *= sourceShape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= sourceShape[i];

        int srcDimSize = sourceShape[actualDim];
        int outDimSize = gradOutput._shape[actualDim];

        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int d = 0; d < srcDimSize; d++)
            {
                int targetIdx = indicesData[d % indicesData.Length];
                if (targetIdx < 0 || targetIdx >= outDimSize) continue;

                int count = countsData[targetIdx];
                T divisor = count > 0 ? numOps.FromDouble(count) : numOps.One;

                for (int inner = 0; inner < innerSize; inner++)
                {
                    int srcIdx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int dstIdx = outer * outDimSize * innerSize + targetIdx * innerSize + inner;
                    gradInputData[srcIdx] = numOps.Divide(gradOutData[dstIdx], divisor);
                }
            }
        }

        return TensorAllocator.Rent<T>(sourceShape, new Vector<T>(gradInputData));
    }

    /// <summary>
    /// Scatter max: Aggregates source values at indices using maximum.
    /// </summary>
    public Tensor<T> ScatterMax<T>(Tensor<T> source, Tensor<int> indices, out Tensor<int>? argmax, int dim = 0, int? outputSize = null)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? source.Rank + dim : dim;

        var indicesData = indices.GetFlattenedData();
        int maxIndex = 0;
        for (int i = 0; i < indicesData.Length; i++)
        {
            if (indicesData[i] > maxIndex) maxIndex = indicesData[i];
        }
        int outDimSize = outputSize ?? (maxIndex + 1);

        var outputShape = source.Shape.ToArray();
        outputShape[actualDim] = outDimSize;

        var sourceData = source.GetFlattenedData();
        int outputLength = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputLength];
        var argmaxData = new int[outputLength];

        // Initialize to negative infinity and -1
        T negInf = numOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < outputData.Length; i++)
        {
            outputData[i] = negInf;
            argmaxData[i] = -1;
        }

        int innerSize = 1;
        for (int i = actualDim + 1; i < source.Rank; i++)
            innerSize *= source._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= source._shape[i];

        int srcDimSize = source._shape[actualDim];

        // Find max
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int d = 0; d < srcDimSize; d++)
            {
                int targetIdx = indicesData[d % indicesData.Length];
                if (targetIdx < 0 || targetIdx >= outDimSize) continue;

                for (int inner = 0; inner < innerSize; inner++)
                {
                    int srcIdx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int dstIdx = outer * outDimSize * innerSize + targetIdx * innerSize + inner;

                    if (numOps.GreaterThan(sourceData[srcIdx], outputData[dstIdx]))
                    {
                        outputData[dstIdx] = sourceData[srcIdx];
                        argmaxData[dstIdx] = d;
                    }
                }
            }
        }

        argmax = new Tensor<int>(outputShape, new Vector<int>(argmaxData));
        var scatterMaxResult = TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("ScatterMax", scatterMaxResult, source, BackwardFunctions<T>.ScatterMaxBackward, new object[] { indices, argmax });
        return scatterMaxResult;
    }

    /// <summary>
    /// Backward pass for scatter max.
    /// </summary>
    public Tensor<T> ScatterMaxBackward<T>(Tensor<T> gradOutput, Tensor<int> argmax, int[] sourceShape, int dim = 0)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (argmax == null)
            throw new ArgumentNullException(nameof(argmax));

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? sourceShape.Length + dim : dim;

        var gradOutData = gradOutput.GetFlattenedData();
        var argmaxData = argmax.GetFlattenedData();
        var gradInputData = new T[sourceShape.Aggregate(1, (a, b) => a * b)];

        // Initialize to zero
        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        int innerSize = 1;
        for (int i = actualDim + 1; i < sourceShape.Length; i++)
            innerSize *= sourceShape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= sourceShape[i];

        int srcDimSize = sourceShape[actualDim];
        int outDimSize = gradOutput._shape[actualDim];

        // Route gradients to argmax positions
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int d = 0; d < outDimSize; d++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int dstIdx = outer * outDimSize * innerSize + d * innerSize + inner;
                    int srcD = argmaxData[dstIdx];
                    if (srcD >= 0 && srcD < srcDimSize)
                    {
                        int srcIdx = outer * srcDimSize * innerSize + srcD * innerSize + inner;
                        gradInputData[srcIdx] = gradOutData[dstIdx];
                    }
                }
            }
        }

        return TensorAllocator.Rent<T>(sourceShape, new Vector<T>(gradInputData));
    }

    /// <summary>
    /// Scatter softmax: Applies softmax within each group defined by indices.
    /// </summary>
    public Tensor<T> ScatterSoftmax<T>(Tensor<T> source, Tensor<int> indices, int dim = 0, int? outputSize = null)
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? source.Rank + dim : dim;

        var indicesData = indices.GetFlattenedData();
        int maxIndex = 0;
        for (int i = 0; i < indicesData.Length; i++)
        {
            if (indicesData[i] > maxIndex) maxIndex = indicesData[i];
        }
        int numGroups = outputSize ?? (maxIndex + 1);

        var sourceData = source.GetFlattenedData();
        var outputData = new T[source.Length];

        int innerSize = 1;
        for (int i = actualDim + 1; i < source.Rank; i++)
            innerSize *= source._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= source._shape[i];

        int srcDimSize = source._shape[actualDim];

        // For each outer/inner position, compute softmax per group
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Find max per group for numerical stability
                var maxPerGroup = new T[numGroups];
                T negInf = numOps.FromDouble(double.NegativeInfinity);
                for (int g = 0; g < numGroups; g++) maxPerGroup[g] = negInf;

                for (int d = 0; d < srcDimSize; d++)
                {
                    int idx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int group = indicesData[d % indicesData.Length];
                    if (group >= 0 && group < numGroups && numOps.GreaterThan(sourceData[idx], maxPerGroup[group]))
                    {
                        maxPerGroup[group] = sourceData[idx];
                    }
                }

                // Compute exp and sum per group
                var sumPerGroup = new T[numGroups];
                for (int g = 0; g < numGroups; g++) sumPerGroup[g] = numOps.Zero;

                for (int d = 0; d < srcDimSize; d++)
                {
                    int idx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int group = indicesData[d % indicesData.Length];
                    if (group >= 0 && group < numGroups)
                    {
                        T expVal = numOps.Exp(numOps.Subtract(sourceData[idx], maxPerGroup[group]));
                        outputData[idx] = expVal;
                        sumPerGroup[group] = numOps.Add(sumPerGroup[group], expVal);
                    }
                }

                // Normalize
                for (int d = 0; d < srcDimSize; d++)
                {
                    int idx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int group = indicesData[d % indicesData.Length];
                    if (group >= 0 && group < numGroups && !numOps.Equals(sumPerGroup[group], numOps.Zero))
                    {
                        outputData[idx] = numOps.Divide(outputData[idx], sumPerGroup[group]);
                    }
                }
            }
        }

        var scatterSmResult = TensorAllocator.Rent<T>(source._shape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("ScatterSoftmax", scatterSmResult, source, BackwardFunctions<T>.ScatterSoftmaxBackward, new object[] { indices });
        return scatterSmResult;
    }

    /// <summary>
    /// Backward pass for scatter softmax.
    /// </summary>
    public Tensor<T> ScatterSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, Tensor<int> indices, int dim = 0)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (output == null)
            throw new ArgumentNullException(nameof(output));
        if (indices == null)
            throw new ArgumentNullException(nameof(indices));
        if (!output.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");

        var numOps = MathHelper.GetNumericOperations<T>();

        int actualDim = dim < 0 ? output.Rank + dim : dim;

        var indicesData = indices.GetFlattenedData();
        int maxIndex = 0;
        for (int i = 0; i < indicesData.Length; i++)
        {
            if (indicesData[i] > maxIndex) maxIndex = indicesData[i];
        }
        int numGroups = maxIndex + 1;

        var gradOutData = gradOutput.GetFlattenedData();
        var outData = output.GetDataArray();
        var gradInputData = new T[gradOutData.Length];

        int innerSize = 1;
        for (int i = actualDim + 1; i < output.Rank; i++)
            innerSize *= output._shape[i];

        int outerSize = 1;
        for (int i = 0; i < actualDim; i++)
            outerSize *= output._shape[i];

        int srcDimSize = output._shape[actualDim];

        // Softmax backward: grad_input = output * (grad_output - sum(output * grad_output))
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Compute sum(output * grad_output) per group
                var sumPerGroup = new T[numGroups];
                for (int g = 0; g < numGroups; g++) sumPerGroup[g] = numOps.Zero;

                for (int d = 0; d < srcDimSize; d++)
                {
                    int idx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int group = indicesData[d % indicesData.Length];
                    if (group >= 0 && group < numGroups)
                    {
                        sumPerGroup[group] = numOps.Add(sumPerGroup[group], numOps.Multiply(outData[idx], gradOutData[idx]));
                    }
                }

                // Compute gradient
                for (int d = 0; d < srcDimSize; d++)
                {
                    int idx = outer * srcDimSize * innerSize + d * innerSize + inner;
                    int group = indicesData[d % indicesData.Length];
                    if (group >= 0 && group < numGroups)
                    {
                        gradInputData[idx] = numOps.Multiply(outData[idx], numOps.Subtract(gradOutData[idx], sumPerGroup[group]));
                    }
                }
            }
        }

        return TensorAllocator.Rent<T>(output._shape, new Vector<T>(gradInputData));
    }

    #endregion


    #region Tensor Reduction Operations

    /// <summary>
    /// Validates and normalizes reduction axes.
    /// </summary>
    /// <param name="axes">The axes to validate</param>
    /// <param name="rank">The tensor rank</param>
    /// <returns>Normalized, validated, and sorted unique axes</returns>
    private static int[] ValidateAndNormalizeAxes(int[] axes, int rank)
    {
        if (axes == null)
            throw new ArgumentNullException(nameof(axes), "Axes cannot be null");

        if (axes.Length == 0)
            throw new ArgumentException("Axes array cannot be empty", nameof(axes));

        var normalizedAxes = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            int axis = axes[i];
            // Normalize negative indices
            int normalized = axis < 0 ? rank + axis : axis;

            if (normalized < 0 || normalized >= rank)
                throw new ArgumentOutOfRangeException(nameof(axes), $"Axis {axis} is out of range for tensor with rank {rank}. Valid range is [{-rank}, {rank - 1}].");

            normalizedAxes[i] = normalized;
        }

        // Check for duplicates
        var uniqueAxes = normalizedAxes.Distinct().ToArray();
        if (uniqueAxes.Length != axes.Length)
            throw new ArgumentException("Duplicate axes are not allowed", nameof(axes));

        return uniqueAxes.OrderBy(a => a).ToArray();
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMax<T>(Tensor<T> input, int[] axes, bool keepDims, out int[] maxIndices)
    {
        // Stride-aware single-axis max
        if (!input.IsContiguous && axes.Length == 1)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            int axis = axes[0] < 0 ? input.Rank + axes[0] : axes[0];
            if (axis < 0 || axis >= input.Rank) throw new ArgumentOutOfRangeException(nameof(axes), $"Axis {axes[0]} out of range for rank {input.Rank}");
            var (outerSize, axisSize, innerSize) = input.GetReductionDims(axis);
            var outShape = new List<int>();
            for (int d = 0; d < input.Rank; d++) { if (d == axis) { if (keepDims) outShape.Add(1); } else outShape.Add(input._shape[d]); }
            var result = TensorAllocator.RentUninitialized<T>(outShape.ToArray());
            var rArr = result.GetDataArray(); var srcArr = input._storage.GetDataArray();
            maxIndices = new int[outerSize * innerSize];
            for (int o = 0; o < outerSize; o++)
                for (int inner = 0; inner < innerSize; inner++)
                {
                    T best = srcArr[input.ReductionStorageIndex(o, 0, inner, axis)]; int bestIdx = 0;
                    for (int a = 1; a < axisSize; a++) { T v = srcArr[input.ReductionStorageIndex(o, a, inner, axis)]; if (ops.GreaterThan(v, best)) { best = v; bestIdx = a; } }
                    rArr[o * innerSize + inner] = best;
                    maxIndices[o * innerSize + inner] = bestIdx;
                }
            return result;
        }
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input._shape;
        var inputData = input.GetFlattenedData();

        // Validate and normalize axes
        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        // Compute output shape
        var outputShapeList = new List<int>();
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (normalizedAxes.Contains(i))
            {
                if (keepDims) outputShapeList.Add(1);
            }
            else
            {
                outputShapeList.Add(inputShape[i]);
            }
        }
        var outputShape = outputShapeList.Count > 0 ? outputShapeList.ToArray() : [1];

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];
        maxIndices = new int[outputSize];

        // Initialize with minimum values
        T minVal = numOps.MinValue;
        for (int i = 0; i < outputSize; i++)
        {
            outputData[i] = minVal;
            maxIndices[i] = -1;
        }

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(outputShape);

        for (int i = 0; i < input.Length; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (keepDims) outputMultiIndex.Add(0);
                }
                else
                {
                    outputMultiIndex.Add(multiIndex[d]);
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);

            if (numOps.GreaterThan(inputData[i], outputData[outputIdx]))
            {
                outputData[outputIdx] = inputData[i];
                maxIndices[outputIdx] = i;
            }
        }

        var reduceMaxResult = TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("ReduceMax", reduceMaxResult, input, BackwardFunctions<T>.ReduceMaxBackward, new object[] { maxIndices });
        return reduceMaxResult;
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMaxBackward<T>(Tensor<T> gradOutput, int[] maxIndices, int[] inputShape)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int inputSize = inputShape.Aggregate(1, (a, b) => a * b);
        var gradInputData = new T[inputSize];

        for (int i = 0; i < inputSize; i++)
            gradInputData[i] = numOps.Zero;

        var gradOutputData = gradOutput.GetFlattenedData();

        for (int i = 0; i < maxIndices.Length; i++)
        {
            if (maxIndices[i] >= 0 && maxIndices[i] < inputSize)
            {
                gradInputData[maxIndices[i]] = numOps.Add(gradInputData[maxIndices[i]], gradOutputData[i]);
            }
        }

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ReduceMean<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        // Stride-aware single-axis mean
        if (!input.IsContiguous && axes.Length == 1)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            int axis = axes[0] < 0 ? input.Rank + axes[0] : axes[0];
            if (axis < 0 || axis >= input.Rank) throw new ArgumentOutOfRangeException(nameof(axes), $"Axis {axes[0]} out of range for rank {input.Rank}");
            var (outerSize, axisSize, innerSize) = input.GetReductionDims(axis);
            var outShape = new List<int>();
            for (int d = 0; d < input.Rank; d++) { if (d == axis) { if (keepDims) outShape.Add(1); } else outShape.Add(input._shape[d]); }
            var earlyResult = TensorAllocator.RentUninitialized<T>(outShape.ToArray());
            var rArr = earlyResult.GetDataArray(); var srcArr = input._storage.GetDataArray();
            T divisor = ops.FromDouble(axisSize);
            for (int o = 0; o < outerSize; o++)
                for (int inner = 0; inner < innerSize; inner++)
                {
                    T sum = ops.Zero;
                    for (int a = 0; a < axisSize; a++) sum = ops.Add(sum, srcArr[input.ReductionStorageIndex(o, a, inner, axis)]);
                    rArr[o * innerSize + inner] = ops.Divide(sum, divisor);
                }
            DifferentiableOps.RecordUnary("ReduceMean", earlyResult, input,
                BackwardFunctions<T>.ReduceMeanBackward,
                savedState: new object[] { new[] { axis } });
            return earlyResult;
        }
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputShape = input._shape;
        var inputData = input.GetFlattenedData();

        // Validate and normalize axes
        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        var outputShapeList = new List<int>();
        for (int i = 0; i < inputShape.Length; i++)
        {
            if (normalizedAxes.Contains(i))
            {
                if (keepDims) outputShapeList.Add(1);
            }
            else
            {
                outputShapeList.Add(inputShape[i]);
            }
        }
        var outputShape = outputShapeList.Count > 0 ? outputShapeList.ToArray() : [1];

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];
        var counts = new int[outputSize];

        for (int i = 0; i < outputSize; i++)
        {
            outputData[i] = numOps.Zero;
            counts[i] = 0;
        }

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(outputShape);

        for (int i = 0; i < input.Length; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (keepDims) outputMultiIndex.Add(0);
                }
                else
                {
                    outputMultiIndex.Add(multiIndex[d]);
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);
            outputData[outputIdx] = numOps.Add(outputData[outputIdx], inputData[i]);
            counts[outputIdx]++;
        }

        for (int i = 0; i < outputSize; i++)
        {
            if (counts[i] > 0)
            {
                outputData[i] = numOps.Divide(outputData[i], numOps.FromDouble(counts[i]));
            }
        }

        var result = TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("ReduceMean", result, input,
            BackwardFunctions<T>.ReduceMeanBackward,
            savedState: new object[] { normalizedAxes.ToArray() });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceMeanBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] axes)
    {
        if (inputShape == null || inputShape.Length == 0)
            throw new ArgumentNullException(nameof(inputShape), "inputShape cannot be null or empty");

        var numOps = MathHelper.GetNumericOperations<T>();
        int inputSize = inputShape.Aggregate(1, (a, b) => a * b);
        var gradInputData = new T[inputSize];

        // Validate and normalize axes
        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        int reduceCount = 1;
        foreach (var ax in normalizedAxes)
        {
            reduceCount *= inputShape[ax];
        }
        T scale = numOps.Divide(numOps.One, numOps.FromDouble(reduceCount));

        var gradOutputData = gradOutput.GetFlattenedData();
        var gradOutputShape = gradOutput._shape;
        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(gradOutputShape);

        for (int i = 0; i < inputSize; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            int d2 = 0;
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (d2 < gradOutputShape.Length && gradOutputShape[d2] == 1)
                    {
                        outputMultiIndex.Add(0);
                        d2++;
                    }
                }
                else
                {
                    if (d2 < gradOutputShape.Length)
                    {
                        outputMultiIndex.Add(multiIndex[d]);
                        d2++;
                    }
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            while (outputMultiIndex.Count < gradOutputShape.Length)
                outputMultiIndex.Add(0);
            while (outputMultiIndex.Count > gradOutputShape.Length)
                outputMultiIndex.RemoveAt(outputMultiIndex.Count - 1);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], gradOutputShape, outputStrides);
            if (outputIdx < 0 || outputIdx >= gradOutputData.Length)
                throw new InvalidOperationException($"Output index {outputIdx} out of range [0, {gradOutputData.Length}). This indicates a shape mismatch between gradOutput and the expected shape.");
            gradInputData[i] = numOps.Multiply(gradOutputData[outputIdx], scale);
        }

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceVariance<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        // Stride-aware: compute mean then variance with stride math
        if (!input.IsContiguous && axes.Length == 1)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            int axis = axes[0] < 0 ? input.Rank + axes[0] : axes[0];
            if (axis < 0 || axis >= input.Rank) throw new ArgumentOutOfRangeException(nameof(axes), $"Axis {axes[0]} out of range for rank {input.Rank}");
            var (outerSize, axisSize, innerSize) = input.GetReductionDims(axis);
            var outShape = new List<int>();
            for (int d = 0; d < input.Rank; d++) { if (d == axis) { if (keepDims) outShape.Add(1); } else outShape.Add(input._shape[d]); }
            var result = TensorAllocator.RentUninitialized<T>(outShape.ToArray());
            var rArr = result.GetDataArray(); var srcArr = input._storage.GetDataArray();
            T divisor = ops.FromDouble(axisSize);
            for (int o = 0; o < outerSize; o++)
                for (int inner = 0; inner < innerSize; inner++)
                {
                    // Mean
                    T sum = ops.Zero;
                    for (int a = 0; a < axisSize; a++) sum = ops.Add(sum, srcArr[input.ReductionStorageIndex(o, a, inner, axis)]);
                    T meanVal = ops.Divide(sum, divisor);
                    // Variance
                    T varSum = ops.Zero;
                    for (int a = 0; a < axisSize; a++) { T diff = ops.Subtract(srcArr[input.ReductionStorageIndex(o, a, inner, axis)], meanVal); varSum = ops.Add(varSum, ops.Multiply(diff, diff)); }
                    rArr[o * innerSize + inner] = ops.Divide(varSum, divisor);
                }
            return result;
        }
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.GetFlattenedData();
        var inputShape = input._shape;

        // First compute the mean
        var mean = ReduceMean(input, axes, keepDims: true);
        var meanData = mean.GetDataArray();
        var meanShape = mean._shape;

        // Normalize axes
        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        // Compute output shape
        var outputShapeList = new List<int>();
        for (int d = 0; d < inputShape.Length; d++)
        {
            if (normalizedAxes.Contains(d))
            {
                if (keepDims) outputShapeList.Add(1);
            }
            else
            {
                outputShapeList.Add(inputShape[d]);
            }
        }
        if (outputShapeList.Count == 0) outputShapeList.Add(1);
        var outputShape = outputShapeList.ToArray();

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];

        // Compute reduction count (number of elements being reduced)
        int reduceCount = 1;
        foreach (var ax in normalizedAxes)
        {
            reduceCount *= inputShape[ax];
        }
        T scale = numOps.Divide(numOps.One, numOps.FromDouble(reduceCount));

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(outputShape);
        var meanStrides = ComputeStrides(meanShape);

        // Accumulate squared differences from mean
        int inputSize = input.Length;
        for (int i = 0; i < inputSize; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            var outputMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (keepDims) outputMultiIndex.Add(0);
                }
                else
                {
                    outputMultiIndex.Add(multiIndex[d]);
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);

            // Get mean value (mean tensor has keepDims=true shape)
            var meanMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                    meanMultiIndex.Add(0);
                else
                    meanMultiIndex.Add(multiIndex[d]);
            }
            int meanIdx = MultiToFlatIndex([.. meanMultiIndex], meanShape, meanStrides);

            T diff = numOps.Subtract(inputData[i], meanData[meanIdx]);
            T squaredDiff = numOps.Multiply(diff, diff);
            outputData[outputIdx] = numOps.Add(outputData[outputIdx], squaredDiff);
        }

        // Divide by count to get variance
        for (int i = 0; i < outputSize; i++)
        {
            outputData[i] = numOps.Multiply(outputData[i], scale);
        }

        return TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceVarianceBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, int[] axes)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (mean == null)
            throw new ArgumentNullException(nameof(mean));

        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.GetFlattenedData();
        var inputShape = input._shape;
        var meanData = mean.GetDataArray();
        var meanShape = mean._shape;
        var gradOutputData = gradOutput.GetFlattenedData();
        var gradOutputShape = gradOutput._shape;

        int inputSize = input.Length;
        var gradInputData = new T[inputSize];

        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        int reduceCount = 1;
        foreach (var ax in normalizedAxes)
        {
            reduceCount *= inputShape[ax];
        }
        T scale = numOps.Divide(numOps.FromDouble(2.0), numOps.FromDouble(reduceCount));

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(gradOutputShape);
        var meanStrides = ComputeStrides(meanShape);

        for (int i = 0; i < inputSize; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            // Map to output and mean indices using helper methods
            int outputIdx = MapToReducedIndex(multiIndex, inputShape, gradOutputShape, normalizedAxes, outputStrides);
            int meanIdx = MapToMeanIndex(multiIndex, inputShape, meanShape, normalizedAxes, meanStrides);

            // gradient = 2 * (x - mean) * gradOutput / N
            T diff = numOps.Subtract(inputData[i], meanData[meanIdx]);
            gradInputData[i] = numOps.Multiply(numOps.Multiply(diff, scale), gradOutputData[outputIdx]);
        }

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceLogVariance<T>(Tensor<T> input, int[] axes, bool keepDims, double epsilon = 1e-8)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        // Compute variance first
        var variance = ReduceVariance(input, axes, keepDims);
        var varianceData = variance.GetDataArray();

        // Apply log(variance + epsilon)
        T eps = numOps.FromDouble(epsilon);
        for (int i = 0; i < varianceData.Length; i++)
        {
            varianceData[i] = numOps.Log(numOps.Add(varianceData[i], eps));
        }

        var logVarResult = TensorAllocator.Rent<T>(variance._shape, new Vector<T>(varianceData));
        // Compute mean inside NoGradScope to avoid adding disconnected tape entries
        Tensor<T> mean;
        using (new NoGradScope<T>())
        {
            mean = ReduceMean(input, axes, keepDims);
        }
        DifferentiableOps.RecordUnary("ReduceLogVariance", logVarResult, input, BackwardFunctions<T>.ReduceLogVarianceBackward, new object[] { axes, mean, variance });
        return logVarResult;
    }

    /// <inheritdoc/>
    public Tensor<T> ReduceLogVarianceBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, Tensor<T> variance, int[] axes)
    {
        if (gradOutput == null)
            throw new ArgumentNullException(nameof(gradOutput));
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (mean == null)
            throw new ArgumentNullException(nameof(mean));
        if (variance == null)
            throw new ArgumentNullException(nameof(variance));
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.GetFlattenedData();
        var inputShape = input._shape;
        var meanData = mean.GetDataArray();
        var meanShape = mean._shape;
        var varianceData = variance.GetDataArray();
        var varianceShape = variance._shape;
        var gradOutputData = gradOutput.GetFlattenedData();
        var gradOutputShape = gradOutput._shape;

        int inputSize = input.Length;
        var gradInputData = new T[inputSize];

        var normalizedAxes = ValidateAndNormalizeAxes(axes, inputShape.Length);

        int reduceCount = 1;
        foreach (var ax in normalizedAxes)
        {
            reduceCount *= inputShape[ax];
        }
        T scale = numOps.Divide(numOps.FromDouble(2.0), numOps.FromDouble(reduceCount));

        var inputStrides = ComputeStrides(inputShape);
        var outputStrides = ComputeStrides(gradOutputShape);
        var meanStrides = ComputeStrides(meanShape);
        var varianceStrides = ComputeStrides(varianceShape);

        for (int i = 0; i < inputSize; i++)
        {
            var multiIndex = FlatToMultiIndex(i, inputShape, inputStrides);

            // Map to output/variance index
            var outputMultiIndex = new List<int>();
            int d2 = 0;
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                {
                    if (d2 < gradOutputShape.Length && gradOutputShape[d2] == 1)
                    {
                        outputMultiIndex.Add(0);
                        d2++;
                    }
                }
                else
                {
                    if (d2 < gradOutputShape.Length)
                    {
                        outputMultiIndex.Add(multiIndex[d]);
                        d2++;
                    }
                }
            }
            if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);
            while (outputMultiIndex.Count < gradOutputShape.Length) outputMultiIndex.Add(0);
            while (outputMultiIndex.Count > gradOutputShape.Length) outputMultiIndex.RemoveAt(outputMultiIndex.Count - 1);

            int outputIdx = MultiToFlatIndex([.. outputMultiIndex], gradOutputShape, outputStrides);

            // Map to mean index
            var meanMultiIndex = new List<int>();
            for (int d = 0; d < inputShape.Length; d++)
            {
                if (normalizedAxes.Contains(d))
                    meanMultiIndex.Add(0);
                else
                    meanMultiIndex.Add(multiIndex[d]);
            }
            int meanIdx = MultiToFlatIndex([.. meanMultiIndex], meanShape, meanStrides);
            int varianceIdx = MultiToFlatIndex([.. outputMultiIndex], varianceShape, varianceStrides);

            // gradient = 2 * (x - mean) / (N * variance) * gradOutput
            // = (x - mean) * scale / variance * gradOutput
            T diff = numOps.Subtract(inputData[i], meanData[meanIdx]);
            T varianceVal = varianceData[varianceIdx];
            // Avoid division by zero
            if (numOps.LessThanOrEquals(varianceVal, numOps.Zero))
                varianceVal = numOps.FromDouble(1e-8);
            T gradScale = numOps.Divide(scale, varianceVal);
            gradInputData[i] = numOps.Multiply(numOps.Multiply(diff, gradScale), gradOutputData[outputIdx]);
        }

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    // Helper methods for reduction operations
    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int[] FlatToMultiIndex(int flatIndex, int[] shape, int[] strides)
    {
        var multiIndex = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            multiIndex[i] = flatIndex / strides[i];
            flatIndex %= strides[i];
        }
        return multiIndex;
    }

    private static int MultiToFlatIndex(int[] multiIndex, int[] shape, int[] strides)
    {
        int flatIndex = 0;
        for (int i = 0; i < multiIndex.Length; i++)
        {
            flatIndex += multiIndex[i] * strides[i];
        }
        return flatIndex;
    }

    /// <summary>
    /// Maps a multi-index from input space to reduced output space for variance backward pass.
    /// </summary>
    private static int MapToReducedIndex(int[] multiIndex, int[] inputShape, int[] outputShape, int[] normalizedAxes, int[] outputStrides)
    {
        var outputMultiIndex = new List<int>();
        int d2 = 0;
        for (int d = 0; d < inputShape.Length; d++)
        {
            if (Array.IndexOf(normalizedAxes, d) >= 0)
            {
                if (d2 < outputShape.Length && outputShape[d2] == 1)
                {
                    outputMultiIndex.Add(0);
                    d2++;
                }
            }
            else
            {
                if (d2 < outputShape.Length)
                {
                    outputMultiIndex.Add(multiIndex[d]);
                    d2++;
                }
            }
        }
        if (outputMultiIndex.Count == 0) outputMultiIndex.Add(0);
        while (outputMultiIndex.Count < outputShape.Length) outputMultiIndex.Add(0);
        while (outputMultiIndex.Count > outputShape.Length) outputMultiIndex.RemoveAt(outputMultiIndex.Count - 1);

        return MultiToFlatIndex([.. outputMultiIndex], outputShape, outputStrides);
    }

    /// <summary>
    /// Maps a multi-index from input space to mean tensor space for variance backward pass.
    /// </summary>
    private static int MapToMeanIndex(int[] multiIndex, int[] inputShape, int[] meanShape, int[] normalizedAxes, int[] meanStrides)
    {
        var meanMultiIndex = new List<int>();
        for (int d = 0; d < inputShape.Length; d++)
        {
            if (Array.IndexOf(normalizedAxes, d) >= 0)
                meanMultiIndex.Add(0);
            else
                meanMultiIndex.Add(multiIndex[d]);
        }
        return MultiToFlatIndex([.. meanMultiIndex], meanShape, meanStrides);
    }

    #endregion

    #region Spatial Operations

    /// <inheritdoc/>
    public virtual Tensor<T> Upsample<T>(Tensor<T> input, int scaleH, int scaleW)
    {
        var shape = input._shape;
        if (shape.Length < 2)
            throw new ArgumentException("Upsample requires tensor with at least 2 dimensions for height and width.");

        // Industry-standard: last two dimensions are height and width
        int heightIdx = shape.Length - 2;
        int widthIdx = shape.Length - 1;
        int height = shape[heightIdx];
        int width = shape[widthIdx];

        // Flatten all leading dimensions into a single "batch" dimension
        int flatBatch = 1;
        for (int i = 0; i < shape.Length - 2; i++)
        {
            flatBatch *= shape[i];
        }

        int newHeight = height * scaleH;
        int newWidth = width * scaleW;

        var inputData = input.GetFlattenedData();
        var outputData = new T[flatBatch * newHeight * newWidth];

        Parallel.For(0, flatBatch, fb =>
        {
            for (int oh = 0; oh < newHeight; oh++)
            {
                int ih = oh / scaleH;
                for (int ow = 0; ow < newWidth; ow++)
                {
                    int iw = ow / scaleW;
                    int inputIdx = (fb * height + ih) * width + iw;
                    int outputIdx = (fb * newHeight + oh) * newWidth + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        });

        // Create output shape preserving all leading dimensions
        var outputShape = new int[shape.Length];
        for (int i = 0; i < shape.Length - 2; i++)
        {
            outputShape[i] = shape[i];
        }
        outputShape[heightIdx] = newHeight;
        outputShape[widthIdx] = newWidth;

        var upsampleResult = TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("Upsample", upsampleResult, input,
            BackwardFunctions<T>.UpsampleBackward,
            savedState: new object[] { scaleH, scaleW });
        return upsampleResult;
    }

    /// <inheritdoc/>
    public Tensor<T> UpsampleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int scaleH, int scaleW)
    {
        if (inputShape.Length < 2)
            throw new ArgumentException("UpsampleBackward requires inputShape with at least 2 dimensions for height and width.");

        var numOps = MathHelper.GetNumericOperations<T>();

        // Industry-standard: last two dimensions are height and width
        int heightIdx = inputShape.Length - 2;
        int widthIdx = inputShape.Length - 1;
        int height = inputShape[heightIdx];
        int width = inputShape[widthIdx];

        // Flatten all leading dimensions into a single "batch" dimension
        int flatBatch = 1;
        int totalInput = 1;
        for (int i = 0; i < inputShape.Length; i++)
        {
            totalInput *= inputShape[i];
            if (i < inputShape.Length - 2)
            {
                flatBatch *= inputShape[i];
            }
        }

        int newHeight = height * scaleH;
        int newWidth = width * scaleW;

        var gradOutputData = gradOutput.GetFlattenedData();
        var gradInputData = new T[totalInput];

        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        Parallel.For(0, flatBatch, fb =>
        {
            for (int oh = 0; oh < newHeight; oh++)
            {
                int ih = oh / scaleH;
                for (int ow = 0; ow < newWidth; ow++)
                {
                    int iw = ow / scaleW;
                    int gradOutputIdx = (fb * newHeight + oh) * newWidth + ow;
                    int gradInputIdx = (fb * height + ih) * width + iw;
                    // No lock needed - each flatBatch partition owns disjoint gradInput slices
                    gradInputData[gradInputIdx] = numOps.Add(gradInputData[gradInputIdx], gradOutputData[gradOutputIdx]);
                }
            }
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> PixelShuffle<T>(Tensor<T> input, int upscaleFactor)
    {
        var shape = input._shape;
        if (shape.Length != 4)
            throw new ArgumentException("PixelShuffle expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int height = shape[2];
        int width = shape[3];

        int r = upscaleFactor;
        if (channels % (r * r) != 0)
            throw new ArgumentException($"Number of channels ({channels}) must be divisible by r^2 ({r * r})");

        int newChannels = channels / (r * r);
        int newHeight = height * r;
        int newWidth = width * r;

        var inputData = input.GetFlattenedData();
        var outputData = new T[batch * newChannels * newHeight * newWidth];

        Parallel.For(0, batch, b =>
        {
            for (int oc = 0; oc < newChannels; oc++)
            {
                for (int oh = 0; oh < newHeight; oh++)
                {
                    for (int ow = 0; ow < newWidth; ow++)
                    {
                        int ih = oh / r;
                        int iw = ow / r;
                        int subH = oh % r;
                        int subW = ow % r;
                        int ic = oc * r * r + subH * r + subW;

                        int inputIdx = ((b * channels + ic) * height + ih) * width + iw;
                        int outputIdx = ((b * newChannels + oc) * newHeight + oh) * newWidth + ow;
                        outputData[outputIdx] = inputData[inputIdx];
                    }
                }
            }
        });

        var psResult = TensorAllocator.Rent<T>([batch, newChannels, newHeight, newWidth], new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("PixelShuffle", psResult, input, BackwardFunctions<T>.PixelShuffleBackward, new object[] { upscaleFactor });
        return psResult;
    }

    /// <inheritdoc/>
    public Tensor<T> PixelShuffleBackward<T>(Tensor<T> gradOutput, int[] inputShape, int upscaleFactor)
    {
        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];

        int r = upscaleFactor;
        int newChannels = channels / (r * r);
        int newHeight = height * r;
        int newWidth = width * r;

        var gradOutputData = gradOutput.GetFlattenedData();
        var gradInputData = new T[batch * channels * height * width];

        Parallel.For(0, batch, b =>
        {
            for (int oc = 0; oc < newChannels; oc++)
            {
                for (int oh = 0; oh < newHeight; oh++)
                {
                    for (int ow = 0; ow < newWidth; ow++)
                    {
                        int ih = oh / r;
                        int iw = ow / r;
                        int subH = oh % r;
                        int subW = ow % r;
                        int ic = oc * r * r + subH * r + subW;

                        int gradInputIdx = ((b * channels + ic) * height + ih) * width + iw;
                        int gradOutputIdx = ((b * newChannels + oc) * newHeight + oh) * newWidth + ow;
                        gradInputData[gradInputIdx] = gradOutputData[gradOutputIdx];
                    }
                }
            }
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    public Tensor<T> AffineGrid<T>(Tensor<T> theta, int outputHeight, int outputWidth)
    {
        if (theta == null) throw new ArgumentNullException(nameof(theta));
        if (theta._shape.Length != 3 || theta._shape[1] != 2 || theta._shape[2] != 3)
            throw new ArgumentException("AffineGrid expects theta shape [batch, 2, 3]");

        int batchSize = theta._shape[0];
        var grid = TensorAllocator.Rent<T>([batchSize, outputHeight, outputWidth, 2]);
        var numOps = MathHelper.GetNumericOperations<T>();

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < outputHeight; h++)
            {
                T yNorm = outputHeight == 1
                    ? numOps.Zero
                    : numOps.FromDouble((double)h / (outputHeight - 1) * 2.0 - 1.0);

                for (int w = 0; w < outputWidth; w++)
                {
                    T xNorm = outputWidth == 1
                        ? numOps.Zero
                        : numOps.FromDouble((double)w / (outputWidth - 1) * 2.0 - 1.0);

                    T xPrime = numOps.Add(
                        numOps.Add(
                            numOps.Multiply(theta[b, 0, 0], xNorm),
                            numOps.Multiply(theta[b, 0, 1], yNorm)),
                        theta[b, 0, 2]);

                    T yPrime = numOps.Add(
                        numOps.Add(
                            numOps.Multiply(theta[b, 1, 0], xNorm),
                            numOps.Multiply(theta[b, 1, 1], yNorm)),
                        theta[b, 1, 2]);

                    grid[b, h, w, 0] = xPrime;
                    grid[b, h, w, 1] = yPrime;
                }
            }
        }

        return grid;
    }

    public virtual Tensor<T> GridSample<T>(Tensor<T> input, Tensor<T> grid)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (grid == null) throw new ArgumentNullException(nameof(grid));

        if (input._shape.Length != 4)
            throw new ArgumentException("GridSample expects input shape [batch, height, width, channels]");
        if (grid._shape.Length != 4 || grid._shape[3] != 2)
            throw new ArgumentException("GridSample expects grid shape [batch, outH, outW, 2]");
        if (input._shape[0] != grid._shape[0])
            throw new ArgumentException("GridSample batch size mismatch between input and grid");

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int inH = input._shape[1];
        int inW = input._shape[2];
        int channels = input._shape[3];
        int outH = grid._shape[1];
        int outW = grid._shape[2];

        var output = TensorAllocator.Rent<T>([batch, outH, outW, channels]);

        T heightScale = numOps.FromDouble((inH - 1) / 2.0);
        T widthScale = numOps.FromDouble((inW - 1) / 2.0);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < outH; h++)
            {
                for (int w = 0; w < outW; w++)
                {
                    T gridX = grid[b, h, w, 0];
                    T gridY = grid[b, h, w, 1];

                    T srcX = numOps.Multiply(numOps.Add(gridX, numOps.One), widthScale);
                    T srcY = numOps.Multiply(numOps.Add(gridY, numOps.One), heightScale);

                    double srcXDouble = Convert.ToDouble(srcX);
                    double srcYDouble = Convert.ToDouble(srcY);
                    int x0 = Math.Max(0, Math.Min((int)Math.Floor(srcXDouble), inW - 1));
                    int x1 = Math.Max(0, Math.Min(x0 + 1, inW - 1));
                    int y0 = Math.Max(0, Math.Min((int)Math.Floor(srcYDouble), inH - 1));
                    int y1 = Math.Max(0, Math.Min(y0 + 1, inH - 1));

                    T wx1 = numOps.Subtract(srcX, numOps.FromDouble(x0));
                    T wx0 = numOps.Subtract(numOps.One, wx1);
                    T wy1 = numOps.Subtract(srcY, numOps.FromDouble(y0));
                    T wy0 = numOps.Subtract(numOps.One, wy1);

                    for (int c = 0; c < channels; c++)
                    {
                        T v00 = input[b, y0, x0, c];
                        T v01 = input[b, y0, x1, c];
                        T v10 = input[b, y1, x0, c];
                        T v11 = input[b, y1, x1, c];

                        T interp = numOps.Add(
                            numOps.Add(
                                numOps.Multiply(numOps.Multiply(v00, wx0), wy0),
                                numOps.Multiply(numOps.Multiply(v01, wx1), wy0)),
                            numOps.Add(
                                numOps.Multiply(numOps.Multiply(v10, wx0), wy1),
                                numOps.Multiply(numOps.Multiply(v11, wx1), wy1)));

                        output[b, h, w, c] = interp;
                    }
                }
            }
        }

        DifferentiableOps.RecordBinary("GridSample", output, input, grid, BackwardFunctions<T>.GridSampleBackward);
        return output;
    }

    /// <summary>
    /// Extracts sliding local blocks (im2col) from a batched input tensor.
    /// Input: [batch, channels, height, width] -> Output: [batch, channels * kH * kW, L]
    /// </summary>
    public virtual Tensor<T> Unfold<T>(Tensor<T> input, int[] kernelSize, int[] stride, int[] padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernelSize == null || kernelSize.Length < 2) throw new ArgumentException("kernelSize must have at least 2 elements.", nameof(kernelSize));
        if (stride == null || stride.Length < 2) throw new ArgumentException("stride must have at least 2 elements.", nameof(stride));
        if (padding == null || padding.Length < 2) throw new ArgumentException("padding must have at least 2 elements.", nameof(padding));
        if (input.Rank != 4) throw new ArgumentException($"Unfold requires 4D input [batch, channels, height, width]. Got rank {input.Rank}.");
        if (kernelSize[0] <= 0 || kernelSize[1] <= 0) throw new ArgumentException("Kernel size elements must be positive.", nameof(kernelSize));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive.", nameof(stride));
        if (padding[0] < 0 || padding[1] < 0) throw new ArgumentException("Padding elements must be non-negative.", nameof(padding));

        // Save original input for tape before potential Contiguous() replacement
        var originalUnfoldInput = input;
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int channels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];
        int kH = kernelSize[0], kW = kernelSize[1];
        int sH = stride[0], sW = stride[1];
        int pH = padding[0], pW = padding[1];

        int outH = (int)Math.Floor((height + 2.0 * pH - kH) / sH) + 1;
        int outW = (int)Math.Floor((width + 2.0 * pW - kW) / sW) + 1;
        if (outH <= 0 || outW <= 0)
            throw new ArgumentException($"Invalid Unfold dimensions: output would be {outH}x{outW}. Check kernel, stride, and padding.");
        int colLength = outH * outW;
        int colChannels = channels * kH * kW;

        var result = TensorAllocator.Rent<T>(new[] { batch, colChannels, colLength });
        var inputData = input.GetDataArray();
        var resultData = result.GetDataArray();

        for (int b = 0; b < batch; b++)
        {
            int batchInputOffset = b * channels * height * width;
            int batchOutputOffset = b * colChannels * colLength;

            for (int c = 0; c < channels; c++)
            {
                for (int ki = 0; ki < kH; ki++)
                {
                    for (int kj = 0; kj < kW; kj++)
                    {
                        int colRow = (c * kH + ki) * kW + kj;
                        for (int oh = 0; oh < outH; oh++)
                        {
                            int ih = oh * sH + ki - pH;
                            for (int ow = 0; ow < outW; ow++)
                            {
                                int iw = ow * sW + kj - pW;
                                int colIdx = batchOutputOffset + colRow * colLength + oh * outW + ow;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                    resultData[colIdx] = inputData[batchInputOffset + c * height * width + ih * width + iw];
                                else
                                    resultData[colIdx] = numOps.Zero;
                            }
                        }
                    }
                }
            }
        }

        if (GradientTape<T>.Current is not null)
            DifferentiableOps.RecordUnary("Unfold", result, originalUnfoldInput, BackwardFunctions<T>.UnfoldBackward,
                new object[] { (int[])kernelSize.Clone(), (int[])stride.Clone(), (int[])padding.Clone() });
        return result;
    }

    /// <summary>
    /// Combines sliding local blocks back into a full tensor (col2im).
    /// Input: [batch, channels * kH * kW, L] -> Output: [batch, channels, outputH, outputW]
    /// </summary>
    public virtual Tensor<T> Fold<T>(Tensor<T> input, int[] outputSize, int[] kernelSize, int[] stride, int[] padding)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (outputSize == null || outputSize.Length < 2) throw new ArgumentException("outputSize must have at least 2 elements.", nameof(outputSize));
        if (kernelSize == null || kernelSize.Length < 2) throw new ArgumentException("kernelSize must have at least 2 elements.", nameof(kernelSize));
        if (stride == null || stride.Length < 2) throw new ArgumentException("stride must have at least 2 elements.", nameof(stride));
        if (padding == null || padding.Length < 2) throw new ArgumentException("padding must have at least 2 elements.", nameof(padding));
        if (input.Rank != 3) throw new ArgumentException($"Fold requires 3D input [batch, C*kH*kW, L]. Got rank {input.Rank}.");
        if (kernelSize[0] <= 0 || kernelSize[1] <= 0) throw new ArgumentException("Kernel size elements must be positive.", nameof(kernelSize));
        if (stride[0] <= 0 || stride[1] <= 0) throw new ArgumentException("Stride elements must be positive.", nameof(stride));
        if (padding[0] < 0 || padding[1] < 0) throw new ArgumentException("Padding elements must be non-negative.", nameof(padding));

        var originalFoldInput = input;
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int kH = kernelSize[0], kW = kernelSize[1];
        int sH = stride[0], sW = stride[1];
        int pH = padding[0], pW = padding[1];
        int outH = outputSize[0], outW = outputSize[1];
        int colChannels = input._shape[1];
        int kernelElements = kH * kW;
        if (kernelElements == 0 || colChannels % kernelElements != 0)
            throw new ArgumentException($"Input channels ({colChannels}) must be divisible by kernel elements ({kernelElements}).");
        int channels = colChannels / kernelElements;
        int colLength = input._shape[2];
        int unfoldH = (int)Math.Floor((outH + 2.0 * pH - kH) / sH) + 1;
        int unfoldW = (int)Math.Floor((outW + 2.0 * pW - kW) / sW) + 1;
        int expectedColLength = unfoldH * unfoldW;
        if (colLength != expectedColLength)
            throw new ArgumentException($"Column length {colLength} doesn't match expected {expectedColLength} for output {outH}x{outW} with kernel {kH}x{kW}, stride {sH}x{sW}, pad {pH}x{pW}.");

        var result = TensorAllocator.Rent<T>(new[] { batch, channels, outH, outW });
        var inputData = input.GetDataArray();
        var resultData = result.GetDataArray();

        // Initialize to zero
        for (int i = 0; i < resultData.Length; i++)
            resultData[i] = numOps.Zero;

        for (int b = 0; b < batch; b++)
        {
            int batchInputOffset = b * colChannels * colLength;
            int batchOutputOffset = b * channels * outH * outW;

            for (int c = 0; c < channels; c++)
            {
                for (int ki = 0; ki < kH; ki++)
                {
                    for (int kj = 0; kj < kW; kj++)
                    {
                        int colRow = (c * kH + ki) * kW + kj;
                        for (int oh = 0; oh < unfoldH; oh++)
                        {
                            int ih = oh * sH + ki - pH;
                            for (int ow = 0; ow < unfoldW; ow++)
                            {
                                int iw = ow * sW + kj - pW;
                                if (ih >= 0 && ih < outH && iw >= 0 && iw < outW)
                                {
                                    int colIdx = batchInputOffset + colRow * colLength + oh * unfoldW + ow;
                                    int outIdx = batchOutputOffset + c * outH * outW + ih * outW + iw;
                                    resultData[outIdx] = numOps.Add(resultData[outIdx], inputData[colIdx]);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (GradientTape<T>.Current is not null)
            DifferentiableOps.RecordUnary("Fold", result, originalFoldInput, BackwardFunctions<T>.FoldBackward,
                new object[] { (int[])kernelSize.Clone(), (int[])stride.Clone(), (int[])padding.Clone() });
        return result;
    }

    public (Tensor<T> real, Tensor<T> imag) ComplexMatMul<T>(Tensor<T> aReal, Tensor<T> aImag, Tensor<T> bReal, Tensor<T> bImag)
    {
        if (aReal == null || aImag == null || bReal == null || bImag == null)
            throw new ArgumentNullException("ComplexMatMul inputs cannot be null");
        var aShape = aReal._shape;
        var bShape = bReal._shape;
        if (aShape.Length != 2 || bShape.Length != 2 || aShape[1] != bShape[0])
            throw new ArgumentException("ComplexMatMul expects shapes [M,K] x [K,N]");

        int m = aShape[0];
        int k = aShape[1];
        int n = bShape[1];

        var realOut = TensorAllocator.Rent<T>([m, n]);
        var imagOut = TensorAllocator.Rent<T>([m, n]);

        var numOps = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T realSum = numOps.Zero;
                T imagSum = numOps.Zero;
                for (int kk = 0; kk < k; kk++)
                {
                    var ar = aReal[i, kk];
                    var ai = aImag[i, kk];
                    var br = bReal[kk, j];
                    var bi = bImag[kk, j];
                    // (ar + i ai) * (br + i bi) = (ar*br - ai*bi) + i(ar*bi + ai*br)
                    realSum = numOps.Add(realSum, numOps.Subtract(numOps.Multiply(ar, br), numOps.Multiply(ai, bi)));
                    imagSum = numOps.Add(imagSum, numOps.Add(numOps.Multiply(ar, bi), numOps.Multiply(ai, br)));
                }
                realOut[i, j] = realSum;
                imagOut[i, j] = imagSum;
            }
        }

        return (realOut, imagOut);
    }

    public Tensor<T> ComplexMagnitudeSquared<T>(Tensor<T> real, Tensor<T> imag)
    {
        if (real == null || imag == null)
            throw new ArgumentNullException("ComplexMagnitudeSquared inputs cannot be null");
        if (!real._shape.SequenceEqual(imag._shape))
            throw new ArgumentException("Real and imaginary parts must have the same shape");

        var numOps = MathHelper.GetNumericOperations<T>();
        var output = TensorAllocator.RentUninitialized<T>(real._shape);
        for (int idx = 0; idx < real.Length; idx++)
        {
            var r = real[idx];
            var i = imag[idx];
            output[idx] = numOps.Add(numOps.Multiply(r, r), numOps.Multiply(i, i));
        }
        DifferentiableOps.RecordBinary("ComplexMagnitudeSquared", output, real, imag, BackwardFunctions<T>.ComplexMagnitudeSquaredBackward);
        return output;
    }

    public (Tensor<T> real, Tensor<T> imag) ComplexNormalize<T>(Tensor<T> real, Tensor<T> imag)
    {
        var magSq = ComplexMagnitudeSquared(real, imag);
        var total = TensorSum(magSq);
        var numOps = MathHelper.GetNumericOperations<T>();
        if (numOps.Equals(total, numOps.Zero))
            return (real.Clone(), imag.Clone());
        var denom = numOps.Sqrt(total);
        var denomTensor = TensorAllocator.RentUninitialized<T>(magSq._shape);
        denomTensor.Fill(denom);
        var realNorm = TensorDivide(real, denomTensor);
        var imagNorm = TensorDivide(imag, denomTensor);
        return (realNorm, imagNorm);
    }

    /// <inheritdoc/>
    public Tensor<T> Crop<T>(Tensor<T> input, int top, int left, int height, int width)
    {
        var shape = input._shape;
        if (shape.Length != 4)
            throw new ArgumentException("Crop expects 4D tensor [batch, channels, height, width]");

        int batch = shape[0];
        int channels = shape[1];
        int inputHeight = shape[2];
        int inputWidth = shape[3];

        if (top < 0 || left < 0 || top + height > inputHeight || left + width > inputWidth)
            throw new ArgumentException("Crop region is out of bounds");

        var inputData = input.GetFlattenedData();
        var outputData = new T[batch * channels * height * width];

        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < height; oh++)
            {
                int ih = top + oh;
                for (int ow = 0; ow < width; ow++)
                {
                    int iw = left + ow;
                    int inputIdx = ((b * channels + c) * inputHeight + ih) * inputWidth + iw;
                    int outputIdx = ((b * channels + c) * height + oh) * width + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        });

        var cropResult = TensorAllocator.Rent<T>([batch, channels, height, width], new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("Crop", cropResult, input, BackwardFunctions<T>.CropBackward, new object[] { top, left });
        return cropResult;
    }

    /// <inheritdoc/>
    public Tensor<T> CropBackward<T>(Tensor<T> gradOutput, int[] inputShape, int top, int left)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        int batch = inputShape[0];
        int channels = inputShape[1];
        int inputHeight = inputShape[2];
        int inputWidth = inputShape[3];

        var gradOutputShape = gradOutput._shape;
        int cropHeight = gradOutputShape[2];
        int cropWidth = gradOutputShape[3];

        var gradOutputData = gradOutput.GetFlattenedData();
        var gradInputData = new T[batch * channels * inputHeight * inputWidth];

        for (int i = 0; i < gradInputData.Length; i++)
            gradInputData[i] = numOps.Zero;

        Parallel.For(0, batch * channels, bc =>
        {
            int b = bc / channels;
            int c = bc % channels;

            for (int oh = 0; oh < cropHeight; oh++)
            {
                int ih = top + oh;
                for (int ow = 0; ow < cropWidth; ow++)
                {
                    int iw = left + ow;
                    int gradOutputIdx = ((b * channels + c) * cropHeight + oh) * cropWidth + ow;
                    int gradInputIdx = ((b * channels + c) * inputHeight + ih) * inputWidth + iw;
                    gradInputData[gradInputIdx] = gradOutputData[gradOutputIdx];
                }
            }
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Pad<T>(Tensor<T> input, int padTop, int padBottom, int padLeft, int padRight, T padValue)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        var shape = input._shape;
        if (shape.Length < 2)
            throw new ArgumentException("Pad expects at least 2D tensor");

        int rank = shape.Length;
        int height = shape[rank - 2];
        int width = shape[rank - 1];

        int newHeight = height + padTop + padBottom;
        int newWidth = width + padLeft + padRight;

        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++)
            batchSize *= shape[i];

        var inputData = input.GetFlattenedData();
        var outputData = new T[batchSize * newHeight * newWidth];

        for (int i = 0; i < outputData.Length; i++)
            outputData[i] = padValue;

        Parallel.For(0, batchSize, b =>
        {
            for (int ih = 0; ih < height; ih++)
            {
                int oh = ih + padTop;
                for (int iw = 0; iw < width; iw++)
                {
                    int ow = iw + padLeft;
                    int inputIdx = b * height * width + ih * width + iw;
                    int outputIdx = b * newHeight * newWidth + oh * newWidth + ow;
                    outputData[outputIdx] = inputData[inputIdx];
                }
            }
        });

        var newShape = (int[])shape.Clone();
        newShape[rank - 2] = newHeight;
        newShape[rank - 1] = newWidth;

        var padResult = TensorAllocator.Rent<T>(newShape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("Pad", padResult, input, BackwardFunctions<T>.PadBackward, new object[] { padTop, padLeft });
        return padResult;
    }

    /// <inheritdoc/>
    public Tensor<T> PadBackward<T>(Tensor<T> gradOutput, int padTop, int padLeft, int[] inputShape)
    {
        int rank = inputShape.Length;
        int height = inputShape[rank - 2];
        int width = inputShape[rank - 1];

        int batchSize = 1;
        for (int i = 0; i < rank - 2; i++)
            batchSize *= inputShape[i];

        var gradOutputShape = gradOutput._shape;
        int paddedHeight = gradOutputShape[rank - 2];
        int paddedWidth = gradOutputShape[rank - 1];

        var gradOutputData = gradOutput.GetFlattenedData();
        var gradInputData = new T[batchSize * height * width];

        Parallel.For(0, batchSize, b =>
        {
            for (int ih = 0; ih < height; ih++)
            {
                int oh = ih + padTop;
                for (int iw = 0; iw < width; iw++)
                {
                    int ow = iw + padLeft;
                    int gradOutputIdx = b * paddedHeight * paddedWidth + oh * paddedWidth + ow;
                    int gradInputIdx = b * height * width + ih * width + iw;
                    gradInputData[gradInputIdx] = gradOutputData[gradOutputIdx];
                }
            }
        });

        return TensorAllocator.Rent<T>(inputShape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public Tensor<T> Concat<T>(IReadOnlyList<Tensor<T>> tensors, int axis)
    {
        if (tensors == null || tensors.Count == 0)
            throw new ArgumentException("At least one tensor required for concatenation");

        var firstShape = tensors[0]._shape;
        int rank = firstShape.Length;

        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {rank} dimensions");

        int totalAxisSize = 0;
        foreach (var tensor in tensors)
        {
            if (tensor._shape.Length != rank)
                throw new ArgumentException("All tensors must have the same number of dimensions");

            for (int i = 0; i < rank; i++)
            {
                if (i != axis && tensor._shape[i] != firstShape[i])
                    throw new ArgumentException($"All tensors must have the same shape except along axis {axis}");
            }

            totalAxisSize += tensor._shape[axis];
        }

        var outputShape = (int[])firstShape.Clone();
        outputShape[axis] = totalAxisSize;

        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        var outputData = new T[outputSize];

        var outputStrides = ComputeStrides(outputShape);

        int axisOffset = 0;
        foreach (var tensor in tensors)
        {
            var tensorData = tensor.GetDataArray();
            var tensorShape = tensor._shape;
            var tensorStrides = ComputeStrides(tensorShape);

            for (int i = 0; i < tensor.Length; i++)
            {
                var multiIndex = FlatToMultiIndex(i, tensorShape, tensorStrides);
                multiIndex[axis] += axisOffset;
                int outputIdx = MultiToFlatIndex(multiIndex, outputShape, outputStrides);
                outputData[outputIdx] = tensorData[i];
            }

            axisOffset += tensor._shape[axis];
        }

        var result = TensorAllocator.Rent<T>(outputShape, new Vector<T>(outputData));
        if (GradientTape<T>.Current is not null)
        {
            DifferentiableOps.RecordIfActive("Concat", result, tensors.ToArray(),
                BackwardFunctions<T>.ConcatenateBackward, new object[] { axis });
        }
        return result;
    }

    /// <inheritdoc/>
    public unsafe T TensorSumOfSquares<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(float))
        {
            T[] arr = tensor.GetFlattenedData();
            float[] fArr = Unsafe.As<T[], float[]>(ref arr);
            int length = tensor.Length;
            float result;

            fixed (float* ptr = fArr)
            {
                result = SumOfSquaresUnsafe(ptr, length);
            }
            return Unsafe.As<float, T>(ref result);
        }
#endif

        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Dot(tensor.AsSpan(), tensor.AsSpan());
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// SIMD sum of squares: sum(x[i] * x[i]) with 4x unrolled FMA accumulation.
    /// </summary>
    private static unsafe float SumOfSquaresUnsafe(float* data, int length)
    {
        int i = 0;
        float result = 0f;

        if (System.Runtime.Intrinsics.X86.Fma.IsSupported && length >= 32)
        {
            var vsum0 = System.Runtime.Intrinsics.Vector256<float>.Zero;
            var vsum1 = vsum0; var vsum2 = vsum0; var vsum3 = vsum0;
            int simdLen = length & ~31;
            for (; i < simdLen; i += 32)
            {
                var v0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(data + i);
                var v1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(data + i + 8);
                var v2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(data + i + 16);
                var v3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(data + i + 24);
                vsum0 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(v0, v0, vsum0);
                vsum1 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(v1, v1, vsum1);
                vsum2 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(v2, v2, vsum2);
                vsum3 = System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(v3, v3, vsum3);
            }
            vsum0 = System.Runtime.Intrinsics.X86.Avx.Add(
                System.Runtime.Intrinsics.X86.Avx.Add(vsum0, vsum1),
                System.Runtime.Intrinsics.X86.Avx.Add(vsum2, vsum3));
            // Horizontal sum
            var hi = System.Runtime.Intrinsics.X86.Avx.ExtractVector128(vsum0, 1);
            var lo = System.Runtime.Intrinsics.X86.Avx.ExtractVector128(vsum0, 0);
            var s128 = System.Runtime.Intrinsics.X86.Sse.Add(lo, hi);
            var shuf = System.Runtime.Intrinsics.X86.Sse.Shuffle(s128, s128, 0b_01_00_11_10);
            s128 = System.Runtime.Intrinsics.X86.Sse.Add(s128, shuf);
            shuf = System.Runtime.Intrinsics.X86.Sse.Shuffle(s128, s128, 0b_10_11_00_01);
            s128 = System.Runtime.Intrinsics.X86.Sse.Add(s128, shuf);
            float tmp;
            System.Runtime.Intrinsics.X86.Sse.StoreScalar(&tmp, s128);
            result = tmp;
        }

        for (; i < length; i++)
            result += data[i] * data[i];
        return result;
    }
#endif

    /// <inheritdoc/>
    public Tensor<TValue> TensorEmbeddingLookup<TValue, TIndex>(Tensor<TValue> embeddings, Tensor<TIndex> indices)
        where TIndex : unmanaged
    {
        if (embeddings == null) throw new ArgumentNullException(nameof(embeddings));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (embeddings.Rank != 2)
            throw new ArgumentException($"Embeddings must be a 2D tensor [vocab_size, embedding_dim]. Got rank {embeddings.Rank}.");

        int vocabSize = embeddings._shape[0];
        int embeddingDim = embeddings._shape[1];
        int numIndices = indices.Length;

        // Output shape is [*indices.shape, embedding_dim]
        var outputShape = new int[indices.Rank + 1];
        for (int i = 0; i < indices.Rank; i++)
        {
            outputShape[i] = indices._shape[i];
        }
        outputShape[indices.Rank] = embeddingDim;

        var result = new Tensor<TValue>(outputShape);
        var embData = embeddings.GetDataArray();
        var resultData = result.GetDataArray();
        var idxData = indices.GetDataArray();

        // For each index, copy the entire embedding row
        for (int i = 0; i < numIndices; i++)
        {
            int tokenIdx = Convert.ToInt32(idxData[i]);

            // Bounds check for index
            if (tokenIdx < 0 || tokenIdx >= vocabSize)
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {tokenIdx} at position {i} is out of bounds for embedding table with vocabulary size {vocabSize}.");

            int srcOffset = tokenIdx * embeddingDim;
            int dstOffset = i * embeddingDim;

            // Direct array copy of embedding row
            Array.Copy(embData, srcOffset, resultData, dstOffset, embeddingDim);
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<TValue> TensorEmbeddingLookupBackward<TValue, TIndex>(Tensor<TValue> gradOutput, Tensor<TIndex> indices, int vocabSize, int embeddingDim)
        where TIndex : unmanaged
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (vocabSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabSize), "Vocabulary size must be positive.");
        if (embeddingDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingDim), "Embedding dimension must be positive.");

        var numOps = MathHelper.GetNumericOperations<TValue>();
        var gradEmbeddings = new Tensor<TValue>(new[] { vocabSize, embeddingDim });
        var gradEmbData = gradEmbeddings.GetDataArray();

        var gradData = gradOutput.GetDataArray();
        var idxData = indices.GetDataArray();
        int numIndices = indices.Length;

        // Scatter-add: for each index, accumulate gradients to the embedding row
        for (int i = 0; i < numIndices; i++)
        {
            int tokenIdx = Convert.ToInt32(idxData[i]);

            // Bounds check for index
            if (tokenIdx < 0 || tokenIdx >= vocabSize)
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {tokenIdx} at position {i} is out of bounds for vocabulary size {vocabSize}.");

            int srcOffset = i * embeddingDim;
            int dstOffset = tokenIdx * embeddingDim;

            // Accumulate gradient for this embedding row - direct array access
            for (int d = 0; d < embeddingDim; d++)
            {
                gradEmbData[dstOffset + d] = numOps.Add(gradEmbData[dstOffset + d], gradData[srcOffset + d]);
            }
        }

        return gradEmbeddings;
    }

    /// <inheritdoc/>
    public Tensor<T> RBFKernel<T>(Tensor<T> input, Tensor<T> centers, Tensor<T> epsilons)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (centers == null) throw new ArgumentNullException(nameof(centers));
        if (epsilons == null) throw new ArgumentNullException(nameof(epsilons));

        if (input.Rank != 2)
            throw new ArgumentException($"input must be 2D [batch, features], got rank {input.Rank}", nameof(input));
        if (centers.Rank != 2)
            throw new ArgumentException($"centers must be 2D [numCenters, features], got rank {centers.Rank}", nameof(centers));
        if (epsilons.Rank != 1)
            throw new ArgumentException($"epsilons must be 1D [numCenters], got rank {epsilons.Rank}", nameof(epsilons));

        var numOps = MathHelper.GetNumericOperations<T>();
        int batchSize = input._shape[0];
        int features = input._shape[1];
        int numCenters = centers._shape[0];

        if (centers._shape[1] != features)
            throw new ArgumentException($"centers features dimension ({centers._shape[1]}) must match input features ({features})", nameof(centers));
        if (epsilons._shape[0] != numCenters)
            throw new ArgumentException($"epsilons length ({epsilons._shape[0]}) must match number of centers ({numCenters})", nameof(epsilons));

        var output = TensorAllocator.Rent<T>([batchSize, numCenters]);
        var inputData = input.GetFlattenedData();
        var centersData = centers.GetFlattenedData();
        var epsilonsData = epsilons.GetFlattenedData();

        // Compute exp(-epsilon * ||x - center||Ã‚Â²) for each (sample, center) pair
        for (int b = 0; b < batchSize; b++)
        {
            int inputOffset = b * features;
            for (int c = 0; c < numCenters; c++)
            {
                int centerOffset = c * features;
                T distSquared = numOps.Zero;

                // Compute ||x - center||Ã‚Â²
                for (int f = 0; f < features; f++)
                {
                    T diff = numOps.Subtract(inputData[inputOffset + f], centersData[centerOffset + f]);
                    distSquared = numOps.Add(distSquared, numOps.Multiply(diff, diff));
                }

                // Compute exp(-epsilon * distSquared)
                T negEpsDist = numOps.Negate(numOps.Multiply(epsilonsData[c], distSquared));
                output[b, c] = numOps.Exp(negEpsDist);
            }
        }

        DifferentiableOps.RecordIfActive("RBFKernel", output, new[] { input, centers }, BackwardFunctions<T>.RBFKernelBackward, new object[] { numOps.ToDouble(epsilonsData[0]) });
        return output;
    }

    /// <inheritdoc/>
    public (Tensor<T> gradInput, Tensor<T> gradCenters, Tensor<T> gradEpsilons) RBFKernelBackward<T>(
        Tensor<T> gradOutput, Tensor<T> input, Tensor<T> centers, Tensor<T> epsilons, Tensor<T> output)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (centers == null) throw new ArgumentNullException(nameof(centers));
        if (epsilons == null) throw new ArgumentNullException(nameof(epsilons));
        if (output == null) throw new ArgumentNullException(nameof(output));

        if (input.Rank != 2)
            throw new ArgumentException($"input must be 2D [batch, features], got rank {input.Rank}", nameof(input));
        if (centers.Rank != 2)
            throw new ArgumentException($"centers must be 2D [numCenters, features], got rank {centers.Rank}", nameof(centers));
        if (epsilons.Rank != 1)
            throw new ArgumentException($"epsilons must be 1D [numCenters], got rank {epsilons.Rank}", nameof(epsilons));
        if (gradOutput.Rank != 2)
            throw new ArgumentException($"gradOutput must be 2D [batch, numCenters], got rank {gradOutput.Rank}", nameof(gradOutput));
        if (output.Rank != 2)
            throw new ArgumentException($"output must be 2D [batch, numCenters], got rank {output.Rank}", nameof(output));

        var numOps = MathHelper.GetNumericOperations<T>();
        int batchSize = input._shape[0];
        int features = input._shape[1];
        int numCenters = centers._shape[0];

        if (centers._shape[1] != features)
            throw new ArgumentException($"centers features dimension ({centers._shape[1]}) must match input features ({features})", nameof(centers));
        if (epsilons._shape[0] != numCenters)
            throw new ArgumentException($"epsilons length ({epsilons._shape[0]}) must match number of centers ({numCenters})", nameof(epsilons));
        if (gradOutput._shape[0] != batchSize || gradOutput._shape[1] != numCenters)
            throw new ArgumentException($"gradOutput shape [{gradOutput._shape[0]}, {gradOutput._shape[1]}] must be [{batchSize}, {numCenters}]", nameof(gradOutput));
        if (output._shape[0] != batchSize || output._shape[1] != numCenters)
            throw new ArgumentException($"output shape [{output._shape[0]}, {output._shape[1]}] must be [{batchSize}, {numCenters}]", nameof(output));

        var gradInput = TensorAllocator.RentUninitialized<T>(input._shape);
        var gradCenters = TensorAllocator.RentUninitialized<T>(centers._shape);
        var gradEpsilons = TensorAllocator.RentUninitialized<T>(epsilons._shape);

        var inputData = input.GetFlattenedData();
        var centersData = centers.GetFlattenedData();
        var epsilonsData = epsilons.GetFlattenedData();
        var outputData = output.GetDataArray();
        var gradOutputData = gradOutput.GetDataArray();
        var gradInputData = gradInput.GetDataArray();
        var gradCentersData = gradCenters.GetDataArray();

        // For RBF: K = exp(-epsilon * ||x - c||Ã‚Â²)
        // dK/dx = K * (-epsilon) * 2 * (x - c) = -2 * epsilon * K * (x - c)
        // dK/dc = K * (-epsilon) * (-2) * (x - c) = 2 * epsilon * K * (x - c) = -dK/dx
        // dK/depsilon = K * (-||x - c||Ã‚Â²)

        for (int b = 0; b < batchSize; b++)
        {
            int inputOffset = b * features;
            for (int c = 0; c < numCenters; c++)
            {
                int centerOffset = c * features;
                int outIdx = b * numCenters + c;
                T K = outputData[outIdx];
                T eps = epsilonsData[c];
                T dL_dK = gradOutputData[outIdx];

                // Compute ||x - c||Ã‚Â² for epsilon gradient
                T distSquared = numOps.Zero;
                for (int f = 0; f < features; f++)
                {
                    T diff = numOps.Subtract(inputData[inputOffset + f], centersData[centerOffset + f]);
                    distSquared = numOps.Add(distSquared, numOps.Multiply(diff, diff));
                }

                // dL/depsilon += dL/dK * dK/depsilon = dL/dK * K * (-distSquared)
                T gradEps = numOps.Multiply(dL_dK, numOps.Multiply(K, numOps.Negate(distSquared)));
                gradEpsilons[c] = numOps.Add(gradEpsilons[c], gradEps);

                // Common factor: -2 * epsilon * K * dL/dK
                T commonFactor = numOps.Multiply(
                    numOps.Multiply(numOps.FromDouble(-2.0), eps),
                    numOps.Multiply(K, dL_dK));

                for (int f = 0; f < features; f++)
                {
                    T diff = numOps.Subtract(inputData[inputOffset + f], centersData[centerOffset + f]);
                    T grad = numOps.Multiply(commonFactor, diff);

                    // dL/dx = commonFactor * (x - c)
                    int inputIdx = b * features + f;
                    gradInputData[inputIdx] = numOps.Add(gradInputData[inputIdx], grad);

                    // dL/dc = -dL/dx = -commonFactor * (x - c)
                    int centerIdx = c * features + f;
                    gradCentersData[centerIdx] = numOps.Subtract(gradCentersData[centerIdx], grad);
                }
            }
        }

        return (gradInput, gradCenters, gradEpsilons);
    }

    #endregion

    #region Tensor Shape Operations

    /// <inheritdoc/>
    public Tensor<T> TensorRepeatElements<T>(Tensor<T> tensor, int repeats, int axis = 0)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (repeats < 1) throw new ArgumentOutOfRangeException(nameof(repeats), "Repeats must be at least 1");
        if (axis < 0 || axis >= tensor._shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} out of range for tensor with {tensor._shape.Length} dimensions");

        // Calculate output shape
        var outputShape = new int[tensor._shape.Length];
        Array.Copy(tensor._shape, outputShape, tensor._shape.Length);
        outputShape[axis] *= repeats;

        var result = TensorAllocator.RentUninitialized<T>(outputShape);

        // Calculate strides for the tensor
        int outerSize = 1;
        for (int i = 0; i < axis; i++)
            outerSize *= tensor._shape[i];

        int axisSize = tensor._shape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < tensor._shape.Length; i++)
            innerSize *= tensor._shape[i];

        var tensorData = tensor.GetFlattenedData();
        var resultData = result.GetDataArray();

        // Perform the repeat operation
        Parallel.For(0, outerSize, outer =>
        {
            for (int a = 0; a < axisSize; a++)
            {
                int srcBase = (outer * axisSize + a) * innerSize;
                int dstBase = (outer * axisSize * repeats + a * repeats) * innerSize;

                for (int r = 0; r < repeats; r++)
                {
                    int dstOffset = dstBase + r * innerSize;
                    Array.Copy(tensorData, srcBase, resultData, dstOffset, innerSize);
                }
            }
        });

        DifferentiableOps.RecordUnary("TensorRepeatElements", result, tensor, BackwardFunctions<T>.RepeatElementsBackward, new object[] { repeats, axis });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTile<T>(Tensor<T> tensor, int[] multiples)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (multiples == null) throw new ArgumentNullException(nameof(multiples));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (multiples.Length != tensor._shape.Length)
            throw new ArgumentException($"Multiples length ({multiples.Length}) must match tensor dimensions ({tensor._shape.Length})");

        // Calculate output shape
        var outputShape = new int[tensor._shape.Length];
        for (int i = 0; i < tensor._shape.Length; i++)
        {
            if (multiples[i] < 1)
                throw new ArgumentOutOfRangeException(nameof(multiples), $"Multiple at index {i} must be at least 1");
            outputShape[i] = tensor._shape[i] * multiples[i];
        }

        var result = TensorAllocator.RentUninitialized<T>(outputShape);
        int totalElements = result._shape.Aggregate(1, (a, b) => a * b);
        var tensorData = tensor.GetFlattenedData();
        var resultData = result.GetDataArray();

        // For each output element, find the corresponding input element
        Parallel.For(0, totalElements, flatIdx =>
        {
            // Convert flat index to multi-dimensional indices
            var outputIndices = new int[outputShape.Length];
            int remaining = flatIdx;
            for (int d = outputShape.Length - 1; d >= 0; d--)
            {
                outputIndices[d] = remaining % outputShape[d];
                remaining /= outputShape[d];
            }

            // Map to input indices (modulo original size)
            int inputFlat = 0;
            int stride = 1;
            for (int d = tensor._shape.Length - 1; d >= 0; d--)
            {
                inputFlat += (outputIndices[d] % tensor._shape[d]) * stride;
                stride *= tensor._shape[d];
            }

            resultData[flatIdx] = tensorData[inputFlat];
        });

        DifferentiableOps.RecordUnary("Tile", result, tensor,
            BackwardFunctions<T>.TileBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSlice<T>(Tensor<T> tensor, int[] start, int[] length)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (start == null) throw new ArgumentNullException(nameof(start));
        if (length == null) throw new ArgumentNullException(nameof(length));
        if (start.Length != tensor._shape.Length)
            throw new ArgumentException($"Start length ({start.Length}) must match tensor dimensions ({tensor._shape.Length})");
        if (length.Length != tensor._shape.Length)
            throw new ArgumentException($"Length length ({length.Length}) must match tensor dimensions ({tensor._shape.Length})");

        // Validate bounds
        for (int i = 0; i < tensor._shape.Length; i++)
        {
            if (start[i] < 0 || start[i] >= tensor._shape[i])
                throw new ArgumentOutOfRangeException(nameof(start), $"Start index {start[i]} out of range for axis {i} with size {tensor._shape[i]}");
            if (length[i] < 1 || start[i] + length[i] > tensor._shape[i])
                throw new ArgumentOutOfRangeException(nameof(length), $"Slice length {length[i]} starting at {start[i]} exceeds axis {i} size {tensor._shape[i]}");
        }

        var result = TensorAllocator.RentUninitialized<T>(length);
        int totalElements = length.Aggregate(1, (a, b) => a * b);
        var tensorData = tensor.GetFlattenedData();
        var resultData = result.GetDataArray();

        // For each output element, find the corresponding input element
        Parallel.For(0, totalElements, flatIdx =>
        {
            // Convert flat index to output indices and map to input flat index
            int remaining = flatIdx;
            int inputFlat = 0;
            int stride = 1;
            for (int d = tensor._shape.Length - 1; d >= 0; d--)
            {
                int outputIdx = remaining % length[d];
                remaining /= length[d];
                inputFlat += (start[d] + outputIdx) * stride;
                stride *= tensor._shape[d];
            }

            resultData[flatIdx] = tensorData[inputFlat];
        });

        DifferentiableOps.RecordUnary("TensorSlice", result, tensor, BackwardFunctions<T>.SliceBackward, new object[] { start });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSetSlice<T>(Tensor<T> destination, Tensor<T> source, int[] start)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (start == null) throw new ArgumentNullException(nameof(start));
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (start.Length != destination._shape.Length)
            throw new ArgumentException($"Start length ({start.Length}) must match destination dimensions ({destination._shape.Length})");
        if (source._shape.Length != destination._shape.Length)
            throw new ArgumentException($"Source dimensions ({source._shape.Length}) must match destination dimensions ({destination._shape.Length})");

        // Validate bounds
        for (int i = 0; i < destination._shape.Length; i++)
        {
            if (start[i] < 0 || start[i] + source._shape[i] > destination._shape[i])
                throw new ArgumentOutOfRangeException(nameof(start), $"Slice starting at {start[i]} with size {source._shape[i]} exceeds destination axis {i} size {destination._shape[i]}");
        }

        // Create a copy of destination to avoid modifying the original
        var result = TensorAllocator.RentUninitialized<T>(destination._shape);
        var destData = destination.GetDataArray();
        var resultData = result.GetDataArray();
        Array.Copy(destData, resultData, destData.Length);

        int sourceTotal = source._shape.Aggregate(1, (a, b) => a * b);
        var sourceData = source.GetFlattenedData();

        // Set the slice values
        Parallel.For(0, sourceTotal, flatIdx =>
        {
            // Convert flat index to source indices and map to dest flat index
            int remaining = flatIdx;
            int destFlat = 0;
            int stride = 1;
            for (int d = destination._shape.Length - 1; d >= 0; d--)
            {
                int srcIdx = remaining % source._shape[d];
                remaining /= source._shape[d];
                destFlat += (start[d] + srcIdx) * stride;
                stride *= destination._shape[d];
            }

            resultData[destFlat] = sourceData[flatIdx];
        });

        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorWhere<T>(Tensor<T> condition, Tensor<T> x, Tensor<T> y)
    {
        if (condition == null) throw new ArgumentNullException(nameof(condition));
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));
        if (!y.IsContiguous) y = y.Contiguous();
        if (!x.IsContiguous) x = x.Contiguous();
        if (!condition.IsContiguous) condition = condition.Contiguous();

        // All tensors must have the same shape (or be broadcastable, but we'll require same shape for simplicity)
        if (!condition._shape.SequenceEqual(x._shape) || !condition._shape.SequenceEqual(y._shape))
            throw new ArgumentException("All tensors must have the same shape");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(condition._shape);
        var condSpan = condition.AsSpan();
        var xSpan = x.AsSpan();
        var ySpan = y.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < condSpan.Length; i++)
        {
            bool isTrue = !numOps.Equals(condSpan[i], numOps.Zero);
            dest[i] = isTrue ? xSpan[i] : ySpan[i];
        }

        return result;
    }

    #endregion

    #region Loop Elimination Operations

    /// <inheritdoc/>
    public void TensorCopy<T>(Tensor<T> source, Tensor<T> destination)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (!source.IsContiguous) source = source.Contiguous();
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");
        if (source.Length != destination.Length)
            throw new ArgumentException($"Tensor lengths must match. Got {source.Length} and {destination.Length}");

        var sourceArray = source.GetFlattenedData();
        var destArray = destination.GetDataArray();
        Array.Copy(sourceArray, destArray, sourceArray.Length);
    }

    /// <inheritdoc/>
    public void TensorFill<T>(Tensor<T> tensor, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        tensor.Fill(value);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorOuterProduct<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        // Flatten both tensors to 1D
        int n = a.Length;
        int m = b.Length;
        var result = TensorAllocator.Rent<T>([n, m]);
        var numOps = MathHelper.GetNumericOperations<T>();

        var aData = a.GetFlattenedData();
        var bData = b.GetFlattenedData();
        var rData = result.GetDataArray();

        Parallel.For(0, n, i =>
        {
            T ai = aData[i];
            int rowOffset = i * m;
            for (int j = 0; j < m; j++)
            {
                rData[rowOffset + j] = numOps.Multiply(ai, bData[j]);
            }
        });

        DifferentiableOps.RecordBinary("TensorOuterProduct", result, a, b, BackwardFunctions<T>.OuterProductBackward);
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorBatchOuterProduct<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a._shape.Length != 2 || b._shape.Length != 2)
            throw new ArgumentException("Both tensors must be 2D [batch, features]");
        if (a._shape[0] != b._shape[0])
            throw new ArgumentException("Batch sizes must match");

        int batch = a._shape[0];
        int n = a._shape[1];
        int m = b._shape[1];
        var result = TensorAllocator.Rent<T>([batch, n, m]);
        var numOps = MathHelper.GetNumericOperations<T>();

        Parallel.For(0, batch, bIdx =>
        {
            for (int i = 0; i < n; i++)
            {
                T ai = a[bIdx, i];
                for (int j = 0; j < m; j++)
                {
                    result[bIdx, i, j] = numOps.Multiply(ai, b[bIdx, j]);
                }
            }
        });

        DifferentiableOps.RecordBinary("TensorBatchOuterProduct", result, a, b, BackwardFunctions<T>.OuterProductBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPermute<T>(Tensor<T> tensor, int[] axes)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (axes == null) throw new ArgumentNullException(nameof(axes));
        if (axes.Length != tensor._shape.Length)
            throw new ArgumentException("Axes length must match tensor rank");

        // Use tensor's built-in Transpose method
        var result = tensor.Transpose(axes);
        DifferentiableOps.RecordUnary("TensorPermute", result, tensor, BackwardFunctions<T>.PermuteBackward, new object[] { axes });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorExpandDims<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Normalize negative axis
        int rank = tensor._shape.Length;
        if (axis < 0) axis = rank + 1 + axis;
        if (axis < 0 || axis > rank)
            throw new ArgumentOutOfRangeException(nameof(axis), "Axis out of range");

        // Build new shape with 1 inserted at axis
        var newShape = new int[rank + 1];
        for (int i = 0; i < axis; i++)
            newShape[i] = tensor._shape[i];
        newShape[axis] = 1;
        for (int i = axis; i < rank; i++)
            newShape[i + 1] = tensor._shape[i];

        var result = tensor.Reshape(newShape);
        DifferentiableOps.RecordUnary("TensorExpandDims", result, tensor, BackwardFunctions<T>.ExpandDimsBackward, new object[] { axis });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSqueeze<T>(Tensor<T> tensor, int axis = -1)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var shape = tensor._shape.ToList();

        if (axis == -1)
        {
            // Remove all singleton dimensions
            shape = shape.Where(s => s != 1).ToList();
            if (shape.Count == 0) shape.Add(1); // Keep at least one dimension
        }
        else
        {
            // Normalize axis
            if (axis < 0) axis = shape.Count + axis;
            if (axis < 0 || axis >= shape.Count)
                throw new ArgumentOutOfRangeException(nameof(axis));
            if (shape[axis] != 1)
                throw new ArgumentException($"Cannot squeeze axis {axis} with size {shape[axis]} (must be 1)");
            shape.RemoveAt(axis);
            if (shape.Count == 0) shape.Add(1);
        }

        var squeezeResult = tensor.Reshape(shape.ToArray());
        DifferentiableOps.RecordUnary("TensorSqueeze", squeezeResult, tensor, BackwardFunctions<T>.SqueezeBackward, new object[] { axis });
        return squeezeResult;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorScatterAdd<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> updates, int axis = 0)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (updates == null) throw new ArgumentNullException(nameof(updates));
        if (!destination.IsContiguous) throw new InvalidOperationException("Output tensor must be contiguous.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(destination._shape);
        TensorCopy(destination, result);

        // Simple 1D scatter-add for now (most common use case: embeddings)
        if (axis == 0 && destination._shape.Length == 2)
        {
            int embeddingDim = destination._shape[1];
            var indicesData = indices.GetFlattenedData();
            var resultData = result.GetDataArray();
            var updatesData = updates.GetFlattenedData();
            for (int i = 0; i < indices.Length; i++)
            {
                int idx = indicesData[i];
                if (idx >= 0 && idx < destination._shape[0])
                {
                    int resultOffset = idx * embeddingDim;
                    int updateOffset = i * embeddingDim;
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        resultData[resultOffset + j] = numOps.Add(resultData[resultOffset + j], updatesData[updateOffset + j]);
                    }
                }
            }
        }
        else
        {
            throw new NotImplementedException("Scatter-add only implemented for axis=0 with 2D destination");
        }

        DifferentiableOps.RecordIfActive("TensorScatterAdd", result, new[] { destination, updates },
            BackwardFunctions<T>.ScatterAddBackward, new object[] { indices, axis });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorGather<T>(Tensor<T> source, Tensor<int> indices, int axis = 0)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (!indices.IsContiguous) indices = indices.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        // Simple 1D gather for embedding lookups
        if (axis == 0 && source._shape.Length == 2)
        {
            int embeddingDim = source._shape[1];
            int numIndices = indices.Length;
            var result = TensorAllocator.Rent<T>([numIndices, embeddingDim]);

            var indicesData = indices.GetFlattenedData();
            var sourceData = source.GetFlattenedData();
            var resultData = result.GetDataArray();
            Parallel.For(0, numIndices, i =>
            {
                int idx = indicesData[i];
                if (idx >= 0 && idx < source._shape[0])
                {
                    Array.Copy(sourceData, idx * embeddingDim, resultData, i * embeddingDim, embeddingDim);
                }
            });

            DifferentiableOps.RecordUnary("TensorGather", result, source, BackwardFunctions<T>.GatherBackward, new object[] { indices, axis });
            return result;
        }
        else
        {
            throw new NotImplementedException("Gather only implemented for axis=0 with 2D source");
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorCumSum<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);

        // Normalize axis
        if (axis < 0) axis = tensor._shape.Length + axis;

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= tensor._shape[i];

        int axisSize = tensor._shape[axis];

        int innerSize = 1;
        for (int i = axis + 1; i < tensor._shape.Length; i++) innerSize *= tensor._shape[i];

        var tensorData = tensor.GetFlattenedData();
        var resultData = result.GetDataArray();

        Parallel.For(0, outerSize * innerSize, flatIdx =>
        {
            int outer = flatIdx / innerSize;
            int inner = flatIdx % innerSize;

            T cumSum = numOps.Zero;
            for (int a = 0; a < axisSize; a++)
            {
                int srcIdx = (outer * axisSize + a) * innerSize + inner;
                cumSum = numOps.Add(cumSum, tensorData[srcIdx]);
                resultData[srcIdx] = cumSum;
            }
        });

        DifferentiableOps.RecordUnary("TensorCumSum", result, tensor, BackwardFunctions<T>.CumSumBackward, new object[] { axis });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogSumExp<T>(Tensor<T> tensor, int axis, bool keepDims = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = tensor._shape.Length + axis;

        // Compute max along axis for numerical stability
        var maxVals = ReduceMax(tensor, new[] { axis }, keepDims: true, out _);

        // Compute exp(x - max)
        var shifted = TensorBroadcastSubtract(tensor, maxVals);
        var expShifted = TensorExp(shifted);

        // Sum along axis
        var sumExp = ReduceSum(expShifted, new[] { axis }, keepDims: keepDims);

        // log(sum) + max
        var logSum = TensorLog(sumExp);

        if (keepDims)
        {
            return TensorAdd(logSum, maxVals);
        }
        else
        {
            var maxSqueezed = TensorSqueeze(maxVals, axis);
            return TensorAdd(logSum, maxSqueezed);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorRandomUniform<T>(int[] shape)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        return Tensor<T>.CreateRandom(shape);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorRandomNormal<T>(int[] shape, T mean, T stddev)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));

        var numOps = MathHelper.GetNumericOperations<T>();
        var random = RandomHelper.ThreadSafeRandom;
        var result = TensorAllocator.RentUninitialized<T>(shape);
        int totalElements = shape.Aggregate(1, (a, b) => a * b);

        var resultData = result.GetDataArray();
        double meanD = numOps.ToDouble(mean);
        double stdD = numOps.ToDouble(stddev);

        // Box-Muller transform for normal distribution
        for (int i = 0; i < totalElements; i += 2)
        {
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();

            // Guard against log(0) which produces -Infinity and then NaN
            u1 = Math.Max(u1, double.Epsilon);

            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            double z0 = mag * Math.Cos(2.0 * Math.PI * u2);
            double z1 = mag * Math.Sin(2.0 * Math.PI * u2);

            resultData[i] = numOps.FromDouble(z0 * stdD + meanD);
            if (i + 1 < totalElements)
                resultData[i + 1] = numOps.FromDouble(z1 * stdD + meanD);
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorRandomUniformRange<T>(int[] shape, T min, T max, int? seed = null)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));

        var numOps = MathHelper.GetNumericOperations<T>();
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.ThreadSafeRandom;
        var result = TensorAllocator.RentUninitialized<T>(shape);
        int totalElements = shape.Aggregate(1, (a, b) => a * b);

        double minD = numOps.ToDouble(min);
        double maxD = numOps.ToDouble(max);
        double range = maxD - minD;

        var resultData = result.GetDataArray();

        if (totalElements > 10000)
        {
            // For seeded random with parallelism, we need thread-local randoms
            if (seed.HasValue)
            {
                var baseRandom = RandomHelper.CreateSeededRandom(seed.Value);
                var seeds = new int[CpuParallelSettings.MaxDegreeOfParallelism];
                for (int i = 0; i < seeds.Length; i++)
                    seeds[i] = baseRandom.Next();

                Parallel.For(0, totalElements, () => RandomHelper.CreateSeededRandom(seeds[Thread.CurrentThread.ManagedThreadId % seeds.Length]),
                    (i, state, localRandom) =>
                    {
                        resultData[i] = numOps.FromDouble(localRandom.NextDouble() * range + minD);
                        return localRandom;
                    },
                    _ => { });
            }
            else
            {
                Parallel.For(0, totalElements, i =>
                {
                    resultData[i] = numOps.FromDouble(RandomHelper.ThreadSafeRandom.NextDouble() * range + minD);
                });
            }
        }
        else
        {
            for (int i = 0; i < totalElements; i++)
            {
                resultData[i] = numOps.FromDouble(random.NextDouble() * range + minD);
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorDropoutMask<T>(int[] shape, T dropoutRate, T scale, int? seed = null)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));

        var numOps = MathHelper.GetNumericOperations<T>();
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.ThreadSafeRandom;
        var result = TensorAllocator.RentUninitialized<T>(shape);
        int totalElements = shape.Aggregate(1, (a, b) => a * b);

        double dropoutRateD = numOps.ToDouble(dropoutRate);
        T zero = numOps.Zero;

        var resultData = result.GetDataArray();

        if (totalElements > 10000)
        {
            // For seeded random with parallelism, we need thread-local randoms
            if (seed.HasValue)
            {
                var baseRandom = RandomHelper.CreateSeededRandom(seed.Value);
                var seeds = new int[CpuParallelSettings.MaxDegreeOfParallelism];
                for (int i = 0; i < seeds.Length; i++)
                    seeds[i] = baseRandom.Next();

                Parallel.For(0, totalElements, () => RandomHelper.CreateSeededRandom(seeds[Thread.CurrentThread.ManagedThreadId % seeds.Length]),
                    (i, state, localRandom) =>
                    {
                        resultData[i] = localRandom.NextDouble() < dropoutRateD ? zero : scale;
                        return localRandom;
                    },
                    _ => { });
            }
            else
            {
                Parallel.For(0, totalElements, i =>
                {
                    resultData[i] = RandomHelper.ThreadSafeRandom.NextDouble() < dropoutRateD ? zero : scale;
                });
            }
        }
        else
        {
            for (int i = 0; i < totalElements; i++)
            {
                resultData[i] = random.NextDouble() < dropoutRateD ? zero : scale;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> ScalarMinusTensor<T>(T scalar, Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);

        // scalar - tensor = -(tensor - scalar) = negate(tensor) + scalar
        numOps.Negate(tensor.AsSpan(), result.AsWritableSpan());
        numOps.AddScalar(result.AsSpan(), scalar, result.AsWritableSpan());

        // d(scalar - x)/dx = -1, so gradient is negated
        DifferentiableOps.RecordUnary("ScalarMinusTensor", result, tensor, BackwardFunctions<T>.NegateBackward);

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorEye<T>(int size)
    {
        if (size <= 0) throw new ArgumentException("Size must be positive", nameof(size));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.Rent<T>([size, size]);
        result.Fill(numOps.Zero);

        for (int i = 0; i < size; i++)
        {
            result[i, i] = numOps.One;
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorDiag<T>(Tensor<T> diagonal)
    {
        if (diagonal == null) throw new ArgumentNullException(nameof(diagonal));

        var numOps = MathHelper.GetNumericOperations<T>();
        int n = diagonal.Length;
        var result = TensorAllocator.Rent<T>([n, n]);
        result.Fill(numOps.Zero);

        var diagData = diagonal.GetFlattenedData();
        var resultData = result.GetDataArray();

        for (int i = 0; i < n; i++)
        {
            resultData[i * n + i] = diagData[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorDiagonal<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor._shape.Length != 2)
            throw new ArgumentException("Tensor must be 2D");

        int n = Math.Min(tensor._shape[0], tensor._shape[1]);
        var result = TensorAllocator.RentUninitialized<T>([n]);

        for (int i = 0; i < n; i++)
        {
            result[i] = tensor[i, i];
        }

        DifferentiableOps.RecordUnary("TensorDiagonal", result, tensor, BackwardFunctions<T>.DiagonalBackward);
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorEinsum<T>(string subscripts, params Tensor<T>[] tensors)
    {
        if (subscripts == null) throw new ArgumentNullException(nameof(subscripts));
        if (tensors == null || tensors.Length == 0) throw new ArgumentNullException(nameof(tensors));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Parse einsum notation
        var parts = subscripts.Replace(" ", "").Split(new[] { "->" }, StringSplitOptions.None);
        if (parts.Length != 2)
            throw new ArgumentException("Einsum subscripts must contain '->'");

        var inputSubscripts = parts[0].Split(new[] { ',' }, StringSplitOptions.None);
        var outputSubscripts = parts[1];

        if (inputSubscripts.Length != tensors.Length)
            throw new ArgumentException($"Expected {inputSubscripts.Length} tensors but got {tensors.Length}");

        // Handle common cases directly for efficiency
        if (tensors.Length == 2)
        {
            // Batched matrix multiplication: bij,bjk->bik
            if (subscripts == "bij,bjk->bik")
            {
                return BatchMatMul(tensors[0], tensors[1]);
            }
            // Matrix multiplication: ij,jk->ik
            if (subscripts == "ij,jk->ik")
            {
                return TensorMatMul(tensors[0], tensors[1]);
            }
            // Batched outer product: bi,bj->bij
            if (subscripts == "bi,bj->bij")
            {
                return TensorBatchOuterProduct(tensors[0], tensors[1]);
            }
            // Outer product: i,j->ij
            if (subscripts == "i,j->ij")
            {
                return TensorOuterProduct(tensors[0], tensors[1]);
            }
            // Batched dot: bi,bi->b
            if (subscripts == "bi,bi->b")
            {
                int batch = tensors[0]._shape[0];
                int n = tensors[0]._shape[1];
                var result = TensorAllocator.RentUninitialized<T>([batch]);
                Parallel.For(0, batch, b =>
                {
                    T sum = numOps.Zero;
                    for (int i = 0; i < n; i++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(tensors[0][b, i], tensors[1][b, i]));
                    }
                    result[b] = sum;
                });
                return result;
            }
        }

        // General einsum implementation is complex - throw for unsupported patterns
        throw new NotImplementedException($"Einsum pattern '{subscripts}' not implemented. Use specific tensor operations instead.");
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAddScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        if (tensor.IsContiguous) { numOps.AddScalar(tensor.AsSpan(), scalar, result.AsWritableSpan()); }
        else { var src = tensor._storage.GetDataArray(); var dst = result.GetDataArray(); for (int i = 0; i < tensor.Length; i++) dst[i] = numOps.Add(src[tensor.LogicalToStorageIndex(i)], scalar); }

        DifferentiableOps.RecordUnary("TensorAddScalar", result, tensor, BackwardFunctions<T>.AddScalarBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSubtractScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        if (tensor.IsContiguous) { numOps.SubtractScalar(tensor.AsSpan(), scalar, result.AsWritableSpan()); }
        else { var src = tensor._storage.GetDataArray(); var dst = result.GetDataArray(); for (int i = 0; i < tensor.Length; i++) dst[i] = numOps.Subtract(src[tensor.LogicalToStorageIndex(i)], scalar); }

        DifferentiableOps.RecordUnary("TensorSubtractScalar", result, tensor, BackwardFunctions<T>.SubtractScalarBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDivideScalar<T>(Tensor<T> tensor, T scalar)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        if (tensor.IsContiguous) { numOps.DivideScalar(tensor.AsSpan(), scalar, result.AsWritableSpan()); }
        else { var src = tensor._storage.GetDataArray(); var dst = result.GetDataArray(); for (int i = 0; i < tensor.Length; i++) dst[i] = numOps.Divide(src[tensor.LogicalToStorageIndex(i)], scalar); }

        if (scalar is not null)
            DifferentiableOps.RecordUnary("TensorDivideScalar", result, tensor, BackwardFunctions<T>.DivideScalarBackward, new object[] { scalar });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TanhDerivative<T>(Tensor<T> tanhOutput)
    {
        if (tanhOutput == null) throw new ArgumentNullException(nameof(tanhOutput));
        if (!tanhOutput.IsContiguous) tanhOutput = tanhOutput.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tanhOutput._shape);

        // d/dx tanh(x) = 1 - tanh(x)^2: compute y*y then subtract from 1
        // Use span-based: result = 1 - y*y
        var y = tanhOutput.AsSpan();
        var dest = result.AsWritableSpan();
        numOps.Multiply(y, y, dest);       // dest = y^2
        numOps.Negate(dest, dest);          // dest = -y^2
        numOps.AddScalar(dest, numOps.One, dest); // dest = 1 - y^2

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> SigmoidDerivative<T>(Tensor<T> sigmoidOutput)
    {
        if (sigmoidOutput == null) throw new ArgumentNullException(nameof(sigmoidOutput));
        if (!sigmoidOutput.IsContiguous) sigmoidOutput = sigmoidOutput.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(sigmoidOutput._shape);
        var temp = TensorAllocator.RentUninitialized<T>(sigmoidOutput._shape);

        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        var y = sigmoidOutput.AsSpan();
        var oneMinusY = temp.AsWritableSpan();
        numOps.Negate(y, oneMinusY);                     // oneMinusY = -y
        numOps.AddScalar(oneMinusY, numOps.One, oneMinusY); // oneMinusY = 1 - y
        numOps.Multiply(y, oneMinusY, result.AsWritableSpan()); // result = y * (1-y)

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> ReLUDerivative<T>(Tensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(input._shape);
        int totalElements = input.Length;

        // ReLU derivative: 1 if x > 0, else 0
        var src = input.AsSpan();
        var dest = result.AsWritableSpan();
        for (int i = 0; i < totalElements; i++)
        {
            dest[i] = numOps.ToDouble(src[i]) > 0 ? numOps.One : numOps.Zero;
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorTriangularMask<T>(int size, bool upper = false, int diagonal = 0)
    {
        if (size <= 0) throw new ArgumentException("Size must be positive", nameof(size));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.Rent<T>([size, size]);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                bool inMask = upper ? (j >= i + diagonal) : (j <= i + diagonal);
                result[i, j] = inMask ? numOps.One : numOps.Zero;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSquash<T>(Tensor<T> tensor, int axis = -1)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = tensor._shape.Length + axis;

        // Compute squared norm along axis
        var squared = TensorMultiply(tensor, tensor);
        var normSquared = ReduceSum(squared, new[] { axis }, keepDims: true);

        // Compute scale factor: ||v||^2 / (1 + ||v||^2)
        var one = TensorAllocator.RentUninitialized<T>(normSquared._shape);
        one.Fill(numOps.One);
        var denom = TensorAdd(one, normSquared);
        var scale = TensorDivide(normSquared, denom);

        // Compute ||v||
        var norm = TensorSqrt(normSquared);
        var epsilon = TensorAllocator.RentUninitialized<T>(norm._shape);
        epsilon.Fill(numOps.FromDouble(1e-8));
        norm = TensorAdd(norm, epsilon);

        // Normalize: v / ||v||
        var normalized = TensorBroadcastDivide(tensor, norm);

        // Apply scale
        return TensorMultiply(scale, normalized);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSquashBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, int axis = -1)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (output == null) throw new ArgumentNullException(nameof(output));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = input._shape.Length + axis;

        // This is a simplified gradient - full implementation would require proper Jacobian
        // For now, approximate with element-wise gradient scaling
        var squared = TensorMultiply(input, input);
        var normSquared = ReduceSum(squared, new[] { axis }, keepDims: true);
        var one = TensorAllocator.RentUninitialized<T>(normSquared._shape);
        one.Fill(numOps.One);
        var denom = TensorAdd(one, normSquared);
        var scale = TensorDivide(one, denom);

        return TensorMultiply(gradOutput, scale);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorNorm<T>(Tensor<T> tensor, int axis, bool keepDims = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Compute squared values
        var squared = TensorMultiply(tensor, tensor);
        // Sum along axis
        var sumSquared = ReduceSum(squared, new[] { axis }, keepDims: keepDims);
        // Square root
        return TensorSqrt(sumSquared);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorNormalize<T>(Tensor<T> tensor, int axis, T epsilon)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        // Compute norm
        var norm = TensorNorm(tensor, axis, keepDims: true);

        // Add epsilon for numerical stability
        var epsArray = TensorAllocator.RentUninitialized<T>(norm._shape);
        epsArray.Fill(epsilon);
        norm = TensorAdd(norm, epsArray);

        // Divide
        return TensorBroadcastDivide(tensor, norm);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorClip<T>(Tensor<T> tensor, T minValue, T maxValue)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        return TensorClamp(tensor, minValue, maxValue);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorConcatenate<T>(Tensor<T>[] tensors, int axis = 0)
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentNullException(nameof(tensors));

        var result = Tensor<T>.Concatenate(tensors, axis);
        DifferentiableOps.RecordIfActive("Concatenate", result, (Tensor<T>[])tensors.Clone(),
            BackwardFunctions<T>.ConcatenateBackward,
            savedState: new object[] { axis });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorSplit<T>(Tensor<T> tensor, int numSplits, int axis = 0)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (numSplits <= 0) throw new ArgumentException("Number of splits must be positive", nameof(numSplits));

        // Normalize axis
        if (axis < 0) axis = tensor._shape.Length + axis;

        int axisSize = tensor._shape[axis];
        if (axisSize % numSplits != 0)
            throw new ArgumentException($"Cannot split axis of size {axisSize} into {numSplits} equal parts");

        int splitSize = axisSize / numSplits;
        var results = new Tensor<T>[numSplits];

        for (int i = 0; i < numSplits; i++)
        {
            int start = i * splitSize;
            results[i] = tensor.Slice(axis, start, start + splitSize);
            DifferentiableOps.RecordUnary("Split", results[i], tensor,
                BackwardFunctions<T>.SplitBackward,
                savedState: new object[] { axis, start, tensor._shape.ToArray() });
        }

        return results;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorOneHot<T>(Tensor<int> indices, int depth)
    {
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (depth <= 0) throw new ArgumentException("Depth must be positive", nameof(depth));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numIndices = indices.Length;
        var result = TensorAllocator.Rent<T>([numIndices, depth]);
        result.Fill(numOps.Zero);

        var idxData = indices.GetFlattenedData();
        var resultData = result.GetDataArray();

        for (int i = 0; i < numIndices; i++)
        {
            int idx = idxData[i];
            if (idx >= 0 && idx < depth)
            {
                resultData[i * depth + idx] = numOps.One;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<int> TensorArgMax<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = tensor._shape.Length + axis;

        // Build output shape (remove axis dimension)
        var outputShape = tensor._shape.Where((_, i) => i != axis).ToArray();
        if (outputShape.Length == 0) outputShape = new[] { 1 };
        var result = new Tensor<int>(outputShape);

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= tensor._shape[i];

        int axisSize = tensor._shape[axis];

        int innerSize = 1;
        for (int i = axis + 1; i < tensor._shape.Length; i++) innerSize *= tensor._shape[i];

        var tensorData = tensor.GetFlattenedData();
        var resultData = result.GetDataArray();

        Parallel.For(0, outerSize * innerSize, flatIdx =>
        {
            int outer = flatIdx / innerSize;
            int inner = flatIdx % innerSize;

            T maxVal = tensorData[(outer * axisSize) * innerSize + inner];
            int maxIdx = 0;

            for (int a = 1; a < axisSize; a++)
            {
                int srcIdx = (outer * axisSize + a) * innerSize + inner;
                T val = tensorData[srcIdx];
                if (numOps.ToDouble(val) > numOps.ToDouble(maxVal))
                {
                    maxVal = val;
                    maxIdx = a;
                }
            }

            resultData[flatIdx] = maxIdx;
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<int> TensorArgMin<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();

        // Normalize axis
        if (axis < 0) axis = tensor._shape.Length + axis;

        // Build output shape (remove axis dimension)
        var outputShape = tensor._shape.Where((_, i) => i != axis).ToArray();
        if (outputShape.Length == 0) outputShape = new[] { 1 };
        var result = new Tensor<int>(outputShape);

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= tensor._shape[i];

        int axisSize = tensor._shape[axis];

        int innerSize = 1;
        for (int i = axis + 1; i < tensor._shape.Length; i++) innerSize *= tensor._shape[i];

        var tensorData = tensor.GetFlattenedData();
        var resultData = result.GetDataArray();

        Parallel.For(0, outerSize * innerSize, flatIdx =>
        {
            int outer = flatIdx / innerSize;
            int inner = flatIdx % innerSize;

            T minVal = tensorData[(outer * axisSize) * innerSize + inner];
            int minIdx = 0;

            for (int a = 1; a < axisSize; a++)
            {
                int srcIdx = (outer * axisSize + a) * innerSize + inner;
                T val = tensorData[srcIdx];
                if (numOps.ToDouble(val) < numOps.ToDouble(minVal))
                {
                    minVal = val;
                    minIdx = a;
                }
            }

            resultData[flatIdx] = minIdx;
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorBinaryCrossEntropy<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));
        if (!predictions.IsContiguous) predictions = predictions.Contiguous();
        if (!targets.IsContiguous) targets = targets.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(predictions._shape);
        var predSpan = predictions.AsSpan();
        var targetSpan = targets.AsSpan();
        var dest = result.AsWritableSpan();
        T upperBound = numOps.Subtract(numOps.One, epsilon);
        double upperVal = numOps.ToDouble(upperBound);
        double epsVal = numOps.ToDouble(epsilon);

        for (int i = 0; i < predSpan.Length; i++)
        {
            T p = predSpan[i];
            T t = targetSpan[i];

            // Clip for numerical stability
            double pVal = numOps.ToDouble(p);
            T clippedUpper = pVal < upperVal ? p : upperBound;
            double clippedUpperVal = numOps.ToDouble(clippedUpper);
            T pClipped = clippedUpperVal > epsVal ? clippedUpper : epsilon;

            // -[t * log(p) + (1-t) * log(1-p)]
            T logP = numOps.Log(pClipped);
            T log1MinusP = numOps.Log(numOps.Subtract(numOps.One, pClipped));
            T oneMinusT = numOps.Subtract(numOps.One, t);

            dest[i] = numOps.Negate(numOps.Add(
                numOps.Multiply(t, logP),
                numOps.Multiply(oneMinusT, log1MinusP)));
        }

        DifferentiableOps.RecordBinary("TensorBinaryCrossEntropy", result, predictions, targets, BackwardFunctions<T>.BinaryCrossEntropyBackward, new object[] { numOps.ToDouble(epsilon) });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorBinaryCrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));
        if (!predictions.IsContiguous) predictions = predictions.Contiguous();
        if (!targets.IsContiguous) targets = targets.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(predictions._shape);
        var predSpan = predictions.AsSpan();
        var targetSpan = targets.AsSpan();
        var dest = result.AsWritableSpan();
        T upperBound = numOps.Subtract(numOps.One, epsilon);
        double upperVal = numOps.ToDouble(upperBound);
        double epsVal = numOps.ToDouble(epsilon);

        for (int i = 0; i < predSpan.Length; i++)
        {
            T p = predSpan[i];
            T t = targetSpan[i];

            // Clip for numerical stability
            double pVal = numOps.ToDouble(p);
            T clippedUpper = pVal < upperVal ? p : upperBound;
            double clippedUpperVal = numOps.ToDouble(clippedUpper);
            T pClipped = clippedUpperVal > epsVal ? clippedUpper : epsilon;

            // Gradient: -t/p + (1-t)/(1-p)
            T termA = numOps.Divide(t, pClipped);
            T oneMinusP = numOps.Subtract(numOps.One, pClipped);
            T oneMinusT = numOps.Subtract(numOps.One, t);
            T termB = numOps.Divide(oneMinusT, oneMinusP);

            dest[i] = numOps.Subtract(termB, termA);
        }

        return result;
    }

    /// <inheritdoc/>
    public (Tensor<T> X, Tensor<T> Y) TensorMeshgrid<T>(Tensor<T> x, Tensor<T> y)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));

        int width = x.Length;
        int height = y.Length;

        var X = TensorAllocator.Rent<T>([height, width]);
        var Y = TensorAllocator.Rent<T>([height, width]);
        var xData = x.GetDataArray();
        var yData = y.GetDataArray();
        var XData = X.GetDataArray();
        var YData = Y.GetDataArray();

        Parallel.For(0, height, row =>
        {
            T yVal = yData[row];
            int rowOffset = row * width;
            Array.Copy(xData, 0, XData, rowOffset, width);
            for (int col = 0; col < width; col++)
            {
                YData[rowOffset + col] = yVal;
            }
        });

        return (X, Y);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSliceAxis<T>(Tensor<T> tensor, int axis, int index)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank != 3) throw new ArgumentException("TensorSliceAxis currently only supports 3D tensors.");

        int dim0 = tensor._shape[0];
        int dim1 = tensor._shape[1];
        int dim2 = tensor._shape[2];

        Tensor<T> result;
        var tensorData = tensor.GetFlattenedData();

        switch (axis)
        {
            case 0:
                // Slice along first axis: result[j,k] = tensor[index, j, k]
                result = TensorAllocator.Rent<T>([dim1, dim2]);
                // tensor flat: index * dim1 * dim2 + j * dim2 + k => contiguous block
                Array.Copy(tensorData, index * dim1 * dim2, result.GetDataArray(), 0, dim1 * dim2);
                break;

            case 1:
                // Slice along second axis: result[i,k] = tensor[i, index, k]
                result = TensorAllocator.Rent<T>([dim0, dim2]);
                var resultData1 = result.GetDataArray();
                Parallel.For(0, dim0, i =>
                {
                    int srcOffset = i * dim1 * dim2 + index * dim2;
                    Array.Copy(tensorData, srcOffset, resultData1, i * dim2, dim2);
                });
                break;

            case 2:
                // Slice along third axis: result[i,j] = tensor[i, j, index]
                result = TensorAllocator.Rent<T>([dim0, dim1]);
                var resultData2 = result.GetDataArray();
                Parallel.For(0, dim0, i =>
                {
                    for (int j = 0; j < dim1; j++)
                    {
                        resultData2[i * dim1 + j] = tensorData[i * dim1 * dim2 + j * dim2 + index];
                    }
                });
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(axis), "Axis must be 0, 1, or 2 for 3D tensors.");
        }

        DifferentiableOps.RecordUnary("TensorSliceAxis", result, tensor, BackwardFunctions<T>.SliceAxisBackward, new object[] { axis, index });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorLinspace<T>(T start, T end, int count)
    {
        if (count < 2) throw new ArgumentException("Count must be at least 2.", nameof(count));

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>([count]);

        T range = numOps.Subtract(end, start);
        T divisor = numOps.FromDouble(count - 1);
        T step = numOps.Divide(range, divisor);

        var resultData = result.GetDataArray();
        Parallel.For(0, count, i =>
        {
            T value = numOps.Add(start, numOps.Multiply(numOps.FromDouble(i), step));
            resultData[i] = value;
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorBatchMatMul<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        // Stride-aware: only materialize views that aren't simple transposes
        // Simple transposes can be handled by GEMM transA/transB flags (zero-copy)
        if (!a.IsContiguous && !a.IsSimpleTranspose) a = a.Contiguous();
        if (!b.IsContiguous && !b.IsSimpleTranspose) b = b.Contiguous();

        // If both tensors are 3D, delegate to BatchMatMul
        if (a.Rank == 3 && b.Rank == 3)
        {
            // Delegate to BatchMatMul which already records to the tape — do NOT record again
            return BatchMatMul(a, b);
        }

        // Handle broadcasting case where b is 2D [K, N]
        if (a.Rank != 3 || b.Rank != 2)
        {
            throw new ArgumentException(
                $"TensorBatchMatMul requires a to be 3D and b to be 2D or 3D. Got ranks {a.Rank} and {b.Rank}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();

        // a: [batch, M, K], b: [K, N]
        int batch = a._shape[0];
        int M = a._shape[1];
        int K = a._shape[2];
        int N = b._shape[1];

        if (b._shape[0] != K)
        {
            throw new ArgumentException(
                $"Matrix dimensions incompatible. a has shape [{batch}, {M}, {K}], b has shape [{b._shape[0]}, {N}]. Inner dimensions must match.");
        }

        var result = TensorAllocator.Rent<T>([batch, M, N]);
        var aData = a.GetDataArray();
        var bData = b.GetDataArray();
        var resultData = result.GetDataArray();

        Parallel.For(0, batch, batchIdx =>
        {
            int aBase = batchIdx * M * K;
            int rBase = batchIdx * M * N;
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    T sum = numOps.Zero;
                    for (int k = 0; k < K; k++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(aData[aBase + i * K + k], bData[k * N + j]));
                    }
                    resultData[rBase + i * N + j] = sum;
                }
            }
        });

        DifferentiableOps.RecordBinary("BatchMatMul", result, a, b,
            BackwardFunctions<T>.BatchMatMulBackward);
        return result;
    }

    /// <inheritdoc/>
    public void TensorSetSliceAxis<T>(Tensor<T> destination, Tensor<T> source, int axis, int index)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (source == null) throw new ArgumentNullException(nameof(source));

        // For 3D tensors
        if (destination.Rank == 3)
        {
            int dim0 = destination._shape[0];
            int dim1 = destination._shape[1];
            int dim2 = destination._shape[2];

            switch (axis)
            {
                case 0:
                    // Set destination[index, :, :] = source
                    Parallel.For(0, dim1, j =>
                    {
                        for (int k = 0; k < dim2; k++)
                        {
                            destination[index, j, k] = source[j, k];
                        }
                    });
                    break;

                case 1:
                    Parallel.For(0, dim0, i =>
                    {
                        for (int k = 0; k < dim2; k++)
                        {
                            destination[i, index, k] = source[i, k];
                        }
                    });
                    break;

                case 2:
                    Parallel.For(0, dim0, i =>
                    {
                        for (int j = 0; j < dim1; j++)
                        {
                            destination[i, j, index] = source[i, j];
                        }
                    });
                    break;

                default:
                    throw new ArgumentOutOfRangeException(nameof(axis));
            }
        }
        else
        {
            throw new NotSupportedException("TensorSetSliceAxis currently only supports 3D tensors.");
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSoftmax<T>(Tensor<T> tensor, int axis)
    {
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        // Delegate to Softmax which has the same functionality
        return Softmax(tensor, axis);
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSoftmaxBackward<T>(Tensor<T> softmaxOutput, Tensor<T> outputGradient, int axis)
    {
        if (!softmaxOutput.IsContiguous) softmaxOutput = softmaxOutput.Contiguous();
        if (!outputGradient.IsContiguous) outputGradient = outputGradient.Contiguous();
        // Delegate to SoftmaxBackward with reordered parameters (grad first, then output)
        return SoftmaxBackward(outputGradient, softmaxOutput, axis);
    }

    /// <inheritdoc/>
    public virtual unsafe Tensor<T> TensorLogSoftmax<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        int rank = tensor.Rank;
        if (axis < 0) axis = rank + axis;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {rank} dimensions");

        // Compute outer and inner sizes relative to the axis
        int outerSize = 1, axisSize = tensor._shape[axis], innerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= tensor._shape[i];
        for (int i = axis + 1; i < rank; i++) innerSize *= tensor._shape[i];

        // Fast SIMD path for float when log_softmax is on the last axis (innerSize==1)
        if (typeof(T) == typeof(float) && innerSize == 1)
        {
            // Use Pin() for NativeMemory compatibility
            var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
            using var pinIn = tensor.Data.Pin();
            using var pinOut = result.Data.Pin();
            unsafe
            {
                LogSoftmaxFloatFastPtr((float*)pinIn.Pointer, (float*)pinOut.Pointer, outerSize, axisSize);
            }
            DifferentiableOps.RecordUnary("LogSoftmax", result, tensor,
                BackwardFunctions<T>.LogSoftmaxBackward);
            return result;
        }

        // Generic scalar fallback
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = tensor.GetFlattenedData();
        var outputData = new T[tensor.Length];

        Parallel.For(0, outerSize * innerSize, idx =>
        {
            int outer = idx / innerSize;
            int inner = idx % innerSize;

            // Find max for numerical stability
            T maxVal = numOps.MinValue;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                if (numOps.GreaterThan(inputData[flatIdx], maxVal))
                    maxVal = inputData[flatIdx];
            }

            // Compute exp(x - max) and sum
            T sumExp = numOps.Zero;
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                T shifted = numOps.Subtract(inputData[flatIdx], maxVal);
                sumExp = numOps.Add(sumExp, numOps.Exp(shifted));
            }

            // log_softmax = (x - max) - log(sum(exp(x - max)))
            T logSumExp = numOps.Log(sumExp);
            for (int i = 0; i < axisSize; i++)
            {
                int flatIdx = (outer * axisSize + i) * innerSize + inner;
                T shifted = numOps.Subtract(inputData[flatIdx], maxVal);
                outputData[flatIdx] = numOps.Subtract(shifted, logSumExp);
            }
        });

        var logSoftmaxResult = TensorAllocator.Rent<T>(tensor._shape, new Vector<T>(outputData));
        DifferentiableOps.RecordUnary("LogSoftmax", logSoftmaxResult, tensor,
            BackwardFunctions<T>.LogSoftmaxBackward);
        return logSoftmaxResult;
    }

    /// <summary>
    /// Fast SIMD log_softmax for float arrays on last axis.
    /// log_softmax(x) = (x - max) - log(sum(exp(x - max)))
    /// </summary>
    /// <summary>Pointer-based log-softmax — works with both managed and NativeMemory.</summary>
    private static unsafe void LogSoftmaxFloatFastPtr(float* pIn, float* pOut, int outerSize, int axisSize)
    {
        int maxThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        bool useParallel = outerSize >= 4 && outerSize * axisSize >= 32768;

        if (useParallel)
        {
            float* pInCap = pIn;
            float* pOutCap = pOut;
            int axisSz = axisSize;
            int outerSz = outerSize;
            int numChunks = Math.Min(maxThreads, outerSize);
            int rowsPerChunk = (outerSize + numChunks - 1) / numChunks;

            PersistentParallelExecutor.Instance.Execute(numChunks, chunk =>
            {
                int startRow = chunk * rowsPerChunk;
                int endRow = Math.Min(startRow + rowsPerChunk, outerSz);
                for (int row = startRow; row < endRow; row++)
                {
                    SimdKernels.LogSoftmaxRowUnsafe(pInCap + row * axisSz, pOutCap + row * axisSz, axisSz);
                }
            });
        }
        else
        {
            for (int row = 0; row < outerSize; row++)
                SimdKernels.LogSoftmaxRowUnsafe(pIn + row * axisSize, pOut + row * axisSize, axisSize);
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorTopK<T>(Tensor<T> tensor, int k, int axis, out Tensor<int> indices)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (k <= 0) throw new ArgumentException("k must be positive.", nameof(k));

        var numOps = MathHelper.GetNumericOperations<T>();
        if (axis < 0) axis = tensor.Rank + axis;

        // For simplicity, handle 2D case: [batch, features]
        if (tensor.Rank == 2 && axis == 1)
        {
            int batch = tensor._shape[0];
            int features = tensor._shape[1];
            k = Math.Min(k, features);

            var resultValues = TensorAllocator.Rent<T>([batch, k]);
            var indicesResult = new Tensor<int>([batch, k]);

            Parallel.For(0, batch, b =>
            {
                // Extract row and sort with indices
                var rowWithIndices = new (T value, int index)[features];
                for (int i = 0; i < features; i++)
                {
                    rowWithIndices[i] = (tensor[b, i], i);
                }

                // Sort descending by value using GreaterThan
                Array.Sort(rowWithIndices, (a, x) =>
                {
                    if (numOps.GreaterThan(x.value, a.value)) return 1;
                    if (numOps.LessThan(x.value, a.value)) return -1;
                    return 0;
                });

                // Take top-k
                for (int i = 0; i < k; i++)
                {
                    resultValues[b, i] = rowWithIndices[i].value;
                    indicesResult[b, i] = rowWithIndices[i].index;
                }
            });

            indices = indicesResult;
            return resultValues;
        }
        else
        {
            throw new NotSupportedException("TensorTopK currently only supports 2D tensors with axis=1.");
        }
    }

    /// <inheritdoc/>
    public Tensor<T> TensorScatter<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> source, int axis)
    {
        if (destination == null) throw new ArgumentNullException(nameof(destination));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (source == null) throw new ArgumentNullException(nameof(source));

        var result = destination.Clone();

        // For 2D with axis=1: result[b, indices[b, i]] = source[b, i]
        if (result.Rank == 2 && axis == 1)
        {
            int batch = result._shape[0];
            int numIndices = indices._shape[1];

            Parallel.For(0, batch, b =>
            {
                for (int i = 0; i < numIndices; i++)
                {
                    int idx = indices[b, i];
                    result[b, idx] = source[b, i];
                }
            });
        }
        else
        {
            throw new NotSupportedException("TensorScatter currently only supports 2D tensors with axis=1.");
        }

        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndexSelect<T>(Tensor<T> tensor, Tensor<int> indices, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));

        if (axis < 0) axis = tensor.Rank + axis;

        // For 2D tensor with axis=0: select rows
        if (tensor.Rank == 2 && axis == 0)
        {
            int numIndices = indices.Length;
            int cols = tensor._shape[1];
            var result = TensorAllocator.Rent<T>([numIndices, cols]);

            var indicesData = indices.GetFlattenedData();
            var tensorData = tensor.GetFlattenedData();
            var resultData = result.GetDataArray();
            Parallel.For(0, numIndices, i =>
            {
                int idx = indicesData[i];
                Array.Copy(tensorData, idx * cols, resultData, i * cols, cols);
            });

            DifferentiableOps.RecordUnary("TensorIndexSelect", result, tensor,
                BackwardFunctions<T>.IndexSelectBackward, new object[] { indices, axis });
            return result;
        }
        else if (tensor.Rank == 2 && axis == 1)
        {
            int rows = tensor._shape[0];
            int numIndices = indices.Length;
            var result = TensorAllocator.Rent<T>([rows, numIndices]);
            var indicesData = indices.GetFlattenedData();
            var tensorData = tensor.GetFlattenedData();
            var resultData = result.GetDataArray();

            Parallel.For(0, rows, i =>
            {
                int rowOffset = i * numIndices;
                int tensorRowOffset = i * tensor._shape[1];
                for (int j = 0; j < numIndices; j++)
                {
                    resultData[rowOffset + j] = tensorData[tensorRowOffset + indicesData[j]];
                }
            });

            DifferentiableOps.RecordUnary("TensorIndexSelect", result, tensor,
                BackwardFunctions<T>.IndexSelectBackward, new object[] { indices, axis });
            return result;
        }
        else
        {
            throw new NotSupportedException("TensorIndexSelect currently only supports 2D tensors.");
        }
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorStack<T>(Tensor<T>[] tensors, int axis)
    {
        if (tensors == null || tensors.Length == 0)
            throw new ArgumentException("Tensors array must not be empty.", nameof(tensors));

        int numTensors = tensors.Length;
        var firstShape = tensors[0]._shape;

        // Validate all tensors have same shape
        for (int i = 1; i < numTensors; i++)
        {
            if (!tensors[i]._shape.SequenceEqual(firstShape))
                throw new ArgumentException("All tensors must have the same shape.");
        }

        if (axis < 0) axis = firstShape.Length + 1 + axis;

        // New shape: insert numTensors at axis position
        var newShape = new int[firstShape.Length + 1];
        for (int i = 0; i < axis; i++) newShape[i] = firstShape[i];
        newShape[axis] = numTensors;
        for (int i = axis; i < firstShape.Length; i++) newShape[i + 1] = firstShape[i];

        var result = TensorAllocator.RentUninitialized<T>(newShape);

        var resultData = result.GetDataArray();

        // Copy each tensor
        Parallel.For(0, numTensors, t =>
        {
            var tensor = tensors[t];
            var tensorData = tensor.GetFlattenedData();
            int sliceSize = 1;
            for (int i = axis + 1; i < newShape.Length; i++) sliceSize *= newShape[i];

            int outerSize = 1;
            for (int i = 0; i < axis; i++) outerSize *= newShape[i];

            for (int outer = 0; outer < outerSize; outer++)
            {
                int srcOffset = outer * sliceSize;
                int dstOffset = (outer * numTensors + t) * sliceSize;
                Array.Copy(tensorData, srcOffset, resultData, dstOffset, sliceSize);
            }
        });

        DifferentiableOps.RecordIfActive("TensorStack", result, tensors,
            BackwardFunctions<T>.StackBackward, new object[] { axis });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T>[] TensorUnstack<T>(Tensor<T> tensor, int axis)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));

        if (axis < 0) axis = tensor.Rank + axis;

        int numSlices = tensor._shape[axis];
        var result = new Tensor<T>[numSlices];

        // New shape: remove the axis dimension
        var newShape = new int[tensor.Rank - 1];
        for (int i = 0, j = 0; i < tensor.Rank; i++)
        {
            if (i != axis) newShape[j++] = tensor._shape[i];
        }

        Parallel.For(0, numSlices, i =>
        {
            result[i] = TensorSliceAxis(tensor, axis, i);
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMap<T>(Tensor<T> tensor, Func<T, T> func)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (func == null) throw new ArgumentNullException(nameof(func));

        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        try
        {
            var src = tensor.GetFlattenedData();
            var dest = result.GetDataArray();

            for (int i = 0; i < tensor.Length; i++)
                dest[i] = func(src[i]);

            return result;
        }
        catch
        {
            TensorAllocator.Return(result);
            throw;
        }
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMaskedFill<T>(Tensor<T> tensor, Tensor<bool> mask, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (mask == null) throw new ArgumentNullException(nameof(mask));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var result = tensor.Clone();
        var maskSpan = mask.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < maskSpan.Length; i++)
        {
            if (maskSpan[i])
                dest[i] = value;
        }

        DifferentiableOps.RecordUnary("TensorMaskedFill", result, tensor,
            BackwardFunctions<T>.MaskedFillBackward, new object[] { mask });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorMaskedFill<T>(Tensor<T> tensor, Tensor<Bit> mask, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (mask == null) throw new ArgumentNullException(nameof(mask));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!tensor._shape.SequenceEqual(mask._shape))
            throw new ArgumentException($"Tensor shape [{string.Join(", ", tensor._shape)}] must match mask shape [{string.Join(", ", mask._shape)}].");

        var result = tensor.Clone();
        var maskSpan = mask.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < maskSpan.Length; i++)
        {
            if ((bool)maskSpan[i])
                dest[i] = value;
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorWhere<T>(Tensor<bool> condition, Tensor<T> x, Tensor<T> y)
    {
        if (condition == null) throw new ArgumentNullException(nameof(condition));
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));

        var result = TensorAllocator.RentUninitialized<T>(x._shape);
        var condData = condition.GetFlattenedData();
        var xData = x.GetFlattenedData();
        var yData = y.GetFlattenedData();
        var rData = result.GetDataArray();

        Parallel.For(0, x.Length, i =>
        {
            rData[i] = condData[i] ? xData[i] : yData[i];
        });

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorWhere<T>(Tensor<Bit> condition, Tensor<T> x, Tensor<T> y)
    {
        if (condition == null) throw new ArgumentNullException(nameof(condition));
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (y == null) throw new ArgumentNullException(nameof(y));
        if (!y.IsContiguous) y = y.Contiguous();
        if (!x.IsContiguous) x = x.Contiguous();
        if (!condition.IsContiguous) condition = condition.Contiguous();
        if (x.Length != y.Length || x.Length != condition.Length)
            throw new ArgumentException("All tensors must have the same length.");

        var result = TensorAllocator.RentUninitialized<T>(x._shape);
        var condData = condition.GetFlattenedData();
        var xData = x.GetFlattenedData();
        var yData = y.GetFlattenedData();
        var rData = result.GetDataArray();

        Parallel.For(0, x.Length, i =>
        {
            rData[i] = (bool)condData[i] ? xData[i] : yData[i];
        });

        return result;
    }

    #endregion

    #region Neural Radiance Fields Operations

    /// <inheritdoc/>
    public Tensor<T> PositionalEncoding<T>(Tensor<T> positions, int numFrequencies)
    {
        return NeRFOperations.PositionalEncoding(positions, numFrequencies);
    }

    /// <inheritdoc/>
    public Tensor<T> PositionalEncodingBackward<T>(Tensor<T> positions, Tensor<T> encodedGradient, int numFrequencies)
    {
        return NeRFOperations.PositionalEncodingBackward(positions, encodedGradient, numFrequencies);
    }

    /// <inheritdoc/>
    public Tensor<T> VolumeRendering<T>(Tensor<T> rgbSamples, Tensor<T> densitySamples, Tensor<T> tValues)
    {
        return NeRFOperations.VolumeRendering(rgbSamples, densitySamples, tValues);
    }

    /// <inheritdoc/>
    public void VolumeRenderingBackward<T>(
        Tensor<T> rgbSamples,
        Tensor<T> densitySamples,
        Tensor<T> tValues,
        Tensor<T> outputGradient,
        out Tensor<T> rgbGradient,
        out Tensor<T> densityGradient)
    {
        NeRFOperations.VolumeRenderingBackward(
            rgbSamples, densitySamples, tValues, outputGradient,
            out rgbGradient, out densityGradient);
    }

    /// <inheritdoc/>
    public (Tensor<T> positions, Tensor<T> directions, Tensor<T> tValues) SampleRayPoints<T>(
        Tensor<T> rayOrigins,
        Tensor<T> rayDirections,
        T nearBound,
        T farBound,
        int numSamples,
        bool stratified = true)
    {
        return NeRFOperations.SampleRayPoints(
            rayOrigins, rayDirections, nearBound, farBound, numSamples, stratified);
    }

    /// <inheritdoc/>
    public Tensor<T> ImportanceSampling<T>(Tensor<T> tValuesCoarse, Tensor<T> weightsCoarse, int numFineSamples)
    {
        return NeRFOperations.ImportanceSampling(tValuesCoarse, weightsCoarse, numFineSamples);
    }

    /// <inheritdoc/>
    public (Tensor<T> origins, Tensor<T> directions) GenerateCameraRays<T>(
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        int imageWidth,
        int imageHeight,
        T focalLength)
    {
        return NeRFOperations.GenerateCameraRays(
            cameraPosition, cameraRotation, imageWidth, imageHeight, focalLength);
    }

    #endregion

    #region Gaussian Splatting Operations

    /// <inheritdoc/>
    public void ProjectGaussians3DTo2D<T>(
        Tensor<T> means3D,
        Tensor<T> covariances3D,
        Matrix<T> viewMatrix,
        Matrix<T> projMatrix,
        int imageWidth,
        int imageHeight,
        out Tensor<T> means2D,
        out Tensor<T> covariances2D,
        out Tensor<T> depths,
        out Tensor<bool> visible)
    {
        GaussianSplattingOperations.ProjectGaussians3DTo2D(
            means3D, covariances3D, viewMatrix, projMatrix,
            imageWidth, imageHeight,
            out means2D, out covariances2D, out depths, out visible);
    }

    /// <inheritdoc/>
    public Tensor<T> RasterizeGaussians<T>(
        Tensor<T> means2D,
        Tensor<T> covariances2D,
        Tensor<T> colors,
        Tensor<T> opacities,
        Tensor<T> depths,
        int imageWidth,
        int imageHeight,
        int tileSize = 16)
    {
        return GaussianSplattingOperations.RasterizeGaussians(
            means2D, covariances2D, colors, opacities, depths,
            imageWidth, imageHeight, tileSize);
    }

    /// <inheritdoc/>
    public void RasterizeGaussiansBackward<T>(
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
        out Tensor<T> opacitiesGrad)
    {
        GaussianSplattingOperations.RasterizeGaussiansBackward(
            means2D, covariances2D, colors, opacities, depths,
            imageWidth, imageHeight, outputGradient, tileSize,
            out means2DGrad, out covariances2DGrad, out colorsGrad, out opacitiesGrad);
    }

    /// <inheritdoc/>
    public Tensor<T> EvaluateSphericalHarmonics<T>(Tensor<T> shCoefficients, Tensor<T> viewDirections, int degree)
    {
        return GaussianSplattingOperations.EvaluateSphericalHarmonics(shCoefficients, viewDirections, degree);
    }

    /// <inheritdoc/>
    public Tensor<T> EvaluateSphericalHarmonicsBackward<T>(
        Tensor<T> shCoefficients,
        Tensor<T> viewDirections,
        int degree,
        Tensor<T> outputGradient)
    {
        return GaussianSplattingOperations.EvaluateSphericalHarmonicsBackward(
            shCoefficients, viewDirections, degree, outputGradient);
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeGaussianCovariance<T>(Tensor<T> rotations, Tensor<T> scales)
    {
        return GaussianSplattingOperations.ComputeGaussianCovariance(rotations, scales);
    }

    /// <inheritdoc/>
    public void ComputeGaussianCovarianceBackward<T>(
        Tensor<T> rotations,
        Tensor<T> scales,
        Tensor<T> covarianceGradient,
        out Tensor<T> rotationsGrad,
        out Tensor<T> scalesGrad)
    {
        GaussianSplattingOperations.ComputeGaussianCovarianceBackward(
            rotations, scales, covarianceGradient,
            out rotationsGrad, out scalesGrad);
    }

    #endregion

    #region Instant-NGP Operations

    /// <inheritdoc/>
    public Tensor<T> MultiresolutionHashEncoding<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel)
    {
        return InstantNGPOperations.MultiresolutionHashEncoding(
            positions, hashTables, resolutions, featuresPerLevel);
    }

    /// <inheritdoc/>
    public Tensor<T>[] MultiresolutionHashEncodingBackward<T>(
        Tensor<T> positions,
        Tensor<T>[] hashTables,
        int[] resolutions,
        int featuresPerLevel,
        Tensor<T> outputGradient)
    {
        return InstantNGPOperations.MultiresolutionHashEncodingBackward(
            positions, hashTables, resolutions, featuresPerLevel, outputGradient);
    }

    /// <inheritdoc/>
    public Tensor<T> UpdateOccupancyGrid<T>(
        Tensor<T> occupancyGrid,
        Tensor<T> densities,
        Tensor<T> positions,
        int gridSize,
        T threshold,
        T decayFactor)
    {
        return InstantNGPOperations.UpdateOccupancyGrid(
            occupancyGrid, densities, positions, gridSize, threshold, decayFactor);
    }

    /// <inheritdoc/>
    public (Tensor<T> positions, Tensor<T> directions, Tensor<bool> validMask, Tensor<T> tValues) SampleRaysWithOccupancy<T>(
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
        return InstantNGPOperations.SampleRaysWithOccupancy(
            rayOrigins, rayDirections, occupancyBitfield, gridSize,
            sceneBoundsMin, sceneBoundsMax, nearBound, farBound, maxSamples);
    }

    #endregion

    #region Mesh Convolution Operations

    /// <inheritdoc/>
    public Tensor<T> SpiralConv<T>(
        Tensor<T> vertexFeatures,
        Tensor<int> spiralIndices,
        Tensor<T> weights,
        Tensor<T> biases)
    {
        return MeshConvolutionOperations.SpiralConv(vertexFeatures, spiralIndices, weights, biases);
    }

    /// <inheritdoc/>
    public Tensor<T> SpiralConvBackwardInput<T>(
        Tensor<T> outputGradient,
        Tensor<int> spiralIndices,
        Tensor<T> weights,
        int inputChannels)
    {
        return MeshConvolutionOperations.SpiralConvBackwardInput(outputGradient, spiralIndices, weights, inputChannels);
    }

    /// <inheritdoc/>
    public Tensor<T> SpiralConvBackwardWeights<T>(
        Tensor<T> outputGradient,
        Tensor<T> vertexFeatures,
        Tensor<int> spiralIndices)
    {
        return MeshConvolutionOperations.SpiralConvBackwardWeights(outputGradient, vertexFeatures, spiralIndices);
    }

    /// <inheritdoc/>
    public Tensor<T> SpiralConvBackwardBias<T>(Tensor<T> outputGradient)
    {
        return MeshConvolutionOperations.SpiralConvBackwardBias(outputGradient);
    }

    /// <inheritdoc/>
    public Tensor<T> DiffusionConv<T>(
        Tensor<T> vertexFeatures,
        Tensor<T> laplacian,
        Tensor<T> weights,
        Tensor<T> biases,
        T diffusionTime)
    {
        return MeshConvolutionOperations.DiffusionConv(vertexFeatures, laplacian, weights, biases, diffusionTime);
    }

    /// <inheritdoc/>
    public (Tensor<T> inputGrad, Tensor<T> weightGrad, Tensor<T> biasGrad) DiffusionConvBackward<T>(
        Tensor<T> outputGradient,
        Tensor<T> vertexFeatures,
        Tensor<T> laplacian,
        Tensor<T> weights,
        T diffusionTime)
    {
        return MeshConvolutionOperations.DiffusionConvBackward(outputGradient, vertexFeatures, laplacian, weights, diffusionTime);
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeMeshLaplacian<T>(
        Tensor<T> vertices,
        Tensor<int> faces,
        LaplacianType laplacianType = LaplacianType.Cotangent)
    {
        return MeshConvolutionOperations.ComputeMeshLaplacian(vertices, faces, laplacianType);
    }

    /// <inheritdoc/>
    public Tensor<int> GenerateSpiralIndices<T>(
        Tensor<T> vertices,
        Tensor<int> faces,
        int spiralLength)
    {
        return MeshConvolutionOperations.GenerateSpiralIndices(vertices, faces, spiralLength);
    }

    #endregion

    #region Advanced Vectorization Operations

    /// <inheritdoc/>
    public Tensor<T> PairwiseDistanceSquared<T>(Tensor<T> x, Tensor<T> y)
    {
        if (x._shape.Length != 2 || y._shape.Length != 2)
            throw new ArgumentException("Input tensors must be 2D [N, D]");
        if (x._shape[1] != y._shape[1])
            throw new ArgumentException("Input tensors must have the same dimensionality");

        int n = x._shape[0];
        int m = y._shape[0];
        int d = x._shape[1];

        var result = TensorAllocator.Rent<T>([n, m]);
        var numOps = MathHelper.GetNumericOperations<T>();

        // Use the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
        // Compute ||x||^2 for each row of x
        var xNormSq = new T[n];
        for (int i = 0; i < n; i++)
        {
            T sum = numOps.Zero;
            for (int k = 0; k < d; k++)
            {
                T val = x[i, k];
                sum = numOps.Add(sum, numOps.Multiply(val, val));
            }
            xNormSq[i] = sum;
        }

        // Compute ||y||^2 for each row of y
        var yNormSq = new T[m];
        for (int j = 0; j < m; j++)
        {
            T sum = numOps.Zero;
            for (int k = 0; k < d; k++)
            {
                T val = y[j, k];
                sum = numOps.Add(sum, numOps.Multiply(val, val));
            }
            yNormSq[j] = sum;
        }

        // Compute result[i,j] = ||x[i]||^2 + ||y[j]||^2 - 2 * x[i].y[j]
        T two = numOps.FromDouble(2.0);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                // Compute dot product x[i].y[j]
                T dot = numOps.Zero;
                for (int k = 0; k < d; k++)
                {
                    dot = numOps.Add(dot, numOps.Multiply(x[i, k], y[j, k]));
                }
                // ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
                T dist = numOps.Subtract(numOps.Add(xNormSq[i], yNormSq[j]), numOps.Multiply(two, dot));
                result[i, j] = dist;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> PairwiseDistance<T>(Tensor<T> x, Tensor<T> y)
    {
        if (!x.IsContiguous) x = x.Contiguous();
        if (!y.IsContiguous) y = y.Contiguous();
        var distSq = PairwiseDistanceSquared(x, y);
        return TensorSqrt(distSq);
    }

    /// <inheritdoc/>
    public (Tensor<T> values, Tensor<int> indices) TopK<T>(Tensor<T> input, int k, int axis = -1, bool largest = true)
    {
        if (axis < 0) axis = input._shape.Length + axis;
        if (axis < 0 || axis >= input._shape.Length)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {input._shape.Length} dimensions");

        int axisSize = input._shape[axis];
        if (k > axisSize)
            throw new ArgumentException($"k ({k}) cannot be greater than axis size ({axisSize})");

        var numOps = MathHelper.GetNumericOperations<T>();

        // Calculate output shape (same as input but with axis dimension = k)
        var outputShape = (int[])input._shape.Clone();
        outputShape[axis] = k;

        var values = TensorAllocator.RentUninitialized<T>(outputShape);
        var indices = new Tensor<int>(outputShape);
        var inputData = input.GetDataArray();
        var valuesData = values.GetDataArray();
        var indicesData = indices.GetDataArray();

        // Calculate strides for axis iteration
        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input._shape[i];
        int innerSize = 1;
        for (int i = axis + 1; i < input._shape.Length; i++) innerSize *= input._shape[i];

        // Process each "slice" along the axis
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                // Extract values along axis
                var axisValues = new (T value, int index)[axisSize];
                for (int a = 0; a < axisSize; a++)
                {
                    int flatIndex = outer * axisSize * innerSize + a * innerSize + inner;
                    axisValues[a] = (inputData[flatIndex], a);
                }

                // Sort by value
                if (largest)
                {
                    Array.Sort(axisValues, (x, y) => numOps.Compare(y.value, x.value));
                }
                else
                {
                    Array.Sort(axisValues, (x, y) => numOps.Compare(x.value, y.value));
                }

                // Take top k
                for (int i = 0; i < k; i++)
                {
                    int outputFlatIndex = outer * k * innerSize + i * innerSize + inner;
                    valuesData[outputFlatIndex] = axisValues[i].value;
                    indicesData[outputFlatIndex] = axisValues[i].index;
                }
            }
        }

        return (values, indices);
    }

    /// <inheritdoc/>
    public Tensor<int> ArgSort<T>(Tensor<T> input, int axis = -1, bool descending = false)
    {
        if (axis < 0) axis = input._shape.Length + axis;
        if (axis < 0 || axis >= input._shape.Length)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {input._shape.Length} dimensions");

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<int>(input._shape);
        var inputData = input.GetFlattenedData();
        var resultData = result.GetDataArray();

        int axisSize = input._shape[axis];
        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input._shape[i];
        int innerSize = 1;
        for (int i = axis + 1; i < input._shape.Length; i++) innerSize *= input._shape[i];

        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                var axisValues = new (T value, int index)[axisSize];
                for (int a = 0; a < axisSize; a++)
                {
                    int flatIndex = outer * axisSize * innerSize + a * innerSize + inner;
                    axisValues[a] = (inputData[flatIndex], a);
                }

                if (descending)
                {
                    Array.Sort(axisValues, (x, y) => numOps.Compare(y.value, x.value));
                }
                else
                {
                    Array.Sort(axisValues, (x, y) => numOps.Compare(x.value, y.value));
                }

                for (int a = 0; a < axisSize; a++)
                {
                    int flatIndex = outer * axisSize * innerSize + a * innerSize + inner;
                    resultData[flatIndex] = axisValues[a].index;
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Gather<T>(Tensor<T> input, Tensor<int> indices, int axis)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        if (axis < 0) axis = input._shape.Length + axis;
        if (axis < 0 || axis >= input._shape.Length)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {input._shape.Length} dimensions");

        // Output shape: input.Shape with axis dimension replaced by indices.Length
        var outputShape = new int[input._shape.Length];
        for (int i = 0; i < input._shape.Length; i++)
        {
            outputShape[i] = i == axis ? indices.Length : input._shape[i];
        }

        var result = TensorAllocator.RentUninitialized<T>(outputShape);
        var inputData = input.GetFlattenedData();
        var indicesData = indices.GetFlattenedData();
        var resultData = result.GetDataArray();

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input._shape[i];
        int axisSize = input._shape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < input._shape.Length; i++) innerSize *= input._shape[i];

        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int idx = 0; idx < indices.Length; idx++)
            {
                int srcIdx = indicesData[idx];
                if (srcIdx < 0 || srcIdx >= axisSize)
                    throw new ArgumentException($"Index {srcIdx} is out of bounds for axis size {axisSize}");

                int srcBase = outer * axisSize * innerSize + srcIdx * innerSize;
                int dstBase = outer * indices.Length * innerSize + idx * innerSize;
                Array.Copy(inputData, srcBase, resultData, dstBase, innerSize);
            }
        }

        DifferentiableOps.RecordUnary("Gather", result, input, BackwardFunctions<T>.GatherBackward, new object[] { indices, axis });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> Scatter<T>(Tensor<T> input, Tensor<int> indices, Tensor<T> values, int axis)
    {
        if (!indices.IsContiguous) indices = indices.Contiguous();
        if (axis < 0) axis = input._shape.Length + axis;
        if (axis < 0 || axis >= input._shape.Length)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {input._shape.Length} dimensions");

        // Create a copy of input
        var result = TensorAllocator.RentUninitialized<T>(input._shape);
        var inputData = input.GetFlattenedData();
        var resultData = result.GetDataArray();
        Array.Copy(inputData, resultData, input.Length);

        var indicesData = indices.GetFlattenedData();
        var valuesData = values.GetFlattenedData();

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input._shape[i];
        int axisSize = input._shape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < input._shape.Length; i++) innerSize *= input._shape[i];

        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int idx = 0; idx < indices.Length; idx++)
            {
                int dstIdx = indicesData[idx];
                if (dstIdx < 0 || dstIdx >= axisSize)
                    throw new ArgumentException($"Index {dstIdx} is out of bounds for axis size {axisSize}");

                int dstBase = outer * axisSize * innerSize + dstIdx * innerSize;
                int srcBase = outer * indices.Length * innerSize + idx * innerSize;
                Array.Copy(valuesData, srcBase, resultData, dstBase, innerSize);
            }
        }

        DifferentiableOps.RecordBinary("Scatter", result, input, values, BackwardFunctions<T>.ScatterBackward, new object[] { indices, axis });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> ScatterAdd<T>(Tensor<T> input, Tensor<int> indices, Tensor<T> values, int axis)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        if (!indices.IsContiguous) indices = indices.Contiguous();
        if (!values.IsContiguous) values = values.Contiguous();
        if (axis < 0) axis = input._shape.Length + axis;
        if (axis < 0 || axis >= input._shape.Length)
            throw new ArgumentException($"Invalid axis {axis} for tensor with {input._shape.Length} dimensions");

        var numOps = MathHelper.GetNumericOperations<T>();

        // Create a copy of input
        var result = TensorAllocator.RentUninitialized<T>(input._shape);
        var inputData = input.GetFlattenedData();
        var resultData = result.GetDataArray();
        Array.Copy(inputData, resultData, input.Length);

        var indicesData = indices.GetFlattenedData();
        var valuesData = values.GetFlattenedData();

        int outerSize = 1;
        for (int i = 0; i < axis; i++) outerSize *= input._shape[i];
        int axisSize = input._shape[axis];
        int innerSize = 1;
        for (int i = axis + 1; i < input._shape.Length; i++) innerSize *= input._shape[i];

        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int idx = 0; idx < indices.Length; idx++)
            {
                int dstIdx = indicesData[idx];
                if (dstIdx < 0 || dstIdx >= axisSize)
                    throw new ArgumentException($"Index {dstIdx} is out of bounds for axis size {axisSize}");

                for (int inner = 0; inner < innerSize; inner++)
                {
                    int dstFlatIndex = outer * axisSize * innerSize + dstIdx * innerSize + inner;
                    int srcFlatIndex = outer * indices.Length * innerSize + idx * innerSize + inner;
                    resultData[dstFlatIndex] = numOps.Add(resultData[dstFlatIndex], valuesData[srcFlatIndex]);
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorCosh<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var srcData = tensor.GetFlattenedData();
        var dstData = result.GetDataArray();

        int length = tensor.Length;
        for (int i = 0; i < length; i++)
        {
            dstData[i] = numOps.FromDouble(Math.Cosh(numOps.ToDouble(srcData[i])));
        }

        DifferentiableOps.RecordUnary("TensorCosh", result, tensor, BackwardFunctions<T>.CoshBackward);
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorSinh<T>(Tensor<T> tensor)
    {
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var srcData = tensor.GetFlattenedData();
        var dstData = result.GetDataArray();

        int length = tensor.Length;
        for (int i = 0; i < length; i++)
        {
            dstData[i] = numOps.FromDouble(Math.Sinh(numOps.ToDouble(srcData[i])));
        }

        DifferentiableOps.RecordUnary("TensorSinh", result, tensor, BackwardFunctions<T>.SinhBackward);
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> TensorOuter<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a._shape.Length != 1 || b._shape.Length != 1)
            throw new ArgumentException("Both inputs must be 1D tensors");

        int n = a._shape[0];
        int m = b._shape[0];
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.Rent<T>([n, m]);

        for (int i = 0; i < n; i++)
        {
            T aVal = a[i];
            for (int j = 0; j < m; j++)
            {
                result[i, j] = numOps.Multiply(aVal, b[j]);
            }
        }

        DifferentiableOps.RecordBinary("TensorOuter", result, a, b, BackwardFunctions<T>.OuterProductBackward);
        return result;
    }

    #endregion

    #region Fused Operations

    /// <inheritdoc/>
    #if !NETFRAMEWORK
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
#endif
    public Tensor<T> FusedLinear<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, FusedActivationType activation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weights == null) throw new ArgumentNullException(nameof(weights));

        // When a gradient tape is active, decompose into tape-recorded primitives so
        // backward can trace the dependency chain from loss -> output -> parameters.
        // The BLAS fast path below bypasses the tape (operates on raw arrays), so we
        // must use the recorded path during training.
        if (Autodiff.GradientTape<T>.Current is not null && !Autodiff.NoGradScope<T>.IsSuppressed)
        {
            // Fused tape path: compute via BLAS, record ONE entry instead of 2-3.
            // This cuts tape overhead by 2-3x vs decomposed (MatMul + Add + Activation).
            if (activation == FusedActivationType.None || activation == FusedActivationType.ReLU)
            {
                // Compute result using BLAS (same path as inference)
                if (!input.IsContiguous) input = input.Contiguous();
                if (!weights.IsContiguous) weights = weights.Contiguous();

                Tensor<T> result;
                if (input.Rank == 2 && weights.Rank == 2 && typeof(T) == typeof(float))
                {
                    int M = input._shape[0], K = input._shape[1], N = weights._shape[1];
                    result = TensorAllocator.RentUninitialized<T>(new[] { M, N });
                    var inArr = (float[])(object)input.GetDataArray();
                    var wArr = (float[])(object)weights.GetDataArray();
                    var outArr = (float[])(object)result.GetDataArray();

                    if (!BlasProvider.TryGemm(M, N, K, inArr, 0, K, wArr, 0, N, outArr, 0, N))
                        Simd.SimdGemm.Sgemm(inArr.AsSpan(0, M * K), wArr.AsSpan(0, K * N), outArr.AsSpan(0, M * N), M, K, N);

                    if (bias != null)
                        CpuFusedOperations.ApplyBiasActivationInPlace(outArr,
                            (float[])(object)bias.GetDataArray(), M, N, activation);
                    else if (activation != FusedActivationType.None)
                        ApplyFusedActivationInPlace(result, activation);
                }
                else
                {
                    // Non-float or non-2D: fall through to decomposed
                    result = TensorMatMul(input, weights);
                    if (bias != null) result = TensorBroadcastAdd(result, bias);
                    if (activation != FusedActivationType.None) ApplyFusedActivationInPlace(result, activation);
                    // Remove decomposed entries, record single fused
                    RemoveLastNTapeEntries<T>(bias != null ? 2 : 1);
                }

                // Record single fused entry with FusedLinearBackward
                var inputs = bias != null ? new[] { input, weights, bias } : new[] { input, weights };
                Autodiff.DifferentiableOps.RecordIfActive("FusedLinear", result, inputs,
                    Autodiff.BackwardFunctions<T>.FusedLinearBackward);
                return result;
            }

            // For non-trivial activations, decompose (activation backward needs its own entry)
            {
                var result = TensorMatMul(input, weights);
                if (bias != null)
                    result = TensorBroadcastAdd(result, bias);
                if (activation != FusedActivationType.None)
                    result = ApplyActivationRecorded(result, activation);
                return result;
            }
        }

        if (!input.IsContiguous) input = input.Contiguous();
        if (!weights.IsContiguous) weights = weights.Contiguous();

        // For 2D inputs (batch x features), use optimized fused GEMM for float/double
        if (input.Rank == 2 && weights.Rank == 2)
        {
            int M = input._shape[0];  // Batch size
            int K = input._shape[1];  // Input features
            int N = weights._shape[1]; // Output features

            if (weights._shape[0] != K)
                throw new ArgumentException($"Weight matrix shape mismatch: expected [{K}, N], got [{weights._shape[0]}, {weights._shape[1]}]");

            // Ultra-fast path for float: zero-alloc arena + direct BLAS pointers
            if (typeof(T) == typeof(float))
            {
                var result = TensorAllocator.RentUninitialized<T>(new[] { M, N });
                var inArr = (float[])(object)input.GetDataArray();
                var wArr = (float[])(object)weights.GetDataArray();
                var outArr = (float[])(object)result.GetDataArray();

                bool blasDone = false;
#if NET5_0_OR_GREATER
                // Tier 0: Raw function pointer — zero overhead calli (no delegate, no P/Invoke)
                if (BlasProvider.HasRawSgemm)
                {
                    unsafe
                    {
                        fixed (float* pIn = inArr, pW = wArr, pOut = outArr)
                            BlasProvider.SgemmRaw(M, N, K, pIn, K, pW, N, pOut, N);
                    }
                    blasDone = true;
                }
#endif
                if (!blasDone)
                {
                    // Tier 1: Native delegate dispatch
                    if (BlasProvider.HasNativeSgemm)
                    {
                        unsafe
                        {
                            fixed (float* pIn = inArr, pW = wArr, pOut = outArr)
                                BlasProvider.SgemmDirect(M, N, K, pIn, K, pW, N, pOut, N);
                        }
                    }
                    // Tier 2: MKL verified
                    else if (BlasProvider.IsMklVerified)
                    {
                        BlasProvider.MklSgemmZeroOffset(M, N, K, inArr, K, wArr, N, outArr, N);
                    }
                    // Tier 3: Any BLAS with validation
                    else if (!BlasProvider.TryGemm(M, N, K, inArr, 0, K, wArr, 0, N, outArr, 0, N))
                    {
                        Simd.SimdGemm.Sgemm(inArr.AsSpan(0, M * K), wArr.AsSpan(0, K * N), outArr.AsSpan(0, M * N), M, K, N);
                    }
                }

                // Fused bias + activation in one pass
                if (bias != null || activation != FusedActivationType.None)
                {
                    var bArr = bias != null ? (float[])(object)bias.GetDataArray() : null;
                    CpuFusedOperations.ApplyBiasActivationInPlace(outArr, bArr, M, N, activation);
                }

                return result;
            }

            // Ultra-fast path for double: zero-alloc arena + direct BLAS pointers
            if (typeof(T) == typeof(double))
            {
                var result = TensorAllocator.RentUninitialized<T>(new[] { M, N });
                var inArr = (double[])(object)input.GetDataArray();
                var wArr = (double[])(object)weights.GetDataArray();
                var outArr = (double[])(object)result.GetDataArray();

                bool dBlasDone = false;
#if NET5_0_OR_GREATER
                if (BlasProvider.HasRawDgemm)
                {
                    unsafe
                    {
                        fixed (double* pIn = inArr, pW = wArr, pOut = outArr)
                            BlasProvider.DgemmRaw(M, N, K, pIn, K, pW, N, pOut, N);
                    }
                    dBlasDone = true;
                }
#endif
                if (!dBlasDone && BlasProvider.HasNativeDgemm)
                {
                    unsafe
                    {
                        fixed (double* pIn = inArr, pW = wArr, pOut = outArr)
                            BlasProvider.DgemmDirect(M, N, K, pIn, K, pW, N, pOut, N);
                    }
                    dBlasDone = true;
                }
                if (!dBlasDone && BlasProvider.IsMklVerified)
                {
                    BlasProvider.MklDgemmZeroOffset(M, N, K, inArr, K, wArr, N, outArr, N);
                    dBlasDone = true;
                }
                if (!dBlasDone && !BlasProvider.TryGemm(M, N, K, inArr, 0, K, wArr, 0, N, outArr, 0, N))
                {
                    CpuFusedOperations.FusedGemmBiasActivation(inArr, wArr,
                        bias != null ? (double[])(object)bias.GetDataArray() : null,
                        outArr, M, N, K, activation);
                    return result;
                }

                if (bias != null || activation != FusedActivationType.None)
                {
                    var bArr = bias != null ? (double[])(object)bias.GetDataArray() : null;
                    CpuFusedOperations.ApplyBiasActivationInPlaceDouble(outArr, bArr, M, N, activation);
                }

                return result;
            }
        }

        // Fallback: sequential operations for other types or higher-rank tensors
        var fallbackResult = TensorMatMul(input, weights);

        if (bias != null)
        {
            fallbackResult = TensorBroadcastAdd(fallbackResult, bias);
        }

        // Apply activation in-place (fallbackResult is a fresh tensor)
        ApplyFusedActivationInPlace(fallbackResult, activation);

        return fallbackResult;
    }

    /// <inheritdoc/>
    public Tensor<T> FusedLinearBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> weights,
        Tensor<T> preActivation,
        FusedActivationType activation,
        out Tensor<T> gradWeights,
        out Tensor<T>? gradBias)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (preActivation == null) throw new ArgumentNullException(nameof(preActivation));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Step 1: Compute activation gradient
        var gradActivation = ApplyFusedActivationBackward(gradOutput, preActivation, activation);

        // Step 2: Compute bias gradient (sum along batch dimension)
        if (gradActivation._shape.Length == 2)
        {
            int batchSize = gradActivation._shape[0];
            int outputSize = gradActivation._shape[1];
            var biasGrad = TensorAllocator.RentUninitialized<T>([outputSize]);

            for (int j = 0; j < outputSize; j++)
            {
                T sum = numOps.Zero;
                for (int i = 0; i < batchSize; i++)
                {
                    sum = numOps.Add(sum, gradActivation[i, j]);
                }
                biasGrad[j] = sum;
            }
            gradBias = biasGrad;
        }
        else
        {
            gradBias = null;
        }

        // Step 3: Compute weight gradient: input^T @ gradActivation
        var inputT = TensorTranspose(input);
        gradWeights = TensorMatMul(inputT, gradActivation);

        // Step 4: Compute input gradient: gradActivation @ weights^T
        var weightsT = TensorTranspose(weights);
        var gradInput = TensorMatMul(gradActivation, weightsT);

        return gradInput;
    }

    /// <inheritdoc/>
    public Tensor<T> FusedConv2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH,
        int strideW,
        int padH,
        int padW,
        int dilationH,
        int dilationW,
        FusedActivationType activation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));

        // CPU implementation: sequential operations
        // Step 1: Conv2D
        var result = Conv2D(input, kernel, new[] { strideH, strideW }, new[] { padH, padW }, new[] { dilationH, dilationW });

        // Step 2: Add bias if provided
        if (bias != null)
        {
            result = TensorBroadcastAdd(result, bias);
        }

        // Step 3: Apply activation in-place (result is a fresh tensor, no need to allocate another)
        ApplyFusedActivationInPlace(result, activation);

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> FusedConv3D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW,
        FusedActivationType activation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));

        // CPU implementation: sequential operations
        // Step 1: Conv3D
        var result = Conv3D(input, kernel, new[] { strideD, strideH, strideW }, new[] { padD, padH, padW }, new[] { dilationD, dilationH, dilationW });

        // Step 2: Add bias if provided (reshape to [1, outChannels, 1, 1, 1] for NCDHW broadcast)
        if (bias != null)
        {
            var biasExpanded = bias.Reshape(1, bias._shape[0], 1, 1, 1);
            result = TensorBroadcastAdd(result, biasExpanded);
        }

        // Step 3: Apply activation in-place (result is a fresh tensor)
        ApplyFusedActivationInPlace(result, activation);

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> FusedConvTranspose2D<T>(
        Tensor<T> input,
        Tensor<T> kernel,
        Tensor<T>? bias,
        int strideH, int strideW,
        int padH, int padW,
        int outputPadH, int outputPadW,
        FusedActivationType activation)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));

        // CPU implementation: sequential operations
        // Step 1: ConvTranspose2D
        var result = ConvTranspose2D(input, kernel, new[] { strideH, strideW }, new[] { padH, padW }, new[] { outputPadH, outputPadW });

        // Step 2: Add bias if provided (reshape to [1, outChannels, 1, 1] for NCHW broadcast)
        if (bias != null)
        {
            var biasExpanded = bias.Reshape(1, bias._shape[0], 1, 1);
            result = TensorBroadcastAdd(result, biasExpanded);
        }

        // Step 3: Apply activation in-place (result is a fresh tensor)
        ApplyFusedActivationInPlace(result, activation);

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> FusedBatchNorm<T>(
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
        out Tensor<T> saveVar)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gamma == null) throw new ArgumentNullException(nameof(gamma));
        if (beta == null) throw new ArgumentNullException(nameof(beta));
        if (runningMean == null) throw new ArgumentNullException(nameof(runningMean));
        if (runningVar == null) throw new ArgumentNullException(nameof(runningVar));

        // CPU implementation: sequential operations
        // Note: Running mean/var and momentum are handled externally in training mode
        // This CPU implementation just does the batch normalization
        var result = BatchNorm(input, gamma, beta, epsilon, out saveMean, out saveVar);

        // Step 2: Apply activation in-place (result is a fresh tensor)
        ApplyFusedActivationInPlace(result, activation);

        return result;
    }

    /// <summary>
    /// Applies the specified activation function to the tensor.
    /// </summary>
    private Tensor<T> ApplyFusedActivation<T>(Tensor<T> input, FusedActivationType activation)
    {
        var handler = ActivationRegistry.Get(activation);
        if (handler is null) return input; // None
        return handler.Apply(this, input);
    }

    /// <summary>
    /// Applies activation in-place on a tensor that was just allocated.
    /// Eliminates one tensor allocation compared to ApplyFusedActivation.
    /// Handlers with true in-place support (ReLU, Sigmoid) override ApplyInPlace.
    /// Others fall back to allocate + copy.
    /// </summary>
    private void ApplyFusedActivationInPlace<T>(Tensor<T> tensor, FusedActivationType activation)
    {
        var handler = ActivationRegistry.Get(activation);
        if (handler is null) return; // None
        handler.ApplyInPlace(this, tensor);
    }

    /// <summary>
    /// Applies activation through the engine's recorded activation methods (tape-aware).
    /// Uses ActivationRegistry dispatch — no switch statement, open/closed compliant.
    /// </summary>
    private Tensor<T> ApplyActivationRecorded<T>(Tensor<T> tensor, FusedActivationType activation)
    {
        var handler = ActivationRegistry.Get(activation);
        if (handler is null) return tensor;
        return handler.Apply(this, tensor);
    }

    /// <summary>
    /// Computes the backward pass for the specified activation function.
    /// </summary>
    private Tensor<T> ApplyFusedActivationBackward<T>(Tensor<T> gradOutput, Tensor<T> preActivation, FusedActivationType activation)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        return activation switch
        {
            FusedActivationType.None => gradOutput,
            FusedActivationType.ReLU => TensorMultiply(gradOutput, ReLUDerivative(preActivation)),
            FusedActivationType.GELU => TensorMultiply(gradOutput, GELUDerivative(preActivation)),
            FusedActivationType.Sigmoid => TensorMultiply(gradOutput, SigmoidDerivative(Sigmoid(preActivation))),
            FusedActivationType.Tanh => TensorMultiply(gradOutput, TanhDerivative(Tanh(preActivation))),
            FusedActivationType.LeakyReLU => TensorMultiply(gradOutput, LeakyReLUDerivative(preActivation, numOps.FromDouble(0.01))),
            FusedActivationType.Swish => TensorMultiply(gradOutput, SwishDerivative(preActivation)),
            FusedActivationType.Softmax => throw new NotSupportedException("Softmax backward requires special handling via cross-entropy"),
            _ => throw new ArgumentException($"Unknown activation type: {activation}")
        };
    }

    /// <summary>
    /// Computes the derivative of Swish activation: f'(x) = f(x) + sigmoid(x) * (1 - f(x))
    /// </summary>
    private Tensor<T> SwishDerivative<T>(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(input._shape);
        var src = input.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
        {
            double x = numOps.ToDouble(src[i]);
            double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
            double swish = x * sigmoid;
            dest[i] = numOps.FromDouble(swish + sigmoid * (1.0 - swish));
        }

        return result;
    }

    /// <summary>
    /// Computes the derivative of Leaky ReLU activation.
    /// </summary>
    private Tensor<T> LeakyReLUDerivative<T>(Tensor<T> input, T alpha)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(input._shape);
        var src = input.AsSpan();
        var dest = result.AsWritableSpan();

        for (int i = 0; i < src.Length; i++)
            dest[i] = numOps.ToDouble(src[i]) > 0 ? numOps.One : alpha;

        return result;
    }

    /// <summary>
    /// Computes the derivative of GELU activation.
    /// </summary>
    private Tensor<T> GELUDerivative<T>(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(input._shape);
        var src = input.AsSpan();
        var dest = result.AsWritableSpan();

        const double sqrtTwoOverPi = 0.7978845608028654; // sqrt(2/pi)
        const double coeff = 0.044715;

        for (int i = 0; i < src.Length; i++)
        {
            double x = numOps.ToDouble(src[i]);
            double x3 = x * x * x;
            double inner = sqrtTwoOverPi * (x + coeff * x3);
            double tanh = Math.Tanh(inner);
            double sech2 = 1.0 - tanh * tanh;
            double innerDeriv = sqrtTwoOverPi * (1.0 + 3.0 * coeff * x * x);
            dest[i] = numOps.FromDouble(0.5 * (1.0 + tanh) + 0.5 * x * sech2 * innerDeriv);
        }

        return result;
    }

    #endregion

    #region Persistent Tensor Management

    /// <inheritdoc/>
    /// <remarks>
    /// On CPU, this is a no-op. Persistent tensor management only provides benefits
    /// on GPU where data transfer between host and device is expensive.
    /// </remarks>
    public void RegisterPersistentTensor<T>(Tensor<T> tensor, PersistentTensorRole role)
    {
        // No-op on CPU: persistence only benefits GPU by avoiding repeated transfers
        // The tensor is already in CPU memory, so there's nothing to cache
    }

    /// <inheritdoc/>
    /// <remarks>
    /// On CPU, this is a no-op. See <see cref="RegisterPersistentTensor{T}"/>.
    /// </remarks>
    public void UnregisterPersistentTensor<T>(Tensor<T> tensor)
    {
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        // No-op on CPU
    }

    /// <inheritdoc/>
    /// <remarks>
    /// On CPU, this is a no-op. See <see cref="RegisterPersistentTensor{T}"/>.
    /// </remarks>
    public void InvalidatePersistentTensor<T>(Tensor<T> tensor)
    {
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        // No-op on CPU
    }

    #endregion

    #region FFT and Signal Processing

    /// <inheritdoc/>
    public Tensor<T> RFFT<T>(Tensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        var numOps = MathHelper.GetNumericOperations<T>();
        int n = input._shape[^1]; // Last dimension is the signal length

        // Pad to next power of 2 if needed
        int nFft = NextPowerOf2(n);

        // Output has (nFft/2 + 1) complex values, stored as interleaved real/imag
        int numFreqs = nFft / 2 + 1;

        // Compute shape for output (last dim becomes 2 * numFreqs for interleaved complex)
        var outputShape = input.Shape.ToArray();
        outputShape[^1] = numFreqs * 2; // Interleaved real/imag
        var result = TensorAllocator.RentUninitialized<T>(outputShape);
        var inputData = input.GetFlattenedData();
        var resultData = result.GetDataArray();

        // Handle batched input
        int batchSize = input.Length / n;

        Parallel.For(0, batchSize, batchIdx =>
        {
            // Extract signal for this batch
            var signal = new Vector<T>(nFft);
            int inputOffset = batchIdx * n;
            for (int i = 0; i < n; i++)
                signal[i] = inputData[inputOffset + i];
            for (int i = n; i < nFft; i++)
                signal[i] = numOps.Zero;

            // Compute FFT using Cooley-Tukey algorithm
            var (realOut, imagOut) = FFTCore<T>(signal, inverse: false);

            // Copy only positive frequencies (0 to Nyquist)
            int outputOffset = batchIdx * numFreqs * 2;
            for (int k = 0; k < numFreqs; k++)
            {
                resultData[outputOffset + k * 2] = realOut[k];
                resultData[outputOffset + k * 2 + 1] = imagOut[k];
            }
        });

        DifferentiableOps.RecordUnary("RFFT", result, input, static (gradOutput, inputs, output, savedState, engine, grads) =>
        {
            // RFFT backward is IRFFT
            var nFftSaved = (int)savedState[0];
            var grad = engine.IRFFT(gradOutput, nFftSaved);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
        }, new object[] { n });
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> IRFFT<T>(Tensor<T> input, int outputLength)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();

        var numOps = MathHelper.GetNumericOperations<T>();
        int numFreqs = input._shape[^1] / 2; // Interleaved real/imag
        int nFft = (numFreqs - 1) * 2;

        // Output shape
        var outputShape = input.Shape.ToArray();
        outputShape[^1] = outputLength;
        var result = TensorAllocator.RentUninitialized<T>(outputShape);
        var inputData = input.GetFlattenedData();
        var resultData = result.GetDataArray();

        // Handle batched input
        int batchSize = input.Length / (numFreqs * 2);

        Parallel.For(0, batchSize, batchIdx =>
        {
            // Reconstruct full spectrum using conjugate symmetry
            var realIn = new Vector<T>(nFft);
            var imagIn = new Vector<T>(nFft);
            int inputOffset = batchIdx * numFreqs * 2;

            // Copy positive frequencies
            for (int k = 0; k < numFreqs; k++)
            {
                realIn[k] = inputData[inputOffset + k * 2];
                imagIn[k] = inputData[inputOffset + k * 2 + 1];
            }

            // Conjugate symmetry for negative frequencies
            for (int k = 1; k < numFreqs - 1; k++)
            {
                realIn[nFft - k] = realIn[k];
                imagIn[nFft - k] = numOps.Negate(imagIn[k]);
            }

            // Compute inverse FFT
            var (realOut, _) = FFTCore<T>(realIn, imagIn, inverse: true);

            // Copy result with normalization
            int outputOffset = batchIdx * outputLength;
            T scale = numOps.FromDouble(1.0 / nFft);
            for (int i = 0; i < outputLength; i++)
            {
                resultData[outputOffset + i] = numOps.Multiply(realOut[i], scale);
            }
        });

        DifferentiableOps.RecordUnary("IRFFT", result, input, static (gradOutput, inputs, output, savedState, engine, grads) =>
        {
            // IRFFT backward is RFFT
            var grad = engine.RFFT(gradOutput);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, engine);
        }, Array.Empty<object>());
        return result;
    }

    /// <inheritdoc/>
    public void FFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (inputReal == null) throw new ArgumentNullException(nameof(inputReal));
        if (inputImag == null) throw new ArgumentNullException(nameof(inputImag));
        if (!inputReal._shape.SequenceEqual(inputImag._shape))
            throw new ArgumentException("Input real and imaginary parts must have the same shape");

        int n = inputReal._shape[^1];

        // Create local variables to use in lambda (out params can't be captured)
        var outReal = TensorAllocator.RentUninitialized<T>(inputReal._shape);
        var outImag = TensorAllocator.RentUninitialized<T>(inputImag._shape);
        var inputRealData = inputReal.GetFlattenedData();
        var inputImagData = inputImag.GetFlattenedData();
        var outRealData = outReal.GetDataArray();
        var outImagData = outImag.GetDataArray();

        int batchSize = inputReal.Length / n;

        Parallel.For(0, batchSize, batchIdx =>
        {
            var realIn = new Vector<T>(n);
            var imagIn = new Vector<T>(n);
            int offset = batchIdx * n;

            for (int i = 0; i < n; i++)
            {
                realIn[i] = inputRealData[offset + i];
                imagIn[i] = inputImagData[offset + i];
            }

            var (realOut, imagOut) = FFTCore<T>(realIn, imagIn, inverse: false);

            for (int i = 0; i < n; i++)
            {
                outRealData[offset + i] = realOut[i];
                outImagData[offset + i] = imagOut[i];
            }
        });

        outputReal = outReal;
        outputImag = outImag;
    }

    /// <inheritdoc/>
    public void IFFT<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (inputReal == null) throw new ArgumentNullException(nameof(inputReal));
        if (inputImag == null) throw new ArgumentNullException(nameof(inputImag));
        if (!inputReal.IsContiguous) inputReal = inputReal.Contiguous();
        if (!inputImag.IsContiguous) inputImag = inputImag.Contiguous();
        if (!inputReal._shape.SequenceEqual(inputImag._shape))
            throw new ArgumentException("Input real and imaginary parts must have the same shape");

        var numOps = MathHelper.GetNumericOperations<T>();
        int n = inputReal._shape[^1];

        // Create local variables to use in lambda (out params can't be captured)
        var outReal = TensorAllocator.RentUninitialized<T>(inputReal._shape);
        var outImag = TensorAllocator.RentUninitialized<T>(inputImag._shape);
        var inputRealData = inputReal.GetFlattenedData();
        var inputImagData = inputImag.GetFlattenedData();
        var outRealData = outReal.GetDataArray();
        var outImagData = outImag.GetDataArray();

        int batchSize = inputReal.Length / n;
        T scale = numOps.FromDouble(1.0 / n);

        Parallel.For(0, batchSize, batchIdx =>
        {
            var realIn = new Vector<T>(n);
            var imagIn = new Vector<T>(n);
            int offset = batchIdx * n;

            for (int i = 0; i < n; i++)
            {
                realIn[i] = inputRealData[offset + i];
                imagIn[i] = inputImagData[offset + i];
            }

            var (realOut, imagOut) = FFTCore<T>(realIn, imagIn, inverse: true);

            for (int i = 0; i < n; i++)
            {
                outRealData[offset + i] = numOps.Multiply(realOut[i], scale);
                outImagData[offset + i] = numOps.Multiply(imagOut[i], scale);
            }
        });

        outputReal = outReal;
        outputImag = outImag;
    }

    /// <inheritdoc/>
    public void FFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (inputReal == null) throw new ArgumentNullException(nameof(inputReal));
        if (inputImag == null) throw new ArgumentNullException(nameof(inputImag));
        if (!inputReal.IsContiguous) inputReal = inputReal.Contiguous();
        if (!inputImag.IsContiguous) inputImag = inputImag.Contiguous();
        if (inputReal._shape.Length < 2)
            throw new ArgumentException("Input must be at least 2D");

        // FFT along last dimension (columns)
        FFT(inputReal, inputImag, out var tempReal, out var tempImag);

        // Transpose, FFT along rows, transpose back
        int height = inputReal._shape[^2];
        int width = inputReal._shape[^1];

        // Create local variables to use in lambda (out params can't be captured)
        var outReal = TensorAllocator.RentUninitialized<T>(inputReal._shape);
        var outImag = TensorAllocator.RentUninitialized<T>(inputImag._shape);
        var tempRealData = tempReal.GetDataArray();
        var tempImagData = tempImag.GetDataArray();
        var outRealData = outReal.GetDataArray();
        var outImagData = outImag.GetDataArray();

        // FFT along rows (second-to-last dimension)
        int batchSize = inputReal.Length / (height * width);

        Parallel.For(0, batchSize, batchIdx =>
        {
            // Process each column
            for (int col = 0; col < width; col++)
            {
                var realIn = new Vector<T>(height);
                var imagIn = new Vector<T>(height);

                for (int row = 0; row < height; row++)
                {
                    int idx = batchIdx * height * width + row * width + col;
                    realIn[row] = tempRealData[idx];
                    imagIn[row] = tempImagData[idx];
                }

                var (realOut, imagOut) = FFTCore<T>(realIn, imagIn, inverse: false);

                for (int row = 0; row < height; row++)
                {
                    int idx = batchIdx * height * width + row * width + col;
                    outRealData[idx] = realOut[row];
                    outImagData[idx] = imagOut[row];
                }
            }
        });

        outputReal = outReal;
        outputImag = outImag;
    }

    /// <inheritdoc/>
    public void IFFT2D<T>(Tensor<T> inputReal, Tensor<T> inputImag, out Tensor<T> outputReal, out Tensor<T> outputImag)
    {
        if (inputReal == null) throw new ArgumentNullException(nameof(inputReal));
        if (inputImag == null) throw new ArgumentNullException(nameof(inputImag));
        if (inputReal._shape.Length < 2)
            throw new ArgumentException("Input must be at least 2D");

        var numOps = MathHelper.GetNumericOperations<T>();
        int height = inputReal._shape[^2];
        int width = inputReal._shape[^1];

        var tempReal = TensorAllocator.RentUninitialized<T>(inputReal._shape);
        var tempImag = TensorAllocator.RentUninitialized<T>(inputImag._shape);

        // Create local variables to use in lambda (out params can't be captured)
        var outReal = TensorAllocator.RentUninitialized<T>(inputReal._shape);
        var outImag = TensorAllocator.RentUninitialized<T>(inputImag._shape);
        var inputRealData = inputReal.GetFlattenedData();
        var inputImagData = inputImag.GetFlattenedData();
        var tempRealData = tempReal.GetDataArray();
        var tempImagData = tempImag.GetDataArray();
        var outRealData = outReal.GetDataArray();
        var outImagData = outImag.GetDataArray();

        int batchSize = inputReal.Length / (height * width);
        T scale = numOps.FromDouble(1.0 / (height * width));

        Parallel.For(0, batchSize, batchIdx =>
        {
            // IFFT along columns first
            for (int col = 0; col < width; col++)
            {
                var realIn = new Vector<T>(height);
                var imagIn = new Vector<T>(height);

                for (int row = 0; row < height; row++)
                {
                    int idx = batchIdx * height * width + row * width + col;
                    realIn[row] = inputRealData[idx];
                    imagIn[row] = inputImagData[idx];
                }

                var (realOut, imagOut) = FFTCore<T>(realIn, imagIn, inverse: true);

                for (int row = 0; row < height; row++)
                {
                    int idx = batchIdx * height * width + row * width + col;
                    tempRealData[idx] = realOut[row];
                    tempImagData[idx] = imagOut[row];
                }
            }

            // IFFT along rows
            for (int row = 0; row < height; row++)
            {
                var realIn = new Vector<T>(width);
                var imagIn = new Vector<T>(width);
                int rowOffset = batchIdx * height * width + row * width;

                for (int col = 0; col < width; col++)
                {
                    realIn[col] = tempRealData[rowOffset + col];
                    imagIn[col] = tempImagData[rowOffset + col];
                }

                var (realOut, imagOut) = FFTCore<T>(realIn, imagIn, inverse: true);

                for (int col = 0; col < width; col++)
                {
                    outRealData[rowOffset + col] = numOps.Multiply(realOut[col], scale);
                    outImagData[rowOffset + col] = numOps.Multiply(imagOut[col], scale);
                }
            }
        });

        outputReal = outReal;
        outputImag = outImag;
    }

    /// <inheritdoc/>
    public void STFT<T>(
        Tensor<T> input,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        out Tensor<T> magnitudeOut,
        out Tensor<T> phaseOut)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (window == null) throw new ArgumentNullException(nameof(window));
        if (window.Length != nFft)
            throw new ArgumentException($"Window length {window.Length} must equal nFft {nFft}");

        var numOps = MathHelper.GetNumericOperations<T>();
        int signalLength = input._shape[^1];

        // Apply centering (reflection padding)
        Tensor<T> paddedInput;
        if (center)
        {
            int padAmount = nFft / 2;
            var paddedShape = input.Shape.ToArray();
            paddedShape[^1] = signalLength + 2 * padAmount;
            paddedInput = TensorAllocator.RentUninitialized<T>(paddedShape);
            var inputData = input.GetDataArray();
            var paddedData = paddedInput.GetDataArray();

            int batchSize = input.Length / signalLength;
            for (int b = 0; b < batchSize; b++)
            {
                int inputOffset = b * signalLength;
                int outputOffset = b * paddedShape[^1];

                // Reflect padding at start
                for (int i = 0; i < padAmount; i++)
                    paddedData[outputOffset + i] = inputData[inputOffset + padAmount - i];

                // Copy original
                Array.Copy(inputData, inputOffset, paddedData, outputOffset + padAmount, signalLength);

                // Reflect padding at end
                for (int i = 0; i < padAmount; i++)
                    paddedData[outputOffset + padAmount + signalLength + i] =
                        inputData[inputOffset + signalLength - 2 - i];
            }
            signalLength = paddedShape[^1];
        }
        else
        {
            paddedInput = input;
        }

        // Calculate number of frames
        int numFrames = (signalLength - nFft) / hopLength + 1;
        int numFreqs = nFft / 2 + 1;

        // Output shapes
        var outputShape = input.Shape.ToArray();
        outputShape[^1] = numFrames;
        var newShape = new int[outputShape.Length + 1];
        Array.Copy(outputShape, newShape, outputShape.Length - 1);
        newShape[^2] = numFreqs;
        newShape[^1] = numFrames;

        // Create local variables to use in lambda (out params can't be captured)
        var magOut = TensorAllocator.RentUninitialized<T>(newShape);
        var phOut = TensorAllocator.RentUninitialized<T>(newShape);
        var paddedInputData = paddedInput.GetDataArray();
        var windowData = window.GetDataArray();
        var magOutData = magOut.GetDataArray();
        var phOutData = phOut.GetDataArray();

        int batchSizeStft = paddedInput.Length / signalLength;

        Parallel.For(0, batchSizeStft, batchIdx =>
        {
            for (int frame = 0; frame < numFrames; frame++)
            {
                int start = frame * hopLength;

                // Extract windowed frame
                var frameData = new Vector<T>(nFft);
                int inputOffset = batchIdx * signalLength;
                for (int i = 0; i < nFft; i++)
                {
                    frameData[i] = numOps.Multiply(paddedInputData[inputOffset + start + i], windowData[i]);
                }

                // Compute FFT
                var (realOut, imagOut) = FFTCore<T>(frameData, inverse: false);

                // Compute magnitude and phase for positive frequencies
                int outputOffset = batchIdx * numFreqs * numFrames;
                for (int k = 0; k < numFreqs; k++)
                {
                    double re = numOps.ToDouble(realOut[k]);
                    double im = numOps.ToDouble(imagOut[k]);
                    double mag = Math.Sqrt(re * re + im * im);
                    double ph = Math.Atan2(im, re);

                    magOutData[outputOffset + k * numFrames + frame] = numOps.FromDouble(mag);
                    phOutData[outputOffset + k * numFrames + frame] = numOps.FromDouble(ph);
                }
            }
        });

        magnitudeOut = magOut;
        phaseOut = phOut;
    }

    /// <inheritdoc/>
    public Tensor<T> ISTFT<T>(
        Tensor<T> magnitude,
        Tensor<T> phase,
        int nFft,
        int hopLength,
        Tensor<T> window,
        bool center,
        int? length = null)
    {
        if (magnitude == null) throw new ArgumentNullException(nameof(magnitude));
        if (phase == null) throw new ArgumentNullException(nameof(phase));
        if (window == null) throw new ArgumentNullException(nameof(window));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numFreqs = magnitude._shape[^2];
        int numFrames = magnitude._shape[^1];

        // Calculate output length
        int outputLength = length ?? (numFrames - 1) * hopLength + nFft;
        if (center)
            outputLength -= nFft; // Remove padding

        // Output shape
        var outputShape = magnitude._shape.Take(magnitude._shape.Length - 2).ToArray();
        outputShape = outputShape.Length > 0 ? outputShape.Append(outputLength).ToArray() : new[] { outputLength };
        var result = TensorAllocator.RentUninitialized<T>(outputShape);
        var windowSum = TensorAllocator.RentUninitialized<T>(outputShape);
        var magData = magnitude.GetDataArray();
        var phaseData = phase.GetDataArray();
        var windowData = window.GetDataArray();
        var resultData = result.GetDataArray();
        var windowSumData = windowSum.GetDataArray();

        int batchSize = magnitude.Length / (numFreqs * numFrames);

        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++)
        {
            int magOffset = batchIdx * numFreqs * numFrames;
            int outputOffset = batchIdx * outputLength;

            for (int frame = 0; frame < numFrames; frame++)
            {
                // Reconstruct complex spectrum from magnitude and phase
                var realIn = new Vector<T>(nFft);
                var imagIn = new Vector<T>(nFft);

                // Positive frequencies
                for (int k = 0; k < numFreqs; k++)
                {
                    double mag = numOps.ToDouble(magData[magOffset + k * numFrames + frame]);
                    double ph = numOps.ToDouble(phaseData[magOffset + k * numFrames + frame]);
                    realIn[k] = numOps.FromDouble(mag * Math.Cos(ph));
                    imagIn[k] = numOps.FromDouble(mag * Math.Sin(ph));
                }

                // Reconstruct negative frequencies using conjugate symmetry
                for (int k = 1; k < numFreqs - 1; k++)
                {
                    realIn[nFft - k] = realIn[k];
                    imagIn[nFft - k] = numOps.Negate(imagIn[k]);
                }

                // Inverse FFT
                var (realOut, _) = FFTCore<T>(realIn, imagIn, inverse: true);
                T scale = numOps.FromDouble(1.0 / nFft);

                // Overlap-add
                int writeStart = center ? Math.Max(0, frame * hopLength - nFft / 2) : frame * hopLength;

                for (int i = 0; i < nFft; i++)
                {
                    int outIdx = writeStart + i;
                    if (outIdx >= 0 && outIdx < outputLength)
                    {
                        int resultIdx = outputOffset + outIdx;
                        T sample = numOps.Multiply(numOps.Multiply(realOut[i], scale), windowData[i]);
                        resultData[resultIdx] = numOps.Add(resultData[resultIdx], sample);

                        T winSquared = numOps.Multiply(windowData[i], windowData[i]);
                        windowSumData[resultIdx] = numOps.Add(windowSumData[resultIdx], winSquared);
                    }
                }
            }

            // Normalize by window sum
            for (int i = 0; i < outputLength; i++)
            {
                int idx = outputOffset + i;
                double winSumD = numOps.ToDouble(windowSumData[idx]);
                if (winSumD > 1e-8)
                {
                    resultData[idx] = numOps.Divide(resultData[idx], windowSumData[idx]);
                }
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> MelSpectrogram<T>(
        Tensor<T> input,
        int sampleRate,
        int nFft,
        int hopLength,
        int nMels,
        T fMin,
        T fMax,
        Tensor<T> window,
        bool powerToDb = true)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (window == null) throw new ArgumentNullException(nameof(window));

        var numOps = MathHelper.GetNumericOperations<T>();

        // Compute STFT
        STFT(input, nFft, hopLength, window, center: true, out var magnitude, out _);

        // Convert to power spectrum (magnitude squared)
        var powerSpec = TensorMultiply(magnitude, magnitude);

        // Create mel filterbank
        var melFilterbank = CreateMelFilterbank<T>(nMels, nFft, sampleRate, fMin, fMax);

        // Apply filterbank: [nMels, numFreqs] @ [numFreqs, numFrames] -> [nMels, numFrames]
        int numFreqs = nFft / 2 + 1;
        int numFrames = magnitude._shape[^1];
        int batchSize = magnitude.Length / (numFreqs * numFrames);

        var melShape = magnitude.Shape.ToArray();
        melShape[^2] = nMels;
        var melSpec = TensorAllocator.RentUninitialized<T>(melShape);
        var melFilterData = melFilterbank.GetDataArray();
        var powerSpecData = powerSpec.GetDataArray();
        var melSpecData = melSpec.GetDataArray();

        for (int batchIdx = 0; batchIdx < batchSize; batchIdx++)
        {
            int inputOffset = batchIdx * numFreqs * numFrames;
            int outputOffset = batchIdx * nMels * numFrames;

            for (int m = 0; m < nMels; m++)
            {
                for (int t = 0; t < numFrames; t++)
                {
                    T sum = numOps.Zero;
                    for (int f = 0; f < numFreqs; f++)
                    {
                        sum = numOps.Add(sum, numOps.Multiply(
                            melFilterData[m * numFreqs + f],
                            powerSpecData[inputOffset + f * numFrames + t]));
                    }
                    melSpecData[outputOffset + m * numFrames + t] = sum;
                }
            }
        }

        // Convert to dB scale if requested
        if (powerToDb)
        {
            double minDbD = -80.0;
            double epsilonD = 1e-10;

            for (int i = 0; i < melSpec.Length; i++)
            {
                double valD = numOps.ToDouble(melSpecData[i]);
                valD = Math.Max(valD, epsilonD);
                double db = 10.0 * Math.Log10(valD);
                db = Math.Max(db, minDbD);
                melSpecData[i] = numOps.FromDouble(db);
            }
        }

        return melSpec;
    }

    /// <inheritdoc/>
    public Tensor<T> GriffinLim<T>(
        Tensor<T> magnitude,
        int nFft,
        int hopLength,
        Tensor<T> window,
        int iterations = 60,
        double momentum = 0.99,
        int? length = null)
    {
        if (magnitude == null) throw new ArgumentNullException(nameof(magnitude));
        if (window == null) throw new ArgumentNullException(nameof(window));

        var numOps = MathHelper.GetNumericOperations<T>();
        int numFreqs = magnitude._shape[^2];
        int numFrames = magnitude._shape[^1];

        // Initialize with random phase
        var random = RandomHelper.ThreadSafeRandom;
        var phase = TensorAllocator.RentUninitialized<T>(magnitude._shape);
        var phaseData = phase.GetDataArray();
        for (int i = 0; i < phase.Length; i++)
        {
            double randomPhase = random.NextDouble() * 2.0 * Math.PI - Math.PI;
            phaseData[i] = numOps.FromDouble(randomPhase);
        }

        Tensor<T>? previousPhase = null;

        for (int iter = 0; iter < iterations; iter++)
        {
            // Reconstruct signal from magnitude and estimated phase
            var reconstructed = ISTFT(magnitude, phase, nFft, hopLength, window, center: true, length: length);

            // Re-compute STFT to get new phase estimate
            STFT(reconstructed, nFft, hopLength, window, center: true, out _, out var newPhase);

            // Apply momentum for faster convergence
            if (previousPhase != null && momentum > 0)
            {
                var newPhaseData = newPhase.GetDataArray();
                var prevPhaseData = previousPhase.GetDataArray();
                phaseData = phase.GetDataArray();
                for (int i = 0; i < phase.Length; i++)
                {
                    double currPh = numOps.ToDouble(newPhaseData[i]);
                    double prevPh = numOps.ToDouble(prevPhaseData[i]);
                    double diff = currPh - prevPh;
                    // Wrap to [-pi, pi]
                    while (diff > Math.PI) diff -= 2 * Math.PI;
                    while (diff < -Math.PI) diff += 2 * Math.PI;
                    double accelerated = prevPh + diff * (1 + momentum);
                    phaseData[i] = numOps.FromDouble(accelerated);
                }
            }
            else
            {
                TensorCopy(newPhase, phase);
            }

            previousPhase = newPhase;
        }

        // Final reconstruction
        return ISTFT(magnitude, phase, nFft, hopLength, window, center: true, length: length);
    }

    /// <inheritdoc/>
    public Tensor<T> CreateMelFilterbank<T>(int nMels, int nFft, int sampleRate, T fMin, T fMax)
    {
        if (nMels <= 0) throw new ArgumentException("nMels must be positive", nameof(nMels));
        if (nFft <= 0) throw new ArgumentException("nFft must be positive", nameof(nFft));
        if (sampleRate <= 0) throw new ArgumentException("sampleRate must be positive", nameof(sampleRate));

        var numOps = MathHelper.GetNumericOperations<T>();
        double fMinHz = numOps.ToDouble(fMin);
        double fMaxHz = numOps.ToDouble(fMax);

        int numFreqs = nFft / 2 + 1;
        var filterbank = TensorAllocator.Rent<T>([nMels, numFreqs]);
        filterbank.Fill(numOps.Zero);

        // Convert Hz to Mel scale
        double melMin = HzToMel(fMinHz);
        double melMax = HzToMel(fMaxHz);

        // Create mel points evenly spaced in mel scale
        var melPoints = new double[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
        {
            melPoints[i] = melMin + i * (melMax - melMin) / (nMels + 1);
        }

        // Convert back to Hz and then to FFT bin indices
        var binIndices = new int[nMels + 2];
        for (int i = 0; i < nMels + 2; i++)
        {
            double hz = MelToHz(melPoints[i]);
            binIndices[i] = (int)Math.Floor((nFft + 1) * hz / sampleRate);
            binIndices[i] = Math.Max(0, Math.Min(numFreqs - 1, binIndices[i]));
        }

        // Create triangular filters
        for (int m = 0; m < nMels; m++)
        {
            int fStart = binIndices[m];
            int fCenter = binIndices[m + 1];
            int fEnd = binIndices[m + 2];

            // Rising edge
            for (int f = fStart; f < fCenter; f++)
            {
                if (fCenter != fStart)
                {
                    double weight = (double)(f - fStart) / (fCenter - fStart);
                    filterbank[m, f] = numOps.FromDouble(weight);
                }
            }

            // Falling edge
            for (int f = fCenter; f < fEnd; f++)
            {
                if (fEnd != fCenter)
                {
                    double weight = (double)(fEnd - f) / (fEnd - fCenter);
                    filterbank[m, f] = numOps.FromDouble(weight);
                }
            }
        }

        return filterbank;
    }

    /// <inheritdoc/>
    public Tensor<T> CreateWindow<T>(string windowType, int length)
    {
        if (length <= 0) throw new ArgumentException("Length must be positive", nameof(length));

        var numOps = MathHelper.GetNumericOperations<T>();
        var window = TensorAllocator.RentUninitialized<T>([length]);

        switch (windowType.ToLowerInvariant())
        {
            case "hann":
            case "hanning":
                for (int i = 0; i < length; i++)
                {
                    double val = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (length - 1)));
                    window[i] = numOps.FromDouble(val);
                }
                break;

            case "hamming":
                for (int i = 0; i < length; i++)
                {
                    double val = 0.54 - 0.46 * Math.Cos(2.0 * Math.PI * i / (length - 1));
                    window[i] = numOps.FromDouble(val);
                }
                break;

            case "blackman":
                for (int i = 0; i < length; i++)
                {
                    double val = 0.42 - 0.5 * Math.Cos(2.0 * Math.PI * i / (length - 1))
                                      + 0.08 * Math.Cos(4.0 * Math.PI * i / (length - 1));
                    window[i] = numOps.FromDouble(val);
                }
                break;

            case "bartlett":
            case "triangular":
                for (int i = 0; i < length; i++)
                {
                    double val = 1.0 - Math.Abs((2.0 * i - (length - 1)) / (length - 1));
                    window[i] = numOps.FromDouble(val);
                }
                break;

            case "rectangular":
            case "boxcar":
                for (int i = 0; i < length; i++)
                {
                    window[i] = numOps.One;
                }
                break;

            default:
                throw new ArgumentException($"Unknown window type: {windowType}. Supported: hann, hamming, blackman, bartlett, rectangular");
        }

        return window;
    }

    #region FFT Helper Methods

    /// <summary>
    /// Core FFT computation using Cooley-Tukey algorithm.
    /// </summary>
    private static (Vector<T> real, Vector<T> imag) FFTCore<T>(Vector<T> realInput, bool inverse)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = realInput.Length;
        var imagInput = new Vector<T>(n);
        for (int i = 0; i < n; i++)
            imagInput[i] = numOps.Zero;
        return FFTCore(realInput, imagInput, inverse);
    }

    /// <summary>
    /// Core FFT computation using Cooley-Tukey algorithm with complex input.
    /// </summary>
    private static (Vector<T> real, Vector<T> imag) FFTCore<T>(Vector<T> realInput, Vector<T> imagInput, bool inverse)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = realInput.Length;
        if (n <= 1)
            return (new Vector<T>(realInput.GetDataArray()), new Vector<T>(imagInput.GetDataArray()));

        // Pad to power of 2 if needed
        int nPadded = NextPowerOf2(n);
        Vector<T> realWork, imagWork;
        if (nPadded != n)
        {
            realWork = new Vector<T>(nPadded);
            imagWork = new Vector<T>(nPadded);
            for (int i = 0; i < n; i++)
            {
                realWork[i] = realInput[i];
                imagWork[i] = imagInput[i];
            }
            for (int i = n; i < nPadded; i++)
            {
                realWork[i] = numOps.Zero;
                imagWork[i] = numOps.Zero;
            }
            n = nPadded;
        }
        else
        {
            realWork = new Vector<T>(realInput.GetDataArray());
            imagWork = new Vector<T>(imagInput.GetDataArray());
        }

        // Bit-reversal permutation
        var real = new Vector<T>(n);
        var imag = new Vector<T>(n);
        int bits = (int)MathHelper.Log2(n);

        for (int i = 0; i < n; i++)
        {
            int j = BitReverse(i, bits);
            real[j] = realWork[i];
            imag[j] = imagWork[i];
        }

        // Cooley-Tukey iterative FFT
        double angleSign = inverse ? 1.0 : -1.0;

        for (int size = 2; size <= n; size *= 2)
        {
            int halfSize = size / 2;
            double angleStep = angleSign * 2.0 * Math.PI / size;

            for (int i = 0; i < n; i += size)
            {
                for (int k = 0; k < halfSize; k++)
                {
                    double angle = angleStep * k;
                    double cos = Math.Cos(angle);
                    double sin = Math.Sin(angle);

                    int evenIdx = i + k;
                    int oddIdx = i + k + halfSize;

                    double tReal = numOps.ToDouble(real[oddIdx]) * cos - numOps.ToDouble(imag[oddIdx]) * sin;
                    double tImag = numOps.ToDouble(real[oddIdx]) * sin + numOps.ToDouble(imag[oddIdx]) * cos;

                    double evenReal = numOps.ToDouble(real[evenIdx]);
                    double evenImag = numOps.ToDouble(imag[evenIdx]);

                    real[evenIdx] = numOps.FromDouble(evenReal + tReal);
                    imag[evenIdx] = numOps.FromDouble(evenImag + tImag);
                    real[oddIdx] = numOps.FromDouble(evenReal - tReal);
                    imag[oddIdx] = numOps.FromDouble(evenImag - tImag);
                }
            }
        }

        return (real, imag);
    }

    /// <summary>
    /// Returns the smallest power of 2 >= n.
    /// </summary>
    private static int NextPowerOf2(int n)
    {
        if (n <= 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }

    /// <summary>
    /// Reverses the bits of an integer for FFT bit-reversal permutation.
    /// </summary>
    private static int BitReverse(int x, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }

    /// <summary>
    /// Converts frequency from Hz to Mel scale.
    /// </summary>
    private static double HzToMel(double hz)
    {
        return 2595.0 * Math.Log10(1.0 + hz / 700.0);
    }

    /// <summary>
    /// Converts frequency from Mel scale to Hz.
    /// </summary>
    private static double MelToHz(double mel)
    {
        return 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);
    }

    #endregion

    #endregion

    #region GPU-Accelerated Operations (CPU Fallback Implementations)

    /// <inheritdoc/>
    public virtual Tensor<T> Softplus<T>(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var data = input.GetFlattenedData();
        var result = new T[data.Length];
        int length = data.Length;

        if (data is float[] dF && result is float[] rF)
        {
            Parallel.For(0, length, i =>
            {
                float x = dF[i];
                rF[i] = x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
            });
        }
        else
        {
            for (int i = 0; i < length; i++)
            {
                double x = numOps.ToDouble(data[i]);
                result[i] = numOps.FromDouble(x > 20.0 ? x : Math.Log(1.0 + Math.Exp(x)));
            }
        }

        var softplusResult = TensorAllocator.Rent<T>(input._shape, new Vector<T>(result));
        DifferentiableOps.RecordUnary("Softplus", softplusResult, input,
            BackwardFunctions<T>.SoftplusBackward);
        return softplusResult;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> HardSwish<T>(Tensor<T> input)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        var data = input.GetFlattenedData();
        var result = new T[data.Length];
        int length = data.Length;

        if (data is float[] dF && result is float[] rF)
        {
            Parallel.For(0, length, i =>
            {
                float x = dF[i];
                float clip = MathF.Min(MathF.Max(x + 3f, 0f), 6f);
                rF[i] = x * clip / 6f;
            });
        }
        else
        {
            for (int i = 0; i < length; i++)
            {
                double x = numOps.ToDouble(data[i]);
                double clip = Math.Min(Math.Max(x + 3.0, 0.0), 6.0);
                result[i] = numOps.FromDouble(x * clip / 6.0);
            }
        }

        var hardSwishResult = TensorAllocator.Rent<T>(input._shape, new Vector<T>(result));
        DifferentiableOps.RecordUnary("HardSwish", hardSwishResult, input,
            BackwardFunctions<T>.HardSwishBackward);
        return hardSwishResult;
    }

    /// <inheritdoc/>
    public virtual unsafe Tensor<T> ReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gradOutput.Length != input.Length)
            throw new ArgumentException($"Shape mismatch: gradOutput length {gradOutput.Length} != input length {input.Length}");

        var numOps = MathHelper.GetNumericOperations<T>();
        if (!input.IsContiguous) input = input.Contiguous();
        int length = input.Length;

        // Fused scalar path: when gradOutput is a uniform-fill tensor (e.g., from MeanBackward),
        // every element is the same value. Use scalar kernel that reads only the input array,
        // eliminating 4MB of grad reads and the fill allocation for 1M-element tensors.
        if (typeof(T) == typeof(float) && gradOutput.UniformFillValue.HasValue)
        {
            float scale = (float)gradOutput.UniformFillValue.Value;
            var resultTensor = TensorAllocator.RentUninitialized<T>(input._shape);
            var iArr = (float[])(object)input.GetDataArray();
            var rArr = (float[])(object)resultTensor.GetDataArray();
            fixed (float* pI = iArr, pR = rArr)
                SimdKernels.ReluBackwardScalarUnsafe(scale, pI, pR, length);
            return resultTensor;
        }

        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();

        // Allocating path: new output buffer
        var resultTensor2 = TensorAllocator.RentUninitialized<T>(input._shape);

        if (typeof(T) == typeof(float))
        {
            var gArr = (float[])(object)gradOutput.GetDataArray();
            var iArr = (float[])(object)input.GetDataArray();
            var rArr = (float[])(object)resultTensor2.GetDataArray();
            fixed (float* pG = gArr, pI = iArr, pR = rArr)
                SimdKernels.ReluBackwardUnsafe(pG, pI, pR, length);
        }
        else if (typeof(T) == typeof(double))
        {
            // Double SIMD backward (AVX2: 4 doubles per vector)
            var gMem = AsDoubleMemory(gradOutput.Data);
            var iMem = AsDoubleMemory(input.Data);
            var rMem = AsDoubleMemory(resultTensor2.Data);
            using var pinG = gMem.Pin();
            using var pinI = iMem.Pin();
            using var pinR = rMem.Pin();
            unsafe
            {
                SimdKernels.ReluBackwardDouble(
                    (double*)pinG.Pointer, (double*)pinI.Pointer,
                    (double*)pinR.Pointer, length);
            }
        }
        else
        {
            var gradData = gradOutput.GetDataArray();
            var inputData = input.GetDataArray();
            var resultData = resultTensor2.GetDataArray();
            for (int i = 0; i < length; i++)
            {
                double inputVal = numOps.ToDouble(inputData[i]);
                resultData[i] = inputVal > 0 ? gradData[i] : numOps.Zero;
            }
        }

        return resultTensor2;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> SigmoidBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var gradData = gradOutput.GetFlattenedData();
        var outData = output.GetDataArray();
        int length = gradOutput.Length;
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!output.IsContiguous) output = output.Contiguous();

        // Float fast path: SIMD grad * sigmoid * (1 - sigmoid)
        if (gradData is float[] gF && outData is float[] oF)
        {
#if NET5_0_OR_GREATER
            var resultArr = GC.AllocateUninitializedArray<float>(length);
#else
            var resultArr = new float[length];
#endif
            SigmoidBackwardFloat(gF, oF, resultArr);
            return (Tensor<T>)(object)TensorAllocator.Rent<T>(gradOutput._shape, (Vector<T>)(object)Vector<float>.FromMemory(resultArr));
        }

        // Double SIMD path
        if (typeof(T) == typeof(double))
        {
            var resultTensor = TensorAllocator.RentUninitialized<T>(gradOutput._shape);
            var gArr = (double[])(object)gradOutput.GetFlattenedData();
            var oArr = (double[])(object)output.GetDataArray();
            var rArr = (double[])(object)resultTensor.GetDataArray();
            unsafe
            {
                fixed (double* pG = gArr, pO = oArr, pR = rArr)
                    SimdKernels.SigmoidBackwardDouble(pG, pO, pR, length);
            }
            return resultTensor;
        }

        // Generic scalar fallback
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            double s = numOps.ToDouble(outData[i]);
            double grad = numOps.ToDouble(gradData[i]);
            result[i] = numOps.FromDouble(grad * s * (1.0 - s));
        }

        return TensorAllocator.Rent<T>(gradOutput._shape, new Vector<T>(result));
    }

    private static unsafe void SigmoidBackwardFloat(float[] grad, float[] sigmoid, float[] result)
    {
        int length = grad.Length;
        int i = 0;
#if NET5_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && length >= 32)
        {
            fixed (float* gp = grad, sp = sigmoid, rp = result)
            {
                var one = System.Runtime.Intrinsics.Vector256.Create(1.0f);
                int simdLen = length & ~31;
                for (; i < simdLen; i += 32)
                {
                    var g0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(gp + i);
                    var s0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(sp + i);
                    System.Runtime.Intrinsics.X86.Avx.Store(rp + i, System.Runtime.Intrinsics.X86.Avx.Multiply(System.Runtime.Intrinsics.X86.Avx.Multiply(g0, s0), System.Runtime.Intrinsics.X86.Avx.Subtract(one, s0)));

                    var g1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(gp + i + 8);
                    var s1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(sp + i + 8);
                    System.Runtime.Intrinsics.X86.Avx.Store(rp + i + 8, System.Runtime.Intrinsics.X86.Avx.Multiply(System.Runtime.Intrinsics.X86.Avx.Multiply(g1, s1), System.Runtime.Intrinsics.X86.Avx.Subtract(one, s1)));

                    var g2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(gp + i + 16);
                    var s2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(sp + i + 16);
                    System.Runtime.Intrinsics.X86.Avx.Store(rp + i + 16, System.Runtime.Intrinsics.X86.Avx.Multiply(System.Runtime.Intrinsics.X86.Avx.Multiply(g2, s2), System.Runtime.Intrinsics.X86.Avx.Subtract(one, s2)));

                    var g3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(gp + i + 24);
                    var s3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(sp + i + 24);
                    System.Runtime.Intrinsics.X86.Avx.Store(rp + i + 24, System.Runtime.Intrinsics.X86.Avx.Multiply(System.Runtime.Intrinsics.X86.Avx.Multiply(g3, s3), System.Runtime.Intrinsics.X86.Avx.Subtract(one, s3)));
                }
            }
        }
#endif
        for (; i < length; i++)
            result[i] = grad[i] * sigmoid[i] * (1f - sigmoid[i]);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TanhBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var gradData = gradOutput.GetFlattenedData();
        var outData = output.GetDataArray();
        int length = gradOutput.Length;
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!output.IsContiguous) output = output.Contiguous();

        // Float fast path: SIMD grad * (1 - tanh^2)
        if (gradData is float[] gF && outData is float[] oF)
        {
#if NET5_0_OR_GREATER
            var resultArr = GC.AllocateUninitializedArray<float>(length);
#else
            var resultArr = new float[length];
#endif
            TanhBackwardFloat(gF, oF, resultArr);
            return (Tensor<T>)(object)TensorAllocator.Rent<T>(gradOutput._shape, (Vector<T>)(object)Vector<float>.FromMemory(resultArr));
        }

        // Double SIMD path
        if (typeof(T) == typeof(double))
        {
            var resultTensor = TensorAllocator.RentUninitialized<T>(gradOutput._shape);
            var gArr = (double[])(object)gradOutput.GetFlattenedData();
            var oArr = (double[])(object)output.GetDataArray();
            var rArr = (double[])(object)resultTensor.GetDataArray();
            unsafe
            {
                fixed (double* pG = gArr, pO = oArr, pR = rArr)
                    SimdKernels.TanhBackwardDouble(pG, pO, pR, length);
            }
            return resultTensor;
        }

        // Generic scalar fallback
        var result = new T[length];

        for (int i = 0; i < length; i++)
        {
            double t = numOps.ToDouble(outData[i]);
            double grad = numOps.ToDouble(gradData[i]);
            result[i] = numOps.FromDouble(grad * (1.0 - t * t));
        }

        return TensorAllocator.Rent<T>(gradOutput._shape, new Vector<T>(result));
    }

    private static unsafe void TanhBackwardFloat(float[] grad, float[] tanh, float[] result)
    {
        int length = grad.Length;
        int i = 0;
#if NET5_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported && length >= 32)
        {
            fixed (float* gp = grad, tp = tanh, rp = result)
            {
                var one = System.Runtime.Intrinsics.Vector256.Create(1.0f);
                int simdLen = length & ~31;
                for (; i < simdLen; i += 32)
                {
                    var g0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(gp + i);
                    var t0 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(tp + i);
                    System.Runtime.Intrinsics.X86.Avx.Store(rp + i, System.Runtime.Intrinsics.X86.Avx.Multiply(g0, System.Runtime.Intrinsics.X86.Avx.Subtract(one, System.Runtime.Intrinsics.X86.Avx.Multiply(t0, t0))));

                    var g1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(gp + i + 8);
                    var t1 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(tp + i + 8);
                    System.Runtime.Intrinsics.X86.Avx.Store(rp + i + 8, System.Runtime.Intrinsics.X86.Avx.Multiply(g1, System.Runtime.Intrinsics.X86.Avx.Subtract(one, System.Runtime.Intrinsics.X86.Avx.Multiply(t1, t1))));

                    var g2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(gp + i + 16);
                    var t2 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(tp + i + 16);
                    System.Runtime.Intrinsics.X86.Avx.Store(rp + i + 16, System.Runtime.Intrinsics.X86.Avx.Multiply(g2, System.Runtime.Intrinsics.X86.Avx.Subtract(one, System.Runtime.Intrinsics.X86.Avx.Multiply(t2, t2))));

                    var g3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(gp + i + 24);
                    var t3 = System.Runtime.Intrinsics.X86.Avx.LoadVector256(tp + i + 24);
                    System.Runtime.Intrinsics.X86.Avx.Store(rp + i + 24, System.Runtime.Intrinsics.X86.Avx.Multiply(g3, System.Runtime.Intrinsics.X86.Avx.Subtract(one, System.Runtime.Intrinsics.X86.Avx.Multiply(t3, t3))));
                }
            }
        }
#endif
        for (; i < length; i++)
            result[i] = grad[i] * (1f - tanh[i] * tanh[i]);
    }

    /// <inheritdoc/>
    public virtual unsafe Tensor<T> GeluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (gradOutput.Length != input.Length)
            throw new ArgumentException($"Shape mismatch: gradOutput length {gradOutput.Length} != input length {input.Length}");

        var numOps = MathHelper.GetNumericOperations<T>();
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!input.IsContiguous) input = input.Contiguous();
        int length = gradOutput.Length;

        var resultTensor = TensorAllocator.RentUninitialized<T>(gradOutput._shape);

        if (typeof(T) == typeof(float))
        {
            var gMem = AsFloatMemory(gradOutput.Data);
            var iMem = AsFloatMemory(input.Data);
            var rMem = AsFloatMemory(resultTensor.Data);
            using var pinG = gMem.Pin();
            using var pinI = iMem.Pin();
            using var pinR = rMem.Pin();
            float* pG = (float*)pinG.Pointer;
            float* pI = (float*)pinI.Pointer;
            float* pR = (float*)pinR.Pointer;
            SimdKernels.GeluBackwardUnsafe(pG, pI, pR, length);
        }
        else
        {
            const double sqrtTwoPi = 0.7978845608028654;
            const double coeff = 0.044715;
            var gradData = gradOutput.GetDataArray();
            var inputData = input.GetDataArray();
            var resultData = resultTensor.GetDataArray();
            for (int i = 0; i < length; i++)
            {
                double x = numOps.ToDouble(inputData[i]);
                double grad = numOps.ToDouble(gradData[i]);
                double tanhArg = sqrtTwoPi * (x + coeff * x * x * x);
                double tanhVal = Math.Tanh(tanhArg);
                double sechSq = 1.0 - tanhVal * tanhVal;
                double derivative = 0.5 * (1.0 + tanhVal) + 0.5 * x * sechSq * sqrtTwoPi * (1.0 + 3.0 * coeff * x * x);
                resultData[i] = numOps.FromDouble(grad * derivative);
            }
        }

        return resultTensor;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> LeakyReluBackward<T>(Tensor<T> gradOutput, Tensor<T> input, double negativeSlope)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var gradData = gradOutput.GetDataArray();
        var inputData = input.GetDataArray();
        var result = new T[gradOutput.Length];
        int length = gradOutput.Length;
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!input.IsContiguous) input = input.Contiguous();

        if (gradData is float[] gF && inputData is float[] iF && result is float[] rF)
        {
            float slopeF = (float)negativeSlope;
            Parallel.For(0, length, i => { rF[i] = iF[i] > 0f ? gF[i] : gF[i] * slopeF; });
        }
        else
        {
            for (int i = 0; i < length; i++)
            {
                double inputVal = numOps.ToDouble(inputData[i]);
                double grad = numOps.ToDouble(gradData[i]);
                result[i] = numOps.FromDouble(inputVal > 0 ? grad : grad * negativeSlope);
            }
        }

        return TensorAllocator.Rent<T>(gradOutput._shape, new Vector<T>(result));
    }

    // ──────────────────────────────────────────────────────────────
    // Activation backward methods — virtual for GPU override dispatch
    // ──────────────────────────────────────────────────────────────

    /// <inheritdoc/>
    public virtual Tensor<T> SwishBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        if (gradOutput.Length != input.Length)
            throw new ArgumentException($"Shape mismatch: gradOutput length {gradOutput.Length} != input length {input.Length}");
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            double sig = 1.0 / (1.0 + Math.Exp(-x));
            double deriv = sig + x * sig * (1.0 - sig);
            result[i] = numOps.FromDouble(numOps.ToDouble(gData[i]) * deriv);
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> MishBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            double sp = Math.Log(1.0 + Math.Exp(x));
            double tsp = Math.Tanh(sp);
            double sig = 1.0 / (1.0 + Math.Exp(-x));
            double deriv = tsp + x * sig * (1.0 - tsp * tsp);
            result[i] = numOps.FromDouble(numOps.ToDouble(gData[i]) * deriv);
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> SoftplusBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            double sig = 1.0 / (1.0 + Math.Exp(-x));
            result[i] = numOps.FromDouble(numOps.ToDouble(gData[i]) * sig);
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> HardswishBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            double deriv = x <= -3.0 ? 0.0 : x >= 3.0 ? 1.0 : (2.0 * x + 3.0) / 6.0;
            result[i] = numOps.FromDouble(numOps.ToDouble(gData[i]) * deriv);
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> SeluBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        const double alpha = 1.6732632423543772;
        const double scale = 1.0507009873554805;
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            double deriv = x >= 0 ? scale : scale * alpha * Math.Exp(x);
            result[i] = numOps.FromDouble(numOps.ToDouble(gData[i]) * deriv);
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> HardsigmoidBackward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            double deriv = (x > -3.0 && x < 3.0) ? 1.0 / 6.0 : 0.0;
            result[i] = numOps.FromDouble(numOps.ToDouble(gData[i]) * deriv);
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> Relu6Backward<T>(Tensor<T> gradOutput, Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            result[i] = (x > 0 && x < 6) ? gData[i] : numOps.Zero;
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> EluBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> output, double alpha)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        var oData = output.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            double deriv = x >= 0 ? 1.0 : numOps.ToDouble(oData[i]) + alpha;
            result[i] = numOps.FromDouble(numOps.ToDouble(gData[i]) * deriv);
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ThresholdBackward<T>(Tensor<T> gradOutput, Tensor<T> input, double threshold)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            result[i] = x > threshold ? gData[i] : numOps.Zero;
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ReciprocalBackward<T>(Tensor<T> gradOutput, Tensor<T> output)
    {
        // d/dx(1/x) = -1/x^2 = -(1/x)^2 = -output^2
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new T[gradOutput.Length];
        var gData = gradOutput.GetDataArray();
        var oData = output.GetDataArray();
        for (int i = 0; i < result.Length; i++)
        {
            double o = numOps.ToDouble(oData[i]);
            result[i] = numOps.FromDouble(numOps.ToDouble(gData[i]) * (-o * o));
        }
        return new Tensor<T>(result, gradOutput.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual (Tensor<T> inputGrad, Tensor<T> alphaGrad) PReLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> alpha)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var xGrad = new T[input.Length];
        var aGrad = new T[alpha.Length];
        var gData = gradOutput.GetDataArray();
        var xData = input.GetDataArray();
        var aData = alpha.GetDataArray();
        int alphaSize = alpha.Length;
        // For NCHW layout, alpha is per-channel: index = (i / spatialSize) % channels
        int spatialSize = input.Rank >= 4
            ? input.Shape._dims[input.Rank - 2] * input.Shape._dims[input.Rank - 1]
            : 1;
        for (int i = 0; i < input.Length; i++)
        {
            double val = numOps.ToDouble(xData[i]);
            int aIdx = alphaSize == 1 ? 0 : (i / spatialSize) % alphaSize;
            double a = numOps.ToDouble(aData[aIdx]);
            double g = numOps.ToDouble(gData[i]);
            xGrad[i] = val >= 0 ? gData[i] : numOps.FromDouble(a * g);
            if (val < 0)
                aGrad[aIdx] = numOps.Add(aGrad[aIdx], numOps.FromDouble(val * g));
        }
        return (new Tensor<T>(xGrad, input.Shape.ToArray()), new Tensor<T>(aGrad, alpha.Shape.ToArray()));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> VarBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, int[] axes)
    {
        if (axes is { Length: > 0 })
            throw new NotSupportedException("VarBackward with non-global axes is not yet supported. Use global variance backward.");
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = input.Length;
        var result = new T[n];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        var mData = mean.GetDataArray();
        double gOut = numOps.ToDouble(gData[0]);
        double m = numOps.ToDouble(mData[0]);
        for (int i = 0; i < n; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            result[i] = numOps.FromDouble(gOut * 2.0 * (x - m) / n);
        }
        return new Tensor<T>(result, input.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> StdBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> mean, Tensor<T> std, int[] axes)
    {
        if (axes is { Length: > 0 })
            throw new NotSupportedException("StdBackward with non-global axes is not yet supported. Use global std backward.");
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = input.Length;
        var result = new T[n];
        var gData = gradOutput.GetDataArray();
        var iData = input.GetDataArray();
        var mData = mean.GetDataArray();
        var sData = std.GetDataArray();
        double gOut = numOps.ToDouble(gData[0]);
        double m = numOps.ToDouble(mData[0]);
        double s = Math.Max(numOps.ToDouble(sData[0]), 1e-8);
        for (int i = 0; i < n; i++)
        {
            double x = numOps.ToDouble(iData[i]);
            result[i] = numOps.FromDouble(gOut * (x - m) / (n * s));
        }
        return new Tensor<T>(result, input.Shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> InstanceNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon, out Tensor<T> mean, out Tensor<T> variance)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int channels = input._shape[1];
        int spatialSize = 1;
        for (int i = 2; i < input.Rank; i++) spatialSize *= input._shape[i];

        var inputData = input.GetFlattenedData();
        var gammaData = gamma.GetDataArray();
        var betaData = beta.GetDataArray();
        var meanData = new T[batch * channels];
        var varData = new T[batch * channels];
        var resultData = new T[input.Length];

        if (inputData is float[] iF && gammaData is float[] gF && betaData is float[] bF
            && meanData is float[] mF && varData is float[] vF && resultData is float[] rF)
        {
            float epsF = (float)epsilon;
            Parallel.For(0, batch * channels, idx =>
            {
                int b2 = idx / channels;
                int c2 = idx % channels;
                int offset = (b2 * channels + c2) * spatialSize;

                float sum = 0f;
                for (int s = 0; s < spatialSize; s++)
                    sum += iF[offset + s];
                float meanVal = sum / spatialSize;
                mF[idx] = meanVal;

                float sumSq = 0f;
                for (int s = 0; s < spatialSize; s++)
                {
                    float diff = iF[offset + s] - meanVal;
                    sumSq += diff * diff;
                }
                float varVal = sumSq / spatialSize;
                vF[idx] = varVal;
                float invStd = 1f / MathF.Sqrt(varVal + epsF);
                float gVal = gF[c2];
                float bVal = bF[c2];

                for (int s = 0; s < spatialSize; s++)
                    rF[offset + s] = (iF[offset + s] - meanVal) * invStd * gVal + bVal;
            });
        }
        else
        {
            for (int b2 = 0; b2 < batch; b2++)
            {
                for (int c2 = 0; c2 < channels; c2++)
                {
                    int offset = (b2 * channels + c2) * spatialSize;
                    double sum = 0;
                    for (int s = 0; s < spatialSize; s++)
                        sum += numOps.ToDouble(inputData[offset + s]);
                    double meanVal = sum / spatialSize;
                    meanData[b2 * channels + c2] = numOps.FromDouble(meanVal);

                    double sumSq = 0;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        double diff = numOps.ToDouble(inputData[offset + s]) - meanVal;
                        sumSq += diff * diff;
                    }
                    double varVal = sumSq / spatialSize;
                    varData[b2 * channels + c2] = numOps.FromDouble(varVal);
                    double invStd = 1.0 / Math.Sqrt(varVal + epsilon);

                    double g = numOps.ToDouble(gammaData[c2]);
                    double be = numOps.ToDouble(betaData[c2]);
                    for (int s = 0; s < spatialSize; s++)
                    {
                        double x = numOps.ToDouble(inputData[offset + s]);
                        resultData[offset + s] = numOps.FromDouble((x - meanVal) * invStd * g + be);
                    }
                }
            }
        }

        mean = TensorAllocator.Rent<T>([batch, channels], new Vector<T>(meanData));
        variance = TensorAllocator.Rent<T>([batch, channels], new Vector<T>(varData));
        var inResult = TensorAllocator.Rent<T>(input._shape, new Vector<T>(resultData));
        DifferentiableOps.RecordIfActive("InstanceNorm", inResult, new[] { input, gamma, beta },
            BackwardFunctions<T>.InstanceNormBackward, new object[] { mean, variance, epsilon });
        return inResult;
    }

    /// <inheritdoc/>
    public Tensor<T> InstanceNormBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> gamma, Tensor<T> mean, Tensor<T> variance, double epsilon, out Tensor<T> gradGamma, out Tensor<T> gradBeta)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int channels = input._shape[1];
        int spatialSize = 1;
        for (int i = 2; i < input.Rank; i++) spatialSize *= input._shape[i];

        var gradOutData = gradOutput.GetDataArray();
        var inputData = input.GetFlattenedData();
        var gammaData = gamma.GetDataArray();
        var meanData = mean.GetDataArray();
        var varData = variance.GetDataArray();
        var gradInputData = new T[input.Length];
        var gradGammaData = new T[channels];
        var gradBetaData = new T[channels];

        // Instance normalization backward pass with correct gradient computation.
        // The gradient includes correction terms for the dependency of mean and variance on input.
        // Formula: dx = (1/N) * invStd * (N * δ - sum(δ) - xNorm * sum(δ * xNorm))
        // where δ = gradOutput * γ, N = spatialSize

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int offset = (b * channels + c) * spatialSize;
                double meanVal = numOps.ToDouble(meanData[b * channels + c]);
                double varVal = numOps.ToDouble(varData[b * channels + c]);
                double invStd = 1.0 / Math.Sqrt(varVal + epsilon);
                double g = numOps.ToDouble(gammaData[c]);

                // First pass: compute gradGamma, gradBeta, and accumulate sums for gradient correction
                double sumDelta = 0.0;
                double sumDeltaXNorm = 0.0;
                for (int s = 0; s < spatialSize; s++)
                {
                    double go = numOps.ToDouble(gradOutData[offset + s]);
                    double x = numOps.ToDouble(inputData[offset + s]);
                    double xNorm = (x - meanVal) * invStd;
                    double delta = go * g;

                    gradGammaData[c] = numOps.Add(gradGammaData[c], numOps.FromDouble(go * xNorm));
                    gradBetaData[c] = numOps.Add(gradBetaData[c], numOps.FromDouble(go));

                    sumDelta += delta;
                    sumDeltaXNorm += delta * xNorm;
                }

                // Second pass: compute gradInput with proper correction terms
                double invN = 1.0 / spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    double go = numOps.ToDouble(gradOutData[offset + s]);
                    double x = numOps.ToDouble(inputData[offset + s]);
                    double xNorm = (x - meanVal) * invStd;
                    double delta = go * g;

                    // dx = invStd * invN * (N * δ - sum(δ) - xNorm * sum(δ * xNorm))
                    double gradInput = invStd * invN * (spatialSize * delta - sumDelta - xNorm * sumDeltaXNorm);
                    gradInputData[offset + s] = numOps.FromDouble(gradInput);
                }
            }
        }

        gradGamma = TensorAllocator.Rent<T>([channels], new Vector<T>(gradGammaData));
        gradBeta = TensorAllocator.Rent<T>([channels], new Vector<T>(gradBetaData));
        return TensorAllocator.Rent<T>(input._shape, new Vector<T>(gradInputData));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> Dropout<T>(Tensor<T> input, double dropoutRate, bool training, out Tensor<T> mask)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var inputData = input.GetFlattenedData();
        var maskData = new T[input.Length];
        var resultData = new T[input.Length];

        if (!training || dropoutRate <= 0)
        {
            Array.Copy(inputData, resultData, input.Length);
            for (int i = 0; i < maskData.Length; i++)
                maskData[i] = numOps.One;
        }
        else
        {
            var rand = RandomHelper.CreateSeededRandom((int)(DateTime.UtcNow.Ticks % int.MaxValue));
            double scale = 1.0 / (1.0 - dropoutRate);
            for (int i = 0; i < input.Length; i++)
            {
                if (rand.NextDouble() >= dropoutRate)
                {
                    maskData[i] = numOps.FromDouble(scale);
                    resultData[i] = numOps.FromDouble(numOps.ToDouble(inputData[i]) * scale);
                }
                else
                {
                    maskData[i] = numOps.Zero;
                    resultData[i] = numOps.Zero;
                }
            }
        }

        mask = TensorAllocator.Rent<T>(input._shape, new Vector<T>(maskData));
        var dropResult = TensorAllocator.Rent<T>(input._shape, new Vector<T>(resultData));
        DifferentiableOps.RecordUnary("Dropout", dropResult, input, BackwardFunctions<T>.DropoutBackward, new object[] { mask, dropoutRate });
        return dropResult;
    }

    /// <inheritdoc/>
    public Tensor<T> DropoutBackward<T>(Tensor<T> gradOutput, Tensor<T> mask, double dropoutRate)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var gradData = gradOutput.GetFlattenedData();
        var maskData = mask.GetDataArray();
        var resultData = new T[gradOutput.Length];

        for (int i = 0; i < gradOutput.Length; i++)
            resultData[i] = numOps.Multiply(gradData[i], maskData[i]);

        return TensorAllocator.Rent<T>(gradOutput._shape, new Vector<T>(resultData));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> Embedding<T>(Tensor<int> indices, Tensor<T> embeddingTable)
    {
        int vocabSize = embeddingTable._shape[0];
        int embeddingDim = embeddingTable._shape[^1];
        int numIndices = indices.Length;
        var tableData = embeddingTable.GetFlattenedData();
        var indicesData = indices.GetFlattenedData();
        var resultData = new T[numIndices * embeddingDim];

        for (int i = 0; i < numIndices; i++)
        {
            int idx = indicesData[i];
            if (idx < 0 || idx >= vocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    $"Embedding index {idx} at position {i} is out of bounds. Valid range: [0, {vocabSize - 1}].");
            }

            int srcOffset = idx * embeddingDim;
            int dstOffset = i * embeddingDim;
            for (int j = 0; j < embeddingDim; j++)
                resultData[dstOffset + j] = tableData[srcOffset + j];
        }

        var outputShape = new int[indices._shape.Length + 1];
        for (int i = 0; i < indices._shape.Length; i++)
            outputShape[i] = indices._shape[i];
        outputShape[^1] = embeddingDim;

        var embResult = TensorAllocator.Rent<T>(outputShape, new Vector<T>(resultData));
        DifferentiableOps.RecordUnary("Embedding", embResult, embeddingTable, BackwardFunctions<T>.EmbeddingBackward,
            new object[] { indices, vocabSize, embeddingDim });
        return embResult;
    }

    /// <inheritdoc/>
    public Tensor<T> EmbeddingBackward<T>(Tensor<T> gradOutput, Tensor<int> indices, int vocabSize, int embeddingDim)
    {
        if (!gradOutput.IsContiguous) gradOutput = gradOutput.Contiguous();
        if (!indices.IsContiguous) indices = indices.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        var gradData = gradOutput.GetFlattenedData();
        var indicesData = indices.GetFlattenedData();
        var resultData = new T[vocabSize * embeddingDim];

        for (int i = 0; i < indices.Length; i++)
        {
            int idx = indicesData[i];
            if (idx < 0 || idx >= vocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(indices),
                    $"Embedding index {idx} at position {i} is out of bounds. Valid range: [0, {vocabSize - 1}].");
            }

            int srcOffset = i * embeddingDim;
            int dstOffset = idx * embeddingDim;
            for (int j = 0; j < embeddingDim; j++)
                resultData[dstOffset + j] = numOps.Add(resultData[dstOffset + j], gradData[srcOffset + j]);
        }

        return TensorAllocator.Rent<T>([vocabSize, embeddingDim], new Vector<T>(resultData));
    }

    /// <inheritdoc/>
    public T CrossEntropyLoss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int batchSize = predictions._shape[0];
        int numClasses = predictions._shape[1];

        var predData = predictions.GetFlattenedData();
        var targetData = targets.GetFlattenedData();

        // Parallel batch processing with thread-local accumulators
        double totalLoss = 0;
        bool sparseTargets = targets.Rank == 1;

        if (batchSize > 32)
        {
            object lockObj = new object();
            Parallel.For(0, batchSize, () => 0.0, (b, _, localLoss) =>
            {
                return localLoss + ComputeCrossEntropyBatch(numOps, predData, targetData, b, numClasses, sparseTargets);
            },
            localLoss => { lock (lockObj) { totalLoss += localLoss; } });
        }
        else
        {
            for (int b = 0; b < batchSize; b++)
            {
                totalLoss += ComputeCrossEntropyBatch(numOps, predData, targetData, b, numClasses, sparseTargets);
            }
        }

        return numOps.FromDouble(totalLoss / batchSize);
    }

    private static double ComputeCrossEntropyBatch<T>(INumericOperations<T> numOps, T[] predData, T[] targetData, int b, int numClasses, bool sparseTargets)
    {
        int offset = b * numClasses;

        // Compute log-sum-exp for numerical stability
        double maxVal = double.NegativeInfinity;
        for (int c = 0; c < numClasses; c++)
        {
            double val = numOps.ToDouble(predData[offset + c]);
            if (val > maxVal) maxVal = val;
        }

        double sumExp = 0;
        for (int c = 0; c < numClasses; c++)
        {
            sumExp += Math.Exp(numOps.ToDouble(predData[offset + c]) - maxVal);
        }
        double logSumExp = maxVal + Math.Log(sumExp);

        double loss = 0;
        if (sparseTargets)
        {
            int targetClass = (int)numOps.ToDouble(targetData[b]);
            if (targetClass < 0 || targetClass >= numClasses)
                throw new ArgumentOutOfRangeException(nameof(targetData), $"Target class {targetClass} is out of range [0, {numClasses}).");
            loss = -(numOps.ToDouble(predData[offset + targetClass]) - logSumExp);
        }
        else
        {
            for (int c = 0; c < numClasses; c++)
            {
                double targetVal = numOps.ToDouble(targetData[b * numClasses + c]);
                if (targetVal > 0)
                {
                    loss -= targetVal * (numOps.ToDouble(predData[offset + c]) - logSumExp);
                }
            }
        }
        return loss;
    }

    /// <inheritdoc/>
    public Tensor<T> CrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!predictions.IsContiguous) predictions = predictions.Contiguous();
        if (!targets.IsContiguous) targets = targets.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        int batchSize = predictions._shape[0];
        int numClasses = predictions._shape[1];

        var predData = predictions.GetFlattenedData();
        var targetData = targets.GetFlattenedData();
        var gradData = new T[predictions.Length];
        bool sparseTargets = targets.Rank == 1;

        // Parallelize batch processing
        Action<int> processBatch = b =>
        {
            int offset = b * numClasses;

            // Compute softmax
            double maxVal = double.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                double val = numOps.ToDouble(predData[offset + c]);
                if (val > maxVal) maxVal = val;
            }

            double sumExp = 0;
            var probs = new double[numClasses];
            for (int c = 0; c < numClasses; c++)
            {
                probs[c] = Math.Exp(numOps.ToDouble(predData[offset + c]) - maxVal);
                sumExp += probs[c];
            }
            double invSumExp = 1.0 / sumExp;
            for (int c = 0; c < numClasses; c++)
                probs[c] *= invSumExp;

            // Gradient
            if (sparseTargets)
            {
                int targetClass = (int)numOps.ToDouble(targetData[b]);
                for (int c = 0; c < numClasses; c++)
                {
                    double grad = probs[c] - (c == targetClass ? 1.0 : 0.0);
                    gradData[offset + c] = numOps.FromDouble(grad / batchSize);
                }
            }
            else
            {
                for (int c = 0; c < numClasses; c++)
                {
                    double targetVal = numOps.ToDouble(targetData[b * numClasses + c]);
                    double grad = probs[c] - targetVal;
                    gradData[offset + c] = numOps.FromDouble(grad / batchSize);
                }
            }
        };

        if (batchSize > 32)
        {
            Parallel.For(0, batchSize, processBatch);
        }
        else
        {
            for (int b = 0; b < batchSize; b++) processBatch(b);
        }

        return TensorAllocator.Rent<T>(predictions._shape, new Vector<T>(gradData));
    }

    /// <inheritdoc/>
    public T MseLoss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!predictions.IsContiguous) predictions = predictions.Contiguous();
        if (!targets.IsContiguous) targets = targets.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();

        // MSE = mean((pred - target)^2) = (dot(diff, diff)) / N
        // Use span-based subtract + dot product instead of ToArray + ToDouble per element
        var diff = TensorAllocator.RentUninitialized<T>(predictions._shape);
        numOps.Subtract(predictions.AsSpan(), targets.AsSpan(), diff.AsWritableSpan());
        T sumSq = numOps.Dot(diff.AsSpan(), diff.AsSpan());

        return numOps.Divide(sumSq, numOps.FromDouble(predictions.Length));
    }

    /// <inheritdoc/>
    public Tensor<T> MseBackward<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (!predictions.IsContiguous) predictions = predictions.Contiguous();
        if (!targets.IsContiguous) targets = targets.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();

        // grad = 2 * (pred - target) / N
        var result = TensorAllocator.RentUninitialized<T>(predictions._shape);
        numOps.Subtract(predictions.AsSpan(), targets.AsSpan(), result.AsWritableSpan());
        T scale = numOps.FromDouble(2.0 / predictions.Length);
        numOps.MultiplyScalar(result.AsSpan(), scale, result.AsWritableSpan());

        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> GlobalAvgPool2D<T>(Tensor<T> input)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int channels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];
        int spatialSize = height * width;

        var inputData = input.GetFlattenedData();
        var resultData = new T[batch * channels];
        int totalChannels = batch * channels;

        // Parallelize across batch*channels (each channel is independent)
        Action<int> processChannel = bc =>
        {
            int offset = bc * spatialSize;
            double sum = 0;
            for (int s = 0; s < spatialSize; s++)
                sum += numOps.ToDouble(inputData[offset + s]);
            resultData[bc] = numOps.FromDouble(sum / spatialSize);
        };

        if (totalChannels > 32)
            Parallel.For(0, totalChannels, processChannel);
        else
            for (int bc = 0; bc < totalChannels; bc++) processChannel(bc);

        return TensorAllocator.Rent<T>([batch, channels, 1, 1], new Vector<T>(resultData));
    }

    /// <inheritdoc/>
    public Tensor<T> GlobalMaxPool2D<T>(Tensor<T> input)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int channels = input._shape[1];
        int height = input._shape[2];
        int width = input._shape[3];
        int spatialSize = height * width;

        var inputData = input.GetFlattenedData();
        var resultData = new T[batch * channels];
        int totalChannels = batch * channels;

        Action<int> processChannel = bc =>
        {
            int offset = bc * spatialSize;
            double maxVal = numOps.ToDouble(inputData[offset]);
            for (int s = 1; s < spatialSize; s++)
            {
                double val = numOps.ToDouble(inputData[offset + s]);
                if (val > maxVal) maxVal = val;
            }
            resultData[bc] = numOps.FromDouble(maxVal);
        };

        if (totalChannels > 32)
            Parallel.For(0, totalChannels, processChannel);
        else
            for (int bc = 0; bc < totalChannels; bc++) processChannel(bc);

        return TensorAllocator.Rent<T>([batch, channels, 1, 1], new Vector<T>(resultData));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> AdaptiveAvgPool2D<T>(Tensor<T> input, int outputHeight, int outputWidth)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0];
        int channels = input._shape[1];
        int inHeight = input._shape[2];
        int inWidth = input._shape[3];

        var inputData = input.GetFlattenedData();
        var result = TensorAllocator.Rent<T>([batch, channels, outputHeight, outputWidth]);
        var resultData = result.GetDataArray();

        Parallel.For(0, batch * channels, idx =>
        {
            int b = idx / channels;
            int c = idx % channels;
            int inputBaseOffset = (b * channels + c) * inHeight * inWidth;
            int outputBaseOffset = (b * channels + c) * outputHeight * outputWidth;

            for (int oh = 0; oh < outputHeight; oh++)
            {
                int startH = (int)Math.Floor((double)oh * inHeight / outputHeight);
                int endH = (int)Math.Ceiling((double)(oh + 1) * inHeight / outputHeight);

                for (int ow = 0; ow < outputWidth; ow++)
                {
                    int startW = (int)Math.Floor((double)ow * inWidth / outputWidth);
                    int endW = (int)Math.Ceiling((double)(ow + 1) * inWidth / outputWidth);

                    double sum = 0;
                    int count = 0;
                    for (int ih = startH; ih < endH; ih++)
                    {
                        for (int iw = startW; iw < endW; iw++)
                        {
                            sum += numOps.ToDouble(inputData[inputBaseOffset + ih * inWidth + iw]);
                            count++;
                        }
                    }
                    resultData[outputBaseOffset + oh * outputWidth + ow] = numOps.FromDouble(sum / count);
                }
            }
        });

        DifferentiableOps.RecordUnary("AdaptiveAvgPool2D", result, input,
            BackwardFunctions<T>.AdaptiveAvgPool2DBackward,
            savedState: new object[] { outputHeight, outputWidth });
        return result;
    }

    #endregion

    #region JIT Dispatch Helpers

    /// <summary>
    /// Dispatches a JIT-compiled binary operation with multi-threaded parallelism for large arrays.
    /// For arrays >= 2M elements, splits work across threads with per-chunk JIT kernels.
    /// For smaller arrays, runs single-threaded JIT kernel.
    /// </summary>
    private static unsafe void JitBinaryDispatch(float* pA, float* pB, float* pR, int length, JitBinaryOp op)
    {
        // For large arrays, parallelize across threads
        // Use 500K threshold for bandwidth-bound binary ops
        int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 500_000));
        if (numChunks >= 2)
        {
            int chunkSize = (length + numChunks - 1) / numChunks;
            // Align to 32-float boundary for SIMD
            chunkSize = (chunkSize + 31) & ~31;

            Parallel.For(0, numChunks, chunk =>
            {
                int start = chunk * chunkSize;
                int count = Math.Min(chunkSize, length - start);
                if (count > 0)
                {
                    var kernel = CpuJitKernels.GetBinaryKernel(op, count);
                    kernel(pA + start, pB + start, pR + start, count);
                }
            });
        }
        else
        {
            var kernel = CpuJitKernels.GetBinaryKernel(op, length);
            kernel(pA, pB, pR, length);
        }
    }

    /// <summary>
    /// Dispatches a JIT-compiled ReLU with multi-threaded parallelism for large arrays.
    /// </summary>
    private static unsafe void JitUnaryDispatch(float* pSrc, float* pDst, int length)
    {
        int numChunks = Math.Min(CpuParallelSettings.MaxDegreeOfParallelism, Math.Max(1, length / 500_000));
        if (numChunks >= 2)
        {
            int chunkSize = (length + numChunks - 1) / numChunks;
            chunkSize = (chunkSize + 31) & ~31;

            Parallel.For(0, numChunks, chunk =>
            {
                int start = chunk * chunkSize;
                int count = Math.Min(chunkSize, length - start);
                if (count > 0)
                {
                    var kernel = CpuJitKernels.GetReLUKernel(count);
                    kernel(pSrc + start, pDst + start, count);
                }
            });
        }
        else
        {
            var kernel = CpuJitKernels.GetReLUKernel(length);
            kernel(pSrc, pDst, length);
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets the underlying array from a Memory without copying if possible.
    /// Falls back to ToArray() if the Memory is not backed by an array at offset 0.
    /// </summary>
    /// <typeparam name="TSource">The source generic type parameter.</typeparam>
    /// <typeparam name="TDest">The destination type (float or double).</typeparam>
    /// <param name="memory">The memory to extract the array from.</param>
    /// <returns>The underlying array or a copy.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TDest[] GetUnderlyingArrayOrCopy<TSource, TDest>(Memory<TSource> memory)
        where TDest : unmanaged
    {
        // Reinterpret the Memory<TSource> as Memory<TDest> using Unsafe.As
        // This is safe when called after checking typeof(TSource) == typeof(TDest)
        var destMemory = Unsafe.As<Memory<TSource>, Memory<TDest>>(ref memory);

        // Try to get the underlying array without copying
        if (MemoryMarshal.TryGetArray((ReadOnlyMemory<TDest>)destMemory, out var segment)
            && segment.Offset == 0
            && segment.Array != null)
        {
            return segment.Array;
        }

        // Fall back to creating a copy
        return destMemory.ToArray();
    }

    #endregion

    #region Tensor-Level Activation Aliases

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSigmoid<T>(Tensor<T> tensor)
    {
        return Sigmoid(tensor);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorReLU<T>(Tensor<T> tensor)
    {
        return ReLU(tensor);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorGELU<T>(Tensor<T> tensor)
    {
        return GELU(tensor);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSiLU<T>(Tensor<T> tensor)
    {
        // SiLU (Sigmoid Linear Unit) is mathematically equivalent to Swish
        return Swish(tensor);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTanh<T>(Tensor<T> tensor)
    {
        return Tanh(tensor);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLeakyReLU<T>(Tensor<T> tensor, T alpha)
    {
        return LeakyReLU(tensor, alpha);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMish<T>(Tensor<T> tensor)
    {
        return Mish(tensor);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHardSwish<T>(Tensor<T> tensor)
    {
        return HardSwish(tensor);
    }

    #endregion

    #region Tensor-Level Composite Operations

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLayerNorm<T>(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta, double epsilon = 1e-5)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        return LayerNorm(input, gamma, beta, epsilon, out _, out _);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> ReduceStd<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        // ReduceVariance is already stride-aware — no Contiguous() needed here

        var numOps = MathHelper.GetNumericOperations<T>();

        // std = sqrt(variance)
        var variance = ReduceVariance(input, axes, keepDims);
        var varianceData = variance.GetDataArray();
        var resultData = new T[varianceData.Length];

        for (int i = 0; i < varianceData.Length; i++)
        {
            // Clamp variance to zero before sqrt to handle floating-point roundoff
            // that can produce tiny negative values from reduction operations
            T v = numOps.LessThan(varianceData[i], numOps.Zero) ? numOps.Zero : varianceData[i];
            resultData[i] = numOps.Sqrt(v);
        }

        return TensorAllocator.Rent<T>(variance._shape, new Vector<T>(resultData));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLerp<T>(Tensor<T> a, Tensor<T> b, T t)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        // lerp(a, b, t) = a + t * (b - a) = (1-t)*a + t*b
        // Using a + t*(b-a) is more numerically stable and requires fewer ops
        var diff = TensorSubtract(b, a);  // b - a
        var scaled = TensorMultiplyScalar(diff, t);  // t * (b - a)
        return TensorAdd(a, scaled);  // a + t * (b - a)
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAddScaled<T>(Tensor<T> a, Tensor<T> b, T scaleA, T scaleB)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!ShapesMatch(a._shape, b._shape))
        {
            throw new ArgumentException(
                $"Tensor shapes must match. Got {FormatShape(a._shape)} and {FormatShape(b._shape)}.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(a._shape);
        var aData = a.GetFlattenedData();
        var bData = b.GetFlattenedData();
        var rData = result.GetDataArray();

        // Single pass: result[i] = scaleA * a[i] + scaleB * b[i]
        for (int i = 0; i < a.Length; i++)
        {
            rData[i] = numOps.Add(
                numOps.Multiply(aData[i], scaleA),
                numOps.Multiply(bData[i], scaleB));
        }

        object boxedScaleA = scaleA is not null ? (object)scaleA : throw new InvalidOperationException("scaleA cannot be null");
        object boxedScaleB = scaleB is not null ? (object)scaleB : throw new InvalidOperationException("scaleB cannot be null");
        DifferentiableOps.RecordBinary("TensorAddScaled", result, a, b, BackwardFunctions<T>.AddScaledBackward, new object[] { boxedScaleA, boxedScaleB });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMaxPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        return MaxPool2D(input, poolSize, stride, padding);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAvgPool2D<T>(Tensor<T> input, int poolSize, int stride = 0, int padding = 0)
    {
        return AvgPool2D(input, poolSize, stride, padding);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorConv2D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int dilation = 1)
    {
        if (!input.IsContiguous) input = input.Contiguous();
        return Conv2D(input, kernel, stride, padding, dilation);
    }

    #endregion

    /// <summary>
    /// Parallel dispatch for compute-bound unary float ops (Exp, Log, Sqrt, Tanh, etc.).
    /// Unlike bandwidth-bound ops (Add/Mul), compute-bound ops benefit from multi-threading
    /// even at smaller sizes because the bottleneck is ALU, not memory bandwidth.
    /// Threshold: 256K elements (~1MB of floats).
    /// </summary>
    private unsafe delegate void UnsafeUnaryKernel(float* input, float* output, int length);

    private static unsafe void ParallelComputeBound(float* input, float* output, int length, UnsafeUnaryKernel kernel)
    {
        // For compute-bound ops, parallelize at 256K+ elements
        const int parallelThreshold = 256 * 1024;
        int maxThreads = CpuParallelSettings.MaxDegreeOfParallelism;
        int chunks = Math.Min(maxThreads, Math.Max(1, length / parallelThreshold));

        if (chunks >= 2)
        {
            int chunkSize = (length + chunks - 1) / chunks;
            chunkSize = (chunkSize + 31) & ~31; // Align to AVX boundary

            // Can't capture float* in lambda — use IntPtr
            IntPtr pIn = (IntPtr)input;
            IntPtr pOut = (IntPtr)output;
            int totalLength = length;

            Parallel.For(0, chunks, chunk =>
            {
                int start = chunk * chunkSize;
                int count = Math.Min(chunkSize, totalLength - start);
                if (count > 0)
                {
                    kernel((float*)pIn + start, (float*)pOut + start, count);
                }
            });
        }
        else
        {
            kernel(input, output, length);
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Differentiable loss functions (return scalar Tensor<T> for tape)
    // ──────────────────────────────────────────────────────────────

    /// <summary>MSE loss: mean((pred - target)^2). Returns scalar tensor for tape.</summary>
    public virtual Tensor<T> TensorMSELoss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T mean;
        using (new NoGradScope<T>())
        {
            var diff = TensorSubtract(predictions, targets);
            var sq = TensorMultiply(diff, diff);
            T sum = TensorSum(sq);
            mean = numOps.Divide(sum, numOps.FromDouble(predictions.Length));
        }
        var result = new Tensor<T>(new[] { mean }, [1]);
        DifferentiableOps.RecordBinary("MSELoss", result, predictions, targets, BackwardFunctions<T>.MSELossBackward);
        return result;
    }

    /// <summary>L1 loss: mean(|pred - target|). Returns scalar tensor for tape.</summary>
    public virtual Tensor<T> TensorL1Loss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T mean;
        using (new NoGradScope<T>())
        {
            var diff = TensorSubtract(predictions, targets);
            var absDiff = TensorAbs(diff);
            T sum = TensorSum(absDiff);
            mean = numOps.Divide(sum, numOps.FromDouble(predictions.Length));
        }
        var result = new Tensor<T>(new[] { mean }, [1]);
        DifferentiableOps.RecordBinary("L1Loss", result, predictions, targets, BackwardFunctions<T>.L1LossBackward);
        return result;
    }

    /// <summary>Huber loss: smooth L1 that transitions from L2 to L1 at delta=1.</summary>
    public virtual Tensor<T> TensorHuberLoss<T>(Tensor<T> predictions, Tensor<T> targets, double delta = 1.0)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        Tensor<T> diff;
        using (new NoGradScope<T>()) { diff = TensorSubtract(predictions, targets); }
        T sum = numOps.Zero;
        for (int i = 0; i < diff.Length; i++)
        {
            double d = numOps.ToDouble(diff[i]);
            sum = numOps.Add(sum, Math.Abs(d) <= delta
                ? numOps.FromDouble(0.5 * d * d)
                : numOps.FromDouble(delta * (Math.Abs(d) - 0.5 * delta)));
        }
        T mean = numOps.Divide(sum, numOps.FromDouble(predictions.Length));
        var result = new Tensor<T>(new[] { mean }, [1]);
        DifferentiableOps.RecordBinary("HuberLoss", result, predictions, targets,
            BackwardFunctions<T>.HuberLossBackward, savedState: new object[] { delta });
        return result;
    }

    /// <summary>BCE with logits: sigmoid cross-entropy loss.</summary>
    public virtual Tensor<T> TensorBCEWithLogitsLoss<T>(Tensor<T> logits, Tensor<T> targets)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            double x = numOps.ToDouble(logits[i]);
            double t = numOps.ToDouble(targets[i]);
            // Numerically stable: max(x,0) - x*t + log(1+exp(-|x|))
            double loss = Math.Max(x, 0) - x * t + Math.Log(1 + Math.Exp(-Math.Abs(x)));
            sum = numOps.Add(sum, numOps.FromDouble(loss));
        }
        T mean = numOps.Divide(sum, numOps.FromDouble(logits.Length));
        var result = new Tensor<T>(new[] { mean }, [1]);
        DifferentiableOps.RecordBinary("BCEWithLogitsLoss", result, logits, targets,
            BackwardFunctions<T>.BCEWithLogitsLossBackward);
        return result;
    }

    /// <summary>Cross-entropy loss with softmax (differentiable). Returns scalar tensor.</summary>
    public virtual Tensor<T> TensorCrossEntropyLoss<T>(Tensor<T> logits, Tensor<T> targets)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int batchSize = logits._shape[0];
        int numClasses = logits._shape[1];
        T totalLoss = numOps.Zero;
        bool sparse = targets.Rank == 1;

        for (int b = 0; b < batchSize; b++)
        {
            totalLoss = numOps.Add(totalLoss,
                numOps.FromDouble(ComputeCrossEntropyBatch(numOps, logits.GetFlattenedData(), targets.GetFlattenedData(), b, numClasses, sparse)));
        }
        T mean = numOps.Divide(totalLoss, numOps.FromDouble(batchSize));
        var result = new Tensor<T>(new[] { mean }, [1]);
        DifferentiableOps.RecordBinary("CrossEntropyLoss", result, logits, targets,
            BackwardFunctions<T>.CrossEntropyLossBackward);
        return result;
    }

    /// <summary>NLL loss: -sum(target * log_probs) / batch_size. Expects log-probabilities.</summary>
    public virtual Tensor<T> TensorNLLLoss<T>(Tensor<T> logProbs, Tensor<T> targets)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int batchSize = logProbs._shape[0];
        int numClasses = logProbs._shape[1];
        T totalLoss = numOps.Zero;
        var lpData = logProbs.GetFlattenedData();
        var tData = targets.GetFlattenedData();

        for (int b = 0; b < batchSize; b++)
        {
            int targetClass = (int)numOps.ToDouble(tData[b]);
            if (targetClass >= 0 && targetClass < numClasses)
                totalLoss = numOps.Subtract(totalLoss, lpData[b * numClasses + targetClass]);
        }
        T mean = numOps.Divide(totalLoss, numOps.FromDouble(batchSize));
        var result = new Tensor<T>(new[] { mean }, [1]);
        DifferentiableOps.RecordBinary("NLLLoss", result, logProbs, targets,
            BackwardFunctions<T>.NLLLossBackward);
        return result;
    }

    /// <summary>KL divergence loss: sum(target * (log(target) - input)).</summary>
    public virtual Tensor<T> TensorKLDivLoss<T>(Tensor<T> input, Tensor<T> target)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = numOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            double t = numOps.ToDouble(target[i]);
            if (t > 0)
            {
                double x = numOps.ToDouble(input[i]);
                sum = numOps.Add(sum, numOps.FromDouble(t * (Math.Log(t) - x)));
            }
        }
        T mean = numOps.Divide(sum, numOps.FromDouble(input.Length));
        var result = new Tensor<T>(new[] { mean }, [1]);
        DifferentiableOps.RecordBinary("KLDivLoss", result, input, target,
            BackwardFunctions<T>.KLDivLossBackward);
        return result;
    }

    /// <summary>Cosine similarity loss between two tensors.</summary>
    public Tensor<T> TensorCosineSimilarityLoss<T>(Tensor<T> a, Tensor<T> b)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double dotProd = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double va = numOps.ToDouble(a[i]);
            double vb = numOps.ToDouble(b[i]);
            dotProd += va * vb;
            normA += va * va;
            normB += vb * vb;
        }
        double sim = dotProd / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-8);
        var result = new Tensor<T>(new[] { numOps.FromDouble(sim) }, [1]);
        DifferentiableOps.RecordBinary("CosineSimilarity", result, a, b,
            BackwardFunctions<T>.CosineSimilarityBackward);
        return result;
    }

    // ──────────────────────────────────────────────────────────────
    // Differentiable activations (missing from existing API)
    // ──────────────────────────────────────────────────────────────

    /// <summary>SELU activation: scale * (max(0,x) + min(0, alpha*(exp(x)-1)))</summary>
    public virtual Tensor<T> TensorSELU<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        const double alpha = 1.6732632423543772;
        const double scale = 1.0507009873554805;
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double x = numOps.ToDouble(tensor[i]);
            double val = x > 0 ? scale * x : scale * alpha * (Math.Exp(x) - 1);
            result[i] = numOps.FromDouble(val);
        }
        DifferentiableOps.RecordUnary("SELU", result, tensor, BackwardFunctions<T>.SELUBackward);
        return result;
    }

    /// <summary>HardSigmoid: clamp(x/6 + 0.5, 0, 1)</summary>
    public virtual Tensor<T> TensorHardSigmoid<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double x = numOps.ToDouble(tensor[i]);
            double val = Math.Max(0, Math.Min(1, x / 6.0 + 0.5));
            result[i] = numOps.FromDouble(val);
        }
        DifferentiableOps.RecordUnary("HardSigmoid", result, tensor, BackwardFunctions<T>.HardSigmoidBackward);
        return result;
    }

    /// <summary>ReLU6: min(max(0, x), 6)</summary>
    public virtual Tensor<T> TensorReLU6<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double x = numOps.ToDouble(tensor[i]);
            result[i] = numOps.FromDouble(Math.Max(0, Math.Min(6, x)));
        }
        DifferentiableOps.RecordUnary("ReLU6", result, tensor, BackwardFunctions<T>.ReLU6Backward);
        return result;
    }

    /// <summary>PReLU: max(0,x) + alpha * min(0,x) where alpha is a learnable parameter</summary>
    public virtual Tensor<T> TensorPReLU<T>(Tensor<T> tensor, Tensor<T> alpha)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        // Compute channel-aware alpha indexing for NCHW tensors
        int channels = alpha.Length;
        int spatialSize = tensor.Rank >= 4 ? tensor._shape[^2] * tensor._shape[^1] : 1;
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double x = numOps.ToDouble(tensor[i]);
            int channelIdx = channels == 1 ? 0 : (i / spatialSize) % channels;
            double a = numOps.ToDouble(alpha[channelIdx]);
            result[i] = numOps.FromDouble(x >= 0 ? x : a * x);
        }
        DifferentiableOps.RecordBinary("PReLU", result, tensor, alpha,
            BackwardFunctions<T>.PReLUBackward,
            savedState: new object[] { channels, spatialSize });
        return result;
    }

    /// <summary>RReLU: random leaky ReLU (lower, upper bounds)</summary>
    public virtual Tensor<T> TensorRReLU<T>(Tensor<T> tensor, double lower = 1.0/8, double upper = 1.0/3, bool training = true)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        var noise = new Tensor<T>(tensor._shape);
        var rng = new Random();
        for (int i = 0; i < tensor.Length; i++)
        {
            double x = numOps.ToDouble(tensor[i]);
            double a = training ? lower + rng.NextDouble() * (upper - lower) : (lower + upper) / 2.0;
            noise[i] = numOps.FromDouble(a);
            result[i] = numOps.FromDouble(x >= 0 ? x : a * x);
        }
        DifferentiableOps.RecordUnary("RReLU", result, tensor, BackwardFunctions<T>.RReLUBackward, savedState: new object[] { noise });
        return result;
    }

    /// <summary>Threshold: x if x > threshold, else value</summary>
    public virtual Tensor<T> TensorThreshold<T>(Tensor<T> tensor, T threshold, T value)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            result[i] = numOps.GreaterThan(tensor[i], threshold) ? tensor[i] : value;
        }
        DifferentiableOps.RecordUnary("Threshold", result, tensor, BackwardFunctions<T>.ThresholdBackward,
            savedState: new object[] { numOps.ToDouble(threshold) });
        return result;
    }

    /// <summary>Element-wise reciprocal: 1/x</summary>
    public Tensor<T> TensorReciprocal<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            result[i] = numOps.Divide(numOps.One, tensor[i]);
        }
        DifferentiableOps.RecordUnary("Reciprocal", result, tensor, BackwardFunctions<T>.ReciprocalBackward);
        return result;
    }

    /// <summary>Element-wise sign: -1, 0, or 1</summary>
    public virtual Tensor<T> TensorSign<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = TensorAllocator.RentUninitialized<T>(tensor._shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double x = numOps.ToDouble(tensor[i]);
            result[i] = numOps.FromDouble(Math.Sign(x));
        }
        DifferentiableOps.RecordUnary("Sign", result, tensor, BackwardFunctions<T>.SignBackward);
        return result;
    }

    // ──────────────────────────────────────────────────────────────
    // Differentiable shape/indexing ops (missing tape hooks)
    // ──────────────────────────────────────────────────────────────

    /// <summary>Flatten tensor to 1D.</summary>
    public virtual Tensor<T> TensorFlatten<T>(Tensor<T> tensor)
    {
        var result = tensor.Reshape([tensor.Length]);
        DifferentiableOps.RecordUnary("Flatten", result, tensor, BackwardFunctions<T>.FlattenBackward);
        return result;
    }

    /// <summary>Narrow (slice along one axis).</summary>
    public virtual Tensor<T> TensorNarrow<T>(Tensor<T> tensor, int dim, int start, int length)
    {
        var result = tensor.Slice(dim, start, start + length);
        DifferentiableOps.RecordUnary("Narrow", result, tensor, BackwardFunctions<T>.NarrowBackward,
            savedState: new object[] { dim, start, length });
        return result;
    }

    /// <summary>IndexSelect: select indices along a given axis (differentiable).</summary>
    public Tensor<T> TensorIndexSelectDiff<T>(Tensor<T> source, Tensor<int> indices, int axis)
    {
        // TensorIndexSelect already records to the tape — no additional recording needed
        return TensorIndexSelect(source, indices, axis);
    }

    /// <summary>Constant padding for N-dimensional tensors.</summary>
    public virtual Tensor<T> TensorConstantPad<T>(Tensor<T> tensor, int[] padding, T value)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int rank = tensor.Rank;
        // Padding is [before_last, after_last, before_second_last, after_second_last, ...]
        var newShape = tensor._shape.ToArray();
        for (int i = 0; i < padding.Length / 2 && i < rank; i++)
        {
            int dim = rank - 1 - i;
            newShape[dim] += padding[2 * i] + padding[2 * i + 1];
        }
        var result = new Tensor<T>(newShape);
        result.Fill(value);
        // Copy original data
        CopyTensorRegion(tensor, result, padding, rank);
        DifferentiableOps.RecordUnary("ConstantPad", result, tensor, BackwardFunctions<T>.ConstantPadBackward,
            savedState: new object[] { padding });
        return result;
    }

    /// <summary>Upsample using bilinear interpolation (4D: NCHW).</summary>
    public virtual Tensor<T> TensorUpsampleBilinear<T>(Tensor<T> input, int[] outputSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = input._shape[0], c = input._shape[1], h = input._shape[2], w = input._shape[3];
        int outH = outputSize[0], outW = outputSize[1];
        var result = TensorAllocator.Rent<T>([n, c, outH, outW]);
        var inData = input.GetFlattenedData();
        var outData = result.GetDataArray();

        for (int batch = 0; batch < n; batch++)
            for (int ch = 0; ch < c; ch++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        double srcH = (oh + 0.5) * h / outH - 0.5;
                        double srcW = (ow + 0.5) * w / outW - 0.5;
                        int h0 = Math.Max(0, (int)Math.Floor(srcH));
                        int w0 = Math.Max(0, (int)Math.Floor(srcW));
                        int h1 = Math.Min(h - 1, h0 + 1);
                        int w1 = Math.Min(w - 1, w0 + 1);
                        double fh = srcH - h0, fw = srcW - w0;
                        int baseIdx = (batch * c + ch) * h * w;
                        double val = (1 - fh) * (1 - fw) * numOps.ToDouble(inData[baseIdx + h0 * w + w0])
                                   + (1 - fh) * fw * numOps.ToDouble(inData[baseIdx + h0 * w + w1])
                                   + fh * (1 - fw) * numOps.ToDouble(inData[baseIdx + h1 * w + w0])
                                   + fh * fw * numOps.ToDouble(inData[baseIdx + h1 * w + w1]);
                        outData[(batch * c + ch) * outH * outW + oh * outW + ow] = numOps.FromDouble(val);
                    }
        DifferentiableOps.RecordUnary("UpsampleBilinear", result, input, BackwardFunctions<T>.UpsampleBilinearBackward,
            savedState: new object[] { new[] { h, w } });
        return result;
    }

    // ──────────────────────────────────────────────────────────────
    // Differentiable pooling ops
    // ──────────────────────────────────────────────────────────────

    /// <summary>1D average pooling.</summary>
    public virtual Tensor<T> TensorAvgPool1D<T>(Tensor<T> input, int kernelSize, int stride)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0], channels = input._shape[1], width = input._shape[2];
        int outW = (width - kernelSize) / stride + 1;
        var result = TensorAllocator.Rent<T>([batch, channels, outW]);
        var inData = input.GetFlattenedData();
        var outData = result.GetDataArray();

        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int ow = 0; ow < outW; ow++)
                {
                    T sum = numOps.Zero;
                    int startW = ow * stride;
                    for (int k = 0; k < kernelSize; k++)
                        sum = numOps.Add(sum, inData[(b * channels + c) * width + startW + k]);
                    outData[(b * channels + c) * outW + ow] = numOps.Divide(sum, numOps.FromDouble(kernelSize));
                }
        DifferentiableOps.RecordUnary("AvgPool1D", result, input, BackwardFunctions<T>.AvgPool1DBackward,
            savedState: new object[] { kernelSize, stride });
        return result;
    }

    /// <summary>1D max pooling with argmax tracking.</summary>
    public virtual Tensor<T> TensorMaxPool1D<T>(Tensor<T> input, int kernelSize, int stride)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = input._shape[0], channels = input._shape[1], width = input._shape[2];
        int outW = (width - kernelSize) / stride + 1;
        var result = TensorAllocator.Rent<T>([batch, channels, outW]);
        var indices = new int[batch * channels * outW];
        var inData = input.GetFlattenedData();
        var outData = result.GetDataArray();

        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int ow = 0; ow < outW; ow++)
                {
                    int startW = ow * stride;
                    T maxVal = inData[(b * channels + c) * width + startW];
                    int maxIdx = startW;
                    for (int k = 1; k < kernelSize; k++)
                    {
                        int idx = startW + k;
                        T val = inData[(b * channels + c) * width + idx];
                        if (numOps.GreaterThan(val, maxVal)) { maxVal = val; maxIdx = idx; }
                    }
                    int outIdx = (b * channels + c) * outW + ow;
                    outData[outIdx] = maxVal;
                    indices[outIdx] = maxIdx;
                }
        DifferentiableOps.RecordUnary("MaxPool1D", result, input, BackwardFunctions<T>.MaxPool1DBackward,
            savedState: new object[] { indices, kernelSize, stride });
        return result;
    }

    // ──────────────────────────────────────────────────────────────
    // Differentiable tensor mean (returns Tensor<T> for tape)
    // ──────────────────────────────────────────────────────────────

    /// <summary>Full mean reduction returning scalar tensor for tape.</summary>
    public Tensor<T> TensorMeanDiff<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T sum = TensorSum(tensor);
        T mean = numOps.Divide(sum, numOps.FromDouble(tensor.Length));
        var result = new Tensor<T>(new[] { mean }, [1]);
        DifferentiableOps.RecordUnary("Mean", result, tensor, BackwardFunctions<T>.MeanBackward);
        return result;
    }

    // ──────────────────────────────────────────────────────────────
    // Missing Phase 1 ops
    // ──────────────────────────────────────────────────────────────

    /// <summary>Stack tensors along a new axis (differentiable).</summary>
    public Tensor<T> TensorStackDiff<T>(Tensor<T>[] tensors, int axis = 0)
    {
        // TensorStack already records to the tape — no additional recording needed
        return TensorStack(tensors, axis);
    }

    /// <summary>Variance of all elements, returns scalar tensor.</summary>
    public virtual Tensor<T> TensorVar<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double mean = 0;
        for (int i = 0; i < tensor.Length; i++) mean += numOps.ToDouble(tensor[i]);
        mean /= tensor.Length;
        double variance = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            double d = numOps.ToDouble(tensor[i]) - mean;
            variance += d * d;
        }
        variance /= tensor.Length;
        var result = new Tensor<T>(new[] { numOps.FromDouble(variance) }, [1]);
        DifferentiableOps.RecordUnary("Var", result, tensor, BackwardFunctions<T>.VarBackward);
        return result;
    }

    /// <summary>Standard deviation of all elements, returns scalar tensor.</summary>
    public virtual Tensor<T> TensorStd<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double mean = 0;
        for (int i = 0; i < tensor.Length; i++) mean += numOps.ToDouble(tensor[i]);
        mean /= tensor.Length;
        double variance = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            double d = numOps.ToDouble(tensor[i]) - mean;
            variance += d * d;
        }
        variance /= tensor.Length;
        var result = new Tensor<T>(new[] { numOps.FromDouble(Math.Sqrt(variance)) }, [1]);
        DifferentiableOps.RecordUnary("Std", result, tensor, BackwardFunctions<T>.StdBackward);
        return result;
    }

    /// <summary>Element-wise square: x^2.</summary>
    public Tensor<T> TensorSquare<T>(Tensor<T> tensor)
    {
        var result = TensorMultiply(tensor, tensor);
        // Replace the TensorMultiply tape entry with Square for cleaner backward
        // Actually, TensorMultiply already records and self-multiply gives correct 2x gradient
        // But let's record explicitly for semantics
        return result;
    }

    /// <summary>LogSumExp: log(sum(exp(x))). Numerically stable.</summary>
    public Tensor<T> TensorLogSumExp<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double maxVal = double.NegativeInfinity;
        for (int i = 0; i < tensor.Length; i++)
            maxVal = Math.Max(maxVal, numOps.ToDouble(tensor[i]));
        double sumExp = 0;
        for (int i = 0; i < tensor.Length; i++)
            sumExp += Math.Exp(numOps.ToDouble(tensor[i]) - maxVal);
        double lse = maxVal + Math.Log(sumExp);
        var result = new Tensor<T>(new[] { numOps.FromDouble(lse) }, [1]);
        DifferentiableOps.RecordUnary("LogSumExp", result, tensor, BackwardFunctions<T>.LogSumExpBackward);
        return result;
    }

    /// <summary>L2 norm: sqrt(sum(x^2)).</summary>
    public Tensor<T> TensorNorm<T>(Tensor<T> tensor)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double sumSq = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            double v = numOps.ToDouble(tensor[i]);
            sumSq += v * v;
        }
        var result = new Tensor<T>(new[] { numOps.FromDouble(Math.Sqrt(sumSq)) }, [1]);
        DifferentiableOps.RecordUnary("Norm", result, tensor, BackwardFunctions<T>.NormBackward);
        return result;
    }

    /// <summary>Adaptive max pool 2D with argmax tracking.</summary>
    public virtual Tensor<T> TensorAdaptiveMaxPool2D<T>(Tensor<T> input, int[] outputSize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = input._shape[0], c = input._shape[1], h = input._shape[2], w = input._shape[3];
        int outH = outputSize[0], outW = outputSize[1];
        var result = TensorAllocator.Rent<T>([n, c, outH, outW]);
        var argmax = new int[n * c * outH * outW];
        var inData = input.GetFlattenedData();
        var outData = result.GetDataArray();

        for (int batch = 0; batch < n; batch++)
            for (int ch = 0; ch < c; ch++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int hStart = oh * h / outH, hEnd = (oh + 1) * h / outH;
                        int wStart = ow * w / outW, wEnd = (ow + 1) * w / outW;
                        int baseIdx = (batch * c + ch) * h * w;
                        double maxV = double.NegativeInfinity;
                        int maxI = baseIdx + hStart * w + wStart;
                        for (int ih = hStart; ih < hEnd; ih++)
                            for (int iw = wStart; iw < wEnd; iw++)
                            {
                                int idx = baseIdx + ih * w + iw;
                                double v = numOps.ToDouble(inData[idx]);
                                if (v > maxV) { maxV = v; maxI = idx; }
                            }
                        int outIdx = (batch * c + ch) * outH * outW + oh * outW + ow;
                        outData[outIdx] = numOps.FromDouble(maxV);
                        argmax[outIdx] = maxI;
                    }
        DifferentiableOps.RecordUnary("AdaptiveMaxPool2D", result, input,
            BackwardFunctions<T>.AdaptiveMaxPool2DBackward, savedState: new object[] { argmax });
        return result;
    }

    /// <summary>Where: select elements from x or y based on condition.</summary>
    public Tensor<T> TensorWhere<T>(bool[] condition, Tensor<T> x, Tensor<T> y)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++)
            result[i] = condition[i] ? x[i] : y[i];
        DifferentiableOps.RecordBinary("Where", result, x, y,
            BackwardFunctions<T>.WhereBackward, savedState: new object[] { condition });
        return result;
    }

    /// <summary>MaskedFill: fill elements where mask is true with value.</summary>
    public Tensor<T> TensorMaskedFill<T>(Tensor<T> tensor, bool[] mask, T value)
    {
        var result = new Tensor<T>(tensor.Shape.ToArray());
        for (int i = 0; i < tensor.Length; i++)
            result[i] = mask[i] ? value : tensor[i];
        DifferentiableOps.RecordUnary("MaskedFill", result, tensor,
            BackwardFunctions<T>.MaskedFillBackward, savedState: new object[] { mask });
        return result;
    }

    /// <summary>Scaled dot-product attention: softmax(Q@K^T/sqrt(dk)) @ V</summary>
    public Tensor<T> TensorScaledDotProductAttention<T>(Tensor<T> query, Tensor<T> key, Tensor<T> value)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int dk = query._shape[^1];
        T scale = numOps.FromDouble(1.0 / Math.Sqrt(dk));

        // Q @ K^T
        var keyT = TensorTranspose(key);
        var scores = TensorMatMul(query, keyT);
        // Scale
        var scaledScores = TensorMultiplyScalar(scores, scale);
        // Softmax along last axis
        int axis = scaledScores.Rank - 1;
        var attnWeights = TensorSoftmax(scaledScores, axis);
        // @ V
        var output = TensorMatMul(attnWeights, value);
        return output;
    }

    private static void CopyTensorRegion<T>(Tensor<T> src, Tensor<T> dst, int[] padding, int rank)
    {
        var srcData = src.GetFlattenedData();
        var dstData = dst.GetDataArray();
        var srcShape = src._shape;
        var dstShape = dst._shape;

        for (int i = 0; i < src.Length; i++)
        {
            var srcIndices = new int[rank];
            int remaining = i;
            for (int d = rank - 1; d >= 0; d--)
            {
                srcIndices[d] = remaining % srcShape[d];
                remaining /= srcShape[d];
            }

            var dstIndices = new int[rank];
            for (int d = 0; d < rank; d++)
            {
                int padDimIdx = rank - 1 - d;
                int padBefore = padDimIdx < padding.Length / 2 ? padding[2 * padDimIdx] : 0;
                dstIndices[d] = srcIndices[d] + padBefore;
            }

            int dstFlat = 0;
            int stride = 1;
            for (int d = rank - 1; d >= 0; d--)
            {
                dstFlat += dstIndices[d] * stride;
                stride *= dstShape[d];
            }

            dstData[dstFlat] = srcData[i];
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Fused Linear + Activation Operations
    // Single tape entry instead of 3, fused backward for performance.
    // ═══════════════════════════════════════════════════════════════════

    /// <inheritdoc/>
    public virtual Tensor<T> FusedLinearReLU<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        var linear = TensorMatMul(input, weight);
        var biased = TensorBroadcastAdd(linear, bias);
        var preActivation = biased;
        var result = ReLU(biased);

        // Replace the 3 separate tape entries with a single fused entry
        RemoveLastNTapeEntries<T>(3);
        DifferentiableOps.RecordIfActive("FusedLinearReLU", result,
            new[] { input, weight, bias },
            BackwardFunctions<T>.FusedMatMulAddReLUBackward,
            new object[] { preActivation });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> FusedLinearSigmoid<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        var linear = TensorMatMul(input, weight);
        var biased = TensorBroadcastAdd(linear, bias);
        var result = TensorSigmoid(biased);

        RemoveLastNTapeEntries<T>(3);
        DifferentiableOps.RecordIfActive("FusedLinearSigmoid", result,
            new[] { input, weight, bias },
            BackwardFunctions<T>.FusedMatMulAddSigmoidBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> FusedLinearTanh<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        var linear = TensorMatMul(input, weight);
        var biased = TensorBroadcastAdd(linear, bias);
        var result = Tanh(biased);

        RemoveLastNTapeEntries<T>(3);
        DifferentiableOps.RecordIfActive("FusedLinearTanh", result,
            new[] { input, weight, bias },
            BackwardFunctions<T>.FusedMatMulAddTanhBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> FusedLinearGELU<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        var linear = TensorMatMul(input, weight);
        var biased = TensorBroadcastAdd(linear, bias);
        var preActivation = biased;
        var result = GELU(biased);

        RemoveLastNTapeEntries<T>(3);
        DifferentiableOps.RecordIfActive("FusedLinearGELU", result,
            new[] { input, weight, bias },
            BackwardFunctions<T>.FusedMatMulAddGELUBackward,
            new object[] { preActivation });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> FusedLinearSwish<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        var linear = TensorMatMul(input, weight);
        var biased = TensorBroadcastAdd(linear, bias);
        var preActivation = biased;
        var result = Swish(biased);

        RemoveLastNTapeEntries<T>(3);
        DifferentiableOps.RecordIfActive("FusedLinearSwish", result,
            new[] { input, weight, bias },
            BackwardFunctions<T>.FusedMatMulAddSwishBackward,
            new object[] { preActivation });
        return result;
    }

    /// <summary>
    /// Removes the last N entries from the active tape. Used by fused ops to replace
    /// individual entries with a single fused entry.
    /// </summary>
    private static void RemoveLastNTapeEntries<T>(int n)
    {
        if (NoGradScope<T>.IsSuppressed) return;
        var tape = GradientTape<T>.Current;
        if (tape is null) return;
        for (int i = 0; i < n && tape.EntryCount > 0; i++)
            tape.RemoveLastEntry();
    }

    // ═══════════════════════════════════════════════════════════════════
    // IoU Loss Operations — Differentiable bounding box regression losses
    // Composed from existing tape-tracked engine ops for automatic backward.
    // ═══════════════════════════════════════════════════════════════════

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIoULoss<T>(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (target == null) throw new ArgumentNullException(nameof(target));
        if (predicted.Shape.Length != 2 || predicted.Shape[1] != 4)
            throw new ArgumentException("Predicted must be [N, 4] in (x1, y1, x2, y2) format.", nameof(predicted));
        if (target.Shape.Length != 2 || target.Shape[1] != 4)
            throw new ArgumentException("Target must be [N, 4] in (x1, y1, x2, y2) format.", nameof(target));
        if (target.Shape[0] != predicted.Shape[0])
            throw new ArgumentException("Target batch size must match predicted batch size.", nameof(target));

        int n = predicted.Shape[0];

        // Extract coordinates: [N] tensors for each coordinate
        // TensorSlice(tensor, start, length) — length is always [n, 1] for single column
        var px1 = TensorSlice(predicted, new[] { 0, 0 }, new[] { n, 1 });
        var py1 = TensorSlice(predicted, new[] { 0, 1 }, new[] { n, 1 });
        var px2 = TensorSlice(predicted, new[] { 0, 2 }, new[] { n, 1 });
        var py2 = TensorSlice(predicted, new[] { 0, 3 }, new[] { n, 1 });

        var tx1 = TensorSlice(target, new[] { 0, 0 }, new[] { n, 1 });
        var ty1 = TensorSlice(target, new[] { 0, 1 }, new[] { n, 1 });
        var tx2 = TensorSlice(target, new[] { 0, 2 }, new[] { n, 1 });
        var ty2 = TensorSlice(target, new[] { 0, 3 }, new[] { n, 1 });

        // Intersection: max(0, min(x2_p, x2_t) - max(x1_p, x1_t)) * max(0, min(y2_p, y2_t) - max(y1_p, y1_t))
        var interX1 = TensorMax(px1, tx1);
        var interY1 = TensorMax(py1, ty1);
        // min(a,b) = -max(-a,-b)
        var interX2 = TensorNegate(TensorMax(TensorNegate(px2), TensorNegate(tx2)));
        var interY2 = TensorNegate(TensorMax(TensorNegate(py2), TensorNegate(ty2)));

        var interW = ReLU(TensorSubtract(interX2, interX1));
        var interH = ReLU(TensorSubtract(interY2, interY1));
        var interArea = TensorMultiply(interW, interH);

        // Union: area_p + area_t - intersection, clamp widths/heights >= 0
        var predW = ReLU(TensorSubtract(px2, px1));
        var predH = ReLU(TensorSubtract(py2, py1));
        var predArea = TensorMultiply(predW, predH);
        var targW = ReLU(TensorSubtract(tx2, tx1));
        var targH = ReLU(TensorSubtract(ty2, ty1));
        var targArea = TensorMultiply(targW, targH);
        var eps = MathHelper.GetNumericOperations<T>().FromDouble(1e-7);
        var unionArea = TensorAddScalar(TensorSubtract(TensorAdd(predArea, targArea), interArea), eps);

        // IoU = intersection / union
        var iou = TensorDivide(interArea, unionArea);

        // Loss = 1 - IoU, reshape from [N,1] to [N] via tape-tracked engine op
        var loss = ScalarMinusTensor(MathHelper.GetNumericOperations<T>().One, iou);
        return Reshape(loss, new[] { n });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorGIoULoss<T>(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (target == null) throw new ArgumentNullException(nameof(target));
        if (predicted.Shape.Length != 2 || predicted.Shape[1] != 4)
            throw new ArgumentException("Predicted must be [N, 4] in (x1, y1, x2, y2) format.", nameof(predicted));
        if (target.Shape.Length != 2 || target.Shape[1] != 4)
            throw new ArgumentException("Target must be [N, 4] in (x1, y1, x2, y2) format.", nameof(target));
        if (target.Shape[0] != predicted.Shape[0])
            throw new ArgumentException("Target batch size must match predicted batch size.", nameof(target));

        var numOps = MathHelper.GetNumericOperations<T>();
        int n = predicted.Shape[0];

        var px1 = TensorSlice(predicted, new[] { 0, 0 }, new[] { n, 1 });
        var py1 = TensorSlice(predicted, new[] { 0, 1 }, new[] { n, 1 });
        var px2 = TensorSlice(predicted, new[] { 0, 2 }, new[] { n, 1 });
        var py2 = TensorSlice(predicted, new[] { 0, 3 }, new[] { n, 1 });

        var tx1 = TensorSlice(target, new[] { 0, 0 }, new[] { n, 1 });
        var ty1 = TensorSlice(target, new[] { 0, 1 }, new[] { n, 1 });
        var tx2 = TensorSlice(target, new[] { 0, 2 }, new[] { n, 1 });
        var ty2 = TensorSlice(target, new[] { 0, 3 }, new[] { n, 1 });

        // Intersection
        var interX1 = TensorMax(px1, tx1);
        var interY1 = TensorMax(py1, ty1);
        var interX2 = TensorNegate(TensorMax(TensorNegate(px2), TensorNegate(tx2)));
        var interY2 = TensorNegate(TensorMax(TensorNegate(py2), TensorNegate(ty2)));
        var interW = ReLU(TensorSubtract(interX2, interX1));
        var interH = ReLU(TensorSubtract(interY2, interY1));
        var interArea = TensorMultiply(interW, interH);

        // Union with clamped widths/heights
        var predArea = TensorMultiply(ReLU(TensorSubtract(px2, px1)), ReLU(TensorSubtract(py2, py1)));
        var targArea = TensorMultiply(ReLU(TensorSubtract(tx2, tx1)), ReLU(TensorSubtract(ty2, ty1)));
        var eps = numOps.FromDouble(1e-7);
        var unionArea = TensorAddScalar(TensorSubtract(TensorAdd(predArea, targArea), interArea), eps);

        // IoU
        var iou = TensorDivide(interArea, unionArea);

        // Enclosing box
        var encX1 = TensorNegate(TensorMax(TensorNegate(px1), TensorNegate(tx1)));
        var encY1 = TensorNegate(TensorMax(TensorNegate(py1), TensorNegate(ty1)));
        var encX2 = TensorMax(px2, tx2);
        var encY2 = TensorMax(py2, ty2);
        var encArea = TensorAddScalar(
            TensorMultiply(TensorSubtract(encX2, encX1), TensorSubtract(encY2, encY1)), eps);

        // GIoU = IoU - (enclosing_area - union_area) / enclosing_area
        var fillRatio = TensorDivide(TensorSubtract(encArea, unionArea), encArea);
        var giou = TensorSubtract(iou, fillRatio);

        // Loss = 1 - GIoU, reshape from [N,1] to [N]
        return Reshape(ScalarMinusTensor(numOps.One, giou), new[] { n });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDIoULoss<T>(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (target == null) throw new ArgumentNullException(nameof(target));
        if (predicted.Shape.Length != 2 || predicted.Shape[1] != 4)
            throw new ArgumentException("Predicted must be [N, 4] in (x1, y1, x2, y2) format.", nameof(predicted));
        if (target.Shape.Length != 2 || target.Shape[1] != 4)
            throw new ArgumentException("Target must be [N, 4] in (x1, y1, x2, y2) format.", nameof(target));
        if (target.Shape[0] != predicted.Shape[0])
            throw new ArgumentException("Target batch size must match predicted batch size.", nameof(target));

        var numOps = MathHelper.GetNumericOperations<T>();
        int n = predicted.Shape[0];

        var px1 = TensorSlice(predicted, new[] { 0, 0 }, new[] { n, 1 });
        var py1 = TensorSlice(predicted, new[] { 0, 1 }, new[] { n, 1 });
        var px2 = TensorSlice(predicted, new[] { 0, 2 }, new[] { n, 1 });
        var py2 = TensorSlice(predicted, new[] { 0, 3 }, new[] { n, 1 });

        var tx1 = TensorSlice(target, new[] { 0, 0 }, new[] { n, 1 });
        var ty1 = TensorSlice(target, new[] { 0, 1 }, new[] { n, 1 });
        var tx2 = TensorSlice(target, new[] { 0, 2 }, new[] { n, 1 });
        var ty2 = TensorSlice(target, new[] { 0, 3 }, new[] { n, 1 });

        // IoU with clamped areas
        var interX1 = TensorMax(px1, tx1);
        var interY1 = TensorMax(py1, ty1);
        var interX2 = TensorNegate(TensorMax(TensorNegate(px2), TensorNegate(tx2)));
        var interY2 = TensorNegate(TensorMax(TensorNegate(py2), TensorNegate(ty2)));
        var interW = ReLU(TensorSubtract(interX2, interX1));
        var interH = ReLU(TensorSubtract(interY2, interY1));
        var interArea = TensorMultiply(interW, interH);
        var predArea = TensorMultiply(ReLU(TensorSubtract(px2, px1)), ReLU(TensorSubtract(py2, py1)));
        var targArea = TensorMultiply(ReLU(TensorSubtract(tx2, tx1)), ReLU(TensorSubtract(ty2, ty1)));
        var eps = numOps.FromDouble(1e-7);
        var unionArea = TensorAddScalar(TensorSubtract(TensorAdd(predArea, targArea), interArea), eps);
        var iou = TensorDivide(interArea, unionArea);

        // Center distance squared
        var half = numOps.FromDouble(0.5);
        var pcx = TensorMultiplyScalar(TensorAdd(px1, px2), half);
        var pcy = TensorMultiplyScalar(TensorAdd(py1, py2), half);
        var tcx = TensorMultiplyScalar(TensorAdd(tx1, tx2), half);
        var tcy = TensorMultiplyScalar(TensorAdd(ty1, ty2), half);
        var dx = TensorSubtract(pcx, tcx);
        var dy = TensorSubtract(pcy, tcy);
        var centerDistSq = TensorAdd(TensorMultiply(dx, dx), TensorMultiply(dy, dy));

        // Enclosing diagonal squared
        var encX1 = TensorNegate(TensorMax(TensorNegate(px1), TensorNegate(tx1)));
        var encY1 = TensorNegate(TensorMax(TensorNegate(py1), TensorNegate(ty1)));
        var encX2 = TensorMax(px2, tx2);
        var encY2 = TensorMax(py2, ty2);
        var encDx = TensorSubtract(encX2, encX1);
        var encDy = TensorSubtract(encY2, encY1);
        var diagSq = TensorAddScalar(TensorAdd(TensorMultiply(encDx, encDx), TensorMultiply(encDy, encDy)), eps);

        // DIoU = IoU - centerDistSq / diagSq
        var penalty = TensorDivide(centerDistSq, diagSq);
        var diou = TensorSubtract(iou, penalty);

        // Reshape from [N,1] to [N]
        return Reshape(ScalarMinusTensor(numOps.One, diou), new[] { n });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCIoULoss<T>(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted == null) throw new ArgumentNullException(nameof(predicted));
        if (target == null) throw new ArgumentNullException(nameof(target));
        if (predicted.Shape.Length != 2 || predicted.Shape[1] != 4)
            throw new ArgumentException("Predicted must be [N, 4] in (x1, y1, x2, y2) format.", nameof(predicted));
        if (target.Shape.Length != 2 || target.Shape[1] != 4)
            throw new ArgumentException("Target must be [N, 4] in (x1, y1, x2, y2) format.", nameof(target));
        if (target.Shape[0] != predicted.Shape[0])
            throw new ArgumentException("Target batch size must match predicted batch size.", nameof(target));

        var numOps = MathHelper.GetNumericOperations<T>();
        int n = predicted.Shape[0];

        var px1 = TensorSlice(predicted, new[] { 0, 0 }, new[] { n, 1 });
        var py1 = TensorSlice(predicted, new[] { 0, 1 }, new[] { n, 1 });
        var px2 = TensorSlice(predicted, new[] { 0, 2 }, new[] { n, 1 });
        var py2 = TensorSlice(predicted, new[] { 0, 3 }, new[] { n, 1 });

        var tx1 = TensorSlice(target, new[] { 0, 0 }, new[] { n, 1 });
        var ty1 = TensorSlice(target, new[] { 0, 1 }, new[] { n, 1 });
        var tx2 = TensorSlice(target, new[] { 0, 2 }, new[] { n, 1 });
        var ty2 = TensorSlice(target, new[] { 0, 3 }, new[] { n, 1 });

        // IoU with clamped areas
        var interX1 = TensorMax(px1, tx1);
        var interY1 = TensorMax(py1, ty1);
        var interX2 = TensorNegate(TensorMax(TensorNegate(px2), TensorNegate(tx2)));
        var interY2 = TensorNegate(TensorMax(TensorNegate(py2), TensorNegate(ty2)));
        var interW = ReLU(TensorSubtract(interX2, interX1));
        var interH = ReLU(TensorSubtract(interY2, interY1));
        var interArea = TensorMultiply(interW, interH);
        var predArea = TensorMultiply(ReLU(TensorSubtract(px2, px1)), ReLU(TensorSubtract(py2, py1)));
        var targArea = TensorMultiply(ReLU(TensorSubtract(tx2, tx1)), ReLU(TensorSubtract(ty2, ty1)));
        var eps = numOps.FromDouble(1e-7);
        var unionArea = TensorAddScalar(TensorSubtract(TensorAdd(predArea, targArea), interArea), eps);
        var iou = TensorDivide(interArea, unionArea);

        // Center distance
        var half = numOps.FromDouble(0.5);
        var pcx = TensorMultiplyScalar(TensorAdd(px1, px2), half);
        var pcy = TensorMultiplyScalar(TensorAdd(py1, py2), half);
        var tcx = TensorMultiplyScalar(TensorAdd(tx1, tx2), half);
        var tcy = TensorMultiplyScalar(TensorAdd(ty1, ty2), half);
        var dx = TensorSubtract(pcx, tcx);
        var dy = TensorSubtract(pcy, tcy);
        var centerDistSq = TensorAdd(TensorMultiply(dx, dx), TensorMultiply(dy, dy));

        // Enclosing diagonal
        var encX1 = TensorNegate(TensorMax(TensorNegate(px1), TensorNegate(tx1)));
        var encY1 = TensorNegate(TensorMax(TensorNegate(py1), TensorNegate(ty1)));
        var encX2 = TensorMax(px2, tx2);
        var encY2 = TensorMax(py2, ty2);
        var encDx = TensorSubtract(encX2, encX1);
        var encDy = TensorSubtract(encY2, encY1);
        var diagSq = TensorAddScalar(TensorAdd(TensorMultiply(encDx, encDx), TensorMultiply(encDy, encDy)), eps);

        // DIoU penalty
        var distPenalty = TensorDivide(centerDistSq, diagSq);

        // Aspect ratio consistency: v = (4/pi^2) * (atan(w_t/h_t) - atan(w_p/h_p))^2
        // Use proper atan per the CIoU paper (Zheng et al., 2020)
        var predW = TensorAddScalar(ReLU(TensorSubtract(px2, px1)), eps);
        var predH = TensorAddScalar(ReLU(TensorSubtract(py2, py1)), eps);
        var targW = TensorAddScalar(ReLU(TensorSubtract(tx2, tx1)), eps);
        var targH = TensorAddScalar(ReLU(TensorSubtract(ty2, ty1)), eps);
        var predRatio = TensorDivide(predW, predH);
        var targRatio = TensorDivide(targW, targH);

        // Compute atan element-wise (no tensor atan op available, compute per-element)
        var vData = new T[n];
        var alphaData = new T[n];
        var fourOverPiSq = 4.0 / (Math.PI * Math.PI);
        for (int i = 0; i < n; i++)
        {
            double predAtanVal = Math.Atan(numOps.ToDouble(predRatio[i]));
            double targAtanVal = Math.Atan(numOps.ToDouble(targRatio[i]));
            double atanDiff = targAtanVal - predAtanVal;
            double vVal = fourOverPiSq * atanDiff * atanDiff;
            vData[i] = numOps.FromDouble(vVal);
            // alpha = v / (1 - IoU + v + eps), detached from gradient per CIoU paper
            double iouVal = numOps.ToDouble(iou[i]);
            alphaData[i] = numOps.FromDouble(vVal / (1.0 - iouVal + vVal + 1e-7));
        }
        var v = new Tensor<T>(vData, new[] { n, 1 });
        // alpha is detached from gradient (constant w.r.t. backward)
        var alpha = StopGradient(new Tensor<T>(alphaData, new[] { n, 1 }));

        // CIoU = IoU - distPenalty - alpha * v
        var aspectPenalty = TensorMultiply(alpha, v);
        var ciou = TensorSubtract(TensorSubtract(iou, distPenalty), aspectPenalty);

        // Reshape from [N,1] to [N]
        return Reshape(ScalarMinusTensor(numOps.One, ciou), new[] { n });
    }

    private static Memory<float> AsFloatMemory<T>(Memory<T> data)
    {
        return Unsafe.As<Memory<T>, Memory<float>>(ref data);
    }

    private static Memory<double> AsDoubleMemory<T>(Memory<T> data)
    {
        return Unsafe.As<Memory<T>, Memory<double>>(ref data);
    }

    #region Hyperbolic Manifold Operations

    private static readonly CpuHyperbolicManifoldEngine _hyperbolicEngine = CpuHyperbolicManifoldEngine.Instance;

    /// <inheritdoc />
    public Vector<T> PoincareExpMap<T>(Vector<T> basePoint, Vector<T> tangentVector, T curvature)
        => _hyperbolicEngine.PoincareExpMap(basePoint, tangentVector, curvature);

    /// <inheritdoc />
    public Vector<T> PoincareLogMap<T>(Vector<T> basePoint, Vector<T> targetPoint, T curvature)
        => _hyperbolicEngine.PoincareLogMap(basePoint, targetPoint, curvature);

    /// <inheritdoc />
    public Vector<T> MobiusAdd<T>(Vector<T> x, Vector<T> y, T curvature)
        => _hyperbolicEngine.MobiusAdd(x, y, curvature);

    /// <inheritdoc />
    public T PoincareDistance<T>(Vector<T> x, Vector<T> y, T curvature)
        => _hyperbolicEngine.PoincareDistance(x, y, curvature);

    /// <inheritdoc />
    public Vector<T> PoincareParallelTransport<T>(Vector<T> x, Vector<T> y, Vector<T> v, T curvature)
        => _hyperbolicEngine.PoincareParallelTransport(x, y, v, curvature);

    /// <inheritdoc />
    public Vector<T> PoincareProject<T>(Vector<T> point, T curvature, T epsilon)
        => _hyperbolicEngine.PoincareProject(point, curvature, epsilon);

    /// <inheritdoc />
    public Vector<T> HyperboloidExpMap<T>(Vector<T> basePoint, Vector<T> tangentVector, T curvature)
        => _hyperbolicEngine.HyperboloidExpMap(basePoint, tangentVector, curvature);

    /// <inheritdoc />
    public Vector<T> HyperboloidLogMap<T>(Vector<T> basePoint, Vector<T> targetPoint, T curvature)
        => _hyperbolicEngine.HyperboloidLogMap(basePoint, targetPoint, curvature);

    /// <inheritdoc />
    public T HyperboloidDistance<T>(Vector<T> x, Vector<T> y, T curvature)
        => _hyperbolicEngine.HyperboloidDistance(x, y, curvature);

    /// <inheritdoc />
    public Vector<T> HyperboloidProject<T>(Vector<T> point, T curvature)
        => _hyperbolicEngine.HyperboloidProject(point, curvature);

    /// <inheritdoc />
    public Vector<T> PoincareToHyperboloid<T>(Vector<T> poincarePoint, T curvature)
        => _hyperbolicEngine.PoincareToHyperboloid(poincarePoint, curvature);

    /// <inheritdoc />
    public Vector<T> HyperboloidToPoincare<T>(Vector<T> hyperboloidPoint, T curvature)
        => _hyperbolicEngine.HyperboloidToPoincare(hyperboloidPoint, curvature);

    /// <inheritdoc />
    public Matrix<T> PoincareExpMapBatch<T>(Matrix<T> basePoints, Matrix<T> tangentVectors, T curvature)
        => _hyperbolicEngine.PoincareExpMapBatch(basePoints, tangentVectors, curvature);

    /// <inheritdoc />
    public Vector<T> PoincareDistanceBatch<T>(Matrix<T> x, Matrix<T> y, T curvature)
        => _hyperbolicEngine.PoincareDistanceBatch(x, y, curvature);

    #endregion

    #region Advanced Algebra Operations

    private static readonly CpuAdvancedAlgebraEngine _algebraEngine = CpuAdvancedAlgebraEngine.Instance;

    /// <inheritdoc />
    public Octonion<T>[] OctonionMultiplyBatch<T>(Octonion<T>[] left, Octonion<T>[] right)
        => _algebraEngine.OctonionMultiplyBatch(left, right);

    /// <inheritdoc />
    public Octonion<T>[] OctonionAddBatch<T>(Octonion<T>[] left, Octonion<T>[] right)
        => _algebraEngine.OctonionAddBatch(left, right);

    /// <inheritdoc />
    public Octonion<T>[] OctonionConjugateBatch<T>(Octonion<T>[] octonions)
        => _algebraEngine.OctonionConjugateBatch(octonions);

    /// <inheritdoc />
    public T[] OctonionNormBatch<T>(Octonion<T>[] octonions)
        => _algebraEngine.OctonionNormBatch(octonions);

    /// <inheritdoc />
    public Octonion<T>[,] OctonionMatMul<T>(Octonion<T>[,] input, Octonion<T>[,] weight)
        => _algebraEngine.OctonionMatMul(input, weight);

    /// <inheritdoc />
    public Multivector<T>[] GeometricProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right)
        => _algebraEngine.GeometricProductBatch(left, right);

    /// <inheritdoc />
    public Multivector<T>[] WedgeProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right)
        => _algebraEngine.WedgeProductBatch(left, right);

    /// <inheritdoc />
    public Multivector<T>[] InnerProductBatch<T>(Multivector<T>[] left, Multivector<T>[] right)
        => _algebraEngine.InnerProductBatch(left, right);

    /// <inheritdoc />
    public Multivector<T>[] MultivectorAddBatch<T>(Multivector<T>[] left, Multivector<T>[] right)
        => _algebraEngine.MultivectorAddBatch(left, right);

    /// <inheritdoc />
    public Multivector<T>[] MultivectorReverseBatch<T>(Multivector<T>[] multivectors)
        => _algebraEngine.MultivectorReverseBatch(multivectors);

    /// <inheritdoc />
    public Multivector<T>[] GradeProjectBatch<T>(Multivector<T>[] multivectors, int grade)
        => _algebraEngine.GradeProjectBatch(multivectors, grade);

    /// <inheritdoc />
    public So3<T>[] So3ExpBatch<T>(So3Group<T> group, Vector<T>[] tangentVectors)
        => _algebraEngine.So3ExpBatch(group, tangentVectors);

    /// <inheritdoc />
    public Vector<T>[] So3LogBatch<T>(So3Group<T> group, So3<T>[] rotations)
        => _algebraEngine.So3LogBatch(group, rotations);

    /// <inheritdoc />
    public So3<T>[] So3ComposeBatch<T>(So3Group<T> group, So3<T>[] left, So3<T>[] right)
        => _algebraEngine.So3ComposeBatch(group, left, right);

    /// <inheritdoc />
    public Se3<T>[] Se3ExpBatch<T>(Se3Group<T> group, Vector<T>[] tangentVectors)
        => _algebraEngine.Se3ExpBatch(group, tangentVectors);

    /// <inheritdoc />
    public Vector<T>[] Se3LogBatch<T>(Se3Group<T> group, Se3<T>[] transforms)
        => _algebraEngine.Se3LogBatch(group, transforms);

    /// <inheritdoc />
    public Se3<T>[] Se3ComposeBatch<T>(Se3Group<T> group, Se3<T>[] left, Se3<T>[] right)
        => _algebraEngine.Se3ComposeBatch(group, left, right);

    /// <inheritdoc />
    public Matrix<T>[] So3AdjointBatch<T>(So3Group<T> group, So3<T>[] rotations)
        => _algebraEngine.So3AdjointBatch(group, rotations);

    /// <inheritdoc />
    public Tensor<T> OctonionMatMulTensor<T>(Tensor<T> input, Tensor<T> weight)
    {
        if (input.Rank != 3 || input._shape[2] != 8)
            throw new ArgumentException($"Input must be rank-3 with last dim 8, got shape [{string.Join(", ", input._shape)}].", nameof(input));
        if (weight.Rank != 3 || weight._shape[2] != 8)
            throw new ArgumentException($"Weight must be rank-3 with last dim 8, got shape [{string.Join(", ", weight._shape)}].", nameof(weight));
        if (input._shape[1] != weight._shape[1])
            throw new ArgumentException($"Input features ({input._shape[1]}) must match weight input dimension ({weight._shape[1]}).");

        var numOps = MathHelper.GetNumericOperations<T>();

        // input: [batch, inputFeatures, 8], weight: [outputFeatures, inputFeatures, 8]
        int batch = input._shape[0];
        int inputFeatures = input._shape[1];
        int outputFeatures = weight._shape[0];

        var output = new Tensor<T>(new[] { batch, outputFeatures, 8 });

        // Pre-allocate reusable buffers outside inner loops to avoid GC pressure.
        // OctonionMultiply writes into prod; a/w are loaded per iteration.
        var accum = new T[8];
        var a = new T[8];
        var w = new T[8];
        var prod = new T[8];

        for (int b = 0; b < batch; b++)
        {
            for (int o = 0; o < outputFeatures; o++)
            {
                for (int c = 0; c < 8; c++) accum[c] = default!;

                for (int i = 0; i < inputFeatures; i++)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        a[c] = input[b, i, c];
                        w[c] = weight[o, i, c];
                    }

                    // Octonion multiplication (Cayley-Dickson): result = w * a
                    // Matches CpuAdvancedAlgebraEngine.OctonionMatMul which uses weight * input.
                    // Octonion multiply is non-commutative, so order matters.
                    OctonionMultiplyInPlace(w, a, prod, numOps);

                    for (int c = 0; c < 8; c++)
                        accum[c] = numOps.Add(accum[c], prod[c]);
                }

                for (int c = 0; c < 8; c++)
                    output[b, o, c] = accum[c];
            }
        }

        DifferentiableOps.RecordBinary("OctonionMatMulTensor", output, input, weight, BackwardFunctions<T>.MatMulBackward);
        return output;
    }

    /// <inheritdoc />
    public Tensor<T> OctonionAddTensor<T>(Tensor<T> a, Tensor<T> b)
    {
        // Element-wise addition — last dimension is 8 (octonion components)
        return TensorAdd(a, b);
    }

    /// <summary>
    /// Multiplies two octonions represented as T[8] arrays using the Cayley-Dickson construction.
    /// </summary>
    private static T[] OctonionMultiply<T>(T[] a, T[] b, INumericOperations<T> ops)
    {
        // Cayley-Dickson: split each octonion into two quaternion halves
        // a = (a0..a3, a4..a7) = (p, q)
        // b = (b0..b3, b4..b7) = (r, s)
        // a*b = (p*r - conj(s)*q, s*p + q*conj(r))

        // Quaternion multiply helper
        static T[] QuatMul(T[] x, T[] y, INumericOperations<T> o)
        {
            return new T[]
            {
                o.Subtract(o.Subtract(o.Subtract(o.Multiply(x[0], y[0]), o.Multiply(x[1], y[1])), o.Multiply(x[2], y[2])), o.Multiply(x[3], y[3])),
                o.Add(o.Subtract(o.Add(o.Multiply(x[0], y[1]), o.Multiply(x[1], y[0])), o.Multiply(x[2], y[3])), o.Multiply(x[3], y[2])),
                o.Add(o.Add(o.Subtract(o.Multiply(x[0], y[2]), o.Multiply(x[1], y[3])), o.Multiply(x[2], y[0])), o.Multiply(x[3], y[1])),
                o.Subtract(o.Add(o.Add(o.Multiply(x[0], y[3]), o.Multiply(x[1], y[2])), o.Multiply(x[3], y[0])), o.Multiply(x[2], y[1]))
            };
        }

        static T[] QuatConj(T[] x, INumericOperations<T> o)
        {
            return new T[] { x[0], o.Negate(x[1]), o.Negate(x[2]), o.Negate(x[3]) };
        }

        static T[] QuatSub(T[] x, T[] y, INumericOperations<T> o)
        {
            return new T[] { o.Subtract(x[0], y[0]), o.Subtract(x[1], y[1]), o.Subtract(x[2], y[2]), o.Subtract(x[3], y[3]) };
        }

        static T[] QuatAdd(T[] x, T[] y, INumericOperations<T> o)
        {
            return new T[] { o.Add(x[0], y[0]), o.Add(x[1], y[1]), o.Add(x[2], y[2]), o.Add(x[3], y[3]) };
        }

        var p = new T[] { a[0], a[1], a[2], a[3] };
        var q = new T[] { a[4], a[5], a[6], a[7] };
        var r = new T[] { b[0], b[1], b[2], b[3] };
        var s = new T[] { b[4], b[5], b[6], b[7] };

        // a*b = (p*r - conj(s)*q, s*p + q*conj(r))
        var pr = QuatMul(p, r, ops);
        var conjS_q = QuatMul(QuatConj(s, ops), q, ops);
        var sp = QuatMul(s, p, ops);
        var q_conjR = QuatMul(q, QuatConj(r, ops), ops);

        var first = QuatSub(pr, conjS_q, ops);
        var second = QuatAdd(sp, q_conjR, ops);

        return new T[] { first[0], first[1], first[2], first[3], second[0], second[1], second[2], second[3] };
    }

    /// <summary>
    /// Zero-allocation octonion multiply that writes result into a pre-allocated destination buffer.
    /// Uses direct multiplication table instead of Cayley-Dickson decomposition to avoid
    /// intermediate quaternion allocations.
    /// </summary>
    private static void OctonionMultiplyInPlace<T>(T[] a, T[] b, T[] result, INumericOperations<T> ops)
    {
        T a0=a[0],a1=a[1],a2=a[2],a3=a[3],a4=a[4],a5=a[5],a6=a[6],a7=a[7];
        T b0=b[0],b1=b[1],b2=b[2],b3=b[3],b4=b[4],b5=b[5],b6=b[6],b7=b[7];

        // e0
        result[0] = ops.Subtract(ops.Subtract(ops.Subtract(ops.Subtract(
            ops.Subtract(ops.Subtract(ops.Subtract(ops.Multiply(a0,b0), ops.Multiply(a1,b1)),
            ops.Multiply(a2,b2)), ops.Multiply(a3,b3)), ops.Multiply(a4,b4)),
            ops.Multiply(a5,b5)), ops.Multiply(a6,b6)), ops.Multiply(a7,b7));
        // e1
        result[1] = ops.Add(ops.Subtract(ops.Add(ops.Add(
            ops.Subtract(ops.Add(ops.Add(ops.Multiply(a0,b1), ops.Multiply(a1,b0)),
            ops.Multiply(a2,b3)), ops.Multiply(a3,b2)), ops.Multiply(a4,b5)),
            ops.Multiply(a5,b4)), ops.Multiply(a6,b7)), ops.Multiply(a7,b6));
        // e2
        result[2] = ops.Subtract(ops.Subtract(ops.Add(ops.Add(
            ops.Add(ops.Subtract(ops.Add(ops.Multiply(a0,b2), ops.Multiply(a2,b0)),
            ops.Multiply(a1,b3)), ops.Multiply(a3,b1)), ops.Multiply(a4,b6)),
            ops.Multiply(a5,b7)), ops.Multiply(a6,b4)), ops.Multiply(a7,b5));
        // e3
        result[3] = ops.Subtract(ops.Add(ops.Subtract(ops.Add(
            ops.Add(ops.Subtract(ops.Add(ops.Multiply(a0,b3), ops.Multiply(a3,b0)),
            ops.Multiply(a2,b1)), ops.Multiply(a1,b2)), ops.Multiply(a4,b7)),
            ops.Multiply(a5,b6)), ops.Multiply(a6,b5)), ops.Multiply(a7,b4));
        // e4
        result[4] = ops.Add(ops.Add(ops.Add(ops.Subtract(
            ops.Subtract(ops.Subtract(ops.Add(ops.Multiply(a0,b4), ops.Multiply(a4,b0)),
            ops.Multiply(a1,b5)), ops.Multiply(a2,b6)), ops.Multiply(a3,b7)),
            ops.Multiply(a5,b1)), ops.Multiply(a6,b2)), ops.Multiply(a7,b3));
        // e5
        result[5] = ops.Add(ops.Subtract(ops.Subtract(ops.Add(
            ops.Add(ops.Subtract(ops.Add(ops.Multiply(a0,b5), ops.Multiply(a5,b0)),
            ops.Multiply(a4,b1)), ops.Multiply(a1,b4)), ops.Multiply(a3,b6)),
            ops.Multiply(a2,b7)), ops.Multiply(a6,b3)), ops.Multiply(a7,b2));
        // e6
        result[6] = ops.Subtract(ops.Add(ops.Subtract(ops.Add(
            ops.Subtract(ops.Add(ops.Add(ops.Multiply(a0,b6), ops.Multiply(a6,b0)),
            ops.Multiply(a1,b7)), ops.Multiply(a4,b2)), ops.Multiply(a2,b4)),
            ops.Multiply(a5,b3)), ops.Multiply(a3,b5)), ops.Multiply(a7,b1));
        // e7
        result[7] = ops.Add(ops.Subtract(ops.Subtract(ops.Subtract(
            ops.Add(ops.Add(ops.Add(ops.Multiply(a0,b7), ops.Multiply(a7,b0)),
            ops.Multiply(a2,b5)), ops.Multiply(a3,b4)), ops.Multiply(a1,b6)),
            ops.Multiply(a4,b3)), ops.Multiply(a5,b2)), ops.Multiply(a6,b1));
    }

    #endregion

    #region Complex Tensor Operations

    /// <inheritdoc />
    public Tensor<T> TensorComplexMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Tensor lengths must match: {a.Length} vs {b.Length}");
        if (a.Length % 2 != 0 || (a.Rank > 0 && a._shape[a.Rank - 1] % 2 != 0))
            throw new ArgumentException("Complex tensors must have even length with the last axis divisible by 2 (interleaved re/im).");

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a._shape);
        int pairs = a.Length / 2;

        for (int i = 0; i < pairs; i++)
        {
            int idx = i * 2;
            T aRe = a.GetFlat(idx), aIm = a.GetFlat(idx + 1);
            T bRe = b.GetFlat(idx), bIm = b.GetFlat(idx + 1);
            // (aRe + aIm*i) * (bRe + bIm*i) = (aRe*bRe - aIm*bIm) + (aRe*bIm + aIm*bRe)*i
            result.SetFlat(idx, ops.Subtract(ops.Multiply(aRe, bRe), ops.Multiply(aIm, bIm)));
            result.SetFlat(idx + 1, ops.Add(ops.Multiply(aRe, bIm), ops.Multiply(aIm, bRe)));
        }

        Autodiff.DifferentiableOps.RecordIfActive("ComplexMultiply", result,
            new[] { a, b }, Autodiff.BackwardFunctions<T>.ComplexMultiplyBackward);

        return result;
    }

    /// <inheritdoc />
    public Tensor<T> TensorComplexConjugate<T>(Tensor<T> a)
    {
        if (a.Length % 2 != 0)
            throw new ArgumentException("Complex tensors must have even length (interleaved re/im).");

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(a._shape);

        for (int i = 0; i < a.Length; i++)
        {
            result.SetFlat(i, (i % 2 == 1) ? ops.Negate(a.GetFlat(i)) : a.GetFlat(i));
        }

        Autodiff.DifferentiableOps.RecordIfActive("ComplexConjugate", result,
            new[] { a }, Autodiff.BackwardFunctions<T>.ComplexConjugateBackward);

        return result;
    }

    /// <inheritdoc />
    public Tensor<T> TensorComplexMagnitude<T>(Tensor<T> a)
    {
        if (a.Length % 2 != 0)
            throw new ArgumentException("Complex tensors must have even length (interleaved re/im).");

        var ops = MathHelper.GetNumericOperations<T>();
        int pairs = a.Length / 2;
        var result = new Tensor<T>(new[] { pairs });

        for (int i = 0; i < pairs; i++)
        {
            int idx = i * 2;
            T re = a.GetFlat(idx), im = a.GetFlat(idx + 1);
            T magSq = ops.Add(ops.Multiply(re, re), ops.Multiply(im, im));
            result.SetFlat(i, ops.Sqrt(magSq));
        }

        Autodiff.DifferentiableOps.RecordIfActive("ComplexMagnitude", result,
            new[] { a }, Autodiff.BackwardFunctions<T>.ComplexMagnitudeBackward,
            new object[] { a });

        return result;
    }

    #endregion

    #region CTC Loss

    /// <inheritdoc />
    public Tensor<T> TensorCTCLoss<T>(Tensor<T> logProbs, Tensor<int> targets, int[] inputLengths, int[] targetLengths, int blank = 0)
    {
        if (logProbs.Rank != 3)
            throw new ArgumentException("logProbs must be 3D [T, N, C].");

        var ops = MathHelper.GetNumericOperations<T>();
        int maxT = logProbs._shape[0];
        int batchSize = logProbs._shape[1];
        int numClasses = logProbs._shape[2];

        if (inputLengths.Length != batchSize || targetLengths.Length != batchSize)
            throw new ArgumentException("inputLengths and targetLengths must have length == batchSize.");

        if (blank < 0 || blank >= numClasses)
            throw new ArgumentOutOfRangeException(nameof(blank), $"Blank index {blank} must be in [0, {numClasses}).");

        // Validate target bounds
        int totalTargets = 0;
        for (int n = 0; n < batchSize; n++)
            totalTargets += targetLengths[n];
        if (totalTargets > targets.Length)
            throw new ArgumentException($"Sum of targetLengths ({totalTargets}) exceeds targets length ({targets.Length}).");

        var losses = new Tensor<T>(new[] { batchSize });

        // Forward-backward algorithm per batch element
        int targetOffset = 0;
        for (int n = 0; n < batchSize; n++)
        {
            int T_n = inputLengths[n];
            int U_n = targetLengths[n];

            if (T_n < 0 || T_n > maxT)
                throw new ArgumentOutOfRangeException(nameof(inputLengths), $"inputLengths[{n}]={T_n} must be in [0, {maxT}].");
            if (U_n < 0)
                throw new ArgumentOutOfRangeException(nameof(targetLengths), $"targetLengths[{n}]={U_n} must be non-negative.");

            int S = 2 * U_n + 1; // expanded label length with blanks

            // Build expanded label sequence: blank, l1, blank, l2, blank, ...
            var expandedLabels = new int[S];
            for (int s = 0; s < S; s++)
            {
                int label = (s % 2 == 0) ? blank : targets.GetFlat(targetOffset + s / 2);
                if (label < 0 || label >= numClasses)
                    throw new ArgumentOutOfRangeException(nameof(targets), $"Target label {label} must be in [0, {numClasses}).");
                expandedLabels[s] = label;
            }

            // Alpha (forward) pass — log domain for numerical stability
            var alpha = new double[T_n, S];
            double negInf = double.NegativeInfinity;
            for (int t = 0; t < T_n; t++)
                for (int s = 0; s < S; s++)
                    alpha[t, s] = negInf;

            // Initialize
            alpha[0, 0] = ops.ToDouble(logProbs[0, n, expandedLabels[0]]);
            if (S > 1)
                alpha[0, 1] = ops.ToDouble(logProbs[0, n, expandedLabels[1]]);

            // Forward recursion
            for (int t = 1; t < T_n; t++)
            {
                for (int s = 0; s < S; s++)
                {
                    double logProbTS = ops.ToDouble(logProbs[t, n, expandedLabels[s]]);
                    double prevSum = alpha[t - 1, s];
                    if (s >= 1)
                        prevSum = LogSumExp(prevSum, alpha[t - 1, s - 1]);
                    if (s >= 2 && expandedLabels[s] != blank && expandedLabels[s] != expandedLabels[s - 2])
                        prevSum = LogSumExp(prevSum, alpha[t - 1, s - 2]);
                    alpha[t, s] = prevSum + logProbTS;
                }
            }

            // Total log-probability
            double logProb = alpha[T_n - 1, S - 1];
            if (S >= 2)
                logProb = LogSumExp(logProb, alpha[T_n - 1, S - 2]);

            // CTC loss = -log P(labels | input)
            losses.SetFlat(n, ops.FromDouble(-logProb));
            targetOffset += U_n;
        }

        Autodiff.DifferentiableOps.RecordIfActive("CTCLoss", losses,
            new[] { logProbs }, Autodiff.BackwardFunctions<T>.CTCLossBackward,
            new object[] { logProbs, targets, inputLengths, targetLengths, blank });

        return losses;
    }

    private static double LogSumExp(double a, double b)
    {
        if (double.IsNegativeInfinity(a)) return b;
        if (double.IsNegativeInfinity(b)) return a;
        double max = Math.Max(a, b);
        return max + Math.Log(Math.Exp(a - max) + Math.Exp(b - max));
    }

    #endregion
}
