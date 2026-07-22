#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Experimental host-facing sparse engine backed by the exact direct-PTX
/// #852 specialization. Unsupported dtypes, shapes, thresholds, axes, active
/// tapes, architectures, or disabled feature gates fall back to the canonical
/// CPU engine without changing public semantics.
/// </summary>
/// <remarks>
/// The caller owns the supplied <see cref="CudaBackend"/>. This first surface
/// proves every <see cref="ISparseEngine"/> operation can reach its own PTX
/// module; persistent sparse index/value residency is a later promotion gate.
/// </remarks>
public sealed class CudaPtxSparseEngine : ISparseEngine
{
    private readonly CudaBackend _backend;
    private readonly ISparseEngine _fallback;

    public CudaPtxSparseEngine(CudaBackend backend, ISparseEngine? fallback = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _fallback = fallback ?? CpuSparseEngine.Instance;
    }

    public Vector<T> SpMV<T>(SparseTensor<T> sparse, Vector<T> dense)
    {
        if (!CanUse(sparse) || dense is null || dense.Length != PtxSparseEngineF32Kernel.Inner)
            return _fallback.SpMV(sparse, dense!);
        SparseTensor<float> csr = AsFloat(sparse).ToCsr();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.SpMV,
            F(csr.Values), I(csr.ColumnIndices), I(csr.RowPointers), F(((Vector<float>)(object)dense).AsSpan().ToArray()), O(PtxSparseEngineF32Kernel.Rows));
        return result is null ? _fallback.SpMV(sparse, dense) : (Vector<T>)(object)new Vector<float>((float[])result[4]!);
    }

    public Vector<T> SpMVTranspose<T>(SparseTensor<T> sparse, Vector<T> dense)
    {
        if (!CanUse(sparse) || dense is null || dense.Length != PtxSparseEngineF32Kernel.Rows)
            return _fallback.SpMVTranspose(sparse, dense!);
        SparseTensor<float> csr = AsFloat(sparse).ToCsr();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.SpMVTranspose,
            F(csr.Values), I(csr.ColumnIndices), I(csr.RowPointers), F(((Vector<float>)(object)dense).AsSpan().ToArray()), O(PtxSparseEngineF32Kernel.Inner));
        return result is null ? _fallback.SpMVTranspose(sparse, dense) : (Vector<T>)(object)new Vector<float>((float[])result[4]!);
    }

    public Matrix<T> SpMM<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        if (!CanUse(sparse) || !IsProductRhs(dense)) return _fallback.SpMM(sparse, dense);
        float[]? output = RunCsrProduct(DirectPtxSparseEngineOperation.SpMM, AsFloat(sparse), ((Matrix<float>)(object)dense).AsSpan().ToArray());
        return output is null ? _fallback.SpMM(sparse, dense) : (Matrix<T>)(object)ProductMatrix(output);
    }

    public SparseTensor<T> SpSpMM<T>(SparseTensor<T> a, SparseTensor<T> b)
    {
        if (!CanUse(a) || !CanUse(b)) return _fallback.SpSpMM(a, b);
        float[]? output = RunSparseProduct(DirectPtxSparseEngineOperation.SpSpMM, AsFloat(a), AsFloat(b));
        if (output is null) return _fallback.SpSpMM(a, b);
        SparseTensor<float> sparse = CpuSparseEngine.Instance.DenseToSparse(DenseMatrix(output), 0f);
        return (SparseTensor<T>)(object)sparse;
    }

    public Matrix<T> AddSparseDense<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        if (!CanUse(sparse) || !IsCanonicalDense(dense)) return _fallback.AddSparseDense(sparse, dense);
        SparseTensor<float> coo = AsFloat(sparse).ToCoo();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.AddSparseDense,
            I(coo.RowIndices), I(coo.ColumnIndices), F(coo.Values), F(((Matrix<float>)(object)dense).AsSpan().ToArray()), O(PtxSparseEngineF32Kernel.DenseElements));
        return result is null ? _fallback.AddSparseDense(sparse, dense) : (Matrix<T>)(object)DenseMatrix((float[])result[4]!);
    }

    public SparseTensor<T> MultiplySparseDense<T>(SparseTensor<T> sparse, Matrix<T> dense)
    {
        if (!CanUse(sparse) || !IsCanonicalDense(dense)) return _fallback.MultiplySparseDense(sparse, dense);
        SparseTensor<float> coo = AsFloat(sparse).ToCoo();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.MultiplySparseDense,
            I(coo.RowIndices), I(coo.ColumnIndices), F(coo.Values), F(((Matrix<float>)(object)dense).AsSpan().ToArray()), O(PtxSparseEngineF32Kernel.NonZeros));
        if (result is null) return _fallback.MultiplySparseDense(sparse, dense);
        return (SparseTensor<T>)(object)new SparseTensor<float>(coo.Rows, coo.Columns,
            (int[])coo.RowIndices.Clone(), (int[])coo.ColumnIndices.Clone(), (float[])result[4]!);
    }

    public Vector<T> SparseGather<T>(Matrix<T> source, SparseTensor<T> indices)
    {
        if (!CanUse(indices) || !IsCanonicalDense(source)) return _fallback.SparseGather(source, indices);
        SparseTensor<float> coo = AsFloat(indices).ToCoo();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.SparseGather,
            I(coo.RowIndices), I(coo.ColumnIndices), F(((Matrix<float>)(object)source).AsSpan().ToArray()), O(PtxSparseEngineF32Kernel.NonZeros));
        return result is null ? _fallback.SparseGather(source, indices) : (Vector<T>)(object)new Vector<float>((float[])result[3]!);
    }

    public Matrix<T> SparseScatter<T>(Vector<T> values, SparseTensor<T> indices, int rows, int cols)
    {
        if (!CanUse(indices) || values is null || values.Length != PtxSparseEngineF32Kernel.NonZeros ||
            rows != PtxSparseEngineF32Kernel.Rows || cols != PtxSparseEngineF32Kernel.Inner)
            return _fallback.SparseScatter(values!, indices, rows, cols);
        SparseTensor<float> coo = AsFloat(indices).ToCoo();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.SparseScatter,
            I(coo.RowIndices), I(coo.ColumnIndices), F(((Vector<float>)(object)values).AsSpan().ToArray()), O(PtxSparseEngineF32Kernel.DenseElements));
        return result is null ? _fallback.SparseScatter(values, indices, rows, cols) : (Matrix<T>)(object)DenseMatrix((float[])result[3]!);
    }

    public void SparseScatterAdd<T>(Vector<T> values, (int[] rows, int[] cols) indices, Matrix<T> target)
    {
        if (!CanUse<T>() || values is null || indices.rows is null || indices.cols is null ||
            values.Length != PtxSparseEngineF32Kernel.NonZeros || indices.rows.Length != PtxSparseEngineF32Kernel.NonZeros ||
            indices.cols.Length != PtxSparseEngineF32Kernel.NonZeros ||
            !CoordinatesInRange(indices.rows, indices.cols, PtxSparseEngineF32Kernel.Rows, PtxSparseEngineF32Kernel.Inner) ||
            !IsCanonicalDense(target))
        {
            _fallback.SparseScatterAdd(values!, indices, target); return;
        }
        Matrix<float> targetFloat = (Matrix<float>)(object)target;
        object?[]? result = Execute(DirectPtxSparseEngineOperation.SparseScatterAdd,
            I(indices.rows), I(indices.cols), F(((Vector<float>)(object)values).AsSpan().ToArray()), IO(targetFloat.AsSpan().ToArray()));
        if (result is null) { _fallback.SparseScatterAdd(values, indices, target); return; }
        ((float[])result[3]!).AsSpan().CopyTo(targetFloat.AsWritableSpan());
    }

    public Matrix<T> SparseToDense<T>(SparseTensor<T> sparse)
    {
        if (!CanUse(sparse)) return _fallback.SparseToDense(sparse);
        SparseTensor<float> coo = AsFloat(sparse).ToCoo();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.SparseToDense,
            I(coo.RowIndices), I(coo.ColumnIndices), F(coo.Values), O(PtxSparseEngineF32Kernel.DenseElements));
        return result is null ? _fallback.SparseToDense(sparse) : (Matrix<T>)(object)DenseMatrix((float[])result[3]!);
    }

    public SparseTensor<T> DenseToSparse<T>(Matrix<T> dense, T threshold)
    {
        if (!CanUse<T>() || !IsCanonicalDense(dense) ||
            BitConverter.SingleToInt32Bits((float)(object)threshold!) != 0)
            return _fallback.DenseToSparse(dense, threshold);
        object?[]? result = Execute(DirectPtxSparseEngineOperation.DenseToSparse,
            F(((Matrix<float>)(object)dense).AsSpan().ToArray()), OI(PtxSparseEngineF32Kernel.DenseElements),
            OI(PtxSparseEngineF32Kernel.DenseElements), O(PtxSparseEngineF32Kernel.DenseElements), OI(PtxSparseEngineF32Kernel.DenseElements));
        if (result is null) return _fallback.DenseToSparse(dense, threshold);
        SparseTensor<float> sparse = Compact(PtxSparseEngineF32Kernel.Rows, PtxSparseEngineF32Kernel.Inner,
            (int[])result[1]!, (int[])result[2]!, (float[])result[3]!, (int[])result[4]!);
        return (SparseTensor<T>)(object)sparse;
    }

    public SparseTensor<T> Coalesce<T>(SparseTensor<T> sparse)
    {
        if (!CanUse(sparse)) return _fallback.Coalesce(sparse);
        SparseTensor<float> coo = AsFloat(sparse).ToCoo();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.Coalesce,
            I(coo.RowIndices), I(coo.ColumnIndices), F(coo.Values), OI(PtxSparseEngineF32Kernel.NonZeros),
            OI(PtxSparseEngineF32Kernel.NonZeros), O(PtxSparseEngineF32Kernel.NonZeros), OI(PtxSparseEngineF32Kernel.NonZeros));
        if (result is null) return _fallback.Coalesce(sparse);
        return (SparseTensor<T>)(object)Compact(coo.Rows, coo.Columns,
            (int[])result[3]!, (int[])result[4]!, (float[])result[5]!, (int[])result[6]!).Coalesce();
    }

    public SparseTensor<T> SparseTranspose<T>(SparseTensor<T> sparse)
    {
        if (!CanUse(sparse)) return _fallback.SparseTranspose(sparse);
        SparseTensor<float> coo = AsFloat(sparse).ToCoo();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.SparseTranspose,
            I(coo.RowIndices), I(coo.ColumnIndices), F(coo.Values), OI(PtxSparseEngineF32Kernel.NonZeros),
            OI(PtxSparseEngineF32Kernel.NonZeros), O(PtxSparseEngineF32Kernel.NonZeros));
        if (result is null) return _fallback.SparseTranspose(sparse);
        var transposed = new SparseTensor<float>(coo.Columns, coo.Rows,
            (int[])result[3]!, (int[])result[4]!, (float[])result[5]!);
        SparseTensor<float> formatted = sparse.Format switch
        {
            SparseStorageFormat.Csr => transposed.ToCsc(),
            SparseStorageFormat.Csc => transposed.ToCsr(),
            _ => transposed
        };
        return (SparseTensor<T>)(object)formatted;
    }

    public Tensor<T> SparseMatMul<T>(SparseTensor<T> a, Tensor<T> b) =>
        SparseMatMulCore(a, b, DirectPtxSparseEngineOperation.SparseMatMul,
            () => _fallback.SparseMatMul(a, b));

    public Tensor<T> SparseMatMulPatternPreserving<T>(SparseTensor<T> a, Tensor<T> b) =>
        SparseMatMulCore(a, b, DirectPtxSparseEngineOperation.SparseMatMulPatternPreserving,
            () => _fallback.SparseMatMulPatternPreserving(a, b));

    public Tensor<T> SparseAddMM<T>(Tensor<T> c, SparseTensor<T> a, Tensor<T> b, T alpha, T beta)
    {
        if (!CanUseNoTape(a) || !IsProductTensor(b) || !IsProductTensor(c) ||
            BitConverter.SingleToInt32Bits((float)(object)alpha!) != BitConverter.SingleToInt32Bits(1f) ||
            BitConverter.SingleToInt32Bits((float)(object)beta!) != BitConverter.SingleToInt32Bits(1f))
            return _fallback.SparseAddMM(c, a, b, alpha, beta);
        SparseTensor<float> csr = AsFloat(a).ToCsr();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.SparseAddMM,
            F(csr.Values), I(csr.ColumnIndices), I(csr.RowPointers), F(((Tensor<float>)(object)b).ToArray()),
            F(((Tensor<float>)(object)c).ToArray()), O(PtxSparseEngineF32Kernel.ProductElements));
        return result is null ? _fallback.SparseAddMM(c, a, b, alpha, beta) :
            (Tensor<T>)(object)new Tensor<float>((float[])result[5]!, new[] { PtxSparseEngineF32Kernel.Rows, PtxSparseEngineF32Kernel.Columns });
    }

    public Tensor<T> SparseSampledAddMM<T>(SparseTensor<T> pattern, Tensor<T> a, Tensor<T> b, Tensor<T> c, T alpha, T beta)
    {
        if (!CanUseNoTape<T>() || !CanUsePattern(pattern) || !IsCanonicalTensor(a) || !IsProductTensor(b) || !IsProductTensor(c) ||
            BitConverter.SingleToInt32Bits((float)(object)alpha!) != BitConverter.SingleToInt32Bits(1f) ||
            BitConverter.SingleToInt32Bits((float)(object)beta!) != BitConverter.SingleToInt32Bits(1f))
            return _fallback.SparseSampledAddMM(pattern, a, b, c, alpha, beta);
        SparseTensor<float> coo = AsFloat(pattern).ToCoo();
        object?[]? result = Execute(DirectPtxSparseEngineOperation.SparseSampledAddMM,
            I(coo.RowIndices), I(coo.ColumnIndices), F(((Tensor<float>)(object)a).ToArray()),
            F(((Tensor<float>)(object)b).ToArray()), F(((Tensor<float>)(object)c).ToArray()), O(PtxSparseEngineF32Kernel.ProductElements));
        return result is null ? _fallback.SparseSampledAddMM(pattern, a, b, c, alpha, beta) :
            (Tensor<T>)(object)new Tensor<float>((float[])result[5]!, new[] { PtxSparseEngineF32Kernel.Rows, PtxSparseEngineF32Kernel.Columns });
    }

    public Tensor<T> SparseSpGeMM<T>(SparseTensor<T> a, SparseTensor<T> b)
    {
        if (!CanUseNoTape(a) || !CanUse(b)) return _fallback.SparseSpGeMM(a, b);
        float[]? output = RunSparseProduct(DirectPtxSparseEngineOperation.SparseSpGeMM, AsFloat(a), AsFloat(b));
        return output is null ? _fallback.SparseSpGeMM(a, b) :
            (Tensor<T>)(object)new Tensor<float>(output, new[] { PtxSparseEngineF32Kernel.Rows, PtxSparseEngineF32Kernel.Inner });
    }

    public Tensor<T> SparseSum<T>(SparseTensor<T> a, int? axis = null) =>
        SparseScalar(a, axis, DirectPtxSparseEngineOperation.SparseSum, () => _fallback.SparseSum(a, axis));

    public Tensor<T> SparseMean<T>(SparseTensor<T> a, int? axis = null) =>
        SparseScalar(a, axis, DirectPtxSparseEngineOperation.SparseMean, () => _fallback.SparseMean(a, axis));

    public Tensor<T> SparseSoftmax<T>(SparseTensor<T> a) => SparseSoftmaxCore(
        a, DirectPtxSparseEngineOperation.SparseSoftmax, () => _fallback.SparseSoftmax(a));

    public Tensor<T> SparseLogSoftmax<T>(SparseTensor<T> a) => SparseSoftmaxCore(
        a, DirectPtxSparseEngineOperation.SparseLogSoftmax, () => _fallback.SparseLogSoftmax(a));

    private Tensor<T> SparseMatMulCore<T>(SparseTensor<T> a, Tensor<T> b,
        DirectPtxSparseEngineOperation operation, Func<Tensor<T>> fallback)
    {
        if (!CanUseNoTape(a) || !IsProductTensor(b)) return fallback();
        float[]? output = RunCsrProduct(operation, AsFloat(a), ((Tensor<float>)(object)b).ToArray());
        return output is null ? fallback() : (Tensor<T>)(object)new Tensor<float>(output,
            new[] { PtxSparseEngineF32Kernel.Rows, PtxSparseEngineF32Kernel.Columns });
    }

    private Tensor<T> SparseScalar<T>(SparseTensor<T> a, int? axis,
        DirectPtxSparseEngineOperation operation, Func<Tensor<T>> fallback)
    {
        if (!CanUseNoTape(a) || axis is not null) return fallback();
        object?[]? result = Execute(operation, F(AsFloat(a).Values), O(1));
        return result is null ? fallback() : (Tensor<T>)(object)new Tensor<float>((float[])result[1]!, new[] { 1 });
    }

    private Tensor<T> SparseSoftmaxCore<T>(SparseTensor<T> a,
        DirectPtxSparseEngineOperation operation, Func<Tensor<T>> fallback)
    {
        if (!CanUseNoTape(a)) return fallback();
        SparseTensor<float> coo = AsFloat(a).ToCoo();
        object?[]? result = Execute(operation, I(coo.RowIndices), F(coo.Values), O(PtxSparseEngineF32Kernel.NonZeros));
        if (result is null) return fallback();
        var dense = new float[PtxSparseEngineF32Kernel.DenseElements];
        float[] values = (float[])result[2]!;
        for (int i = 0; i < values.Length; i++) dense[coo.RowIndices[i] * PtxSparseEngineF32Kernel.Inner + coo.ColumnIndices[i]] = values[i];
        return (Tensor<T>)(object)new Tensor<float>(dense, new[] { PtxSparseEngineF32Kernel.Rows, PtxSparseEngineF32Kernel.Inner });
    }

    private float[]? RunCsrProduct(DirectPtxSparseEngineOperation operation, SparseTensor<float> sparse, float[] dense)
    {
        SparseTensor<float> csr = sparse.ToCsr();
        object?[]? result = Execute(operation, F(csr.Values), I(csr.ColumnIndices), I(csr.RowPointers), F(dense), O(PtxSparseEngineF32Kernel.ProductElements));
        return result is null ? null : (float[])result[4]!;
    }

    private float[]? RunSparseProduct(DirectPtxSparseEngineOperation operation, SparseTensor<float> a, SparseTensor<float> b)
    {
        SparseTensor<float> ac = a.ToCsr(), bc = b.ToCsr();
        object?[]? result = Execute(operation, F(ac.Values), I(ac.ColumnIndices), I(ac.RowPointers),
            F(bc.Values), I(bc.ColumnIndices), I(bc.RowPointers), O(PtxSparseEngineF32Kernel.DenseElements));
        return result is null ? null : (float[])result[6]!;
    }

    private object?[]? Execute(DirectPtxSparseEngineOperation operation, params HostArgument[] arguments)
    {
        var buffers = new IGpuBuffer[arguments.Length];
        try
        {
            for (int i = 0; i < arguments.Length; i++)
            {
                HostArgument arg = arguments[i];
                if (arg.FloatData is not null) buffers[i] = _backend.AllocateBuffer(arg.FloatData);
                else
                {
                    buffers[i] = _backend.AllocateBuffer(arg.Elements);
                    if (arg.IntData is not null) _backend.UploadIntBufferInPlace(arg.IntData, buffers[i]);
                }
            }
            bool launched = _backend.TryDirectPtxSparseEngine(operation, buffers[0],
                At(buffers, 1), At(buffers, 2), At(buffers, 3), At(buffers, 4), At(buffers, 5), At(buffers, 6));
            if (!launched) return null;
            var result = new object?[arguments.Length];
            for (int i = 0; i < arguments.Length; i++)
            {
                if (!arguments[i].Download) continue;
                if (arguments[i].Integer)
                {
                    byte[] bytes = _backend.DownloadByteBuffer(buffers[i], checked(arguments[i].Elements * sizeof(int)));
                    var ints = new int[arguments[i].Elements]; Buffer.BlockCopy(bytes, 0, ints, 0, bytes.Length); result[i] = ints;
                }
                else result[i] = _backend.DownloadBuffer(buffers[i]);
            }
            return result;
        }
        finally
        {
            foreach (IGpuBuffer? buffer in buffers) buffer?.Dispose();
        }
    }

    private bool CanUse<T>() => typeof(T) == typeof(float) && _backend.IsDirectPtxSparseGraphEnabled &&
        !_backend.IsStreamCapturing();
    private bool CanUse<T>(SparseTensor<T>? sparse) => CanUse<T>() && sparse is not null &&
        sparse.Rows == PtxSparseEngineF32Kernel.Rows && sparse.Columns == PtxSparseEngineF32Kernel.Inner &&
        sparse.NonZeroCount == PtxSparseEngineF32Kernel.NonZeros && HasValidCoordinates(sparse, requireUnique: false);
    private bool CanUsePattern<T>(SparseTensor<T>? sparse) => CanUse<T>() && sparse is not null &&
        sparse.Rows == PtxSparseEngineF32Kernel.Rows && sparse.Columns == PtxSparseEngineF32Kernel.Columns &&
        sparse.NonZeroCount == PtxSparseEngineF32Kernel.NonZeros && HasValidCoordinates(sparse, requireUnique: true);
    private bool CanUseNoTape<T>() => CanUse<T>() && !DifferentiableOps.IsTapeActiveForThread<T>();
    private bool CanUseNoTape<T>(SparseTensor<T>? sparse) => CanUse(sparse) && !DifferentiableOps.IsTapeActiveForThread<T>();
    private static SparseTensor<float> AsFloat<T>(SparseTensor<T> value) => (SparseTensor<float>)(object)value;
    private static bool IsCanonicalDense<T>(Matrix<T>? value) => typeof(T) == typeof(float) && value is not null &&
        value.Rows == PtxSparseEngineF32Kernel.Rows && value.Columns == PtxSparseEngineF32Kernel.Inner;
    private static bool IsProductRhs<T>(Matrix<T>? value) => typeof(T) == typeof(float) && value is not null &&
        value.Rows == PtxSparseEngineF32Kernel.Inner && value.Columns == PtxSparseEngineF32Kernel.Columns;
    private static bool IsCanonicalTensor<T>(Tensor<T>? value) => typeof(T) == typeof(float) && value is not null && value.IsContiguous &&
        value.Shape.Length == 2 && value.Shape[0] == PtxSparseEngineF32Kernel.Rows && value.Shape[1] == PtxSparseEngineF32Kernel.Inner;
    private static bool IsProductTensor<T>(Tensor<T>? value) => typeof(T) == typeof(float) && value is not null && value.IsContiguous &&
        value.Shape.Length == 2 && value.Shape[0] == PtxSparseEngineF32Kernel.Inner && value.Shape[1] == PtxSparseEngineF32Kernel.Columns;
    private static Matrix<float> DenseMatrix(float[] values) => Matrix<float>.FromMemory(values.AsMemory(), PtxSparseEngineF32Kernel.Rows, PtxSparseEngineF32Kernel.Inner);
    private static Matrix<float> ProductMatrix(float[] values) => Matrix<float>.FromMemory(values.AsMemory(), PtxSparseEngineF32Kernel.Rows, PtxSparseEngineF32Kernel.Columns);
    private static IGpuBuffer? At(IGpuBuffer[] buffers, int index) => index < buffers.Length ? buffers[index] : null;

    private static bool HasValidCoordinates<T>(SparseTensor<T> sparse, bool requireUnique)
    {
        try
        {
            SparseTensor<T> coo = sparse.ToCoo();
            if (!CoordinatesInRange(coo.RowIndices, coo.ColumnIndices, sparse.Rows, sparse.Columns)) return false;
            if (!requireUnique) return true;
            var seen = new HashSet<long>();
            for (int i = 0; i < coo.RowIndices.Length; i++)
            {
                long coordinate = ((long)coo.RowIndices[i] << 32) | (uint)coo.ColumnIndices[i];
                if (!seen.Add(coordinate)) return false;
            }
            return true;
        }
        catch (ArgumentException)
        {
            return false;
        }
        catch (IndexOutOfRangeException)
        {
            return false;
        }
    }

    private static bool CoordinatesInRange(int[] rows, int[] columns, int rowCount, int columnCount)
    {
        for (int i = 0; i < rows.Length; i++)
            if ((uint)rows[i] >= (uint)rowCount || (uint)columns[i] >= (uint)columnCount) return false;
        return true;
    }

    private static SparseTensor<float> Compact(int rows, int cols, int[] sourceRows, int[] sourceCols, float[] sourceValues, int[] flags)
    {
        int count = 0; for (int i = 0; i < flags.Length; i++) if (flags[i] != 0) count++;
        var outputRows = new int[count]; var outputCols = new int[count]; var outputValues = new float[count];
        int cursor = 0;
        for (int i = 0; i < flags.Length; i++)
        {
            if (flags[i] == 0) continue;
            outputRows[cursor] = sourceRows[i]; outputCols[cursor] = sourceCols[i]; outputValues[cursor] = sourceValues[i]; cursor++;
        }
        return new SparseTensor<float>(rows, cols, outputRows, outputCols, outputValues);
    }

    private static HostArgument F(float[] data) => new(data, null, data.Length, false, false);
    private static HostArgument I(int[] data) => new(null, data, data.Length, false, true);
    private static HostArgument O(int elements) => new(null, null, elements, true, false);
    private static HostArgument OI(int elements) => new(null, null, elements, true, true);
    private static HostArgument IO(float[] data) => new(data, null, data.Length, true, false);
    private readonly record struct HostArgument(float[]? FloatData, int[]? IntData, int Elements, bool Download, bool Integer);
}
#endif
