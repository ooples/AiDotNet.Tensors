using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;

namespace AiDotNet.Tensors.Onnx;

/// <summary>
/// Converts an ONNX <see cref="TensorProto"/> initializer into a
/// <see cref="Tensor{T}"/>. ONNX encodes initializer data in two places:
/// the typed <c>*_data</c> fields (<c>float_data</c>, <c>int32_data</c>,
/// <c>int64_data</c>, <c>double_data</c>) or the <c>raw_data</c> byte blob
/// (always little-endian per spec). This loader handles both.
/// </summary>
internal static class InitializerLoader
{
    // ONNX TensorProto.DataType enum values (from onnx.proto3).
    private const int FLOAT = 1;
    private const int UINT8 = 2;
    private const int INT8 = 3;
    private const int INT32 = 6;
    private const int INT64 = 7;
    private const int DOUBLE = 11;

    internal static Tensor<T> Load<T>(TensorProto proto) where T : unmanaged
    {
        int[] shape = new int[proto.Dims.Count];
        long total = 1;
        for (int i = 0; i < proto.Dims.Count; i++)
        {
            shape[i] = checked((int)proto.Dims[i]);
            total = checked(total * shape[i]);
        }
        if (total < 0 || total > int.MaxValue)
            throw new InvalidDataException(
                $"ONNX initializer '{proto.Name}' declares {total} elements; " +
                "too large for a single-tensor import.");
        int n = (int)total;

        // Scalar tensors carry Dims.Count == 0 but a single element.
        if (shape.Length == 0) { shape = new[] { 1 }; n = 1; }

        var tensor = new Tensor<T>(shape);
        var dst = tensor.AsWritableSpan();

        // Fast path: direct element-type match between ONNX and T.
        if (typeof(T) == typeof(float) && proto.DataType == FLOAT)
        {
            var floatDst = MemoryMarshal.Cast<T, float>(dst);
            LoadFloat(proto, floatDst, n);
            return tensor;
        }
        if (typeof(T) == typeof(double) && proto.DataType == DOUBLE)
        {
            var doubleDst = MemoryMarshal.Cast<T, double>(dst);
            LoadDouble(proto, doubleDst, n);
            return tensor;
        }

        // Cross-type: ONNX float → T=double (widening) is safe; narrowing
        // and other mixes go through the numeric-ops conversion path.
        if (typeof(T) == typeof(double) && proto.DataType == FLOAT)
        {
            var tmp = new float[n];
            LoadFloat(proto, tmp, n);
            var doubleDst = MemoryMarshal.Cast<T, double>(dst);
            for (int i = 0; i < n; i++) doubleDst[i] = tmp[i];
            return tensor;
        }
        if (typeof(T) == typeof(float) && proto.DataType == DOUBLE)
        {
            var tmp = new double[n];
            LoadDouble(proto, tmp, n);
            var floatDst = MemoryMarshal.Cast<T, float>(dst);
            for (int i = 0; i < n; i++) floatDst[i] = (float)tmp[i];
            return tensor;
        }

        // Integer initializers (attention masks, indices, shape tensors).
        // Convert to T via float intermediary.
        if (proto.DataType == INT32 || proto.DataType == INT64 ||
            proto.DataType == INT8 || proto.DataType == UINT8)
        {
            LoadInt<T>(proto, dst, n);
            return tensor;
        }

        throw new NotSupportedException(
            $"ONNX initializer '{proto.Name}' has element type {proto.DataType} " +
            $"but T = {typeof(T).Name}. Phase 1 supports FLOAT/DOUBLE/INT* → float or double conversions.");
    }

    private static void LoadFloat(TensorProto proto, Span<float> dst, int n)
    {
        if (!proto.RawData.IsEmpty)
        {
            if (proto.RawData.Length != n * sizeof(float))
                throw new InvalidDataException(
                    $"ONNX initializer '{proto.Name}' raw_data size {proto.RawData.Length} does not match " +
                    $"expected {n * sizeof(float)} bytes.");
            var src = proto.RawData.Span;
            // ONNX raw_data is always little-endian. Direct memcpy works on
            // the overwhelming majority of targets (x86/x64/ARM64 LE).
            MemoryMarshal.Cast<byte, float>(src).CopyTo(dst);
            return;
        }
        if (proto.FloatData.Count == n)
        {
            for (int i = 0; i < n; i++) dst[i] = proto.FloatData[i];
            return;
        }
        throw new InvalidDataException(
            $"ONNX initializer '{proto.Name}' has no float data: raw_data empty and float_data count " +
            $"{proto.FloatData.Count} != expected {n}.");
    }

    private static void LoadDouble(TensorProto proto, Span<double> dst, int n)
    {
        if (!proto.RawData.IsEmpty)
        {
            if (proto.RawData.Length != n * sizeof(double))
                throw new InvalidDataException(
                    $"ONNX initializer '{proto.Name}' raw_data size {proto.RawData.Length} does not match " +
                    $"expected {n * sizeof(double)} bytes.");
            MemoryMarshal.Cast<byte, double>(proto.RawData.Span).CopyTo(dst);
            return;
        }
        if (proto.DoubleData.Count == n)
        {
            for (int i = 0; i < n; i++) dst[i] = proto.DoubleData[i];
            return;
        }
        throw new InvalidDataException(
            $"ONNX initializer '{proto.Name}' has no double data.");
    }

    private static void ValidateRawDataSize(TensorProto proto, int expectedBytes)
    {
        if (proto.RawData.Length != expectedBytes)
            throw new InvalidDataException(
                $"ONNX initializer '{proto.Name}' raw_data size {proto.RawData.Length} does not match " +
                $"expected {expectedBytes} bytes.");
    }

    private static void LoadInt<T>(TensorProto proto, Span<T> dst, int n) where T : unmanaged
    {
        // Integer initializers are almost always small (shape tensors, axes,
        // masks). Unpack to long first, then convert to T via the numeric
        // ops registry so T = float/double/Half all work.
        var asLong = new long[n];
        switch (proto.DataType)
        {
            case INT32:
                if (!proto.RawData.IsEmpty)
                {
                    ValidateRawDataSize(proto, n * sizeof(int));
                    var srcI = MemoryMarshal.Cast<byte, int>(proto.RawData.Span);
                    for (int i = 0; i < n; i++) asLong[i] = srcI[i];
                }
                else
                {
                    for (int i = 0; i < n; i++) asLong[i] = proto.Int32Data[i];
                }
                break;
            case INT64:
                if (!proto.RawData.IsEmpty)
                {
                    ValidateRawDataSize(proto, n * sizeof(long));
                    var srcL = MemoryMarshal.Cast<byte, long>(proto.RawData.Span);
                    for (int i = 0; i < n; i++) asLong[i] = srcL[i];
                }
                else
                {
                    for (int i = 0; i < n; i++) asLong[i] = proto.Int64Data[i];
                }
                break;
            case INT8:
                if (!proto.RawData.IsEmpty)
                {
                    ValidateRawDataSize(proto, n * sizeof(sbyte));
                    var srcS = MemoryMarshal.Cast<byte, sbyte>(proto.RawData.Span);
                    for (int i = 0; i < n; i++) asLong[i] = srcS[i];
                }
                else
                {
                    for (int i = 0; i < n; i++) asLong[i] = proto.Int32Data[i];
                }
                break;
            case UINT8:
                if (!proto.RawData.IsEmpty)
                {
                    ValidateRawDataSize(proto, n * sizeof(byte));
                    var srcB = proto.RawData.Span;
                    for (int i = 0; i < n; i++) asLong[i] = srcB[i];
                }
                else
                {
                    for (int i = 0; i < n; i++) asLong[i] = (uint)proto.Int32Data[i];
                }
                break;
            default:
                throw new NotSupportedException($"Unsupported integer ONNX type {proto.DataType}.");
        }

        // Convert long → T. The common float/double paths write directly
        // via MemoryMarshal for zero-alloc; every other T uses the numeric
        // ops registry's FromDouble, which handles int8/int16/int32/int64,
        // uint8/16/32/64, Half, BFloat16, decimal — every type exposed
        // through MathHelper.GetNumericOperations.
        if (typeof(T) == typeof(float))
        {
            var fDst = MemoryMarshal.Cast<T, float>(dst);
            for (int i = 0; i < n; i++) fDst[i] = asLong[i];
        }
        else if (typeof(T) == typeof(double))
        {
            var dDst = MemoryMarshal.Cast<T, double>(dst);
            for (int i = 0; i < n; i++) dDst[i] = asLong[i];
        }
        else
        {
            var ops = Helpers.MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < n; i++) dst[i] = ops.FromDouble(asLong[i]);
        }
    }
}
