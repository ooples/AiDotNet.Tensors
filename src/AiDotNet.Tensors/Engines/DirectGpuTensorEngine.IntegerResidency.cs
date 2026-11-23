// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Device residency for integer tensors (sbyte, byte, short, int, long).
//
// The same contract as the double routes (DirectGpuTensorEngine.Float64Residency), on the same registry, pool and
// budget: a result stays on the device in a buffer of the tensor's own element type and reaches the host on the first
// host read; an op goes to the device only when that costs no more host<->device copies than the CPU engine would.
// The kernels compute in the element type and wrap as unchecked host arithmetic does, so the values are the CPU
// engine's bit for bit. Nothing is routed through a float.
public partial class DirectGpuTensorEngine
{
    /// <summary>The shortest integer tensor an element-wise op or a resident reduction runs on the device.</summary>
    internal const int IntegerElementwiseMinLength = 1 << 17;

    private static GpuIntegerBinaryOperation? IntegerBinaryOperationFor(string callerName) => callerName switch
    {
        nameof(TensorAdd) => GpuIntegerBinaryOperation.Add,
        nameof(TensorSubtract) => GpuIntegerBinaryOperation.Subtract,
        nameof(TensorMultiply) => GpuIntegerBinaryOperation.Multiply,
        _ => null,
    };

    // Only the type-preserving policy takes these routes; a speed-first scope narrows to float on its own path.
    private static bool IntegerDeviceRouteAllowed(Type elementType)
        => !AutocastScope.IsEnabled
           && GpuExecutionPolicyScope.EffectivePolicyFor(elementType).AccuracyMode == GpuAccuracyMode.PreserveInputType;

    private bool TryGetIntegerElementwise(GpuElementType type, Type elementType, out IDirectGpuBackend backend, out IGpuIntegerElementwise elementwise)
    {
        elementwise = null!;
        if (ResidentStepActive || !IntegerDeviceRouteAllowed(elementType) || !TryGetBackend(out backend))
        {
            backend = null!;
            return false;
        }

        if (backend is not IGpuIntegerElementwise candidate || !backend.SupportsElementType(type) || !candidate.SupportsIntegerElementwise(type))
            return false;
        elementwise = candidate;
        return true;
    }

    /// <summary>
    /// <c>left (op) right</c> for integer operands, on the device with the result left there. Null when this op, type,
    /// backend, policy or scope has no integer route, or the CPU engine is the cheaper one for these operands.
    /// </summary>
    private Tensor<T>? TryBinaryIntegerAny<T>(Tensor<T> left, Tensor<T> right, string callerName)
    {
        if (IntegerBinaryOperationFor(callerName) is not { } operation) return null;
        if (left is Tensor<int> leftInt && right is Tensor<int> rightInt)
            return TryBinaryInteger(leftInt, rightInt, GpuElementType.Int32, operation, callerName) as Tensor<T>;
        if (left is Tensor<long> leftLong && right is Tensor<long> rightLong)
            return TryBinaryInteger(leftLong, rightLong, GpuElementType.Int64, operation, callerName) as Tensor<T>;
        if (left is Tensor<short> leftShort && right is Tensor<short> rightShort)
            return TryBinaryInteger(leftShort, rightShort, GpuElementType.Int16, operation, callerName) as Tensor<T>;
        if (left is Tensor<sbyte> leftSByte && right is Tensor<sbyte> rightSByte)
            return TryBinaryInteger(leftSByte, rightSByte, GpuElementType.Int8, operation, callerName) as Tensor<T>;
        if (left is Tensor<byte> leftByte && right is Tensor<byte> rightByte)
            return TryBinaryInteger(leftByte, rightByte, GpuElementType.UInt8, operation, callerName) as Tensor<T>;
        return null;
    }

    private Tensor<T>? TryBinaryInteger<T>(Tensor<T> left, Tensor<T> right, GpuElementType type, GpuIntegerBinaryOperation operation,
        string callerName)
        where T : unmanaged
    {
        int length = left.Length;
        if (length < IntegerElementwiseMinLength || right.Length != length) return null;
        // Element i of one operand meets element i of the other in storage order: operands laid out differently are the
        // CPU engine's, which reorders them.
        if (left.Layout != right.Layout) return null;
        if (!TryGetIntegerElementwise(type, typeof(T), out var backend, out var elementwise)) return null;

        lock (_typedLock)
        {
            CountResidentTransfers(left, out int uploads, out int downloads);
            if (!ReferenceEquals(left, right))
            {
                CountResidentTransfers(right, out int rightUploads, out int rightDownloads);
                uploads += rightUploads;
                downloads += rightDownloads;
            }

            if (uploads > downloads) return null;

            using var a = ResolveResidentOperand(backend, left, type);
            using var b = ReferenceEquals(left, right) ? new OwnedBuffer(a.Buffer, ownsBuffer: false) : ResolveResidentOperand(backend, right, type);
            var output = RentResidentBuffer(backend, type, length, left.DataVector, right.DataVector);
            try
            {
                elementwise.BinaryInteger(operation, a.Buffer, b.Buffer, output, length);
                var result = RegisterResidentResult<T>(backend, output, length, left.Shape.ToArray(), type);
                result.Layout = left.Layout;
                return result;
            }
            catch (NotSupportedException capabilityGap)
            {
                output.Dispose();
                GpuLaunchProbe.OnFallback(callerName, capabilityGap);
                return null;
            }
            catch
            {
                output.Dispose();
                throw;
            }
        }
    }

    /// <summary><c>-input</c> for an integer tensor on the device. Null when the CPU engine takes it.</summary>
    private Tensor<T>? TryUnaryIntegerAny<T>(Tensor<T> input, string callerName)
    {
        if (callerName != nameof(TensorNegate)) return null;
        const GpuIntegerUnaryOperation Operation = GpuIntegerUnaryOperation.Negate;
        if (input is Tensor<int> ints)
            return TryMapInteger(ints, GpuElementType.Int32, callerName, (e, a, o, n) => e.UnaryInteger(Operation, a, o, n)) as Tensor<T>;
        if (input is Tensor<long> longs)
            return TryMapInteger(longs, GpuElementType.Int64, callerName, (e, a, o, n) => e.UnaryInteger(Operation, a, o, n)) as Tensor<T>;
        if (input is Tensor<short> shorts)
            return TryMapInteger(shorts, GpuElementType.Int16, callerName, (e, a, o, n) => e.UnaryInteger(Operation, a, o, n)) as Tensor<T>;
        if (input is Tensor<sbyte> sbytes)
            return TryMapInteger(sbytes, GpuElementType.Int8, callerName, (e, a, o, n) => e.UnaryInteger(Operation, a, o, n)) as Tensor<T>;
        return null;
    }

    /// <summary>
    /// <c>input (op) scalar</c> for an integer tensor on the device. Null when the CPU engine takes it, and under a
    /// graph trace or anomaly mode (which keep the CPU engine's path, as the double route does).
    /// </summary>
    private Tensor<T>? TryScalarIntegerAny<T>(Tensor<T> input, T scalar, GpuIntegerBinaryOperation operation, string callerName)
    {
        if (Compilation.GraphMode.IsActive || Autodiff.AnomalyModeScope.IsActive) return null;
        if (input is Tensor<int> ints && scalar is int intScalar)
            return TryMapInteger(ints, GpuElementType.Int32, callerName, (e, a, o, n) => e.ScalarInteger(operation, a, intScalar, o, n)) as Tensor<T>;
        if (input is Tensor<long> longs && scalar is long longScalar)
            return TryMapInteger(longs, GpuElementType.Int64, callerName, (e, a, o, n) => e.ScalarInteger(operation, a, longScalar, o, n)) as Tensor<T>;
        if (input is Tensor<short> shorts && scalar is short shortScalar)
            return TryMapInteger(shorts, GpuElementType.Int16, callerName, (e, a, o, n) => e.ScalarInteger(operation, a, shortScalar, o, n)) as Tensor<T>;
        if (input is Tensor<sbyte> sbytes && scalar is sbyte sbyteScalar)
            return TryMapInteger(sbytes, GpuElementType.Int8, callerName, (e, a, o, n) => e.ScalarInteger(operation, a, sbyteScalar, o, n)) as Tensor<T>;
        if (input is Tensor<byte> bytes && scalar is byte byteScalar)
            return TryMapInteger(bytes, GpuElementType.UInt8, callerName, (e, a, o, n) => e.ScalarInteger(operation, a, byteScalar, o, n)) as Tensor<T>;
        return null;
    }

    private Tensor<T>? TryMapInteger<T>(Tensor<T> input, GpuElementType type, string callerName,
        Action<IGpuIntegerElementwise, IGpuBuffer, IGpuBuffer, int> launch)
        where T : unmanaged
    {
        int length = input.Length;
        if (length < IntegerElementwiseMinLength) return null;
        if (!TryGetIntegerElementwise(type, typeof(T), out var backend, out var elementwise)) return null;

        lock (_typedLock)
        {
            CountResidentTransfers(input, out int uploads, out int downloads);
            if (uploads > downloads) return null;

            using var a = ResolveResidentOperand(backend, input, type);
            var output = RentResidentBuffer(backend, type, length, input.DataVector, input.DataVector);
            try
            {
                launch(elementwise, a.Buffer, output, length);
                var result = RegisterResidentResult<T>(backend, output, length, input.Shape.ToArray(), type);
                // The kernel maps the elements in storage order, so the result is laid out as its input is.
                result.Layout = input.Layout;
                return result;
            }
            catch (NotSupportedException capabilityGap)
            {
                output.Dispose();
                GpuLaunchProbe.OnFallback(callerName, capabilityGap);
                return null;
            }
            catch
            {
                output.Dispose();
                throw;
            }
        }
    }

    /// <summary>
    /// The sum of an integer tensor whose values are on the device, computed there in its own type (wrapping as the
    /// host's unchecked sum does). False when the CPU engine takes it.
    /// </summary>
    private bool TrySumIntegerAny<T>(Tensor<T> tensor, out T sum)
    {
        sum = default!;
        long wide;
        if (tensor is Tensor<int> ints)
        {
            if (!TrySumInteger(ints, GpuElementType.Int32, out wide)) return false;
            sum = (T)(object)unchecked((int)wide);
        }
        else if (tensor is Tensor<long> longs)
        {
            if (!TrySumInteger(longs, GpuElementType.Int64, out wide)) return false;
            sum = (T)(object)wide;
        }
        else if (tensor is Tensor<short> shorts)
        {
            if (!TrySumInteger(shorts, GpuElementType.Int16, out wide)) return false;
            sum = (T)(object)unchecked((short)wide);
        }
        else if (tensor is Tensor<sbyte> sbytes)
        {
            if (!TrySumInteger(sbytes, GpuElementType.Int8, out wide)) return false;
            sum = (T)(object)unchecked((sbyte)wide);
        }
        else if (tensor is Tensor<byte> bytes)
        {
            if (!TrySumInteger(bytes, GpuElementType.UInt8, out wide)) return false;
            sum = (T)(object)unchecked((byte)wide);
        }
        else
        {
            return false;
        }

        return true;
    }

    private bool TrySumInteger<T>(Tensor<T> tensor, GpuElementType type, out long sum)
        where T : unmanaged
    {
        sum = 0;
        int length = tensor.Length;
        if (length == 0 || ResidentStepActive || !IntegerDeviceRouteAllowed(typeof(T)) || !TryGetBackend(out var backend)) return false;
        if (backend is not IGpuIntegerReductions reductions || !backend.SupportsElementType(type) || !reductions.SupportsIntegerReductions(type))
            return false;
        try
        {
            lock (_typedLock)
            {
                // Only a tensor already on the device is reduced there: uploading a host tensor to sum it costs more
                // than the CPU engine's sum.
                if (FindResident(tensor) is not { } resident || !ReferenceEquals(resident.Backend, backend)) return false;
                sum = reductions.SumInteger(resident.Buffer, length);
                return true;
            }
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TensorSumInteger", capabilityGap);
            return false;
        }
    }

    /// <summary>Frees the device copy of a persistent integer tensor whose host values changed or that was unregistered.</summary>
    private void DropIntegerResident<T>(Tensor<T> tensor)
    {
        if (tensor is Tensor<int> || tensor is Tensor<long> || tensor is Tensor<short> || tensor is Tensor<sbyte> || tensor is Tensor<byte>)
            DropResident(tensor);
    }
}
