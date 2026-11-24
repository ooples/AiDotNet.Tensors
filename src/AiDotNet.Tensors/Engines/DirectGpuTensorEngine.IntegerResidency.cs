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
        nameof(TensorMax) => GpuIntegerBinaryOperation.Max,
        nameof(TensorMin) => GpuIntegerBinaryOperation.Min,
        // Both are computed on the device behind a check of the operands there (IntegerOperandsRaiseOnHost): the
        // CPU engine raises for a zero divisor and for the most negative value over -1, and a device cannot.
        nameof(TensorDivide) => GpuIntegerBinaryOperation.Divide,
        nameof(TensorRemainder) => GpuIntegerBinaryOperation.Remainder,
        "TensorEquals" => GpuIntegerBinaryOperation.Equal,
        "TensorNotEquals" => GpuIntegerBinaryOperation.NotEqual,
        "TensorGreaterThan" => GpuIntegerBinaryOperation.Greater,
        "TensorLessThan" => GpuIntegerBinaryOperation.Less,
        _ => null,
    };

    // Only the type-preserving policy takes these routes; a speed-first scope narrows to float on its own path.
    private static bool IntegerDeviceRouteAllowed(Type elementType)
        => !AutocastScope.IsEnabled
           && GpuExecutionPolicyScope.EffectivePolicyFor(elementType).AccuracyMode == GpuAccuracyMode.PreserveInputType;

    // The backend a typed (integer or 16-bit float) route may use now, or null: not inside a resident step, and only
    // under the type-preserving policy.
    private IDirectGpuBackend? TypedRouteBackend(Type elementType)
    {
        if (ResidentStepActive || !IntegerDeviceRouteAllowed(elementType)) return null;
        return TryGetBackend(out var backend) ? backend : null;
    }

    private static Action<IGpuBuffer, IGpuBuffer, IGpuBuffer, int>? IntegerBinaryKernel(IDirectGpuBackend backend, GpuElementType type,
        GpuIntegerBinaryOperation operation)
    {
        if (backend is not IGpuIntegerElementwise elementwise || !backend.SupportsElementType(type) || !elementwise.SupportsIntegerElementwise(type))
            return null;
        if (operation is GpuIntegerBinaryOperation.Divide or GpuIntegerBinaryOperation.Remainder)
        {
            if (backend is not IGpuIntegerReductions reductions || !reductions.SupportsIntegerReductions(type)) return null;
            return (a, b, output, size) =>
            {
                // Zero divisors raise in the CPU engine's divide; its remainder gives 0 for them, as the kernel does.
                bool zeroRaises = operation == GpuIntegerBinaryOperation.Divide;
                if (IntegerOperandsRaiseOnHost(backend, elementwise, reductions, type, a, b, output, size, zeroRaises))
                    throw new IntegerHostRaisesException(operation.ToString());
                elementwise.BinaryInteger(operation, a, b, output, size);
            };
        }

        return (a, b, output, size) => elementwise.BinaryInteger(operation, a, b, output, size);
    }

    /// <summary>
    /// The operands hold a value the CPU engine raises for (a zero divisor, the most negative value over -1, the most
    /// negative value's absolute value). The route catches it as a capability gap, so the CPU engine runs the
    /// operation and raises the exception it raises for those operands; no result is produced either way.
    /// </summary>
    private sealed class IntegerHostRaisesException : NotSupportedException
    {
        public IntegerHostRaisesException(string operation)
            : base($"{operation}: an operand the CPU engine raises for; it runs the operation to raise it.")
        {
        }
    }

    // The most negative value of a 32- or 64-bit type, whose negation overflows (the CPU engine raises); null for the
    // narrower types, whose quotient is computed in 32 bits and wraps without raising.
    private static long? OverflowingDividend(GpuElementType type) => type switch
    {
        GpuElementType.Int32 => int.MinValue,
        GpuElementType.Int64 => long.MinValue,
        _ => null,
    };

    private static long MostNegative(GpuElementType type) => type switch
    {
        GpuElementType.Int8 => sbyte.MinValue,
        GpuElementType.Int16 => short.MinValue,
        GpuElementType.Int32 => int.MinValue,
        _ => long.MinValue,
    };

    /// <summary>
    /// Whether <c>a / b</c> raises on the host for some element, decided on the device: a zero in <paramref name="b"/>
    /// (when <paramref name="zeroRaises"/>), or the most negative value of a 32- or 64-bit type over -1.
    /// <paramref name="scratch"/> is the launch's output buffer, overwritten afterwards by the operation itself.
    /// </summary>
    private static bool IntegerOperandsRaiseOnHost(IDirectGpuBackend backend, IGpuIntegerElementwise elementwise, IGpuIntegerReductions reductions,
        GpuElementType type, IGpuBuffer a, IGpuBuffer b, IGpuBuffer scratch, int size, bool zeroRaises)
    {
        if (zeroRaises)
        {
            elementwise.ScalarInteger(GpuIntegerBinaryOperation.Equal, b, 0, scratch, size);
            if (reductions.MaxInteger(scratch, size) != 0) return true;
        }

        // Only where the dividend holds the most negative value at all, which one reduction tells.
        if (OverflowingDividend(type) is not { } lowest || reductions.MinInteger(a, size) != lowest) return false;
        using var dividendIsLowest = backend.AllocateBuffer(size, type);
        using var divisorIsMinusOne = backend.AllocateBuffer(size, type);
        elementwise.ScalarInteger(GpuIntegerBinaryOperation.Equal, a, lowest, dividendIsLowest, size);
        elementwise.ScalarInteger(GpuIntegerBinaryOperation.Equal, b, -1, divisorIsMinusOne, size);
        elementwise.BinaryInteger(GpuIntegerBinaryOperation.And, dividendIsLowest, divisorIsMinusOne, scratch, size);
        return reductions.MaxInteger(scratch, size) != 0;
    }

    // A unary launch; the absolute value first asks the device whether the most negative value is there, for which
    // the CPU engine raises.
    private static void LaunchIntegerUnary(IGpuIntegerElementwise elementwise, GpuIntegerUnaryOperation operation, GpuElementType type,
        IGpuBuffer a, IGpuBuffer output, int size)
    {
        if (operation == GpuIntegerUnaryOperation.Abs)
        {
            if (elementwise is not IGpuIntegerReductions reductions || !reductions.SupportsIntegerReductions(type))
                throw new GpuCapabilityUnavailableException("The absolute value needs the device's minimum reduction.");
            if (reductions.MinInteger(a, size) == MostNegative(type)) throw new IntegerHostRaisesException(nameof(TensorAbs));
        }

        elementwise.UnaryInteger(operation, a, output, size);
    }

    // A scalar launch; a division by -1 of a 32- or 64-bit type first asks the device whether the most negative value
    // is there, for which the CPU engine raises.
    private static void LaunchIntegerScalar(IGpuIntegerElementwise elementwise, GpuIntegerBinaryOperation operation, GpuElementType type,
        IGpuBuffer a, long scalar, IGpuBuffer output, int size)
    {
        if (operation == GpuIntegerBinaryOperation.Divide && scalar == -1 && OverflowingDividend(type) is { } lowest)
        {
            if (elementwise is not IGpuIntegerReductions reductions || !reductions.SupportsIntegerReductions(type))
                throw new GpuCapabilityUnavailableException("A division by -1 needs the device's minimum reduction.");
            if (reductions.MinInteger(a, size) == lowest) throw new IntegerHostRaisesException(nameof(TensorDivideScalar));
        }

        elementwise.ScalarInteger(operation, a, scalar, output, size);
    }

    private static GpuNarrowFloatBinaryOperation? NarrowFloatBinaryOperationFor(string callerName) => callerName switch
    {
        nameof(TensorAdd) => GpuNarrowFloatBinaryOperation.Add,
        nameof(TensorSubtract) => GpuNarrowFloatBinaryOperation.Subtract,
        nameof(TensorMultiply) => GpuNarrowFloatBinaryOperation.Multiply,
        _ => null,
    };

    private static GpuNarrowFloatUnaryOperation? NarrowFloatUnaryOperationFor(string callerName) => callerName switch
    {
        nameof(TensorNegate) => GpuNarrowFloatUnaryOperation.Negate,
        nameof(TensorReLU) => GpuNarrowFloatUnaryOperation.Relu,
        nameof(TensorGELU) => GpuNarrowFloatUnaryOperation.Gelu,
        _ => null,
    };

    /// <summary>
    /// <c>left (op) right</c> for BFloat16 operands on the device, computed in FP32 and stored as BFloat16, the result
    /// left there. Null when the CPU engine takes it. (Half tensors keep the FP16 routes they already have.)
    /// </summary>
    private Tensor<T>? TryBinaryNarrowFloatAny<T>(Tensor<T> left, Tensor<T> right, string callerName)
    {
        if (left is not Tensor<NumericOperations.BFloat16> leftValues || right is not Tensor<NumericOperations.BFloat16> rightValues) return null;
        if (NarrowFloatBinaryOperationFor(callerName) is not { } operation) return null;
        const GpuElementType Type = GpuElementType.BFloat16;
        return TryBinaryTyped(leftValues, rightValues, Type, callerName, backend =>
            backend is IGpuNarrowFloatElementwise elementwise && backend.SupportsElementType(Type)
            && elementwise.SupportsNarrowFloatBinary(operation, Type)
                ? (a, b, output, size) => elementwise.BinaryNarrowFloat(operation, Type, a, b, output, size)
                : null) as Tensor<T>;
    }

    /// <summary>Negate, ReLU or GELU of a BFloat16 tensor on the device. Null when the CPU engine takes it.</summary>
    private Tensor<T>? TryUnaryNarrowFloatAny<T>(Tensor<T> input, string callerName)
    {
        if (input is not Tensor<NumericOperations.BFloat16> values) return null;
        if (NarrowFloatUnaryOperationFor(callerName) is not { } operation) return null;
        const GpuElementType Type = GpuElementType.BFloat16;
        return TryMapTyped(values, Type, callerName, backend =>
            backend is IGpuNarrowFloatElementwise elementwise && backend.SupportsElementType(Type)
            && elementwise.SupportsNarrowFloatUnary(operation, Type)
                ? (a, output, size) => elementwise.UnaryNarrowFloat(operation, Type, a, output, size)
                : null) as Tensor<T>;
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
        => TryBinaryTyped(left, right, type, callerName, backend => IntegerBinaryKernel(backend, type, operation));

    // One element-wise kernel over two typed operands, the result left on the device. `kernelFor` gives the launch for
    // a backend, or null when that backend has no such kernel.
    private Tensor<T>? TryBinaryTyped<T>(Tensor<T> left, Tensor<T> right, GpuElementType type, string callerName,
        Func<IDirectGpuBackend, Action<IGpuBuffer, IGpuBuffer, IGpuBuffer, int>?> kernelFor)
        where T : unmanaged
    {
        int length = left.Length;
        if (length < IntegerElementwiseMinLength || right.Length != length) return null;
        // Element i of one operand meets element i of the other in storage order: operands laid out differently are the
        // CPU engine's, which reorders them.
        if (left.Layout != right.Layout) return null;
        if (TypedRouteBackend(typeof(T)) is not { } backend || kernelFor(backend) is not { } kernel) return null;

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
            using var b = ReferenceEquals(left, right)
                ? new OwnedBuffer(a.Buffer, ownsBuffer: false)
                : ResolveResidentOperand(backend, right, type, left.DataVector);
            var output = RentResidentBuffer(backend, type, length, left.DataVector, right.DataVector);
            try
            {
                kernel(a.Buffer, b.Buffer, output, length);
                // An operand uploaded for this launch is still being read by it: released in queue order, not now.
                a.DisposeAfterQueuedUse(backend);
                b.DisposeAfterQueuedUse(backend);
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
        GpuIntegerUnaryOperation Operation;
        switch (callerName)
        {
            case nameof(TensorNegate): Operation = GpuIntegerUnaryOperation.Negate; break;
            case nameof(TensorAbs): Operation = GpuIntegerUnaryOperation.Abs; break;
            case nameof(TensorSign): Operation = GpuIntegerUnaryOperation.Sign; break;
            default: return null;
        }

        if (input is Tensor<int> ints)
            return TryMapInteger(ints, GpuElementType.Int32, callerName, (e, a, o, n) => LaunchIntegerUnary(e, Operation, GpuElementType.Int32, a, o, n)) as Tensor<T>;
        if (input is Tensor<long> longs)
            return TryMapInteger(longs, GpuElementType.Int64, callerName, (e, a, o, n) => LaunchIntegerUnary(e, Operation, GpuElementType.Int64, a, o, n)) as Tensor<T>;
        if (input is Tensor<short> shorts)
            return TryMapInteger(shorts, GpuElementType.Int16, callerName, (e, a, o, n) => LaunchIntegerUnary(e, Operation, GpuElementType.Int16, a, o, n)) as Tensor<T>;
        if (input is Tensor<sbyte> sbytes)
            return TryMapInteger(sbytes, GpuElementType.Int8, callerName, (e, a, o, n) => LaunchIntegerUnary(e, Operation, GpuElementType.Int8, a, o, n)) as Tensor<T>;
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
            return TryMapInteger(ints, GpuElementType.Int32, callerName, (e, a, o, n) => LaunchIntegerScalar(e, operation, GpuElementType.Int32, a, intScalar, o, n)) as Tensor<T>;
        if (input is Tensor<long> longs && scalar is long longScalar)
            return TryMapInteger(longs, GpuElementType.Int64, callerName, (e, a, o, n) => LaunchIntegerScalar(e, operation, GpuElementType.Int64, a, longScalar, o, n)) as Tensor<T>;
        if (input is Tensor<short> shorts && scalar is short shortScalar)
            return TryMapInteger(shorts, GpuElementType.Int16, callerName, (e, a, o, n) => LaunchIntegerScalar(e, operation, GpuElementType.Int16, a, shortScalar, o, n)) as Tensor<T>;
        if (input is Tensor<sbyte> sbytes && scalar is sbyte sbyteScalar)
            return TryMapInteger(sbytes, GpuElementType.Int8, callerName, (e, a, o, n) => LaunchIntegerScalar(e, operation, GpuElementType.Int8, a, sbyteScalar, o, n)) as Tensor<T>;
        if (input is Tensor<byte> bytes && scalar is byte byteScalar)
            return TryMapInteger(bytes, GpuElementType.UInt8, callerName, (e, a, o, n) => LaunchIntegerScalar(e, operation, GpuElementType.UInt8, a, byteScalar, o, n)) as Tensor<T>;
        return null;
    }

    /// <summary>
    /// Whether <paramref name="scalar"/> is a divisor: zero is not, and the CPU engine raises for it at its first
    /// element, so nothing is launched.
    /// </summary>
    private static bool IsIntegerDivisor<T>(T scalar) => scalar switch
    {
        int value => value != 0,
        long value => value != 0,
        short value => value != 0,
        sbyte value => value != 0,
        byte value => value != 0,
        _ => false,
    };

    private Tensor<T>? TryMapInteger<T>(Tensor<T> input, GpuElementType type, string callerName,
        Action<IGpuIntegerElementwise, IGpuBuffer, IGpuBuffer, int> launch)
        where T : unmanaged
        => TryMapTyped(input, type, callerName, backend =>
            backend is IGpuIntegerElementwise elementwise && backend.SupportsElementType(type) && elementwise.SupportsIntegerElementwise(type)
                ? (a, output, size) => launch(elementwise, a, output, size)
                : null);

    // One element-wise kernel over one typed operand, the result left on the device.
    private Tensor<T>? TryMapTyped<T>(Tensor<T> input, GpuElementType type, string callerName,
        Func<IDirectGpuBackend, Action<IGpuBuffer, IGpuBuffer, int>?> kernelFor)
        where T : unmanaged
    {
        int length = input.Length;
        if (length < IntegerElementwiseMinLength) return null;
        if (TypedRouteBackend(typeof(T)) is not { } backend || kernelFor(backend) is not { } kernel) return null;

        lock (_typedLock)
        {
            CountResidentTransfers(input, out int uploads, out int downloads);
            if (uploads > downloads) return null;

            using var a = ResolveResidentOperand(backend, input, type);
            var output = RentResidentBuffer(backend, type, length, input.DataVector, input.DataVector);
            try
            {
                kernel(a.Buffer, output, length);
                a.DisposeAfterQueuedUse(backend);
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
    /// host's unchecked sum does), boxed. Null when the CPU engine takes it.
    /// </summary>
    private object? TrySumIntegerAny<T>(Tensor<T> tensor)
    {
        if (tensor is Tensor<int> ints) return TrySumInteger(ints, GpuElementType.Int32, out long wide) ? unchecked((int)wide) : null;
        if (tensor is Tensor<long> longs) return TrySumInteger(longs, GpuElementType.Int64, out long wide) ? wide : null;
        if (tensor is Tensor<short> shorts) return TrySumInteger(shorts, GpuElementType.Int16, out long wide) ? unchecked((short)wide) : null;
        if (tensor is Tensor<sbyte> sbytes) return TrySumInteger(sbytes, GpuElementType.Int8, out long wide) ? unchecked((sbyte)wide) : null;
        if (tensor is Tensor<byte> bytes) return TrySumInteger(bytes, GpuElementType.UInt8, out long wide) ? unchecked((byte)wide) : null;
        return null;
    }

    /// <summary>
    /// The smallest or largest element of an integer tensor whose values are on the device, found there, boxed. Null
    /// when the CPU engine takes it.
    /// </summary>
    private object? TryExtremeIntegerAny<T>(Tensor<T> tensor, GpuIntegerReduction reduction)
    {
        if (tensor is Tensor<int> ints) return TrySumInteger(ints, GpuElementType.Int32, out long wide, reduction) ? (int)wide : null;
        if (tensor is Tensor<long> longs) return TrySumInteger(longs, GpuElementType.Int64, out long wide, reduction) ? wide : null;
        if (tensor is Tensor<short> shorts) return TrySumInteger(shorts, GpuElementType.Int16, out long wide, reduction) ? (short)wide : null;
        if (tensor is Tensor<sbyte> sbytes) return TrySumInteger(sbytes, GpuElementType.Int8, out long wide, reduction) ? (sbyte)wide : null;
        if (tensor is Tensor<byte> bytes) return TrySumInteger(bytes, GpuElementType.UInt8, out long wide, reduction) ? (byte)wide : null;
        return null;
    }

    private bool TrySumInteger<T>(Tensor<T> tensor, GpuElementType type, out long sum, GpuIntegerReduction reduction = GpuIntegerReduction.Sum)
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
                sum = reduction switch
                {
                    GpuIntegerReduction.Min => reductions.MinInteger(resident.Buffer, length),
                    GpuIntegerReduction.Max => reductions.MaxInteger(resident.Buffer, length),
                    _ => reductions.SumInteger(resident.Buffer, length),
                };
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
        if (tensor is Tensor<int> || tensor is Tensor<long> || tensor is Tensor<short> || tensor is Tensor<sbyte> || tensor is Tensor<byte>
            || tensor is Tensor<NumericOperations.BFloat16>)
            DropResident(tensor);
    }
}
