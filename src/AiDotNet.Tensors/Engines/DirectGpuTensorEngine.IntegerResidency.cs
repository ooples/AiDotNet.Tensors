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
        return (a, b, output, size) => elementwise.BinaryInteger(operation, a, b, output, size);
    }

    private static GpuNarrowFloatBinaryOperation? NarrowFloatBinaryOperationFor(string callerName) => callerName switch
    {
        nameof(TensorAdd) => GpuNarrowFloatBinaryOperation.Add,
        nameof(TensorSubtract) => GpuNarrowFloatBinaryOperation.Subtract,
        nameof(TensorMultiply) => GpuNarrowFloatBinaryOperation.Multiply,
        nameof(TensorDivide) => GpuNarrowFloatBinaryOperation.Divide,
        _ => null,
    };

    private static GpuNarrowFloatUnaryOperation? NarrowFloatUnaryOperationFor(string callerName) => callerName switch
    {
        nameof(TensorNegate) => GpuNarrowFloatUnaryOperation.Negate,
        nameof(TensorReLU) => GpuNarrowFloatUnaryOperation.Relu,
        nameof(TensorGELU) => GpuNarrowFloatUnaryOperation.Gelu,
        nameof(TensorAbs) => GpuNarrowFloatUnaryOperation.Abs,
        nameof(TensorSqrt) => GpuNarrowFloatUnaryOperation.Sqrt,
        nameof(TensorExp) => GpuNarrowFloatUnaryOperation.Exp,
        nameof(TensorLog) => GpuNarrowFloatUnaryOperation.Log,
        nameof(TensorTanh) => GpuNarrowFloatUnaryOperation.Tanh,
        nameof(TensorSigmoid) or nameof(Sigmoid) => GpuNarrowFloatUnaryOperation.Sigmoid,
        _ => null,
    };

    /// <summary>
    /// Whether a <c>Half</c> tensor takes the typed routes below. The FP16 activation store keeps its own registry of
    /// device halves (and its own kernels) for the tensors it holds; where it is on, they stay its.
    /// </summary>
    private static bool HalfTakesTypedRoute => !Fp16ActEnabled && !s_fp16FwdStore;

    /// <summary>
    /// <c>left (op) right</c> for 16-bit float operands (BFloat16, Half) on the device, computed in FP32 and stored in
    /// the operands' type, the result left there. Null when the CPU engine takes it.
    /// </summary>
    private Tensor<T>? TryBinaryNarrowFloatAny<T>(Tensor<T> left, Tensor<T> right, string callerName)
    {
        if (NarrowFloatBinaryOperationFor(callerName) is not { } operation) return null;
        if (left is Tensor<NumericOperations.BFloat16> leftValues && right is Tensor<NumericOperations.BFloat16> rightValues)
            return TryBinaryNarrowFloat(leftValues, rightValues, GpuElementType.BFloat16, operation, callerName) as Tensor<T>;
#if NET5_0_OR_GREATER
        if (HalfTakesTypedRoute && left is Tensor<Half> leftHalves && right is Tensor<Half> rightHalves)
            return TryBinaryNarrowFloat(leftHalves, rightHalves, GpuElementType.Float16, operation, callerName) as Tensor<T>;
#endif
        return null;
    }

    private Tensor<T>? TryBinaryNarrowFloat<T>(Tensor<T> left, Tensor<T> right, GpuElementType type, GpuNarrowFloatBinaryOperation operation,
        string callerName)
        where T : unmanaged
        => TryBinaryTyped(left, right, type, callerName, backend =>
            backend is IGpuNarrowFloatElementwise elementwise && backend.SupportsElementType(type)
            && elementwise.SupportsNarrowFloatBinary(operation, type)
                ? (a, b, output, size) => elementwise.BinaryNarrowFloat(operation, type, a, b, output, size)
                : null);

    /// <summary>
    /// <c>input (op) scalar</c> for a 16-bit float tensor on the device, computed in FP32 and stored in its type. Null
    /// when the CPU engine takes it, and under a graph trace or anomaly mode (which keep the CPU engine's path, as the
    /// double route does).
    /// </summary>
    private Tensor<T>? TryScalarNarrowFloatAny<T>(Tensor<T> input, T scalar, GpuNarrowFloatBinaryOperation operation, string callerName)
    {
        if (Compilation.GraphMode.IsActive || Autodiff.AnomalyModeScope.IsActive) return null;
        if (input is Tensor<NumericOperations.BFloat16> values && scalar is NumericOperations.BFloat16 narrowScalar)
            return TryScalarNarrowFloat(values, GpuElementType.BFloat16, NumericOperations.BFloat16.ToFloat(narrowScalar), operation, callerName) as Tensor<T>;
#if NET5_0_OR_GREATER
        if (HalfTakesTypedRoute && input is Tensor<Half> halves && scalar is Half halfScalar)
            return TryScalarNarrowFloat(halves, GpuElementType.Float16, (float)halfScalar, operation, callerName) as Tensor<T>;
#endif
        return null;
    }

    private Tensor<T>? TryScalarNarrowFloat<T>(Tensor<T> input, GpuElementType type, float scalar, GpuNarrowFloatBinaryOperation operation,
        string callerName)
        where T : unmanaged
        => TryMapTyped(input, type, callerName, backend =>
            backend is IGpuNarrowFloatElementwise elementwise && backend.SupportsElementType(type)
            && elementwise.SupportsNarrowFloatBinary(operation, type)
                ? (a, output, size) => elementwise.ScalarNarrowFloat(operation, type, a, scalar, output, size)
                : null);

    /// <summary>A unary operation of a 16-bit float tensor on the device. Null when the CPU engine takes it.</summary>
    private Tensor<T>? TryUnaryNarrowFloatAny<T>(Tensor<T> input, string callerName)
    {
        if (NarrowFloatUnaryOperationFor(callerName) is not { } operation) return null;
        if (input is Tensor<NumericOperations.BFloat16> values)
            return TryUnaryNarrowFloat(values, GpuElementType.BFloat16, operation, callerName) as Tensor<T>;
#if NET5_0_OR_GREATER
        if (HalfTakesTypedRoute && input is Tensor<Half> halves)
            return TryUnaryNarrowFloat(halves, GpuElementType.Float16, operation, callerName) as Tensor<T>;
#endif
        return null;
    }

    private Tensor<T>? TryUnaryNarrowFloat<T>(Tensor<T> input, GpuElementType type, GpuNarrowFloatUnaryOperation operation, string callerName)
        where T : unmanaged
        => TryMapTyped(input, type, callerName, backend =>
            backend is IGpuNarrowFloatElementwise elementwise && backend.SupportsElementType(type)
            && elementwise.SupportsNarrowFloatUnary(operation, type)
                ? (a, output, size) => elementwise.UnaryNarrowFloat(operation, type, a, output, size)
                : null);

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
    /// The sum of a 16-bit float tensor whose values are on the device, accumulated there in FP32 (the device's blocked
    /// sum, see <see cref="IGpuNarrowFloatReductions"/>) and rounded once into the type, boxed. Null when the CPU engine
    /// takes it.
    /// </summary>
    private object? TrySumNarrowFloat<T>(Tensor<T> tensor)
    {
        if (tensor is Tensor<NumericOperations.BFloat16> values)
            return TrySumNarrowFloat(values, GpuElementType.BFloat16, out float sum) ? NumericOperations.BFloat16.FromFloat(sum) : null;
#if NET5_0_OR_GREATER
        if (HalfTakesTypedRoute && tensor is Tensor<Half> halves)
            return TrySumNarrowFloat(halves, GpuElementType.Float16, out float sum) ? (Half)sum : null;
#endif
        return null;
    }

    private bool TrySumNarrowFloat<T>(Tensor<T> values, GpuElementType type, out float sum)
        where T : unmanaged
    {
        sum = 0f;
        int length = values.Length;
        if (length == 0 || ResidentStepActive || !IntegerDeviceRouteAllowed(typeof(T)) || !TryGetBackend(out var backend)) return false;
        if (backend is not IGpuNarrowFloatReductions reductions || !backend.SupportsElementType(type) || !reductions.SupportsNarrowFloatSum(type))
            return false;
        try
        {
            lock (_typedLock)
            {
                // Only a tensor already on the device is reduced there, as for the integer types.
                if (FindResident(values) is not { } resident || !ReferenceEquals(resident.Backend, backend)) return false;
                sum = reductions.SumNarrowFloat(type, resident.Buffer, length);
                return true;
            }
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("TensorSumNarrowFloat", capabilityGap);
            return false;
        }
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
        if (tensor is Tensor<int> || tensor is Tensor<long> || tensor is Tensor<short> || tensor is Tensor<sbyte> || tensor is Tensor<byte>
            || tensor is Tensor<NumericOperations.BFloat16>)
            DropResident(tensor);
    }
}
