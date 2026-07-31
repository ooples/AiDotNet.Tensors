using System;

namespace AiDotNet.Tensors;

/// <summary>
/// Thrown when an operation receives operand tensors that live on different devices and the active
/// <see cref="DeviceDispatchMode"/> is <see cref="DeviceDispatchMode.Strict"/>. Mirrors PyTorch's
/// "Expected all tensors to be on the same device" error, but is a strongly-typed exception carrying the
/// offending <see cref="TensorDevice"/> values (no string parsing) and a concrete <c>.To(...)</c> fix.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Tensors can live on the CPU or on a GPU. An operation like matrix multiply
/// runs on ONE device, so all of its inputs must be on the same device. If you mix (say a CPU weight with a
/// GPU input), this exception tells you exactly which tensors disagree and how to fix it — move one with
/// <c>tensor.To(DeviceInfo.Cuda(0))</c> (or <c>model.To(...)</c> to move a whole model).</para>
/// </remarks>
[Serializable]
public sealed class DeviceMismatchException : InvalidOperationException
{
    /// <summary>The device of the first operand (the one others are compared against).</summary>
    public DeviceInfo Expected { get; }

    /// <summary>The device of the operand that did not match <see cref="Expected"/>.</summary>
    public DeviceInfo Actual { get; }

    /// <summary>Creates a device-mismatch error describing the two disagreeing devices and the fix.</summary>
    public DeviceMismatchException(DeviceInfo expected, DeviceInfo actual)
        : base($"All operand tensors must be on the same device, but found {expected} and {actual}. " +
               $"Move one so they match, e.g. tensor.To({expected}) or tensor.To({actual}) " +
               $"(or model.To(device) to move a whole model).")
    {
        Expected = expected;
        Actual = actual;
    }

    /// <summary>Creates a device-mismatch error with a caller-supplied message.</summary>
    public DeviceMismatchException(DeviceInfo expected, DeviceInfo actual, string message)
        : base(message)
    {
        Expected = expected;
        Actual = actual;
    }
}
