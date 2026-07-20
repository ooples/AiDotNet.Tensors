using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors;

/// <summary>
/// How strictly an operation must agree on the device of its operand tensors.
/// </summary>
public enum DeviceDispatchMode
{
    /// <summary>
    /// A CPU operand mixed with a GPU operand is uploaded to the GPU (the historical auto-upload behavior).
    /// Keeps every existing caller working while the op surface is migrated to device-keyed dispatch.
    /// </summary>
    Permissive = 0,

    /// <summary>
    /// An operation runs on its operands' device; a CPU operand mixed with a GPU operand (or two operands on
    /// different GPUs) throws <see cref="DeviceMismatchException"/>. The PyTorch-equivalent strict rule.
    /// </summary>
    Strict = 1,
}

/// <summary>
/// The outcome of resolving a set of operand tensors to the single device an operation should run on.
/// </summary>
public readonly struct DeviceResolution
{
    /// <summary>The resolved device type the op should execute on.</summary>
    public TensorDevice Device { get; }

    /// <summary>The resolved device index (for multi-GPU); 0 for CPU / single GPU.</summary>
    public int Index { get; }

    /// <summary>Creates a resolution for the given device and index.</summary>
    public DeviceResolution(TensorDevice device, int index)
    {
        Device = device;
        Index = index;
    }

    /// <summary>True when the resolved device is a GPU (anything other than <see cref="TensorDevice.CPU"/>).</summary>
    public bool IsGpu => Device != TensorDevice.CPU;

    /// <summary>The resolution as a <see cref="DeviceInfo"/>.</summary>
    public DeviceInfo ToDeviceInfo() => new DeviceInfo(Device, Index);
}

/// <summary>
/// The single, central policy that decides which device an operation runs on from the devices of its operand
/// tensors. Every device-aware engine op resolves through here instead of ad-hoc residency checks, so device
/// dispatch is defined in one place and strictness is a single global switch.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When you multiply two tensors, the library must decide whether to run on the CPU
/// or a GPU. That decision should depend on where the tensors already live — not on some hidden global setting.
/// This helper looks at every input's <see cref="TensorBase{T}.DeviceInfo"/> and returns the one device they all
/// agree on. If they disagree, it either uploads the CPU one to the GPU (permissive) or tells you to fix it
/// (strict), based on <see cref="Mode"/>.</para>
/// </remarks>
public static class DeviceDispatch
{
    /// <summary>
    /// The active strictness. Defaults to <see cref="DeviceDispatchMode.Permissive"/> so wiring ops through
    /// <see cref="Resolve{T}"/> changes no behavior until the whole op surface is migrated, at which point this
    /// is flipped to <see cref="DeviceDispatchMode.Strict"/>.
    /// </summary>
    public static DeviceDispatchMode Mode { get; set; } = DeviceDispatchMode.Permissive;

    /// <summary>
    /// Resolves the device an op should run on from its operand tensors.
    /// <list type="bullet">
    /// <item>No GPU operands ⇒ <see cref="TensorDevice.CPU"/>.</item>
    /// <item>All GPU operands on the same (device, index), no CPU operand ⇒ that GPU.</item>
    /// <item>GPU + CPU mix ⇒ that GPU in <see cref="DeviceDispatchMode.Permissive"/> (caller uploads the CPU
    /// operand); <see cref="DeviceMismatchException"/> in <see cref="DeviceDispatchMode.Strict"/>.</item>
    /// <item>Two GPU operands on different devices/indices ⇒ always <see cref="DeviceMismatchException"/>
    /// (an op cannot span two GPUs).</item>
    /// </list>
    /// Null and zero-length (deferred, not-yet-materialized) operands are ignored — they carry no device.
    /// </summary>
    public static DeviceResolution Resolve<T>(params Tensor<T>[] operands)
    {
        bool sawGpu = false;
        bool sawCpu = false;
        TensorDevice gpuDevice = TensorDevice.CPU;
        int gpuIndex = 0;

        if (operands is not null)
        {
            for (int i = 0; i < operands.Length; i++)
            {
                var t = operands[i];
                if (t is null || t.Length == 0) continue;

                if (t.Device == TensorDevice.CPU)
                {
                    sawCpu = true;
                    continue;
                }

                if (!sawGpu)
                {
                    sawGpu = true;
                    gpuDevice = t.Device;
                    gpuIndex = t.DeviceInfo.Index;
                }
                else if (t.Device != gpuDevice || t.DeviceInfo.Index != gpuIndex)
                {
                    // Two different GPUs in the same op is never runnable — always an error.
                    throw new DeviceMismatchException(new DeviceInfo(gpuDevice, gpuIndex), t.DeviceInfo);
                }
            }
        }

        if (!sawGpu)
            return new DeviceResolution(TensorDevice.CPU, 0);

        if (sawCpu && Mode == DeviceDispatchMode.Strict)
            throw new DeviceMismatchException(new DeviceInfo(gpuDevice, gpuIndex), DeviceInfo.CPU);

        // Permissive GPU+CPU mix, or a clean all-GPU op: run on the GPU.
        return new DeviceResolution(gpuDevice, gpuIndex);
    }
}
