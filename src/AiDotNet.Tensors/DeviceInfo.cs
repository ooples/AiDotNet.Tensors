namespace AiDotNet.Tensors;

/// <summary>
/// Represents a specific compute device with type and index, equivalent to PyTorch's
/// <c>torch.device('cuda:0')</c>. Supports multi-GPU scenarios where tensors can be
/// placed on specific GPU devices.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> If you have multiple GPUs in your computer, you can
/// specify which one to use. <c>DeviceInfo.Cuda(0)</c> means the first NVIDIA GPU,
/// <c>DeviceInfo.Cuda(1)</c> means the second, etc.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// var device = DeviceInfo.Cuda(0);     // First NVIDIA GPU
/// var tensor = new Tensor&lt;float&gt;(data, shape).To(device);
/// Console.WriteLine(tensor.DeviceInfo); // "cuda:0"
/// </code>
/// </remarks>
public readonly struct DeviceInfo : IEquatable<DeviceInfo>
{
    /// <summary>The type of device (CPU, CUDA, OpenCL, etc.).</summary>
    public TensorDevice Type { get; }

    /// <summary>
    /// The device index for multi-device scenarios. 0 for the first device.
    /// -1 for CPU (device index is not applicable).
    /// </summary>
    public int Index { get; }

    /// <summary>
    /// Creates a DeviceInfo with the specified type and index.
    /// </summary>
    public DeviceInfo(TensorDevice type, int index = 0)
    {
        Type = type;
        Index = type == TensorDevice.CPU ? -1 : index;
    }

    /// <summary>CPU device (no index).</summary>
    public static DeviceInfo CPU => new(TensorDevice.CPU);

    /// <summary>Creates a CUDA device with the specified index.</summary>
    public static DeviceInfo Cuda(int index = 0) => new(TensorDevice.CUDA, index);

    /// <summary>Creates an OpenCL device with the specified index.</summary>
    public static DeviceInfo OpenCL(int index = 0) => new(TensorDevice.OpenCL, index);

    /// <summary>Creates a HIP/ROCm device with the specified index.</summary>
    public static DeviceInfo Hip(int index = 0) => new(TensorDevice.HIP, index);

    /// <summary>Creates a Vulkan device with the specified index.</summary>
    public static DeviceInfo Vulkan(int index = 0) => new(TensorDevice.Vulkan, index);

    /// <summary>Creates a Metal device with the specified index.</summary>
    public static DeviceInfo Metal(int index = 0) => new(TensorDevice.Metal, index);

    /// <summary>Creates a WebGPU device with the specified index.</summary>
    public static DeviceInfo WebGPU(int index = 0) => new(TensorDevice.WebGPU, index);

    /// <summary>Creates a DirectML device with the specified index.</summary>
    public static DeviceInfo DirectML(int index = 0) => new(TensorDevice.DirectML, index);

    /// <summary>Creates an NPU device with the specified index.</summary>
    public static DeviceInfo NPU(int index = 0) => new(TensorDevice.NPU, index);

    /// <summary>
    /// Parses a device string like "cuda:0", "cpu", "opencl:1".
    /// </summary>
    public static DeviceInfo Parse(string deviceString)
    {
        if (string.IsNullOrWhiteSpace(deviceString))
            return CPU;

        var parts = deviceString.Trim().ToLowerInvariant().Split(':');
        var typeName = parts[0];
        int index = parts.Length > 1 && int.TryParse(parts[1], out var idx) ? idx : 0;

        var type = typeName switch
        {
            "cpu" => TensorDevice.CPU,
            "cuda" or "nvidia" => TensorDevice.CUDA,
            "opencl" or "ocl" => TensorDevice.OpenCL,
            "hip" or "rocm" => TensorDevice.HIP,
            "vulkan" or "vk" => TensorDevice.Vulkan,
            "metal" or "mps" => TensorDevice.Metal,
            "webgpu" => TensorDevice.WebGPU,
            "directml" or "dml" => TensorDevice.DirectML,
            "npu" or "tpu" => TensorDevice.NPU,
            _ => throw new ArgumentException($"Unknown device type: {typeName}")
        };

        return new DeviceInfo(type, index);
    }

    /// <summary>Returns "cuda:0", "cpu", etc.</summary>
    public override string ToString()
    {
        if (Type == TensorDevice.CPU)
            return "cpu";

        var typeName = Type switch
        {
            TensorDevice.CUDA => "cuda",
            TensorDevice.OpenCL => "opencl",
            TensorDevice.HIP => "hip",
            TensorDevice.Vulkan => "vulkan",
            TensorDevice.Metal => "metal",
            TensorDevice.WebGPU => "webgpu",
            TensorDevice.DirectML => "directml",
            TensorDevice.NPU => "npu",
            _ => Type.ToString().ToLowerInvariant()
        };

        return $"{typeName}:{Index}";
    }

    public bool Equals(DeviceInfo other) => Type == other.Type && Index == other.Index;
    public override bool Equals(object? obj) => obj is DeviceInfo other && Equals(other);
    public override int GetHashCode() => ((int)Type * 397) ^ Index;
    public static bool operator ==(DeviceInfo left, DeviceInfo right) => left.Equals(right);
    public static bool operator !=(DeviceInfo left, DeviceInfo right) => !left.Equals(right);
}
