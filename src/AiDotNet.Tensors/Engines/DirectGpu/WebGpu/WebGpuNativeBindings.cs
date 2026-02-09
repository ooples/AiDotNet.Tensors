// Copyright (c) AiDotNet. All rights reserved.
// WebGPU JavaScript interop bindings for Blazor WebAssembly.
// Only available in .NET 7+ with Blazor WebAssembly.

#if NET7_0_OR_GREATER
using System;
using System.Runtime.InteropServices.JavaScript;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// JavaScript interop bindings for WebGPU API.
/// These methods bridge C# to the browser's WebGPU implementation.
/// </summary>
/// <remarks>
/// <para><b>WebGPU Architecture:</b></para>
/// <para>
/// WebGPU is a modern graphics API providing:
/// </para>
/// <list type="bullet">
/// <item><b>GPU Adapter</b>: Physical GPU abstraction</item>
/// <item><b>GPU Device</b>: Logical connection to the adapter</item>
/// <item><b>GPU Buffer</b>: Memory on the GPU for compute data</item>
/// <item><b>GPU Shader Module</b>: Compiled WGSL compute shaders</item>
/// <item><b>GPU Compute Pipeline</b>: Configured compute operation</item>
/// <item><b>GPU Command Encoder</b>: Records GPU commands</item>
/// </list>
/// </remarks>
public static partial class WebGpuNativeBindings
{
    private const string ModuleName = "./js/webgpu-tensors.js";

    #region Initialization

    /// <summary>
    /// Checks if WebGPU is supported in the current browser.
    /// </summary>
    [JSImport("isWebGpuSupported", ModuleName)]
    public static partial bool IsWebGpuSupported();

    /// <summary>
    /// Initializes WebGPU and requests an adapter and device.
    /// </summary>
    /// <returns>True if initialization succeeded.</returns>
    [JSImport("initializeWebGpu", ModuleName)]
    public static partial Task<bool> InitializeWebGpuAsync();

    /// <summary>
    /// Gets the adapter name/description.
    /// </summary>
    [JSImport("getAdapterInfo", ModuleName)]
    public static partial string GetAdapterInfo();

    /// <summary>
    /// Gets device limits (max buffer size, max compute invocations, etc.).
    /// </summary>
    [JSImport("getDeviceLimits", ModuleName)]
    public static partial string GetDeviceLimitsJson();

    /// <summary>
    /// Destroys the WebGPU device and releases resources.
    /// </summary>
    [JSImport("destroyDevice", ModuleName)]
    public static partial void DestroyDevice();

    #endregion

    #region Buffer Operations

    /// <summary>
    /// Creates a GPU buffer with the specified size.
    /// </summary>
    /// <param name="sizeBytes">Size in bytes.</param>
    /// <param name="usage">Buffer usage flags (STORAGE, COPY_SRC, COPY_DST, MAP_READ, MAP_WRITE).</param>
    /// <returns>Buffer handle ID.</returns>
    [JSImport("createBuffer", ModuleName)]
    public static partial int CreateBuffer(int sizeBytes, int usage);

    /// <summary>
    /// Destroys a GPU buffer.
    /// </summary>
    /// <param name="bufferId">Buffer handle ID.</param>
    [JSImport("destroyBuffer", ModuleName)]
    public static partial void DestroyBuffer(int bufferId);

    /// <summary>
    /// Writes data to a GPU buffer (internal double[] version for JS interop).
    /// </summary>
    [JSImport("writeBuffer", ModuleName)]
    internal static partial void WriteBufferInternal(int bufferId, [JSMarshalAs<JSType.Array<JSType.Number>>] double[] data, int offsetBytes);

    /// <summary>
    /// Writes data to a GPU buffer.
    /// </summary>
    /// <param name="bufferId">Buffer handle ID.</param>
    /// <param name="data">Float array data to write.</param>
    /// <param name="offsetBytes">Offset in bytes.</param>
    public static void WriteBuffer(int bufferId, float[] data, int offsetBytes)
    {
        var doubleData = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            doubleData[i] = data[i];
        }
        WriteBufferInternal(bufferId, doubleData, offsetBytes);
    }

    /// <summary>
    /// Reads data from a GPU buffer. Returns JSON string of the data array.
    /// </summary>
    /// <remarks>
    /// Since Task{T[]} is not supported by JS interop for complex types,
    /// we return a JSON string that can be parsed on the C# side.
    /// </remarks>
    [JSImport("readBufferAsJson", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.String>>]
    internal static partial Task<string> ReadBufferAsJsonAsync(int bufferId, int sizeBytes, int offsetBytes);

    /// <summary>
    /// Reads data from a GPU buffer.
    /// </summary>
    /// <param name="bufferId">Buffer handle ID.</param>
    /// <param name="sizeBytes">Size to read in bytes.</param>
    /// <param name="offsetBytes">Offset in bytes.</param>
    /// <returns>Float array data.</returns>
    public static async Task<float[]> ReadBufferAsync(int bufferId, int sizeBytes, int offsetBytes)
    {
        var jsonData = await ReadBufferAsJsonAsync(bufferId, sizeBytes, offsetBytes);
        if (string.IsNullOrEmpty(jsonData))
        {
            return Array.Empty<float>();
        }

        // Parse JSON array of numbers
        var values = System.Text.Json.JsonSerializer.Deserialize<double[]>(jsonData);
        if (values is null)
        {
            return Array.Empty<float>();
        }

        var floatData = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            floatData[i] = (float)values[i];
        }
        return floatData;
    }

    /// <summary>
    /// Copies data between GPU buffers.
    /// </summary>
    [JSImport("copyBufferToBuffer", ModuleName)]
    public static partial void CopyBufferToBuffer(int srcBufferId, int srcOffset, int dstBufferId, int dstOffset, int sizeBytes);

    #endregion

    #region Shader Operations

    /// <summary>
    /// Creates a shader module from WGSL source code.
    /// </summary>
    /// <param name="wgslSource">WGSL shader source code.</param>
    /// <returns>Shader module ID.</returns>
    [JSImport("createShaderModule", ModuleName)]
    public static partial int CreateShaderModule(string wgslSource);

    /// <summary>
    /// Creates a compute pipeline from a shader module.
    /// </summary>
    /// <param name="shaderModuleId">Shader module ID.</param>
    /// <param name="entryPoint">Entry point function name.</param>
    /// <returns>Pipeline ID.</returns>
    [JSImport("createComputePipeline", ModuleName)]
    public static partial Task<int> CreateComputePipelineAsync(int shaderModuleId, string entryPoint);

    /// <summary>
    /// Destroys a shader module.
    /// </summary>
    [JSImport("destroyShaderModule", ModuleName)]
    public static partial void DestroyShaderModule(int shaderModuleId);

    /// <summary>
    /// Destroys a compute pipeline.
    /// </summary>
    [JSImport("destroyPipeline", ModuleName)]
    public static partial void DestroyPipeline(int pipelineId);

    #endregion

    #region Compute Dispatch

    /// <summary>
    /// Creates a bind group for a compute operation (internal double[] version for JS interop).
    /// </summary>
    [JSImport("createBindGroup", ModuleName)]
    internal static partial int CreateBindGroupInternal(int pipelineId, [JSMarshalAs<JSType.Array<JSType.Number>>] double[] bufferIds);

    /// <summary>
    /// Creates a bind group for a compute operation.
    /// </summary>
    /// <param name="pipelineId">Pipeline ID.</param>
    /// <param name="bufferIds">Array of buffer IDs to bind.</param>
    /// <returns>Bind group ID.</returns>
    public static int CreateBindGroup(int pipelineId, int[] bufferIds)
    {
        var doubleIds = new double[bufferIds.Length];
        for (int i = 0; i < bufferIds.Length; i++)
        {
            doubleIds[i] = bufferIds[i];
        }
        return CreateBindGroupInternal(pipelineId, doubleIds);
    }

    /// <summary>
    /// Destroys a bind group.
    /// </summary>
    [JSImport("destroyBindGroup", ModuleName)]
    public static partial void DestroyBindGroup(int bindGroupId);

    /// <summary>
    /// Dispatches a compute operation.
    /// </summary>
    /// <param name="pipelineId">Pipeline ID.</param>
    /// <param name="bindGroupId">Bind group ID.</param>
    /// <param name="workgroupsX">Number of workgroups in X dimension.</param>
    /// <param name="workgroupsY">Number of workgroups in Y dimension.</param>
    /// <param name="workgroupsZ">Number of workgroups in Z dimension.</param>
    [JSImport("dispatchCompute", ModuleName)]
    public static partial Task DispatchComputeAsync(int pipelineId, int bindGroupId, int workgroupsX, int workgroupsY, int workgroupsZ);

    /// <summary>
    /// Dispatches a compute operation with uniform buffer.
    /// </summary>
    /// <param name="pipelineId">Pipeline ID.</param>
    /// <param name="bindGroupId">Bind group ID.</param>
    /// <param name="uniformBufferId">Uniform buffer ID.</param>
    /// <param name="workgroupsX">Number of workgroups in X dimension.</param>
    /// <param name="workgroupsY">Number of workgroups in Y dimension.</param>
    /// <param name="workgroupsZ">Number of workgroups in Z dimension.</param>
    [JSImport("dispatchComputeWithUniforms", ModuleName)]
    public static partial Task DispatchComputeWithUniformsAsync(int pipelineId, int bindGroupId, int uniformBufferId, int workgroupsX, int workgroupsY, int workgroupsZ);

    /// <summary>
    /// Submits all pending GPU commands and waits for completion.
    /// </summary>
    [JSImport("submitAndWait", ModuleName)]
    public static partial Task SubmitAndWaitAsync();

    #endregion

    #region Synchronization

    /// <summary>
    /// Creates a GPU timestamp query for profiling.
    /// </summary>
    [JSImport("createTimestampQuery", ModuleName)]
    public static partial int CreateTimestampQuery();

    /// <summary>
    /// Reads timestamp query results in nanoseconds.
    /// </summary>
    [JSImport("readTimestampQuery", ModuleName)]
    public static partial Task<double> ReadTimestampQueryAsync(int queryId);

    /// <summary>
    /// Destroys a timestamp query.
    /// </summary>
    [JSImport("destroyTimestampQuery", ModuleName)]
    public static partial void DestroyTimestampQuery(int queryId);

    #endregion
}

/// <summary>
/// WebGPU buffer usage flags.
/// </summary>
[Flags]
public enum WebGpuBufferUsage : int
{
    /// <summary>None.</summary>
    None = 0,

    /// <summary>Buffer can be mapped for reading.</summary>
    MapRead = 1,

    /// <summary>Buffer can be mapped for writing.</summary>
    MapWrite = 2,

    /// <summary>Buffer can be used as a copy source.</summary>
    CopySrc = 4,

    /// <summary>Buffer can be used as a copy destination.</summary>
    CopyDst = 8,

    /// <summary>Buffer can be used as a vertex buffer.</summary>
    Vertex = 16,

    /// <summary>Buffer can be used as an index buffer.</summary>
    Index = 32,

    /// <summary>Buffer can be used as a uniform buffer.</summary>
    Uniform = 64,

    /// <summary>Buffer can be used as a storage buffer.</summary>
    Storage = 128,

    /// <summary>Buffer can be used for indirect draw/dispatch.</summary>
    Indirect = 256,

    /// <summary>Buffer can be used for query resolve.</summary>
    QueryResolve = 512
}

/// <summary>
/// WebGPU device limits.
/// </summary>
public sealed class WebGpuDeviceLimits
{
    /// <summary>Maximum buffer size in bytes.</summary>
    public long MaxBufferSize { get; set; }

    /// <summary>Maximum storage buffer binding size.</summary>
    public long MaxStorageBufferBindingSize { get; set; }

    /// <summary>Maximum uniform buffer binding size.</summary>
    public int MaxUniformBufferBindingSize { get; set; }

    /// <summary>Maximum compute workgroup size X.</summary>
    public int MaxComputeWorkgroupSizeX { get; set; }

    /// <summary>Maximum compute workgroup size Y.</summary>
    public int MaxComputeWorkgroupSizeY { get; set; }

    /// <summary>Maximum compute workgroup size Z.</summary>
    public int MaxComputeWorkgroupSizeZ { get; set; }

    /// <summary>Maximum compute invocations per workgroup.</summary>
    public int MaxComputeInvocationsPerWorkgroup { get; set; }

    /// <summary>Maximum compute workgroups per dimension.</summary>
    public int MaxComputeWorkgroupsPerDimension { get; set; }

    /// <summary>Maximum bind groups.</summary>
    public int MaxBindGroups { get; set; }

    /// <summary>Maximum bindings per bind group.</summary>
    public int MaxBindingsPerBindGroup { get; set; }

    /// <summary>Maximum storage buffers per shader stage.</summary>
    public int MaxStorageBuffersPerShaderStage { get; set; }
}
#endif
