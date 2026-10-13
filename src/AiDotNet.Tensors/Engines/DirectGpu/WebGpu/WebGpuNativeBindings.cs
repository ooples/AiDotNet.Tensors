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

    /// <summary>
    /// When set, every call below goes to this host and not to the browser's JS runtime. A host runs the same
    /// <c>webgpu-tensors.js</c> module somewhere other than a Blazor WebAssembly page (the tests run it in Node on
    /// Dawn), which is the only way the backend's own dispatch executes outside a browser. Null in production.
    /// </summary>
    internal static IWebGpuJsHost? Host { get; set; }

    // NaN and the infinities have no JSON number form; both sides carry them as the strings "NaN", "Infinity" and
    // "-Infinity", which is also what a JS typed array constructor turns back into the right value. A read also
    // sends negative zero as the string "-0", because JSON.stringify writes it as 0.
    private static readonly System.Text.Json.JsonSerializerOptions HostJson = new()
    {
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
            | System.Text.Json.Serialization.JsonNumberHandling.AllowReadingFromString,
    };

    private static string Args(params object?[] arguments) => System.Text.Json.JsonSerializer.Serialize(arguments, HostJson);

    private static T HostValue<T>(string json) => System.Text.Json.JsonSerializer.Deserialize<T>(json, HostJson)!;

    #region Initialization

    /// <summary>
    /// Checks if WebGPU is supported in the current browser.
    /// </summary>
    public static bool IsWebGpuSupported()
        => Host is { } host ? HostValue<bool>(host.Invoke("isWebGpuSupported", Args())) : JsIsWebGpuSupported();

    [JSImport("isWebGpuSupported", ModuleName)]
    private static partial bool JsIsWebGpuSupported();

    /// <summary>
    /// Initializes WebGPU and requests an adapter and device.
    /// </summary>
    /// <returns>True if initialization succeeded.</returns>
    public static Task<bool> InitializeWebGpuAsync()
        => Host is { } host ? Task.FromResult(HostValue<bool>(host.Invoke("initializeWebGpu", Args()))) : JsInitializeWebGpuAsync();

    [JSImport("initializeWebGpu", ModuleName)]
    private static partial Task<bool> JsInitializeWebGpuAsync();

    /// <summary>
    /// Gets the adapter name/description.
    /// </summary>
    public static string GetAdapterInfo()
        => Host is { } host ? HostValue<string>(host.Invoke("getAdapterInfo", Args())) : JsGetAdapterInfo();

    [JSImport("getAdapterInfo", ModuleName)]
    private static partial string JsGetAdapterInfo();

    /// <summary>
    /// Gets device limits (max buffer size, max compute invocations, etc.).
    /// </summary>
    public static string GetDeviceLimitsJson()
        => Host is { } host ? HostValue<string>(host.Invoke("getDeviceLimitsJson", Args())) : JsGetDeviceLimitsJson();

    [JSImport("getDeviceLimitsJson", ModuleName)]
    private static partial string JsGetDeviceLimitsJson();

    /// <summary>
    /// Destroys the WebGPU device and releases resources.
    /// </summary>
    public static void DestroyDevice()
    {
        if (Host is { } host) host.Invoke("destroyDevice", Args());
        else JsDestroyDevice();
    }

    [JSImport("destroyDevice", ModuleName)]
    private static partial void JsDestroyDevice();

    #endregion

    #region Buffer Operations

    /// <summary>
    /// Creates a GPU buffer with the specified size.
    /// </summary>
    /// <param name="sizeBytes">Size in bytes.</param>
    /// <param name="usage">Buffer usage flags (STORAGE, COPY_SRC, COPY_DST, MAP_READ, MAP_WRITE).</param>
    /// <returns>Buffer handle ID.</returns>
    public static int CreateBuffer(int sizeBytes, int usage)
        => Host is { } host ? HostValue<int>(host.Invoke("createBuffer", Args(sizeBytes, usage))) : JsCreateBuffer(sizeBytes, usage);

    [JSImport("createBuffer", ModuleName)]
    private static partial int JsCreateBuffer(int sizeBytes, int usage);

    /// <summary>
    /// Destroys a GPU buffer.
    /// </summary>
    /// <param name="bufferId">Buffer handle ID.</param>
    public static void DestroyBuffer(int bufferId)
    {
        if (Host is { } host) host.Invoke("destroyBuffer", Args(bufferId));
        else JsDestroyBuffer(bufferId);
    }

    [JSImport("destroyBuffer", ModuleName)]
    private static partial void JsDestroyBuffer(int bufferId);

    /// <summary>
    /// Writes data to a GPU buffer (internal double[] version for JS interop).
    /// </summary>
    internal static void WriteBufferInternal(int bufferId, double[] data, int offsetBytes)
    {
        if (Host is { } host) host.Invoke("writeBuffer", Args(bufferId, data, offsetBytes));
        else JsWriteBufferInternal(bufferId, data, offsetBytes);
    }

    [JSImport("writeBuffer", ModuleName)]
    private static partial void JsWriteBufferInternal(int bufferId, [JSMarshalAs<JSType.Array<JSType.Number>>] double[] data, int offsetBytes);

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
    /// Writes raw bytes to a GPU buffer (internal double[] version for JS interop).
    /// </summary>
    internal static void WriteByteBufferInternal(int bufferId, double[] data, int offsetBytes)
    {
        if (Host is { } host) host.Invoke("writeByteBuffer", Args(bufferId, data, offsetBytes));
        else JsWriteByteBufferInternal(bufferId, data, offsetBytes);
    }

    [JSImport("writeByteBuffer", ModuleName)]
    private static partial void JsWriteByteBufferInternal(int bufferId, [JSMarshalAs<JSType.Array<JSType.Number>>] double[] data, int offsetBytes);

    /// <summary>
    /// Writes raw bytes to a GPU buffer.
    /// </summary>
    public static void WriteByteBuffer(int bufferId, byte[] data, int offsetBytes)
    {
        var doubleData = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            doubleData[i] = data[i];
        }
        WriteByteBufferInternal(bufferId, doubleData, offsetBytes);
    }

    /// <summary>
    /// Reads data from a GPU buffer. Returns JSON string of the data array.
    /// </summary>
    /// <remarks>
    /// Since Task{T[]} is not supported by JS interop for complex types,
    /// we return a JSON string that can be parsed on the C# side.
    /// </remarks>
    internal static Task<string> ReadBufferAsJsonAsync(int bufferId, int sizeBytes, int offsetBytes)
        => Host is { } host ? Task.FromResult(HostValue<string>(host.Invoke("readBufferAsJson", Args(bufferId, sizeBytes, offsetBytes)))) : JsReadBufferAsJsonAsync(bufferId, sizeBytes, offsetBytes);

    [JSImport("readBufferAsJson", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.String>>]
    private static partial Task<string> JsReadBufferAsJsonAsync(int bufferId, int sizeBytes, int offsetBytes);

    /// <summary>
    /// Reads raw bytes from a GPU buffer. Returns JSON string of byte values.
    /// </summary>
    internal static Task<string> ReadByteBufferAsJsonAsync(int bufferId, int sizeBytes, int offsetBytes)
        => Host is { } host ? Task.FromResult(HostValue<string>(host.Invoke("readByteBufferAsJson", Args(bufferId, sizeBytes, offsetBytes)))) : JsReadByteBufferAsJsonAsync(bufferId, sizeBytes, offsetBytes);

    [JSImport("readByteBufferAsJson", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.String>>]
    private static partial Task<string> JsReadByteBufferAsJsonAsync(int bufferId, int sizeBytes, int offsetBytes);

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
        var values = System.Text.Json.JsonSerializer.Deserialize<double[]>(jsonData, HostJson);
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
    /// Reads raw bytes from a GPU buffer.
    /// </summary>
    public static async Task<byte[]> ReadByteBufferAsync(int bufferId, int sizeBytes, int offsetBytes)
    {
        var jsonData = await ReadByteBufferAsJsonAsync(bufferId, sizeBytes, offsetBytes);
        if (string.IsNullOrEmpty(jsonData))
        {
            return Array.Empty<byte>();
        }

        var values = System.Text.Json.JsonSerializer.Deserialize<double[]>(jsonData, HostJson);
        if (values is null)
        {
            return Array.Empty<byte>();
        }

        var byteData = new byte[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            byteData[i] = checked((byte)values[i]);
        }
        return byteData;
    }

    /// <summary>
    /// Copies data between GPU buffers.
    /// </summary>
    public static void CopyBufferToBuffer(int srcBufferId, int srcOffset, int dstBufferId, int dstOffset, int sizeBytes)
    {
        if (Host is { } host) host.Invoke("copyBufferToBuffer", Args(srcBufferId, srcOffset, dstBufferId, dstOffset, sizeBytes));
        else JsCopyBufferToBuffer(srcBufferId, srcOffset, dstBufferId, dstOffset, sizeBytes);
    }

    [JSImport("copyBufferToBuffer", ModuleName)]
    private static partial void JsCopyBufferToBuffer(int srcBufferId, int srcOffset, int dstBufferId, int dstOffset, int sizeBytes);

    #endregion

    #region Shader Operations

    /// <summary>
    /// Creates a shader module from WGSL source code.
    /// </summary>
    /// <param name="wgslSource">WGSL shader source code.</param>
    /// <returns>Shader module ID.</returns>
    public static int CreateShaderModule(string wgslSource)
        => Host is { } host ? HostValue<int>(host.Invoke("createShaderModule", Args(wgslSource))) : JsCreateShaderModule(wgslSource);

    [JSImport("createShaderModule", ModuleName)]
    private static partial int JsCreateShaderModule(string wgslSource);

    /// <summary>
    /// Creates a compute pipeline from a shader module.
    /// </summary>
    /// <param name="shaderModuleId">Shader module ID.</param>
    /// <param name="entryPoint">Entry point function name.</param>
    /// <returns>Pipeline ID.</returns>
    public static Task<int> CreateComputePipelineAsync(int shaderModuleId, string entryPoint)
        => Host is { } host ? Task.FromResult(HostValue<int>(host.Invoke("createComputePipeline", Args(shaderModuleId, entryPoint)))) : JsCreateComputePipelineAsync(shaderModuleId, entryPoint);

    [JSImport("createComputePipeline", ModuleName)]
    private static partial Task<int> JsCreateComputePipelineAsync(int shaderModuleId, string entryPoint);

    /// <summary>
    /// Destroys a shader module.
    /// </summary>
    public static void DestroyShaderModule(int shaderModuleId)
    {
        if (Host is { } host) host.Invoke("destroyShaderModule", Args(shaderModuleId));
        else JsDestroyShaderModule(shaderModuleId);
    }

    [JSImport("destroyShaderModule", ModuleName)]
    private static partial void JsDestroyShaderModule(int shaderModuleId);

    /// <summary>
    /// Destroys a compute pipeline.
    /// </summary>
    public static void DestroyPipeline(int pipelineId)
    {
        if (Host is { } host) host.Invoke("destroyPipeline", Args(pipelineId));
        else JsDestroyPipeline(pipelineId);
    }

    [JSImport("destroyPipeline", ModuleName)]
    private static partial void JsDestroyPipeline(int pipelineId);

    #endregion

    #region Compute Dispatch

    /// <summary>
    /// Creates a bind group for a compute operation (internal double[] version for JS interop).
    /// </summary>
    internal static int CreateBindGroupInternal(int pipelineId, double[] bufferIds)
        => Host is { } host ? HostValue<int>(host.Invoke("createBindGroup", Args(pipelineId, bufferIds))) : JsCreateBindGroupInternal(pipelineId, bufferIds);

    [JSImport("createBindGroup", ModuleName)]
    private static partial int JsCreateBindGroupInternal(int pipelineId, [JSMarshalAs<JSType.Array<JSType.Number>>] double[] bufferIds);

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

    internal static int CreateBindGroupRangesInternal(int pipelineId, double[] bufferIds, double[] offsetBytes, double[] sizeBytes)
        => Host is { } host ? HostValue<int>(host.Invoke("createBindGroupRanges", Args(pipelineId, bufferIds, offsetBytes, sizeBytes))) : JsCreateBindGroupRangesInternal(pipelineId, bufferIds, offsetBytes, sizeBytes);

    [JSImport("createBindGroupRanges", ModuleName)]
    private static partial int JsCreateBindGroupRangesInternal(int pipelineId,
        [JSMarshalAs<JSType.Array<JSType.Number>>] double[] bufferIds,
        [JSMarshalAs<JSType.Array<JSType.Number>>] double[] offsetBytes,
        [JSMarshalAs<JSType.Array<JSType.Number>>] double[] sizeBytes);

    /// <summary>
    /// Creates a bind group whose entries bind a byte range of each buffer (buffer views, #1142).
    /// </summary>
    public static int CreateBindGroupRanges(int pipelineId, int[] bufferIds, int[] offsetBytes, int[] sizeBytes)
    {
        if (offsetBytes.Length != bufferIds.Length || sizeBytes.Length != bufferIds.Length)
            throw new ArgumentException("Every bound buffer needs one offset and one size.", nameof(offsetBytes));
        var ids = new double[bufferIds.Length];
        var offsets = new double[bufferIds.Length];
        var sizes = new double[bufferIds.Length];
        for (int i = 0; i < bufferIds.Length; i++)
        {
            ids[i] = bufferIds[i];
            offsets[i] = offsetBytes[i];
            sizes[i] = sizeBytes[i];
        }
        return CreateBindGroupRangesInternal(pipelineId, ids, offsets, sizes);
    }
    /// <summary>
    /// Destroys a bind group.
    /// </summary>
    public static void DestroyBindGroup(int bindGroupId)
    {
        if (Host is { } host) host.Invoke("destroyBindGroup", Args(bindGroupId));
        else JsDestroyBindGroup(bindGroupId);
    }

    [JSImport("destroyBindGroup", ModuleName)]
    private static partial void JsDestroyBindGroup(int bindGroupId);

    /// <summary>
    /// Dispatches a compute operation.
    /// </summary>
    /// <param name="pipelineId">Pipeline ID.</param>
    /// <param name="bindGroupId">Bind group ID.</param>
    /// <param name="workgroupsX">Number of workgroups in X dimension.</param>
    /// <param name="workgroupsY">Number of workgroups in Y dimension.</param>
    /// <param name="workgroupsZ">Number of workgroups in Z dimension.</param>
    public static Task DispatchComputeAsync(int pipelineId, int bindGroupId, int workgroupsX, int workgroupsY, int workgroupsZ)
    {
        if (Host is not { } host) return JsDispatchComputeAsync(pipelineId, bindGroupId, workgroupsX, workgroupsY, workgroupsZ);
        host.Invoke("dispatchCompute", Args(pipelineId, bindGroupId, workgroupsX, workgroupsY, workgroupsZ));
        return Task.CompletedTask;
    }

    [JSImport("dispatchCompute", ModuleName)]
    private static partial Task JsDispatchComputeAsync(int pipelineId, int bindGroupId, int workgroupsX, int workgroupsY, int workgroupsZ);

    /// <summary>
    /// Dispatches a compute operation with uniform buffer.
    /// </summary>
    /// <param name="pipelineId">Pipeline ID.</param>
    /// <param name="bindGroupId">Bind group ID.</param>
    /// <param name="uniformBufferId">Uniform buffer ID.</param>
    /// <param name="workgroupsX">Number of workgroups in X dimension.</param>
    /// <param name="workgroupsY">Number of workgroups in Y dimension.</param>
    /// <param name="workgroupsZ">Number of workgroups in Z dimension.</param>
    public static Task DispatchComputeWithUniformsAsync(int pipelineId, int bindGroupId, int uniformBufferId, int workgroupsX, int workgroupsY, int workgroupsZ)
    {
        if (Host is not { } host) return JsDispatchComputeWithUniformsAsync(pipelineId, bindGroupId, uniformBufferId, workgroupsX, workgroupsY, workgroupsZ);
        host.Invoke("dispatchComputeWithUniforms", Args(pipelineId, bindGroupId, uniformBufferId, workgroupsX, workgroupsY, workgroupsZ));
        return Task.CompletedTask;
    }

    [JSImport("dispatchComputeWithUniforms", ModuleName)]
    private static partial Task JsDispatchComputeWithUniformsAsync(int pipelineId, int bindGroupId, int uniformBufferId, int workgroupsX, int workgroupsY, int workgroupsZ);

    /// <summary>
    /// Submits all pending GPU commands and waits for completion.
    /// </summary>
    public static Task SubmitAndWaitAsync()
    {
        if (Host is not { } host) return JsSubmitAndWaitAsync();
        host.Invoke("submitAndWait", Args());
        return Task.CompletedTask;
    }

    [JSImport("submitAndWait", ModuleName)]
    private static partial Task JsSubmitAndWaitAsync();

    #endregion

    #region Synchronization

    /// <summary>
    /// Creates a GPU timestamp query for profiling.
    /// </summary>
    public static int CreateTimestampQuery()
        => Host is { } host ? HostValue<int>(host.Invoke("createTimestampQuery", Args())) : JsCreateTimestampQuery();

    [JSImport("createTimestampQuery", ModuleName)]
    private static partial int JsCreateTimestampQuery();

    /// <summary>
    /// Reads timestamp query results in nanoseconds.
    /// </summary>
    public static Task<double> ReadTimestampQueryAsync(int queryId)
        => Host is { } host ? Task.FromResult(HostValue<double>(host.Invoke("readTimestampQuery", Args(queryId)))) : JsReadTimestampQueryAsync(queryId);

    [JSImport("readTimestampQuery", ModuleName)]
    private static partial Task<double> JsReadTimestampQueryAsync(int queryId);

    /// <summary>
    /// Destroys a timestamp query.
    /// </summary>
    public static void DestroyTimestampQuery(int queryId)
    {
        if (Host is { } host) host.Invoke("destroyTimestampQuery", Args(queryId));
        else JsDestroyTimestampQuery(queryId);
    }

    [JSImport("destroyTimestampQuery", ModuleName)]
    private static partial void JsDestroyTimestampQuery(int queryId);

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
