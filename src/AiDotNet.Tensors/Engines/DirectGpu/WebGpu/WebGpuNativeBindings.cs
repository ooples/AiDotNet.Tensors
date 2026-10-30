// Copyright (c) CrowdTrainer LLC. All rights reserved.
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
        // Every module and pipeline id belongs to the device.
        s_moduleSources.Clear();
        s_pipelineEntries.Clear();
        s_foldedPipelines.Clear();
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

    // Buffer contents cross to JavaScript as base64 text, not as arrays of numbers. A number array cannot carry
    // every bit pattern: an int reinterpreted as a float (how int buffers and the integer fields of kernel
    // parameters are uploaded) is a NaN whenever it is negative, and a NaN's payload does not survive the trip
    // through a JS number. Base64 is exact for any bytes, and it is one string to marshal instead of one boxed
    // number per element.

    /// <summary>Writes <paramref name="base64"/> (the bytes to store) into a buffer at <paramref name="offsetBytes"/>.</summary>
    internal static void WriteBufferBase64(int bufferId, string base64, int offsetBytes)
    {
        if (Host is { } host) host.Invoke("writeBufferBase64", Args(bufferId, base64, offsetBytes));
        else JsWriteBufferBase64(bufferId, base64, offsetBytes);
    }

    [JSImport("writeBufferBase64", ModuleName)]
    private static partial void JsWriteBufferBase64(int bufferId, string base64, int offsetBytes);

    /// <summary>Reads <paramref name="sizeBytes"/> bytes of a buffer as base64 text.</summary>
    internal static Task<string> ReadBufferBase64Async(int bufferId, int sizeBytes, int offsetBytes)
        => Host is { } host ? Task.FromResult(HostValue<string>(host.Invoke("readBufferBase64", Args(bufferId, sizeBytes, offsetBytes)))) : JsReadBufferBase64Async(bufferId, sizeBytes, offsetBytes);

    [JSImport("readBufferBase64", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.String>>]
    private static partial Task<string> JsReadBufferBase64Async(int bufferId, int sizeBytes, int offsetBytes);

    /// <summary>
    /// Writes float data to a GPU buffer, bit for bit.
    /// </summary>
    /// <param name="bufferId">Buffer handle ID.</param>
    /// <param name="data">Data to write.</param>
    /// <param name="offsetBytes">Offset in bytes.</param>
    public static void WriteBuffer(int bufferId, float[] data, int offsetBytes)
    {
        var bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        WriteBufferBase64(bufferId, Convert.ToBase64String(bytes), offsetBytes);
    }

    /// <summary>
    /// Writes raw bytes to a GPU buffer. A length that is not a multiple of four is padded with zeros to the next
    /// word, which the buffer must have room for.
    /// </summary>
    /// <param name="bufferId">Buffer handle ID.</param>
    /// <param name="data">Bytes to write.</param>
    /// <param name="offsetBytes">Offset in bytes.</param>
    public static void WriteByteBuffer(int bufferId, byte[] data, int offsetBytes)
        => WriteBufferBase64(bufferId, Convert.ToBase64String(data), offsetBytes);

    /// <summary>
    /// Reads float data from a GPU buffer, bit for bit.
    /// </summary>
    /// <param name="bufferId">Buffer handle ID.</param>
    /// <param name="sizeBytes">Number of bytes to read.</param>
    /// <param name="offsetBytes">Offset in bytes.</param>
    /// <returns>Float array containing the data.</returns>
    public static async Task<float[]> ReadBufferAsync(int bufferId, int sizeBytes, int offsetBytes)
    {
        byte[] bytes = await ReadByteBufferAsync(bufferId, sizeBytes, offsetBytes);
        var floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, floats.Length * sizeof(float));
        return floats;
    }

    /// <summary>
    /// Reads raw bytes from a GPU buffer.
    /// </summary>
    /// <param name="bufferId">Buffer handle ID.</param>
    /// <param name="sizeBytes">Number of bytes to read.</param>
    /// <param name="offsetBytes">Offset in bytes.</param>
    /// <returns>The bytes read.</returns>
    public static async Task<byte[]> ReadByteBufferAsync(int bufferId, int sizeBytes, int offsetBytes)
    {
        string base64 = await ReadBufferBase64Async(bufferId, sizeBytes, offsetBytes);
        return string.IsNullOrEmpty(base64) ? Array.Empty<byte>() : Convert.FromBase64String(base64);
    }

    /// <summary>
    /// Submits a copy of <paramref name="sizeBytes"/> bytes to a staging buffer and starts mapping it, without waiting;
    /// returns the token <see cref="FinishReadbackAsync"/> takes (0 when nothing was started).
    /// </summary>
    internal static int BeginReadback(int bufferId, int sizeBytes, int offsetBytes)
        => Host is { } host ? HostValue<int>(host.Invoke("beginReadback", Args(bufferId, sizeBytes, offsetBytes)))
            : JsBeginReadback(bufferId, sizeBytes, offsetBytes);

    [JSImport("beginReadback", ModuleName)]
    private static partial int JsBeginReadback(int bufferId, int sizeBytes, int offsetBytes);

    /// <summary>Waits for the copy <see cref="BeginReadback"/> started and returns its bytes.</summary>
    internal static async Task<byte[]> FinishReadbackAsync(int token)
    {
        string base64 = Host is { } host ? HostValue<string>(host.Invoke("finishReadbackBase64", Args(token)))
            : await JsFinishReadbackBase64Async(token);
        return string.IsNullOrEmpty(base64) ? Array.Empty<byte>() : Convert.FromBase64String(base64);
    }

    [JSImport("finishReadbackBase64", ModuleName)]
    [return: JSMarshalAs<JSType.Promise<JSType.String>>]
    private static partial Task<string> JsFinishReadbackBase64Async(int token);

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
    {
        int id = Host is { } host ? HostValue<int>(host.Invoke("createShaderModule", Args(wgslSource))) : JsCreateShaderModule(wgslSource);
        if (id >= 0) s_moduleSources[id] = wgslSource;
        return id;
    }

    [JSImport("createShaderModule", ModuleName)]
    private static partial int JsCreateShaderModule(string wgslSource);

    /// <summary>
    /// Creates a compute pipeline from a shader module.
    /// </summary>
    /// <param name="shaderModuleId">Shader module ID.</param>
    /// <param name="entryPoint">Entry point function name.</param>
    /// <returns>Pipeline ID.</returns>
    public static async Task<int> CreateComputePipelineAsync(int shaderModuleId, string entryPoint)
    {
        int id = Host is { } host
            ? HostValue<int>(host.Invoke("createComputePipeline", Args(shaderModuleId, entryPoint)))
            : await JsCreateComputePipelineAsync(shaderModuleId, entryPoint).ConfigureAwait(false);
        if (id >= 0 && s_moduleSources.TryGetValue(shaderModuleId, out var source)) s_pipelineEntries[id] = (source, entryPoint);
        return id;
    }

    [JSImport("createComputePipeline", ModuleName)]
    private static partial Task<int> JsCreateComputePipelineAsync(int shaderModuleId, string entryPoint);

    /// <summary>
    /// Destroys a shader module.
    /// </summary>
    public static void DestroyShaderModule(int shaderModuleId)
    {
        s_moduleSources.TryRemove(shaderModuleId, out _);
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
        s_pipelineEntries.TryRemove(pipelineId, out _);
        if (s_foldedPipelines.TryRemove(pipelineId, out var folded) && folded >= 0) DestroyPipeline(folded);
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
        if (NeedsFold(workgroupsX, workgroupsY, workgroupsZ)) return DispatchFoldedAsync(pipelineId, bindGroupId, -1, workgroupsX);
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
        if (NeedsFold(workgroupsX, workgroupsY, workgroupsZ)) return DispatchFoldedAsync(pipelineId, bindGroupId, uniformBufferId, workgroupsX);
        if (Host is not { } host) return JsDispatchComputeWithUniformsAsync(pipelineId, bindGroupId, uniformBufferId, workgroupsX, workgroupsY, workgroupsZ);
        host.Invoke("dispatchComputeWithUniforms", Args(pipelineId, bindGroupId, uniformBufferId, workgroupsX, workgroupsY, workgroupsZ));
        return Task.CompletedTask;
    }

    [JSImport("dispatchComputeWithUniforms", ModuleName)]
    private static partial Task JsDispatchComputeWithUniformsAsync(int pipelineId, int bindGroupId, int uniformBufferId, int workgroupsX, int workgroupsY, int workgroupsZ);

    /// <summary>
    /// Dispatches a chain of compute operations, each with its uniform buffer, in one compute pass submitted once
    /// (#1452). Element <c>i</c> of every array describes dispatch <c>i</c>; the X workgroup count only (Y and Z 1).
    /// </summary>
    public static Task DispatchComputeChainWithUniformsAsync(int[] pipelineIds, int[] bindGroupIds, int[] uniformBufferIds, int[] workgroupsX)
    {
        if (pipelineIds.Length != bindGroupIds.Length || pipelineIds.Length != uniformBufferIds.Length || pipelineIds.Length != workgroupsX.Length)
            throw new ArgumentException("Every dispatch needs a pipeline, a bind group, a uniform buffer and a workgroup count.");
        foreach (int x in workgroupsX)
            if (NeedsFold(x, 1, 1)) return DispatchChainOneByOneAsync(pipelineIds, bindGroupIds, uniformBufferIds, workgroupsX);
        if (Host is not { } host)
            return JsDispatchComputeChainWithUniformsAsync(ToDoubles(pipelineIds), ToDoubles(bindGroupIds), ToDoubles(uniformBufferIds), ToDoubles(workgroupsX));
        host.Invoke("dispatchComputeChainWithUniforms", Args(pipelineIds, bindGroupIds, uniformBufferIds, workgroupsX));
        return Task.CompletedTask;
    }

    private static double[] ToDoubles(int[] values)
    {
        var doubles = new double[values.Length];
        for (int i = 0; i < values.Length; i++) doubles[i] = values[i];
        return doubles;
    }

    [JSImport("dispatchComputeChainWithUniforms", ModuleName)]
    private static partial Task JsDispatchComputeChainWithUniformsAsync([JSMarshalAs<JSType.Array<JSType.Number>>] double[] pipelineIds,
        [JSMarshalAs<JSType.Array<JSType.Number>>] double[] bindGroupIds, [JSMarshalAs<JSType.Array<JSType.Number>>] double[] uniformBufferIds,
        [JSMarshalAs<JSType.Array<JSType.Number>>] double[] workgroupsX);

    // A 1-D dispatch past the per-dimension limit runs the entry point's folded copy on a 2-D grid
    // (WebGpuDispatchFold), so no caller has to size its work to 65,535 workgroups. Each pipeline remembers its source
    // and entry point for that; its folded pipeline is built on first need (-1: the entry point cannot be folded).
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, string> s_moduleSources = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, (string Source, string EntryPoint)> s_pipelineEntries = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, int> s_foldedPipelines = new();

    /// <summary>The device's workgroups per dimension (WebGpuDevice sets it from the adapter's limits).</summary>
    internal static int MaxWorkgroupsPerDimension { get; set; } = 65535;

    private static bool NeedsFold(int workgroupsX, int workgroupsY, int workgroupsZ)
        => workgroupsX > MaxWorkgroupsPerDimension && workgroupsY == 1 && workgroupsZ == 1;

    private static async Task DispatchFoldedAsync(int pipelineId, int bindGroupId, int uniformBufferId, int workgroups)
    {
        var (x, y) = WebGpuDispatchFold.Grid(workgroups, MaxWorkgroupsPerDimension);
        int folded = await FoldedPipelineAsync(pipelineId).ConfigureAwait(false);
        int groups = CreateBuffer(16, (int)(WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst));
        if (groups < 0) throw new InvalidOperationException("Could not create the folded dispatch's uniform buffer.");
        try
        {
            var count = new byte[16];
            BitConverter.GetBytes((uint)workgroups).CopyTo(count, 0);
            WriteBufferBase64(groups, Convert.ToBase64String(count), 0);
            if (Host is { } host)
                host.Invoke("dispatchComputeFolded", Args(folded, bindGroupId, uniformBufferId, groups, x, y));
            else
                await JsDispatchComputeFoldedAsync(folded, bindGroupId, uniformBufferId, groups, x, y).ConfigureAwait(false);
        }
        finally
        {
            // Like a kernel's uniform buffer: work already submitted keeps it alive.
            DestroyBuffer(groups);
        }
    }

    private static async Task<int> FoldedPipelineAsync(int pipelineId)
    {
        if (!s_foldedPipelines.TryGetValue(pipelineId, out int folded))
        {
            folded = -1;
            if (s_pipelineEntries.TryGetValue(pipelineId, out var entry)
                && WebGpuDispatchFold.FoldSource(entry.Source, entry.EntryPoint) is { } source)
            {
                int module = CreateShaderModule(source);
                if (module >= 0) folded = await CreateComputePipelineAsync(module, entry.EntryPoint).ConfigureAwait(false);
            }
            folded = s_foldedPipelines.GetOrAdd(pipelineId, folded);
        }
        if (folded < 0)
            throw new GpuCapabilityUnavailableException(
                $"Pipeline {pipelineId} needs more workgroups than the device dispatches per dimension ({MaxWorkgroupsPerDimension}), and its entry point cannot be folded onto a 2-D grid.");
        return folded;
    }

    // A chain with a stage past the limit: the stages one dispatch each, in order (the queue keeps them ordered).
    private static async Task DispatchChainOneByOneAsync(int[] pipelineIds, int[] bindGroupIds, int[] uniformBufferIds, int[] workgroupsX)
    {
        for (int i = 0; i < pipelineIds.Length; i++)
            await DispatchComputeWithUniformsAsync(pipelineIds[i], bindGroupIds[i], uniformBufferIds[i], workgroupsX[i], 1, 1).ConfigureAwait(false);
    }

    [JSImport("dispatchComputeFolded", ModuleName)]
    private static partial Task JsDispatchComputeFoldedAsync(int pipelineId, int bindGroupId, int uniformBufferId, int groupsBufferId, int workgroupsX, int workgroupsY);

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
    /// Writes the GPU timestamp <paramref name="index"/> (0 or 1) of a query from <see cref="CreateTimestampQuery"/>, in a
    /// submission of its own: work submitted between the two writes is what the query times.
    /// </summary>
    public static void WriteTimestamp(int queryId, int index)
    {
        if (Host is { } host) host.Invoke("writeTimestamp", Args(queryId, index));
        else JsWriteTimestamp(queryId, index);
    }

    [JSImport("writeTimestamp", ModuleName)]
    private static partial void JsWriteTimestamp(int queryId, int index);

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
