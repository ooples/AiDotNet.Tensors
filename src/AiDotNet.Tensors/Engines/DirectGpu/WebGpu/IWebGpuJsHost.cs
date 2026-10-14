// Copyright (c) AiDotNet. All rights reserved.
#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

/// <summary>
/// Something that can run the functions <c>webgpu-tensors.js</c> exports. In a browser that is the JS runtime, reached
/// through <see cref="WebGpuNativeBindings"/>'s imports. Outside one, a host set on
/// <see cref="WebGpuNativeBindings.Host"/> runs the same module elsewhere, so the WebGPU backend can execute on a
/// desktop WebGPU implementation.
/// </summary>
internal interface IWebGpuJsHost
{
    /// <summary>
    /// Calls the module export named <paramref name="function"/> with the JSON array
    /// <paramref name="argumentsJson"/> as its arguments, waits for it (awaiting a returned promise), and returns its
    /// result as JSON (<c>null</c> for a function that returns nothing). Throws when the module has no such export or
    /// the call fails.
    /// </summary>
    string Invoke(string function, string argumentsJson);
}
#endif
