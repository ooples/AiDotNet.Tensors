using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// Runtime GLSL-to-SPIR-V compiler using libshaderc.
/// Compiles GLSL compute shader source strings to SPIR-V binary at runtime,
/// eliminating the need for hand-assembled SPIR-V bytecode arrays.
/// Caches compiled results for reuse across kernel launches.
/// </summary>
internal sealed class VulkanGlslCompiler : IDisposable
{
    private IntPtr _compiler;
    private IntPtr _options;
    private readonly ConcurrentDictionary<string, uint[]> _cache = new(StringComparer.Ordinal);
    private bool _available;

    public bool IsAvailable => _available;

    public VulkanGlslCompiler()
    {
        try
        {
            _compiler = ShadercNativeBindings.shaderc_compiler_initialize();
            if (_compiler == IntPtr.Zero)
            {
                _available = false;
                return;
            }

            _options = ShadercNativeBindings.shaderc_compile_options_initialize();
            if (_options != IntPtr.Zero)
            {
                ShadercNativeBindings.shaderc_compile_options_set_optimization_level(
                    _options, ShadercNativeBindings.shaderc_optimization_level_performance);
                ShadercNativeBindings.shaderc_compile_options_set_target_env(
                    _options, ShadercNativeBindings.shaderc_target_env_vulkan,
                    ShadercNativeBindings.shaderc_env_version_vulkan_1_0);
            }

            _available = true;
        }
        catch
        {
            _available = false;
        }
    }

    /// <summary>
    /// Compiles a GLSL compute shader to SPIR-V binary.
    /// Returns cached result if the same source was compiled before.
    /// </summary>
    /// <param name="glslSource">Complete GLSL compute shader source (must include #version directive).</param>
    /// <param name="entryPoint">Entry point function name (typically "main").</param>
    /// <returns>SPIR-V binary as uint array, or null if compilation failed.</returns>
    public uint[]? CompileToSpirv(string glslSource, string entryPoint = "main")
    {
        if (!_available)
            return null;

        if (_cache.TryGetValue(glslSource, out var cached))
            return cached;

        IntPtr result = IntPtr.Zero;
        try
        {
            result = ShadercNativeBindings.shaderc_compile_into_spv(
                _compiler,
                glslSource,
                (UIntPtr)glslSource.Length,
                ShadercNativeBindings.shaderc_compute_shader,
                "kernel.comp",
                entryPoint,
                _options);

            if (result == IntPtr.Zero)
                return null;

            int status = ShadercNativeBindings.shaderc_result_get_compilation_status(result);
            if (status != ShadercNativeBindings.shaderc_compilation_status_success)
            {
                IntPtr errPtr = ShadercNativeBindings.shaderc_result_get_error_message(result);
                string errorMsg = errPtr != IntPtr.Zero ? Marshal.PtrToStringAnsi(errPtr) ?? "Unknown error" : "Unknown error";
                System.Diagnostics.Debug.WriteLine($"[VulkanGlslCompiler] GLSL compilation failed: {errorMsg}");
                return null;
            }

            UIntPtr byteLength = ShadercNativeBindings.shaderc_result_get_length(result);
            IntPtr bytesPtr = ShadercNativeBindings.shaderc_result_get_bytes(result);

            if (bytesPtr == IntPtr.Zero || (int)byteLength == 0)
                return null;

            int wordCount = (int)byteLength / sizeof(uint);
            var spirv = new uint[wordCount];
            Marshal.Copy(bytesPtr, (int[])(object)spirv, 0, wordCount);

            _cache.TryAdd(glslSource, spirv);
            return spirv;
        }
        finally
        {
            if (result != IntPtr.Zero)
                ShadercNativeBindings.shaderc_result_release(result);
        }
    }

    /// <summary>
    /// Compiles a GLSL compute shader and creates a VulkanShaderModule from it.
    /// </summary>
    public VulkanShaderModule? CompileToShaderModule(string glslSource, string entryPoint = "main")
    {
        var spirv = CompileToSpirv(glslSource, entryPoint);
        if (spirv is null)
            return null;

        return VulkanShaderModule.Create(spirv);
    }

    public void Dispose()
    {
        if (_options != IntPtr.Zero)
        {
            ShadercNativeBindings.shaderc_compile_options_release(_options);
            _options = IntPtr.Zero;
        }

        if (_compiler != IntPtr.Zero)
        {
            ShadercNativeBindings.shaderc_compiler_release(_compiler);
            _compiler = IntPtr.Zero;
        }

        _available = false;
    }
}
