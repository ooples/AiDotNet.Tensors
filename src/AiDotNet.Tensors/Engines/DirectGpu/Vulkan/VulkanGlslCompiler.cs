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
    private volatile string? _lastError;

    public bool IsAvailable => _available;

    /// <summary>
    /// The reason the most recent compile attempt failed, or <see langword="null"/> when the
    /// last attempt succeeded. shaderc's diagnostics used to be written to
    /// <see cref="System.Diagnostics.Debug"/> and discarded in Release builds, which made a
    /// genuine GLSL syntax error indistinguishable from a missing libshaderc.
    /// </summary>
    public string? LastError => _lastError;

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
        {
            _lastError = "libshaderc is unavailable: shaderc_compiler_initialize failed or the native library could not be loaded.";
            return null;
        }

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
            {
                _lastError = "shaderc_compile_into_spv returned no result handle.";
                return null;
            }

            int status = ShadercNativeBindings.shaderc_result_get_compilation_status(result);
            if (status != ShadercNativeBindings.shaderc_compilation_status_success)
            {
                IntPtr errPtr = ShadercNativeBindings.shaderc_result_get_error_message(result);
                string errorMsg = errPtr != IntPtr.Zero ? Marshal.PtrToStringAnsi(errPtr) ?? "Unknown error" : "Unknown error";
                _lastError = $"GLSL compilation failed (shaderc status {status}): {errorMsg}";
                System.Diagnostics.Debug.WriteLine($"[VulkanGlslCompiler] {_lastError}");
                return null;
            }

            UIntPtr byteLength = ShadercNativeBindings.shaderc_result_get_length(result);
            IntPtr bytesPtr = ShadercNativeBindings.shaderc_result_get_bytes(result);

            if (bytesPtr == IntPtr.Zero || (int)byteLength == 0)
            {
                _lastError = "shaderc reported success but produced an empty SPIR-V module.";
                return null;
            }

            int byteCount = (int)byteLength;
            int wordCount = byteCount / sizeof(uint);
            var spirv = new uint[wordCount];
            var bytes = new byte[byteCount];
            Marshal.Copy(bytesPtr, bytes, 0, byteCount);
            Buffer.BlockCopy(bytes, 0, spirv, 0, byteCount);

            _lastError = null;
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

        var module = VulkanShaderModule.Create(spirv);
        if (module is null)
            _lastError = "vkCreateShaderModule failed for the compiled SPIR-V module.";
        return module;
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
