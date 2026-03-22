using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// P/Invoke bindings for libshaderc (Google's GLSL-to-SPIR-V compiler).
/// Ships with the Vulkan SDK. Enables runtime compilation of GLSL compute
/// shaders to SPIR-V, eliminating the need for hand-assembled bytecode.
/// </summary>
internal static class ShadercNativeBindings
{
    // shaderc ships as shaderc_shared.dll (Windows), libshaderc_shared.so (Linux), libshaderc_shared.dylib (macOS)
    private const string ShadercWindows = "shaderc_shared";
    private const string ShadercLinux = "libshaderc_shared";

    // Shader kind for compute shaders
    public const int shaderc_compute_shader = 5;

    // Compilation status
    public const int shaderc_compilation_status_success = 0;

    // Optimization level
    public const int shaderc_optimization_level_performance = 2;

    // --- Compiler lifecycle ---

    [DllImport(ShadercWindows, EntryPoint = "shaderc_compiler_initialize")]
    private static extern IntPtr shaderc_compiler_initialize_Windows();

    [DllImport(ShadercLinux, EntryPoint = "shaderc_compiler_initialize")]
    private static extern IntPtr shaderc_compiler_initialize_Linux();

    public static IntPtr shaderc_compiler_initialize()
    {
        try { return shaderc_compiler_initialize_Windows(); }
        catch (DllNotFoundException) { return shaderc_compiler_initialize_Linux(); }
    }

    [DllImport(ShadercWindows, EntryPoint = "shaderc_compiler_release")]
    private static extern void shaderc_compiler_release_Windows(IntPtr compiler);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_compiler_release")]
    private static extern void shaderc_compiler_release_Linux(IntPtr compiler);

    public static void shaderc_compiler_release(IntPtr compiler)
    {
        try { shaderc_compiler_release_Windows(compiler); }
        catch (DllNotFoundException) { shaderc_compiler_release_Linux(compiler); }
    }

    // --- Compile options ---

    [DllImport(ShadercWindows, EntryPoint = "shaderc_compile_options_initialize")]
    private static extern IntPtr shaderc_compile_options_initialize_Windows();

    [DllImport(ShadercLinux, EntryPoint = "shaderc_compile_options_initialize")]
    private static extern IntPtr shaderc_compile_options_initialize_Linux();

    public static IntPtr shaderc_compile_options_initialize()
    {
        try { return shaderc_compile_options_initialize_Windows(); }
        catch (DllNotFoundException) { return shaderc_compile_options_initialize_Linux(); }
    }

    [DllImport(ShadercWindows, EntryPoint = "shaderc_compile_options_release")]
    private static extern void shaderc_compile_options_release_Windows(IntPtr options);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_compile_options_release")]
    private static extern void shaderc_compile_options_release_Linux(IntPtr options);

    public static void shaderc_compile_options_release(IntPtr options)
    {
        try { shaderc_compile_options_release_Windows(options); }
        catch (DllNotFoundException) { shaderc_compile_options_release_Linux(options); }
    }

    [DllImport(ShadercWindows, EntryPoint = "shaderc_compile_options_set_optimization_level")]
    private static extern void shaderc_compile_options_set_optimization_level_Windows(IntPtr options, int level);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_compile_options_set_optimization_level")]
    private static extern void shaderc_compile_options_set_optimization_level_Linux(IntPtr options, int level);

    public static void shaderc_compile_options_set_optimization_level(IntPtr options, int level)
    {
        try { shaderc_compile_options_set_optimization_level_Windows(options, level); }
        catch (DllNotFoundException) { shaderc_compile_options_set_optimization_level_Linux(options, level); }
    }

    [DllImport(ShadercWindows, EntryPoint = "shaderc_compile_options_set_target_env")]
    private static extern void shaderc_compile_options_set_target_env_Windows(IntPtr options, int env, uint version);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_compile_options_set_target_env")]
    private static extern void shaderc_compile_options_set_target_env_Linux(IntPtr options, int env, uint version);

    public static void shaderc_compile_options_set_target_env(IntPtr options, int env, uint version)
    {
        try { shaderc_compile_options_set_target_env_Windows(options, env, version); }
        catch (DllNotFoundException) { shaderc_compile_options_set_target_env_Linux(options, env, version); }
    }

    // --- Compilation ---

    [DllImport(ShadercWindows, EntryPoint = "shaderc_compile_into_spv")]
    private static extern IntPtr shaderc_compile_into_spv_Windows(
        IntPtr compiler, string source, UIntPtr sourceSize,
        int shaderKind, string inputFileName, string entryPointName, IntPtr options);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_compile_into_spv")]
    private static extern IntPtr shaderc_compile_into_spv_Linux(
        IntPtr compiler, string source, UIntPtr sourceSize,
        int shaderKind, string inputFileName, string entryPointName, IntPtr options);

    public static IntPtr shaderc_compile_into_spv(
        IntPtr compiler, string source, UIntPtr sourceSize,
        int shaderKind, string inputFileName, string entryPointName, IntPtr options)
    {
        try { return shaderc_compile_into_spv_Windows(compiler, source, sourceSize, shaderKind, inputFileName, entryPointName, options); }
        catch (DllNotFoundException) { return shaderc_compile_into_spv_Linux(compiler, source, sourceSize, shaderKind, inputFileName, entryPointName, options); }
    }

    // --- Result inspection ---

    [DllImport(ShadercWindows, EntryPoint = "shaderc_result_get_compilation_status")]
    private static extern int shaderc_result_get_compilation_status_Windows(IntPtr result);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_result_get_compilation_status")]
    private static extern int shaderc_result_get_compilation_status_Linux(IntPtr result);

    public static int shaderc_result_get_compilation_status(IntPtr result)
    {
        try { return shaderc_result_get_compilation_status_Windows(result); }
        catch (DllNotFoundException) { return shaderc_result_get_compilation_status_Linux(result); }
    }

    [DllImport(ShadercWindows, EntryPoint = "shaderc_result_get_length")]
    private static extern UIntPtr shaderc_result_get_length_Windows(IntPtr result);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_result_get_length")]
    private static extern UIntPtr shaderc_result_get_length_Linux(IntPtr result);

    public static UIntPtr shaderc_result_get_length(IntPtr result)
    {
        try { return shaderc_result_get_length_Windows(result); }
        catch (DllNotFoundException) { return shaderc_result_get_length_Linux(result); }
    }

    [DllImport(ShadercWindows, EntryPoint = "shaderc_result_get_bytes")]
    private static extern IntPtr shaderc_result_get_bytes_Windows(IntPtr result);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_result_get_bytes")]
    private static extern IntPtr shaderc_result_get_bytes_Linux(IntPtr result);

    public static IntPtr shaderc_result_get_bytes(IntPtr result)
    {
        try { return shaderc_result_get_bytes_Windows(result); }
        catch (DllNotFoundException) { return shaderc_result_get_bytes_Linux(result); }
    }

    [DllImport(ShadercWindows, EntryPoint = "shaderc_result_get_error_message")]
    private static extern IntPtr shaderc_result_get_error_message_Windows(IntPtr result);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_result_get_error_message")]
    private static extern IntPtr shaderc_result_get_error_message_Linux(IntPtr result);

    public static IntPtr shaderc_result_get_error_message(IntPtr result)
    {
        try { return shaderc_result_get_error_message_Windows(result); }
        catch (DllNotFoundException) { return shaderc_result_get_error_message_Linux(result); }
    }

    [DllImport(ShadercWindows, EntryPoint = "shaderc_result_release")]
    private static extern void shaderc_result_release_Windows(IntPtr result);

    [DllImport(ShadercLinux, EntryPoint = "shaderc_result_release")]
    private static extern void shaderc_result_release_Linux(IntPtr result);

    public static void shaderc_result_release(IntPtr result)
    {
        try { shaderc_result_release_Windows(result); }
        catch (DllNotFoundException) { shaderc_result_release_Linux(result); }
    }

    // Environment constants
    public const int shaderc_target_env_vulkan = 0;
    public const uint shaderc_env_version_vulkan_1_0 = (1u << 22);
    public const uint shaderc_env_version_vulkan_1_1 = (1u << 22) | (1u << 12);
    public const uint shaderc_env_version_vulkan_1_2 = (1u << 22) | (2u << 12);
}
