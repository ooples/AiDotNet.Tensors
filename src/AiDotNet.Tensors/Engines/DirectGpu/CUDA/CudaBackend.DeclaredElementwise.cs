// Copyright (c) AiDotNet. All rights reserved.
// The pointwise family on declared kernels (#890 phase 4a): every elementwise launch -- unary, binary, scalar-parameter,
// fused add + activation, bias broadcast -- runs a kernel emitted from a declared body (PtxAffineEmitter.EmitElementwise)
// whose scalar math calls the function CudaMathLibrary derived from the hand-written kernel it replaces, linked from
// that module's relocatable library. One kernel per operator, of runtime length; a kernel that is not derivable keeps
// its hand-written launch.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;
using AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private sealed class DeclaredElementwise : IDisposable
    {
        internal DeclaredElementwise(DirectPtxModule module, IntPtr function, CodegenElementwiseSpec spec)
        {
            Module = module;
            Function = function;
            Spec = spec;
        }

        internal DirectPtxModule Module { get; }
        internal IntPtr Function { get; }
        internal CodegenElementwiseSpec Spec { get; }

        public void Dispose() => Module.Dispose();
    }

    // The module each loaded kernel came from: recorded by CompileKernelModule, last definer wins.
    private readonly Dictionary<string, CudaMathModule> _kernelModules = new(StringComparer.Ordinal);

    /// <summary>The library function derived from the copy of <paramref name="kernel"/> this backend runs, or null.</summary>
    internal CudaMathFunction? ResolveElementwise(string kernel)
    {
        CudaMathModule? module;
        lock (_kernelModules) _kernelModules.TryGetValue(kernel, out module);
        return module is null ? null : CudaMathLibrary.Resolve(kernel, module);
    }

    /// <summary>Why <paramref name="kernel"/> has no derived function, for diagnostics.</summary>
    internal string ElementwiseRefusal(string kernel)
    {
        CudaMathModule? module;
        lock (_kernelModules) _kernelModules.TryGetValue(kernel, out module);
        return module is null ? kernel + ": not loaded by this backend" : CudaMathLibrary.RefusalOf(kernel, module) ?? "derivable";
    }

    /// <summary>Every loaded kernel with a derived function, by module.</summary>
    internal IReadOnlyList<CudaMathFunction> DerivableElementwise()
    {
        List<string> kernels;
        lock (_kernelModules) kernels = new List<string>(_kernelModules.Keys);
        kernels.Sort(StringComparer.Ordinal);
        var result = new List<CudaMathFunction>();
        foreach (string kernel in kernels)
            if (ResolveElementwise(kernel) is { } function) result.Add(function);
        return result;
    }

    // Compiled elementwise operators by key; a null entry records one that cannot be built (it is not retried).
    private readonly Dictionary<string, DeclaredElementwise?> _declaredElementwise = new(StringComparer.Ordinal);

    // Relocatable math libraries by module; a null entry records a module whose library does not compile.
    private readonly Dictionary<string, byte[]?> _mathLibraries = new(StringComparer.Ordinal);

    // The NVRTC log of each library that did not compile.
    private readonly Dictionary<string, string> _mathLibraryLogs = new(StringComparer.Ordinal);

    /// <summary>Why <paramref name="moduleName"/>'s math library did not compile, or null.</summary>
    internal string? MathLibraryLog(string moduleName)
    {
        lock (_directPtxLock) return _mathLibraryLogs.TryGetValue(moduleName, out string? log) ? log : null;
    }

    /// <summary>Why the last elementwise operator that could not be built was refused, for diagnostics.</summary>
    internal string? LastElementwiseDecline { get; private set; }

    /// <summary>Number of elementwise calls served by declared kernels.</summary>
    internal long DeclaredElementwiseLaunches => System.Threading.Interlocked.Read(ref _declaredElementwiseLaunches);
    private long _declaredElementwiseLaunches;

    /// <summary>
    /// <paramref name="kernel"/>'s operation on <paramref name="inputs"/> through a declared kernel. False -- having
    /// launched nothing -- when the route is off, the kernel is not derivable, or its operator cannot be built.
    /// </summary>
    internal bool TryDeclaredElementwise(string kernel, ReadOnlySpan<IGpuBuffer> inputs, IGpuBuffer output,
        ReadOnlySpan<float> scalars, int size)
    {
        if (!DeclaredKernelsEnabled || size <= 0) return false;
        var function = ResolveElementwise(kernel);
        if (function is null || function.Inputs != inputs.Length || function.Scalars.Length != scalars.Length) return false;
        var operator_ = ElementwiseOperator("call:" + kernel, () =>
        {
            var arguments = new CodegenExpr[function.Inputs + function.Scalars.Length];
            for (int i = 0; i < function.Inputs; i++) arguments[i] = CodegenExpr.Operand(i);
            for (int s = 0; s < function.Scalars.Length; s++) arguments[function.Inputs + s] = CodegenExpr.Scalar(s);
            return new CodegenElementwiseSpec(kernel, function.Inputs, CodegenExpr.Call(function.Handle, arguments));
        });
        return operator_ is not null && LaunchElementwise(operator_, inputs, output, scalars, size, columns: 0);
    }

    /// <summary>
    /// <c>activation(a + b)</c> in ONE declared kernel, where the hand-written path launched add_vectors then the
    /// activation: the same two library functions, composed, so the result is the two launches' result.
    /// </summary>
    internal bool TryDeclaredAddActivation(string activationKernel, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        if (!DeclaredKernelsEnabled || size <= 0) return false;
        var add = ResolveElementwise("add_vectors");
        var activation = ResolveElementwise(activationKernel);
        if (add is null || activation is null || activation.Inputs != 1 || activation.Scalars.Length != 0 ||
            add.Module.Name != activation.Module.Name) return false;
        var operator_ = ElementwiseOperator("add+" + activationKernel, () => new CodegenElementwiseSpec("add_" + activationKernel, 2,
            CodegenExpr.Call(activation.Handle, CodegenExpr.Call(add.Handle, CodegenExpr.Operand(0), CodegenExpr.Operand(1)))));
        return operator_ is not null && LaunchElementwise(operator_, new[] { a, b }, output, ReadOnlySpan<float>.Empty, size, columns: 0);
    }

    /// <summary>
    /// <c>C[r, c] = A[r, c] + bias[c]</c>, then relu when asked, as bias_add_out: add_vectors with the bias read across
    /// the last axis, then relu -- the same functions the hand-written kernel's expression is.
    /// </summary>
    internal bool TryDeclaredBiasAdd(IGpuBuffer a, IGpuBuffer bias, IGpuBuffer output, int rows, int columns, bool relu)
    {
        if (!DeclaredKernelsEnabled || rows <= 0 || columns <= 0 || (long)rows * columns > int.MaxValue) return false;
        var add = ResolveElementwise("add_vectors");
        var activation = relu ? ResolveElementwise("relu") : null;
        if (add is null || (relu && (activation is null || activation.Module.Name != add.Module.Name))) return false;
        var operator_ = ElementwiseOperator(relu ? "bias+relu" : "bias", () =>
        {
            var sum = CodegenExpr.Call(add.Handle, CodegenExpr.Operand(0), CodegenExpr.Operand(1));
            return new CodegenElementwiseSpec(relu ? "bias_add_relu" : "bias_add", 2,
                activation is null ? sum : CodegenExpr.Call(activation.Handle, sum), ColumnBroadcastInput: 1);
        });
        if (bias.Size < columns) throw new ArgumentException("The bias holds " + bias.Size + " elements; " + columns + " columns.", nameof(bias));
        return operator_ is not null && LaunchElementwise(operator_, new[] { a, bias }, output, ReadOnlySpan<float>.Empty,
            rows * columns, columns);
    }

    private DeclaredElementwise? ElementwiseOperator(string key, Func<CodegenElementwiseSpec> build)
    {
        lock (_directPtxLock)
        {
            if (_declaredElementwise.TryGetValue(key, out var known)) return known;
            // Building compiles and links, which graph capture forbids: decline now, build on a call outside capture.
            if (IsStreamCapturing()) return null;
            DeclaredElementwise? built = null;
            try
            {
                var spec = build();
                var emitter = new PtxAffineEmitter();
                string entry = "declared_" + SafeName(spec.Name);
                string ptx = emitter.EmitElementwise(spec, entry, _ccMajor, _ccMinor);
                var objects = new List<byte[]>();
                foreach (var called in emitter.CalledFunctions)
                {
                    var library = MathLibrary(called.Library)
                        ?? throw new NotSupportedException("The " + called.Library + " library did not compile.");
                    if (!objects.Contains(library)) objects.Add(library);
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _mainStream);
                var module = objects.Count == 0
                    ? _directPtxRuntime.LoadModule(ptx, allowExperimentalJitFallback: true)
                    : _directPtxRuntime.LoadLinkedModule(ptx, objects);
                try
                {
                    built = new DeclaredElementwise(module, module.GetFunction(entry, out _), spec);
                }
                catch
                {
                    module.Dispose();
                    throw;
                }
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException or ArgumentException)
            {
                LastElementwiseDecline = key + ": " + ex.Message;
            }
            _declaredElementwise[key] = built;
            return built;
        }
    }

    // The relocatable library of a module's derived functions, compiled once per process (and cached on disk).
    private byte[]? MathLibrary(string moduleName)
    {
        if (_mathLibraries.TryGetValue(moduleName, out var known)) return known;
        byte[]? image = null;
        // The module's library holds every derivable kernel this backend runs FROM that module.
        var functions = new List<CudaMathFunction>();
        foreach (var function in DerivableElementwise())
            if (function.Module.Name == moduleName) functions.Add(function);
        var module = functions.Count == 0 ? null : functions[0].Module;
        if (module is not null)
        {
            string source = CudaMathLibrary.LibrarySource(module, functions);
            var options = new List<string>(NvrtcOptions(_ccMajor, _ccMinor, module.FastMath)) { "--relocatable-device-code=true" };
            string? cacheDir = GetKernelCacheDirectory();
            string key = ComputeCacheKey(source, string.Join(" ", options) + " " + NvrtcNativeBindings.VersionTag, module.FastMath);
            string? cacheFile = cacheDir is null ? null : Path.Combine(cacheDir, "mathlib_" + moduleName + "_" + key + ".cubin");
            if (cacheFile is not null && File.Exists(cacheFile)) image = File.ReadAllBytes(cacheFile);
            else
            {
                image = CompileRelocatable(source, "mathlib_" + moduleName, options.ToArray(), out string log);
                if (image is null)
                {
                    LastElementwiseDecline = moduleName + " library: " + log;
                    _mathLibraryLogs[moduleName] = log;
                }
                else if (cacheFile is not null)
                {
                    try { File.WriteAllBytes(cacheFile, image); } catch (IOException) { } catch (UnauthorizedAccessException) { }
                }
            }
        }
        _mathLibraries[moduleName] = image;
        return image;
    }

    private static byte[]? CompileRelocatable(string source, string name, string[] options, out string log)
    {
        IntPtr program = IntPtr.Zero;
        log = "";
        if (NvrtcNativeBindings.nvrtcCreateProgram(ref program, source, name + ".cu", 0, IntPtr.Zero, IntPtr.Zero) != NvrtcResult.Success)
        {
            log = "nvrtcCreateProgram failed";
            return null;
        }
        try
        {
            if (NvrtcNativeBindings.nvrtcCompileProgram(program, options.Length, options) != NvrtcResult.Success)
            {
                log = GetNvrtcLog(program);
                return null;
            }
            if (NvrtcNativeBindings.nvrtcGetCUBINSize(program, out UIntPtr size) != NvrtcResult.Success || size == UIntPtr.Zero)
            {
                log = "no relocatable cubin";
                return null;
            }
            IntPtr buffer = Marshal.AllocHGlobal((int)size);
            try
            {
                if (NvrtcNativeBindings.nvrtcGetCUBIN(program, buffer) != NvrtcResult.Success)
                {
                    log = "nvrtcGetCUBIN failed";
                    return null;
                }
                var bytes = new byte[(int)size];
                Marshal.Copy(buffer, bytes, 0, bytes.Length);
                return bytes;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }
        finally
        {
            NvrtcNativeBindings.nvrtcDestroyProgram(ref program);
        }
    }

    private unsafe bool LaunchElementwise(DeclaredElementwise operator_, ReadOnlySpan<IGpuBuffer> inputs, IGpuBuffer output,
        ReadOnlySpan<float> scalars, int size, int columns)
    {
        var spec = operator_.Spec;
        if (output.Size < size) throw new ArgumentException("The output holds " + output.Size + " elements; the call needs " + size + ".", nameof(output));
        bool aligned = size % 4 == 0 && ((long)output.Handle & 15) == 0 && (spec.ColumnBroadcastInput < 0 || columns % 4 == 0);
        for (int i = 0; i < inputs.Length; i++)
        {
            if (inputs[i].Handle == IntPtr.Zero) throw new InvalidOperationException("Elementwise operand " + i + " was disposed.");
            if (i != spec.ColumnBroadcastInput && inputs[i].Size < size)
                throw new ArgumentException("Operand " + i + " holds " + inputs[i].Size + " elements; the call needs " + size + ".", nameof(inputs));
            aligned &= ((long)inputs[i].Handle & 15) == 0;
        }
        int vectorized = aligned ? 1 : 0;
        int count = inputs.Length + 3 + (spec.ColumnBroadcastInput >= 0 ? 1 : 0) + scalars.Length;
        IntPtr* pointers = stackalloc IntPtr[inputs.Length + 1];
        int* integers = stackalloc int[3];
        float* floats = stackalloc float[Math.Max(1, scalars.Length)];
        void** argv = stackalloc void*[count];
        int a = 0;
        for (int i = 0; i < inputs.Length; i++) { pointers[i] = inputs[i].Handle; argv[a++] = &pointers[i]; }
        pointers[inputs.Length] = output.Handle;
        argv[a++] = &pointers[inputs.Length];
        integers[0] = size; argv[a++] = &integers[0];
        integers[1] = vectorized; argv[a++] = &integers[1];
        if (spec.ColumnBroadcastInput >= 0) { integers[2] = columns; argv[a++] = &integers[2]; }
        for (int s = 0; s < scalars.Length; s++) { floats[s] = scalars[s]; argv[a++] = &floats[s]; }

        long work = aligned ? size / 4 : size;
        uint grid = (uint)((work + PtxAffineEmitter.ElementwiseBlockThreads - 1) / PtxAffineEmitter.ElementwiseBlockThreads);
        using var _ = PushContext();
        operator_.Module.LaunchCurrentContext(operator_.Function, grid, 1, 1, PtxAffineEmitter.ElementwiseBlockThreads, 1, 1, 0, argv);
        System.Threading.Interlocked.Increment(ref _declaredElementwiseLaunches);
        return true;
    }

    /// <summary>
    /// Launches the HAND-WRITTEN kernel <paramref name="kernel"/> with its own argument order: the incumbent a declared
    /// elementwise kernel is verified against. False when the kernel is not loaded or not derivable.
    /// </summary>
    internal unsafe bool LaunchHandWrittenElementwise(string kernel, ReadOnlySpan<IGpuBuffer> inputs, IGpuBuffer output,
        ReadOnlySpan<float> scalars, int size)
    {
        var function = ResolveElementwise(kernel);
        if (function is null || !_kernelCache.TryGetValue(kernel, out var handle) ||
            function.Inputs != inputs.Length || function.Scalars.Length != scalars.Length) return false;
        string order = function.Parameters;
        IntPtr* pointers = stackalloc IntPtr[inputs.Length + 1];
        float* floats = stackalloc float[Math.Max(1, scalars.Length)];
        int n = size;
        void** argv = stackalloc void*[order.Length];
        int input = 0, scalar = 0;
        for (int a = 0; a < order.Length; a++)
        {
            switch (order[a])
            {
                case 'i': pointers[input] = inputs[input].Handle; argv[a] = &pointers[input]; input++; break;
                case 'o': pointers[inputs.Length] = output.Handle; argv[a] = &pointers[inputs.Length]; break;
                case 's': floats[scalar] = scalars[scalar]; argv[a] = &floats[scalar]; scalar++; break;
                default: argv[a] = &n; break;
            }
        }
        using var _ = PushContext();
        LaunchKernel(handle, (uint)((size + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, argv);
        return true;
    }

    private static string SafeName(string name)
    {
        var sb = new System.Text.StringBuilder(name.Length);
        foreach (char ch in name) sb.Append(char.IsLetterOrDigit(ch) ? ch : '_');
        return sb.ToString();
    }

    private void DisposeDeclaredElementwise()
    {
        foreach (var operator_ in _declaredElementwise.Values) operator_?.Dispose();
        _declaredElementwise.Clear();
    }
}
