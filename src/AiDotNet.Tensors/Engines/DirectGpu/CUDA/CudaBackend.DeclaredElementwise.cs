// Copyright (c) CrowdTrainer LLC. All rights reserved.
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

    // The cache file each in-memory library image was READ from, so a link failure can discard a corrupt one. Only
    // images that came off disk are listed; a freshly compiled image cannot be the cache's fault.
    private readonly Dictionary<string, string> _mathLibraryCacheFiles = new(StringComparer.Ordinal);

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

    // The compiled operator for each hand-written kernel name ("call:" operators), looked up without building a key.
    private readonly Dictionary<string, DeclaredElementwise?> _elementwiseByKernel = new(StringComparer.Ordinal);

    /// <summary>
    /// <paramref name="kernel"/>'s operation through a declared kernel: up to three inputs (<paramref name="i0"/>..),
    /// up to three scalars (the first <paramref name="scalarCount"/> of <paramref name="s0"/>..), the output and the
    /// length. False -- having launched nothing -- when the route is off, the kernel is not derivable or takes other
    /// operands, or its operator cannot be built. A call allocates nothing once its operator exists.
    /// </summary>
    internal bool TryDeclaredElementwise(string kernel, IGpuBuffer output, int size, IGpuBuffer i0, IGpuBuffer? i1 = null,
        IGpuBuffer? i2 = null, int scalarCount = 0, float s0 = 0, float s1 = 0, float s2 = 0, IGpuBuffer? i3 = null)
    {
        if (!DeclaredKernelsEnabled || size <= 0) return false;
        int inputs = i3 is not null ? 4 : i2 is not null ? 3 : i1 is not null ? 2 : 1;
        DeclaredElementwise? operator_;
        bool known;
        lock (_directPtxLock) known = _elementwiseByKernel.TryGetValue(kernel, out operator_);
        if (!known)
        {
            var function = ResolveElementwise(kernel);
            operator_ = function is null ? null : ElementwiseOperator("call:" + kernel, () =>
            {
                var arguments = new CodegenExpr[function.Inputs + function.Scalars.Length];
                for (int i = 0; i < function.Inputs; i++) arguments[i] = CodegenExpr.Operand(i);
                for (int s = 0; s < function.Scalars.Length; s++) arguments[function.Inputs + s] = CodegenExpr.Scalar(s);
                return new CodegenElementwiseSpec(kernel, function.Inputs, CodegenExpr.Call(function.Handle, arguments));
            });
            // Under capture nothing was built: leave the kernel unrecorded, so the first call outside builds it.
            if (operator_ is not null || !IsStreamCapturing())
                lock (_directPtxLock) _elementwiseByKernel[kernel] = operator_;
        }
        if (operator_ is null || operator_.Spec.Inputs != inputs || operator_.Spec.Scalars != scalarCount) return false;
        return LaunchElementwise(operator_, output, size, 0, i0, i1, i2, scalarCount, s0, s1, s2, i3);
    }

    /// <summary>The span form of <see cref="TryDeclaredElementwise(string, IGpuBuffer, int, IGpuBuffer, IGpuBuffer?, IGpuBuffer?, int, float, float, float)"/>, for tools and tests.</summary>
    internal bool TryDeclaredElementwise(string kernel, ReadOnlySpan<IGpuBuffer> inputs, IGpuBuffer output,
        ReadOnlySpan<float> scalars, int size)
    {
        if (inputs.Length < 1 || inputs.Length > 4 || scalars.Length > 3) return false;
        return TryDeclaredElementwise(kernel, output, size, inputs[0], inputs.Length > 1 ? inputs[1] : null,
            inputs.Length > 2 ? inputs[2] : null, scalars.Length,
            scalars.Length > 0 ? scalars[0] : 0, scalars.Length > 1 ? scalars[1] : 0, scalars.Length > 2 ? scalars[2] : 0,
            inputs.Length > 3 ? inputs[3] : null);
    }

    // The composites' operators by their constant keys.
    private DeclaredElementwise? _addRelu, _addSigmoid, _addGelu, _biasAdd, _biasAddRelu;
    private bool _addReluTried, _addSigmoidTried, _addGeluTried, _biasAddTried, _biasAddReluTried;

    /// <summary>
    /// <c>activation(a + b)</c> in ONE declared kernel, where the hand-written path launched add_vectors then the
    /// activation: the same two library functions, composed, so the result is the two launches' result.
    /// </summary>
    internal bool TryDeclaredAddActivation(string activationKernel, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int size)
    {
        if (!DeclaredKernelsEnabled || size <= 0) return false;
        ref DeclaredElementwise? slot = ref _addRelu;
        ref bool tried = ref _addReluTried;
        if (activationKernel == "sigmoid") { slot = ref _addSigmoid; tried = ref _addSigmoidTried; }
        else if (activationKernel == "gelu") { slot = ref _addGelu; tried = ref _addGeluTried; }
        else if (activationKernel != "relu") return false;
        if (!tried)
        {
            var add = ResolveElementwise("add_vectors");
            var activation = ResolveElementwise(activationKernel);
            if (add is null || activation is null || activation.Inputs != 1 || activation.Scalars.Length != 0 ||
                add.Module.Name != activation.Module.Name) { tried = true; return false; }
            slot = ElementwiseOperator("add+" + activationKernel, () => new CodegenElementwiseSpec("add_" + activationKernel, 2,
                CodegenExpr.Call(activation.Handle, CodegenExpr.Call(add.Handle, CodegenExpr.Operand(0), CodegenExpr.Operand(1)))));
            tried = slot is not null || !IsStreamCapturing();
        }
        return slot is not null && LaunchElementwise(slot, output, size, 0, a, b, null, 0, 0, 0, 0);
    }

    /// <summary>
    /// <c>C[r, c] = A[r, c] + bias[c]</c>, then relu when asked, as bias_add_out: add_vectors with the bias read across
    /// the last axis, then relu -- the same functions the hand-written kernel's expression is.
    /// </summary>
    internal bool TryDeclaredBiasAdd(IGpuBuffer a, IGpuBuffer bias, IGpuBuffer output, int rows, int columns, bool relu)
    {
        if (!DeclaredKernelsEnabled || rows <= 0 || columns <= 0 || (long)rows * columns > int.MaxValue) return false;
        ref DeclaredElementwise? slot = ref relu ? ref _biasAddRelu : ref _biasAdd;
        ref bool tried = ref relu ? ref _biasAddReluTried : ref _biasAddTried;
        if (!tried)
        {
            var add = ResolveElementwise("add_vectors");
            var activation = relu ? ResolveElementwise("relu") : null;
            if (add is null || (relu && (activation is null || activation.Module.Name != add.Module.Name))) { tried = true; return false; }
            slot = ElementwiseOperator(relu ? "bias+relu" : "bias", () =>
            {
                var sum = CodegenExpr.Call(add.Handle, CodegenExpr.Operand(0), CodegenExpr.Operand(1));
                return new CodegenElementwiseSpec(relu ? "bias_add_relu" : "bias_add", 2,
                    activation is null ? sum : CodegenExpr.Call(activation.Handle, sum), ColumnBroadcastInput: 1);
            });
            tried = slot is not null || !IsStreamCapturing();
        }
        if (slot is null) return false;
        if (bias.Size < columns) throw new ArgumentException("The bias holds " + bias.Size + " elements; " + columns + " columns.", nameof(bias));
        return LaunchElementwise(slot, output, rows * columns, columns, a, bias, null, 0, 0, 0, 0);
    }

    private DeclaredElementwise? ElementwiseOperator(string key, Func<CodegenElementwiseSpec> build)
    {
        lock (_directPtxLock)
        {
            if (_declaredElementwise.TryGetValue(key, out var known)) return known;
            // Building compiles and links, which graph capture forbids: decline now, build on a call outside capture.
            if (IsStreamCapturing()) return null;
            // Twice at most: a link failure against a library image that came off disk discards the cached cubin and
            // recompiles, so a truncated cache file costs one rebuild instead of the route for every later process.
            for (int attempt = 0; ; attempt++)
            {
                DeclaredElementwise? built = null;
                // Only a LINK failure against an image that came off disk is evidence of a torn cache file. A later
                // GetFunction rejection (the local-memory budget) is the kernel's own spill: discarding every cached
                // cubin for it would delete good caches and retry into the same rejection.
                bool linkFailedOnDiskImage = false;
                try
                {
                    var spec = build();
                    var emitter = new PtxAffineEmitter();
                    string entry = "declared_" + SafeName(spec.Name);
                    string ptx = emitter.EmitElementwise(spec, entry, _ccMajor, _ccMinor);
                    var objects = new List<byte[]>();
                    bool anyFromDisk = false;
                    foreach (var called in emitter.CalledFunctions)
                    {
                        var library = MathLibrary(called.Library)
                            ?? throw new NotSupportedException("The " + called.Library + " library did not compile.");
                        anyFromDisk |= _mathLibraryCacheFiles.ContainsKey(called.Library);
                        if (!objects.Contains(library)) objects.Add(library);
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _mainStream);
                    DirectPtxModule module;
                    if (objects.Count == 0)
                        module = _directPtxRuntime.LoadModule(ptx, allowExperimentalJitFallback: true);
                    else
                    {
                        try
                        {
                            module = _directPtxRuntime.LoadLinkedModule(ptx, objects);
                        }
                        catch (Exception ex) when (anyFromDisk &&
                            ex is NotSupportedException or InvalidOperationException or ArgumentException)
                        {
                            linkFailedOnDiskImage = true;
                            throw;
                        }
                    }
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
                    if (attempt == 0 && linkFailedOnDiskImage && DiscardCachedMathLibraries()) continue;
                }
                _declaredElementwise[key] = built;
                return built;
            }
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
            string source;
            try
            {
                source = CudaMathLibrary.LibrarySource(module, functions);
            }
            catch (System.Text.RegularExpressions.RegexMatchTimeoutException ex)
            {
                // The library declines like a failed compile: no image, and every caller keeps its hand-written path.
                System.Diagnostics.Trace.TraceWarning("[CudaBackend] " + moduleName + " library: source analysis timed out: " + ex);
                const string timedOut = "the source analysis exceeded its pattern timeout";
                LastElementwiseDecline = moduleName + " library: " + timedOut;
                _mathLibraryLogs[moduleName] = timedOut;
                _mathLibraries[moduleName] = null;
                return null;
            }
            var options = new List<string>(NvrtcOptions(_ccMajor, _ccMinor, module.FastMath)) { "--relocatable-device-code=true" };
            string? cacheDir = GetKernelCacheDirectory();
            string key = ComputeCacheKey(source, string.Join(" ", options) + " " + NvrtcNativeBindings.VersionTag, module.FastMath);
            string? cacheFile = cacheDir is null ? null : Path.Combine(cacheDir, "mathlib_" + moduleName + "_" + key + ".cubin");
            if (cacheFile is not null && TryReadCachedImage(cacheFile, out byte[]? cached))
            {
                image = cached;
                _mathLibraryCacheFiles[moduleName] = cacheFile;
            }
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
                    WriteCachedImage(cacheFile, image);
                }
            }
        }
        _mathLibraries[moduleName] = image;
        return image;
    }

    // A cached image that cannot be read is a cache MISS, not a failure: the library is recompiled and rewritten.
    private static bool TryReadCachedImage(string cacheFile, out byte[]? image)
    {
        image = null;
        try
        {
            if (!File.Exists(cacheFile)) return false;
            image = File.ReadAllBytes(cacheFile);
        }
        catch (IOException) { return false; }
        catch (UnauthorizedAccessException) { return false; }
        // A zero-length file is a write that was interrupted before any bytes landed.
        return image is { Length: > 0 };
    }

    // Written through a temporary and moved into place: File.WriteAllBytes writes IN PLACE, so two processes
    // compiling the same library at once - or one killed mid-write - could leave a truncated cubin that every later
    // process would then trust, losing the declared elementwise route and the routes that link the library.
    private static void WriteCachedImage(string cacheFile, byte[] image)
    {
        string temp = cacheFile + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            File.WriteAllBytes(temp, image);
            // Another process may have finished first; its image is the same content-addressed library.
            if (!File.Exists(cacheFile)) File.Move(temp, cacheFile);
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
        finally
        {
            try { if (File.Exists(temp)) File.Delete(temp); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    // Discards every library image that came off disk, file and all, so the next build recompiles instead of linking
    // the same bad image again. Returns whether anything was discarded. The caller holds _directPtxLock.
    private bool DiscardCachedMathLibraries()
    {
        if (_mathLibraryCacheFiles.Count == 0) return false;
        foreach (var pair in _mathLibraryCacheFiles)
        {
            try { File.Delete(pair.Value); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            _mathLibraries.Remove(pair.Key);
        }
        _mathLibraryCacheFiles.Clear();
        return true;
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

    private unsafe bool LaunchElementwise(DeclaredElementwise operator_, IGpuBuffer output, int size, int columns,
        IGpuBuffer i0, IGpuBuffer? i1, IGpuBuffer? i2, int scalarCount, float s0, float s1, float s2, IGpuBuffer? i3 = null)
    {
        var spec = operator_.Spec;
        if (output.Size < size) throw new ArgumentException("The output holds " + output.Size + " elements; the call needs " + size + ".", nameof(output));
        int inputCount = spec.Inputs;
        IntPtr* pointers = stackalloc IntPtr[5];
        bool aligned = size % 4 == 0 && ((long)output.Handle & 15) == 0 && (spec.ColumnBroadcastInput < 0 || columns % 4 == 0);
        for (int i = 0; i < inputCount; i++)
        {
            var input = (i == 0 ? i0 : i == 1 ? i1 : i == 2 ? i2 : i3)
                ?? throw new ArgumentException("Elementwise operand " + i + " is missing.");
            if (input.Handle == IntPtr.Zero) throw new InvalidOperationException("Elementwise operand " + i + " was disposed.");
            if (i != spec.ColumnBroadcastInput && input.Size < size)
                throw new ArgumentException("Operand " + i + " holds " + input.Size + " elements; the call needs " + size + ".");
            aligned &= ((long)input.Handle & 15) == 0;
            pointers[i] = input.Handle;
        }
        pointers[inputCount] = output.Handle;
        int* integers = stackalloc int[3];
        float* floats = stackalloc float[3];
        floats[0] = s0; floats[1] = s1; floats[2] = s2;
        void** argv = stackalloc void*[inputCount + 4 + scalarCount];
        int a = 0;
        for (int i = 0; i <= inputCount; i++) argv[a++] = &pointers[i];
        integers[0] = size; argv[a++] = &integers[0];
        integers[1] = aligned ? 1 : 0; argv[a++] = &integers[1];
        if (spec.ColumnBroadcastInput >= 0) { integers[2] = columns; argv[a++] = &integers[2]; }
        for (int s = 0; s < scalarCount; s++) argv[a++] = &floats[s];

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
        // The production launchers take the _vec4 variant when the length and pointers allow: so does the incumbent.
        bool aligned = size % 4 == 0 && ((long)output.Handle & 15) == 0;
        foreach (var input_ in inputs) aligned &= ((long)input_.Handle & 15) == 0;
        if (aligned && scalars.Length == 0 && _kernelCache.TryGetValue(kernel + "_vec4", out var vec4))
        {
            n = size / 4;
            handle = vec4;
        }
        using var _ = PushContext();
        LaunchKernel(handle, (uint)((n + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, argv);
        return true;
    }

    // The relocatable libraries a kernel's calls come from, each once.
    private List<byte[]> LibrariesFor(IReadOnlyList<CodegenLibraryFunction> calls)
    {
        var objects = new List<byte[]>();
        foreach (var call in calls)
        {
            var library = MathLibrary(call.Library)
                ?? throw new NotSupportedException("The " + call.Library + " library did not compile: " + MathLibraryLog(call.Library));
            if (!objects.Contains(library)) objects.Add(library);
        }
        return objects;
    }

    /// <summary>Loads diagnostic PTX (verification tooling) into this backend's context.</summary>
    internal (DirectPtxModule Module, IntPtr Function) LoadDiagnosticKernel(string ptx, string entry)
    {
        lock (_directPtxLock)
        {
            _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _mainStream);
            var module = _directPtxRuntime.LoadModule(ptx, allowExperimentalJitFallback: true);
            return (module, module.GetFunction(entry, out _));
        }
    }

    /// <summary>Launches a diagnostic kernel on this backend's stream.</summary>
    internal unsafe void LaunchDiagnostic(DirectPtxModule module, IntPtr function, uint grid, uint block, void** args)
    {
        using var _ = PushContext();
        module.LaunchCurrentContext(function, grid, 1, 1, block, 1, 1, 0, args);
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
