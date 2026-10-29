using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Interop;
using AiDotNet.Tensors.LinearAlgebra;
using Python.Runtime;
using Xunit;

namespace AiDotNet.Tensors.Tests.Interop;

/// <summary>
/// DLPack round trips with real PyTorch (#1574): <c>torch.from_dlpack</c> on a Tensors export, and a torch export
/// (<c>torch.utils.dlpack.to_dlpack</c> / <c>__dlpack__</c>) imported by Tensors, on CPU and CUDA. DLPack shares a
/// pointer, so torch runs in this process: the test embeds the CPython that <c>AIDOTNET_TORCH_PYTHON</c> names (a
/// python with torch installed) and skips when it is unset. The parity lane sets it.
/// </summary>
[Collection(TorchPython.Collection)]
public class DLPackTorchInteropTests
{
    // Producer and consumer halves of the DLPack exchange, written against the CPython C API through ctypes.
    // A consumer renames the capsule it takes ("used_dltensor[_versioned]"), as torch does, so the capsule's
    // owner never calls the deleter a second time.
    private const string Bridge = """
        import ctypes, gc, torch
        from torch.utils import dlpack as tdl
        api = ctypes.pythonapi
        api.PyCapsule_New.restype = ctypes.py_object
        api.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        api.PyCapsule_GetPointer.restype = ctypes.c_void_p
        api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        api.PyCapsule_SetName.restype = ctypes.c_int
        api.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
        live = {}

        class Producer:
            # The __dlpack__ protocol object torch.from_dlpack negotiates with; it hands over a versioned capsule.
            def __init__(self, ptr, device):
                self.ptr, self.device = ptr, device
            def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):
                if max_version is None or max_version[0] < 1:
                    raise BufferError("the producer only offers DLPack 1.x")
                return api.PyCapsule_New(self.ptr, b"dltensor_versioned", None)
            def __dlpack_device__(self):
                return self.device

        def from_tensors_legacy(key, ptr):
            live[key] = tdl.from_dlpack(api.PyCapsule_New(ptr, b"dltensor", None))

        def from_tensors_versioned(key, ptr, device_type, device_id):
            live[key] = torch.from_dlpack(Producer(ptr, (device_type, device_id)))

        def export(key, versioned):
            t = live[key]
            if versioned:
                cap, name = t.__dlpack__(max_version=(1, 0)), b"dltensor_versioned"
            else:
                cap, name = tdl.to_dlpack(t), b"dltensor"
            ptr = api.PyCapsule_GetPointer(cap, name)
            api.PyCapsule_SetName(cap, b"used_" + name)
            return ptr

        def make(key, n, cuda):
            live[key] = torch.arange(n, dtype=torch.float32, device="cuda" if cuda else "cpu") * 0.5 - 3.0

        def values(key):
            return live[key].detach().cpu().reshape(-1).tolist()

        def shape(key):
            return list(live[key].shape)

        def is_cuda(key):
            return live[key].is_cuda

        def scale(key, factor):
            live[key].mul_(factor)
            if live[key].is_cuda:
                torch.cuda.synchronize()

        def drop(key):
            del live[key]
            gc.collect()
        """;

    private static readonly Lazy<PyModule> Module = new(() =>
    {
        using (Py.GIL()) return PyModule.FromString("aidotnet_dlpack_bridge", Bridge);
    });

    private static float[] Iota(int n) => Enumerable.Range(0, n).Select(i => i * 0.5f - 3f).ToArray();

    private static T Call<T>(string fn, params object[] args)
    {
        using (Py.GIL())
        {
            var pyArgs = args.Select(a => a.ToPython()).ToArray();
            return Module.Value.InvokeMethod(fn, pyArgs).As<T>();
        }
    }

    private static void Call(string fn, params object[] args)
    {
        using (Py.GIL()) Module.Value.InvokeMethod(fn, args.Select(a => a.ToPython()).ToArray()).Dispose();
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Cpu_TensorsToTorch_IsZeroCopy_AndTorchReleasesTheExport(bool versioned)
    {
        Skip.If(TorchPython.Unavailable is not null, TorchPython.Unavailable);
        int liveBefore = DLPack.LiveExportCount;
        var source = new Tensor<float>(new[] { 3, 4 }, new Vector<float>(Iota(12)));
        string key = "cpu_in_" + versioned;
        if (versioned)
            Call("from_tensors_versioned", key, source.ToDLPack().ToInt64(), 1, 0);
        else
            Call("from_tensors_legacy", key, source.ToDLPackLegacy().ToInt64());

        Assert.Equal(new[] { 3, 4 }, Call<int[]>("shape", key));
        Assert.Equal(Iota(12), Call<float[]>("values", key));
        Call("scale", key, 2.0);   // a write through torch is a write to the Tensors storage
        Assert.Equal(Iota(12).Select(v => v * 2f).ToArray(), source.ToArray());

        Assert.Equal(liveBefore + 1, DLPack.LiveExportCount);
        Call("drop", key);         // torch runs the deleter when its tensor dies
        Assert.Equal(liveBefore, DLPack.LiveExportCount);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Cpu_TorchToTensors_IsZeroCopy_AndReleasesTorchsTensorOnDispose(bool versioned)
    {
        Skip.If(TorchPython.Unavailable is not null, TorchPython.Unavailable);
        string key = "cpu_out_" + versioned;
        Call("make", key, 10, false);
        var ptr = new IntPtr(Call<long>("export", key, versioned));
        var imported = versioned ? Tensor<float>.FromDLPack(ptr) : Tensor<float>.FromDLPackLegacy(ptr);
        Assert.Equal(Iota(10), imported.ToArray());

        Call("scale", key, 3.0);   // torch's in-place write is visible through the import: no copy was made
        Assert.Equal(Iota(10).Select(v => v * 3f).ToArray(), imported.ToArray());
        imported.Dispose();        // runs torch's deleter; torch's own tensor stays valid
        Assert.Equal(Iota(10).Select(v => v * 3f).ToArray(), Call<float[]>("values", key));
        Call("drop", key);
    }

    [SkippableFact]
    public void Cuda_RoundTripsThroughTorch_OnTheDevice()
    {
        Skip.If(TorchPython.Unavailable is not null, TorchPython.Unavailable);
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        Skip.IfNot(TorchPython.CudaAvailable, "torch was built without CUDA or sees no device.");
        var previous = AiDotNetEngine.Current;
        var gpu = new DirectGpuTensorEngine();
        try
        {
            Skip.IfNot(gpu.IsGpuAvailable && gpu.PlacementBackend is CudaBackend, "The active GPU backend is not CUDA.");
            AiDotNetEngine.Current = gpu;
            int device = ((CudaBackend)gpu.PlacementBackend).DeviceOrdinal;
            int liveBefore = DLPack.LiveExportCount;

            // Tensors -> torch: torch's legacy default stream (DLPack value 1) waits on the export.
            var source = new Tensor<float>(new[] { 4, 8 }, new Vector<float>(Iota(32))).Gpu();
            Call("from_tensors_versioned", "cuda_in", source.ToDLPack(DLPackStream.LegacyDefault).ToInt64(), 2, device);
            Assert.True(Call<bool>("is_cuda", "cuda_in"));
            Assert.Equal(Iota(32), Call<float[]>("values", "cuda_in"));
            Call("drop", "cuda_in");
            Assert.Equal(liveBefore, DLPack.LiveExportCount);
            source.Dispose();

            // torch -> Tensors: the import wraps torch's device pointer as a GPU buffer.
            Call("make", "cuda_out", 16, true);
            var imported = Tensor<float>.FromDLPack(new IntPtr(Call<long>("export", "cuda_out", true)));
            Assert.True(imported.IsGpuResident);
            Assert.Equal(Iota(16), imported.ToArray());
            imported.Dispose();
            Call("drop", "cuda_out");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }
}

/// <summary>Starts the CPython named by <c>AIDOTNET_TORCH_PYTHON</c> once per test process.</summary>
internal static class TorchPython
{
    internal const string Collection = "Embedded CPython + torch";

    private static readonly Lazy<string?> Failure = new(Start);

    /// <summary>Why torch is unavailable, or <c>null</c> once the interpreter is up and torch imports.</summary>
    internal static string? Unavailable => Failure.Value;

    internal static bool CudaAvailable
    {
        get
        {
            if (Unavailable is not null) return false;
            using (Py.GIL()) return Py.Import("torch").GetAttr("cuda").InvokeMethod("is_available").As<bool>();
        }
    }

    private static string? Start()
    {
        var exe = Environment.GetEnvironmentVariable("AIDOTNET_TORCH_PYTHON");
        if (string.IsNullOrWhiteSpace(exe)) return "AIDOTNET_TORCH_PYTHON is not set (the python executable of an environment with torch).";
        if (!File.Exists(exe)) return $"AIDOTNET_TORCH_PYTHON points at {exe}, which does not exist.";

        // The interpreter reports its own shared library, home and module path; a venv's site-packages comes with it.
        const string probe = "import json, os, sys, sysconfig\n"
            + "lib = os.path.join(sys.base_prefix, 'python%d%d.dll' % sys.version_info[:2]) if os.name == 'nt' "
            + "else os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY'))\n"
            + "print(json.dumps({'lib': lib, 'home': sys.base_prefix, 'path': sys.path}))";
        var psi = new ProcessStartInfo(exe) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        psi.ArgumentList.Add("-c");
        psi.ArgumentList.Add(probe);
        using var p = Process.Start(psi)!;
        string stdout = p.StandardOutput.ReadToEnd();
        string stderr = p.StandardError.ReadToEnd();
        p.WaitForExit();
        if (p.ExitCode != 0) throw new InvalidOperationException($"Probing {exe} failed: {stderr}");
        using var info = JsonDocument.Parse(stdout);
        var lib = info.RootElement.GetProperty("lib").GetString()!;
        if (!File.Exists(lib)) throw new InvalidOperationException($"{exe} reports its shared library at {lib}, which does not exist.");

        Runtime.PythonDLL = lib;
        PythonEngine.PythonHome = info.RootElement.GetProperty("home").GetString()!;
        PythonEngine.Initialize();
        PythonEngine.BeginAllowThreads();
        using (Py.GIL())
        {
            using var sys = Py.Import("sys");
            using var path = new PyList(info.RootElement.GetProperty("path").EnumerateArray()
                .Select(e => (PyObject)new PyString(e.GetString()!)).ToArray());
            sys.SetAttr("path", path);
            // A missing torch is a broken AIDOTNET_TORCH_PYTHON, not a reason to skip: this throws.
            Py.Import("torch").Dispose();
        }
        return null;
    }
}

[CollectionDefinition(TorchPython.Collection, DisableParallelization = true)]
public sealed class TorchPythonCollection
{
}
