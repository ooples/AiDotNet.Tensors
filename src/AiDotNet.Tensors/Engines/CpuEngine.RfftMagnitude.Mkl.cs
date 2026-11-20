using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;
#if NET5_0_OR_GREATER
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
#endif

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    internal static bool MklRfftMagnitudeEnabled = Environment.GetEnvironmentVariable("AIDOTNET_MKL_RFFT") == "1";
    [ThreadStatic] internal static int MklRfftForwardRuns;
    [ThreadStatic] internal static int MklRfftBackwardRuns;

    private static unsafe bool TryMklRfftMagnitudeRows<T>(ReadOnlyMemory<T> source, ReadOnlyMemory<T> upstream,
        Memory<T> destination, int rows, int length, int bins, double epsilon, bool backward, bool accumulate)
    {
#if NET5_0_OR_GREATER
        if (!MklRfftMagnitudeEnabled || typeof(T) != typeof(float)) return false;
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || RuntimeInformation.ProcessArchitecture != Architecture.X64)
            throw new PlatformNotSupportedException("The experimental MKL RFFT binding currently requires Windows x64.");
        var input = AsFloatReadOnlyMemory(source);
        var gradient = backward ? AsFloatReadOnlyMemory(upstream) : default;
        var output = AsFloatMemory(destination);
        // Bound each complex work buffer to 4 MiB, except when one transform
        // itself needs more. A large batch never retains a batch-sized spectrum.
        int chunkRows = Math.Max(1, Math.Min(rows, 1048576 / checked(2 * length)));
        for (int first = 0; first < rows; first += chunkRows)
        {
            int count = Math.Min(chunkRows, rows - first);
            RunMklRfftMagnitudeFloatRows(input.Slice(first * length, count * length),
                backward ? gradient.Slice(first * bins, count * bins) : default,
                output.Slice(first * (backward ? length : bins), count * (backward ? length : bins)),
                count, length, bins, epsilon, backward, accumulate);
        }
        return true;
#else
        return false;
#endif
    }

#if NET5_0_OR_GREATER
    private static unsafe bool RunMklRfftMagnitudeFloatRows(ReadOnlyMemory<float> source, ReadOnlyMemory<float> upstream,
        Memory<float> destination, int rows, int length, int bins, double epsilon, bool backward, bool accumulate)
    {
        var api = MklRfftApi.Instance.Value;
        int threads = Math.Max(1, Math.Min(Environment.ProcessorCount, CpuParallelSettings.MaxDegreeOfParallelism));
        var forward = MklRfftPlan.Get(api, rows, length, complex: false, threads);
        int spectrumLength = checked(2 * rows * bins);
        var spectrum = CpuStorageAllocationDiagnostics.RentWorkspace<float>(spectrumLength, "workspace.mkl_rfft_spectrum");
        try
        {
            var input = AsFloatReadOnlyMemory(source).Span;
            var output = AsFloatMemory(destination).Span;
            fixed (float* inputPointer = input)
            fixed (float* spectrumPointer = spectrum)
                api.Check(api.Forward(forward.Handle, (IntPtr)inputPointer, (IntPtr)spectrumPointer));
            float eps = (float)epsilon;
            if (!backward)
            {
                for (int i = 0; i < rows * bins; i++)
                    output[i] = MklFloatMagnitude(spectrum[2 * i], spectrum[2 * i + 1], eps);
                MklRfftForwardRuns++;
                if (MklRfftForwardRuns == 1 && Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
                    AiDotNetEngine.LogDiagnostic("[MKL-RFFT] forward rows=" + rows + " length=" + length + " threads=" + threads);
                return true;
            }

            int complexLength = checked(2 * rows * length);
            var seed = CpuStorageAllocationDiagnostics.RentWorkspace<float>(complexLength, "workspace.mkl_rfft_adjoint_seed");
            try
            {
                var inverse = CpuStorageAllocationDiagnostics.RentWorkspace<float>(complexLength, "workspace.mkl_rfft_adjoint_output");
                try
                {
                    seed.AsSpan(0, complexLength).Clear();
                    var gradient = AsFloatReadOnlyMemory(upstream).Span;
                    for (int row = 0; row < rows; row++)
                        for (int bin = 0; bin < bins; bin++)
                        {
                            int observed = row * bins + bin, full = row * length + bin;
                            float real = spectrum[2 * observed], imaginary = spectrum[2 * observed + 1];
                            float scale = gradient[observed] / MklFloatMagnitude(real, imaginary, eps);
                            seed[2 * full] = real * scale;
                            seed[2 * full + 1] = imaginary * scale;
                        }
                    var adjoint = MklRfftPlan.Get(api, rows, length, complex: true, threads);
                    fixed (float* seedPointer = seed)
                    fixed (float* inversePointer = inverse)
                        api.Check(api.Backward(adjoint.Handle, (IntPtr)seedPointer, (IntPtr)inversePointer));
                    for (int i = 0; i < rows * length; i++)
                        output[i] = (accumulate ? output[i] : 0f) + inverse[2 * i];
                    MklRfftBackwardRuns++;
                    if (MklRfftBackwardRuns == 1 && Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
                        AiDotNetEngine.LogDiagnostic("[MKL-RFFT] backward rows=" + rows + " length=" + length + " threads=" + threads);
                    return true;
                }
                finally { ArrayPool<float>.Shared.Return(inverse); }
            }
            finally { ArrayPool<float>.Shared.Return(seed); }
        }
        finally { ArrayPool<float>.Shared.Return(spectrum); }
    }
#endif

#if NET5_0_OR_GREATER
    private static float MklFloatMagnitude(float real, float imaginary, float epsilon)
    {
        float squared = real * real;
        squared += imaginary * imaginary;
        squared += epsilon;
        return MathF.Sqrt(squared);
    }

    // DFTI's Windows LP64 ABI uses four-byte MKL_LONG. Bind its fixed one-dimensional
    // argument forms explicitly; descriptor ownership and cache eviction stay private.
    private sealed unsafe class MklRfftApi
    {
        internal static readonly Lazy<MklRfftApi> Instance = new(() => new MklRfftApi());
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate int CreateDescriptor(out IntPtr descriptor, int precision, int domain, int dimensions, int length);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate int SetInteger(IntPtr descriptor, int parameter, int value);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate int SetStrides(IntPtr descriptor, int parameter, int* value);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate int CommitDescriptor(IntPtr descriptor);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate int Compute(IntPtr descriptor, IntPtr input, IntPtr output);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate int FreeDescriptor(ref IntPtr descriptor);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate IntPtr ErrorMessage(int status);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void GetVersion(IntPtr buffer, int length);
        internal readonly CreateDescriptor Create;
        internal readonly SetInteger Set;
        internal readonly SetStrides Strides;
        internal readonly CommitDescriptor Commit;
        internal readonly Compute Forward, Backward;
        internal readonly FreeDescriptor Free;
        private readonly ErrorMessage _error;
        private readonly IntPtr _library;
        private MklRfftApi()
        {
            string path = Environment.GetEnvironmentVariable("AIDOTNET_MKL_RFFT_LIBRARY") ?? "";
            if (!Path.IsPathRooted(path) || !File.Exists(path))
                throw new InvalidOperationException("MKL RFFT opt-in requires an existing absolute AIDOTNET_MKL_RFFT_LIBRARY path.");
            _library = System.Runtime.InteropServices.NativeLibrary.Load(path);
            D Bind<D>(string name) where D : Delegate => Marshal.GetDelegateForFunctionPointer<D>(
                System.Runtime.InteropServices.NativeLibrary.GetExport(_library, name));
            Create = Bind<CreateDescriptor>("DftiCreateDescriptor");
            Set = Bind<SetInteger>("DftiSetValue"); Strides = Bind<SetStrides>("DftiSetValue");
            Commit = Bind<CommitDescriptor>("DftiCommitDescriptor");
            Forward = Bind<Compute>("DftiComputeForward"); Backward = Bind<Compute>("DftiComputeBackward");
            Free = Bind<FreeDescriptor>("DftiFreeDescriptor"); _error = Bind<ErrorMessage>("DftiErrorMessage");
            if (Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_DIAG") == "1")
            {
                var version = Bind<GetVersion>("MKL_Get_Version_String");
                byte* buffer = stackalloc byte[256];
                version((IntPtr)buffer, 256);
                using var stream = File.OpenRead(path);
                using var sha = SHA256.Create();
                AiDotNetEngine.LogDiagnostic("[MKL-RFFT] library=" + path + " sha256=" + Convert.ToHexString(sha.ComputeHash(stream))
                    + " version=" + Marshal.PtrToStringAnsi((IntPtr)buffer));
            }
        }
        internal void Check(int status)
        {
            if (status != 0) throw new InvalidOperationException("MKL DFTI: " + Marshal.PtrToStringAnsi(_error(status)));
        }
    }

    private sealed unsafe class MklRfftPlan : IDisposable
    {
        [ThreadStatic] private static Dictionary<(int Rows, int Length, bool Complex, int Threads), MklRfftPlan>? s_plans;
        private readonly MklRfftApi _api;
        internal IntPtr Handle;
        private MklRfftPlan(MklRfftApi api, int rows, int length, bool complex, int threads)
        {
            _api = api;
            api.Check(api.Create(out Handle, 35, complex ? 32 : 33, 1, length)); // SINGLE, COMPLEX/REAL
            try
            {
                api.Check(api.Set(Handle, 11, 44)); // out of place
                api.Check(api.Set(Handle, 7, rows));
                api.Check(api.Set(Handle, 14, length));
                api.Check(api.Set(Handle, 15, complex ? length : length / 2 + 1));
                api.Check(api.Set(Handle, 27, threads));
                int* strides = stackalloc int[2] { 0, 1 };
                api.Check(api.Strides(Handle, 12, strides)); api.Check(api.Strides(Handle, 13, strides));
                if (!complex) api.Check(api.Set(Handle, 10, 39)); // conjugate-even interleaved complex
                api.Check(api.Commit(Handle));
            }
            catch { Dispose(); throw; }
        }
        internal static MklRfftPlan Get(MklRfftApi api, int rows, int length, bool complex, int threads)
        {
            var cache = s_plans ??= new Dictionary<(int, int, bool, int), MklRfftPlan>();
            var key = (rows, length, complex, threads);
            if (cache.TryGetValue(key, out var plan)) return plan;
            if (cache.Count >= 32)
                foreach (var item in cache) { cache.Remove(item.Key); item.Value.Dispose(); break; }
            plan = new MklRfftPlan(api, rows, length, complex, threads);
            cache.Add(key, plan);
            return plan;
        }
        public void Dispose()
        {
            if (Handle != IntPtr.Zero) _api.Free(ref Handle);
            GC.SuppressFinalize(this);
        }
        ~MklRfftPlan() { if (Handle != IntPtr.Zero) _api.Free(ref Handle); }
    }
#endif
}
