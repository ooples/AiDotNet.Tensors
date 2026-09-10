// Copyright (c) AiDotNet. All rights reserved.
// Pure P/Invoke OpenCL program - no managed GPU runtime dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>How an OpenCL program reached driver-executable form.</summary>
    public enum OpenClProgramBuildOrigin
    {
        /// <summary>The device driver compiled OpenCL C source during this process.</summary>
        SourceCompilation = 0,
        /// <summary>The driver loaded its previously compiled device-native binary.</summary>
        NativeBinaryCache = 1
    }

    /// <summary>The OpenCL driver's reported form of a built program binary.</summary>
    public enum OpenClProgramBinaryType
    {
        /// <summary>The driver did not expose a recognized binary type.</summary>
        Unavailable = 0,
        /// <summary>A relocatable compiled object that is not directly executable.</summary>
        CompiledObject = 1,
        /// <summary>A linked library that is not directly executable.</summary>
        Library = 2,
        /// <summary>A device executable accepted by the selected OpenCL driver.</summary>
        Executable = 3
    }

    /// <summary>
    /// OpenCL program wrapper using pure P/Invoke. No managed GPU runtime dependency.
    /// Supports disk-based binary caching to avoid recompilation on startup.
    /// </summary>
    internal sealed class DirectOpenClProgram : IDisposable
    {
        private IntPtr _program;
        private readonly DirectOpenClContext _context;
        private bool _disposed;
        private string _sourceHash = string.Empty;

        /// <summary>Gets whether this program was compiled from source or loaded as a native binary.</summary>
        internal OpenClProgramBuildOrigin BuildOrigin { get; }

        /// <summary>Gets the driver's typed classification of the built program artifact.</summary>
        internal OpenClProgramBinaryType BinaryType
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(DirectOpenClProgram));
                if (!OpenClNativeBindings.TryGetProgramBuildInfoUInt(
                        _program,
                        _context.Device,
                        OpenClNativeBindings.CL_PROGRAM_BINARY_TYPE,
                        out uint binaryType))
                {
                    return OpenClProgramBinaryType.Unavailable;
                }
                return binaryType switch
                {
                    OpenClNativeBindings.CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT =>
                        OpenClProgramBinaryType.CompiledObject,
                    OpenClNativeBindings.CL_PROGRAM_BINARY_TYPE_LIBRARY =>
                        OpenClProgramBinaryType.Library,
                    OpenClNativeBindings.CL_PROGRAM_BINARY_TYPE_EXECUTABLE =>
                        OpenClProgramBinaryType.Executable,
                    _ => OpenClProgramBinaryType.Unavailable
                };
            }
        }

        /// <summary>Gets the byte size of the selected device's compiled program artifact.</summary>
        internal long NativeBinarySizeBytes
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(DirectOpenClProgram));

                int err = OpenClNativeBindings.GetProgramInfo(
                    _program,
                    OpenClNativeBindings.CL_PROGRAM_BINARY_SIZES,
                    UIntPtr.Zero,
                    IntPtr.Zero,
                    out UIntPtr sizeNeeded);
                if (err != OpenClNativeBindings.CL_SUCCESS ||
                    (ulong)sizeNeeded < (ulong)UIntPtr.Size ||
                    (ulong)sizeNeeded > int.MaxValue)
                {
                    return 0;
                }

                IntPtr sizes = Marshal.AllocHGlobal(checked((int)(ulong)sizeNeeded));
                try
                {
                    err = OpenClNativeBindings.GetProgramInfo(
                        _program,
                        OpenClNativeBindings.CL_PROGRAM_BINARY_SIZES,
                        sizeNeeded,
                        sizes,
                        out _);
                    if (err != OpenClNativeBindings.CL_SUCCESS)
                        return 0;

                    ulong binarySize = UIntPtr.Size == 8
                        ? unchecked((ulong)Marshal.ReadInt64(sizes))
                        : unchecked((uint)Marshal.ReadInt32(sizes));
                    return binarySize > long.MaxValue ? long.MaxValue : (long)binarySize;
                }
                finally
                {
                    Marshal.FreeHGlobal(sizes);
                }
            }
        }

        /// <summary>
        /// Enable or disable binary caching. Defaults to true.
        /// Can be disabled via AIDOTNET_DISABLE_KERNEL_CACHE=1 environment variable.
        /// </summary>
        public static bool EnableBinaryCache { get; set; } = !IsEnvTrue("AIDOTNET_DISABLE_KERNEL_CACHE");

        public IntPtr Handle
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(DirectOpenClProgram));
                return _program;
            }
        }

        public DirectOpenClProgram(DirectOpenClContext context, string source)
        {
            _context = context;
            BuildOrigin = OpenClProgramBuildOrigin.SourceCompilation;
            _sourceHash = ComputeHash(source);

            var sources = new string[] { source };
            var lengths = new UIntPtr[] { (UIntPtr)source.Length };

            _program = OpenClNativeBindings.CreateProgramWithSource(
                context.Context,
                1,
                sources,
                lengths,
                out int err);

            if (err != OpenClNativeBindings.CL_SUCCESS || _program == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL program: {err}");
        }

        private DirectOpenClProgram(DirectOpenClContext context, IntPtr program, string sourceHash)
        {
            _context = context;
            _program = program;
            _sourceHash = sourceHash;
            BuildOrigin = OpenClProgramBuildOrigin.NativeBinaryCache;
        }

        /// <summary>
        /// Builds the program for the context's device, using binary cache if available.
        /// </summary>
        public void Build(string options = "")
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(DirectOpenClProgram));

            var devices = new IntPtr[] { _context.Device };
            int err = OpenClNativeBindings.BuildProgram(_program, 1, devices, options, IntPtr.Zero, IntPtr.Zero);

            if (err != OpenClNativeBindings.CL_SUCCESS)
            {
                string buildLog = OpenClNativeBindings.GetBuildLog(_program, _context.Device);
                throw new InvalidOperationException($"Failed to build OpenCL program (error {err}):\n{buildLog}");
            }

            // Save binary to disk cache after successful compilation
            if (EnableBinaryCache && !string.IsNullOrEmpty(_sourceHash))
            {
                try
                {
                    SaveBinaryToCache(_sourceHash, options);
                }
                catch
                {
                    // Non-fatal: cache write failure should not break compilation
                }
            }
        }

        /// <summary>
        /// Attempts to create a program from cached binary. Returns null if cache miss.
        /// </summary>
        public static DirectOpenClProgram? TryCreateFromCache(DirectOpenClContext context, string source, string buildOptions)
        {
            if (!EnableBinaryCache) return null;

            string hash = ComputeHash(source);
            string deviceKey = GetDeviceCacheKey(context);
            string cachePath = GetCachePath(hash, buildOptions, deviceKey);

            if (!File.Exists(cachePath)) return null;

            try
            {
                byte[] binary = File.ReadAllBytes(cachePath);
                if (binary.Length == 0)
                {
                    TryDeleteInvalidCacheEntry(cachePath);
                    return null;
                }

                var devices = new IntPtr[] { context.Device };
                var lengths = new UIntPtr[] { (UIntPtr)binary.Length };
                var binaryStatus = new int[1];

                GCHandle pinnedBinary = GCHandle.Alloc(binary, GCHandleType.Pinned);
                try
                {
                    var binaries = new IntPtr[] { pinnedBinary.AddrOfPinnedObject() };
                    IntPtr program = OpenClNativeBindings.CreateProgramWithBinary(
                        context.Context,
                        1,
                        devices,
                        lengths,
                        binaries,
                        binaryStatus,
                        out int err);

                    if (err != OpenClNativeBindings.CL_SUCCESS ||
                        binaryStatus[0] != OpenClNativeBindings.CL_SUCCESS ||
                        program == IntPtr.Zero)
                    {
                        if (program != IntPtr.Zero)
                            OpenClNativeBindings.ReleaseProgram(program);
                        TryDeleteInvalidCacheEntry(cachePath);
                        return null;
                    }

                    // Build the binary program (required by OpenCL spec)
                    int buildErr = OpenClNativeBindings.BuildProgram(program, 1, devices, buildOptions, IntPtr.Zero, IntPtr.Zero);
                    if (buildErr != OpenClNativeBindings.CL_SUCCESS)
                    {
                        OpenClNativeBindings.ReleaseProgram(program);
                        TryDeleteInvalidCacheEntry(cachePath);
                        return null;
                    }

                    return new DirectOpenClProgram(context, program, hash);
                }
                finally
                {
                    pinnedBinary.Free();
                }
            }
            catch
            {
                // A truncated or otherwise unreadable artifact must not permanently poison this key.
                TryDeleteInvalidCacheEntry(cachePath);
                return null;
            }
        }

        private void SaveBinaryToCache(string hash, string buildOptions)
        {
            string deviceKey = GetDeviceCacheKey(_context);
            string cachePath = GetCachePath(hash, buildOptions, deviceKey);

            // Get binary size — pass paramValueSize=0 when paramValue is null (OpenCL spec requirement)
            int err = OpenClNativeBindings.GetProgramInfo(
                _program,
                OpenClNativeBindings.CL_PROGRAM_BINARY_SIZES,
                UIntPtr.Zero,
                IntPtr.Zero,
                out UIntPtr sizeNeeded);

            if (err != OpenClNativeBindings.CL_SUCCESS) return;

            // Allocate for binary sizes array (one per device)
            if ((ulong)sizeNeeded > int.MaxValue) return;
            IntPtr sizesPtr = Marshal.AllocHGlobal(checked((int)(ulong)sizeNeeded));
            try
            {
                err = OpenClNativeBindings.GetProgramInfo(
                    _program,
                    OpenClNativeBindings.CL_PROGRAM_BINARY_SIZES,
                    sizeNeeded,
                    sizesPtr,
                    out _);

                if (err != OpenClNativeBindings.CL_SUCCESS) return;

                UIntPtr binarySize;
                if (UIntPtr.Size == 8)
                    binarySize = (UIntPtr)(ulong)Marshal.ReadInt64(sizesPtr);
                else
                    binarySize = (UIntPtr)(uint)Marshal.ReadInt32(sizesPtr);

                if ((ulong)binarySize == 0 || (ulong)binarySize > int.MaxValue) return;

                // Allocate buffer for the binary
                IntPtr binaryPtr = Marshal.AllocHGlobal(checked((int)(ulong)binarySize));
                try
                {
                    // Get the binary: pass array of pointers to binaries
                    IntPtr binariesArrayPtr = Marshal.AllocHGlobal(IntPtr.Size);
                    try
                    {
                        Marshal.WriteIntPtr(binariesArrayPtr, binaryPtr);

                        err = OpenClNativeBindings.GetProgramInfo(
                            _program,
                            OpenClNativeBindings.CL_PROGRAM_BINARIES,
                            (UIntPtr)IntPtr.Size,
                            binariesArrayPtr,
                            out _);

                        if (err != OpenClNativeBindings.CL_SUCCESS) return;

                        // Copy to managed array and write to disk
                        byte[] binary = new byte[(int)(ulong)binarySize];
                        Marshal.Copy(binaryPtr, binary, 0, binary.Length);

                        string dir = Path.GetDirectoryName(cachePath) ?? GetCacheDirectory();
                        if (!Directory.Exists(dir))
                            Directory.CreateDirectory(dir);

                        // Write to temp file then rename for atomic cache update
                        string tempPath = cachePath + ".tmp." + Guid.NewGuid().ToString("N");
                        try
                        {
                            File.WriteAllBytes(tempPath, binary);
                            // The artifact is deterministic for this source/options/device key. Never
                            // delete a completed winner: File.Move is the atomic first-writer-wins commit
                            // supported by every target framework in this package.
                            File.Move(tempPath, cachePath);
                        }
                        catch (IOException)
                        {
                            // Another process may have written the cache file concurrently — that's fine
                            try { File.Delete(tempPath); } catch { /* best effort cleanup */ }
                        }
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(binariesArrayPtr);
                    }
                }
                finally
                {
                    Marshal.FreeHGlobal(binaryPtr);
                }
            }
            finally
            {
                Marshal.FreeHGlobal(sizesPtr);
            }
        }

        private static string ComputeHash(string source)
        {
            using (var sha = SHA256.Create())
            {
                byte[] hashBytes = sha.ComputeHash(Encoding.UTF8.GetBytes(source));
                var sb = new StringBuilder(hashBytes.Length * 2);
                for (int i = 0; i < hashBytes.Length; i++)
                    sb.Append(hashBytes[i].ToString("x2"));
                return sb.ToString();
            }
        }

        private static string GetCachePath(string sourceHash, string buildOptions, string deviceKey = "")
        {
            // Device executables are not portable across source, compiler flags, devices, or
            // drivers. Hash the complete identity once so the filename remains short enough for
            // net471-era Windows paths without weakening collision resistance through truncation.
            string cacheIdentity = string.Join("|", new[]
            {
                "opencl-native-binary-v2",
                sourceHash,
                buildOptions ?? string.Empty,
                deviceKey ?? string.Empty
            });
            return Path.Combine(GetCacheDirectory(), ComputeHash(cacheIdentity) + ".clbin");
        }

        private static string GetDeviceCacheKey(DirectOpenClContext context)
        {
            return string.Join("|", new[]
            {
                context.DeviceVendor,
                context.DeviceName,
                context.DeviceBoardName,
                context.DriverVersion,
                context.OpenClVersion
            });
        }

        private static string GetCacheDirectory()
        {
            string? customDir = Environment.GetEnvironmentVariable("AIDOTNET_KERNEL_CACHE_DIR");
            if (!string.IsNullOrWhiteSpace(customDir))
                return customDir;

            string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(appData, "AiDotNet", "KernelCache");
        }

        private static void TryDeleteInvalidCacheEntry(string cachePath)
        {
            try { File.Delete(cachePath); }
            catch { /* another process may be replacing or reading it */ }
        }

        private static bool IsEnvTrue(string name)
        {
            string? val = Environment.GetEnvironmentVariable(name);
            return string.Equals(val, "1", StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(val, "true", StringComparison.OrdinalIgnoreCase);
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_program != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseProgram(_program);
                _program = IntPtr.Zero;
            }

            _disposed = true;
        }
    }
}
