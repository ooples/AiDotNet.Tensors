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

        /// <summary>
        /// Enable or disable binary caching. Defaults to true.
        /// Can be disabled via AIDOTNET_DISABLE_KERNEL_CACHE=1 environment variable.
        /// </summary>
        public static bool EnableBinaryCache { get; set; } = !IsEnvTrue("AIDOTNET_DISABLE_KERNEL_CACHE");

        private static readonly string CacheDirectory = GetCacheDirectory();

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
                if (binary.Length == 0) return null;

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

                    if (err != OpenClNativeBindings.CL_SUCCESS || program == IntPtr.Zero)
                        return null;

                    // Build the binary program (required by OpenCL spec)
                    int buildErr = OpenClNativeBindings.BuildProgram(program, 1, devices, buildOptions, IntPtr.Zero, IntPtr.Zero);
                    if (buildErr != OpenClNativeBindings.CL_SUCCESS)
                    {
                        OpenClNativeBindings.ReleaseProgram(program);
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
                // Cache read failure is non-fatal
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
            IntPtr sizesPtr = Marshal.AllocHGlobal((int)sizeNeeded);
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

                if ((ulong)binarySize == 0) return;

                // Allocate buffer for the binary
                IntPtr binaryPtr = Marshal.AllocHGlobal((int)(ulong)binarySize);
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

                        string dir = Path.GetDirectoryName(cachePath) ?? CacheDirectory;
                        if (!Directory.Exists(dir))
                            Directory.CreateDirectory(dir);

                        // Write to temp file then rename for atomic cache update
                        string tempPath = cachePath + ".tmp." + Guid.NewGuid().ToString("N");
                        try
                        {
                            File.WriteAllBytes(tempPath, binary);
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
            // Include build options and device info in cache key since binaries are not portable across devices/drivers
            string optionsHash = string.IsNullOrEmpty(buildOptions) ? "default" : ComputeHash(buildOptions).Substring(0, 8);
            string deviceSuffix = string.IsNullOrEmpty(deviceKey) ? "" : $"_{ComputeHash(deviceKey).Substring(0, 8)}";
            return Path.Combine(CacheDirectory, $"{sourceHash.Substring(0, 16)}_{optionsHash}{deviceSuffix}.clbin");
        }

        private static string GetDeviceCacheKey(DirectOpenClContext context)
        {
            return $"{context.DeviceName}|{context.DriverVersion}";
        }

        private static string GetCacheDirectory()
        {
            string? customDir = Environment.GetEnvironmentVariable("AIDOTNET_KERNEL_CACHE_DIR");
            if (!string.IsNullOrWhiteSpace(customDir))
                return customDir;

            string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(appData, "AiDotNet", "KernelCache");
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
