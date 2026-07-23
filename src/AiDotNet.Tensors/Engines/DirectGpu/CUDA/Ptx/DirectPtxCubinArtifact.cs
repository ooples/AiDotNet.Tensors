using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxModuleImageKind
{
    EmbeddedCubin,
    DiskCacheCubin,
    DriverLinkedCubin,
    DriverJitPtx
}

/// <summary>
/// Architecture-specific executable produced from a direct-PTX module. The
/// driver linker returns the exact cubin that is loaded, so validation can
/// preserve and disassemble the same machine-code container used at runtime.
/// </summary>
internal sealed class DirectPtxCubinArtifact
{
    internal DirectPtxCubinArtifact(
        byte[] image,
        string sourceKey,
        string cubinSha256,
        DirectPtxModuleImageKind imageKind,
        string? path,
        string compilerLog)
    {
        Image = image;
        SourceKey = sourceKey;
        CubinSha256 = cubinSha256;
        ImageKind = imageKind;
        Path = path;
        CompilerLog = compilerLog;
    }

    internal byte[] Image { get; }
    internal string SourceKey { get; }
    internal string CubinSha256 { get; }
    internal DirectPtxModuleImageKind ImageKind { get; }
    internal string? Path { get; }
    internal string CompilerLog { get; }
}

/// <summary>
/// Compiles PTX with the CUDA Driver linker, verifies the returned cubin, and
/// resolves artifacts in production order: embedded package resource, verified
/// disk cache, then a new driver-link compilation. PTX text is never passed to
/// cuModuleLoadData on this path.
/// </summary>
internal static class DirectPtxCubinArtifactCache
{
    internal const int PipelineVersion = 3;
    private const int PtxInputType = 1; // CU_JIT_INPUT_PTX
    private const int LogBytes = 16 * 1024;
    private const string CacheEnvironmentVariable = "AIDOTNET_DIRECT_PTX_CACHE_PATH";
    private static readonly object Sync = new();
    [ThreadStatic]
    private static int _freshCompileScopeDepth;
    private static readonly Lazy<IReadOnlyDictionary<string, EmbeddedManifestEntry>>
        EmbeddedManifest = new(ReadEmbeddedManifest);

    private sealed record EmbeddedManifestEntry(
        string CubinSha256,
        string LinkerLogSha256);

    internal static DirectPtxCubinArtifact Resolve(DirectPtxRuntime runtime, string ptx)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        PtxCompat.ThrowIfNullOrWhiteSpace(ptx, nameof(ptx));
        ptx = CanonicalizePtx(ptx);
        string sourceKey = ComputeSourceKey(
            ptx, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);

        lock (Sync)
        {
            if (_freshCompileScopeDepth != 0)
                return Compile(runtime, ptx, sourceKey, cachePath: null);
            DirectPtxCubinArtifact? embedded = TryReadEmbedded(runtime, sourceKey);
            if (embedded != null)
                return embedded;

            string? cachePath = GetCachePath(runtime, sourceKey);
            DirectPtxCubinArtifact? cached = TryReadDisk(
                runtime, cachePath, sourceKey);
            if (cached != null)
                return cached;

            DirectPtxCubinArtifact compiled = Compile(runtime, ptx, sourceKey, cachePath);
            TryWriteDisk(compiled, runtime);
            return compiled;
        }
    }

    internal static string ComputeSourceKey(string ptx, int major, int minor)
    {
        ptx = CanonicalizePtx(ptx);
        string identity = "direct-ptx-cubin-v" +
            PipelineVersion.ToString(CultureInfo.InvariantCulture) + "|sm" +
            major.ToString(CultureInfo.InvariantCulture) +
            minor.ToString(CultureInfo.InvariantCulture) + "|" + ptx;
        return Sha256(Encoding.UTF8.GetBytes(identity));
    }

    internal static string ComputePtxSha256(string ptx) =>
        Sha256(Encoding.UTF8.GetBytes(CanonicalizePtx(ptx)));

    /// <summary>
    /// Produces a fresh exact cubin for a release exporter. Unlike normal
    /// resolution, this deliberately bypasses embedded and disk artifacts so
    /// the emitted linker log and binary come from the current driver linker.
    /// </summary>
    internal static DirectPtxCubinArtifact CompileExact(
        DirectPtxRuntime runtime,
        string ptx)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        PtxCompat.ThrowIfNullOrWhiteSpace(ptx, nameof(ptx));
        ptx = CanonicalizePtx(ptx);
        string sourceKey = ComputeSourceKey(
            ptx, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        lock (Sync)
            return Compile(runtime, ptx, sourceKey, cachePath: null);
    }

    /// <summary>
    /// Makes kernel constructors use a fresh driver link while a release
    /// exporter validates their live blueprint metadata. This deliberately
    /// bypasses a stale embedded manifest so the exporter can replace it; the
    /// normal production resolver remains strict.
    /// </summary>
    internal static IDisposable EnterFreshCompileScope()
    {
        checked { _freshCompileScopeDepth++; }
        return new FreshCompileScope();
    }

    internal static string FormatLinkerLog(
        string linkerInfoLog,
        DirectPtxRuntime runtime) =>
        "pipeline-version=" + PipelineVersion.ToString(CultureInfo.InvariantCulture) + "\n" +
        "target=sm" + runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
        runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture) + "\n" +
        "driver-version=" + runtime.DriverVersion.ToString(CultureInfo.InvariantCulture) + "\n" +
        "cuda-driver-linker-info-log:\n" + NormalizeLinkerInfoLog(linkerInfoLog);

    /// <summary>
    /// CUDA 13.3 on Windows can print an uninitialized signed integer in the
    /// informational "used N barriers" field for kernels that use no named
    /// barriers. Preserve valid 0..16 values and make only impossible values
    /// deterministic; resource enforcement continues to use driver function
    /// attributes and final SASS rather than this advisory text.
    /// </summary>
    internal static string NormalizeLinkerInfoLog(string? linkerInfoLog) =>
        Regex.Replace(
            linkerInfoLog ?? string.Empty,
            @"used (-?\d+) barriers",
            match => int.TryParse(
                    match.Groups[1].Value,
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out int barriers) && barriers >= 0 && barriers <= 16
                ? match.Value
                : "used unavailable barriers",
            RegexOptions.CultureInvariant);

    internal static string CanonicalizePtx(string ptx)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(ptx, nameof(ptx));
        if (ptx.IndexOf('\r') < 0)
            return ptx;
        return ptx.Replace("\r\n", "\n").Replace('\r', '\n');
    }

    private static DirectPtxCubinArtifact? TryReadEmbedded(
        DirectPtxRuntime runtime, string sourceKey)
    {
        string suffix = ".Artifacts.sm" +
            runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
            runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture) + "." +
            sourceKey + ".cubin";
        Assembly assembly = typeof(DirectPtxCubinArtifactCache).Assembly;
        string? resourceName = null;
        foreach (string candidate in assembly.GetManifestResourceNames())
        {
            if (candidate.EndsWith(suffix, StringComparison.Ordinal))
            {
                resourceName = candidate;
                break;
            }
        }
        if (resourceName == null)
            return null;

        using Stream? stream = assembly.GetManifestResourceStream(resourceName);
        if (stream == null)
            throw new InvalidDataException(
                "The embedded direct-PTX cubin resource could not be opened: " + resourceName);
        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        byte[] image = memory.ToArray();
        ValidateCubin(image, "embedded resource " + resourceName);
        string cubinHash = Sha256(image);
        string manifestKey = "sm" +
            runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
            runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture) + "|" + sourceKey;
        if (!EmbeddedManifest.Value.TryGetValue(
                manifestKey, out EmbeddedManifestEntry? manifest) ||
            !string.Equals(
                manifest.CubinSha256, cubinHash, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException(
                "Embedded direct-PTX cubin failed its release-manifest hash: " + resourceName);
        string linkerResourceName = resourceName.Substring(
            0, resourceName.Length - ".cubin".Length) + ".linker.txt";
        using Stream? linkerStream = assembly.GetManifestResourceStream(linkerResourceName);
        if (linkerStream == null)
            throw new InvalidDataException(
                "Embedded direct-PTX cubin is missing its linker-log sidecar: " + resourceName);
        using var linkerMemory = new MemoryStream();
        linkerStream.CopyTo(linkerMemory);
        byte[] linkerBytes = linkerMemory.ToArray();
        if (!string.Equals(
                manifest.LinkerLogSha256, Sha256(linkerBytes),
                StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException(
                "Embedded direct-PTX linker log failed its release-manifest hash: " +
                linkerResourceName);
        return new DirectPtxCubinArtifact(
            image, sourceKey, cubinHash, DirectPtxModuleImageKind.EmbeddedCubin,
            resourceName, Encoding.UTF8.GetString(linkerBytes));
    }

    private static IReadOnlyDictionary<string, EmbeddedManifestEntry>
        ReadEmbeddedManifest()
    {
        var result = new Dictionary<string, EmbeddedManifestEntry>(StringComparer.Ordinal);
        Assembly assembly = typeof(DirectPtxCubinArtifactCache).Assembly;
        foreach (string resourceName in assembly.GetManifestResourceNames())
        {
            const string suffix = "-cubins.tsv";
            if (!resourceName.EndsWith(suffix, StringComparison.Ordinal))
                continue;
            string marker = ".Artifacts.sm";
            int markerIndex = resourceName.IndexOf(marker, StringComparison.Ordinal);
            if (markerIndex < 0)
                continue;
            int architectureStart = markerIndex + marker.Length;
            int architectureEnd = resourceName.IndexOf('.', architectureStart);
            if (architectureEnd <= architectureStart)
                continue;
            string architecture = "sm" + resourceName.Substring(
                architectureStart, architectureEnd - architectureStart);
            using Stream? stream = assembly.GetManifestResourceStream(resourceName);
            if (stream == null)
                continue;
            using var reader = new StreamReader(stream, Encoding.UTF8, true, 1024, leaveOpen: false);
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (line.Length == 0 || line[0] == '#' || line.StartsWith("blueprint-id", StringComparison.Ordinal))
                    continue;
                string[] columns = line.Split('\t');
                if (columns.Length != 6)
                    throw new InvalidDataException(
                        "Malformed embedded direct-PTX cubin manifest row in " +
                        resourceName + ": " + line);
                string key = architecture + "|" + columns[2];
                var entry = new EmbeddedManifestEntry(columns[3], columns[5]);
                if (result.TryGetValue(
                        key, out EmbeddedManifestEntry? existing) &&
                    (!string.Equals(
                        existing.CubinSha256, entry.CubinSha256,
                        StringComparison.OrdinalIgnoreCase) ||
                     !string.Equals(
                        existing.LinkerLogSha256, entry.LinkerLogSha256,
                        StringComparison.OrdinalIgnoreCase)))
                    throw new InvalidDataException(
                        "Conflicting embedded direct-PTX cubin manifests for " + key + ".");
                result[key] = entry;
            }
        }
        return result;
    }

    private static DirectPtxCubinArtifact? TryReadDisk(
        DirectPtxRuntime runtime,
        string? path,
        string sourceKey)
    {
        if (path == null || !File.Exists(path))
            return null;
        try
        {
            byte[] image = File.ReadAllBytes(path);
            ValidateCubin(image, path);
            string cubinHash = Sha256(image);
            string hashPath = path + ".sha256";
            if (!File.Exists(hashPath) ||
                !string.Equals(File.ReadAllText(hashPath).Trim(), cubinHash,
                    StringComparison.OrdinalIgnoreCase))
                return null;
            string linkerPath = path + ".linker.txt";
            if (!File.Exists(linkerPath))
                return null;
            byte[] linkerBytes = File.ReadAllBytes(linkerPath);
            string auditPath = path + ".audit.txt";
            if (!File.Exists(auditPath) ||
                !DiskAuditMatches(
                    auditPath, runtime, sourceKey, cubinHash,
                    Sha256(linkerBytes)))
                return null;
            return new DirectPtxCubinArtifact(
                image, sourceKey, cubinHash, DirectPtxModuleImageKind.DiskCacheCubin,
                path, Encoding.UTF8.GetString(linkerBytes));
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
        catch (InvalidDataException)
        {
            // A partial or stale cache entry is not authoritative. Recompile
            // from the canonical PTX and replace it below.
            return null;
        }
    }

    private static bool DiskAuditMatches(
        string path,
        DirectPtxRuntime runtime,
        string sourceKey,
        string cubinHash,
        string linkerLogHash)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(path))
        {
            int separator = line.IndexOf('=');
            if (separator <= 0 || separator == line.Length - 1)
                return false;
            string key = line.Substring(0, separator);
            if (values.ContainsKey(key))
                return false;
            values.Add(key, line.Substring(separator + 1));
        }
        string target = "sm" +
            runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
            runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture);
        return values.Count == 6 &&
            values.TryGetValue("pipeline-version", out string? pipeline) &&
            string.Equals(
                pipeline, PipelineVersion.ToString(CultureInfo.InvariantCulture),
                StringComparison.Ordinal) &&
            values.TryGetValue("source-key", out string? recordedSource) &&
            string.Equals(recordedSource, sourceKey, StringComparison.Ordinal) &&
            values.TryGetValue("cubin-sha256", out string? recordedHash) &&
            string.Equals(recordedHash, cubinHash, StringComparison.OrdinalIgnoreCase) &&
            values.TryGetValue("target", out string? recordedTarget) &&
            string.Equals(recordedTarget, target, StringComparison.Ordinal) &&
            values.TryGetValue("driver-version", out string? recordedDriver) &&
            string.Equals(
                recordedDriver,
                runtime.DriverVersion.ToString(CultureInfo.InvariantCulture),
                StringComparison.Ordinal) &&
            values.TryGetValue(
                "linker-log-sha256", out string? recordedLinkerLogHash) &&
            string.Equals(
                recordedLinkerLogHash, linkerLogHash,
                StringComparison.OrdinalIgnoreCase);
    }

    private static unsafe DirectPtxCubinArtifact Compile(
        DirectPtxRuntime runtime, string ptx, string sourceKey, string? cachePath)
    {
        using var _ = runtime.Enter();
        IntPtr infoLog = IntPtr.Zero;
        IntPtr errorLog = IntPtr.Zero;
        IntPtr ptxBuffer = IntPtr.Zero;
        IntPtr linkState = IntPtr.Zero;
        try
        {
            infoLog = Marshal.AllocHGlobal(LogBytes);
            errorLog = Marshal.AllocHGlobal(LogBytes);
            new Span<byte>((void*)infoLog, LogBytes).Clear();
            new Span<byte>((void*)errorLog, LogBytes).Clear();
            int[] options = [3, 4, 5, 6, 12]; // logs plus CU_JIT_LOG_VERBOSE
            IntPtr[] values =
                [infoLog, (IntPtr)LogBytes, errorLog, (IntPtr)LogBytes, (IntPtr)1];
            DirectPtxRuntime.Check(
                CudaNativeBindings.cuLinkCreate(
                    (uint)options.Length, options, values, out linkState),
                "cuLinkCreate(direct PTX)");

            byte[] ptxBytes = Encoding.ASCII.GetBytes(ptx + "\0");
            ptxBuffer = Marshal.AllocHGlobal(ptxBytes.Length);
            Marshal.Copy(ptxBytes, 0, ptxBuffer, ptxBytes.Length);
            CudaResult addResult = CudaNativeBindings.cuLinkAddData(
                linkState, PtxInputType, ptxBuffer, (UIntPtr)(uint)ptxBytes.Length,
                "direct-ptx.ptx", 0, IntPtr.Zero, IntPtr.Zero);
            if (addResult != CudaResult.Success)
                throw LinkFailure("cuLinkAddData(PTX)", addResult, errorLog, infoLog);

            CudaResult completeResult = CudaNativeBindings.cuLinkComplete(
                linkState, out IntPtr cubin, out UIntPtr cubinSize);
            if (completeResult != CudaResult.Success)
                throw LinkFailure("cuLinkComplete(PTX)", completeResult, errorLog, infoLog);
            ulong length64 = cubinSize.ToUInt64();
            if (length64 == 0 || length64 > int.MaxValue)
                throw new InvalidDataException(
                    "CUDA linker returned an invalid cubin length: " +
                    length64.ToString(CultureInfo.InvariantCulture));
            byte[] image = new byte[(int)length64];
            Marshal.Copy(cubin, image, 0, image.Length);
            ValidateCubin(image, "CUDA driver linker output");
            string compilerLog = Marshal.PtrToStringAnsi(infoLog) ?? string.Empty;
            return new DirectPtxCubinArtifact(
                image, sourceKey, Sha256(image), DirectPtxModuleImageKind.DriverLinkedCubin,
                cachePath, compilerLog);
        }
        finally
        {
            if (linkState != IntPtr.Zero)
                CudaNativeBindings.cuLinkDestroy(linkState);
            if (ptxBuffer != IntPtr.Zero)
                Marshal.FreeHGlobal(ptxBuffer);
            if (errorLog != IntPtr.Zero)
                Marshal.FreeHGlobal(errorLog);
            if (infoLog != IntPtr.Zero)
                Marshal.FreeHGlobal(infoLog);
        }
    }

    private static InvalidOperationException LinkFailure(
        string operation, CudaResult result, IntPtr errorLog, IntPtr infoLog) =>
        new(operation + " failed with CUDA driver status " +
            ((int)result).ToString(CultureInfo.InvariantCulture) + " (" + result + ").\n" +
            "Linker error log:\n" + (Marshal.PtrToStringAnsi(errorLog) ?? string.Empty) +
            "\nLinker info log:\n" + (Marshal.PtrToStringAnsi(infoLog) ?? string.Empty));

    private static string? GetCachePath(DirectPtxRuntime runtime, string sourceKey)
    {
        try
        {
            string? directory = Environment.GetEnvironmentVariable(CacheEnvironmentVariable);
            if (string.IsNullOrWhiteSpace(directory))
            {
                string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                if (string.IsNullOrWhiteSpace(local))
                    return null;
                directory = System.IO.Path.Combine(local, "AiDotNet", "Tensors", "DirectPtx");
            }
            directory = System.IO.Path.GetFullPath(directory);
            string fileName = "sm" +
                runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
                runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture) + "-drv" +
                runtime.DriverVersion.ToString(CultureInfo.InvariantCulture) + "-" + sourceKey + ".cubin";
            return System.IO.Path.Combine(directory, fileName);
        }
        catch
        {
            return null;
        }
    }

    private static void TryWriteDisk(
        DirectPtxCubinArtifact artifact, DirectPtxRuntime runtime)
    {
        string? path = artifact.Path;
        if (path == null)
            return;
        string? temporary = null;
        try
        {
            string? directory = System.IO.Path.GetDirectoryName(path);
            if (string.IsNullOrWhiteSpace(directory))
                return;
            Directory.CreateDirectory(directory);
            temporary = path + ".tmp-" + Guid.NewGuid().ToString("N");
            File.WriteAllBytes(temporary, artifact.Image);
            File.Copy(temporary, path, overwrite: true);
            File.Delete(temporary);
            temporary = null;
            File.WriteAllText(path + ".sha256", artifact.CubinSha256 + Environment.NewLine);
            string linkerLog = FormatLinkerLog(artifact.CompilerLog, runtime);
            byte[] linkerBytes = Encoding.UTF8.GetBytes(linkerLog);
            File.WriteAllBytes(path + ".linker.txt", linkerBytes);
            // Write the audit marker last. Readers reject partial cache entries,
            // including a binary paired with a stale or altered linker log.
            File.WriteAllText(path + ".audit.txt",
                "pipeline-version=" + PipelineVersion.ToString(CultureInfo.InvariantCulture) + Environment.NewLine +
                "source-key=" + artifact.SourceKey + Environment.NewLine +
                "cubin-sha256=" + artifact.CubinSha256 + Environment.NewLine +
                "target=sm" + runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
                runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture) + Environment.NewLine +
                "driver-version=" + runtime.DriverVersion.ToString(CultureInfo.InvariantCulture) + Environment.NewLine +
                "linker-log-sha256=" + Sha256(linkerBytes) + Environment.NewLine);
        }
        catch (IOException)
        {
            // A read-only cache does not prevent use of the in-memory cubin.
        }
        catch (UnauthorizedAccessException)
        {
            // A read-only cache does not prevent use of the in-memory cubin.
        }
        finally
        {
            if (temporary != null)
            {
                try { File.Delete(temporary); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
    }

    private static void ValidateCubin(byte[] image, string source)
    {
        if (image.Length < 64 || image[0] != 0x7f || image[1] != (byte)'E' ||
            image[2] != (byte)'L' || image[3] != (byte)'F')
            throw new InvalidDataException(
                "Direct-PTX compiled artifact is not a CUDA ELF cubin: " + source);
    }

    private static string Sha256(byte[] bytes)
    {
        using SHA256 sha = SHA256.Create();
        return PtxCompat.ToHexString(sha.ComputeHash(bytes)).ToLowerInvariant();
    }

    private sealed class FreshCompileScope : IDisposable
    {
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            if (_freshCompileScopeDepth <= 0)
                throw new InvalidOperationException(
                    "Direct-PTX fresh-compile scope depth is unbalanced.");
            _freshCompileScopeDepth--;
        }
    }
}
