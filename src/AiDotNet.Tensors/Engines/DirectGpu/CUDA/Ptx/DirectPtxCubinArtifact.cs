using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
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
    private static readonly Lazy<IReadOnlyDictionary<string, EmbeddedArtifact>> EmbeddedArtifacts =
        new(ReadEmbeddedArtifacts);

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
            DirectPtxCubinArtifact? embedded = TryReadEmbedded(
                runtime.ComputeCapabilityMajor,
                runtime.ComputeCapabilityMinor,
                sourceKey,
                ComputePtxSha256(ptx));
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

    internal static DirectPtxCubinArtifact? TryResolveEmbedded(
        string ptx,
        int computeCapabilityMajor,
        int computeCapabilityMinor)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(ptx, nameof(ptx));
        ptx = CanonicalizePtx(ptx);
        return TryReadEmbedded(
            computeCapabilityMajor,
            computeCapabilityMinor,
            ComputeSourceKey(ptx, computeCapabilityMajor, computeCapabilityMinor),
            ComputePtxSha256(ptx));
    }

    internal static string CanonicalizePtx(string ptx)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(ptx, nameof(ptx));
        if (ptx.IndexOf('\r') < 0)
            return ptx;
        return ptx.Replace("\r\n", "\n").Replace('\r', '\n');
    }

    private static DirectPtxCubinArtifact? TryReadEmbedded(
        int computeCapabilityMajor,
        int computeCapabilityMinor,
        string sourceKey,
        string ptxSha256)
    {
        string architecture = "sm" +
            computeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
            computeCapabilityMinor.ToString(CultureInfo.InvariantCulture);
        string sourceIdentity = architecture + "|source|" + sourceKey;
        string ptxIdentity = architecture + "|ptx|" + ptxSha256;
        if (!EmbeddedArtifacts.Value.TryGetValue(sourceIdentity, out EmbeddedArtifact? artifact) &&
            !EmbeddedArtifacts.Value.TryGetValue(ptxIdentity, out artifact))
            return null;

        Assembly assembly = typeof(DirectPtxCubinArtifactCache).Assembly;
        using Stream? stream = assembly.GetManifestResourceStream(artifact.ResourceName);
        if (stream == null)
            throw new InvalidDataException(
                "The embedded direct-PTX cubin resource could not be opened: " + artifact.ResourceName);
        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        byte[] image = memory.ToArray();
        ValidateCubin(image, "embedded resource " + artifact.ResourceName);
        string cubinHash = Sha256(image);
        if (!string.Equals(artifact.CubinSha256, cubinHash, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException(
                "Embedded direct-PTX cubin failed its release-manifest hash: " + artifact.ResourceName);
        string compilerLog = "precompiled package cubin";
        if (artifact.LinkerLogSha256 != null)
        {
            if (artifact.LinkerResourceName == null)
                throw new InvalidDataException(
                    "Embedded direct-PTX cubin is missing its linker-log sidecar: " +
                    artifact.ResourceName);
            using Stream? linkerStream = assembly.GetManifestResourceStream(
                artifact.LinkerResourceName);
            if (linkerStream == null)
                throw new InvalidDataException(
                    "The embedded direct-PTX linker-log resource could not be opened: " +
                    artifact.LinkerResourceName);
            using var linkerMemory = new MemoryStream();
            linkerStream.CopyTo(linkerMemory);
            byte[] linkerBytes = linkerMemory.ToArray();
            if (!string.Equals(
                    artifact.LinkerLogSha256, Sha256(linkerBytes),
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Embedded direct-PTX linker log failed its release-manifest hash: " +
                    artifact.LinkerResourceName);
            compilerLog = Encoding.UTF8.GetString(linkerBytes);
        }
        return new DirectPtxCubinArtifact(
            image, sourceKey, cubinHash, DirectPtxModuleImageKind.EmbeddedCubin,
            artifact.ResourceName, compilerLog);
    }

    private static IReadOnlyDictionary<string, EmbeddedArtifact> ReadEmbeddedArtifacts()
    {
        var result = new Dictionary<string, EmbeddedArtifact>(StringComparer.Ordinal);
        Assembly assembly = typeof(DirectPtxCubinArtifactCache).Assembly;
        string[] orderedResourceNames = assembly.GetManifestResourceNames()
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();
        foreach (string resourceName in orderedResourceNames)
        {
            if (!resourceName.EndsWith(".tsv", StringComparison.Ordinal) ||
                !TryParseEmbeddedArtifactArchitecture(resourceName, out string architecture))
                continue;

            using Stream? stream = assembly.GetManifestResourceStream(resourceName);
            if (stream == null)
                throw new InvalidDataException(
                    "The embedded direct-PTX manifest could not be opened: " + resourceName);
            using var reader = new StreamReader(stream, Encoding.UTF8, true, 1024, leaveOpen: false);
            string[]? header = null;
            int ptxIndex = -1;
            int sourceIndex = -1;
            int cubinIndex = -1;
            int fileIndex = -1;
            int linkerLogIndex = -1;
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (line.Length == 0 || line[0] == '#')
                    continue;
                if (header == null)
                {
                    header = line.Split('\t');
                    ptxIndex = Array.IndexOf(header, "ptx-sha256");
                    sourceIndex = Array.IndexOf(header, "source-key");
                    cubinIndex = Array.IndexOf(header, "cubin-sha256");
                    fileIndex = Array.IndexOf(header, "file");
                    linkerLogIndex = Array.IndexOf(header, "linker-log-sha256");
                    // Artifact directories can also contain non-manifest TSV
                    // evidence. Only a table with the release-manifest identity
                    // columns participates in executable resolution.
                    if (ptxIndex < 0 || cubinIndex < 0 || fileIndex < 0)
                        break;
                    continue;
                }

                string[] columns = line.Split('\t');
                if (columns.Length <= Math.Max(fileIndex, Math.Max(ptxIndex, cubinIndex)))
                    throw new InvalidDataException(
                        "Malformed embedded direct-PTX manifest row in " + resourceName + ": " + line);

                string? cubinResource = FindEmbeddedCubinResource(
                    orderedResourceNames, architecture, columns[fileIndex]);
                if (cubinResource == null)
                    throw new InvalidDataException(
                        "Embedded direct-PTX manifest references a missing cubin: " +
                        resourceName + " -> " + columns[fileIndex]);
                string? linkerLogHash = null;
                string? linkerResource = null;
                if (linkerLogIndex >= 0)
                {
                    if (linkerLogIndex >= columns.Length ||
                        string.IsNullOrWhiteSpace(columns[linkerLogIndex]) ||
                        !columns[fileIndex].EndsWith(".cubin", StringComparison.Ordinal))
                        throw new InvalidDataException(
                            "Malformed embedded direct-PTX linker-log manifest row in " +
                            resourceName + ": " + line);
                    linkerLogHash = columns[linkerLogIndex];
                    string linkerFile = columns[fileIndex].Substring(
                        0, columns[fileIndex].Length - ".cubin".Length) + ".linker.txt";
                    linkerResource = FindEmbeddedCubinResource(
                        orderedResourceNames, architecture, linkerFile);
                    if (linkerResource == null)
                        throw new InvalidDataException(
                            "Embedded direct-PTX manifest references a missing linker-log sidecar: " +
                            resourceName + " -> " + linkerFile);
                }
                var artifact = new EmbeddedArtifact(
                    columns[cubinIndex], cubinResource, linkerLogHash, linkerResource);
                AddEmbeddedArtifact(result, architecture + "|ptx|" + columns[ptxIndex], artifact);
                if (sourceIndex >= 0 && sourceIndex < columns.Length &&
                    !string.IsNullOrWhiteSpace(columns[sourceIndex]))
                    AddEmbeddedArtifact(
                        result, architecture + "|source|" + columns[sourceIndex], artifact);
            }
        }
        return result;
    }

    internal static bool TryParseEmbeddedArtifactArchitecture(
        string resourceName,
        out string architecture)
    {
        const string marker = ".Artifacts.sm";
        int markerIndex = resourceName.IndexOf(marker, StringComparison.Ordinal);
        if (markerIndex < 0)
        {
            architecture = string.Empty;
            return false;
        }

        int architectureStart = markerIndex + ".Artifacts.".Length;
        int architectureEnd = resourceName.IndexOf('.', architectureStart);
        if (architectureEnd <= architectureStart)
        {
            architecture = string.Empty;
            return false;
        }

        architecture = resourceName.Substring(
            architectureStart, architectureEnd - architectureStart);
        return true;
    }

    private static string? FindEmbeddedCubinResource(
        IReadOnlyList<string> orderedResourceNames,
        string architecture,
        string fileName)
    {
        string architectureMarker = ".Artifacts." + architecture + ".";
        string suffix = "." + fileName;
        foreach (string candidate in orderedResourceNames)
        {
            if (candidate.IndexOf(architectureMarker, StringComparison.Ordinal) >= 0 &&
                candidate.EndsWith(suffix, StringComparison.Ordinal))
                return candidate;
        }
        return null;
    }

    private static void AddEmbeddedArtifact(
        IDictionary<string, EmbeddedArtifact> artifacts,
        string identity,
        EmbeddedArtifact artifact)
    {
        if (artifacts.TryGetValue(identity, out EmbeddedArtifact? existing))
        {
            if (!string.Equals(existing.CubinSha256, artifact.CubinSha256,
                    StringComparison.OrdinalIgnoreCase) ||
                !string.Equals(existing.LinkerLogSha256, artifact.LinkerLogSha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Embedded direct-PTX manifests disagree for identity " + identity + ".");
            return;
        }
        artifacts.Add(identity, artifact);
    }

    private sealed record EmbeddedArtifact(
        string CubinSha256,
        string ResourceName,
        string? LinkerLogSha256,
        string? LinkerResourceName);

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
