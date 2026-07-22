using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

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
    private const int PipelineVersion = 1;
    private const int PtxInputType = 1; // CU_JIT_INPUT_PTX
    private const int LogBytes = 16 * 1024;
    private const string CacheEnvironmentVariable = "AIDOTNET_DIRECT_PTX_CACHE_PATH";
    private static readonly object Sync = new();
    private static readonly Lazy<IReadOnlyDictionary<string, string>> EmbeddedHashes =
        new(ReadEmbeddedHashes);

    internal static DirectPtxCubinArtifact Resolve(DirectPtxRuntime runtime, string ptx)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        PtxCompat.ThrowIfNullOrWhiteSpace(ptx, nameof(ptx));
        string sourceKey = ComputeSourceKey(
            ptx, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);

        lock (Sync)
        {
            DirectPtxCubinArtifact? embedded = TryReadEmbedded(runtime, sourceKey);
            if (embedded != null)
                return embedded;

            string? cachePath = GetCachePath(runtime, sourceKey);
            DirectPtxCubinArtifact? cached = TryReadDisk(cachePath, sourceKey);
            if (cached != null)
                return cached;

            DirectPtxCubinArtifact compiled = Compile(runtime, ptx, sourceKey, cachePath);
            TryWriteDisk(compiled, runtime);
            return compiled;
        }
    }

    internal static string ComputeSourceKey(string ptx, int major, int minor)
    {
        string identity = "direct-ptx-cubin-v" +
            PipelineVersion.ToString(CultureInfo.InvariantCulture) + "|sm" +
            major.ToString(CultureInfo.InvariantCulture) +
            minor.ToString(CultureInfo.InvariantCulture) + "|" + ptx;
        return Sha256(Encoding.UTF8.GetBytes(identity));
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
        if (!EmbeddedHashes.Value.TryGetValue(manifestKey, out string? expectedHash) ||
            !string.Equals(expectedHash, cubinHash, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException(
                "Embedded direct-PTX cubin failed its release-manifest hash: " + resourceName);
        return new DirectPtxCubinArtifact(
            image, sourceKey, cubinHash, DirectPtxModuleImageKind.EmbeddedCubin,
            resourceName, "precompiled package cubin");
    }

    private static IReadOnlyDictionary<string, string> ReadEmbeddedHashes()
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        Assembly assembly = typeof(DirectPtxCubinArtifactCache).Assembly;
        foreach (string resourceName in assembly.GetManifestResourceNames())
        {
            const string suffix = ".normalization-cubins.tsv";
            if (!resourceName.EndsWith(suffix, StringComparison.Ordinal))
                continue;
            string marker = ".Artifacts.sm";
            int markerIndex = resourceName.IndexOf(marker, StringComparison.Ordinal);
            int suffixIndex = resourceName.Length - suffix.Length;
            if (markerIndex < 0 || suffixIndex <= markerIndex + marker.Length)
                continue;
            string architecture = "sm" + resourceName.Substring(
                markerIndex + marker.Length, suffixIndex - markerIndex - marker.Length);
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
                if (columns.Length >= 4)
                    result[architecture + "|" + columns[2]] = columns[3];
            }
        }
        return result;
    }

    private static DirectPtxCubinArtifact? TryReadDisk(string? path, string sourceKey)
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
            return new DirectPtxCubinArtifact(
                image, sourceKey, cubinHash, DirectPtxModuleImageKind.DiskCacheCubin,
                path, "verified driver-link disk cache");
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static unsafe DirectPtxCubinArtifact Compile(
        DirectPtxRuntime runtime, string ptx, string sourceKey, string? cachePath)
    {
        using var _ = runtime.Enter();
        IntPtr infoLog = Marshal.AllocHGlobal(LogBytes);
        IntPtr errorLog = Marshal.AllocHGlobal(LogBytes);
        IntPtr ptxBuffer = IntPtr.Zero;
        IntPtr linkState = IntPtr.Zero;
        try
        {
            new Span<byte>((void*)infoLog, LogBytes).Clear();
            new Span<byte>((void*)errorLog, LogBytes).Clear();
            int[] options = [3, 4, 5, 6]; // CU_JIT_INFO/ERROR_LOG_BUFFER(_SIZE_BYTES)
            IntPtr[] values = [infoLog, (IntPtr)LogBytes, errorLog, (IntPtr)LogBytes];
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
            Marshal.FreeHGlobal(errorLog);
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
        try
        {
            string? directory = System.IO.Path.GetDirectoryName(path);
            if (string.IsNullOrWhiteSpace(directory))
                return;
            Directory.CreateDirectory(directory);
            string temporary = path + ".tmp-" + Guid.NewGuid().ToString("N");
            File.WriteAllBytes(temporary, artifact.Image);
            if (!File.Exists(path))
                File.Move(temporary, path);
            else
                File.Delete(temporary);
            File.WriteAllText(path + ".sha256", artifact.CubinSha256 + Environment.NewLine);
            File.WriteAllText(path + ".audit.txt",
                "pipeline-version=" + PipelineVersion.ToString(CultureInfo.InvariantCulture) + Environment.NewLine +
                "source-key=" + artifact.SourceKey + Environment.NewLine +
                "cubin-sha256=" + artifact.CubinSha256 + Environment.NewLine +
                "target=sm" + runtime.ComputeCapabilityMajor.ToString(CultureInfo.InvariantCulture) +
                runtime.ComputeCapabilityMinor.ToString(CultureInfo.InvariantCulture) + Environment.NewLine +
                "driver-version=" + runtime.DriverVersion.ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
        }
        catch (IOException)
        {
            // A read-only cache does not prevent use of the in-memory cubin.
        }
        catch (UnauthorizedAccessException)
        {
            // A read-only cache does not prevent use of the in-memory cubin.
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
}
