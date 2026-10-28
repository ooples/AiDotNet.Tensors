using System;

namespace AiDotNet.Tensors.Interop;

/// <summary>
/// The consumer's stream for a DLPack export: the <c>stream</c> argument of the array API's
/// <c>__dlpack__(stream=...)</c>.
/// </summary>
/// <remarks>
/// <para>For a CUDA tensor the producer must make the data safe to use on the consumer's stream before handing it
/// over. AiDotNet.Tensors records an event on the stream that produced the data and makes the consumer's stream wait
/// on it (a device-side wait; the host does not block). <see cref="NoSync"/> skips that, for a consumer that
/// synchronizes on its own.</para>
/// <para>CPU tensors need no stream; every value is accepted and ignored for them.</para>
/// <para><c>default(DLPackStream)</c> is <see cref="Default"/>: the array API's <c>stream=None</c>, which for CUDA
/// means the legacy default stream.</para>
/// </remarks>
public readonly struct DLPackStream : IEquatable<DLPackStream>
{
    private readonly DLPackStreamKind _kind;
    private readonly IntPtr _handle;

    private DLPackStream(DLPackStreamKind kind, IntPtr handle)
    {
        _kind = kind;
        _handle = handle;
    }

    /// <summary>No stream given (<c>stream=None</c>): for CUDA, the legacy default stream.</summary>
    public static DLPackStream Default => default;

    /// <summary>The consumer synchronizes itself (<c>stream=-1</c>); the producer adds no ordering.</summary>
    public static DLPackStream NoSync => new(DLPackStreamKind.NoSync, IntPtr.Zero);

    /// <summary>The CUDA legacy default stream (<c>stream=1</c>, <c>CU_STREAM_LEGACY</c>).</summary>
    public static DLPackStream LegacyDefault => new(DLPackStreamKind.LegacyDefault, IntPtr.Zero);

    /// <summary>The CUDA per-thread default stream (<c>stream=2</c>, <c>CU_STREAM_PER_THREAD</c>).</summary>
    public static DLPackStream PerThreadDefault => new(DLPackStreamKind.PerThreadDefault, IntPtr.Zero);

    /// <summary>
    /// A specific stream: a <c>cudaStream_t</c> / <c>CUstream</c> handle, such as
    /// <c>torch.cuda.current_stream().cuda_stream</c>.
    /// </summary>
    /// <exception cref="ArgumentException">The handle is 0, 1 or 2: the array API reserves 0 as ambiguous, and 1 and
    /// 2 are <see cref="LegacyDefault"/> and <see cref="PerThreadDefault"/>.</exception>
    public static DLPackStream FromHandle(IntPtr streamHandle)
    {
        long value = streamHandle.ToInt64();
        if (value == 0)
            throw new ArgumentException(
                "Stream 0 is ambiguous under the DLPack array API; pass LegacyDefault or PerThreadDefault instead.",
                nameof(streamHandle));
        if (value == 1) return LegacyDefault;
        if (value == 2) return PerThreadDefault;
        return new DLPackStream(DLPackStreamKind.Handle, streamHandle);
    }

    /// <summary>True when the producer must not add any stream ordering.</summary>
    public bool IsNoSync => _kind == DLPackStreamKind.NoSync;

    /// <summary>The CUDA stream handle the consumer's work is ordered on, or <see cref="IntPtr.Zero"/> for
    /// <see cref="NoSync"/>.</summary>
    internal IntPtr CudaStreamHandle => _kind switch
    {
        DLPackStreamKind.NoSync => IntPtr.Zero,
        DLPackStreamKind.PerThreadDefault => DLPackNative.CudaPerThreadDefaultStream,
        DLPackStreamKind.Handle => _handle,
        _ => DLPackNative.CudaLegacyDefaultStream,   // Default and LegacyDefault
    };

    /// <inheritdoc/>
    public bool Equals(DLPackStream other) => CudaStreamHandle == other.CudaStreamHandle && IsNoSync == other.IsNoSync;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is DLPackStream other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode() => CudaStreamHandle.GetHashCode() ^ (IsNoSync ? 1 : 0);

    /// <inheritdoc/>
    public override string ToString() => _kind switch
    {
        DLPackStreamKind.NoSync => "NoSync",
        DLPackStreamKind.LegacyDefault => "LegacyDefault",
        DLPackStreamKind.PerThreadDefault => "PerThreadDefault",
        DLPackStreamKind.Handle => "0x" + _handle.ToInt64().ToString("x"),
        _ => "Default",
    };

    /// <summary>Equality.</summary>
    public static bool operator ==(DLPackStream left, DLPackStream right) => left.Equals(right);

    /// <summary>Inequality.</summary>
    public static bool operator !=(DLPackStream left, DLPackStream right) => !left.Equals(right);

    private enum DLPackStreamKind
    {
        Default = 0,
        NoSync,
        LegacyDefault,
        PerThreadDefault,
        Handle,
    }
}
