using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Layer 5 of the BlasManaged allocator stack: carves sub-spans out of a
/// caller-supplied <see cref="BlasOptions{T}.Workspace"/> byte buffer.
/// Provides a zero-allocation path for benchmarks, inference servers, and
/// AOT-published binaries.
///
/// <para>
/// The carver maintains a monotonic cursor; successive
/// <see cref="TryCarve"/> calls return non-overlapping sub-spans. When the
/// workspace can't satisfy a request (cursor would exceed length), returns
/// <see cref="Span{Byte}.Empty"/> — the caller then falls through to a
/// lower allocator layer.
/// </para>
///
/// <para>
/// As a <c>ref struct</c> the carver can hold a <see cref="Span{Byte}"/>
/// directly but must remain on the stack — appropriate for the scope of a
/// single Gemm call.
/// </para>
/// </summary>
internal ref struct WorkspaceCarver
{
    private readonly Span<byte> _workspace;
    private int _offset;

    public WorkspaceCarver(Span<byte> workspace)
    {
        _workspace = workspace;
        _offset = 0;
    }

    /// <summary>
    /// Try to carve a <paramref name="bytes"/>-byte sub-span out of the
    /// remaining workspace. Returns empty if not enough remains, OR if the
    /// caller passed a non-positive byte count.
    /// </summary>
    public Span<byte> TryCarve(int bytes)
    {
        if (bytes <= 0) return Span<byte>.Empty;
        int remaining = _workspace.Length - _offset;
        if (remaining < bytes) return Span<byte>.Empty;
        var slice = _workspace.Slice(_offset, bytes);
        _offset += bytes;
        return slice;
    }

    /// <summary>Bytes still available in the workspace.</summary>
    public int RemainingBytes => _workspace.Length - _offset;

    /// <summary>Total workspace size (constant for the carver's lifetime).</summary>
    public int TotalBytes => _workspace.Length;

    /// <summary>True if the carver has a non-empty workspace.</summary>
    public bool HasWorkspace => _workspace.Length > 0;
}
