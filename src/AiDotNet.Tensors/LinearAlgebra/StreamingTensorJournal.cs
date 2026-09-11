// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>Opaque, element-type-safe handle for one losslessly journaled tensor.</summary>
public sealed class StreamingTensorJournalRecord<T>
{
    private readonly int[] _shape;
    private readonly IReadOnlyList<int> _publicShape;

    internal StreamingTensorJournalRecord(long ownerId, JournalSegment[] segments, int[] shape, int elementCount)
    {
        OwnerId = ownerId;
        Segments = segments;
        _shape = shape;
        _publicShape = Array.AsReadOnly(_shape);
        ElementCount = elementCount;
    }

    internal long OwnerId { get; }
    internal JournalSegment[] Segments { get; }
    internal int[] ShapeInternal => _shape;

    /// <summary>Shape captured when the tensor was appended.</summary>
    public IReadOnlyList<int> Shape => _publicShape;

    /// <summary>Number of tensor elements in this record.</summary>
    public int ElementCount { get; }
}

/// <summary>Bounded tensor-payload and backing-store telemetry for a tensor journal.</summary>
public sealed class StreamingTensorJournalReport
{
    /// <summary>
    /// Configured hard cap for resident tensor payload plus the reusable typed replay tensor.
    /// Runtime object metadata and framework-owned file buffers are intentionally excluded.
    /// </summary>
    public long MaxResidentBytes { get; init; }

    /// <summary>Current resident tensor payload plus reusable typed replay-tensor bytes.</summary>
    public long ResidentBytes { get; init; }

    /// <summary>Maximum resident tensor-payload and typed replay-tensor bytes observed.</summary>
    public long ResidentBytesPeak { get; init; }

    /// <summary>Current allocated length of the reusable backing file.</summary>
    public long BackingStoreBytes { get; init; }

    /// <summary>Number of live tensor records.</summary>
    public int LiveRecordCount { get; init; }

    /// <summary>Total bytes written to the backing store.</summary>
    public long DiskWriteBytes { get; init; }

    /// <summary>Total bytes read from the backing store.</summary>
    public long DiskReadBytes { get; init; }
}

/// <summary>
/// Consumes one ephemeral, typed replay chunk. Only the first <paramref name="elementCount"/>
/// values are part of the record. The callback is synchronous: all CPU and device work that reads
/// the buffer must finish before it returns. The buffer is owned and reused by the journal and must
/// not be retained after the callback returns.
/// </summary>
public delegate void StreamingTensorJournalChunkConsumer<T>(
    int elementOffset,
    Tensor<T> buffer,
    int elementCount);

internal sealed class JournalSegment
{
    public byte[]? ResidentData { get; set; }
    public long FileOffset { get; set; } = -1;
    public int ByteCount { get; set; }
}

internal enum StreamingTensorJournalLifecycle
{
    Active,
    Poisoned,
    Disposed,
}

internal enum StreamingTensorJournalFileOperation
{
    Write,
    ReleaseRange,
}

/// <summary>A lossless tensor journal with hard-bounded resident tensor payload.</summary>
/// <remarks>
/// The configured budget covers the reusable typed replay tensor and resident record payload.
/// It intentionally excludes fixed runtime overhead such as object metadata, collection capacity,
/// and framework-owned file buffers; record metadata grows with the number of live records.
/// Excess tensor payload is written to a private reusable backing file.
/// Records are replayed from the exact native bytes that were appended. Removed file ranges are
/// reused and trailing free ranges truncate the file, so repeated append/remove cycles do not
/// grow disk usage without bound. All operations are serialized per journal.
///
/// <para>Float, double, and signed 64-bit integer tensors are supported. Inputs and restore destinations must be contiguous;
/// contiguous offset views are supported, while non-contiguous and sparse layouts are rejected
/// explicitly instead of allocating a hidden full-tensor copy.</para>
/// </remarks>
public sealed class StreamingTensorJournal<T> : IDisposable
{
    private const int DefaultIoBufferBytes = 1024 * 1024;
    private const string BackingDirectoryPrefix = "aidotnet-tensor-journal-";
    private static long _nextOwnerId;

    private readonly object _gate = new();
    private readonly long _ownerId = Interlocked.Increment(ref _nextOwnerId);
    private readonly long _maxResidentBytes;
    private readonly long _payloadResidentBudget;
    private Tensor<T>? _replayBuffer;
    private readonly int _elementSize;
    private readonly int _replayBufferBytes;
    private readonly string _backingDirectory;
    private readonly string _backingFilePath;
    private readonly HashSet<StreamingTensorJournalRecord<T>> _liveRecords = new();
    private readonly List<FreeFileRange> _freeFileRanges = new();
    // Per-instance, internal fault seam; production journals do not allocate a callback.
    private readonly Action<StreamingTensorJournalFileOperation>? _beforeFileOperation;
#if NETFRAMEWORK
    // FileStream on .NET Framework has no Span-based API. One reusable bounded adapter avoids the
    // previous byte-at-a-time virtual calls without allocating per record.
    private readonly byte[] _frameworkIoBuffer;
#endif

    private FileStream? _backingFile;
    private long _residentPayloadBytes;
    private long _residentBytesPeak;
    private long _backingLength;
    private long _diskWriteBytes;
    private long _diskReadBytes;
    private int _replayInProgress;
    private StreamingTensorJournalLifecycle _lifecycle = StreamingTensorJournalLifecycle.Active;

    ~StreamingTensorJournal()
    {
        Dispose();
    }

    /// <summary>
    /// Creates a journal whose resident tensor payload and typed replay tensor are hard-bounded.
    /// </summary>
    public StreamingTensorJournal(long maxResidentBytes, string? backingStorePath = null)
    {
        if (maxResidentBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxResidentBytes),
                "The journal resident-byte budget must be positive.");
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double) && typeof(T) != typeof(long))
            throw new NotSupportedException(
                $"Streaming tensor journals support float, double, and Int64, not {typeof(T).Name}.");

        _elementSize = typeof(T) == typeof(float) ? sizeof(float) : sizeof(long);
        if (maxResidentBytes < _elementSize)
            throw new ArgumentOutOfRangeException(nameof(maxResidentBytes),
                $"The journal budget must hold at least one {typeof(T).Name} value ({_elementSize} bytes).");

        _maxResidentBytes = maxResidentBytes;
        // Reserve at most half the budget for replay so ordinary small records can still remain
        // resident. A one-element budget necessarily dedicates that element to replay and spills
        // records, which is the only way to preserve the hard journal-owned payload bound.
        long replayBudget = maxResidentBytes >= 2L * _elementSize
            ? maxResidentBytes / 2
            : maxResidentBytes;
        _replayBufferBytes = checked((int)Math.Min(DefaultIoBufferBytes, replayBudget));
        _replayBufferBytes -= _replayBufferBytes % _elementSize;
        if (_replayBufferBytes == 0) _replayBufferBytes = _elementSize;
        _replayBuffer = new Tensor<T>(new[] { _replayBufferBytes / _elementSize });
        _payloadResidentBudget = maxResidentBytes - _replayBufferBytes;
        _residentBytesPeak = _replayBufferBytes;
#if NETFRAMEWORK
        // Span-based FileStream APIs are unavailable on net471. Keep one fixed-size adapter buffer
        // independent of the tensor payload budget: tying it to a valid four-byte budget degraded a
        // multi-megabyte spill into one managed FileStream call per float.
        _frameworkIoBuffer = new byte[DefaultIoBufferBytes];
#endif

        string root = string.IsNullOrWhiteSpace(backingStorePath)
            ? Path.GetTempPath()
            : Path.GetFullPath(backingStorePath);
        Directory.CreateDirectory(root);
        _backingDirectory = Path.Combine(root, BackingDirectoryPrefix + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_backingDirectory);
        _backingFilePath = Path.Combine(_backingDirectory, "journal.bin");
    }

    internal StreamingTensorJournal(
        long maxResidentBytes,
        string? backingStorePath,
        Action<StreamingTensorJournalFileOperation> beforeFileOperation)
        : this(maxResidentBytes, backingStorePath)
    {
        _beforeFileOperation = beforeFileOperation ?? throw new ArgumentNullException(nameof(beforeFileOperation));
    }

    /// <summary>Number of live journal records.</summary>
    public int Count
    {
        get
        {
            ThrowIfReplayInProgress();
            lock (_gate)
            {
                ThrowIfReplayInProgress();
                ThrowIfDisposed();
                return _liveRecords.Count;
            }
        }
    }

    /// <summary>Current resident tensor payload plus reusable typed replay-tensor bytes.</summary>
    public long ResidentBytes
    {
        get
        {
            ThrowIfReplayInProgress();
            lock (_gate)
            {
                ThrowIfReplayInProgress();
                ThrowIfDisposed();
                return CurrentResidentBytes;
            }
        }
    }

    /// <summary>Returns resident-memory and backing-store telemetry.</summary>
    public StreamingTensorJournalReport GetReport()
    {
        ThrowIfReplayInProgress();
        lock (_gate)
        {
            ThrowIfReplayInProgress();
            ThrowIfDisposed();
            return new StreamingTensorJournalReport
            {
                MaxResidentBytes = _maxResidentBytes,
                ResidentBytes = CurrentResidentBytes,
                ResidentBytesPeak = _residentBytesPeak,
                BackingStoreBytes = _backingLength,
                LiveRecordCount = _liveRecords.Count,
                DiskWriteBytes = _diskWriteBytes,
                DiskReadBytes = _diskReadBytes,
            };
        }
    }

    /// <summary>Appends one contiguous tensor losslessly and returns an opaque typed handle.</summary>
    public StreamingTensorJournalRecord<T> Append(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        ThrowIfReplayInProgress();
        lock (_gate)
        {
            ThrowIfReplayInProgress();
            ThrowIfDisposed();
            if (!tensor.IsContiguous)
                throw new NotSupportedException(
                    "Streaming tensor journals require a contiguous tensor. Call Contiguous() explicitly first.");

            if (tensor.Length > int.MaxValue / _elementSize)
                throw new NotSupportedException(
                    "A journal record cannot exceed Int32.MaxValue bytes. Chunk the tensor before appending.");
            ReadOnlySpan<byte> sourceBytes = GetReadOnlyBytes(tensor);
            // One tensor owns at most one storage extent. Segmenting by the resident budget made
            // metadata grow as O(tensorBytes / budget), which was unbounded and catastrophic for a
            // four-byte budget. Chunking belongs to replay, where one reusable bounded buffer is
            // sufficient; storage itself is one resident payload or one contiguous file range.
            int segmentCount = sourceBytes.Length == 0 ? 0 : 1;
            var segments = new JournalSegment[segmentCount];
            int completedSegments = 0;
            try
            {
                if (segmentCount != 0)
                {
                    var segment = new JournalSegment { ByteCount = sourceBytes.Length };
                    segments[0] = segment;

                    if (_residentPayloadBytes + sourceBytes.Length <= _payloadResidentBudget)
                    {
                        byte[] resident = new byte[sourceBytes.Length];
                        sourceBytes.CopyTo(resident);
                        segment.ResidentData = resident;
                        _residentPayloadBytes += sourceBytes.Length;
                        UpdateResidentPeak();
                    }
                    else
                    {
                        long fileOffset = AllocateFileRange(sourceBytes.Length);
                        try
                        {
                            WriteFileRange(fileOffset, sourceBytes);
                            segment.FileOffset = fileOffset;
                        }
                        catch
                        {
                            try
                            {
                                ReleaseFileRange(fileOffset, sourceBytes.Length);
                            }
                            catch
                            {
                                PoisonAndReleaseAllRecords();
                            }
                            throw;
                        }
                    }
                    completedSegments = 1;
                }

                var record = new StreamingTensorJournalRecord<T>(
                    _ownerId,
                    segments,
                    (int[])tensor._shape.Clone(),
                    tensor.Length);
                _liveRecords.Add(record);
                return record;
            }
            catch
            {
                try
                {
                    for (int i = 0; i < completedSegments; i++) ReleaseSegment(segments[i]);
                }
                catch
                {
                    PoisonAndReleaseAllRecords();
                }
                throw;
            }
        }
    }

    /// <summary>
    /// Replays a record through one reusable resident buffer without materializing the complete
    /// tensor. The callback is synchronous: all work consuming a chunk, including queued device
    /// work, must complete before the callback returns. Reentrant or concurrent journal operations
    /// are rejected until replay completes, and the supplied buffer must not be retained.
    /// </summary>
    public void Replay(
        StreamingTensorJournalRecord<T> record,
        StreamingTensorJournalChunkConsumer<T> consume)
    {
        if (consume is null) throw new ArgumentNullException(nameof(consume));
        ThrowIfReplayInProgress();
        lock (_gate)
        {
            ThrowIfReplayInProgress();
            Validate(record);
            if (Interlocked.CompareExchange(ref _replayInProgress, 1, 0) != 0)
                throw new InvalidOperationException("A tensor-journal replay is already in progress.");
        }

        try
        {
            Tensor<T> replayBuffer = _replayBuffer
                ?? throw new InvalidOperationException("The replay buffer is unavailable.");
            int capacity = replayBuffer.Length;
            int elementOffset = 0;
            while (elementOffset < record.ElementCount)
            {
                int elementCount = Math.Min(capacity, record.ElementCount - elementOffset);
                int byteCount = checked(elementCount * _elementSize);
                ReadRecordRangeInto(
                    record,
                    checked(elementOffset * _elementSize),
                    GetWritableBytes(replayBuffer).Slice(0, byteCount));
                replayBuffer.IncrementVersion();
                consume(elementOffset, replayBuffer, elementCount);
                elementOffset += elementCount;
            }
        }
        finally
        {
            Volatile.Write(ref _replayInProgress, 0);
        }
    }

    /// <summary>
    /// Materializes a fresh tensor containing the exact journaled values. This convenience API
    /// allocates memory proportional to the complete record and is not a bounded replay operation.
    /// </summary>
    public Tensor<T> Read(StreamingTensorJournalRecord<T> record)
    {
        ThrowIfReplayInProgress();
        lock (_gate)
        {
            ThrowIfReplayInProgress();
            Validate(record);
            var tensor = new Tensor<T>((int[])record.ShapeInternal.Clone());
            ReadRecordInto(record, tensor);
            tensor.IncrementVersion();
            return tensor;
        }
    }

    /// <summary>
    /// Atomically restores the journaled value into an existing same-shaped contiguous tensor.
    /// The destination is not mutated unless the complete record has first been read successfully.
    /// Atomic destination replacement requires a complete O(record-size) staging tensor; callers
    /// that need bounded consumption should use <see cref="Replay"/> instead.
    /// </summary>
    public void Restore(StreamingTensorJournalRecord<T> record, Tensor<T> destination)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        ThrowIfReplayInProgress();
        lock (_gate)
        {
            ThrowIfReplayInProgress();
            Validate(record);
            ValidateDestination(record, destination);
            var staging = new Tensor<T>((int[])record.ShapeInternal.Clone());
            ReadRecordInto(record, staging);
            staging.ReadOnlyData.Span.CopyTo(destination.Data.Span);
            destination.IncrementVersion();
        }
    }

    /// <summary>Removes a record and immediately makes its resident and disk ranges reusable.</summary>
    public void Remove(StreamingTensorJournalRecord<T> record)
    {
        ThrowIfReplayInProgress();
        lock (_gate)
        {
            ThrowIfReplayInProgress();
            Validate(record);
            for (int i = 0; i < record.Segments.Length; i++) ReleaseSegment(record.Segments[i]);
            _liveRecords.Remove(record);
        }
    }

    private static ReadOnlySpan<byte> GetReadOnlyBytes(Tensor<T> tensor)
    {
        if (typeof(T) == typeof(float))
        {
            var typed = (Tensor<float>)(object)tensor;
            return MemoryMarshal.AsBytes(typed.ReadOnlyData.Span);
        }

        if (typeof(T) == typeof(double))
        {
            var doubleTensor = (Tensor<double>)(object)tensor;
            return MemoryMarshal.AsBytes(doubleTensor.ReadOnlyData.Span);
        }

        var longTensor = (Tensor<long>)(object)tensor;
        return MemoryMarshal.AsBytes(longTensor.ReadOnlyData.Span);
    }

    private void ReadRecordInto(StreamingTensorJournalRecord<T> record, Tensor<T> destination)
    {
        ReadRecordRangeInto(record, 0, GetWritableBytes(destination));
    }

    private void ReadRecordRangeInto(
        StreamingTensorJournalRecord<T> record,
        int sourceByteOffset,
        Span<byte> destination)
    {
        if (destination.Length == 0) return;
        int remainingOffset = sourceByteOffset;
        int destinationOffset = 0;
        for (int i = 0; i < record.Segments.Length && destinationOffset < destination.Length; i++)
        {
            JournalSegment segment = record.Segments[i];
            if (remainingOffset >= segment.ByteCount)
            {
                remainingOffset -= segment.ByteCount;
                continue;
            }

            int count = Math.Min(segment.ByteCount - remainingOffset, destination.Length - destinationOffset);
            Span<byte> target = destination.Slice(destinationOffset, count);
            if (segment.ResidentData is not null)
            {
                segment.ResidentData.AsSpan(remainingOffset, count).CopyTo(target);
            }
            else
            {
                ReadFileRange(segment.FileOffset + remainingOffset, target);
            }
            destinationOffset += count;
            remainingOffset = 0;
        }

        if (destinationOffset != destination.Length)
            throw new InvalidOperationException("Streaming tensor journal record ended before the requested range.");
    }

    private static Span<byte> GetWritableBytes(Tensor<T> tensor)
    {
        if (typeof(T) == typeof(float))
            return MemoryMarshal.AsBytes(((Tensor<float>)(object)tensor).Data.Span);

        if (typeof(T) == typeof(double))
            return MemoryMarshal.AsBytes(((Tensor<double>)(object)tensor).Data.Span);

        return MemoryMarshal.AsBytes(((Tensor<long>)(object)tensor).Data.Span);
    }

    private static void ValidateDestination(
        StreamingTensorJournalRecord<T> record,
        Tensor<T> destination)
    {
        if (!destination.IsContiguous)
            throw new NotSupportedException(
                "Streaming tensor journal restore requires a contiguous destination.");
        if (destination.Length != record.ElementCount)
            throw new ArgumentException(
                $"Destination length {destination.Length} does not match journal record length " +
                $"{record.ElementCount}.", nameof(destination));
        if (destination.Rank != record.ShapeInternal.Length)
            throw new ArgumentException("Destination rank does not match the journal record.", nameof(destination));
        for (int i = 0; i < destination.Rank; i++)
        {
            if (destination._shape[i] != record.ShapeInternal[i])
                throw new ArgumentException("Destination shape does not match the journal record.", nameof(destination));
        }
    }

    private void Validate(StreamingTensorJournalRecord<T> record)
    {
        ThrowIfDisposed();
        if (record is null) throw new ArgumentNullException(nameof(record));
        if (record.OwnerId != _ownerId)
            throw new ArgumentException("The record belongs to a different tensor journal.", nameof(record));
        if (!_liveRecords.Contains(record))
            throw new InvalidOperationException("The tensor journal record has already been removed.");
    }

    private FileStream BackingFile() => _backingFile ??= new FileStream(
        _backingFilePath,
        FileMode.OpenOrCreate,
        FileAccess.ReadWrite,
        FileShare.Read,
        4096,
        FileOptions.DeleteOnClose);

    private long AllocateFileRange(int byteCount)
    {
        for (int i = 0; i < _freeFileRanges.Count; i++)
        {
            FreeFileRange range = _freeFileRanges[i];
            if (range.Length < byteCount) continue;
            long offset = range.Offset;
            if (range.Length == byteCount)
            {
                _freeFileRanges.RemoveAt(i);
            }
            else
            {
                _freeFileRanges[i] = new FreeFileRange(
                    range.Offset + byteCount,
                    range.Length - byteCount);
            }
            return offset;
        }

        long appendedOffset = _backingLength;
        _backingLength = checked(_backingLength + byteCount);
        return appendedOffset;
    }

    private void WriteFileRange(long fileOffset, ReadOnlySpan<byte> source)
    {
        _beforeFileOperation?.Invoke(StreamingTensorJournalFileOperation.Write);
        FileStream stream = BackingFile();
        stream.Seek(fileOffset, SeekOrigin.Begin);
#if NETFRAMEWORK
        int written = 0;
        while (written < source.Length)
        {
            int count = Math.Min(_frameworkIoBuffer.Length, source.Length - written);
            source.Slice(written, count).CopyTo(_frameworkIoBuffer);
            stream.Write(_frameworkIoBuffer, 0, count);
            written += count;
        }
#else
        stream.Write(source);
#endif
        _diskWriteBytes += source.Length;
    }

    private void ReadFileRange(long fileOffset, Span<byte> destination)
    {
        FileStream stream = BackingFile();
        stream.Seek(fileOffset, SeekOrigin.Begin);
        int read = 0;
        while (read < destination.Length)
        {
#if NETFRAMEWORK
            int requested = Math.Min(_frameworkIoBuffer.Length, destination.Length - read);
            int count = stream.Read(_frameworkIoBuffer, 0, requested);
            if (count <= 0)
                throw new InvalidOperationException(
                    $"Streaming tensor journal encountered a short read at offset {fileOffset + read}.");
            _frameworkIoBuffer.AsSpan(0, count).CopyTo(destination.Slice(read, count));
            read += count;
#else
            int count = stream.Read(destination.Slice(read));
            if (count <= 0)
                throw new InvalidOperationException(
                    $"Streaming tensor journal encountered a short read at offset {fileOffset + read}.");
            read += count;
#endif
        }
        _diskReadBytes += destination.Length;
    }

    private void ReleaseSegment(JournalSegment segment)
    {
        if (segment.ResidentData is not null)
        {
            _residentPayloadBytes -= segment.ResidentData.Length;
            segment.ResidentData = null;
        }
        else if (segment.FileOffset >= 0)
        {
            ReleaseFileRange(segment.FileOffset, segment.ByteCount);
            segment.FileOffset = -1;
        }
    }

    private void ReleaseFileRange(long offset, long length)
    {
        if (offset < 0 || length <= 0) return;
        _beforeFileOperation?.Invoke(StreamingTensorJournalFileOperation.ReleaseRange);
        int insertion = 0;
        while (insertion < _freeFileRanges.Count && _freeFileRanges[insertion].Offset < offset)
            insertion++;
        _freeFileRanges.Insert(insertion, new FreeFileRange(offset, length));

        for (int i = Math.Max(0, insertion - 1); i + 1 < _freeFileRanges.Count;)
        {
            FreeFileRange current = _freeFileRanges[i];
            FreeFileRange next = _freeFileRanges[i + 1];
            if (current.Offset + current.Length < next.Offset)
            {
                i++;
                continue;
            }
            long end = Math.Max(current.Offset + current.Length, next.Offset + next.Length);
            _freeFileRanges[i] = new FreeFileRange(current.Offset, end - current.Offset);
            _freeFileRanges.RemoveAt(i + 1);
        }

        long truncatedLength = _backingLength;
        int firstTrimmedRange = _freeFileRanges.Count;
        for (int i = _freeFileRanges.Count - 1; i >= 0; i--)
        {
            FreeFileRange range = _freeFileRanges[i];
            if (range.Offset + range.Length != truncatedLength) break;
            truncatedLength = range.Offset;
            firstTrimmedRange = i;
        }

        if (truncatedLength == _backingLength) return;
        if (_backingFile is not null && _backingFile.Length != truncatedLength)
        {
            try
            {
                _backingFile.SetLength(truncatedLength);
            }
            catch
            {
                // The free range remains reusable even when the operating system cannot physically
                // truncate the file. Do not publish a shorter logical length or corrupt a live record.
                return;
            }
        }

        _backingLength = truncatedLength;
        if (firstTrimmedRange < _freeFileRanges.Count)
            _freeFileRanges.RemoveRange(
                firstTrimmedRange,
                _freeFileRanges.Count - firstTrimmedRange);
    }

    private long CurrentResidentBytes => checked(_replayBufferBytes + _residentPayloadBytes);

    private void UpdateResidentPeak()
    {
        long current = CurrentResidentBytes;
        if (current > _maxResidentBytes)
            throw new InvalidOperationException("Streaming tensor journal exceeded its resident-byte budget.");
        if (current > _residentBytesPeak) _residentBytesPeak = current;
    }

    private void ThrowIfReplayInProgress()
    {
        if (Volatile.Read(ref _replayInProgress) != 0)
            throw new InvalidOperationException(
                "Tensor-journal operations are not allowed while a replay callback is active.");
    }

    private void ThrowIfDisposed()
    {
        if (_lifecycle == StreamingTensorJournalLifecycle.Disposed)
            throw new ObjectDisposedException(nameof(StreamingTensorJournal<T>));
        if (_lifecycle == StreamingTensorJournalLifecycle.Poisoned)
            throw new InvalidOperationException(
                "The streaming tensor journal is unavailable after an append cleanup failure.");
    }

    private void PoisonAndReleaseAllRecords()
    {
        _lifecycle = StreamingTensorJournalLifecycle.Poisoned;
        foreach (StreamingTensorJournalRecord<T> record in _liveRecords)
        {
            for (int i = 0; i < record.Segments.Length; i++)
            {
                record.Segments[i].ResidentData = null;
                record.Segments[i].FileOffset = -1;
            }
        }
        _liveRecords.Clear();
        _freeFileRanges.Clear();
        _residentPayloadBytes = 0;
        _replayBuffer = null;
        try { _backingFile?.Dispose(); } catch { }
        _backingFile = null;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        ThrowIfReplayInProgress();
        bool directoryRemoved;
        lock (_gate)
        {
            ThrowIfReplayInProgress();
            if (_lifecycle != StreamingTensorJournalLifecycle.Disposed)
            {
                foreach (StreamingTensorJournalRecord<T> record in _liveRecords)
                {
                    for (int i = 0; i < record.Segments.Length; i++)
                    {
                        record.Segments[i].ResidentData = null;
                        record.Segments[i].FileOffset = -1;
                    }
                }
                _lifecycle = StreamingTensorJournalLifecycle.Disposed;
                _liveRecords.Clear();
                _freeFileRanges.Clear();
                _residentPayloadBytes = 0;
                _replayBuffer = null;
                try { _backingFile?.Dispose(); } catch { }
                _backingFile = null;
            }
        }
        directoryRemoved = TryRemoveBackingDirectory();
        // If the OS temporarily refuses directory deletion, leave the finalizer registered so it
        // can make one later best-effort retry after this instance becomes unreachable.
        if (directoryRemoved) GC.SuppressFinalize(this);
    }

    private bool TryRemoveBackingDirectory()
    {
        try
        {
            if (Directory.Exists(_backingDirectory))
                Directory.Delete(_backingDirectory, recursive: true);
            return !Directory.Exists(_backingDirectory);
        }
        catch
        {
            // DeleteOnClose removes the data file even when a best-effort directory cleanup fails.
            return false;
        }
    }

    private readonly struct FreeFileRange
    {
        public FreeFileRange(long offset, long length)
        {
            Offset = offset;
            Length = length;
        }

        public long Offset { get; }
        public long Length { get; }
    }
}
