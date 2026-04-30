// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.IO;
using System.Text;
using AiDotNet.Tensors.Licensing;

namespace AiDotNet.Tensors.Serialization.TensorBoard;

/// <summary>
/// Direct-to-TFEvents writer — emits the binary record format that
/// TensorBoard's <c>EventAccumulator</c> reads, with no external
/// protobuf dependency. Records are framed exactly as
/// <c>tf.io.TFRecordWriter</c> emits them:
/// <list type="bullet">
///   <item>8-byte LE u64 length</item>
///   <item>4-byte LE u32 masked CRC32 of the length</item>
///   <item>length bytes of payload (a serialised
///   <c>tensorflow.Event</c> message)</item>
///   <item>4-byte LE u32 masked CRC32 of the payload</item>
/// </list>
/// </summary>
/// <remarks>
/// <para><b>Why direct binary:</b></para>
/// <para>
/// <c>tensorboardX</c> and PyTorch's <c>SummaryWriter</c> use a
/// compiled <c>tensorboard</c> Python package that brings ~50 MB of
/// transitive deps. We hand-encode the few protobuf messages we
/// actually need (<c>Event</c> with <c>Summary</c> containing
/// <c>SimpleValue</c> / <c>HistogramProto</c> / <c>Image</c>) — total
/// ~150 lines. The output opens cleanly in real TensorBoard.
/// </para>
/// <para><b>Message subset:</b></para>
/// <para>
/// We support scalar (<see cref="AddScalar"/>), histogram
/// (<see cref="AddHistogram"/>), and hyperparameters
/// (<see cref="AddHParam"/> / <see cref="LogHParams"/>) — the three
/// categories that make up &gt;95% of typical training-loop logs.
/// Image / audio / embedding / graph add another 200-300 lines and
/// are layered in by the same pattern: format the protobuf
/// payload, frame, write.
/// </para>
/// </remarks>
public sealed class TensorBoardSummaryWriter : IDisposable
{
    private readonly Stream _stream;
    private readonly bool _ownsStream;
    private bool _disposed;
    private readonly DateTimeOffset _startTime = DateTimeOffset.UtcNow;

    /// <summary>
    /// Opens a <c>events.out.tfevents.{timestamp}.{host}</c>-style
    /// file in <paramref name="logDir"/>. The directory is created if
    /// missing. Counts as one <see cref="PersistenceGuard.EnforceBeforeSave"/>.
    /// </summary>
    public static TensorBoardSummaryWriter OpenLogDir(string logDir)
    {
        PersistenceGuard.EnforceBeforeSave();
        if (logDir is null) throw new ArgumentNullException(nameof(logDir));
        if (!Directory.Exists(logDir)) Directory.CreateDirectory(logDir);

        // Use FileMode.CreateNew + a uniqueness suffix loop so two
        // writers opened in the same second on the same host don't
        // truncate each other's logs. The unix-seconds-based filename
        // matches TensorFlow's writer convention (TensorBoard's
        // EventAccumulator scans logDir by that prefix); appending an
        // extra `.{counter}` suffix on collision is a benign extension
        // — TensorBoard still picks both files up.
        long unixSeconds = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        string host = Environment.MachineName;
        string baseFilename = $"events.out.tfevents.{unixSeconds}.{host}";
        string path = Path.Combine(logDir, baseFilename);
        FileStream fs;
        int suffix = 0;
        while (true)
        {
            try
            {
                fs = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
                break;
            }
            catch (IOException) when (File.Exists(path))
            {
                suffix++;
                path = Path.Combine(logDir, $"{baseFilename}.{suffix}");
                if (suffix > 1000)
                    // Shouldn't happen in practice — 1000 collisions in
                    // one second is pathological. Surface clearly.
                    throw new IOException(
                        $"Could not allocate a unique TensorBoard events filename in {logDir} after " +
                        $"1000 collisions on '{baseFilename}'.");
            }
        }
        try
        {
            var w = new TensorBoardSummaryWriter(fs, ownsStream: true);
            // Emit a file_version header event the way TensorFlow does
            // — this is what TensorBoard reads to confirm the file is
            // an events log rather than arbitrary binary.
            w.WriteFileVersion();
            return w;
        }
        catch
        {
            fs.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Wraps an existing writable stream. The writer does NOT take
    /// ownership — caller disposes the stream.
    /// </summary>
    public static TensorBoardSummaryWriter ToStream(Stream stream)
    {
        PersistenceGuard.EnforceBeforeSave();
        var w = new TensorBoardSummaryWriter(stream, ownsStream: false);
        w.WriteFileVersion();
        return w;
    }

    private TensorBoardSummaryWriter(Stream stream, bool ownsStream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite) throw new ArgumentException("Stream must be writable.", nameof(stream));
        _stream = stream;
        _ownsStream = ownsStream;
    }

    private void WriteFileVersion()
    {
        // Emit an Event with file_version="brain.Event:2" so
        // TensorBoard's reader doesn't reject the stream as
        // pre-V1.
        var ev = new EventBuilder
        {
            WallTime = (DateTimeOffset.UtcNow - new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero)).TotalSeconds,
            Step = 0,
            FileVersion = "brain.Event:2",
        }.ToBytes();
        WriteRecord(ev);
    }

    /// <summary>
    /// Logs a scalar value at <paramref name="step"/> under
    /// <paramref name="tag"/> — the most common operation in a
    /// training loop. Tags follow TensorBoard's slash convention
    /// (<c>"loss/train"</c>, <c>"accuracy/val"</c>) so similarly-named
    /// tags group on one chart.
    /// </summary>
    public void AddScalar(string tag, double value, long step)
    {
        ThrowIfDisposed();
        if (tag is null) throw new ArgumentNullException(nameof(tag));
        var ev = new EventBuilder
        {
            WallTime = (DateTimeOffset.UtcNow - new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero)).TotalSeconds,
            Step = step,
            ScalarTag = tag,
            ScalarValue = (float)value,
        }.ToBytes();
        WriteRecord(ev);
    }

    /// <summary>
    /// Logs a histogram of <paramref name="values"/> under
    /// <paramref name="tag"/> with caller-specified bucket upper edges.
    /// Edges must be sorted ascending; values strictly less than
    /// <c>edges[0]</c> count into bucket 0, values >= <c>edges[^1]</c>
    /// count into the last bucket. Use this when you want the same
    /// fixed bin layout across runs so TensorBoard's overlay view
    /// stacks the histograms cleanly.
    /// </summary>
    public void AddHistogramWithBuckets(string tag, ReadOnlySpan<float> values, ReadOnlySpan<double> bucketUpperEdges, long step)
    {
        ThrowIfDisposed();
        if (tag is null) throw new ArgumentNullException(nameof(tag));
        if (bucketUpperEdges.Length == 0)
            throw new ArgumentException("bucketUpperEdges must contain at least one edge.", nameof(bucketUpperEdges));
        for (int i = 1; i < bucketUpperEdges.Length; i++)
        {
            if (!(bucketUpperEdges[i] > bucketUpperEdges[i - 1]))
                throw new ArgumentException(
                    $"bucketUpperEdges must be strictly ascending; edge[{i}]={bucketUpperEdges[i]} <= edge[{i - 1}]={bucketUpperEdges[i - 1]}.",
                    nameof(bucketUpperEdges));
        }
        if (values.Length == 0) return;

        double min = values[0], max = values[0], sum = 0, sumSq = 0;
        for (int i = 0; i < values.Length; i++)
        {
            double v = values[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumSq += v * v;
        }

        int n = bucketUpperEdges.Length;
        var bucketEdges = new double[n];
        var bucketCounts = new double[n];
        for (int i = 0; i < n; i++) bucketEdges[i] = bucketUpperEdges[i];
        // Binary search per value — O(N log B) which beats the naive
        // O(N×B) when B is large (e.g., 256 fine-grained edges).
        for (int i = 0; i < values.Length; i++)
        {
            double v = values[i];
            int lo = 0, hi = n - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (v < bucketEdges[mid]) hi = mid;
                else lo = mid + 1;
            }
            // Final bucket holds the >=last-edge tail.
            if (v >= bucketEdges[n - 1]) lo = n - 1;
            bucketCounts[lo]++;
        }

        var ev = new EventBuilder
        {
            WallTime = (DateTimeOffset.UtcNow - new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero)).TotalSeconds,
            Step = step,
            HistogramTag = tag,
            HistogramMin = min,
            HistogramMax = max,
            HistogramNum = values.Length,
            HistogramSum = sum,
            HistogramSumSq = sumSq,
            HistogramBucketLimits = bucketEdges,
            HistogramBucketCounts = bucketCounts,
        }.ToBytes();
        WriteRecord(ev);
    }

    /// <summary>
    /// Logs a histogram of <paramref name="values"/> under
    /// <paramref name="tag"/>. Bucket count is auto-selected (30
    /// equal-width bins from min to max). For caller-specified edges
    /// use <see cref="AddHistogramWithBuckets"/>.
    /// </summary>
    public void AddHistogram(string tag, ReadOnlySpan<float> values, long step)
    {
        ThrowIfDisposed();
        if (tag is null) throw new ArgumentNullException(nameof(tag));
        if (values.Length == 0) return;

        // Compute summary stats + auto-bin into 30 equal-width buckets
        // — TensorBoard's default-renderer histogram view.
        double min = values[0], max = values[0], sum = 0, sumSq = 0;
        for (int i = 0; i < values.Length; i++)
        {
            double v = values[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumSq += v * v;
        }
        // HistogramProto's bucket_limit is parallel to bucket — both
        // arrays must have the same length, and bucket_limit[i] is
        // the *upper* edge of bucket i (not the lower).
        double[] bucketEdges;
        double[] bucketCounts;
        if (max == min)
        {
            // Constant-value distribution: a single bucket whose upper
            // limit is the value itself. Inventing a fake range would
            // make TensorBoard render a spread-out histogram for what
            // is structurally a delta function.
            bucketEdges = new double[] { max };
            bucketCounts = new double[] { values.Length };
        }
        else
        {
            const int Buckets = 30;
            double width = (max - min) / Buckets;
            bucketEdges = new double[Buckets];
            bucketCounts = new double[Buckets];
            for (int i = 0; i < Buckets; i++) bucketEdges[i] = min + (i + 1) * width;
            for (int i = 0; i < values.Length; i++)
            {
                int idx = (int)((values[i] - min) / width);
                if (idx >= Buckets) idx = Buckets - 1;
                if (idx < 0) idx = 0;
                bucketCounts[idx]++;
            }
        }

        var ev = new EventBuilder
        {
            WallTime = (DateTimeOffset.UtcNow - new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero)).TotalSeconds,
            Step = step,
            HistogramTag = tag,
            HistogramMin = min,
            HistogramMax = max,
            HistogramNum = values.Length,
            HistogramSum = sum,
            HistogramSumSq = sumSq,
            HistogramBucketLimits = bucketEdges,
            HistogramBucketCounts = bucketCounts,
        }.ToBytes();
        WriteRecord(ev);
    }

    /// <summary>
    /// Logs a single PNG-encoded image under <paramref name="tag"/>.
    /// TensorBoard's Image dashboard accepts encoded PNG / JPEG bytes
    /// directly — the caller is responsible for the encoding step
    /// (<see cref="System.Drawing"/> isn't a portable .NET dependency,
    /// and we don't want to ship a per-platform image-codec pin).
    /// </summary>
    /// <param name="tag">Tag under which the image appears in the dashboard.</param>
    /// <param name="encodedImage">PNG or JPEG byte payload.</param>
    /// <param name="width">Image width in pixels (for the protobuf metadata).</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="colorspace">3 = RGB, 4 = RGBA, 1 = grayscale, 2 = grayscale+alpha. Defaults to 3.</param>
    /// <param name="step">Training step the image was captured at.</param>
    public void AddImage(string tag, byte[] encodedImage, int width, int height, long step, int colorspace = 3)
    {
        ThrowIfDisposed();
        if (tag is null) throw new ArgumentNullException(nameof(tag));
        if (encodedImage is null) throw new ArgumentNullException(nameof(encodedImage));
        if (encodedImage.Length == 0) throw new ArgumentException("Encoded image cannot be empty.", nameof(encodedImage));
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (colorspace < 1 || colorspace > 4) throw new ArgumentOutOfRangeException(nameof(colorspace));

        var ev = new EventBuilder
        {
            WallTime = UnixSeconds(),
            Step = step,
            ImageTag = tag,
            ImageBytes = encodedImage,
            ImageWidth = width,
            ImageHeight = height,
            ImageColorspace = colorspace,
        }.ToBytes();
        WriteRecord(ev);
    }

    /// <summary>
    /// Logs an embedding projector record. TensorBoard's Projector
    /// dashboard reads <c>tensors.tsv</c> + <c>metadata.tsv</c> from
    /// disk via a <c>projector_config.pbtxt</c>; this helper materialises
    /// those files in <paramref name="logDir"/> alongside the events
    /// stream so the dashboard picks them up automatically.
    /// </summary>
    /// <param name="logDir">Directory where the events file lives. Required because
    /// projector files are written next to the events file, not into the events stream.</param>
    /// <param name="tag">Sub-folder name under <c>logDir/</c> that holds this projection.</param>
    /// <param name="embeddings">2D array — rows × dim. Each row is one point in the projection.</param>
    /// <param name="metadataLabels">Optional per-row labels written to <c>metadata.tsv</c>.</param>
    public void AddEmbedding(string logDir, string tag, float[,] embeddings, string[]? metadataLabels = null)
    {
        ThrowIfDisposed();
        if (logDir is null) throw new ArgumentNullException(nameof(logDir));
        if (tag is null) throw new ArgumentNullException(nameof(tag));
        if (embeddings is null) throw new ArgumentNullException(nameof(embeddings));
        int rows = embeddings.GetLength(0);
        int dim = embeddings.GetLength(1);
        if (rows == 0 || dim == 0)
            throw new ArgumentException("Embeddings must be non-empty (rows > 0, dim > 0).", nameof(embeddings));
        if (metadataLabels is not null && metadataLabels.Length != rows)
            throw new ArgumentException(
                $"metadataLabels.Length ({metadataLabels.Length}) must equal embeddings rows ({rows}).",
                nameof(metadataLabels));

        // Write tag-specific subdir with tensors.tsv + metadata.tsv.
        string subDir = Path.Combine(logDir, tag);
        if (!Directory.Exists(subDir)) Directory.CreateDirectory(subDir);
        string tensorsPath = Path.Combine(subDir, "tensors.tsv");
        string metadataPath = Path.Combine(subDir, "metadata.tsv");

        // tensors.tsv — tab-separated floats per row, one row per
        // embedding point. TensorBoard's projector reads this format.
        using (var w = new StreamWriter(tensorsPath))
        {
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < dim; c++)
                {
                    if (c > 0) w.Write('\t');
                    w.Write(embeddings[r, c].ToString("R", System.Globalization.CultureInfo.InvariantCulture));
                }
                w.WriteLine();
            }
        }

        if (metadataLabels is not null)
        {
            using var w = new StreamWriter(metadataPath);
            foreach (var label in metadataLabels) w.WriteLine(label ?? string.Empty);
        }

        // projector_config.pbtxt — text-format protobuf the dashboard
        // reads. We append rather than overwrite so multiple
        // AddEmbedding calls can coexist in the same logDir.
        string configPath = Path.Combine(logDir, "projector_config.pbtxt");
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("embeddings {");
        sb.AppendLine($"  tensor_name: \"{tag}\"");
        sb.AppendLine($"  tensor_path: \"{tag}/tensors.tsv\"");
        if (metadataLabels is not null)
            sb.AppendLine($"  metadata_path: \"{tag}/metadata.tsv\"");
        sb.AppendLine("}");
        File.AppendAllText(configPath, sb.ToString());
    }

    /// <summary>
    /// Logs a model-graph summary — the GraphDef bytes that
    /// TensorBoard's Graphs dashboard renders. The caller supplies
    /// already-serialised <c>tensorflow.GraphDef</c> protobuf bytes;
    /// AiDotNet doesn't ship a native TF graph emitter (we'd need
    /// the entire <c>tensorflow</c> package's protobuf surface), so
    /// this method exists to let callers who have a GraphDef from a
    /// different source (e.g. an exported ONNX → TF transform) pipe
    /// it through.
    /// </summary>
    /// <param name="graphDefBytes">Serialised tensorflow.GraphDef protobuf.</param>
    public void AddGraph(byte[] graphDefBytes)
    {
        ThrowIfDisposed();
        if (graphDefBytes is null) throw new ArgumentNullException(nameof(graphDefBytes));
        if (graphDefBytes.Length == 0)
            throw new ArgumentException("GraphDef bytes cannot be empty.", nameof(graphDefBytes));
        // Event.graph_def = field 4, length-delimited.
        var ev = new EventBuilder
        {
            WallTime = UnixSeconds(),
            Step = 0,
            GraphDefBytes = graphDefBytes,
        }.ToBytes();
        WriteRecord(ev);
    }

    /// <summary>
    /// Logs a numeric hyperparameter value as an ordinary scalar
    /// summary under tag <c>hparams/{name}</c>. Visible in
    /// TensorBoard's <b>Scalars</b> dashboard alongside other tagged
    /// scalars; <b>not</b> automatically displayed in TensorBoard's
    /// dedicated <b>HParams</b> dashboard.
    /// </summary>
    /// <remarks>
    /// <para><b>Why this is a plain scalar, not a real HParams record:</b></para>
    /// <para>
    /// TensorBoard's HParams plugin requires dedicated session-metadata
    /// summaries (<c>session_start_info</c> / <c>session_end_info</c>
    /// + an experiment configuration) — it doesn't recognise
    /// arbitrary <c>hparams/*</c>-prefixed scalars. Implementing the
    /// metadata-summaries surface adds a meaningful amount of
    /// protobuf shape we'd need to vend; until that lands, this
    /// method ships as a "scalar log under a conventional prefix"
    /// helper, with the docs being explicit about the limitation
    /// rather than misleading callers into thinking the dashboard
    /// will pick it up.
    /// </para>
    /// </remarks>
    public void AddHParam(string name, double value)
    {
        ThrowIfDisposed();
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException(
                "HParam name must be a non-empty, non-whitespace string. A blank name " +
                "would degrade into an unnameable 'hparams/' tag that's hard to detect downstream.",
                nameof(name));
        AddScalar("hparams/" + name, value, step: 0);
    }

    /// <summary>
    /// Logs every entry of <paramref name="hparams"/> as a scalar
    /// under <c>hparams/{key}</c>. Convenience for "log all
    /// hyperparameters once at start of training". See
    /// <see cref="AddHParam"/> for the relationship to TensorBoard's
    /// HParams dashboard (these are scalar logs, not HParams plugin
    /// records).
    /// </summary>
    public void LogHParams(IReadOnlyDictionary<string, double> hparams)
    {
        // Guard up front: an empty dictionary used to silently succeed
        // on a disposed writer because the per-entry AddHParam call
        // never ran. Now ThrowIfDisposed fires before any work.
        ThrowIfDisposed();
        if (hparams is null) throw new ArgumentNullException(nameof(hparams));
        foreach (var kv in hparams) AddHParam(kv.Key, kv.Value);
    }

    /// <summary>Flushes the underlying stream.</summary>
    public void Flush()
    {
        ThrowIfDisposed();
        _stream.Flush();
    }

    /// <summary>
    /// Seconds since Unix epoch as a double — what the TFEvents
    /// <c>Event.wall_time</c> field expects.
    /// </summary>
    private static double UnixSeconds()
        => (DateTimeOffset.UtcNow - new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero)).TotalSeconds;

    private void WriteRecord(byte[] payload)
    {
        // TFRecord framing:
        //   uint64 LE  length
        //   uint32 LE  masked_crc32(length)
        //   payload
        //   uint32 LE  masked_crc32(payload)
        Span<byte> lenBytes = stackalloc byte[8];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(lenBytes, (ulong)payload.Length);
        var lenArr = lenBytes.ToArray();
        _stream.Write(lenArr, 0, 8);

        uint maskedLenCrc = MaskedCrc32C(lenArr);
        Span<byte> u32 = stackalloc byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(u32, maskedLenCrc);
        _stream.Write(u32.ToArray(), 0, 4);

        _stream.Write(payload, 0, payload.Length);

        uint maskedPayloadCrc = MaskedCrc32C(payload);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(u32, maskedPayloadCrc);
        _stream.Write(u32.ToArray(), 0, 4);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TensorBoardSummaryWriter));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try { _stream.Flush(); } catch { }
        if (_ownsStream) _stream.Dispose();
    }

    // ─── CRC32C with the TFRecord mask ─────────────────────────────

    /// <summary>
    /// CRC32C (Castagnoli polynomial) hashed over the Tensorflow-
    /// canonical mask <c>((crc &gt;&gt; 15) | (crc &lt;&lt; 17)) +
    /// 0xa282ead8</c>. TFRecord readers expect the masked variant —
    /// passing a raw CRC fails verification.
    /// </summary>
    internal static uint MaskedCrc32C(byte[] data)
    {
        uint crc = Crc32C(data);
        return ((crc >> 15) | (crc << 17)) + 0xa282ead8u;
    }

    private static readonly uint[] _crc32cTable = BuildCrc32cTable();
    private static uint[] BuildCrc32cTable()
    {
        const uint Polynomial = 0x82F63B78u; // CRC32C (Castagnoli) reversed polynomial
        var table = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint c = i;
            for (int k = 0; k < 8; k++)
                c = ((c & 1) != 0) ? (Polynomial ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        return table;
    }

    internal static uint Crc32C(byte[] data)
    {
        uint crc = 0xFFFFFFFFu;
        for (int i = 0; i < data.Length; i++)
            crc = _crc32cTable[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
        return ~crc;
    }

    // ─── Hand-rolled protobuf encoder for the Event subset we use ──
    //
    // Field tags taken from the canonical .proto:
    //   message Event {
    //     double wall_time = 1;
    //     int64 step = 2;
    //     oneof what {
    //       string file_version = 3;
    //       Summary summary = 5;
    //     }
    //   }
    //   message Summary {
    //     repeated Value value = 1;
    //   }
    //   message Summary.Value {
    //     string tag = 1;
    //     oneof value {
    //       float simple_value = 2;
    //       HistogramProto histo = 5;
    //     }
    //   }
    //   message HistogramProto {
    //     double min = 1;
    //     double max = 2;
    //     double num = 3;
    //     double sum = 4;
    //     double sum_squares = 5;
    //     repeated double bucket_limit = 6 [packed=true];
    //     repeated double bucket = 7 [packed=true];
    //   }

    private sealed class EventBuilder
    {
        public double WallTime;
        public long Step;
        public string? FileVersion;
        public string? ScalarTag;
        public float? ScalarValue;
        public string? HistogramTag;
        public double HistogramMin;
        public double HistogramMax;
        public long HistogramNum;
        public double HistogramSum;
        public double HistogramSumSq;
        public double[]? HistogramBucketLimits;
        public double[]? HistogramBucketCounts;
        public string? ImageTag;
        public byte[]? ImageBytes;
        public int ImageWidth;
        public int ImageHeight;
        public int ImageColorspace;
        public byte[]? GraphDefBytes;

        public byte[] ToBytes()
        {
            using var ms = new MemoryStream();
            // wall_time = 1 (double, fixed64)
            WriteTag(ms, fieldNumber: 1, wireType: 1);
            WriteFixed64(ms, BitConverter.DoubleToInt64Bits(WallTime));
            // step = 2 (int64, varint)
            WriteTag(ms, fieldNumber: 2, wireType: 0);
            WriteVarintI64(ms, Step);

            if (FileVersion is not null)
            {
                // file_version = 3 (string, length-delimited)
                WriteTag(ms, fieldNumber: 3, wireType: 2);
                var bytes = Encoding.UTF8.GetBytes(FileVersion);
                WriteVarintU64(ms, (ulong)bytes.Length);
                ms.Write(bytes, 0, bytes.Length);
            }
            else if (ScalarTag is not null && ScalarValue.HasValue)
            {
                // summary = 5 (Summary, length-delimited)
                var summaryBytes = BuildScalarSummary(ScalarTag, ScalarValue.Value);
                WriteTag(ms, fieldNumber: 5, wireType: 2);
                WriteVarintU64(ms, (ulong)summaryBytes.Length);
                ms.Write(summaryBytes, 0, summaryBytes.Length);
            }
            else if (HistogramTag is not null && HistogramBucketLimits is not null && HistogramBucketCounts is not null)
            {
                var summaryBytes = BuildHistogramSummary(
                    HistogramTag, HistogramMin, HistogramMax, HistogramNum,
                    HistogramSum, HistogramSumSq, HistogramBucketLimits, HistogramBucketCounts);
                WriteTag(ms, fieldNumber: 5, wireType: 2);
                WriteVarintU64(ms, (ulong)summaryBytes.Length);
                ms.Write(summaryBytes, 0, summaryBytes.Length);
            }
            else if (ImageTag is not null && ImageBytes is not null)
            {
                var summaryBytes = BuildImageSummary(
                    ImageTag, ImageBytes, ImageWidth, ImageHeight, ImageColorspace);
                WriteTag(ms, fieldNumber: 5, wireType: 2);
                WriteVarintU64(ms, (ulong)summaryBytes.Length);
                ms.Write(summaryBytes, 0, summaryBytes.Length);
            }
            else if (GraphDefBytes is not null)
            {
                // graph_def = 4 (length-delimited bytes)
                WriteTag(ms, fieldNumber: 4, wireType: 2);
                WriteVarintU64(ms, (ulong)GraphDefBytes.Length);
                ms.Write(GraphDefBytes, 0, GraphDefBytes.Length);
            }

            return ms.ToArray();
        }

        // Summary.Image protobuf:
        //   message Image {
        //     int32 height = 1;
        //     int32 width = 2;
        //     int32 colorspace = 3;
        //     bytes encoded_image_string = 4;
        //   }
        // Wrapped in Summary.Value with field 4 (image) of type Image.
        private static byte[] BuildImageSummary(
            string tag, byte[] encoded, int width, int height, int colorspace)
        {
            using var imgStream = new MemoryStream();
            WriteTag(imgStream, 1, 0); WriteVarintI64(imgStream, height);
            WriteTag(imgStream, 2, 0); WriteVarintI64(imgStream, width);
            WriteTag(imgStream, 3, 0); WriteVarintI64(imgStream, colorspace);
            WriteTag(imgStream, 4, 2);
            WriteVarintU64(imgStream, (ulong)encoded.Length);
            imgStream.Write(encoded, 0, encoded.Length);
            var imgBytes = imgStream.ToArray();

            using var valueStream = new MemoryStream();
            // Value.tag = 1
            WriteTag(valueStream, 1, 2);
            var tagBytes = Encoding.UTF8.GetBytes(tag);
            WriteVarintU64(valueStream, (ulong)tagBytes.Length);
            valueStream.Write(tagBytes, 0, tagBytes.Length);
            // Value.image = 4 (Image, length-delimited)
            WriteTag(valueStream, 4, 2);
            WriteVarintU64(valueStream, (ulong)imgBytes.Length);
            valueStream.Write(imgBytes, 0, imgBytes.Length);
            var valueBytes = valueStream.ToArray();

            using var summaryStream = new MemoryStream();
            WriteTag(summaryStream, 1, 2);
            WriteVarintU64(summaryStream, (ulong)valueBytes.Length);
            summaryStream.Write(valueBytes, 0, valueBytes.Length);
            return summaryStream.ToArray();
        }

        private static byte[] BuildScalarSummary(string tag, float value)
        {
            using var summaryStream = new MemoryStream();
            // Summary.value (Value) = 1 — repeated, but we emit one.
            using var valueStream = new MemoryStream();
            // Value.tag = 1 (string)
            WriteTag(valueStream, fieldNumber: 1, wireType: 2);
            var tagBytes = Encoding.UTF8.GetBytes(tag);
            WriteVarintU64(valueStream, (ulong)tagBytes.Length);
            valueStream.Write(tagBytes, 0, tagBytes.Length);
            // Value.simple_value = 2 (float, fixed32)
            WriteTag(valueStream, fieldNumber: 2, wireType: 5);
            WriteFixed32(valueStream, value.SingleToInt32BitsCompat());

            var valueBytes = valueStream.ToArray();
            WriteTag(summaryStream, fieldNumber: 1, wireType: 2);
            WriteVarintU64(summaryStream, (ulong)valueBytes.Length);
            summaryStream.Write(valueBytes, 0, valueBytes.Length);
            return summaryStream.ToArray();
        }

        private static byte[] BuildHistogramSummary(
            string tag, double min, double max, long num, double sum, double sumSq,
            double[] bucketLimits, double[] bucketCounts)
        {
            using var histoStream = new MemoryStream();
            // HistogramProto fields 1..5
            WriteTag(histoStream, 1, 1); WriteFixed64(histoStream, BitConverter.DoubleToInt64Bits(min));
            WriteTag(histoStream, 2, 1); WriteFixed64(histoStream, BitConverter.DoubleToInt64Bits(max));
            WriteTag(histoStream, 3, 1); WriteFixed64(histoStream, BitConverter.DoubleToInt64Bits(num));
            WriteTag(histoStream, 4, 1); WriteFixed64(histoStream, BitConverter.DoubleToInt64Bits(sum));
            WriteTag(histoStream, 5, 1); WriteFixed64(histoStream, BitConverter.DoubleToInt64Bits(sumSq));
            // bucket_limit = 6, packed double — emit as a single
            // length-delimited field whose body is the concatenated
            // fixed64 doubles.
            WriteTag(histoStream, 6, 2);
            WriteVarintU64(histoStream, (ulong)bucketLimits.Length * 8);
            for (int i = 0; i < bucketLimits.Length; i++)
                WriteFixed64(histoStream, BitConverter.DoubleToInt64Bits(bucketLimits[i]));
            // bucket = 7, packed double
            WriteTag(histoStream, 7, 2);
            WriteVarintU64(histoStream, (ulong)bucketCounts.Length * 8);
            for (int i = 0; i < bucketCounts.Length; i++)
                WriteFixed64(histoStream, BitConverter.DoubleToInt64Bits(bucketCounts[i]));

            var histoBytes = histoStream.ToArray();

            using var valueStream = new MemoryStream();
            // Value.tag = 1
            WriteTag(valueStream, 1, 2);
            var tagBytes = Encoding.UTF8.GetBytes(tag);
            WriteVarintU64(valueStream, (ulong)tagBytes.Length);
            valueStream.Write(tagBytes, 0, tagBytes.Length);
            // Value.histo = 5 (HistogramProto, length-delimited)
            WriteTag(valueStream, 5, 2);
            WriteVarintU64(valueStream, (ulong)histoBytes.Length);
            valueStream.Write(histoBytes, 0, histoBytes.Length);
            var valueBytes = valueStream.ToArray();

            using var summaryStream = new MemoryStream();
            WriteTag(summaryStream, 1, 2);
            WriteVarintU64(summaryStream, (ulong)valueBytes.Length);
            summaryStream.Write(valueBytes, 0, valueBytes.Length);
            return summaryStream.ToArray();
        }

        private static void WriteTag(Stream s, int fieldNumber, int wireType)
            => WriteVarintU64(s, (ulong)((fieldNumber << 3) | wireType));

        private static void WriteVarintU64(Stream s, ulong v)
        {
            while ((v & ~0x7FUL) != 0)
            {
                s.WriteByte((byte)((v & 0x7FUL) | 0x80UL));
                v >>= 7;
            }
            s.WriteByte((byte)v);
        }

        private static void WriteVarintI64(Stream s, long v) => WriteVarintU64(s, (ulong)v);

        private static void WriteFixed32(Stream s, int v)
        {
            Span<byte> buf = stackalloc byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(buf, v);
            s.Write(buf.ToArray(), 0, 4);
        }

        private static void WriteFixed64(Stream s, long v)
        {
            Span<byte> buf = stackalloc byte[8];
            System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(buf, v);
            s.Write(buf.ToArray(), 0, 8);
        }
    }
}

/// <summary>
/// Polyfill for <c>BitConverter.SingleToInt32Bits</c> which is .NET
/// 6+; net471 needs the byte-array detour.
/// </summary>
internal static class BitConverterPolyfill
{
    public static int SingleToInt32BitsCompat(this float value)
    {
#if NET6_0_OR_GREATER
        return BitConverter.SingleToInt32Bits(value);
#else
        return BitConverter.ToInt32(BitConverter.GetBytes(value), 0);
#endif
    }
}
