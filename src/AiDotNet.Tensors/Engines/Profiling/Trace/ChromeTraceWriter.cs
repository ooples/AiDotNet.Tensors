// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace AiDotNet.Tensors.Engines.Profiling.Trace;

/// <summary>
/// Emits a sequence of <see cref="TraceEvent"/> as the Chrome Trace Event
/// Format JSON object that <c>chrome://tracing</c> and Perfetto UI accept
/// without further processing.
///
/// <para><b>Format:</b>
/// <code>
/// { "traceEvents": [ { "name":"...", "cat":"...", "ph":"X", "ts":..., "dur":..., "pid":..., "tid":..., "args":{...} }, ... ],
///   "displayTimeUnit": "ns" }
/// </code></para>
///
/// <para>We hand-roll the JSON instead of going through
/// <c>System.Text.Json</c>: events are simple, the schema is fixed, and
/// avoiding the reflection/serializer warm-up time means a 1-million-event
/// flush stays in the tens-of-millis range.</para>
///
/// <para>Strings are JSON-escaped with backslash sequences for the seven
/// characters the spec requires (<c>" \ / \b \f \n \r \t</c>) plus the
/// <c>\uXXXX</c> escape for any control character. Non-ASCII characters pass
/// through as UTF-8 bytes — Chrome and Perfetto both accept that.</para>
/// </summary>
public static class ChromeTraceWriter
{
    /// <summary>Writes <paramref name="events"/> to <paramref name="path"/>. If
    /// <paramref name="path"/> ends in <c>.gz</c>, the output is gzipped (the
    /// extension Chrome's tracing UI expects for compressed traces).</summary>
    public static void Write(string path, IReadOnlyList<TraceEvent> events)
    {
        bool gzip = path.EndsWith(".gz", System.StringComparison.OrdinalIgnoreCase);
        using var fileStream = File.Create(path);
        if (gzip)
        {
            using var gz = new GZipStream(fileStream, CompressionLevel.Fastest, leaveOpen: false);
            Write(gz, events);
        }
        else
        {
            Write(fileStream, events);
        }
    }

    /// <summary>Writes events to <paramref name="stream"/> as raw JSON.
    /// The stream is left open; the caller owns its lifetime.</summary>
    public static void Write(Stream stream, IReadOnlyList<TraceEvent> events)
    {
        // BufferedStream + UTF-8 writer keeps the per-event cost dominated by
        // string formatting, not syscalls. 64 KB matches the default file-cache
        // page size on Windows and Linux — larger buffers don't measurably help.
        using var buffered = new BufferedStream(stream, bufferSize: 64 * 1024);
        using var writer = new StreamWriter(buffered, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false), bufferSize: 64 * 1024, leaveOpen: true);

        writer.Write("{\"traceEvents\":[");
        for (int i = 0; i < events.Count; i++)
        {
            if (i > 0) writer.Write(',');
            WriteEvent(writer, events[i]);
        }
        writer.Write("],\"displayTimeUnit\":\"ns\"}");
        writer.Flush();
    }

    private static void WriteEvent(TextWriter w, TraceEvent e)
    {
        w.Write("{\"name\":\"");
        WriteEscaped(w, e.Name);
        w.Write("\",\"cat\":\"");
        WriteEscaped(w, e.Category);
        w.Write("\",\"ph\":\"");
        w.Write(e.Phase);
        w.Write("\",\"ts\":");
        w.Write(e.TimestampMicros.ToString(System.Globalization.CultureInfo.InvariantCulture));

        // Duration is only emitted on Complete events ('X'). Chrome ignores
        // 'dur' on other phases but Perfetto warns; keep the output clean.
        if (e.Phase == 'X')
        {
            w.Write(",\"dur\":");
            w.Write(e.DurationMicros.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }

        w.Write(",\"pid\":");
        w.Write(e.ProcessId.ToString(System.Globalization.CultureInfo.InvariantCulture));
        w.Write(",\"tid\":");
        w.Write(e.ThreadId.ToString(System.Globalization.CultureInfo.InvariantCulture));

        if (e.Args is { Count: > 0 })
        {
            w.Write(",\"args\":{");
            bool first = true;
            foreach (var kv in e.Args)
            {
                if (!first) w.Write(',');
                first = false;
                w.Write('"');
                WriteEscaped(w, kv.Key);
                w.Write("\":\"");
                WriteEscaped(w, kv.Value ?? "");
                w.Write('"');
            }
            w.Write('}');
        }

        w.Write('}');
    }

    private static void WriteEscaped(TextWriter w, string s)
    {
        // Hot path: most names contain no escape-worthy characters. Stream the
        // common run with Write(s) only when no escape was hit; otherwise fall
        // back to per-char emission. Keeps single-allocation strings flowing
        // through one Write call when possible.
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            switch (c)
            {
                case '\\': w.Write("\\\\"); break;
                case '"':  w.Write("\\\""); break;
                case '\b': w.Write("\\b"); break;
                case '\f': w.Write("\\f"); break;
                case '\n': w.Write("\\n"); break;
                case '\r': w.Write("\\r"); break;
                case '\t': w.Write("\\t"); break;
                default:
                    if (c < 0x20)
                    {
                        // Spec requires \uXXXX for any control character below 0x20.
                        w.Write("\\u");
                        w.Write(((int)c).ToString("X4", System.Globalization.CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        w.Write(c);
                    }
                    break;
            }
        }
    }
}
