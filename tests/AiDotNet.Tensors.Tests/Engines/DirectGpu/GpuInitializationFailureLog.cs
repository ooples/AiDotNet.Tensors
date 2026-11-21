using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A GPU lane fails when a test of its own backend skips, and the usual reason is only "CUDA is not available": the
/// backend records why it could not initialize, but writes it to the trace, which a test run discards. With
/// <c>AIDOTNET_GPU_INIT_FAILURE_LOG</c> set to a file path, every backend initialization failure the process traces is
/// appended to that file, so the lane can print the cause next to the skips.
/// </summary>
internal static class GpuInitializationFailureLog
{
    [ModuleInitializer]
    internal static void Attach()
    {
        string? path = Environment.GetEnvironmentVariable("AIDOTNET_GPU_INIT_FAILURE_LOG");
        if (path is null || string.IsNullOrWhiteSpace(path)) return;
        Trace.Listeners.Add(new FailureListener(path));
    }

    private sealed class FailureListener : TraceListener
    {
        private readonly string _path;
        private readonly object _gate = new();

        public FailureListener(string path) => _path = path;

        public override void Write(string? message)
        {
            // Failures are traced as whole lines; a partial write is never one.
        }

        public override void WriteLine(string? message)
        {
            if (message is null || message.IndexOf("initialization failed", StringComparison.OrdinalIgnoreCase) < 0) return;
            try
            {
                lock (_gate)
                {
                    File.AppendAllText(_path, $"{DateTime.UtcNow:O} {message}{Environment.NewLine}");
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                // Another test host holds the file, or the path is gone. The diagnostic must never fail a test, and the
                // trace is the only channel here, so report it to the debugger output rather than re-enter the listeners.
                Debug.WriteLine($"GPU initialization failure log could not be written: {ex.Message}");
            }
        }
    }
}
