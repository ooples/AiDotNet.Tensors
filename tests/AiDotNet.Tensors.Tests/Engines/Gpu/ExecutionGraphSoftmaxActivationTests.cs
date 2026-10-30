#if NET6_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu.Graph;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Gpu;

/// <summary>
/// The linear layer's separate-node softmax normalizes [rows, features] row by row. Out of place it writes straight
/// into the output; in place (input == output, the separate-node path) it must not bind one buffer as both the kernel's
/// input and output, so it normalizes into scratch, copies back and frees the scratch. Checked with a recording backend
/// (DispatchProxy, .NET 5+), so it runs without a GPU.
/// </summary>
public sealed class ExecutionGraphSoftmaxActivationTests
{
    [Fact]
    public void OutOfPlace_RunsOneRowWiseSoftmaxIntoTheOutput()
    {
        var (backend, log) = RecordingBackend.Create();
        var input = new FakeBuffer(12);
        var output = new FakeBuffer(12);
        ExecutionGraphBuilderExtensions.ApplyActivation(backend, input, output, FusedActivationType.Softmax, rows: 3, features: 4);
        Assert.Equal(new[] { "Softmax(in,out,3,4)" }, log.Calls(input, output));
    }

    [Fact]
    public void InPlace_NormalizesIntoScratchThenCopiesBackAndFreesIt()
    {
        var (backend, log) = RecordingBackend.Create();
        var buffer = new FakeBuffer(12);
        ExecutionGraphBuilderExtensions.ApplyActivation(backend, buffer, buffer, FusedActivationType.Softmax, rows: 3, features: 4);
        Assert.Equal(new[] { "AllocateBuffer(12)", "Softmax(in,scratch,3,4)", "Copy(scratch,out,12)" }, log.Calls(buffer, buffer));
        Assert.True(log.Scratch is { Disposed: true }, "the scratch buffer was not disposed");
    }

    private sealed class FakeBuffer : IGpuBuffer
    {
        public FakeBuffer(int size) { Size = size; }
        public int Size { get; }
        public long SizeInBytes => Size * 4L;
        public IntPtr Handle => IntPtr.Zero;
        public bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }

    private sealed class CallLog
    {
        private readonly List<(string Name, object?[] Args)> _calls = new();
        public FakeBuffer? Scratch { get; set; }
        public void Add(string name, object?[] args) => _calls.Add((name, args));

        /// <summary>Each call with its buffers named: in place the one buffer is "in" as the first argument (read) and
        /// "out" elsewhere (written).</summary>
        public string[] Calls(IGpuBuffer input, IGpuBuffer output)
        {
            var result = new string[_calls.Count];
            for (int i = 0; i < _calls.Count; i++)
            {
                var (name, args) = _calls[i];
                var parts = new string[args.Length];
                for (int a = 0; a < args.Length; a++)
                {
                    object? o = args[a];
                    parts[a] = ReferenceEquals(o, Scratch) ? "scratch"
                        : ReferenceEquals(input, output) && ReferenceEquals(o, input) ? (a == 0 ? "in" : "out")
                        : ReferenceEquals(o, input) ? "in"
                        : ReferenceEquals(o, output) ? "out"
                        : o?.ToString() ?? "null";
                }
                result[i] = $"{name}({string.Join(",", parts)})";
            }
            return result;
        }
    }
    private class RecordingBackend : DispatchProxy
    {
        private CallLog _log = null!;

        public static (IDirectGpuBackend Backend, CallLog Log) Create()
        {
            var backend = Create<IDirectGpuBackend, RecordingBackend>();
            var log = new CallLog();
            ((RecordingBackend)(object)backend)._log = log;
            return (backend, log);
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            args ??= Array.Empty<object?>();
            switch (targetMethod?.Name)
            {
                case "AllocateBuffer" when args.Length == 1 && args[0] is int size:
                    _log.Add("AllocateBuffer", args);
                    return _log.Scratch = new FakeBuffer(size);
                case "Softmax":
                case "Copy":
                    _log.Add(targetMethod!.Name, args);
                    return null;
                default:
                    throw new NotSupportedException($"unexpected backend call {targetMethod?.Name}");
            }
        }
    }
}
#endif
