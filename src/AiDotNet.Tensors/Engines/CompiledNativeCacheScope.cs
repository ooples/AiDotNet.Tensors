using System;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines;

// Only a permanently uncaptured compiled body may borrow larger idle native cells. Queue affinity and
// physical-capacity accounting remain owned by the native pool; nested capturable bodies suppress this scope.
internal static class CompiledNativeCacheScope
{
    [ThreadStatic] internal static IDirectGpuBackend? Current;
    internal static Lease Enter(IDirectGpuBackend? backend) => new(backend);
    internal readonly struct Lease : IDisposable
    {
        private readonly IDirectGpuBackend? _previous;
        internal Lease(IDirectGpuBackend? backend) { _previous = Current; Current = backend; }
        public void Dispose() => Current = _previous;
    }
}
