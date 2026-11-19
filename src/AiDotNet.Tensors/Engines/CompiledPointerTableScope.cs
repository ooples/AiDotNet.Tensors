using System;

namespace AiDotNet.Tensors.Engines;

// Only a completed, explicitly uncaptured compiled plan may request mutable native pointer tables.
// The native backend still checks stream capture and keeps this ownership separate from ordinary/captured tables.
internal static class CompiledPointerTableScope
{
    [ThreadStatic] internal static long Current;
    internal readonly struct Lease : IDisposable
    {
        private readonly long _previous;
        internal Lease(long previous) => _previous = previous;
        public void Dispose() => Current = _previous;
    }
    internal static Lease Enter(long owner)
    {
        long previous = Current;
        Current = owner;
        return new Lease(previous);
    }
}
