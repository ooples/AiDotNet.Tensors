using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// Drop-in replacement for the <see cref="Dictionary{TKey,TValue}"/> that backed the OpenCL kernel
    /// cache. Behaviour is identical EXCEPT the throwing indexer getter records a kernel-not-found event
    /// to <see cref="GpuLaunchProbe.OnKernelMiss"/> before throwing. That is the single choke point for
    /// the "hollow GPU override" signal (#775): a <c>backend.Op()</c> that indexes a kernel name which was
    /// never registered throws here, its caller's <c>catch</c> silently falls back to the CPU, and the
    /// op looks GPU-covered to reflection while doing zero GPU work. The residency guard resets the probe,
    /// runs one op, and an op with zero launches AND a recorded miss is a hollow-override bug (vs a
    /// legitimately zero-compute view, which misses nothing). <see cref="TryGetValue"/> is a CHECKED access
    /// whose misses the caller handles, so it does NOT record — only the throwing indexer does.
    /// </summary>
    internal sealed class OpenClKernelCache
    {
        private readonly Dictionary<string, DirectOpenClKernel> _map = new();

        public DirectOpenClKernel this[string name]
        {
            get
            {
                if (_map.TryGetValue(name, out var kernel)) return kernel;
                GpuLaunchProbe.OnKernelMiss(name);
                throw new KeyNotFoundException($"The given key '{name}' was not present in the dictionary.");
            }
            set => _map[name] = value;
        }

        public bool TryGetValue(string name, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out DirectOpenClKernel value)
            => _map.TryGetValue(name, out value);
        public bool ContainsKey(string name) => _map.ContainsKey(name);
        public int Count => _map.Count;
        public Dictionary<string, DirectOpenClKernel>.ValueCollection Values => _map.Values;
        public void Clear() => _map.Clear();
    }
}
