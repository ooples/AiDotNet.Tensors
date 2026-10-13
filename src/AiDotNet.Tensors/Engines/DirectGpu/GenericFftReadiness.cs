// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Whether a launcher backend's generic FFT can run (#963, #965): it reports the capability only after every kernel
/// <see cref="PortableGenericFft"/> launches has compiled, remembers the first failure, and turns an explicit request
/// for an unavailable transform into a <see cref="GpuCapabilityUnavailableException"/> carrying that error.
/// </summary>
internal sealed class GenericFftReadiness
{
    private readonly Func<string, string?> _compile;
    private readonly object _lock = new object();
    private bool? _ready;
    private string? _error;

    /// <param name="compile">Builds the named kernel's pipeline; returns null on success, else the compiler's error.</param>
    public GenericFftReadiness(Func<string, string?> compile)
        => _compile = compile ?? throw new ArgumentNullException(nameof(compile));

    /// <summary>The first compile error, once one has occurred.</summary>
    public string? Error
    {
        get { lock (_lock) return _error; }
    }

    public bool IsReady(FftElementType type)
    {
        if (!PortableGenericFft.Supports(type)) return false;
        lock (_lock)
        {
            if (_ready is bool known) return known;
            // Every storage type runs the same kernels (the storage mode is a launch argument), so one check covers all.
            foreach (string kernel in PortableGenericFft.KernelNames)
            {
                string? error = _compile(kernel);
                if (error is null) continue;
                _error = $"{kernel}: {error}";
                _ready = false;
                return false;
            }
            _ready = true;
            return true;
        }
    }

    public void Require(FftElementType type, string backendName)
    {
        if (IsReady(type)) return;
        throw new GpuCapabilityUnavailableException(
            $"{backendName} generic FFT with {type} storage is unavailable" + (Error is { } e ? ": " + e : "."));
    }
}