# Parity-210 native Metal kernels

This directory is the landing zone for native Metal kernels that implement
issue #210's new ops. Currently empty — every parity-210 op reaches Metal
callers through inheritance (`DirectGpuTensorEngine : CpuEngine`), so
they execute on CPU when this backend is active.

See `Engines/DirectGpu/Parity210Kernels.cs` for the rollout priority
list and the skip-with-reason convention for ops a backend genuinely
can't express.
