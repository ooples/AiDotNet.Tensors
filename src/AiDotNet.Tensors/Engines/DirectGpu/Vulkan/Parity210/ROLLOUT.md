# Parity-210 native Vulkan kernels

This directory is the landing zone for native Vulkan kernels that implement
issue #210's new ops. Currently empty — every parity-210 op reaches Vulkan
callers through inheritance (`DirectGpuTensorEngine : CpuEngine`), so
they execute on CPU when this backend is active.

See `Engines/DirectGpu/Parity210Kernels.cs` for the rollout priority
list and the skip-with-reason convention for ops a backend genuinely
can't express.
