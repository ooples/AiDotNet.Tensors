# AiDotNet.Tensors.Gpu

GPU meta-package for [AiDotNet.Tensors](https://github.com/ooples/AiDotNet.Tensors).

Installing this package gives you **AiDotNet.Tensors + the full NVIDIA CUDA runtime** (cuBLAS, cuBLASLt, cudart, and the nvRTC runtime kernel compiler) so NVIDIA GPU acceleration works **out of the box — no CUDA Toolkit install required**.

```
dotnet add package AiDotNet.Tensors.Gpu
```

The GPU engine auto-adopts on a supported NVIDIA card and runs training/inference on the device by default (generic GPU dispatch + CUDA-graph whole-step capture are on by default; opt out with `AIDOTNET_GPU_PREFER_GENERIC=0` / `AIDOTNET_CUDA_GRAPH_STEP=0`).

## When NOT to use this

CPU-only deployments should install the lean base **AiDotNet.Tensors** package instead — this meta-package pulls ~900 MB of CUDA native DLLs.

## Notes

- Windows x64 today; Linux x64 runtimes are staged in the native package and can be enabled the same way.
- A driver new enough for CUDA 12 is required (the CUDA *Toolkit* is not — the nvRTC compiler ships in the package).
- For absolute peak throughput, ensure the GPU's power limit is at its default maximum (`nvidia-smi -q -d POWER`); a manually lowered power cap throttles the clock.
