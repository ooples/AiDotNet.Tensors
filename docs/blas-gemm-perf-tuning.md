# CPU GEMM performance tuning (FP32)

AiDotNet.Tensors ships a **portable, deterministic, managed** FP32 GEMM (the GotoBLAS-style
`GotoGemmFp32` + JIT microkernels). It needs no native dependency and is bit-reproducible across
machines and thread counts. For most workloads it is the right default.

On large multi-core x86 it does **not** fully match hand-tuned native BLAS (OpenBLAS / Intel MKL)
on the medium/large "diffusion" shapes — measured ~50–72% of OpenBLAS on a 16-core Zen2 for those
shapes (it matches or wins on tiny GEMM and large squares). The remaining gap is the native
libraries' assembly-tuned multithreaded packed-GEMM scheduling. Two opt-in environment switches let
you trade portability/precision for that performance when you need it.

## `AIDOTNET_USE_BLAS=1` — native OpenBLAS fast path (opt-in)

Set `AIDOTNET_USE_BLAS=1` (also accepts `true` / `yes` / `on`) **at process start** to route FP32/FP64
GEMM through native OpenBLAS when a compatible binary is present.

- **When to use:** you need maximum CPU throughput on the diffusion/medium GEMM shapes today and can
  accept a native dependency. ~1.4–2× the managed path on those shapes.
- **Cost:** adds a native dependency; native BLAS is **not bit-deterministic** across thread counts,
  so combine with the determinism settings only if your build's native provider guarantees it.
- **Opt back out:** `AIDOTNET_USE_BLAS=0` forces the managed path even if a native binary is present.
- Provider selection: `AIDOTNET_BLAS_PROVIDER` (advanced; see `BlasProvider`).

The managed path remains the default precisely because it is portable and deterministic — native BLAS
is the explicit performance opt-in, not the baseline.

## `AIDOTNET_GEMM_BF16=1` — bf16-B large-GEMM bandwidth path (opt-in, precision-changing)

Set `AIDOTNET_GEMM_BF16=1` to store the streamed **B** operand as bf16 (half the bytes → ~2×
bandwidth headroom) on large bandwidth-bound shapes; A stays fp32 and accumulation stays fp32.

- **When it engages:** only the large bandwidth-bound regime (total work ≥ 4·10⁹, M ≥ 512, N < 2·K).
  Smaller and wide-N/short-M shapes stay on exact fp32 — bf16 there only adds pack overhead and
  precision loss (it measured *slower* on the compute-bound 1024³).
- **Measured:** ~1.13× (2048³) to ~1.19× (4096³) over the fp32 managed path, and it **beats Intel
  MKL** on the large squares (≈147% of MKL at 4096³).
- **Precision:** bf16 keeps ~3 significant digits; the K-deep fp32 accumulate keeps the result
  error small *relative to the result scale* (≈10⁻³). This is a **lossy** mode — never enabled by
  default. Appropriate for inference where bf16 precision is acceptable (common in diffusion);
  do not use it where exact fp32 GEMM is required.

`AIDOTNET_GEMM_BF16` only affects the managed `GotoGemmFp32` path (it is ignored when
`AIDOTNET_USE_BLAS=1` routes to native BLAS, and on the smaller shapes that don't meet the gate).

## Summary

| Setting                  | Default | Effect                                              | Trade-off                          |
|--------------------------|---------|-----------------------------------------------------|------------------------------------|
| (none)                   | ✅      | Portable, deterministic managed FP32/FP64 GEMM      | ~50–72% of native BLAS on big shapes |
| `AIDOTNET_USE_BLAS=1`    |         | Native OpenBLAS GEMM (~1.4–2× on diffusion shapes)  | native dependency, non-deterministic |
| `AIDOTNET_GEMM_BF16=1`   |         | bf16-B on large GEMMs (beats MKL on big squares)    | lossy (~3 sig figs), large shapes only |
