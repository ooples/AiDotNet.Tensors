# CPU GEMM performance tuning (FP32)

AiDotNet.Tensors ships a **portable, deterministic, managed** FP32 GEMM (the GotoBLAS-style
`GotoGemmFp32` + JIT microkernels). It needs no native dependency and is bit-reproducible across
machines and thread counts. For most workloads it is the right default.

It is the only CPU GEMM: the library ships no native BLAS (OpenBLAS and Intel MKL were removed under #368).
Measured against OpenBLAS on a 16-core Zen 2 (79 catalog shapes, alternating arms, two runs), the managed
GEMM's median time was 0.90-0.93x OpenBLAS's: FP32 median 0.96x, FP64 0.59x, faster on about 65% of shapes,
with the worst remaining shapes around 1.1-1.35x. One opt-in switch trades precision for bandwidth on large
shapes.

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

`AIDOTNET_GEMM_BF16` only affects the `GotoGemmFp32` path, and only on shapes that meet the gate.

## Summary

| Setting                  | Default | Effect                                              | Trade-off                          |
|--------------------------|---------|-----------------------------------------------------|------------------------------------|
| (none)                   | ✅      | Portable, deterministic managed FP32/FP64 GEMM      | —                                  |
| `AIDOTNET_GEMM_BF16=1`   |         | bf16-B on large GEMMs (beats MKL on big squares)    | lossy (~3 sig figs), large shapes only |
