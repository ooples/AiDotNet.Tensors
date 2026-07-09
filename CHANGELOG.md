# Changelog

## [0.111.1](https://github.com/ooples/AiDotNet.Tensors/compare/v0.111.0...v0.111.1) (2026-07-09)


### Bug Fixes

* **autodiff:** make 2D AffineGrid tape-aware + vectorize (fixes STN untrainable) ([#751](https://github.com/ooples/AiDotNet.Tensors/issues/751)) ([890ed53](https://github.com/ooples/AiDotNet.Tensors/commit/890ed53656bc409da8930380085feeb59390adcf))
* **compiled-training:** keep plan activation buffers alive across replay — unfreezes fused training at large vocab ([#754](https://github.com/ooples/AiDotNet.Tensors/issues/754)) ([687a4c6](https://github.com/ooples/AiDotNet.Tensors/commit/687a4c6ad2a031c9ad5f5caa91c0df009e0abdd9))


### Performance

* **arena:** recycle transformer-layer scratch between layers for memory-bounded inference (AiDotNet[#1824](https://github.com/ooples/AiDotNet.Tensors/issues/1824)) ([#753](https://github.com/ooples/AiDotNet.Tensors/issues/753)) ([2572bd8](https://github.com/ooples/AiDotNet.Tensors/commit/2572bd80e4b9014ffb9b3ee5ad2d9ad8f79fe44b))

## [0.111.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.110.1...v0.111.0) (2026-07-07)


### Features

* **simd:** dual-path Adam kernel — bit-exact default + AIDOTNET_FAST_MATH FMA fast path ([#738](https://github.com/ooples/AiDotNet.Tensors/issues/738)) ([d670ce3](https://github.com/ooples/AiDotNet.Tensors/commit/d670ce31750d09311125761c9d3db8ed2dcc7bdf))


### Bug Fixes

* **autodiff:** make 2D AffineGrid differentiable w.r.t. theta ([#747](https://github.com/ooples/AiDotNet.Tensors/issues/747)) ([72e4116](https://github.com/ooples/AiDotNet.Tensors/commit/72e4116efcf33c11310f54fe0401f3597477db06))
* **bench:** gate gemm-bench on the median per-round paired ON/OFF ratio ([#741](https://github.com/ooples/AiDotNet.Tensors/issues/741)) ([39194bb](https://github.com/ooples/AiDotNet.Tensors/commit/39194bbf96e44f40c5d13969b05a565f732405b1))
* **gpu:** fused-optimizer weight-coherence — refresh resident device buffer after in-place update ([#739](https://github.com/ooples/AiDotNet.Tensors/issues/739)) ([8e61dc4](https://github.com/ooples/AiDotNet.Tensors/commit/8e61dc48df46d7981044fe98b22dc1b6a453218c))
* **gpu:** hardware-independent determinism for scatter-add grad + cuBLAS under SetDeterministicMode (Closes [#742](https://github.com/ooples/AiDotNet.Tensors/issues/742)) ([#744](https://github.com/ooples/AiDotNet.Tensors/issues/744)) ([b815872](https://github.com/ooples/AiDotNet.Tensors/commit/b8158726d08bc72b85bef25fabffef811aef6938))


### Performance

* **gemm:** beta=0 write-first path — skip redundant C-zeroing on pure C:=A·B ([#743](https://github.com/ooples/AiDotNet.Tensors/issues/743)) ([2aac704](https://github.com/ooples/AiDotNet.Tensors/commit/2aac704a5d9a55ff2407903a95ad74bae164e18b))
* kill training spin-storm — lower resident-pool MRES spin (2047 -&gt; 32) ([#736](https://github.com/ooples/AiDotNet.Tensors/issues/736)) ([8ede01f](https://github.com/ooples/AiDotNet.Tensors/commit/8ede01f1c9802bbafbc6eed2fd7c6db2e94a00da))
* zero-alloc training via TensorArena reuse + shape-safe ring (DRAFT) ([#734](https://github.com/ooples/AiDotNet.Tensors/issues/734)) ([3f7178a](https://github.com/ooples/AiDotNet.Tensors/commit/3f7178af342e4ed33871501ae160fd51479e858d))

## [0.103.0](https://github.com/ooples/AiDotNet.Tensors/compare/v0.102.16...v0.103.0) (2026-07-07)


### Features

* **cuda:** wire BatchNorm to cuDNN + fix cuDNN context binding (AiDotNet[#1159](https://github.com/ooples/AiDotNet.Tensors/issues/1159)) ([#726](https://github.com/ooples/AiDotNet.Tensors/issues/726)) ([5ff173c](https://github.com/ooples/AiDotNet.Tensors/commit/5ff173c746319fcc7e00b1c2843cdad0c2683856))
* **dcnv3:** fused grouped/depthwise deformable convolution kernel [WIP] ([#691](https://github.com/ooples/AiDotNet.Tensors/issues/691)) ([c16345f](https://github.com/ooples/AiDotNet.Tensors/commit/c16345f007f7eb8fb8b95b45a87f04849b823199))
* **dcnv3:** single-launch fused grouped deformable backward (GPU) + CPU follow-up ([#693](https://github.com/ooples/AiDotNet.Tensors/issues/693)) ([4006015](https://github.com/ooples/AiDotNet.Tensors/commit/4006015f4cac96785b7a6b46c4c4941cde920415))
* **fused:** bfloat16 moment storage for fused Adam/AdamW (CPU) ([#713](https://github.com/ooples/AiDotNet.Tensors/issues/713)) ([02655de](https://github.com/ooples/AiDotNet.Tensors/commit/02655dea06ffba03c143e761f88e28e32a8e8f6b))
* **gemm:** opt-in bf16-B GEMM + fused epilogue on the fast GotoGemm path ([#706](https://github.com/ooples/AiDotNet.Tensors/issues/706)) ([5fa4cff](https://github.com/ooples/AiDotNet.Tensors/commit/5fa4cfffafd870b89734584646f995f3fb851cbb))
* **simd:** shared bit-identical simd adam/amsgrad step kernel ([#1757](https://github.com/ooples/AiDotNet.Tensors/issues/1757)) ([#721](https://github.com/ooples/AiDotNet.Tensors/issues/721)) ([628abd5](https://github.com/ooples/AiDotNet.Tensors/commit/628abd516a4029917861df88c92f94b4098f2df9))
* **streaming:** RAM-aware Auto store — step bf16 -&gt; int8 -&gt; int4 to fit resident cap in inference ([#720](https://github.com/ooples/AiDotNet.Tensors/issues/720)) ([c8681be](https://github.com/ooples/AiDotNet.Tensors/commit/c8681be3e8c303f19570c9ab6d2c96cdbda4d01d))


### Bug Fixes

* **#638:** CUDA-graph capture training now LEARNS (device barrier + deterministic scratch pool + LN resident-weight gate) ([#705](https://github.com/ooples/AiDotNet.Tensors/issues/705)) ([25132d1](https://github.com/ooples/AiDotNet.Tensors/commit/25132d1681130f40ef06f31b2eb97c61e3f7fe6d))
* **autodiff:** route backward row-parallelism through the cooperative pool (fixes single-core training) ([#729](https://github.com/ooples/AiDotNet.Tensors/issues/729)) ([c2b85f2](https://github.com/ooples/AiDotNet.Tensors/commit/c2b85f2110ccd523db01675956084601017d990c))
* **blas:** default to managed BlasManaged; make native CPU BLAS opt-in ([#686](https://github.com/ooples/AiDotNet.Tensors/issues/686)) ([5ca9f14](https://github.com/ooples/AiDotNet.Tensors/commit/5ca9f14476cfbff89352f872a396819d7c2eb06c))
* **blas:** real fix for the Linux FP32 GEMM crash — add a SysV-ABI tile kernel (GotoGemm now cross-platform) ([#699](https://github.com/ooples/AiDotNet.Tensors/issues/699)) ([b7d4582](https://github.com/ooples/AiDotNet.Tensors/commit/b7d4582c12a41fb2f7f288176e0aac658ffcf402))
* **build:** cold builds fail — split LicenseValidator Restore/Build into distinct-property MSBuild calls ([#724](https://github.com/ooples/AiDotNet.Tensors/issues/724)) ([ec9b889](https://github.com/ooples/AiDotNet.Tensors/commit/ec9b8892d6f42fb8146d6abd7a08e140dd0f8303))
* **ci:** harden historical shard regressions ([#718](https://github.com/ooples/AiDotNet.Tensors/issues/718)) ([6ac5bc6](https://github.com/ooples/AiDotNet.Tensors/commit/6ac5bc6628ec370d6f8c9a13eeeaa2c867421577))
* **compilation:** fused optimizer NREs on params with no materialized weights (kicks lazy/multi-modal models off the fused path) ([#712](https://github.com/ooples/AiDotNet.Tensors/issues/712)) ([b54086e](https://github.com/ooples/AiDotNet.Tensors/commit/b54086e3190aaa7fc56c5e3c04929f8fa6d0d73f))
* **compilation:** stop topo-sort dropping in-place op chains (BatchNorm running-stat staleness) ([#704](https://github.com/ooples/AiDotNet.Tensors/issues/704)) ([0f764be](https://github.com/ooples/AiDotNet.Tensors/commit/0f764bec1d5741f31a7dd8fbeef6de6eb6baed99))
* **compile:** refresh norm statistics in the fused training forward (GroupNorm + BatchNorm) ([#722](https://github.com/ooples/AiDotNet.Tensors/issues/722)) ([7f2c94d](https://github.com/ooples/AiDotNet.Tensors/commit/7f2c94da78329d8cc6bf6e2316f26ce3690a3f87))
* **cpu:** invalidate per-array derived-weight caches on invalidatepersistenttensor ([#708](https://github.com/ooples/AiDotNet.Tensors/issues/708)) ([d03ea35](https://github.com/ooples/AiDotNet.Tensors/commit/d03ea35f36dd8ab31ad1cf18b6c62667e505667f))
* **gpu:** apply softmax row-wise in fused-linear epilogue (was crash/garbage) ([#711](https://github.com/ooples/AiDotNet.Tensors/issues/711)) ([e7fbc3e](https://github.com/ooples/AiDotNet.Tensors/commit/e7fbc3e3d995d43a821b23c2e3c439e5ec6e9b9d))
* **gpu:** clamp box w/h in IoU-family loss kernels so degenerate boxes match CPU ([#709](https://github.com/ooples/AiDotNet.Tensors/issues/709)) ([deaf1ea](https://github.com/ooples/AiDotNet.Tensors/commit/deaf1eae23ebfe011fb3bb93253b6762649085d4))
* **gpu:** CUDA Dropout dropped 100% of activations (mask never generated) + stale seed ([#732](https://github.com/ooples/AiDotNet.Tensors/issues/732)) ([41d923f](https://github.com/ooples/AiDotNet.Tensors/commit/41d923f7c11313f5f398ed4642bfcc4aaad6a243))
* **gpu:** DetectVendor must prefer CUDA over OpenCL device-0 vendor on mixed boxes ([#733](https://github.com/ooples/AiDotNet.Tensors/issues/733)) ([e8a4fb4](https://github.com/ooples/AiDotNet.Tensors/commit/e8a4fb4b3160fb50bfe5a3e9a3cc7d1ef2e71d15))
* **gpu:** LayerNorm/RMSNorm normalize over gamma dims for rank &gt; 2 (fixes GPU transformer forward) ([#730](https://github.com/ooples/AiDotNet.Tensors/issues/730)) ([82464c2](https://github.com/ooples/AiDotNet.Tensors/commit/82464c2f8da9be6b50c3f940e5d19322178dbf45))
* **streaming:** defer ReleaseToPool drop on shared storage (refcount &gt; 1) ([#692](https://github.com/ooples/AiDotNet.Tensors/issues/692)) ([0cf29fd](https://github.com/ooples/AiDotNet.Tensors/commit/0cf29fd18830aefd52cbe07e64f5c8d0f10c82bf))
* **streaming:** quiesce in-flight prefetch workers before pool teardown ([#687](https://github.com/ooples/AiDotNet.Tensors/issues/687)) ([f833a92](https://github.com/ooples/AiDotNet.Tensors/commit/f833a92af0f9e8ee164f3586853a7779eaac20a1))
* **streaming:** reclaim WeightRegistry entries for GC'd owners ([#714](https://github.com/ooples/AiDotNet.Tensors/issues/714)) ([#716](https://github.com/ooples/AiDotNet.Tensors/issues/716)) ([be97676](https://github.com/ooples/AiDotNet.Tensors/commit/be97676be3916d1a4e59a832b9d312bdf46f4477))


### Performance

* **#1650/#638:** diffusion UNet forward CUDA-graph capture ENGAGES — whole forward resident, ~3.2x (replay bit-exact) ([#671](https://github.com/ooples/AiDotNet.Tensors/issues/671)) ([1eec88e](https://github.com/ooples/AiDotNet.Tensors/commit/1eec88e6ae30124d6e98c1b38a684cbf3366f03b))
* **#1672:** *Into resident-scratch engine kernels (SDPA/AdaLN/FusedLinear) for diffusion inference ([#696](https://github.com/ooples/AiDotNet.Tensors/issues/696)) ([3652dc7](https://github.com/ooples/AiDotNet.Tensors/commit/3652dc7e427251d34442ac7e4315d209900cd58f))
* **#1715:** bound streaming-pool resident set on the GetParameters/register path (foundation-scale OOM) ([#707](https://github.com/ooples/AiDotNet.Tensors/issues/707)) ([170517e](https://github.com/ooples/AiDotNet.Tensors/commit/170517ef49ae10eec762f095c30ce5ee1df9f809))
* **#653:** route StreamingStrategy GEMM dispatch onto low-latency PersistentParallelExecutor ([#688](https://github.com/ooples/AiDotNet.Tensors/issues/688)) ([d9e7462](https://github.com/ooples/AiDotNet.Tensors/commit/d9e7462e32a75bb06bff70d6b53737d9d1a85ba2))
* **attn:** scatter-lock GC sweep — GQA backward 6.4x less GC + 3D-op locks ([#698](https://github.com/ooples/AiDotNet.Tensors/issues/698)) ([f9596de](https://github.com/ooples/AiDotNet.Tensors/commit/f9596dee4b86f63ca7d11cc51ada0f53d771c287))
* **blas:** managed FP32 GEMM — GotoBLAS rewrite + CCX-2D-NUMA + short-M campaign (DiT ~2x; sq4096 &gt; throttled MKL) ([#695](https://github.com/ooples/AiDotNet.Tensors/issues/695)) ([5f19acb](https://github.com/ooples/AiDotNet.Tensors/commit/5f19acb843653604bec84c4d4d368a3c878c810a))
* cache per-op env-var reads + skip zero-fill on small RentUninitialized ([#731](https://github.com/ooples/AiDotNet.Tensors/issues/731)) ([61ab50f](https://github.com/ooples/AiDotNet.Tensors/commit/61ab50fae556adc9c47fbbc239d5e69b5667d823))
* **conv:** batch=1 conv-backward-kernel writes gradient straight into dest ([#689](https://github.com/ooples/AiDotNet.Tensors/issues/689)) ([a1d61e0](https://github.com/ooples/AiDotNet.Tensors/commit/a1d61e016e6c68f336bcf9edd635e60c18b6b214))
* **conv:** bounded large-array pool for conv-backward stacks — ~8x less GC traffic ([#1691](https://github.com/ooples/AiDotNet.Tensors/issues/1691) follow-up) ([#694](https://github.com/ooples/AiDotNet.Tensors/issues/694)) ([aeaa5ea](https://github.com/ooples/AiDotNet.Tensors/commit/aeaa5ea360fa7802660bda21861ce8e79b896896))
* **conv:** route large-spatial 3x3 convs to implicit-GEMM (24x forward, 8x CNN step) ([#690](https://github.com/ooples/AiDotNet.Tensors/issues/690)) ([7a8a727](https://github.com/ooples/AiDotNet.Tensors/commit/7a8a727e3b66766fcf8c5b0e9c90029c9b3f6011))
* **cpu:** parallelize 3 linear-attn scans + fused fp16-weight GEMM ([#723](https://github.com/ooples/AiDotNet.Tensors/issues/723)) ([c840501](https://github.com/ooples/AiDotNet.Tensors/commit/c84050122773ac7d35eea46735c71effa6cd4415))
* **dcnv3:** make grouped deformable-conv backward ~2.9× faster (GC + dispatch + lock contention) ([#697](https://github.com/ooples/AiDotNet.Tensors/issues/697)) ([9d60add](https://github.com/ooples/AiDotNet.Tensors/commit/9d60addd4cd8fe939838d6ec59eaacf94b91fc89))
* **gemm:** exceed OpenBLAS — JIT small-GEMM (1.5–6×) + scientific multi-thread routing/pinning wins on diffusion shapes ([#700](https://github.com/ooples/AiDotNet.Tensors/issues/700)) ([ce14327](https://github.com/ooples/AiDotNet.Tensors/commit/ce14327c77f06ea39190b0f55586714ed3fd5b21))
* **geometry:** kill per-element indexer/numOps dispatch in sampling+distance kernels (GridSample ~50x +3) ([#702](https://github.com/ooples/AiDotNet.Tensors/issues/702)) ([1224c7d](https://github.com/ooples/AiDotNet.Tensors/commit/1224c7d16cf5a28bc8d3620ce852a3df9e66ae54))
* **ssm:** parallelize fused SSM/linear-attention scan kernels (Mamba, GLA, RWKV-7, GatedDeltaNet, xLSTM) ([#703](https://github.com/ooples/AiDotNet.Tensors/issues/703)) ([26fb3f4](https://github.com/ooples/AiDotNet.Tensors/commit/26fb3f47f4e7e0d65a7ca79a0a7d527373fc0095))
