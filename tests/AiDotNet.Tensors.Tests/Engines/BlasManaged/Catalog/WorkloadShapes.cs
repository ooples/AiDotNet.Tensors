using System.Collections.Generic;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;

/// <summary>
/// Sub-issue A (#369) task A.3: curated standard ML workload shapes.
///
/// <para>
/// These shapes are always included in <see cref="ShapeCatalog.All"/> regardless of
/// instrumentation frequency, so the bench forces coverage of canonical architectures
/// even on test runs that don't exercise them. Sources span 4 architecture families
/// (BERT, GPT-2 medium, ResNet50, MobileNetV2) plus FP64 scientific workloads and
/// transposed backward-pass shapes.
/// </para>
///
/// <para>
/// Frequency is 0 because these shapes are workload-defined, not observation-defined.
/// The merge logic in <see cref="ShapeCatalog"/> keeps the workload version when the
/// same (M, N, K, TransA, TransB, Dtype) tuple also appears in the instrumented set.
/// </para>
/// </summary>
public static class WorkloadShapes
{
    /// <summary>
    /// 25 representative shapes: BERT-base (FFN/attention), GPT-2 medium (FFN/attention),
    /// ResNet50 (im2col-ed forward + backward), MobileNetV2 (pointwise + FC head),
    /// FP64 scientific (regression / PCA / QR), tiny shapes (LSTM cell / embedding).
    /// </summary>
    public static IReadOnlyList<Shape> All { get; } = new[]
    {
        // ── BERT-base ────────────────────────────────────────────────────
        // hidden=768, intermediate=3072, heads=12, head_dim=64, seq=128, batch=8 → M=1024
        new Shape("BERT_FFN_up_1024x3072x768",      1024, 3072, 768,  false, false, DType.Single, 0, "workload:BERT-base FFN expansion"),
        new Shape("BERT_FFN_down_1024x768x3072",    1024, 768,  3072, false, false, DType.Single, 0, "workload:BERT-base FFN contraction"),
        new Shape("BERT_Attn_QKV_1024x768x768",     1024, 768,  768,  false, false, DType.Single, 0, "workload:BERT-base attention QKV proj"),
        new Shape("BERT_Attn_score_96x128x64",      96,   128,  64,   false, false, DType.Single, 0, "workload:BERT-base attention score"),
        new Shape("BERT_Attn_ctx_96x64x128",        96,   64,   128,  false, false, DType.Single, 0, "workload:BERT-base attention context"),

        // ── GPT-2 medium ─────────────────────────────────────────────────
        // hidden=1024, intermediate=4096, batch=4, seq=128 → M=512
        new Shape("GPT2med_FFN_up_512x4096x1024",   512,  4096, 1024, false, false, DType.Single, 0, "workload:GPT-2 medium FFN up"),
        new Shape("GPT2med_FFN_down_512x1024x4096", 512,  1024, 4096, false, false, DType.Single, 0, "workload:GPT-2 medium FFN down"),
        new Shape("GPT2med_Attn_proj_512x1024x1024",512,  1024, 1024, false, false, DType.Single, 0, "workload:GPT-2 medium attention proj"),

        // ── ResNet50 forward (im2col-ed conv) ───────────────────────────
        // M = output H*W, N = output channels, K = kernel * input channels
        new Shape("ResNet50_conv1_3136x64x147",     3136, 64,   147,  false, false, DType.Single, 0, "workload:ResNet50 conv1 7x7"),
        new Shape("ResNet50_layer1_3136x64x64",     3136, 64,   64,   false, false, DType.Single, 0, "workload:ResNet50 layer1 1x1"),
        new Shape("ResNet50_layer2_784x128x128",    784,  128,  128,  false, false, DType.Single, 0, "workload:ResNet50 layer2 3x3"),
        new Shape("ResNet50_layer3_196x256x256",    196,  256,  256,  false, false, DType.Single, 0, "workload:ResNet50 layer3 3x3"),
        new Shape("ResNet50_layer4_49x512x512",     49,   512,  512,  false, false, DType.Single, 0, "workload:ResNet50 layer4 3x3"),
        new Shape("ResNet50_fc_1x1000x2048",        1,    1000, 2048, false, false, DType.Single, 0, "workload:ResNet50 FC head"),

        // ── MobileNetV2 (pointwise convolutions = pure GEMM) ────────────
        new Shape("MobileNetV2_pw_3136x32x32",      3136, 32,   32,   false, false, DType.Single, 0, "workload:MobileNetV2 PW 1x1"),
        new Shape("MobileNetV2_pw_784x144x24",      784,  144,  24,   false, false, DType.Single, 0, "workload:MobileNetV2 PW expand"),
        new Shape("MobileNetV2_pw_196x96x96",       196,  96,   96,   false, false, DType.Single, 0, "workload:MobileNetV2 PW mid"),
        new Shape("MobileNetV2_fc_1x1000x1280",     1,    1000, 1280, false, false, DType.Single, 0, "workload:MobileNetV2 FC head"),

        // ── FP64 scientific workloads ────────────────────────────────────
        new Shape("FP64_Linreg_4096x1024x1024",     4096, 1024, 1024, false, false, DType.Double, 0, "workload:linear regression"),
        new Shape("FP64_PCA_512x512x4096",          512,  512,  4096, false, false, DType.Double, 0, "workload:PCA covariance"),
        new Shape("FP64_QR_2048x2048x256",          2048, 2048, 256,  false, false, DType.Double, 0, "workload:QR decomposition panel"),

        // ── Backward-pass shapes (transposed) ────────────────────────────
        new Shape("BERT_FFN_bwd_dW_3072x768x1024",  3072, 768,  1024, true,  false, DType.Single, 0, "workload:BERT FFN backward dW"),
        new Shape("ResNet50_bwd_dW_64x147x3136",    64,   147,  3136, true,  false, DType.Single, 0, "workload:ResNet50 conv1 backward"),

        // ── Tiny shapes (LSTM cell, embedding lookup) ───────────────────
        new Shape("LSTM_cell_1x256x256",            1,    256,  256,  false, false, DType.Single, 0, "workload:LSTM cell per-timestep"),
        new Shape("LSTM_cell_1x512x512",            1,    512,  512,  false, false, DType.Single, 0, "workload:LSTM cell large"),
        new Shape("GRU_cell_1x768x768",             1,    768,  768,  false, false, DType.Single, 0, "workload:GRU cell"),
        new Shape("Embedding_proj_8x768x768",       8,    768,  768,  false, false, DType.Single, 0, "workload:Embedding projection"),

        // ── BERT batch variants (inference sequence-1, training batch-32) ──
        new Shape("BERT_inf_seq1_1x768x768",        1,    768,  768,  false, false, DType.Single, 0, "workload:BERT inference seq-1"),
        new Shape("BERT_train_b32_4096x3072x768",   4096, 3072, 768,  false, false, DType.Single, 0, "workload:BERT training batch-32"),

        // ── BERT-large (hidden=1024) ────────────────────────────────────
        new Shape("BERTlarge_FFN_up_1024x4096x1024",   1024, 4096, 1024, false, false, DType.Single, 0, "workload:BERT-large FFN up"),
        new Shape("BERTlarge_Attn_QKV_1024x1024x1024", 1024, 1024, 1024, false, false, DType.Single, 0, "workload:BERT-large attention QKV"),

        // ── GPT-2 small (hidden=768) ────────────────────────────────────
        new Shape("GPT2sm_FFN_up_512x3072x768",     512,  3072, 768,  false, false, DType.Single, 0, "workload:GPT-2 small FFN up"),
        new Shape("GPT2sm_Attn_proj_512x768x768",   512,  768,  768,  false, false, DType.Single, 0, "workload:GPT-2 small attention proj"),

        // ── Vision Transformer (ViT-Base, 14x14 patches = 196, +1 cls = 197) ──
        new Shape("ViT_patch_proj_196x768x768",     196,  768,  768,  false, false, DType.Single, 0, "workload:ViT-Base patch projection"),
        new Shape("ViT_Attn_QKV_197x2304x768",      197,  2304, 768,  false, false, DType.Single, 0, "workload:ViT-Base attention QKV (3*768)"),

        // ── Pure compute-bound large shapes (stress test) ───────────────
        new Shape("LargeSquare_2048sq",             2048, 2048, 2048, false, false, DType.Single, 0, "workload:large square FP32"),
        new Shape("LargeSquare_FP64_1024sq",        1024, 1024, 1024, false, false, DType.Double, 0, "workload:large square FP64"),

        // ── Llama 7B (hidden=4096, intermediate=11008, heads=32, head_dim=128) ──
        // SwiGLU FFN = 2 up-projections + 1 down-projection at intermediate=11008.
        // M = batch * seq. Inference (seq=1, batch=1) and training (seq=2048) covered.
        new Shape("Llama7B_FFN_up_2048x11008x4096",   2048, 11008, 4096, false, false, DType.Single, 0, "workload:Llama-7B SwiGLU up (train seq=2048)"),
        new Shape("Llama7B_FFN_down_2048x4096x11008", 2048, 4096,  11008, false, false, DType.Single, 0, "workload:Llama-7B FFN down (train seq=2048)"),
        new Shape("Llama7B_Attn_QKV_2048x12288x4096", 2048, 12288, 4096, false, false, DType.Single, 0, "workload:Llama-7B fused QKV (3*4096)"),
        new Shape("Llama7B_Attn_O_2048x4096x4096",    2048, 4096,  4096, false, false, DType.Single, 0, "workload:Llama-7B attention output"),
        new Shape("Llama7B_inf_seq1_1x4096x4096",     1,    4096,  4096, false, false, DType.Single, 0, "workload:Llama-7B inference seq-1"),
        new Shape("Llama7B_inf_FFN_1x11008x4096",     1,    11008, 4096, false, false, DType.Single, 0, "workload:Llama-7B inference FFN up seq-1"),

        // ── Stable Diffusion UNet (hidden=320 at base, scales to 1280; cross-attn=768) ──
        // Spatial attention shapes at the typical 64x64 latent resolution (4096 tokens).
        // The big GEMMs are inside the ResBlocks (1280→1280) and the FFN inside the
        // Transformer blocks at each resolution.
        new Shape("SD_UNet_resblock_4096x1280x1280",  4096, 1280, 1280, false, false, DType.Single, 0, "workload:SD UNet ResBlock 1280x1280"),
        new Shape("SD_UNet_attn_QKV_4096x960x320",    4096, 960,  320,  false, false, DType.Single, 0, "workload:SD UNet self-attn QKV (3*320)"),
        new Shape("SD_UNet_xattn_4096x768x320",       4096, 768,  320,  false, false, DType.Single, 0, "workload:SD UNet cross-attn K from text"),
        new Shape("SD_UNet_FFN_4096x1280x320",        4096, 1280, 320,  false, false, DType.Single, 0, "workload:SD UNet FFN expansion"),

        // ── DiT / DiT-XL (Stable Diffusion 3 family, hidden=1152, patch_size=2) ──
        new Shape("DiTXL_FFN_up_1024x4608x1152",      1024, 4608, 1152, false, false, DType.Single, 0, "workload:DiT-XL FFN up (4x expansion)"),
        new Shape("DiTXL_Attn_QKV_1024x3456x1152",    1024, 3456, 1152, false, false, DType.Single, 0, "workload:DiT-XL fused QKV"),

        // ── MoE expert routing (typical Mixtral 8x7B expert: hidden=4096, expert_dim=14336) ──
        // Per-expert FFN only sees fraction of tokens. Active fraction ~25% of seq=2048.
        new Shape("MoE_expert_up_512x14336x4096",     512,  14336, 4096, false, false, DType.Single, 0, "workload:Mixtral expert FFN up (25% routing)"),
        new Shape("MoE_expert_down_512x4096x14336",   512,  4096,  14336, false, false, DType.Single, 0, "workload:Mixtral expert FFN down"),

        // ── Whisper encoder (hidden=1280, ~1500 audio tokens, MLP=5120) ──
        new Shape("Whisper_enc_FFN_1500x5120x1280",   1500, 5120, 1280, false, false, DType.Single, 0, "workload:Whisper encoder FFN"),
        new Shape("Whisper_enc_Attn_1500x1280x1280",  1500, 1280, 1280, false, false, DType.Single, 0, "workload:Whisper encoder self-attn"),

        // ── ConvNeXt (modern CNN with depthwise+pointwise+MLP; replaces ResNet50 on some benchmarks) ──
        new Shape("ConvNeXt_stage2_PW_3136x96x96",    3136, 96,   96,   false, false, DType.Single, 0, "workload:ConvNeXt stage-2 pointwise"),
        new Shape("ConvNeXt_stage4_PW_49x768x768",    49,   768,  768,  false, false, DType.Single, 0, "workload:ConvNeXt stage-4 pointwise"),

        // ── EfficientNet B0 (compound-scaled CNN, MBConv pointwise blocks) ──
        new Shape("EffNetB0_PW_3136x16x32",           3136, 16,   32,   false, false, DType.Single, 0, "workload:EfficientNet-B0 PW project"),
        new Shape("EffNetB0_PW_196x1280x192",         196,  1280, 192,  false, false, DType.Single, 0, "workload:EfficientNet-B0 PW final"),

        // ── Backward-pass shapes for modern architectures (transposed) ──
        // dW shapes have inverted (M, K) vs the forward; M = output cols.
        new Shape("Llama7B_FFN_bwd_dW_11008x4096x2048", 11008, 4096, 2048, true,  false, DType.Single, 0, "workload:Llama-7B FFN backward dW"),
        new Shape("SD_UNet_bwd_dW_1280x1280x4096",      1280,  1280, 4096, true,  false, DType.Single, 0, "workload:SD UNet ResBlock backward dW"),
    };
}
