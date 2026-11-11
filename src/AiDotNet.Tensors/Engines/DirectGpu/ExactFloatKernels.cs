// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Exact float comparison and selection helpers for GPU kernel sources, one copy per kernel dialect.
/// </summary>
/// <remarks>
/// <para>
/// Comparisons, min/max, clamp, sign, abs, where-select and arg-extrema must give the CPU engine's answer for EVERY
/// float: signed zeros, subnormals, infinities and NaN. Float instructions cannot guarantee that on a GPU. Fast-math
/// builds (OpenCL <c>-cl-fast-relaxed-math -cl-finite-math-only</c>, CUDA/HIP <c>--use_fast_math</c>) flush
/// subnormals to zero in comparisons and min/max and assume NaN and infinity never occur, and Vulkan and WebGPU let the
/// hardware flush subnormals whatever the shader says. On an NVIDIA GPU under OpenCL, <c>1.4e-45f &gt; 0.0f</c> was
/// false while the CPU said true.
/// </para>
/// <para>
/// These helpers therefore work on the IEEE-754 bit pattern with integer instructions only. The compiler's float
/// assumptions do not apply to integers, and the hardware never flushes an integer. Every value is ordered by
/// <c>xf_key</c>: the magnitude bits, negated when the sign bit is set. That ordering is monotone over all non-NaN
/// floats, and it maps -0 and +0 to the same key, so they compare equal as IEEE says. NaN is detected as a magnitude
/// above the infinity pattern and is unordered: every comparison involving it is false.
/// </para>
/// <para>
/// The CPU semantics the helpers reproduce:
/// <list type="bullet">
/// <item><c>max(a, b) = a &gt; b ? a : b</c> and <c>min(a, b) = a &lt; b ? a : b</c>: x86 MAXPS/MINPS, which the
/// CPU engine's SIMD kernels run. The second operand wins when either is NaN and when both are zeros.</item>
/// <item><c>clamp(x, lo, hi)</c>: <c>c = x &lt; lo ? lo : x</c>, then <c>c &gt; hi ? hi : c</c>. A NaN input stays
/// NaN, and a NaN bound is ignored.</item>
/// <item><c>sign(x)</c>: 1 for x &gt; 0, -1 for x &lt; 0, otherwise +0. That covers both zeros and NaN.</item>
/// <item><c>abs(x)</c>: the sign bit cleared. <c>-x</c>: the sign bit flipped (IEEE negate, which amin = -max(-x)
/// relies on).</item>
/// <item>max reductions: from -inf, a value is taken only when strictly greater, so NaN is skipped and an all-NaN
/// input gives -inf; arg-max / arg-min start at the first element, so the first of equal extrema wins.</item>
/// <item>where/select: the condition is true when it is not ±0 (NaN counts as true). The chosen operand is passed
/// through bit for bit.</item>
/// </list>
/// Values are selected as bit patterns, never through a float move a compiler could turn into a flushing min/max.
/// </para>
/// <para>
/// The integer instructions cost about the same as the float compare they replace. The arithmetic kernels that
/// legitimately use fast math are untouched. The C-family copies carry an include guard, so a translation unit that
/// concatenates two sources holding them still compiles.
/// </para>
/// </remarks>
internal static class ExactFloatKernels
{
    /// <summary>The helpers in OpenCL C. Operands and results are <c>uint</c> bit patterns (<c>as_uint</c>/<c>as_float</c>).</summary>
    public const string OpenCl = @"
#ifndef AIDN_EXACT_FLOAT_HELPERS
#define AIDN_EXACT_FLOAT_HELPERS
// Exact float predicates on the IEEE-754 bit pattern (integer instructions only): immune to fast/finite-math flags
// and to subnormal flushing. See ExactFloatKernels.cs.
inline int xf_isnan(uint b) { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
inline int xf_key(uint b) { int m = (int)(b & 0x7FFFFFFFu); return (b & 0x80000000u) != 0u ? -m : m; }
inline int xf_gt(uint a, uint b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) > xf_key(b); }
inline int xf_lt(uint a, uint b) { return xf_gt(b, a); }
inline int xf_ge(uint a, uint b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) >= xf_key(b); }
inline int xf_le(uint a, uint b) { return xf_ge(b, a); }
inline int xf_eq(uint a, uint b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) == xf_key(b); }
inline uint xf_max(uint a, uint b) { return xf_gt(a, b) ? a : b; }
inline uint xf_min(uint a, uint b) { return xf_lt(a, b) ? a : b; }
inline uint xf_clamp(uint x, uint lo, uint hi) { uint c = xf_lt(x, lo) ? lo : x; return xf_gt(c, hi) ? hi : c; }
inline uint xf_sign(uint x) { return xf_isnan(x) ? 0u : (xf_key(x) > 0 ? 0x3F800000u : (xf_key(x) < 0 ? 0xBF800000u : 0u)); }
inline uint xf_abs(uint x) { return x & 0x7FFFFFFFu; }
inline int xf_truthy(uint c) { return (c & 0x7FFFFFFFu) != 0u; }
inline float xf_bool(int c) { return c ? 1.0f : 0.0f; }
#endif
";

    /// <summary>
    /// The helpers in CUDA C++, which HIP compiles unchanged. Operands and results are <c>unsigned int</c> bit
    /// patterns (<c>__float_as_uint</c>/<c>__uint_as_float</c>).
    /// </summary>
    public const string Cuda = @"
#ifndef AIDN_EXACT_FLOAT_HELPERS
#define AIDN_EXACT_FLOAT_HELPERS
// Exact float predicates on the IEEE-754 bit pattern (integer instructions only): immune to --use_fast_math (which
// flushes subnormals in float compares and fminf/fmaxf) and to finite-math assumptions. See ExactFloatKernels.cs.
__device__ __forceinline__ bool xf_isnan(unsigned int b) { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
__device__ __forceinline__ int xf_key(unsigned int b) { int m = (int)(b & 0x7FFFFFFFu); return (b & 0x80000000u) != 0u ? -m : m; }
__device__ __forceinline__ bool xf_gt(unsigned int a, unsigned int b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) > xf_key(b); }
__device__ __forceinline__ bool xf_lt(unsigned int a, unsigned int b) { return xf_gt(b, a); }
__device__ __forceinline__ bool xf_ge(unsigned int a, unsigned int b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) >= xf_key(b); }
__device__ __forceinline__ bool xf_le(unsigned int a, unsigned int b) { return xf_ge(b, a); }
__device__ __forceinline__ bool xf_eq(unsigned int a, unsigned int b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) == xf_key(b); }
__device__ __forceinline__ unsigned int xf_max(unsigned int a, unsigned int b) { return xf_gt(a, b) ? a : b; }
__device__ __forceinline__ unsigned int xf_min(unsigned int a, unsigned int b) { return xf_lt(a, b) ? a : b; }
__device__ __forceinline__ unsigned int xf_clamp(unsigned int x, unsigned int lo, unsigned int hi) { unsigned int c = xf_lt(x, lo) ? lo : x; return xf_gt(c, hi) ? hi : c; }
__device__ __forceinline__ unsigned int xf_sign(unsigned int x) { return xf_isnan(x) ? 0u : (xf_key(x) > 0 ? 0x3F800000u : (xf_key(x) < 0 ? 0xBF800000u : 0u)); }
__device__ __forceinline__ unsigned int xf_abs(unsigned int x) { return x & 0x7FFFFFFFu; }
__device__ __forceinline__ bool xf_truthy(unsigned int c) { return (c & 0x7FFFFFFFu) != 0u; }
// Float-in / float-out forms for elementwise bodies: the operands only move, they are never computed on.
__device__ __forceinline__ float xf_maxf(float a, float b) { return __uint_as_float(xf_max(__float_as_uint(a), __float_as_uint(b))); }
__device__ __forceinline__ float xf_minf(float a, float b) { return __uint_as_float(xf_min(__float_as_uint(a), __float_as_uint(b))); }
__device__ __forceinline__ float xf_clampf(float x, float lo, float hi) { return __uint_as_float(xf_clamp(__float_as_uint(x), __float_as_uint(lo), __float_as_uint(hi))); }
__device__ __forceinline__ float xf_signf(float x) { return __uint_as_float(xf_sign(__float_as_uint(x))); }
__device__ __forceinline__ float xf_absf(float x) { return __uint_as_float(xf_abs(__float_as_uint(x))); }
__device__ __forceinline__ float xf_negf(float x) { return __uint_as_float(__float_as_uint(x) ^ 0x80000000u); }
__device__ __forceinline__ float xf_gtf(float a, float b) { return xf_gt(__float_as_uint(a), __float_as_uint(b)) ? 1.0f : 0.0f; }
__device__ __forceinline__ float xf_ltf(float a, float b) { return xf_lt(__float_as_uint(a), __float_as_uint(b)) ? 1.0f : 0.0f; }
__device__ __forceinline__ float xf_selectf(float c, float a, float b) { return __uint_as_float(xf_truthy(__float_as_uint(c)) ? __float_as_uint(a) : __float_as_uint(b)); }
#endif
";

    /// <summary>
    /// The helpers in GLSL 450 for the Vulkan shaders, which go after the <c>#version</c> line. Operands and results
    /// are <c>uint</c> bit patterns (<c>floatBitsToUint</c>/<c>uintBitsToFloat</c>). Vulkan lets the hardware flush
    /// subnormals in float compares and min/max whatever the shader asks, so only integer compares are exact there.
    /// </summary>
    public const string Glsl = @"
#ifndef AIDN_EXACT_FLOAT_HELPERS
#define AIDN_EXACT_FLOAT_HELPERS
bool xf_isnan(uint b) { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
int xf_key(uint b) { int m = int(b & 0x7FFFFFFFu); return (b & 0x80000000u) != 0u ? -m : m; }
bool xf_gt(uint a, uint b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) > xf_key(b); }
bool xf_lt(uint a, uint b) { return xf_gt(b, a); }
bool xf_eq(uint a, uint b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) == xf_key(b); }
uint xf_max(uint a, uint b) { return xf_gt(a, b) ? a : b; }
uint xf_min(uint a, uint b) { return xf_lt(a, b) ? a : b; }
uint xf_clamp(uint x, uint lo, uint hi) { uint c = xf_lt(x, lo) ? lo : x; return xf_gt(c, hi) ? hi : c; }
uint xf_sign(uint x) { return xf_isnan(x) ? 0u : (xf_key(x) > 0 ? 0x3F800000u : (xf_key(x) < 0 ? 0xBF800000u : 0u)); }
bool xf_truthy(uint c) { return (c & 0x7FFFFFFFu) != 0u; }
#endif
";

    /// <summary>
    /// The helpers in Metal Shading Language. Operands and results are <c>uint</c> bit patterns
    /// (<c>as_type&lt;uint&gt;</c>/<c>as_type&lt;float&gt;</c>), with float-in / float-out forms for elementwise bodies.
    /// MSL leaves fp32 subnormal handling to the GPU, which may flush them in compares and min/max.
    /// </summary>
    public const string Msl = @"
#ifndef AIDN_EXACT_FLOAT_HELPERS
#define AIDN_EXACT_FLOAT_HELPERS
inline bool xf_isnan(uint b) { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
inline int xf_key(uint b) { int m = int(b & 0x7FFFFFFFu); return (b & 0x80000000u) != 0u ? -m : m; }
inline bool xf_gt(uint a, uint b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) > xf_key(b); }
inline bool xf_lt(uint a, uint b) { return xf_gt(b, a); }
inline bool xf_eq(uint a, uint b) { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) == xf_key(b); }
inline uint xf_max(uint a, uint b) { return xf_gt(a, b) ? a : b; }
inline uint xf_min(uint a, uint b) { return xf_lt(a, b) ? a : b; }
inline uint xf_clamp(uint x, uint lo, uint hi) { uint c = xf_lt(x, lo) ? lo : x; return xf_gt(c, hi) ? hi : c; }
inline uint xf_sign(uint x) { return xf_isnan(x) ? 0u : (xf_key(x) > 0 ? 0x3F800000u : (xf_key(x) < 0 ? 0xBF800000u : 0u)); }
inline bool xf_truthy(uint c) { return (c & 0x7FFFFFFFu) != 0u; }
inline float xf_maxf(float a, float b) { return as_type<float>(xf_max(as_type<uint>(a), as_type<uint>(b))); }
inline float xf_minf(float a, float b) { return as_type<float>(xf_min(as_type<uint>(a), as_type<uint>(b))); }
inline float xf_clampf(float x, float lo, float hi) { return as_type<float>(xf_clamp(as_type<uint>(x), as_type<uint>(lo), as_type<uint>(hi))); }
inline float xf_signf(float x) { return as_type<float>(xf_sign(as_type<uint>(x))); }
inline float xf_absf(float x) { return as_type<float>(as_type<uint>(x) & 0x7FFFFFFFu); }
inline float xf_negf(float x) { return as_type<float>(as_type<uint>(x) ^ 0x80000000u); }
inline bool xf_gtf(float a, float b) { return xf_gt(as_type<uint>(a), as_type<uint>(b)); }
inline bool xf_ltf(float a, float b) { return xf_lt(as_type<uint>(a), as_type<uint>(b)); }
inline float xf_selectf(float c, float a, float b) { return as_type<float>(xf_truthy(as_type<uint>(c)) ? as_type<uint>(a) : as_type<uint>(b)); }
#endif
";

    /// <summary>
    /// The helpers in WGSL. Operands and results are <c>u32</c> bit patterns (<c>bitcast&lt;u32&gt;</c> and
    /// <c>bitcast&lt;f32&gt;</c>). WGSL allows an implementation to flush subnormals and to assume that NaN and infinity
    /// never occur, so it may fold a float NaN test away. WGSL has no preprocessor, so a module includes this text
    /// exactly once.
    /// </summary>
    public const string Wgsl = @"
// Exact float predicates on the IEEE-754 bit pattern (integer instructions only). See ExactFloatKernels.cs.
fn xf_isnan(b: u32) -> bool { return (b & 0x7FFFFFFFu) > 0x7F800000u; }
fn xf_key(b: u32) -> i32 { let m = i32(b & 0x7FFFFFFFu); return select(m, -m, (b & 0x80000000u) != 0u); }
fn xf_gt(a: u32, b: u32) -> bool { return !xf_isnan(a) && !xf_isnan(b) && xf_key(a) > xf_key(b); }
fn xf_lt(a: u32, b: u32) -> bool { return xf_gt(b, a); }
fn xf_max(a: u32, b: u32) -> u32 { return select(b, a, xf_gt(a, b)); }
fn xf_min(a: u32, b: u32) -> u32 { return select(b, a, xf_lt(a, b)); }
fn xf_clamp(x: u32, lo: u32, hi: u32) -> u32 { let c = select(x, lo, xf_lt(x, lo)); return select(c, hi, xf_gt(c, hi)); }
fn xf_sign(x: u32) -> u32 {
    if (xf_isnan(x)) { return 0u; }
    let k = xf_key(x);
    if (k > 0) { return 0x3F800000u; }
    if (k < 0) { return 0xBF800000u; }
    return 0u;
}
fn xf_truthy(c: u32) -> bool { return (c & 0x7FFFFFFFu) != 0u; }
";
}
