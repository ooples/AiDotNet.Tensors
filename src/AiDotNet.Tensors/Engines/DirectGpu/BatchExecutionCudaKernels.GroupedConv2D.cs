using System.Text.RegularExpressions;

namespace AiDotNet.Tensors.Engines.DirectGpu;

internal static partial class BatchExecutionCudaKernels
{
    internal const string GroupedConv2D = "bx_grouped_conv2d";
    internal const string GroupedConv2DInputGrad = "bx_grouped_conv2d_dx";
    internal const string GroupedConv2DKernelGrad = "bx_grouped_conv2d_dw";
    internal static readonly string GroupedConv2DSource =
        GroupedConv2DShaderBodies.CudaKernel(GroupedConv2D, GroupedConv2DShaderBodies.Forward, false)
        + GroupedConv2DShaderBodies.CudaKernel(GroupedConv2DInputGrad, GroupedConv2DShaderBodies.InputGrad, true)
        + GroupedConv2DShaderBodies.CudaKernel(GroupedConv2DKernelGrad, GroupedConv2DShaderBodies.KernelGrad, true);
}

// The three targets share one index formula. The WGSL formatter handles only
// declarations, fixed for-loop headers and uniform names in these bodies.
internal static class GroupedConv2DShaderBodies
{
    internal static readonly string[] Parameters = { "batch", "inChannels", "inHeight", "inWidth", "outChannels",
        "outHeight", "outWidth", "kernelH", "kernelW", "strideH", "strideW", "padH", "padW", "dilationH", "dilationW", "groups" };
    internal const string Forward = @"
int i = bx_gid();
if (i >= batch * outChannels * outHeight * outWidth) return;
int ow = i % outWidth;
int oh = (i / outWidth) % outHeight;
int oc = (i / (outWidth * outHeight)) % outChannels;
int bn = i / (outWidth * outHeight * outChannels);
int cg = inChannels / groups;
int kg = outChannels / groups;
int group = oc / kg;
float sum = 0.0f;
for (int ci = 0; ci < cg; ci += 1) {
    for (int kh = 0; kh < kernelH; kh += 1) {
        int ih = oh * strideH - padH + kh * dilationH;
        if (ih < 0 || ih >= inHeight) continue;
        for (int kw = 0; kw < kernelW; kw += 1) {
            int iw = ow * strideW - padW + kw * dilationW;
            if (iw < 0 || iw >= inWidth) continue;
            int x = ((bn * inChannels + group * cg + ci) * inHeight + ih) * inWidth + iw;
            int w = ((oc * cg + ci) * kernelH + kh) * kernelW + kw;
            sum += first[x] * second[w];
        }
    }
}
resultValues[i] = sum;
";
    internal const string InputGrad = @"
int i = bx_gid();
if (i >= batch * inChannels * inHeight * inWidth) return;
int iw = i % inWidth;
int ih = (i / inWidth) % inHeight;
int ic = (i / (inWidth * inHeight)) % inChannels;
int bn = i / (inWidth * inHeight * inChannels);
int cg = inChannels / groups;
int kg = outChannels / groups;
int group = ic / cg;
int ci = ic % cg;
float sum = 0.0f;
for (int ko = 0; ko < kg; ko += 1) {
    int oc = group * kg + ko;
    for (int kh = 0; kh < kernelH; kh += 1) {
        int ny = ih + padH - kh * dilationH;
        if (ny < 0 || ny % strideH != 0) continue;
        int oh = ny / strideH;
        if (oh >= outHeight) continue;
        for (int kw = 0; kw < kernelW; kw += 1) {
            int nx = iw + padW - kw * dilationW;
            if (nx < 0 || nx % strideW != 0) continue;
            int ow = nx / strideW;
            if (ow >= outWidth) continue;
            int dy = ((bn * outChannels + oc) * outHeight + oh) * outWidth + ow;
            int w = ((oc * cg + ci) * kernelH + kh) * kernelW + kw;
            sum += first[dy] * second[w];
        }
    }
}
if (accumulate != 0) { resultValues[i] += sum; }
else { resultValues[i] = sum; }
";
    internal const string KernelGrad = @"
int i = bx_gid();
int cg = inChannels / groups;
if (i >= outChannels * cg * kernelH * kernelW) return;
int kw = i % kernelW;
int kh = (i / kernelW) % kernelH;
int ci = (i / (kernelW * kernelH)) % cg;
int oc = i / (kernelW * kernelH * cg);
int group = oc / (outChannels / groups);
int ic = group * cg + ci;
float sum = 0.0f;
for (int bn = 0; bn < batch; bn += 1) {
    for (int oh = 0; oh < outHeight; oh += 1) {
        int ih = oh * strideH - padH + kh * dilationH;
        if (ih < 0 || ih >= inHeight) continue;
        for (int ow = 0; ow < outWidth; ow += 1) {
            int iw = ow * strideW - padW + kw * dilationW;
            if (iw < 0 || iw >= inWidth) continue;
            int x = ((bn * inChannels + ic) * inHeight + ih) * inWidth + iw;
            int dy = ((bn * outChannels + oc) * outHeight + oh) * outWidth + ow;
            sum += first[x] * second[dy];
        }
    }
}
if (accumulate != 0) { resultValues[i] += sum; }
else { resultValues[i] = sum; }
";

    internal static string ScalarDeclarations(bool backward)
        => "int " + string.Join("; int ", Parameters) + (backward ? "; int accumulate;" : ";");

    internal static string CudaKernel(string name, string body, bool backward)
        => "\nextern \"C\" __global__ __launch_bounds__(256) void " + name
            + "(const float* first, const float* second, float* resultValues, int "
            + string.Join(", int ", Parameters) + (backward ? ", int accumulate" : "") + ") {\n" + body + "}\n";

    internal static string Wgsl(string body, bool backward)
    {
        string fields = string.Join(", ", System.Array.ConvertAll(Parameters, name => name + ": i32"));
        string formatted = body.Replace("0.0f", "0.0").Replace("bx_gid()", "i32(gid.x)");
        // WGSL requires a compound statement even for a one-statement guard.
        formatted = Regex.Replace(formatted, @"(?m)^([ \t]*if \([^\r\n]*\)) +(return|continue);[ \t]*$", "$1 { $2; }");
        formatted = Regex.Replace(formatted, @"for \(int (\w+) = 0;", "for (var $1: i32 = 0;");
        formatted = Regex.Replace(formatted, @"\bint (\w+) = ", "let $1: i32 = ");
        formatted = Regex.Replace(formatted, @"\bfloat (\w+) = ", "var $1: f32 = ");
        foreach (string name in Parameters)
            formatted = Regex.Replace(formatted, @"\b" + name + @"\b", "params." + name);
        if (backward) formatted = Regex.Replace(formatted, @"\baccumulate\b", "params.accumulate");
        return "struct Params { " + fields + (backward ? ", accumulate: i32" : "") + " };\n"
            + "@group(0) @binding(0) var<storage, read> first: array<f32>;\n"
            + "@group(0) @binding(1) var<storage, read> second: array<f32>;\n"
            + "@group(0) @binding(2) var<storage, read_write> resultValues: array<f32>;\n"
            + "@group(0) @binding(3) var<uniform> params: Params;\n"
            + "@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" + formatted + "}\n";
    }
}
