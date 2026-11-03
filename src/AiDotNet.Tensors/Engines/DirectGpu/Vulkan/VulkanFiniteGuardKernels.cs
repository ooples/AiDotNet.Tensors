// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1521: the device-side finiteness guard on Vulkan, the GLSL counterpart of OpenCL's (#1502). The multi-tensor sum of
// squares accumulates a float sum and a 2^-80-scaled sum per tensor, then finishes as the double IDeviceScalarOps
// promises (NaN when any element was not finite). The guarded optimizer shaders are the plain ones with the guard bound
// after their buffers and read before any write, so a finite guard leaves the arithmetic untouched.
internal static class VulkanFiniteGuardKernels
{
    // One workgroup of 256 per tensor (the host dispatches 256 threads): each invocation strides the tensor, then
    // invocation 0 adds the partials in order to acc[0] (the sum of squares) and acc[1] (the sum of (x * 2^-80)^2), or
    // sets acc[1] to NaN bits when any element was not finite. Dispatches on the one queue run in order behind a
    // compute barrier, so no atomics are needed. Finiteness is read from the exponent bits, and the NaN mark is kept by
    // its bits rather than by float propagation, which SPIR-V does not guarantee.
    // The scaled sum carries a norm past the float range: when acc[0] overflows (the sum is at least 2^128) the scaled
    // sum is at least 2^-32, a normal float, and it cannot itself overflow below ~4e9 elements at FLT_MAX.
    public const string Accumulate = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) buffer Acc { uint acc[]; };
layout(push_constant) uniform Params { uint n; };
shared float partialSum[256];
shared float partialScaled[256];
shared uint partialBad[256];
void main(){
    if(gl_WorkGroupID.x!=0u)return;
    uint lid=gl_LocalInvocationID.x;
    float down=uintBitsToFloat(0x17800000u);
    float v=0.0;float w=0.0;uint bad=0u;
    for(uint i=lid;i<n;i+=256u){float e=x[i];float d=e*down;v+=e*e;w+=d*d;bad|=(floatBitsToUint(e)&0x7f800000u)==0x7f800000u?1u:0u;}
    partialSum[lid]=v;partialScaled[lid]=w;partialBad[lid]=bad;
    memoryBarrierShared();barrier();
    if(lid==0u){
        float s=0.0;float t=0.0;uint f=0u;
        for(uint j=0u;j<256u;++j){s+=partialSum[j];t+=partialScaled[j];f|=partialBad[j];}
        acc[0]=floatBitsToUint(uintBitsToFloat(acc[0])+s);
        uint prev=acc[1];
        acc[1]=(f!=0u||(prev&0x7f800000u)==0x7f800000u)?0x7fc00000u:floatBitsToUint(uintBitsToFloat(prev)+t);
    }
}
";

    // The accumulated (float sum, scaled sum) as a double built from bits: NaN when the scaled sum is NaN (an element
    // was not finite), the scaled sum times 2^160 when only the float sum overflowed, else the float sum exactly (a
    // subnormal normalised into the double's wider exponent). Writing DBL_MAX for an overflow made the clip
    // coefficient ~0 and erased the gradients, where CUDA scales a 3e19 element to ~0.7.
    public const string Finish = @"#version 450
layout(local_size_x = 256) in;
layout(set=0,binding=0) buffer Acc { uint acc[]; };
void main(){
    if(gl_GlobalInvocationID.x!=0u)return;
    uint u=acc[0];uint e=(u>>23)&0xffu,mant=u&0x7fffffu;uint hi,lo;
    uint su=acc[1];uint se=(su>>23)&0xffu,smant=su&0x7fffffu;
    if(se==0xffu){hi=0x7ff80000u;lo=0u;}
    else if(e==0xffu){hi=se==0u?0x7fefffffu:((se+896u+160u)<<20)|(smant>>3);lo=se==0u?0xffffffffu:smant<<29;}
    else if(e==0u&&mant==0u){hi=0u;lo=0u;}
    else if(e==0u){int shift=23-findMSB(mant);uint m=(mant<<uint(shift))&0x7fffffu;hi=(uint(1-shift-127+1023)<<20)|(m>>3);lo=m<<29;}
    else{hi=((e+896u)<<20)|(mant>>3);lo=mant<<29;}
    acc[0]=lo;acc[1]=hi;
}
";

    private const string GuardTest = "(guardBits[1]&0x7ff00000u)==0x7ff00000u";

    // A plain optimizer shader with the guard (the double's two words) bound at the next binding and tested where the
    // shader drops out-of-range invocations.
    private static string Guarded(string shader, int guardBinding)
    {
        const string Bounds = "if(i>=size)return;";
        int at = shader.IndexOf(Bounds, System.StringComparison.Ordinal);
        if (at < 0 || shader.IndexOf(Bounds, at + 1, System.StringComparison.Ordinal) >= 0)
            throw new System.InvalidOperationException("The optimizer shader's bounds test moved; update the guarded form.");
        const string Header = "layout(local_size_x = 256) in;\n";
        int header = shader.IndexOf(Header, System.StringComparison.Ordinal);
        if (header < 0) throw new System.InvalidOperationException("The optimizer shader's header moved.");
        string guarded = shader.Substring(0, at) + "if(i>=size||" + GuardTest + ")return;" + shader.Substring(at + Bounds.Length);
        return guarded.Insert(header + Header.Length,
            "layout(set=0,binding=" + guardBinding + ") readonly buffer Guard { uint guardBits[]; };\n");
    }

    public static readonly string SgdGuarded = Guarded(VulkanOptimizerKernels.TwoBuffer, 2);
    public static readonly string AdamGuarded = Guarded(VulkanOptimizerKernels.FourBuffer, 4);
}
