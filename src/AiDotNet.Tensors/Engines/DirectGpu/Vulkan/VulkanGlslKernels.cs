namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GLSL compute shader source strings for Vulkan fused kernels.
/// Compiled to SPIR-V at runtime via VulkanGlslCompiler (libshaderc).
/// Each kernel is a standalone compute shader with push constants for parameters.
/// </summary>
public static class VulkanGlslKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    private const string TwoBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) writeonly buffer B { float b[]; };
";

    private const string ThreeBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float bdata[]; };
layout(set = 0, binding = 2) writeonly buffer C { float c[]; };
";

    public static string UnaryElementwise => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; uint op; uint p0; uint p1; uint p2; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx];
    float v0 = uintBitsToFloat(p0), v1 = uintBitsToFloat(p1), v2 = uintBitsToFloat(p2);
    float y = 0.0;
    switch (op) {
        case 0u: y = pow(x, v0); break;
        case 1u: y = abs(x); break;
        case 2u: y = exp(x); break;
        case 3u: y = exp2(x); break;
        case 4u: y = pow(10.0, x); break;
        case 5u: y = exp(x) - 1.0; break;
        case 6u: y = log(x); break;
        case 7u: y = log2(x); break;
        case 8u: y = log(1.0 + x); break;
        case 9u: y = sqrt(x); break;
        case 10u: y = sign(x); break;
        case 11u: y = 0.5*x*(1.0+tanh(0.7978845608*(x+0.044715*x*x*x))); break;
        case 12u: y = sin(x); break;
        case 13u: y = cos(x); break;
        case 14u: y = tan(x); break;
        case 15u: y = asin(x); break;
        case 16u: y = acos(x); break;
        case 17u: y = atan(x); break;
        case 18u: y = sinh(x); break;
        case 19u: y = cosh(x); break;
        case 20u: y = asinh(x); break;
        case 21u: y = acosh(x); break;
        case 22u: y = atanh(x); break;
        case 23u: y = 1.0 / x; break;
        case 24u: y = sign(x) * pow(abs(x), 0.3333333333333333); break;
        case 25u: y = log(x) * 0.4342944819032518; break;
        case 26u: y = -x; break;
        case 27u: y = floor(x); break;
        case 28u: y = ceil(x); break;
        case 29u: y = roundEven(x); break;
        case 30u: y = trunc(x); break;
        case 31u: y = x >= 0.0 ? x : v0 * x; break;
        case 32u: y = x >= 0.0 ? x : v0 * (exp(x) - 1.0); break;
        case 33u: y = x / (1.0 + exp(-x)); break;
        case 34u: y = x * tanh(log(1.0 + exp(x))); break;
        case 35u: y = log(1.0 + exp(x)); break;
        case 36u: y = x <= -3.0 ? 0.0 : (x >= 3.0 ? x : x * (x + 3.0) / 6.0); break;
        case 37u: y = v1 * (x >= 0.0 ? x : v0 * (exp(x) - 1.0)); break;
        case 38u: y = clamp(x / 6.0 + 0.5, 0.0, 1.0); break;
        case 39u: y = clamp(x, v0, v1); break;
        case 40u: {
            uint xb = floatBitsToUint(x), vb = p0;
            uint xa = xb & 0x7FFFFFFFu, va = vb & 0x7FFFFFFFu;
            bool equal = xa <= 0x7F800000u && va <= 0x7F800000u
                && (xb == vb || ((xa | va) == 0u));
            y = equal ? 0.0 : 1.0;
            break;
        }
        case 41u: y = max(v1, 10.0 * log(max(1e-10, x / v0)) * 0.4342944819032518); break;
        case 42u: y = v0 * pow(10.0, x / 10.0); break;
        default: y = x; break;
    }
    b[idx] = y;
}";

    public static string BinaryElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; uint op; uint p0; uint p1; uint p2; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx], y = bdata[idx];
    float v0 = uintBitsToFloat(p0), v1 = uintBitsToFloat(p1);
    float z = 0.0;
    switch (op) {
        case 0u: z = min(x, y); break;
        case 1u: z = max(x, y); break;
        case 2u: z = x * (y >= 0.0 ? 1.0 : v0); break;
        case 3u: z = y > 0.0 ? x : 0.0; break;
        case 4u: z = x * y * (1.0 - y); break;
        case 5u: z = x * (1.0 - y * y); break;
        case 6u: z = x / (1.0 + exp(-y)); break;
        case 7u: z = x * (y <= -3.0 ? 0.0 : (y >= 3.0 ? 1.0 : (2.0*y+3.0)/6.0)); break;
        case 8u: z = x * v1 * (y >= 0.0 ? 1.0 : v0 * exp(y)); break;
        case 9u: z = x * (y > -3.0 && y < 3.0 ? 0.1666666666666667 : 0.0); break;
        case 10u: z = y > v0 && y < v1 ? x : 0.0; break;
        case 11u: z = x > y ? 1.0 : 0.0; break;
        case 12u: z = x < y ? 1.0 : 0.0; break;
        case 13u: {
            uint xb = floatBitsToUint(x), yb = floatBitsToUint(y);
            uint xa = xb & 0x7FFFFFFFu, ya = yb & 0x7FFFFFFFu;
            bool equal = xa <= 0x7F800000u && ya <= 0x7F800000u
                && (xb == yb || ((xa | ya) == 0u));
            z = equal ? 1.0 : 0.0;
            break;
        }
        case 14u: z = fma(v0, y - x, x); break;
        case 15u: z = fma(v0, x, v1 * y); break;
        case 16u: z = sqrt(x*x + y*y); break;
        case 17u: z = atan(y, x); break;
        case 18u: z = x * y; break;
        case 19u: {
            float sig = 1.0 / (1.0 + exp(-y));
            z = x * (sig + y * sig * (1.0 - sig));
            break;
        }
        case 20u: {
            float t = tanh(0.7978845608 * (y + 0.044715 * y*y*y));
            float dtdy = 0.7978845608 * (1.0 + 0.134145 * y*y) * (1.0 - t*t);
            z = x * (0.5 * (1.0 + t) + 0.5 * y * dtdy);
            break;
        }
        case 21u: {
            float sp = log(1.0 + exp(y));
            float tsp = tanh(sp);
            float sig = 1.0 / (1.0 + exp(-y));
            z = x * (tsp + y * sig * (1.0 - tsp*tsp));
            break;
        }
        default: z = x; break;
    }
    c[idx] = z;
}";

    public static string EluBackward => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint size; uint alphaBits; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float alpha = uintBitsToFloat(alphaBits);
    d[idx] = a[idx] * (bdata[idx] >= 0.0 ? 1.0 : c[idx] + alpha);
}";

    public static string Fma => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    d[idx] = a[idx] * bdata[idx] + c[idx];
}";

    public static string Where => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    d[idx] = a[idx] != 0.0 ? bdata[idx] : c[idx];
}";

    public static string SoftmaxRows => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint rows; uint cols; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= rows) return;
    uint offset = row * cols;
    float rowMax = -3.402823466e+38;
    for (uint col = 0u; col < cols; ++col) rowMax = max(rowMax, a[offset + col]);
    float sum = 0.0;
    for (uint col = 0u; col < cols; ++col) sum += exp(a[offset + col] - rowMax);
    for (uint col = 0u; col < cols; ++col) b[offset + col] = exp(a[offset + col] - rowMax) / sum;
}";

    public static string SoftmaxBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint rows; uint cols; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= rows) return;
    uint offset = row * cols;
    float dot = 0.0;
    for (uint col = 0u; col < cols; ++col) dot += a[offset + col] * bdata[offset + col];
    for (uint col = 0u; col < cols; ++col) {
        uint idx = offset + col;
        c[idx] = bdata[idx] * (a[idx] - dot);
    }
}";

    public static string BiasAdd => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; uint cols; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = a[idx] + bdata[idx % cols];
}";

    public static string Conv2DBiasAdd => Header + @"
layout(set = 0, binding = 0) buffer Output { float outputData[]; };
layout(set = 0, binding = 1) readonly buffer Bias { float biasData[]; };
layout(push_constant) uniform Params { uint size; uint channels; uint spatialSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    uint channel = (idx / spatialSize) % channels;
    outputData[idx] += biasData[channel];
}";

    public static string StridedDot => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint aSize; uint bSize; int bOffset; int bStride; };
void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    float sum = 0.0;
    for (uint i = 0u; i < aSize; ++i) {
        int bIndex = bOffset + int(i) * bStride;
        if (bIndex >= 0 && bIndex < int(bSize)) sum += a[i] * bdata[bIndex];
    }
    c[0] = sum;
}";

    public static string Transpose => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint rows; uint cols; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint matrixSize = rows * cols;
    if (idx >= batch * matrixSize) return;
    uint matrix = idx / matrixSize;
    uint local = idx % matrixSize;
    uint row = local / cols;
    uint col = local % cols;
    b[matrix * matrixSize + col * rows + row] = a[idx];
}";

    public static string Copy2DStrided => Header + @"
layout(set = 0, binding = 0) readonly buffer Source { float sourceData[]; };
layout(set = 0, binding = 1) buffer Destination { float destinationData[]; };
layout(push_constant) uniform Params { uint rows; uint srcCols; uint destCols; uint destOffset; uint copyCols; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rows * copyCols) return;
    uint row = idx / copyCols;
    uint col = idx % copyCols;
    destinationData[row * destCols + destOffset + col] = sourceData[row * srcCols + col];
}";

    public static string NearestNeighborUpsample => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batchChannels; uint height; uint width; uint scaleFactor; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint outHeight = height * scaleFactor, outWidth = width * scaleFactor;
    if (idx >= batchChannels * outHeight * outWidth) return;
    uint plane = idx / (outHeight * outWidth);
    uint local = idx % (outHeight * outWidth);
    uint outRow = local / outWidth, outCol = local % outWidth;
    b[idx] = a[plane * height * width + (outRow / scaleFactor) * width + outCol / scaleFactor];
}";

    public static string NearestNeighborUpsampleBackward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batchChannels; uint height; uint width; uint scaleFactor; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchChannels * height * width) return;
    uint plane = idx / (height * width);
    uint local = idx % (height * width);
    uint inRow = local / width, inCol = local % width;
    uint outHeight = height * scaleFactor, outWidth = width * scaleFactor;
    float sum = 0.0;
    for (uint dy = 0u; dy < scaleFactor; ++dy)
        for (uint dx = 0u; dx < scaleFactor; ++dx)
            sum += a[plane * outHeight * outWidth + (inRow * scaleFactor + dy) * outWidth + inCol * scaleFactor + dx];
    b[idx] = sum;
}";

    public static string NearestNeighborUpsample3D => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint depth; uint height; uint width; uint scaleD; uint scaleH; uint scaleW; };
void main(){uint idx=gl_GlobalInvocationID.x;uint outD=depth*scaleD,outH=height*scaleH,outW=width*scaleW;if(idx>=batch*channels*outD*outH*outW)return;uint ow=idx%outW,q=idx/outW,oh=q%outH;q/=outH;uint od=q%outD;q/=outD;uint channel=q%channels,batchIndex=q/channels;b[idx]=a[(((batchIndex*channels+channel)*depth+od/scaleD)*height+oh/scaleH)*width+ow/scaleW];}";

    public static string NearestNeighborUpsample3DBackward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint depth; uint height; uint width; uint scaleD; uint scaleH; uint scaleW; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*channels*depth*height*width)return;uint iw=idx%width,q=idx/width,ih=q%height;q/=height;uint id=q%depth;q/=depth;uint channel=q%channels,batchIndex=q/channels;uint outD=depth*scaleD,outH=height*scaleH,outW=width*scaleW;float sum=0.0;for(uint dz=0u;dz<scaleD;++dz)for(uint dy=0u;dy<scaleH;++dy)for(uint dx=0u;dx<scaleW;++dx)sum+=a[(((batchIndex*channels+channel)*outD+id*scaleD+dz)*outH+ih*scaleH+dy)*outW+iw*scaleW+dx];b[idx]=sum;}";

    public static string AffineGrid2D => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint outputHeight; uint outputWidth; };
void main(){uint point=gl_GlobalInvocationID.x;if(point>=batch*outputHeight*outputWidth)return;uint w=point%outputWidth,q=point/outputWidth,h=q%outputHeight,batchIndex=q/outputHeight;float ny=outputHeight>1u?2.0*float(h)/float(outputHeight-1u)-1.0:0.0;float nx=outputWidth>1u?2.0*float(w)/float(outputWidth-1u)-1.0:0.0;uint t=batchIndex*6u;b[point*2u]=a[t]*nx+a[t+1u]*ny+a[t+2u];b[point*2u+1u]=a[t+3u]*nx+a[t+4u]*ny+a[t+5u];}";

    private const string GridSampleHelpers = @"
int gridPad(int coordinate,int size,uint paddingMode){if(coordinate>=0&&coordinate<size)return coordinate;if(paddingMode==0u)return -1;if(paddingMode==1u)return clamp(coordinate,0,size-1);if(size<=1)return 0;int period=2*(size-1),value=coordinate%period;if(value<0)value+=period;return value<size?value:period-value;}
";

    public static string GridSampleBackwardInput => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inHeight; uint inWidth; uint outHeight; uint outWidth; uint paddingMode; uint alignCorners; };
" + GridSampleHelpers + @"
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*channels*inHeight*inWidth)return;uint ixTarget=idx%inWidth,q=idx/inWidth,iyTarget=q%inHeight;q/=inHeight;uint channel=q%channels,batchIndex=q/channels;float sum=0.0;for(uint oh=0u;oh<outHeight;++oh)for(uint ow=0u;ow<outWidth;++ow){uint gridOffset=(batchIndex*outHeight*outWidth+oh*outWidth+ow)*2u;float gx=bdata[gridOffset],gy=bdata[gridOffset+1u];float ix=alignCorners!=0u?(gx+1.0)*0.5*float(inWidth-1u):(gx+1.0)*0.5*float(inWidth)-0.5;float iy=alignCorners!=0u?(gy+1.0)*0.5*float(inHeight-1u):(gy+1.0)*0.5*float(inHeight)-0.5;int ix0=int(floor(ix)),iy0=int(floor(iy));float dx=ix-float(ix0),dy=iy-float(iy0);for(int jy=0;jy<=1;++jy)for(int jx=0;jx<=1;++jx){int py=gridPad(iy0+jy,int(inHeight),paddingMode),px=gridPad(ix0+jx,int(inWidth),paddingMode);if(py==int(iyTarget)&&px==int(ixTarget)){float wy=jy==0?1.0-dy:dy,wx=jx==0?1.0-dx:dx;sum+=a[((batchIndex*channels+channel)*outHeight+oh)*outWidth+ow]*wy*wx;}}}c[idx]=sum;}";

    public static string GridSampleBackwardGrid => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inHeight; uint inWidth; uint outHeight; uint outWidth; uint paddingMode; uint alignCorners; };
" + GridSampleHelpers + @"
void main(){uint point=gl_GlobalInvocationID.x;if(point>=batch*outHeight*outWidth)return;uint ow=point%outWidth,q=point/outWidth,oh=q%outHeight,batchIndex=q/outHeight;float gx=c[point*2u],gy=c[point*2u+1u];float ix=alignCorners!=0u?(gx+1.0)*0.5*float(inWidth-1u):(gx+1.0)*0.5*float(inWidth)-0.5;float iy=alignCorners!=0u?(gy+1.0)*0.5*float(inHeight-1u):(gy+1.0)*0.5*float(inHeight)-0.5;float dix=alignCorners!=0u?0.5*float(inWidth-1u):0.5*float(inWidth),diy=alignCorners!=0u?0.5*float(inHeight-1u):0.5*float(inHeight);int ix0=int(floor(ix)),iy0=int(floor(iy));float dx=ix-float(ix0),dy=iy-float(iy0),gradX=0.0,gradY=0.0;for(uint channel=0u;channel<channels;++channel){float grad=a[((batchIndex*channels+channel)*outHeight+oh)*outWidth+ow];for(int jy=0;jy<=1;++jy)for(int jx=0;jx<=1;++jx){int py=gridPad(iy0+jy,int(inHeight),paddingMode),px=gridPad(ix0+jx,int(inWidth),paddingMode);if(py>=0&&px>=0){float wy=jy==0?1.0-dy:dy,wx=jx==0?1.0-dx:dx,pixel=bdata[((batchIndex*channels+channel)*inHeight+uint(py))*inWidth+uint(px)];gradY+=grad*pixel*(jy==0?-1.0:1.0)*wx;gradX+=grad*pixel*wy*(jx==0?-1.0:1.0);}}}d[point*2u]=gradX*dix;d[point*2u+1u]=gradY*diy;}";

    public static string TileBatch => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint repeats; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= repeats * innerSize) return;
    b[idx] = a[idx % innerSize];
}";

    public static string CapsuleSquash => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint numCapsules; uint capsuleDim; float epsilon; };
void main() {
    uint capsule = gl_GlobalInvocationID.x;
    if (capsule >= numCapsules) return;
    uint offset = capsule * capsuleDim;
    float squaredNorm = 0.0;
    for (uint d0 = 0u; d0 < capsuleDim; ++d0) squaredNorm += a[offset + d0] * a[offset + d0];
    float scale = squaredNorm / ((1.0 + squaredNorm) * sqrt(squaredNorm + epsilon));
    for (uint d0 = 0u; d0 < capsuleDim; ++d0) b[offset + d0] = a[offset + d0] * scale;
}";

    public static string CapsuleSquashBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint numCapsules; uint capsuleDim; float epsilon; };
void main() {
    uint capsule = gl_GlobalInvocationID.x;
    if (capsule >= numCapsules) return;
    uint offset = capsule * capsuleDim;
    float squaredNorm = 0.0;
    for (uint d0 = 0u; d0 < capsuleDim; ++d0) squaredNorm += bdata[offset + d0] * bdata[offset + d0];
    float norm = sqrt(squaredNorm + epsilon);
    float denominator = (1.0 + squaredNorm) * norm;
    float factor = (squaredNorm + 2.0 * squaredNorm / (1.0 + squaredNorm)) / (denominator * denominator);
    for (uint d0 = 0u; d0 < capsuleDim; ++d0) c[offset + d0] = a[offset + d0] * factor;
}";

    public static string CapsulePredictions => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint inputCapsules; uint inputDim; uint outputCapsules; uint outputDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = batchSize * inputCapsules * outputCapsules * outputDim;
    if (idx >= total) return;
    uint outputDimIndex = idx % outputDim;
    uint q = idx / outputDim;
    uint outputCapsule = q % outputCapsules;
    q /= outputCapsules;
    uint inputCapsule = q % inputCapsules;
    uint batch = q / inputCapsules;
    float sum = 0.0;
    for (uint inputDimIndex = 0u; inputDimIndex < inputDim; ++inputDimIndex)
        sum += a[(batch * inputCapsules + inputCapsule) * inputDim + inputDimIndex] *
            bdata[((inputCapsule * outputCapsules + outputCapsule) * outputDim + outputDimIndex) * inputDim + inputDimIndex];
    c[idx] = sum;
}";

    public static string CapsuleWeightedSum => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint inputCapsules; uint outputCapsules; uint capsuleDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = batchSize * outputCapsules * capsuleDim;
    if (idx >= total) return;
    uint dim = idx % capsuleDim;
    uint q = idx / capsuleDim;
    uint outputCapsule = q % outputCapsules;
    uint batch = q / outputCapsules;
    float sum = 0.0;
    for (uint inputCapsule = 0u; inputCapsule < inputCapsules; ++inputCapsule)
        sum += a[(batch * inputCapsules + inputCapsule) * outputCapsules + outputCapsule] *
            bdata[((batch * inputCapsules + inputCapsule) * outputCapsules + outputCapsule) * capsuleDim + dim];
    c[idx] = sum;
}";

    public static string CapsuleAgreement => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint inputCapsules; uint outputCapsules; uint capsuleDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = batchSize * inputCapsules * outputCapsules;
    if (idx >= total) return;
    uint outputCapsule = idx % outputCapsules;
    uint q = idx / outputCapsules;
    uint inputCapsule = q % inputCapsules;
    uint batch = q / inputCapsules;
    float dot = 0.0;
    for (uint dim = 0u; dim < capsuleDim; ++dim)
        dot += a[((batch * inputCapsules + inputCapsule) * outputCapsules + outputCapsule) * capsuleDim + dim] *
            bdata[(batch * outputCapsules + outputCapsule) * capsuleDim + dim];
    c[idx] = dot;
}";

    public static string AttentionForward => Header + @"
layout(set = 0, binding = 0) readonly buffer Query { float queryData[]; };
layout(set = 0, binding = 1) readonly buffer Key { float keyData[]; };
layout(set = 0, binding = 2) readonly buffer Value { float valueData[]; };
layout(set = 0, binding = 3) writeonly buffer Output { float outputData[]; };
layout(set = 0, binding = 4) writeonly buffer Weights { float weightData[]; };
layout(set = 0, binding = 5) readonly buffer Mask { float maskData[]; };
layout(push_constant) uniform Params {
    uint batch; uint queryHeads; uint kvHeads; uint seqQ; uint seqK; uint headDim;
    float scale; uint causal; uint writeWeights; uint maskMode; uint maskBatchStride;
};
float attentionScore(uint batchIndex, uint queryHead, uint kvHead, uint queryIndex, uint keyIndex) {
    uint queryOffset = ((batchIndex * queryHeads + queryHead) * seqQ + queryIndex) * headDim;
    uint keyOffset = ((batchIndex * kvHeads + kvHead) * seqK + keyIndex) * headDim;
    float score = 0.0;
    for (uint d0 = 0u; d0 < headDim; ++d0) score += queryData[queryOffset + d0] * keyData[keyOffset + d0];
    return score * scale;
}
bool attentionMasked(uint batchIndex, uint queryHead, uint queryIndex, uint keyIndex) {
    if (maskMode == 0u) return false;
    uint maskOffset = maskMode == 2u ? batchIndex * maskBatchStride + queryHead * seqQ * seqK : 0u;
    return maskData[maskOffset + queryIndex * seqK + keyIndex] == 0.0;
}
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= batch * queryHeads * seqQ) return;
    uint queryIndex = row % seqQ;
    uint q = row / seqQ;
    uint queryHead = q % queryHeads;
    uint batchIndex = q / queryHeads;
    uint queriesPerKv = queryHeads / kvHeads;
    uint kvHead = queryHead / queriesPerKv;
    float rowMax = -3.402823466e+38;
    for (uint keyIndex = 0u; keyIndex < seqK; ++keyIndex) {
        if ((causal != 0u && keyIndex > queryIndex) || attentionMasked(batchIndex, queryHead, queryIndex, keyIndex)) continue;
        rowMax = max(rowMax, attentionScore(batchIndex, queryHead, kvHead, queryIndex, keyIndex));
    }
    float denominator = 0.0;
    for (uint keyIndex = 0u; keyIndex < seqK; ++keyIndex) {
        if ((causal != 0u && keyIndex > queryIndex) || attentionMasked(batchIndex, queryHead, queryIndex, keyIndex)) continue;
        denominator += exp(attentionScore(batchIndex, queryHead, kvHead, queryIndex, keyIndex) - rowMax);
    }
    uint weightOffset = row * seqK;
    if (writeWeights != 0u) {
        for (uint keyIndex = 0u; keyIndex < seqK; ++keyIndex) {
            float weight = 0.0;
            if ((causal == 0u || keyIndex <= queryIndex) && !attentionMasked(batchIndex, queryHead, queryIndex, keyIndex) && denominator > 0.0)
                weight = exp(attentionScore(batchIndex, queryHead, kvHead, queryIndex, keyIndex) - rowMax) / denominator;
            weightData[weightOffset + keyIndex] = weight;
        }
    }
    uint valueOffset = (batchIndex * kvHeads + kvHead) * seqK * headDim;
    uint outputOffset = row * headDim;
    for (uint d0 = 0u; d0 < headDim; ++d0) {
        float sum = 0.0;
        for (uint keyIndex = 0u; keyIndex < seqK; ++keyIndex) {
            if ((causal != 0u && keyIndex > queryIndex) || attentionMasked(batchIndex, queryHead, queryIndex, keyIndex) || denominator <= 0.0) continue;
            float weight = exp(attentionScore(batchIndex, queryHead, kvHead, queryIndex, keyIndex) - rowMax) / denominator;
            sum += weight * valueData[valueOffset + keyIndex * headDim + d0];
        }
        outputData[outputOffset + d0] = sum;
    }
}";

    public static string AttentionStats => Header + @"
layout(set = 0, binding = 0) readonly buffer Query { float queryData[]; };
layout(set = 0, binding = 1) readonly buffer Key { float keyData[]; };
layout(set = 0, binding = 2) readonly buffer Bias { float biasData[]; };
layout(set = 0, binding = 3) writeonly buffer Stats { float statsData[]; };
layout(push_constant) uniform Params {
    uint batch; uint heads; uint seqQ; uint seqK; uint headDim; float scale;
    uint causal; uint hasBias; uint biasBatchStride;
};
void main(){
    uint row=gl_GlobalInvocationID.x;if(row>=batch*heads*seqQ)return;
    uint queryIndex=row%seqQ,q=row/seqQ,head=q%heads,batchIndex=q/heads;
    uint queryOffset=((batchIndex*heads+head)*seqQ+queryIndex)*headDim;
    uint keyBase=(batchIndex*heads+head)*seqK*headDim;
    uint biasBase=batchIndex*biasBatchStride+head*seqQ*seqK+queryIndex*seqK;
    float rowMax=-3.402823466e+38;
    for(uint keyIndex=0u;keyIndex<seqK;++keyIndex){if(causal!=0u&&keyIndex>queryIndex)continue;float score=0.0;for(uint d=0u;d<headDim;++d)score+=queryData[queryOffset+d]*keyData[keyBase+keyIndex*headDim+d];score*=scale;if(hasBias!=0u)score+=biasData[biasBase+keyIndex];rowMax=max(rowMax,score);}
    float rowSum=0.0;
    for(uint keyIndex=0u;keyIndex<seqK;++keyIndex){if(causal!=0u&&keyIndex>queryIndex)continue;float score=0.0;for(uint d=0u;d<headDim;++d)score+=queryData[queryOffset+d]*keyData[keyBase+keyIndex*headDim+d];score*=scale;if(hasBias!=0u)score+=biasData[biasBase+keyIndex];rowSum+=exp(score-rowMax);}
    statsData[row]=rowMax+log(max(rowSum,1.0e-20));
}";

    public static string AttentionBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GradOutput { float gradOutputData[]; };
layout(set = 0, binding = 1) readonly buffer Query { float queryData[]; };
layout(set = 0, binding = 2) readonly buffer Key { float keyData[]; };
layout(set = 0, binding = 3) readonly buffer Value { float valueData[]; };
layout(set = 0, binding = 4) readonly buffer Weights { float weightData[]; };
layout(set = 0, binding = 5) writeonly buffer Grad { float gradData[]; };
layout(push_constant) uniform Params {
    uint batch; uint queryHeads; uint kvHeads; uint seqQ; uint seqK; uint headDim;
    float scale; uint causal; uint operation;
};
float gradAttention(uint batchIndex, uint queryHead, uint queryIndex, uint keyIndex, uint kvHead) {
    uint goOffset = ((batchIndex * queryHeads + queryHead) * seqQ + queryIndex) * headDim;
    uint valueOffset = ((batchIndex * kvHeads + kvHead) * seqK + keyIndex) * headDim;
    float sum = 0.0;
    for (uint d0 = 0u; d0 < headDim; ++d0) sum += gradOutputData[goOffset + d0] * valueData[valueOffset + d0];
    return sum;
}
float gradScore(uint batchIndex, uint queryHead, uint queryIndex, uint keyIndex, uint kvHead) {
    if (causal != 0u && keyIndex > queryIndex) return 0.0;
    uint weightOffset = ((batchIndex * queryHeads + queryHead) * seqQ + queryIndex) * seqK;
    float dot = 0.0;
    for (uint j = 0u; j < seqK; ++j)
        dot += gradAttention(batchIndex, queryHead, queryIndex, j, kvHead) * weightData[weightOffset + j];
    return weightData[weightOffset + keyIndex] *
        (gradAttention(batchIndex, queryHead, queryIndex, keyIndex, kvHead) - dot);
}
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint queriesPerKv = queryHeads / kvHeads;
    if (operation == 0u) {
        if (idx >= batch * queryHeads * seqQ * headDim) return;
        uint dim = idx % headDim;
        uint q = idx / headDim;
        uint queryIndex = q % seqQ; q /= seqQ;
        uint queryHead = q % queryHeads; uint batchIndex = q / queryHeads;
        uint kvHead = queryHead / queriesPerKv;
        uint keyBase = (batchIndex * kvHeads + kvHead) * seqK * headDim;
        float sum = 0.0;
        for (uint keyIndex = 0u; keyIndex < seqK; ++keyIndex)
            sum += gradScore(batchIndex, queryHead, queryIndex, keyIndex, kvHead) * keyData[keyBase + keyIndex * headDim + dim];
        gradData[idx] = sum * scale;
    } else {
        if (idx >= batch * kvHeads * seqK * headDim) return;
        uint dim = idx % headDim;
        uint q = idx / headDim;
        uint keyIndex = q % seqK; q /= seqK;
        uint kvHead = q % kvHeads; uint batchIndex = q / kvHeads;
        float sum = 0.0;
        for (uint queryHead = kvHead * queriesPerKv; queryHead < (kvHead + 1u) * queriesPerKv; ++queryHead) {
            uint weightHeadOffset = (batchIndex * queryHeads + queryHead) * seqQ * seqK;
            for (uint queryIndex = 0u; queryIndex < seqQ; ++queryIndex) {
                if (operation == 2u) {
                    uint goOffset = ((batchIndex * queryHeads + queryHead) * seqQ + queryIndex) * headDim;
                    sum += weightData[weightHeadOffset + queryIndex * seqK + keyIndex] * gradOutputData[goOffset + dim];
                } else {
                    uint queryOffset = ((batchIndex * queryHeads + queryHead) * seqQ + queryIndex) * headDim;
                    sum += gradScore(batchIndex, queryHead, queryIndex, keyIndex, kvHead) * queryData[queryOffset + dim] * scale;
                }
            }
        }
        gradData[idx] = sum;
    }
}";

    public static string ScatterAdd => Header + @"
layout(set = 0, binding = 0) readonly buffer Source { float sourceData[]; };
layout(set = 0, binding = 1) readonly buffer Indices { float indexData[]; };
layout(set = 0, binding = 2) buffer Destination { float destinationData[]; };
layout(push_constant) uniform Params { uint sourceSize; uint destSize; };
void main() {
    uint destination = gl_GlobalInvocationID.x;
    if (destination >= destSize) return;
    float sum = destinationData[destination];
    for (uint i = 0u; i < sourceSize; ++i)
        if (floatBitsToInt(indexData[i]) == int(destination)) sum += sourceData[i];
    destinationData[destination] = sum;
}";

    public static string GatherRows => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint numIndices; uint featureSize; uint sourceRows; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= numIndices * featureSize) return;
    uint indexPosition = idx / featureSize;
    uint feature = idx % featureSize;
    int sourceRow = floatBitsToInt(bdata[indexPosition]);
    c[idx] = sourceRow >= 0 && sourceRow < int(sourceRows) ? a[uint(sourceRow) * featureSize + feature] : 0.0;
}";

    public static string EmbeddingBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint numIndices; uint embeddingDim; uint vocabSize; };
void main(){
    uint idx=gl_GlobalInvocationID.x;
    if(idx>=vocabSize*embeddingDim)return;
    uint word=idx/embeddingDim,dimension=idx%embeddingDim;float sum=0.0;
    for(uint i=0u;i<numIndices;++i)if(floatBitsToUint(bdata[i])==word)sum+=a[i*embeddingDim+dimension];
    c[idx]=sum;
}";

    public static string GenericLoss => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; uint operation; float p0; float p1; };
float lossValue(float prediction, float target) {
    float difference = prediction - target;
    if (operation == 0u) return difference * difference;
    if (operation == 1u) return abs(difference);
    if (operation == 2u) { float cp = clamp(prediction, 1e-7, 1.0-1e-7); return -(target*log(cp)+(1.0-target)*log(1.0-cp)); }
    if (operation == 3u) { float d=abs(difference); return d<p0 ? 0.5*d*d/p0 : d-0.5*p0; }
    if (operation == 4u) { float cp=clamp(prediction,1e-7,1.0-1e-7); float pt=target>0.5?cp:1.0-cp; return -p0*pow(1.0-pt,p1)*log(pt); }
    if (operation == 5u) return log(cosh(difference));
    if (operation == 6u) { float d=target-prediction; return d>=0.0?p0*d:(p0-1.0)*d; }
    if (operation == 7u) return max(0.0,1.0-target*prediction);
    if (operation == 8u) { float h=max(0.0,1.0-target*prediction); return h*h; }
    if (operation == 9u) return exp(prediction)-target*prediction;
    if (operation == 10u) return exp(-target*prediction);
    if (operation == 11u) { float yp=target*prediction; float h=max(0.0,1.0-yp); return yp>=-1.0?h*h:-4.0*yp; }
    if (operation == 12u) return -target*log(max(1e-7,prediction));
    if (operation == 13u) return sqrt(difference*difference+p0*p0);
    return p0*abs(difference)+p1*difference*difference;
}
void main() {
    uint idx=gl_GlobalInvocationID.x;
    if(idx>=size)return;
    c[idx]=lossValue(a[idx],bdata[idx]);
}";

    public static string GenericLossBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; uint operation; float p0; float p1; };
float lossGradient(float prediction, float target) {
    float difference=prediction-target;
    if(operation==0u)return 2.0*difference;
    if(operation==1u)return sign(difference);
    if(operation==2u){float cp=clamp(prediction,1e-7,1.0-1e-7);return -target/cp+(1.0-target)/(1.0-cp);}
    if(operation==3u)return abs(difference)<p0?difference/p0:sign(difference);
    if(operation==4u){float cp=clamp(prediction,1e-7,1.0-1e-7);bool positive=target>0.5;float pt=positive?cp:1.0-cp;float direction=positive?-1.0:1.0;return -p0*direction*pow(1.0-pt,p1)*(p1*log(pt)+1.0/pt);}
    if(operation==5u)return tanh(difference);
    if(operation==6u)return target-prediction>=0.0?-p0:1.0-p0;
    if(operation==7u)return target*prediction<1.0?-target:0.0;
    if(operation==8u){float h=max(0.0,1.0-target*prediction);return h>0.0?-2.0*h*target:0.0;}
    if(operation==9u)return exp(prediction)-target;
    if(operation==10u)return -target*exp(-target*prediction);
    if(operation==11u){float yp=target*prediction;return yp>=1.0?0.0:(yp>=-1.0?-2.0*target*(1.0-yp):-4.0*target);}
    if(operation==12u)return -target/max(1e-7,prediction);
    if(operation==13u)return difference/sqrt(difference*difference+p0*p0);
    return p0*sign(difference)+2.0*p1*difference;
}
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=size)return;c[idx]=lossGradient(a[idx],bdata[idx])/float(size);}";

    public static string SparseCrossEntropyRows => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main(){
    uint row=gl_GlobalInvocationID.x;if(row>=batchSize)return;
    uint offset=row*numClasses;float rowMax=-3.402823466e+38;
    for(uint cls=0u;cls<numClasses;++cls)rowMax=max(rowMax,a[offset+cls]);
    float sum=0.0;for(uint cls=0u;cls<numClasses;++cls)sum+=exp(a[offset+cls]-rowMax);
    uint target=floatBitsToUint(bdata[row]);
    c[row]=target<numClasses?-(a[offset+target]-(rowMax+log(sum))):uintBitsToFloat(0x7fc00000u);
}";

    public static string SparseCrossEntropyBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main(){
    uint idx=gl_GlobalInvocationID.x;if(idx>=batchSize*numClasses)return;
    uint row=idx/numClasses,cls=idx%numClasses,offset=row*numClasses;float rowMax=-3.402823466e+38;
    for(uint j=0u;j<numClasses;++j)rowMax=max(rowMax,a[offset+j]);
    float sum=0.0;for(uint j=0u;j<numClasses;++j)sum+=exp(a[offset+j]-rowMax);
    uint target=floatBitsToUint(bdata[row]);
    c[idx]=(exp(a[idx]-rowMax)/sum-(cls==target?1.0:0.0))/float(batchSize);
}";

    public static string TripletLossRows => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint embeddingDim; float margin; };
void main(){uint row=gl_GlobalInvocationID.x;if(row>=batchSize)return;uint offset=row*embeddingDim;float dp=0.0,dn=0.0;for(uint j=0u;j<embeddingDim;++j){float x=a[offset+j]-bdata[offset+j];dp+=x*x;x=a[offset+j]-c[offset+j];dn+=x*x;}d[row]=max(0.0,sqrt(dp)-sqrt(dn)+margin);}";

    public static string TripletLossBackward => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint embeddingDim; float margin; uint operation; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batchSize*embeddingDim)return;uint row=idx/embeddingDim,offset=row*embeddingDim;float dp=0.0,dn=0.0;for(uint j=0u;j<embeddingDim;++j){float x=a[offset+j]-bdata[offset+j];dp+=x*x;x=a[offset+j]-c[offset+j];dn+=x*x;}float distP=sqrt(dp+1e-8),distN=sqrt(dn+1e-8),value=0.0;if(distP-distN+margin>0.0){float diffP=a[idx]-bdata[idx],diffN=a[idx]-c[idx];value=operation==0u?diffP/distP-diffN/distN:(operation==1u?-diffP/distP:diffN/distN);}d[idx]=value/float(batchSize);}";

    public static string ContrastiveLossRows => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint embeddingDim; float margin; };
void main(){uint row=gl_GlobalInvocationID.x;if(row>=batchSize)return;uint offset=row*embeddingDim;float ds=0.0;for(uint j=0u;j<embeddingDim;++j){float x=a[offset+j]-bdata[offset+j];ds+=x*x;}float dist=sqrt(ds),h=max(0.0,margin-dist),label=c[row];d[row]=0.5*(label*ds+(1.0-label)*h*h);}";

    public static string ContrastiveLossBackward => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint embeddingDim; float margin; uint operation; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batchSize*embeddingDim)return;uint row=idx/embeddingDim,offset=row*embeddingDim;float ds=0.0;for(uint j=0u;j<embeddingDim;++j){float x=a[offset+j]-bdata[offset+j];ds+=x*x;}float dist=sqrt(ds+1e-8),diff=a[idx]-bdata[idx],label=c[row];float value=label*diff;if(margin-dist>0.0)value-=(1.0-label)*(margin-dist)*diff/dist;d[idx]=(operation==0u?value:-value)/float(batchSize);}";

    public static string NormalizationForward => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) writeonly buffer Output { float outputData[]; };
layout(set=0,binding=2) readonly buffer Gamma { float gammaData[]; };
layout(set=0,binding=3) readonly buffer Beta { float betaData[]; };
layout(set=0,binding=4) buffer RunningMean { float runningMeanData[]; };
layout(set=0,binding=5) buffer RunningVariance { float runningVarianceData[]; };
layout(set=0,binding=6) writeonly buffer SavedMean { float savedMeanData[]; };
layout(set=0,binding=7) writeonly buffer SavedScale { float savedScaleData[]; };
layout(push_constant) uniform Params { uint mode; uint batch; uint channels; uint spatial; uint groups; float epsilon; float momentum; uint training; };
void main(){
    uint unit=gl_GlobalInvocationID.x;
    if(mode==0u){
        if(unit>=channels)return;float mean,variance;
        if(training!=0u){mean=0.0;for(uint b0=0u;b0<batch;++b0)for(uint s=0u;s<spatial;++s)mean+=inputData[(b0*channels+unit)*spatial+s];mean/=float(batch*spatial);variance=0.0;for(uint b0=0u;b0<batch;++b0)for(uint s=0u;s<spatial;++s){float x=inputData[(b0*channels+unit)*spatial+s]-mean;variance+=x*x;}variance/=float(batch*spatial);runningMeanData[unit]=(1.0-momentum)*runningMeanData[unit]+momentum*mean;runningVarianceData[unit]=(1.0-momentum)*runningVarianceData[unit]+momentum*variance;}else{mean=runningMeanData[unit];variance=runningVarianceData[unit];}
        float inv=1.0/sqrt(variance+epsilon);savedMeanData[unit]=mean;savedScaleData[unit]=inv;for(uint b0=0u;b0<batch;++b0)for(uint s=0u;s<spatial;++s){uint idx=(b0*channels+unit)*spatial+s;outputData[idx]=gammaData[unit]*(inputData[idx]-mean)*inv+betaData[unit];}
    }else if(mode==1u){
        if(unit>=batch)return;uint offset=unit*channels;float mean=0.0;for(uint i=0u;i<channels;++i)mean+=inputData[offset+i];mean/=float(channels);float variance=0.0;for(uint i=0u;i<channels;++i){float x=inputData[offset+i]-mean;variance+=x*x;}variance/=float(channels);float inv=1.0/sqrt(variance+epsilon);savedMeanData[unit]=mean;savedScaleData[unit]=inv;for(uint i=0u;i<channels;++i)outputData[offset+i]=gammaData[i]*(inputData[offset+i]-mean)*inv+betaData[i];
    }else if(mode==2u){
        if(unit>=batch*groups)return;uint batchIndex=unit/groups,group=unit%groups,channelsPerGroup=channels/groups,groupSize=channelsPerGroup*spatial;float mean=0.0;for(uint c0=group*channelsPerGroup;c0<(group+1u)*channelsPerGroup;++c0)for(uint s=0u;s<spatial;++s)mean+=inputData[(batchIndex*channels+c0)*spatial+s];mean/=float(groupSize);float variance=0.0;for(uint c0=group*channelsPerGroup;c0<(group+1u)*channelsPerGroup;++c0)for(uint s=0u;s<spatial;++s){float x=inputData[(batchIndex*channels+c0)*spatial+s]-mean;variance+=x*x;}variance/=float(groupSize);float inv=1.0/sqrt(variance+epsilon);savedMeanData[unit]=mean;savedScaleData[unit]=inv;for(uint c0=group*channelsPerGroup;c0<(group+1u)*channelsPerGroup;++c0)for(uint s=0u;s<spatial;++s){uint idx=(batchIndex*channels+c0)*spatial+s;outputData[idx]=gammaData[c0]*(inputData[idx]-mean)*inv+betaData[c0];}
    }else{
        if(unit>=batch)return;uint offset=unit*channels;float sumSquares=0.0;for(uint i=0u;i<channels;++i)sumSquares+=inputData[offset+i]*inputData[offset+i];float rms=sqrt(sumSquares/float(channels)+epsilon);savedScaleData[unit]=rms;for(uint i=0u;i<channels;++i)outputData[offset+i]=gammaData[i]*inputData[offset+i]/rms;
    }
}";

    public static string NormalizationBackward => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float go[]; };
layout(set=0,binding=1) readonly buffer Input { float inputData[]; };
layout(set=0,binding=2) readonly buffer Gamma { float gammaData[]; };
layout(set=0,binding=3) readonly buffer SavedMean { float savedMeanData[]; };
layout(set=0,binding=4) readonly buffer SavedScale { float savedScaleData[]; };
layout(set=0,binding=5) writeonly buffer GradInput { float gi[]; };
layout(set=0,binding=6) writeonly buffer GradGamma { float gg[]; };
layout(set=0,binding=7) writeonly buffer GradBeta { float gb[]; };
layout(push_constant) uniform Params { uint mode; uint operation; uint batch; uint channels; uint spatial; };
void main(){
    uint unit=gl_GlobalInvocationID.x;
    if(mode==0u){if(unit>=channels)return;float mean=savedMeanData[unit],inv=savedScaleData[unit],sumGrad=0.0,sumGradX=0.0;for(uint b0=0u;b0<batch;++b0)for(uint s=0u;s<spatial;++s){uint idx=(b0*channels+unit)*spatial+s;float xhat=(inputData[idx]-mean)*inv;sumGrad+=go[idx];sumGradX+=go[idx]*xhat;}gg[unit]=sumGradX;gb[unit]=sumGrad;float count=float(batch*spatial);for(uint b0=0u;b0<batch;++b0)for(uint s=0u;s<spatial;++s){uint idx=(b0*channels+unit)*spatial+s;float xhat=(inputData[idx]-mean)*inv;gi[idx]=gammaData[unit]*inv/count*(count*go[idx]-sumGrad-xhat*sumGradX);}return;}
    if(mode==1u){if(operation==0u){if(unit>=batch)return;uint offset=unit*channels;float mean=savedMeanData[unit],inv=savedScaleData[unit],sumGrad=0.0,sumGradX=0.0;for(uint i=0u;i<channels;++i){float xhat=(inputData[offset+i]-mean)*inv;sumGrad+=go[offset+i]*gammaData[i];sumGradX+=go[offset+i]*gammaData[i]*xhat;}for(uint i=0u;i<channels;++i){float xhat=(inputData[offset+i]-mean)*inv;gi[offset+i]=inv/float(channels)*(float(channels)*go[offset+i]*gammaData[i]-sumGrad-xhat*sumGradX);}}else{if(unit>=channels)return;float gradGamma=0.0,gradBeta=0.0;for(uint b0=0u;b0<batch;++b0){uint idx=b0*channels+unit;gradGamma+=go[idx]*(inputData[idx]-savedMeanData[b0])*savedScaleData[b0];gradBeta+=go[idx];}gg[unit]=gradGamma;gb[unit]=gradBeta;}return;}
    if(mode==2u){if(operation==0u){if(unit>=batch*channels)return;uint batchIndex=unit/channels,channel=unit%channels,offset=(batchIndex*channels+channel)*spatial,stat=batchIndex*channels+channel;float mean=savedMeanData[stat],inv=savedScaleData[stat],sumGrad=0.0,sumGradX=0.0;for(uint s=0u;s<spatial;++s){float xhat=(inputData[offset+s]-mean)*inv;sumGrad+=go[offset+s];sumGradX+=go[offset+s]*xhat;}for(uint s=0u;s<spatial;++s){float xhat=(inputData[offset+s]-mean)*inv;gi[offset+s]=gammaData[channel]*inv/float(spatial)*(float(spatial)*go[offset+s]-sumGrad-xhat*sumGradX);}}else{if(unit>=channels)return;float gradGamma=0.0,gradBeta=0.0;for(uint b0=0u;b0<batch;++b0){uint stat=b0*channels+unit,offset=(b0*channels+unit)*spatial;for(uint s=0u;s<spatial;++s){gradGamma+=go[offset+s]*(inputData[offset+s]-savedMeanData[stat])*savedScaleData[stat];gradBeta+=go[offset+s];}}gg[unit]=gradGamma;gb[unit]=gradBeta;}return;}
    if(operation==0u){if(unit>=batch)return;uint offset=unit*channels;float rms=savedScaleData[unit],inv=1.0/rms,sumGradX=0.0;for(uint i=0u;i<channels;++i)sumGradX+=go[offset+i]*gammaData[i]*inputData[offset+i];sumGradX*=inv*inv/float(channels);for(uint i=0u;i<channels;++i)gi[offset+i]=(go[offset+i]*gammaData[i]-inputData[offset+i]*sumGradX)*inv;}else{if(unit>=channels)return;float gradGamma=0.0;for(uint b0=0u;b0<batch;++b0){uint idx=b0*channels+unit;gradGamma+=go[idx]*inputData[idx]/savedScaleData[b0];}gg[unit]=gradGamma;}
}";

    // =====================================================================
    // Scalar ops
    // =====================================================================

    // In-place multiply of every element by a DEVICE-resident scalar (bdata[0]). Vulkan/GLSL counterpart of
    // the CUDA scale_by_device_scalar. The in-place buffer is bound to BOTH binding 0 (read) and binding 2
    // (write); each invocation reads and writes the same index, so the read/write alias is hazard-free.
    public static string ScaleByDeviceScalar => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = a[idx] * bdata[0];
}";

    // =====================================================================
    // Reductions
    // =====================================================================

    public static string MeanAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    float sum = 0.0;
    uint base_idx = idx * reduceSize;
    for (uint j = 0; j < reduceSize; j++) sum += a[base_idx + j];
    b[idx] = sum / float(reduceSize);
}";

    public static string VarianceAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) sum += a[base_idx + j];
    float mean = sum / float(reduceSize);
    float var_sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float d = a[base_idx + j] - mean; var_sum += d * d; }
    b[idx] = var_sum / float(reduceSize);
}";

    public static string StdAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) sum += a[base_idx + j];
    float mean = sum / float(reduceSize);
    float var_sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float d = a[base_idx + j] - mean; var_sum += d * d; }
    b[idx] = sqrt(var_sum / float(reduceSize));
}";

    public static string NormAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float sum_sq = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float v = a[base_idx + j]; sum_sq += v * v; }
    b[idx] = sqrt(sum_sq);
}";

    public static string SumOfSquaresAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float sum_sq = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float v = a[base_idx + j]; sum_sq += v * v; }
    b[idx] = sum_sq;
}";

    public static string MaxMagnitudeAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float mx = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float v = abs(a[base_idx + j]); mx = max(mx, v); }
    b[idx] = mx;
}";

    public static string MinMagnitudeAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float mn = abs(a[base_idx]);
    for (uint j = 1; j < reduceSize; j++) { float v = abs(a[base_idx + j]); mn = min(mn, v); }
    b[idx] = mn;
}";

    public static string LogVarianceAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) sum += a[base_idx + j];
    float mean = sum / float(reduceSize);
    float var_sum = 0.0;
    for (uint j = 0; j < reduceSize; j++) { float d = a[base_idx + j] - mean; var_sum += d * d; }
    b[idx] = log(var_sum / float(reduceSize) + 1e-8);
}";

    public static string LogSumExpAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float max_val = a[base_idx];
    for (uint j = 1; j < reduceSize; j++) max_val = max(max_val, a[base_idx + j]);
    float sum_exp = 0.0;
    for (uint j = 0; j < reduceSize; j++) sum_exp += exp(a[base_idx + j] - max_val);
    b[idx] = max_val + log(sum_exp);
}";

    public static string CumSumAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * innerSize;
    float running = 0.0;
    for (uint j = 0; j < innerSize; j++) { running += a[base_idx + j]; b[base_idx + j] = running; }
}";

    public static string ProductAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    uint base_idx = idx * reduceSize;
    float prod = 1.0;
    for (uint j = 0; j < reduceSize; j++) prod *= a[base_idx + j];
    b[idx] = prod;
}";

    public static string NormalizeL2 => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;
    float sum_sq = 0.0;
    for (uint j = 0; j < innerSize; j++) { float v = a[base_idx + j]; sum_sq += v * v; }
    float inv_norm = 1.0 / (sqrt(sum_sq) + 1e-12);
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] = a[base_idx + j] * inv_norm;
}";

    // =====================================================================
    // Broadcast / Scalar
    // =====================================================================

    public static string ScalarMinusTensor => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float scalar; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = scalar - a[idx];
}";

    public static string BroadcastAddLast => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] + bdata[idx % innerSize];
}";

    public static string BroadcastMulLast => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] * bdata[idx % innerSize];
}";

    public static string AddScalar => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float scalar; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] + scalar;
}";

    public static string ClipKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float minVal; float maxVal; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = clamp(a[idx], minVal, maxVal);
}";

    public static string RsqrtKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = inversesqrt(a[idx] + 1e-12);
}";

    // =====================================================================
    // Gated Activations
    // =====================================================================

    public static string GluForward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float value = a[outer * fullDim + d];
    float gate = a[outer * fullDim + halfDim + d];
    b[idx] = value * (1.0 / (1.0 + exp(-gate)));
}";

    public static string GeGluForward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float value = a[outer * fullDim + d];
    float gate = a[outer * fullDim + halfDim + d];
    float x3 = value * value * value;
    float gelu = 0.5 * value * (1.0 + tanh(0.7978845608 * (value + 0.044715 * x3)));
    b[idx] = gelu * gate;
}";

    public static string ReGluForward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float value = a[outer * fullDim + d];
    float gate = a[outer * fullDim + halfDim + d];
    b[idx] = max(value, 0.0) * gate;
}";

    public static string SwiGluForward => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float value = a[outer * fullDim + d];
    float gate = a[outer * fullDim + halfDim + d];
    float sig = 1.0 / (1.0 + exp(-value));
    b[idx] = value * sig * gate;
}";

    // Backward gated activation GLSL kernels.
    // a[] = gradOutput [outerSize * halfDim]
    // bdata[] = original input [outerSize * halfDim * 2]  (first half = value, second half = gate)
    // c[] = gradInput [outerSize * halfDim * 2]  (first half = dValue, second half = dGate)
    // Each thread computes ONE output pair (dValue[idx], dGate[idx]) from gradOutput[idx] and input.

    public static string GluBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float grad = a[idx];
    float value = bdata[outer * fullDim + d];
    float gate = bdata[outer * fullDim + halfDim + d];
    float sig = 1.0 / (1.0 + exp(-gate));
    c[outer * fullDim + d] = grad * sig;
    c[outer * fullDim + halfDim + d] = grad * value * sig * (1.0 - sig);
}";

    public static string GeGluBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float grad = a[idx];
    float value = bdata[outer * fullDim + d];
    float gate = bdata[outer * fullDim + halfDim + d];
    float x3 = value * value * value;
    float z = 0.7978845608 * (value + 0.044715 * x3);
    float tanh_z = tanh(z);
    float gelu = 0.5 * value * (1.0 + tanh_z);
    // GELU derivative: 0.5*(1+tanh(z)) + 0.5*x*sech^2(z)*z'
    // where z' = 0.7978845608 * (1 + 3*0.044715*x^2)
    float sech2 = 1.0 - tanh_z * tanh_z;
    float z_deriv = 0.7978845608 * (1.0 + 3.0 * 0.044715 * value * value);
    float gelu_deriv = 0.5 * (1.0 + tanh_z) + 0.5 * value * sech2 * z_deriv;
    c[outer * fullDim + d] = grad * gate * gelu_deriv;
    c[outer * fullDim + halfDim + d] = grad * gelu;
}";

    public static string ReGluBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float grad = a[idx];
    float value = bdata[outer * fullDim + d];
    float gate = bdata[outer * fullDim + halfDim + d];
    c[outer * fullDim + d] = grad * gate * (value > 0.0 ? 1.0 : 0.0);
    c[outer * fullDim + halfDim + d] = grad * max(value, 0.0);
}";

    public static string SwiGluBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint halfDim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * halfDim;
    if (idx >= total) return;
    uint outer = idx / halfDim; uint d = idx % halfDim; uint fullDim = halfDim * 2;
    float grad = a[idx];
    float value = bdata[outer * fullDim + d];
    float gate = bdata[outer * fullDim + halfDim + d];
    float sig = 1.0 / (1.0 + exp(-value));
    float swish = value * sig;
    float swish_deriv = sig + value * sig * (1.0 - sig);
    c[outer * fullDim + d] = grad * gate * swish_deriv;
    c[outer * fullDim + halfDim + d] = grad * swish;
}";

    public static string ReluDerivative => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] > 0.0 ? 1.0 : 0.0;
}";

    public static string SigmoidDerivative => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float s = a[idx]; b[idx] = s * (1.0 - s);
}";

    public static string TanhDerivative => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float t = a[idx]; b[idx] = 1.0 - t * t;
}";

    // =====================================================================
    // Softmax Variants + Distance
    // =====================================================================

    public static string LogSoftmax => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;
    float max_val = a[base_idx];
    for (uint j = 1; j < innerSize; j++) max_val = max(max_val, a[base_idx + j]);
    float sum_exp = 0.0;
    for (uint j = 0; j < innerSize; j++) sum_exp += exp(a[base_idx + j] - max_val);
    float log_sum = log(sum_exp);
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] = a[base_idx + j] - max_val - log_sum;
}";

    public static string OuterProduct => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint M; uint N; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= M * N) return;
    c[idx] = a[idx / N] * bdata[idx % N];
}";

    public static string CosineSimilarity => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint dim; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float dot_val = 0.0, na = 0.0, nb = 0.0;
    for (uint i = 0; i < dim; i++) {
        float ai = a[batch * dim + i]; float bi = bdata[batch * dim + i];
        dot_val += ai * bi; na += ai * ai; nb += bi * bi;
    }
    c[batch] = dot_val / (sqrt(na) * sqrt(nb) + 1e-8);
}";

    // =====================================================================
    // Shape / Layout
    // =====================================================================

    public static string EyeKernel => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n * n) return;
    b[idx] = (idx / n == idx % n) ? 1.0 : 0.0;
}";

    public static string OneHotKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize * numClasses) return;
    uint batch = idx / numClasses; uint cls = idx % numClasses;
    b[idx] = (uint(a[batch]) == cls) ? 1.0 : 0.0;
}";

    public static string DiagKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n * n) return;
    b[idx] = (idx / n == idx % n) ? a[idx / n] : 0.0;
}";

    public static string TriangularMask => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint rows; uint cols; int diagonal; float maskValue; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rows * cols) return;
    uint row = idx / cols; uint col = idx % cols;
    b[idx] = (col > row + uint(diagonal)) ? maskValue : 0.0;
}";

    // =====================================================================
    // Loss
    // =====================================================================

    public static string CrossEntropyLoss => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float loss = 0.0;
    for (uint cls = 0; cls < numClasses; cls++) {
        float target = bdata[batch * numClasses + cls];
        if (target > 0.0) loss -= target * log(max(a[batch * numClasses + cls], 1e-7));
    }
    c[batch] = loss;
}";

    public static string MseLoss => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numFeatures; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float sum_sq = 0.0;
    for (uint f = 0; f < numFeatures; f++) {
        float d = a[batch * numFeatures + f] - bdata[batch * numFeatures + f];
        sum_sq += d * d;
    }
    c[batch] = sum_sq / float(numFeatures);
}";

    // Shape ops
    public static string ConcatAxisGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint aInnerSize; uint bInnerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint totalInner = aInnerSize + bInnerSize;
    if (idx >= outerSize * totalInner) return;
    uint outer = idx / totalInner; uint inner = idx % totalInner;
    c[idx] = (inner < aInnerSize) ? a[outer * aInnerSize + inner] : bdata[outer * bInnerSize + (inner - aInnerSize)];
}";

    public static string SliceLastAxisGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint inputInnerSize; uint start; uint sliceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * sliceSize) return;
    b[idx] = a[(idx / sliceSize) * inputInnerSize + start + (idx % sliceSize)];
}";

    /// <summary>
    /// General axis slice: out[outer * stride + s] = in[outer * axisSize * stride + index * stride + s]
    /// Works for any axis, not just the last one.
    /// </summary>
    public static string SliceAxisGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint axisSize; uint stride; uint index; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * stride;
    if (idx >= total) return;
    uint outer = idx / stride;
    uint s = idx % stride;
    b[idx] = a[outer * axisSize * stride + index * stride + s];
}";

    /// <summary>
    /// General axis set-slice: out[outer * axisSize * stride + index * stride + s] = in[outer * stride + s]
    /// Inverse of SliceAxisGlsl.
    /// </summary>
    public static string SetSliceAxisGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint axisSize; uint stride; uint index; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * stride;
    if (idx >= total) return;
    uint outer = idx / stride;
    uint s = idx % stride;
    b[outer * axisSize * stride + index * stride + s] = a[idx];
}";

    public static string Pad2DGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint outH; uint outW; uint padTop; uint padLeft; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batch * channels * outH * outW) return;
    uint w = idx % outW; uint temp = idx / outW; uint h = temp % outH; temp /= outH; uint ch = temp % channels; uint ba = temp / channels;
    int srcH = int(h) - int(padTop); int srcW = int(w) - int(padLeft);
    b[idx] = (srcH >= 0 && srcH < int(inH) && srcW >= 0 && srcW < int(inW)) ? a[((ba * channels + ch) * inH + uint(srcH)) * inW + uint(srcW)] : 0.0;
}";

    public static string TileLastAxisGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; uint repeats; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint tiledInner = innerSize * repeats;
    if (idx >= outerSize * tiledInner) return;
    b[idx] = a[(idx / tiledInner) * innerSize + ((idx % tiledInner) % innerSize)];
}";

    public static string SumAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize) return;
    float sum = 0.0;
    uint offset = idx * reduceSize;
    for (uint j = 0u; j < reduceSize; ++j) sum += a[offset + j];
    b[idx] = sum;
}";

    public static string ScalarReduce => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; uint op; };
void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    if (op == 0u) {
        float value = 0.0;
        for (uint i = 0u; i < size; ++i) value += a[i];
        b[0] = value;
    } else if (op == 1u) {
        float value = -3.402823466e+38;
        for (uint i = 0u; i < size; ++i) value = max(value, a[i]);
        b[0] = value;
    } else if (op == 2u) {
        float value = 3.402823466e+38;
        for (uint i = 0u; i < size; ++i) value = min(value, a[i]);
        b[0] = value;
    } else if (op == 3u) {
        float mean = 0.0;
        for (uint i = 0u; i < size; ++i) mean += a[i];
        mean /= float(size);
        float variance = 0.0;
        for (uint i = 0u; i < size; ++i) { float delta = a[i] - mean; variance += delta * delta; }
        b[0] = sqrt(max(0.0, variance / float(size)));
    } else {
        float sumSquares = 0.0;
        for (uint i = 0u; i < size; ++i) sumSquares += a[i] * a[i];
        b[0] = sqrt(sumSquares);
    }
}";

    public static string VarianceAxisFromMean => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint outer = gl_GlobalInvocationID.x;
    if (outer >= outerSize) return;
    float variance = 0.0;
    uint offset = outer * reduceSize;
    for (uint j = 0u; j < reduceSize; ++j) { float delta = a[offset + j] - bdata[outer]; variance += delta * delta; }
    c[outer] = variance / float(reduceSize);
}";

    public static string ArgExtremaAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; uint findMax; };
void main() {
    uint outer = gl_GlobalInvocationID.x;
    if (outer >= outerSize) return;
    uint offset = outer * reduceSize;
    float best = findMax != 0u ? -3.402823466e+38 : 3.402823466e+38;
    uint bestIndex = 0u;
    for (uint j = 0u; j < reduceSize; ++j) {
        float value = a[offset + j];
        bool better = findMax != 0u ? value > best : value < best;
        if (better) { best = value; bestIndex = j; }
    }
    b[outer] = float(bestIndex);
}";

    public static string MaxAxis => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint outer = gl_GlobalInvocationID.x;
    if (outer >= outerSize) return;
    float value = -3.402823466e+38;
    uint offset = outer * reduceSize;
    for (uint j = 0u; j < reduceSize; ++j) if (a[offset + j] > value) value = a[offset + j];
    b[outer] = value;
}";

    public static string TopKAxis => Header + @"
layout(set = 0, binding = 0) readonly buffer Input { float inputData[]; };
layout(set = 0, binding = 1) writeonly buffer Values { float values[]; };
layout(set = 0, binding = 2) writeonly buffer Indices { float indices[]; };
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; uint k; };
bool isNanValue(float value) { uint bits = floatBitsToUint(value); return (bits & 0x7fffffffu) > 0x7f800000u; }
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    uint outer = idx / reduceSize;
    uint local = idx % reduceSize;
    float value = inputData[idx];
    bool valueNan = isNanValue(value);
    uint rank = 0u;
    uint offset = outer * reduceSize;
    for (uint j = 0u; j < reduceSize; ++j) {
        float other = inputData[offset + j];
        bool otherNan = isNanValue(other);
        bool before = (!otherNan && valueNan) ||
            (otherNan == valueNan && (other > value || (other == value && j < local)));
        if (before) ++rank;
    }
    if (rank < k) { values[outer * k + rank] = value; indices[outer * k + rank] = uintBitsToFloat(local); }
}";

    public static string BroadcastMultiplyLastAxis => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] * bdata[idx % innerSize];
}";

    public static string TileAxisGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint axisSize; uint innerSize; uint repeats; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint expandedAxis = axisSize * repeats;
    uint total = outerSize * expandedAxis * innerSize;
    if (idx >= total) return;
    uint inner = idx % innerSize;
    uint expandedIndex = (idx / innerSize) % expandedAxis;
    uint outer = idx / (expandedAxis * innerSize);
    uint axis = expandedIndex / repeats;
    b[idx] = a[(outer * axisSize + axis) * innerSize + inner];
}";

    public static string FillGlsl => Header + @"
layout(set = 0, binding = 0) buffer O { float o[]; };
layout(push_constant) uniform Params { float value; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < size) o[idx] = value;
}";

    public static string PixelShuffleGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint scale; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint outH = inH * scale; uint outW = inW * scale;
    if (idx >= batch * channels * outH * outW) return;
    uint ow = idx % outW; uint temp = idx / outW; uint oh = temp % outH; temp /= outH; uint oc = temp % channels; uint ba = temp / channels;
    uint srcC = oc * scale * scale + (oh % scale) * scale + (ow % scale);
    b[idx] = a[((ba * channels * scale * scale + srcC) * inH + oh / scale) * inW + ow / scale];
}";

    public static string EyeKernelGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n * n) return;
    b[idx] = (idx / n == idx % n) ? 1.0 : 0.0;
}";

    public static string LinspaceKernelGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { float start; float step; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = start + step * float(idx);
}";

    public static string DiagKernelGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint n; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n * n) return;
    b[idx] = (idx / n == idx % n) ? a[idx / n] : 0.0;
}";

    public static string TriangularMaskGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint rows; uint cols; int diagonal; float maskValue; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= rows * cols) return;
    b[idx] = (int(idx % cols) > int(idx / cols) + diagonal) ? maskValue : 0.0;
}";

    public static string IndexSelectGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint numIndices; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= numIndices * innerSize) return;
    c[idx] = a[uint(bdata[idx / innerSize]) * innerSize + (idx % innerSize)];
}";

    public static string BatchDotProductGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint dim; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float sum = 0.0;
    for (uint i = 0; i < dim; i++) sum += a[batch * dim + i] * bdata[batch * dim + i];
    c[batch] = sum;
}";

    public static string PairwiseDistanceGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint M; uint N; uint dim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= M * N) return;
    uint i = idx / N; uint j = idx % N;
    float d = 0.0;
    for (uint k = 0; k < dim; k++) { float diff = a[i*dim+k] - bdata[j*dim+k]; d += diff*diff; }
    c[idx] = sqrt(d);
}";

    public static string BceLossGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float p = clamp(a[idx], 1e-7, 1.0 - 1e-7);
    float t = bdata[idx];
    c[idx] = -(t * log(p) + (1.0 - t) * log(1.0 - p));
}";

    public static string CrossEntropyLossGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float loss = 0.0;
    for (uint cls = 0; cls < numClasses; cls++) {
        float target = bdata[batch * numClasses + cls];
        if (target > 0.0) loss -= target * log(max(a[batch * numClasses + cls], 1e-7));
    }
    c[batch] = loss;
}";

    // =====================================================================
    // Additional Scalar / Broadcast kernels
    // =====================================================================

    public static string SubScalar => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float scalar; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] - scalar;
}";

    public static string DivScalar => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float scalar; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] / scalar;
}";

    public static string PowScalar => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { float exponent; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = pow(a[idx], exponent);
}";

    public static string FracKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = fract(a[idx]);
}";

    public static string SinKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = sin(a[idx]);
}";

    public static string CosKernel => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = cos(a[idx]);
}";

    public static string BroadcastSubLast => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] - bdata[idx % innerSize];
}";

    public static string BroadcastDivLast => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] / (bdata[idx % innerSize] + 1e-12);
}";

    public static string BroadcastAddFirst => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx / innerSize] + bdata[idx];
}";

    public static string BroadcastMulFirst => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx / innerSize] * bdata[idx];
}";

    public static string BroadcastMultiplyFirstAxis => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * innerSize) return;
    c[idx] = a[idx] * bdata[idx / innerSize];
}";

    public static string EqualsKernel => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = (a[idx] == bdata[idx]) ? 1.0 : 0.0;
}";

    public static string NotEqualsKernel => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = (a[idx] != bdata[idx]) ? 1.0 : 0.0;
}";

    public static string MaskedFillKernel => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { float fillValue; uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = (bdata[idx] != 0.0) ? fillValue : a[idx];
}";

    // =====================================================================
    // Additional Shape / Layout kernels
    // =====================================================================

    public static string SetSliceLastAxisGlsl => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) buffer B { float b[]; };
layout(push_constant) uniform Params { uint outerSize; uint outputInnerSize; uint start; uint sliceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * sliceSize) return;
    uint outer = idx / sliceSize; uint inner = idx % sliceSize;
    b[outer * outputInnerSize + start + inner] = a[outer * sliceSize + inner];
}";

    public static string Stack2Glsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size * 2) return;
    c[idx] = ((idx % 2) == 0) ? a[idx / 2] : bdata[idx / 2];
}";

    public static string Pad2DBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint outH; uint outW; uint padTop; uint padLeft; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batch * channels * inH * inW) return;
    uint w = idx % inW; uint temp = idx / inW; uint h = temp % inH; temp /= inH; uint ch = temp % channels; uint ba = temp / channels;
    b[idx] = a[((ba * channels + ch) * outH + h + padTop) * outW + w + padLeft];
}";

    public static string RepeatElementsGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint totalElements; uint innerSize; uint repeats; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= totalElements) return;
    b[idx] = a[idx / repeats];
}";

    public static string PixelShuffleBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint scale; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint totalInputChannels = channels * scale * scale;
    if (idx >= batch * totalInputChannels * inH * inW) return;
    uint iw = idx % inW; uint temp = idx / inW; uint ih = temp % inH; temp /= inH; uint ic = temp % totalInputChannels; uint ba = temp / totalInputChannels;
    uint oc = ic / (scale * scale); uint sub = ic % (scale * scale); uint subH = sub / scale; uint subW = sub % scale;
    uint outH = inH * scale; uint outW = inW * scale;
    b[idx] = a[((ba * channels + oc) * outH + ih * scale + subH) * outW + iw * scale + subW];
}";

    public static string Crop2DGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint outH; uint outW; uint offsetH; uint offsetW; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batch * channels * outH * outW) return;
    uint w = idx % outW; uint temp = idx / outW; uint h = temp % outH; temp /= outH; uint ch = temp % channels; uint ba = temp / channels;
    b[idx] = a[((ba * channels + ch) * inH + h + offsetH) * inW + w + offsetW];
}";

    public static string Crop2DBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint outH; uint outW; uint offsetH; uint offsetW; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batch * channels * outH * outW) return;
    uint w = idx % outW; uint temp = idx / outW; uint h = temp % outH; temp /= outH; uint ch = temp % channels; uint ba = temp / channels;
    // b = gradInput (full size), a = gradOutput (cropped size)
    // Map output position back to input position
    b[((ba * channels + ch) * inH + h + offsetH) * inW + w + offsetW] = a[idx];
}";

    public static string ExtractDiagKernelGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint n; uint cols; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    b[idx] = a[idx * cols + idx];
}";

    // =====================================================================
    // Backward kernels
    // =====================================================================

    public static string ReduceSumBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    b[idx] = a[idx / reduceSize];
}";

    public static string ReduceMeanBackwardGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    b[idx] = a[idx / reduceSize] / float(reduceSize);
}";

    private const string FourBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float bdata[]; };
layout(set = 0, binding = 2) readonly buffer C { float c[]; };
layout(set = 0, binding = 3) writeonly buffer D { float d[]; };
";

    private const string FiveBufferLayout = @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float bdata[]; };
layout(set = 0, binding = 2) readonly buffer C { float c[]; };
layout(set = 0, binding = 3) readonly buffer D { float d[]; };
layout(set = 0, binding = 4) writeonly buffer E { float e[]; };
";

    public static string ReduceMaxBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    // a=gradOutput, bdata=input, c=maxValues, d=gradInput
    d[idx] = (bdata[idx] == c[idx / reduceSize]) ? a[idx / reduceSize] : 0.0;
}";

    public static string ReduceVarianceBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    // a=gradOutput, bdata=input, c=means, d=gradInput
    uint outer = idx / reduceSize;
    d[idx] = a[outer] * 2.0 * (bdata[idx] - c[outer]) / float(reduceSize);
}";

    public static string ReduceLogVarianceBackwardGlsl => Header + FiveBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outerSize * reduceSize) return;
    // a=gradOutput, bdata=input, c=means, d=variances, e=gradInput
    uint outer = idx / reduceSize;
    e[idx] = a[outer] * (1.0 / (d[outer] + 1e-8)) * 2.0 * (bdata[idx] - c[outer]) / float(reduceSize);
}";

    // =====================================================================
    // Softmax Variants
    // =====================================================================

    public static string TaylorSoftmaxGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;
    float sum = 0.0;
    for (uint j = 0; j < innerSize; j++) {
        float x = a[base_idx + j];
        float t = max(1.0 + x + 0.5 * x * x, 1e-10);
        b[base_idx + j] = t;
        sum += t;
    }
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] /= sum;
}";

    // Spherical softmax: normalize by exp-weighted L2 norm (not regular softmax)
    // output[i] = exp(x[i]) / sqrt(sum(exp(2*x[j])))
    public static string SphericalSoftmaxGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;
    float max_val = a[base_idx];
    for (uint j = 1; j < innerSize; j++) max_val = max(max_val, a[base_idx + j]);
    float sum_sq = 0.0;
    for (uint j = 0; j < innerSize; j++) {
        float e = exp(a[base_idx + j] - max_val);
        b[base_idx + j] = e;
        sum_sq += e * e;
    }
    float inv_norm = 1.0 / (sqrt(sum_sq) + 1e-10);
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] *= inv_norm;
}";

    // =====================================================================
    // Distance
    // =====================================================================

    public static string PairwiseDistanceSquaredGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint M; uint N; uint dim; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= M * N) return;
    uint i = idx / N; uint j = idx % N;
    float d = 0.0;
    for (uint k = 0; k < dim; k++) { float diff = a[i*dim+k] - bdata[j*dim+k]; d += diff*diff; }
    c[idx] = d;
}";

    public static string BatchOuterProductGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint M; uint N; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize * M * N) return;
    uint ba = idx / (M * N); uint rem = idx % (M * N);
    uint i = rem / N; uint j = rem % N;
    c[idx] = a[ba * M + i] * bdata[ba * N + j];
}";

    public static string MseLossGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numFeatures; };
void main() {
    uint batch = gl_GlobalInvocationID.x;
    if (batch >= batchSize) return;
    float sum_sq = 0.0;
    for (uint f = 0; f < numFeatures; f++) {
        float d = a[batch * numFeatures + f] - bdata[batch * numFeatures + f];
        sum_sq += d * d;
    }
    c[batch] = sum_sq / float(numFeatures);
}";

    // =====================================================================
    // Philox counter-based PRNG kernels (GPU-native RNG)
    // =====================================================================

    // Philox 2x32-10 PRNG: deterministic, parallel-safe counter-based RNG
    private const string PhiloxPrng = @"
// Philox 2x32 round function
uvec2 philox2x32_round(uvec2 ctr, uint key) {
    uint hi = uint((uint64_t(ctr.x) * uint64_t(0xD256D193u)) >> 32);
    uint lo = ctr.x * 0xD256D193u;
    return uvec2(hi ^ key ^ ctr.y, lo);
}
// Philox 2x32-10: 10 rounds for full period
uvec2 philox2x32(uvec2 ctr, uint key) {
    for (int i = 0; i < 10; i++) {
        ctr = philox2x32_round(ctr, key);
        key += 0x9E3779B9u; // golden ratio
    }
    return ctr;
}
// Convert uint to float in [0,1)
float uint_to_float01(uint x) { return float(x >> 8) * (1.0 / 16777216.0); }
";

    public static string DropoutMaskGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint size; float keepProb; uint seedLo; uint seedHi; };
" + PhiloxPrng + @"
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    uvec2 rng = philox2x32(uvec2(idx, seedHi), seedLo);
    float u = uint_to_float01(rng.x);
    float inv = 1.0 / max(keepProb, 1e-7);
    b[idx] = (u < keepProb) ? inv : 0.0;
}";

    public static string GaussianNoiseGlsl => Header + @"
layout(set = 0, binding = 0) writeonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint size; float mean; float stddev; uint seedLo; uint seedHi; };
" + PhiloxPrng + @"
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    uvec2 rng = philox2x32(uvec2(idx, seedHi), seedLo);
    float u1 = max(uint_to_float01(rng.x), 1e-10);
    float u2 = uint_to_float01(rng.y);
    // Box-Muller transform
    float z = sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
    b[idx] = mean + stddev * z;
}";

    public static string GumbelSoftmaxGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; float temperature; uint seedLo; uint seedHi; };
" + PhiloxPrng + @"
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    uint base_idx = row * innerSize;

    // Add Gumbel noise and divide by temperature
    float max_val = -1e30;
    for (uint j = 0; j < innerSize; j++) {
        uvec2 rng = philox2x32(uvec2(base_idx + j, seedHi), seedLo);
        float u = max(uint_to_float01(rng.x), 1e-10);
        float gumbel = -log(-log(u));
        float val = (a[base_idx + j] + gumbel) / max(temperature, 1e-7);
        b[base_idx + j] = val;
        max_val = max(max_val, val);
    }
    // Softmax
    float sum_exp = 0.0;
    for (uint j = 0; j < innerSize; j++) {
        b[base_idx + j] = exp(b[base_idx + j] - max_val);
        sum_exp += b[base_idx + j];
    }
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] /= (sum_exp + 1e-10);
}";

    public static string SparsemaxGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint innerSize; };
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= outerSize) return;
    if (innerSize > 256) return; // Safety: skip rows with too many classes
    uint base_idx = row * innerSize;

    // Copy to output and sort in-place (each thread owns its row — no shared memory race)
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] = a[base_idx + j];
    // Sort descending (bubble sort — OK for small innerSize typical in sparsemax)
    for (uint i = 0; i < innerSize; i++) {
        for (uint j = i + 1; j < innerSize; j++) {
            if (b[base_idx + j] > b[base_idx + i]) { float tmp = b[base_idx + i]; b[base_idx + i] = b[base_idx + j]; b[base_idx + j] = tmp; }
        }
    }
    // Find tau using sorted values (now in b[base_idx..])
    float cs = 0.0, tau = 0.0;
    for (uint k = 0; k < innerSize; k++) {
        cs += b[base_idx + k];
        float cand = (cs - 1.0) / float(k + 1);
        if (b[base_idx + k] > cand) tau = cand;
    }
    // Apply sparsemax using original unsorted values
    for (uint j = 0; j < innerSize; j++) b[base_idx + j] = max(a[base_idx + j] - tau, 0.0);
}";

    // =====================================================================
    // Activation kernels (forward + backward)
    // =====================================================================

    public static string Relu6 => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx];
    b[idx] = min(max(x, 0.0), 6.0);
}";

    public static string Relu6Backward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = bdata[idx];
    c[idx] = (x > 0.0 && x < 6.0) ? a[idx] : 0.0;
}";

    public static string PRelu => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; uint alphaSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx];
    float alpha = bdata[idx % alphaSize];
    c[idx] = x >= 0.0 ? x : alpha * x;
}";

    public static string PReluBackwardInput => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer INP { float inp[]; };
layout(set = 0, binding = 2) readonly buffer ALPHA { float alpha[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; uint alphaSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = inp[idx];
    float a = alpha[idx % alphaSize];
    gradIn[idx] = x >= 0.0 ? gradOut[idx] : a * gradOut[idx];
}";

    public static string PReluBackwardAlpha => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer INP { float inp[]; };
layout(set = 0, binding = 2) writeonly buffer GA { float gradAlpha[]; };
layout(push_constant) uniform Params { uint size; uint alphaSize; };
void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= alphaSize) return;
    float sum = 0.0;
    for (uint i = c; i < size; i += alphaSize) {
        if (inp[i] < 0.0) sum += inp[i] * gradOut[i];
    }
    gradAlpha[c] = sum;
}";

    public static string RRelu => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx];
    c[idx] = x >= 0.0 ? x : bdata[idx] * x;
}";

    public static string RReluBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer INP { float inp[]; };
layout(set = 0, binding = 2) readonly buffer NOISE { float noise[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = inp[idx];
    gradIn[idx] = gradOut[idx] * (x >= 0.0 ? 1.0 : noise[idx]);
}";

    public static string Threshold => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint size; float thresh; float val; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    b[idx] = a[idx] > thresh ? a[idx] : val;
}";

    public static string ThresholdBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; float thresh; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = bdata[idx] > thresh ? a[idx] : 0.0;
}";

    public static string ReciprocalBackward => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = bdata[idx];
    c[idx] = -a[idx] / (x * x);
}";

    public static string VarBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * reduceSize;
    if (idx >= total) return;
    uint outer = idx / reduceSize;
    d[idx] = a[outer] * 2.0 * (bdata[idx] - c[outer]) / float(reduceSize);
}";

    public static string StdBackwardGlsl => Header + FiveBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * reduceSize;
    if (idx >= total) return;
    uint outer = idx / reduceSize;
    float s = max(d[outer], 1e-8);
    e[idx] = a[outer] * (bdata[idx] - c[outer]) / (float(reduceSize) * s);
}";

    public static string MaskedFillBackwardGlsl => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = (bdata[idx] != 0.0) ? 0.0 : a[idx];
}";

    public static string WhereBackwardGlsl => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float bdata[]; };
layout(set = 0, binding = 2) buffer C { float c[]; };
layout(set = 0, binding = 3) buffer D { float d[]; };
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float cond = bdata[idx];
    c[idx] = (cond != 0.0) ? a[idx] : 0.0;
    d[idx] = (cond != 0.0) ? 0.0 : a[idx];
}";

    public static string NormBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * reduceSize;
    if (idx >= total) return;
    uint outer = idx / reduceSize;
    float n = max(c[outer], 1e-8);
    d[idx] = a[outer] * bdata[idx] / n;
}";

    public static string LogSumExpBackwardGlsl => Header + FourBufferLayout + @"
layout(push_constant) uniform Params { uint outerSize; uint reduceSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = outerSize * reduceSize;
    if (idx >= total) return;
    uint outer = idx / reduceSize;
    float softmax_val = exp(bdata[idx] - c[outer]);
    d[idx] = a[outer] * softmax_val;
}";

    public static string AvgPool1DGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inLength; uint outLength; uint kernelSize; uint stride; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = batch * channels * outLength;
    if (idx >= total) return;
    uint o = idx % outLength; uint ch = (idx / outLength) % channels; uint b = idx / (outLength * channels);
    uint inOff = (b * channels + ch) * inLength;
    float sum = 0.0; uint cnt = 0;
    for (uint k = 0; k < kernelSize; k++) { uint pos = o * stride + k; if (pos < inLength) { sum += a[inOff + pos]; cnt++; } }
    b[idx] = cnt > 0 ? sum / float(cnt) : 0.0;
}";

    public static string MaxPool1DGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inLength; uint outLength; uint kernelSize; uint stride; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = batch * channels * outLength;
    if (idx >= total) return;
    uint o = idx % outLength; uint ch = (idx / outLength) % channels; uint b = idx / (outLength * channels);
    uint inOff = (b * channels + ch) * inLength;
    float maxVal = -3.402823466e+38;
    for (uint k = 0; k < kernelSize; k++) { uint pos = o * stride + k; if (pos < inLength) maxVal = max(maxVal, a[inOff + pos]); }
    b[idx] = maxVal;
}";

    public static string BilinearUpsample2DGlsl => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batch; uint channels; uint inH; uint inW; uint outH; uint outW; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = batch * channels * outH * outW;
    if (idx >= total) return;
    uint ow = idx % outW; uint oh = (idx / outW) % outH; uint ch = (idx / (outW * outH)) % channels; uint b = idx / (outW * outH * channels);
    float h_ratio = (outH > 1) ? float(inH - 1) / float(outH - 1) : 0.0;
    float w_ratio = (outW > 1) ? float(inW - 1) / float(outW - 1) : 0.0;
    float h_in = float(oh) * h_ratio; float w_in = float(ow) * w_ratio;
    uint h0 = uint(h_in); uint h1 = min(h0 + 1, inH - 1); uint w0 = uint(w_in); uint w1 = min(w0 + 1, inW - 1);
    float hd = h_in - float(h0); float wd = w_in - float(w0);
    uint base_idx = (b * channels + ch) * inH * inW;
    b[idx] = (1-hd)*(1-wd)*a[base_idx+h0*inW+w0] + (1-hd)*wd*a[base_idx+h0*inW+w1] + hd*(1-wd)*a[base_idx+h1*inW+w0] + hd*wd*a[base_idx+h1*inW+w1];
}";

    // ScatterMean — atomic CAS variant. NON-DETERMINISTIC across runs (issue #382);
    // see ScatterMeanDeterministicGlsl below for the atomic-free variant.
    public static string ScatterMeanGlsl => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { int bdata[]; };
layout(set = 0, binding = 2) buffer C { uint c_bits[]; };
layout(set = 0, binding = 3) buffer D { uint d_counts[]; };
layout(push_constant) uniform Params { uint sourceSize; uint featureSize; };

// CAS-based atomic float add using uint reinterpretation (no GL_EXT_shader_atomic_float needed)
void atomicAddFloat(uint index, float val) {
    uint oldBits = c_bits[index];
    uint newBits;
    do {
        float oldVal = uintBitsToFloat(oldBits);
        newBits = floatBitsToUint(oldVal + val);
    } while ((oldBits = atomicCompSwap(c_bits[index], oldBits, newBits)) != oldBits);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= sourceSize) return;
    uint row = idx / featureSize; uint col = idx % featureSize;
    uint targetRow = uint(bdata[row]);
    atomicAddFloat(targetRow * featureSize + col, a[idx]);
    if (col == 0) atomicAdd(d_counts[targetRow], 1u);
}";

    // ScatterMean — bit-deterministic variant (issue #382). One work-item per
    // (dstRow, col) output cell scans numSrcRows in fixed ascending order.
    // No atomic CAS loop; accumulation order is identical across runs.
    // Uses a raw float buffer for output (not uint-bits) since no atomics are needed.
    public static string ScatterMeanDeterministicGlsl => Header + @"
layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { int bdata[]; };
layout(set = 0, binding = 2) buffer C { float c[]; };
layout(set = 0, binding = 3) buffer D { uint d_counts[]; };
layout(push_constant) uniform Params { uint sourceSize; uint outputSize; uint featureSize; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = outputSize * featureSize;
    if (gid >= total) return;
    uint dstRow = gid / featureSize;
    uint col = gid % featureSize;
    uint numSrcRows = sourceSize / featureSize;
    float sum = 0.0;
    uint cnt = 0u;
    for (uint srcRow = 0u; srcRow < numSrcRows; srcRow++) {
        if (uint(bdata[srcRow]) == dstRow) {
            sum += a[srcRow * featureSize + col];
            if (col == 0u) cnt++;
        }
    }
    c[dstRow * featureSize + col] = sum;
    if (col == 0u) d_counts[dstRow] = cnt;
}";

    public static string ScatterMeanDivideGlsl => Header + @"
layout(set = 0, binding = 0) buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float b[]; };
layout(push_constant) uniform Params { uint outputSize; uint featureSize; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= outputSize) return;
    uint row = idx / featureSize;
    float cnt = b[row];
    if (cnt > 0.0) a[idx] /= cnt;
}";

    // =====================================================================
    // Loss function kernels (forward + backward)
    // =====================================================================

    public static string MseLossElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float d = a[idx] - bdata[idx];
    c[idx] = d * d;
}";

    public static string L1LossElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    c[idx] = abs(a[idx] - bdata[idx]);
}";

    public static string HuberLossElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; float delta; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float d = a[idx] - bdata[idx];
    float ad = abs(d);
    c[idx] = (ad <= delta) ? (0.5 * d * d) : (delta * (ad - 0.5 * delta));
}";

    public static string BceWithLogitsElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float x = a[idx]; float t = bdata[idx];
    float ax = abs(x);
    c[idx] = max(x, 0.0) - x * t + log(1.0 + exp(-ax));
}";

    public static string L1LossBatch => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numFeatures; };
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= batchSize) return;
    float sum_abs = 0.0;
    for (uint f = 0; f < numFeatures; f++) sum_abs += abs(a[b * numFeatures + f] - bdata[b * numFeatures + f]);
    c[b] = sum_abs / float(numFeatures);
}";

    public static string HuberLossBatch => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numFeatures; float delta; };
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= batchSize) return;
    float sum = 0.0;
    for (uint f = 0; f < numFeatures; f++) {
        float d = a[b * numFeatures + f] - bdata[b * numFeatures + f];
        float ad = abs(d);
        sum += (ad <= delta) ? (0.5 * d * d) : (delta * (ad - 0.5 * delta));
    }
    c[b] = sum / float(numFeatures);
}";

    public static string NllLossBatch => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint numClasses; };
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= batchSize) return;
    int tc = int(bdata[b]);
    c[b] = (tc >= 0 && tc < int(numClasses)) ? -a[b * numClasses + uint(tc)] : 0.0;
}";

    public static string KlDivElementwise => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint size; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float t = bdata[idx];
    c[idx] = (t > 0.0) ? t * (log(t) - a[idx]) : 0.0;
}";

    public static string HuberLossBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer P { float pred[]; };
layout(set = 0, binding = 2) readonly buffer T { float tgt[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; float invN; float delta; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float d = pred[idx] - tgt[idx];
    float ad = abs(d);
    float grad = (ad <= delta) ? d : (delta * ((d > 0.0) ? 1.0 : -1.0));
    gradIn[idx] = gradOut[0] * grad * invN;
}";

    public static string MseLossBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer P { float pred[]; };
layout(set = 0, binding = 2) readonly buffer T { float tgt[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; float invN; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    gradIn[idx] = gradOut[0] * 2.0 * (pred[idx] - tgt[idx]) * invN;
}";

    public static string L1LossBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer P { float pred[]; };
layout(set = 0, binding = 2) readonly buffer T { float tgt[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; float invN; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float d = pred[idx] - tgt[idx];
    float s = (d > 0.0) ? 1.0 : ((d < 0.0) ? -1.0 : 0.0);
    gradIn[idx] = gradOut[0] * s * invN;
}";

    public static string BceWithLogitsBackward => Header + @"
layout(set = 0, binding = 0) readonly buffer GO { float gradOut[]; };
layout(set = 0, binding = 1) readonly buffer L { float logits[]; };
layout(set = 0, binding = 2) readonly buffer T { float tgt[]; };
layout(set = 0, binding = 3) writeonly buffer GI { float gradIn[]; };
layout(push_constant) uniform Params { uint size; float invN; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= size) return;
    float sig = 1.0 / (1.0 + exp(-logits[idx]));
    gradIn[idx] = gradOut[0] * (sig - tgt[idx]) * invN;
}";

    // =====================================================================
    // Hyperbolic Geometry Operations (Poincaré Ball Model)
    // =====================================================================

    /// <summary>
    /// Projects points onto the Poincaré ball: clamps ||x|| &lt; 1/sqrt(c) - eps.
    /// Push constants: batchSize, dim, curvature, epsilon.
    /// Buffers: binding(0) = input [batchSize * dim], binding(1) = output [batchSize * dim].
    /// One thread per batch element.
    /// </summary>
    public static string PoincareProject => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint dim; float curvature; float epsilon; };
void main() {
    uint bi = gl_GlobalInvocationID.x;
    if (bi >= batchSize) return;
    uint off = bi * dim;
    float sqNorm = 0.0;
    for (uint d = 0; d < dim; d++) sqNorm += a[off + d] * a[off + d];
    float maxNorm = (1.0 / sqrt(curvature)) - epsilon;
    float maxNormSq = maxNorm * maxNorm;
    float scale = sqNorm >= maxNormSq ? maxNorm / sqrt(sqNorm) : 1.0;
    for (uint d = 0; d < dim; d++) b[off + d] = a[off + d] * scale;
}";

    /// <summary>
    /// Möbius addition: x ⊕_c y in the Poincaré ball.
    /// Push constants: batchSize, dim, curvature.
    /// Buffers: binding(0) = x, binding(1) = y, binding(2) = output. All [batchSize * dim].
    /// </summary>
    public static string MobiusAdd => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint dim; float curvature; };
void main() {
    uint bi = gl_GlobalInvocationID.x;
    if (bi >= batchSize) return;
    uint off = bi * dim;
    float cv = curvature;
    float xSq = 0.0, ySq = 0.0, xy = 0.0;
    for (uint d = 0; d < dim; d++) {
        float xv = a[off + d], yv = bdata[off + d];
        xSq += xv * xv; ySq += yv * yv; xy += xv * yv;
    }
    float denom = 1.0 + 2.0 * cv * xy + cv * cv * xSq * ySq;
    denom = max(abs(denom), 1e-10);
    float numX = 1.0 + 2.0 * cv * xy + cv * ySq;
    float numY = 1.0 - cv * xSq;
    for (uint d = 0; d < dim; d++)
        c[off + d] = (numX * a[off + d] + numY * bdata[off + d]) / denom;
}";

    /// <summary>
    /// Poincaré exponential map: exp_x(v) = x ⊕_c tanh(√c·λ_x·||v||/2)·v/(√c·||v||).
    /// Push constants: batchSize, dim, curvature.
    /// Buffers: binding(0) = basePoint, binding(1) = tangentVec, binding(2) = output.
    /// </summary>
    public static string PoincareExpMap => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint dim; float curvature; };
void main() {
    uint bi = gl_GlobalInvocationID.x;
    if (bi >= batchSize) return;
    uint off = bi * dim;
    float cv = curvature;
    float sqrtC = sqrt(cv);
    // Compute ||base||^2 and ||tangent||
    float bpSq = 0.0, tvSq = 0.0;
    for (uint d = 0; d < dim; d++) {
        bpSq += a[off + d] * a[off + d];
        tvSq += bdata[off + d] * bdata[off + d];
    }
    float tvNorm = sqrt(max(tvSq, 1e-20));
    float lambda = 2.0 / max(1.0 - cv * bpSq, 1e-10);
    // If tangent is near-zero, output = base point
    if (tvNorm < 1e-10) {
        for (uint d = 0; d < dim; d++) c[off + d] = a[off + d];
        return;
    }
    float t = tanh(sqrtC * lambda * tvNorm / 2.0) / (sqrtC * tvNorm);
    // scaled = t * tangentVec, then Mobius add base + scaled
    float sSq = 0.0, bs = 0.0;
    for (uint d = 0; d < dim; d++) {
        float sv = t * bdata[off + d];
        sSq += sv * sv;
        bs += a[off + d] * sv;
    }
    float denom = 1.0 + 2.0 * cv * bs + cv * cv * bpSq * sSq;
    denom = max(abs(denom), 1e-10);
    float numX = 1.0 + 2.0 * cv * bs + cv * sSq;
    float numY = 1.0 - cv * bpSq;
    for (uint d = 0; d < dim; d++)
        c[off + d] = (numX * a[off + d] + numY * t * bdata[off + d]) / denom;
}";

    /// <summary>
    /// Poincaré distance: d(x,y) = (2/√c) · arctanh(√c · ||(-x) ⊕ y||).
    /// Push constants: batchSize, dim, curvature.
    /// Buffers: binding(0) = x [batchSize*dim], binding(1) = y [batchSize*dim], binding(2) = output [batchSize].
    /// </summary>
    public static string PoincareDistance => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint dim; float curvature; };
void main() {
    uint b = gl_GlobalInvocationID.x;
    if (b >= batchSize) return;
    uint off = b * dim;
    float cv = curvature;
    float diffSq = 0.0, xSq = 0.0, ySq = 0.0;
    for (uint d = 0; d < dim; d++) {
        float diff = a[off + d] - bdata[off + d];
        diffSq += diff * diff;
        xSq += a[off + d] * a[off + d];
        ySq += bdata[off + d] * bdata[off + d];
    }
    float denomX = max(abs(1.0 - cv * xSq), 1e-10);
    float denomY = max(abs(1.0 - cv * ySq), 1e-10);
    float arg = max(1.0 + 2.0 * cv * diffSq / (denomX * denomY), 1.0);
    // acosh(x) = log(x + sqrt(x*x - 1))
    float dist = log(arg + sqrt(max(arg * arg - 1.0, 0.0))) / sqrt(cv);
    c[b] = dist;
}";

    // =====================================================================
    // Octonion Algebra Operations
    // =====================================================================

    /// <summary>
    /// Octonion element-wise multiplication using Cayley-Dickson construction.
    /// Push constants: count (number of octonion pairs).
    /// Buffers: binding(0) = a [count*8], binding(1) = b [count*8], binding(2) = output [count*8].
    /// One thread per octonion.
    /// </summary>
    public static string OctonionMultiply => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint count; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= count) return;
    uint o = i * 8;
    float a0=a[o],a1=a[o+1],a2=a[o+2],a3=a[o+3],a4=a[o+4],a5=a[o+5],a6=a[o+6],a7=a[o+7];
    float b0=bdata[o],b1=bdata[o+1],b2=bdata[o+2],b3=bdata[o+3],b4=bdata[o+4],b5=bdata[o+5],b6=bdata[o+6],b7=bdata[o+7];
    c[o+0] = a0*b0-a1*b1-a2*b2-a3*b3-a4*b4-a5*b5-a6*b6-a7*b7;
    c[o+1] = a0*b1+a1*b0+a2*b3-a3*b2+a4*b5-a5*b4-a6*b7+a7*b6;
    c[o+2] = a0*b2-a1*b3+a2*b0+a3*b1+a4*b6+a5*b7-a6*b4-a7*b5;
    c[o+3] = a0*b3+a1*b2-a2*b1+a3*b0+a4*b7-a5*b6+a6*b5-a7*b4;
    c[o+4] = a0*b4-a1*b5-a2*b6-a3*b7+a4*b0+a5*b1+a6*b2+a7*b3;
    c[o+5] = a0*b5+a1*b4-a2*b7+a3*b6-a4*b1+a5*b0-a6*b3+a7*b2;
    c[o+6] = a0*b6+a1*b7+a2*b4-a3*b5-a4*b2+a5*b3+a6*b0-a7*b1;
    c[o+7] = a0*b7-a1*b6+a2*b5+a3*b4-a4*b3-a5*b2+a6*b1+a7*b0;
}";

    /// <summary>
    /// Octonion linear layer forward: output[b,o] = sum_i(weight[o,i] * input[b,i]) + bias[o].
    /// Push constants: batchSize, inputFeatures, outputFeatures.
    /// Buffers: binding(0) = input [B*I*8], binding(1) = weights [O*I*8],
    ///          binding(2) = biases [O*8], binding(3) = output [B*O*8].
    /// One thread per (batch, outputFeature) pair.
    /// </summary>
    public static string OctonionLinearForward => Header + @"
layout(set = 0, binding = 0) readonly buffer IN { float inp[]; };
layout(set = 0, binding = 1) readonly buffer W  { float wt[]; };
layout(set = 0, binding = 2) readonly buffer BI { float bias[]; };
layout(set = 0, binding = 3) writeonly buffer OUT { float outp[]; };
layout(push_constant) uniform Params { uint batchSize; uint inputFeatures; uint outputFeatures; };

// Octonion multiply: r = a * b (Cayley-Dickson)
void oct_mul(float a0,float a1,float a2,float a3,float a4,float a5,float a6,float a7,
             float b0,float b1,float b2,float b3,float b4,float b5,float b6,float b7,
             out float r0,out float r1,out float r2,out float r3,
             out float r4,out float r5,out float r6,out float r7) {
    r0=a0*b0-a1*b1-a2*b2-a3*b3-a4*b4-a5*b5-a6*b6-a7*b7;
    r1=a0*b1+a1*b0+a2*b3-a3*b2+a4*b5-a5*b4-a6*b7+a7*b6;
    r2=a0*b2-a1*b3+a2*b0+a3*b1+a4*b6+a5*b7-a6*b4-a7*b5;
    r3=a0*b3+a1*b2-a2*b1+a3*b0+a4*b7-a5*b6+a6*b5-a7*b4;
    r4=a0*b4-a1*b5-a2*b6-a3*b7+a4*b0+a5*b1+a6*b2+a7*b3;
    r5=a0*b5+a1*b4-a2*b7+a3*b6-a4*b1+a5*b0-a6*b3+a7*b2;
    r6=a0*b6+a1*b7+a2*b4-a3*b5-a4*b2+a5*b3+a6*b0-a7*b1;
    r7=a0*b7-a1*b6+a2*b5+a3*b4-a4*b3-a5*b2+a6*b1+a7*b0;
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint totalPairs = batchSize * outputFeatures;
    if (tid >= totalPairs) return;
    uint b = tid / outputFeatures;
    uint o = tid % outputFeatures;

    // Accumulate: sum_i weight[o,i] * input[b,i]
    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    for (uint i = 0; i < inputFeatures; i++) {
        uint wi = (o * inputFeatures + i) * 8;
        uint ii = (b * inputFeatures + i) * 8;
        float r0,r1,r2,r3,r4,r5,r6,r7;
        // weight * input (non-commutative order matches CpuEngine)
        oct_mul(wt[wi],wt[wi+1],wt[wi+2],wt[wi+3],wt[wi+4],wt[wi+5],wt[wi+6],wt[wi+7],
                inp[ii],inp[ii+1],inp[ii+2],inp[ii+3],inp[ii+4],inp[ii+5],inp[ii+6],inp[ii+7],
                r0,r1,r2,r3,r4,r5,r6,r7);
        s0+=r0;s1+=r1;s2+=r2;s3+=r3;s4+=r4;s5+=r5;s6+=r6;s7+=r7;
    }
    // Add bias
    uint bo = o * 8;
    uint oo = (b * outputFeatures + o) * 8;
    outp[oo]=s0+bias[bo]; outp[oo+1]=s1+bias[bo+1]; outp[oo+2]=s2+bias[bo+2]; outp[oo+3]=s3+bias[bo+3];
    outp[oo+4]=s4+bias[bo+4]; outp[oo+5]=s5+bias[bo+5]; outp[oo+6]=s6+bias[bo+6]; outp[oo+7]=s7+bias[bo+7];
}";

    /// <summary>
    /// Octonion linear backward (input gradient) using multiplication table Jacobian.
    /// Push constants: batchSize, inputFeatures, outputFeatures.
    /// Buffers: binding(0)=gradOut [B*O*8], binding(1)=weights [O*I*8], binding(2)=gradInput [B*I*8].
    /// One thread per (batch, inputFeature).
    /// </summary>
    public static string OctonionLinearBackwardInput => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint inputFeatures; uint outputFeatures; };
void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint totalPairs = batchSize * inputFeatures;
    if (tid >= totalPairs) return;
    uint b = tid / inputFeatures;
    uint i = tid % inputFeatures;

    // Accumulate gradient via full Jacobian transpose of d(w*a)/da
    float ga0=0,ga1=0,ga2=0,ga3=0,ga4=0,ga5=0,ga6=0,ga7=0;
    for (uint o = 0; o < outputFeatures; o++) {
        uint goOff = (b * outputFeatures + o) * 8;
        uint wOff = (o * inputFeatures + i) * 8;
        float w0=bdata[wOff],w1=bdata[wOff+1],w2=bdata[wOff+2],w3=bdata[wOff+3],
              w4=bdata[wOff+4],w5=bdata[wOff+5],w6=bdata[wOff+6],w7=bdata[wOff+7];
        float g0=a[goOff],g1=a[goOff+1],g2=a[goOff+2],g3=a[goOff+3],
              g4=a[goOff+4],g5=a[goOff+5],g6=a[goOff+6],g7=a[goOff+7];
        // Jacobian transpose of d(w*a)/da from the octonion multiplication table
        ga0+=g0*w0+g1*w1+g2*w2+g3*w3+g4*w4+g5*w5+g6*w6+g7*w7;
        ga1+=g0*(-w1)+g1*w0+g2*(-w3)+g3*w2+g4*(-w5)+g5*w4+g6*w7+g7*(-w6);
        ga2+=g0*(-w2)+g1*w3+g2*w0+g3*(-w1)+g4*(-w6)+g5*(-w7)+g6*w4+g7*w5;
        ga3+=g0*(-w3)+g1*(-w2)+g2*w1+g3*w0+g4*(-w7)+g5*w6+g6*(-w5)+g7*w4;
        ga4+=g0*(-w4)+g1*w5+g2*w6+g3*w7+g4*w0+g5*(-w1)+g6*(-w2)+g7*(-w3);
        ga5+=g0*(-w5)+g1*(-w4)+g2*w7+g3*(-w6)+g4*w1+g5*w0+g6*w3+g7*(-w2);
        ga6+=g0*(-w6)+g1*(-w7)+g2*(-w4)+g3*w5+g4*w2+g5*(-w3)+g6*w0+g7*w1;
        ga7+=g0*(-w7)+g1*w6+g2*(-w5)+g3*(-w4)+g4*w3+g5*w2+g6*(-w1)+g7*w0;
    }
    uint giOff = (b * inputFeatures + i) * 8;
    c[giOff]=ga0; c[giOff+1]=ga1; c[giOff+2]=ga2; c[giOff+3]=ga3;
    c[giOff+4]=ga4; c[giOff+5]=ga5; c[giOff+6]=ga6; c[giOff+7]=ga7;
}";

    /// <summary>
    /// Octonion linear backward (weight gradient): gradW[o,i] = sum_b gradOut[b,o]^T * input[b,i].
    /// Push constants: batchSize, inputFeatures, outputFeatures.
    /// Buffers: binding(0)=gradOut [B*O*8], binding(1)=input [B*I*8], binding(2)=gradWeights [O*I*8].
    /// One thread per (outputFeature, inputFeature).
    /// </summary>
    public static string OctonionLinearBackwardWeights => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint inputFeatures; uint outputFeatures; };
void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint totalPairs = outputFeatures * inputFeatures;
    if (tid >= totalPairs) return;
    uint o = tid / inputFeatures;
    uint i = tid % inputFeatures;
    // Accumulate gradient via full Jacobian transpose of d(w*a)/dw
    float gw0=0,gw1=0,gw2=0,gw3=0,gw4=0,gw5=0,gw6=0,gw7=0;
    for (uint b = 0; b < batchSize; b++) {
        uint inOff = (b * inputFeatures + i) * 8;
        uint goOff = (b * outputFeatures + o) * 8;
        float a0=bdata[inOff],a1=bdata[inOff+1],a2=bdata[inOff+2],a3=bdata[inOff+3],
              a4=bdata[inOff+4],a5=bdata[inOff+5],a6=bdata[inOff+6],a7=bdata[inOff+7];
        float g0=a[goOff],g1=a[goOff+1],g2=a[goOff+2],g3=a[goOff+3],
              g4=a[goOff+4],g5=a[goOff+5],g6=a[goOff+6],g7=a[goOff+7];
        // Jacobian transpose of d(w*a)/dw from the octonion multiplication table
        gw0+=g0*a0+g1*a1+g2*a2+g3*a3+g4*a4+g5*a5+g6*a6+g7*a7;
        gw1+=g0*(-a1)+g1*a0+g2*a3+g3*(-a2)+g4*a5+g5*(-a4)+g6*(-a7)+g7*a6;
        gw2+=g0*(-a2)+g1*(-a3)+g2*a0+g3*a1+g4*a6+g5*a7+g6*(-a4)+g7*(-a5);
        gw3+=g0*(-a3)+g1*a2+g2*(-a1)+g3*a0+g4*a7+g5*(-a6)+g6*a5+g7*(-a4);
        gw4+=g0*(-a4)+g1*(-a5)+g2*(-a6)+g3*(-a7)+g4*a0+g5*a1+g6*a2+g7*a3;
        gw5+=g0*(-a5)+g1*a4+g2*(-a7)+g3*a6+g4*(-a1)+g5*a0+g6*(-a3)+g7*a2;
        gw6+=g0*(-a6)+g1*a7+g2*a4+g3*(-a5)+g4*(-a2)+g5*a3+g6*a0+g7*(-a1);
        gw7+=g0*(-a7)+g1*(-a6)+g2*a5+g3*a4+g4*(-a3)+g5*(-a2)+g6*a1+g7*a0;
    }
    uint gwOff = (o * inputFeatures + i) * 8;
    c[gwOff]=gw0; c[gwOff+1]=gw1; c[gwOff+2]=gw2; c[gwOff+3]=gw3;
    c[gwOff+4]=gw4; c[gwOff+5]=gw5; c[gwOff+6]=gw6; c[gwOff+7]=gw7;
}";

    /// <summary>
    /// Octonion linear backward (bias gradient): gradBias[o] = sum_b gradOut[b,o].
    /// Push constants: batchSize, outputFeatures.
    /// Buffers: binding(0)=gradOut [B*O*8], binding(1)=gradBiases [O*8].
    /// One thread per outputFeature.
    /// </summary>
    public static string OctonionLinearBackwardBiases => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint batchSize; uint outputFeatures; };
void main() {
    uint o = gl_GlobalInvocationID.x;
    if (o >= outputFeatures) return;
    for (uint cc = 0; cc < 8; cc++) {
        float sum = 0.0;
        for (uint bat = 0; bat < batchSize; bat++)
            sum += a[(bat * outputFeatures + o) * 8 + cc];
        b[o * 8 + cc] = sum;
    }
}";

    /// <summary>
    /// Hyperbolic linear forward: Euclidean matmul → bias add → Poincaré projection.
    /// Push constants: batchSize, inputFeatures, outputFeatures, curvature, epsilon.
    /// Buffers: binding(0)=input [B*I], binding(1)=weights [O*I], binding(2)=biases [O],
    ///          binding(3)=output [B*O].
    /// One thread per (batch, outputFeature).
    /// </summary>
    /// <summary>
    /// Non-fused hyperbolic linear: matmul + bias only. Projection done separately.
    /// </summary>
    public static string HyperbolicLinearForward => Header + @"
layout(set = 0, binding = 0) readonly buffer IN { float inp[]; };
layout(set = 0, binding = 1) readonly buffer W  { float wt[]; };
layout(set = 0, binding = 2) readonly buffer BI { float bias[]; };
layout(set = 0, binding = 3) writeonly buffer OUT { float outp[]; };
layout(push_constant) uniform Params { uint batchSize; uint inputFeatures; uint outputFeatures; float curvature; float epsilon; };
void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint total = batchSize * outputFeatures;
    if (tid >= total) return;
    uint b = tid / outputFeatures;
    uint o = tid % outputFeatures;
    float val = bias[o];
    for (uint i = 0; i < inputFeatures; i++)
        val += inp[b * inputFeatures + i] * wt[o * inputFeatures + i];
    outp[b * outputFeatures + o] = val;
}";

    /// <summary>
    /// Fused octonion linear forward + ReLU activation.
    /// Combines octonion matmul + bias + ReLU in a single kernel launch.
    /// ReLU is applied component-wise to each of the 8 octonion components.
    /// </summary>
    public static string OctonionLinearForwardFusedReLU => Header + @"
layout(set = 0, binding = 0) readonly buffer IN { float inp[]; };
layout(set = 0, binding = 1) readonly buffer W  { float wt[]; };
layout(set = 0, binding = 2) readonly buffer BI { float bias[]; };
layout(set = 0, binding = 3) writeonly buffer OUT { float outp[]; };
layout(push_constant) uniform Params { uint batchSize; uint inputFeatures; uint outputFeatures; };

void oct_mul(float a0,float a1,float a2,float a3,float a4,float a5,float a6,float a7,
             float b0,float b1,float b2,float b3,float b4,float b5,float b6,float b7,
             out float r0,out float r1,out float r2,out float r3,
             out float r4,out float r5,out float r6,out float r7) {
    r0=a0*b0-a1*b1-a2*b2-a3*b3-a4*b4-a5*b5-a6*b6-a7*b7;
    r1=a0*b1+a1*b0+a2*b3-a3*b2+a4*b5-a5*b4-a6*b7+a7*b6;
    r2=a0*b2-a1*b3+a2*b0+a3*b1+a4*b6+a5*b7-a6*b4-a7*b5;
    r3=a0*b3+a1*b2-a2*b1+a3*b0+a4*b7-a5*b6+a6*b5-a7*b4;
    r4=a0*b4-a1*b5-a2*b6-a3*b7+a4*b0+a5*b1+a6*b2+a7*b3;
    r5=a0*b5+a1*b4-a2*b7+a3*b6-a4*b1+a5*b0-a6*b3+a7*b2;
    r6=a0*b6+a1*b7+a2*b4-a3*b5-a4*b2+a5*b3+a6*b0-a7*b1;
    r7=a0*b7-a1*b6+a2*b5+a3*b4-a4*b3-a5*b2+a6*b1+a7*b0;
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint totalPairs = batchSize * outputFeatures;
    if (tid >= totalPairs) return;
    uint b = tid / outputFeatures;
    uint o = tid % outputFeatures;
    float s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    for (uint i = 0; i < inputFeatures; i++) {
        uint wi = (o * inputFeatures + i) * 8;
        uint ii = (b * inputFeatures + i) * 8;
        float r0,r1,r2,r3,r4,r5,r6,r7;
        oct_mul(wt[wi],wt[wi+1],wt[wi+2],wt[wi+3],wt[wi+4],wt[wi+5],wt[wi+6],wt[wi+7],
                inp[ii],inp[ii+1],inp[ii+2],inp[ii+3],inp[ii+4],inp[ii+5],inp[ii+6],inp[ii+7],
                r0,r1,r2,r3,r4,r5,r6,r7);
        s0+=r0;s1+=r1;s2+=r2;s3+=r3;s4+=r4;s5+=r5;s6+=r6;s7+=r7;
    }
    uint bo = o * 8;
    uint oo = (b * outputFeatures + o) * 8;
    // Fused bias + ReLU
    outp[oo]  =max(s0+bias[bo],  0.0); outp[oo+1]=max(s1+bias[bo+1],0.0);
    outp[oo+2]=max(s2+bias[bo+2],0.0); outp[oo+3]=max(s3+bias[bo+3],0.0);
    outp[oo+4]=max(s4+bias[bo+4],0.0); outp[oo+5]=max(s5+bias[bo+5],0.0);
    outp[oo+6]=max(s6+bias[bo+6],0.0); outp[oo+7]=max(s7+bias[bo+7],0.0);
}";

    // Fused Linear + Activation GLSL kernels
    private const string FusedLinearLayout = @"
layout(set = 0, binding = 0) readonly buffer Input { float inp[]; };
layout(set = 0, binding = 1) readonly buffer Weight { float wt[]; };
layout(set = 0, binding = 2) readonly buffer Bias { float bias[]; };
layout(set = 0, binding = 3) writeonly buffer Output { float outp[]; };
layout(push_constant) uniform Params { uint batchSize; uint inFeatures; uint outFeatures; };
";

    public static string FusedLinearReLU => Header + FusedLinearLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*outFeatures)return; uint b=idx/outFeatures,j=idx%outFeatures; float sum=bias[j]; for(uint k=0;k<inFeatures;k++)sum+=inp[b*inFeatures+k]*wt[k*outFeatures+j]; outp[idx]=max(sum,0.0); }";
    public static string FusedLinearSigmoid => Header + FusedLinearLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*outFeatures)return; uint b=idx/outFeatures,j=idx%outFeatures; float sum=bias[j]; for(uint k=0;k<inFeatures;k++)sum+=inp[b*inFeatures+k]*wt[k*outFeatures+j]; outp[idx]=1.0/(1.0+exp(-sum)); }";
    public static string FusedLinearTanh => Header + FusedLinearLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*outFeatures)return; uint b=idx/outFeatures,j=idx%outFeatures; float sum=bias[j]; for(uint k=0;k<inFeatures;k++)sum+=inp[b*inFeatures+k]*wt[k*outFeatures+j]; outp[idx]=tanh(sum); }";
    public static string FusedLinearGELU => Header + FusedLinearLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*outFeatures)return; uint b=idx/outFeatures,j=idx%outFeatures; float sum=bias[j]; for(uint k=0;k<inFeatures;k++)sum+=inp[b*inFeatures+k]*wt[k*outFeatures+j]; outp[idx]=0.5*sum*(1.0+tanh(0.7978845608*(sum+0.044715*sum*sum*sum))); }";
    public static string FusedLinearSwish => Header + FusedLinearLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*outFeatures)return; uint b=idx/outFeatures,j=idx%outFeatures; float sum=bias[j]; for(uint k=0;k<inFeatures;k++)sum+=inp[b*inFeatures+k]*wt[k*outFeatures+j]; outp[idx]=sum/(1.0+exp(-sum)); }";
    public static string FusedLinearIdentity => Header + FusedLinearLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*outFeatures)return; uint b=idx/outFeatures,j=idx%outFeatures; float sum=bias[j]; for(uint k=0;k<inFeatures;k++)sum+=inp[b*inFeatures+k]*wt[k*outFeatures+j]; outp[idx]=sum; }";

    private const string FusedLinearLeakyReluLayout = @"
layout(set = 0, binding = 0) readonly buffer Input { float inp[]; };
layout(set = 0, binding = 1) readonly buffer Weight { float wt[]; };
layout(set = 0, binding = 2) readonly buffer Bias { float bias[]; };
layout(set = 0, binding = 3) writeonly buffer Output { float outp[]; };
layout(push_constant) uniform Params { uint batchSize; uint inFeatures; uint outFeatures; float alpha; };
";
    public static string FusedLinearLeakyRelu => Header + FusedLinearLeakyReluLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*outFeatures)return; uint b=idx/outFeatures,j=idx%outFeatures; float sum=bias[j]; for(uint k=0;k<inFeatures;k++)sum+=inp[b*inFeatures+k]*wt[k*outFeatures+j]; outp[idx]=sum>=0.0?sum:alpha*sum; }";

    // IoU Loss GLSL kernels
    private const string IoULayout = @"
layout(set = 0, binding = 0) readonly buffer Pred { float pred[]; };
layout(set = 0, binding = 1) readonly buffer Target { float targ[]; };
layout(set = 0, binding = 2) writeonly buffer Loss { float loss[]; };
layout(push_constant) uniform Params { uint numBoxes; };
";

    public static string IoULoss => Header + IoULayout + @"void main() { uint i=gl_GlobalInvocationID.x; if(i>=numBoxes)return; uint o=i*4; float px1=pred[o],py1=pred[o+1],px2=pred[o+2],py2=pred[o+3]; float tx1=targ[o],ty1=targ[o+1],tx2=targ[o+2],ty2=targ[o+3]; float iW=max(0.0,min(px2,tx2)-max(px1,tx1)),iH=max(0.0,min(py2,ty2)-max(py1,ty1)); float iA=iW*iH,pA=max(0.0,px2-px1)*max(0.0,py2-py1),tA=max(0.0,tx2-tx1)*max(0.0,ty2-ty1),uA=pA+tA-iA+1e-7; loss[i]=1.0-iA/uA; }";
    public static string GIoULoss => Header + IoULayout + @"void main() { uint i=gl_GlobalInvocationID.x; if(i>=numBoxes)return; uint o=i*4; float px1=pred[o],py1=pred[o+1],px2=pred[o+2],py2=pred[o+3]; float tx1=targ[o],ty1=targ[o+1],tx2=targ[o+2],ty2=targ[o+3]; float iW=max(0.0,min(px2,tx2)-max(px1,tx1)),iH=max(0.0,min(py2,ty2)-max(py1,ty1)); float iA=iW*iH,pA=max(0.0,px2-px1)*max(0.0,py2-py1),tA=max(0.0,tx2-tx1)*max(0.0,ty2-ty1),uA=pA+tA-iA+1e-7; float iou=iA/uA; float eA=(max(px2,tx2)-min(px1,tx1))*(max(py2,ty2)-min(py1,ty1))+1e-7; loss[i]=1.0-(iou-(eA-uA)/eA); }";
    public static string DIoULoss => Header + IoULayout + @"void main() { uint i=gl_GlobalInvocationID.x; if(i>=numBoxes)return; uint o=i*4; float px1=pred[o],py1=pred[o+1],px2=pred[o+2],py2=pred[o+3]; float tx1=targ[o],ty1=targ[o+1],tx2=targ[o+2],ty2=targ[o+3]; float iW=max(0.0,min(px2,tx2)-max(px1,tx1)),iH=max(0.0,min(py2,ty2)-max(py1,ty1)); float iA=iW*iH,pA=max(0.0,px2-px1)*max(0.0,py2-py1),tA=max(0.0,tx2-tx1)*max(0.0,ty2-ty1),uA=pA+tA-iA+1e-7; float iou=iA/uA; float dx=0.5*(px1+px2)-0.5*(tx1+tx2),dy=0.5*(py1+py2)-0.5*(ty1+ty2); float cds=dx*dx+dy*dy; float eDx=max(px2,tx2)-min(px1,tx1),eDy=max(py2,ty2)-min(py1,ty1); float ds=eDx*eDx+eDy*eDy+1e-7; loss[i]=1.0-(iou-cds/ds); }";
    public static string CIoULoss => Header + IoULayout + @"void main() { uint i=gl_GlobalInvocationID.x; if(i>=numBoxes)return; uint o=i*4; float px1=pred[o],py1=pred[o+1],px2=pred[o+2],py2=pred[o+3]; float tx1=targ[o],ty1=targ[o+1],tx2=targ[o+2],ty2=targ[o+3]; float iW=max(0.0,min(px2,tx2)-max(px1,tx1)),iH=max(0.0,min(py2,ty2)-max(py1,ty1)); float iA=iW*iH,pA=max(0.0,px2-px1)*max(0.0,py2-py1),tA=max(0.0,tx2-tx1)*max(0.0,ty2-ty1),uA=pA+tA-iA+1e-7; float iou=iA/uA; float dx=0.5*(px1+px2)-0.5*(tx1+tx2),dy=0.5*(py1+py2)-0.5*(ty1+ty2); float cds=dx*dx+dy*dy; float eDx=max(px2,tx2)-min(px1,tx1),eDy=max(py2,ty2)-min(py1,ty1); float ds=eDx*eDx+eDy*eDy+1e-7; float pw=max(px2-px1,1e-7),ph=max(py2-py1,1e-7),tw=max(tx2-tx1,1e-7),th=max(ty2-ty1,1e-7); float rd=atan(tw/th)-atan(pw/ph); float v=(4.0/(3.14159265*3.14159265))*rd*rd; float alpha=v/(1.0-iou+v+1e-7); loss[i]=1.0-(iou-cds/ds-alpha*v); }";

    // Fused Linear Backward — Grad Input kernels (4 buffers: gradOutput, weight, saved, gradInput)
    private const string FusedLinearBackwardGradInputLayout = @"
layout(set = 0, binding = 0) readonly buffer GradOutput { float go[]; };
layout(set = 0, binding = 1) readonly buffer Weight { float wt[]; };
layout(set = 0, binding = 2) readonly buffer Saved { float saved[]; };
layout(set = 0, binding = 3) writeonly buffer GradInput { float gi[]; };
layout(push_constant) uniform Params { uint batchSize; uint inFeatures; uint outFeatures; };
";

    public static string FusedLinearReLUBackwardGradInput => Header + FusedLinearBackwardGradInputLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*inFeatures)return; uint b=idx/inFeatures,i=idx%inFeatures; float s=0.0; for(uint j=0;j<outFeatures;j++){float g=go[b*outFeatures+j]; float masked=saved[b*outFeatures+j]>0.0?g:0.0; s+=masked*wt[i*outFeatures+j];} gi[idx]=s; }";
    public static string FusedLinearSigmoidBackwardGradInput => Header + FusedLinearBackwardGradInputLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*inFeatures)return; uint b=idx/inFeatures,i=idx%inFeatures; float s=0.0; for(uint j=0;j<outFeatures;j++){float g=go[b*outFeatures+j]; float o=saved[b*outFeatures+j]; float masked=g*o*(1.0-o); s+=masked*wt[i*outFeatures+j];} gi[idx]=s; }";
    public static string FusedLinearTanhBackwardGradInput => Header + FusedLinearBackwardGradInputLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*inFeatures)return; uint b=idx/inFeatures,i=idx%inFeatures; float s=0.0; for(uint j=0;j<outFeatures;j++){float g=go[b*outFeatures+j]; float o=saved[b*outFeatures+j]; float masked=g*(1.0-o*o); s+=masked*wt[i*outFeatures+j];} gi[idx]=s; }";
    public static string FusedLinearGELUBackwardGradInput => Header + FusedLinearBackwardGradInputLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*inFeatures)return; uint b=idx/inFeatures,i=idx%inFeatures; float s=0.0; for(uint j=0;j<outFeatures;j++){float g=go[b*outFeatures+j]; float p=saved[b*outFeatures+j]; float t=tanh(0.7978845608*(p+0.044715*p*p*p)); float dt=0.7978845608*(1.0+3.0*0.044715*p*p)*(1.0-t*t); float masked=g*(0.5*(1.0+t)+0.5*p*dt); s+=masked*wt[i*outFeatures+j];} gi[idx]=s; }";
    public static string FusedLinearSwishBackwardGradInput => Header + FusedLinearBackwardGradInputLayout + @"void main() { uint idx=gl_GlobalInvocationID.x; if(idx>=batchSize*inFeatures)return; uint b=idx/inFeatures,i=idx%inFeatures; float s=0.0; for(uint j=0;j<outFeatures;j++){float g=go[b*outFeatures+j]; float p=saved[b*outFeatures+j]; float sig=1.0/(1.0+exp(-p)); float masked=g*(sig+p*sig*(1.0-sig)); s+=masked*wt[i*outFeatures+j];} gi[idx]=s; }";

    // Fused Linear Backward — Weight Gradient kernel (4 buffers: gradOutput, input, saved, gradWeight)
    // activationType: 0=ReLU, 1=Sigmoid, 2=Tanh, 3=GELU, 4=Swish
    private const string FusedLinearBackwardWeightGradLayout = @"
layout(set = 0, binding = 0) readonly buffer GradOutput { float go[]; };
layout(set = 0, binding = 1) readonly buffer Input { float inp[]; };
layout(set = 0, binding = 2) readonly buffer Saved { float saved[]; };
layout(set = 0, binding = 3) writeonly buffer GradWeight { float gw[]; };
layout(push_constant) uniform Params { uint batchSize; uint inFeatures; uint outFeatures; uint activationType; };
";

    public static string FusedLinearWeightGrad => Header + FusedLinearBackwardWeightGradLayout + @"
float applyBackward(float g, float s, uint act) {
    if(act==0u) return s>0.0?g:0.0;
    if(act==1u) return g*s*(1.0-s);
    if(act==2u) return g*(1.0-s*s);
    if(act==3u){float t=tanh(0.7978845608*(s+0.044715*s*s*s)); float dt=0.7978845608*(1.0+3.0*0.044715*s*s)*(1.0-t*t); return g*(0.5*(1.0+t)+0.5*s*dt);}
    if(act==4u){float sig=1.0/(1.0+exp(-s)); return g*(sig+s*sig*(1.0-sig));}
    return g;
}
void main() { uint idx=gl_GlobalInvocationID.x; uint total=inFeatures*outFeatures; if(idx>=total)return; uint i=idx/outFeatures,j=idx%outFeatures; float s=0.0; for(uint b=0;b<batchSize;b++){float masked=applyBackward(go[b*outFeatures+j],saved[b*outFeatures+j],activationType); s+=inp[b*inFeatures+i]*masked;} gw[idx]=s; }";

    // Fused Linear Backward — Bias Gradient kernel (3 buffers: gradOutput, saved, gradBias)
    private const string FusedLinearBackwardBiasGradLayout = @"
layout(set = 0, binding = 0) readonly buffer GradOutput { float go[]; };
layout(set = 0, binding = 1) readonly buffer Saved { float saved[]; };
layout(set = 0, binding = 2) writeonly buffer GradBias { float gb[]; };
layout(push_constant) uniform Params { uint batchSize; uint outFeatures; uint activationType; };
";

    public static string FusedLinearBiasGrad => Header + FusedLinearBackwardBiasGradLayout + @"
float applyBackwardBias(float g, float s, uint act) {
    if(act==0u) return s>0.0?g:0.0;
    if(act==1u) return g*s*(1.0-s);
    if(act==2u) return g*(1.0-s*s);
    if(act==3u){float t=tanh(0.7978845608*(s+0.044715*s*s*s)); float dt=0.7978845608*(1.0+3.0*0.044715*s*s)*(1.0-t*t); return g*(0.5*(1.0+t)+0.5*s*dt);}
    if(act==4u){float sig=1.0/(1.0+exp(-s)); return g*(sig+s*sig*(1.0-sig));}
    return g;
}
void main() { uint j=gl_GlobalInvocationID.x; if(j>=outFeatures)return; float s=0.0; for(uint b=0;b<batchSize;b++){s+=applyBackwardBias(go[b*outFeatures+j],saved[b*outFeatures+j],activationType);} gb[j]=s; }";

    // IoU Loss Backward GLSL kernels (4 buffers: gradOutput, predicted, target, gradPredicted)
    private const string IoUBackwardLayout = @"
layout(set = 0, binding = 0) readonly buffer GradOutput { float goB[]; };
layout(set = 0, binding = 1) readonly buffer Pred { float pred[]; };
layout(set = 0, binding = 2) readonly buffer Target { float targ[]; };
layout(set = 0, binding = 3) writeonly buffer GradPred { float gp[]; };
layout(push_constant) uniform Params { uint numBoxes; };
";

    private const string IoUGradHelper = @"
void computeIoUGrad(float px1,float py1,float px2,float py2,float tx1,float ty1,float tx2,float ty2,float goVal,out float g0,out float g1,out float g2,out float g3) {
    float iw=max(0.0,min(px2,tx2)-max(px1,tx1)),ih=max(0.0,min(py2,ty2)-max(py1,ty1));
    float iA=iw*ih,pw=max(0.0,px2-px1),ph=max(0.0,py2-py1),pA=pw*ph,tA=max(0.0,tx2-tx1)*max(0.0,ty2-ty1),uA=pA+tA-iA+1e-7;
    float hi=(iw>0.0&&ih>0.0)?1.0:0.0;
    float dI0=hi*(-(px1>tx1?1.0:0.0))*ih, dI1=hi*(-(py1>ty1?1.0:0.0))*iw;
    float dI2=hi*((px2<tx2?1.0:0.0))*ih, dI3=hi*((py2<ty2?1.0:0.0))*iw;
    float dU0=-ph-dI0, dU1=-pw-dI1, dU2=ph-dI2, dU3=pw-dI3;
    float uSq=uA*uA;
    g0=goVal*(-(dI0*uA-iA*dU0)/uSq); g1=goVal*(-(dI1*uA-iA*dU1)/uSq);
    g2=goVal*(-(dI2*uA-iA*dU2)/uSq); g3=goVal*(-(dI3*uA-iA*dU3)/uSq);
}
";

    public static string IoULossBackward => Header + IoUBackwardLayout + IoUGradHelper + @"void main() { uint i=gl_GlobalInvocationID.x; if(i>=numBoxes)return; uint o=i*4; float g0,g1,g2,g3; computeIoUGrad(pred[o],pred[o+1],pred[o+2],pred[o+3],targ[o],targ[o+1],targ[o+2],targ[o+3],goB[i],g0,g1,g2,g3); gp[o]=g0;gp[o+1]=g1;gp[o+2]=g2;gp[o+3]=g3; }";

    public static string GIoULossBackward => Header + IoUBackwardLayout + IoUGradHelper + @"void main() { uint i=gl_GlobalInvocationID.x; if(i>=numBoxes)return; uint o=i*4;
float px1=pred[o],py1=pred[o+1],px2=pred[o+2],py2=pred[o+3]; float tx1=targ[o],ty1=targ[o+1],tx2=targ[o+2],ty2=targ[o+3];
float ig0,ig1,ig2,ig3; computeIoUGrad(px1,py1,px2,py2,tx1,ty1,tx2,ty2,1.0,ig0,ig1,ig2,ig3);
float encW=max(px2,tx2)-min(px1,tx1),encH=max(py2,ty2)-min(py1,ty1),encA=encW*encH+1e-7;
float iw=max(0.0,min(px2,tx2)-max(px1,tx1)),ih=max(0.0,min(py2,ty2)-max(py1,ty1)),iA=iw*ih;
float pw=px2-px1,ph=py2-py1,uA=pw*ph+(tx2-tx1)*(ty2-ty1)-iA+1e-7;
float hi=(iw>0.0&&ih>0.0)?1.0:0.0;
float dI0=hi*(-(px1>tx1?1.0:0.0))*ih,dI1=hi*(-(py1>ty1?1.0:0.0))*iw,dI2=hi*((px2<tx2?1.0:0.0))*ih,dI3=hi*((py2<ty2?1.0:0.0))*iw;
float dU0=-ph-dI0,dU1=-pw-dI1,dU2=ph-dI2,dU3=pw-dI3;
float dEncA0=-(px1<tx1?1.0:0.0)*encH,dEncA1=-(py1<ty1?1.0:0.0)*encW,dEncA2=(px2>tx2?1.0:0.0)*encH,dEncA3=(py2>ty2?1.0:0.0)*encW;
float encASq=encA*encA;
float dP0=-(dU0*encA-uA*dEncA0)/encASq,dP1=-(dU1*encA-uA*dEncA1)/encASq,dP2=-(dU2*encA-uA*dEncA2)/encASq,dP3=-(dU3*encA-uA*dEncA3)/encASq;
gp[o]=goB[i]*(-ig0+dP0);gp[o+1]=goB[i]*(-ig1+dP1);gp[o+2]=goB[i]*(-ig2+dP2);gp[o+3]=goB[i]*(-ig3+dP3); }";

    public static string DIoULossBackward => Header + IoUBackwardLayout + IoUGradHelper + @"void main() { uint i=gl_GlobalInvocationID.x; if(i>=numBoxes)return; uint o=i*4;
float px1=pred[o],py1=pred[o+1],px2=pred[o+2],py2=pred[o+3]; float tx1=targ[o],ty1=targ[o+1],tx2=targ[o+2],ty2=targ[o+3];
float ig0,ig1,ig2,ig3; computeIoUGrad(px1,py1,px2,py2,tx1,ty1,tx2,ty2,1.0,ig0,ig1,ig2,ig3);
float dx=0.5*(px1+px2)-0.5*(tx1+tx2),dy=0.5*(py1+py2)-0.5*(ty1+ty2);
float rhoSq=dx*dx+dy*dy; float dRho0=dx,dRho1=dy,dRho2=dx,dRho3=dy;
float encDx=max(px2,tx2)-min(px1,tx1),encDy=max(py2,ty2)-min(py1,ty1);
float cSq=encDx*encDx+encDy*encDy+1e-7,cSqSq=cSq*cSq;
float dCSq0=2.0*encDx*(-(px1<tx1?1.0:0.0)),dCSq1=2.0*encDy*(-(py1<ty1?1.0:0.0)),dCSq2=2.0*encDx*((px2>tx2?1.0:0.0)),dCSq3=2.0*encDy*((py2>ty2?1.0:0.0));
float dp0=(dRho0*cSq-rhoSq*dCSq0)/cSqSq,dp1=(dRho1*cSq-rhoSq*dCSq1)/cSqSq,dp2=(dRho2*cSq-rhoSq*dCSq2)/cSqSq,dp3=(dRho3*cSq-rhoSq*dCSq3)/cSqSq;
gp[o]=goB[i]*(-ig0+dp0);gp[o+1]=goB[i]*(-ig1+dp1);gp[o+2]=goB[i]*(-ig2+dp2);gp[o+3]=goB[i]*(-ig3+dp3); }";

    public static string CIoULossBackward => Header + IoUBackwardLayout + IoUGradHelper + @"void main() { uint i=gl_GlobalInvocationID.x; if(i>=numBoxes)return; uint o=i*4;
float px1=pred[o],py1=pred[o+1],px2=pred[o+2],py2=pred[o+3]; float tx1=targ[o],ty1=targ[o+1],tx2=targ[o+2],ty2=targ[o+3];
float ig0,ig1,ig2,ig3; computeIoUGrad(px1,py1,px2,py2,tx1,ty1,tx2,ty2,1.0,ig0,ig1,ig2,ig3);
float dx=0.5*(px1+px2)-0.5*(tx1+tx2),dy=0.5*(py1+py2)-0.5*(ty1+ty2);
float rhoSq=dx*dx+dy*dy; float dRho0=dx,dRho1=dy,dRho2=dx,dRho3=dy;
float encDx=max(px2,tx2)-min(px1,tx1),encDy=max(py2,ty2)-min(py1,ty1);
float cSq=encDx*encDx+encDy*encDy+1e-7,cSqSq=cSq*cSq;
float dCSq0=2.0*encDx*(-(px1<tx1?1.0:0.0)),dCSq1=2.0*encDy*(-(py1<ty1?1.0:0.0)),dCSq2=2.0*encDx*((px2>tx2?1.0:0.0)),dCSq3=2.0*encDy*((py2>ty2?1.0:0.0));
float pw=max(px2-px1,1e-7),ph=max(py2-py1,1e-7),tw=max(tx2-tx1,1e-7),th=max(ty2-ty1,1e-7);
float atanDiff=atan(tw/th)-atan(pw/ph); float fourOverPiSq=4.0/(3.14159265*3.14159265);
float v=fourOverPiSq*atanDiff*atanDiff;
float iw=max(0.0,min(px2,tx2)-max(px1,tx1)),ih=max(0.0,min(py2,ty2)-max(py1,ty1));
float iou=iw*ih/(pw*ph+(tx2-tx1)*(ty2-ty1)-iw*ih+1e-7);
float alpha=v/(1.0-iou+v+1e-7);
float rss=pw*pw+ph*ph,dAtanPw=ph/rss,dAtanPh=-pw/rss;
float dV0=2.0*fourOverPiSq*atanDiff*dAtanPw,dV1=2.0*fourOverPiSq*atanDiff*dAtanPh,dV2=2.0*fourOverPiSq*atanDiff*(-dAtanPw),dV3=2.0*fourOverPiSq*atanDiff*(-dAtanPh);
float dp0=(dRho0*cSq-rhoSq*dCSq0)/cSqSq,dp1=(dRho1*cSq-rhoSq*dCSq1)/cSqSq,dp2=(dRho2*cSq-rhoSq*dCSq2)/cSqSq,dp3=(dRho3*cSq-rhoSq*dCSq3)/cSqSq;
gp[o]=goB[i]*(-ig0+dp0+alpha*dV0);gp[o+1]=goB[i]*(-ig1+dp1+alpha*dV1);gp[o+2]=goB[i]*(-ig2+dp2+alpha*dV2);gp[o+3]=goB[i]*(-ig3+dp3+alpha*dV3); }";

    // =====================================================================
    // Complex Tensor Operations
    // =====================================================================

    public static string ComplexMultiply => Header + ThreeBufferLayout + @"
layout(push_constant) uniform Params { uint numPairs; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= numPairs) return;
    uint o = i * 2;
    float aRe=a[o], aIm=a[o+1], bRe=bdata[o], bIm=bdata[o+1];
    c[o]   = aRe*bRe - aIm*bIm;
    c[o+1] = aRe*bIm + aIm*bRe;
}";

    public static string ComplexConjugate => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint numPairs; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= numPairs) return;
    uint o = i * 2;
    b[o]   = a[o];
    b[o+1] = -a[o+1];
}";

    public static string ComplexMagnitude => Header + TwoBufferLayout + @"
layout(push_constant) uniform Params { uint numPairs; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= numPairs) return;
    uint o = i * 2;
    b[i] = sqrt(a[o]*a[o] + a[o+1]*a[o+1]);
}";

    public static string SplitComplexTopK => Header + @"
layout(set = 0, binding = 0) readonly buffer InReal { float inReal[]; };
layout(set = 0, binding = 1) readonly buffer InImag { float inImag[]; };
layout(set = 0, binding = 2) writeonly buffer OutReal { float outReal[]; };
layout(set = 0, binding = 3) writeonly buffer OutImag { float outImag[]; };
layout(push_constant) uniform Params { uint n; uint k; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= n) return;
    float re = inReal[idx], im = inImag[idx];
    precise float magSq = re * re + im * im;
    bool magIsNan = (floatBitsToUint(magSq) & 0x7fffffffu) > 0x7f800000u;
    uint rank = 0u;
    for (uint other = 0u; other < n; ++other) {
        float otherRe = inReal[other], otherIm = inImag[other];
        precise float otherMagSq = otherRe * otherRe + otherIm * otherIm;
        bool otherIsNan = (floatBitsToUint(otherMagSq) & 0x7fffffffu) > 0x7f800000u;
        bool otherBefore = magIsNan
            ? (!otherIsNan || (otherIsNan && other < idx))
            : (!otherIsNan && (otherMagSq > magSq || (otherMagSq == magSq && other < idx)));
        rank += otherBefore ? 1u : 0u;
    }
    bool keep = rank < k;
    outReal[idx] = keep ? re : 0.0;
    outImag[idx] = keep ? im : 0.0;
}";
}
