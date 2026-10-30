// Copyright (c) AiDotNet. All rights reserved.
// Correctness-first resident Vulkan kernels for operations that do not yet have tuned SPIR-V.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

internal static class VulkanResidentKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    public static string RandomGenerate => Header + @"
layout(set=0,binding=0) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint count; uint seedLow; uint seedHigh; uint mode; uint minimumBits; uint maximumBits; uint meanBits; uint stdDevBits; };
uint mulHigh(uint a,uint b){uint al=a&65535u,ah=a>>16u,bl=b&65535u,bh=b>>16u;uint ll=al*bl,lh=al*bh,hl=ah*bl,hh=ah*bh,mid=(ll>>16u)+(lh&65535u)+(hl&65535u);return hh+(lh>>16u)+(hl>>16u)+(mid>>16u);}
uvec4 philoxRound(uint c0,uint c1,uint k0,uint k1){const uint m0=0xd2511f53u,m1=0xcd9e8d57u;uint lo0=c0*m0,hi0=mulHigh(c0,m0),lo1=c1*m1,hi1=mulHigh(c1,m1);return uvec4(hi1^k0,lo1,hi0^k1,lo0);}
uvec4 philox(uvec4 counter,uvec2 key){uvec4 c=counter;uvec2 k=key;for(uint i=0u;i<10u;++i){c=philoxRound(c.x,c.z,k.x,k.y);k.x+=0x9e3779b9u;k.y+=0xbb67ae85u;}return c;}
float uniform01(uint value){return float(value>>8u)*(1.0/16777216.0);}
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=count)return;uvec2 key=uvec2(seedLow,seedHigh);
    if(mode==0u){uint seed=seedLow^seedHigh;uint state=idx*747796405u+seed+2891336453u;uint word=((state>>((state>>28u)+4u))^state)*277803737u;uint sampleValue=(word>>22u)^word;float u=float(sampleValue>>8u)*(1.0/16777216.0);float minimum=uintBitsToFloat(minimumBits),maximum=uintBitsToFloat(maximumBits);outputData[idx]=fma(u,maximum-minimum,minimum);return;}
    uint pairIndex=idx/2u,pairBlock=pairIndex/2u,pairLane=pairIndex%2u;uvec4 randomPair=philox(uvec4(pairBlock,1u,0u,0u),key);float u1=max(uniform01(randomPair[pairLane*2u]),1.0e-10),u2=uniform01(randomPair[pairLane*2u+1u]);float magnitude=uintBitsToFloat(stdDevBits)*sqrt(-2.0*log(u1)),angle=6.283185307*u2,mean=uintBitsToFloat(meanBits);outputData[idx]=mean+magnitude*((idx&1u)==0u?cos(angle):sin(angle));
}";

    public static string StatelessDropoutMask => Header + @"
layout(set=0,binding=0) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint count; uint threshold; uint scaleBits; uint seed; };
void main(){
    uint idx=gl_GlobalInvocationID.x;if(idx>=count)return;
    uint state=idx*747796405u+seed+2891336453u;
    uint word=((state>>((state>>28u)+4u))^state)*277803737u;
    uint sampleValue=(word>>22u)^word;
    outputData[idx]=sampleValue<threshold?0.0:uintBitsToFloat(scaleBits);
}";

    public static string Enforce2x4 => Header + @"
layout(set=0,binding=0) readonly buffer Dense { float dense[]; };
layout(set=0,binding=1) writeonly buffer Values { float values[]; };
layout(set=0,binding=2) writeonly buffer Indices { uint indices[]; };
layout(push_constant) uniform Params { uint M; uint K; uint groups; };
void main(){
    uint word=gl_GlobalInvocationID.x,first=word*4u;if(first>=groups)return;uint packed=0u;
    for(uint slot=0u;slot<4u;++slot){uint group=first+slot;if(group>=groups)break;uint row=group/(K/4u),block=group%(K/4u),base=(row*K)+(block*4u);uint i0=0u,i1=1u;
        if(abs(dense[base+i1])>abs(dense[base+i0])){uint t=i0;i0=i1;i1=t;}
        for(uint j=2u;j<4u;++j){float a=abs(dense[base+j]);if(a>abs(dense[base+i0])){i1=i0;i0=j;}else if(a>abs(dense[base+i1]))i1=j;}
        values[group*2u]=dense[base+i0];values[group*2u+1u]=dense[base+i1];packed|=(i0|(i1<<2u))<<(slot*8u);
    }indices[word]=packed;
}";

    public static string Decompress2x4 => Header + @"
layout(set=0,binding=0) readonly buffer Values { float values[]; };
layout(set=0,binding=1) readonly buffer Indices { uint indices[]; };
layout(set=0,binding=2) writeonly buffer Dense { float dense[]; };
layout(push_constant) uniform Params { uint M; uint K; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=M*K)return;uint group=idx/4u,local=idx%4u,byteValue=(indices[group/4u]>>((group%4u)*8u))&255u;uint i0=byteValue&3u,i1=(byteValue>>2u)&3u;dense[idx]=local==i0?values[group*2u]:(local==i1?values[group*2u+1u]:0.0);}";

    public static string ScatterAddEdges => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) readonly buffer Sources { float sources[]; };
layout(set=0,binding=2) readonly buffer Targets { float targets[]; };
layout(set=0,binding=3) readonly buffer EdgeValues { float edgeValues[]; };
layout(set=0,binding=4) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint nodes; uint edges; uint features; uint weighted; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=nodes*features)return;uint node=idx/features,feature=idx%features;float sum=inputData[idx];for(uint e=0u;e<edges;++e){uint target=floatBitsToUint(targets[e]);if(target==node){uint source=floatBitsToUint(sources[e]);if(source<nodes)sum+=(weighted!=0u?edgeValues[e]:1.0)*inputData[source*features+feature];}}outputData[idx]=sum;}";

    public static string MaxPool2DBackward => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Indices { int indices[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
layout(push_constant) uniform Params { uint batch; uint channels; uint inHeight; uint inWidth; uint outHeight; uint outWidth; };
void main(){uint idx=gl_GlobalInvocationID.x,inputSpatial=inHeight*inWidth,total=batch*channels*inputSpatial;if(idx>=total)return;uint bc=idx/inputSpatial,local=idx%inputSpatial,outputOffset=bc*outHeight*outWidth;float sum=0.0;for(uint oh=0u;oh<outHeight;++oh)for(uint ow=0u;ow<outWidth;++ow){uint outIndex=outputOffset+oh*outWidth+ow;if(indices[outIndex]==int(local))sum+=gradOutput[outIndex];}gradInput[idx]=sum;}";

    public static string MaxPool3DBackward => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Indices { int indices[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
layout(push_constant) uniform Params { uint batch; uint channels; uint inDepth; uint inHeight; uint inWidth; uint outDepth; uint outHeight; uint outWidth; };
void main(){uint idx=gl_GlobalInvocationID.x,inputSpatial=inDepth*inHeight*inWidth,total=batch*channels*inputSpatial;if(idx>=total)return;uint bc=idx/inputSpatial,local=idx%inputSpatial,outputSpatial=outDepth*outHeight*outWidth,outputOffset=bc*outputSpatial;float sum=0.0;for(uint od=0u;od<outDepth;++od)for(uint oh=0u;oh<outHeight;++oh)for(uint ow=0u;ow<outWidth;++ow){uint outIndex=outputOffset+(od*outHeight+oh)*outWidth+ow;if(indices[outIndex]==int(local))sum+=gradOutput[outIndex];}gradInput[idx]=sum;}";

    public static string CsrSegmented => Header + @"
layout(set=0,binding=0) readonly buffer Columns { float columns[]; };
layout(set=0,binding=1) readonly buffer Rows { float rows[]; };
layout(set=0,binding=2) readonly buffer Input { float inputData[]; };
layout(set=0,binding=3) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint M; uint K; uint N; uint op; float epsilon; };
void main(){uint outIndex=gl_GlobalInvocationID.x;if(outIndex>=M*N)return;uint row=outIndex/N,feature=outIndex%N,start=floatBitsToUint(rows[row]),end=floatBitsToUint(rows[row+1u]);if(start>=end){outputData[outIndex]=0.0;return;}if(op<2u){float value=op==0u?-3.402823466e38:3.402823466e38;for(uint p=start;p<end;++p){uint col=floatBitsToUint(columns[p]);if(col<K)value=op==0u?max(value,inputData[col*N+feature]):min(value,inputData[col*N+feature]);}outputData[outIndex]=value;return;}float mean=0.0;for(uint p=start;p<end;++p){uint col=floatBitsToUint(columns[p]);if(col<K)mean+=inputData[col*N+feature];}mean/=float(end-start);float variance=0.0;for(uint p=start;p<end;++p){uint col=floatBitsToUint(columns[p]);if(col<K){float d=inputData[col*N+feature]-mean;variance+=d*d;}}outputData[outIndex]=sqrt(variance/float(end-start)+epsilon);}";

    public static string RbfForward => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) readonly buffer Centers { float centers[]; };
layout(set=0,binding=2) readonly buffer Epsilons { float epsilons[]; };
layout(set=0,binding=3) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint batch; uint centerCount; uint inputDim; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*centerCount)return;uint b=idx/centerCount,c=idx%centerCount;float distance=0.0;for(uint d=0u;d<inputDim;++d){float x=inputData[b*inputDim+d]-centers[c*inputDim+d];distance+=x*x;}outputData[idx]=exp(-epsilons[c]*distance);}";

    public static string StdpUpdate => Header + @"
layout(set=0,binding=0) buffer Weights { float weights[]; };
layout(set=0,binding=1) readonly buffer PreTrace { float preTrace[]; };
layout(set=0,binding=2) readonly buffer PostTrace { float postTrace[]; };
layout(set=0,binding=3) readonly buffer PreSpike { float preSpike[]; };
layout(set=0,binding=4) readonly buffer PostSpike { float postSpike[]; };
layout(push_constant) uniform Params { uint numPre; uint numPost; float ltp; float ltd; float minWeight; float maxWeight; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=numPre*numPost)return;uint i=idx/numPost,j=idx%numPost;float value=weights[idx];if(postSpike[j]>0.5)value+=ltp*preTrace[i];if(preSpike[i]>0.5)value-=ltd*postTrace[j];weights[idx]=clamp(value,minWeight,maxWeight);}";

    public static string UpdateTraces => Header + @"
layout(set=0,binding=0) buffer Traces { float traces[]; };
layout(set=0,binding=1) readonly buffer Spikes { float spikes[]; };
layout(push_constant) uniform Params { uint size; float decay; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx<size)traces[idx]=decay*traces[idx]+(spikes[idx]>0.5?1.0:0.0);}";

    public static string HyperbolicBackwardInput => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Weights { float weights[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
layout(push_constant) uniform Params { uint batch; uint inputFeatures; uint outputFeatures; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*inputFeatures)return;uint b=idx/inputFeatures,i=idx%inputFeatures;float sum=0.0;for(uint o=0u;o<outputFeatures;++o)sum+=gradOutput[b*outputFeatures+o]*weights[o*inputFeatures+i];gradInput[idx]=sum;}";

    public static string HyperbolicBackwardWeights => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer Input { float inputData[]; };
layout(set=0,binding=2) writeonly buffer GradWeights { float gradWeights[]; };
layout(push_constant) uniform Params { uint batch; uint inputFeatures; uint outputFeatures; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=outputFeatures*inputFeatures)return;uint o=idx/inputFeatures,i=idx%inputFeatures;float sum=0.0;for(uint b=0u;b<batch;++b)sum+=gradOutput[b*outputFeatures+o]*inputData[b*inputFeatures+i];gradWeights[idx]=sum;}";

    public static string HyperbolicBackwardBiases => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) writeonly buffer GradBiases { float gradBiases[]; };
layout(push_constant) uniform Params { uint batch; uint outputFeatures; };
void main(){uint o=gl_GlobalInvocationID.x;if(o>=outputFeatures)return;float sum=0.0;for(uint b=0u;b<batch;++b)sum+=gradOutput[b*outputFeatures+o];gradBiases[o]=sum;}";

    public static string QuantumMeasurement => Header + @"
layout(set=0,binding=0) readonly buffer Real { float realData[]; };
layout(set=0,binding=1) readonly buffer Imag { float imagData[]; };
layout(set=0,binding=2) writeonly buffer Probabilities { float probabilities[]; };
layout(push_constant) uniform Params { uint size; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx<size)probabilities[idx]=realData[idx]*realData[idx]+imagData[idx]*imagData[idx];}";

    public static string NormalizeProbabilities => Header + @"
layout(set=0,binding=0) buffer Probabilities { float probabilities[]; };
layout(push_constant) uniform Params { uint batch; uint stateSize; };
void main(){uint b=gl_GlobalInvocationID.x;if(b>=batch)return;uint offset=b*stateSize;float sum=0.0;for(uint s=0u;s<stateSize;++s)sum+=probabilities[offset+s];if(sum>0.0)for(uint s=0u;s<stateSize;++s)probabilities[offset+s]/=sum;}";

    public static string ComplexMatVec => Header + @"
layout(set=0,binding=0) readonly buffer MatrixReal { float matrixReal[]; };
layout(set=0,binding=1) readonly buffer MatrixImag { float matrixImag[]; };
layout(set=0,binding=2) readonly buffer VectorReal { float vectorReal[]; };
layout(set=0,binding=3) readonly buffer VectorImag { float vectorImag[]; };
layout(set=0,binding=4) writeonly buffer OutputReal { float outputReal[]; };
layout(set=0,binding=5) writeonly buffer OutputImag { float outputImag[]; };
layout(push_constant) uniform Params { uint batch; uint dim; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*dim)return;uint b=idx/dim,row=idx%dim,matrixOffset=b*dim*dim,vectorOffset=b*dim;float realSum=0.0,imagSum=0.0;for(uint j=0u;j<dim;++j){uint m=matrixOffset+row*dim+j;realSum+=matrixReal[m]*vectorReal[vectorOffset+j]-matrixImag[m]*vectorImag[vectorOffset+j];imagSum+=matrixReal[m]*vectorImag[vectorOffset+j]+matrixImag[m]*vectorReal[vectorOffset+j];}outputReal[idx]=realSum;outputImag[idx]=imagSum;}";

    public static string QuantumRotation => Header + @"
layout(set=0,binding=0) readonly buffer StateReal { float stateReal[]; };
layout(set=0,binding=1) readonly buffer StateImag { float stateImag[]; };
layout(set=0,binding=2) readonly buffer Angles { float angles[]; };
layout(set=0,binding=3) writeonly buffer OutputReal { float outputReal[]; };
layout(set=0,binding=4) writeonly buffer OutputImag { float outputImag[]; };
layout(push_constant) uniform Params { uint batch; uint stateSize; uint angleCount; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*stateSize)return;uint b=idx/stateSize;float angle=angles[b%angleCount],c=cos(angle),s=sin(angle);outputReal[idx]=c*stateReal[idx]-s*stateImag[idx];outputImag[idx]=s*stateReal[idx]+c*stateImag[idx];}";

    public static string MeasurementForward => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint batch; uint stateSize; };
void main(){uint b=gl_GlobalInvocationID.x;if(b>=batch)return;uint offset=b*stateSize;float sum=0.0;for(uint s=0u;s<stateSize;++s){float value=inputData[offset+s]*inputData[offset+s];outputData[offset+s]=value;sum+=value;}if(sum>0.0)for(uint s=0u;s<stateSize;++s)outputData[offset+s]/=sum;}";

    public static string SplitDft => Header + @"
layout(set=0,binding=0) readonly buffer InputReal { float inputReal[]; };
layout(set=0,binding=1) readonly buffer InputImag { float inputImag[]; };
layout(set=0,binding=2) writeonly buffer OutputReal { float outputReal[]; };
layout(set=0,binding=3) writeonly buffer OutputImag { float outputImag[]; };
layout(push_constant) uniform Params { uint batch; uint n; uint inverse; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*n)return;uint b=idx/n,k=idx%n,offset=b*n;float realSum=0.0,imagSum=0.0,sign=inverse!=0u?1.0:-1.0;for(uint j=0u;j<n;++j){float angle=sign*6.283185307179586*float(k*j)/float(n),c=cos(angle),s=sin(angle);realSum+=inputReal[offset+j]*c-inputImag[offset+j]*s;imagSum+=inputReal[offset+j]*s+inputImag[offset+j]*c;}if(inverse!=0u){realSum/=float(n);imagSum/=float(n);}outputReal[idx]=realSum;outputImag[idx]=imagSum;}";

    public static string SplitDft2D => Header + @"
layout(set=0,binding=0) readonly buffer InputReal { float inputReal[]; };
layout(set=0,binding=1) readonly buffer InputImag { float inputImag[]; };
layout(set=0,binding=2) writeonly buffer OutputReal { float outputReal[]; };
layout(set=0,binding=3) writeonly buffer OutputImag { float outputImag[]; };
layout(push_constant) uniform Params { uint batch; uint height; uint width; uint inverse; };
void main(){uint idx=gl_GlobalInvocationID.x,slice=height*width;if(idx>=batch*slice)return;uint b=idx/slice,local=idx%slice,ky=local/width,kx=local%width,offset=b*slice;float realSum=0.0,imagSum=0.0,sign=inverse!=0u?1.0:-1.0;for(uint y=0u;y<height;++y)for(uint x=0u;x<width;++x){float angle=sign*6.283185307179586*(float(ky*y)/float(height)+float(kx*x)/float(width)),c=cos(angle),s=sin(angle);uint source=offset+y*width+x;realSum+=inputReal[source]*c-inputImag[source]*s;imagSum+=inputReal[source]*s+inputImag[source]*c;}if(inverse!=0u){realSum/=float(slice);imagSum/=float(slice);}outputReal[idx]=realSum;outputImag[idx]=imagSum;}";

    public static string PolarToComplex => Header + @"
layout(set=0,binding=0) readonly buffer Magnitude { float magnitude[]; };
layout(set=0,binding=1) readonly buffer Phase { float phase[]; };
layout(set=0,binding=2) writeonly buffer Real { float realData[]; };
layout(set=0,binding=3) writeonly buffer Imag { float imagData[]; };
layout(push_constant) uniform Params { uint size; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx<size){realData[idx]=magnitude[idx]*cos(phase[idx]);imagData[idx]=magnitude[idx]*sin(phase[idx]);}}";

    public static string MelFilterbank => Header + @"
layout(set=0,binding=0) readonly buffer Power { float power[]; };
layout(set=0,binding=1) readonly buffer Filterbank { float filterbank[]; };
layout(set=0,binding=2) writeonly buffer Mel { float mel[]; };
layout(push_constant) uniform Params { uint frames; uint frequencies; uint bands; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=frames*bands)return;uint frame=idx/bands,band=idx%bands;float sum=0.0;for(uint k=0u;k<frequencies;++k)sum+=power[frame*frequencies+k]*filterbank[band*frequencies+k];mel[idx]=sum;}";

    // Recurrent sequences run as one dispatch per timestep (plus whole-sequence projections and gradient reductions), with
    // an invocation per (batch row, hidden unit). They used to run each batch row's whole sequence in one invocation and
    // the whole BPTT in ONE invocation: a 64-row, 32-step LSTM backward took 5.5 s, the driver's watchdog reset the
    // device, and training failed with VK_ERROR_DEVICE_LOST at the next submission.
    // Shared layout: LSTM input/output [B, T, *] (batch-major), GRU input/output [T, B, *] (time-major); allH/allC
    // [T + 1, B, H]; push constants seqLen, batch, inputSize, hiddenSize, then the step t where a kernel takes one.

    // LSTM gate pre-activations from the input and both biases, for every step at once: gates [T, B, 4H].
    public static string LstmSequenceInputProjection => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) readonly buffer WeightsIh { float weightsIh[]; };
layout(set=0,binding=2) readonly buffer BiasIh { float biasIh[]; };
layout(set=0,binding=3) readonly buffer BiasHh { float biasHh[]; };
layout(set=0,binding=4) writeonly buffer Gates { float gates[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
void main(){uint idx=gl_GlobalInvocationID.x,gateSize=4u*hiddenSize;if(idx>=seqLen*batch*gateSize)return;uint g=idx%gateSize,row=idx/gateSize,t=row/batch,b=row%batch,inputOffset=(b*seqLen+t)*inputSize;
    float sum=biasIh[g]+biasHh[g];for(uint k=0u;k<inputSize;++k)sum+=inputData[inputOffset+k]*weightsIh[g*inputSize+k];gates[idx]=sum;}";

    // One LSTM step: adds the recurrent term to unit i's four gate pre-activations, then writes c and h for step t + 1.
    public static string LstmSequenceStep => Header + @"
layout(set=0,binding=0) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=1) buffer Gates { float gates[]; };
layout(set=0,binding=2) buffer AllH { float allH[]; };
layout(set=0,binding=3) buffer AllC { float allC[]; };
layout(set=0,binding=4) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; uint t; };
float sigmoid(float x){return 1.0/(1.0+exp(-x));}
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*hiddenSize)return;uint b=idx/hiddenSize,i=idx%hiddenSize,H=hiddenSize,cellSize=batch*H;
    uint hPrev=t*cellSize+b*H,gateOffset=(t*batch+b)*4u*H;float pi=gates[gateOffset+i],pf=gates[gateOffset+H+i],pg=gates[gateOffset+2u*H+i],po=gates[gateOffset+3u*H+i];
    for(uint k=0u;k<H;++k){float hk=allH[hPrev+k];pi+=hk*weightsHh[i*H+k];pf+=hk*weightsHh[(H+i)*H+k];pg+=hk*weightsHh[(2u*H+i)*H+k];po+=hk*weightsHh[(3u*H+i)*H+k];}
    gates[gateOffset+i]=pi;gates[gateOffset+H+i]=pf;gates[gateOffset+2u*H+i]=pg;gates[gateOffset+3u*H+i]=po;
    float ct=sigmoid(pf)*allC[hPrev+i]+sigmoid(pi)*tanh(pg),ht=sigmoid(po)*tanh(ct);allC[(t+1u)*cellSize+b*H+i]=ct;allH[(t+1u)*cellSize+b*H+i]=ht;outputData[(b*seqLen+t)*H+i]=ht;}";

    // LSTM backward, step t: gate gradients into dGates [T, B, 4H] from dh = dy + dhCarry and the cell carry dcCarry,
    // then dcCarry = dc * f. dhCarry is rewritten by LstmSequenceBackwardHidden.
    public static string LstmSequenceBackwardCell => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer AllC { float allC[]; };
layout(set=0,binding=2) readonly buffer Gates { float gates[]; };
layout(set=0,binding=3) writeonly buffer DGates { float dGates[]; };
layout(set=0,binding=4) readonly buffer DhCarry { float dhCarry[]; };
layout(set=0,binding=5) buffer DcCarry { float dcCarry[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; uint t; };
float sigmoid(float x){return 1.0/(1.0+exp(-x));}
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*hiddenSize)return;uint b=idx/hiddenSize,i=idx%hiddenSize,H=hiddenSize,cellSize=batch*H,state=b*H+i;
    uint gateOffset=(t*batch+b)*4u*H;float ig=sigmoid(gates[gateOffset+i]),fg=sigmoid(gates[gateOffset+H+i]),gg=tanh(gates[gateOffset+2u*H+i]),og=sigmoid(gates[gateOffset+3u*H+i]);
    float tanhCt=tanh(allC[(t+1u)*cellSize+state]),dh=gradOutput[(b*seqLen+t)*H+i]+dhCarry[state],dc=dh*og*(1.0-tanhCt*tanhCt)+dcCarry[state];
    dGates[gateOffset+i]=dc*gg*ig*(1.0-ig);dGates[gateOffset+H+i]=dc*allC[t*cellSize+state]*fg*(1.0-fg);dGates[gateOffset+2u*H+i]=dc*ig*(1.0-gg*gg);dGates[gateOffset+3u*H+i]=dh*tanhCt*og*(1.0-og);
    dcCarry[state]=dc*fg;}";

    // LSTM backward, step t: dhCarry[b, k] = sum over the 4H gate rows of dGates * W_hh, the gradient reaching h(t - 1).
    public static string LstmSequenceBackwardHidden => Header + @"
layout(set=0,binding=0) readonly buffer DGates { float dGates[]; };
layout(set=0,binding=1) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=2) writeonly buffer DhCarry { float dhCarry[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; uint t; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*hiddenSize)return;uint b=idx/hiddenSize,k=idx%hiddenSize,gateSize=4u*hiddenSize,gateOffset=(t*batch+b)*gateSize;
    float sum=0.0;for(uint g=0u;g<gateSize;++g)sum+=dGates[gateOffset+g]*weightsHh[g*hiddenSize+k];dhCarry[idx]=sum;}";

    // LSTM input gradient for every step at once: gradInput [B, T, In].
    public static string LstmSequenceInputGradient => Header + @"
layout(set=0,binding=0) readonly buffer DGates { float dGates[]; };
layout(set=0,binding=1) readonly buffer WeightsIh { float weightsIh[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*seqLen*inputSize)return;uint k=idx%inputSize,row=idx/inputSize,b=row/seqLen,t=row%seqLen,gateSize=4u*hiddenSize,gateOffset=(t*batch+b)*gateSize;
    float sum=0.0;for(uint g=0u;g<gateSize;++g)sum+=dGates[gateOffset+g]*weightsIh[g*inputSize+k];gradInput[idx]=sum;}";

    // LSTM weight gradients over every step and batch row: gradWeightsIh [4H, In] from the input, gradWeightsHh [4H, H]
    // from h(t - 1). mode 0 = input-hidden, 1 = hidden-hidden.
    public static string LstmSequenceWeightGradient => Header + @"
layout(set=0,binding=0) readonly buffer DGates { float dGates[]; };
layout(set=0,binding=1) readonly buffer Source { float source[]; };
layout(set=0,binding=2) writeonly buffer GradWeights { float gradWeights[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; uint mode; };
void main(){uint width=mode==0u?inputSize:hiddenSize,gateSize=4u*hiddenSize,idx=gl_GlobalInvocationID.x;if(idx>=gateSize*width)return;uint g=idx/width,k=idx%width;
    float sum=0.0;for(uint t=0u;t<seqLen;++t)for(uint b=0u;b<batch;++b){uint src=mode==0u?(b*seqLen+t)*inputSize+k:(t*batch+b)*hiddenSize+k;sum+=dGates[(t*batch+b)*gateSize+g]*source[src];}gradWeights[idx]=sum;}";

    // LSTM bias gradient (the same for both biases, which the forward sums): the gate gradients summed over steps and rows.
    public static string LstmSequenceBiasGradient => Header + @"
layout(set=0,binding=0) readonly buffer DGates { float dGates[]; };
layout(set=0,binding=1) writeonly buffer GradBiasIh { float gradBiasIh[]; };
layout(set=0,binding=2) writeonly buffer GradBiasHh { float gradBiasHh[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
void main(){uint gateSize=4u*hiddenSize,g=gl_GlobalInvocationID.x;if(g>=gateSize)return;float sum=0.0;for(uint row=0u;row<seqLen*batch;++row)sum+=dGates[row*gateSize+g];gradBiasIh[g]=sum;gradBiasHh[g]=sum;}";

    // GRU, PyTorch's formulation (the reset gate scales W_hn h + b_hn), as the other backends compute it:
    //   r = sigmoid(W_ir x + b_ir + W_hr h + b_hr), z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    //   n = tanh(W_in x + b_in + r * (W_hn h + b_hn)), h' = (1 - z) * n + z * h
    // cacheGates [T, B, 3H] holds r, z and W_hn h + b_hn. The projection first stores W_i* x + b_i* there.
    public static string GruSequenceInputProjection => Header + @"
layout(set=0,binding=0) readonly buffer Input { float inputData[]; };
layout(set=0,binding=1) readonly buffer WeightsIh { float weightsIh[]; };
layout(set=0,binding=2) readonly buffer BiasIh { float biasIh[]; };
layout(set=0,binding=3) writeonly buffer Gates { float gates[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
void main(){uint idx=gl_GlobalInvocationID.x,gateSize=3u*hiddenSize;if(idx>=seqLen*batch*gateSize)return;uint g=idx%gateSize,row=idx/gateSize;
    float sum=biasIh[g];for(uint k=0u;k<inputSize;++k)sum+=inputData[row*inputSize+k]*weightsIh[g*inputSize+k];gates[idx]=sum;}";

    // One GRU step for unit j of batch row b; replaces the projected input terms in cacheGates with r, z, W_hn h + b_hn.
    public static string GruSequenceStep => Header + @"
layout(set=0,binding=0) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=1) readonly buffer BiasHh { float biasHh[]; };
layout(set=0,binding=2) buffer Gates { float gates[]; };
layout(set=0,binding=3) buffer AllH { float allH[]; };
layout(set=0,binding=4) writeonly buffer Output { float outputData[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; uint t; };
float sigmoid(float x){return 1.0/(1.0+exp(-x));}
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*hiddenSize)return;uint b=idx/hiddenSize,j=idx%hiddenSize,H=hiddenSize,cellSize=batch*H,row=t*batch+b,gateOffset=row*3u*H,hPrev=t*cellSize+b*H;
    float hr=biasHh[j],hz=biasHh[H+j],hn=biasHh[2u*H+j];for(uint k=0u;k<H;++k){float hk=allH[hPrev+k];hr+=weightsHh[j*H+k]*hk;hz+=weightsHh[(H+j)*H+k]*hk;hn+=weightsHh[(2u*H+j)*H+k]*hk;}
    float r=sigmoid(gates[gateOffset+j]+hr),z=sigmoid(gates[gateOffset+H+j]+hz),n=tanh(gates[gateOffset+2u*H+j]+r*hn),hNew=(1.0-z)*n+z*allH[hPrev+j];
    gates[gateOffset+j]=r;gates[gateOffset+H+j]=z;gates[gateOffset+2u*H+j]=hn;allH[(t+1u)*cellSize+b*H+j]=hNew;outputData[row*H+j]=hNew;}";

    // GRU backward, step t: dGates [T, B, 4H] gets dr, dz, dn (pre-activation) and dn * r (the W_hn h + b_hn term's
    // gradient) from dh = dy + carry; carry becomes dh * z, the direct path. n is recovered through (1 - z) * n = h' - z h.
    public static string GruSequenceBackwardCell => Header + @"
layout(set=0,binding=0) readonly buffer GradOutput { float gradOutput[]; };
layout(set=0,binding=1) readonly buffer AllH { float allH[]; };
layout(set=0,binding=2) readonly buffer Gates { float gates[]; };
layout(set=0,binding=3) writeonly buffer DGates { float dGates[]; };
layout(set=0,binding=4) buffer Carry { float carry[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; uint t; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*hiddenSize)return;uint b=idx/hiddenSize,j=idx%hiddenSize,H=hiddenSize,cellSize=batch*H,row=t*batch+b,gateOffset=row*3u*H,dOffset=row*4u*H;
    float r=gates[gateOffset+j],z=gates[gateOffset+H+j],hn=gates[gateOffset+2u*H+j],hPrev=allH[t*cellSize+b*H+j],hCur=allH[(t+1u)*cellSize+b*H+j];
    float dh=gradOutput[row*H+j]+carry[idx],oneMinusZ=1.0-z,nScaled=hCur-z*hPrev,dn=oneMinusZ>1e-12?dh*(oneMinusZ-nScaled*nScaled/oneMinusZ):0.0;
    dGates[dOffset+j]=dn*hn*r*(1.0-r);dGates[dOffset+H+j]=dh*z*(hPrev-hCur);dGates[dOffset+2u*H+j]=dn;dGates[dOffset+3u*H+j]=dn*r;carry[idx]=dh*z;}";

    // GRU backward, step t: carry[b, j] += the matrix paths into h(t - 1) through W_hr, W_hz and W_hn.
    public static string GruSequenceBackwardHidden => Header + @"
layout(set=0,binding=0) readonly buffer DGates { float dGates[]; };
layout(set=0,binding=1) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=2) buffer Carry { float carry[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; uint t; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*hiddenSize)return;uint b=idx/hiddenSize,j=idx%hiddenSize,H=hiddenSize,dOffset=(t*batch+b)*4u*H;
    float sum=0.0;for(uint m=0u;m<H;++m)sum+=weightsHh[m*H+j]*dGates[dOffset+m]+weightsHh[(H+m)*H+j]*dGates[dOffset+H+m]+weightsHh[(2u*H+m)*H+j]*dGates[dOffset+3u*H+m];carry[idx]+=sum;}";

    // GRU input gradient for every step at once: gradInput [T, B, In].
    public static string GruSequenceInputGradient => Header + @"
layout(set=0,binding=0) readonly buffer DGates { float dGates[]; };
layout(set=0,binding=1) readonly buffer WeightsIh { float weightsIh[]; };
layout(set=0,binding=2) writeonly buffer GradInput { float gradInput[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=seqLen*batch*inputSize)return;uint i=idx%inputSize,row=idx/inputSize,H=hiddenSize,dOffset=row*4u*H;
    float sum=0.0;for(uint m=0u;m<H;++m)sum+=weightsIh[m*inputSize+i]*dGates[dOffset+m]+weightsIh[(H+m)*inputSize+i]*dGates[dOffset+H+m]+weightsIh[(2u*H+m)*inputSize+i]*dGates[dOffset+2u*H+m];gradInput[idx]=sum;}";

    // GRU weight gradients over every step and batch row. mode 0: gradWeightsIh [3H, In] from the input with dr, dz, dn;
    // mode 1: gradWeightsHh [3H, H] from h(t - 1) with dr, dz, dn * r.
    public static string GruSequenceWeightGradient => Header + @"
layout(set=0,binding=0) readonly buffer DGates { float dGates[]; };
layout(set=0,binding=1) readonly buffer Source { float source[]; };
layout(set=0,binding=2) writeonly buffer GradWeights { float gradWeights[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; uint mode; };
void main(){uint H=hiddenSize,width=mode==0u?inputSize:H,idx=gl_GlobalInvocationID.x;if(idx>=3u*H*width)return;uint gate=idx/width,k=idx%width,gi=gate/H,j=gate%H,slot=(mode==1u&&gi==2u)?3u:gi;
    float sum=0.0;for(uint row=0u;row<seqLen*batch;++row)sum+=dGates[row*4u*H+slot*H+j]*source[row*width+k];gradWeights[idx]=sum;}";

    // GRU bias gradients: b_i* takes dr, dz, dn; b_h* takes dr, dz, dn * r.
    public static string GruSequenceBiasGradient => Header + @"
layout(set=0,binding=0) readonly buffer DGates { float dGates[]; };
layout(set=0,binding=1) writeonly buffer GradBiasIh { float gradBiasIh[]; };
layout(set=0,binding=2) writeonly buffer GradBiasHh { float gradBiasHh[]; };
layout(push_constant) uniform Params { uint seqLen; uint batch; uint inputSize; uint hiddenSize; };
void main(){uint H=hiddenSize,g=gl_GlobalInvocationID.x;if(g>=3u*H)return;uint gi=g/H,j=g%H;float sumIh=0.0,sumHh=0.0;
    for(uint row=0u;row<seqLen*batch;++row){uint o=row*4u*H;sumIh+=dGates[o+gi*H+j];sumHh+=dGates[o+(gi==2u?3u:gi)*H+j];}gradBiasIh[g]=sumIh;gradBiasHh[g]=sumHh;}";

    public static string GruCellBackward => Header + @"
layout(set=0,binding=0) readonly buffer GradH { float gradH[]; };
layout(set=0,binding=1) readonly buffer GateR { float gateR[]; };
layout(set=0,binding=2) readonly buffer GateZ { float gateZ[]; };
layout(set=0,binding=3) readonly buffer GateN { float gateN[]; };
layout(set=0,binding=4) readonly buffer PrevH { float prevH[]; };
layout(set=0,binding=5) readonly buffer WeightsHh { float weightsHh[]; };
layout(set=0,binding=6) writeonly buffer GradPrevH { float gradPrevH[]; };
layout(set=0,binding=7) writeonly buffer GradGateR { float gradGateR[]; };
layout(set=0,binding=8) writeonly buffer GradGateZ { float gradGateZ[]; };
layout(set=0,binding=9) writeonly buffer GradGateN { float gradGateN[]; };
layout(push_constant) uniform Params { uint batch; uint hiddenSize; };
void main(){uint idx=gl_GlobalInvocationID.x;if(idx>=batch*hiddenSize)return;uint offset=(idx/hiddenSize)*hiddenSize,i=idx%hiddenSize;float r=gateR[idx],z=gateZ[idx],n=gateN[idx],previous=prevH[idx],localDh=gradH[idx],dRPre=0.0;for(uint j=0u;j<hiddenSize;++j){float dN=gradH[offset+j]*(1.0-gateZ[offset+j])*(1.0-gateN[offset+j]*gateN[offset+j]);dRPre+=dN*weightsHh[(2u*hiddenSize+j)*hiddenSize+i]*previous;}gradGateZ[idx]=localDh*(previous-n)*z*(1.0-z);gradGateN[idx]=localDh*(1.0-z)*(1.0-n*n);gradGateR[idx]=dRPre*r*(1.0-r);gradPrevH[idx]=localDh*z;}";
}
