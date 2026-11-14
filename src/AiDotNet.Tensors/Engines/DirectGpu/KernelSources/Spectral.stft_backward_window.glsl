#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float frameGrad[]; };
layout(set=0, binding=1) buffer B1 { float signal[]; };
layout(set=0, binding=2) buffer B2 { float gradWindow[]; };
layout(push_constant) uniform PC {
    int batch;
    int sigLen;
    int pad;
    int nFft;
    int hop;
    int numFrames;
};
#define P210_PI 3.14159265358979323846
void main() {
    int idx = int(gl_GlobalInvocationID.x); if (idx >= nFft) return;
    int i = idx; float acc = 0.0;
    for (int b = 0; b < batch; b++) {
        for (int f = 0; f < numFrames; f++) {
            int j = f*hop + i;
            int src = j < pad ? pad - j : (j < pad + sigLen ? j - pad : sigLen - 2 - (j - pad - sigLen));
            acc += frameGrad[(b*numFrames + f)*nFft + i] * signal[b*sigLen + src];
        }
    }
    gradWindow[i] = acc;
}
