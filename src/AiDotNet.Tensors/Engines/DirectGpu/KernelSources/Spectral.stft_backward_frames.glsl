#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float grad[]; };
layout(set=0, binding=1) buffer B1 { float magnitude[]; };
layout(set=0, binding=2) buffer B2 { float phase[]; };
layout(set=0, binding=3) buffer B3 { float frameGrad[]; };
layout(push_constant) uniform PC {
    int batch;
    int nFft;
    int numFrames;
    int numFreqs;
    int phaseTarget;
};
#define P210_PI 3.14159265358979323846
// Phase-gradient magnitude floor: BackwardFunctions.PhaseGradientMagnitudeFloor. DC and the even-length Nyquist bin
// are real for a real frame, so their phase is piecewise constant (0 or pi) and its gradient is exactly zero.
const float STFT_BWD_PHASE_FLOOR = 1e-12;
void main() {
    int idx = int(gl_GlobalInvocationID.x); if (idx >= batch*numFrames*nFft) return;
    int i = idx % nFft; int tmp = idx / nFft; int frame = tmp % numFrames; int b = tmp / numFrames;
    int specOff = b*numFreqs*numFrames + frame;
    // Twiddle e^(+2*pi*i*m/nFft) from the integer phase m = k*i mod nFft, quarter turns exact, as in the forward.
    float tw = 2.0 * P210_PI / float(nFft); float acc = 0.0; int m = 0;
    for (int k = 0; k < numFreqs; k++) {
        int bin = specOff + k*numFrames; float g = grad[bin]; float p = phase[bin];
        float cp = cos(p); float sp = sin(p); float sr; float si;
        if (phaseTarget != 0) { float mg = magnitude[bin]; float r = (k != 0 && 2 * k != nFft && mg > STFT_BWD_PHASE_FLOOR) ? g / mg : 0.0; sr = -r*sp; si = r*cp; }
        else { sr = g*cp; si = g*sp; }
        float c; float s;
        if (m == 0) { c = 1.0; s = 0.0; }
        else if (2 * m == nFft) { c = -1.0; s = 0.0; }
        else if (4 * m == nFft) { c = 0.0; s = 1.0; }
        else if (4 * m == 3 * nFft) { c = 0.0; s = -1.0; }
        else { float a = tw * float(2 * m > nFft ? m - nFft : m); c = cos(a); s = sin(a); }
        acc += sr*c - si*s;
        m += i; if (m >= nFft) m -= nFft;
    }
    frameGrad[idx] = acc;
}
