export void sigmoid_f32_ispc(uniform float x[],
                            uniform float y[],
                            uniform int n) {
    foreach (i = 0 ... n) {
        y[i] = 1.0f / (1.0f + exp(-x[i]));
    }
}
