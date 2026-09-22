export void relu_f32_ispc(uniform float x[],
                          uniform float y[],
                          uniform int n) {
    foreach (i = 0 ... n) {
        float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}
