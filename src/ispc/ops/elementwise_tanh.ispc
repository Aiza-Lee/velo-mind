export void tanh_f32_ispc(uniform float x[],
                          uniform float y[],
                          uniform int n) {
    foreach (i = 0 ... n) {
        float e2x = exp(2.0f * x[i]);
        y[i] = (e2x - 1.0f) / (e2x + 1.0f);
    }
}
