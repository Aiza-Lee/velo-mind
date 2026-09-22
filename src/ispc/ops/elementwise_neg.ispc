export void neg_f32_ispc(uniform float x[],
                         uniform float y[],
                         uniform int n) {
    foreach (i = 0 ... n) {
        y[i] = -x[i];
    }
}
