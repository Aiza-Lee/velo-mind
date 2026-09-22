task void transpose_task(uniform const float x[],
                         uniform float y[],
                         uniform const int out_strides[],
                         uniform const int in_strides[],
                         uniform const int perm[],
                         uniform int rank,
                         uniform int total,
                         uniform int span) {
    uniform int start = taskIndex * span;
    uniform int end = min(start + span, total);

    for (uniform int flat = start; flat < end; ++flat) {
        uniform int rem = flat;
        uniform int in_flat = 0;
        for (uniform int ax = 0; ax < rank; ++ax) {
            uniform int coord = rem / out_strides[ax];
            rem = rem % out_strides[ax];
            uniform int in_axis = perm[ax];
            in_flat += coord * in_strides[in_axis];
        }
        y[flat] = x[in_flat];
    }
}

export void transpose_f32_ispc(uniform const float x[],
                              uniform float y[],
                              uniform const int out_strides[],
                              uniform const int in_strides[],
                              uniform const int perm[],
                              uniform int rank,
                              uniform int total) {
    // 小算子串行阈值：小张量 (<= 256) 直接在当前线程串行执行
    if (total <= 256) {
        for (uniform int flat = 0; flat < total; ++flat) {
            uniform int rem = flat;
            uniform int in_flat = 0;
            for (uniform int ax = 0; ax < rank; ++ax) {
                uniform int coord = rem / out_strides[ax];
                rem = rem % out_strides[ax];
                uniform int in_axis = perm[ax];
                in_flat += coord * in_strides[in_axis];
            }
            y[flat] = x[in_flat];
        }
        return;
    }

    uniform int num_tasks = total > 128 ? 32 : 1;
    uniform int span = (total + num_tasks - 1) / num_tasks;
    launch[num_tasks] transpose_task(x, y, out_strides, in_strides, perm, rank, total, span);
    sync;
}
