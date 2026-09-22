task void rsqrt_task(uniform const float x[],
                     uniform float y[],
                     uniform int n,
                     uniform int span) {
    uniform int start = taskIndex * span;
    uniform int end = min(start + span, n);
    foreach (i = start ... end) {
        y[i] = 1.0f / sqrt(x[i]);
    }
}

export void rsqrt_f32_ispc(uniform const float x[],
                           uniform float y[],
                           uniform int n) {
    // 小算子串行阈值：n <= 1024 直接向量化执行，避免 launch 与任务系统开销
    if (n <= 1024) {
        foreach (i = 0 ... n) {
            y[i] = 1.0f / sqrt(x[i]);
        }
        return;
    }

    uniform int num_tasks = 16;
    uniform int span = (n + num_tasks - 1) / num_tasks;
    launch[num_tasks] rsqrt_task(x, y, n, span);
    sync;
}
