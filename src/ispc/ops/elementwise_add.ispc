task void add_task(uniform float a[],
                   uniform float b[],
                   uniform float c[],
                   uniform int n,
                   uniform int span) {
    uniform int start = taskIndex * span;
    uniform int end = min(start + span, n);
    foreach (i = start ... end) {
        c[i] = a[i] + b[i];
    }
}

export void add_f32_ispc(uniform float a[],
                         uniform float b[],
                         uniform float c[],
                         uniform int n) {
    // 小算子串行阈值：n <= 2048 直接向量化执行，避免 launch 与任务系统开销
    if (n <= 2048) {
        foreach (i = 0 ... n) {
            c[i] = a[i] + b[i];
        }
        return;
    }

    uniform int num_tasks = 16;
    uniform int span = (n + num_tasks - 1) / num_tasks;
    launch[num_tasks] add_task(a, b, c, n, span);
    sync;
}
