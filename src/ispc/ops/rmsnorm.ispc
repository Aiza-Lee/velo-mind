task void rmsnorm_task(uniform const float x[],
                       uniform const float weight[],
                       uniform float y[],
                       uniform int reduce_dim,
                       uniform float eps,
                       uniform int total_rows,
                       uniform int rows_per_task) {
    uniform int start_row = taskIndex * rows_per_task;
    uniform int end_row = min(start_row + rows_per_task, total_rows);

    for (uniform int r = start_row; r < end_row; ++r) {
        uniform const float * uniform x_row = x + r * reduce_dim;
        uniform float * uniform y_row = y + r * reduce_dim;

        float sumsq = 0.0f;
        foreach (i = 0 ... reduce_dim) {
            float v = x_row[i];
            sumsq += v * v;
        }
        uniform float total_sum = reduce_add(sumsq);
        uniform float mean_sq = total_sum / (float)reduce_dim;
        uniform float inv_rms = 1.0f / sqrt(mean_sq + eps);

        foreach (i = 0 ... reduce_dim) {
            y_row[i] = x_row[i] * inv_rms * weight[i];
        }
    }
}

export void rmsnorm_f32_ispc(uniform const float x[],
                             uniform const float weight[],
                             uniform float y[],
                             uniform int reduce_dim,
                             uniform float eps,
                             uniform int total_rows) {
    // 小算子串行阈值：小批次/单行直接在当前线程串行执行，避免任务启动与线程同步开销
    if (total_rows <= 2) {
        for (uniform int r = 0; r < total_rows; ++r) {
            uniform const float * uniform x_row = x + r * reduce_dim;
            uniform float * uniform y_row = y + r * reduce_dim;

            float sumsq = 0.0f;
            foreach (i = 0 ... reduce_dim) {
                float v = x_row[i];
                sumsq += v * v;
            }
            uniform float total_sum = reduce_add(sumsq);
            uniform float mean_sq = total_sum / (float)reduce_dim;
            uniform float inv_rms = 1.0f / sqrt(mean_sq + eps);

            foreach (i = 0 ... reduce_dim) {
                y_row[i] = x_row[i] * inv_rms * weight[i];
            }
        }
        return;
    }

    uniform int num_tasks = total_rows > 32 ? 32 : total_rows;
    if (num_tasks < 1) num_tasks = 1;
    uniform int rows_per_task = (total_rows + num_tasks - 1) / num_tasks;

    launch[num_tasks] rmsnorm_task(x, weight, y, reduce_dim, eps, total_rows, rows_per_task);
    sync;
}
