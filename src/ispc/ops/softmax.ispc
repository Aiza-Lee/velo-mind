task void softmax_task(uniform const float x[],
                       uniform float y[],
                       uniform int cols,
                       uniform int total_rows,
                       uniform int rows_per_task, uniform int inner) {
    uniform int start_row = taskIndex * rows_per_task;
    uniform int end_row = min(start_row + rows_per_task, total_rows);

    for (uniform int r = start_row; r < end_row; ++r) {
        uniform const float * uniform x_row = x + (r / inner) * cols * inner + r % inner;
        uniform float * uniform y_row = y + (r / inner) * cols * inner + r % inner;

        float max_v = x_row[0];
        foreach (i = 0 ... cols) {
            if (x_row[i * inner] > max_v) max_v = x_row[i * inner];
        }
        uniform float row_max = reduce_max(max_v);

        float sum_exp = 0.0f;
        foreach (i = 0 ... cols) {
            sum_exp += exp(x_row[i * inner] - row_max);
        }
        uniform float row_sum = reduce_add(sum_exp);
        uniform float inv_sum = 1.0f / row_sum;

        foreach (i = 0 ... cols) {
            y_row[i * inner] = exp(x_row[i * inner] - row_max) * inv_sum;
        }
    }
}

export void softmax_f32_ispc(uniform const float x[],
                             uniform float y[],
                             uniform int cols,
                             uniform int total_rows, uniform int inner) {
    // 小算子串行阈值：小批次/单行直接在当前线程串行执行，避免任务启动与线程同步开销
    if (total_rows <= 2) {
        for (uniform int r = 0; r < total_rows; ++r) {
            uniform const float * uniform x_row = x + (r / inner) * cols * inner + r % inner;
            uniform float * uniform y_row = y + (r / inner) * cols * inner + r % inner;

            float max_v = x_row[0];
            foreach (i = 0 ... cols) {
                if (x_row[i * inner] > max_v) max_v = x_row[i * inner];
            }
            uniform float row_max = reduce_max(max_v);

            float sum_exp = 0.0f;
            foreach (i = 0 ... cols) {
                sum_exp += exp(x_row[i * inner] - row_max);
            }
            uniform float row_sum = reduce_add(sum_exp);
            uniform float inv_sum = 1.0f / row_sum;

            foreach (i = 0 ... cols) {
                y_row[i * inner] = exp(x_row[i * inner] - row_max) * inv_sum;
            }
        }
        return;
    }

    uniform int num_tasks = total_rows > 32 ? 32 : total_rows;
    if (num_tasks < 1) num_tasks = 1;
    uniform int rows_per_task = (total_rows + num_tasks - 1) / num_tasks;

    launch[num_tasks] softmax_task(x, y, cols, total_rows, rows_per_task, inner);
    sync;
}
