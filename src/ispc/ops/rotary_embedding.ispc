task void rotary_embedding_task(uniform const float x[],
                                uniform const float cos_cache[],
                                uniform const float sin_cache[],
                                uniform float y[],
                                uniform int head_dim,
                                uniform int half_dim,
                                uniform int cos_rows,
                                uniform int total_rows,
                                uniform int rows_per_task) {
    uniform int start_row = taskIndex * rows_per_task;
    uniform int end_row = min(start_row + rows_per_task, total_rows);

    for (uniform int r = start_row; r < end_row; ++r) {
        uniform const float * uniform x_row = x + r * head_dim;
        uniform float * uniform y_row = y + r * head_dim;
        uniform int pos = (cos_rows > 1) ? (r % cos_rows) : 0;
        uniform const float * uniform cos_row = cos_cache + pos * half_dim;
        uniform const float * uniform sin_row = sin_cache + pos * half_dim;

        foreach (j = 0 ... half_dim) {
            float a = x_row[j];
            float b = x_row[j + half_dim];
            float c = cos_row[j];
            float s = sin_row[j];
            y_row[j]            = a * c - b * s;
            y_row[j + half_dim] = b * c + a * s;
        }
    }
}

export void rotary_embedding_f32_ispc(uniform const float x[],
                                     uniform const float cos_cache[],
                                     uniform const float sin_cache[],
                                     uniform float y[],
                                     uniform int head_dim,
                                     uniform int cos_rows,
                                     uniform int total_rows) {
    uniform int half_dim = head_dim / 2;

    // 小算子串行阈值：单步 decode / 小头数直接在当前线程串行执行，避免任务启动与线程同步开销
    if (total_rows <= 4) {
        for (uniform int r = 0; r < total_rows; ++r) {
            uniform const float * uniform x_row = x + r * head_dim;
            uniform float * uniform y_row = y + r * head_dim;
            uniform int pos = (cos_rows > 1) ? (r % cos_rows) : 0;
            uniform const float * uniform cos_row = cos_cache + pos * half_dim;
            uniform const float * uniform sin_row = sin_cache + pos * half_dim;

            foreach (j = 0 ... half_dim) {
                float a = x_row[j];
                float b = x_row[j + half_dim];
                float c = cos_row[j];
                float s = sin_row[j];
                y_row[j]            = a * c - b * s;
                y_row[j + half_dim] = b * c + a * s;
            }
        }
        return;
    }

    uniform int num_tasks = total_rows > 32 ? 32 : total_rows;
    if (num_tasks < 1) num_tasks = 1;
    uniform int rows_per_task = (total_rows + num_tasks - 1) / num_tasks;

    launch[num_tasks] rotary_embedding_task(x, cos_cache, sin_cache, y,
                                            head_dim, half_dim, cos_rows,
                                            total_rows, rows_per_task);
    sync;
}
