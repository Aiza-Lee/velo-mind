task void embedding_task(uniform const float table[],
                         uniform const int indices[],
                         uniform float y[],
                         uniform int hidden,
                         uniform int total_tokens,
                         uniform int tokens_per_task) {
    uniform int start_tok = taskIndex * tokens_per_task;
    uniform int end_tok = min(start_tok + tokens_per_task, total_tokens);

    for (uniform int t = start_tok; t < end_tok; ++t) {
        uniform int token = indices[t];
        uniform const float * uniform table_row = table + token * hidden;
        uniform float * uniform y_row = y + t * hidden;

        foreach (j = 0 ... hidden) {
            y_row[j] = table_row[j];
        }
    }
}

export void embedding_f32_i32_ispc(uniform const float table[],
                                   uniform const int indices[],
                                   uniform float y[],
                                   uniform int hidden,
                                   uniform int total_tokens) {
    // 小算子串行阈值：少于等于 4 个 token 直接在当前线程执行
    if (total_tokens <= 4) {
        for (uniform int t = 0; t < total_tokens; ++t) {
            uniform int token = indices[t];
            uniform const float * uniform table_row = table + token * hidden;
            uniform float * uniform y_row = y + t * hidden;

            foreach (j = 0 ... hidden) {
                y_row[j] = table_row[j];
            }
        }
        return;
    }

    uniform int num_tasks = total_tokens > 32 ? 32 : total_tokens;
    if (num_tasks < 1) num_tasks = 1;
    uniform int tokens_per_task = (total_tokens + num_tasks - 1) / num_tasks;

    launch[num_tasks] embedding_task(table, indices, y, hidden, total_tokens, tokens_per_task);
    sync;
}
