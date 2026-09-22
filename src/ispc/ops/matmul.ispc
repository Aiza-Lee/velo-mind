task void matmul_tile_task(uniform const float A[],
                           uniform const float B[],
                           uniform float C[],
                           uniform int batch,
                           uniform int M,
                           uniform int K,
                           uniform int N,
                           uniform int n_tiles,
                           uniform int N_step,
                           uniform int total_tiles,
                           uniform int tiles_per_task) {
    uniform int start_tile = taskIndex * tiles_per_task;
    uniform int end_tile = min(start_tile + tiles_per_task, total_tiles);

    for (uniform int tile_idx = start_tile; tile_idx < end_tile; ++tile_idx) {
        uniform int r = tile_idx / n_tiles;
        uniform int c_tile = tile_idx % n_tiles;

        uniform int b = r / M;
        uniform int m = r % M;

        uniform int n_start = c_tile * N_step;
        uniform int n_end = min(n_start + N_step, N);

        uniform const float * uniform a_row = A + (b * M + m) * K;
        uniform const float * uniform b_mat = B + b * (K * N);
        uniform float * uniform c_row = C + (b * M + m) * N;

        foreach (n = n_start ... n_end) {
            c_row[n] = 0.0f;
        }

        uniform int k = 0;
        for (; k + 3 < K; k += 4) {
            uniform float a0 = a_row[k];
            uniform float a1 = a_row[k + 1];
            uniform float a2 = a_row[k + 2];
            uniform float a3 = a_row[k + 3];
            uniform const float * uniform b0 = b_mat + k * N;
            uniform const float * uniform b1 = b_mat + (k + 1) * N;
            uniform const float * uniform b2 = b_mat + (k + 2) * N;
            uniform const float * uniform b3 = b_mat + (k + 3) * N;
            foreach (n = n_start ... n_end) {
                c_row[n] += a0 * b0[n] + a1 * b1[n] + a2 * b2[n] + a3 * b3[n];
            }
        }
        for (; k < K; ++k) {
            uniform float a_val = a_row[k];
            uniform const float * uniform b_row = b_mat + k * N;
            foreach (n = n_start ... n_end) {
                c_row[n] += a_val * b_row[n];
            }
        }
    }
}

export void matmul_f32_ispc(uniform const float A[],
                            uniform const float B[],
                            uniform float C[],
                            uniform int batch,
                            uniform int M,
                            uniform int K,
                            uniform int N) {
    uniform int total_rows = batch * M;
    // 每个任务处理若干 64 列分块，列内由 ISPC gang 向量化。
    uniform int N_step = 64;
    uniform int n_tiles = (N + N_step - 1) / N_step;
    uniform int total_tiles = total_rows * n_tiles;

    // 小算子串行阈值：单 tile 或极小规模矩阵直接在当前线程串行执行，避免任务调度与同步开销
    if (total_tiles <= 1) {
        for (uniform int r = 0; r < total_rows; ++r) {
            uniform int b = r / M;
            uniform int m = r % M;
            uniform const float * uniform a_row = A + (b * M + m) * K;
            uniform const float * uniform b_mat = B + b * (K * N);
            uniform float * uniform c_row = C + (b * M + m) * N;

            foreach (n = 0 ... N) {
                c_row[n] = 0.0f;
            }

            uniform int k = 0;
            for (; k + 3 < K; k += 4) {
                uniform float a0 = a_row[k];
                uniform float a1 = a_row[k + 1];
                uniform float a2 = a_row[k + 2];
                uniform float a3 = a_row[k + 3];
                uniform const float * uniform b0 = b_mat + k * N;
                uniform const float * uniform b1 = b_mat + (k + 1) * N;
                uniform const float * uniform b2 = b_mat + (k + 2) * N;
                uniform const float * uniform b3 = b_mat + (k + 3) * N;
                foreach (n = 0 ... N) {
                    c_row[n] += a0 * b0[n] + a1 * b1[n] + a2 * b2[n] + a3 * b3[n];
                }
            }
            for (; k < K; ++k) {
                uniform float a_val = a_row[k];
                uniform const float * uniform b_row = b_mat + k * N;
                foreach (n = 0 ... N) {
                    c_row[n] += a_val * b_row[n];
                }
            }
        }
        return;
    }

    uniform int num_tasks = total_tiles > 64 ? 64 : total_tiles;
    if (num_tasks < 1) num_tasks = 1;
    uniform int tiles_per_task = (total_tiles + num_tasks - 1) / num_tasks;

    launch[num_tasks] matmul_tile_task(A, B, C, batch, M, K, N, n_tiles, N_step, total_tiles, tiles_per_task);
    sync;
}
