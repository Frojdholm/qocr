extern "C" __global__
void batched_warp_perspective_tensor_kernel(
    const unsigned char* __restrict__ img,
    const float* __restrict__ M_inv,
    const int* __restrict__ widths,
    const int* __restrict__ angles,
    float* __restrict__ output,
    int B,
    int img_h,
    int img_w,
    int crop_h,
    int max_w,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b
) {
    int total_threads = B * crop_h * max_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_threads) return;

    int plane = crop_h * max_w;
    int b = idx / plane;
    int rem = idx % plane;
    int y_dst = rem / max_w;
    int x_dst = rem % max_w;

    int box_w = widths[b];
    if (x_dst >= box_w) return;

    int angle = angles ? angles[b] : 0;
    float x_eval = (angle == 180) ? (float)(box_w - 1 - x_dst) : (float)x_dst;
    float y_eval = (angle == 180) ? (float)(crop_h - 1 - y_dst) : (float)y_dst;

    const float* m = M_inv + b * 9;
    float u = m[0] * x_eval + m[1] * y_eval + m[2];
    float v = m[3] * x_eval + m[4] * y_eval + m[5];
    float w = m[6] * x_eval + m[7] * y_eval + m[8];

    float inv_w = (w != 0.0f) ? (1.0f / w) : 0.0f;
    float x_src = u * inv_w;
    float y_src = v * inv_w;

    // Bicubic interpolation (OpenCV cubic spline, a = -0.75f)
    int x0 = (int)floorf(x_src);
    int y0 = (int)floorf(y_src);
    float fx = x_src - (float)x0;
    float fy = y_src - (float)y0;

    const float a = -0.75f;
    float wx[4];
    float wy[4];

    // 1D cubic spline weights for x
    {
        float t = fx;
        float t2 = t * t;
        float t3 = t2 * t;
        wx[0] = ((a * (t + 1.0f) - 5.0f * a) * (t + 1.0f) + 8.0f * a) * (t + 1.0f) - 4.0f * a;
        wx[1] = ((a + 2.0f) * t - (a + 3.0f)) * t2 + 1.0f;
        wx[2] = ((a + 2.0f) * (1.0f - t) - (a + 3.0f)) * (1.0f - t) * (1.0f - t) + 1.0f;
        wx[3] = ((a * (2.0f - t) - 5.0f * a) * (2.0f - t) + 8.0f * a) * (2.0f - t) - 4.0f * a;
    }

    // 1D cubic spline weights for y
    {
        float t = fy;
        float t2 = t * t;
        float t3 = t2 * t;
        wy[0] = ((a * (t + 1.0f) - 5.0f * a) * (t + 1.0f) + 8.0f * a) * (t + 1.0f) - 4.0f * a;
        wy[1] = ((a + 2.0f) * t - (a + 3.0f)) * t2 + 1.0f;
        wy[2] = ((a + 2.0f) * (1.0f - t) - (a + 3.0f)) * (1.0f - t) * (1.0f - t) + 1.0f;
        wy[3] = ((a * (2.0f - t) - 5.0f * a) * (2.0f - t) + 8.0f * a) * (2.0f - t) - 4.0f * a;
    }

    float r_sum = 0.0f;
    float g_sum = 0.0f;
    float b_sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int py = max(0, min(y0 - 1 + j, img_h - 1));
        int row_offset = py * img_w * 3;
        float w_y = wy[j];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int px = max(0, min(x0 - 1 + i, img_w - 1));
            int idx_px = row_offset + px * 3;
            float w_xy = w_y * wx[i];

            r_sum += w_xy * (float)img[idx_px + 0];
            g_sum += w_xy * (float)img[idx_px + 1];
            b_sum += w_xy * (float)img[idx_px + 2];
        }
    }

    float r_val = max(0.0f, min(r_sum, 255.0f));
    float g_val = max(0.0f, min(g_sum, 255.0f));
    float b_val = max(0.0f, min(b_sum, 255.0f));

    float norm_r = ((r_val / 255.0f) - mean_r) / std_r;
    float norm_g = ((g_val / 255.0f) - mean_g) / std_g;
    float norm_b = ((b_val / 255.0f) - mean_b) / std_b;

    int out_b_offset = b * 3 * plane;
    output[out_b_offset + 0 * plane + rem] = norm_r;
    output[out_b_offset + 1 * plane + rem] = norm_g;
    output[out_b_offset + 2 * plane + rem] = norm_b;
}
