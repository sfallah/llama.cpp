#include "ggml/src/ggml-metal/ggml-metal-device.h"
#include "models.h"

// Implementation based on approach suggested by Acly
// See: https://github.com/ggml-org/llama.cpp/pull/17383#issuecomment-3554227091
static ggml_tensor * window_partition(ggml_context * ctx0, ggml_tensor * x, const int window) {
    auto [c, w, h, b] = x->ne;
    // same as
    // x = ggml_win_part(m, x, window);
    // x = ggml_reshape_3d(m, x, c, window * window, x->ne[3]);

    const int64_t px  = (window - w % window) % window;
    const int64_t py  = (window - h % window) % window;
    const int64_t npw = (w + px) / window;
    const int64_t nph = (h + py) / window;

    ggml_tensor * cur = x;
    if (px > 0 || py > 0) {
        cur = ggml_pad(ctx0, cur, 0, static_cast<int>(px), static_cast<int>(py), 0);
    }
    cur = ggml_reshape_4d(ctx0, cur, c * window, npw, window, nph * b);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_4d(ctx0, cur, c, window, window, npw * nph * b);
    return cur;
}

// Implementation based on approach suggested by Acly
// See: https://github.com/ggml-org/llama.cpp/pull/17383#issuecomment-3554227091
static ggml_tensor * window_unpartition(ggml_context * ctx0,
                                        ggml_tensor *  x,
                                        const int      w,
                                        const int      h,
                                        const int      window) {
    const int64_t c = x->ne[0];
    // same as
    // x = ggml_reshape_4d(m, x, c, window, window, x->ne[2]);
    // x = ggml_win_unpart(m, x, w, h, window);

    const int64_t px  = (window - w % window) % window;
    const int64_t py  = (window - h % window) % window;
    const int64_t npw = (w + px) / window;
    const int64_t nph = (h + py) / window;

    const int64_t b   = x->ne[3] / (npw * nph);
    ggml_tensor * cur = x;
    cur               = ggml_reshape_4d(ctx0, cur, c * window, window, npw, nph * b);
    cur               = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur               = ggml_reshape_4d(ctx0, cur, c, w + px, h + py, b);
    cur               = ggml_view_4d(ctx0, cur, cur->ne[0], w, h, cur->ne[3], cur->nb[1], cur->nb[2], cur->nb[3], 0);
    cur               = ggml_cont(ctx0, cur);
    return cur;
}

static ggml_tensor * get_rel_pos(ggml_context * ctx0,
                                 ggml_tensor *  rel_pos,  // [L, C]
                                 ggml_tensor *  indices,  // [q_size, k_size]
                                 const int      q_size,
                                 const int      k_size) {
    const int64_t C = rel_pos->ne[0];  // channels
    const int64_t L = rel_pos->ne[1];  // length

    GGML_ASSERT(indices != nullptr);
    GGML_ASSERT(indices->type == GGML_TYPE_I32);
    GGML_ASSERT(indices->ne[0] == k_size);
    GGML_ASSERT(indices->ne[1] == q_size);

    const auto    max_rel_dist = 2 * std::max(q_size, k_size) - 1;
    ggml_tensor * cur          = rel_pos;

    if (max_rel_dist != L) {
        // Linear interpolation
        const int64_t ne0 = cur->ne[0];
        const int64_t ne1 = cur->ne[1];
        const int64_t ne2 = cur->ne[2];
        const int64_t ne3 = cur->ne[3];

        cur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3)), ne1, 1, ne0 * ne2 * ne3);
        cur = ggml_reshape_4d(
            ctx0, ggml_interpolate(ctx0, cur, max_rel_dist, 1, ne0 * ne2 * ne3, 1, GGML_SCALE_MODE_BILINEAR),
            max_rel_dist, ne0, ne2, ne3);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    }

    // Flatten indices to 1D for ggml_get_rows
    const int qk = q_size * k_size;

    cur = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, cur, ggml_reshape_1d(ctx0, indices, qk)), C, k_size, q_size);

    return cur;  // [C, k_size, q_size]
}

ggml_cgraph * clip_graph_deepseekocr2::build() {
    // patch embedding
    ggml_tensor * inp_raw = build_inp_raw();

    ggml_tensor * sam_out = build_sam(inp_raw);

    ggml_tensor * qwen2_out;
    // Building Qwen2 Decoder2Encoder
    {
        ggml_tensor * inp;

        inp = ggml_cpy(ctx0, sam_out, ggml_dup_tensor(ctx0, sam_out));
        inp = ggml_reshape_2d(ctx0, inp, inp->ne[0] * inp->ne[1], inp->ne[2]);  // H*W, C
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

        auto num_image_tokens = inp->ne[1];  // H*W

        // query based on numbers of image tokens (in SAM output)
        // 16x16 -> query_1024 (1024x1024 images)
        // 12x12 -> query_768 (768x768 images)

        ggml_tensor * query_embed = model.resample_query_1024;
        int           num_queries = 256;

        if (num_image_tokens == 144) {
            query_embed = model.resample_query_768;
            num_queries = 144;
        }

        // Concatenate: image tokens + query tokens
        // Shape: (B, num_image_tokens + num_queries, C)
        inp = ggml_concat(ctx0, inp, ggml_cast(ctx0, query_embed, inp->type), 1);

        auto seq_len = inp->ne[1];

        // attention mask for the qwen2 encoder, see CustomQwen2Decoder._create_custom_4d_mask
        // sequence layout is [image tokens | query tokens]; the values are computed on
        // the host and uploaded in clip.cpp set_input() (see "qwen2_attn_mask")
        ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_len, seq_len);
        ggml_set_name(attn_mask, "qwen2_attn_mask");
        ggml_set_input(attn_mask);

        ggml_tensor * cur;
        ggml_tensor * inpL = inp;

        ggml_tensor * inp_pos = ggml_cast(ctx0, ggml_arange(ctx0, 0, seq_len, 1), GGML_TYPE_I32);

        int  num_heads = 14;
        auto head_dim  = n_embd / num_heads;
        int  kv_heads  = 2;

        // qwen2 layers
        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * inpSA = inpL;
            const auto &  layer = model.layers[il];

            // norm
            cur = build_norm(inpL, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_RMS, eps, il);
            cb(cur, "attn_norm", il);

            auto n_tokens = cur->ne[1];

            // self-attention
            {
                // compute Q and K and RoPE them
                ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.q_w, cur), layer.q_b);
                ggml_tensor * Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.k_w, cur), layer.k_b);
                ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.v_w, cur), layer.v_b);

                Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, num_heads, n_tokens);
                Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, kv_heads, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, kv_heads, n_tokens);

                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 131072, 1000000, 1, 0,
                                     1, 0, 0);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 131072, 1000000, 1, 0,
                                     1, 0, 0);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur, attn_mask, 1.0f / sqrtf(float(d_head)), il);
            }
            ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = build_norm(ffn_inp, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_RMS, eps, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur, layer.ff_up_w, nullptr, layer.ff_gate_w, nullptr, layer.ff_down_w, nullptr, FFN_SILU,
                            il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);
            inpL = cur;
        }
        cur = inpL;

        cur = build_norm(cur, model.post_ln_w, model.post_ln_b, NORM_TYPE_RMS, eps, -1);
        cur = ggml_cont(ctx0,
                        ggml_view_2d(ctx0, cur, cur->ne[0], num_queries, cur->nb[1],
                                     cur->nb[1] * (cur->ne[1] - num_queries)));  // only take query tokens for output

        ggml_build_forward_expand(gf, cur);
        qwen2_out = cur;
    }

    ggml_tensor * cur;

    cur = ggml_mul_mat(ctx0, model.mm_fc_w, qwen2_out);
    cur = ggml_add(ctx0, cur, model.mm_fc_b);

    const auto n_dim = cur->ne[0];

    ggml_tensor * vs;
    vs  = ggml_reshape_2d(ctx0, model.view_seperator, n_dim, 1);  // (n_dim, 1)
    cur = ggml_concat(ctx0, cur, vs, 1);                          // (n_dim, h*(w+1) + 1)

    cb(cur, "dsocr2_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
