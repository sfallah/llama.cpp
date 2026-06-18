#include "grounding.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace grounding {

namespace {

// softmax of one logit row (numerically stable)
std::vector<float> softmax_row(const float * logits, int n_vocab) {
    float maxl = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        maxl = std::max(maxl, logits[i]);
    }
    std::vector<float> p(n_vocab);
    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        float e = std::exp(logits[i] - maxl);
        p[i] = e;
        sum += e;
    }
    const float inv = (float) (1.0 / sum);
    for (int i = 0; i < n_vocab; i++) {
        p[i] *= inv;
    }
    return p;
}

llama_token argmax_row(const float * logits, int n_vocab) {
    llama_token best = 0;
    float bestv = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > bestv) {
            bestv = logits[i];
            best  = i;
        }
    }
    return best;
}

struct topk_entry { llama_token id; float prob; };

// top-k of a probability row, sorted by probability descending (ties broken by id, like torch.topk)
std::vector<topk_entry> topk_row(const std::vector<float> & probs, int k) {
    const int n = (int) probs.size();
    std::vector<llama_token> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    k = std::min(k, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](llama_token a, llama_token b) {
            if (probs[a] != probs[b]) return probs[a] > probs[b];
            return a < b;
        });
    std::vector<topk_entry> out(k);
    for (int i = 0; i < k; i++) out[i] = { idx[i], probs[idx[i]] };
    return out;
}

// is_valid_box_frame: classify the 6-row block as an empty box / a legal box / illegal
enum class frame { EMPTY, LEGAL, ILLEGAL };
frame box_frame(const std::vector<std::vector<float>> & probs, const token_ids & t,
                float start_thresh, float end_thresh) {
    if (probs[0][t.box_start] >= start_thresh) {
        if (probs[1][t.none_tok] > 0.2f && probs[2][t.box_end] > 0.2f &&
            probs[3][t.null_tok] > 0.1f && probs[4][t.null_tok] > 0.1f) {
            return frame::EMPTY;
        }
    }
    const float end_score = probs[5][t.box_end] + probs[5][t.null_tok] + probs[5][t.im_end];
    return end_score >= end_thresh ? frame::LEGAL : frame::ILLEGAL;
}

// decode_bbox_avg: returns the 6 committed tokens of a box, or nullopt if this block is not a box.
std::optional<std::vector<llama_token>> decode_bbox_avg(
        const std::vector<std::vector<float>> & probs, const token_ids & t, mode m, int keep_k) {
    const frame f = box_frame(probs, t, /*start*/ 0.7f, /*end*/ 0.2f);
    if (f == frame::EMPTY) {
        return std::vector<llama_token>{ t.box_start, t.none_tok, t.box_end, t.null_tok, t.null_tok, t.null_tok };
    }
    if (f == frame::ILLEGAL) {
        return std::nullopt;
    }

    llama_token coords[4];
    for (int pos = 0; pos < 4; pos++) { // block rows 1..4
        const auto tk = topk_row(probs[pos + 1], keep_k);
        // first valid (coordinate) token in the top-k
        int first = -1, valid_count = 0;
        llama_token vmax = 0, vmin = 0;
        float first_prob = 0.0f;
        for (int i = 0; i < (int) tk.size(); i++) {
            if (t.is_coord(tk[i].id)) {
                if (first < 0) { first = i; first_prob = tk[i].prob; vmax = vmin = tk[i].id; }
                else { vmax = std::max(vmax, tk[i].id); vmin = std::min(vmin, tk[i].id); }
                valid_count++;
            }
        }
        if (first < 0) {
            return std::nullopt; // a position with no coordinate candidate -> not a box
        }
        llama_token c = tk[first].id;
        if (m == mode::HYBRID) {
            const bool abnormal = first_prob < 0.9f && valid_count > 1 && (vmax - vmin) > 60;
            if (abnormal) c = 0; // mark uncertain -> handle_pattern will route to AR fallback
        }
        coords[pos] = c;
    }
    return std::vector<llama_token>{ t.box_start, coords[0], coords[1], coords[2], coords[3], t.box_end };
}

// decode_ref: returns [ref_start, semantic tokens...] if the block looks like a <ref> span, else nullopt.
std::optional<std::vector<llama_token>> decode_ref(
        const std::vector<std::vector<float>> & probs, const token_ids & t, int keep_k) {
    if (probs[0][t.ref_start] < 0.6f) {
        return std::nullopt;
    }
    std::vector<llama_token> out{ t.ref_start };
    for (int pos = 1; pos < (int) probs.size(); pos++) {
        const auto tk = topk_row(probs[pos], keep_k);
        int first = -1;
        for (int i = 0; i < (int) tk.size(); i++) {
            if (!t.is_coord(tk[i].id)) { first = i; break; } // first non-coordinate token
        }
        if (first < 0) {
            return std::nullopt;
        }
        out.push_back(tk[first].id);
    }
    return out;
}

block_result handle_pattern(const std::vector<llama_token> & v, const token_ids & t, mode m) {
    auto term = [&]() {
        return block_result{ block_type::IM_END, { t.im_end }, false, true };
    };
    if (v.empty() || v[0] == t.null_tok || v[0] == t.im_end) {
        return term();
    }
    if (v.size() >= 2 && v[0] == t.box_start && v[1] == t.none_tok) {
        return block_result{ block_type::EMPTY_BOX, { t.box_start, t.none_tok, t.box_end }, false, false };
    }
    if (v[0] == t.box_start) {
        int coord_ix = 1;
        for (int i = 1; i <= 4 && i < (int) v.size(); i++) {
            if (t.is_coord(v[i])) coord_ix++;
            else break;
        }
        if (coord_ix == 5 && v.size() >= 6 && v[5] == t.box_end) {
            return block_result{ block_type::COORD_BOX, v, false, false };
        }
        if (coord_ix == 3 && v.size() >= 4 && v[3] == t.box_end) {
            return block_result{ block_type::POINT_BOX, { v.begin(), v.begin() + 4 }, false, false };
        }
        if (m == mode::FAST) {
            return block_result{ block_type::COORD_BOX, v, false, false };
        }
        return block_result{ block_type::ERROR_BOX, { v.begin(), v.begin() + coord_ix }, true, false };
    }
    // ref_object: truncate at the first null token, drop a duplicated trailing </ref>
    std::vector<llama_token> r;
    for (llama_token id : v) {
        if (id == t.null_tok) break;
        r.push_back(id);
    }
    if (r.size() >= 2 && r[r.size() - 1] == t.ref_end && r[r.size() - 2] == t.ref_end) {
        r.pop_back();
    }
    return block_result{ block_type::REF_OBJECT, r, false, false };
}

} // namespace

block_result handle_block(const float * logits, int n_rows, int n_vocab, const token_ids & t, mode m) {
    // greedy top-1 per row + per-row softmax (needed for the probability thresholds)
    std::vector<llama_token> x0(n_rows);
    std::vector<std::vector<float>> probs(n_rows);
    for (int r = 0; r < n_rows; r++) {
        const float * row = logits + (size_t) r * n_vocab;
        x0[r]    = argmax_row(row, n_vocab);
        probs[r] = softmax_row(row, n_vocab);
    }

    // keep_k_avg defaults to 4 in the model's sample_tokens()
    auto box_avg = decode_bbox_avg(probs, t, m, /*keep_k*/ 4);
    if (!box_avg) {
        box_avg = decode_ref(probs, t, /*keep_k*/ 5);
    }
    const bool is_box_empty = !box_avg.has_value(); // fell through to the all-zero fallback
    const std::vector<llama_token> new_tokens = is_box_empty ? x0 : *box_avg;
    return handle_pattern(new_tokens, t, m);
}

} // namespace grounding
