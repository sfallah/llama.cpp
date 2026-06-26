// DEBUG-ONLY instrumentation for the DeepSeek-OCR single-view regression hunt.
// Throwaway: lives only on the sf/dsocr-debug-* branches, gated by DSOCR_DEBUG_DIR.
// Dumps intermediate tensors as NumPy .npy (float32, C-order) for element-wise diff.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// write a float32 buffer as a NumPy v1.0 .npy file (shape given in C-order)
inline void dsocr_debug_write_npy(const std::string & path, const float * data, const std::vector<int64_t> & shape) {
    std::string shape_str;
    int64_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size()) shape_str += ", ";
        n *= shape[i];
    }
    if (shape.size() == 1) shape_str += ",";  // numpy wants (N,) for 1-D

    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + shape_str + "), }";
    // pad with spaces so 10 (magic+version+len) + dict + '\n' is a 64-byte multiple
    const size_t unpadded = 10 + dict.size() + 1;
    const size_t pad      = (64 - (unpadded % 64)) % 64;
    dict.append(pad, ' ');
    dict += '\n';

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[dsocr-debug] could not open %s for writing\n", path.c_str());
        return;
    }
    const unsigned char magic[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00 };
    const uint16_t header_len = (uint16_t) dict.size();
    fwrite(magic, 1, 8, f);
    fwrite(&header_len, sizeof(uint16_t), 1, f);  // little-endian host (x86/arm)
    fwrite(dict.data(), 1, dict.size(), f);
    fwrite(data, sizeof(float), (size_t) n, f);
    fclose(f);
}

// read a computed ggml tensor to host, convert to f32, and dump as <dir>/<name>.npy
inline void dsocr_debug_dump_tensor(const std::string & dir, const ggml_tensor * t) {
    const int n_dims = ggml_n_dims(t);
    std::vector<int64_t> shape;  // ggml ne[0] is innermost -> numpy last axis
    for (int i = n_dims - 1; i >= 0; --i) shape.push_back(t->ne[i]);

    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, f.data(), 0, n * sizeof(float));
    } else {
        std::vector<char> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
        const auto * tt = ggml_get_type_traits(t->type);
        if (tt && tt->to_float) {
            tt->to_float(raw.data(), f.data(), n);
        } else {
            fprintf(stderr, "[dsocr-debug] no to_float for type %d (%s), skipping\n", t->type, t->name);
            return;
        }
    }

    const std::string path = dir + "/" + std::string(t->name) + ".npy";
    dsocr_debug_write_npy(path, f.data(), shape);

    fprintf(stderr, "[dsocr-debug] dumped %-13s shape=[", t->name);
    for (size_t i = 0; i < shape.size(); ++i) fprintf(stderr, "%s%lld", i ? "," : "", (long long) shape[i]);
    fprintf(stderr, "] type=%s -> %s\n", ggml_type_name(t->type), path.c_str());
}

// ggml_backend_sched eval callback: dump the three named encoder-stage tensors
inline bool dsocr_debug_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool wanted = strcmp(t->name, "sam_output")   == 0 ||
                        strcmp(t->name, "clip_out")     == 0 ||
                        strcmp(t->name, "dsocr_output") == 0;
    if (ask) {
        return wanted;  // only observe (get a data callback for) the wanted tensors
    }
    if (wanted) {
        dsocr_debug_dump_tensor((const char *) user_data, t);
    }
    return true;  // continue graph evaluation
}
