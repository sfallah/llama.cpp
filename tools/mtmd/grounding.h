#pragma once

// LocateAnything Parallel Box Decoding (PBD / "MTP fast mode").
//
// This is a faithful C++ port of the model's generate_utils.py block decode: the model predicts a
// whole box/point/ref in one bidirectional block of `block_size` masked positions instead of one
// token at a time. These are pure functions over a block of logits; the KV/forward plumbing lives
// in the caller (mtmd-cli).

#include "llama.h"

#include <vector>

namespace grounding {

enum class mode {
    SLOW,    // pure autoregressive next-token prediction (no PBD)
    FAST,    // PBD only, never fall back to AR
    HYBRID,  // PBD first, fall back to AR on an uncertain/malformed box
};

// Model-specific control token ids (LocateAnythingConfig). Defaults match the released config.
struct token_ids {
    llama_token box_start   = 151668; // <box>
    llama_token box_end     = 151669; // </box>
    llama_token ref_start   = 151672; // <ref>
    llama_token ref_end     = 151673; // </ref>
    llama_token coord_start = 151677; // <0>
    llama_token coord_end   = 152677; // <1000>
    llama_token text_mask   = 151676; // <text_mask>
    llama_token none_tok    = 4064;   // "none"
    llama_token null_tok    = 152678; // <null>
    llama_token switch_tok  = 152679; // <switch>
    llama_token im_end      = 151645; // <|im_end|> (eos)

    bool is_coord(llama_token id) const { return id >= coord_start && id <= coord_end; }
};

enum class block_type {
    COORD_BOX,   // <box> x1 x2 x3 x4 </box>
    POINT_BOX,   // <box> x y </box>
    EMPTY_BOX,   // <box> none </box>
    REF_OBJECT,  // <ref> ...semantic tokens...
    IM_END,      // terminal
    ERROR_BOX,   // malformed -> hybrid switches to AR
};

struct block_result {
    block_type type;
    std::vector<llama_token> tokens; // tokens to commit for this block
    bool need_switch_to_ar = false;
    bool is_terminal       = false;
};

// block_size logit rows of n_vocab each (row-major). Returns the classified block + committed tokens.
block_result handle_block(const float * logits, int n_rows, int n_vocab, const token_ids & t, mode m);

} // namespace grounding
