# DeepSeek-OCR regression test

`test-deepseek-ocr.py` exercises the llama.cpp DeepSeek-OCR port end-to-end:
it runs `mtmd-cli` on a fixture image, scores the output against a
ground-truth transcript with OCR-appropriate metrics, and gates pass/fail on
**parity with the upstream HuggingFace reference model**.

Exit code is `0` on PASS and `1` on FAIL, so the script slots into CI as-is.

---

## Quick start

One-time setup (using [`uv`](https://docs.astral.sh/uv/)):

```sh
uv venv tools/mtmd/tests/.venv --python 3.11
VIRTUAL_ENV="$(pwd)/tools/mtmd/tests/.venv" \
    uv pip install -r tools/mtmd/tests/tests-requirements.txt
```

Run from the repo root:

```sh
tools/mtmd/tests/.venv/bin/python tools/mtmd/tests/test-deepseek-ocr.py
```

The test takes a couple of minutes (one mtmd-cli inference pass + an embedding
model download/inference on first run).

---

## What the test gates on

The default fixture (`../test-1.jpeg`) is a deliberately low-quality scan of a
1969 NYT moon-landing front page. Even the upstream `deepseek-ai/DeepSeek-OCR`
HuggingFace model can't transcribe it cleanly, so an absolute "good OCR"
threshold would always FAIL. Instead the test asks:

> Is llama.cpp's output at least as good as the upstream HF model's output,
> when both are scored against the same ground truth?

That's a *parity gate*. It catches regressions in the C++ port without
demanding the impossible from the underlying model. If you swap the test
image for a high-quality scan, llama.cpp's CER drops to 0 — see
[Sanity check on the positive image](#sanity-check-on-the-positive-image) below
for a one-shot check you can run by hand.

---

## Metrics

| Metric | Library | Direction | Role |
|---|---|---|---|
| **CER** (character error rate) | `jiwer` | lower is better, 0 = perfect | gates PASS/FAIL |
| **chrF** (character n-gram F-score) | `sacrebleu` | higher is better, 100 = perfect | gates PASS/FAIL |
| Embedding cosine similarity | `sentence-transformers` (`google/embeddinggemma-300m`) | higher is better | **informational only** |

Why CER + chrF instead of embedding cosine alone? Sentence embeddings score
*meaning*, not character fidelity. A 24% CER transcript and a 0% CER
transcript can land within 0.05 of each other on cosine similarity, so the
score has too little dynamic range to act as a gate. CER tells you directly
how many characters need to be edited; chrF is a more forgiving partner that
handles small alignment drift. The embedding score is kept as a sanity floor
(catches the case where the OCR transcribes something completely unrelated).

### Local alignment

The ground truth covers only the article body — not the page masthead —
while `mtmd-cli` transcribes the whole page. `rapidfuzz.fuzz.partial_ratio_alignment`
locates the span of the OCR output that best matches the reference, and all
three metrics are computed on that aligned window. This removes the need for
brittle keyword-anchor trimming and means the masthead doesn't penalise
the score.

Both texts are NFC-normalised and whitespace-collapsed before scoring, so
cosmetic line-wrap differences don't show up as character errors.

---

## The parity baseline

Hard-coded in `test-deepseek-ocr.py`:

```python
HF_REFERENCE_CER  = 0.3030   # deepseek-ai/DeepSeek-OCR on test-1.jpeg
HF_REFERENCE_CHRF = 67.52    # scored against test-1-ground-truth.txt
CER_TOLERANCE     = 0.02
CHRF_TOLERANCE    = 2.0
```

The default thresholds derive directly:

- `--cer-threshold  = HF_REFERENCE_CER + CER_TOLERANCE  = 0.3230`
- `--chrf-threshold = HF_REFERENCE_CHRF − CHRF_TOLERANCE = 65.52`

llama.cpp currently scores `CER 0.2441` / `chrF 71.35` against the same
ground truth, comfortably *better* than the HF reference, so the test passes
with margin.

### Re-measuring the baseline

If you change the test image or ground truth, the baseline is stale. To
re-measure:

1. Run `deepseek-ai/DeepSeek-OCR` (the upstream HF model) directly on the new
   image with the same `Free OCR.` prompt and save the output to
   `test-1-extracted.txt`.
2. Compute CER and chrF of that output against the new ground truth using
   the same `normalize_text` + `locally_align` pipeline this test uses, e.g.:

   ```python
   import unicodedata, jiwer
   from rapidfuzz import fuzz
   from sacrebleu.metrics import CHRF

   def norm(t): return " ".join(unicodedata.normalize("NFC", t).split())

   with open("tools/mtmd/tests/test-1-ground-truth.txt") as f:
       ref = norm(f.read())
   with open("tools/mtmd/tests/test-1-extracted.txt") as f:
       hyp = norm(f.read())

   alignment = fuzz.partial_ratio_alignment(ref, hyp)
   aligned = hyp[alignment.dest_start:alignment.dest_end]

   print("CER ", jiwer.cer(ref, aligned))
   print("chrF", CHRF().sentence_score(aligned, [ref]).score)
   ```

3. Update `HF_REFERENCE_CER` and `HF_REFERENCE_CHRF` in the script.

The tolerances exist to absorb floating-point / tokeniser noise; tighten
them if you want a stricter parity gate.

---

## Fixtures

| File | What it is |
|---|---|
| `../test-1.jpeg` | **Negative case** — the full NYT front page at low resolution, with smudged, faded type. This is the default input: it stresses the OCR on a hard image. |
| `test-1-positive.png` | **Positive case** — a clean, high-resolution crop of just the "A Powdery Surface" article body from the same front page. Same article text as the ground truth, but easy to read. Used for the sanity-check invocation below. |
| `test-1-ground-truth.txt` | Clean, human-corrected transcript of the "A Powdery Surface" article body. Applies to **both** images: the negative image contains this text (among the rest of the front page) at bad quality, and the positive image is just this text at good quality. |
| `test-1-extracted.txt` | Output of upstream `deepseek-ai/DeepSeek-OCR` on `test-1.jpeg`, used to derive the parity baseline. |

The same ground truth is reused on purpose: it makes the pair a controlled
A/B — only image quality changes between the negative and positive run, so
any difference in CER/chrF is attributable to OCR difficulty rather than to
the metric or the fixture.

The ground truth covers only the article body; it does not include the page
masthead. `locally_align()` handles the resulting size mismatch on the
negative image (the masthead is excluded from scoring).

---

## Useful flags

```
--image PATH             Test image (relative to tools/mtmd or absolute)
--expected-text PATH     Ground-truth transcript (relative to this dir or absolute)
--cer-threshold FLOAT    Override CER gate (default: HF reference + 0.02)
--chrf-threshold FLOAT   Override chrF gate (default: HF reference − 2.0)
--sim-threshold FLOAT    Override embedding-cosine floor (informational; default 0.70)
--verbose                DEBUG-level logging: resolved input paths, full
                         mtmd-cli command, reference and aligned-hypothesis
                         spans.
--trace                  Like --verbose, plus third-party library logging
                         (HF Hub HTTP traffic, sentence-transformers, ...).
--llama-bin PATH         Override llama-mtmd-cli path
--llama-model PATH       Override DeepSeek-OCR GGUF path
--mmproj PATH            Override mmproj GGUF path
```

If you pass `--image`, the parity baseline no longer applies — pass
`--cer-threshold` and `--chrf-threshold` too.

### Sanity check on the positive image

To verify the pipeline itself (alignment, normalisation, metric code) is
behaving correctly, point the test at the committed clean crop of the same
article — `tests/test-1-positive.png` — and tighten the gates accordingly:

```sh
tools/mtmd/tests/.venv/bin/python tools/mtmd/tests/test-deepseek-ocr.py \
    --image tests/test-1-positive.png \
    --cer-threshold 0.05 \
    --chrf-threshold 90
```

(`--image` is resolved against `tools/mtmd/`, so `tests/test-1-positive.png`
finds the fixture next to this README.)

You should see something close to `CER 0.0000`, `chrF 100.00`, `Embedding
cosine 1.0000` — confirming that when the input is legible, the model and the
test pipeline produce a character-perfect match. If this run *doesn't* pass,
the regression is in the pipeline (or the C++ port), not the negative
fixture's image quality.

---

## Prerequisites

The script expects (paths relative to the repo root, all overridable):

- `build/bin/llama-mtmd-cli` — built llama.cpp binary
- `gguf_models/deepseek-ai/deepseek-ocr-bf16.gguf` — model
- `gguf_models/deepseek-ai/mmproj-deepseek-ocr-bf16.gguf` — multimodal projector

Missing any of these prints a clear `Error: <thing> not found: <path>` and
exits with code 1.

---

## Troubleshooting

- **`Error: binary not found`** — build llama.cpp first (`cmake --build build -j`).
- **`Error: model not found`** — convert the HF weights with `convert_hf_to_gguf.py` and place them under `gguf_models/deepseek-ai/`, or pass `--llama-model` / `--mmproj`.
- **Embedding model download stalls** — `embeddinggemma-300m` is fetched from HuggingFace on first run; ensure outbound HTTPS works and you've accepted the model's terms on HF Hub if required.
- **Metric numbers move slightly between runs** — `mtmd-cli` is invoked with `--temp 0`, so OCR output is deterministic; small drift would come from upstream sentence-transformers / torch updates. If embedding cosine moves but CER/chrF don't, you're fine (only CER/chrF gate the test).
