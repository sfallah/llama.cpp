#!/usr/bin/env python3
"""
Regression test for the llama.cpp DeepSeek-OCR port.

Runs mtmd-cli on a test image, locally aligns the output against a ground-truth
transcript, and gates on parity with the upstream HuggingFace reference model
(CER and chrF). See README.md in this directory for the full methodology and
how to re-measure the baseline.
"""

import argparse
import logging
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger("deepseek-ocr-test")

# Embedding model used for the informational cosine-similarity sanity check.
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
EMBEDDING_PREFIX = "task: sentence similarity | query: "
EMBEDDING_MAX_SEQ_LENGTH = 2048
EMBEDDING_LOWERCASE = True

# Default test fixtures. The image is resolved against tools/mtmd; the
# ground-truth file is resolved against tools/mtmd/tests (this script's dir).
# Override on the command line with --image / --expected-text to run the same
# harness against a clean "positive" image, etc. -- but if you change the
# image the parity baselines below no longer apply, so override the thresholds
# too.
DEFAULT_IMAGE = "test-1.jpeg"
DEFAULT_EXPECTED_TEXT = "test-1-ground-truth.txt"
RUN_TIMEOUT = 300

# HuggingFace `deepseek-ai/DeepSeek-OCR` reference model's quality on
# test-1.jpeg, measured against test-1-ground-truth.txt with the same
# normalise + local-align + CER/chrF pipeline used here. The reference
# transcript itself is saved at tests/test-1-extracted.txt for inspection.
# Re-measure (and update these constants) if the test image, ground truth,
# or upstream model output changes.
HF_REFERENCE_CER = 0.3030
HF_REFERENCE_CHRF = 67.52

# How much worse than the HF reference llama.cpp is allowed to be before the
# test fails. Set to small positive values so floating-point / tokeniser noise
# doesn't flip a passing run -- tighten if you want a stricter parity gate.
CER_TOLERANCE = 0.02
CHRF_TOLERANCE = 2.0

# Third-party loggers silenced outside --trace mode.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub",
                 "sentence_transformers", "transformers", "filelock", "fsspec")


class Thresholds(NamedTuple):
    cer: float
    chrf: float
    sim: float


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def normalize_text(text: str) -> str:
    """NFC-normalize and collapse whitespace runs so cosmetic line-wrap /
    spacing differences don't show up as CER errors."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def locally_align(reference: str, hypothesis: str) -> str:
    """Return the span of `hypothesis` that best matches `reference`.

    The ground-truth fixture covers only the article body, while mtmd-cli
    transcribes the whole page (masthead included). A fuzzy partial-ratio
    alignment locates the window of the hypothesis that best matches the
    reference, so CER / chrF aren't dominated by the unmatched material.
    """
    from rapidfuzz import fuzz
    alignment = fuzz.partial_ratio_alignment(reference, hypothesis)
    if alignment is None or alignment.dest_end <= alignment.dest_start:
        return hypothesis
    return hypothesis[alignment.dest_start:alignment.dest_end]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. 0 is perfect; lower is better."""
    import jiwer
    return jiwer.cer(reference, hypothesis)


def compute_chrf(reference: str, hypothesis: str) -> float:
    """chrF score on 0-100. Higher is better; robust to small alignment drift."""
    from sacrebleu.metrics import CHRF
    return CHRF().sentence_score(hypothesis, [reference]).score


def run_mtmd_deepseek_ocr(model_path, mmproj_path, image_path, bin_path, prompt="Free OCR. ") -> str:
    """Run inference using llama.cpp mtmd-cli and return the generated text."""
    cmd = [
        str(bin_path),
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--image", str(image_path),
        "-p", prompt,
        "--chat-template", "deepseek-ocr",
        "--temp", "0",
        "--flash-attn", "off",  # match the HF "eager" attention reference
        "--no-warmup",
    ]
    logger.debug(f"  command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=False, timeout=RUN_TIMEOUT)
    if result.returncode != 0:
        logger.error("llama.cpp stderr:\n%s", result.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"llama-mtmd-cli failed with code {result.returncode}")

    output = result.stdout.decode("utf-8", errors="replace").strip()
    logger.info(f"  output: {len(output)} chars")
    return output


def _chunk_by_tokens(text: str, tokenizer, max_tokens: int, stride: int) -> list[str]:
    """Split text into overlapping token windows that fit the embedding model's context."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return [""]
    chunks = []
    start = 0
    while start < len(ids):
        chunks.append(tokenizer.decode(ids[start:start + max_tokens], skip_special_tokens=True))
        if start + max_tokens >= len(ids):
            break
        start += stride
    return chunks


def _doc_embedding(text: str, embed_model, max_tokens: int, stride: int, prefix: str = ""):
    """Mean-pool the (normalized) per-chunk embeddings, then re-normalize."""
    import torch

    chunks = _chunk_by_tokens(text, embed_model.tokenizer, max_tokens, stride)
    chunk_embs = embed_model.encode(
        [prefix + c for c in chunks],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return torch.nn.functional.normalize(chunk_embs.mean(dim=0, keepdim=True), p=2, dim=1)


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_embedding_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between two (possibly long) texts using a chunked doc embedding."""
    from sentence_transformers import SentenceTransformer

    device = _pick_device()
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (device={device})")
    if EMBEDDING_LOWERCASE:
        text1, text2 = text1.lower(), text2.lower()

    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    embed_model.eval()
    # The sentence-transformers config sometimes caps this lower than the
    # architecture's max_position_embeddings; use the configured value.
    embed_model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH

    # Reserve room for special tokens and the per-chunk instruction prefix.
    prefix_len = len(embed_model.tokenizer.encode(EMBEDDING_PREFIX, add_special_tokens=False))
    max_tokens = max(8, embed_model.get_max_seq_length() - 2 - prefix_len)
    stride = max(1, max_tokens // 2)  # 50% overlap so chunk boundaries keep context

    logger.info("Computing embeddings...")
    emb1 = _doc_embedding(text1, embed_model, max_tokens, stride, EMBEDDING_PREFIX)
    emb2 = _doc_embedding(text2, embed_model, max_tokens, stride, EMBEDDING_PREFIX)
    return float(embed_model.similarity(emb1, emb2).item())


def read_expected_output(file_path: str) -> str:
    """Read expected OCR output from a file path (absolute or pre-resolved)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def evaluate(label: str, reference: str, hypothesis: str, thresholds: Thresholds) -> bool:
    """Compute CER, chrF, and embedding similarity on the locally-aligned span,
    log a report, and return True iff CER and chrF both pass."""
    reference_n = normalize_text(reference)
    hypothesis_n = normalize_text(hypothesis)
    aligned = locally_align(reference_n, hypothesis_n)

    cer = compute_cer(reference_n, aligned)
    chrf = compute_chrf(reference_n, aligned)
    similarity = compute_embedding_similarity(reference_n, aligned)

    logger.debug(f"\n--- {label}: reference (normalized) ---\n{reference_n}")
    logger.debug(f"\n--- {label}: hypothesis aligned span (normalized) ---\n{aligned}")

    cer_pass = cer <= thresholds.cer
    chrf_pass = chrf >= thresholds.chrf
    sim_pass = similarity >= thresholds.sim
    passed = cer_pass and chrf_pass

    logger.info("")
    logger.info("=" * 60)
    logger.info(f" {label}")
    logger.info("=" * 60)
    logger.info(f"  CER               {cer:>7.4f}    (<= {thresholds.cer:>7.4f}  -> {verdict(cer_pass)})")
    logger.info(f"  chrF (0-100)      {chrf:>7.2f}    (>= {thresholds.chrf:>7.2f}  -> {verdict(chrf_pass)})")
    logger.info(f"  Embedding cosine  {similarity:>7.4f}    (>= {thresholds.sim:>7.2f}  -> {verdict(sim_pass)}, informational)")
    logger.info(f"  Reference chars   {len(reference_n):>7}")
    logger.info(f"  Aligned chars     {len(aligned):>7} (of {len(hypothesis_n)} hypothesis chars)")
    logger.info("")
    logger.info(f"  Result: {verdict(passed)}")
    logger.info("=" * 60)
    return passed


def build_argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare llama.cpp DeepSeek-OCR output with a ground-truth transcript")
    ap.add_argument("--llama-model", default="gguf_models/deepseek-ai/deepseek-ocr-bf16.gguf",
                    help="Path to llama.cpp GGUF model (relative to repo root or absolute)")
    ap.add_argument("--mmproj", default="gguf_models/deepseek-ai/mmproj-deepseek-ocr-bf16.gguf",
                    help="Path to mmproj GGUF file (relative to repo root or absolute)")
    ap.add_argument("--llama-bin", default="build/bin/llama-mtmd-cli",
                    help="Path to llama-mtmd-cli binary (relative to repo root or absolute)")
    ap.add_argument("--image", default=DEFAULT_IMAGE,
                    help="Path to the test image (relative to tools/mtmd or absolute). "
                         "Use a clean, high-quality image to sanity-check the metrics.")
    ap.add_argument("--expected-text", default=DEFAULT_EXPECTED_TEXT,
                    help="Ground-truth plain-text transcript (relative to this script's dir or absolute)")
    ap.add_argument("--cer-threshold", type=float,
                    default=HF_REFERENCE_CER + CER_TOLERANCE,
                    help=("Maximum CER for PASS (lower is better). "
                          f"Default {HF_REFERENCE_CER + CER_TOLERANCE:.4f} = HF reference CER "
                          f"({HF_REFERENCE_CER:.4f}) + tolerance ({CER_TOLERANCE:.2f})."))
    ap.add_argument("--chrf-threshold", type=float,
                    default=HF_REFERENCE_CHRF - CHRF_TOLERANCE,
                    help=("Minimum chrF (0-100) for PASS (higher is better). "
                          f"Default {HF_REFERENCE_CHRF - CHRF_TOLERANCE:.2f} = HF reference chrF "
                          f"({HF_REFERENCE_CHRF:.2f}) - tolerance ({CHRF_TOLERANCE:.1f})."))
    ap.add_argument("--sim-threshold", type=float, default=0.7,
                    help="Embedding cosine similarity floor (informational only; does not gate PASS)")
    ap.add_argument("--verbose", action="store_true",
                    help="Log the reference and aligned-hypothesis texts at DEBUG level")
    ap.add_argument("--trace", action="store_true",
                    help="TRACE mode: --verbose plus third-party library logging "
                         "(HF Hub HTTP traffic, sentence-transformers, transformers, ...)")
    return ap


def configure_logging(verbose: bool, trace: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if (verbose or trace) else logging.INFO,
                        format="%(message)s")
    if not trace:
        for noisy in NOISY_LOGGERS:
            logging.getLogger(noisy).setLevel(logging.WARNING)


def resolve_path(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def main() -> int:
    args = build_argument_parser().parse_args()
    configure_logging(args.verbose, args.trace)

    thresholds = Thresholds(cer=args.cer_threshold, chrf=args.chrf_threshold, sim=args.sim_threshold)

    tests_dir = Path(__file__).parent          # tools/mtmd/tests
    mtmd_dir = tests_dir.parent                # tools/mtmd
    repo_root = mtmd_dir.parent.parent         # repo root

    inputs = [
        ("image",         resolve_path(args.image,         mtmd_dir)),
        ("expected-text", resolve_path(args.expected_text, tests_dir)),
        ("model",         resolve_path(args.llama_model,   repo_root)),
        ("mmproj",        resolve_path(args.mmproj,        repo_root)),
        ("binary",        resolve_path(args.llama_bin,     repo_root)),
    ]
    for label, path in inputs:
        if not path.exists():
            logger.error(f"Error: {label} not found: {path}")
            return 1
    paths = dict(inputs)

    # Header banner.
    logger.info("=" * 60)
    logger.info("DeepSeek-OCR: llama.cpp vs ground-truth comparison")
    logger.info("=" * 60)
    logger.info(f"Embedding model:       {EMBEDDING_MODEL_NAME} (max_seq_length={EMBEDDING_MAX_SEQ_LENGTH})")
    logger.info(f"HF reference baseline: CER {HF_REFERENCE_CER:.4f}, chrF {HF_REFERENCE_CHRF:.2f}")
    logger.info(f"Parity gates:          CER <= {thresholds.cer:.4f}, chrF >= {thresholds.chrf:.2f}")

    logger.debug("")
    logger.debug("Resolved test inputs:")
    for label, path in inputs:
        logger.debug(f"  {label:<14} {path}")

    logger.info("")
    logger.info("[1/3] Running llama.cpp 'Free OCR'")
    llama_free_ocr = run_mtmd_deepseek_ocr(paths["model"], paths["mmproj"],
                                           paths["image"], paths["binary"])

    logger.info("")
    logger.info("[2/3] Reading expected output")
    expected_free_ocr = read_expected_output(paths["expected-text"])
    logger.info(f"  reference: {len(expected_free_ocr)} chars")

    logger.info("")
    logger.info("[3/3] Computing OCR quality metrics")
    ok = evaluate("Free OCR", expected_free_ocr, llama_free_ocr, thresholds)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
