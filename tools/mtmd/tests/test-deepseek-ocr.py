#!/usr/bin/env python3
"""
Evaluates llama.cpp's DeepSeek-OCR by comparing its output for a test
image to the actual text in part of that image.

Runs the test image through mtmd-cli for both DeepSeek-OCR and
DeepSeek-OCR-2, calculates CER and chrF for each output, and holds them
against the HF model's scores. Exits non-zero if either variant fails.
"""

import argparse
import logging
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("deepseek-ocr-test")

DEFAULT_IMAGE = "test-1.jpeg"
DEFAULT_EXPECTED_TEXT = "test-1-ground-truth.txt"
RUN_TIMEOUT = 300


@dataclass
class ModelSpec:
    """A DeepSeek-OCR variant: CLI flags, default paths, and its parity baseline.

    The gate is parity with the upstream HuggingFace model on test-1.jpeg: CER
    must stay within `cer_tol` above the HF model's CER, and chrF within
    `chrf_tol` below it.
    """
    key: str             # short id, e.g. "v1"
    label: str           # human-readable name
    model_arg: str       # CLI flag overriding the model GGUF path
    mmproj_arg: str      # CLI flag overriding the mmproj GGUF path
    model_default: str   # default model GGUF path, relative to the repo root
    mmproj_default: str  # default mmproj GGUF path, relative to the repo root
    hf_cer: float        # upstream HF model's CER on the test image
    hf_chrf: float       # upstream HF model's chrF on the test image
    cer_tol: float       # allowed CER slack above the HF reference
    chrf_tol: float      # allowed chrF slack below the HF reference

    @property
    def cer_max(self) -> float:
        return self.hf_cer + self.cer_tol

    @property
    def chrf_min(self) -> float:
        return self.hf_chrf - self.chrf_tol


MODELS = [
    ModelSpec(
        key="v1", label="DeepSeek-OCR",
        model_arg="--llama-model", mmproj_arg="--mmproj",
        model_default="gguf_models/deepseek-ai/deepseek-ocr-bf16.gguf",
        mmproj_default="gguf_models/deepseek-ai/mmproj-deepseek-ocr-bf16.gguf",
        # deepseek-ai/DeepSeek-OCR (greedy) on test-1.jpeg vs test-1-ground-truth.txt.
        # llama.cpp scores better than this (CER ~0.24), so the gate has margin.
        hf_cer=0.3030, hf_chrf=67.52, cer_tol=0.02, chrf_tol=2.0,
    ),
    ModelSpec(
        key="v2", label="DeepSeek-OCR-2",
        model_arg="--llama-model-2", mmproj_arg="--mmproj-2",
        model_default="gguf_models/deepseek-ai/deepseek-ocr-2-bf16.gguf",
        mmproj_default="gguf_models/deepseek-ai/mmproj-deepseek-ocr-2-bf16.gguf",
        # deepseek-ai/DeepSeek-OCR-2 on test-1.jpeg. Both the HF model and llama.cpp
        # fail this low-quality full-page scan: they read the headlines but cannot
        # resolve the small body text and hallucinate it. The HF reference decodes
        # with no_repeat_ngram_size; run_mtmd_cli matches that with the DRY sampler,
        # so the two are compared on equal footing. chrF is the load-bearing gate
        # here -- it sits at ~34 when the hallucinated body does not loop and
        # craters to ~24 if it does. llama.cpp currently scores CER ~0.77 / chrF ~34.
        hf_cer=0.6894, hf_chrf=34.60, cer_tol=0.12, chrf_tol=8.0,
    ),
]


def arg_dest(flag: str) -> str:
    """argparse destination name for a long option, e.g. --llama-model -> llama_model."""
    return flag.lstrip("-").replace("-", "_")


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def normalize_text(text: str) -> str:
    """NFC-normalize and collapse whitespace, so line-wrap and spacing
    don't count as CER errors."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def locally_align(expected: str, ocr_out: str) -> str:
    """Return the span of `ocr_out` that best matches `expected`.

    The ground truth covers part of the article body.
    But the test image includes half of the newspaper's front page.
    Fuzzy partial-ratio matching picks out
    the body so the unrelated text doesn't disturb CER / chrF.
    """
    from rapidfuzz import fuzz
    alignment = fuzz.partial_ratio_alignment(expected, ocr_out)
    if alignment is None or alignment.dest_end <= alignment.dest_start:
        return ocr_out
    return ocr_out[alignment.dest_start:alignment.dest_end]


def compute_cer(expected: str, ocr_out: str) -> float:
    """Character Error Rate. Lower is better.
    CER: fraction of characters you'd insert/delete/substitute to fix the output; 0 = perfect."""
    import jiwer
    return jiwer.cer(expected, ocr_out)


def compute_chrf(expected: str, ocr_out: str) -> float:
    """chrF score on 0-100. Higher is better.
    chrF: F-score over shared character n-grams; more forgiving of small word/spacing drift than CER.
    """
    from sacrebleu.metrics import CHRF
    return CHRF().sentence_score(ocr_out, [expected]).score


def run_mtmd_cli(model_path, mmproj_path, image_path, bin_path) -> str:
    """Run mtmd-cli on the image and return its output."""
    cmd = [
        str(bin_path),
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--image", str(image_path),
        "-p", "Free OCR. ",
        "--chat-template", "deepseek-ocr",
        "--temp", "0",
        "--flash-attn", "off",  # match the HF "eager" attention reference
        "--no-warmup",
        "-n", "512",  # cap generation: enough for a full transcription, and bounds
                      # a model that loops (DeepSeek-OCR-2 loops on hard images, as
                      # does the HF reference -- without a cap the KV cache fills)
        # The HF reference always decodes with no_repeat_ngram_size (20/35). Without
        # equivalent repetition control, greedy llama.cpp loops verbatim on a hard
        # image and its CER is not comparable to the HF score. DRY is llama.cpp's
        # analog: it penalises only long verbatim repeats, so a clean transcription
        # is left intact. The default DRY sequence breakers include "\n", which
        # would stop it from seeing multi-line loops, so they are cleared with
        # --dry-sequence-breaker none.
        "--dry-multiplier", "0.8",
        "--dry-base", "1.75",
        "--dry-allowed-length", "2",
        "--dry-penalty-last-n", "-1",
        "--dry-sequence-breaker", "none",
    ]
    logger.debug(f"  command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=False, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        if e.stderr:
            logger.error("llama.cpp stderr:\n%s", e.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"llama-mtmd-cli timed out after {RUN_TIMEOUT}s")

    if result.returncode != 0:
        logger.error("llama.cpp stderr:\n%s", result.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"llama-mtmd-cli failed with code {result.returncode}")

    output = result.stdout.decode("utf-8", errors="replace").strip()
    if not output:
        raise RuntimeError("llama-mtmd-cli produced no output on stdout")
    logger.info(f"  output: {len(output)} chars")
    return output


def read_expected_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def evaluate(spec: "ModelSpec", expected: str, ocr_out: str) -> bool:
    expected = normalize_text(expected)
    ocr_out = normalize_text(ocr_out)
    aligned = locally_align(expected, ocr_out)

    logger.debug(f"\n--- expected (normalized) ---\n{expected}")
    logger.debug(f"\n--- OCR output (normalized) ---\n{ocr_out}")
    logger.debug(f"\n--- aligned span ---\n{aligned}")

    cer = compute_cer(expected, aligned)
    chrf = compute_chrf(expected, aligned)

    cer_pass = cer <= spec.cer_max
    chrf_pass = chrf >= spec.chrf_min
    passed = cer_pass and chrf_pass

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"{spec.label}: Free OCR evaluation")
    logger.info("=" * 60)
    logger.info(f"  CER               {cer:>7.4f}    (HF {spec.hf_cer:.4f}, <= {spec.cer_max:>7.4f}  -> {verdict(cer_pass)})")
    logger.info(f"  chrF (0-100)      {chrf:>7.2f}    (HF {spec.hf_chrf:.2f}, >= {spec.chrf_min:>7.2f}  -> {verdict(chrf_pass)})")
    logger.info(f"  Expected chars    {len(expected):>7}")
    logger.info(f"  Aligned chars     {len(aligned):>7} (of {len(ocr_out)} OCR chars)")
    logger.info("")
    logger.info(f"  Result: {verdict(passed)}")
    logger.info("=" * 60)
    return passed


def argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare llama.cpp DeepSeek-OCR output with a ground-truth transcript")
    ap.add_argument("--llama-bin", default="build/bin/llama-mtmd-cli",
                    help="Path to llama-mtmd-cli binary (relative to repo root or absolute)")
    for spec in MODELS:
        ap.add_argument(spec.model_arg, default=spec.model_default,
                        help=f"Path to the {spec.label} GGUF model (relative to repo root or absolute)")
        ap.add_argument(spec.mmproj_arg, default=spec.mmproj_default,
                        help=f"Path to the {spec.label} mmproj GGUF file (relative to repo root or absolute)")
    ap.add_argument("--verbose", action="store_true",
                    help="Also log the expected, OCR, and aligned text")
    return ap


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(message)s")


def resolve_path(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def main() -> int:
    args = argument_parser().parse_args()
    configure_logging(args.verbose)

    tests_dir = Path(__file__).parent  # tools/mtmd/tests
    mtmd_dir = tests_dir.parent  # tools/mtmd
    repo_root = mtmd_dir.parent.parent  # repo root

    image = resolve_path(DEFAULT_IMAGE, mtmd_dir)
    expected_text_path = resolve_path(DEFAULT_EXPECTED_TEXT, tests_dir)
    binary = resolve_path(args.llama_bin, repo_root)

    # shared inputs must exist before any model can run
    for label, path in [("image", image),
                        ("expected-text", expected_text_path),
                        ("binary", binary)]:
        if not path.exists():
            logger.error(f"Error: {label} not found: {path}")
            return 1

    logger.info("=" * 60)
    logger.info("DeepSeek-OCR: llama.cpp vs HF parity check")
    logger.info("=" * 60)

    expected = read_expected_text(expected_text_path)
    logger.info(f"Expected text: {len(expected)} chars")

    results: dict[str, bool] = {}
    for spec in MODELS:
        model = resolve_path(getattr(args, arg_dest(spec.model_arg)), repo_root)
        mmproj = resolve_path(getattr(args, arg_dest(spec.mmproj_arg)), repo_root)

        logger.info("")
        logger.info("#" * 60)
        logger.info(f"# {spec.label}")
        logger.info("#" * 60)

        missing = [(lbl, p) for lbl, p in [("model", model), ("mmproj", mmproj)]
                   if not p.exists()]
        if missing:
            for lbl, p in missing:
                logger.error(f"  Error: {spec.label} {lbl} not found: {p}")
            results[spec.label] = False
            continue

        logger.info("  Running llama.cpp 'Free OCR'")
        try:
            ocr_out = run_mtmd_cli(model, mmproj, image, binary)
        except RuntimeError as e:
            logger.error(f"  Error: {e}")
            results[spec.label] = False
            continue

        results[spec.label] = evaluate(spec, expected, ocr_out)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for label, ok in results.items():
        logger.info(f"  {label:<16} {verdict(ok)}")
    all_passed = all(results.values())
    logger.info("")
    logger.info(f"  Overall: {verdict(all_passed)}")
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
