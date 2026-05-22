#!/usr/bin/env python3
"""
Evaluates llama.cpp's DeepSeek-OCR by comparing its output for a test
image to the actual text in part of that image.

Each test case runs one image through mtmd-cli for one DeepSeek-OCR
variant, calculates CER and chrF, and holds them against the HF model's
scores. The cases cover:

  - DeepSeek-OCR   on a single-view scan (test-1.jpeg, 640x488)
  - DeepSeek-OCR-2 on the same single-view scan
  - DeepSeek-OCR-2 on a tall crop (test-1-positive.png, 429x806) that is
    large enough to exercise the multi-tile dynamic-resolution path

Exits non-zero if any case fails its parity gate.
"""

import argparse
import logging
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("deepseek-ocr-test")

RUN_TIMEOUT = 300


@dataclass
class ModelSpec:
    """A DeepSeek-OCR variant: CLI flags and default GGUF paths."""
    key: str             # short id, e.g. "v1"
    label: str           # human-readable name
    model_arg: str       # CLI flag overriding the model GGUF path
    mmproj_arg: str      # CLI flag overriding the mmproj GGUF path
    model_default: str   # default model GGUF path, relative to the repo root
    mmproj_default: str  # default mmproj GGUF path, relative to the repo root


@dataclass
class TestCase:
    """One image run for one model, with a parity gate vs the HF reference.

    The gate is parity with the upstream HuggingFace model: CER must stay
    within `cer_tol` above the HF model's CER, and chrF within `chrf_tol`
    below it.
    """
    model_key: str       # ModelSpec.key this case runs against
    label: str           # human-readable case name
    image: str           # image path, relative to the repo root
    ground_truth: str    # ground-truth transcript path, relative to the repo root
    hf_cer: float        # upstream HF model's CER on this image
    hf_chrf: float       # upstream HF model's chrF on this image
    cer_tol: float       # allowed CER slack above the HF reference
    chrf_tol: float      # allowed chrF slack below the HF reference

    @property
    def cer_max(self) -> float:
        return self.hf_cer + self.cer_tol

    @property
    def chrf_min(self) -> float:
        return self.hf_chrf - self.chrf_tol


MODELS = {
    "v1": ModelSpec(
        key="v1", label="DeepSeek-OCR",
        model_arg="--llama-model", mmproj_arg="--mmproj",
        model_default="gguf_models/deepseek-ai/deepseek-ocr-bf16.gguf",
        mmproj_default="gguf_models/deepseek-ai/mmproj-deepseek-ocr-bf16.gguf",
    ),
    "v2": ModelSpec(
        key="v2", label="DeepSeek-OCR-2",
        model_arg="--llama-model-2", mmproj_arg="--mmproj-2",
        model_default="gguf_models/deepseek-ai/deepseek-ocr-2-bf16.gguf",
        mmproj_default="gguf_models/deepseek-ai/mmproj-deepseek-ocr-2-bf16.gguf",
    ),
}

CASES = [
    TestCase(
        model_key="v1", label="single-view scan",
        image="tools/mtmd/test-1.jpeg",
        ground_truth="tools/mtmd/tests/test-1-ground-truth.txt",
        # deepseek-ai/DeepSeek-OCR (greedy) on test-1.jpeg vs test-1-ground-truth.txt.
        # llama.cpp scores better than this (CER ~0.24), so the gate has margin.
        hf_cer=0.3030, hf_chrf=67.52, cer_tol=0.02, chrf_tol=2.0,
    ),
    TestCase(
        model_key="v2", label="single-view scan",
        image="tools/mtmd/test-1.jpeg",
        ground_truth="tools/mtmd/tests/test-1-ground-truth.txt",
        # deepseek-ai/DeepSeek-OCR-2 on test-1.jpeg. The image is 640x488 -- below
        # the 768 tiling threshold -- so both the HF model and llama.cpp take the
        # single 1024 global-view path. Both fail this low-quality full-page scan:
        # they read the headlines but cannot resolve the small body text and
        # hallucinate it. The HF reference decodes with no_repeat_ngram_size;
        # run_mtmd_cli matches that with the DRY sampler, so the two are compared
        # on equal footing. chrF is the load-bearing gate here -- it sits at ~34
        # when the hallucinated body does not loop and craters to ~24 if it does.
        # llama.cpp currently scores CER ~0.77 / chrF ~34.
        hf_cer=0.6894, hf_chrf=34.60, cer_tol=0.12, chrf_tol=8.0,
    ),
    TestCase(
        model_key="v2", label="multi-tile (dynamic resolution)",
        image="tools/mtmd/tests/test-1-positive.png",
        ground_truth="tools/mtmd/tests/test-1-ground-truth.txt",
        # deepseek-ai/DeepSeek-OCR-2 on test-1-positive.png, a 429x806 crop of the
        # same article. At 806 px tall it crosses the 768 threshold, so HF and
        # llama.cpp both take the multi-tile path: dynamic_preprocess picks a (1,2)
        # grid -> 2 local 768 tiles + 1 global 1024 view = 545 image tokens. This
        # is the regression guard for the tiling preprocessing -- a broken tile
        # path craters the score (cf. the 0.77 CER on the un-tiled low-res scan).
        # The crop is high quality, so both models score near-perfect: HF
        # CER 0.0236 / chrF 97.05, llama.cpp CER ~0.017 / chrF ~96.8.
        hf_cer=0.0236, hf_chrf=97.05, cer_tol=0.03, chrf_tol=3.0,
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


def evaluate(case: "TestCase", title: str, expected: str, ocr_out: str) -> bool:
    expected = normalize_text(expected)
    ocr_out = normalize_text(ocr_out)
    aligned = locally_align(expected, ocr_out)

    logger.debug(f"\n--- expected (normalized) ---\n{expected}")
    logger.debug(f"\n--- OCR output (normalized) ---\n{ocr_out}")
    logger.debug(f"\n--- aligned span ---\n{aligned}")

    cer = compute_cer(expected, aligned)
    chrf = compute_chrf(expected, aligned)

    cer_pass = cer <= case.cer_max
    chrf_pass = chrf >= case.chrf_min
    passed = cer_pass and chrf_pass

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"{title}: Free OCR evaluation")
    logger.info("=" * 60)
    logger.info(f"  CER               {cer:>7.4f}    (HF {case.hf_cer:.4f}, <= {case.cer_max:>7.4f}  -> {verdict(cer_pass)})")
    logger.info(f"  chrF (0-100)      {chrf:>7.2f}    (HF {case.hf_chrf:.2f}, >= {case.chrf_min:>7.2f}  -> {verdict(chrf_pass)})")
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
    for spec in MODELS.values():
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

    repo_root = Path(__file__).resolve().parents[3]  # tests -> mtmd -> tools -> repo root
    binary = resolve_path(args.llama_bin, repo_root)

    if not binary.exists():
        logger.error(f"Error: binary not found: {binary}")
        return 1

    logger.info("=" * 60)
    logger.info("DeepSeek-OCR: llama.cpp vs HF parity check")
    logger.info("=" * 60)

    results: dict[str, bool] = {}
    for case in CASES:
        model_spec = MODELS[case.model_key]
        title = f"{model_spec.label} -- {case.label}"

        logger.info("")
        logger.info("#" * 60)
        logger.info(f"# {title}")
        logger.info("#" * 60)

        model = resolve_path(getattr(args, arg_dest(model_spec.model_arg)), repo_root)
        mmproj = resolve_path(getattr(args, arg_dest(model_spec.mmproj_arg)), repo_root)
        image = resolve_path(case.image, repo_root)
        ground_truth = resolve_path(case.ground_truth, repo_root)

        missing = [(lbl, p) for lbl, p in [("model", model), ("mmproj", mmproj),
                                           ("image", image), ("ground-truth", ground_truth)]
                   if not p.exists()]
        if missing:
            for lbl, p in missing:
                logger.error(f"  Error: {lbl} not found: {p}")
            results[title] = False
            continue

        expected = read_expected_text(ground_truth)
        logger.info(f"  Image: {case.image}")
        logger.info(f"  Expected text: {len(expected)} chars")
        logger.info("  Running llama.cpp 'Free OCR'")
        try:
            ocr_out = run_mtmd_cli(model, mmproj, image, binary)
        except RuntimeError as e:
            logger.error(f"  Error: {e}")
            results[title] = False
            continue

        results[title] = evaluate(case, title, expected, ocr_out)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for title, ok in results.items():
        logger.info(f"  {title:<48} {verdict(ok)}")
    all_passed = all(results.values())
    logger.info("")
    logger.info(f"  Overall: {verdict(all_passed)}")
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
