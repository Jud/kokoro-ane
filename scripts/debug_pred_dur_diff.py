#!/usr/bin/env python3
"""Diff predicted token durations between patched-PyTorch and CoreML.

Verifies whether the duration predictor produces stable per-token durations
across pad lengths after the CoreMLPackedBidirLSTM patch lands. If durations
are stable across totals (real_n, 8, 16, 24, 32, 64, 100), the export is now
length-aware. If they drift, the patch is incomplete.

Usage:
    .venv/bin/python scripts/debug_pred_dur_diff.py [--coreml] [--voice bf_lily]

By default runs PyTorch only. Pass --coreml to also load the deployed CoreML
frontend and compare side by side.
"""
import argparse
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from export_coreml import (  # noqa: E402
    KokoroModelA, NUM_HARMONICS, _tokenize, load_kokoro_model,
)
from reference import (  # noqa: E402
    patch_pack_padded_sequence, patch_sinegen_for_export,
)

DEFAULT_FRONTEND = os.path.expanduser(
    "~/Library/Application Support/com.operator.app/models/kokoro/kokoro_frontend.mlmodelc"
)


def make_inputs(token_ids, total_n):
    real_n = len(token_ids)
    ids = np.zeros((1, total_n), dtype=np.int32)
    mask = np.zeros((1, total_n), dtype=np.int32)
    ids[0, :real_n] = np.array(token_ids, dtype=np.int32)
    mask[0, :real_n] = 1
    return ids, mask


def run_torch(frontend, token_ids, total_n, ref_s):
    ids_np, mask_np = make_inputs(token_ids, total_n)
    with torch.no_grad():
        *_, pred_dur = frontend(
            torch.from_numpy(ids_np).long(),
            torch.from_numpy(mask_np).long(),
            ref_s,
            torch.tensor([1.0]),
            torch.zeros(1, NUM_HARMONICS),
        )
    return pred_dur.flatten().cpu().numpy()


def run_coreml(fe, token_ids, total_n, ref_s):
    ids_np, mask_np = make_inputs(token_ids, total_n)
    out = fe.predict({
        "input_ids": ids_np,
        "attention_mask": mask_np,
        "ref_s": ref_s.detach().numpy().astype(np.float32),
        "speed": np.array([1.0], dtype=np.float32),
        "random_phases": np.zeros((1, NUM_HARMONICS), dtype=np.float32),
    })
    return out["pred_dur_clamped"].flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coreml", action="store_true")
    parser.add_argument("--frontend-path", default=DEFAULT_FRONTEND)
    parser.add_argument("--voice", default="bf_lily")
    parser.add_argument(
        "--texts", nargs="+",
        default=["4.", "Beauty.", "Switch", "Hello world."])
    args = parser.parse_args()

    pipeline, model = load_kokoro_model()
    set_phases = patch_sinegen_for_export(model)
    patch_pack_padded_sequence()

    fe_torch = KokoroModelA(model, 100, 267000, set_phases).eval()

    fe_coreml = None
    if args.coreml:
        import coremltools as ct
        if args.frontend_path.endswith(".mlmodelc"):
            fe_coreml = ct.models.CompiledMLModel(
                args.frontend_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        else:
            fe_coreml = ct.models.MLModel(
                args.frontend_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    voice = pipeline.load_voice(args.voice)

    for text in args.texts:
        token_ids = _tokenize(text, pipeline)
        real_n = len(token_ids)
        # Match pipeline.py: pack[len(ps)-1] where ps is the phoneme string
        # (token_ids minus the BOS/EOS bracketing). real_n = len(ps) + 2.
        ref_s = voice[real_n - 3]
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)

        totals = sorted({real_n, 8, 16, 24, 32, 64, 100})
        totals = [n for n in totals if n >= real_n]

        print(f"\n{text!r} real_n={real_n} ids={token_ids}")
        for total_n in totals:
            pt = run_torch(fe_torch, token_ids, total_n, ref_s)
            line = (
                f"  N={total_n:3d}  pt={pt[:real_n].astype(int).tolist()} "
                f"sum={int(pt[:real_n].sum())}"
            )
            if fe_coreml is not None:
                cm = run_coreml(fe_coreml, token_ids, total_n, ref_s)
                line += (
                    f" | cm={cm[:real_n].astype(int).tolist()} "
                    f"sum={int(cm[:real_n].sum())}"
                )
            print(line)


if __name__ == "__main__":
    main()
