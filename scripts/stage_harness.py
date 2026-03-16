#!/usr/bin/env python3
"""Stage-by-stage CoreML comparison harness for Kokoro TTS.

Isolates each model stage into a standalone module, converts to CoreML,
and compares PyTorch vs CoreML outputs. Stages don't cascade — each
uses PyTorch-computed intermediates as input.

Usage:
    .venv/bin/python scripts/stage_harness.py
    .venv/bin/python scripts/stage_harness.py --stage 1    # run single stage
    .venv/bin/python scripts/stage_harness.py --json        # machine-readable output
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Patches (must happen before model load)
# ---------------------------------------------------------------------------
nn.utils.rnn.pack_padded_sequence = lambda x, lengths, **kw: x
nn.utils.rnn.pad_packed_sequence = lambda x, **kw: (x, None)

import coremltools as ct
from coreml_ops import register_missing_torch_ops

register_missing_torch_ops()

sys.path.insert(0, os.path.dirname(__file__))
from export_coreml import patch_sinegen_for_export, SAMPLES_PER_FRAME, SineGen
from export_coreml import GeneratorFrontEnd, GeneratorBackEnd, DecoderBackEnd

# ---------------------------------------------------------------------------
# Stage wrapper modules
# ---------------------------------------------------------------------------

class Stage1_BERT(nn.Module):
    """BERT + BertEncoder: input_ids → d_en"""
    def __init__(self, model):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder

    def forward(self, input_ids, attention_mask):
        bert_dur = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        return d_en


class Stage2_Duration(nn.Module):
    """Duration prediction: d_en, style → pred_dur"""
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.predictor.text_encoder
        self.lstm = model.predictor.lstm
        self.duration_proj = model.predictor.duration_proj

    def forward(self, d_en, s, input_lengths, text_mask, speed, attention_mask):
        d = self.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.lstm(d)
        duration = self.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed[0]
        pred_dur = torch.round(duration).clamp(min=1).long()
        pred_dur = pred_dur * attention_mask.long()
        return pred_dur, d


class Stage3_Alignment(nn.Module):
    """Alignment matrix: pred_dur → pred_aln_trg (pure math)"""
    def __init__(self, max_frames):
        super().__init__()
        self.max_frames = max_frames

    def forward(self, pred_dur):
        cumsum = torch.cumsum(pred_dur, dim=-1)
        frame_indices = torch.arange(
            self.max_frames, device=pred_dur.device
        ).unsqueeze(0)
        starts = F.pad(cumsum[:, :-1], (1, 0))
        pred_aln_trg = (
            (frame_indices.unsqueeze(1) >= starts.unsqueeze(2)) &
            (frame_indices.unsqueeze(1) < cumsum.unsqueeze(2))
        ).float()
        # total_frames derivable from pred_dur: cumsum[0, -1]
        # Return as 1D tensor to avoid scalar output ordering issues in CoreML
        total_frames = cumsum[:, -1:].float()  # [1, 1]
        return pred_aln_trg, total_frames


class Stage4_F0N(nn.Module):
    """F0/N prediction: d, pred_aln_trg, style → F0, N"""
    def __init__(self, model):
        super().__init__()
        self.F0Ntrain = model.predictor.F0Ntrain
        # Bind the shared LSTM and F0/N modules
        self.shared = model.predictor.shared
        self.F0_blocks = model.predictor.F0
        self.N_blocks = model.predictor.N
        self.F0_proj = model.predictor.F0_proj
        self.N_proj = model.predictor.N_proj

    def forward(self, d, pred_aln_trg, s):
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.F0Ntrain(en, s)
        return F0_pred, N_pred


class Stage5_TextEncoder(nn.Module):
    """Text encoding + alignment: input_ids → asr"""
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(self, input_ids, input_lengths, text_mask, pred_aln_trg):
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        return asr


class Stage6_Decoder(nn.Module):
    """Full decoder: asr, F0, N, style → audio"""
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr, F0_pred, N_pred, s_content):
        audio = self.decoder(asr, F0_pred, N_pred, s_content)
        return audio


class Stage7_Generator(nn.Module):
    """Generator only (inside decoder): x, style, F0_curve → audio"""
    def __init__(self, model):
        super().__init__()
        self.generator = model.decoder.generator

    def forward(self, x, s, f0_curve):
        return self.generator(x, s, f0_curve)


class SplitA_Predictor(nn.Module):
    """Split model A: BERT + Duration + Alignment + F0/N + TextEncoder (stages 1-5)."""
    def __init__(self, model, max_frames):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.predictor = model.predictor
        self.text_encoder = model.text_encoder
        self.max_frames = max_frames

    def forward(self, input_ids, attention_mask, ref_s, speed):
        input_lengths = attention_mask.sum(dim=1).long()
        text_mask = (attention_mask == 0)

        bert_dur = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, 128:]

        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed[0]
        pred_dur = torch.round(duration).clamp(min=1).long()
        pred_dur = pred_dur * attention_mask.long()

        cumsum = torch.cumsum(pred_dur, dim=-1)
        total_frames = cumsum[:, -1:]  # [1, 1]
        frame_indices = torch.arange(
            self.max_frames, device=input_ids.device).unsqueeze(0)
        starts = F.pad(cumsum[:, :-1], (1, 0))
        pred_aln_trg = (
            (frame_indices.unsqueeze(1) >= starts.unsqueeze(2)) &
            (frame_indices.unsqueeze(1) < cumsum.unsqueeze(2))
        ).float()

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        return asr, F0_pred, N_pred, total_frames


class SplitB_Decoder(nn.Module):
    """Split model B: Decoder (stage 6)."""
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr, F0_pred, N_pred, s_content):
        return self.decoder(asr, F0_pred, N_pred, s_content)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def compare_tensors(py_out, coreml_out, active_length=None):
    """Compare PyTorch and CoreML output tensors.

    If active_length is set, compare only the first active_length samples
    (ignores garbage tail from fixed-size audio output).
    """
    py_np = py_out.numpy().flatten() if isinstance(py_out, torch.Tensor) else py_out.flatten()
    cml_np = coreml_out.flatten()
    ml = min(len(py_np), len(cml_np))
    if active_length is not None:
        ml = min(ml, active_length)
    if ml == 0:
        return {"corr": 0, "mse": 0, "max_diff": 0, "shape_match": False}

    py_np = py_np[:ml].astype(np.float64)
    cml_np = cml_np[:ml].astype(np.float64)

    mse = float(np.mean((py_np - cml_np) ** 2))
    max_diff = float(np.max(np.abs(py_np - cml_np)))

    if np.std(py_np) < 1e-10 or np.std(cml_np) < 1e-10:
        corr = 1.0 if mse < 1e-10 else 0.0
    else:
        corr = float(np.corrcoef(py_np, cml_np)[0, 1])

    return {"corr": corr, "mse": mse, "max_diff": max_diff, "len": ml}


def test_stage(name, module, example_inputs, input_specs, output_names=None, active_length=None):
    """Trace, convert, and compare a single stage.

    Returns dict with metrics or error info.
    """
    result = {"name": name, "status": "unknown"}
    module.eval()

    # PyTorch forward
    try:
        with torch.no_grad():
            py_outputs = module(*example_inputs)
        if not isinstance(py_outputs, tuple):
            py_outputs = (py_outputs,)
        result["py_shapes"] = [list(o.shape) for o in py_outputs]
    except Exception as e:
        result["status"] = "pytorch_error"
        result["error"] = str(e)
        return result, None

    # Trace
    try:
        t0 = time.time()
        with torch.no_grad():
            traced = torch.jit.trace(module, example_inputs, check_trace=False)
        result["trace_time"] = round(time.time() - t0, 2)
    except Exception as e:
        result["status"] = "trace_error"
        result["error"] = str(e)
        return result, py_outputs

    # Convert
    try:
        t0 = time.time()
        mlmodel = ct.convert(
            traced,
            inputs=input_specs,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT32,
            convert_to="mlprogram",
        )
        result["convert_time"] = round(time.time() - t0, 2)
    except Exception as e:
        result["status"] = "convert_error"
        result["error"] = str(e)
        return result, py_outputs

    # Build feed dict
    feed = {}
    for spec, tensor in zip(input_specs, example_inputs):
        feed[spec.name] = tensor.numpy().astype(np.float32) \
            if tensor.dtype in (torch.float32, torch.float64) \
            else tensor.numpy().astype(np.int32)

    # Save to temp file so we can reload with different compute units
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "model.mlpackage")
        mlmodel.save(tmp_path)

        WARMUP_RUNS = 2
        TIMED_RUNS = 3

        # Run on ALL (ANE-eligible)
        ane_out = None
        try:
            ml_all = ct.models.MLModel(tmp_path, compute_units=ct.ComputeUnit.ALL)
            # Cold run
            t0 = time.time()
            ane_out = ml_all.predict(feed)
            result["predict_time_ane_cold"] = round(time.time() - t0, 3)
            # Warm up
            for _ in range(WARMUP_RUNS):
                ml_all.predict(feed)
            # Timed warm runs
            t0 = time.time()
            for _ in range(TIMED_RUNS):
                ane_out = ml_all.predict(feed)
            result["predict_time_ane"] = round((time.time() - t0) / TIMED_RUNS, 3)
            result["ane_ok"] = True
        except Exception:
            result["ane_ok"] = False

        # Run on CPU_ONLY
        cpu_out = None
        try:
            ml_cpu = ct.models.MLModel(tmp_path, compute_units=ct.ComputeUnit.CPU_ONLY)
            # Cold run
            t0 = time.time()
            cpu_out = ml_cpu.predict(feed)
            result["predict_time_cpu_cold"] = round(time.time() - t0, 3)
            # Warm up
            for _ in range(WARMUP_RUNS):
                ml_cpu.predict(feed)
            # Timed warm runs
            t0 = time.time()
            for _ in range(TIMED_RUNS):
                cpu_out = ml_cpu.predict(feed)
            result["predict_time_cpu"] = round((time.time() - t0) / TIMED_RUNS, 3)
            result["cpu_ok"] = True
        except Exception:
            result["cpu_ok"] = False

    if ane_out is None and cpu_out is None:
        result["status"] = "predict_error"
        result["error"] = "both ANE and CPU failed"
        return result, py_outputs

    def _match_and_compare(py_outputs, coreml_out, output_names, active_length=None):
        """Match CoreML outputs to PyTorch outputs by shape, return comparisons."""
        coreml_items = list(coreml_out.items())
        comparisons = []
        for i, py_out in enumerate(py_outputs):
            py_shape = tuple(py_out.shape)
            oname = output_names[i] if output_names and i < len(output_names) else f"out_{i}"
            matched = None
            for cml_name, cml_val in coreml_items:
                if tuple(cml_val.shape) == py_shape:
                    matched = (cml_name, cml_val)
                    coreml_items.remove((cml_name, cml_val))
                    break
            if matched is None:
                py_size = py_out.numel()
                for cml_name, cml_val in coreml_items:
                    if cml_val.size == py_size:
                        matched = (cml_name, cml_val)
                        coreml_items.remove((cml_name, cml_val))
                        break
            if matched:
                # Use active_length for first output (audio) only
                al = active_length if i == 0 else None
                comp = compare_tensors(py_out, matched[1], active_length=al)
                comp["name"] = oname
                comparisons.append(comp)
            else:
                comparisons.append({"name": oname, "corr": 0, "mse": 0, "max_diff": 0, "error": "no_shape_match"})
        return comparisons

    # PyTorch vs CPU
    if cpu_out is not None:
        cpu_comps = _match_and_compare(py_outputs, cpu_out, output_names, active_length=active_length)
        result["cpu_corr"] = cpu_comps[0]["corr"] if cpu_comps else 0
        result["cpu_comparisons"] = cpu_comps

    # PyTorch vs ANE
    if ane_out is not None:
        ane_comps = _match_and_compare(py_outputs, ane_out, output_names, active_length=active_length)
        result["ane_corr"] = ane_comps[0]["corr"] if ane_comps else 0
        result["comparisons"] = ane_comps  # primary comparisons = ANE
    elif cpu_out is not None:
        result["comparisons"] = result.get("cpu_comparisons", [])

    result["status"] = "ok"
    result["corr"] = result.get("ane_corr", result.get("cpu_corr", 0))

    return result, py_outputs


def test_pipeline_stage(name, frontend_module, backend_module,
                        frontend_inputs, frontend_specs,
                        backend_extra_inputs, backend_extra_specs,
                        py_reference_output, output_names=None,
                        active_length=None):
    """Test a two-model pipeline: frontend on CPU_ONLY, backend on ALL.

    Correlation is measured against py_reference_output.
    """
    result = {"name": name, "status": "unknown"}
    frontend_module.eval()
    backend_module.eval()

    # Trace and convert frontend (CPU-only)
    try:
        with torch.no_grad():
            fe_traced = torch.jit.trace(frontend_module, frontend_inputs, check_trace=False)
        fe_model = ct.convert(
            fe_traced, inputs=frontend_specs,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT32,
            convert_to="mlprogram",
        )
    except Exception as e:
        result["status"] = "frontend_convert_error"
        result["error"] = str(e)
        return result

    # Trace and convert backend (ANE-eligible)
    try:
        with torch.no_grad():
            fe_py_out = frontend_module(*frontend_inputs)
        backend_inputs = backend_extra_inputs + (fe_py_out,)
        backend_specs_full = backend_extra_specs + [
            ct.TensorType(name="har", shape=fe_py_out.shape, dtype=np.float32)]

        with torch.no_grad():
            be_traced = torch.jit.trace(backend_module, backend_inputs, check_trace=False)
        be_model = ct.convert(
            be_traced, inputs=backend_specs_full,
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT32,
            convert_to="mlprogram",
        )
    except Exception as e:
        result["status"] = "backend_convert_error"
        result["error"] = str(e)
        return result

    # Build feed dicts
    fe_feed = {}
    for spec, tensor in zip(frontend_specs, frontend_inputs):
        fe_feed[spec.name] = tensor.numpy().astype(np.float32) \
            if tensor.dtype in (torch.float32, torch.float64) \
            else tensor.numpy().astype(np.int32)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        fe_path = os.path.join(tmpdir, "frontend.mlpackage")
        be_path = os.path.join(tmpdir, "backend.mlpackage")
        fe_model.save(fe_path)
        be_model.save(be_path)

        WARMUP_RUNS = 2
        TIMED_RUNS = 3

        # Load frontend CPU-only, backend ALL (ANE)
        ml_fe = ct.models.MLModel(fe_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        ml_be = ct.models.MLModel(be_path, compute_units=ct.ComputeUnit.ALL)

        # Build backend feed (extra inputs + frontend output)
        def _run_pipeline():
            fe_out = ml_fe.predict(fe_feed)
            har_val = list(fe_out.values())[0]  # single output
            be_feed = {}
            for spec, tensor in zip(backend_extra_specs, backend_extra_inputs):
                be_feed[spec.name] = tensor.numpy().astype(np.float32)
            be_feed["har"] = har_val.astype(np.float32)
            return ml_be.predict(be_feed)

        # Cold run
        t0 = time.time()
        ane_out = _run_pipeline()
        result["predict_time_ane_cold"] = round(time.time() - t0, 3)

        # Warm up
        for _ in range(WARMUP_RUNS):
            _run_pipeline()

        # Timed runs
        t0 = time.time()
        for _ in range(TIMED_RUNS):
            ane_out = _run_pipeline()
        result["predict_time_ane"] = round((time.time() - t0) / TIMED_RUNS, 3)
        result["ane_ok"] = True

        # Also time CPU-only backend for comparison
        ml_be_cpu = ct.models.MLModel(be_path, compute_units=ct.ComputeUnit.CPU_ONLY)

        def _run_pipeline_cpu():
            fe_out = ml_fe.predict(fe_feed)
            har_val = list(fe_out.values())[0]
            be_feed = {}
            for spec, tensor in zip(backend_extra_specs, backend_extra_inputs):
                be_feed[spec.name] = tensor.numpy().astype(np.float32)
            be_feed["har"] = har_val.astype(np.float32)
            return ml_be_cpu.predict(be_feed)

        t0 = time.time()
        cpu_out = _run_pipeline_cpu()
        result["predict_time_cpu_cold"] = round(time.time() - t0, 3)
        for _ in range(WARMUP_RUNS):
            _run_pipeline_cpu()
        t0 = time.time()
        for _ in range(TIMED_RUNS):
            cpu_out = _run_pipeline_cpu()
        result["predict_time_cpu"] = round((time.time() - t0) / TIMED_RUNS, 3)
        result["cpu_ok"] = True

    # Compare against PyTorch reference
    py_outputs = (py_reference_output,) if not isinstance(py_reference_output, tuple) else py_reference_output

    if ane_out is not None:
        ane_comps = _match_and_compare(py_outputs, ane_out, output_names, active_length=active_length)
        result["ane_corr"] = ane_comps[0]["corr"] if ane_comps else 0
        result["comparisons"] = ane_comps

    if cpu_out is not None:
        cpu_comps = _match_and_compare(py_outputs, cpu_out, output_names, active_length=active_length)
        result["cpu_corr"] = cpu_comps[0]["corr"] if cpu_comps else 0
        result["cpu_comparisons"] = cpu_comps

    result["status"] = "ok"
    result["corr"] = result.get("ane_corr", result.get("cpu_corr", 0))
    return result


def _match_and_compare(py_outputs, coreml_out, output_names, active_length=None):
    """Match CoreML outputs to PyTorch outputs by shape, return comparisons."""
    coreml_items = list(coreml_out.items())
    comparisons = []
    for i, py_out in enumerate(py_outputs):
        py_shape = tuple(py_out.shape)
        oname = output_names[i] if output_names and i < len(output_names) else f"out_{i}"
        matched = None
        for cml_name, cml_val in coreml_items:
            if tuple(cml_val.shape) == py_shape:
                matched = (cml_name, cml_val)
                coreml_items.remove((cml_name, cml_val))
                break
        if matched is None:
            py_size = py_out.numel()
            for cml_name, cml_val in coreml_items:
                if cml_val.size == py_size:
                    matched = (cml_name, cml_val)
                    coreml_items.remove((cml_name, cml_val))
                    break
        if matched:
            al = active_length if i == 0 else None
            comp = compare_tensors(py_out, matched[1], active_length=al)
            comp["name"] = oname
            comparisons.append(comp)
        else:
            comparisons.append({"name": oname, "corr": 0, "mse": 0, "max_diff": 0, "error": "no_shape_match"})
    return comparisons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model():
    from kokoro import KPipeline
    from export_coreml import CustomSTFT

    pipeline = KPipeline(lang_code="a")
    model = pipeline.model
    model.eval()

    gen = model.decoder.generator
    gen.stft = CustomSTFT(
        filter_length=gen.stft.filter_length,
        hop_length=gen.stft.hop_length,
        win_length=gen.stft.win_length,
    )

    set_phases_fn = patch_sinegen_for_export(model)
    return pipeline, model, set_phases_fn


def run_pytorch_pipeline(model, set_phases_fn, token_ids, ref_s, max_tokens, max_frames):
    """Run full pipeline in PyTorch, capturing all intermediates."""
    intermediates = {}

    input_ids = torch.zeros(1, max_tokens, dtype=torch.int64)
    seq_len = min(len(token_ids), max_tokens)
    input_ids[0, :seq_len] = torch.tensor(token_ids[:seq_len])
    attention_mask = torch.zeros(1, max_tokens, dtype=torch.int64)
    attention_mask[0, :seq_len] = 1
    speed = torch.tensor([1.0])
    phases = torch.rand(1, 9) * 2 * torch.pi

    set_phases_fn(model.decoder, phases)

    with torch.no_grad():
        input_lengths = attention_mask.sum(dim=1).long()
        text_mask = (attention_mask == 0)

        # Stage 1: BERT
        bert_dur = model.bert(input_ids, attention_mask=attention_mask)
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        intermediates["d_en"] = d_en

        s = ref_s[:, 128:]
        s_content = ref_s[:, :128]
        intermediates["s"] = s
        intermediates["s_content"] = s_content

        # Stage 2: Duration
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        intermediates["d"] = d
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed[0]
        pred_dur = torch.round(duration).clamp(min=1).long()
        pred_dur = pred_dur * attention_mask.long()
        intermediates["pred_dur"] = pred_dur

        # Stage 3: Alignment
        cumsum = torch.cumsum(pred_dur, dim=-1)
        total_frames = cumsum[0, -1]
        frame_indices = torch.arange(max_frames, device=input_ids.device).unsqueeze(0)
        starts = F.pad(cumsum[:, :-1], (1, 0))
        pred_aln_trg = (
            (frame_indices.unsqueeze(1) >= starts.unsqueeze(2)) &
            (frame_indices.unsqueeze(1) < cumsum.unsqueeze(2))
        ).float()
        intermediates["pred_aln_trg"] = pred_aln_trg
        intermediates["total_frames"] = total_frames

        # Stage 4: F0/N
        en = d.transpose(-1, -2) @ pred_aln_trg
        intermediates["en"] = en
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        intermediates["F0_pred"] = F0_pred
        intermediates["N_pred"] = N_pred

        # Stage 5: Text encoder
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        intermediates["t_en"] = t_en
        asr = t_en @ pred_aln_trg
        intermediates["asr"] = asr

        # Stage 6: Decoder
        audio = model.decoder(asr, F0_pred, N_pred, s_content)
        intermediates["audio"] = audio

    # Store inputs for stage testing
    intermediates["input_ids"] = input_ids
    intermediates["attention_mask"] = attention_mask
    intermediates["input_lengths"] = input_lengths
    intermediates["text_mask"] = text_mask
    intermediates["speed"] = speed
    intermediates["phases"] = phases
    intermediates["ref_s"] = ref_s

    return intermediates


def _build_stage_configs(model, im, max_frames):
    """Build stage test configurations. Returns list of (stage_num, name, module, inputs, specs, output_names)."""
    stages = []

    stages.append((1, "1_bert", Stage1_BERT(model),
        (im["input_ids"], im["attention_mask"]),
        [ct.TensorType(name="input_ids", shape=im["input_ids"].shape, dtype=np.int32),
         ct.TensorType(name="attention_mask", shape=im["attention_mask"].shape, dtype=np.int32)],
        ["d_en"]))

    stages.append((2, "2_duration", Stage2_Duration(model),
        (im["d_en"], im["s"], im["input_lengths"], im["text_mask"], im["speed"], im["attention_mask"]),
        [ct.TensorType(name="d_en", shape=im["d_en"].shape, dtype=np.float32),
         ct.TensorType(name="s", shape=im["s"].shape, dtype=np.float32),
         ct.TensorType(name="input_lengths", shape=im["input_lengths"].shape, dtype=np.int32),
         ct.TensorType(name="text_mask", shape=im["text_mask"].shape, dtype=np.int32),
         ct.TensorType(name="speed", shape=im["speed"].shape, dtype=np.float32),
         ct.TensorType(name="attention_mask", shape=im["attention_mask"].shape, dtype=np.int32)],
        ["pred_dur", "d"]))

    stages.append((3, "3_alignment", Stage3_Alignment(max_frames),
        (im["pred_dur"],),
        [ct.TensorType(name="pred_dur", shape=im["pred_dur"].shape, dtype=np.int32)],
        ["pred_aln_trg", "total_frames"]))

    stages.append((4, "4_f0n", Stage4_F0N(model),
        (im["d"], im["pred_aln_trg"], im["s"]),
        [ct.TensorType(name="d", shape=im["d"].shape, dtype=np.float32),
         ct.TensorType(name="pred_aln_trg", shape=im["pred_aln_trg"].shape, dtype=np.float32),
         ct.TensorType(name="s", shape=im["s"].shape, dtype=np.float32)],
        ["F0_pred", "N_pred"]))

    stages.append((5, "5_text_enc", Stage5_TextEncoder(model),
        (im["input_ids"], im["input_lengths"], im["text_mask"], im["pred_aln_trg"]),
        [ct.TensorType(name="input_ids", shape=im["input_ids"].shape, dtype=np.int32),
         ct.TensorType(name="input_lengths", shape=im["input_lengths"].shape, dtype=np.int32),
         ct.TensorType(name="text_mask", shape=im["text_mask"].shape, dtype=np.int32),
         ct.TensorType(name="pred_aln_trg", shape=im["pred_aln_trg"].shape, dtype=np.float32)],
        ["asr"]))

    stages.append((6, "6_decoder", Stage6_Decoder(model),
        (im["asr"], im["F0_pred"], im["N_pred"], im["s_content"]),
        [ct.TensorType(name="asr", shape=im["asr"].shape, dtype=np.float32),
         ct.TensorType(name="F0_pred", shape=im["F0_pred"].shape, dtype=np.float32),
         ct.TensorType(name="N_pred", shape=im["N_pred"].shape, dtype=np.float32),
         ct.TensorType(name="s_content", shape=im["s_content"].shape, dtype=np.float32)],
        ["audio"]))

    # Stage 7 needs the decoder's pre-generator output
    try:
        with torch.no_grad():
            dec = model.decoder
            F0 = dec.F0_conv(im["F0_pred"].unsqueeze(1))
            N = dec.N_conv(im["N_pred"].unsqueeze(1))
            x = torch.cat([im["asr"], F0, N], axis=1)
            x = dec.encode(x, im["s_content"])
            asr_res = dec.asr_res(im["asr"])
            res = True
            for block in dec.decode:
                if res:
                    x = torch.cat([x, asr_res, F0, N], axis=1)
                x = block(x, im["s_content"])
                if block.upsample_type != "none":
                    res = False
            gen_input = x

        stages.append((7, "7_generator", Stage7_Generator(model),
            (gen_input, im["s_content"], im["F0_pred"]),
            [ct.TensorType(name="x", shape=gen_input.shape, dtype=np.float32),
             ct.TensorType(name="s", shape=im["s_content"].shape, dtype=np.float32),
             ct.TensorType(name="f0_curve", shape=im["F0_pred"].shape, dtype=np.float32)],
            ["audio"]))
    except Exception as e:
        stages.append((7, "7_generator", None, None, None, None))

    # ---- Split models ----
    # Split A: stages 1-5 combined (predictor)
    stages.append((8, "split_A", SplitA_Predictor(model, max_frames),
        (im["input_ids"], im["attention_mask"], im["ref_s"], im["speed"]),
        [ct.TensorType(name="input_ids", shape=im["input_ids"].shape, dtype=np.int32),
         ct.TensorType(name="attention_mask", shape=im["attention_mask"].shape, dtype=np.int32),
         ct.TensorType(name="ref_s", shape=im["ref_s"].shape, dtype=np.float32),
         ct.TensorType(name="speed", shape=im["speed"].shape, dtype=np.float32)],
        ["asr", "F0_pred", "N_pred", "total_frames"]))

    # Split B: stage 6 (decoder)
    stages.append((9, "split_B", SplitB_Decoder(model),
        (im["asr"], im["F0_pred"], im["N_pred"], im["s_content"]),
        [ct.TensorType(name="asr", shape=im["asr"].shape, dtype=np.float32),
         ct.TensorType(name="F0_pred", shape=im["F0_pred"].shape, dtype=np.float32),
         ct.TensorType(name="N_pred", shape=im["N_pred"].shape, dtype=np.float32),
         ct.TensorType(name="s_content", shape=im["s_content"].shape, dtype=np.float32)],
        ["audio"]))

    return stages


def run_all_stages(model, intermediates, max_frames, only_stage=None):
    """Run all stage tests (or a single stage), return results list."""
    configs = _build_stage_configs(model, intermediates, max_frames)
    results = []

    # Active audio length for trimming garbage tail in correlation
    active_audio = int(intermediates["total_frames"].item()) * SAMPLES_PER_FRAME
    # Stages that output audio (6=decoder, 7=generator, 9=split_B)
    audio_stages = {6, 7, 9}

    for stage_num, name, module, inputs, specs, output_names in configs:
        if only_stage is not None and only_stage != stage_num:
            continue

        if module is None:
            results.append({"name": name, "status": "setup_error", "error": "failed to build"})
            continue

        al = active_audio if stage_num in audio_stages else None
        r, _ = test_stage(name, module, inputs, specs, output_names=output_names, active_length=al)
        results.append(r)

    # Pipeline split stages (stage 10 = generator pipeline)
    if only_stage is None or only_stage == 10:
        try:
            gen = model.decoder.generator
            fe = GeneratorFrontEnd(gen)
            be = GeneratorBackEnd(gen)

            # Compute pre-generator input (same as stage 7 setup)
            with torch.no_grad():
                dec = model.decoder
                F0 = dec.F0_conv(intermediates["F0_pred"].unsqueeze(1))
                N = dec.N_conv(intermediates["N_pred"].unsqueeze(1))
                x = torch.cat([intermediates["asr"], F0, N], axis=1)
                x = dec.encode(x, intermediates["s_content"])
                asr_res = dec.asr_res(intermediates["asr"])
                res = True
                for block in dec.decode:
                    if res:
                        x = torch.cat([x, asr_res, F0, N], axis=1)
                    x = block(x, intermediates["s_content"])
                    if block.upsample_type != "none":
                        res = False
                gen_input = x

            r = test_pipeline_stage(
                "10_gen_pipe",
                fe, be,
                frontend_inputs=(intermediates["F0_pred"],),
                frontend_specs=[
                    ct.TensorType(name="f0_curve", shape=intermediates["F0_pred"].shape,
                                  dtype=np.float32)],
                backend_extra_inputs=(gen_input, intermediates["s_content"]),
                backend_extra_specs=[
                    ct.TensorType(name="x", shape=gen_input.shape, dtype=np.float32),
                    ct.TensorType(name="s", shape=intermediates["s_content"].shape,
                                  dtype=np.float32)],
                py_reference_output=intermediates["audio"],
                output_names=["audio"],
                active_length=active_audio,
            )
            results.append(r)
        except Exception as e:
            results.append({"name": "10_gen_pipe", "status": "setup_error",
                           "error": str(e)[:80]})

    # Stage 11: Full decoder pipeline (frontend CPU + decoder backend ANE)
    if only_stage is None or only_stage == 11:
        try:
            gen = model.decoder.generator
            fe = GeneratorFrontEnd(gen)
            be = DecoderBackEnd(model.decoder)

            r = test_pipeline_stage(
                "11_dec_pipe",
                fe, be,
                frontend_inputs=(intermediates["F0_pred"],),
                frontend_specs=[
                    ct.TensorType(name="f0_curve", shape=intermediates["F0_pred"].shape,
                                  dtype=np.float32)],
                backend_extra_inputs=(
                    intermediates["asr"], intermediates["F0_pred"],
                    intermediates["N_pred"], intermediates["s_content"]),
                backend_extra_specs=[
                    ct.TensorType(name="asr", shape=intermediates["asr"].shape, dtype=np.float32),
                    ct.TensorType(name="F0_curve", shape=intermediates["F0_pred"].shape, dtype=np.float32),
                    ct.TensorType(name="N", shape=intermediates["N_pred"].shape, dtype=np.float32),
                    ct.TensorType(name="s", shape=intermediates["s_content"].shape, dtype=np.float32)],
                py_reference_output=intermediates["audio"],
                output_names=["audio"],
                active_length=active_audio,
            )
            results.append(r)
        except Exception as e:
            results.append({"name": "11_dec_pipe", "status": "setup_error",
                           "error": str(e)[:80]})

    return results


def print_results(results):
    """Pretty-print stage results."""
    w = 88
    print(f"\n{'='*w}")
    print(f"{'Stage':<16} {'Status':<8} "
          f"{'CPU Corr':>9} {'Cold':>6} {'Warm':>6}  "
          f"{'ANE Corr':>9} {'Cold':>6} {'Warm':>6}")
    print(f"{'-'*w}")

    for r in results:
        name = r["name"]
        status = r["status"]

        if status == "ok":
            ane_ok = r.get("ane_ok", False)
            cpu_ok = r.get("cpu_ok", False)
            cpu_corr = r.get("cpu_corr", None)
            ane_corr = r.get("ane_corr", None)
            ane_cold = r.get("predict_time_ane_cold", 0)
            ane_warm = r.get("predict_time_ane", 0)
            cpu_cold = r.get("predict_time_cpu_cold", 0)
            cpu_warm = r.get("predict_time_cpu", 0)

            # FAIL if ANE doesn't run
            if not ane_ok:
                flag = "FAIL"
            elif ane_corr is not None and ane_corr > 0.99:
                flag = "PASS"
            elif ane_corr is not None and ane_corr > 0.9:
                flag = "WARN"
            else:
                flag = "FAIL"

            def _fmt_corr(c):
                return f"{c:.4f}" if c is not None else "---"
            def _fmt_time(t, ok):
                return f"{t*1000:.0f}ms" if ok else "FAIL"

            print(f"{name:<16} {flag:<8} "
                  f"{_fmt_corr(cpu_corr):>9} {_fmt_time(cpu_cold, cpu_ok):>6} {_fmt_time(cpu_warm, cpu_ok):>6}  "
                  f"{_fmt_corr(ane_corr):>9} {_fmt_time(ane_cold, ane_ok):>6} {_fmt_time(ane_warm, ane_ok):>6}")

            # Sub-outputs (from ANE comparisons)
            c = r.get("comparisons", [])
            for ci in c[1:]:
                oname = ci.get("name", "?")
                print(f"  └─ {oname:<39}  "
                      f"{ci['corr']:>9.4f}")
        else:
            err = r.get("error", "")[:40]
            print(f"{name:<16} {status:<8} {'':>9} {'':>6} {'':>6}  "
                  f"{'':>9} {'':>6} {'':>6}  {err}")

    print(f"{'='*w}")


TEST_SENTENCES = {
    "short":  "Hello world.",
    "medium": "She sells seashells by the seashore, and the shells she sells are seashells for sure.",
    "long":   "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity.",
}


def _tokenize(text, pipeline, model):
    """Convert text to token IDs with start/end tokens."""
    phonemes, _ = pipeline.g2p(text)
    raw = list(filter(lambda i: i is not None,
        map(lambda p: model.vocab.get(p), phonemes)))
    return [0] + raw + [0]


def _merge_results(all_runs):
    """Merge results across multiple sentences — report worst-case corr, avg timing."""
    merged = {}
    for results in all_runs:
        for r in results:
            name = r["name"]
            if name not in merged:
                merged[name] = {**r, "_runs": 1}
                continue

            m = merged[name]
            m["_runs"] += 1

            if r["status"] != "ok":
                m["status"] = r["status"]
                continue

            # Worst-case correlations
            for key in ("cpu_corr", "ane_corr"):
                if key in r and key in m:
                    if r[key] is not None and m[key] is not None:
                        m[key] = min(m[key], r[key])

            # Average timings
            for key in ("predict_time_cpu", "predict_time_ane",
                        "predict_time_cpu_cold", "predict_time_ane_cold"):
                if key in r and key in m:
                    m[key] = (m[key] * (m["_runs"] - 1) + r[key]) / m["_runs"]

            # Worst-case ANE/CPU ok
            for key in ("ane_ok", "cpu_ok"):
                if key in r:
                    m[key] = m.get(key, True) and r[key]

            # Update overall corr to worst ANE
            m["corr"] = m.get("ane_corr", m.get("cpu_corr", 0))

    return list(merged.values())


MINIMUM_EXPERIMENT_SECONDS = 300  # 5 minutes


def _emit_results(all_runs, sentences, args):
    """Format and print results (text or JSON)."""
    if not args.json:
        for label, results in zip(sentences.keys(), all_runs):
            print(f"\n--- {label} ---")
            print_results(results)

        # Also print worst-case summary if multiple sentences
        if len(all_runs) > 1:
            merged = _merge_results(all_runs)
            print(f"\n--- WORST CASE across {len(sentences)} sentences ---")
            print_results(merged)
    else:
        out = {}
        for label, results in zip(sentences.keys(), all_runs):
            for r in results:
                for c in r.get("comparisons", []):
                    for k in list(c.keys()):
                        if isinstance(c[k], (np.floating, np.integer)):
                            c[k] = float(c[k])
            out[label] = results
        if len(all_runs) > 1:
            merged = _merge_results(all_runs)
            for r in merged:
                r.pop("_runs", None)
            out["worst_case"] = merged
        print(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Stage-by-stage CoreML harness")
    parser.add_argument("--stage", type=int,
                        help="Run single stage (1-7, 8=split_A, 9=split_B)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--text", default=None,
                        help="Single test sentence (overrides multi-sentence)")
    args = parser.parse_args()

    experiment_start = time.time()

    print("Loading model...")
    pipeline, model, set_phases_fn = load_model()

    from export_coreml import BUCKETS
    bucket = BUCKETS["kokoro_21_5s"]
    max_tokens = bucket["max_tokens"]
    max_audio = bucket["max_audio"]
    max_frames = max_audio // SAMPLES_PER_FRAME

    voice_pack = pipeline.load_voice("af_heart")

    # Build sentence list
    if args.text:
        sentences = {"custom": args.text}
    else:
        sentences = TEST_SENTENCES

    all_runs = []
    for label, text in sentences.items():
        token_ids = _tokenize(text, pipeline, model)
        seq_len = min(len(token_ids), max_tokens)

        ref_s = voice_pack[seq_len]
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)

        print(f"\n[{label}] \"{text[:60]}{'...' if len(text)>60 else ''}\" "
              f"({seq_len} tokens)")

        intermediates = run_pytorch_pipeline(
            model, set_phases_fn, token_ids, ref_s, max_tokens, max_frames
        )
        total = int(intermediates["total_frames"].item())
        print(f"  frames: {total}, audio: {total * SAMPLES_PER_FRAME} samples")

        results = run_all_stages(
            model, intermediates, max_frames, only_stage=args.stage
        )
        all_runs.append(results)

    # Enforce minimum experiment time before revealing results
    elapsed = time.time() - experiment_start
    remaining = MINIMUM_EXPERIMENT_SECONDS - elapsed
    if remaining > 0:
        print(f"\n{'='*88}")
        print(f"Experiment complete. Withholding results until minimum experiment")
        print(f"time of {MINIMUM_EXPERIMENT_SECONDS // 60} minutes has elapsed.")
        print(f"Elapsed: {elapsed:.0f}s — waiting {remaining:.0f}s more...")
        print(f"{'='*88}")
        sys.stdout.flush()
        time.sleep(remaining)
        print(f"\nExperiment time reached. Releasing results.\n")

    _emit_results(all_runs, sentences, args)


if __name__ == "__main__":
    main()
