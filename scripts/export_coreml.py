#!/usr/bin/env python3
"""Export Kokoro-82M from PyTorch to split CoreML models for kokoro-coreml.

Creates frontend + backend CoreML model pairs per bucket:
  - {bucket}_frontend.mlmodelc  (predictor, CPU_ONLY, ~5ms)
  - {bucket}_backend.mlmodelc   (decoder, ANE-eligible, ~112ms)

Usage:
    .venv/bin/python scripts/export_coreml.py [--output-dir ./models_export]
    .venv/bin/python scripts/export_coreml.py --verify
    .venv/bin/python scripts/export_coreml.py --bucket kokoro_21_5s
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from coreml_ops import register_missing_torch_ops

# Reference implementations — DO NOT MODIFY these imports.
# CustomSTFT, SineGen, patch_sinegen_for_export, and patch_pack_padded_sequence
# define the PyTorch reference. They live in reference.py (read-only).
from reference import (  # noqa: F401 — re-exported for harness/other scripts
    CustomSTFT, SineGen, patch_sinegen_for_export, patch_sinegen_for_production,
    patch_pack_padded_sequence,
)

register_missing_torch_ops()


# ---------------------------------------------------------------------------
# CoreML-compatible STFT override
# ---------------------------------------------------------------------------
# Replace index_put_ (phase[mask] = pi) with torch.where for CoreML export.
# This does NOT change the PyTorch reference — it only affects the traced graph.
def _stft_transform_coreml(self, waveform):
    if self.center:
        pad_len = self.n_fft // 2
        waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
    x = waveform.unsqueeze(1)
    ri = F.conv1d(x, self.weight_forward_fused, bias=None,
                  stride=self.hop_length, padding=0)
    real_out = ri[:, :self.freq_bins, :]
    imag_out = ri[:, self.freq_bins:, :]
    magnitude = torch.sqrt(real_out * real_out + imag_out * imag_out + 1e-14)
    phase = torch.atan2(imag_out, real_out)
    correction_mask = (imag_out == 0) & (real_out < 0)
    phase = torch.where(correction_mask, torch.tensor(torch.pi), phase)
    return magnitude, phase


# ---------------------------------------------------------------------------
# Pipeline split modules
# ---------------------------------------------------------------------------
class GeneratorFrontEnd(nn.Module):
    """SineGen + STFT transform -> har conditioning."""
    def __init__(self, generator):
        super().__init__()
        self.f0_upsamp = generator.f0_upsamp
        self.m_source = generator.m_source
        self.stft = generator.stft

    def forward(self, f0_curve):
        f0 = self.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
        har_source, noi_source, uv = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = self.stft.transform(har_source)
        return torch.cat([har_spec, har_phase], dim=1)


class GeneratorBackEnd(nn.Module):
    """Upsampling cascade + inverse STFT -> audio.
    Takes precomputed har conditioning.

    Uses ANE-compatible replacements:
    - F.leaky_relu → relu decomposition (ANELeakyReLU-style inline)
    - torch.sin (phase) → _ane_sin polynomial approximation
    """
    def __init__(self, generator):
        super().__init__()
        self.num_upsamples = generator.num_upsamples
        self.num_kernels = generator.num_kernels
        self.noise_convs = generator.noise_convs
        self.noise_res = generator.noise_res
        self.ups = generator.ups
        self.resblocks = generator.resblocks
        self.reflection_pad = generator.reflection_pad
        self.conv_post = generator.conv_post
        self.post_n_fft = generator.post_n_fft
        self.stft = generator.stft
        self._leaky_relu = ANELeakyReLU(0.1)
        self._leaky_relu_final = ANELeakyReLU(0.01)  # default negative_slope

    def forward(self, x, s, har):
        for i in range(self.num_upsamples):
            x = self._leaky_relu(x)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        x = self._leaky_relu_final(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = _ane_sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)


class DecoderBackEnd(nn.Module):
    """Full decoder with precomputed har conditioning.
    Decoder preprocessing + GeneratorBackEnd."""
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.encode = decoder.encode
        self.decode = decoder.decode
        self.asr_res = decoder.asr_res
        self.gen_backend = GeneratorBackEnd(decoder.generator)

    def forward(self, asr, F0_curve, N, s, har):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N_out], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        return self.gen_backend(x, s, har)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
BUCKETS = {
    "kokoro_21_5s":  {"max_tokens": 124, "max_audio": 175_800},
    "kokoro_24_10s": {"max_tokens": 242, "max_audio": 240_000},
    "kokoro_25_20s": {"max_tokens": 510, "max_audio": 510_000},
}

SAMPLES_PER_FRAME = 600  # decoder upsample (2x) * generator (10*6*5=300)
NUM_HARMONICS = 9        # fundamental + 8 overtones
S_CONTENT_DIM = 128      # first half of 256-dim style vector (content style)


# ---------------------------------------------------------------------------
# ANE-compatible op replacements
# ---------------------------------------------------------------------------

def _ane_sin(x):
    """ANE-compatible sin approximation using 9th-order Taylor polynomial.

    Wraps input to [-pi, pi] using round (ANE-native), then applies a
    polynomial that closely approximates sin over that range.
    All ops (mul, sub, round, add) are ANE-native.
    """
    TWO_PI = 2.0 * math.pi
    x = x - torch.round(x / TWO_PI) * TWO_PI
    x2 = x * x
    # 9th-order Taylor: x - x³/3! + x⁵/5! - x⁷/7! + x⁹/9!
    return x * (1.0 - x2 * (0.16666666666 - x2 * (0.00833333333 - x2 * (0.000198412698 - x2 * 0.00000275573192))))


def _ane_cos(x):
    """ANE-compatible cos via sin(x + pi/2)."""
    return _ane_sin(x + (math.pi / 2.0))


class ANELeakyReLU(nn.Module):
    """ANE-compatible LeakyReLU using only relu/mul/sub.

    LeakyReLU(x) = relu(x) + alpha * (x - relu(x))
                 = relu(x) - alpha * relu(-x)
    relu and mul are ANE-native; avoids the leakyRelu op which causes
    CPU/GPU fallback.
    """
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.relu(x) + self.negative_slope * (x - F.relu(x))


class ANEInstanceNorm1d(nn.Module):
    """ANE-compatible InstanceNorm1d decomposed into native ops.

    Uses reduceMean, mul, sub, rsqrt, add — all ANE-native.
    Replaces nn.InstanceNorm1d which maps to the instanceNorm MIL op
    that has poor ANE support.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x shape: [B, C, L]
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        var = (x_centered * x_centered).mean(dim=-1, keepdim=True)
        x_norm = x_centered * torch.rsqrt(var + self.eps)
        if self.affine:
            x_norm = x_norm * self.weight.unsqueeze(-1) + self.bias.unsqueeze(-1)
        return x_norm


def patch_ane_ops(model):
    """Apply all ANE-compatibility patches to the backend model.

    Replaces instanceNorm, leakyRelu, sin, and cos with ANE-native
    equivalents via monkey-patching. Must be called after
    patch_sinegen_for_export (which handles Snake pow→mul).
    """
    _patch_instance_norm(model)
    _patch_leaky_relu(model)
    _patch_sin_cos(model)


def _patch_instance_norm(model):
    """Replace all nn.InstanceNorm1d with ANEInstanceNorm1d."""
    def _replace_instance_norm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.InstanceNorm1d):
                ane_norm = ANEInstanceNorm1d(
                    child.num_features,
                    eps=child.eps,
                    affine=child.affine,
                )
                # Copy learned parameters if affine
                if child.affine and child.weight is not None:
                    ane_norm.weight.data.copy_(child.weight.data)
                    ane_norm.bias.data.copy_(child.bias.data)
                setattr(module, name, ane_norm)
            else:
                _replace_instance_norm(child)

    _replace_instance_norm(model)


def _patch_leaky_relu(model):
    """Replace leakyRelu in backend paths with ANE-compatible relu decomposition.

    Two sources of leakyRelu in the backend:
    1. F.leaky_relu() calls in GeneratorBackEnd.forward (patched via method override)
    2. nn.LeakyReLU modules stored as .actv in AdainResBlk1d (patched via module swap)
    """
    from kokoro.istftnet import AdainResBlk1d

    # Replace nn.LeakyReLU module instances in the model tree
    def _replace_leaky_relu_modules(module):
        for name, child in module.named_children():
            if isinstance(child, nn.LeakyReLU):
                setattr(module, name, ANELeakyReLU(child.negative_slope))
            else:
                _replace_leaky_relu_modules(child)

    _replace_leaky_relu_modules(model)


def _patch_sin_cos(model):
    """Replace torch.sin/cos in backend with polynomial approximations.

    Three sources in the backend:
    1. Snake activation sin in AdaINResBlock1 — patched via method override
    2. phase = torch.sin(...) in Generator.forward (line 324 of istftnet.py)
       → patched via GeneratorBackEnd.forward override
    3. cos/sin in CustomSTFT.inverse — patched via method override
    """
    from kokoro.istftnet import AdaINResBlock1

    # 1. Patch Snake activations to use _ane_sin instead of torch.sin
    def _snake_ane_forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2,
            self.alpha1, self.alpha2
        ):
            xt = n1(x, s)
            s1 = _ane_sin(a1 * xt)
            xt = xt + (1.0 / a1) * (s1 * s1)
            xt = c1(xt)
            xt = n2(xt, s)
            s2 = _ane_sin(a2 * xt)
            xt = xt + (1.0 / a2) * (s2 * s2)
            xt = c2(xt)
            x = xt + x
        return x

    AdaINResBlock1.forward = _snake_ane_forward

    # 2+3. STFT inverse keeps exact torch.sin/cos — it's the final output
    # stage so accuracy matters more than ANE eligibility here. Any CPU
    # fallback only affects this last step.


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_kokoro_model():
    from kokoro import KPipeline
    from kokoro.istftnet import CustomSTFT

    pipeline = KPipeline(lang_code="a")
    model = pipeline.model
    model.eval()

    # Swap TorchSTFT → CustomSTFT (conv1d-based, no complex ops, CoreML-safe)
    gen = model.decoder.generator
    gen.stft = CustomSTFT(
        filter_length=gen.stft.filter_length,
        hop_length=gen.stft.hop_length,
        win_length=gen.stft.win_length,
    )

    return pipeline, model


# ---------------------------------------------------------------------------
# CoreML-friendly length-aware bidirectional LSTM
# ---------------------------------------------------------------------------
# StyleTTS2's text encoder, duration encoder, and prosody predictor all use
# pack_padded_sequence around bidirectional LSTMs. CoreML cannot convert
# pack_padded_sequence, so the export historically patched it to a no-op
# (reference.patch_pack_padded_sequence). That removed length-awareness — the
# reverse direction of every bidir LSTM consumed pad tokens before reaching the
# real prefix, contaminating real-token hidden states. Symptom: short utterances
# get distorted durations.
#
# CoreMLPackedBidirLSTM splits a bidir LSTM into two unidirectional LSTMs and
# reverses only the valid prefix using arange + where + gather — all static ops
# that survive jit.trace + coremltools convert. Math is equivalent to the
# original packed bidir LSTM under right-padding (our case).
class CoreMLPackedBidirLSTM(nn.Module):
    def __init__(self, src):
        super().__init__()
        if not (src.bidirectional and src.num_layers == 1 and src.batch_first):
            raise ValueError(
                "CoreMLPackedBidirLSTM expects a 1-layer bidirectional batch_first LSTM")
        self.hidden_size = src.hidden_size
        self.fwd = nn.LSTM(
            src.input_size, src.hidden_size, 1,
            batch_first=True, bias=src.bias)
        self.rev = nn.LSTM(
            src.input_size, src.hidden_size, 1,
            batch_first=True, bias=src.bias)

        def copy(dst, suffix):
            for name in ("weight_ih_l0", "weight_hh_l0"):
                getattr(dst, name).data.copy_(getattr(src, name + suffix).data)
            if src.bias:
                for name in ("bias_ih_l0", "bias_hh_l0"):
                    getattr(dst, name).data.copy_(getattr(src, name + suffix).data)

        copy(self.fwd, "")
        copy(self.rev, "_reverse")

    def forward(self, x, lengths, pad_mask):
        valid = (~pad_mask).to(x.dtype).unsqueeze(-1)
        x = x * valid

        y_f, _ = self.fwd(x)

        # Length-aware reverse via permutation-matrix matmul.
        # gather_along_axis on dynamic shapes loses output shape info in
        # coremltools (becomes `tensor<fp32, [?, ?, ?]>`) which prevents the
        # downstream LSTM from compiling (model-load error -14). matmul
        # carries shape forward cleanly. The permutation matrix selects
        # x[rev_idx[i]] for each output position i, where rev_idx flips only
        # the valid prefix (positions 0..L-1) and leaves pads in place. The
        # same permutation undoes itself, so we use it for both directions.
        b, t, c = x.shape
        idx = torch.arange(t, device=x.device, dtype=lengths.dtype).unsqueeze(0).expand(b, t)
        l = lengths.unsqueeze(1)
        rev_idx = torch.where(idx < l, l - 1 - idx, idx)
        positions = torch.arange(t, device=x.device, dtype=rev_idx.dtype).unsqueeze(0)
        perm = (positions.unsqueeze(0) == rev_idx.unsqueeze(-1)).to(x.dtype)

        x_rev = torch.matmul(perm, x)
        y_rev_seq, _ = self.rev(x_rev)
        y_r = torch.matmul(perm, y_rev_seq)

        return torch.cat([y_f, y_r], dim=-1) * valid


class CoreMLTextEncoder(nn.Module):
    """Length-aware replacement for kokoro.modules.TextEncoder."""

    def __init__(self, src):
        super().__init__()
        self.embedding = src.embedding
        self.cnn = src.cnn
        self.lstm = CoreMLPackedBidirLSTM(src.lstm)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m1 = m.unsqueeze(1)
        x = x.masked_fill(m1, 0.0)
        for c in self.cnn:
            x = c(x)
            x = x.masked_fill(m1, 0.0)
        x = x.transpose(1, 2)
        x = self.lstm(x, input_lengths, m)
        x = x.transpose(-1, -2)
        x = x.masked_fill(m1, 0.0)
        return x


class CoreMLDurationEncoder(nn.Module):
    """Length-aware replacement for kokoro.modules.DurationEncoder."""

    def __init__(self, src):
        super().__init__()
        from kokoro.modules import AdaLayerNorm
        self.lstms = nn.ModuleList()
        for block in src.lstms:
            if isinstance(block, AdaLayerNorm):
                self.lstms.append(block)
            else:
                self.lstms.append(CoreMLPackedBidirLSTM(block))
        self.dropout = src.dropout
        self.d_model = src.d_model
        self.sty_dim = src.sty_dim

    def forward(self, x, style, text_lengths, m):
        from kokoro.modules import AdaLayerNorm
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x = x.masked_fill(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x = x.masked_fill(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = block(x, text_lengths, m)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)


# ---------------------------------------------------------------------------
# Split pipeline wrappers for export
# ---------------------------------------------------------------------------
class KokoroModelA(nn.Module):
    """Frontend: predictor (stages 1-5) + GeneratorFrontEnd (SineGen/STFT).

    Runs on CPU_ONLY. Keeps atan2 (in STFT.transform) off the ANE.
    Matches stageharness dec_pipe frontend + SplitA_Predictor.
    """

    def __init__(self, model, max_tokens, max_audio, set_phases_fn):
        super().__init__()
        self.set_phases_fn = set_phases_fn

        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        # Hold the whole predictor so jit registers all its sub-modules
        # (predictor.shared / .F0 / .N / .F0_proj / .N_proj are reached via
        # self.predictor.F0Ntrain in forward). The text_encoder and lstm
        # inside model.predictor are bypassed at runtime by the CoreML*
        # wrappers below — their weights are double-registered in state_dict
        # but never traced.
        self.predictor = model.predictor
        self.predictor_text_encoder = CoreMLDurationEncoder(model.predictor.text_encoder)
        self.predictor_duration_lstm = CoreMLPackedBidirLSTM(model.predictor.lstm)
        self.text_encoder = CoreMLTextEncoder(model.text_encoder)
        self.gen_frontend = GeneratorFrontEnd(model.decoder.generator)

    def forward(self, input_ids, attention_mask, ref_s, speed, random_phases):
        """
        Args:
            input_ids:       [1, N] int32 (dynamic N)
            attention_mask:  [1, N] int32
            ref_s:           [1, 256] float32
            speed:           [1] float32
            random_phases:   [1, 9] float32
        Returns:
            asr:                  [1, 512, F] float32 (dynamic F = total frames)
            F0_pred:              [1, 2F] float32
            N_pred:               [1, 2F] float32
            har:                  [1, 22, H] float32
            audio_length_samples: [1] int32
            pred_dur_clamped:     [1, N] float32
        """
        self.set_phases_fn(self.gen_frontend, random_phases)

        input_lengths = attention_mask.sum(dim=1).long()
        text_mask = (attention_mask == 0)

        bert_dur = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, S_CONTENT_DIM:]

        d = self.predictor_text_encoder(d_en, s, input_lengths, text_mask)
        x = self.predictor_duration_lstm(d, input_lengths, text_mask)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed[0]
        pred_dur = torch.round(duration).clamp(min=1).long()
        pred_dur = pred_dur * attention_mask.long()

        # Dynamic alignment: frame count derived from predicted durations
        cumsum = torch.cumsum(pred_dur, dim=-1)
        total_frames = cumsum[0, -1]
        frame_indices = torch.arange(
            total_frames, device=input_ids.device
        ).unsqueeze(0)
        starts = F.pad(cumsum[:, :-1], (1, 0))
        pred_aln_trg = (
            (frame_indices.unsqueeze(1) >= starts.unsqueeze(2)) &
            (frame_indices.unsqueeze(1) < cumsum.unsqueeze(2))
        ).float()

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        # GeneratorFrontEnd: SineGen + STFT.transform (atan2 stays on CPU)
        har = self.gen_frontend(F0_pred)

        audio_length = (total_frames.float() * float(SAMPLES_PER_FRAME)).int()

        return asr, F0_pred, N_pred, har, audio_length, pred_dur.float()


class KokoroModelB(nn.Module):
    """Backend: DecoderBackEnd (decoder preprocessing + generator backend).

    Runs on ALL (ANE-eligible). No atan2/SineGen/STFT.transform — those
    are in Model A. Matches stageharness DecoderBackEnd.
    """

    def __init__(self, model):
        super().__init__()
        self.decoder_backend = DecoderBackEnd(model.decoder)

    def forward(self, asr, F0_pred, N_pred, s_content, har):
        """
        Args:
            asr:        [1, 512, max_frames] float32
            F0_pred:    [1, F0_frames] float32
            N_pred:     [1, F0_frames] float32
            s_content:  [1, 128] float32
            har:        [1, har_channels, har_frames] float32
        Returns:
            audio:      [1, 1, max_audio] float32
        """
        return self.decoder_backend(asr, F0_pred, N_pred, s_content, har)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _compile_model(pkg_path, output_dir, name):
    """Compile an .mlpackage to .mlmodelc."""
    print(f"  Compiling {name}...")
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", pkg_path, output_dir],
        capture_output=True, text=True)
    compiled = os.path.join(output_dir,
                            os.path.basename(pkg_path).replace(".mlpackage", ".mlmodelc"))
    if result.returncode == 0 and os.path.exists(compiled):
        print(f"  Compiled: {compiled}")
    else:
        print(f"  Compilation failed: {result.stderr[:200]}")


def _tokenize(text, pipeline):
    """Convert text to token IDs with BOS/EOS."""
    phonemes, _ = pipeline.g2p(text)
    raw = list(filter(lambda i: i is not None,
        map(lambda p: pipeline.model.vocab.get(p), phonemes)))
    return [0] + raw + [0]


def _quantize_int8(model, label):
    """Apply int8 linear symmetric weight quantization to a CoreML model."""
    import coremltools.optimize.coreml as cto
    config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    print(f"  Quantizing {label} weights to int8...")
    return cto.linear_quantize_weights(model, config)


def convert_voices_to_binary(voice_dir, output_dir):
    """Convert voice JSON files to float32 binary format (~5x smaller)."""
    import struct
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(voice_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(voice_dir, fname)) as f:
            data = json.load(f)
        dim = len(data["embedding"])
        keys = {}
        for k, v in data.items():
            key_id = 0 if k == "embedding" else int(k)
            keys[key_id] = np.array(v, dtype=np.float32)
        outpath = os.path.join(output_dir, fname.replace(".json", ".bin"))
        with open(outpath, "wb") as f:
            f.write(struct.pack("<HH", len(keys), dim))
            for key_id in sorted(keys.keys()):
                f.write(struct.pack("<H", key_id))
                f.write(keys[key_id].tobytes())
    count = len([f for f in os.listdir(output_dir) if f.endswith(".bin")])
    print(f"  Converted {count} voice files to binary")


def export_bucket(pipeline, model, set_phases_fn, bucket_name, bucket_config,
                  output_dir, verify=False):
    max_tokens = bucket_config["max_tokens"]
    max_audio = bucket_config["max_audio"]
    max_frames = max_audio // SAMPLES_PER_FRAME

    print(f"\nExporting {bucket_name} (max_tokens={max_tokens}, max_audio={max_audio})")

    # Override STFT transform for CoreML compatibility (torch.where instead of index_put_)
    CustomSTFT.transform = _stft_transform_coreml

    # --- Frontend (Model A): predictor + GeneratorFrontEnd on CPU_ONLY ---
    frontend = KokoroModelA(model, max_tokens, max_audio, set_phases_fn)
    frontend.eval()

    fe_ids = torch.zeros(1, max_tokens, dtype=torch.int64)
    fe_ids[0, :3] = torch.tensor([0, 50, 1])
    fe_mask = torch.zeros(1, max_tokens, dtype=torch.int64)
    fe_mask[0, :3] = 1
    fe_ref_s = torch.randn(1, 256)
    fe_speed = torch.tensor([1.0])
    fe_phases = torch.rand(1, NUM_HARMONICS) * 2 * torch.pi

    print("  Tracing frontend...")
    with torch.no_grad():
        fe_traced = torch.jit.trace(
            frontend, (fe_ids, fe_mask, fe_ref_s, fe_speed, fe_phases),
            check_trace=False,
        )

    print("  Converting frontend to CoreML...")
    fe_model = ct.convert(
        fe_traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_tokens), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_tokens),
                          dtype=np.int32),
            ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
            ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
            ct.TensorType(name="random_phases", shape=(1, NUM_HARMONICS),
                          dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="asr", dtype=np.float32),
            ct.TensorType(name="F0_pred", dtype=np.float32),
            ct.TensorType(name="N_pred", dtype=np.float32),
            ct.TensorType(name="har", dtype=np.float32),
            ct.TensorType(name="audio_length_samples", dtype=np.int32),
            ct.TensorType(name="pred_dur_clamped", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )

    fe_model = _quantize_int8(fe_model, "frontend")
    fe_pkg = os.path.join(output_dir, f"{bucket_name}_frontend.mlpackage")
    fe_model.save(fe_pkg)
    print(f"  Saved {fe_pkg}")

    _compile_model(fe_pkg, output_dir, "frontend")

    # --- Backend (Model B): DecoderBackEnd on ALL (ANE-eligible, no atan2) ---
    # Run frontend to get intermediate tensor shapes for tracing
    with torch.no_grad():
        asr, F0_pred, N_pred, har, _, _ = frontend(
            fe_ids, fe_mask, fe_ref_s, fe_speed, fe_phases)

    backend = KokoroModelB(model)
    backend.eval()

    be_s_content = torch.randn(1, S_CONTENT_DIM)

    print("  Tracing backend...")
    with torch.no_grad():
        be_traced = torch.jit.trace(
            backend, (asr, F0_pred, N_pred, be_s_content, har),
            check_trace=False,
        )

    print(f"  Backend input shapes: asr={tuple(asr.shape)}, "
          f"F0_pred={tuple(F0_pred.shape)}, har={tuple(har.shape)}")

    # Convert backend on a separate thread to avoid CoreML thread-local
    # state corruption between frontend and backend conversions.
    import threading
    _be_result = {}
    def _convert_backend():
        print("  Converting backend to CoreML...")
        _be_result["model"] = ct.convert(
            be_traced,
            inputs=[
                ct.TensorType(name="asr", shape=tuple(asr.shape),
                              dtype=np.float32),
                ct.TensorType(name="F0_pred", shape=tuple(F0_pred.shape),
                              dtype=np.float32),
                ct.TensorType(name="N_pred", shape=tuple(N_pred.shape),
                              dtype=np.float32),
                ct.TensorType(name="s_content", shape=(1, S_CONTENT_DIM), dtype=np.float32),
                ct.TensorType(name="har", shape=tuple(har.shape),
                              dtype=np.float32),
            ],
            outputs=[
                ct.TensorType(name="audio", dtype=np.float32),
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT32,
            convert_to="mlprogram",
        )
    _t = threading.Thread(target=_convert_backend)
    _t.start()
    _t.join()
    be_model = _be_result["model"]

    be_model = _quantize_int8(be_model, "backend")
    be_pkg = os.path.join(output_dir, f"{bucket_name}_backend.mlpackage")
    be_model.save(be_pkg)
    print(f"  Saved {be_pkg}")

    _compile_model(be_pkg, output_dir, "backend")

    if verify:
        verify_export(pipeline, model, set_phases_fn, fe_model, be_model,
                      frontend, backend,
                      bucket_name, max_tokens, max_frames)

    return fe_model, be_model


def verify_export(pipeline, pytorch_model, set_phases_fn, fe_coreml,
                  be_coreml, frontend_py, backend_py,
                  bucket_name, max_tokens, max_frames):
    """Compare PyTorch split wrappers vs CoreML split models (apples-to-apples)."""
    print(f"\n  Verifying {bucket_name}...")

    text = "Hello world, this is a test."
    phonemes, _ = pipeline.g2p(text)
    vocab = pytorch_model.vocab
    raw = list(filter(lambda i: i is not None,
        map(lambda p: vocab.get(p), phonemes)))
    token_ids = [0] + raw + [0]

    seq_len = min(len(token_ids), max_tokens)
    token_ids = token_ids[:seq_len]

    voice_pack = pipeline.load_voice("af_heart")
    ref_s_tensor = voice_pack[seq_len]
    if ref_s_tensor.dim() == 1:
        ref_s_tensor = ref_s_tensor.unsqueeze(0)

    phases = torch.rand(1, NUM_HARMONICS) * 2 * torch.pi

    # Prepare fixed-size inputs
    input_ids_t = torch.zeros(1, max_tokens, dtype=torch.int64)
    input_ids_t[0, :seq_len] = torch.tensor(token_ids[:seq_len])
    mask_t = torch.zeros(1, max_tokens, dtype=torch.int64)
    mask_t[0, :seq_len] = 1

    # Run PyTorch split pipeline as reference (reuse already-constructed wrappers)
    with torch.no_grad():
        asr, F0_pred, N_pred, har, audio_len, pred_dur = frontend_py(
            input_ids_t, mask_t, ref_s_tensor, torch.tensor([1.0]), phases)
        py_audio = backend_py(
            asr, F0_pred, N_pred, ref_s_tensor[:, :S_CONTENT_DIM], har)

    py_len = int(audio_len[0].item())
    py_audio_np = py_audio.flatten().numpy()[:py_len]

    # Reload with correct compute units (frontend CPU, backend ANE).
    # mlProgram models require weights on disk, so save/reload via tmpdir.
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        fe_path = os.path.join(tmpdir, "fe.mlpackage")
        be_path = os.path.join(tmpdir, "be.mlpackage")
        fe_coreml.save(fe_path)
        be_coreml.save(be_path)
        fe_cpu = ct.models.MLModel(fe_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        be_ane = ct.models.MLModel(be_path, compute_units=ct.ComputeUnit.ALL)

    # Run CoreML split pipeline
    input_ids_np = input_ids_t.numpy().astype(np.int32)
    mask_np = mask_t.numpy().astype(np.int32)

    fe_out = fe_cpu.predict({
        "input_ids": input_ids_np,
        "attention_mask": mask_np,
        "ref_s": ref_s_tensor.numpy().astype(np.float32),
        "speed": np.array([1.0], dtype=np.float32),
        "random_phases": phases.numpy().astype(np.float32),
    })

    s_content = ref_s_tensor[:, :S_CONTENT_DIM].numpy().astype(np.float32)
    be_out = be_ane.predict({
        "asr": fe_out["asr"].astype(np.float32),
        "F0_pred": fe_out["F0_pred"].astype(np.float32),
        "N_pred": fe_out["N_pred"].astype(np.float32),
        "s_content": s_content,
        "har": fe_out["har"].astype(np.float32),
    })

    coreml_len = int(fe_out["audio_length_samples"].flatten()[0])
    coreml_audio = be_out["audio"].flatten()[:coreml_len]

    min_len = min(len(py_audio_np), len(coreml_audio))
    if min_len == 0:
        print("  ERROR: empty audio")
        return

    diff = py_audio_np[:min_len] - coreml_audio[:min_len]
    mse = float(np.mean(diff ** 2))
    max_diff = float(np.max(np.abs(diff)))
    corr = float(np.corrcoef(py_audio_np[:min_len], coreml_audio[:min_len])[0, 1])

    print(f"  PyTorch: {len(py_audio_np)} samples, "
          f"CoreML: {len(coreml_audio)} samples")
    print(f"  MSE: {mse:.8f}, Max diff: {max_diff:.6f}, Correlation: {corr:.6f}")

    if corr > 0.99:
        print(f"  PASS (correlation {corr:.4f})")
    elif corr > 0.95:
        print(f"  WARN (correlation {corr:.4f})")
    else:
        print(f"  FAIL (correlation {corr:.4f})")

    import soundfile as sf
    os.makedirs("verify_output", exist_ok=True)
    sf.write(f"verify_output/{bucket_name}_pytorch.wav", py_audio_np, 24000)
    sf.write(f"verify_output/{bucket_name}_coreml.wav", coreml_audio, 24000)
    print(f"  Saved to verify_output/{bucket_name}_{{pytorch,coreml}}.wav")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def export_dynamic(pipeline, model, set_phases_fn, output_dir,
                   trace_tokens=100, max_tokens=512, verify=False):
    """Export a single dynamic frontend + backend model pair.

    No fixed buckets, no padding required. Frontend accepts 3-max_tokens tokens.
    Backend accepts variable frame counts. No ANE patches applied.

    The alignment matrix is sized for max_tokens (not trace_tokens) so that
    any token count up to max_tokens produces full-length audio.
    """
    # Alignment is now dynamic (derived from predicted durations), so trace_audio
    # only affects the trace-time tensor sizes, not the runtime output.
    trace_audio = ((trace_tokens * 1600 + 599) // 600) * 600

    print(f"\nExporting dynamic model (trace={trace_tokens} tokens, range=3-{max_tokens})")

    CustomSTFT.transform = _stft_transform_coreml

    # --- Frontend ---
    frontend = KokoroModelA(model, trace_tokens, trace_audio, set_phases_fn)
    frontend.eval()

    fe_ids = torch.zeros(1, trace_tokens, dtype=torch.int64)
    fe_ids[0, :3] = torch.tensor([0, 50, 1])
    fe_mask = torch.zeros(1, trace_tokens, dtype=torch.int64)
    fe_mask[0, :3] = 1
    fe_ref_s = torch.randn(1, 256)
    fe_speed = torch.tensor([1.0])
    fe_phases = torch.zeros(1, NUM_HARMONICS)

    print("  Tracing frontend...")
    with torch.no_grad():
        fe_traced = torch.jit.trace(
            frontend, (fe_ids, fe_mask, fe_ref_s, fe_speed, fe_phases),
            check_trace=False,
        )

    print("  Converting frontend to CoreML (dynamic shapes)...")
    seq_dim = ct.RangeDim(lower_bound=3, upper_bound=max_tokens, default=trace_tokens)
    fe_model = ct.convert(
        fe_traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_dim), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_dim), dtype=np.int32),
            ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
            ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
            ct.TensorType(name="random_phases", shape=(1, NUM_HARMONICS),
                          dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="asr", dtype=np.float32),
            ct.TensorType(name="F0_pred", dtype=np.float32),
            ct.TensorType(name="N_pred", dtype=np.float32),
            ct.TensorType(name="har", dtype=np.float32),
            ct.TensorType(name="audio_length_samples", dtype=np.int32),
            ct.TensorType(name="pred_dur_clamped", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )

    fe_pkg = os.path.join(output_dir, "kokoro_frontend.mlpackage")
    fe_model.save(fe_pkg)
    print(f"  Saved {fe_pkg}")

    _compile_model(fe_pkg, output_dir, "frontend")

    # --- Backend (dynamic shapes) ---
    with torch.no_grad():
        asr, F0_pred, N_pred, har, _, _ = frontend(
            fe_ids, fe_mask, fe_ref_s, fe_speed, fe_phases)

    backend = KokoroModelB(model)
    backend.eval()

    be_s_content = torch.randn(1, S_CONTENT_DIM)

    print("  Tracing backend...")
    with torch.no_grad():
        be_traced = torch.jit.trace(
            backend, (asr, F0_pred, N_pred, be_s_content, har),
            check_trace=False,
        )

    frames_dim = ct.RangeDim(lower_bound=1, upper_bound=2000, default=asr.shape[2])
    f0_dim = ct.RangeDim(lower_bound=1, upper_bound=4000, default=F0_pred.shape[1])
    har_dim = ct.RangeDim(lower_bound=1, upper_bound=500000, default=har.shape[2])

    import threading
    _be_result = {}
    def _convert_backend():
        print("  Converting backend to CoreML (dynamic shapes)...")
        _be_result["model"] = ct.convert(
            be_traced,
            inputs=[
                ct.TensorType(name="asr", shape=(1, 512, frames_dim), dtype=np.float32),
                ct.TensorType(name="F0_pred", shape=(1, f0_dim), dtype=np.float32),
                ct.TensorType(name="N_pred", shape=(1, f0_dim), dtype=np.float32),
                ct.TensorType(name="s_content", shape=(1, S_CONTENT_DIM), dtype=np.float32),
                ct.TensorType(name="har", shape=(1, 22, har_dim), dtype=np.float32),
            ],
            outputs=[
                ct.TensorType(name="audio", dtype=np.float32),
            ],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT32,
            convert_to="mlprogram",
        )
    _t = threading.Thread(target=_convert_backend)
    _t.start()
    _t.join()
    be_model = _be_result["model"]

    be_pkg = os.path.join(output_dir, "kokoro_backend.mlpackage")
    be_model.save(be_pkg)
    print(f"  Saved {be_pkg}")

    _compile_model(be_pkg, output_dir, "backend")

    return fe_model, be_model


def _export_palettized(output_dir):
    """Create 8-bit palettized copies of the dynamic models for quality tracking."""
    import coremltools.optimize.coreml as cto
    config = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(nbits=8))

    for name in ["kokoro_frontend", "kokoro_backend"]:
        src = os.path.join(output_dir, f"{name}.mlpackage")
        dst = os.path.join(output_dir, f"{name}_pal8.mlpackage")
        if not os.path.exists(src):
            continue
        print(f"  Palettizing {name} (8-bit)...")
        model = ct.models.MLModel(src)
        quantized = cto.palettize_weights(model, config)
        quantized.save(dst)
        src_size = os.path.getsize(os.path.join(src, "Data/com.apple.CoreML/weights/weight.bin"))
        dst_size = os.path.getsize(os.path.join(dst, "Data/com.apple.CoreML/weights/weight.bin"))
        print(f"    {src_size/1024/1024:.0f}MB → {dst_size/1024/1024:.0f}MB")


def main():
    parser = argparse.ArgumentParser(description="Export Kokoro-82M to CoreML")
    parser.add_argument("--output-dir", default="./models_export")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="Zero noise and phases for reproducible testing (harness mode)")
    parser.add_argument("--skip-palettize", action="store_true",
                        help="Skip palettized model export")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Kokoro model...")
    pipeline, model = load_kokoro_model()

    if args.deterministic:
        print("Patching SineGen for CoreML (deterministic — zeroed noise/phases)...")
        set_phases_fn = patch_sinegen_for_export(model)
    else:
        print("Patching SineGen for CoreML (production — natural randomness)...")
        set_phases_fn = patch_sinegen_for_production(model)

    # pack_padded_sequence no-op — needed for tracing (CoreML can't convert it),
    # but doesn't affect output when there's no padding (batch=1, no masked tokens)
    patch_pack_padded_sequence()

    # Dynamic export — no ANE patches, no fixed padding
    export_dynamic(pipeline, model, set_phases_fn, args.output_dir,
                   verify=args.verify)

    # Also export palettized variants for quality tracking
    if not args.skip_palettize:
        _export_palettized(args.output_dir)

    print(f"\nDone. Models in {args.output_dir}/")


if __name__ == "__main__":
    main()
