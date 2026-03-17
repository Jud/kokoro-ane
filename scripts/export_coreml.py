#!/usr/bin/env python3
"""Export Kokoro-82M from PyTorch to split CoreML models for kokoro-tts-swift.

Creates frontend + backend CoreML model pairs per bucket:
  - {bucket}_frontend.mlmodelc  (predictor, CPU_ONLY, ~5ms)
  - {bucket}_backend.mlmodelc   (decoder, ANE-eligible, ~112ms)

Usage:
    .venv/bin/python scripts/export_coreml.py [--output-dir ./models_export]
    .venv/bin/python scripts/export_coreml.py --verify
    .venv/bin/python scripts/export_coreml.py --bucket kokoro_21_5s
"""
import argparse
import os
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from coreml_ops import register_missing_torch_ops

# CoreML-compatible _f02sine for SineGen.
# Only _f02sine is used — it gets monkey-patched onto the original SineGen class.
# The original __init__, _f02uv, and forward are kept on the original class.

class SineGen:
    @staticmethod
    def _f02sine(self, f0_values):
        """CoreML-compatible phase computation: fractional freq → pool → cumsum → interpolate → sin."""
        val = f0_values / self.sampling_rate
        rad_values = val - torch.floor(val)

        if not self.flag_for_pulse:
            K = int(self.upsample_scale)

            # Work in [B, D, L] channels-first format to minimize transposes
            x = rad_values.transpose(1, 2)                       # [B, D, L]
            x = F.avg_pool1d(x, kernel_size=K, stride=K)         # [B, D, N]
            x = torch.cumsum(x, dim=2)                           # [B, D, N]
            x = x * (2.0 * torch.pi * K)
            x = F.interpolate(x, scale_factor=float(K),
                              mode='linear', align_corners=True)  # [B, D, N*K]
            phase = x.transpose(1, 2)                             # [B, N*K, D]

            # Wrap phase to [0, 2π)
            two_pi = 2.0 * torch.pi
            phase = phase - two_pi * torch.floor(phase / two_pi)

            sines = torch.sin(phase)

        return sines

# CustomSTFT inlined from kokoro.custom_stft.

class CustomSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window="hann", center=True, pad_mode="replicate"):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = filter_length
        self.center = center
        self.pad_mode = pad_mode
        self.freq_bins = self.n_fft // 2 + 1

        assert window == 'hann', window
        window_tensor = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        if self.win_length < self.n_fft:
            window_tensor = F.pad(window_tensor, (0, self.n_fft - self.win_length))
        elif self.win_length > self.n_fft:
            window_tensor = window_tensor[:self.n_fft]
        # window_tensor used only during __init__ to construct DFT matrices
        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2 * np.pi * np.outer(k, n) / self.n_fft
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)

        forward_window = window_tensor.numpy()
        forward_real = dft_real * forward_window
        forward_imag = dft_imag * forward_window

        # Fused forward weight: cat([W_real, W_imag]) — single conv1d for both
        self.register_buffer("weight_forward_fused",
                             torch.cat([
                                 torch.from_numpy(forward_real).float().unsqueeze(1),
                                 torch.from_numpy(forward_imag).float().unsqueeze(1),
                             ], dim=0))

        inv_scale = 1.0 / self.n_fft
        angle_t = 2 * np.pi * np.outer(n, k) / self.n_fft
        idft_cos = np.cos(angle_t).T
        idft_sin = np.sin(angle_t).T
        inv_window = window_tensor.numpy() * inv_scale

        # Fused inverse weight: cat([W_real, -W_imag]) — single conv_transpose1d
        self.register_buffer("weight_backward_fused",
                             torch.cat([
                                 torch.from_numpy(idft_cos * inv_window).float().unsqueeze(1),
                                 -torch.from_numpy(idft_sin * inv_window).float().unsqueeze(1),
                             ], dim=0))

    def transform(self, waveform):
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
        x = waveform.unsqueeze(1)
        # Single fused conv1d for both real and imag
        ri = F.conv1d(x, self.weight_forward_fused, bias=None,
                      stride=self.hop_length, padding=0)
        real_out = ri[:, :self.freq_bins, :]
        imag_out = ri[:, self.freq_bins:, :]
        magnitude = torch.sqrt(real_out * real_out + imag_out * imag_out + 1e-14)
        phase = torch.atan2(imag_out, real_out)
        correction_mask = (imag_out == 0) & (real_out < 0)
        phase[correction_mask] = torch.pi
        return magnitude, phase

    def inverse(self, magnitude, phase, length=None):
        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        combined = torch.cat([real_part, imag_part], dim=1)
        waveform = F.conv_transpose1d(combined, self.weight_backward_fused,
                                      bias=None, stride=self.hop_length, padding=0)
        if self.center:
            pad_len = self.n_fft // 2
            waveform = waveform[..., pad_len:-pad_len]
        if length is not None:
            waveform = waveform[..., :length]
        return waveform

    def forward(self, x):
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])

register_missing_torch_ops()


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
    Takes precomputed har conditioning."""
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

    def forward(self, x, s, har):
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
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
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
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
}

SAMPLES_PER_FRAME = 600  # decoder upsample (2x) * generator (10*6*5=300)
NUM_HARMONICS = 9        # fundamental + 8 overtones
S_CONTENT_DIM = 128      # first half of 256-dim style vector (content style)


# ---------------------------------------------------------------------------
# Monkey-patches for CoreML compatibility
# ---------------------------------------------------------------------------
def patch_pack_padded_sequence():
    """Replace pack_padded_sequence/pad_packed_sequence with no-ops.

    CoreML cannot convert these ops. PyTorch audio quality is identical
    without them (verified empirically).
    """
    nn.utils.rnn.pack_padded_sequence = lambda x, lengths, **kw: x
    nn.utils.rnn.pad_packed_sequence = lambda x, **kw: (x, None)


def _patch_snake_mul(model):
    """Replace pow(sin(x), 2) with sin(x) * sin(x) in Snake activations.

    pow is not in ANE's native op list. Explicit mul is ANE-native
    and avoids the pow decomposition overhead.
    """
    from kokoro.istftnet import AdaINResBlock1

    def _snake_mul_forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2,
            self.alpha1, self.alpha2
        ):
            xt = n1(x, s)
            s1 = torch.sin(a1 * xt)
            xt = xt + (1.0 / a1) * (s1 * s1)
            xt = c1(xt)
            xt = n2(xt, s)
            s2 = torch.sin(a2 * xt)
            xt = xt + (1.0 / a2) * (s2 * s2)
            xt = c2(xt)
            x = xt + x
        return x

    AdaINResBlock1.forward = _snake_mul_forward


def patch_sinegen_for_export(model):
    """Apply inlined SineGen to the model and return a phases helper.

    The inlined SineGen class (above) has CoreML-compatible _f02sine built in.
    This function applies it to the loaded model's SineGen instances via
    class-level method replacement, and provides the set_phases helper.
    """
    from kokoro.istftnet import SineGen as OriginalSineGen

    # Replace _f02sine on the original class
    OriginalSineGen._f02sine = SineGen._f02sine

    # Skip dead noise computation entirely.
    # The original forward computes noise = noise_amp * randn_like(sines) then
    # sines = sines * uv + noise. With zero noise, this simplifies to sines * uv.
    # Eliminating the dead code produces a cleaner, smaller traced graph.
    def _zero_noise_forward(self, f0):
        fn = torch.multiply(f0, torch.FloatTensor(
            [[range(1, self.harmonic_num + 2)]]).to(f0.device))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        sine_waves = sine_waves * uv
        return sine_waves, uv, torch.zeros_like(sine_waves)
    OriginalSineGen.forward = _zero_noise_forward

    # Replace pow(sin,2) with mul in Snake activations
    _patch_snake_mul(model)

    # Eliminate dead noise in SourceModuleHnNSF — noi_source is never used
    # in the generator, and randn_like in the trace creates unnecessary ops
    from kokoro.istftnet import SourceModuleHnNSF
    def _no_noise_source_forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, torch.zeros_like(uv), uv
    SourceModuleHnNSF.forward = _no_noise_source_forward

    # Set external phases on all SineGen instances
    def set_phases(module, phases):
        for m in module.modules():
            if isinstance(m, OriginalSineGen):
                m._external_phases = torch.zeros_like(phases)

    return set_phases


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
# Split pipeline wrappers for export
# ---------------------------------------------------------------------------
class KokoroModelA(nn.Module):
    """Frontend: predictor (stages 1-5) + GeneratorFrontEnd (SineGen/STFT).

    Runs on CPU_ONLY. Keeps atan2 (in STFT.transform) off the ANE.
    Matches stageharness dec_pipe frontend + SplitA_Predictor.
    """

    def __init__(self, model, max_tokens, max_audio, set_phases_fn):
        super().__init__()
        self.max_tokens = max_tokens
        self.max_frames = max_audio // SAMPLES_PER_FRAME
        self.set_phases_fn = set_phases_fn

        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.predictor = model.predictor
        self.text_encoder = model.text_encoder
        self.gen_frontend = GeneratorFrontEnd(model.decoder.generator)

    def forward(self, input_ids, attention_mask, ref_s, speed, random_phases):
        """
        Args:
            input_ids:       [1, max_tokens] int32
            attention_mask:  [1, max_tokens] int32
            ref_s:           [1, 256] float32
            speed:           [1] float32
            random_phases:   [1, 9] float32
        Returns:
            asr:                  [1, 512, max_frames] float32
            F0_pred:              [1, F0_frames] float32
            N_pred:               [1, F0_frames] float32
            har:                  [1, har_channels, har_frames] float32
            audio_length_samples: [1] int32
            pred_dur_clamped:     [1, max_tokens] float32
        """
        self.set_phases_fn(self.gen_frontend, random_phases)

        input_lengths = attention_mask.sum(dim=1).long()
        text_mask = (attention_mask == 0)

        bert_dur = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, S_CONTENT_DIM:]

        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed[0]
        pred_dur = torch.round(duration).clamp(min=1).long()
        pred_dur = pred_dur * attention_mask.long()

        cumsum = torch.cumsum(pred_dur, dim=-1)
        frame_indices = torch.arange(
            self.max_frames, device=input_ids.device
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

        audio_length = (cumsum[:, -1].float() * float(SAMPLES_PER_FRAME)).int()

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
def export_bucket(pipeline, model, set_phases_fn, bucket_name, bucket_config,
                  output_dir, verify=False):
    max_tokens = bucket_config["max_tokens"]
    max_audio = bucket_config["max_audio"]
    max_frames = max_audio // SAMPLES_PER_FRAME

    print(f"\nExporting {bucket_name} (max_tokens={max_tokens}, max_audio={max_audio})")

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

    fe_pkg = os.path.join(output_dir, f"{bucket_name}_frontend.mlpackage")
    fe_model.save(fe_pkg)
    print(f"  Saved {fe_pkg}")

    def _compile(pkg_path, name):
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

    _compile(fe_pkg, "frontend")

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

    print("  Converting backend to CoreML...")
    be_model = ct.convert(
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

    be_pkg = os.path.join(output_dir, f"{bucket_name}_backend.mlpackage")
    be_model.save(be_pkg)
    print(f"  Saved {be_pkg}")

    _compile(be_pkg, "backend")

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
def main():
    parser = argparse.ArgumentParser(description="Export Kokoro-82M to CoreML")
    parser.add_argument("--output-dir", default="./models_export")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bucket", choices=list(BUCKETS.keys()))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Applying CoreML compatibility patches...")
    patch_pack_padded_sequence()

    print("Loading Kokoro model...")
    pipeline, model = load_kokoro_model()

    print("Patching SineGen for CoreML...")
    set_phases_fn = patch_sinegen_for_export(model)

    buckets = {args.bucket: BUCKETS[args.bucket]} if args.bucket else BUCKETS
    for name, config in buckets.items():
        export_bucket(pipeline, model, set_phases_fn, name, config,
                      args.output_dir, verify=args.verify)

    print(f"\nDone. Models in {args.output_dir}/")


if __name__ == "__main__":
    main()
