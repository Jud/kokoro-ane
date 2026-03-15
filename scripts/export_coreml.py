#!/usr/bin/env python3
"""Export Kokoro-82M from PyTorch to CoreML models for kokoro-tts-swift.

Creates two fixed-size CoreML models:
  - kokoro_21_5s.mlmodelc  (max 124 tokens, ~5s audio)
  - kokoro_24_10s.mlmodelc (max 242 tokens, ~10s audio)

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

# SineGen inlined from kokoro.istftnet with CoreML-compatible _f02sine.
# The harness imports this class, so changes here are traced natively.
# Original _f02sine uses: % 1, in-place ops, F.interpolate(scale_factor=1/K)
# This version fixes: fmod→floor, in-place→functional, fractional interp→avg_pool1d

class SineGen(nn.Module):
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0, flag_for_pulse=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        # Instantaneous frequency as fraction of sample rate
        # Use x - floor(x) instead of % 1 (CoreML's floor_mod has fmod semantics)
        val = f0_values / self.sampling_rate
        rad_values = val - torch.floor(val)

        # Random initial phase (functional ops only — no in-place assignment)
        if hasattr(self, '_external_phases') and self._external_phases is not None:
            rand_ini = self._external_phases[:, :f0_values.shape[2]]
        else:
            rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2],
                                  device=f0_values.device)

        # Zero fundamental's phase offset (functional: cat zeros with rest)
        rand_ini = torch.cat([
            torch.zeros(rand_ini.shape[0], 1, device=rand_ini.device),
            rand_ini[:, 1:]
        ], dim=1)

        # Add phase offset to first time step only (functional: pad + add)
        offset = F.pad(
            rand_ini.unsqueeze(1),           # [B, 1, D]
            (0, 0, 0, f0_values.shape[1] - 1)  # pad dim1: 0 left, L-1 right
        )  # [B, L, D]
        rad_values = rad_values + offset

        if not self.flag_for_pulse:
            K = int(self.upsample_scale)

            # Downscale: avg_pool1d replaces F.interpolate(scale_factor=1/K)
            rad_t = rad_values.transpose(1, 2)                         # [B, D, L]
            rad_down_t = F.avg_pool1d(rad_t, kernel_size=K, stride=K)  # [B, D, N]
            rad_down = rad_down_t.transpose(1, 2)                      # [B, N, D]

            # Cumulative phase at frame rate
            phase_down = torch.cumsum(rad_down, dim=1)  # [B, N, D]

            # Scale by 2πK and upscale with integer scale factor
            phase_scaled = phase_down.transpose(1, 2) * (2.0 * torch.pi * K)
            phase_up = F.interpolate(
                phase_scaled,
                scale_factor=float(K),
                mode='linear',
                align_corners=True
            )  # [B, D, N*K]
            phase = phase_up.transpose(1, 2)  # [B, N*K, D]

            # Wrap phase to [0, 2π) — CoreML's sin() loses precision for
            # large arguments (higher harmonics reach 25,000+ radians)
            two_pi = 2.0 * torch.pi
            phase = phase - two_pi * torch.floor(phase / two_pi)

            sines = torch.sin(phase)

        return sines

    def forward(self, f0):
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise

# CustomSTFT copied from kokoro.custom_stft with ANE-compatible transform.
# The harness imports this class, so changes here are traced natively.

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
        self.register_buffer("window", window_tensor)

        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2 * np.pi * np.outer(k, n) / self.n_fft
        dft_real = np.cos(angle)
        dft_imag = -np.sin(angle)

        forward_window = window_tensor.numpy()
        forward_real = dft_real * forward_window
        forward_imag = dft_imag * forward_window

        self.register_buffer("weight_forward_real",
                             torch.from_numpy(forward_real).float().unsqueeze(1))
        self.register_buffer("weight_forward_imag",
                             torch.from_numpy(forward_imag).float().unsqueeze(1))

        inv_scale = 1.0 / self.n_fft
        angle_t = 2 * np.pi * np.outer(n, k) / self.n_fft
        idft_cos = np.cos(angle_t).T
        idft_sin = np.sin(angle_t).T
        inv_window = window_tensor.numpy() * inv_scale

        self.register_buffer("weight_backward_real",
                             torch.from_numpy(idft_cos * inv_window).float().unsqueeze(1))
        self.register_buffer("weight_backward_imag",
                             torch.from_numpy(idft_sin * inv_window).float().unsqueeze(1))

        # Fused inverse weight: single conv_transpose1d eliminates
        # cancellation error from separate real_rec - imag_rec subtraction
        self.register_buffer("weight_backward_fused",
                             torch.cat([
                                 torch.from_numpy(idft_cos * inv_window).float().unsqueeze(1),
                                 torch.from_numpy(-idft_sin * inv_window).float().unsqueeze(1),
                             ], dim=0))

    def transform(self, waveform):
        if self.center:
            pad_len = self.n_fft // 2
            waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
        x = waveform.unsqueeze(1)
        real_out = F.conv1d(x, self.weight_forward_real, bias=None,
                            stride=self.hop_length, padding=0)
        imag_out = F.conv1d(x, self.weight_forward_imag, bias=None,
                            stride=self.hop_length, padding=0)
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
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
# Pipeline split modules for ANE-compatible decoder
# ---------------------------------------------------------------------------
class GeneratorFrontEnd(nn.Module):
    """SineGen + STFT transform → har conditioning.

    This portion contains atan2 + correction_mask which must run on CPU.
    Split out so the rest of the generator can run on ANE.
    """
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
        har = torch.cat([har_spec, har_phase], dim=1)
        return har


class GeneratorBackEnd(nn.Module):
    """Upsampling cascade + inverse STFT → audio.

    Takes precomputed har conditioning. No atan2 — runs fully on ANE.
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

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
BUCKETS = {
    "kokoro_21_5s":  {"max_tokens": 124, "max_audio": 175_800},
    "kokoro_24_10s": {"max_tokens": 242, "max_audio": 240_000},
}

SAMPLES_PER_FRAME = 600  # decoder upsample (2x) * generator (10*6*5=300)
NUM_HARMONICS = 9        # fundamental + 8 overtones


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


def patch_sinegen_for_export(model):
    """Apply inlined SineGen to the model and return a phases helper.

    The inlined SineGen class (above) has CoreML-compatible _f02sine built in.
    This function applies it to the loaded model's SineGen instances via
    class-level method replacement, and provides the set_phases helper.
    """
    from kokoro.istftnet import SineGen as OriginalSineGen

    # Apply inlined _f02sine to the original class so existing instances use it
    OriginalSineGen._f02sine = SineGen._f02sine

    # Patch forward to use deterministic noise (zeros instead of randn)
    # so PyTorch reference and CoreML produce identical outputs.
    _orig_forward = OriginalSineGen.forward
    def _deterministic_forward(self, f0):
        # Temporarily replace randn_like with zeros_like
        _real_randn = torch.randn_like
        torch.randn_like = torch.zeros_like
        try:
            return _orig_forward(self, f0)
        finally:
            torch.randn_like = _real_randn
    OriginalSineGen.forward = _deterministic_forward

    # Helper to set external phases on all SineGen instances
    def set_phases(module, phases):
        for m in module.modules():
            if isinstance(m, OriginalSineGen):
                m._external_phases = phases

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
# Static wrapper for CoreML export
# ---------------------------------------------------------------------------
class KokoroStaticWrapper(nn.Module):
    """Fixed-shape wrapper for CoreML export.

    Takes raw token IDs and produces audio. All intermediate tensors
    use fixed maximum sizes determined by the bucket config.
    """

    def __init__(self, model, max_tokens, max_audio, set_phases_fn):
        super().__init__()
        self.max_tokens = max_tokens
        self.max_audio = max_audio
        self.max_frames = max_audio // SAMPLES_PER_FRAME
        self.set_phases_fn = set_phases_fn

        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.predictor = model.predictor
        self.text_encoder = model.text_encoder
        self.decoder = model.decoder

    def forward(self, input_ids, attention_mask, ref_s, speed, random_phases):
        """
        Args:
            input_ids:       [1, max_tokens] int32 — token IDs (0-padded)
            attention_mask:  [1, max_tokens] int32 — 1=real, 0=padding
            ref_s:           [1, 256] float32 — voice style embedding
            speed:           [1] float32 — speed factor
            random_phases:   [1, 9] float32 — harmonic phase offsets
        Returns:
            audio:           [1, 1, max_audio] float32 — waveform (silence-padded)
            audio_length:    [1] int32 — number of valid audio samples
            pred_dur:        [1, max_tokens] float32 — per-token durations
        """
        # Set external phases on all SineGen instances
        self.set_phases_fn(self.decoder, random_phases)

        input_lengths = attention_mask.sum(dim=1).long()
        text_mask = (attention_mask == 0)  # True for padding positions

        # BERT encoding
        bert_dur = self.bert(input_ids, attention_mask=attention_mask)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        s = ref_s[:, 128:]  # style for predictor

        # Duration prediction
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed[0]
        pred_dur = torch.round(duration).clamp(min=1).long()

        # Zero out padding token durations
        pred_dur = pred_dur * attention_mask.long()

        # Static alignment matrix using cumsum (avoids dynamic repeat_interleave)
        cumsum = torch.cumsum(pred_dur, dim=-1)  # [1, max_tokens]
        total_frames = cumsum[0, -1]

        # Fixed-size frame indices (max_frames)
        frame_indices = torch.arange(
            self.max_frames, device=input_ids.device
        ).unsqueeze(0)  # [1, max_frames]

        starts = F.pad(cumsum[:, :-1], (1, 0))  # [1, max_tokens]

        # Alignment: pred_aln_trg[b, t, f] = 1 iff frame f belongs to token t
        pred_aln_trg = (
            (frame_indices.unsqueeze(1) >= starts.unsqueeze(2)) &
            (frame_indices.unsqueeze(1) < cumsum.unsqueeze(2))
        ).float()  # [1, max_tokens, max_frames]

        # Apply alignment to get frame-level features
        en = d.transpose(-1, -2) @ pred_aln_trg  # [1, 640, max_frames]
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg  # [1, 512, max_frames]

        # Decode to audio (fixed output length = max_frames * SAMPLES_PER_FRAME)
        audio = self.decoder(
            asr, F0_pred, N_pred, ref_s[:, :128]
        )  # [1, 1, max_audio]

        # Audio length from predicted durations
        audio_length = (total_frames * SAMPLES_PER_FRAME).int().unsqueeze(0)

        return audio, audio_length, pred_dur.float()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_bucket(pipeline, model, set_phases_fn, bucket_name, bucket_config,
                  output_dir, verify=False):
    max_tokens = bucket_config["max_tokens"]
    max_audio = bucket_config["max_audio"]

    print(f"\nExporting {bucket_name} (max_tokens={max_tokens}, max_audio={max_audio})")

    wrapper = KokoroStaticWrapper(model, max_tokens, max_audio, set_phases_fn)
    wrapper.eval()

    # Example inputs for tracing
    example_ids = torch.zeros(1, max_tokens, dtype=torch.int64)
    example_ids[0, :3] = torch.tensor([0, 50, 1])
    example_mask = torch.zeros(1, max_tokens, dtype=torch.int64)
    example_mask[0, :3] = 1
    example_ref_s = torch.randn(1, 256)
    example_speed = torch.tensor([1.0])
    example_phases = torch.rand(1, NUM_HARMONICS) * 2 * torch.pi

    print("  Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (example_ids, example_mask, example_ref_s,
             example_speed, example_phases),
            check_trace=False,
        )

    print("  Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
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
            ct.TensorType(name="audio", dtype=np.float32),
            ct.TensorType(name="audio_length_samples", dtype=np.int32),
            ct.TensorType(name="pred_dur_clamped", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        convert_to="mlprogram",
    )

    mlpackage_path = os.path.join(output_dir, f"{bucket_name}.mlpackage")
    mlmodel.save(mlpackage_path)
    print(f"  Saved {mlpackage_path}")

    print("  Compiling to .mlmodelc...")
    compile_result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", mlpackage_path, output_dir],
        capture_output=True, text=True
    )
    compiled_path = os.path.join(output_dir, f"{bucket_name}.mlmodelc")
    if compile_result.returncode == 0 and os.path.exists(compiled_path):
        print(f"  Compiled: {compiled_path}")
    else:
        print(f"  Compilation failed: {compile_result.stderr[:200]}")

    if verify:
        verify_export(pipeline, model, set_phases_fn, mlmodel,
                      bucket_name, max_tokens)

    return mlmodel


def verify_export(pipeline, pytorch_model, set_phases_fn, coreml_model,
                  bucket_name, max_tokens):
    print(f"\n  Verifying {bucket_name}...")

    text = "Hello world, this is a test."
    phonemes, _ = pipeline.g2p(text)
    token_ids = pipeline.tokenize(phonemes)
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()

    seq_len = min(len(token_ids), max_tokens)
    token_ids = token_ids[:seq_len]

    voice_pack = pipeline.load_voice("af_heart")
    ref_s_tensor = voice_pack[len(token_ids)]
    if ref_s_tensor.dim() == 1:
        ref_s_tensor = ref_s_tensor.unsqueeze(0)

    # Fixed phases for deterministic comparison
    phases = torch.rand(1, NUM_HARMONICS) * 2 * torch.pi

    # Run PyTorch reference
    set_phases_fn(pytorch_model.decoder, phases)
    with torch.no_grad():
        py_audio, py_dur = pytorch_model.forward_with_tokens(
            torch.tensor([token_ids], dtype=torch.int64),
            ref_s_tensor, 1.0,
        )
    py_audio_np = py_audio.numpy()

    # Run CoreML
    input_ids = np.zeros((1, max_tokens), dtype=np.int32)
    input_ids[0, :seq_len] = token_ids
    mask = np.zeros((1, max_tokens), dtype=np.int32)
    mask[0, :seq_len] = 1

    coreml_out = coreml_model.predict({
        "input_ids": input_ids,
        "attention_mask": mask,
        "ref_s": ref_s_tensor.numpy().astype(np.float32),
        "speed": np.array([1.0], dtype=np.float32),
        "random_phases": phases.numpy().astype(np.float32),
    })

    coreml_audio = coreml_out["audio"].flatten()
    coreml_len = int(coreml_out["audio_length_samples"].flatten()[0])
    coreml_audio = coreml_audio[:coreml_len]

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

    # Save both for listening
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
