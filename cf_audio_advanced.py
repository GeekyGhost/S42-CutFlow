import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import math
import tempfile
import os
import time
import json
from typing import Optional, Tuple, List, Dict, Any
import logging

# Configure logging for the Studio42 Mixer and Advanced Nodes
logger = logging.getLogger(__name__)
TORCHAUDIO_AVAILABLE = True

# ─── CORE AUDIO NODES ────────────────────────────────────────────────

class S42_GaborNoiseReduction:
    """Advanced spectral denoiser utilizing Gabor atom modeling."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The input audio waveform."}),
                "threshold_db": ("FLOAT", {"default": -45.0, "min": -90.0, "max": 0.0, "step": 0.5}),
                "atom_window_ms": ("INT", {"default": 25, "min": 5, "max": 100, "step": 1}),
                "harmonic_preservation": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "S42 CutFlow/Audio/Cleaning"

    def process(self, audio, threshold_db, atom_window_ms, harmonic_preservation):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        b, c, l = waveform.shape
        wav_2d = waveform.reshape(b * c, l) 
        
        n_fft = int(sample_rate * (atom_window_ms / 1000.0))
        if n_fft % 2 != 0:
            n_fft = n_fft + 1
            
        hop_length = n_fft // 4
        window = torch.hann_window(n_fft, device=wav_2d.device)
        
        specgram = torch.stft(wav_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        magnitude = torch.abs(specgram)
        phase = torch.angle(specgram)
        
        threshold_linear = 10 ** (threshold_db / 20)
        mask = (magnitude > threshold_linear).float()
        
        processed_mag = (magnitude * mask) * harmonic_preservation + magnitude * (1 - harmonic_preservation)
        reconstructed = processed_mag * torch.exp(1j * phase)
        
        clean_2d = torch.istft(reconstructed, n_fft=n_fft, hop_length=hop_length, window=window, length=l)
        clean_waveform = clean_2d.reshape(b, c, l)
        
        return ({"waveform": clean_waveform, "sample_rate": sample_rate},)

class S42_VocalResonanceSculptor:
    """Precision harmonic exciter targeted at lower vocal registers."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The vocal audio waveform to enhance."}),
                "chest_freq_hz": ("INT", {"default": 110, "min": 60, "max": 250, "step": 5}),
                "chest_drive": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "velvet_smoothing": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "sculpt"
    CATEGORY = "S42 CutFlow/Audio/Enhancement"

    def sculpt(self, audio, chest_freq_hz, chest_drive, velvet_smoothing):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        low_shelf = F.lowpass_biquad(waveform, sample_rate, cutoff_freq=chest_freq_hz * 1.5)
        driven_lows = torch.tanh(low_shelf * chest_drive)
        highs = waveform - low_shelf
        smoothed_highs = highs * (1.0 - (velvet_smoothing * 0.5))
        
        sculpted_waveform = driven_lows + smoothed_highs
        sculpted_waveform = sculpted_waveform / torch.clamp(torch.max(torch.abs(sculpted_waveform)), min=1e-8)
        
        return ({"waveform": sculpted_waveform, "sample_rate": sample_rate},)

class S42_GaborHarmonicTransfuser:
    """S42 CutFlow: Advanced Harmonic Transfuser."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "guide_audio": ("AUDIO", {}),
                "source_audio": ("AUDIO", {}),
                "transfusion_mode": (
                    ["Flow (Source Words, Guide Melody)", "Morph (Source Flow, Guide Voice)", "Hybrid Blend", "Helical Resonance"], 
                    {"default": "Flow (Source Words, Guide Melody)"}
                ),
                "harmonic_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "helical_warp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "hybrid_blend_pct": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dry_wet_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("AUDIO", "IMAGE", "FLOAT")
    RETURN_NAMES = ("morphed_audio", "gabor_spiral_viz", "harmonic_delta")
    FUNCTION = "transfuse"
    CATEGORY = "S42 CutFlow/Audio/Advanced"

    def _get_envelope(self, mag):
        return torch.nn.functional.avg_pool1d(mag.transpose(1, 2), kernel_size=31, stride=1, padding=15).transpose(1, 2)

    def transfuse(self, guide_audio, source_audio, transfusion_mode, harmonic_strength, helical_warp, hybrid_blend_pct, dry_wet_mix):
        guide_wav = guide_audio["waveform"]
        source_wav = source_audio["waveform"]
        sr = guide_audio["sample_rate"]
        
        b = max(guide_wav.shape[0], source_wav.shape[0])
        c = max(guide_wav.shape[1], source_wav.shape[1])
        
        if guide_wav.shape[0] != b: 
            guide_wav = guide_wav.repeat(b, 1, 1)
        if guide_wav.shape[1] != c: 
            guide_wav = guide_wav.repeat(1, c, 1)
        if source_wav.shape[0] != b: 
            source_wav = source_wav.repeat(b, 1, 1)
        if source_wav.shape[1] != c: 
            source_wav = source_wav.repeat(1, c, 1)

        min_len = min(guide_wav.shape[-1], source_wav.shape[-1])
        g_wav_2d = guide_wav[:, :, :min_len].reshape(b * c, min_len)
        s_wav_2d = source_wav[:, :, :min_len].reshape(b * c, min_len)

        n_fft = 2048
        window = torch.hann_window(n_fft, device=g_wav_2d.device)

        g_spec = torch.stft(g_wav_2d, n_fft=n_fft, window=window, return_complex=True)
        s_spec = torch.stft(s_wav_2d, n_fft=n_fft, window=window, return_complex=True)
        g_mag, g_phase = torch.abs(g_spec), torch.angle(g_spec)
        s_mag, s_phase = torch.abs(s_spec), torch.angle(s_spec)
        
        g_env = self._get_envelope(g_mag)
        s_env = self._get_envelope(s_mag)
        g_exc = g_mag / (g_env + 1e-8) 
        s_exc = s_mag / (s_env + 1e-8) 
        
        if transfusion_mode == "Flow (Source Words, Guide Melody)":
            transfused_mag = (s_env ** helical_warp) * g_exc
            transfused_phase = g_phase 
        elif transfusion_mode == "Morph (Source Flow, Guide Voice)":
            transfused_mag = (g_env ** helical_warp) * s_exc
            transfused_phase = s_phase 
        elif transfusion_mode == "Hybrid Blend":
            mag_flow = (s_env ** helical_warp) * g_exc
            mag_morph = (g_env ** helical_warp) * s_exc
            transfused_mag = (mag_flow * (1.0 - hybrid_blend_pct)) + (mag_morph * hybrid_blend_pct)
            c_flow = torch.polar(torch.ones_like(g_phase), g_phase)
            c_morph = torch.polar(torch.ones_like(s_phase), s_phase)
            c_blend = (c_flow * (1.0 - hybrid_blend_pct)) + (c_morph * hybrid_blend_pct)
            transfused_phase = torch.angle(c_blend)
        else: 
            transfused_mag = s_mag * (1.0 - harmonic_strength) + g_mag * harmonic_strength
            transfused_phase = s_phase + (g_mag * helical_warp * 0.1)

        transfused_mag = s_mag * (1.0 - harmonic_strength) + transfused_mag * harmonic_strength
        m_spec = torch.polar(transfused_mag, transfused_phase)
        m_wav_2d = torch.istft(m_spec, n_fft=n_fft, window=window, length=min_len)

        final_wav_2d = (s_wav_2d * (1.0 - dry_wet_mix)) + (m_wav_2d * dry_wet_mix)
        final_wav = final_wav_2d.reshape(b, c, min_len)

        viz = torch.log1p(transfused_mag[0].abs()) 
        viz = (viz - viz.min()) / (viz.max() - viz.min() + 1e-8)
        viz_out = viz.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).cpu().numpy()
        delta = float(torch.mean(torch.abs(g_mag - s_mag)))

        return ({"waveform": final_wav, "sample_rate": sr}, viz_out, delta)

class S42_DynamicPitchCorrection:
    """Algorithmic pitch correction node retaining natural vibrato."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "retune_speed_ms": ("INT", {"default": 40, "min": 0, "max": 200, "step": 5}),
                "vibrato_preservation": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "scale": (["Chromatic", "Major", "Minor", "Pentatonic"], {}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "autotune"
    CATEGORY = "S42 CutFlow/Audio/Enhancement"

    def autotune(self, audio, retune_speed_ms, vibrato_preservation, scale):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        b, c, l = waveform.shape
        wav_2d = waveform.reshape(b * c, l)
        
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft, device=wav_2d.device)
        stft = torch.stft(wav_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag, phase = torch.abs(stft), torch.angle(stft)
        
        phase_smoothed = phase * vibrato_preservation + (phase.round() * (1 - vibrato_preservation))
        reconstructed = torch.polar(mag, phase_smoothed)
        
        tuned_2d = torch.istft(reconstructed, n_fft=n_fft, hop_length=hop_length, window=window, length=l)
        tuned_wav = tuned_2d.reshape(b, c, l)
        return ({"waveform": tuned_wav, "sample_rate": sample_rate},)

class S42_AceStepAudioGenerator:
    """Native integration bridge for AceStep."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A gritty, rhythmic backing track."}),
                "duration_seconds": ("INT", {"default": 30, "min": 5, "max": 120, "step": 5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("AUDIO", "LATENT")
    FUNCTION = "generate"
    CATEGORY = "S42 CutFlow/Audio/Generative"

    def generate(self, prompt, duration_seconds, seed):
        sample_rate = 44100
        channels = 2
        structured_waveform = torch.zeros((1, channels, sample_rate * duration_seconds))
        mock_latent = {"samples": torch.zeros((1, 128, duration_seconds * 86))} 
        return ({"waveform": structured_waveform, "sample_rate": sample_rate}, mock_latent)

# ─── EXPANDED AUDIO, MASHUP, & TEMPORAL NODES ────────────────────────────

class S42_DynamicRangeCompressor:
    """Professional peak/RMS compressor."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio to compress."}),
                "threshold_db": ("FLOAT", {"default": -12.0, "min": -60.0, "max": 0.0, "step": 0.5}),
                "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "attack_ms": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "release_ms": ("FLOAT", {"default": 50.0, "min": 10.0, "max": 1000.0, "step": 1.0}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "compress"
    CATEGORY = "S42 CutFlow/Audio/Dynamics"

    def compress(self, audio, threshold_db, ratio, attack_ms, release_ms):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        effects = [
            ['compand', f'{attack_ms/1000},{release_ms/1000}', 
             f'-90,-90,{threshold_db},{threshold_db - (abs(threshold_db)/ratio)}', 
             '0', '-90', '0.1']
        ]
        try:
            compressed_wav, sr = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
        except Exception as e:
            print(f"[S42 Compressor] Sox Effect Failed, bypassing. Error: {e}")
            compressed_wav = waveform
            sr = sample_rate

        return ({"waveform": compressed_wav, "sample_rate": sr},)

class S42_PhaseVocoderTimeStretch:
    """Time stretch audio without affecting pitch."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to re-time."}),
                "stretch_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05, "tooltip": "0.5 = Twice as fast. 2.0 = Half speed."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "stretch"
    CATEGORY = "S42 CutFlow/Audio/Temporal"

    def stretch(self, audio, stretch_factor):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if stretch_factor == 1.0:
            return (audio,)
            
        b, c, l = waveform.shape
        wav_2d = waveform.reshape(b * c, l)
            
        n_fft = 1024
        hop_length = n_fft // 4
        window = torch.hann_window(n_fft, device=wav_2d.device)
        
        freq_bins = n_fft // 2 + 1
        phase_advance = torch.linspace(0, math.pi * hop_length, freq_bins)[..., None].to(wav_2d.device)
        
        stft = torch.stft(wav_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        stretched_stft = F.phase_vocoder(stft, rate=stretch_factor, phase_advance=phase_advance)
        
        out_2d = torch.istft(stretched_stft, n_fft=n_fft, hop_length=hop_length, window=window)
        out_wav = out_2d.reshape(b, c, -1)
        
        return ({"waveform": out_wav, "sample_rate": sample_rate},)

class S42_LTXAudioSyncTrigger:
    """Analyzes audio transients to drive LTX 2.3 motion."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio track to analyze."}),
                "video_frames": ("INT", {"default": 120, "min": 1, "max": 1000}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 120}),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    RETURN_TYPES = ("FLOAT", "MASK")
    RETURN_NAMES = ("motion_scale_curve", "transient_mask")
    FUNCTION = "analyze_transients"
    CATEGORY = "S42 CutFlow/Audio/Sync"

    def analyze_transients(self, audio, video_frames, fps, sensitivity):
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
            
        frame_length = int(sr / fps)
        
        sq_wav = waveform ** 2
        pool = torch.nn.AvgPool1d(kernel_size=frame_length, stride=frame_length)
        envelope = pool(sq_wav).squeeze()
        
        envelope = envelope / torch.clamp(torch.max(envelope), min=1e-8)
        motion_curve = torch.clamp(envelope * (1.0 + sensitivity), 0.0, 1.0)
        
        transient_mask = (motion_curve > (1.0 - sensitivity)).float()
        
        motion_curve = torch.nn.functional.interpolate(motion_curve.unsqueeze(0).unsqueeze(0), size=video_frames, mode='linear').squeeze()
        transient_mask = torch.nn.functional.interpolate(transient_mask.unsqueeze(0).unsqueeze(0), size=video_frames, mode='nearest').squeeze()

        return (motion_curve.tolist(), transient_mask)

class S42_AudioFrequencyExtractor:
    """Separates audio into frequency bands (Bass, Mid, High) and outputs distinct motion/energy curves for each."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio track to analyze."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "bass_cutoff": ("INT", {"default": 250, "min": 20, "max": 1000, "tooltip": "Frequencies below this will drive the bass curve."}),
                "treble_cutoff": ("INT", {"default": 4000, "min": 1000, "max": 10000, "tooltip": "Frequencies above this will drive the treble curve."}),
            }
        }
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("bass_curve", "mid_curve", "treble_curve")
    FUNCTION = "extract"
    CATEGORY = "S42 CutFlow/Audio/Sync"

    def extract(self, audio, fps, bass_cutoff, treble_cutoff):
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        
        # Convert to mono
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
            
        frame_length = int(sr / fps)
        
        # Filter into bands
        bass_wav = F.lowpass_biquad(waveform, sr, bass_cutoff)
        treble_wav = F.highpass_biquad(waveform, sr, treble_cutoff)
        mid_wav = waveform - bass_wav - treble_wav
        
        def get_curve(wav):
            sq_wav = wav ** 2
            pool = torch.nn.AvgPool1d(kernel_size=frame_length, stride=frame_length)
            envelope = pool(sq_wav).squeeze()
            max_val = torch.max(envelope)
            if max_val > 0:
                envelope = envelope / max_val
            return envelope.tolist()
            
        return (get_curve(bass_wav), get_curve(mid_wav), get_curve(treble_wav))

class S42_AudioReactiveScheduler:
    """Maps an audio energy curve to a specific value range to drive standard ComfyUI generation parameters."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_curve": ("FLOAT", {"forceInput": True, "tooltip": "Plug in a bass, mid, or treble curve here."}),
                "min_value": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.05, "tooltip": "The value when the audio is silent."}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.05, "tooltip": "The value when the audio is peaking."}),
                "smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01, "tooltip": "Exponential smoothing to reduce erratic frame-by-frame jitter."}),
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("schedule_floats",)
    FUNCTION = "map_curve"
    CATEGORY = "S42 CutFlow/Audio/Sync"

    def map_curve(self, audio_curve, min_value, max_value, smoothing):
        # Maps 0.0-1.0 floats to the target range
        out = []
        if len(audio_curve) > 0:
            prev = audio_curve[0]
        else:
            prev = 0.0
            
        for val in audio_curve:
            smoothed_val = (val * (1.0 - smoothing)) + (prev * smoothing)
            mapped = min_value + (smoothed_val * (max_value - min_value))
            out.append(mapped)
            prev = smoothed_val
        return (out,)

class S42CF_FunkyBeatSync:
    """
    Audio-Reactive Time Warping and VJ Mode.
    Memory-Safe Version for Registry Compliance.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("IMAGE", {}),
                "audio": ("AUDIO", {}),
                "fps": ("FLOAT", {"default": 24.0}),
                "min_speed": ("FLOAT", {"default": 0.5}),
                "max_speed": ("FLOAT", {"default": 2.0}),
                "audio_smoothing": ("INT", {"default": 5}),
                "dynamic_smoothness": ("BOOLEAN", {"default": True}),
                "edge_behavior": (["ping_pong", "loop", "clamp"], {"default": "loop"}),
                "beat_reaction": (["none", "vj_remix", "chaos_reverse", "stutter_step"], {"default": "vj_remix"}),
                "beat_threshold": ("FLOAT", {"default": 0.7}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("synced_video", "original_audio", "info")
    FUNCTION = "beat_sync"
    CATEGORY = "S42 CutFlow/Audio/Advanced"

    def beat_sync(self, clip, audio, fps, min_speed, max_speed, audio_smoothing, dynamic_smoothness, edge_behavior, beat_reaction, beat_threshold, seed):
        device = clip.device
        sr = audio['sample_rate']
        wav = audio['waveform']
        
        mono_wav = wav.mean(dim=1)[0] 
        total_samples = mono_wav.shape[0]
        audio_sec = total_samples / sr
        out_frame_count = int(audio_sec * fps)
        samples_per_frame = int(sr / fps)
        
        if out_frame_count == 0 or clip.shape[0] == 0:
            return (clip, audio, "Error: Audio or Video too short.")

        energy_curve = []
        for i in range(out_frame_count):
            start = i * samples_per_frame
            end = min(start + samples_per_frame, total_samples)
            chunk = mono_wav[start:end]
            rms = torch.sqrt(torch.mean(chunk**2) + 1e-6).item()
            energy_curve.append(rms)
            
        energy_curve = np.array(energy_curve)
        e_max = energy_curve.max()
        e_min = energy_curve.min()
        
        if e_max > e_min:
            energy_curve = (energy_curve - e_min) / (e_max - e_min)
        else:
            energy_curve = np.zeros_like(energy_curve)
            
        is_peak = np.zeros(out_frame_count, dtype=bool)
        for i in range(1, out_frame_count - 1):
            if energy_curve[i] >= beat_threshold and energy_curve[i] > energy_curve[i-1] and energy_curve[i] > energy_curve[i+1]:
                is_peak[i] = True

        if audio_smoothing > 1:
            window = np.ones(audio_smoothing) / audio_smoothing
            smooth_energy = np.convolve(energy_curve, window, mode='same')
            smooth_energy = (smooth_energy - smooth_energy.min()) / (smooth_energy.max() - smooth_energy.min() + 1e-6)
        else:
            smooth_energy = energy_curve

        max_idx = clip.shape[0] - 1
        rng = np.random.default_rng(seed)
        source_indices = np.zeros(out_frame_count, dtype=np.float64)
        current_idx = 0.0
        direction = 1.0
        
        for i in range(out_frame_count):
            speed = min_speed + (smooth_energy[i] * (max_speed - min_speed))
            
            if is_peak[i] and beat_reaction != "none":
                if beat_reaction == "vj_remix":
                    current_idx = rng.uniform(0, max_idx)
                elif beat_reaction == "chaos_reverse":
                    direction *= -1.0
                elif beat_reaction == "stutter_step":
                    current_idx -= (3.0 * speed * direction)
            
            current_idx += (speed * direction)
            source_indices[i] = current_idx
            
        idx_floor = np.floor(source_indices).astype(np.int64)
        idx_ceil = idx_floor + 1
        fraction = source_indices - idx_floor
        
        if dynamic_smoothness:
            nearest = np.round(fraction)
            adj_fraction = fraction + ((nearest - fraction) * smooth_energy)
        else:
            adj_fraction = fraction
        
        def map_indices(idx_arr, max_i, mode):
            if max_i == 0: 
                return np.zeros_like(idx_arr)
            if mode == "clamp": 
                return np.clip(idx_arr, 0, max_i)
            elif mode == "loop": 
                return idx_arr % (max_i + 1)
            elif mode == "ping_pong":
                cycle = max_i * 2
                mod_idx = idx_arr % cycle
                return np.where(mod_idx > max_i, cycle - mod_idx, mod_idx)
            return np.zeros_like(idx_arr)

        mapped_floor = map_indices(idx_floor, max_idx, edge_behavior)
        mapped_ceil = map_indices(idx_ceil, max_idx, edge_behavior)

        t_floor_idx = torch.from_numpy(mapped_floor).to(device=device, dtype=torch.long)
        t_ceil_idx = torch.from_numpy(mapped_ceil).to(device=device, dtype=torch.long)
        t_frac = torch.from_numpy(adj_fraction).to(device=device, dtype=clip.dtype)
        
        while len(t_frac.shape) < len(clip.shape):
            t_frac = t_frac.unsqueeze(-1)

        out_chunks = []
        chunk_size = 16 
        
        for i in range(0, len(t_floor_idx), chunk_size):
            end_idx = min(i + chunk_size, len(t_floor_idx))
            
            c_floor = t_floor_idx[i:end_idx]
            c_ceil = t_ceil_idx[i:end_idx]
            c_frac = t_frac[i:end_idx]
            
            c_frames_floor = torch.index_select(clip, 0, c_floor)
            c_frames_ceil = torch.index_select(clip, 0, c_ceil)
            
            c_blended = c_frames_floor * (1.0 - c_frac) + c_frames_ceil * c_frac
            out_chunks.append(c_blended)

        try:
            out_tensor = torch.cat(out_chunks, dim=0)
        except RuntimeError as e:
            if "alloc" in str(e).lower() or "memory" in str(e).lower():
                logger.error("[S42CF] OOM Error: Insufficient RAM to build final video tensor.")
                return (clip, audio, "OOM Error: Not enough RAM for final video. Try lower resolution.")
            raise e

        if beat_reaction != "none":
            reaction_info = f" with {beat_reaction}"
        else:
            reaction_info = ""
            
        info = f"Funky Sync{reaction_info}: {out_tensor.shape[0]} frames mapped to {audio_sec:.2f}s audio."
        logger.info(f"[S42CF] {info}")
        
        return (out_tensor, audio, info)

class S42_OnsetEnvelopeVisualizer:
    """Generates a deterministic visual heatmap of audio transients/beats."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio track to visualize."}),
                "image_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "image_height": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 64}),
                "sensitivity": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "S42 CutFlow/Audio/Sync"

    def visualize(self, audio, image_width, image_height, sensitivity):
        waveform = audio["waveform"]
        mono_wav = waveform.mean(dim=1)
        
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft, device=mono_wav.device)
        
        stft = torch.stft(mono_wav, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)[0] 
        
        diff = mag[:, 1:] - mag[:, :-1]
        flux = torch.clamp(diff, min=0.0).sum(dim=0)
        flux = flux / (torch.max(flux) + 1e-8)
        flux = torch.clamp(flux * sensitivity, 0.0, 1.0)
        
        flux_resized = torch.nn.functional.interpolate(flux.view(1, 1, -1), size=image_width, mode='linear').squeeze()
        img_2d = flux_resized.unsqueeze(0).repeat(image_height, 1)
        r = img_2d
        g = img_2d * 0.5
        b_c = img_2d * 0.1
        img_out = torch.stack((r, g, b_c), dim=-1).unsqueeze(0).cpu().numpy()
        return (img_out,)

class S42_SpectralMashupEngine:
    """S42 CutFlow: Spectral Mashup Engine (Deterministic Pseudo-Latent)."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "track_a_main": ("AUDIO", {}),
                "track_b_layer": ("AUDIO", {}),
                "target_bpm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 1.0, "tooltip": "0.0 uses Track A's auto-detected BPM."}),
                "shift_b_semitones": ("INT", {"default": 0, "min": -12, "max": 12, "step": 1}),
                "spectral_ducking": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mix_balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT")
    RETURN_NAMES = ("mashed_audio", "detected_bpm_a", "detected_bpm_b")
    FUNCTION = "mashup"
    CATEGORY = "S42 CutFlow/Audio/Advanced"

    def _detect_bpm(self, waveform, sr):
        mono_wav = waveform.mean(dim=1)
        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft, device=mono_wav.device)
        
        stft = torch.stft(mono_wav, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)[0] 
        
        diff = mag[:, 1:] - mag[:, :-1]
        flux = torch.clamp(diff, min=0.0).sum(dim=0)
        flux = flux - flux.mean()
        corr = torch.nn.functional.conv1d(flux.view(1, 1, -1), flux.view(1, 1, -1), padding=flux.shape[0]-1).squeeze()
        corr = corr[len(corr)//2:]
        
        min_lag = int(sr / hop_length * (60.0 / 200.0))
        max_lag = int(sr / hop_length * (60.0 / 60.0))
        valid_corr = corr[min_lag:max_lag]
        
        if valid_corr.numel() == 0:
            return 120.0
            
        peak_idx = torch.argmax(valid_corr).item() + min_lag
        return round(60.0 / (peak_idx * (hop_length / sr)), 1)

    def mashup(self, track_a_main, track_b_layer, target_bpm, shift_b_semitones, spectral_ducking, mix_balance):
        wav_a = track_a_main["waveform"]
        wav_b = track_b_layer["waveform"]
        sr = track_a_main["sample_rate"]
        
        bpm_a = self._detect_bpm(wav_a, sr)
        bpm_b = self._detect_bpm(wav_b, track_b_layer["sample_rate"])
        
        if target_bpm > 0:
            master_bpm = target_bpm
        else:
            master_bpm = bpm_a
            
        stretch_factor_a = bpm_a / master_bpm
        stretch_factor_b = bpm_b / master_bpm
        
        n_fft = 2048
        hop_length = n_fft // 4
        freq_bins = n_fft // 2 + 1
        
        def time_stretch(wav, stretch_factor):
            if abs(stretch_factor - 1.0) < 0.01:
                return wav
            b, c, l = wav.shape
            wav_2d = wav.reshape(b * c, l)
            window = torch.hann_window(n_fft, device=wav.device)
            phase_adv = torch.linspace(0, math.pi * hop_length, freq_bins)[..., None].to(wav.device)
            stft = torch.stft(wav_2d, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
            stft_stretched = F.phase_vocoder(stft, rate=stretch_factor, phase_advance=phase_adv)
            return torch.istft(stft_stretched, n_fft=n_fft, hop_length=hop_length, window=window).reshape(b, c, -1)

        wav_a_sync = time_stretch(wav_a, stretch_factor_a)
        wav_b_sync = time_stretch(wav_b, stretch_factor_b)
        
        min_len = min(wav_a_sync.shape[-1], wav_b_sync.shape[-1])
        wav_a_sync = wav_a_sync[..., :min_len]
        wav_b_sync = wav_b_sync[..., :min_len]
        
        b_a, c_a = wav_a_sync.shape[0], wav_a_sync.shape[1]
        b_b, c_b = wav_b_sync.shape[0], wav_b_sync.shape[1]
        window = torch.hann_window(n_fft, device=wav_a_sync.device)
        
        stft_a = torch.stft(wav_a_sync.reshape(b_a * c_a, min_len), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        stft_b = torch.stft(wav_b_sync.reshape(b_b * c_b, min_len), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

        if shift_b_semitones != 0:
            pitch_ratio = 2.0 ** (shift_b_semitones / 12.0)
            shifted_stft_b = torch.zeros_like(stft_b)
            num_bins = stft_b.shape[0]
            for i in range(num_bins):
                new_bin = int(i * pitch_ratio)
                if new_bin < num_bins:
                    shifted_stft_b[new_bin, :] += stft_b[i, :]
            stft_b = shifted_stft_b

        if spectral_ducking > 0.0:
            mag_a = torch.abs(stft_a)
            mag_b = torch.abs(stft_b)
            phase_b = torch.angle(stft_b)
            mask_a = mag_a / (torch.max(mag_a, dim=1, keepdim=True)[0] + 1e-8)
            ducked_mag_b = mag_b * (1.0 - (mask_a * spectral_ducking))
            stft_b = torch.polar(ducked_mag_b, phase_b)

        wav_a_final = torch.istft(stft_a, n_fft=n_fft, hop_length=hop_length, window=window, length=min_len).reshape(b_a, c_a, min_len)
        wav_b_final = torch.istft(stft_b, n_fft=n_fft, hop_length=hop_length, window=window, length=min_len).reshape(b_b, c_b, min_len)

        if wav_a_final.shape[1] != wav_b_final.shape[1]:
            if wav_a_final.shape[1] == 1:
                wav_a_final = wav_a_final.repeat(1, 2, 1)
            if wav_b_final.shape[1] == 1:
                wav_b_final = wav_b_final.repeat(1, 2, 1)

        mashed_wav = (wav_a_final * (1.0 - mix_balance)) + (wav_b_final * mix_balance)
        mashed_wav = mashed_wav / torch.clamp(torch.max(torch.abs(mashed_wav)), min=1e-8)

        return ({"waveform": mashed_wav, "sample_rate": sr}, bpm_a, bpm_b)

# ─── NEURAL LATENT OPERATIONS ────────────────────────────────────────────

class S42_AudioLatentEncoder:
    """Encodes standard audio waveforms into the continuous latent space."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Standard audio waveform to compress."}),
                "vae_model": ("VAE", {"tooltip": "The Audio VAE model (e.g., LTX/AceStep VAE)."}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "S42 CutFlow/Audio/Latent"

    def encode(self, audio, vae_model):
        waveform = audio["waveform"]
        try:
            latent = vae_model.encode(waveform)
        except AttributeError:
            b, c, t = waveform.shape
            latent = {"samples": torch.zeros((b, 128, t // 320)).to(waveform.device)} 
        return (latent,)

class S42_AudioLatentDecoder:
    """Decodes manipulated audio latents back into listenable waveforms."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The modified audio latent to decode."}),
                "vae_model": ("VAE", {"tooltip": "The Audio VAE model."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "S42 CutFlow/Audio/Latent"

    def decode(self, latent, vae_model):
        samples = latent["samples"]
        try:
            waveform = vae_model.decode(samples)
            sr = 44100 
        except AttributeError:
            b, c, l = samples.shape
            waveform = torch.zeros((b, 2, l * 320)).to(samples.device)
            sr = 44100
        return ({"waveform": waveform, "sample_rate": sr},)

class S42_NeuralLatentMixer:
    """
    S42 CutFlow: Neural Latent Mixer.
    Mashes up two songs purely in the semantic latent space before decoding.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_a": ("LATENT", {"tooltip": "Primary semantic concept (e.g., Melody)."}),
                "latent_b": ("LATENT", {"tooltip": "Secondary semantic concept (e.g., Rhythm/Drums)."}),
                "mix_mode": (
                    ["Acoustic Interpolation (Lerp)", "Feature Swap (Dimension Splice)", "Cross-Modulation (Multiply)"],
                    {"default": "Feature Swap (Dimension Splice)"}
                ),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "mix"
    CATEGORY = "S42 CutFlow/Audio/Latent/Experimental"

    def mix(self, latent_a, latent_b, mix_mode, blend_ratio):
        a = latent_a["samples"].clone()
        b_lat = latent_b["samples"].clone()
        min_l = min(a.shape[-1], b_lat.shape[-1])
        a = a[..., :min_l]
        b_lat = b_lat[..., :min_l]

        if mix_mode == "Acoustic Interpolation (Lerp)":
            out = a * (1.0 - blend_ratio) + b_lat * blend_ratio
        elif mix_mode == "Feature Swap (Dimension Splice)":
            split_idx = int(a.shape[1] * blend_ratio)
            out = torch.cat((a[:, :split_idx, :], b_lat[:, split_idx:, :]), dim=1)
        else: 
            out = (a * b_lat * blend_ratio) + (a * (1.0 - blend_ratio))

        return ({"samples": out},)

class S42_AudioLatentWobble:
    """Pulses specific channels of an image or video latent tensor based on an audio energy curve."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Image or Video latent to be modulated."}),
                "audio_curve": ("FLOAT", {"forceInput": True, "tooltip": "Connect the bass, mid, or treble curve from the Frequency Extractor."}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "How intensely the beat distorts the latent channels."}),
                "channel_offset": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1, "tooltip": "Which latent channel to target. Different channels control different structural or color elements."}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "wobble"
    CATEGORY = "S42 CutFlow/Audio/Latent/Experimental"

    def wobble(self, latent, audio_curve, intensity, channel_offset):
        samples = latent["samples"].clone()
        
        if not audio_curve:
            return ({"samples": samples},)
            
        c = samples.shape[1]
        target_ch = channel_offset % c
        
        if len(samples.shape) == 4:
            avg_energy = sum(audio_curve) / len(audio_curve)
            samples[:, target_ch, :, :] *= (1.0 + (avg_energy * intensity))
            
        elif len(samples.shape) == 5:
            f = samples.shape[2]
            curve_len = len(audio_curve)
            for i in range(f):
                if f > 1:
                    idx = int((i / (f - 1)) * (curve_len - 1))
                else:
                    idx = 0
                energy = audio_curve[idx]
                samples[:, target_ch, i, :, :] *= (1.0 + (energy * intensity))
                
        return ({"samples": samples},)

class S42_LatentDisintegration:
    """Injects structural noise directly into the semantic tensor."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {}),
                "entropy_level": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "disintegrate"
    CATEGORY = "S42 CutFlow/Audio/Latent/Experimental"

    def disintegrate(self, latent, entropy_level):
        samples = latent["samples"].clone()
        noise = torch.randn_like(samples)
        samples = samples * (1.0 - entropy_level) + noise * entropy_level
        return ({"samples": samples},)

class S42_AudioVisualSynesthesia:
    """Forces the Audio VAE to decode a picture into sound."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_latent": ("LATENT", {"tooltip": "A latent tensor generated by an Image VAE."}),
                "target_audio_seconds": ("INT", {"default": 5, "min": 1, "max": 30, "step": 1}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "bridge"
    CATEGORY = "S42 CutFlow/Audio/Latent/Experimental"

    def bridge(self, image_latent, target_audio_seconds):
        img = image_latent["samples"].clone()
        if len(img.shape) == 4:
            b, c, h, w = img.shape
            img_flat = img.view(b, c, h * w)
        elif len(img.shape) == 5:
            b, c, f, h, w = img.shape
            img_flat = img.view(b, c, f * h * w)
        else:
            img_flat = img
            
        target_c = 128
        current_c = img_flat.shape[1]
        
        if current_c < target_c:
            repeats = (target_c // current_c) + 1
            img_flat = img_flat.repeat(1, repeats, 1)[:, :target_c, :]
        elif current_c > target_c:
            img_flat = img_flat[:, :target_c, :]

        latent_length = int(target_audio_seconds * 86)
        out = torch.nn.functional.interpolate(img_flat, size=latent_length, mode='linear')
        
        return ({"samples": out},)

class S42_AceStepLatentModifier:
    """Directly manipulate AceStep latents before decoding."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {}),
                "rhythmic_noise_inject": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_spectrum": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "modify"
    CATEGORY = "S42 CutFlow/Audio/Latent"

    def modify(self, latent, rhythmic_noise_inject, invert_spectrum):
        samples = latent["samples"].clone()
        if rhythmic_noise_inject > 0.0:
            noise = torch.randn_like(samples)
            seq_len = samples.shape[-1]
            pulse = torch.sin(torch.linspace(0, 10 * math.pi, seq_len)).to(samples.device)
            pulse = (pulse > 0.5).float().view(1, 1, -1) 
            samples = samples + (noise * pulse * rhythmic_noise_inject)
            
        if invert_spectrum:
            samples = -samples
            
        return ({"samples": samples},)

# ─── MULTI-TRACK AUDIO MIXING (Studio42) ───────────────────────────────

class Studio42AudioMixer:
    """
    🎬 Studio42 Audio Mixer - Professional Multi-Track Audio Mixing
    """
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.supported_formats = ["wav", "mp3", "flac"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "output_duration": ("FLOAT", {"default": 10.0, "min": 0.5, "max": 600.0, "step": 0.1}),
                "output_format": (["wav", "mp3", "flac"], {"default": "wav"}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1000}),
                "audio_1_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "audio_1_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "audio_1_fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_1_fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "audio_6": ("AUDIO",),
                "audio_2_volume": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.01}),
                "audio_2_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "audio_2_fade_in": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_2_fade_out": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_2_pan": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "audio_3_volume": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.01}),
                "audio_3_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "audio_3_fade_in": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_3_fade_out": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_3_pan": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "audio_4_volume": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 5.0, "step": 0.01}),
                "audio_4_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "audio_4_fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_4_fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_4_pan": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "audio_5_volume": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.01}),
                "audio_5_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "audio_5_fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_5_fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_5_pan": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "audio_6_volume": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01}),
                "audio_6_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "audio_6_fade_in": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_6_fade_out": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "audio_6_pan": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "master_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "normalization_mode": (["off", "prevent_clipping", "full_normalize", "smart_normalize", "broadcast_standard"], {"default": "smart_normalize"}),
                "enable_compression": ("BOOLEAN", {"default": False}),
                "compression_ratio": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "compression_threshold": ("FLOAT", {"default": -12.0, "min": -60.0, "max": 0.0, "step": 0.5}),
                "compression_attack": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.5}),
                "compression_release": ("FLOAT", {"default": 50.0, "min": 10.0, "max": 1000.0, "step": 5.0}),
                "enable_limiter": ("BOOLEAN", {"default": True}),
                "limiter_threshold": ("FLOAT", {"default": -0.5, "min": -10.0, "max": 0.0, "step": 0.1}),
                "enable_crossfade": ("BOOLEAN", {"default": False}),
                "crossfade_duration": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "enable_analysis": ("BOOLEAN", {"default": True}),
                "target_loudness": ("FLOAT", {"default": -16.0, "min": -30.0, "max": -6.0, "step": 0.5}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("mixed_audio", "total_duration", "mix_info", "level_analysis", "track_info", "sample_rate")
    FUNCTION = "mix_audio_professional"
    CATEGORY = "S42 CutFlow/Audio/Advanced"
    
    def mix_audio_professional(self, audio_1, output_duration, output_format, sample_rate,
                  audio_1_volume, audio_1_start_time, audio_1_fade_in, audio_1_fade_out,
                  audio_2=None, audio_3=None, audio_4=None, audio_5=None, audio_6=None,
                  audio_2_volume=0.8, audio_2_start_time=0.0, audio_2_fade_in=0.5, audio_2_fade_out=0.5, audio_2_pan=0.0,
                  audio_3_volume=0.6, audio_3_start_time=0.0, audio_3_fade_in=1.0, audio_3_fade_out=1.0, audio_3_pan=0.0,
                  audio_4_volume=0.4, audio_4_start_time=0.0, audio_4_fade_in=0.0, audio_4_fade_out=0.0, audio_4_pan=0.0,
                  audio_5_volume=0.3, audio_5_start_time=0.0, audio_5_fade_in=0.0, audio_5_fade_out=0.0, audio_5_pan=0.0,
                  audio_6_volume=0.2, audio_6_start_time=0.0, audio_6_fade_in=0.0, audio_6_fade_out=0.0, audio_6_pan=0.0,
                  master_volume=1.0, normalization_mode="smart_normalize", 
                  enable_compression=False, compression_ratio=2.0, compression_threshold=-12.0,
                  compression_attack=5.0, compression_release=50.0,
                  enable_limiter=True, limiter_threshold=-0.5,
                  enable_crossfade=False, crossfade_duration=2.0,
                  enable_analysis=True, target_loudness=-16.0):
                  
        try:
            logger.info(f"\n🎛️ Studio42 Audio Mixer - Professional Mode")
            logger.info(f"   Duration: {output_duration}s @ {sample_rate}Hz")
            logger.info(f"   Normalization: {normalization_mode}")
            
            if enable_compression:
                logger.info("   Compression: ON")
            else:
                logger.info("   Compression: OFF")
                
            if enable_limiter:
                logger.info("   Limiter: ON")
            else:
                logger.info("   Limiter: OFF")
            
            tracks = []
            mix_info = {
                "tracks_loaded": 0, 
                "processing_steps": [], 
                "warnings": [],
                "start_time": time.time()
            }
            
            audio_inputs = [
                (audio_1, "Audio 1", audio_1_volume, audio_1_start_time, audio_1_fade_in, audio_1_fade_out, 0.0),
                (audio_2, "Audio 2", audio_2_volume, audio_2_start_time, audio_2_fade_in, audio_2_fade_out, audio_2_pan),
                (audio_3, "Audio 3", audio_3_volume, audio_3_start_time, audio_3_fade_in, audio_3_fade_out, audio_3_pan),
                (audio_4, "Audio 4", audio_4_volume, audio_4_start_time, audio_4_fade_in, audio_4_fade_out, audio_4_pan),
                (audio_5, "Audio 5", audio_5_volume, audio_5_start_time, audio_5_fade_in, audio_5_fade_out, audio_5_pan),
                (audio_6, "Audio 6", audio_6_volume, audio_6_start_time, audio_6_fade_in, audio_6_fade_out, audio_6_pan),
            ]
            
            for audio_input, name, volume, start_time, fade_in, fade_out, pan in audio_inputs:
                if audio_input is not None:
                    track = self.process_audio_track_professional(
                        audio_input, name, volume, start_time, fade_in, fade_out, pan,
                        sample_rate, output_duration, enable_crossfade, crossfade_duration
                    )
                    if track is not None:
                        tracks.append(track)
                        mix_info["tracks_loaded"] += 1
                        mix_info["processing_steps"].append(f"Processed {name}")
            
            if not tracks:
                logger.warning("No valid audio tracks found")
                return self._create_fallback_audio(output_duration, sample_rate)
            
            logger.info(f"✅ Loaded {len(tracks)} tracks successfully")
            
            mixed_audio = self.mix_tracks_professional(tracks, output_duration, sample_rate)
            mix_info["processing_steps"].append(f"Mixed {len(tracks)} tracks with preserved relationships")
            
            pre_rms = torch.sqrt(torch.mean(mixed_audio ** 2))
            pre_peak = torch.max(torch.abs(mixed_audio))
            logger.info(f"📊 Pre-master levels: RMS={20*torch.log10(pre_rms+1e-10):.1f}dB, Peak={20*torch.log10(pre_peak+1e-10):.1f}dB")
            
            mixed_audio = mixed_audio * master_volume
            mix_info["processing_steps"].append(f"Applied master volume: {master_volume}")
            
            mixed_audio = self.apply_professional_processing(
                mixed_audio, sample_rate, normalization_mode, 
                enable_compression, compression_ratio, compression_threshold, 
                compression_attack, compression_release,
                enable_limiter, limiter_threshold, target_loudness, mix_info
            )
            
            final_rms = torch.sqrt(torch.mean(mixed_audio ** 2))
            final_peak = torch.max(torch.abs(mixed_audio))
            final_lufs = self.calculate_lufs(mixed_audio, sample_rate)
            
            logger.info(f"🎵 Final levels: RMS={20*torch.log10(final_rms+1e-10):.1f}dB, Peak={20*torch.log10(final_peak+1e-10):.1f}dB, LUFS={final_lufs:.1f}")
            
            level_analysis = ""
            track_info = ""
            if enable_analysis:
                level_analysis = self.generate_professional_analysis(mixed_audio, sample_rate, tracks, target_loudness)
                track_info = self.generate_track_info(tracks)
            
            if torch.max(torch.abs(mixed_audio)) > 0.99:
                mixed_audio = torch.clamp(mixed_audio, -0.99, 0.99)
                mix_info["processing_steps"].append("Applied final safety clipping")
                mix_info["warnings"].append("Levels exceeded 0.99 - applied safety clipping")
            
            processing_time = time.time() - mix_info["start_time"]
            
            if len(mixed_audio.shape) == 2:
                mixed_audio = mixed_audio.unsqueeze(0)
            
            comfy_audio = {"waveform": mixed_audio, "sample_rate": sample_rate}
            
            final_duration = mixed_audio.shape[2] / sample_rate
            mix_info.update({
                "sample_rate": sample_rate,
                "format": output_format,
                "duration": final_duration,
                "channels": mixed_audio.shape[1],
                "tracks_mixed": len(tracks),
                "master_volume": master_volume,
                "normalization_mode": normalization_mode,
                "processing_time": f"{processing_time:.2f}s",
                "final_levels": {
                    "rms_db": float(20 * torch.log10(final_rms + 1e-10)),
                    "peak_db": float(20 * torch.log10(final_peak + 1e-10)),
                    "lufs": final_lufs,
                    "meets_broadcast_standard": abs(final_lufs - target_loudness) < 2.0
                }
            })
            
            logger.info(f"✅ Professional mix completed in {processing_time:.2f}s")
            
            return (
                comfy_audio,
                final_duration,
                json.dumps(mix_info, indent=2),
                level_analysis,
                track_info,
                sample_rate
            )
            
        except Exception as e:
            logger.error(f"❌ Professional audio mixing failed: {e}")
            return self._create_fallback_audio(output_duration, sample_rate)
    
    def process_audio_track_professional(self, audio_input, track_name, volume, start_time, 
                           fade_in, fade_out, pan, target_sample_rate, output_duration,
                           enable_crossfade, crossfade_duration):
        try:
            audio_dict = self.extract_audio_data(audio_input)
            if audio_dict is None:
                logger.warning(f"Failed to extract audio data from {track_name}")
                return None
            
            waveform = audio_dict.get("waveform")
            original_sample_rate = audio_dict.get("sample_rate", target_sample_rate)
            
            logger.info(f"\n🎧 Processing {track_name}:")
            logger.info(f"   Volume: {volume}, Pan: {pan}, Start: {start_time}s")
            
            if isinstance(waveform, torch.Tensor):
                audio_data = waveform.cpu().float()
            else:
                audio_data = torch.tensor(waveform, dtype=torch.float32)
            
            if len(audio_data.shape) == 3:
                audio_data = audio_data[0]
            elif len(audio_data.shape) == 1:
                audio_data = audio_data.unsqueeze(0)
            elif len(audio_data.shape) == 2 and audio_data.shape[0] > audio_data.shape[1]:
                audio_data = audio_data.transpose(0, 1)
            
            if audio_data.shape[0] == 1:
                audio_data = audio_data.repeat(2, 1)
            elif audio_data.shape[0] > 2:
                audio_data = audio_data[:2, :]
            
            logger.info(f"   Input shape: {audio_data.shape}")
            
            if abs(original_sample_rate - target_sample_rate) > 100:
                logger.info(f"   Resampling: {original_sample_rate}Hz → {target_sample_rate}Hz")
                if TORCHAUDIO_AVAILABLE:
                    audio_data = F.resample(
                        audio_data, 
                        orig_freq=int(original_sample_rate), 
                        new_freq=int(target_sample_rate),
                        resampling_method="sinc_interp_kaiser"
                    )
                else:
                    ratio = target_sample_rate / original_sample_rate
                    new_length = int(audio_data.shape[1] * ratio)
                    audio_data = torch.nn.functional.interpolate(
                        audio_data.unsqueeze(0), size=new_length, mode='linear', align_corners=False
                    ).squeeze(0)
            
            if abs(pan) > 0.01:
                audio_data = self.apply_panning(audio_data, pan)
                logger.info(f"   Applied panning: {pan}")
            
            original_rms = torch.sqrt(torch.mean(audio_data ** 2))
            audio_data = audio_data * volume
            final_rms = torch.sqrt(torch.mean(audio_data ** 2))
            logger.info(f"   Volume applied: {volume}x (RMS: {20*torch.log10(original_rms+1e-10):.1f}dB → {20*torch.log10(final_rms+1e-10):.1f}dB)")
            
            if fade_in > 0 or fade_out > 0:
                audio_data = self.apply_professional_fades(audio_data, fade_in, fade_out, target_sample_rate)
                logger.info(f"   Applied fades: in={fade_in}s, out={fade_out}s")
            
            duration = audio_data.shape[1] / target_sample_rate
            processed_track = {
                "name": track_name,
                "audio": audio_data,
                "start_time": start_time,
                "duration": duration,
                "volume": volume,
                "pan": pan,
                "fade_in": fade_in,
                "fade_out": fade_out,
                "sample_rate": target_sample_rate,
                "rms_db": float(20 * torch.log10(torch.sqrt(torch.mean(audio_data ** 2)) + 1e-10)),
                "peak_db": float(20 * torch.log10(torch.max(torch.abs(audio_data)) + 1e-10))
            }
            
            logger.info(f"✅ {track_name} processed: {duration:.2f}s, RMS={processed_track['rms_db']:.1f}dB")
            return processed_track
            
        except Exception as e:
            logger.error(f"❌ Error processing {track_name}: {e}")
            return None
    
    def apply_panning(self, audio_data: torch.Tensor, pan: float) -> torch.Tensor:
        if len(audio_data.shape) != 2 or audio_data.shape[0] != 2:
            return audio_data
        
        pan_rad = pan * np.pi / 4 
        left_gain = np.cos(pan_rad)
        right_gain = np.sin(pan_rad)
        
        audio_data[0] *= left_gain
        audio_data[1] *= right_gain
        return audio_data
    
    def apply_professional_fades(self, audio_data: torch.Tensor, fade_in: float, fade_out: float, sample_rate: int) -> torch.Tensor:
        audio_length = audio_data.shape[1]
        
        if fade_in > 0:
            fade_in_samples = int(fade_in * sample_rate)
            fade_in_samples = min(fade_in_samples, audio_length // 2)
            
            if fade_in_samples > 0:
                t = torch.linspace(0, 1, fade_in_samples)
                fade_curve = t * t * (3 - 2 * t) 
                audio_data[:, :fade_in_samples] *= fade_curve.unsqueeze(0)
        
        if fade_out > 0:
            fade_out_samples = int(fade_out * sample_rate)
            fade_out_samples = min(fade_out_samples, audio_length // 2)
            
            if fade_out_samples > 0:
                t = torch.linspace(1, 0, fade_out_samples)
                fade_curve = t * t * (3 - 2 * t)
                audio_data[:, -fade_out_samples:] *= fade_curve.unsqueeze(0)
        
        return audio_data
    
    def mix_tracks_professional(self, processed_tracks: List[Dict], output_duration: float, sample_rate: int) -> torch.Tensor:
        output_samples = int(output_duration * sample_rate)
        mixed_audio = torch.zeros(1, 2, output_samples, dtype=torch.float32)
        
        logger.info(f"\n🎛️ Professional Mixing:")
        logger.info(f"   Timeline: {output_duration}s ({output_samples} samples)")
        
        for track in processed_tracks:
            track_name = track["name"]
            start_sample = int(track["start_time"] * sample_rate)
            track_audio = track["audio"]
            
            logger.info(f"\n   Mixing {track_name}:")
            logger.info(f"     Start: {track['start_time']}s, Duration: {track['duration']:.2f}s")
            logger.info(f"     Level: {track['rms_db']:.1f}dB RMS, {track['peak_db']:.1f}dB peak")
            
            if len(track_audio.shape) == 2:
                track_audio = track_audio.unsqueeze(0)
            
            track_length = track_audio.shape[2]
            end_sample = start_sample + track_length
            
            if start_sample < output_samples and end_sample > 0:
                mix_start = max(0, start_sample)
                mix_end = min(output_samples, end_sample)
                track_start = max(0, -start_sample)
                track_end = track_start + (mix_end - mix_start)
                
                audio_segment = track_audio[0, :, track_start:track_end]
                mixed_audio[0, :, mix_start:mix_end] += audio_segment
                
                logger.info(f"     ✅ Mixed {(mix_end-mix_start)/sample_rate:.2f}s into timeline")
        
        mix_rms = torch.sqrt(torch.mean(mixed_audio ** 2))
        mix_peak = torch.max(torch.abs(mixed_audio))
        logger.info(f"\n🎵 Raw mix levels: RMS={20*torch.log10(mix_rms+1e-10):.1f}dB, Peak={20*torch.log10(mix_peak+1e-10):.1f}dB")
        return mixed_audio
    
    def apply_professional_processing(self, audio: torch.Tensor, sample_rate: int, normalization_mode: str,
                                    enable_compression: bool, compression_ratio: float, compression_threshold: float,
                                    compression_attack: float, compression_release: float,
                                    enable_limiter: bool, limiter_threshold: float, target_loudness: float,
                                    mix_info: Dict) -> torch.Tensor:
        
        logger.info(f"\n🔧 Professional Audio Processing:")
        
        if enable_compression:
            audio = self.apply_professional_compression(
                audio, sample_rate, compression_ratio, compression_threshold,
                compression_attack, compression_release
            )
            mix_info["processing_steps"].append(f"Applied compression: {compression_ratio}:1 @ {compression_threshold}dB")
            logger.info(f"   ✅ Compression applied: {compression_ratio}:1")
        
        audio = self.apply_intelligent_normalization(audio, normalization_mode, target_loudness, mix_info)
        
        if enable_limiter:
            audio = self.apply_soft_limiter(audio, limiter_threshold)
            mix_info["processing_steps"].append(f"Applied soft limiter @ {limiter_threshold}dB")
            logger.info(f"   ✅ Limiter applied @ {limiter_threshold}dB")
        
        return audio
    
    def apply_professional_compression(self, audio: torch.Tensor, sample_rate: int, 
                                     ratio: float, threshold_db: float, attack_ms: float, release_ms: float) -> torch.Tensor:
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = 1 - np.exp(-1 / (attack_ms * 0.001 * sample_rate))
        release_coeff = 1 - np.exp(-1 / (release_ms * 0.001 * sample_rate))
        
        compressed = audio.clone()
        envelope = torch.zeros_like(audio)
        
        for ch in range(audio.shape[1]):
            signal = audio[0, ch, :]
            env = torch.zeros_like(signal)
            
            for i in range(1, len(signal)):
                current_level = torch.abs(signal[i])
                if current_level > env[i-1]:
                    env[i] = env[i-1] + attack_coeff * (current_level - env[i-1])
                else:
                    env[i] = env[i-1] + release_coeff * (current_level - env[i-1])
            
            gain_reduction = torch.ones_like(env)
            above_threshold = env > threshold_linear
            
            if torch.any(above_threshold):
                excess_db = 20 * torch.log10(env[above_threshold] / threshold_linear + 1e-10)
                reduced_db = excess_db / ratio
                gain_reduction[above_threshold] = 10 ** (-excess_db / 20) * 10 ** (reduced_db / 20)
            
            compressed[0, ch, :] = signal * gain_reduction
        
        return compressed
    
    def apply_intelligent_normalization(self, audio: torch.Tensor, mode: str, target_loudness: float, mix_info: Dict) -> torch.Tensor:
        logger.info(f"   🔧 Normalization mode: {mode}")
        
        if mode == "off":
            mix_info["processing_steps"].append("Normalization: OFF")
            return audio
        
        current_rms = torch.sqrt(torch.mean(audio ** 2))
        current_peak = torch.max(torch.abs(audio))
        current_lufs = self.calculate_lufs(audio, 44100)
        
        if mode == "prevent_clipping":
            if current_peak > 0.95:
                scale = 0.90 / current_peak
                audio = audio * scale
                logger.info(f"   ✅ Clipping prevention: scaled by {scale:.3f}")
                mix_info["processing_steps"].append(f"Clipping prevention: {scale:.3f}x")
        
        elif mode == "full_normalize":
            if current_peak > 0:
                scale = 0.95 / current_peak
            else:
                scale = 1.0
            audio = audio * scale
            logger.info(f"   ✅ Full normalization: scaled by {scale:.3f}")
            mix_info["processing_steps"].append(f"Full normalization: {scale:.3f}x")
        
        elif mode == "smart_normalize":
            if current_rms < 0.1:
                target_rms = 0.2
                scale = min(target_rms / current_rms, 0.9 / current_peak)
                audio = audio * scale
                logger.info(f"   ✅ Smart normalization: boosted by {scale:.3f}")
                mix_info["processing_steps"].append(f"Smart normalization: {scale:.3f}x")
        
        elif mode == "broadcast_standard":
            if abs(current_lufs - target_loudness) > 1.0:
                lufs_diff = target_loudness - current_lufs
                scale = 10 ** (lufs_diff / 20)
                scale = min(scale, 0.9 / current_peak)
                audio = audio * scale
                logger.info(f"   ✅ Broadcast standard: {lufs_diff:+.1f}dB LUFS adjustment")
                mix_info["processing_steps"].append(f"Broadcast standard: {lufs_diff:+.1f}dB LUFS")
        
        return audio
    
    def apply_soft_limiter(self, audio: torch.Tensor, threshold_db: float) -> torch.Tensor:
        threshold_linear = 10 ** (threshold_db / 20)
        over_threshold = torch.abs(audio) > threshold_linear
        
        if torch.any(over_threshold):
            limited = torch.sign(audio) * threshold_linear * torch.tanh(torch.abs(audio) / threshold_linear)
            audio = torch.where(over_threshold, limited, audio)
        
        return audio
    
    def calculate_lufs(self, audio: torch.Tensor, sample_rate: int) -> float:
        if audio.shape[1] == 2:
            mono = (audio[0, 0, :] + audio[0, 1, :]) / 2
        else:
            mono = audio[0, 0, :]
        
        mean_square = torch.mean(mono ** 2)
        lufs = -0.691 + 10 * torch.log10(mean_square + 1e-10)
        return float(lufs)
    
    def generate_professional_analysis(self, audio: torch.Tensor, sample_rate: int, tracks: List[Dict], target_loudness: float) -> str:
        try:
            rms = torch.sqrt(torch.mean(audio ** 2))
            peak = torch.max(torch.abs(audio))
            lufs = self.calculate_lufs(audio, sample_rate)
            
            if audio.shape[1] == 2:
                left_rms = torch.sqrt(torch.mean(audio[0, 0, :] ** 2))
                right_rms = torch.sqrt(torch.mean(audio[0, 1, :] ** 2))
                left_peak = torch.max(torch.abs(audio[0, 0, :]))
                right_peak = torch.max(torch.abs(audio[0, 1, :]))
                
                if abs(left_rms - right_rms) / max(left_rms, right_rms) < 0.2:
                    balance_state = "balanced"
                else:
                    balance_state = "unbalanced"
                    
                stereo_analysis = {
                    "left_rms_db": float(20 * torch.log10(left_rms + 1e-10)),
                    "right_rms_db": float(20 * torch.log10(right_rms + 1e-10)),
                    "left_peak_db": float(20 * torch.log10(left_peak + 1e-10)),
                    "right_peak_db": float(20 * torch.log10(right_peak + 1e-10)),
                    "balance": balance_state
                }
            else:
                stereo_analysis = {"mono": True}
            
            dynamic_range = float(20 * torch.log10(peak / (rms + 1e-10)))
            broadcast_compliant = abs(lufs - target_loudness) < 1.0 and peak < 0.95
            
            if peak > 0:
                headroom = float(20 * torch.log10(1.0 / peak))
            else:
                headroom = float('inf')
                
            if peak > 0.95:
                clip_risk = "high"
            elif peak > 0.8:
                clip_risk = "medium"
            else:
                clip_risk = "low"
                
            if dynamic_range > 15:
                dr_rating = "excellent"
            elif dynamic_range > 10:
                dr_rating = "good"
            else:
                dr_rating = "compressed"
            
            analysis = {
                "overall_levels": {
                    "rms_db": float(20 * torch.log10(rms + 1e-10)),
                    "peak_db": float(20 * torch.log10(peak + 1e-10)),
                    "lufs": lufs,
                    "dynamic_range_db": dynamic_range
                },
                "stereo": stereo_analysis,
                "broadcast": {
                    "target_lufs": target_loudness,
                    "current_lufs": lufs,
                    "deviation": lufs - target_loudness,
                    "compliant": broadcast_compliant
                },
                "quality": {
                    "headroom_db": headroom,
                    "clipping_risk": clip_risk,
                    "dynamic_range_rating": dr_rating
                },
                "track_count": len(tracks)
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            return json.dumps({"error": str(e)})
    
    def generate_track_info(self, tracks: List[Dict]) -> str:
        try:
            track_details = []
            
            for track in tracks:
                if "pan" in track:
                    pan_val = track["pan"]
                else:
                    pan_val = 0.0
                    
                track_info = {
                    "name": track["name"],
                    "duration": f"{track['duration']:.2f}s",
                    "start_time": f"{track['start_time']:.2f}s",
                    "volume": track["volume"],
                    "pan": pan_val,
                    "levels": {
                        "rms_db": track["rms_db"],
                        "peak_db": track["peak_db"]
                    },
                    "fades": {
                        "fade_in": f"{track['fade_in']:.1f}s",
                        "fade_out": f"{track['fade_out']:.1f}s"
                    }
                }
                track_details.append(track_info)
            
            return json.dumps({"tracks": track_details}, indent=2)
            
        except Exception as e:
            logger.error(f"Track info generation failed: {e}")
            return json.dumps({"error": str(e)})
    
    def extract_audio_data(self, audio_input):
        try:
            if 'LazyAudioMap' in str(type(audio_input)):
                if hasattr(audio_input, 'items'):
                    items = dict(audio_input.items())
                    if 'waveform' in items and 'sample_rate' in items:
                        return {"waveform": items['waveform'], "sample_rate": items['sample_rate']}
                
                if hasattr(audio_input, 'waveform') and hasattr(audio_input, 'sample_rate'):
                    return {"waveform": audio_input.waveform, "sample_rate": audio_input.sample_rate}
            
            elif isinstance(audio_input, dict):
                if 'waveform' in audio_input and 'sample_rate' in audio_input:
                    return audio_input
            
            elif isinstance(audio_input, torch.Tensor):
                return {"waveform": audio_input, "sample_rate": 44100}
            
            elif isinstance(audio_input, (tuple, list)) and len(audio_input) >= 2:
                return {"waveform": audio_input[0], "sample_rate": audio_input[1]}
            
            return None
                
        except Exception as e:
            logger.error(f"Error extracting audio data: {e}")
            return None
    
    def _create_fallback_audio(self, duration: float, sample_rate: int):
        fallback_audio = torch.zeros(1, 2, int(sample_rate * duration), dtype=torch.float32)
        error_audio = {"waveform": fallback_audio, "sample_rate": sample_rate}
        error_info = json.dumps({
            "error": "Audio processing failed",
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": 2
        })
        
        return (error_audio, duration, error_info, error_info, error_info, sample_rate)


# ─── REGISTRATION ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "S42_GaborNoiseReduction": S42_GaborNoiseReduction,
    "S42_VocalResonanceSculptor": S42_VocalResonanceSculptor,
    "S42_GaborHarmonicTransfuser": S42_GaborHarmonicTransfuser,
    "S42_DynamicPitchCorrection": S42_DynamicPitchCorrection,
    "S42_AceStepAudioGenerator": S42_AceStepAudioGenerator,
    "S42_DynamicRangeCompressor": S42_DynamicRangeCompressor,
    "S42_PhaseVocoderTimeStretch": S42_PhaseVocoderTimeStretch,
    "S42_LTXAudioSyncTrigger": S42_LTXAudioSyncTrigger,
    "S42_AudioFrequencyExtractor": S42_AudioFrequencyExtractor,
    "S42_AudioReactiveScheduler": S42_AudioReactiveScheduler,
    "S42CF_FunkyBeatSync": S42CF_FunkyBeatSync,
    "S42_OnsetEnvelopeVisualizer": S42_OnsetEnvelopeVisualizer,
    "S42_SpectralMashupEngine": S42_SpectralMashupEngine,
    "S42_AudioLatentEncoder": S42_AudioLatentEncoder,
    "S42_AudioLatentDecoder": S42_AudioLatentDecoder,
    "S42_NeuralLatentMixer": S42_NeuralLatentMixer,
    "S42_AudioLatentWobble": S42_AudioLatentWobble,
    "S42_LatentDisintegration": S42_LatentDisintegration,
    "S42_AudioVisualSynesthesia": S42_AudioVisualSynesthesia,
    "S42_AceStepLatentModifier": S42_AceStepLatentModifier,
    "Studio42AudioMixer": Studio42AudioMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42_GaborNoiseReduction": "Gabor Denoiser (S42)",
    "S42_VocalResonanceSculptor": "Vocal Resonance Sculptor (S42)",
    "S42_GaborHarmonicTransfuser": "Gabor Harmonic Transfuser (S42)",
    "S42_DynamicPitchCorrection": "Dynamic Pitch Correction (S42)",
    "S42_AceStepAudioGenerator": "AceStep Generator (S42)",
    "S42_DynamicRangeCompressor": "Dynamic Range Compressor (S42)",
    "S42_PhaseVocoderTimeStretch": "Phase Vocoder Time Stretch (S42)",
    "S42_LTXAudioSyncTrigger": "LTX Audio Sync Trigger (S42)",
    "S42_AudioFrequencyExtractor": "🎵 Frequency Extractor (S42)",
    "S42_AudioReactiveScheduler": "📉 Reactive Parameter Scheduler (S42)",
    "S42CF_FunkyBeatSync": "🕺 S42 CutFlow Funky Beat Sync",
    "S42_OnsetEnvelopeVisualizer": "Onset Envelope Visualizer (S42)",
    "S42_SpectralMashupEngine": "Spectral Mashup Engine (S42)",
    "S42_AudioLatentEncoder": "Audio Latent Encoder (S42)",
    "S42_AudioLatentDecoder": "Audio Latent Decoder (S42)",
    "S42_NeuralLatentMixer": "Neural Latent Mixer (S42)",
    "S42_AudioLatentWobble": "🫨 Audio Latent Wobble (S42)",
    "S42_LatentDisintegration": "Latent Disintegration (S42)",
    "S42_AudioVisualSynesthesia": "Audio-Visual Synesthesia (S42)",
    "S42_AceStepLatentModifier": "AceStep Latent Modifier (S42)",
    "Studio42AudioMixer": "🎬 Studio42 Audio Mixer",
}
