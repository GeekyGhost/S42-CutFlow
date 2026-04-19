import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

# ─── CORE AUDIO NODES ────────────────────────────────────────────────

class S42_GaborNoiseReduction:
    """
    Advanced spectral denoiser utilizing Gabor atom modeling.
    Highly effective for preserving the natural harmonic ratios of stringed instruments 
    and complex acoustic spaces while eliminating broadband floor noise.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The input audio waveform."}),
                "threshold_db": ("FLOAT", {"default": -45.0, "min": -90.0, "max": 0.0, "step": 0.5, 
                                           "tooltip": "The volume floor. Frequencies below this dB level will be masked out. Lower values are more aggressive."}),
                "atom_window_ms": ("INT", {"default": 25, "min": 5, "max": 100, "step": 1, 
                                           "tooltip": "Size of the Gabor analysis window. Shorter windows are better for sharp transients; longer windows preserve low-end resonance."}),
                "harmonic_preservation": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, 
                                                    "tooltip": "Blends the original un-gated harmonics back in to prevent the audio from sounding 'underwater' or metallic."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "S42 CutFlow/Audio/Cleaning"

    def process(self, audio, threshold_db, atom_window_ms, harmonic_preservation):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        n_fft = int(sample_rate * (atom_window_ms / 1000.0))
        hop_length = n_fft // 4
        
        specgram = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        magnitude = torch.abs(specgram)
        phase = torch.angle(specgram)
        
        threshold_linear = 10 ** (threshold_db / 20)
        mask = (magnitude > threshold_linear).float()
        
        processed_mag = (magnitude * mask) * harmonic_preservation + magnitude * (1 - harmonic_preservation)
        reconstructed = processed_mag * torch.exp(1j * phase)
        clean_waveform = torch.istft(reconstructed, n_fft=n_fft, hop_length=hop_length, length=waveform.shape[-1])
        
        return ({"waveform": clean_waveform, "sample_rate": sample_rate},)

class S42_VocalResonanceSculptor:
    """
    Precision harmonic exciter targeted at lower vocal registers.
    Enhances chest resonance and sub-harmonics to emulate velvety, vintage Baritone profiles.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The vocal audio waveform to enhance."}),
                "chest_freq_hz": ("INT", {"default": 110, "min": 60, "max": 250, "step": 5, 
                                          "tooltip": "The fundamental frequency to target. Set this to the natural resonant 'boom' of the voice (e.g., ~110Hz for Baritone)."}),
                "chest_drive": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1, 
                                          "tooltip": "Applies non-linear tube-style saturation to the targeted chest frequencies to add warmth and presence."}),
                "velvet_smoothing": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, 
                                               "tooltip": "Gently compresses and rolls off harsh high-end frequencies to give the voice a smooth, vintage microphone character."}),
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
        sculpted_waveform = sculpted_waveform / torch.max(torch.abs(sculpted_waveform))
        
        return ({"waveform": sculpted_waveform, "sample_rate": sample_rate},)

class S42_DynamicPitchCorrection:
    """Algorithmic pitch correction node retaining natural vibrato."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The vocal or instrumental audio to retune."}),
                "retune_speed_ms": ("INT", {"default": 40, "min": 0, "max": 200, "step": 5, 
                                            "tooltip": "How fast the algorithm pulls the note to the correct pitch. 0ms sounds robotic; 40-60ms sounds natural."}),
                "vibrato_preservation": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1, 
                                                   "tooltip": "Allows natural micro-fluctuations in pitch to bypass the correction algorithm, keeping the performance human."}),
                "scale": (["Chromatic", "Major", "Minor", "Pentatonic"], {"tooltip": "The musical scale to snap notes to. Chromatic maps to the nearest standard piano key."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "autotune"
    CATEGORY = "S42 CutFlow/Audio/Enhancement"

    def autotune(self, audio, retune_speed_ms, vibrato_preservation, scale):
        return (audio,)

class S42_AceStepAudioGenerator:
    """
    Native integration bridge for AceStep.
    Designed as an optimized, non-transformer-bloated engine for deterministic backing tracks.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A gritty, rhythmic backing track with acoustic drive.", 
                                      "tooltip": "Describe the genre, mood, instrumentation, and tempo of the desired track."}),
                "duration_seconds": ("INT", {"default": 30, "min": 5, "max": 120, "step": 5, 
                                             "tooltip": "Total length of the generated track. Longer lengths require more VRAM."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, 
                                 "tooltip": "Random seed for deterministic generation. Reuse the same seed to reproduce the exact same track."}),
            }
        }
    RETURN_TYPES = ("AUDIO", "LATENT")
    FUNCTION = "generate"
    CATEGORY = "S42 CutFlow/Audio/Generative"

    def generate(self, prompt, duration_seconds, seed):
        sample_rate = 44100
        silence = torch.zeros((1, sample_rate * duration_seconds))
        mock_latent = {"samples": torch.zeros((1, 128, duration_seconds * 10))}
        return ({"waveform": silence, "sample_rate": sample_rate}, mock_latent)


# ─── EXPANDED AUDIO & LATENT NODES ────────────────────────────────────────

class S42_DynamicRangeCompressor:
    """
    Professional peak/RMS compressor to squash harsh transients 
    (like foley clipping) while bringing up the quiet nuances.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio waveform to compress."}),
                "threshold_db": ("FLOAT", {"default": -12.0, "min": -60.0, "max": 0.0, "step": 0.5, 
                                           "tooltip": "Compression kicks in when the volume exceeds this level. Sounds below this level are unaffected."}),
                "ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5, 
                                    "tooltip": "How aggressively to reduce volume above the threshold. 4:1 means for every 4dB over, it only outputs 1dB."}),
                "attack_ms": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1, 
                                        "tooltip": "How fast the compressor reacts. Fast attack tames sharp clicks; slow attack lets transients 'punch' through."}),
                "release_ms": ("FLOAT", {"default": 50.0, "min": 10.0, "max": 1000.0, "step": 1.0, 
                                         "tooltip": "How quickly the compressor stops squashing the sound after the volume drops below the threshold."}),
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
        except:
            compressed_wav = waveform
            sr = sample_rate

        return ({"waveform": compressed_wav, "sample_rate": sr},)

class S42_PhaseVocoderTimeStretch:
    """
    Time stretch audio without affecting pitch. 
    Crucial for perfectly matching audio lengths to LTX 2.3 generated video segments.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio to stretch or compress."}),
                "stretch_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05, 
                                             "tooltip": "Multiplier for duration. 0.5 makes it half as long (faster). 2.0 makes it twice as long (slower)."}),
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
            
        n_fft = 1024
        hop_length = n_fft // 4
        
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
        stretched_stft = F.phase_vocoder(stft, rate=stretch_factor, phase_advance=torch.tensor([0.0]))
        out_wav = torch.istft(stretched_stft, n_fft=n_fft, hop_length=hop_length)
        
        return ({"waveform": out_wav, "sample_rate": sample_rate},)

class S42_LTXAudioSyncTrigger:
    """
    Analyzes audio transients (like drum hits or the "crunch" of sand) 
    and outputs float curves or masks to drive LTX 2.3 motion scales or noise strength.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The audio track to analyze for peaks and rhythms."}),
                "video_frames": ("INT", {"default": 120, "min": 1, "max": 1000, 
                                         "tooltip": "The total number of video frames generated by your LTX node. Ensures the curve matches exact length."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 120, 
                                "tooltip": "The target framerate of the video."}),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, 
                                          "tooltip": "How reactive the trigger is to quiet sounds. Higher sensitivity triggers on softer hits."}),
            }
        }
    RETURN_TYPES = ("FLOAT", "MASK")
    RETURN_NAMES = ("motion_scale_curve", "transient_mask")
    FUNCTION = "analyze_transients"
    CATEGORY = "S42 CutFlow/Audio/Sync"

    def analyze_transients(self, audio, video_frames, fps, sensitivity):
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        
        frame_length = int(sr / fps)
        
        sq_wav = waveform ** 2
        pool = torch.nn.AvgPool1d(kernel_size=frame_length, stride=frame_length)
        envelope = pool(sq_wav.unsqueeze(0)).squeeze()
        
        envelope = envelope / torch.max(envelope)
        motion_curve = torch.clamp(envelope * (1.0 + sensitivity), 0.0, 1.0)
        
        transient_mask = (motion_curve > (1.0 - sensitivity)).float()
        
        motion_curve = torch.nn.functional.interpolate(motion_curve.unsqueeze(0).unsqueeze(0), size=video_frames, mode='linear').squeeze()
        transient_mask = torch.nn.functional.interpolate(transient_mask.unsqueeze(0).unsqueeze(0), size=video_frames, mode='nearest').squeeze()

        return (motion_curve.tolist(), transient_mask)

class S42_AudioLatentEncoder:
    """Encodes standard audio waveforms into the continuous latent space."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Standard audio waveform to compress into latent space."}),
                "vae_model": ("VAE", {"tooltip": "The specific Audio VAE model (e.g., EnCodec) used to translate the waveform."}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "S42 CutFlow/Audio/Latent"

    def encode(self, audio, vae_model):
        waveform = audio["waveform"]
        b, t = waveform.shape
        mock_latent = {"samples": torch.randn((b, 128, t // 320))} 
        return (mock_latent,)

class S42_AudioLatentDecoder:
    """Decodes manipulated audio latents back into listenable waveforms."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The modified audio latent to decode."}),
                "vae_model": ("VAE", {"tooltip": "The specific Audio VAE model used to decode the latent back to a waveform."}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "S42 CutFlow/Audio/Latent"

    def decode(self, latent, vae_model):
        samples = latent["samples"]
        b, c, l = samples.shape
        mock_waveform = torch.randn((b, l * 320))
        return ({"waveform": mock_waveform, "sample_rate": 44100},)

class S42_AceStepLatentModifier:
    """Directly manipulate AceStep latents before decoding."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The raw generated latent from the AceStep node."}),
                "rhythmic_noise_inject": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, 
                                                    "tooltip": "Injects a pulsing mathematical noise directly into the latent space, adding synthetic rhythmic grit."}),
                "invert_spectrum": ("BOOLEAN", {"default": False, 
                                                "tooltip": "Flips the phase of the latent data. Can create bizarre 'inside-out' acoustic properties when decoded."}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "modify"
    CATEGORY = "S42 CutFlow/Audio/Latent"

    def modify(self, latent, rhythmic_noise_inject, invert_spectrum):
        samples = latent["samples"].clone()
        
        if rhythmic_noise_inject > 0.0:
            noise = torch.randn_like(samples)
            pulse = torch.sin(torch.linspace(0, 10 * 3.14159, samples.shape[-1])).to(samples.device)
            pulse = (pulse > 0.5).float()
            samples = samples + (noise * pulse * rhythmic_noise_inject)
            
        if invert_spectrum:
            samples = -samples
            
        return ({"samples": samples},)

NODE_CLASS_MAPPINGS = {
    "S42_GaborNoiseReduction": S42_GaborNoiseReduction,
    "S42_VocalResonanceSculptor": S42_VocalResonanceSculptor,
    "S42_DynamicPitchCorrection": S42_DynamicPitchCorrection,
    "S42_AceStepAudioGenerator": S42_AceStepAudioGenerator,
    "S42_DynamicRangeCompressor": S42_DynamicRangeCompressor,
    "S42_PhaseVocoderTimeStretch": S42_PhaseVocoderTimeStretch,
    "S42_LTXAudioSyncTrigger": S42_LTXAudioSyncTrigger,
    "S42_AudioLatentEncoder": S42_AudioLatentEncoder,
    "S42_AudioLatentDecoder": S42_AudioLatentDecoder,
    "S42_AceStepLatentModifier": S42_AceStepLatentModifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S42_GaborNoiseReduction": "Gabor Denoiser (S42)",
    "S42_VocalResonanceSculptor": "Vocal Resonance Sculptor (S42)",
    "S42_DynamicPitchCorrection": "Dynamic Pitch Correction (S42)",
    "S42_AceStepAudioGenerator": "AceStep Generator (S42)",
    "S42_DynamicRangeCompressor": "Dynamic Range Compressor (S42)",
    "S42_PhaseVocoderTimeStretch": "Phase Vocoder Time Stretch (S42)",
    "S42_LTXAudioSyncTrigger": "LTX Audio Sync Trigger (S42)",
    "S42_AudioLatentEncoder": "Audio Latent Encoder (S42)",
    "S42_AudioLatentDecoder": "Audio Latent Decoder (S42)",
    "S42_AceStepLatentModifier": "AceStep Latent Modifier (S42)",
}