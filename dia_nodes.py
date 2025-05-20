import os
import torch
import numpy as np
import folder_paths
import torchaudio
import soundfile as sf
from pathlib import Path

# Activate virtual environment
from .venv_activator import activate_venv
activate_venv()

# ComfyUI imports
import comfy.utils


# Will be imported when the node is loaded
dia_model = None

class DIAVoiceCloneNode:
    """
    A ComfyUI node for voice cloning using the DIA (Diffusion Image Analysis) model.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcript_text": ("STRING", {"multiline": True, "default": "[S1] This is a voice sample for cloning.", "lines": 10}),
                "tts_prompt_text": ("STRING", {"multiline": True, "default": "[S1] This is the text I want to generate with the cloned voice.", "lines": 25}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.3, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "cfg_filter_top_k": ("INT", {"default": 45, "min": 1, "max": 100, "step": 1}),
                "max_tokens": ("INT", {"default": 2000, "min": 500, "max": 5000, "step": 100}),
                "save_audio_file": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "audio/dia"}),
            },
            "optional": {
                "input_audio": ("AUDIO",),
                "transcript": ("STRING",),  # Optional transcript input that overrides transcript_text
                "tts_prompt": ("STRING",),  # Optional tts_prompt input that overrides tts_prompt_text
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_voice"
    CATEGORY = "audio/dia"
    OUTPUT_NODE = True
    
    def generate_voice(self, transcript_text, tts_prompt_text, cfg_scale, temperature, top_p, cfg_filter_top_k, max_tokens, save_audio_file, filename_prefix, input_audio=None, transcript=None, tts_prompt=None):
        global dia_model
        
        # Use connected inputs if provided, otherwise use text fields
        final_transcript = transcript if transcript is not None else transcript_text
        final_tts_prompt = tts_prompt if tts_prompt is not None else tts_prompt_text
        
        # Initialize the model if it hasn't been loaded yet
        if dia_model is None:
            try:
                from dia.model import Dia
                print("Loading DIA model... This may take a while on first run.")
                # Always use float32 for highest quality as requested
                dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float32")
                print("DIA model loaded successfully.")
            except Exception as e:
                print(f"Error loading DIA model: {e}")
                raise RuntimeError(f"Failed to load DIA model: {e}")
        
        try:
            # Prepare the full prompt with transcript and generation text
            full_prompt = final_transcript + final_tts_prompt
            
            # Process input audio if provided
            audio_prompt = None
            if input_audio is not None:
                waveform = input_audio["waveform"]  # [1, C, N]
                sample_rate = input_audio["sample_rate"]
                
                # Convert to tensor if needed
                if not isinstance(waveform, torch.Tensor):
                    waveform = torch.tensor(waveform)
                
                # Resample if needed
                if sample_rate != 44100:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 44100)
                
                # Convert to mono if stereo
                if waveform.shape[1] == 2:
                    waveform = torch.mean(waveform, dim=1, keepdim=True)
                
                # Move to CUDA directly - no device checks
                waveform = waveform.cuda()
                
                # Encode with DAC
                audio_data = dia_model.dac_model.preprocess(waveform, 44100)
                _, encoded_frame, _, _, _ = dia_model.dac_model.encode(audio_data)
                audio_prompt = encoded_frame.squeeze(0).transpose(0, 1)
            
            print(f"Generating audio with DIA model...")
            print(f"CFG Scale: {cfg_scale}, Temperature: {temperature}, Top P: {top_p}")
            print(f"Max Tokens: {max_tokens}")
            
            # Generate the audio with max_tokens parameter to control length
            output = dia_model.generate(
                text=full_prompt,
                audio_prompt=audio_prompt,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                max_tokens=max_tokens,  # Added max_tokens parameter
                use_torch_compile=False,  # Always disabled as requested
                verbose=True
            )
            
            # Save the audio to file if requested
            if save_audio_file:
                (
                    full_output_folder,
                    filename,
                    counter,
                    subfolder,
                    filename_prefix,
                ) = folder_paths.get_save_image_path(
                    filename_prefix, folder_paths.get_output_directory()
                )
                file = f"{filename}_{counter:05}_.mp3"
                output_path = os.path.join(full_output_folder, file)
                sf.write(output_path, output, 44100)
                print(f"Audio saved to {output_path}")
            
            # Convert numpy array to tensor for ComfyUI AUDIO format
            output_tensor = torch.from_numpy(output).unsqueeze(0).unsqueeze(0)
            
            # Return in ComfyUI AUDIO format
            return ({"waveform": output_tensor, "sample_rate": 44100},)
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            raise RuntimeError(f"Failed to generate audio: {e}")


class DIAVoiceCloneAdvancedNode(DIAVoiceCloneNode):
    """
    An advanced version of the DIA voice clone node with more parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcript_text": ("STRING", {"multiline": True, "default": "[S1] This is a voice sample for cloning.", "lines": 10}),
                "tts_prompt_text": ("STRING", {"multiline": True, "default": "[S1] This is the text I want to generate with the cloned voice.", "lines": 25}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.3, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "cfg_filter_top_k": ("INT", {"default": 45, "min": 1, "max": 100, "step": 1}),
                "max_tokens": ("INT", {"default": 3000, "min": 500, "max": 5000, "step": 100}),
                "speed_factor": ("FLOAT", {"default": 0.94, "min": 0.8, "max": 1.2, "step": 0.01}),
                "save_audio_file": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "audio/dia"}),
            },
            "optional": {
                "input_audio": ("AUDIO",),
                "transcript": ("STRING",),  # Optional transcript input that overrides transcript_text
                "tts_prompt": ("STRING",),  # Optional tts_prompt input that overrides tts_prompt_text
            }
        }
    
    FUNCTION = "generate_voice_advanced"
    
    def generate_voice_advanced(self, transcript_text, tts_prompt_text, cfg_scale, temperature, top_p, cfg_filter_top_k, max_tokens, speed_factor, save_audio_file, filename_prefix, input_audio=None, transcript=None, tts_prompt=None):
        global dia_model
        
        # Use connected inputs if provided, otherwise use text fields
        final_transcript = transcript if transcript is not None else transcript_text
        final_tts_prompt = tts_prompt if tts_prompt is not None else tts_prompt_text
        
        # Initialize the model if it hasn't been loaded yet
        if dia_model is None:
            try:
                from dia.model import Dia
                print("Loading DIA model... This may take a while on first run.")
                # Always use float32 for highest quality as requested
                dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float32")
                print("DIA model loaded successfully.")
            except Exception as e:
                print(f"Error loading DIA model: {e}")
                raise RuntimeError(f"Failed to load DIA model: {e}")
        
        try:
            # Prepare the full prompt with transcript and generation text
            full_prompt = final_transcript + final_tts_prompt
            
            # Process input audio if provided
            audio_prompt = None
            if input_audio is not None:
                waveform = input_audio["waveform"]  # [1, C, N]
                sample_rate = input_audio["sample_rate"]
                
                # Convert to tensor if needed
                if not isinstance(waveform, torch.Tensor):
                    waveform = torch.tensor(waveform)
                
                # Resample if needed
                if sample_rate != 44100:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 44100)
                
                # Convert to mono if stereo
                if waveform.shape[1] == 2:
                    waveform = torch.mean(waveform, dim=1, keepdim=True)
                
                # Move to CUDA directly - no device checks
                waveform = waveform.cuda()
                
                # Encode with DAC
                audio_data = dia_model.dac_model.preprocess(waveform, 44100)
                _, encoded_frame, _, _, _ = dia_model.dac_model.encode(audio_data)
                audio_prompt = encoded_frame.squeeze(0).transpose(0, 1)
            
            print(f"Generating audio with DIA model...")
            print(f"CFG Scale: {cfg_scale}, Temperature: {temperature}, Top P: {top_p}")
            print(f"Max Tokens: {max_tokens}, Speed Factor: {speed_factor}")
            
            # Generate the audio with advanced parameters
            output = dia_model.generate(
                text=full_prompt,
                audio_prompt=audio_prompt,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                max_tokens=max_tokens,
                use_torch_compile=False,  # Always disabled as requested
                verbose=True
            )
            
            # Apply speed adjustment if needed
            if speed_factor != 1.0:
                try:
                    # Simple resampling for speed adjustment
                    original_len = len(output)
                    target_len = int(original_len / speed_factor)
                    
                    # Use numpy interpolation for speed adjustment
                    x_original = np.arange(original_len)
                    x_resampled = np.linspace(0, original_len - 1, target_len)
                    output = np.interp(x_resampled, x_original, output)
                    
                    print(f"Applied speed adjustment: {speed_factor}x")
                except Exception as e:
                    print(f"Warning: Speed adjustment failed: {e}")
            
            # Save the audio to file if requested
            if save_audio_file:
                (
                    full_output_folder,
                    filename,
                    counter,
                    subfolder,
                    filename_prefix,
                ) = folder_paths.get_save_image_path(
                    filename_prefix, folder_paths.get_output_directory()
                )
                file = f"{filename}_{counter:05}_.mp3"
                output_path = os.path.join(full_output_folder, file)
                sf.write(output_path, output, 44100)
                print(f"Audio saved to {output_path}")
            
            # Convert numpy array to tensor for ComfyUI AUDIO format
            output_tensor = torch.from_numpy(output).unsqueeze(0).unsqueeze(0)
            
            # Return in ComfyUI AUDIO format
            return ({"waveform": output_tensor, "sample_rate": 44100},)
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            raise RuntimeError(f"Failed to generate audio: {e}")


class AudioRetimer:
    """
    A node for adjusting the speed of audio with or without preserving pitch.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_audio": ("AUDIO",),
                "preserve_pitch": ("BOOLEAN", {"default": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "retime_audio"
    CATEGORY = "audio/dia"
    OUTPUT_NODE = True

    def retime_audio(self, input_audio, preserve_pitch, speed):
        waveform = input_audio["waveform"]  # [1, C, N]
        sample_rate = input_audio["sample_rate"]

        if not preserve_pitch:
            # Simple time stretching by resampling
            new_sample_rate = int(sample_rate * speed)
            slower_audio = torchaudio.functional.resample(waveform, new_sample_rate, sample_rate)
            return ({"waveform": slower_audio, "sample_rate": sample_rate},)
        else:
            try:
                import librosa
            except ImportError:
                raise ImportError("Librosa must be installed for pitch preservation. Run 'pip install librosa'")

            # Convert to numpy for librosa processing
            waveform_np = waveform.squeeze().cpu().numpy()

            # Time-stretching without pitch alteration
            y_stretched = librosa.effects.time_stretch(waveform_np, rate=speed)

            # Convert back to tensor
            y_stretched_tensor = torch.tensor(y_stretched)

            if y_stretched_tensor.ndim == 1:
                # Mono: [N] → [1, 1, N]
                y_stretched_tensor = y_stretched_tensor.unsqueeze(0).unsqueeze(0)
            elif y_stretched_tensor.ndim == 2:
                # Multi-channel: [C, N] → [1, C, N]
                y_stretched_tensor = y_stretched_tensor.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected tensor shape: {y_stretched_tensor.shape}")

            return ({"waveform": y_stretched_tensor, "sample_rate": sample_rate},)


# Register the nodes
NODE_CLASS_MAPPINGS = {
    "DIAVoiceClone": DIAVoiceCloneNode,
    "DIAVoiceCloneAdvanced": DIAVoiceCloneAdvancedNode,
    "DIAAudioRetimer": AudioRetimer,
}

# Define display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DIAVoiceClone": "DIA Voice Clone",
    "DIAVoiceCloneAdvanced": "DIA Voice Clone (Advanced)",
    "DIAAudioRetimer": "DIA Audio Retimer",
}
