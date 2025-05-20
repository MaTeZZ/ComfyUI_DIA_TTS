# DIA Voice Clone ComfyUI Node

This custom node integrates the [DIA (Diffusion Image Analysis) voice cloning model](https://github.com/nari-labs/dia) into ComfyUI.

## Features

- Voice cloning with transcript input
- TTS prompt generation
- Adjustable parameters:
  - CFG Scale
  - Temperature
  - Top P
  - CFG Filter Top K
  - Max Tokens (Advanced node only)
- MP3 output
- Audio retiming with pitch preservation

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/MaTeZZ/ComfyUI_DIA_TTS.git
```

2. Install the required dependencies:
```bash
cd ComfyUI_DIA_TTS
pip install -r requirements.txt
```

## Usage

### DIA Voice Clone Node

The node accepts AUDIO input and produces AUDIO output compatible with other ComfyUI audio nodes.

1. Add the "DIA Voice Clone" node to your workflow
2. Provide the following inputs:
   - **transcript**: The transcript of the voice sample you want to clone (must use [S1] and [S2] speaker tags)
   - **tts_prompt**: The text you want to generate with the cloned voice
   - **input_audio**: Connect an AUDIO input for voice cloning
   - **cfg_scale**: Controls how closely the output follows the prompt (higher = more faithful)
   - **temperature**: Controls randomness (higher = more random)
   - **top_p**: Controls diversity (lower = more focused)
   - **cfg_filter_top_k**: Number of top logits to consider during CFG filtering
   - **save_audio_file**: Whether to save the generated audio to disk
   - **filename_prefix**: Prefix for the saved audio file
   - **use_torch_compile**: Whether to use torch.compile for faster generation

3. Connect the output to other audio nodes or use it directly

### DIA Voice Clone Advanced Node

Includes all features of the standard node plus:
- **max_tokens**: Maximum number of audio tokens to generate

### DIA Audio Retimer

A utility node for adjusting audio speed:
- **input_audio**: The audio to adjust
- **preserve_pitch**: Whether to maintain the original pitch while changing speed
- **speed**: Speed factor (0.3-3.0)

## Notes

- The first run will download the DIA model and may take some time
- For best results, provide a 5-10 second audio sample with a matching transcript
- Always begin input text with `[S1]`, and alternate between `[S1]` and `[S2]` for dialogue
- The model requires a GPU with sufficient VRAM (10+ GB recommended)

## Requirements

- Python 3.x
- PyTorch 2.x
- ComfyUI
- Other dependencies listed in requirements.txt
