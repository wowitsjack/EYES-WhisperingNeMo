# EYES-WhisperingNeMo

A speech diarization and transcription system using Whisper and NeMo, with GPU acceleration support.

## Features

- Speech and music separation using Demucs
- Transcription using OpenAI's Whisper model
- Speaker diarization using NVIDIA's NeMo framework
- GPU acceleration support with CUDA and cuDNN
- Aligned transcription output with speaker labels

## Requirements

- Python 3.12+
- CUDA 12.x
- cuDNN 9.1.0
- PyTorch with CUDA support
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wowitsjack/EYES-WhisperingNeMo.git
cd EYES-WhisperingNeMo
```

2. Create and activate a virtual environment:
```bash
python -m venv nemo-asr-venv
source nemo-asr-venv/bin/activate  # Linux/Mac
# OR
.\nemo-asr-venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the diarization script with:

```bash
python diarize.py --audio_path <path_to_audio> --whisper_model_name <model_size> --device <cuda/cpu>
```

Parameters:
- `audio_path`: Path to the input audio file
- `whisper_model_name`: Whisper model size (tiny, base, small, medium, large)
- `device`: Device to run on (cuda for GPU, cpu for CPU)

Example:
```bash
python diarize.py --audio_path vocals.mp3 --whisper_model_name medium --device cuda
```

## Output

The script will:
1. Separate speech from music (if present)
2. Transcribe the audio
3. Perform speaker diarization
4. Generate aligned output with speaker labels

## License

MIT License 