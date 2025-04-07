# EYES-WhisperingNeMo 🎙️👥

A powerful audio transcription and speaker diarization tool that combines Whisper, NeMo, and Demucs for accurate transcription with speaker separation. Perfect for podcasts, interviews, meetings, and any multi-speaker audio content.

## Features 🌟

- **High-Quality Transcription**: Uses OpenAI's Whisper models for accurate speech recognition
- **Speaker Diarization**: Identifies and separates different speakers using NVIDIA's NeMo
- **Music Separation**: Isolates speech from background music using Demucs
- **Multiple Languages**: Supports 99+ languages through Whisper
- **GPU Acceleration**: CUDA support for faster processing
- **Flexible Output**: Generates both plain text and SRT subtitle formats
- **Easy to Use**: Simple command-line interface with sensible defaults

## Prerequisites 🔧

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- CUDA 12.x and cuDNN 9.x installed (for GPU acceleration)
- FFmpeg installed

### CUDA Setup

For GPU acceleration, ensure you have:
1. NVIDIA GPU with CUDA support
2. CUDA 12.x installed
3. cuDNN 9.x installed

## Installation 🚀

1. Clone the repository:
   ```bash
   git clone https://github.com/wowitsjack/EYES-WhisperingNeMo.git
   cd EYES-WhisperingNeMo
   ```

2. Run the setup script:
   ```bash
   python run.py --help
   ```
   The script will automatically:
   - Create a virtual environment
   - Install all required dependencies
   - Set up NeMo and other models

## Quick Start 🏃‍♂️

Process an audio file with default settings:
```bash
python run.py path/to/your/audio.mp3
```

This will:
1. Separate speech from music (if any)
2. Transcribe the audio
3. Identify different speakers
4. Generate both .txt and .srt files with speaker labels

## Advanced Usage 🔄

### Command Line Options

```bash
python run.py [audio_file] [options]

Options:
  --whisper_model MODEL    Whisper model to use (tiny, base, small, medium, large-v2)
  --device DEVICE         Processing device (cuda/cpu)
  --language LANG         Language code (e.g., en, ja, zh)
  --batch_size SIZE      Batch size for processing
  --no-stemming          Disable music/voice separation
  --keep-numerals        Keep numerical digits instead of converting to words
```

### Example Commands

1. Use a specific Whisper model with GPU acceleration:
   ```bash
   python run.py interview.mp3 --whisper_model large-v2 --device cuda
   ```

2. Process non-English content:
   ```bash
   python run.py japanese_podcast.mp3 --language ja --whisper_model large-v2
   ```

3. Process clean speech without music separation:
   ```bash
   python run.py meeting_recording.mp3 --no-stemming
   ```

4. Adjust batch size for memory constraints:
   ```bash
   python run.py long_audio.mp3 --batch_size 4
   ```

## Output Files 📄

The tool generates two files for each processed audio:
- `[audio_name].txt`: Plain text transcript with speaker labels
- `[audio_name].srt`: Subtitle file with timestamps and speaker labels

Example output format:
```
Speaker 0: Hello, welcome to the podcast.
Speaker 1: Thanks for having me here today.
Speaker 0: Let's talk about your recent project...
```

## Memory Usage and Performance 🚦

- Memory usage depends on the Whisper model size:
  - tiny: ~1GB
  - base: ~1GB
  - small: ~2GB
  - medium: ~5GB
  - large-v2: ~10GB

- GPU memory requirements:
  - Minimum: 4GB VRAM
  - Recommended: 8GB+ VRAM for large models

- Processing time varies based on:
  - Audio length
  - Model size
  - GPU capabilities
  - Whether music separation is enabled

## Troubleshooting 🔍

1. **CUDA/cuDNN errors**:
   - Ensure CUDA 12.x and cuDNN 9.x are properly installed
   - Verify GPU compatibility
   - Try using `--device cpu` if GPU issues persist

2. **Memory errors**:
   - Reduce batch size: `--batch_size 4`
   - Use a smaller Whisper model
   - Free up system memory/VRAM

3. **Long processing times**:
   - Enable GPU acceleration with `--device cuda`
   - Disable music separation with `--no-stemming` for clean speech
   - Use a smaller Whisper model for faster processing

## License 📜

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- [OpenAI Whisper](https://github.com/openai/whisper)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Demucs](https://github.com/facebookresearch/demucs)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 