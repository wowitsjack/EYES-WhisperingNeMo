#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

def setup_environment():
    """Ensure virtual environment is set up and activated."""
    if not os.path.exists('nemo-asr-venv'):
        print("Setting up virtual environment...")
        os.system('bash auto-venv-setup.sh')
    else:
        print("Virtual environment already exists.")

def run_diarization(args):
    """Run the diarization process with given arguments."""
    cmd = f'python diarize.py --audio_path "{args.audio_path}"'
    
    if args.whisper_model:
        cmd += f' --whisper_model_name {args.whisper_model}'
    if args.device:
        cmd += f' --device {args.device}'
    if args.language:
        cmd += f' --language {args.language}'
    if args.batch_size is not None:
        cmd += f' --batch_size {args.batch_size}'
    if not args.enable_stemming:
        cmd += ' --enable_stemming False'
    if not args.suppress_numerals:
        cmd += ' --suppress_numerals False'
    
    print(f"Running command: {cmd}")
    return os.system(cmd)

def main():
    parser = argparse.ArgumentParser(
        description='Easy-to-use audio transcription and diarization tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python run.py my_audio.mp3

  # Use specific Whisper model and GPU
  python run.py my_audio.mp3 --whisper_model large-v2 --device cuda

  # Process non-English audio
  python run.py foreign_audio.mp3 --language ja --whisper_model large-v2

  # Disable music separation for clean speech
  python run.py speech.mp3 --no-stemming

  # Adjust batch size for memory constraints
  python run.py long_audio.mp3 --batch_size 4
        """
    )
    
    parser.add_argument('audio_path', help='Path to the audio file to process')
    parser.add_argument('--whisper_model', help='Whisper model to use (tiny, base, small, medium, large-v2)')
    parser.add_argument('--device', help='Device to use for processing (cuda/cpu)')
    parser.add_argument('--language', help='Language code (e.g., en, ja, zh)')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing')
    parser.add_argument('--no-stemming', dest='enable_stemming', action='store_false',
                      help='Disable music/voice separation')
    parser.add_argument('--keep-numerals', dest='suppress_numerals', action='store_false',
                      help='Keep numerical digits instead of converting to words')
    
    args = parser.parse_args()
    
    # Convert audio path to absolute path
    args.audio_path = str(Path(args.audio_path).resolve())
    
    # Ensure the audio file exists
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        sys.exit(1)
    
    # Setup environment if needed
    setup_environment()
    
    # Run diarization
    return run_diarization(args)

if __name__ == "__main__":
    sys.exit(main()) 