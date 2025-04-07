#!/usr/bin/env python3
"""
Demo script for the EYES-WhisperingNeMo system.
"""
import os
import sys
import nltk
import urllib.request
import argparse
from diarize import process_audio

def download_sample_audio(output_path):
    """Download a sample audio file if none is provided."""
    print("Downloading a sample audio file...")
    url = "https://github.com/MahmoudAshraf97/whisper-diarization/raw/main/sample.mp3"
    urllib.request.urlretrieve(url, output_path)
    print(f"Sample audio downloaded to {output_path}")
    return output_path

def setup_nltk():
    """Download necessary NLTK data."""
    print("Setting up NLTK data...")
    nltk.download('punkt')
    print("NLTK setup complete.")

def main():
    """Run a demo of the EYES-WhisperingNeMo system."""
    parser = argparse.ArgumentParser(description="Demo for EYES-WhisperingNeMo")
    parser.add_argument("--audio", default=None, help="Path to audio file (will download a sample if not provided)")
    parser.add_argument("--model", default="medium", help="Whisper model to use (tiny, base, small, medium, large-v2)")
    parser.add_argument("--stemming", action="store_true", help="Enable music separation (uses more RAM)")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup NLTK
    setup_nltk()
    
    # Download sample audio if none provided
    audio_path = args.audio
    if audio_path is None:
        audio_path = download_sample_audio("sample.mp3")
    
    # Process the audio
    print(f"Processing audio file: {audio_path}")
    print(f"Using Whisper model: {args.model}")
    print(f"Music separation enabled: {args.stemming}")
    
    txt_path, srt_path = process_audio(
        audio_path=audio_path,
        enable_stemming=args.stemming,
        whisper_model_name=args.model,
        device=args.device
    )
    
    print(f"\nProcessing complete!")
    print(f"Text transcript saved to: {txt_path}")
    print(f"SRT subtitles saved to: {srt_path}")
    
    # Show a sample of the transcript
    print("\nSample of transcript:")
    with open(txt_path, "r", encoding="utf-8-sig") as f:
        transcript = f.read()
        print(transcript[:500] + "..." if len(transcript) > 500 else transcript)

if __name__ == "__main__":
    main() 