#!/usr/bin/env python3
import os
import argparse
import logging
import re
import torch
import torchaudio
import faster_whisper
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
from utils import (
    create_config,
    get_words_speaker_mapping,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    write_srt,
    find_numeral_symbol_tokens,
    cleanup,
    process_language_arg,
    punct_model_langs,
    langs_to_iso,
)


def process_audio(
    audio_path,
    enable_stemming=True,
    whisper_model_name="large-v2",
    suppress_numerals=True,
    batch_size=8,
    language=None,
    device=None,
):
    """
    Process audio file for transcription and speaker diarization.
    
    Args:
        audio_path: Path to the audio file
        enable_stemming: Whether to enable music removal from speech
        whisper_model_name: Whisper model to use
        suppress_numerals: Replace numerical digits with pronounciation
        batch_size: Batch size for inference
        language: Language code or None for auto-detection
        device: Device to use for processing ('cuda' or 'cpu')
        
    Returns:
        Path to the output text and srt files
    """
    logging.basicConfig(level=logging.INFO)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create temp directory for processing files
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    
    # Step 1: Isolate vocals from the rest of the audio if stemming is enabled
    if enable_stemming:
        logging.info("Separating music from speech using Demucs...")
        return_code = os.system(
            f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "{temp_path}" --device "{device}"'
        )

        if return_code != 0:
            logging.warning("Source splitting failed, using original audio file.")
            vocal_target = audio_path
        else:
            vocal_target = os.path.join(
                temp_path,
                "htdemucs",
                os.path.splitext(os.path.basename(audio_path))[0],
                "vocals.wav",
            )
    else:
        vocal_target = audio_path
    
    # Step 2: Transcribe audio using Whisper and realign timestamps using forced alignment
    logging.info(f"Transcribing audio using Whisper {whisper_model_name}...")
    compute_type = "float16" if device == "cuda" else "float32"
    
    whisper_model = faster_whisper.WhisperModel(
        whisper_model_name, device=device, compute_type=compute_type
    )
    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    audio_waveform = faster_whisper.decode_audio(vocal_target)
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )

    if batch_size > 0:
        transcript_segments, info = whisper_pipeline.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size,
            without_timestamps=True,
        )
    else:
        transcript_segments, info = whisper_model.transcribe(
            audio_waveform,
            language,
            suppress_tokens=suppress_tokens,
            without_timestamps=True,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)
    
    # Process detected language
    language = process_language_arg(info.language, whisper_model_name)
    logging.info(f"Detected language: {language}")

    # Clear GPU VRAM
    del whisper_model, whisper_pipeline
    torch.cuda.empty_cache()
    
    # Step 3: Align the transcription with the original audio using forced alignment
    logging.info("Aligning transcription with audio...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    audio_waveform = (
        torch.from_numpy(audio_waveform)
        .to(alignment_model.dtype)
        .to(alignment_model.device)
    )

    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=batch_size
    )

    del alignment_model
    torch.cuda.empty_cache()

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[info.language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    # Step 4: Convert audio to mono for NeMo compatibility
    logging.info("Converting audio to mono format...")
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        audio_waveform.cpu().unsqueeze(0).float(),
        16000,
        channels_first=True,
    )
    
    # Step 5: Perform speaker diarization using NeMo MSDD
    logging.info("Performing speaker diarization...")
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()
    
    # Step 6: Map speakers to sentences based on timestamps
    logging.info("Mapping speakers to sentences...")
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    
    # Step 7: Realign speech segments using punctuation if language supported
    logging.info("Realigning speech segments using punctuation...")
    if info.language in punct_model_langs:
        # Restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))
        labled_words = punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word
    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language. Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
    
    # Step 8: Export results as text and subtitle files
    logging.info("Exporting results...")
    txt_path = f"{os.path.splitext(audio_path)[0]}.txt"
    srt_path = f"{os.path.splitext(audio_path)[0]}.srt"
    
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(srt_path, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)
    
    # Clean up temporary files
    cleanup(temp_path)
    
    logging.info(f"Transcription complete. Outputs saved to {txt_path} and {srt_path}")
    return txt_path, srt_path


def main():
    parser = argparse.ArgumentParser(description="Transcribe and diarize audio files")
    parser.add_argument("--audio_path", required=True, help="Path to the audio file")
    parser.add_argument("--enable_stemming", type=bool, default=True, help="Enable music removal from speech")
    parser.add_argument("--whisper_model_name", default="large-v2", help="Whisper model to use")
    parser.add_argument("--suppress_numerals", type=bool, default=True, help="Replace numerical digits with pronunciation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--language", default=None, help="Language code (auto-detect if not specified)")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    process_audio(
        audio_path=args.audio_path,
        enable_stemming=args.enable_stemming,
        whisper_model_name=args.whisper_model_name,
        suppress_numerals=args.suppress_numerals,
        batch_size=args.batch_size,
        language=args.language,
        device=args.device,
    )


if __name__ == "__main__":
    main() 