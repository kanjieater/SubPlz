import time
import traceback


def transcribe(streams, model, be):
    max_workers = be.threads
    print("📝 Transcribing...")
    args = {
        "language": be.language,
        "initial_prompt": be.initial_prompt,
        "length_penalty": be.length_penalty,
        "temperature": be.temperature,
        "beam_size": be.beam_size,
        "patience": be.patience,
        "suppress_tokens": be.suppress_tokens,
        "prepend_punctuations": be.prepend_punctuations,
        "append_punctuations": be.append_punctuations,
        "compression_ratio_threshold": be.compression_ratio_threshold,
        "logprob_threshold": be.logprob_threshold,
        "condition_on_previous_text": be.condition_on_previous_text,
        "no_speech_threshold": be.no_speech_threshold,
        "word_timestamps": be.word_timestamps,
        "denoiser": be.denoiser,
        "vad": be.vad,
        "onnx": True,
    }
    # TODO: not faster-whisper
    # logprob_threshold

    start_time = time.monotonic()

    for stream_index, stream in enumerate(streams):
        for segment_index, audio in enumerate(stream[2]):
            try:
                audio.transcribe(model, **args)
            except Exception as e:
                print(f"Error transcribing stream {stream_index}, segment {segment_index}: {e}")
                traceback.print_exc()


    print(f"⏱️  Transcribing took: {time.monotonic() - start_time:.2f}s")
    return streams
