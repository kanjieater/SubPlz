from ats.main import Segment
from .logger import logger
from subplz.utils import get_tqdm, get_threads
from subplz.align import shift_align
from subplz.transcribe import transcribe
from subplz.files import get_sources, post_process
from subplz.models import get_model, get_temperature, unload_model

tqdm, trange = get_tqdm()


def gen(source, model, streams, be):
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
    }
    print("🤖 Writing generated subs...")
    segments = []
    offset = 0

    with tqdm(streams) as bar:
        for ai, batches in enumerate(bar):
            for s in streams[ai][2]:
                transcribed_segments = s.transcribe(model, **args).get("segments", [])

                for seg in transcribed_segments:
                    adjusted_segment = Segment(
                        seg["text"], seg["start"] + offset, seg["end"] + offset
                    )
                    segments.append(adjusted_segment)
                offset += s.duration
            if not segments:
                continue

    shifted_segments = shift_align(segments)
    source.writer.write_sub(shifted_segments, source.output_full_paths[ai])


def run_gen(inputs):
    be = inputs.backend
    be.temperature = get_temperature(be)
    be.threads = get_threads(be)
    sources = get_sources(inputs.sources, inputs.cache)
    for source in tqdm(sources):
        print(f"🐼 Starting '{source.audio}'...")
        result = None
        try:
            result = transcribe(source, be)
            if result.success:
                gen(source, result.model, result.streams, be)
            else:
                logger.error(f"Skipping generation for '{source.audio[0]}' due to transcription failure.")
        finally:
            if result and result.model:
                unload_model(result.model)
    post_process(sources, "gen")