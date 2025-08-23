import sys
import time
import traceback
from copy import deepcopy
import torch
from dataclasses import dataclass, field
from .logger import logger
from .models import get_model, unload_model


@dataclass
class TranscriptionResult:
    """A structured return type for the transcribe function."""

    success: bool = False
    streams: list = field(default_factory=list)
    model: object = None


def _get_transcribe_args(be):
    """Helper function to build the arguments dictionary for transcription."""
    # (This function remains unchanged from the previous answer)
    return {
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
    }


def transcribe(source, be) -> TranscriptionResult:
    """
    Orchestrates transcription with a retry-and-fallback mechanism for CUDA OOM errors.
    Returns a single TranscriptionResult object.
    """
    logger.info("📝 Starting transcription process...")

    gpu_retries = 2
    devices_to_try = ["cuda", "cpu"] if be.device == "cuda" else ["cpu"]

    for device in devices_to_try:
        temp_be = deepcopy(be)
        temp_be.device = device

        attempts = gpu_retries + 1 if device == "cuda" else 1

        for attempt in range(attempts):
            current_model = None
            try:
                if attempt > 0:
                    logger.info(
                        f"Retrying transcription on GPU (Attempt {attempt + 1}/{attempts})..."
                    )
                elif device == "cpu":
                    logger.warning(
                        "🚨 GPU attempts failed. Falling back to CPU with the same model. This will be much slower."
                    )

                logger.info(
                    f"Attempting transcription with model '{temp_be.model_name}' on device '{temp_be.device}'..."
                )
                current_model = get_model(temp_be)
                transcribe_args = _get_transcribe_args(temp_be)

                start_time = time.monotonic()
                for stream in source.streams:
                    for audio in stream[2]:
                        audio.transcribe(current_model, **transcribe_args)

                logger.success(
                    f"✅ Transcription successful with model '{temp_be.model_name}' on '{device}'."
                )
                logger.info(
                    f"⏱️  Transcribing took: {time.monotonic() - start_time:.2f}s"
                )

                # On success, return the structured result object
                return TranscriptionResult(
                    success=True, streams=source.streams, model=current_model
                )

            except RuntimeError as e:
                if "CUDA failed with error out of memory" in str(e):
                    logger.warning(
                        f"🔥 CUDA OOM Error on attempt {attempt + 1}. Freeing memory before next attempt."
                    )
                    unload_model(current_model)
                    if torch.cuda.is_available():
                        vram_allocated = torch.cuda.memory_allocated() / 1e9
                        vram_reserved = torch.cuda.memory_reserved() / 1e9
                        logger.debug(
                            f"VRAM after cleanup: {vram_allocated:.2f} GB allocated, {vram_reserved:.2f} GB reserved."
                        )
                    if attempt + 1 >= attempts:
                        break
                    else:
                        time.sleep(2)
                        continue
                else:
                    logger.opt(exception=True).error(
                        "A non-OOM runtime error occurred."
                    )
                    unload_model(current_model)
                    return TranscriptionResult(success=False)

            except Exception as e:
                logger.opt(exception=True).error(
                    "An unexpected error occurred during transcription."
                )
                unload_model(current_model)
                return TranscriptionResult(success=False)

    logger.critical(f"💥 All transcription attempts failed for '{source.audio[0]}'.")
    return TranscriptionResult(success=False)
