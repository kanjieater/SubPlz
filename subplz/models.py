import gc
from faster_whisper import WhisperModel
import whisper
from types import MethodType
import torch
from copy import copy
import numpy as np

from .logger import logger
from .utils import get_tqdm

tqdm = get_tqdm()[0]


def unload_model(model):
    """
    Explicitly unloads a model from VRAM to free up memory.
    """
    if model is None:
        return
    try:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Model successfully unloaded from memory.")
    except Exception as e:
        logger.warning(f"An error occurred while unloading the model: {e}")


def faster_transcribe(self, audio, name, **args):
    args["log_prob_threshold"] = args.pop("logprob_threshold")
    args["beam_size"] = args["beam_size"] if args["beam_size"] else 1
    args["patience"] = args["patience"] if args["patience"] else 1
    args["length_penalty"] = args["length_penalty"] if args["length_penalty"] else 1
    result = self.transcribe(audio, best_of=1, **args)
    # result = self.refine(audio, result, **args)
    result.pad(0.5, 0.5, word_level=False)
    segments, prev_end = [], 0
    with tqdm(total=result.duration, unit_scale=True, unit=" seconds") as pbar:
        pbar.set_description(f"{name}")
        for segment in result.segments:
            segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )
            pbar.update(segment.end - prev_end)
            prev_end = segment.end
        pbar.update(result.duration - prev_end)
        pbar.refresh()

    return {
        "segments": segments,
        "language": args["language"] if "language" in args else result.language,
    }


def get_temperature(inputs):
    temperature = copy(inputs.temperature)
    temperature_increment_on_fallback = copy(inputs.temperature_increment_on_fallback)
    if (temperature_increment_on_fallback) is not None:
        temperature = tuple(
            np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
        )
    else:
        temperature = [temperature]
    return temperature


def get_model(backend):
    model_name = backend.model_name
    device = backend.device
    faster_whisper = backend.faster_whisper
    stable_ts = backend.stable_ts
    local_files_only = backend.local_only
    quantize = backend.quantize
    num_workers = backend.threads
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    logger.info(
        f"🖥️  We're using {device}. Results will be faster using Cuda with GPU than just CPU. Lot's of RAM needed no matter what."
    )
    compute_type = (
        "float32" if not quantize else ("int8" if device == "cpu" else "float16")
    )
    if faster_whisper:
        model = WhisperModel(
            model_name, device, local_files_only, compute_type, num_workers
        )
        model.transcribe2 = model.transcribe
        model.faster_transcribe = MethodType(faster_transcribe, model)
    elif stable_ts:
        import stable_whisper

        model = stable_whisper.load_faster_whisper(
            model_name,
            device=device,
            local_files_only=local_files_only,
            compute_type=compute_type,
            num_workers=num_workers,
        )
        # model.transcribe2 = model.transcribe_stable
        # model.transcribe2 = model.transcribe
        # TODO: Don't monkeypatch this - unnecessary
        model.faster_transcribe = MethodType(faster_transcribe, model)

    else:
        model = whisper.load_model(model).to(device)
        if quantize and device != "cpu":
            model = model.half()
    return model
