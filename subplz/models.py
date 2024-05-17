from faster_whisper import WhisperModel
import whisper
from types import MethodType
import torch
from copy import copy
import numpy as np
from copy import deepcopy

# from huggingface import modify_model
# from quantization import ptdq_linear

from ats.main import faster_transcribe


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
    local_files_only = backend.local_only
    quantize = backend.quantize
    num_workers = backend.threads
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(
        f"We're using {device}. Results should be similar in runtime between CPU & cuda"
    )
    compute_type = (
        'float32' if not quantize else ('int8' if device == 'cpu' else 'float16')
    )
    if faster_whisper:
        model = WhisperModel(
            model_name, device, local_files_only, compute_type, num_workers
        )
        model.transcribe2 = model.transcribe
        model.transcribe = MethodType(faster_transcribe, model)
    else:
        model = whisper.load_model(model).to(device)
        if quantize and device != 'cpu':
            model = model.half()
    return model

    # if args.pop('dynamic_quantization') and device == "cpu" and not faster_whisper:
    #     ptdq_linear(model)

    # overlap, batches = args.pop("fast_decoder_overlap"), args.pop("fast_decoder_batches")
    # if args.pop("fast_decoder") and not faster_whisper:
    #     args["overlap"] = overlap
    #     args["batches"] = batches
    # modify_model(model)
