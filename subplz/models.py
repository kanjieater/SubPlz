from faster_whisper import WhisperModel
import whisper
from types import MethodType
import torch
import numpy as np

# from huggingface import modify_model
# from quantization import ptdq_linear

from ats.main import faster_transcribe

def get_temperature(args):
    print(args)
    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]
    return temperature

def get_model(args, threads):
    model, device = args.get("model"), args.pop('device')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"We're using {device}. Results should be similar in runtime between CPU & cuda")
    faster_whisper = args.pop('faster_whisper')
    local_only = args.pop('local_only')
    quantize = args.pop("quantize")
    if faster_whisper:
        model = WhisperModel(model, device, local_files_only=local_only, compute_type='float32' if not quantize else ('int8' if device == 'cpu' else 'float16'), num_workers=threads)
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