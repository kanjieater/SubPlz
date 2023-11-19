import torch
import torch.nn.functional as F
from whisper.audio import (
    HOP_LENGTH,
    N_FFT,
    N_SAMPLES,
    mel_filters,
)

def pad(audio):
    signal_dim = audio.dim()
    extended_shape = [1] * (3 - signal_dim) + list(audio.size())
    pad = int(N_FFT // 2)
    audio = F.pad(audio.view(extended_shape), [pad, pad], 'reflect')
    return audio.view(audio.shape[-signal_dim:])

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
vad_model = vad_model.cuda()
vad_get_speech_timestamps = utils[0]

def get_speech_timestamps(audio):
    global vad_model
    if audio.device != vad_model.device:
        vad_model = vad_model.cpu()
        # vad_model = vad_model.to(audio.device)
    return vad_get_speech_timestamps(audio, vad_model, 0.25, min_speech_duration_ms=100, min_silence_duration_ms=50) # TODO(YM): play with this idk

def vad_and_write(audio):
    audio = audio.numpy()
    speech_segments = get_speech_timestamps(audio)
    l = [0] + [z for i in speech_segments for z in i.values()] + [audio.shape[-1]]
    # print(l)
    for i in range(0, len(l), 2): audio[l[i]:l[i+1]] = -1

    import wave
    import numpy as np

    samplerate = 16000

    # Convert to (little-endian) 16 bit integers.
    audio = (audio * (2 ** 15 - 1)).astype("<h")

    with wave.open("/tmp/output.wav", "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(audio.tobytes())


def log_mel_spectrogram(audio, apply_silence=False, n_mels=80):
    chunk_size = 10*N_SAMPLES
    max_mel = 0 # Idk why clipping with this is needed in the original implementation, but keep it "consistent" anw
    # vad_and_write(audio)
    audio = pad(audio)
    current = HOP_LENGTH
    while current < len(audio)-3*HOP_LENGTH:
        chunk = audio[current-HOP_LENGTH:current+chunk_size+HOP_LENGTH]
        speech_segments = get_speech_timestamps(chunk)
        speech_segments = [z for i in speech_segments for z in i.values()]
        if apply_silence:
            l = [0] + speech_segments + [len(chunk)]
            for i in range(0, len(l), 2): chunk[l[i]:l[i+1]] = -1

        window = torch.hann_window(N_FFT).to(audio.device)
        stft = torch.stft(chunk, N_FFT, HOP_LENGTH, window=window, return_complex=True, center=False)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = mel_filters(audio.device, n_mels)
        mel_spec = filters @ magnitudes
        old_current = current
        current += mel_spec.shape[-1] * HOP_LENGTH

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if log_spec.max() > max_mel: max_mel = log_spec.max()
        log_spec = torch.maximum(log_spec, max_mel - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        yield log_spec, speech_segments

def compare_spectrogram(audio):
    pass
#     base = original(audio)
#     test = torch.zeros((80, 0))
#     for i in  log_mel_spectrogram(audio):
#         test = torch.concat([test, i[0]], dim=-1)
#     return base.size() == test.size() and base.isclose(test).all().item()
