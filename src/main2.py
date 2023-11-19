import argparse
import warnings
import os
import tqdm
import math
from pprint import pprint

import whisper
from whisper.utils import (
    exact_div,
    format_timestamp,
    get_writer,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)
from whisper.audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    mel_filters,
    pad_or_trim,
    load_audio
)
from whisper.timing import add_word_timestamps
from whisper.decoding import DecodingOptions, DecodingResult, TokenDecoder
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.model import Whisper

import torch
from torch import Tensor
import torch.nn.functional as F

import numpy as np
import ffmpeg
from typing import TYPE_CHECKING, Optional, Union

from audio import log_mel_spectrogram, compare_spectrogram
import decoding
from quantization import ptdq_linear

def segment_text(file, language, progress=True, spaces=False, whatever=False, whitespace=False):
    import pysbd
    seg = pysbd.Segmenter(language=language, clean=False)

    sentences = [" "]
    file = file.strip() if whitespace else file
    with tqdm.tqdm(file.split("\n"), disable=not progress) as bar:
        bar.set_description("Segmenting the script to sentences")
        for line in bar:
            for i in seg.segment(line):
                i = i.strip() if whitespace else i# Ugly
                l = ["」", " ", "　", "’"] if whitespace else ["」", "》", "’"]
                while len(i) and i[0] in l:  # Fix end of quotes
                    sentences[-1] += i[0]
                    i = i[1:]
                # if sentences[-1][0] in ["」", "’"] and len(sentences[-1] + i) < 50:  # Merge short lines with quotes
                #     sentences[-1] += i
                # if len(i):
                if spaces:
                    f =  i.split()
                    if whatever:
                        sentences.extend([k for i in f for z in i.split("。") for k in z.split("、")])
                    else:
                        sentences.extend(f)
                else:
                    sentences.append(i)
    return sentences[1:]
    # return [i + ' ' for i in sentences[1:]]

def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    text_path: str,
    *,
    verbose: Optional[bool] = None,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    **decode_options,
):
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    # audio = audio.to(model.device)
    # print("Spectrograms", compare_spectrogram(audio))
    audio = audio.to(model.device)
    mel_gen = log_mel_spectrogram(audio, apply_silence=False, n_mels=model.dims.n_mels)
    mel, speech_timestamps = next(mel_gen)
    speech_timestamps = [i // HOP_LENGTH  for i in speech_timestamps]
    # print(speech_timestamps)

    if not model.is_multilingual: decode_options["language"] = "en"
    if decode_options.get("language", None) is None:
        if verbose: print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
        mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
        _, probs = model.detect_language(mel_segment)
        decode_options["language"] = max(probs, key=probs.get)
        if verbose is not None: print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")

    language: str = decode_options["language"]
    task: str = "transcribe"
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)
    with open(text_path, encoding="utf-8") as x:
        text = segment_text(x.read(), language)

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)

    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def new_segment(
        *, start: float, end: float, tokens: torch.Tensor, result: DecodingResult
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    decode_options.pop("best_of", None)
    decode_options.pop("beam_size", None)
    decode_options.pop("patience", None)
    options = DecodingOptions(**decode_options, temperature=0)#, without_timestamps=True)
    decoder = decoding.DirectedDecoder(text, tokenizer)


    with tqdm.tqdm(total=len(audio)//HOP_LENGTH, unit="frames", disable=verbose is not False) as pbar:
        last_speech_timestamp = 0.0
        ntimestamps = 0
        speech_start, speech_end = speech_timestamps[ntimestamps], speech_timestamps[ntimestamps+1]
        while mel.shape[-1] > 5:
            if mel.shape[-1] < 2 * N_FRAMES and (x := next(mel_gen, None)) is not None:
                new_speech = [i//HOP_LENGTH + seek + mel.shape[-1] for i in x[1]]
                speech_timestamps += new_speech
                mel = torch.concat([mel, x[0]], dim=-1)

            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            mel_segment = mel[:, : N_FRAMES]
            segment_size = min(N_FRAMES, mel.shape[-1])

            # Not sure if really needed but whatever
            if ntimestamps == len(speech_timestamps) and seek < speech_end:
                speech_start = None

            while ntimestamps < len(speech_timestamps)-2 and seek > speech_end:
                ntimestamps += 2
                speech_start, speech_end = speech_timestamps[ntimestamps], speech_timestamps[ntimestamps+1]

            seek_shift = 0
            # print(seek, speech_start)
            speech_start = None if speech_start is not None and seek > speech_start else speech_start
            if speech_start is not None and seek + segment_size < speech_start:
                seek_shift += segment_size  # fast-forward to the next segment boundary
                pbar.update(seek_shift)
                seek += seek_shift
                mel = mel[:, seek_shift:]
                continue
            decoder.start_timestamp = (speech_start - seek) / N_FRAMES * 30 if speech_start else None
            decoder.timestamps = speech_timestamps[ntimestamps:]
            # decoder.tiktok = False
            decoder.seek = seek
            decoder.ntimestamps = 0

            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)

            result = model.decode(mel_segment, decoder, options)
            tokens = torch.tensor(result.tokens)

            logprob_threshold = None
            if no_speech_threshold is not None:
                # no voice activity check
                # should_skip = result.no_speech_prob > no_speech_threshold
                should_skip = False
                if (
                    logprob_threshold is not None
                    and result.avg_logprob > logprob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek_shift += segment_size  # fast-forward to the next segment boundary
                    pbar.update(seek_shift)
                    seek += seek_shift
                    mel = mel[:, seek_shift:]
                    continue

            current_segments = []

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)
            if len(consecutive) > 0:
                # if the output contains two consecutive timestamp tokens
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    current_segments.append(
                        new_segment(
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    seek_shift += segment_size
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_pos = (
                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    seek_shift += last_timestamp_pos * input_stride
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (
                    len(timestamps) > 0
                    and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    last_timestamp_pos = (
                        timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = last_timestamp_pos * time_precision

                current_segments.append(
                    new_segment(
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                        result=result,
                    )
                )
                seek_shift += segment_size

            if word_timestamps:
                add_word_timestamps(
                    segments=current_segments,
                    model=model,
                    tokenizer=tokenizer,
                    mel=mel_segment,
                    num_frames=segment_size,
                    prepend_punctuations=prepend_punctuations,
                    append_punctuations=append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                    )
                # print([w['word'] for s in current_segments for w in s["words"]])
                word_end_timestamps = [
                    w["end"] for s in current_segments for w in s["words"]
                ]
                if len(word_end_timestamps) > 0:
                    last_speech_timestamp = word_end_timestamps[-1]
                if not single_timestamp_ending and len(word_end_timestamps) > 0:
                    seek_shift = round(
                        (word_end_timestamps[-1] - time_offset) * FRAMES_PER_SECOND
                    )

            if verbose:
                for segment in current_segments:
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                    print(make_safe(line))

            # # if a segment is instantaneous or does not contain text, clear it
            # for i, segment in enumerate(current_segments):
            #     if segment["start"] == segment["end"] or segment["text"].strip() == "":
            #         segment["text"] = ""
            #         segment["tokens"] = []
            #         segment["words"] = []

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(
                        current_segments, start=len(all_segments)
                    )
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            seek_shift = min(mel.shape[-1], seek_shift)
            pbar.update(seek_shift)
            seek += seek_shift
            mel = mel[:, seek_shift:]


    with open(text_path, encoding='utf-8') as x:
        text = segment_text(x.read(), language)

    # for x in all_segments:
    #     if len(x['words']):
    #         x['words'][0]['start'] = x['start']
    #         x['words'][-1]['end'] = x['end']

    new = []
    x = [w for x in all_segments for w in x['words']]
    # print(x)
    s, e, current = 0, 0, 0
    for i in text:
        print("text", i)
        if e == len(x):
            break
        n = len(i)
        while n > current and e < len(x):
            print(n, current, x[e]['word'])
            print(f"words{s}", ''.join([i['word'] for i in x[s:e]]))
            current += len(x[e]['word'])
            e += 1
        # print(i, ''.join([i['word'] for i in x[s:e]]))
        new.append({
            "start": x[s]['start'],
            "end": x[e-1]['end'],
            "words": x[s:e],
            "text": i,
            "tokens": tokenizer.encode(i),
            "temperature": 0,
            "avg_logprob": 0,
            "compression_ratio": 0,
            "no_speech_prob": 0,
        })
        s, e, current = e, e, 0

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=new,#all_segments,
        language=language,
    )

def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--text", type=str, help="text files")
    parser.add_argument("--model", default="small", choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="srt", choices=["txt", "vtt", "srt", "tsv", "json", "all"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    # parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=False, help="whether to perform inference in fp16; True by default")

    # parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    # parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    # parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=True, help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--highlight_words", type=str2bool, default=True, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--dynamic_quantization", "--dq", type=str2bool, default=True, help="Use dynamic quantization")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead."
            )
        args["language"] = "en"

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    Whisper.decode = decoding.decode
    model = whisper.load_model(model_name, device=device, download_root=model_dir)
    dynamic_quantization = args.pop("dynamic_quantization")
    if dynamic_quantization and device == "cpu":
        ptdq_linear(model)
    # model.decode = decoding.decode

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if not args["word_timestamps"]:
        for option in word_options:
            if args[option]:
                # parser.print_help()
                warnings.warn(f"--{option} requires --word_timestamps True")
                # parser.error(f"--{option} requires --word_timestamps True")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}
    text_path = args.pop("text")
    for audio_path in args.pop("audio"):
        result = transcribe(model, audio_path, text_path, **args)
        writer(result, audio_path, writer_args)

if __name__ == "__main__":
    cli()
