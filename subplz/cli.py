import argparse
from types import SimpleNamespace
from typing import List
import multiprocessing
import torch
from pathlib import Path
from dataclasses import dataclass

from subplz.files import get_working_folders


def setup_advanced_cli(parser):
    sp = parser.add_subparsers(
        help="Generate a subtitle file from an file's audio source", required=True
    )
    sync = sp.add_parser(
        "sync",
        help="Sync a text to an audio/video stream",
        usage=""""subplz sync [-h] [-d DIRS [DIRS ...]] [--audio AUDIO [AUDIO ...]] [--text TEXT [TEXT ...]] [--output-dir OUTPUT_DIR] [--output-format OUTPUT_FORMAT]
                   [--language LANGUAGE] [--model MODEL] """,
    )
    gen = sp.add_parser("gen", help="Generate subs from a audio/video stream")
    main_group = sync.add_argument_group("Main arguments")
    optional_group = sync.add_argument_group("Optional arguments")
    advanced_group = sync.add_argument_group("Advanced arguments")

    # Sources
    main_group.add_argument(
        "-d",
        "--dirs",
        dest="dirs",
        default=[],
        required=False,
        type=str,
        nargs="+",
        help="List of folders to pull audio from and generate subs to",
    )
    main_group.add_argument(
        "--audio",
        nargs="+",
        default=[],
        required=False,
        help="list of audio files to process (in the correct order)",
    )
    main_group.add_argument(
        "--text", nargs="+", default=[], required=False, help="path to the script file"
    )
    main_group.add_argument(
        "--output-dir",
        default=None,
        help="Output directory, default uses the directory for the first audio file",
    )
    main_group.add_argument(
        "--output-format",
        default="srt",
        help="Output format, currently only supports vtt and srt",
    )

    # General Whisper
    main_group.add_argument(
        "--language", default='ja', help="language of the script and audio"
    )
    optional_group.add_argument(
        "--model",
        default="tiny",
        help="tiny is best; whisper model to use. can be one of tiny, small, large, huge",
    )

    # Behaviors
    optional_group.add_argument(
        "--respect-grouping",
        default=False,
        help="Keep the lines in the same subtitle together, instead of breaking them apart. ",
        action=argparse.BooleanOptionalAction,
    )
    optional_group.add_argument(
        "--overwrite",
        default=True,
        help="Overwrite any destination files",
        action=argparse.BooleanOptionalAction,
    )

    # Cache Behaviors
    optional_group.add_argument(
        "--use-cache",
        default=True,
        help="whether to use the cache or not",
        action=argparse.BooleanOptionalAction,
    )
    optional_group.add_argument(
        "--cache-dir", default="SyncCache", help="the cache directory"
    )
    optional_group.add_argument(
        "--overwrite-cache",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Always overwrite the cache",
    )
    optional_group.add_argument(
        "--rerun",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Align files even if file is marked as having run",
    )

    # Hardware Inputs
    optional_group.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help=r"number of threads",
    )
    optional_group.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to do inference on",
    )

    # UI
    optional_group.add_argument(
        "--progress",
        default=True,
        help="progress bar on/off",
        action=argparse.BooleanOptionalAction,
    )

    # Faster Whisper
    optional_group.add_argument(
        "--faster-whisper",
        default=False,
        help="Use faster_whisper, doesn't work with hugging face's decoding method currently",
        action=argparse.BooleanOptionalAction,
    )
    optional_group.add_argument(
        "--local-only",
        default=False,
        help="Don't download outside models",
        action=argparse.BooleanOptionalAction,
    )

    # stable-ts
    optional_group.add_argument(
        "--stable-ts",
        default=True,
        help="Use stable-ts",
        action=argparse.BooleanOptionalAction,
    )
    optional_group.add_argument(
        "--vad",
        default=True,
        help="Don't download outside models",
        action=argparse.BooleanOptionalAction,
    )

    # No-compo algo
    optional_group.add_argument(
        "--respect-grouping-count",
        type=int,
        help="Affects how deep to search for matching groups with the respect-grouping flag; Default 6 is good for audiobooks, 15 for subs with animated themes",
        default=6,
    )

    # Advanced Model Inputs
    advanced_group.add_argument(
        "--initial_prompt",
        type=str,
        default=None,
        help="optional text to provide as a prompt for the first window.",
    )
    advanced_group.add_argument(
        "--length_penalty",
        type=float,
        default=None,
        help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default",
    )
    advanced_group.add_argument(
        "--temperature", type=float, default=0, help="temperature to use for sampling"
    )
    advanced_group.add_argument(
        "--temperature_increment_on_fallback",
        type=float,
        default=0.2,
        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
    )
    advanced_group.add_argument(
        "--beam_size",
        type=int,
        default=None,
        help="number of beams in beam search, only applicable when temperature is zero",
    )
    advanced_group.add_argument(
        "--patience",
        type=float,
        default=None,
        help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
    )
    advanced_group.add_argument(
        "--suppress_tokens",
        type=str,
        default=[-1],
        help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
    )
    advanced_group.add_argument(
        "--prepend_punctuations",
        type=str,
        default="\"'“¿([{-『「（〈《〔【｛［‘“〝※",
        help="if word_timestamps is True, merge these punctuation symbols with the next word",
    )
    advanced_group.add_argument(
        "--append_punctuations",
        type=str,
        default="\"'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~",
        help="if word_timestamps is True, merge these punctuation symbols with the previous word",
    )
    advanced_group.add_argument(
        "--nopend_punctuations",
        type=str,
        default="うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20",
        help="TODO",
    )
    advanced_group.add_argument(
        "--compression_ratio_threshold",
        type=float,
        default=2.4,
        help="if the gzip compression ratio is higher than this value, treat the decoding as failed",
    )
    advanced_group.add_argument(
        "--logprob_threshold",
        type=float,
        default=-1.0,
        help="if the average log probability is lower than this value, treat the decoding as failed",
    )
    advanced_group.add_argument(
        "--condition_on_previous_text",
        default=False,
        help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
        action=argparse.BooleanOptionalAction,
    )
    advanced_group.add_argument(
        "--no_speech_threshold",
        type=float,
        default=0.6,
        help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
    )
    advanced_group.add_argument(
        "--word-timestamps",
        default=True,
        help="(experimental) extract word-level timestamps and refine the results based on them",
        action=argparse.BooleanOptionalAction,
    )

    # Experimental Hugging Face
    # Experimental Word Timestamps
    advanced_group.add_argument(
        "--highlight_words",
        default=False,
        help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt",
        action=argparse.BooleanOptionalAction,
    )
    advanced_group.add_argument(
        "--max_line_width",
        type=int,
        default=None,
        help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line",
    )
    advanced_group.add_argument(
        "--max_line_count",
        type=int,
        default=None,
        help="(requires --word_timestamps True) the maximum number of lines in a segment",
    )
    advanced_group.add_argument(
        "--max_words_per_line",
        type=int,
        default=None,
        help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment",
    )

    # Original Whisper Optimizations
    advanced_group.add_argument(
        "--dynamic-quantization",
        "--dq",
        default=False,
        help="Use torch's dynamic quantization (cpu only)",
        action=argparse.BooleanOptionalAction,
    )
    advanced_group.add_argument(
        "--quantize",
        default=True,
        help="use fp16 on gpu or int8 on cpu",
        action=argparse.BooleanOptionalAction,
    )

    # Fast Decoder
    advanced_group.add_argument(
        "--fast-decoder",
        default=False,
        help="Use hugging face's decoding method, currently incomplete",
        action=argparse.BooleanOptionalAction,
    )
    advanced_group.add_argument(
        "--fast-decoder-overlap",
        type=int,
        default=10,
        help="Overlap between each batch",
    )
    advanced_group.add_argument(
        "--fast-decoder-batches",
        type=int,
        default=1,
        help="Number of batches to operate on",
    )

    return parser


def get_args():
    parser = argparse.ArgumentParser(description="Match audio to a transcript")

    parser = setup_advanced_cli(parser)
    args = parser.parse_args()
    return args


def validate_source_inputs(sources):
    has_explicit_params = sources.audio or sources.text or sources.output_dir
    error_text = "You must specify --dirs/-d, or alternatively all three of --audio --text --output_dir or which; not both"
    if sources.dirs:
        if has_explicit_params:
            raise ValueError(error_text)
    elif not has_explicit_params:
        raise ValueError(error_text)


@dataclass
class backendParams:
    # Hardware
    threads: int
    device: str
    # UI
    progress: bool
    # Behavior
    respect_grouping: bool
    respect_grouping_count: int
    # General Whisper
    language: str
    model_name: str
    # Faster Whisper
    faster_whisper: bool
    local_only: bool
    # stable-ts
    stable_ts: bool
    vad: bool
    # Advanced Whisper
    initial_prompt: str
    length_penalty: float
    temperature: float
    temperature_increment_on_fallback: float
    beam_size: int
    patience: float
    suppress_tokens: List[str]
    prepend_punctuations: str
    append_punctuations: str
    nopend_punctuations: str
    compression_ratio_threshold: float
    logprob_threshold: float
    condition_on_previous_text: bool
    no_speech_threshold: float
    word_timestamps: bool
    # Experimental Hugging Face
    # Experimental Word Timestamps
    highlight_words: bool
    max_line_width: int
    max_line_count: int
    max_words_per_line: int
    # Original Whisper Optimizations
    dynamic_quantization: bool
    quantize: bool
    # Fast Decoder
    fast_decoder: bool
    fast_decoder_overlap: int
    fast_decoder_batches: int



# args.progress


def get_inputs():
    args = get_args()
    inputs = SimpleNamespace(
        backend=backendParams(
            threads=args.threads,
            device=args.device,
            progress=args.progress,
            language=args.language,
            model_name=args.model,
            faster_whisper=args.faster_whisper,
            local_only=args.local_only,
            initial_prompt=args.initial_prompt,
            length_penalty=args.length_penalty,
            temperature=args.temperature,
            temperature_increment_on_fallback=args.temperature_increment_on_fallback,
            beam_size=args.beam_size,
            patience=args.patience,
            suppress_tokens=args.suppress_tokens,
            prepend_punctuations=args.prepend_punctuations,
            append_punctuations=args.append_punctuations,
            nopend_punctuations=args.nopend_punctuations,
            logprob_threshold=args.logprob_threshold,
            compression_ratio_threshold=args.compression_ratio_threshold,
            condition_on_previous_text=args.condition_on_previous_text,
            no_speech_threshold=args.no_speech_threshold,
            word_timestamps=args.word_timestamps,
            highlight_words=args.highlight_words,
            max_line_width=args.max_line_width,
            max_line_count=args.max_line_count,
            max_words_per_line=args.max_words_per_line,
            dynamic_quantization=args.dynamic_quantization,
            quantize=args.quantize,
            fast_decoder=args.fast_decoder,
            fast_decoder_overlap=args.fast_decoder_overlap,
            fast_decoder_batches=args.fast_decoder_batches,
            respect_grouping=args.respect_grouping,
            respect_grouping_count=args.respect_grouping_count,
            stable_ts=args.stable_ts,
            vad=args.vad,
        ),
        cache=SimpleNamespace(
            overwrite=args.overwrite_cache,
            enabled=args.use_cache,
            cache_dir=args.cache_dir,
            model_name=args.model,
        ),
        sources=SimpleNamespace(
            dirs=args.dirs,
            audio=args.audio,
            text=args.text,
            output_dir=args.output_dir,
            output_format=args.output_format,
            overwrite=args.overwrite,
            rerun=args.rerun,
            lang=args.language,
        ),
    )
    validate_source_inputs(inputs.sources)

    return inputs
