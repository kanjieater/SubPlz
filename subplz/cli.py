import argparse
from types import SimpleNamespace
from typing import List, Optional
import multiprocessing
import torch
from pathlib import Path
from dataclasses import dataclass, fields, field
import yaml

from subplz.files import get_working_folders


START_PUNC = """『「(（《｟[{"'“¿""" + """'“"¿([{-『「（〈《〔【｛［｟＜<‘“〝※"""
END_PUNC = """'"・.。!！?？:：”>＞⦆)]}』」）〉》〕】｝］’〟／＼～〜~;；─―–-➡"""
OTHER_PUNC = "＊　,，、…"
PUNCTUATION = START_PUNC + END_PUNC + OTHER_PUNC

ARGUMENTS = {
    "dirs": {
        "flags": ["-d", "--dirs"],
        "kwargs": {
            "dest": "dirs",
            "default": [],
            "required": False,
            "type": str,
            "nargs": "+",
            "help": "List of folders to pull audio from and generate subs to",
        },
    },
    "audio": {
        "flags": ["--audio"],
        "kwargs": {
            "nargs": "+",
            "default": [],
            "required": False,
            "help": "list of audio files to process (in the correct order)",
        },
    },
    "lang_ext": {
        "flags": ["--lang-ext"],
        "kwargs": {
            "type": str,
            "default": None,
            "required": False,
            "help": (
                "The subtitle language extension to be used. You can use this to also force "
                "multiple subtitles with different generation methods to be used, "
                "e.g., attackOnTitanEP01.az.srt could be used to indicate AI generated sub"
            ),
        },
    },
    "lang_ext_priority": {
        "flags": ["--lang-ext-priority"],
        "kwargs": {
            "default": [],
            "required": True,
            "type": str,
            "nargs": "+",
            "help": """
                The priority of subtitle extensions to pull from to rename to the prefered copy extension
            """,
        },
    },
    "text": {
        "flags": ["--text"],
        "kwargs": {
            "nargs": "+",
            "default": [],
            "required": False,
            "help": "path to the script file",
        },
    },
    "output_dir": {
        "flags": ["--output-dir"],
        "kwargs": {
            "default": None,
            "help": "Output directory, default uses the directory for the first audio file",
        },
    },
    "output_format": {
        "flags": ["--output-format"],
        "kwargs": {
            "default": "srt",
            "help": "Output format, currently only supports vtt and srt",
        },
    },
    "lang": {
        "flags": ["--lang"],
        "kwargs": {"default": "ja", "help": "language of the script and audio"},
    },
    "language": {
        "flags": ["--language"],
        "kwargs": {"default": "ja", "help": "language of the script and audio"},
    },
    "model_name": {
        "flags": ["--model"],
        "kwargs": {
            "default": "tiny",
            "dest": "model_name",
            "help": "tiny is best; whisper model to use. can be one of tiny, small, large, huge",
        },
    },
    "respect_grouping": {
        "flags": ["--respect-grouping"],
        "kwargs": {
            "default": False,
            "help": (
                "Keep the lines in the same subtitle together, instead of breaking them apart."
            ),
            "action": "store_true",
        },
    },
    "nlp": {
        "flags": ["--nlp"],
        "kwargs": {
            "default": False,
            "help": (
                "Use nlp library for splitting sentences, can be slower."
            ),
            "action": "store_true",
        },
    },
    "alass": {
        "flags": ["--alass"],
        "kwargs": {
            "default": False,
            "help": (
                "Use Alass to extract a subtitle as 'original-lang-ext' subtitle "
                "and use it to align the subtitles"
            ),
            "action": "store_true",
        },
    },
    "lang_ext_incorrect": {
        "flags": ["--lang-ext-incorrect"],
        "kwargs": {
            "type": str,
            "default": None,
            "required": False,
            "help": "The subtitle extension to prioritize to correct",
        },
    },
    "lang_ext_original": {
        "flags": ["--lang-ext-original"],
        "kwargs": {
            "type": str,
            "default": None,
            "required": False,
            "help": (
                "The subtitle language extension to be used for alass upon extraction. "
                "If no embedded subs exist, we'll pull from the sub with this lang extension. "
                "If only lang-ext & lang-ext-original are provided, we'll rename one ext to the other "
                "without changing the contents of the sub."
            ),
        },
    },
    "pipeline": {
        "flags": ["--pipeline"],
        "kwargs": {
            "required": False,
            "action": "append",
            "nargs": "+",
            "help": "Define a command. Use this flag multiple times to build a pipeline to be run in batch",
        },
    },
    "config": {
        "flags": ["-c", "--config"],
        "kwargs": {
            "type": str,
            "default": None,
            "metavar": "PATH",
            "help": "Path to the YAML file to define what should be done with automation and batch",
        },
    },
    "overwrite": {
        "flags": ["--overwrite"],
        "kwargs": {
            "default": None,
            "help": "Overwrite any destination files",
            "action": "store_true",
        },
    },
    "verify": {
        "flags": ["--verify"],
        "kwargs": {
            "default": None,
            "help": "Verify the language of the text",
            "action": "store_true",
        },
    },
    "unique": {
        "flags": ["--unique"],
        "kwargs": {
            "default": None,
            "help": "Only rename if the file is unique to the other subs in the directory",
            "action": "store_true",
        },
    },
    "dry_run": {
        "flags": ["--dry-run"],
        "kwargs": {
            "default": None,
            "help": "If set, we'll only show the changes we'd make instead of writing them",
            "action": "store_true",
        },
    },
    "use_cache": {
        "flags": ["--use-cache"],
        "kwargs": {
            "default": True,
            "help": "whether to use the cache or not",
            "action": "store_true",
        },
    },
    "cache_dir": {
        "flags": ["--cache-dir"],
        "kwargs": {"default": "SyncCache", "help": "the cache directory"},
    },
    "overwrite_cache": {
        "flags": ["--overwrite-cache"],
        "kwargs": {
            "default": False,
            "action": "store_true",
            "help": "Always overwrite the cache",
        },
    },
    "rerun": {
        "flags": ["--rerun"],
        "kwargs": {
            "default": False,
            "action": "store_true",
            "help": "Work on files even if file is marked as having run",
        },
    },
    "threads": {
        "flags": ["--threads"],
        "kwargs": {
            "type": int,
            "default": multiprocessing.cpu_count(),
            "help": "number of threads",
        },
    },
    "device": {
        "flags": ["--device"],
        "kwargs": {
            "default": "cuda" if torch.cuda.is_available() else "cpu",
            "help": "device to do inference on",
        },
    },
    "progress": {
        "flags": ["--progress"],
        "kwargs": {
            "default": True,
            "help": "progress bar on/off",
            "action": "store_true",
        },
    },
    "faster_whisper": {
        "flags": ["--faster-whisper"],
        "kwargs": {
            "default": False,
            "help": "Use faster_whisper; doesn't work with hugging face's decoding method currently",
            "action": "store_true",
        },
    },
    "local_only": {
        "flags": ["--local-only"],
        "kwargs": {
            "default": False,
            "help": "Don't download outside models",
            "action": "store_true",
        },
    },
    "stable_ts": {
        "flags": ["--stable-ts"],
        "kwargs": {"default": True, "help": "Use stable-ts", "action": "store_true"},
    },
    "denoiser": {
        "flags": ["--denoiser"],
        "kwargs": {"type": str, "default": "", "help": "Use stable-ts denoiser"},
    },
    "refinement": {
        "flags": ["--refinement"],
        "kwargs": {
            "default": True,
            "help": "Use stable-ts refinement option",
            "action": "store_true",
        },
    },
    "vad": {
        "flags": ["--vad"],
        "kwargs": {
            "default": True,
            "help": "Use Voice Activity Detection to get more accurate timestamps",
            "action": "store_true",
        },
    },
    "respect_grouping_count": {
        "flags": ["--respect-grouping-count"],
        "kwargs": {
            "type": int,
            "default": 6,
            "help": (
                "Affects how deep to search for matching groups with the respect-grouping flag; "
                "Default 6 is good for audiobooks, 15 for subs with animated themes"
            ),
        },
    },
    "initial_prompt": {
        "flags": ["--initial-prompt"],
        "kwargs": {
            "type": str,
            "default": None,
            "help": "optional text to provide as a prompt for the first window.",
        },
    },
    "length_penalty": {
        "flags": ["--length-penalty"],
        "kwargs": {
            "type": float,
            "default": None,
            "help": "optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default",
        },
    },
    "temperature": {
        "flags": ["--temperature"],
        "kwargs": {
            "type": float,
            "default": 0,
            "help": "temperature to use for sampling",
        },
    },
    "temperature_increment_on_fallback": {
        "flags": ["--temperature-increment-on-fallback"],
        "kwargs": {
            "type": float,
            "default": 0.2,
            "help": "temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
        },
    },
    "beam_size": {
        "flags": ["--beam-size"],
        "kwargs": {
            "type": int,
            "default": None,
            "help": "number of beams in beam search, only applicable when temperature is zero",
        },
    },
    "patience": {
        "flags": ["--patience"],
        "kwargs": {
            "type": float,
            "default": None,
            "help": "optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
        },
    },
    "suppress_tokens": {
        "flags": ["--suppress-tokens"],
        "kwargs": {
            "type": str,
            "default": [-1],
            "help": "comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
        },
    },
    "prepend_punctuations": {
        "flags": ["--prepend-punctuations"],
        "kwargs": {
            "type": str,
            "default": START_PUNC,
            "help": "if word_timestamps is True, merge these punctuation symbols with the next word",
        },
    },
    "append_punctuations": {
        "flags": ["--append-punctuations"],
        "kwargs": {
            "type": str,
            "default": END_PUNC + OTHER_PUNC,
            "help": "if word_timestamps is True, merge these punctuation symbols with the previous word",
        },
    },
    "nopend_punctuations": {
        "flags": ["--nopend-punctuations"],
        "kwargs": {
            "type": str,
            "default": "うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20",
            "help": "TODO",
        },
    },
    "compression_ratio_threshold": {
        "flags": ["--compression-ratio-threshold"],
        "kwargs": {
            "type": float,
            "default": 2.4,
            "help": "if the gzip compression ratio is higher than this value, treat the decoding as failed",
        },
    },
    "logprob_threshold": {
        "flags": ["--logprob-threshold"],
        "kwargs": {
            "type": float,
            "default": -1.0,
            "help": "if the average log probability is lower than this value, treat the decoding as failed",
        },
    },
    "condition_on_previous_text": {
        "flags": ["--condition-on-previous-text"],
        "kwargs": {
            "default": False,
            "help": "if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
            "action": argparse.BooleanOptionalAction,
        },
    },
    "no_speech_threshold": {
        "flags": ["--no-speech-threshold"],
        "kwargs": {
            "type": float,
            "default": 0.6,
            "help": "if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
        },
    },
    "word_timestamps": {
        "flags": ["--word-timestamps"],
        "kwargs": {
            "default": True,
            "help": "(experimental) extract word-level timestamps and refine the results based on them",
            "action": argparse.BooleanOptionalAction,
        },
    },
    "highlight_words": {
        "flags": ["--highlight-words"],
        "kwargs": {
            "default": False,
            "help": "(requires --word_timestamps True) underline each word as it is spoken in srt and vtt",
            "action": argparse.BooleanOptionalAction,
        },
    },
    "max_line_width": {
        "flags": ["--max-line-width"],
        "kwargs": {
            "type": int,
            "default": None,
            "help": "(requires --word_timestamps True) the maximum number of characters in a line before breaking the line",
        },
    },
    "max_line_count": {
        "flags": ["--max-line-count"],
        "kwargs": {
            "type": int,
            "default": None,
            "help": "(requires --word_timestamps True) the maximum number of lines in a segment",
        },
    },
    "max_words_per_line": {
        "flags": ["--max-words-per-line"],
        "kwargs": {
            "type": int,
            "default": None,
            "help": "(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment",
        },
    },
    "dynamic_quantization": {
        "flags": ["--dynamic-quantization"],
        "kwargs": {
            "default": False,
            "help": "Use torch's dynamic quantization (cpu only)",
            "action": argparse.BooleanOptionalAction,
        },
    },
    "quantize": {
        "flags": ["--quantize"],
        "kwargs": {
            "default": True,
            "help": "use fp16 on gpu or int8 on cpu",
            "action": argparse.BooleanOptionalAction,
        },
    },
    "fast_decoder": {
        "flags": ["--fast-decoder"],
        "kwargs": {
            "default": False,
            "help": "Use hugging face's decoding method, currently incomplete",
            "action": argparse.BooleanOptionalAction,
        },
    },
    "fast_decoder_overlap": {
        "flags": ["--fast-decoder-overlap"],
        "kwargs": {
            "type": int,
            "default": 10,
            "help": "Overlap between each batch",
        },
    },
    "fast_decoder_batches": {
        "flags": ["--fast-decoder-batches"],
        "kwargs": {
            "type": int,
            "default": 1,
            "help": "Number of batches to operate on",
        },
    },
}


@dataclass
class SyncParams:
    # Hardware
    threads: int = field(metadata={"category": "optional"})
    device: str = field(metadata={"category": "optional"})
    # UI
    progress: bool = field(metadata={"category": "optional"})
    # Behavior
    respect_grouping: bool = field(metadata={"category": "optional"})
    respect_grouping_count: int = field(metadata={"category": "optional"})
    # General Whisper
    language: str = field(metadata={"category": "optional"})
    model_name: str = field(metadata={"category": "optional"})
    # Faster Whisper
    faster_whisper: bool = field(metadata={"category": "optional"})
    local_only: bool = field(metadata={"category": "optional"})
    # stable-ts
    stable_ts: bool = field(metadata={"category": "optional"})
    vad: bool = field(metadata={"category": "optional"})
    denoiser: str = field(metadata={"category": "optional"})
    refinement: str = field(metadata={"category": "optional"})
    # Advanced Whisper
    initial_prompt: str = field(metadata={"category": "advanced"})
    length_penalty: float = field(metadata={"category": "advanced"})
    temperature: float = field(metadata={"category": "advanced"})
    temperature_increment_on_fallback: float = field(metadata={"category": "advanced"})
    beam_size: int = field(metadata={"category": "advanced"})
    patience: float = field(metadata={"category": "advanced"})
    suppress_tokens: List[str] = field(metadata={"category": "advanced"})
    prepend_punctuations: str = field(metadata={"category": "advanced"})
    append_punctuations: str = field(metadata={"category": "advanced"})
    nopend_punctuations: str = field(metadata={"category": "advanced"})
    compression_ratio_threshold: float = field(metadata={"category": "advanced"})
    logprob_threshold: float = field(metadata={"category": "advanced"})
    condition_on_previous_text: bool = field(metadata={"category": "advanced"})
    no_speech_threshold: float = field(metadata={"category": "advanced"})
    word_timestamps: bool = field(metadata={"category": "advanced"})
    # Experimental Hugging Face
    highlight_words: bool = field(metadata={"category": "advanced"})
    max_line_width: int = field(metadata={"category": "advanced"})
    max_line_count: int = field(metadata={"category": "advanced"})
    max_words_per_line: int = field(metadata={"category": "advanced"})
    # Original Whisper Optimizations
    dynamic_quantization: bool = field(metadata={"category": "advanced"})
    quantize: bool = field(metadata={"category": "advanced"})
    # Fast Decoder
    fast_decoder: bool = field(metadata={"category": "advanced"})
    fast_decoder_overlap: int = field(metadata={"category": "advanced"})
    fast_decoder_batches: int = field(metadata={"category": "advanced"})


@dataclass
class SyncData:
    dirs: List[str] = field(metadata={"category": "main"})
    audio: List[str] = field(metadata={"category": "main"})
    text: List[str] = field(metadata={"category": "main"})
    output_dir: str = field(metadata={"category": "main"})
    output_format: str = field(metadata={"category": "main"})
    rerun: Optional[bool] = field(metadata={"category": "optional"})
    lang: str = field(metadata={"category": "optional"})
    lang_ext: str = field(metadata={"category": "optional"})
    lang_ext_original: Optional[str] = field(metadata={"category": "optional"})
    lang_ext_incorrect: Optional[str] = field(metadata={"category": "optional"})
    subcommand: str = field(metadata={"category": "main"})
    overwrite_cache: bool = field(metadata={"category": "optional"})
    use_cache: bool = field(metadata={"category": "optional"})
    cache_dir: str = field(metadata={"category": "optional"})
    overwrite: bool = field(default=True, metadata={"category": "optional"})
    alass: bool = field(default=False, metadata={"category": "optional"})
    nlp: bool = field(default=False, metadata={"category": "optional"})


@dataclass
class GenData:
    dirs: List[str] = field(metadata={"category": "main"})
    audio: List[str] = field(metadata={"category": "main"})
    output_dir: str = field(metadata={"category": "main"})
    output_format: str = field(metadata={"category": "main"})
    rerun: Optional[bool] = field(metadata={"category": "optional"})
    lang: str = field(metadata={"category": "optional"})
    lang_ext: str = field(metadata={"category": "optional"})
    subcommand: str = field(metadata={"category": "main"})
    overwrite_cache: bool = field(metadata={"category": "optional"})
    use_cache: bool = field(metadata={"category": "optional"})
    cache_dir: str = field(metadata={"category": "optional"})
    overwrite: bool = field(default=False, metadata={"category": "optional"})
    alass: bool = field(default=False, metadata={"category": "optional"})


@dataclass
class GenParams:
    # Hardware
    threads: int = field(metadata={"category": "optional"})
    device: str = field(metadata={"category": "optional"})
    # UI
    progress: bool = field(metadata={"category": "optional"})
    # General Whisper
    language: str = field(metadata={"category": "optional"})
    model_name: str = field(metadata={"category": "optional"})
    # Faster Whisper
    faster_whisper: bool = field(metadata={"category": "optional"})
    local_only: bool = field(metadata={"category": "optional"})
    # stable-ts
    stable_ts: bool = field(metadata={"category": "optional"})
    vad: bool = field(metadata={"category": "optional"})
    denoiser: str = field(metadata={"category": "optional"})
    refinement: str = field(metadata={"category": "optional"})
    # Advanced Whisper
    initial_prompt: str = field(metadata={"category": "advanced"})
    length_penalty: float = field(metadata={"category": "advanced"})
    temperature: float = field(metadata={"category": "advanced"})
    temperature_increment_on_fallback: float = field(metadata={"category": "advanced"})
    beam_size: int = field(metadata={"category": "advanced"})
    patience: float = field(metadata={"category": "advanced"})
    suppress_tokens: List[str] = field(metadata={"category": "advanced"})
    prepend_punctuations: str = field(metadata={"category": "advanced"})
    append_punctuations: str = field(metadata={"category": "advanced"})
    nopend_punctuations: str = field(metadata={"category": "advanced"})
    compression_ratio_threshold: float = field(metadata={"category": "advanced"})
    logprob_threshold: float = field(metadata={"category": "advanced"})
    condition_on_previous_text: bool = field(metadata={"category": "advanced"})
    no_speech_threshold: float = field(metadata={"category": "advanced"})
    word_timestamps: bool = field(metadata={"category": "advanced"})
    # Experimental Hugging Face
    highlight_words: bool = field(metadata={"category": "advanced"})
    max_line_width: int = field(metadata={"category": "advanced"})
    max_line_count: int = field(metadata={"category": "advanced"})
    max_words_per_line: int = field(metadata={"category": "advanced"})
    # Original Whisper Optimizations
    dynamic_quantization: bool = field(metadata={"category": "advanced"})
    quantize: bool = field(metadata={"category": "advanced"})
    # Fast Decoder
    fast_decoder: bool = field(metadata={"category": "advanced"})
    fast_decoder_overlap: int = field(metadata={"category": "advanced"})
    fast_decoder_batches: int = field(metadata={"category": "advanced"})


@dataclass
class FindParams:
    subcommand: str = field(metadata={"category": "main"})
    dirs: List[str] = field(metadata={"category": "main"})


@dataclass
class RenameParams:
    subcommand: str = field(metadata={"category": "main"})
    dirs: List[str] = field(metadata={"category": "main"})
    lang_ext: str = field(metadata={"category": "main"})
    lang_ext_original: Optional[str] = field(metadata={"category": "main"})
    overwrite: bool = field(default=False, metadata={"category": "optional"})
    dry_run: bool = field(default=False, metadata={"category": "optional"})
    unique: bool = field(default=False, metadata={"category": "optional"})

@dataclass
class CopyParams:
    subcommand: str = field(metadata={"category": "main"})
    dirs: List[str] = field(metadata={"category": "main"})
    lang_ext: str = field(metadata={"category": "main"})
    lang_ext_priority: Optional[str] = field(metadata={"category": "main"})
    overwrite: bool = field(default=True, metadata={"category": "optional"})

@dataclass
class ExtractParams:
    subcommand: str = field(metadata={"category": "main"})
    dirs: List[str] = field(metadata={"category": "main"})
    lang_ext: str = field(metadata={"category": "main"})
    lang_ext_original: Optional[str] = field(metadata={"category": "main"})
    overwrite: bool = field(default=False, metadata={"category": "optional"})
    verify: bool = field(default=False, metadata={"category": "optional"})

@dataclass
class BatchParams:
    # --- Required fields first ---
    subcommand: str = field(metadata={"category": "main"})
    dirs: List[str] = field(metadata={"category": "optional"})
    config: Optional[str] = field(default=None, metadata={"category": "main"})
    pipeline: List[List[str]] = field(default_factory=list, metadata={"category": "optional"})


def setup_commands_cli(parser):
    sp = parser.add_subparsers(
        help="Generate a subtitle file from a file's audio source",
        required=True,
        dest="subcommand",
    )

    # Sync command
    sync = sp.add_parser(
        "sync",
        help="Sync a text to an audio/video stream",
        usage=""""subplz sync [-h] [-d DIRS [DIRS ...]] [--audio AUDIO [AUDIO ...]] [--text TEXT [TEXT ...]] [--output-dir OUTPUT_DIR] [--output-format OUTPUT_FORMAT]
                   [--language LANGUAGE] [--model MODEL] """,
    )

    main_group_sync = sync.add_argument_group("Main arguments")
    optional_group_sync = sync.add_argument_group("Optional arguments")
    advanced_group_sync = sync.add_argument_group("Advanced arguments")

    add_arguments_from_dataclass(
        main_group_sync, SyncParams, optional_group_sync, advanced_group_sync
    )
    add_arguments_from_dataclass(
        main_group_sync, SyncData, optional_group_sync, advanced_group_sync
    )

    # Gen command
    gen = sp.add_parser(
        "gen",
        help="Generate subtitles from AI without basing it on existing subtitles",
        usage=""""subplz gen [-h] [-d DIRS [DIRS ...]] [--audio AUDIO [AUDIO ...]] [--output-dir OUTPUT_DIR] [--output-format OUTPUT_FORMAT]
                   [--language LANGUAGE] [--model MODEL] """,
    )

    main_group_gen = gen.add_argument_group("Main arguments")
    optional_group_gen = gen.add_argument_group("Optional arguments")
    advanced_group_gen = gen.add_argument_group("Advanced arguments")

    add_arguments_from_dataclass(
        main_group_gen, GenParams, optional_group_gen, advanced_group_gen
    )
    add_arguments_from_dataclass(
        main_group_gen, GenData, optional_group_gen, advanced_group_gen
    )

    # Find command
    find = sp.add_parser(
        "find",
        help="Find all subdirectories that contain video or audio files",
        usage=""""subplz find [-h] [-d DIRS [DIRS ...]] """,
    )
    main_group_find = find.add_argument_group("Main arguments")
    optional_group_find = find.add_argument_group("Optional arguments")
    advanced_group_find = find.add_argument_group("Advanced arguments")
    add_arguments_from_dataclass(
        main_group_find, FindParams, optional_group_find, advanced_group_find
    )

    # rename command
    rename = sp.add_parser(
        "rename",
        help="Rename one subtitle to another lang ext name",
        usage=""""subplz rename [-h] [-d DIRS [DIRS ...]] """,
    )
    main_group_rename = rename.add_argument_group("Main arguments")
    optional_group_rename = rename.add_argument_group("Optional arguments")
    advanced_group_rename = rename.add_argument_group("Advanced arguments")
    add_arguments_from_dataclass(
        main_group_rename, RenameParams, optional_group_rename, advanced_group_rename
    )

    # copy command
    copy = sp.add_parser(
        "copy",
        help="Copy one subtitle to another lang ext name without removing it",
        usage=""""subplz rename [-h] [-d DIRS [DIRS ...]] """,
    )
    main_group_copy = copy.add_argument_group("Main arguments")
    optional_group_copy = copy.add_argument_group("Optional arguments")
    advanced_group_copy = copy.add_argument_group("Advanced arguments")
    add_arguments_from_dataclass(
        main_group_copy, CopyParams, optional_group_copy, advanced_group_copy
    )


    extract = sp.add_parser(
        "extract",
        help="Extract one subtitle from a video file",
        usage=""""subplz extract [-h] [-d DIRS [DIRS ...]] """,
    )
    main_group_extract = extract.add_argument_group("Main arguments")
    optional_group_extract = extract.add_argument_group("Optional arguments")
    advanced_group_extract = extract.add_argument_group("Advanced arguments")
    add_arguments_from_dataclass(
        main_group_extract, ExtractParams, optional_group_extract, advanced_group_extract
    )

    batch = sp.add_parser(
        "batch",
        help="Run a predefined batch of operations on a directory",
        usage="subplz batch -d DIRS [DIRS ...] [--initial-rename]",
    )
    main_group_batch = batch.add_argument_group("Main arguments")
    optional_group_batch = batch.add_argument_group("Optional arguments")
    advanced_group_batch = batch.add_argument_group("Advanced arguments")
    add_arguments_from_dataclass(
        main_group_batch, BatchParams, optional_group_batch, advanced_group_batch
    )

    return parser


def get_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser = setup_commands_cli(parser)
    # --- Use the provided list, or fall back to sys.argv ---
    args = parser.parse_args(arg_list)
    return args


def validate_source_inputs(sources):
    has_explicit_params = sources.audio or sources.text or sources.output_dir
    error_text = "You must specify --dirs/-d, or alternatively all three of --audio --text --output_dir or which; not both"
    if sources.dirs:
        if has_explicit_params:
            raise ValueError(error_text)
    elif not has_explicit_params:
        raise ValueError(error_text)


def add_arguments_from_dataclass(
    main_group, dataclass_type, optional_group, advanced_group
):
    # Create a mapping of categories to their respective argument groups
    category_groups = {
        "main": main_group,
        "optional": optional_group,
        "advanced": advanced_group,
    }

    for f in fields(dataclass_type):
        arg_config = ARGUMENTS.get(f.name)
        if arg_config:
            category = f.metadata.get("category")
            group = category_groups.get(category)
            if group:
                group.add_argument(*arg_config["flags"], **arg_config["kwargs"])


def get_data(args, subcommand_to_dataclass):
    data_class = subcommand_to_dataclass.get(args.subcommand)
    if data_class is None:
        raise ValueError("Unknown subcommand")
    kwargs = {
        f.name: getattr(args, f.name)
        for f in fields(data_class)
        if hasattr(args, f.name)
    }
    return data_class(**kwargs)


def get_backend_data(args):
    subcommand_to_dataclass = {
        "gen": GenParams,
        "sync": SyncParams,
    }
    return get_data(args, subcommand_to_dataclass)


def get_source_data(args):
    subcommand_to_dataclass = {
        "gen": GenData,
        "sync": SyncData,
    }
    return get_data(args, subcommand_to_dataclass)


def get_inputs(arg_list=None):
    args = get_args(arg_list)
    if args.subcommand in ["sync", "gen"]:
        inputs = SimpleNamespace(
            subcommand=args.subcommand,
            backend=get_backend_data(args),
            cache=SimpleNamespace(
                overwrite=args.overwrite_cache,
                enabled=args.use_cache,
                cache_dir=args.cache_dir,
                model_name=args.model_name,
            ),
            sources=get_source_data(args),
        )
        if inputs.subcommand == "sync":
            validate_source_inputs(inputs.sources)

        return inputs
    else:
        return args
