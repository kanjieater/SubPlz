import argparse
from types import SimpleNamespace
import multiprocessing
import torch
from subplz.files import get_working_folders


def setup_advanced_cli(parser):
    sp = parser.add_subparsers(help='Generate a subtitle file from an file\'s audio source', required=True)
    sync = sp.add_parser('sync',
                         help='Sync a text to an audio/video stream',
                         usage=""""subplz sync [-h] [-d DIRS [DIRS ...]] [--audio AUDIO [AUDIO ...]] [--text TEXT [TEXT ...]] [--output-dir OUTPUT_DIR] [--output-format OUTPUT_FORMAT]
                   [--language LANGUAGE] [--model MODEL] """)
    gen = sp.add_parser('gen', help='Generate subs from a audio/video stream')
    main_group = sync.add_argument_group('Main arguments')
    optional_group = sync.add_argument_group('Optional arguments')
    advanced_group = sync.add_argument_group('Advanced arguments')

    # Sources
    main_group.add_argument(
        "-d",
        "--dirs",
        dest="dirs",
        default=None,
        required=False,
        type=str,
        nargs="+",
        help="List of folders to pull audio from and generate subs to",
    )
    main_group.add_argument("--audio", nargs="+", required=False, help="list of audio files to process (in the correct order)")
    main_group.add_argument("--text", nargs="+", required=False, help="path to the script file")
    main_group.add_argument("--output-dir", default=None, help="Output directory, default uses the directory for the first audio file")
    main_group.add_argument("--output-format", default='srt', help="Output format, currently only supports vtt and srt")

    # General Whisper
    main_group.add_argument("--language", default=None, help="language of the script and audio")
    optional_group.add_argument("--model", default="tiny", help="tiny is best; whisper model to use. can be one of tiny, small, large, huge")

    # Behaviors
    optional_group.add_argument("--overwrite", default=False,  help="Overwrite any destination files", action=argparse.BooleanOptionalAction)

    # Cache Behaviors
    optional_group.add_argument("--use-cache", default=True, help="whether to use the cache or not", action=argparse.BooleanOptionalAction)
    optional_group.add_argument("--cache-dir", default="AudiobookTextSyncCache", help="the cache directory")
    optional_group.add_argument("--overwrite-cache", default=False, action=argparse.BooleanOptionalAction, help="Always overwrite the cache")

    # Hardware Inputs
    optional_group.add_argument("--threads", type=int, default=multiprocessing.cpu_count(), help=r"number of threads")
    optional_group.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to do inference on")

    # UI
    optional_group.add_argument("--progress", default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)

    # Faster Whisper
    optional_group.add_argument("--faster-whisper", default=True, help='Use faster_whisper, doesn\'t work with hugging face\'s decoding method currently', action=argparse.BooleanOptionalAction)
    optional_group.add_argument("--local-only", default=False, help="Don't download outside models", action=argparse.BooleanOptionalAction)

    # Advanced Model Inputs
    advanced_group.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    advanced_group.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")
    advanced_group.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    advanced_group.add_argument("--temperature_increment_on_fallback", type=float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    advanced_group.add_argument("--beam_size", type=int, default=None, help="number of beams in beam search, only applicable when temperature is zero")
    advanced_group.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    advanced_group.add_argument("--suppress_tokens", type=str, default=[-1], help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    advanced_group.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-『「（〈《〔【｛［‘“〝※", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    advanced_group.add_argument("--append_punctuations", type=str, default="\"\'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    advanced_group.add_argument("--nopend_punctuations", type=str, default="うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20", help="TODO")

    # Experimental Hugging Face
    advanced_group.add_argument("--logprob_threshold", type=float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    advanced_group.add_argument("--compression_ratio_threshold", type=float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    advanced_group.add_argument("--condition_on_previous_text", default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop", action=argparse.BooleanOptionalAction)
    advanced_group.add_argument("--no_speech_threshold", type=float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    # Experimental Word Timestamps
    advanced_group.add_argument("--word_timestamps", default=False, help="(experimental) extract word-level timestamps and refine the results based on them", action=argparse.BooleanOptionalAction)
    advanced_group.add_argument("--highlight_words", default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt", action=argparse.BooleanOptionalAction)
    advanced_group.add_argument("--max_line_width", type=int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    advanced_group.add_argument("--max_line_count", type=int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    advanced_group.add_argument("--max_words_per_line", type=int, default=None, help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment")

    # Original Whisper Optimizations
    advanced_group.add_argument("--dynamic-quantization", "--dq", default=False, help="Use torch's dynamic quantization (cpu only)", action=argparse.BooleanOptionalAction)
    advanced_group.add_argument('--quantize', default=True, help="use fp16 on gpu or int8 on cpu", action=argparse.BooleanOptionalAction)


    # Fast Decoder
    advanced_group.add_argument("--fast-decoder", default=False, help="Use hugging face's decoding method, currently incomplete", action=argparse.BooleanOptionalAction)
    advanced_group.add_argument("--fast-decoder-overlap", type=int, default=10,help="Overlap between each batch")
    advanced_group.add_argument("--fast-decoder-batches", type=int, default=1, help="Number of batches to operate on")

    return parser


def get_args():
    parser = argparse.ArgumentParser(
        description="Match audio to a transcript"
    )



    parser = setup_advanced_cli(parser)
    args = parser.parse_args()
    return args

def get_aggregated_inputs(args):
    # args = get_args()
    # print(args)
    working_folders = get_working_folders(args.sources.dirs)
    print(working_folders)
    return args


def validate_source_inputs(sources):
    has_explicit_params = sources.audio or sources.text or sources.output_dir
    error_text = "You must specify --dirs/-d, or alternatively all three of --audio --text --output_dir or which; not both"
    if (sources.dirs):
        if (has_explicit_params):
            raise ValueError(error_text)
    elif (not has_explicit_params):
        raise ValueError(error_text)


def get_inputs():
    args = get_args()
    inputs = SimpleNamespace(
        transcribe=SimpleNamespace(
            ogprob_threshold= args.logprob_threshold,
            beam_size= args.beam_size,
            patience= args.patience,
            length_penalty= args.length_penalty
        ),
        sources=SimpleNamespace(
            dirs=args.dirs,
            audio=args.audio,
            text=args.text,
            output_dir=args.output_dir,
            output_format=args.output_format
        )
    )
    validate_source_inputs(inputs.sources)

    aggr_inputs = get_aggregated_inputs(inputs)
    return inputs

