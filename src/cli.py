import argparse
import multiprocessing
import torch


def setup_advanced_cli(parser):

    parser.add_argument( "--audio", nargs="+", required=True, help="list of audio files to process (in the correct order)")
    parser.add_argument("--text", nargs="+", required=True, help="path to the script file")
    parser.add_argument("--model", default="tiny", help="whisper model to use. can be one of tiny, small, large, huge")
    parser.add_argument("--language", default=None, help="language of the script and audio")
    parser.add_argument("--progress", default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", default=False,  help="Overwrite any destination files", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-cache", default=True, help="whether to use the cache or not", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache-dir", default="AudiobookTextSyncCache", help="the cache directory")
    parser.add_argument("--overwrite-cache", default=False, action=argparse.BooleanOptionalAction, help="Always overwrite the cache")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count(), help=r"number of threads")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to do inference on")
    parser.add_argument("--dynamic-quantization", "--dq", default=False, help="Use torch's dynamic quantization (cpu only)", action=argparse.BooleanOptionalAction)
    parser.add_argument('--quantize', default=True, help="use fp16 on gpu or int8 on cpu", action=argparse.BooleanOptionalAction)

    parser.add_argument("--faster-whisper", default=True, help='Use faster_whisper, doesn\'t work with hugging face\'s decoding method currently', action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast-decoder", default=False, help="Use hugging face's decoding method, currently incomplete", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast-decoder-overlap", type=int, default=10,help="Overlap between each batch")
    parser.add_argument("--fast-decoder-batches", type=int, default=1, help="Number of batches to operate on")

    parser.add_argument("--beam_size", type=int, default=None, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default=[-1], help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop", action=argparse.BooleanOptionalAction)

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--temperature_increment_on_fallback", type=float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", default=False, help="(experimental) extract word-level timestamps and refine the results based on them", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-『「（〈《〔【｛［‘“〝※", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--nopend_punctuations", type=str, default="うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20", help="TODO")
    parser.add_argument("--highlight_words", default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_line_width", type=int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--max_words_per_line", type=int, default=None, help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment")
    parser.add_argument("--output-dir", default=None, help="Output directory, default uses the directory for the first audio file")
    parser.add_argument("--output-format", default='srt', help="Output format, currently only supports vtt and srt")
    parser.add_argument("--local-only", default=False, help="Don't download outside models", action=argparse.BooleanOptionalAction)
    return parser

def get_args():
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument(
        "-d",
        "--dirs",
        dest="dirs",
        default=None,
        required=True,
        type=str,
        nargs="+",
        help="List of folders to run generate subs for",
    )
    parser = setup_advanced_cli(parser)
    args = parser.parse_args().__dict__
    return args
