def run_main():
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument(
        "--audio",
        nargs="+",
        required=True,
        help="list of audio files to process (in the correct order)",
    )
    parser.add_argument(
        "--text", nargs="+", required=True, help="path to the script file"
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="whisper model to use. can be one of tiny, small, large, huge",
    )
    parser.add_argument(
        "--language", default=None, help="language of the script and audio"
    )
    parser.add_argument(
        "--progress",
        default=True,
        help="progress bar on/off",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        help="Overwrite any destination files",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--use-cache",
        default=True,
        help="whether to use the cache or not",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--cache-dir", default="AudiobookTextSyncCache", help="the cache directory"
    )
    parser.add_argument(
        "--overwrite-cache",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Always overwrite the cache",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help=r"number of threads",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to do inference on",
    )
    parser.add_argument(
        "--dynamic-quantization",
        "--dq",
        default=False,
        help="Use torch's dynamic quantization (cpu only)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--quantize",
        default=True,
        help="use fp16 on gpu or int8 on cpu",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--faster-whisper",
        default=True,
        help="Use faster_whisper, doesn't work with hugging face's decoding method currently",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--fast-decoder",
        default=False,
        help="Use hugging face's decoding method, currently incomplete",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--fast-decoder-overlap",
        type=int,
        default=10,
        help="Overlap between each batch",
    )
    parser.add_argument(
        "--fast-decoder-batches",
        type=int,
        default=1,
        help="Number of batches to operate on",
    )

    parser.add_argument(
        "--ignore-tags",
        default=["rt"],
        nargs="+",
        help="Tags to ignore during the epub to text conversion, useful for removing furigana",
    )
    parser.add_argument(
        "--prefix-chapter-name",
        default=True,
        help="Whether to prefix the text of each chapter with its name",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--follow-links",
        default=True,
        help="Whether to follow hrefs or not in the ebook",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--beam_size",
        type=int,
        default=None,
        help="number of beams in beam search, only applicable when temperature is zero",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=None,
        help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=None,
        help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default",
    )

    parser.add_argument(
        "--suppress_tokens",
        type=str,
        default=[-1],
        help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
    )
    parser.add_argument(
        "--initial_prompt",
        type=str,
        default=None,
        help="optional text to provide as a prompt for the first window.",
    )
    parser.add_argument(
        "--condition_on_previous_text",
        default=False,
        help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature to use for sampling"
    )
    parser.add_argument(
        "--temperature_increment_on_fallback",
        type=float,
        default=0.2,
        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below",
    )
    parser.add_argument(
        "--compression_ratio_threshold",
        type=float,
        default=2.4,
        help="if the gzip compression ratio is higher than this value, treat the decoding as failed",
    )
    parser.add_argument(
        "--logprob_threshold",
        type=float,
        default=-1.0,
        help="if the average log probability is lower than this value, treat the decoding as failed",
    )
    parser.add_argument(
        "--no_speech_threshold",
        type=float,
        default=0.6,
        help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
    )
    parser.add_argument(
        "--word_timestamps",
        default=False,
        help="(experimental) extract word-level timestamps and refine the results based on them",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--prepend_punctuations",
        type=str,
        default="\"'“¿([{-『「（〈《〔【｛［‘“〝※",
        help="if word_timestamps is True, merge these punctuation symbols with the next word",
    )
    parser.add_argument(
        "--append_punctuations",
        type=str,
        default="\"'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~",
        help="if word_timestamps is True, merge these punctuation symbols with the previous word",
    )
    parser.add_argument(
        "--nopend_punctuations",
        type=str,
        default="うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20",
        help="TODO",
    )
    parser.add_argument(
        "--highlight_words",
        default=False,
        help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--max_line_width",
        type=int,
        default=None,
        help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line",
    )
    parser.add_argument(
        "--max_line_count",
        type=int,
        default=None,
        help="(requires --word_timestamps True) the maximum number of lines in a segment",
    )
    parser.add_argument(
        "--max_words_per_line",
        type=int,
        default=None,
        help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory, default uses the directory for the first audio file",
    )
    parser.add_argument(
        "--output-format",
        default="srt",
        help="Output format, currently only supports vtt and srt",
    )
    parser.add_argument(
        "--local-only",
        default=False,
        help="Don't download outside models",
        action=argparse.BooleanOptionalAction,
    )

    # parser.add_argument("--split-script", default="", help=r"the regex to split the script with. for monogatari it is something like ^\s[\uFF10-\uFF19]*\s$")

    args = parser.parse_args().__dict__
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.pop("progress"))
    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    output_dir = (
        Path(k) if (k := args.pop("output_dir")) else Path(".")
    )  # os.path.dirname(args['audio'][0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.pop("output_format")

    # model, device = args.pop("model"), args.pop('device')
    # if device == 'cuda' and not torch.cuda.is_available():
    #     device = 'cpu'
    # print(f"We're using {device}")
    # overwrite, overwrite_cache = args.pop('overwrite'), args.pop('overwrite_cache')
    # cache = Cache(model_name=model, enabled=args.pop("use_cache"), cache_dir=args.pop("cache_dir"),
    #               ask=not overwrite_cache, overwrite=overwrite_cache,
    #               memcache={})

    # faster_whisper = args.pop('faster_whisper')
    # local_only = args.pop('local_only')
    # quantize = args.pop("quantize")
    # if faster_whisper:
    #     model = WhisperModel(model, device, local_files_only=local_only, compute_type='float32' if not quantize else ('int8' if device == 'cpu' else 'float16'), num_workers=threads)
    #     model.transcribe2 = model.transcribe
    #     model.transcribe = MethodType(faster_transcribe, model)
    # else:
    #     model = whisper.load_model(model).to(device)
    #     if quantize and device != 'cpu':
    #         model = model.half()

    # if args.pop('dynamic_quantization') and device == "cpu" and not faster_whisper:
    #     ptdq_linear(model)

    # overlap, batches = args.pop("fast_decoder_overlap"), args.pop("fast_decoder_batches")
    # if args.pop("fast_decoder") and not faster_whisper:
    #     args["overlap"] = overlap
    #     args["batches"] = batches
    #     modify_model(model)

    # print("Loading...")
    # streams = [(os.path.basename(f), *AudioStream.from_file(f)) for f in args.pop('audio')]
    # chapters = [(os.path.basename(i), Epub.from_file(i)) if i.split(".")[-1] == 'epub' else (os.path.basename(i), [TextFile(path=i, title=os.path.basename(i))]) for i in args.pop('text')]

    # temperature = args.pop("temperature")
    # if (increment := args.pop("temperature_increment_on_fallback")) is not None:
    #     temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    # else:
    #     temperature = [temperature]

    # word_options = [
    #     "highlight_words",
    #     "max_line_count",
    #     "max_line_width",
    #     "max_words_per_line",
    # ]
    # if not args["word_timestamps"]:
    #     for option in word_options:
    #         if args[option]:
    #             parser.error(f"--{option} requires --word_timestamps True")

    # if args["max_line_count"] and not args["max_line_width"]:
    #     warnings.warn("--max_line_count has no effect without --max_line_width")
    # if args["max_words_per_line"] and args["max_line_width"]:
    #     warnings.warn("--max_words_per_line has no effect with --max_line_width")
    # writer_args = {arg: args.pop(arg) for arg in word_options}
    # word_timestamps = args.pop("word_timestamps")

    # ignore_tags = set(args.pop('ignore_tags'))
    # prefix_chapter_name = args.pop('prefix_chapter_name')
    # follow_links = args.pop('follow_links')

    # nopend = args.pop('nopend_punctuations')

    # print('Transcribing...')
    # s = time.monotonic()
    # with futures.ThreadPoolExecutor(max_workers=threads) as p:
    #     r = []
    #     for i in range(len(streams)):
    #         for j, v in enumerate(streams[i][2]):
    #             r.append(p.submit(lambda x: x.transcribe(model, cache, temperature=temperature, **args), v))
    #     futures.wait(r)
    # print(f"Transcribing took: {time.monotonic()-s:.2f}s")

    # print('Fuzzy matching chapters...')
    # ats, sta = match_start(streams, chapters, cache)
    # audio_batches = expand_matches(streams, chapters, ats, sta)
    # print_batches(audio_batches)

    # print('Syncing...')
    # with tqdm(audio_batches) as bar:
    #     for ai, batches in enumerate(bar):
    #         out = output_dir / (splitext(basename(streams[ai][2][0].path))[0] + '.' + output_format)
    #         if not overwrite and out.exists():
    #             bar.write(f"{out.name} already exsits, skipping.")
    #             continue

    #         bar.set_description(basename(streams[ai][2][0].path))
    #         offset, segments = 0, []
    #         for ajs, (chi, chjs), _ in tqdm(batches):
    #             ach = [streams[ai][2][aj] for aj in ajs]
    #             tch = [chapters[chi][1][chj] for chj in chjs]
    #             if tch:
    #                 acontent = []
    #                 boff = 0
    #                 for a in ach:
    #                     for p in a.transcribe(model, cache, temperature=temperature, **args)['segments']:
    #                         p['start'] += boff
    #                         p['end'] += boff
    #                         acontent.append(p)
    #                     boff += a.duration

    #                 tcontent = [p for t in tch for p in t.text(prefix_chapter_name, ignore=ignore_tags)]
    #                 alignment, references = align.align(model, acontent, tcontent, set(args['prepend_punctuations']), set(args['append_punctuations']), set(nopend))

    #                 # pprint(alignment)
    #                 segments.extend(to_subs(tcontent, acontent, alignment, offset, None))
    #             offset += sum(a.duration for a in ach)

    #         if not segments:
    #             continue

    #         with out.open("w", encoding="utf8") as o:
    #             if output_format == 'srt':
    #                 write_srt(segments, o)
    #             elif output_format == 'vtt':
    #                 write_vtt(segments, o)
