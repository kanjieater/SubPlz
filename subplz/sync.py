
from ats.main import (
    match_start,
    expand_matches,
    print_batches,
    to_subs,
)
import warnings
from ats import align
from ats.lang import get_lang

from subplz.align import nc_align
from subplz.files import sourceData
from subplz.utils import get_tqdm

tqdm = get_tqdm()

# Filter out the specific warning
warnings.filterwarnings("ignore", category=FutureWarning, message="This search incorrectly ignores the root element")


def fuzzy_match_chapters(streams, chapters, cache):
    print("🔍 Fuzzy matching chapters...")
    ats, sta = match_start(streams, chapters, cache)
    audio_batches = expand_matches(streams, chapters, ats, sta)
    # print_batches(audio_batches)
    return audio_batches


def do_batch(ach, tch, prepend, append, nopend, offset):
    acontent = []
    boff = 0
    for a in ach:
        for p in a[0]["segments"]:
            p["start"] += boff
            p["end"] += boff
            acontent.append(p)
        boff += a[1]

    language = get_lang(ach[0][0]["language"])

    tcontent = [p for t in tch for p in t.text()]
    alignment, references = align.align(
        None,
        language,
        [p["text"] for p in acontent],
        [p.text() for p in tcontent],
        [],
        prepend,
        append,
        nopend,
    )
    return to_subs(tcontent, acontent, alignment, offset, None)


def sync(source: sourceData, model, streams, cache, be):
    nopend = set(be.nopend_punctuations)
    chapters = source.chapters
    args = {
        "language": be.language,
        "initial_prompt": be.initial_prompt,
        "length_penalty": be.length_penalty,
        "temperature": be.temperature,
        "beam_size": be.beam_size,
        "patience": be.patience,
        "suppress_tokens": be.suppress_tokens,
        "prepend_punctuations": be.prepend_punctuations,
        "append_punctuations": be.append_punctuations,
        "compression_ratio_threshold": be.compression_ratio_threshold,
        "logprob_threshold": be.logprob_threshold,
        "condition_on_previous_text": be.condition_on_previous_text,
        "no_speech_threshold": be.no_speech_threshold,
        "word_timestamps": be.word_timestamps,
    }

    audio_batches = fuzzy_match_chapters(streams, chapters, cache)
    print("🔄 Syncing...")
    with tqdm(audio_batches) as bar:
        for ai, batches in enumerate(bar):

            # bar.set_description(basename(streams[ai][2][0].path))
            offset, segments = 0, []
            for ajs, (chi, chjs), _ in tqdm(batches):
                ach = [
                    (
                        streams[ai][2][aj].transcribe(
                            model, cache, **args
                        ),
                        streams[ai][2][aj].duration,
                    )
                    for aj in ajs
                ]
                tch = [chapters[chi][1][chj] for chj in chjs]
                if tch:
                    segments.extend(
                        do_batch(
                            ach,
                            tch,
                            set(args["prepend_punctuations"]),
                            set(args["append_punctuations"]),
                            nopend,
                            offset,
                        )
                    )

                offset += sum(a[1] for a in ach)

            if not segments:
                continue
            source.writer.write_sub(segments, source.output_full_paths[ai])
            # if(len(source.chapters) == 1 and be.respect_grouping):
            #     new_segments = nc_align(chapters[0][0], source.output_full_paths[ai])
            #     source.writer.write_sub(new_segments, source.output_full_paths[ai])


