from ats.main import (
    expand_matches,
    to_subs,
)
import warnings
from ats import align
from ats.lang import get_lang
from rapidfuzz import fuzz
from .logger import logger
from subplz.transcribe import transcribe
from subplz.alass import sync_alass
from subplz.files import get_sources, post_process
from subplz.models import get_model, get_temperature, unload_model
from subplz.align import nc_align, shift_align
from subplz.files import sourceData
from subplz.utils import get_tqdm, get_threads
from .sub import write_subfail
from pathlib import Path


tqdm, trange = get_tqdm()

# Filter out the specific warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="This search incorrectly ignores the root element",
)


def match_start(audio, text):
    ats, sta = {}, {}
    textcache = {}
    for ai in trange(len(audio)):
        afn, at, ac = audio[ai]
        for i in trange(len(ac)):
            if (ai, i) in ats:
                continue
            try:
                l = ac[i].transcribe(None)
            except AttributeError as e:
                print(
                    f"🥺 Transcript not found! Attribute error occurred: {e}. If you ran with disabling the cache, delete your cached files."
                )

            lang = get_lang(l["language"])
            acontent = lang.normalize(
                lang.clean("".join(seg["text"] for seg in l["segments"]))
            )
            best = (-1, -1, 0)
            for ti in range(len(text)):
                tfn, tc = text[ti]

                for j in range(len(tc)):
                    if (ti, j) in sta:
                        continue

                    if (ti, j) not in textcache:
                        textcache[ti, j] = lang.normalize(
                            lang.clean("".join(p.text() for p in tc[j].text()))
                        )
                    tcontent = textcache[ti, j]
                    if len(acontent) < 100 or len(tcontent) < 100:
                        continue

                    l = min(len(tcontent), len(acontent), 2000)
                    score = fuzz.ratio(acontent[:l], tcontent[:l])
                    # title = tc[j].titles[0] if hasattr(tc[j], 'titles') else basename(tc[j].path)
                    # tqdm.write(ac[i].cn + ' ' + title + ' ' + str(j) + ' ' + str(score))
                    if score > 40 and score > best[-1]:
                        best = (ti, j, score)

            if best[:-1] in sta:
                tqdm.write("match_start")
            elif best != (-1, -1, 0):
                ats[ai, i] = best
                sta[best[:-1]] = (ai, i, best[-1])

    return ats, sta


def fuzzy_match_chapters(streams, chapters):
    print("🔍 Fuzzy matching chapters...")
    ats, sta = match_start(streams, chapters)
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


def sync(source: sourceData, model, streams, be):
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
    audio_batches = fuzzy_match_chapters(streams, chapters)
    print("🔄 Syncing...")
    with tqdm(audio_batches) as bar:
        for ai, batches in enumerate(bar):
            # bar.set_description(basename(streams[ai][2][0].path))
            offset, segments = 0, []
            for ajs, (chi, chjs), _ in tqdm(batches):
                ach = [
                    (
                        streams[ai][2][aj].transcribe(model, **args),
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
                else:
                    print(
                        f"🫠 No chapters were matched for {streams[ai][2][0].path}. We'll still try to sync..."
                    )

                offset += sum(a[1] for a in ach)

            if not segments:
                continue
            shifted_segments = shift_align(segments)
            source.writer.write_sub(shifted_segments, source.output_full_paths[ai])
            if len(source.chapters) == 1 and be.respect_grouping:
                new_segments = nc_align(
                    chapters[0][0],
                    source.output_full_paths[ai],
                    be.respect_grouping_count,
                )
                source.writer.write_sub(new_segments, source.output_full_paths[ai])


def run_sync(inputs):
    be = inputs.backend
    be.temperature = get_temperature(be)
    be.threads = get_threads(be)
    sources = get_sources(inputs.sources, inputs.cache)

    for source in tqdm(sources):
        print(f"🐼 Starting '{source.audio}'...")
        result = None
        try:
            if source.alass:
                sync_alass(source, inputs.sources, be)
            else:
                result = transcribe(source, be)
                if result.success:
                    sync(source, result.model, result.streams, be)
                else:
                    raise RuntimeError(f"Transcription failed for '{source.audio[0]}'.")

        except Exception as e:
            logger.opt(exception=True).error(
                f"A critical error occurred during sync for '{source.audio[0]}': {e}"
            )
            # Use the single, shared helper function to write the failure log
            if source.output_full_paths:
                output_path = Path(source.output_full_paths[0])
                write_subfail(source.audio[0], output_path, str(e))
            else:
                logger.warning(
                    "Cannot write subfail file because source.output_full_paths is empty."
                )

        finally:
            if result and result.model:
                unload_model(result.model)
    return post_process(sources, "sync")
