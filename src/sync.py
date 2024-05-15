from os.path import basename, splitext

from ats import match_start, expand_matches, print_batches, do_batch, write_srt, write_vtt
from src.utils import get_tqdm

tqdm = get_tqdm()

def fuzzy_match_chapters(streams, chapters, cache):
    print('Fuzzy matching chapters...')
    ats, sta = match_start(streams, chapters, cache)
    audio_batches = expand_matches(streams, chapters, ats, sta)
    print_batches(audio_batches)
    return audio_batches

#TODO decouple output formatting
def sync(output_dir, output_format, streams, chapters, cache, overwrite, temperature, args):
    nopend = set(args.pop('nopend_punctuations'))
    audio_batches = fuzzy_match_chapters(streams, chapters, cache)
    print('Syncing...')
    with tqdm(audio_batches) as bar:
        for ai, batches in enumerate(bar):
            out = output_dir / (splitext(basename(streams[ai][2][0].path))[0] + '.' + output_format)
            if not overwrite and out.exists():
                bar.write(f"{out.name} already exists, skipping.")
                continue

            bar.set_description(basename(streams[ai][2][0].path))
            offset, segments = 0, []
            for ajs, (chi, chjs), _ in tqdm(batches):
                ach = [(streams[ai][2][aj].transcribe(model, cache, temperature=temperature, **args), streams[ai][2][aj].duration) for aj in ajs]
                tch = [chapters[chi][1][chj] for chj in chjs]
                if tch:
                    segments.extend(do_batch(ach, tch, set(args['prepend_punctuations']), set(args['append_punctuations']), nopend, offset))

                offset += sum(a[1] for a in ach)

            if not segments:
                continue

            with out.open("w", encoding="utf8") as o:
                if output_format == 'srt':
                    write_srt(segments, o)
                elif output_format == 'vtt':
                    write_vtt(segments, o)
