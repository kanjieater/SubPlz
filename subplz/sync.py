from os.path import basename, splitext

from ats.main import match_start, expand_matches, print_batches, write_srt, write_vtt, to_subs
from ats import align

from ats.lang import get_lang
from subplz.utils import get_tqdm

tqdm = get_tqdm()

def fuzzy_match_chapters(streams, chapters, cache):
    print('Fuzzy matching chapters...')
    print(streams)
    ats, sta = match_start(streams, chapters, cache)
    audio_batches = expand_matches(streams, chapters, ats, sta)
    print_batches(audio_batches)
    return audio_batches

def do_batch(ach, tch, prepend, append, nopend, offset):
    acontent = []
    boff = 0
    for a in ach:
        for p in a[0]['segments']:
            p['start'] += boff
            p['end'] += boff
            acontent.append(p)
        boff += a[1]

    language = get_lang(ach[0][0]['language'])

    tcontent = [p for t in tch for p in t.text()]
    alignment, references = align.align(None, language, [p['text'] for p in acontent], [p.text() for p in  tcontent], [], prepend, append, nopend)
    return to_subs(tcontent, acontent, alignment, offset, None)

#TODO decouple output formatting
def sync(output_dir, output_format, model, streams, chapters, cache, temperature, args):
    nopend = set(args.pop('nopend_punctuations'))
    audio_batches = fuzzy_match_chapters(streams, chapters, cache)
    print('Syncing...')
    with tqdm(audio_batches) as bar:
        for ai, batches in enumerate(bar):
            out = output_dir / (splitext(basename(streams[ai][2][0].path))[0] + '.' + output_format)
            if not cache.overwrite and out.exists():
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
