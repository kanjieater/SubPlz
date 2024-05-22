import os
import argparse
import subprocess
import tempfile
from pprint import pprint
from types import MethodType
from lang import get_lang
from wcwidth import wcswidth

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell' or shell == 'google.colab._shell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return True  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    from tqdm.notebook import tqdm, trange
else:
    from tqdm import tqdm, trange

from functools import partialmethod, reduce
from itertools import groupby, takewhile, chain
from dataclasses import dataclass
from pathlib import Path

import multiprocessing
import concurrent.futures as futures

import torch
import numpy as np
import whisper

import align
from huggingface import modify_model
from quantization import ptdq_linear
from faster_whisper import WhisperModel

from rapidfuzz import fuzz

from bs4 import element
from bs4 import BeautifulSoup

from os.path import basename, splitext
import time

from audio import AudioFile, TranscribedAudioStream, TranscribedAudioFile
from text import TextFile, SubFile


def sexagesimal(secs, use_comma=False):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    r = f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>6.3f}'
    if use_comma:
        r = r.replace('.', ',')
    return r

@dataclass(eq=True)
class Segment:
    text: str
    # words: Segment
    start: float
    end: float
    def __repr__(self):
        return f"Segment(text='{self.text}', start={sexagesimal(self.start)}, end={sexagesimal(self.end)})"
    def vtt(self, use_comma=False):
        return f"{sexagesimal(self.start, use_comma)} --> {sexagesimal(self.end, use_comma)}\n{self.text}"

def write_srt(segments, o):
    o.write('\n\n'.join(str(i+1)+'\n'+s.vtt(use_comma=True) for i, s in enumerate(segments)))

def write_vtt(segments, o):
    o.write("WEBVTT\n\n"+'\n\n'.join(s.vtt() for s in segments))

@dataclass(eq=True)
class Cache:
    model_name: str
    cache_dir: str
    enabled: bool
    ask: bool
    overwrite: bool

    def get_name(self, filename, chid):
        return filename + '.' + str(chid) +  '.' + self.model_name + ".subs"

    def get(self, filename, chid): # TODO Fix this crap
        if not self.enabled: return
        fn = self.get_name(filename, chid)
        fn2 = filename + '.' + str(chid) +  '.' + 'small' + ".subs"
        fn3 = filename + '.' + str(chid) +  '.' + 'base' + ".subs"
        if not self.enabled: return
        if (q := Path(self.cache_dir) / fn).exists():
            return eval(q.read_bytes().decode("utf-8"))
        if (q := Path(self.cache_dir) / fn2).exists():
            return eval(q.read_bytes().decode("utf-8"))
        if (q := Path(self.cache_dir) / fn3).exists():
            return eval(q.read_bytes().decode("utf-8"))

    def put(self, filename, chid, content):
        if not self.enabled: return content
        cd = Path(self.cache_dir)
        cd.mkdir(parents=True, exist_ok=True)
        fn =  self.get_name(filename, chid)
        p = cd / fn

        if 'text' in content:
            del content['text']
        if 'ori_dict' in content:
            del content['ori_dict']

        # Some of these may be useful but they just take so much space
        for i in content['segments']:
            if 'words' in i:
                del i['words']
            del i['id']
            del i['tokens']
            del i['avg_logprob']
            del i['temperature']
            del i['seek']
            del i['compression_ratio']
            del i['no_speech_prob']

        p.write_bytes(repr(content).encode('utf-8'))
        return content

def match_start(audio, text, prepend, append, nopend):
    ats, sta = {}, {}
    textcache = {}
    for ai, afile in enumerate(tqdm(audio)):
        for i, ach in enumerate(tqdm(afile.chapters)):
            if (ai, i) in ats: continue

            lang = get_lang(ach.language, prepend, append, nopend)
            acontent = lang.normalize(lang.clean(''.join(seg['text'] for seg in ach.segments)))

            best = (-1, -1, 0)
            for ti, tfile in enumerate(text):
                for j, tch in enumerate(tfile.chapters):
                    if (ti, j) in sta: continue

                    if (ti, j) not in textcache:
                        textcache[ti, j] = lang.normalize(lang.clean(''.join(p.text() for p in tfile.chapters[j].text())))
                    tcontent = textcache[ti, j]
                    if len(acontent) < 100 or len(tcontent) < 100: continue

                    limit = min(len(tcontent), len(acontent), 2000)
                    score = fuzz.ratio(acontent[:limit], tcontent[:limit])
                    if score > 40 and score > best[-1]:
                        best = (ti, j, score)

            if best[:-1] in sta:
                tqdm.write("WARNING match_start")
            elif best != (-1, -1, 0):
                ats[ai, i] = best
                sta[best[:-1]] = (ai, i, best[-1])

    return ats, sta

# I hate it
def expand_matches(audio, text, ats, sta):
    audio_batches = []
    for ai, a in enumerate(audio):
        batch = []
        def add(idx, other=[]):
            chi, chj, _ = ats[ai, idx]
            z = [chj] + list(takewhile(lambda j: (chi, j) not in sta, range(chj+1, len(text[chi].chapters))))
            batch.append(([idx]+other, (chi, z), ats[ai, idx][-1]))

        prev = None
        for t, it in groupby(range(len(a.chapters)), key=lambda aj: (ai, aj) in ats):
            k = list(it)
            if t:
                for i in k[:-1]: add(i)
                prev = k[-1]
            elif prev is not None:
                add(prev, k)
            else:
                batch.append((k, (-1, []), None))

        if prev == len(a.chapters)-1:
            add(prev)
        audio_batches.append(batch)
    return audio_batches

# Takes in the original not the transcribed classes
def print_batches(batches, audio, text, spacing=2, sep1='=', sep2='-'):
    rows = [1, ["Audio", "Text", "Score"]]
    width = [wcswidth(h) for h in rows[-1]]

    for ai, batch in enumerate(batches):
        use_audio_header = len(audio[ai].chapters) > 1

        texts = [chi for _, (chi, _), _ in batch if chi != -1]
        text_unique = all([i == texts[0] for i in texts])
        use_text_header = text_unique and len(batch[0][1][1]) > 3

        if use_audio_header or use_text_header:
            rows.append(1)
            rows.append([audio[ai].title, '', ''])
            use_audio_header = True
            if text_unique:
                rows[-1][1] = text[batch[0][1][0]].title
                use_text_header = True
            width[0] = max(width[0], wcswidth(rows[-1][0]))
            width[1] = max(width[1], wcswidth(rows[-1][1]))
        rows.append(1)
        for ajs, (chi, chjs), score in batch:
            a = [audio[ai].chapters[aj] for aj in ajs]
            t = [text[chi].chapters[chj] for chj in chjs]
            for i in range(max(len(a), len(t))):
                row = ['', '' if t else '?', '']
                if i < len(a):
                    row[0] = (audio[ai].title + "::" if not use_audio_header else '') + a[i].title
                    width[0] = max(width[0], wcswidth(row[0]))
                if i < len(t):
                    row[1] = (text[chi].title + "::" if not use_text_header else '') + t[i].title.strip()
                    width[1] = max(width[1], wcswidth(row[1]))
                if i == 0:
                    row[2] = format(score/100, '.2%') if score is not None else '?'
                    width[2] = max(width[2], wcswidth(row[2]))
                rows.append(row)
            rows.append(2)
        rows = rows[:-1]
    rows.append(1)

    for row in rows:
        csep = ' ' * spacing
        if type(row) is int:
            sep = sep1 if row == 1 else sep2
            print(csep.join([sep*w for w in width]))
            continue
        print(csep.join([r.ljust(width[i]-wcswidth(r)+len(r)) for i, r in enumerate(row)]))

def to_epub():
    pass

def to_subs(text, subs, alignment, offset, references):
    alignment = [t + [i] for i, a in enumerate(alignment) for t in a]
    start, end = 0, 0
    segments = []
    for si, s in enumerate(subs):
        while end < len(alignment) and alignment[end][-2] == si:
            end += 1

        r = ''
        for a in alignment[start:end]:
            r += text[a[-1]].text()[a[0]:a[1]]

        if r.strip():
            if False: # Debug
                r = s['text']+'\n'+r
            segments.append(Segment(text=r, start=s['start']+offset, end=s['end']+offset))
        else:
            segments.append(Segment(text='＊'+s['text'], start=s['start']+offset, end=s['end']+offset))

        start = end
    return segments

def do_batch(ach, tch, prepend, append, nopend, offset):
    acontent = []
    boff = 0
    for a in ach:
        for p in a[0].segments:
            p['start'] += boff
            p['end'] += boff
            acontent.append(p)
        boff += a[1]

    language = get_lang(ach[0][0].language, prepend, append, nopend)

    tcontent = [p for t in tch for p in t.text()]
    alignment, references = align.align(None, language, [p['text'] for p in acontent], [p.text() for p in  tcontent], [], set(prepend), set(append), set(nopend))
    return to_subs(tcontent, acontent, alignment, offset, None)

def faster_transcribe(self, audio, **args):
    name = args.pop('name')

    args['log_prob_threshold'] = args.pop('logprob_threshold')
    args['beam_size'] = args['beam_size'] if args['beam_size'] else 1
    args['patience'] = args['patience'] if args['patience'] else 1
    args['length_penalty'] = args['length_penalty'] if args['length_penalty'] else 1

    gen, info = self.transcribe2(audio, best_of=1, **args)

    segments, prev_end = [], 0
    with tqdm(total=info.duration, unit_scale=True, unit=" seconds") as pbar:
        pbar.set_description(f'{name}')
        for segment in gen:
            segments.append(segment._asdict())
            pbar.update(segment.end - prev_end)
            prev_end = segment.end
        pbar.update(info.duration - prev_end)
        pbar.refresh()

    return {'segments': segments, 'language': args['language'] if 'language' in args else info.language}


def parse_indices(s, l):
    ss, r = s.split(), set()
    for a in ss:
        try:
            if a[0] == '^':
                val = int(a[1:])
                r = r.union(range(l)) - {val}
            elif len(k := a.split('-')) > 1:
                val1 = min(int(k[0]), l)
                val2 = min(int(k[1]), l)
                r = r.union(range(val1, val2+1))
            else:
                if (val1 := int(a)) < l:
                    r.add(val1)
        except ValueError:
            return
    return r


def alass(output_dir, alass_path, alass_args, args):
    audio = list(chain.from_iterable(AudioFile.from_dir(f, track=args['language'], whole=True) for f in args.pop('audio')))
    text = list(chain.from_iterable(TextFile.from_dir(f) for f in args.pop('text')))
    if not all(isinstance(t, SubFile) for t in text):
        print('--alass inputs should be subtitle files')
        return
    if len(audio) != len(text):
        print("len(audio) != len(text), input needs to be in order for alass alignment")

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True) # onnx is much faster
    (get_speech_timestamps, _, _, *_) = utils

    with tqdm(zip(audio, text), total=len(audio)) as bar:
        for a, t in bar:
            bar.set_description(f'Running VAD on {a.title}')
            v = get_speech_timestamps(a.audio(), model, sampling_rate=16000, return_seconds=True)
            bar.set_description(f'Aligning {t.title} with {a.title}')
            segments = [Segment(text='h', start=s['start'], end=s['end']) for s in v]
            with tempfile.NamedTemporaryFile(mode="w", suffix='.srt') as f:
                write_srt(segments, f)
                cmd = [alass_path, *['-'+h for h in alass_args], f.name, str(t.path), str(output_dir / (a.path.stem + ''.join(t.path.suffixes)))]
                print(' '.join(cmd))
                try:
                    subprocess.run(cmd)
                except CalledProcessError as e:
                    raise RuntimeError(f"Alass command failed: {e.stderr.decode()}\n args: {' '.join(cmd)}") from e

def main():
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument("--audio", nargs="+", type=Path, required=True, help="list of audio files to process (in the correct order)")
    parser.add_argument("--text", nargs="+", type=Path, required=True, help="path to the script file")

    parser.add_argument("--model", default="tiny", help="whisper model to use. can be one of tiny, small, large, huge")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to do inference on")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count(), help=r"number of threads")
    parser.add_argument("--language", default=None, help="language of the script and audio")
    parser.add_argument("--whole", default=False, help="Do the alignment on whole files, don't split into chapters", action=argparse.BooleanOptionalAction)
    parser.add_argument("--local-only", default=False, help="Don't download outside models", action=argparse.BooleanOptionalAction)

    parser.add_argument("--alass", default=False, help="Use vad+alass to realign, inputs need to be in-order, this is temporary until I figure out something better (implies --whole)", action=argparse.BooleanOptionalAction)
    parser.add_argument("--alass-path", default='alass', help="path to alass")
    parser.add_argument("--alass-args", default=['O0'], nargs="+", help="additional arguments to alass (pass without the dash, eg: O1)")

    parser.add_argument("--progress", default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite", default=False,  help="Overwrite any destination files", action=argparse.BooleanOptionalAction)

    parser.add_argument("--use-cache", default=True, help="whether to use the cache or not", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache-dir", default="AudiobookTextSyncCache", help="the cache directory")
    parser.add_argument("--overwrite-cache", default=False, action=argparse.BooleanOptionalAction, help="Always overwrite the cache")

    parser.add_argument('--quantize', default=True, help="use fp16 on gpu or int8 on cpu", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dynamic-quantization", "--dq", default=False, help="Use torch's dynamic quantization (cpu only)", action=argparse.BooleanOptionalAction)

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

    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-『「（〈《〔【｛［‘“〝※", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--nopend_punctuations", type=str, default="うぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ…\u3000\x20", help="TODO")

    parser.add_argument("--word_timestamps", default=False, help="(experimental) extract word-level timestamps and refine the results based on them", action=argparse.BooleanOptionalAction)
    parser.add_argument("--highlight_words", default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_line_width", type=int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--max_words_per_line", type=int, default=None, help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment")

    parser.add_argument("--output-dir", default=u'.', type=Path, help="Output directory, default uses the directory for the first audio file")
    parser.add_argument("--output-format", default='srt', help="Output format, currently only supports vtt and srt")

    args = parser.parse_args().__dict__
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.pop('progress'))
    if (threads := args.pop("threads")) > 0: torch.set_num_threads(threads)

    output_dir = args.pop('output_dir')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = args.pop('output_format')

    alass_path, alass_args = args.pop('alass_path'), args.pop('alass_args')
    if args.pop('alass'):
        alass(output_dir, alass_path, alass_args, args)
        exit(0)

    model, device = args.pop("model"), args.pop('device')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"Using device: {device}")

    file_overwrite, overwrite_cache = args.pop('overwrite'), args.pop('overwrite_cache')
    cache = Cache(model_name=model, enabled=args.pop("use_cache"), cache_dir=args.pop("cache_dir"),
                  ask=not overwrite_cache, overwrite=overwrite_cache)

    faster_whisper, local_only, quantize = args.pop('faster_whisper'), args.pop('local_only'), args.pop('quantize')
    fast_decoder, overlap, batches = args.pop('fast_decoder'), args.pop("fast_decoder_overlap"), args.pop("fast_decoder_batches")
    dq = args.pop('dynamic_quantization')
    if faster_whisper:
        model = WhisperModel(model, device, local_files_only=local_only, compute_type='float32' if not quantize else ('int8' if device == 'cpu' else 'float16'), num_workers=threads)
        model.transcribe2 = model.transcribe
        model.transcribe = MethodType(faster_transcribe, model)
    else:
        model = whisper.load_model(model, device)
        args['fp16'] = quantize and device != 'cpu'
        if args['fp16']:
            model = model.half()
        elif dq:
            ptdq_linear(model)

        if fast_decoder:
            args["overlap"] = overlap
            args["batches"] = batches
            modify_model(model)

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    word_options = [
        "highlight_words",
        "max_line_count",
        "max_line_width",
        "max_words_per_line",
    ]
    if not args["word_timestamps"]:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} requires --word_timestamps True")

    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    if args["max_words_per_line"] and args["max_line_width"]:
        warnings.warn("--max_words_per_line has no effect with --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}
    word_timestamps = args.pop("word_timestamps")

    prepend, append, nopend = [args.pop(i+'_punctuations') for i in ['prepend', 'append', 'nopend']]
    whole = args.pop('whole')

    print("Loading...")
    audio = list(chain.from_iterable(AudioFile.from_dir(f, track=args['language'], whole=whole) for f in args.pop('audio')))
    text = list(chain.from_iterable(TextFile.from_dir(f) for f in args.pop('text')))

    print('Transcribing...')
    s = time.monotonic()
    transcribed_audio = []

    # Trash code
    # TODO: This really doesn't have much of a perf improvement
    # Get rid of it and update faster-whisper to support batching
    with futures.ThreadPoolExecutor(max_workers=threads) as p:
        in_cache = []
        for i, a in enumerate(audio):
            for j, c in enumerate(a.chapters):
                if cache.get(a.path.name, c.id): # Cache the result here and not in the class?
                    in_cache.append((i, j))

        if cache.ask and len(in_cache):
            for i, v in enumerate(in_cache):
                name = audio[v[0]].title+'/'+audio[v[0]].chapters[v[1]].title
                print(('{0: >' + str(len(str(len(in_cache))))+ '} {1}').format(i, name))
            indices = None
            while indices is None:
                inp = input('Choose cache files to overwrite: (eg: "1 2 3", "1-3", "^4" (empty for none))\n>> ') # Taken from yay
                if (indices := parse_indices(inp, len(in_cache))) is None:
                    print("Parsing failed")
            overwrite = {in_cache[i] for i in indices}
        else:
            overwrite = set(in_cache) if overwrite_cache else set()

        fs = []
        for i, a in enumerate(audio):
            cf = []
            for j, c in enumerate(a.chapters):
                if (i, j) not in overwrite and (t := cache.get(a.path.name, c.id)):
                    l = lambda c=c, t=t: TranscribedAudioStream.from_map(c, t)
                else:
                    l = lambda c=c: TranscribedAudioStream.from_map(c, cache.put(a.path.name, c.id, model.transcribe(c.audio(), name=c.title, temperature=temperature, **args)))
                cf.append(p.submit(l))
            fs.append(cf)

        transcribed_audio =  [TranscribedAudioFile(file=audio[i], chapters=[r.result() for r in f]) for i, f in enumerate(fs)]
    print(f"Transcribing took: {time.monotonic()-s:.2f}s")

    print('Fuzzy matching chapters...')
    ats, sta = match_start(transcribed_audio, text, prepend, append, nopend)
    audio_batches = expand_matches(transcribed_audio, text, ats, sta)
    print_batches(audio_batches, audio, text)

    print('Syncing...')
    with tqdm(audio_batches) as bar:
        for ai, batches in enumerate(bar):
            out = output_dir / (audio[ai].path.stem + '.' + output_format)
            if not file_overwrite and out.exists():
                bar.write(f"{out.name} already exists, skipping.")
                continue

            bar.set_description(audio[ai].path.name)
            offset, segments = 0, []
            for ajs, (chi, chjs), _ in tqdm(batches):
                ach = [(transcribed_audio[ai].chapters[aj], audio[ai].chapters[aj].duration) for aj in ajs]
                tch = [text[chi].chapters[chj] for chj in chjs]
                if tch:
                    segments.extend(do_batch(ach, tch, prepend, append, nopend, offset))

                offset += sum(a[1] for a in ach)

            if not segments:
                continue

            with out.open("w") as o:
                if output_format == 'srt':
                    write_srt(segments, o)
                elif output_format == 'vtt':
                    write_vtt(segments, o)

if __name__ == "__main__":
    main()
