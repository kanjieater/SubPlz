import whisper
from whisper.decoding import DecodingOptions, DecodingResult
# from decoding import DecodingTask

import matplotlib.pyplot as plt
from huggingface import modify_model

import ebooklib
from ebooklib import epub
import xml.etree.ElementTree as ET

import os
import numpy as np
from pprint import pprint
from dataclasses import dataclass, field, replace
from time import time
from tqdm import tqdm
from whisper.tokenizer import get_tokenizer
from whisper import audio
import torch
from torch.distributions import Categorical
import argparse
import ffmpeg
import multiprocessing
from quantization import ptdq_linear
from fuzzywuzzy import fuzz
from itertools import chain
import functools
from functools import partialmethod
from dataclasses import dataclass
from pathlib import Path
import torch.nn.functional as F

from Bio import Align
import regex as re
import math

def sexagesimal(secs):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    return f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>6.3f}'

@dataclass(eq=True, frozen=True)
class Segment:
    text: str
    # words: Segment
    start: float
    end: float
    def __repr__(self):
        return f"Segment(text='{self.text}', start={sexagesimal(self.start)}, end={sexagesimal(self.end)})"
    def vtt(self):
        return f"{sexagesimal(self.start)} --> {sexagesimal(self.end)}\n{self.text}"

@dataclass(eq=True, frozen=True)
class Cache:
    model_name: str
    cache_dir: str
    enabled: bool
    ask: bool
    overwrite: bool

    def get(self, filename, chid):
        if not self.enabled: return
        fn = (filename + '.' + str(chid) +  '.' + self.model_name + ".subs") # Include the hash of the model settings?
        if (q := Path(self.cache_dir) / fn).exists():
            return eval(q.read_bytes().decode("utf-8"))

    def put(self, filename, chid, content):
        if not self.enabled: return content
        cd = Path(self.cache_dir)
        cd.mkdir(parents=True, exist_ok=True)
        q = cd / (filename + '.' + str(chid) +  '.' + self.model_name + ".subs")
        if q.exists():
            if self.ask:
                while (k := input(f"Cache for file {filename}, chapter id {chid} already exists. Overwrite?  [y/n/Y/N] (yes, no, yes/no and don't ask again) ").strip()) not in ['y', 'n', 'Y', 'N']: pass
                self.ask = not (k == 'N' or k == 'Y')
                self.overwrite = k == 'Y' or k == 'y'
            if not self.overwrite: return content
        q.write_bytes(repr(content).encode('utf-8'))
        return content

@dataclass
class AudioStream:
    stream: ffmpeg.Stream
    path: Path
    cn: str
    cid: int

    def audio(self):
        data, _ = self.stream.output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k').run(quiet=True, input='')
        return np.frombuffer(data, np.int16).astype(np.float32) / 32768.0

    def transcribe(self, model, cache, **kwargs):
        if r := cache.get(os.path.basename(self.path), self.cid): return r
        r = model.transcribe(self.audio(), **kwargs)
        return cache.put(os.path.basename(self.path), self.cid, r)

    @classmethod
    def from_file(cls, path):
        info = ffmpeg.probe(path, show_chapters=None)
        title = info.get('format', {}).get('tags', {}).get('title', os.path.basename(path))
        if 'chapters' not in info or len(info['chapters']) < 1:
            return title, [cls(stream=ffmpeg.input(path), path=path, cn=title, cid=0)]
        return title, [cls(stream=ffmpeg.input(path, ss=float(chapter['start_time']), to=float(chapter['end_time'])),
                    path=path,
                    cn=chapter.get('tags', {}).get('title', ''),
                    cid=chapter['id'])
                for chapter in info['chapters']]


@dataclass(eq=True, frozen=True)
class Epub:
    epub: epub.EpubBook
    title: str
    start: int
    end: int

    @functools.lru_cache(maxsize=None)
    def xml(item):
        return ET.fromstring(item.get_content().decode('utf8'))

    # Pray that the xml isn't too large
    @functools.lru_cache(maxsize=None) # probably not very useful
    def find_with_path(xml, tid):
        for i, c in enumerate(xml):
            if 'id' in c.attrib and c.attrib['id'] == tid:
                return (i,), c
            elif (k := Epub.find_with_path(c, tid)) is not None:
                return ((i,) + k[0], k[1])

    def text(self, prefix, ignore=set()):
        def add(o, z, idx):
            if z and (k := ' '.join(i.strip() for i in z.strip().split('\n'))):
                o.append((idx, k))

        def to_text(idx, xml, parent, append=[], append2=[], prepend=[], prepend2=[], skip=set()):
            if idx in skip or xml.tag in ignore or ("}" in xml.tag and xml.tag.split("}")[1] in ignore):
                return [], []

            o, a = [], []
            add(o, xml.text, idx)
            # All of this just for 幼女戦記's audiobooks
            for i, v in enumerate(xml):
                no, na = to_text(idx + (i,), v, parent, skip=skip)
                o.extend(no)
                a.extend(list(map(lambda x: (idx, x), prepend2)))
                a.extend(na)
                a.extend(list(map(lambda x: (idx, x), append2)))
                add(o, v.tail, idx)

            # o.extend(list(zip([idx]*len(prepend), prepend))) # Exactly the same number of chars lol
            o.extend(list(map(lambda x: (idx, x), prepend)))
            o.extend(a)
            o.extend(list(map(lambda x: (idx, x), append)))

            if 'href' in xml.attrib and "#" in xml.attrib['href']: # Make sure it isn't an entire document
                ref, tid = xml.attrib['href'].split("#")
                item = self.epub.get_item_with_href(('' if '/' in ref else (parent + '/')) + ref) # idk check this again
                idx = [i for i, _ in self.epub.spine].index(item.id)
                path, element = Epub.find_with_path(Epub.xml(item), tid)
                skip.add((idx,) + path) # Not sure if this is correct, it only works if 1. the href is in the same file, 2. the content comes *after* the href
                return o, to_text((idx,) + path, element, parent)[0]
            return o, []

        # TODO Youjo senki has quotes on its images, use ocr? add another options to replace the images with some text?
        r = [((self.start,), self.title)] if prefix else []
        for i in range(self.start, self.end):
            id, is_linear = self.epub.spine[i]
            item = self.epub.get_item_with_id(id)
            if is_linear and item.media_type == "application/xhtml+xml":
                x, l = to_text((i,), Epub.xml(item), '/'.join(item.file_name.split("/")[:-1]))
                r.extend(x)
                # r.extend(l)
        return r

    @classmethod
    def from_file(cls, path):
        file = epub.read_epub(path)
        toc = [file.get_item_with_href(x.href.split("#")[0]) for x in file.toc]
        idx, c = [], 0
        # Uhh?? See 【小説29巻】本好きの下剋上～司書になるためには手段を選んでいられません～第五部「女神の化身VIII」.epub
        # TODO: check the audiobook when it gets released
        # for i in range(len(file.spine)):
        #     v = file.spine[i]
        #     if v[0] == toc[c].id:
        #         idx.append(i)
        #         c+=1
        #         if c == len(toc): break

        k = 0
        while len(toc) > c:
            for i in range(idx[-1]+1 if len(idx) else 0, len(file.spine)):
                v = file.spine[i]
                if v[0] == toc[c].id:
                    idx.append(i)
                    c += 1
                    if c == len(toc): break
            idx.append(idx[-1])
            c += 1
            k += 1
        if k > 1: print(file.title, "has a broken toc")
        idx[-1] = len(file.spine)
        return [cls(epub=file, title=file.toc[i].title, start=idx[i], end=idx[i+1]) for i in range(len(toc))]

def match(audio, text):
    ats = {}
    for ai in range(len(audio)):
        afn, at, ac = audio[ai]
        for ti in range(len(text)):
            tfn, tc = text[ti]
            main = fuzz.ratio(tfn, afn)
            if type(tc[0]) != str:
                main = max(main, fuzz.ratio(tfn + tc[0].epub.title, afn + at))
                for aci in range(len(ac)):
                    best = ats.get((ai, aci), (-1, -1, 0))
                    for tci in range(len(tc)):
                        score = fuzz.ratio(afn + at + ac[aci].cn, tfn + tc[tci].epub.title + tc[tci].title)
                        if score > best[-1] and score > main:
                            best = (ti, tci, score)
                    if best != (-1, -1, 0): ats[(ai, aci)] = best
            if main > ats.get((ai, -1), (-1, -1, 0))[-1]:
                ats[(ai, -1)] = (ti, -1, main)

    # for k, v in ats.items():
    #     if k[-1] != -1:
    #         print(audio[k[0]][2][k[1]].cn, text[v[0]][1][v[1]].title)
    #     else:
    #         print(audio[k[0]][1], text[v[0]][1][0].epub.title)
    return {k: v[:-1] for k,v in ats.items()}#, {v[:-1]: k for k,v in ats.items()}

def clean(x):
    ascii_to_wide = dict((i, chr(i + 0xfee0)) for i in range(0x21, 0x7f))
    ascii_to_wide.update({0x20: '\u3000', 0x2D: '\u2212'})  # space and minus
    kata_hira = dict((0x30a1 + i, chr(0x3041 + i)) for i in range(0x56))
    kansuu_to_ascii = dict([(ord('一'), '１'), (ord('二'), '２'), (ord('三'), '３'), (ord('四'), '４'), (ord('五'), '５'), (ord('六'), '６'), (ord('七'), '７'), (ord('八'), '８'), (ord('九'), '９'), (ord('零'), '０'), (ord('十'), '１')])
    allt = kata_hira | kansuu_to_ascii | ascii_to_wide
    return [re.sub("[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}|\s|ー]+", "", i).translate(allt) for i in x]

def align(model, transcript, text):
    transcript_str = [i['text'] for i in transcript['segments']]
    text_str = [i[1] for i in text]
    transcript_str_clean, text_str_clean = clean(transcript_str), clean(text_str)
    for i in range(10):
        print(text_str[i])

    aligner = Align.PairwiseAligner(scoring=None, mode='global', open_gap_score=-1, mismatch_score=-1, extend_gap_score=-1)
    alignment = aligner.align(''.join(text_str_clean), ''.join(transcript_str_clean))[0]
    s, e = 0, 140
    print(alignment[0, s:e].__str__().replace("-", "ー"))
    print(alignment[1, s:e].__str__().replace("-", "ー"))

    coords = alignment.coordinates#.T#[:50]
    ps, pe = 0, 0
    ss, se = 0, 0
    pidx, cidx = 0, 0
    ppos, cpos = 0, 0
    pposr, cposr = 0, 0
    while pe < len(text_str) and se < len(transcript_str):
        if ppos == cpos:
            print(''.join(text_str[ps:pe]))
            print(''.join(transcript_str[ss:se]))
            print(ps, pe)
            print(ss, se)
            print()
            ps = pe
            ss = se
            cposr += len(transcript_str_clean[se])
            cpos += len(transcript_str_clean[se])
            while cposr > coords[1][cidx]:
                cidx += 1
                if coords[1][cidx] == coords[1][cidx-1]: cpos += coords[0][cidx] - coords[0][cidx-1]
            se += 1
            pposr += len(text_str_clean[pe])
            ppos += len(text_str_clean[pe])
            while pposr > coords[0][pidx]:
                pidx += 1
                if coords[0][pidx] == coords[0][pidx-1]: ppos += coords[1][pidx] - coords[1][pidx-1]
            pe += 1
        elif ppos > cpos:
            cposr += len(transcript_str_clean[se])
            cpos += len(transcript_str_clean[se])
            while cposr > coords[1][cidx]:
                cidx += 1
                if coords[1][cidx] == coords[1][cidx-1]: cpos += coords[0][cidx] - coords[0][cidx-1]
            se += 1
        elif cpos > ppos:
            pposr += len(text_str_clean[pe])
            ppos += len(text_str_clean[pe])
            while pposr > coords[0][pidx]:
                # print(coords[0][pidx])
                pidx += 1
                if coords[0][pidx] == coords[0][pidx-1]: ppos += coords[1][pidx] - coords[1][pidx-1]

            # print(coords[0][pidx])
            pe += 1
    print(''.join(text_str[ps:pe]))
    print(''.join(transcript_str[ss:se]))
    print()



    # print(coords)
    # pprint(alignment.coordinates[:, :50])
    # print(alignment[1, s:e])
    # print(alignment[:, :100])

#     pprint(transcript_str)
#     pprint(text_str)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument( "--audio", nargs="+", required=True, help="list of audio files to process (in the correct order)")
    parser.add_argument("--text", nargs="+", required=True, help="path to the script file")
    parser.add_argument("--model", default="tiny", help="whisper model to use. can be one of tiny, small, large, huge")
    parser.add_argument("--language", default="ja", help="language of the script and audio")
    parser.add_argument("--progress", default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-cache", default=True, help="whether to use the cache or not", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache-dir", default="AudiobookTextSyncCache", help="the cache directory")
    parser.add_argument("--overwrite-cache", default=False, action=argparse.BooleanOptionalAction, help="Always overwrite the cache")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count(), help=r"number of threads")
    parser.add_argument("--device", default="cpu", help="device to do inference on")
    parser.add_argument("--dynamic-quantization", "--dq", default=False, help="Use torch's dynamic quantization (cpu only)", action=argparse.BooleanOptionalAction)

    parser.add_argument("--fast-decoder", default=False,help="Use hugging face's decoding method, currenly incomplete", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast-decoder-overlap", type=int, default=10,help="Overlap between each batch")
    parser.add_argument("--fast-decoder-batches", type=int, default=1, help="Number of batches to operate on")

    parser.add_argument("--ignore-tags", default=['rt'], nargs='+', help="Tags to ignore during the epub to text conversion, useful for removing furigana")
    parser.add_argument("--prefix-chapter-name", default=True, help="Whether to prefix the text of each chapter with its name")

    parser.add_argument("--fp16", default=False, help="whether to perform inference in fp16", action=argparse.BooleanOptionalAction)
    parser.add_argument("--beam_size", type=int, default=None, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop", action=argparse.BooleanOptionalAction)

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--temperature_increment_on_fallback", type=float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", default=False, help="(experimental) extract word-level timestamps and refine the results based on them", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--highlight_words", default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max_line_width", type=int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--max_words_per_line", type=int, default=None, help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment")
    # TODO
    # parser.add_argument("--output-file", default=None, help="name of the output subtitle file")
    # parser.add_argument("--split-script", default="", help=r"the regex to split the script with. for monogatari it is something like ^\s[\uFF10-\uFF19]*\s$")
    args = parser.parse_args().__dict__

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)
    # if args.output_file is None:
    #     args.output_file = os.path.splitext(args.audio_files[0])[0] + ".vtt"
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.pop('progress'))

    model = args.pop("model")
    device = args.pop('device')

    overwrite_cache = args.pop('overwrite_cache')
    cache = Cache(model_name=model, enabled=args.pop("use_cache"), cache_dir=args.pop("cache_dir"),
                  ask=not overwrite_cache, overwrite=overwrite_cache)
    model = whisper.load_model(model).to(device)
    if device == "cpu" and args.pop('dynamic_quantization'):
        ptdq_linear(model)

    overlap, batches = args.pop("fast_decoder_overlap"), args.pop("fast_decoder_batches")
    if args.pop("fast_decoder"):
        args["overlap"] = overlap
        args["batches"] = batches
        modify_model(model)

    streams = [(os.path.basename(f), *AudioStream.from_file(f)) for f in args.pop('audio')]
    chapters = [(os.path.basename(i), Epub.from_file(i)) if i.split(".")[-1] == 'epub' else (os.path.basename(i), [Path(i).read_text()])
                for i in args.pop('text')]
    ats = match(streams, chapters)

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

    ignore_tags = set(args.pop('ignore_tags'))
    prefix_chapter_name = args.pop('prefix_chapter_name')
    for i, v in enumerate(streams):
        for j, v in enumerate(v[2]):
            if (i, j) in ats:
                ci, cj = ats[(i, j)]
                text = chapters[ci][1][cj].text(prefix_chapter_name, ignore=ignore_tags)
                transcript = v.transcribe(model, cache, temperature=temperature, **args)
                align(model, transcript, text)
                # pprint(transcript['segments'])
                # pprint(text)
                # exit(0)
