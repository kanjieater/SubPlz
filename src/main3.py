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
# import align
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
        fn = (filename + '.' + str(chid) +  '.' + self.model_name + ".subs") # Include hash of the model settings?
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
        if 'chapters' not in info or len(info['chapters']) < 1:
            return [cls(stream=ffmpeg.input(path),
                        path=path, cn=os.path.basename(path), cid=0)]
        return [cls(stream=ffmepg.input(path, ss=float(chapter['start_time']), to=float(chapter['end_time'])),
                    path=path,
                    cn='' if 'tags' not in chapter or 'title' not in chapter['tags'] else chapter['tags']['title'],
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

    def text(self, ignore=set()):
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
                return o, to_text((idx,) + path, element, parent, skip={})[0]
            return o, []

        r = []
        for i in range(self.start, self.end):
            id, is_linear = self.epub.spine[i]
            item = self.epub.get_item_with_id(id)
            if is_linear and item.media_type == "application/xhtml+xml":
                x, l = to_text((i,), Epub.xml(item), ''.join(item.file_name.split("/")[:-1]))
                r.extend(x)
                # r.extend(l)
        return r

    @classmethod
    def from_file(cls, path):
        file = epub.read_epub(path)
        toc = [file.get_item_with_href(x.href.split("#")[0]) for x in file.toc]
        idx, c = [], 0
        # Uhh?? See 小説29巻】本好きの下剋上～司書になるためには手段を選んでいられません～第五部「女神の化身VIII」.epub
        while len(toc) > c:
            for i in range(idx[-1]+1 if len(idx) else 0, len(file.spine)):
                v = file.spine[i]
                if v[0] == toc[c].id:
                    idx.append(i)
                    pc = c
                    c += 1
                    if c == len(toc): break
            idx.append(idx[-1])
            c+=1
        idx.append(len(file.spine))
        return [cls(epub=file, title=file.toc[i].title, start=idx[i], end=idx[i+1]) for i in range(len(toc))]

def match(audio, text):
    sta = {}
    for i in range(len(scripts)):
        script, best = scripts[i], (-1, -1, 0)
        for j in range(len(streams)):
            if (r := fuzz.ratio(script, streams[j][0])) > best[-1]:
                best = (j, -1, r)
            for k in range(len(streams[j][1])):
                if (r := fuzz.ratio(script, streams[j][1][k].cn)) > best[-1]:
                    best = (j, k, r)
        if best == (-1, -1, 0):
            print("Couldn't find a script match based on filename")
            # TODO(ym): Match based on content? based on the remaining indicies?
            # If I matched based on content then using anything to help decoding doesn't sound viable?
        sta[i] = best
    ats = {(v[0], v[1]): k for k, v in sta.items()}
    return sta, ats

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

    if args.pop("fast_decoder"):
        model = modify_model(model)
    overlap = args.pop("fast_decoder_overlap")
    batches = args.pop("fast_decoder_batches")

    streams = [(os.path.basename(f), AudioStream.from_file(f)) for f in args.pop('audio')]
    scripts = args.pop('text')
    chapters = [z  for i in scripts for z in Epub.from_file(i)]
    for i, v in enumerate(chapters):
        pprint(v.text(ignore={'rt'}))
        # if i == 3: exit(0)
    sta, ats = match(streams, scripts)

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

    for i in streams:
        if hasattr(model, 'huggingface'):
            i[1][0].transcribe(model, cache, batches=batches, overlap=overlap, temperature=temperature, **args)
        else:
            i[1][0].transcribe(model, cache, temperature=temperature, **args)
