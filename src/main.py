import whisper
from whisper.decoding import DecodingOptions, DecodingResult
# from future import fstrings
# from decoding import DecodingTask

import unicodedata
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
from faster_whisper import WhisperModel

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
from types import MethodType
import torch.nn.functional as F

import bs4
from bs4 import BeautifulSoup
from Bio import Align
import regex as re
import math

def sexagesimal(secs):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    return f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>6.3f}'

@dataclass(eq=True)
class Segment:
    text: str
    # words: Segment
    start: float
    end: float
    def __repr__(self):
        return f"Segment(text='{self.text}', start={sexagesimal(self.start)}, end={sexagesimal(self.end)})"
    def vtt(self):
        return f"{sexagesimal(self.start)} --> {sexagesimal(self.end)}\n{self.text}"

@dataclass(eq=True)
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
        # if not self.enabled: return content
        cd = Path(self.cache_dir)
        cd.mkdir(parents=True, exist_ok=True)
        q = cd / (filename + '.' + str(chid) +  '.' + self.model_name + ".subs")
        if q.exists():
            if self.ask:
                prompt = f"Cache for file {filename}, chapter id {chid} already exists. Overwrite?  [y/n/Y/N] (yes, no, yes/no and don't ask again) "
                while (k := input(prompt).strip()) not in ['y', 'n', 'Y', 'N']: pass
                self.ask = not (k == 'N' or k == 'Y')
                self.overwrite = k == 'Y' or k == 'y'
            if not self.overwrite: return content
        q.write_bytes(repr(content).encode('utf-8'))
        return content

@dataclass
class AudioStream:
    stream: ffmpeg.Stream
    path: Path
    duration: float
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
            return title, [cls(stream=ffmpeg.input(path), duration=float(info['streams'][0]['duration']), path=path, cn=title, cid=0)]
        return title, [cls(stream=ffmpeg.input(path, ss=float(chapter['start_time']), to=float(chapter['end_time'])),
                           duration=float(chapter['end_time']) - float(chapter['start_time']),
                           path=path,
                           cn=chapter.get('tags', {}).get('title', ''),
                           cid=chapter['id'])
                       for chapter in info['chapters']]

@dataclass(eq=True, frozen=True)
class Paragraph:
    chapter: int
    element: bs4.element.Tag
    references: list

    def text(self):
        return ''.join(self.element.stripped_strings)


@dataclass(eq=True, frozen=True)
class TextParagraph:
    path: str
    idx: int
    content: str
    references: list

    def text(self):
        return self.content

@dataclass(eq=True, frozen=True)
class TextFile:
    path: str
    title: str
    def text(self, *args, **kwargs):
        return [TextParagraph(path=self.path, idx=i, content=o, references=[]) for i, v in enumerate(Path(self.path).read_text().split('\n'))if (o := v.strip()) != '']

@dataclass(eq=True, frozen=True)
class Epub:
    epub: epub.EpubBook
    title: str
    start: int
    end: int

    def text(self, prefix, follow_links=True, ignore=set()):
        o = []
        for i in range(self.start, self.end):
            id, is_linear = self.epub.spine[i]
            item = self.epub.get_item_with_id(id)
            # https://gitlab.com/smoores/storyteller/-/blob/main/storyteller/synchronize/epub.py?ref_type=heads#L259
            if is_linear and item.media_type == "application/xhtml+xml":
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                paragraphs = soup.find("body").find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"])

                for p in paragraphs:
                    references = []
                    for r in p.find_all(href=True):
                        if "#" not in r['href']: continue
                        path, tid = r['href'].split("#")

                        if os.path.basename(path) != os.path.basename(item.file_name):
                            chapter = self.epub.get_item_with_href(path if '/' in path else (os.path.dirname(item.file_name) + '/' + path))
                            idx = [i for i, _ in self.epub.spine].index(chapter.id)
                            # TODO: cache or whatever and get rid of this if
                            ref = BeautifulSoup(chapter.get_content(), 'html.parser').find(id=tid)
                        else:
                            idx = i
                            ref = soup.find(id=tid)
                        references.append(Paragraph(chapter=idx, element=ref, references=[]))
                    o.append(Paragraph(chapter=i, element=p, references=references))
        return o

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

            audio_full_title = clean(afn+at)
            text_full_title = clean((tfn + tc[0].epub.title) if type(tc[0]) == Epub else at*2)
            main = fuzz.ratio(audio_full_title,text_full_title)

            for i in range(len(ac)):
                best = ats.get((ai, i), (-1, -1, 0))
                for j in range(len(tc)):
                    ach = audio_full_title + clean(ac[i].cn)
                    tch = text_full_title + clean(tc[j].title)
                    score = fuzz.ratio(ach, tch)
                    # print(ach, '-', tch, '-', score, main)
                    if score > best[-1] and score > main:
                        best = (ti, j, score)
                if best != (-1, -1, 0): ats[(ai, i)] = best

    for k, v in ats.items():
        ai, i = k
        ti, tj, s = v
        if ti != -1:
            print(audio[ai][2][i].cn, text[ti][1][tj].title, s)
    return {k: v[:-1] for k,v in ats.items()}#, {v[:-1]: k for k,v in ats.items()}

ascii_to_wide = dict((i, chr(i + 0xfee0)) for i in range(0x21, 0x7f))
# ascii_to_wide.update({0x20: '\u3000', 0x2D: '\u2212'})  # space and minus
kata_hira = dict((0x30a1 + i, chr(0x3041 + i)) for i in range(0x56))
kansuu_to_ascii = dict([(ord('一'), '１'), (ord('二'), '２'), (ord('三'), '３'), (ord('四'), '４'), (ord('五'), '５'), (ord('六'), '６'), (ord('七'), '７'), (ord('八'), '８'), (ord('九'), '９'), (ord('◯'), '０') ,(ord('零'), '０'), (ord('十'), '１')])
allt = kata_hira | kansuu_to_ascii | ascii_to_wide
# allt = kansuu_to_ascii | ascii_to_wide
def clean(s, normalize=True):
    r = r"[\p{C}\p{M}\p{P}\p{S}\p{Z}\sーぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ]+"
    s = re.sub(r, "", s).translate(allt)
    return unicodedata.normalize("NFKD", s) if normalize else s

def align2(coords, *arrs):
    pos, off = [0, 0], [0, 0]
    el, idx = [0, 0], 0

    segments = []
    while el[0] < len(arrs[0]) and el[1] < len(arrs[1]):
        segments.append([*el, *off])
        i = 0 if (len(arrs[0][el[0]])-off[0]) < (len(arrs[1][el[1]])-off[1]) else 1
        smaller = len(arrs[i][el[i]]) - off[i]
        gap = [0, 0]
        while (idx+1) < coords.shape[1] and pos[i] + smaller > coords[i][idx]:
            idx += 1
            if coords[i, idx] == coords[i, idx-1]: gap[i] += coords[1-i, idx] - coords[1-i, idx-1]
            if coords[1-i, idx] == coords[1-i, idx-1]: gap[1-i] += coords[i, idx] - coords[i, idx-1]

        off[i] = 0
        el[i] += 1
        pos[i] += smaller

        advance = smaller + gap[i] - gap[1-i]
        while el[1-i] < len(arrs[1-i]) and (advance - (len(arrs[1-i][el[1-i]]) - off[1-i])) > 0:
            advance -= len(arrs[1-i][el[1-i]]) - off[1-i]
            off[1-i] = 0
            el[1-i] += 1
            segments.append([*el, *off])
        off[1-i] += advance
        pos[1-i] += smaller + gap[i] - gap[1-i]

    return segments

def fix(original, edited, segments):
    s, e = 0, 0
    while e < len(segments):
        e += 1
        if e == len(segments) or segments[s][0] != segments[e][0]:
            orig, cl = original[segments[s][0]].translate(allt), edited[segments[s][0]]
            i, j, jj = 0, 0, s
            while i < len(orig) and j < len(cl) and jj < e:
                while jj < e and j >= segments[jj][2]:
                    segments[jj][2] = i
                    jj += 1
                if orig[i] == cl[j]:
                    j += 1
                i += 1
            while jj < e:
                segments[jj][2] = i
                jj += 1
            s = e

def fix_punc(t, segments, prepend, append):
    # This can probably go into an infinite loop
    for i in range(len(segments)):
        k = segments[i]
        text = t[k[0]]
        while k[2] > 0 and k[2] < len(text):
            if text[k[2]-1] in prepend:
                k[2] -= 1
            elif text[k[2]] in append:
                k[2] += 1
            # Do more testing on this, esp the first one
            # This is more like a hack because I don't want to access text
            elif k[2] == len(text) - 1:
                k[2] = len(text)
            elif k[2] == 1:
                k[2] = 0
            # elif k[2] - 2 >= 0 and text[k[2]-1] not in append and text[k[2]-2] not in "、," and text[k[2]-2] in append:
            elif k[2] - 2 >= 0 and text[k[2]-1] not in prepend and text[k[2]-1] not in append and text[k[2]-2] in append:
                k[2] -= 1
            elif k[2] + 1 < len(text) and text[k[2]] != "が" and text[k[2]] not in prepend and text[k[2]] not in append and text[k[2]+1] in append: # Lol
                k[2] += 1
            # Lol, two "levels" just for …
            elif k[2] - 3 >= 0 and text[k[2]-2] == '…' and text[k[2]-3] in append:
                k[2] -= 2
            elif k[2] + 2 < len(text) and text[k[2]+1] == '…' and text[k[2]+2] in append:
                k[2] += 2
            else:
                break

# Here be dragons
def find(segments, start, end):
    for i, v in enumerate(segments):
        if type(v[0]) != str and v[1] == start:
            break
    for j, v in enumerate(segments[i:]): # First end
        if v[1] == end:
            break
    for k, v in enumerate(segments[i+j:]): # Last end
        if v[1] != end:
            break
    return i, i+j+k

def replace(segments, replacement):
    start, end = replacement[0][1], replacement[-1][1]
    start, end = find(segments, start, end)
    k = start
    while k > 0 and type(segments[k][0]) != str and segments[k][0] == segments[start][0]:
        k -= 1
    k += 1
    new = segments[end][1]
    for i in segments[k:end]:
        i[1] = new
    segments[k:k] = replacement

def align(model, transcript, text, prepend, append):
    transcript_str = [i['text'] for i in transcript['segments']]
    transcript_str_clean = [clean(i, normalize=False) for i in transcript_str]
    transcript_str_joined = ''.join(transcript_str_clean)

    aligner = Align.PairwiseAligner(scoring=None, mode='local', match_score=1, open_gap_score=-1, mismatch_score=-1, extend_gap_score=-1)
    def inner(text):
        text_str = [i.text() for i in text]
        text_str_clean = [clean(i, normalize=False) for i in text_str]
        text_str_joined = ''.join(text_str_clean)

        if not len(text_str_joined) or not len(transcript_str_joined): return []

        coords = aligner.align(text_str_joined, transcript_str_joined)[0].coordinates
        coords = np.concatenate([np.zeros((2, 1)).astype(int), np.array([0, coords[1][0]]).reshape(2, 1),  coords], axis=1)  # Should probably fix align2 to find the start instead of hacking it like this

        segments = align2(coords, text_str_clean, transcript_str_clean)
        fix(text_str, text_str_clean, segments)
        fix_punc(text_str, segments, prepend, append)
        return segments

    references = {k.element.attrs['id']:k for i in text for k in i.references} # Filter out dups

    segments = inner(text)
    for k, v in references.items():
        ref_segments = inner([v])
        i = len(ref_segments)-1
        while i > 0 and ref_segments[i][2] > 0:
            ref_segments[i][0] = k
            i -= 1
        ref_segments[i][0] = k
        re = ref_segments[i:]
        print(k)
        pprint(re)
        replace(segments, re)

    return segments, references

def to_epub():
    pass

def to_subs(text, transcript, alignment, offset, references):
    s, e = 0, 0
    segments = []
    while e < len(alignment):
        e += 1
        if e == len(alignment) or alignment[s][1] != alignment[e][1]:
            r = ''
            for n, k in zip(alignment[s:e], alignment[s+1:e+1]):
                p, o = n[0], n[2]
                p2, o2 = k[0], k[2]
                if type(p) == str:
                    f = references[p].text()[o:]
                    if p == p2:
                        f = f[:o2-o]
                    if p != p2 and type(p2) == str:
                        f += references[p2].text()[:o2]
                    r += f
                elif type(p2) == str:
                    r += text[p].text()[o:]
                else:
                    r += text[p].text()[o:o2] if p == p2 else (text[p].text()[o:] + text[p2].text()[:o2]) #text[p].text()[o:]
            if e == len(alignment): r += text[alignment[-1][0]].text()[alignment[-1][2]:]
            t = transcript['segments'][alignment[s][1]]
            segments.append(Segment(text=t['text']+'\n'+r, start=t['start']+offset, end=t['end']+offset))
            s = e
    return segments


def faster_transcribe(self, audio, **args):
    args.pop('fp16')
    args['log_prob_threshold'] = args.pop('logprob_threshold')
    args['beam_size'] = args['beam_size'] if args['beam_size'] else 1
    args['patience'] = args['patience'] if args['patience'] else 1
    args['length_penalty'] = args['length_penalty'] if args['length_penalty'] else 1
    segments, info = self.transcribe2(audio, best_of=1, **args)
    x = {'segments': []}
    for segment in segments:
        x['segments'].append({'text': segment.text, 'start': segment.start, 'end': segment.end})
    return x

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

    parser.add_argument("--faster-whisper", default=True, help='Use faster_whisper, doesn\'t work with hugging face\'s decoding method currently', action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast-decoder", default=False, help="Use hugging face's decoding method, currently incomplete", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast-decoder-overlap", type=int, default=10,help="Overlap between each batch")
    parser.add_argument("--fast-decoder-batches", type=int, default=1, help="Number of batches to operate on")

    parser.add_argument("--ignore-tags", default=['rt'], nargs='+', help="Tags to ignore during the epub to text conversion, useful for removing furigana")
    parser.add_argument("--prefix-chapter-name", default=True, help="Whether to prefix the text of each chapter with its name", action=argparse.BooleanOptionalAction)
    parser.add_argument("--follow-links", default=True, help="Whether to follow hrefs or not in the ebook", action=argparse.BooleanOptionalAction)

    parser.add_argument("--fp16", default=False, help="whether to perform inference in fp16", action=argparse.BooleanOptionalAction)
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
    # sutegana = 'ぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ'
    sutegana = ''
    parser.add_argument("--append_punctuations", type=str, default="\"\'・.。,，!！?？:：”)]}、』」）〉》〕】｝］’〟／＼～〜~"+sutegana, help="if word_timestamps is True, merge these punctuation symbols with the previous word")
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
    faster_whisper = args.pop('faster_whisper')

    model = args.pop("model")
    device = args.pop('device')

    overwrite_cache = args.pop('overwrite_cache')
    cache = Cache(model_name=model, enabled=args.pop("use_cache"), cache_dir=args.pop("cache_dir"),
                  ask=not overwrite_cache, overwrite=overwrite_cache)
    model = WhisperModel(model, device, local_files_only=True, compute_type='int8' if device == 'cpu' else 'float16') if faster_whisper else whisper.load_model(model).to(device)

    if faster_whisper:
        model.transcribe2 = model.transcribe
        model.transcribe = MethodType(faster_transcribe, model)

    if args.pop('dynamic_quantization') and not faster_whisper and device == "cpu":
        ptdq_linear(model)

    overlap, batches = args.pop("fast_decoder_overlap"), args.pop("fast_decoder_batches")
    if args.pop("fast_decoder") and not faster_whisper:
        args["overlap"] = overlap
        args["batches"] = batches
        modify_model(model)

    streams = [(os.path.basename(f), *AudioStream.from_file(f)) for f in args.pop('audio')]
    chapters = [(os.path.basename(i), Epub.from_file(i)) if i.split(".")[-1] == 'epub' else (os.path.basename(i), [TextFile(path=i, title=os.path.basename(i))]) for i in args.pop('text')]
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
    follow_links = args.pop('follow_links')
    for i, v in enumerate(streams):
        offset = 0
        segments = []
        for j, v in enumerate(v[2]):
            if (i, j) in ats:
                print(i, j)
                ci, cj = ats[(i, j)]
                print(streams[i][2][j].cn)
                print(chapters[ci][1][cj].title)
                text = chapters[ci][1][cj].text(prefix_chapter_name, follow_links=follow_links, ignore=ignore_tags)
                transcript = v.transcribe(model, cache, temperature=temperature, **args)
                alignment, references = align(model, transcript, text, args['prepend_punctuations'], args['append_punctuations'])
                segments.extend(to_subs(text, transcript, alignment, offset, references))
                break
            offset += v.duration
        with open(f"/tmp/out.vtt", "w") as out:
            out.write("WEBVTT\n\n")
            out.write('\n\n'.join([s.vtt() for s in segments]))
        break
