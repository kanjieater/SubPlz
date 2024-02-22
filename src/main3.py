import whisper
from whisper.decoding import DecodingOptions, DecodingResult
from decoding import DecodingTask

import matplotlib.pyplot as plt

import os
import numba
import numpy as np
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

@dataclass
class Cache:
    model_name: str
    cache_dir: str
    enabled: bool
    ask: bool
    overwrite: bool

    def get(self, filename, chid):
        if not self.enabled: return
        fn = (filename + '.' + str(chid) +  '.' + self.model_name + ".subs")
        if (q := Path(self.cache_dir) / fn).exists():
            return eval(q.read_bytes().decode("utf-8"))

    def put(self, filename, chid, content):
        if not self.enabled: return content
        q = Path(self.cache_dir) / (filename + '.' + str(chid) +  '.' + self.model_name + ".subs")
        if q.exists() and self.ask:
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


def lcs(f, s):
    l = [0] * len(s)
    fidx, sidx = (0,0), (0,0)
    for i in range(len(f)):
        for j in reversed(range(len(s))):
            if f[i] == s[j]:
                if i == 0 or j == 0:
                    l[j] = 1
                else:
                    l[j] = l[j - 1] + 1
                if l[j] > fidx[1] - fidx[0]: fidx, sidx = (i-l[j]+1, i+1), (j - l[j]+1, j+1)
                # elif l[j] == (ret[0][1] - ret[0][0]): ret.append(((i-z+1, i+1)) # don't need more than one answer
            else:
                l[j] = 0
    return fidx, sidx


@torch.no_grad()
def decode(model, mel, options, **kwargs):
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result, logits = DecodingTask(model, options).run(mel)
    return result[0] if single else result, logits


def similarity(l1, l2):
    sm = torch.zeros([*l1.shape[:-2], l1.shape[-2], l2.shape[-2]])
    for i in range(l1.shape[-2]): # sm = (l1 * l2).sum(-1) # The dream
        m = l1[:, [i]] * l2
        sm[..., i, :] = -2 * (1 - m.sqrt().sum(-1)).sqrt() + 1
    return sm
    # Some tests with cross entropy
    # k = sm.log()
    # return k
    # absmins = abs(k.reshape(k.shape[0], -1).min(1).values)
    # maxes = k.reshape(k.shape[0], -1).max(1).values
    # return (k / torch.where(absmins > maxes, absmins, maxes).reshape(-1, 1, 1).expand(*k.shape))
    # return (sm / sm.max() - 0.5) * 2


@numba.jit(nopython=True)
def traceback_old(cost, mi, mj):
    result = []
    d = [(-1, -1), (-1, 0), (0, -1)]
    while mi > 0 and mj > 0:
        result.append((mi-1, mj-1))
        c0 = cost[mi - 1, mj - 1]
        c1 = cost[mi - 1, mj]
        c2 = cost[mi, mj - 1]
        n = np.array([c0, c1, c2])
        h = n.argmax()
        if n[h] == 0:
            break
        di, dj = d[h]
        mi, mj = mi + di, mj + dj
    result = np.array(result)
    return result[::-1, :].T

def traceback(c, mi, mj, fl, fs, sl, ss, tokenizer):
    ot = []
    t1, t2 = [], []
    def score(x):
        # return len(x)
        return sum(x)/((5 + len(x))/6)
        # return -np.inf if len(x) == 0 else sum(x)/len(x)
    while mi > 0 and mj > 0:
        f = c[[mi-1, mi, mi-1], [mj-1, mj-1, mj]]
        m = f.argmax()
        if f[m] == 0: break
        if m == 0:
            if len(t1) and len(t2):
                print(t1, tokenizer.decode_with_timestamps(t1))
                print(t2, tokenizer.decode_with_timestamps(t2))
                s1, s2 = [fl[mi+k][t] for k, t in enumerate(reversed(t1))], [sl[mj+k][t] for k, t in enumerate(reversed(t2))]
                ot.extend(t1 if score(s1) > score(s2) else t2)
                t1, t2 = [], []
            t1.append(fs[mi-1])
            t2.append(ss[mj-1])
            mi, mj = mi-1, mj-1
        elif m == 1:
            t2.append(ss[mj-1])
            mj = mj-1
        else:
            t1.append(fs[mi-1])
            mi = mi-1
    if len(t1) and len(t2):
        s1, s2 = [fl[mi+k][t] for k, t in enumerate(reversed(t1))], [sl[mj+k][t] for k, t in enumerate(reversed(t2))]
        ot.extend(t1 if score(s1) > score(s2) else t2)
    return ot[::-1], mi, mj

@numba.jit(nopython=True, parallel=True)
def align(sm: np.ndarray, gap=-1):
    N, M = sm.shape[-2:]
    cost = np.zeros((N+1, M+1), dtype=np.float32)
    m, mi, mj = 0, 0, 0
    for i in range(1, N+1):
        for j in range(1, M+1):
            c0 = cost[i - 1, j - 1] + sm[i-1, j-1]
            c1 = cost[i - 1, j] + gap
            c2 = cost[i, j - 1] + gap
            c = max(c1, c2, c0.item(), 0.0)
            cost[i, j] = c
            if c > m:
                m, mi, mj = c, i, j
    return cost, mi, mj
    # t = traceback(cost, mi, mj)
    # return t, m/t.shape[1]

import gc
def transcribe(model, data, **kwargs):
    data = torch.tensor(data).to(model.device)
    tokenizer = get_tokenizer(model.is_multilingual)
    batches = 1 # This is the max on my pc with small, TODO investigate why? the small model is 4x bigger yeah but this is too much
    overlap = 10
    left = 30 - overlap
    last = torch.zeros((1, 0, model.dims.n_vocab))
    last_tokens = DecodingResult(audio_features=None, language=kwargs['language'])
    for i in range(0, data.shape[0], left * 16000 * batches):
        x = data[i:i+left * 16000 * batches + overlap * 16000]
        mel = audio.log_mel_spectrogram(x)
        mels = []
        for k in range(batches):
            chunk = mel[:, k * left*100: k * left*100 + 3000]
            if chunk.shape[-1] == 0: break
            if chunk.shape[-1] < 3000: chunk = audio.pad_or_trim(chunk, audio.N_FRAMES)
            mels.append(chunk.unsqueeze(0))
        mels = torch.concat(mels, dim=0)
        mels = mels.half() if kwargs['fp16'] else mels
        audio_features = model.encoder(mels)
        result, logits = model.decode(audio_features, DecodingOptions(fp16=kwargs['fp16'], language=kwargs.get('language', None), length_penalty=None, beam_size=None)) # TODO: options
        del audio_features
        gc.collect()
        for i in result:
            print(tokenizer.decode_with_timestamps(i.tokens))

        print("Started aligning")
        ls = logits.shape[1]
        for i in range(logits.shape[0]):
            if i == 0:
                fl, fs = last, np.array(last_tokens.tokens)
            else:
                fl, fs = logits[i-1], np.array(result[i-1].tokens)
            sl, ss = logits[i].clone(), np.array(result[i].tokens)
            fl, sl = fl[3: 3+len(fs)].log_softmax(-1), sl[3: 3+len(ss)].log_softmax(-1)
            if len(ss) > sl.shape[0]: # What? Feels like a bug
                ss = ss[:sl.shape[0]-len(ss)]
            x = sl[ss >= tokenizer.timestamp_begin,  tokenizer.timestamp_begin:int(tokenizer.timestamp_begin + overlap // 0.02+1)]
            sl[ss >= tokenizer.timestamp_begin, int(tokenizer.timestamp_begin + left // 0.02+1): int(tokenizer.timestamp_begin + 30//0.02+1)] = x
            sl[ss >= tokenizer.timestamp_begin, tokenizer.timestamp_begin:int(tokenizer.timestamp_begin + overlap // 0.02+1)]  = -np.inf#sl[sl.ge(tokenizer.timestamp_begin): tokenizer.timestamp_begin + left / 0.02: tokenizer.timestamp_begin + 30/0.02] =
            sm = similarity(fl.unsqueeze(0).exp(), sl.unsqueeze(0).exp())[0].numpy()

            c, mi, mj = align(sm)
            shared, ni, nj = traceback(c, mi, mj, fl, fs, sl, ss, tokenizer)

            print(tokenizer.decode_with_timestamps(shared))
        last = logits[-1]
        last_tokens = result[-1]
        print("End alignment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument( "--audio-files", nargs="+", required=True, help="list of audio files to process (in the correct order)")
    parser.add_argument("--script", nargs="+", required=True, help="path to the script file")
    parser.add_argument("--model", default="tiny", help="whisper model to use. can be one of tiny, small, large, huge")
    parser.add_argument("--language", default="ja", help="language of the script and audio")
    parser.add_argument("--progress", default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-cache", default=True, help="whether to use the cache or not", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache-dir", default="AudiobookTextSyncCache", help="the cache directory")
    parser.add_argument("--overwrite-cache", default=False, action=argparse.BooleanOptionalAction, help="Always overwrite the cache")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count(), help=r"number of threads")
    parser.add_argument("--device", default="cpu", help="device to do inference on")
    parser.add_argument("--dynamic-quantizaiton", "--dq", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--fp16", default=False, help="whether to perform inference in fp16", action=argparse.BooleanOptionalAction)
    # TODO
    # parser.add_argument("--output-file", default=None, help="name of the output subtitle file")
    # parser.add_argument("--split-script", default="", help=r"the regex to split the script with. for monogatari it is something like ^\s[\uFF10-\uFF19]*\s$")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    # if args.output_file is None:
    #     args.output_file = os.path.splitext(args.audio_files[0])[0] + ".vtt"
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.progress)

    setattr(whisper.model.Whisper, 'decode', decode)
    setattr(whisper.model.Whisper, 'transcribe', transcribe)
    model = whisper.load_model(args.model).to(args.device)
    if args.device == "cpu" and args.dynamic_quantizaiton:
        ptdq_linear(model)

    cache = Cache(model_name=args.model, enabled=args.use_cache, cache_dir=args.cache_dir, ask=not args.overwrite_cache, overwrite=args.overwrite_cache)
    streams = [(os.path.basename(f), AudioStream.from_file(f)) for f in args.audio_files]
    scripts = args.script
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

    for i in streams:
        i[1][0].transcribe(model, cache, language=args.language, fp16=args.fp16)
