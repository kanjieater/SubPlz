import whisper
import os
import numpy as np
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

    def transcribe(self, model, cache, language):
        if r := cache.get(os.path.basename(self.path), self.cid): return r
        r = model.transcribe(self.audio(), language=language)
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

def transcribe(self, data, **kwargs):
    language = kwargs['language']
    tokenizer = get_tokenizer(self.is_multilingual, language=kwargs['language'] if 'language' in kwargs else 'en')
    batches = 2
    kv_cache, hooks = self.install_kv_cache_hooks()
    segments = []
    overlap = 7
    left = 30 - overlap
    for i in range(0, data.shape[0], left * 16000 * batches):
        x = data[i:i+left * 16000 * batches + overlap * 16000]
        print(x.shape)
        mel = audio.log_mel_spectrogram(x)
        print(mel.shape)
        mels = []
        for k in range(batches):
            print(k * left*100, k * left*100 + 3000)
            chunk = mel[:, k * left*100: k * left*100 + 3000]
            if chunk.shape[-1] == 0: break
            if chunk.shape[-1] < 3000: chunk = audio.pad_or_trim(chunk, audio.N_FRAMES)
            mels.append(chunk.unsqueeze(0))
        mels = torch.concat(mels, dim=0)
        print(mels.shape)
        initial = [*tokenizer.sot_sequence, tokenizer.no_timestamps]
        tokens = torch.tensor(initial).repeat(mels.shape[0], 1)
        next_tokens = tokens
        audio_features = self.encoder(mels)
        x = tokenizer.encode('(')[0]
        y = tokenizer.encode(')')[0]
        while not (tokens[:, -1] == tokenizer.eot).all() and tokens.shape[-1] < 700:
            for t in tokens.tolist():
                print(tokenizer.decode(t))

            logits = self.decoder(next_tokens, audio_features, kv_cache=kv_cache)
            # logits[:, :, x] -= 3
            # logits[:, :, y] -= 3
            # logits[:, :, tokenizer.eot] -= 2
            # print(torch.topk(logits[:, -1, :], 3, -1))

            # print(logits[:, -1, tokenizer.eot])
            next_tokens = Categorical(logits=logits/0.2).sample()[:, -1:]
            # next_tokens = next_tokens[:, -1:]
            # logits[:, :, tokenizer.eot] = logits[:, :, tokenizer.eot]/1.2
            # next_tokens = logits.argmax(-1)[:, -1:]
            next_tokens[tokens[:, -1] == tokenizer.eot] = tokenizer.eot
            tokens = torch.concat([tokens, next_tokens], dim=-1)


        offset = i / 16000
        print(offset)
        for n, t in enumerate(tokens.tolist()):
            eot = t.index(tokenizer.eot) if tokenizer.eot in t else -1
            t = t[len(initial):]
            text = tokenizer.decode(t)
            segments.append(Segment(text=text, start=offset + n*left, end=offset + n*left + 30))

        for t in tokens.tolist():
            print(tokenizer.decode_with_timestamps(t))
        kv_cache.clear()


    for h in hooks:
        h.remove()

    file = open("/tmp/out.vtt", "w")
    file.write("WEBVTT\n\n")
    for s in segments:
        file.write(s.vtt() + "\n\n")
    file.flush()
    file.close()

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
    parser.add_argument("--dynamic-quantizaiton", "-dq", default=True, action=argparse.BooleanOptionalAction)
    # TODO
    # parser.add_argument("--output-file", default=None, help="name of the output subtitle file")
    # parser.add_argument("--split-script", default="", help=r"the regex to split the script with. for monogatari it is something like ^\s[\uFF10-\uFF19]*\s$")
    args = parser.parse_args()

    print(args.threads)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    # if args.output_file is None:
    #     args.output_file = os.path.splitext(args.audio_files[0])[0] + ".vtt"
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.progress)

    setattr(whisper.model.Whisper, 'transcribe', transcribe)
    model = whisper.load_model(args.model)
    if args.device == "cpu" and args.dynamic_quantizaiton:
        # ptdq_linear(model)
        pass

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
        i[1][0].transcribe(model, cache, args.language)
