from pprint import pprint
import os
# import intel_extension_for_pytorch as ipex
# import faster_whisper
from fuzzywuzzy import fuzz
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
import ffmpeg
import argparse
import whisper
import stable_whisper
from stable_whisper import text_output
import multiprocessing
import os
import numpy as np
from tqdm import tqdm
from functools import partialmethod


# CACHEDIR = "/tmp/AudiobookTextSyncCache/"
CACHEDIR = "/home/ym/fun/AudiobookTextSync/AudiobookTextSyncCache/"

def secs_to_hhmmss(secs):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    return f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>6.3f}'

@dataclass(eq=True, frozen=True)
class Segment:
    text: str
    start: float
    end: float
    def __repr__(self):
        return f"Segment(text='{self.text}', start={secs_to_hhmmss(self.start)}, end={secs_to_hhmmss(self.end)})"
    def vtt(self):
        return f"{secs_to_hhmmss(self.start)} --> {secs_to_hhmmss(self.end)}\n{self.text}"


def find_matches(script, subs, max_merge_count=6):
    max_search_context = max_merge_count * 2  # Should probably just be its own thing

    bar = tqdm(total=len(script))
    bar.set_description("Matching subs and the script")

    def score(script_used, scs, sce, max_subs, sus, sue):
        best_score, best_used_sub = 0, 1
        line = "".join(script[scs:scs+script_used])
        for used_subs in range(1, min(sue - sus, max_subs)+1):
            subtitle = "".join(i.text for i in subs[sus:sus+used_subs])
            score = fuzz.ratio(subtitle, line.replace('「', '').replace('」', '')) / 100.0 * min(len(line), len(subtitle))
            score += best(scs + script_used, sce, sus + used_subs, sue)
            if score > best_score:
                best_score, best_used_sub = score, used_subs
        return (best_score, best_used_sub)

    memo, best_script = {}, {}
    def best(scs, sce, sus, sue):
        if sce == scs: return 0

        key = (scs, sus)
        if key in memo:
            return memo[key][0]

        best_score, best_used_sub, best_used_script = 0, 1, 1
        for used_script in range(1, min(max_merge_count, sce - scs)+1):
            max_subs = max_merge_count if used_script == 1 else 1  # No Idea
            # max_subs = max_merge_count
            current_score, used_sub = score(used_script, scs, sce, max_subs, sus, sue)
            if current_score > best_score:
                best_score, best_used_sub, best_used_script = current_score, used_sub, used_script

        if best_used_script > 1:  # Do one more fitting
            current_score, used_sub = score(best_used_script, scs, sce, max_merge_count, sus, sue)
            if current_score > best_score: best_score, best_used_sub = current_score, used_sub

        memo[key] = (best_score, best_used_sub, best_used_script)
        prev_score, prev_key = best_script.get(scs, (0, None))
        if best_score >= prev_score:
            best_script[scs] = (best_score, key)

        return best_score

    results = []
    used = []
    def find_match(scs, sce, sus, sue):
        if sce == scs or sue == sus: return

        memo.clear()
        best_script.clear()
        mid = (sce + scs) // 2
        s, e = max(scs, mid - max_search_context), min(sce, mid + max_search_context)

        for i in reversed(range(s, e)):
            for j in reversed(range(sus, sue)):
                best(i, e, j, sue)

        key = best_script[s][1]
        _, sub_used, script_used = memo[key]
        scriptp, subp = s, key[1]
        while (scriptp + script_used) < mid and (subp + sub_used) < sue:
            scriptp += script_used
            subp += sub_used
            _, sub_used, script_used = memo[(scriptp, subp)]

        subs_result = subs[subp : subp + sub_used]
        used.extend(script[scriptp : scriptp + script_used])
        out = (
            "".join(script[scriptp : scriptp + script_used]),
            Segment(
                text="".join(i.text for i in subs_result),
                start=subs_result[0].start,
                end=subs_result[-1].end,
            ),
        )

        bar.update(script_used)
        find_match(scs, scriptp, sus, subp)
        results.append(out)
        find_match(scriptp+script_used, sce, subp+sub_used, sue)

    find_match(0, len(script), 0, len(subs))
    pprint(list(set(script) - set(used)))
    bar.close()
    return results


def segment(file, language, progress=True, spaces=True, whatever=False, whitespace=False):
    import pysbd
    seg = pysbd.Segmenter(language=language, clean=False)

    sentences = [" "]
    file = file.strip() if whitespace else file
    with tqdm(file.split("\n"), disable=not progress) as bar:
        bar.set_description("Segmenting the script to sentences")
        for line in bar:
            for i in seg.segment(line):
                i = i.strip() if whitespace else i# Ugly
                l = ["」", " ", "　", "’"] if whitespace else ["」", "’"]
                while len(i) and i[0] in l:  # Fix end of quotes
                    sentences[-1] += i[0]
                    i = i[1:]
                # if sentences[-1][0] in ["」", "’"] and len(sentences[-1] + i) < 50:  # Merge short lines with quotes
                #     sentences[-1] += i
                # if len(i):
                if spaces:
                    f =  i.split()
                    if whatever:
                        sentences.extend([k for i in f for z in i.split("。") for k in z.split("、")])
                    else:
                        sentences.extend(f)
                else:
                    sentences.append(i)
    return sentences[1:]


def process_audio(model, model_name, language, files, use_cache=True):
    def load(file, start, end):
        out, _ = (
            ffmpeg.input(file, ss=start, to=end)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, af="highpass=f=200,lowpass=f=3000", ar="16k")
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def process(data):
        return model.transcribe(
            data,
            fp16=False,
            # demucs=True,
            language=language,
            # suppress_silence=True,
            vad=True,
            regroup=True,
            verbose=None, # IT"S NONE NOT FALSE
            word_timestamps=True,
        )

    # this is neat, but the code is awful, and it dosen't cover all the edge cases.
    # So I'll probably just remove it
    def edit_tags(info):
        info["format"]["tags"] = {k.lower(): v for k, v in info["format"]["tags"].items()}
        info["format"]["tags"].setdefault("title", file)
        if "chapters" not in info or len(info["chapters"]) == 0:
            info["chapters"] = [{}]
        for ch in info["chapters"]:
            ch.setdefault("tags", info["format"]["tags"])
            ch.setdefault("start_time", 0)
            ch.setdefault("end_time", info["format"]["duration"])
            ch["tags"].setdefault("title", info["format"]["filename"])
            ch.setdefault("id", 0)
            ch["tags"] = {k.lower(): v for k, v in ch["tags"].items()}
        return info

    os.makedirs(CACHEDIR, exist_ok=True)

    transcript = []
    h = [0]
    offset = 0
    with tqdm(files) as bar:
        bar.set_description("Processing audio")
        for file in bar:
            info = edit_tags(ffmpeg.probe(file, show_chapters=None))
            for ch in (bar2 := tqdm(info["chapters"])):
                s, e = float(ch["start_time"]), float(ch["end_time"])
                q = Path(CACHEDIR) / (os.path.basename(info["format"]["filename"]) + '.' + str(ch["id"]) +  '.' + model_name + ".subs")
                if q.exists() and use_cache:
                    seg_ts = eval(q.read_bytes().decode('utf-8'))
                else:
                    bar.refresh()  # first bar gets broken if i don't do this
                    bar2.set_description(f"{ch['id']} {ch['tags']['title']}" + (" is more than an hour, this may crash" if e - s > 60 * 60 * 1 else ""))
                    seg_ts = process(load(file, s, e)).to_dict()
                    # pprint(seg_ts)
                    q.write_bytes(repr(seg_ts).encode('utf-8'))

                words = [Segment(text=''.join(seg['text']), start=seg['start']+offset, end=seg['end']+offset) for seg in seg_ts['segments']]
                # words = []
                # for seg in seg_ts['segments']:
                #     z = segment(seg['text'], language, spaces=True, progress=False, whitespace=False)
                #     # pprint(z)
                #     seg['words'][0]['word'] = seg['words'][0]['word'].lstrip() # This sucks
                #     string, start, end = "", seg['words'][0]['start'], seg['words'][0]['end']
                #     for word in seg['words']:
                #         if not z[0].startswith(string + word['word']):
                #             # print("OUT")
                #             # print(repr(string), repr(word['word']))
                #             words.append(Segment(text=string, start=start+offset, end=end+offset))
                #             z = z[1:]
                #             string, start, end = word['word'].lstrip(), word['start'], word['end']
                #         else:
                #             string += word['word']
                #             end = word['end']
                #     words.append(Segment(text=string, start=start+offset, end=end+offset))

                # words = [Segment(text=word['word'], start=word['start']+offset, end=word['end']+offset) for seg in seg_ts['segments'] for word in seg['words']]
                # print('WEBVTT\n\n')
                # print('\n\n'.join(i.vtt() for i in segs))
                # segs = [Segment(text=text, start=seg['start']+offset, end=seg['end']+offset) for text, seg in chain.from_iterable(zip(*i) for i in seg_ts)]
                transcript.extend(words)
                offset += e - s
                h.append(offset)
    # print()
    # pprint(len(transcript))
    # transcript = [i for i in transcript if len(i.text) > 0]
    # with open("whatever.vtt", "w") as f:
    #     f.write('WEBVTT\n\n')
    #     f.write('\n\n'.join(i.vtt() for i in transcript))
    # pprint(transcript)
    # pprint(len(transcript))
    return (transcript, h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument(
        "--audio-files",
        type=str,
        nargs="+",
        required=True,
        help="list of audio files to process (in the correct order)",
    )
    parser.add_argument("--script", required=True, type=str, help="path to the script file")
    parser.add_argument("--language", help="language of the script and audio", type=str, default="ja")
    parser.add_argument("--model", help="whisper model to use. can be one of tiny, small, large, huge", type=str, default="tiny")
    parser.add_argument("--output-file", help="name of the output subtitle file", type=str, default=None)
    parser.add_argument("--progress",default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-cache", default=True, help="whether to use the cache or not", action=argparse.BooleanOptionalAction)
    parser.add_argument("--split-script", help=r"the regex to split the script with. for monogatari it is something like ^\s[\uFF10-\uFF19]*\s$", type=str, default="")
    args = parser.parse_args()

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.progress)

    if args.output_file is None:
        args.output_file = args.script + ".vtt"

    with open(args.script, "r") as f:
        sentences = segment(f.read(), args.language, whatever=False, spaces=True)

    model = stable_whisper.load_model(args.model)
    # model = faster_whisper.WhisperModel(args.model, device='cpu', compute_type="float32")
    # stable_whisper.modify_model(model)
    # model = ipex.optimize(model)
    transcripts, h = process_audio(model, args.model, args.language, args.audio_files, use_cache=args.use_cache)
    results = find_matches(sentences, transcripts)
    # file = open(os.path.splitext(args.audio_files[0])[0] + '.vtt', "w")
    # file.write("WEBVTT\n\n")
    # for i in results:
    #     if i[1].start > h[1]:
    #         file.flush()
    #         file.close()
    #         h = h[1:]
    #         args.audio_files = args.audio_files[1:]
    #         file = open(os.path.splitext(args.audio_files[0])[0] + '.vtt', "w")
    #         file.write("WEBVTT\n\n")

    #     segment = Segment(text=i[0], start=i[1].start-h[0], end=i[1].end-h[0])
    #     file.write(segment.vtt() + "\n\n")

#     # f = open(
    with open(args.output_file, "w") as out:
        out.write("WEBVTT\n\n")
        # for i in results:
        subs = "\n\n".join(Segment(text=t, start=s.start, end=s.end).vtt() for t, s in results)
        out.write(subs)
