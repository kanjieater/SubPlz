from pprint import pprint
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


CACHEDIR = "/tmp/AudiobookTextSyncCache/"

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
        return f"Segment(text={self.text}, start={secs_to_hhmmss(self.start)}, end={secs_to_hhmmss(self.end)})"
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
            score = fuzz.ratio(subtitle, line) / 100.0 * min(len(line), len(subtitle))
            score += best(scs + script_used, sce, sus + used_subs, sue)
            if score > best_score:
                best_score, best_used_sub = score, used_subs
        return (best_score, best_used_sub)

    memo, best_script = {}, {}
    def best(scs, sce, sus, sue):  # Having to use subindex here is just sad
        if (sce - scs) <= 0:
            return 0

        key = (scs, sus)
        if key in memo:
            return memo[key][0]

        best_score, best_used_sub, best_used_script = 0, 1, 1
        for used_script in range(1, min(max_merge_count, sce - scs)+1):
            max_subs = max_merge_count if used_script == 1 else 1  # No Idea
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

    def get_best_sub_path(script_pos, n, last_script_pos, last_sub_to_test):
        _, key = best_script[script_pos]
        ret = []
        sub_pos = key[1]

        i = 0
        while i < n and script_pos < last_script_pos and sub_pos < last_sub_to_test:
            ret.append((script_pos, sub_pos))
            decision = memo[(script_pos, sub_pos)]
            num_used_sub = decision[1]
            num_used_script = decision[2]
            sub_pos += num_used_sub
            script_pos += num_used_script
            i += 1
        return ret

    results = []
    def find_match(scs, sce, sus, sue):
        # print(scs, sce, sus, sue)
        # print()
        if sce == scs or sue == sus: return

        memo.clear()
        best_script.clear()
        mid = (sce + scs) // 2
        s, e = max(scs, mid - max_search_context), min(sce, mid + max_search_context)
        # print("s", s, "e", e)

        for i in reversed(range(s, e)):
            for j in reversed(range(sus, sue)):
                best(i, e, j, sue)

        best_path = get_best_sub_path(s, e - s, e, sue)
        if len(best_path) > 0:
            for p in best_path:
                if p[0] > mid:
                    break
                mid_key = p
        else:
            return
        m = memo[mid_key]
        scriptp, subp = mid_key
        _, sub_used, script_used = m

        # key = best_script[s][1]
        # _, sub_used, script_used = memo[key]
        # scriptp, subp = s, key[1]
        # while (scriptp + script_used) < mid and (subp + sub_used) < sue:
        #     # print(scriptp, subp)
        #     scriptp += script_used
        #     subp += sub_used
        #     _, sub_used, script_used = memo[(scriptp, subp)]
        # print(scriptp, subp, sub_used, script_used)

        subs_result = subs[subp : subp + sub_used]
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
    bar.close()
    return results


def segment(file, language):
    import pysbd
    seg = pysbd.Segmenter(language=language, clean=False)

    sentences = [" "]
    with tqdm(file.strip().split("\n")) as bar:
        bar.set_description("Segmenting the script to sentences")
        for line in bar:
            for i in seg.segment(line):
                i = i.strip()  # Ugly
                while len(i) and i[1] in ["」", " ", "　", "’"]:  # Fix end of quotes
                    sentences[-1] += i[0]
                    i = i[1:]
                if sentences[-1][-1] in ["」", "’"] and len(sentences[-1] + i) < 80:  # Merge short lines with quotes
                    sentences[-1] += i
                elif len(i):
                    sentences.append(i)
    return sentences[1:]


def process_audio(model, model_name, language, files):
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
            language=language,
            suppress_silence=True,
            # vad=True,
            regroup=True,
            word_timestamps=True
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
    offset = 0
    with tqdm(files) as bar:
        bar.set_description("Processing audio")
        for file in bar:
            info = edit_tags(ffmpeg.probe(file, show_chapters=None))
            for ch in (bar2 := tqdm(info["chapters"])):
                s, e = float(ch["start_time"]), float(ch["end_time"])
                q = Path(CACHEDIR) / (os.path.basename(info["format"]["filename"]) + '.' + str(ch["id"]) +  '.' + model_name + ".subs")
                if q.exists():
                    seg_ts = eval(q.read_bytes().decode('utf-8'))
                else:
                    bar.refresh()  # first bar gets broken if i don't do this
                    bar2.set_description(f"{ch['id']} {ch['tags']['title']}" + (" is more than an hour, this may crash" if e - s > 60 * 60 * 1 else ""))
                    seg_ts = process(load(file, s, e)).to_dict()
                    pprint(seg_ts)
                    q.write_bytes(repr(seg_ts).encode('utf-8'))

                # print(seg_ts)
                # segs = text_output.to_word_level_segments()
                # print()
                # print(segs)
                # print()
                # exit()
                segs = [Segment(text=''.join(seg['text']), start=seg['start']+offset, end=seg['end']+offset) for seg in seg_ts['segments']]
                # print('WEBVTT\n\n')
                # print('\n\n'.join(i.vtt() for i in segs))
                # segs = [Segment(text=text, start=seg['start']+offset, end=seg['end']+offset) for text, seg in chain.from_iterable(zip(*i) for i in seg_ts)]
                transcript.extend(segs)
                offset += e - s
    pprint(transcript)
    out = list(sorted(sorted(transcript, key=lambda x: x.start), key=lambda x: x.end))
    print()
    print(out)
    print()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    parser.add_argument("--language", type=str, default="ja")
    parser.add_argument("--model", type=str, default="tiny")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument(
        "--audio-files",
        type=str,
        nargs="+",
        required=True,
        help="List of audio files to process (in the correct order)",
    )
    parser.add_argument(
        "--script", required=True, type=str, help="Path to the script file"
    )
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.script + ".vtt"

    with open(args.script, "r") as f:
        sentences = segment(f.read(), args.language)

    model = stable_whisper.load_model(args.model)
    transcripts = process_audio(model, args.model, args.language, args.audio_files)
    results = find_matches(sentences, transcripts)
    with open(args.output_file, "w") as out:
        out.write("WEBVTT\n\n")
        subs = "\n\n".join(Segment(text=t, start=s.start, end=s.end).vtt() for t, s in results)
        out.write(subs)
