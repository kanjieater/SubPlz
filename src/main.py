from __future__ import annotations
from pprint import pprint
import regex as re
import pyalign
import os
import itertools
from fuzzywuzzy import fuzz
from dataclasses import dataclass
from itertools import chain, count, takewhile
from pathlib import Path
import ffmpeg
import argparse
import whisper
# import stable_whisper
# from stable_whisper import text_output
import whisper
import multiprocessing
import os
import numpy as np
from tqdm import tqdm
from functools import partialmethod
import torch

from sentence_transformers import SentenceTransformer, util
transmodel = SentenceTransformer('all-mpnet-base-v2')

import sudachipy
# from sudachipy import dictionary

CACHEDIR = "/tmp/AudiobookTextSyncCache/"
# CACHEDIR = "/home/ym/fun/AudiobookTextSync/AudiobookTextSyncCache/"

def secs_to_hhmmss(secs):
    mm, ss = divmod(secs, 60)
    hh, mm = divmod(mm, 60)
    return f'{hh:0>2.0f}:{mm:0>2.0f}:{ss:0>6.3f}'


# TODO(YM):
def clean_string(s):
    return s.replace('「', '').replace('」', '').replace("《", "").replace("》", "")

@dataclass(eq=True, frozen=True)
class Segment:
    text: str
    # words: Segment
    start: float
    end: float
    def __repr__(self):
        return f"Segment(text='{self.text}', start={secs_to_hhmmss(self.start)}, end={secs_to_hhmmss(self.end)})"
    def vtt(self):
        return f"{secs_to_hhmmss(self.start)} --> {secs_to_hhmmss(self.end)}\n{self.text}"

#def align(scripts, subs, subi, subj, scripti, scriptj):
#    script = scripts[scriptj]
#    script_clean = re.sub("[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}|\s]+", "", script)
#    starts, ends = subj, subj+1
#    sub_clean = re.sub("[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}|\s]+", "", subs[subj].text)

#    # r = (alignment := pyalign.local_alignment(sub_clean, script_clean))
#                                       #T          S
#    alignment = pyalign.local_alignment(sub_clean, script_clean)
#    mapping = alignment.t_to_s
#    prev_mapping = np.array([])
#    while np.max(mapping) < len(script_clean) - 2 and np.min(mapping) >= 1:
#    # while  ((mapping == -1).sum() >= len(script_clean)//4 and (mapping == -1).sum() != (prev_mapping == -1).sum()) or (len(sub_clean) < len(script_clean)):
#        print(mapping.tolist())
#        print(np.max(mapping), len(script_clean))
#        print(alignment.s_to_t.tolist())
#        print("sub2  ", sub_clean)#''.join(["ー" if k == -1 else sub_clean[k] for k in alignment.t_to_s]))
#        print("script", ''.join(["ー" if k == -1 else script_clean[k] for k in alignment.s_to_t]))#script_clean)
#        if mapping[-1] == -1:
#            ends +=1
#        # if (mapping[:len(mapping)//2] == -1).sum() > (mapping[len(mapping)//2:] == -1).sum():
#        if mapping[0] == -1:
#            starts -= 1
#        if mapping[0] != -1 and mapping[-1] != -1:
#            starts -= 1
#            ends += 1
#        sub_clean = re.sub(r"[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}|\s]+", "", ''.join(i.text for i in subs[starts: ends]))
#        alignment = pyalign.local_alignment(sub_clean, script_clean)
#        prev_mapping = mapping
#        mapping = alignment.t_to_s

#    alignment = pyalign.global_alignment(sub_clean, script_clean)
#    print("-"*80)
#    print(alignment.s_to_t.tolist())
#    print("sub2  ", sub_clean)#''.join(["ー" if k == -1 else sub_clean[k] for k in alignment.t_to_s]))
#    print("script", ''.join(["ー" if k == -1 else script_clean[k] for k in alignment.s_to_t]))#script_clean)
#    # print("sub2", sub_clean)
#    # print("script2", script_clean)

    # print()
    # for i in count(0):
    #     if i > 1:
    #         break
    #     sub = re.sub("[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}]+", "", ''.join(j.text for j in subs[subj-i:subj+i+1]))
    #     if len(sub) == 0:
    #         continue
    #     alignment = pyalign.local_alignment(sub, script_clean)
    #     print(i)
    #     print(sub)
    #     # print(script_clean)
    #     print(alignment.s_to_t)
    #     print(alignment.t_to_s)
    #     n = ''.join(["ー" if k == -1 else script_clean[k] for k in alignment.s_to_t])
    #     # s = ''.join(["ー" if k == -1 else sub[k] for k in alignment.t_to_s])
    #     print(n)
    #     # print(s)
    #     print()
    # pass

# def find_matches(script, subs):
#     segs = []
#     # script = script[:100]
#     # escript = util.normalize_embeddings(transmodel.encode(script, convert_to_tensor=True))

#     # torch.save(escript, "/tmp/tensor.pt")
#     escript = torch.load("/tmp/tensor.pt")
#     # print("loaded escript")
#     # equery = util.normalize_embeddings(transmodel.encode([i.text for i in subs], convert_to_tensor=True))
#     equery = torch.load("/tmp/query.pt")
#     # torch.save(equery, "/tmp/query.pt")
#     print("loaded equery")
#     # subs = subs[:100]
#     n = 0
#     last = 0
#     lastl = 0
#     for l in  range(0, len(subs), 1):# itertools.batched(subs, 3):
#         if len(subs[l].text) < 5:
#             continue
#         # query = util.normalize_embeddings(transmodel.encode([''.join(i.text for i in subs[l:l+1])], convert_to_tensor=True))
#         r = util.semantic_search(equery[l], escript[last:last+50], score_function=util.dot_score)
#         # if r[0][0]['corpus_id'] < 100 and r[0][0]['score'] > 0.85:
#         if r[0][0]['score'] > 0.85 and len(script[last + r[0][0]['corpus_id']]) > 5:
#             print("subs: ", ''.join(i.text for i in subs[l:l+1]))
#             print("script", script[last + r[0][0]['corpus_id']], last + r[0][0]['corpus_id'], r[0][0]['score'])
#             align(script, subs, lastl, l, last, last + r[0][0]['corpus_id'])
#             last += r[0][0]['corpus_id'] + 1
#             lastl = l
#             n += 1
#             print()
#     print(n)
#     return segs



# def find_matches(script, subs, max_merge=3):
#     # script = script[1:]
#     segs = []
#     start, end = 0, 1
#     simp = 0
#     sstart, send = 0, 1
#     while end < len(script):
#         line = ''.join(script[start:end])
#         eline = transmodel.encode(line)
#         sub = ''.join(i.text for i in subs[sstart:send])
#         esubs = transmodel.encode(sub)
#         sim = util.cos_sim(esubs, eline)
#         if sim < 0.4:
#             sstart += 1
#             send += 1
#         print(f"Testing: {line} with {sub} || {sstart} {send} || {sim} {simp}")
#         if simp > sim:
#             out = (
#                 "".join(script[start:end]),
#                 Segment(
#                     text="".join(i.text for i in subs[sstart:send]),
#                     start=subs[sstart].start,
#                     end=subs[send-1].end,
#                 ),
#             )
#             segs.append(out)
#             start, end = end, end+1
#             sstart, send = send, send+1
#             simp = 0
#         else:
#             if send < len(subs):
#                 send += 1
#             else:
#                 out = (
#                     "".join(script[start:end]),
#                     Segment(
#                         text="".join(i.text for i in subs[sstart:send]),
#                         start=subs[sstart].start,
#                         end=subs[send].end,
#                     ),
#                 )
#                 segs.append(out)
#                 break
#             simp = sim
#         if end == len(script):
#             out = (
#                 "".join(script[start:end]),
#                 Segment(
#                     text="".join(i.text for i in subs[sstart:send]),
#                     start=subs[sstart].start,
#                     end=subs[send].end,
#                 ),
#             )
#             segs.append(out)
#     return segs


# def find_matches(script, subs, max_merge_count=6):
#     max_search_context = max_merge_count * 2  # Should probably just be its own thing

#     bar = tqdm(total=len(script))
#     bar.set_description(f"Matching subs and the script {len(script)} {len(subs)}")

#     def score(script_used, scs, sce, max_subs, sus, sue):
#         best_score, best_used_sub = 0, 1
#         line = "".join(script[scs:scs+script_used])
#         for used_subs in range(1, min(sue - sus, max_subs)+1):
#             subtitle = "".join(i.text for i in subs[sus:sus+used_subs])
#             # Sometimes removing the length multiplication leads to better results, we really need a better heuristic that considers more things tbh
#             # Also, we should have the concept of a "barrier", points that this should not go over
#             # For example, when the speaker changes, but i don't know how efficient speaker speaker diarization is for long audiobooks
#             score = fuzz.ratio(subtitle, clean_string(line)) / 100.00 * min(len(line), len(subtitle))
#             # eline, esubtitle = *map(transmodel.encode, [line, subtitle]),
#             # score = util.cos_sim(eline, esubtitle)
#             score += best(scs + script_used, sce, sus + used_subs, sue)
#             if score > best_score:
#                 best_score, best_used_sub = score, used_subs
#         return (best_score, best_used_sub)

#     memo, best_script = {}, {}
#     def best(scs, sce, sus, sue):
#         if sce == scs: return 0

#         key = (scs, sus)
#         if key in memo: return memo[key][0]

#         best_score, best_used_sub, best_used_script = 0, 1, 1
#         for used_script in range(1, min(max_merge_count, sce - scs)+1):
#             max_subs = max_merge_count if used_script == 1 else 1  # No Idea
#             # max_subs = max_merge_count
#             current_score, used_sub = score(used_script, scs, sce, max_subs, sus, sue)
#             if current_score > best_score:
#                 best_score, best_used_sub, best_used_script = current_score, used_sub, used_script

#         if best_used_script > 1:  # Do one more fitting
#             current_score, used_sub = score(best_used_script, scs, sce, max_merge_count, sus, sue)
#             if current_score > best_score: best_score, best_used_sub = current_score, used_sub

#         memo[key] = (best_score, best_used_sub, best_used_script)
#         prev_score, prev_key = best_script.get(scs, (0, None))
#         if best_score >= prev_score:
#             best_script[scs] = (best_score, key)

#         return best_score

#     results = []
#     used = []
#     def find_match(scs, sce, sus, sue):
#         if sce == scs or sue == sus: return

#         memo.clear()
#         best_script.clear()
#         mid = (sce + scs) // 2
#         s, e = max(scs, mid - max_search_context), min(sce, mid + max_search_context)

#         for i in reversed(range(s, e)):
#             for j in reversed(range(sus, sue)):
#                 best(i, e, j, sue)

#         key = best_script[s][1]
#         _, sub_used, script_used = memo[key]
#         scriptp, subp = s, key[1]
#         while (scriptp + script_used) < mid and (subp + sub_used) < sue:
#             scriptp += script_used
#             subp += sub_used
#             _, sub_used, script_used = memo[(scriptp, subp)]

#         subs_result = subs[subp : subp + sub_used]
#         used.extend(script[scriptp : scriptp + script_used])
#         out = (
#             "".join(script[scriptp : scriptp + script_used]),
#             Segment(
#                 text="".join(i.text for i in subs_result),
#                 start=subs_result[0].start,
#                 end=subs_result[-1].end,
#             ),
#         )

#         bar.update(script_used)
#         find_match(scs, scriptp, sus, subp)
#         results.append(out)
#         find_match(scriptp+script_used, sce, subp+sub_used, sue)

#     find_match(0, len(script), 0, len(subs))
#     pprint(list(set(script) - set(used)))
#     bar.close()
#     return results


ascii_to_wide = dict((i, chr(i + 0xfee0)) for i in range(0x21, 0x7f))
ascii_to_wide.update({0x20: '\u3000', 0x2D: '\u2212'})  # space and minus
kata_hira = dict((0x30a1 + i, chr(0x3041 + i)) for i in range(0x56))
kansuu_to_ascii = dict([(ord('一'), '１'), (ord('二'), '２'), (ord('三'), '３'), (ord('四'), '４'), (ord('五'), '５'), (ord('六'), '６'), (ord('七'), '７'), (ord('八'), '８'), (ord('九'), '９'), (ord('零'), '０')])
allt = kata_hira | kansuu_to_ascii | ascii_to_wide


# from suffix_trees import STree
# def maximum_matching_substring(a, b):
#     # Concatenate the strings with a unique separator character
#     concatenated = [a, b]

#     # Build the suffix tree using Ukkonen's algorithm
#     stree = STree.STree(concatenated)

#     # Find the longest common substring using the lcs method
#     lcs_result = stree.lcs()

#     # Extract the maximum matching substring
#     matching_substring = lcs_result[-1]
#     return matching_substring

def clean(x):
    clean = [re.sub("[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}|\s|ー]+", "", i) for i in x]
    # origin = [i for i in range(len(clean)) for _ in range(len(clean[i]))]
    text = ''.join(clean).translate(allt)
    # return origin, text
    return None, text

from Bio import Align
def find_matches(script, subs):
    subs_origin, subs_whole = clean([i.text for i in subs])
    script_origin, script_whole = clean(script)
    # print(maximum_matching_substring(subs_whole, script_whole))

    aligner = Align.PairwiseAligner(scoring=None, mode='global', open_gap_score=-1, mismatch_score=-1, extend_gap_score=-1)
    alignment = aligner.align(script_whole, subs_whole)[0]
    print(alignment.score)
    print(len(alignment.coordinates[0]))
    for i

    # s, e = 60000, 60150
    # print(alignments[0][0, s:e].__str__().replace("-", "ー"))
    # print(alignments[0][1, s:e].__str__().replace("-", "ー"))

#     import gc
#     gc.collect()
#     import time
#     import edlib
#     s = time.time()
#     edlib.align(subs_whole[:200], script_whole[:200])
#     print(time.time() - s)

#     limit = 200
#     script_whole = script_whole2
#     i = 0
#     while i < len(subs_whole):
#         print(script_whole[:1000])
#     # for i in range(0, len(subs_whole), limit):
#         subs_text = subs_whole[i:i+limit]
#         print(subs_text)
#         print(i//limit, len(subs_text), len(script_whole))
#         # if not len(subs_text): break

#         alignment = pyalign.local_alignment(script_whole, subs_text)
#         s, e = 0, 150
#         al = alignment.t_to_s
#         f = len(al)-1
#         while al[f] == -1:
#             f -= 1
#         end = al[f]
#         print(end, f, all(i == -1 for i in al))
#         print(al[s:e])
#         print(subs_text[s:e])
#         print(''.join(["ー" if k == -1 else script_whole[k] for k in al[s:e]]))

#         s, e = limit-150, limit
#         print(al[s:e])
#         print(subs_text[s:e])
#         print(''.join(["ー" if k == -1 else script_whole[k] for k in al[s:e]]))
#         print()
#         script_whole = script_whole[end+1:]
#         i += f

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
                l = ["」", " ", "　", "’"] if whitespace else ["」", "》", "’"]
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


def process_audio(model, model_name, language, files, split_mode=sudachipy.Tokenizer.SplitMode.C, use_cache=True):
    def load(file, start, end):
        # the highpass, lowpass filters should options
        out, _ = (
            ffmpeg.input(file, ss=start, to=end)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, af="highpass=f=200,lowpass=f=3000", ar="16k")
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


    # jptokenizer = dictionary.Dictionary(dict_type="full").create()
    # jptokenizer = None
    # def split_tokens(tokens, tokenizer):
    #     words, word_tokens = stable_whisper.timing._split_tokens(tokens, tokenizer)

    #     if tokenizer.language == 'ja':
    #         jp_words, jp_tokens = [], []
    #         tokenized = list(jptokenizer.tokenize(''.join(words), split_mode))
    #         tstart, wstart = 0,  0
    #         while tstart < len(tokenized) or wstart < len(words):
    #             tend, wend = tstart+1, wstart+1
    #             x, y = tokenized[tstart].surface(), words[wstart]
    #             while len(x) != len(y):
    #                 if len(x) < len(y):
    #                     x += tokenized[tend].surface()
    #                     tend += 1
    #                 else:
    #                     y += words[wend]
    #                     wend += 1
    #             jp_words.append(x)
    #             jp_tokens.append(list(chain.from_iterable(word_tokens[wstart:wend])))
    #             tstart, wstart = tend, wend
    #         words, word_tokens = jp_words, jp_tokens
    #     # print(words)
    #     return words, word_tokens

    def process(data):
        return model.transcribe_stable(data, language=language, vad=True, regroup=True, verbose=None, word_timestamps=True, input_sr=16000)
        # return model.transcribe(
        #     data,
        #     fp16=False,
        #     # demucs=True,
        #     language=language,
        #     # suppress_silence=True,
        #     vad=True,
        #     regroup=True,
        #     verbose=None, # IT"S NONE NOT FALSE
        #     word_timestamps=True,
        #     split_callback=split_tokens,
        # )

    # this is neat, but the code is awful, and it dosen't cover all the edge cases.
    # So I'll probably just remove it
    def edit_tags(info):
        info["format"].setdefault("tags", {})
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

    transcript, h, offset = [], [0], 0
    with tqdm(files) as bar:
        bar.set_description("Processing audio")
        for file in bar:
            info = edit_tags(ffmpeg.probe(file, show_chapters=None))
            for ch in (bar2 := tqdm(info["chapters"])):
                s, e = float(ch["start_time"]), float(ch["end_time"])
                # 幼女戦記 has a 付録 that isn't included in the book
                # This should be more generalized to allow skipping chapters based on regex or something
                if ch['tags']['title'].find("付録") >= 0:
                    print(f"\n\n{ch['tags']['title']}\n\n")
                    offset += e - s
                    h.append(offset)
                    continue

                bar.refresh()  # first bar gets broken if i don't do this
                bar2.set_description(f"{ch['id']} {ch['tags']['title']}" + (" is more than an hour, this may crash" if e - s > 60 * 60 * 1 else ""))
                q = Path(CACHEDIR) / (os.path.basename(info["format"]["filename"]) + '.' + str(ch["id"]) +  '.' + model_name + ".subs")
                if q.exists() and use_cache:
                    seg_ts = eval(q.read_bytes().decode('utf-8'))
                    # pprint(seg_ts)
                    print()
                    print()
                    # print(''.join([i['text'] for i in seg_ts['segments']]))
                    print(len(''.join([i['text'] for i in seg_ts['segments']])))
                    print()
                    print()
                else:
                    seg_ts = process(load(file, s, e)).to_dict()
                    q.write_bytes(repr(seg_ts).encode('utf-8'))
                # pprint(seg_ts)
                # stable_whisper.result_to_ass(seg_ts, os.path.basename(info["format"]["filename"]) + '.' + str(ch["id"]) +  '.' + model_name + ".subs")
                # Random tests
                words = [Segment(text=''.join(seg['text']), start=seg['start']+offset, end=seg['end']+offset) for seg in seg_ts['segments']]
                # pprint(seg_ts)
                # pprint(words)
                # try:
                #     words = []
                #     for seg in seg_ts['segments']:
                #         z = segment(seg['text'], language, spaces=True, progress=False, whitespace=False)
                #         # pprint(z)
                #         seg['words'][0]['word'] = seg['words'][0]['word'].lstrip() # This sucks
                #         string, start, end = "", seg['words'][0]['start'], seg['words'][0]['end']
                #         for word in seg['words']:
                #             if not z[0].startswith(string + word['word']):
                #                 # print("OUT")
                #                 # print(repr(string), repr(word['word']))
                #                 words.append(Segment(text=string, start=start+offset, end=end+offset))
                #                 z = z[1:]
                #                 string, start, end = word['word'].lstrip(), word['start'], word['end']
                #             else:
                #                 string += word['word']
                #                 end = word['end']
                #         words.append(Segment(text=string, start=start+offset, end=end+offset))
                # except:
                #     words = [Segment(text=''.join(seg['text']), start=seg['start']+offset, end=seg['end']+offset) for seg in seg_ts['segments']]

                # words = [Segment(text=word['word'], start=word['start']+offset, end=word['end']+offset) for seg in seg_ts['segments'] for word in seg['words']]
                # print('WEBVTT\n\n')
                # print('\n\n'.join(i.vtt() for i in segs))
                # segs = [Segment(text=text, start=seg['start']+offset, end=seg['end']+offset) for text, seg in chain.from_iterable(zip(*i) for i in seg_ts)]
                transcript.extend(words)
                offset += e - s
                h.append(offset)
    # print(transcript)
    # Debugging stuff
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
    parser.add_argument("--split_mode", type=str, help="Tokenizer split mode", default='C')
    parser.add_argument("--language", help="language of the script and audio", type=str, default="ja")
    parser.add_argument("--model", help="whisper model to use. can be one of tiny, small, large, huge", type=str, default="tiny")
    parser.add_argument("--output-file", help="name of the output subtitle file", type=str, default=None)
    parser.add_argument("--progress",default=True,  help="progress bar on/off", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-cache", default=True, help="whether to use the cache or not", action=argparse.BooleanOptionalAction)
    parser.add_argument("--split-script", help=r"the regex to split the script with. for monogatari it is something like ^\s[\uFF10-\uFF19]*\s$", type=str, default="")
    args = parser.parse_args()

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.progress)

    if args.output_file is None:
        # args.output_file = args.script + '.vtt'
        args.output_file = os.path.splitext(args.audio_files[0])[0] + ".vtt"
        # args.output_file = "whatever.vtt"

    with open(args.script, "r") as f:
        sentences = segment(f.read(), args.language, whatever=False, spaces=True)

    # model = stable_whisper.load_model(args.model)
    # model = stable_whisper.load_faster_whisper(args.model)
    model = None
    transcripts, h = process_audio(model, args.model, args.language, args.audio_files, split_mode=sudachipy.Tokenizer.SplitMode.__getattribute__(args.split_mode), use_cache=args.use_cache)
    results = find_matches(sentences, transcripts)

    # # For split files
    # # file = open(os.path.splitext(args.audio_files[0])[0] + '.vtt', "w")
    # # file.write("WEBVTT\n\n")
    # # for i in results:
    # #     if i[1].start > h[1]:
    # #         file.flush()
    # #         file.close()
    # #         h = h[1:]
    # #         args.audio_files = args.audio_files[1:]
    # #         file = open(os.path.splitext(args.audio_files[0])[0] + '.vtt', "w")
    # #         file.write("WEBVTT\n\n")

    # #     segment = Segment(text=i[0], start=i[1].start-h[0], end=i[1].end-h[0])
    # #     file.write(segment.vtt() + "\n\n")

#     with open(args.output_file, "w") as out:
#         out.write("WEBVTT\n\n")
#         # for i in results:
#         subs = "\n\n".join(Segment(text=t, start=s.start, end=s.end).vtt() for t, s in results)
#         out.write(subs)
