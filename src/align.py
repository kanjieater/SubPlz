import regex as re
import unicodedata

import numpy as np
import sys
from Bio import Align
from pprint import pprint
from tqdm import tqdm
np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)

ascii_to_wide = dict((i, chr(i + 0xfee0)) for i in range(0x21, 0x7f))
kata_hira = dict((0x30a1 + i, chr(0x3041 + i)) for i in range(0x56))

kansuu = '一二三四五六七八九十〇零壱弐参肆伍陸漆捌玖拾'
arabic = '１２３４５６７８９１００'
kansuu_to_arabic = {ord(kansuu[i]): arabic[i%len(arabic)] for i in range(len(kansuu))}

closingpunc =  set('・,，!！?？:：”)]}、』」）〉》〕】｝］')
test =  {ord(i): '。' for i in closingpunc}
test2 = {ord('は'): 'わ', ord('あ'): 'わ'}

allt = kata_hira | kansuu_to_arabic | ascii_to_wide | test | test2
g_unused = []

def clean(s, normalize=True):
    r = r'(?![。])[\p{C}\p{M}\p{P}\p{S}\p{Z}\sー々ゝぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ]+'
    s = re.sub(r, '', s.translate(allt))
    s = re.sub(r'(.)(?=\1+)', '', s) # aaa -> a, Doesn't feel that useful but w/e
    return unicodedata.normalize("NFKD", s) if normalize else s

def align_sub(coords, text, subs, thing=2):
    current = [0, 0]
    pos, toff = np.array([0, 0]), 0
    p, gaps = np.array([0, 0]), np.array([0, 0])
    unused = []
    segments = [[]]
    off, s = 0, 0
    count = 0
    for i in range(coords.shape[1]):
        c = coords[:, i]
        isgap = 0 in (c - p)

        while current[1] < len(subs) and (pos[1] + len(subs[current[1]])) <= c[1]:
            if current[0] >= len(text):
                unused.extend(subs[current[1]:])
                g_unused.append(unused)
                return segments[:len(text)]
            pos[1] += len(subs[current[1]])
            if isgap: gaps += np.clip(pos - p, 0, None)[::-1]

            diff = len(subs[current[1]]) + gaps[1] - gaps[0]
            if diff > len(subs[current[1]])//4:
                target = toff + diff + off
                off = 0
                c2 = 0
                while current[0] < len(text) and target >= len(text[current[0]]):
                    start, end = toff, len(text[current[0]])

                    # ips, ipe = pos[0], pos[0] + (end - start)
                    # for  r, j in gaps_text: # TODO:
                    #     if (r >= ips) and (ipe <= j):
                    #         print(ips, ipe, r, j)

                    if (end - start) != 0:
                        region = text[current[0]][start:end]
                        prev = segments[-1]
                        if end - start < thing or len(set(region) - set('。')) < thing:
                            if len(prev): # and start - prev[-1][1] < 5: # Check for gaps
                                prev[-1][1] = end
                            else:
                                prev.append([start, end, current[1]])
                                # print("Hmm")
                        else:
                            # Two chunks stuck together, fix later
                            if target > end:
                                c2 += 1
                            prev.append([start, end, current[1]])

                    segments.append([])
                    pos[0] += end - start
                    target -= len(text[current[0]])
                    current[0] += 1
                    toff = 0
                if c2:
                    count += 1
                pos[0] += target - toff
                segments[-1].append([toff, target, current[1]])
                toff = target
            else:
                unused.append(subs[current[1]])
                prev = segments[-1]
                if toff >= len(text[current[0]])//2 and len(prev) and len(prev[-1]): # Check for gaps
                    prev[-1][1] += diff
                    toff += diff
                else:
                    off += diff

            current[1] += 1
            s, gaps[...], p[:] = i, 0, np.maximum(pos, p)

        if isgap: gaps += (c - p)[::-1]
        p = c


    # pprint(unused)
    # pprint(segments)
    # print(len(text))
    # print(count)
    # for l, s in enumerate(segments[:len(text)]):
    #     if not l: continue
    #     for ss, ee, sub in s:
    #         print(text[l][ss:ee])
    #         print(subs[sub])
    #         print()
    g_unused.append(unused)
    return segments[:len(text)]

# """"""Heuristics"""""""
def fix_punc(text, segments, prepend, append, nopend):
    for l, s in enumerate(segments):
        if not s: continue
        t = text[l]
        for p, f in zip(s, s[1:] + [s[-1]]):
            connected = f[0] == p[1]
            k = 0
            while True:
                if k > 20:
                    break
                if p[1] < len(t) and t[p[1]] in append:
                    p[1] += 1
                elif t[p[1]-1] in prepend:
                    p[1] -= 1
                elif (p[1] > 0 and t[p[1]-1] in nopend) or (p[1] < len(t) and t[p[1]] in nopend) or (p[1] < len(t)-1 and t[p[1]+1] in nopend):
                    start, end = p[1]-1, p[1]
                    if  p[1] < len(t)-1 and (t[p[1]+1] in nopend and 0x4e00 > ord(t[p[1]]) or ord(t[p[1]]) > 0x9faf): # Bail out if we end on a kanji
                        end += 1

                    while start > 0 and t[start] in nopend:
                        start -= 1
                    while end < len(t)-1 and t[end] in nopend:
                        end += 1


                    if t[start] in prepend:
                        if p[1] == start:
                            break
                        p[1] = start
                    elif t[start] in append:
                        if p[1] == start+1:
                            break
                        p[1] = start+1
                    elif end < len(t) and t[end] in prepend:
                        if p[1] == end:
                            break
                        p[1] = end
                    elif end < len(t) and t[end] in append:
                        if p[1] == end+1:
                            break
                        p[1] = end+1
                    else:
                        break
                    # if t[end] in prepend and t[start] in append:
                    #     tqdm.write(t)
                    #     tqdm.write("wtf")
                else:
                    break
                k += 1
            if connected: f[0] = p[1]

def fix(original, edited, segments):
    for l, s in enumerate(segments):
        o, e = original[l].translate(allt), edited[l]
        m = {}
        oi, ei, ii = 0, 0, 0
        for oi in range(len(o)):
            if ei < len(e) and o[oi] == e[ei]:
                m[ei] = oi
                ei += 1
            oi += 1
        m[ei] = oi # Snap to end
        m[0] = 0 # Snap to zero

        for k, f in enumerate(s):
            f[0] = m[f[0]]
            f[1] = m[f[1]]

def align(model, transcript, text, prepend, append, nopend):
    transcript_str = [i['text'] for i in transcript['segments']]
    transcript_str_clean = [clean(i, normalize=False) for i in transcript_str]
    transcript_str_joined = ''.join(transcript_str_clean)

    aligner = Align.PairwiseAligner(scoring=None, mode='global', match_score=0.8, open_gap_score=-0.8, mismatch_score=-1, extend_gap_score=-0.5)
    def inner(text_str):
        text_str_clean = [clean(i, normalize=False) for i in text_str]
        text_str_joined = ''.join(text_str_clean)

        if not len(text_str_joined) or not len(transcript_str_joined): return []
        alignment = aligner.align(text_str_joined, transcript_str_joined)[0]
        # print(alignment.__str__().replace('-', "ー"))
        coords = alignment.coordinates
        # print(coords)

        segments = align_sub(coords, text_str_clean, transcript_str_clean)
        fix(text_str, text_str_clean, segments)
        fix_punc(text_str, segments, prepend, append, nopend)
        return segments

    references = {k.element.attrs['id']:k for i in text for k in i.references} # Filter out dups

    text_str = [i.text() for i in text]
    segments = inner(text_str)
    # pprint(segments)
    # for k, v in references.items():
    #     ref_segments = inner([v.text()], fix3=True)
    #     pprint(ref_segments)
    #     for i in range(len(ref_segments)):
    #         ref_segments[i][0] = k
    #     replace(segments, text_str, ref_segments)

    return segments, references
