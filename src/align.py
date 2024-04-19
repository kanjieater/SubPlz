import regex as re
import unicodedata

import numpy as np
import sys
from Bio import Align
from pprint import pprint
np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)

ascii_to_wide = dict((i, chr(i + 0xfee0)) for i in range(0x21, 0x7f))
kata_hira = dict((0x30a1 + i, chr(0x3041 + i)) for i in range(0x56))

kansuu = '一二三四五六七八九十〇零壱弐参肆伍陸漆捌玖拾'
arabic = '１２３４５６７８９１００'
kansuu_to_arabic = {ord(kansuu[i]): arabic[i%len(arabic)] for i in range(len(kansuu))}

closingpunc =  set('・,，!！?？:：”)]}、』」）〉》〕】｝］')
test =  {ord(i): '。' for i in closingpunc}
test2 = {ord('は'): "わ"}

allt = kata_hira | kansuu_to_arabic | ascii_to_wide | test | test2

def clean(s, normalize=True):
    r = r"(?![。])[\p{C}\p{M}\p{P}\p{S}\p{Z}\sーぁぃぅぇぉっゃゅょゎゕゖァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇷ゚ㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ]+"
    s = re.sub(r, "", s.translate(allt))
    return unicodedata.normalize("NFKD", s) if normalize else s

def align_sub(coords, text, subs):
    current = [0, 0]
    pos, toff = np.array([0, 0]), 0
    p, gaps = np.array([0, 0]), np.array([0, 0])
    gaps_text = []
    segments, unused = [], []
    off, s = 0, 0
    count = 0
    for i in range(coords.shape[1]):
        c = coords[:, i]
        isgap = 0 in (c - p)

        while current[1] < len(subs) and (pos[1] + len(subs[current[1]])) <= c[1]:
            pos[1] += len(subs[current[1]])
            if isgap: gaps += np.clip(pos - p, 0, None)[::-1]

            diff = len(subs[current[1]]) + gaps[1] - gaps[0]
            if diff > len(subs[current[1]])//4:
                segments.append([*current, toff, 0])
                if toff + diff + off > len(text[current[0]]):
                    print(toff, toff+diff+off, text[current[0]], len(text[current[0]]))
                    count += 1
                    # pass

                pos[0] += diff + off
                toff += diff + off
                off = 0

                f = toff >= len(text[current[0]])
                while current[0] < len(text) and toff >= len(text[current[0]]):
                    toff -= len(text[current[0]])
                    current[0] += 1
                    segments.append([*current, 0, 0])
                # if f and toff != 0:
                #     print("Not zero", toff)
                segments.append([*current, toff, 0])
                gaps = []
            else:
                unused.append(subs[current[1]])
                if toff >= len(text[current[0]])//2:
                    toff += diff
                else:
                    off += diff

            current[1] += 1
            s, gaps[...], p[:] = i, 0, np.maximum(pos, p)

        if isgap: gaps += (c - p)[::-1]
        p = c

    print(count)
    pprint(unused)
    # pprint(segments)
    return segments

def fix(original, edited, segments):
    s, e = 0, 0
    while e < len(segments):
        e += 1
        if e == len(segments) or segments[s][0] != segments[e][0]:
            if segments[s][0] >= len(original):
                break
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

def align(model, transcript, text, prepend, append):
    transcript_str = [i['text'] for i in transcript['segments']]
    transcript_str_clean = [clean(i, normalize=False) for i in transcript_str]
    transcript_str_joined = ''.join(transcript_str_clean)

    aligner = Align.PairwiseAligner(scoring=None, mode='global', match_score=0.8, open_gap_score=-1.2, mismatch_score=-0.8, extend_gap_score=-0.5)
    # aligner = Align.PairwiseAligner(scoring=None, mode='global', match_score=1, open_gap_score=-1, mismatch_score=-1, extend_gap_score=-1)
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
        # fix_punc(text_str, segments, prepend, append)
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
