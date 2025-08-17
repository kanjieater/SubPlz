import numpy as np
from Bio import Align


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
                return segments[: len(text)]
            pos[1] += len(subs[current[1]])
            if isgap:
                gaps += np.clip(pos - p, 0, None)[::-1]

            diff = len(subs[current[1]]) + gaps[1] - gaps[0]
            if diff > len(subs[current[1]]) // 4:
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
                        if end - start < thing or len(set(region) - set("。")) < thing:
                            if len(
                                prev
                            ):  # and start - prev[-1][1] < 5: # Check for gaps
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
                if (
                    toff >= len(text[current[0]]) // 2 and len(prev) and len(prev[-1])
                ):  # Check for gaps
                    prev[-1][1] += diff
                    toff += diff
                else:
                    off += diff

            current[1] += 1
            s, gaps[...], p[:] = i, 0, np.maximum(pos, p)

        if isgap:
            gaps += (c - p)[::-1]
        p = c

    return segments[: len(text)]


# """"""Heuristics"""""""
def fix_punc(text, segments, prepend, append, nopend):
    for l, s in enumerate(segments):
        if not s:
            continue
        t = text[l]
        for p, f in zip(s, s[1:] + [s[-1]]):
            connected = f[0] == p[1]
            loop = 0
            while True:
                if loop > 20:
                    break
                if p[1] < len(t) and t[p[1]] in append:
                    p[1] += 1
                elif t[p[1] - 1] in prepend:
                    p[1] -= 1
                elif (
                    (p[1] > 0 and t[p[1] - 1] in nopend)
                    or (p[1] < len(t) and t[p[1]] in nopend)
                    or (p[1] < len(t) - 1 and t[p[1] + 1] in nopend)
                ):
                    start, end = p[1] - 1, p[1]
                    if p[1] < len(t) - 1 and (
                        t[p[1] + 1] in nopend
                        and 0x4E00 > ord(t[p[1]])
                        or ord(t[p[1]]) > 0x9FAF
                    ):  # Bail out if we end on a kanji
                        end += 1

                    while start > 0 and t[start] in nopend:
                        start -= 1
                    while end < len(t) - 1 and t[end] in nopend:
                        end += 1

                    if t[start] in prepend:
                        if p[1] == start:
                            break
                        p[1] = start
                    elif t[start] in append:
                        if p[1] == start + 1:
                            break
                        p[1] = start + 1
                    elif end < len(t) and t[end] in prepend:
                        if p[1] == end:
                            break
                        p[1] = end
                    elif end < len(t) and t[end] in append:
                        if p[1] == end + 1:
                            break
                        p[1] = end + 1
                    else:
                        break
                else:
                    break
                loop += 1
            if connected:
                f[0] = p[1]


def fix(lang, original, edited, segments):
    for l, s in enumerate(segments):
        o, e = lang.translate(original[l]), edited[l]
        m = {}
        oi, ei, ii = 0, 0, 0
        for oi in range(len(o)):
            if ei < len(e) and o[oi] == e[ei]:
                m[ei] = oi
                ei += 1
            oi += 1
        m[ei] = oi  # Snap to end
        m[0] = 0  # Snap to zero

        last = 0
        for i in range(len(e)):
            if i in m:
                last = i
            else:
                m[i] = m[last]

        for k, f in enumerate(s):
            f[0] = m[min(max(f[0], 0), len(e))]
            f[1] = m[min(max(f[1], 0), len(e))]


# This is structured like this to deal with references later
def align(model, lang, transcript, text, references, prepend, append, nopend):
    aligner = Align.PairwiseAligner(
        mode="global",
        match_score=1,
        open_gap_score=-0.8,
        mismatch_score=-0.6,
        extend_gap_score=-0.5,
    )

    transcript_clean = [lang.clean(i) for i in transcript]
    transcript_joined = "".join(transcript_clean)

    def inner(text):
        text_clean = [lang.clean(i) for i in text]
        text_joined = "".join(text_clean)

        if not len(text_joined) or not len(transcript_joined):
            return []
        alignment = aligner.align(text_joined, transcript_joined)[0]
        coords = alignment.coordinates

        segments = align_sub(coords, text_clean, transcript_clean)
        fix(lang, text, text_clean, segments)
        fix_punc(text, segments, prepend, append, nopend)
        return segments

    return inner(text), []  # references
