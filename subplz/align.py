from fuzzywuzzy import fuzz
import re
from datetime import datetime
from tqdm import tqdm
from ats.main import Segment
from subplz.files import get_tmp_path

MAX_MERGE_COUNT = (
    6
)  # Larger gives better results, but takes longer to process.
MAX_SEARCH_CONTEXT = MAX_MERGE_COUNT * 2

# Trim script for quick testing
# script = script[:500]
# subs = subs[:1000]

# Use dynamic programming to pick best subs mapping
memo = {}


class ScriptLine:
    def __init__(self, line):
        self.text = line
        # self.txt = re.sub("「|」|『|』|、|。|・|？|…|―|─|！|（|）", "", line)

    def __repr__(self):
        return "ScriptLine(%s)" % self.text


def read_script(file):
    for line in file:
        line = line.rstrip("\n")
        if line == "":
            continue
        yield line


def get_script(script, script_pos, num_used, sep=""):
    end = min(len(script), script_pos + num_used)
    return sep.join([sub.text for sub in script[script_pos:end]])


def get_base(subs, sub_pos, num_used, sep=""):
    end = min(len(subs), sub_pos + num_used)
    return sep.join([sub.text for sub in subs[sub_pos:end]])


def get_best_sub_n(
    script, subs, script_pos, num_used_script, last_script_pos, sub_pos, max_subs, last_sub_to_test
):
    t_best_score = 0
    t_best_used_sub = 1

    line = get_script(script, script_pos, num_used_script)

    remaining_subs = last_sub_to_test - sub_pos

    for num_used_sub in range(1, min(max_subs, remaining_subs) + 1):
        base = get_base(subs, sub_pos, num_used_sub)
        curr_score = fuzz.ratio(base, line) / 100.0 * min(len(line), len(base))
        tot_score = curr_score + calc_best_score(
            script,
            subs,
            script_pos + num_used_script,
            last_script_pos,
            sub_pos + num_used_sub,
            last_sub_to_test,
        )
        if tot_score > t_best_score:
            t_best_score = tot_score
            t_best_used_sub = num_used_sub

    return (t_best_score, t_best_used_sub)


best_script_score_and_sub = {}


def calc_best_score(script, subs, script_pos, last_script_pos, sub_pos, last_sub_to_test):
    if script_pos >= len(script) or sub_pos >= len(subs):
        return 0

    key = (script_pos, sub_pos)
    if key in memo:
        return memo[key][0]

    best_score = 0
    best_used_sub = 1
    best_used_script = 1

    remaining_script = last_script_pos - script_pos

    for num_used_script in range(1, min(MAX_MERGE_COUNT, remaining_script) + 1):
        max_subs = MAX_MERGE_COUNT if num_used_script == 1 else 1
        t_best_score, t_best_used_sub = get_best_sub_n(
            script,
            subs,
            script_pos,
            num_used_script,
            last_script_pos,
            sub_pos,
            max_subs,
            last_sub_to_test,
        )

        if t_best_score > best_score:
            best_score = t_best_score
            best_used_sub = t_best_used_sub
            best_used_script = num_used_script

    if best_used_script > 1:
        # Do one more fitting
        t_best_score, t_best_used_sub = get_best_sub_n(
            script,
            subs,
            script_pos,
            best_used_script,
            last_script_pos,
            sub_pos,
            MAX_MERGE_COUNT,
            last_sub_to_test,
        )
        if t_best_score > best_score:
            best_score = t_best_score
            best_used_sub = t_best_used_sub

    key = (script_pos, sub_pos)
    memo[key] = (best_score, best_used_sub, best_used_script)

    # Save best sub pos for this script pos
    best_prev_score, best_sub = best_script_score_and_sub.get(script_pos, (0, None))
    if best_score >= best_prev_score:
        best_script_score_and_sub[script_pos] = (best_score, key)

    return best_score


def get_best_sub_path(script_pos, n, last_script_pos, last_sub_to_test):
    _, key = best_script_score_and_sub[script_pos]
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


def test_sub_pos(script, subs, script_pos, last_script_pos, first_sub_to_test, last_sub_to_test):
    for sub_pos in range(last_sub_to_test - 1, first_sub_to_test - 1, -1):
        calc_best_score(script, subs, script_pos, last_script_pos, sub_pos, last_sub_to_test)


def recursively_find_match(script, subs, result, first_script, last_script, first_sub, last_sub, bar):
    bar.total += 1
    bar.refresh()
    if first_script == last_script or first_sub == last_sub:
        return

    memo.clear()
    best_script_score_and_sub.clear()

    mid = (first_script + last_script) // 2
    start = max(first_script, mid - MAX_SEARCH_CONTEXT)
    end = min(mid + MAX_SEARCH_CONTEXT, last_script)

    # print('testing first %d last %d mid %d' % (first_script, last_script, mid))
    for script_pos in range(end - 1, start - 1, -1):
        test_sub_pos(script, subs, script_pos, end, first_sub, last_sub)

    best_path = get_best_sub_path(start, end - start, end, last_sub)
    if len(best_path) > 0:
        for p in best_path:
            if p[0] > mid:
                break
            mid_key = p

        mid_memo = memo[mid_key]
        script_pos = mid_key[0]
        sub_pos = mid_key[1]
        num_used_script = mid_memo[2]
        num_used_sub = mid_memo[1]

        # Recurse before
        recursively_find_match(
            script, subs, result, first_script, script_pos, first_sub, sub_pos, bar
        )
        bar.update(1)
        scr = get_script(script, script_pos, num_used_script, " ‖ ")
        scr_out = get_script(script, script_pos, num_used_script, "")
        base = get_base(subs, sub_pos, num_used_sub, " ‖ ")

        # print((script_pos, num_used_script, sub_pos, num_used_sub), scr, '==', base)
        result.append((script_pos, num_used_script, sub_pos, num_used_sub))

        # Recurse after
        recursively_find_match(
            script,
            subs,
            result,
            script_pos + num_used_script,
            last_script,
            sub_pos + num_used_sub,
            last_sub,
            bar,
        )
        bar.update(1)
    bar.update(1)
    # t.total = new_total
    # t.refresh()

def remove_tags(line):
    return re.sub("<[^>]*>", "", line)

def get_lines(file):
    for line in file:
        yield line.rstrip("\n")

# class Subtitle:
#     def __init__(self, start, end, line):
#         self.start = start
#         self.end = end
#         self.text = line

# TODO kill this off
def read_vtt(file):
    lines = get_lines(file)

    subs = []
    header = next(lines)
    assert header == "WEBVTT"
    # assert next(lines) == "Kind: captions"
    # assert next(lines).startswith("Language:")
    assert next(lines) == ""

    last_sub = " "

    while True:
        # for t in range(0, 10):
        line = next(lines, None)
        if line == None:  # EOF
            break
        # print(line)
        m = re.findall(
            r"(\d\d:\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)",
            line,
        )
        if not m:
            print(
                f'Warning: Line "{line}" did not look like a valid VTT input. There could be issues parsing this sub'
            )
            continue

        matchPair = [list(filter(None, x)) for x in m][0]
        sub_start = matchPair[0]  # .replace('.', ',')
        sub_end = matchPair[1]

        line = next(lines)
        while line:
            sub = remove_tags(line)
            if last_sub != sub and sub not in [" ", "[音楽]"]:
                last_sub = sub
                # print("sub:", sub_start, sub_end, sub)
                subs.append(Segment(sub, sub_start, sub_end))
            elif last_sub == sub and subs:
                subs[-1].end = sub_end
                # print("Update sub:", subs[-1].start, subs[-1].end, subs[-1].text)
            try:
                line = next(lines)
            except StopIteration:
                line = None

    return subs


def to_float(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
    time_delta = time_obj - datetime(1900, 1, 1)
    float_time = time_delta.total_seconds()
    return float_time



def nc_align(split_script, subs_file):
    with open(split_script, encoding='utf-8') as s:
        script = [ScriptLine(line.strip()) for line in read_script(s)]
    print(subs_file)
    with open(subs_file, encoding='utf-8') as vtt:
        subs = read_vtt(vtt)
    new_subs = []

    result = []
    print("🤝 Grouping based on transcript...")
    bar = tqdm(total=0)
    recursively_find_match(script, subs, result, 0, len(script), 0, len(subs), bar)
    bar.close()
    for i, (script_pos, num_used_script, sub_pos, num_used_sub) in enumerate(
        tqdm(result)
    ):
        if i == 0:
            script_pos = 0
            sub_pos = 0

        if i + 1 < len(result):
            num_used_script = result[i + 1][0] - script_pos
            num_used_sub = result[i + 1][2] - sub_pos
        else:
            num_used_script = len(script) - script_pos
            num_used_sub = len(subs) - sub_pos

        scr_out = get_script(script, script_pos, num_used_script, "")
        scr = get_script(script, script_pos, num_used_script, " ‖ ")
        base = get_base(subs, sub_pos, num_used_sub, " ‖ ")

        # print('Record:', script_pos, scr, '==', base)
        new_subs.append(
            Segment(scr_out, to_float(subs[sub_pos].start), to_float(subs[sub_pos + num_used_sub - 1].end))
        )

    return new_subs
