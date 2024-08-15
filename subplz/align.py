from rapidfuzz import fuzz
import re
from typing import List

from tqdm import tqdm
from ats.main import Segment
from subplz.cli import PUNCTUATION, START_PUNC, END_PUNC

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
    script,
    subs,
    script_pos,
    num_used_script,
    last_script_pos,
    sub_pos,
    max_subs,
    last_sub_to_test,
    max_merge_count
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
            max_merge_count,
        )
        if tot_score > t_best_score:
            t_best_score = tot_score
            t_best_used_sub = num_used_sub

    return (t_best_score, t_best_used_sub)


best_script_score_and_sub = {}


def calc_best_score(
    script, subs, script_pos, last_script_pos, sub_pos, last_sub_to_test, max_merge_count
):
    if script_pos >= len(script) or sub_pos >= len(subs):
        return 0

    key = (script_pos, sub_pos)
    if key in memo:
        return memo[key][0]

    best_score = 0
    best_used_sub = 1
    best_used_script = 1

    remaining_script = last_script_pos - script_pos

    for num_used_script in range(1, min(max_merge_count, remaining_script) + 1):
        max_subs = max_merge_count if num_used_script == 1 else 1
        t_best_score, t_best_used_sub = get_best_sub_n(
            script,
            subs,
            script_pos,
            num_used_script,
            last_script_pos,
            sub_pos,
            max_subs,
            last_sub_to_test,
            max_merge_count,
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
            max_merge_count,
            last_sub_to_test,
            max_merge_count,
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


def test_sub_pos(
    script, subs, script_pos, last_script_pos, first_sub_to_test, last_sub_to_test, max_merge_count
):
    for sub_pos in range(last_sub_to_test - 1, first_sub_to_test - 1, -1):
        calc_best_score(
            script, subs, script_pos, last_script_pos, sub_pos, last_sub_to_test, max_merge_count
        )


def recursively_find_match(
    script, subs, result, first_script, last_script, first_sub, last_sub, max_merge_count, bar=None
):
    if bar is None:
        bar = tqdm(total=1, position=0, leave=True)

    if first_script == last_script or first_sub == last_sub:
        bar.close()
        return

    memo.clear()
    best_script_score_and_sub.clear()
    max_search_context = max_merge_count * 2
    mid = (first_script + last_script) // 2
    start = max(first_script, mid - max_search_context)
    end = min(mid + max_search_context, last_script)

    for script_pos in tqdm(range(end - 1, start - 1, -1), position=1, leave=False):
        test_sub_pos(script, subs, script_pos, end, first_sub, last_sub, max_merge_count)

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

        recursively_find_match(
            script, subs, result, first_script, script_pos, first_sub, sub_pos, max_merge_count, bar
        )

        scr_out = get_script(script, script_pos, num_used_script, "")
        scr = get_script(script, script_pos, num_used_script, " ‖ ")
        base = get_base(subs, sub_pos, num_used_sub, " ‖ ")

        result.append((script_pos, num_used_script, sub_pos, num_used_sub))

        recursively_find_match(
            script,
            subs,
            result,
            script_pos + num_used_script,
            last_script,
            sub_pos + num_used_sub,
            last_sub,
            max_merge_count,
            bar,
        )
    bar.close()


def remove_tags(line):
    return re.sub("<[^>]*>", "", line)


def get_lines(file):
    for line in file:
        yield line.rstrip("\n")


def read_subtitles(file):
    lines = get_lines(file)
    subs = []
    first_line = next(lines)
    is_vtt = first_line == "WEBVTT"
    if is_vtt:
        assert next(lines) == ""
    last_sub = " "
    while True:
        line = next(lines, None)
        if line is None:  # EOF
            break
        # Match timestamp lines for both VTT and SRT formats
        m = re.findall(
            r"(\d\d:\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)|(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)|(\d\d:\d\d,\d\d\d) --> (\d\d:\d\d,\d\d\d)|(\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)",
            line,
        )
        if not m:
            if not line.isdigit() and line:
                print(
                    f'Warning: Line "{line}" did not look like a valid VTT/SRT input. There could be issues parsing this sub'
                )
            continue

        match_pair = [list(filter(None, x)) for x in m][0]
        sub_start = match_pair[0].replace(",", ".")  # Convert SRT to VTT format
        sub_end = match_pair[1].replace(",", ".")

        # Read the subtitle text
        line = next(lines)
        sub_text = []
        while line:
            sub_text.append(remove_tags(line))
            try:
                line = next(lines)
            except StopIteration:
                line = None
            if line == "":
                break

        sub = " ".join(sub_text)
        if sub and last_sub != sub and sub not in [" ", "[音楽]"]:
            last_sub = sub
            subs.append(Segment(sub, sub_start, sub_end))
        elif last_sub == sub and subs:
            subs[-1].end = sub_end

    return subs

def to_float(time_str):
    time_components = time_str.split(":")[::-1]
    total_seconds = 0
    for i, component in enumerate(time_components):
        total_seconds += float(component) * (60 ** i)
    return total_seconds


def nc_align(split_script, subs_file, max_merge_count):
    with open(split_script, encoding="utf-8") as s:
        script = [ScriptLine(line) for line in read_script(s)]
    print(subs_file)
    with open(subs_file, encoding="utf-8") as vtt:
        subs = read_subtitles(vtt)
    new_subs = []

    result = []
    print("🤝 Grouping based on transcript...")
    bar = tqdm(total=0)
    recursively_find_match(script, subs, result, 0, len(script), 0, len(subs), max_merge_count, bar)
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
            Segment(
                scr_out,
                to_float(subs[sub_pos].start),
                to_float(subs[sub_pos + num_used_sub - 1].end),
            )
        )

    return new_subs


def double_check_misaligned_pairs(segments):
    if not segments or len(segments) < 2:
        return segments

    adjusted_segments = []
    for i, segment in enumerate(segments):
        segment = handle_specific_pattern(segment, segments, i)
        segment = handle_starting_punctuation(segment, adjusted_segments, i)
        segment = handle_ending_punctuation(segment, segments, i)
        adjusted_segments.append(segment)

    return adjusted_segments

def handle_specific_pattern(segment, segments, index):
    PATTERN_1 = r'」「(.{1,2})、$'  # Pattern for '」「ばか、'
    PATTERN_2 = r'」「(.{1})、$'  # Pattern for '」「ば、'
    combined_pattern = f'({PATTERN_1}|{PATTERN_2})'
    match = re.search(combined_pattern, segment.text)
    if match:
        matched_text = match.group(0)
        if index < len(segments) - 1:
            segments[index + 1].text = matched_text + segments[index + 1].text
            segment.text = segment.text[:match.start()]

    return segment


def handle_starting_punctuation(segment, adjusted_segments, index):
    if segment.text and segment.text[0] in END_PUNC and index > 0:
        adjusted_segments[-1].text += segment.text[0]
        segment.text = segment.text[1:]
    return segment

def handle_ending_punctuation(segment, segments, index):
    if segment.text and segment.text[-1] in START_PUNC and index < len(segments) - 1:
        segments[index + 1].text = segment.text[-1] + segments[index + 1].text
        segment.text = segment.text[:-1]
    return segment

def find_punctuation_index(s: str) -> int:
    indices = [i for i, char in enumerate(s) if char in PUNCTUATION]
    return indices

def has_ending_punctuation(s: str) -> bool:
    indices = [i for i, char in enumerate(s) if char in END_PUNC]
    return bool(indices)


def has_punctuation(s: str) -> bool:
    return bool(find_punctuation_index(s))

def has_double_comma(str_starts: str, str_ends: str) -> bool:
    comma = """、"""
    return str_starts[-1] == comma and str_ends[-1] == comma



def count_non_punctuation(s: str) -> int:
    return len([char for char in s if char not in PUNCTUATION])


def find_index_with_non_punctuation_start(indices: List[int]) -> List[int]:
    """Removes sequential indices, keeping only the first occurrence in a sequence."""
    if not indices:
        return []

    result = [indices[0]]

    for i in range(1, len(indices)):
        # Add index if it's not consecutive with the previous index
        if indices[i] != indices[i - 1] + 1:
            result.append(indices[i])
        elif i == 1:  # Keep the first in a consecutive sequence
            result.append(indices[i])
        else:  # Replace the last item with the current index
            result[-1] = indices[i]

    return result

def find_index_with_non_punctuation_end(indices: List[int]) -> List[int]:
    """Removes sequential indices, keeping only the last occurrence in a sequence."""
    if not indices:
        return []
    result = []
    for i in range(len(indices) - 1):
        if indices[i] != indices[i + 1] - 1:
            result.append(indices[i])
    result.append(indices[-1])
    return result


def trim_segments(segments: List['Segment']) -> List['Segment']:
    return [Segment(text=segment.text.strip(), start=segment.start, end=segment.end) for segment in segments]


def print_modified_segments(segments, new_segments, final_segments,modified_new_segment_debug_log, modified_final_segment_debug_log):
    print("Modified Start segments:")
    for index in modified_new_segment_debug_log:
        print(f"""
            Original: {segments[index].text}
            Modified: {new_segments[index].text}
            """)

    print("Modified End segments:")
    for index in modified_final_segment_debug_log:
        print(f"""
            Original: {segments[index].text}
            Modified: {final_segments[index].text}
            """)

def shift_align(segments: List[Segment]) -> List[Segment]:
    modified_new_segment_debug_log = []
    modified_final_segment_debug_log = []
    new_segments = []
    for i, segment in enumerate(segments):
        text = segment.text

        # If no punctuation is present, keep the segment unchanged
        if not has_punctuation(text):
            new_segments.append(segment)
            continue

        # Case 2.a: Handle case for starting index
        indices = find_punctuation_index(text)
        if indices:
            start_index = find_index_with_non_punctuation_start(indices)[0]
            # Case 2.c: Handle empty non-punctuation chunks
            # if the substring would result in an empty string if it were removed
            non_punc_count = count_non_punctuation(text[0:start_index])
            if non_punc_count == 0 or \
                count_non_punctuation(text[start_index+1:]) == 0:
                new_segments.append(segment)
                continue
            if non_punc_count <= 2:
                # If the first segment has 2 or fewer non-punctuation characters
                if i > 0 and len(new_segments) > 0 and not has_ending_punctuation(new_segments[-1].text[-1]) and not has_double_comma(new_segments[-1].text, text[:start_index + 1]):
                    # Move part of the text to the previous segment
                    prev_segment = new_segments.pop()
                    prev_segment.text += text[:start_index + 1]  # Include the punctuation
                    new_segments.append(prev_segment)
                    text = text[start_index + 1:]  # Exclude the punctuation
                    modified_new_segment_debug_log.append(i)
        new_segments.append(Segment(text, segment.start, segment.end))

    final_segments = []
    skip_next = False
    for i, segment in enumerate(new_segments):
        if skip_next:
            skip_next = False
            continue
        text = segment.text
        indices = find_punctuation_index(text)
        if indices:
            last_index = find_index_with_non_punctuation_end(indices)[-1]
            non_punc_count = count_non_punctuation(text[last_index:])
            if non_punc_count == 0 or \
                (len(text[last_index:])) == len(text):
                final_segments.append(segment)
                continue
            if count_non_punctuation(text[indices[-1]:]) <= 2:
                # Move part of the text to the next segment
                if i+1 < len(new_segments) and not has_ending_punctuation(new_segments[i+1].text[0]) and not has_double_comma(new_segments[i+1].text[0], text[:start_index + 1]):
                    next_segment = segments[i + 1]
                    next_segment.text = text[last_index+1:] + next_segment.text  # Keep the punctuation
                    text = text[:last_index+1]  # Exclude the punctuation
                    final_segments.append(Segment(text, segment.start, segment.end))
                    final_segments.append(next_segment)
                    skip_next = True
                    modified_final_segment_debug_log.append(i)
                    continue
        final_segments.append(Segment(text, segment.start, segment.end))

    # print_modified_segments(segments, new_segments, final_segments, modified_new_segment_debug_log, modified_final_segment_debug_log)
    return trim_segments(double_check_misaligned_pairs(final_segments))