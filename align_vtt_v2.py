from fuzzywuzzy import fuzz
import argparse
import sys
import re
from utils import Subtitle, read_vtt, write_sub
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Align a script to vtt subs')
parser.add_argument('--mode', dest='mode', type=int, default=2,
                    help='matching mode')
parser.add_argument('--max-merge', dest='max_merge', type=int, default=6,
                    help='max subs to merge into one line')

parser.add_argument('script', type=argparse.FileType('r', encoding='UTF-8'),
                    help='script file path')
parser.add_argument('subs', type=argparse.FileType('r', encoding='UTF-8'),
                    help='.vtt subtitle file path')
parser.add_argument('out', type=argparse.FileType('w', encoding='UTF-8'),
                    help='aligned output file path')

args = parser.parse_args(sys.argv[1:])

MAX_MERGE_COUNT = args.max_merge # Larger gives better results, but takes longer to process.
MAX_SEARCH_CONTEXT = MAX_MERGE_COUNT * 2

class ScriptLine:
    def __init__(self, line):
        self.line = line
        self.txt = re.sub('「|」|『|』|、|。|・|？|…|―|─|！|（|）', '', line)

    def __repr__(self):
        return 'ScriptLine(%s)' % self.line





def read_script(file):
    for line in file:
        line = line.rstrip('\n')
        if line == '':
            continue
        yield line




script = [ScriptLine(line.strip()) for line in read_script(args.script)]
subs = read_vtt(args.subs)

# Trim script for quick testing
#script = script[:500]
#subs = subs[:1000]

# Use dynamic programming to pick best subs mapping
memo = {}

def get_script(script_pos, num_used, sep=''):
    end = min(len(script), script_pos + num_used)
    return sep.join([sub.line for sub in script[script_pos:end]])

def get_base(sub_pos, num_used, sep=''):
    end = min(len(subs), sub_pos + num_used)
    return sep.join([sub.line for sub in subs[sub_pos:end]])

def get_best_sub_n(script_pos, num_used_script, last_script_pos, sub_pos, max_subs, last_sub_to_test):
    t_best_score = 0
    t_best_used_sub = 1

    line = get_script(script_pos, num_used_script)

    remaining_subs = last_sub_to_test - sub_pos

    for num_used_sub in range(1, min(max_subs, remaining_subs) + 1):
        base = get_base(sub_pos, num_used_sub)
        curr_score = fuzz.ratio(base, line) / 100.0 * min(len(line), len(base))
        tot_score = curr_score + calc_best_score(script_pos + num_used_script, last_script_pos, sub_pos + num_used_sub, last_sub_to_test)
        if tot_score > t_best_score:
            t_best_score = tot_score
            t_best_used_sub = num_used_sub

    return (t_best_score, t_best_used_sub)

best_script_score_and_sub = {}

def calc_best_score(script_pos, last_script_pos, sub_pos, last_sub_to_test):
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
        t_best_score, t_best_used_sub = get_best_sub_n(script_pos, num_used_script, last_script_pos, sub_pos, max_subs, last_sub_to_test)

        if t_best_score > best_score:
            best_score = t_best_score
            best_used_sub = t_best_used_sub
            best_used_script = num_used_script

    if best_used_script > 1:
        # Do one more fitting
        t_best_score, t_best_used_sub = get_best_sub_n(script_pos, best_used_script, last_script_pos, sub_pos, MAX_MERGE_COUNT, last_sub_to_test)
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

def test_sub_pos(script_pos, last_script_pos, first_sub_to_test, last_sub_to_test):
    for sub_pos in range(last_sub_to_test - 1, first_sub_to_test - 1, -1):
        calc_best_score(script_pos, last_script_pos, sub_pos, last_sub_to_test)

def recursively_find_match(result, first_script, last_script, first_sub, last_sub, bar):
    bar.total += 1
    bar.refresh()
    if first_script == last_script or first_sub == last_sub:
        return

    memo.clear()
    best_script_score_and_sub.clear()

    mid = (first_script + last_script) // 2
    start = max(first_script, mid - MAX_SEARCH_CONTEXT)
    end = min(mid + MAX_SEARCH_CONTEXT, last_script)

    #print('testing first %d last %d mid %d' % (first_script, last_script, mid))
    for script_pos in range(end - 1, start - 1, -1):
        test_sub_pos(script_pos, end, first_sub, last_sub)

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
        recursively_find_match(result, first_script, script_pos, first_sub, sub_pos, bar)
        bar.update(1)
        scr = get_script(script_pos, num_used_script, ' ‖ ')
        scr_out = get_script(script_pos, num_used_script, '')
        base = get_base(sub_pos, num_used_sub, ' ‖ ')

        # print((script_pos, num_used_script, sub_pos, num_used_sub), scr, '==', base)
        result.append((script_pos, num_used_script, sub_pos, num_used_sub))

        # Recurse after
        recursively_find_match(result, script_pos + num_used_script, last_script, sub_pos + num_used_sub, last_sub, bar)
        bar.update(1)
    bar.update(1)
    # t.total = new_total
    # t.refresh()

new_subs = []

if args.mode == 1:
  last_script_to_test = len(script)
  last_sub_to_test = len(subs)
  first_sub_to_test = 0
  for script_pos in range(len(script) - 1, -1, -1):
      if script_pos == 0:
          first_sub_to_test = 0
      if (script_pos % 10) == 0:
          print("%d/%d testing %d - %d subs " % (script_pos , len(script), first_sub_to_test, last_sub_to_test))

      test_sub_pos(script_pos, last_script_to_test, first_sub_to_test, last_sub_to_test)

  # Construct new subs using the memo trace.
  script_pos = 0
  sub_pos = 0

  while script_pos < len(script) and sub_pos < len(subs):
      try:
          decision = memo[(script_pos, sub_pos)]
      except:
          print("Missing key?", script_pos, sub_pos)
          break
      #print(decision, subs[sub_pos].line)
      num_used_sub = decision[1]
      num_used_script = decision[2]
      scr_out = get_script(script_pos, num_used_script, '')
      scr = get_script(script_pos, num_used_script, ' ‖ ')

      if num_used_sub:
          base = get_base(sub_pos, num_used_sub, ' ‖ ')
          print('Record:', script_pos, scr, '==', base)
          new_subs.append(Subtitle(subs[sub_pos].start, subs[sub_pos+num_used_sub-1].end, scr_out))
          sub_pos += num_used_sub
      else:
          print('Skip:  ', script[script_pos].line)
      script_pos += num_used_script
elif args.mode == 2:
  result = []
  print('Matching subs to sentences. This can take a while...')
  bar = tqdm(total=0)
  recursively_find_match(result, 0, len(script), 0, len(subs), bar)
  bar.close()
  for i, (script_pos, num_used_script, sub_pos, num_used_sub) in enumerate(tqdm(result)):
    if i == 0:
      script_pos = 0
      sub_pos = 0

    if i + 1 < len(result):
      num_used_script = result[i+1][0] - script_pos
      num_used_sub = result[i+1][2] - sub_pos
    else:
      num_used_script = len(script) - script_pos
      num_used_sub = len(subs) - sub_pos

    scr_out = get_script(script_pos, num_used_script, '')
    scr = get_script(script_pos, num_used_script, ' ‖ ')
    base = get_base(sub_pos, num_used_sub, ' ‖ ')

    # print('Record:', script_pos, scr, '==', base)
    new_subs.append(Subtitle(subs[sub_pos].start, subs[sub_pos+num_used_sub-1].end, scr_out))
else:
  sys.exit('Unknown mode %d' % args.mode)

write_sub(args.out, new_subs)
