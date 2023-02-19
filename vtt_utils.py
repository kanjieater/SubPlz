
import re


class Subtitle:
    def __init__(self, start, end, line):
        self.start = start
        self.end = end
        self.line = line


def remove_tags(line):
    return re.sub('<[^>]*>', '', line)


def get_lines(file):
    for line in file:
        yield line.rstrip('\n')


def read_vtt(file):
    lines = get_lines(file)
    subs = []

    # Read header
    assert next(lines) == "WEBVTT"
    # assert next(lines) == "Kind: captions"
    # assert next(lines).startswith("Language:")
    assert next(lines) == ""

    last_sub = ' '

    while True:
    #for t in range(0, 10):
        line = next(lines, None)
        if line == None: # EOF
            break
        # print(line)
        m = re.findall(r'(\d\d:\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)', line)
        assert m
        matchPair = [list(filter(None, x)) for x in m][0]
        sub_start = matchPair[0] #.replace('.', ',')
        sub_end = matchPair[1]

        line = next(lines)
        while line:
            sub = remove_tags(line)
            if last_sub != sub and sub not in [' ', '[音楽]']:
                last_sub = sub
                # print("sub:", sub_start, sub_end, sub)
                subs.append(Subtitle(sub_start, sub_end, sub))
            elif last_sub == sub and subs:
                subs[-1].end = sub_end
                # print("Update sub:", subs[-1].start, subs[-1].end, subs[-1].line)
            try:
                line = next(lines)
            except StopIteration:
                line = None

    return subs


def write_sub(outfile, subs):
    outfile.write('WEBVTT\n\n')
    for n, sub in enumerate(subs):
        # outfile.write('%d\n' % (n + 1))
        outfile.write('%s --> %s\n' % (sub.start, sub.end))
        outfile.write('%s\n\n' % (sub.line))