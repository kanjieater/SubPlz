import pysbd
from tqdm import tqdm

def fix_end_of_quotes(lines):
    fixed = []
    for i, line in enumerate(lines):
        if i > 0 and line and line[0] in ["」", "’"]:
            fixed[-1] += line[0]
            line = line[1:]
        if line:
            fixed.append(line)
    return fixed

def merge_short_lines_with_quotes(lines):
    fixed = []
    for line in lines:
        if fixed and fixed[-1][0] in ["」", "’"] and len(fixed[-1] + line) <= 1:
            fixed[-1] += line
        else:
            fixed.append(line)
    return fixed

def split_sentences(input_file, output_path, lang):
    seg = pysbd.Segmenter(language=lang, clean=False)

    # for file_name in input_file:
    with open(input_file, "r", encoding="UTF-8") as file:
        input_lines = file.readlines()

    lines = []
    print("✂️ Splitting transcript into sentences")
    for text in tqdm(input_lines):
        text = text.rstrip("\n")
        lines += seg.segment(text)

    lines = fix_end_of_quotes(lines)
    lines = merge_short_lines_with_quotes(lines)

    with open(output_path, "w", encoding="utf-8") as fo:
        for line in lines:
            if line:
                fo.write(line + "\n")