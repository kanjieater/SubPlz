# -*- coding: utf-8 -*-

import pysbd
import sys

inputs = [sys.argv[1]]
for file_name in inputs:
    with open(file_name, 'r', encoding='UTF-8') as file:
        ilines = file.readlines()

    seg = pysbd.Segmenter(language="en", clean=False)
    with open(file_name + '.split.txt', 'w', encoding='UTF-8') as fo:
        lines = []
        for i, text in enumerate(ilines):
            if (i % 10) == 0:
                print('%d/%d' % (i, len(ilines)))
            text.rstrip('\n')
            lines += seg.segment(text)

        # Fix end of quotes
        fixed = []
        for i, line in enumerate(lines):
            if i > 0 and len(line) > 0 and line[0] in ['」', '’']:
                fixed[-1] += line[0]
                line = line[1:]
            if len(line):
                fixed.append(line)
        lines = fixed

        # Merge short lines with quotes
        fixed = []
        for i, line in enumerate(lines):
            if len(fixed) > 0:
                if (fixed[-1][0] in ['」', '’']) and len(fixed[-1]+line) <= 80:
                    fixed[-1] += line
                    continue
            fixed.append(line)
        lines = fixed

        for line in lines:
            if line != '':
                fo.write(line + '\n')
