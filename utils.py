import re
from natsort import os_sorted
from glob import glob, escape
import json

audio_formats = [
	'aac',
	'ac3',
	'alac',
	'ape',
	'flac',
	'mp3',
	'm4a',
	'ogg',
	'opus',
	'wav',
	'm4b',
]
video_formats = ['3g2', '3gp', 'avi', 'flv', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'webm']
subtitle_formats = ['ass', 'srt', 'vtt']


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
	header = next(lines)
	assert header == 'WEBVTT'
	# assert next(lines) == "Kind: captions"
	# assert next(lines).startswith("Language:")
	assert next(lines) == ''

	last_sub = ' '

	while True:
		# for t in range(0, 10):
		line = next(lines, None)
		if line == None:  # EOF
			break
		# print(line)
		m = re.findall(
			r'(\d\d:\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d.\d\d\d)|(\d\d:\d\d.\d\d\d) --> (\d\d:\d\d:\d\d.\d\d\d)',
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


def write_sub(output_file_path, subs):
	with open(output_file_path, 'w', encoding='utf-8') as outfile:
		outfile.write('WEBVTT\n\n')
		for n, sub in enumerate(subs):
			# outfile.write('%d\n' % (n + 1))
			outfile.write('%s --> %s\n' % (sub.start, sub.end))
			outfile.write('%s\n\n' % (sub.line))


def grab_files(folder, types, sort=True):
	files = []
	for type in types:
		pattern = f'{escape(folder)}/{type}'
		files.extend(glob(pattern))
	if sort:
		return os_sorted(files)
	return files


def get_mapping(mapping_path):
	with open(mapping_path) as f:
		mapping = json.load(f)
		print(f'Reading mapping: {mapping}')
	return mapping
