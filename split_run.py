from fuzzywuzzy import fuzz
import argparse
import sys
import re
from glob import glob
from os import path
import subprocess
import stable_whisper
from stable_whisper import modify_model
import ffmpeg
from vtt_utils import read_vtt, write_sub
from datetime import datetime, timedelta


parser = argparse.ArgumentParser(description="Match audio to a transcript")
# parser.add_argument('--mode', dest='mode', type=int, default=2,
#                     help='matching mode')
# parser.add_argument('--max-merge', dest='max_merge', type=int, default=6,
#                     help='max subs to merge into one line')

full_folder = [sys.argv[1]][0]
input_folder = path.basename(path.dirname(full_folder))
split_folder = path.join(full_folder, f"{input_folder}_splitted")
print(split_folder)

model = False  # global model preserved between files


def grab_files(folder, types):
    files = []
    for type in types:
        pattern = f"{folder}/{type}"
        files.extend(glob(pattern))
    print(files)
    return files


def run_stable_whisper(audio_file, full_timings_path):
    global model
    if not model:
        model = stable_whisper.load_model("large-v2")
        modify_model(model)
    results = model.transcribe(audio_file, language="ja", suppress_silence=False, ts_num=16)
    stable_whisper.results_to_sentence_word_ass(results, full_timings_path)


def generate_transcript_from_audio(audio_file, full_timings_path):
    run_stable_whisper(audio_file, full_timings_path)


def convert_ass_to_vtt(full_timings_path, full_vtt_path):
    stream = ffmpeg.input(full_timings_path)
    stream = ffmpeg.output(stream, full_vtt_path)
    ffmpeg.run(stream, overwrite_output=True)


# def combine_vtt(vtt_files, output_file_path):
#     streams = []
#     for file in vtt_files:
#         streams.append(ffmpeg.input(file))
#     combined = ffmpeg.concat(*streams)
#     stream = ffmpeg.output(combined, output_file_path)
#     ffmpeg.run(stream, overwrite_output=True)


def get_time_as_delta(time_str):
    try:
        a = datetime.strptime(time_str, "%H:%M:%S.%f")  # '16:31:32.123'
    except ValueError:
        a = datetime.strptime(time_str, "%M:%S.%f")
    ts = timedelta(
        hours=a.hour, minutes=a.minute, seconds=a.second, microseconds=a.microsecond
    )
    return ts


def get_time_str_from_delta(delta):
    s = delta.total_seconds()
    micro = delta.microseconds
    mif = str(micro).rjust(6,'0')[:3]
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_string = '{:02}:{:02}:{:02}.{:s}'.format(int(hours), int(minutes), int(seconds), mif)
    return formatted_string


def adjust_timings(subs, offset):
    ms_offset = offset
    for sub in subs:
        sub.start = get_time_str_from_delta(get_time_as_delta(sub.start) + ms_offset)
        sub.end = get_time_str_from_delta(get_time_as_delta(sub.end) + ms_offset)
    return subs


def combine_vtt(vtt_files, offsets, output_file_path):
    subs = []
    with open(output_file_path, "w") as outfile:
        for n, vtt_file in enumerate(vtt_files):
            with open(vtt_file) as vtt:
                latest_subs = read_vtt(vtt)
                last_offset = offsets[n]
                subs += adjust_timings(latest_subs, last_offset)
        write_sub(outfile, subs)


def get_audio_duration(audio_file_path):
    duration_string = ffmpeg.probe(audio_file_path)['format']['duration']
    duration = timedelta(seconds=float(duration_string))
    return duration


def get_offsets(audio_files):
    offsets = [timedelta(0)]
    # don't need the last one since there's no file after it to offset
    for n, file in enumerate(audio_files[:-1]):
        offsets.append(get_audio_duration(file) + offsets[n])
    return offsets


audio_files = grab_files(split_folder, ["*.mp3", "*.m4b", "*.mp4"])

for audio_file in audio_files:
    file_name = path.splitext(audio_file)[0]
    full_timings_path = path.join(split_folder, f"{file_name}.ass")
    full_vtt_path = path.splitext(full_timings_path)[0] + ".vtt"

    generate_transcript_from_audio(audio_file, full_timings_path)
    convert_ass_to_vtt(full_timings_path, full_vtt_path)

audio_file_offsets = get_offsets(audio_files)

vtt_files = grab_files(split_folder, ["*.vtt"])
combine_vtt(vtt_files, audio_file_offsets, path.join(full_folder, f'{"timings"}.vtt'))
