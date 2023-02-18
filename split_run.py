from fuzzywuzzy import fuzz
import argparse
import sys
import re
from glob import glob
from os import path
import subprocess
import stable_whisper
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
    results = model.transcribe(audio_file, language="ja")
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
    # hf = str(s // 3600).rjust(2, 0)
    # mf = str(s % 3600 // 60).rjust(2, 0)
    # sf = str(s % 60).rjust(2, 0)
    mif = str(micro)[:3].ljust(3,'0')
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_string = '{:02}:{:02}:{:02}.{:s}'.format(int(hours), int(minutes), int(seconds), mif)
    return formatted_string
    # return datetime.strptime(str(delta), "%H:%M:%S.%f")


def adjust_timings(subs, offset):
    ms_offset = get_time_as_delta(offset)
    for sub in subs:
        sub.start = get_time_str_from_delta(get_time_as_delta(sub.start) + ms_offset)
        sub.end = get_time_str_from_delta(get_time_as_delta(sub.end) + ms_offset)
    return subs


def combine_vtt(vtt_files, output_file_path):
    subs = []
    # offsets = get_offsets()
    last_offset = "00:00:00.000"
    with open(output_file_path, "w") as outfile:
        for n, vtt_file in enumerate(vtt_files):
            with open(vtt_file) as vtt:
                latest_subs = read_vtt(vtt)
                subs += adjust_timings(latest_subs, last_offset)
                last_offset = subs[-1].end

                # for sub in subs:
                #     write_sub(outfile, sub)
                # outfile.write(sub)
                # pass
        # offsets = get_offsets(subs)
        # new_subs = adjust_timings(subs,)
        write_sub(outfile, subs)


files_grabbed = grab_files(split_folder, ["*.mp3", "*.m4b", "*.mp4"])
print(files_grabbed)

for audio_file in files_grabbed:
    file_name = path.splitext(audio_file)[0]
    full_timings_path = path.join(split_folder, f"{file_name}.ass")
    full_vtt_path = path.splitext(full_timings_path)[0] + ".vtt"

    # generate_transcript_from_audio(audio_file, full_timings_path)
    # convert_ass_to_vtt(full_timings_path, full_vtt_path)

vtt_files = grab_files(split_folder, ["*.vtt"])
combine_vtt(vtt_files, path.join(full_folder, f'{"timings"}.vtt'))
