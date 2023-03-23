import argparse
import sys, os
from os import path
import stable_whisper
import ffmpeg
from utils import read_vtt, write_sub, grab_files
from datetime import datetime, timedelta
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from pprint import pprint
import multiprocessing

SUPPORTED_FORMATS = ["*.mp3", "*.m4b", "*.mp4"]





def run_stable_whisper(audio_file, full_timings_path):
    global model
    if not model:
        model = stable_whisper.load_model("tiny")
    result = model.transcribe(
        audio_file,
        language="ja",
        suppress_silence=True,
        vad=True,
        regroup=True,
        word_timestamps=True,
    )
    result.to_ass(full_timings_path)


def generate_transcript_from_audio(audio_file, full_timings_path):
    run_stable_whisper(audio_file, full_timings_path)


def convert_ass_to_vtt(full_timings_path, full_vtt_path):
    stream = ffmpeg.input(full_timings_path)
    stream = ffmpeg.output(stream, full_vtt_path, loglevel="error").global_args('-hide_banner')
    ffmpeg.run(stream, overwrite_output=True)


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
    mif = str(micro).rjust(6, "0")[:3]
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_string = "{:02}:{:02}:{:02}.{:s}".format(
        int(hours), int(minutes), int(seconds), mif
    )
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
    duration_string = ffmpeg.probe(audio_file_path)["format"]["duration"]
    duration = timedelta(seconds=float(duration_string))
    # print(duration)
    return duration


def get_offsets(audio_files):
    offsets = [timedelta(0)]
    # don't need the last one since there's no file after it to offset
    for n, file in enumerate(audio_files[:-1]):
        offsets.append(get_audio_duration(file) + offsets[n])
    return offsets


def filter_audio(file_path):
    filename, ext = path.splitext(file_path)
    stream = ffmpeg.input(file_path)
    # stream = ffmpeg.filter('highpass', f='200')
    # stream = ffmpeg.filter('lowpass', f='3000')
    stream = ffmpeg.output(
        stream,
        path.join(path.dirname(file_path), f"{filename}.filtered{ext}"),
        af="highpass=f=200,lowpass=f=3000",
        vn=None,
        loglevel="error"
    ).global_args('-hide_banner')
    # print(str(stream.get_args()))
    return ffmpeg.run(stream, overwrite_output=True)


def prep_audio(file_paths, working_folder):
    process_map(filter_audio, file_paths, max_workers=multiprocessing.cpu_count())
    return grab_files(
        working_folder,
        [format.replace("*", "*.filtered") for format in SUPPORTED_FORMATS],
    )


def remove_temp_files(files):
    for file in files:
        try:
            os.remove(file)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

def generate_transcript_from_audio_wrapper(audio_path_dict):
    audio_file = audio_path_dict['audio_file']
    working_folder = audio_path_dict['working_folder']
    file_name = path.splitext(audio_file)[0]
    full_timings_path = path.join(working_folder, f"{file_name}.ass")
    full_vtt_path = path.splitext(full_timings_path)[0] + ".vtt"

    generate_transcript_from_audio(audio_file, full_timings_path)
    convert_ass_to_vtt(full_timings_path, full_vtt_path)


def run():
    full_folder = [sys.argv[1]][0]
    folder_name =  path.dirname(full_folder)
    content_name = path.basename(folder_name)
    split_folder = path.join(full_folder, f"{content_name}_splitted")

    if path.exists(split_folder) and path.isdir(split_folder):
        working_folder = split_folder
    else:
        working_folder = full_folder
    print(f"Working on {working_folder}")

    global model
    model = False  # global model preserved between files

    temp_files = grab_files(working_folder, ['*.filtered.*'])
    remove_temp_files(temp_files)

    audio_files = grab_files(working_folder, SUPPORTED_FORMATS)
    pprint(f"{len(audio_files)} files will be combined in this order:")
    pprint(audio_files)
    if len(audio_files) == 0:
        raise Exception(f"No audio files found at {working_folder}")

    prepped_audio = prep_audio(audio_files, working_folder)
    # prepped_audio = audio_files
    audio_path_dicts = [{"working_folder":working_folder, "audio_file":af} for af in prepped_audio]
    # process_map(generate_transcript_from_audio, audio_path_dicts, max_workers=multiprocessing.cpu_count())
    for audio_path_dict in tqdm(audio_path_dicts):
        generate_transcript_from_audio_wrapper(audio_path_dict)

    audio_file_offsets = get_offsets(prepped_audio)

    vtt_files = grab_files(working_folder, ["*.vtt"])
    combine_vtt(
        vtt_files, audio_file_offsets, path.join(full_folder, f'{"timings"}.vtt')
    )
    temp_files = grab_files(working_folder, ['*.filtered.*'])
    remove_temp_files(temp_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match audio to a transcript")
    run()
    # parser.add_argument('--mode', dest='mode', type=int, default=2,
    #                     help='matching mode')
    # parser.add_argument('--max-merge', dest='max_merge', type=int, default=6,
    #                     help='max subs to merge into one line')


