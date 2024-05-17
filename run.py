import argparse
import sys, os
import stable_whisper
import ffmpeg
import multiprocessing
from os import path
from pathlib import Path
from datetime import datetime, timedelta
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from pprint import pprint
from utils import read_vtt, write_sub, grab_files
from split_sentences import split_sentences
import align
import traceback


SUPPORTED_FORMATS = ['*.mp3', '*.m4b', '*.mp4']


def get_model(model_type='large-v2'):
    return stable_whisper.load_model(model_type)
    # return stable_whisper.load_faster_whisper(model_type)


def generate_transcript_from_audio(
    audio_file, full_timings_path, model, sub_format='ass', **kwargs
):
    default_args = {
        'language': 'ja',
        'suppress_silence': True,
        # 'vad': True,
        'regroup': True,
        'word_timestamps': True,
        'use_word_position': True,
        # 'only_voice_freq': True,
        'prepend_punctuations': """「"'“¿([{-)""",
        'append_punctuations': """.。,，!！?？:：”)]}、)」""",
    }

    default_args.update(kwargs)

    result = model.transcribe(audio_file, **default_args)
    # if sub_format == 'ass':
    #     result.to_ass(full_timings_path)
    # else:
    result.to_srt_vtt(full_timings_path, word_level=False)


def align_text(model, working_folder, script_file, final):
    file_content = Path(script_file).read_text()
    audio_file = prep_audio(working_folder)
    result = model.align(
        audio_file,
        file_content,
        language='ja',
        original_split=True,
        prepend_punctuations="""「"'“¿([{-)""",
        append_punctuations=""".。,，!！?？:：”)]}、)」""",
        #  use_word_position=True,
        #  max_word_dur=60.0,
        #  word_dur_factor=40.0,
        # vad=True,
    )
    # result = model.refine(audio_file, result, word_level=False)

    result.to_srt_vtt(final, word_level=False)
    return result


def convert_sub_format(full_original_path, full_sub_path):
    stream = ffmpeg.input(full_original_path)
    stream = ffmpeg.output(stream, full_sub_path, loglevel='error').global_args(
        '-hide_banner'
    )
    ffmpeg.run(stream, overwrite_output=True)


def get_time_as_delta(time_str):
    try:
        a = datetime.strptime(time_str, '%H:%M:%S.%f')  # '16:31:32.123'
    except ValueError:
        a = datetime.strptime(time_str, '%M:%S.%f')
    ts = timedelta(
        hours=a.hour, minutes=a.minute, seconds=a.second, microseconds=a.microsecond
    )
    return ts


def get_time_str_from_delta(delta):
    s = delta.total_seconds()
    micro = delta.microseconds
    mif = str(micro).rjust(6, '0')[:3]
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_string = '{:02}:{:02}:{:02}.{:s}'.format(
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

    for n, vtt_file in enumerate(vtt_files):
        with open(vtt_file, encoding='utf-8') as vtt:
            latest_subs = read_vtt(vtt)
            last_offset = offsets[n]
            subs += adjust_timings(latest_subs, last_offset)
    write_sub(output_file_path, subs)


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


def filter_audio(file_path):
    filename, ext = path.splitext(file_path)
    stream = ffmpeg.input(file_path)
    # stream = ffmpeg.filter('highpass', f='200')
    # stream = ffmpeg.filter('lowpass', f='3000')
    stream = ffmpeg.output(
        stream,
        path.join(path.dirname(file_path), f'{filename}.filtered{ext}'),
        af='highpass=f=200,lowpass=f=3000',
        vn=None,
        loglevel='error',
    ).global_args('-hide_banner')
    return ffmpeg.run(stream, overwrite_output=True)


def prep_audio(working_folder, use_cache=False):
    if not use_cache:
        cleanup()

    audio_files = grab_files(working_folder, SUPPORTED_FORMATS)

    # filtered_formats = [
    #     format.replace("*", "*.filtered") for format in SUPPORTED_FORMATS
    # ]
    # cached = grab_files(
    #     working_folder,
    #     filtered_formats,
    # )
    file_paths = audio_files
    # if use_cache and len(cached) != 0:
    #     file_paths = cached

    if len(file_paths) == 0:
        raise Exception(f'No audio files found at {working_folder}')
    if len(file_paths) > 1:
        raise Exception(
            f'Multiple audio files found at {working_folder}. Make sure there is only one, and try again.'
        )
    return audio_files[0]
    # pprint(f"{len(file_paths)} file will be processed:")
    # pprint(file_paths)

    # if not use_cache:
    #     process_map(filter_audio, file_paths, max_workers=multiprocessing.cpu_count())

    # return grab_files(
    #     working_folder,
    #     filtered_formats,
    # )


def remove_files(files):
    for file in files:
        try:
            os.remove(file)
        except OSError as e:
            print('Error: %s - %s.' % (e.filename, e.strerror))


def generate_transcript_from_audio_wrapper(audio_path_dict, model):
    audio_file = audio_path_dict['audio_file']
    working_folder = audio_path_dict['working_folder']
    file_name = path.splitext(audio_file)[0]
    full_timings_path = path.join(working_folder, f'{file_name}')
    full_vtt_path = f'{path.splitext(full_timings_path)[0]}.vtt'.replace(
        '.filtered', ''
    )
    generate_transcript_from_audio(audio_file, full_vtt_path, model)
    # convert_sub_format(full_timings_path, full_vtt_path)
    # remove_files([full_timings_path])


def split_txt(working_folder):
    txt_file = grab_files(
        working_folder,
        ['*.txt'],
    )
    split_file = grab_files(
        working_folder,
        ['*.split.txt'],
    )
    txt_file = list(set(txt_file) - set(split_file))

    if len(txt_file) > 1:
        raise Exception(
            f'Multiple txt files found at {working_folder}. Only one is allowed.'
        )
    split_sentences(txt_file)


def get_content_name(working_folder):
    folder_name = path.dirname(working_folder)
    content_name = path.basename(folder_name)
    return content_name


def get_working_folders(dirs):
    working_folders = []
    for dir in dirs:
        full_folder = os.path.join(dir, '')
        content_name = get_content_name(dir)
        split_folder = path.join(full_folder, f'{content_name}_splitted')

        # if path.exists(split_folder) and path.isdir(split_folder):
        #     working_folder = split_folder
        #     print(
        #         f"Warning: Using split files causes a fixed delay for every split file. This is a known bug. Use the single file method instead"
        #     )
        # else:
        working_folder = full_folder
        working_folders.append(working_folder)
    return working_folders


def cleanup():
    temp_files = grab_files(working_folder, ['*.filtered.*'])
    remove_files(temp_files)


def run(working_folder, use_transcript_cache, use_filtered_cache, model):
    if not use_transcript_cache:
        prepped_audio = prep_audio(working_folder, use_filtered_cache)
        # audio_path_dicts = [
        audio_path_dict = {
            'working_folder': working_folder,
            'audio_file': prepped_audio,
        }  # for af in prepped_audio
        # ]
        # for audio_path_dict in tqdm(audio_path_dicts):
        generate_transcript_from_audio_wrapper(audio_path_dict, model)

    if not use_filtered_cache:
        cleanup()


def align_stable_transcript(working_folder, content_name):
    split_script = grab_files(working_folder, ['*.split.txt'])
    final = path.join(working_folder, f'{content_name}.srt')
    align_text(model, working_folder, split_script[0], final)
    remove_files(split_script)


def align_transcript(working_folder, content_name):
    split_script = grab_files(working_folder, ['*.split.txt'])
    subs_file = grab_files(working_folder, [f'{content_name}.vtt'])
    out = path.join(working_folder, 'matched.vtt')
    align.run(split_script[0], subs_file[0], out)
    final = path.join(working_folder, f'{content_name}.srt')
    convert_sub_format(out, final)
    remove_files(split_script + subs_file + [out])


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Match audio to a transcript")
    # parser.add_argument(
    #     "-d",
    #     "--dirs",
    #     dest="dirs",
    #     default=None,
    #     required=True,
    #     type=str,
    #     nargs="+",
    #     help="List of folders to run generate subs for",
    # )

    # parser.add_argument(
    #     "--use-filtered-cache",
    #     help="Uses cached filtered files and skips cleanup. Skips making the filtered audio files.",
    #     action="store_true",
    #     default=False,
    # )
    # parser.add_argument(
    #     "--use-transcript-cache",
    #     help="Uses cached transcript files and skips cleanup. Skips the whisper step.",
    #     action="store_true",
    #     default=False,
    # )
    # parser.add_argument(
    #     "--use-stable-ts-align",
    #     help="Uses experimental alignment. Twice as fast, but sometimes misses on accuracy for long works",
    #     action="store_true",
    #     default=False,
    # )
    # args = parser.parse_args()
    # working_folders = get_working_folders(args.dirs)
    if args.use_stable_ts_align:
        model = get_model('large-v2')
    else:
        model = get_model('tiny')
    successes = []
    failures = []
    for working_folder in working_folders:
        try:
            print(f'Working on {working_folder}')
            split_txt(working_folder)
            if args.use_stable_ts_align:
                align_stable_transcript(
                    working_folder, get_content_name(working_folder)
                )
            else:
                run(
                    working_folder,
                    args.use_transcript_cache,
                    args.use_filtered_cache,
                    model,
                )
                align_transcript(working_folder, get_content_name(working_folder))
            successes.append(working_folder)
        except Exception as err:
            tb = traceback.format_exc()
            pprint(tb)
            failures.append({'working_folder': working_folder, 'err': err})

    if len(successes) > 0:
        print(f'The following {len(successes)} succeeded:')
        pprint(successes)
    if len(failures) > 0:
        print(f'The following {len(failures)} failed:')
        for failure in failures:
            pprint(failure['working_folder'])
            pprint(failure['err'])
