import argparse
import sys, os
import stable_whisper
import ffmpeg
import multiprocessing
# from subsai import SubsAI
import traceback
from os import path
from pathlib import Path
from datetime import datetime, timedelta
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from pprint import pprint
from utils import read_vtt, write_sub, grab_files
# from split_sentences import split_sentences
from run import get_working_folders, generate_transcript_from_audio, get_model


# SUPPORTED_FORMATS = ["*.mp3", "*.m4b", "*.mp4"]
# extensions = [
#   '3gp', 'aac', 'amr', 'amv', 'ape', 'apng', 'asf', 'ass', 'au', 'av1', 'avi', 'bethsoftvid', 'bfi', 'bfstm', 'bmp_pipe', 'c93', 'caf', 'cavsvideo', 'cdg', 'cdxl', 'chromaprint', 'codec2', 'concat', 'dash', 'daud', 'dcstr', 'dirac', 'dnxhd', 'dpx_pipe', 'dsf', 'dsicin', 'dss', 'dts', 'dtshd', 'dv', 'dvbsub', 'dvbtxt', 'dvd', 'dxa', 'ea', 'ea_cdata', 'eac3', 'f32be', 'f32le', 'f4v', 'f64be', 'f64le', 'fbdev', 'film_cpk', 'filmstrip', 'fits', 'flac', 'flv', 'framecrc', 'framehash', 'framemd5', 'fsb', 'g722', 'g723_1', 'g726', 'g726le', 'g729', 'gif', 'gif_pipe', 'gsm', 'gxf', 'h261', 'h263', 'h264', 'hca', 'hcom', 'hevc', 'hls', 'ico', 'idcin', 'iff', 'ifv', 'ilbc', 'image2', 'image2pipe', 'ipmovie', 'ipod', 'ipu', 'ismv', 'iss', 'ivf', 'ivr', 'jpeg_pipe', 'jpegls_pipe', 'jv', 'kmsgrab', 'kvag', 'latm', 'lavfi', 'libcdio', 'libdc1394', 'libgme', 'libopenmpt', 'live_flv', 'lmlm4', 'loas', 'lrc', 'lvf', 'lxf', 'm4v', 'matroska', 'mca', 'mcc', 'mgsts', 'microdvd', 'mjpeg', 'mjpeg_2000', 'mkvtimestamp_v2', 'mlp', 'mlv', 'mm', 'mmf', 'mods', 'moflex', 'mov', 'mp2', 'mp3', 'mp4', 'm4v', 'mpeg', 'mpeg1video', 'mpeg2video', 'mpegts', 'mpegtsraw', 'mpegvideo', 'mpjpeg', 'mpl2', 'mpsub', 'msf', 'msnwctcp', 'msp', 'mtaf', 'mtv', 'mulaw', 'musx', 'mv', 'mvi', 'mxf', 'mxf_d10', 'mxf_opatom', 'mxg', 'nc', 'nistsphere', 'nsp', 'nsv', 'null', 'nut', 'nuv', 'obu', 'oga', 'ogg', 'ogv', 'oma', 'openal', 'opus', 'oss', 'paf', 'pam_pipe', 'pbm_pipe', 'pcx_pipe', 'pgm_pipe', 'pgmyuv_pipe', 'pgx_pipe', 'photocd_pipe', 'pictor_pipe', 'pjs', 'pmp', 'png_pipe', 'pp_bnk', 'ppm_pipe', 'psd_pipe', 'psp', 'psxstr', 'pulse', 'pva', 'pvf', 'qcp', 'qdraw_pipe', 'r3d', 'rawvideo', 'realtext', 'redspark', 'rl2', 'rm', 'roq', 'rpl', 'rsd', 'rso', 'rtp', 'rtp_mpegts', 'rtsp', 's16be', 's16le', 's24be', 's24le', 's32be', 's32le', 's8', 'sami', 'sap', 'sbc', 'sbg', 'scc', 'sdl', 'sdl2', 'sdp', 'sdr2', 'sds', 'sdx', 'segment', 'ser', 'sgi_pipe', 'shn', 'siff', 'simbiosis_imx', 'singlejpeg', 'sln', 'smjpeg', 'smk', 'smoothstreaming', 'smush', 'sndio', 'sol', 'sox', 'spdif', 'spx', 'srt', 'stl', 'stream_segment', 'streamhash', 'subviewer', 'subviewer1', 'sunrast_pipe', 'sup', 'svag', 'svcd', 'svg_pipe', 'svs', 'swf', 'tak', 'tedcaptions', 'thp', 'tiertexseq', 'tiff_pipe', 'tmv', 'truehd', 'tta', 'ttml', 'tty', 'txd', 'ty', 'u16be', 'u16le', 'u24be', 'u24le', 'u32be', 'u32le', 'u8', 'uncodedframecrc', 'v210', 'v210x', 'vag', 'vc1', 'vc1test', 'vcd', 'vidc', 'vivo', 'vob', 'vobsub', 'voc', 'vpk', 'vqf', 'w64', 'wav', 'wc3movie', 'webm', 'webm_chunk', 'webm_dash_manifest', 'webp', 'webp_pipe', 'webvtt', 'wsaud', 'wsd', 'wsvqa', 'wtv', 'wv', 'wve', 'xa', 'xbin', 'xbm_pipe', 'xmv', 'xpm_pipe', 'xv', 'xvag', 'xwd_pipe', 'xwma', 'yop', 'yuv4mpegpipe'
# ]
audio_formats = ['aac', 'ac3', 'alac', 'ape', 'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'm4b']
video_formats = ['3g2', '3gp', 'avi', 'flv', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'webm']

SUPPORTED_FORMATS = ['*.' + extension for extension in video_formats + audio_formats]

def generate_subs(files, model):
    for file in files:
        try:
            ext = 'srt'
            sub_path = str(Path(file).with_suffix(''))
            generate_transcript_from_audio(file, sub_path, model, ext)
        except Exception as err:
            tb = traceback.format_exc()
            pprint(tb)
            failures.append({"working_folder": file, "err": err})
    return failures
# def generate_subs(files):
#     for file in files:
#         try:
#             ext = '.srt'
#             subs_ai = SubsAI()
#             model = subs_ai.create_model('guillaumekln/faster-whisper', {'model_type': 'base'})
#             subs = subs_ai.transcribe(file, model)
#             sub_path = Path(file).with_suffix(ext)

#             subs.save(sub_path)
#         except Exception as err:
#             pprint(err)
#             failures.append({"working_folder": file, "err": err})
#     return failures



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio for files")
    parser.add_argument(
        "-d",
        "--dirs",
        dest="dirs",
        default=None,
        required=True,
        type=str,
        nargs="+",
        help="List of folders to run generate subs for",
    )

    parser.add_argument(
        "--use-filtered-cache",
        help="Uses cached filtered files and skips cleanup. Skips making the filtered audio files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use-transcript-cache",
        help="Uses cached transcript files and skips cleanup. Skips the whisper step.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    working_folders = get_working_folders(args.dirs)
    global model
    model = False  # global model preserved between files
    model = get_model('large-v2')
    successes = []
    failures = []
    for working_folder in working_folders:
        # try:
        print(working_folder)
        audio_files = grab_files(working_folder, SUPPORTED_FORMATS)
        failures.extend(generate_subs(audio_files, model))
        successes.append(working_folder)
        # except Exception as err:
        #     pprint(err)
        #     failures.append({"working_folder": working_folder, "err": err})

    if len(successes) > 0:
        print(f"The following {len(successes)} succeeded:")
        pprint(successes)
    if len(failures) > 0:
        print(f"The following {len(failures)} failed:")
        for failure in failures:
            pprint(failure['working_folder'])
            pprint(failure['err'])
