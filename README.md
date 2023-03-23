# AudiobookTextSync
https://user-images.githubusercontent.com/32607317/219973521-5a5c2bf2-4df1-422b-874c-5731b4261736.mp4


## Usage
`usage: main.py [-h] [--language LANGUAGE] [--model MODEL] [--output-file OUTPUT_FILE] --audio-files AUDIO_FILES [AUDIO_FILES ...] --script SCRIPT

Match audio to a transcript

options:
  -h, --help            show this help message and exit
  --language LANGUAGE
  --model MODEL
  --output-file OUTPUT_FILE
  --audio-files AUDIO_FILES [AUDIO_FILES ...]
                        List of audio files to process (in the correct order)
  --script SCRIPT       Path to the script file``
`

## Dependencies
- At least python 3.9
- stable-ts
- ffmpeg (optionally compiled with `--enable-libfdk-aac`, which improves the audio quality by a lot)

## Problems
For some reason currently the the sync part of the algorithm behaves differently than kanjieater's *sigh*

## Credits
- KanjiEater
- Me
