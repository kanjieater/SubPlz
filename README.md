# AudiobookTextSync
https://user-images.githubusercontent.com/32607317/219973521-5a5c2bf2-4df1-422b-874c-5731b4261736.mp4


## Usage
```
usage: main.py [-h] --audio-files AUDIO_FILES [AUDIO_FILES ...] --script SCRIPT [--language LANGUAGE] [--model MODEL] [--output-file OUTPUT_FILE]

Match audio to a transcript

options:
  -h, --help            show this help message and exit
  --audio-files AUDIO_FILES [AUDIO_FILES ...]
                        list of audio files to process (in the correct order)
  --script SCRIPT       path to the script file
  --language LANGUAGE   language of the script and audio
  --model MODEL         whisper model to use. can be one of tiny, small, large, huge
  --output-file OUTPUT_FILE
                        name of the output subtitle file
```

## Dependencies
- At least python 3.9
- stable-ts
- ffmpeg (optionally compiled with `--enable-libfdk-aac`, which improves the audio quality by a lot)

## Problems
For some reason currently the the sync part of the algorithm behaves differently than kanjieater's *sigh*

## Credits
- KanjiEater
- Me
