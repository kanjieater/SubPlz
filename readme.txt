# Install

Install `ffmpeg` and make it available on the path

Use python `3.9.9`

`pip install -U openai-whisper`

`pip install git+https://github.com/m-bain/whisperx.git`

`pip install stable-ts`




Run ./run.sh "<folder>"

Transcript to match must be in <folder> and named `script.txt`

Audio file must be in <folder> and named audio.mp3

Examples:

#Single File

./run.sh "$(wslpath -a "D:\Editing\Audiobooks\かがみの孤城\2")"


# Get a single transcript from split files
./split_run.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"

# Combine split files into a single m4b
./combine.sh "/mnt/d/Editing/Audiobooks/ｍｅｄｉｕｍ霊媒探偵城塚翡翠"