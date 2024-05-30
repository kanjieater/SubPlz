# SubPlz🫴: Get Incredibly Accurate Subs for Anything


https://user-images.githubusercontent.com/32607317/219973521-5a5c2bf2-4df1-422b-874c-5731b4261736.mp4


Generate accurate subtitles from audio, align existing subs to videos, generate your own Kindle's Immersion Reading like audiobook subs.

This tool allows you to use AI models to generate subtitles from only audio, then match the subtitles to an accurate text, like a book. It supports syncronizing existing subs as well. Soon, You can also just generate subtitles for videos with it, without needing any existing subtitles. Currently I am only developing this tool for Japanese use, though rumor has it, the `language` flag can be used for other languages too.

It requires a modern GPU with decent VRAM, CPU, and RAM. There's also a community built Google Colab notebook available on discord.

Current State: 
- The subtitle timings will be 99.99% accurate for most intended use cases.
- The timings will be mostly accurate, but may come late or leave early. 
- Occassionally, the first word of the next line will show up in the next subtitle.
- Occassionally, non-spoken things like sound effects in subtitles will be combined with other lines
- Known Issues: RAM usage. 5+ hr audiobooks can take more than 12 GB of RAM. I can't run a 19 hr one with 48GB of RAM. The current work around is to use an epub + chaptered m4b audiobook. Then we can automatically split the ebook text and audiobook chapters to sync in smaller chunks accurately. Alternatively you could use multiple text files and mp3 files to achieve a similar result.
Accuracy has improved tremendously with the latest updates to the AI tooling used. Sometimes the first few lines will be off slightly, but will quickly autocorrect. If it get's off midway, it autocorrects. Sometimes multiple lines get bundled together making large subtitles, but it's not usually an issue.

Support for this tool can be found [on KanjiEater's thread](https://discord.com/channels/617136488840429598/1076966013268148314)  [on The Moe Way Discord](https://learnjapanese.moe/join/)

Support for any tool by KanjiEater can be found [on KanjiEater's Discord](https://discord.com/invite/agbwB4p)

# Support

The Deep Weeb Podcast - Sub Please 😉

<a href="https://youtube.com/c/kanjieater"><img src="https://github.com/gauravghongde/social-icons/blob/master/SVG/Color/Youtube.svg" height="50px" title="YouTube"></a>
<a href="https://tr.ee/-TOCGozNUI" title="Twitter"><img src="https://github.com/gauravghongde/social-icons/blob/master/SVG/Color/Twitter.svg" height="50px"></a>
<a href="https://tr.ee/FlmKJAest5" title="Discord"><img src="https://github.com/gauravghongde/social-icons/blob/master/SVG/Color/Discord.svg" height="50px"></a>

If you find my tools useful please consider supporting via Patreon. I have spent countless hours to make these useful for not only myself but other's as well and am now offering them completely 100% free.

<a href="https://www.patreon.com/kanjieater" rel="nofollow"><img src="https://i.imgur.com/VCTLqLj.png"></a>

If you can't contribute monetarily please consider following on a social platform, joining the discord and sharing this with a friend.


# How to use

## Quick Guide

1. Put an audio/video file and a text file in a folder.
   1. Audio / Video files: `m4b`, `mkv` or any other audio/video file
   2. Text files: `srt`, `vtt`, `ass`, `txt`, or `epub`
```bash
/sync/
└── /Harry Potter 1/
   ├── Im an audio file.m4b
   └── Harry Potter.epub
└── /Harry Potter 2 The Spooky Sequel/
   ├── Harry Potter 2 The Spooky Sequel.mp3
   └── script.txt
```
2. List the directories you want to run this on. The `-d` parameter can multiple audiobooks to process like: `subplz sync -d "/mnt/d/sync/Harry Potter 1/" "/mnt/d/sync/Harry Potter 2 The Spooky Sequel/"`
3. Run `subplz sync -d "<full folder path>"` like `/mnt/d/sync/Harry Potter 1"`
4. From there, use a [texthooker](https://github.com/Renji-XD/texthooker-ui) with something like [mpv_websocket](https://github.com/kuroahna/mpv_websocket) and enjoy Immersion Reading.


# Install

Currently supports Docker (preferred), Windows, and unix based OS's like Ubuntu 22.04 on WSL2. Primarily supports Japanese, but other languages may work as well with limited dev support.

## Run from Colab
1. Open this [Colab](https://colab.research.google.com/drive/1LOu6tffvYiOqzrSMH6Pe91Eka55uOuT3?usp=sharing)
1. In Google Drive, create a folder named `sync` on the root of MyDrive
1. Upload the audio/video file and supported text to your `sync` folder
1. Open the colab, you can change the last line if you want, like `-d "/content/drive/MyDrive/sync/Harry Potter 1/"` for the quick guide example
1. In the upper menu, click Runtime > run all, give the necessary permissions and wait for it to finish, should take some 30 min for your average book

## Running from Docker

1. Install [Docker](https://docs.docker.com/desktop/install/windows-install/)
2. ```bash
   docker run -it --rm --name subplz \
   -v <full path to up to content folder>:/sync \
   -v <your folder path>:/SyncCache \
   kanjieater/subplz:latest \
   sync -d "/sync/<content folder>/"
   ```

   Example:

   ```bash
   /mnt/d/sync/
            └── /変な家/
                  ├── 変な家.m4b
                  └── 変な家.epub
   ```
   ```bash
   docker run -it --rm --name subplz \
   --gpus all \
   -v /mnt/d/sync:/sync \
   -v /mnt/d/SyncCache:/app/SyncCache \
   kanjieater/subplz:latest \
   sync -d "/sync/変な家/"
   ```
   a. Optional: `--gpus all` will allow you to run with GPU. If this doesn't work make sure you've enabled your GPU in docker (outside the scope of this project)

   b. `-v <your folder path>:/sync` ex: `-v /mnt/d/sync:/sync` This is where your files that you want to sync are at. The part to the left of the `:` if your machine, the part to the right is what the app will see as the folder name.

   c. The SyncCache part is the same thing as the folder syncing. This is just mapping where things are locally to your machine. As long as the app can find the SyncCache folder, it will be able to resync things much faster.

   d. `<command> <params>` ex: `sync -d /sync/変な家/`, this runs a `subplz <command> <params>` as you would outside of docker



## Setup from source

1. Install `ffmpeg` and make it available on the path

1. `git clone https://github.com/kanjieater/SubPlz.git`

2. Use python >= `3.11.2` (latest working version is always specified in `pyproject.toml`)

3. `pip install .`

4. You can get a full list of cli params from `subplz sync -h `
5. If you're using a single file for the entire audiobook with chapters you are good to go. If an file with audio is too long it may use up all of your RAM. You can use the docker image  [`m4b-tool`](https://github.com/sandreas/m4b-tool#installation) to make a chaptered audio file. Trust me, you want the improved codec's that are included in the docker image. I tested both and noticed a huge drop in sound quality without them. When lossy formats like mp3 are transcoded they lose quality so it's important to use the docker image to retain the best quality if you plan to listen to the audio file.


## Note
- This can be GPU intense, RAM intense, and CPU intense script part. `subplz sync -d "<full folder path>"` eg `subplz sync -d "/mnt/d/Editing/Audiobooks/かがみの孤城/"`. This runs each file to get a character level transcript. It then creates a sub format that can be matched to the `script.txt`. Each character level subtitle is merged into a phrase level, and your result should be a `<name>.srt` file. The video or audio file then can be watched with `MPV`, playing audio in time with the subtitle.
- Users with a modern CPU with lots of threads won't notice much of a difference between using CUDA/GPU & CPU

## Sort Order
By default, the `-d` parameter will pick up the supported files in the directory(s) given. Ensure that your OS sorts them in an order that you would want them to be patched together in. Sort them by name, and as long as all of the audio files are in order and the all of the text files are in the same order, they'll be "zipped" up individually with each other.

## Overwrite
By default the tool will overwrite any existing srt named after the audio file's name. If you don't want it to do this you must explicitly tell it not to.

`subplz sync -d "/mnt/v/somefolder" --no-overwrite`

## Only Running for the Files It Needs
For subtitles, SubPlz renames matching sub files to the audio with the `<audiofile>.old.<sub ext>` naming. This ensures that subplz runs once and only once per directory for your content. If you want to rerun the SubPlz syncing, you can use the flag `--rerun` to use the matching `.old` file and ignore all subs that aren't `.old`.

## Respect Transcript Grouping
If you want to allow the tool to break lines up into smaller chunks, you can use this flag. `--no-respect-grouping`

## Tuning Recommendations
For different use cases, different parameters may be optimal.

### For Audiobooks
- Recommended: `subplz sync -d "/mnt/d/sync/Harry Potter"`
- A chapter `m4b` file will allow us to split up the audio and do things in parallel
- There can be slight variations between `epub` and `txt` files, like where full character spaces aren't pickedup in `epub` but are in `txt`. A chaptered `epub` may be faster, but you can have more control over what text gets synced from a `txt` file if you need to manually remove things (but `epub` is still probably the easier option, and very reliable)
- If the audio and the text differ greatly - like full sections of the book are read in different order, you will want to use `--no-respect-grouping` to let the algorithm remove content for you
- The default `--model "tiny"` seems to work well, and is much faster than other models. If your transcript is inaccurate, consider using a larger model to compensate

### For Realigning Subtitles
- Recommended: `subplz sync --model large-v3 -d "/mnt/v/Videos/J-Anime Shows/Sousou no Frieren"`
- Highly recommend running with something like `--model "large-v3"` as subtitles often have sound effects or other things that won't be picked up by transcription models. By using a large model, it will take much longer (a 24 min episode can go from 30 seconds to 4 mins for me), but it will be much more accurate.
- Subs can be cut off in strange ways if you have an unreliable transcript, so you may want to use `--respect-grouping`. If you find your subs frequently have very long subtitle lines, consider using `--no-respect-grouping`

# FAQ
## Can I run this with multiple Audio files and _One_ script?
It's not recommended. You will have a bad time.

If your audiobook is huge (eg 38 hours long & 31 audio files), then break up each section into an m4b or audio file with a text file for it: one text file per one audio file. This will work fine.

But it _can_ work in very specific circumstances. The exception to the Sort Order rule, is if we find one transcript and multiple audio files. We'll assume that's something like a bunch of `mp3`s or other audio files that you want to sync to a single transcript like an `epub`. This only works if the `epub` chapters and the `mp3` match. `Txt ` files don't work very well for this case currently. I still don't recommend it.

## How do I get things into one file then?
Please use m4b for audiobooks. I know you may have gotten them in mp3 and it's an extra step, but it's _the_ audiobook format. 

I've heard of people using https://github.com/yermak/AudioBookConverter

Personally, I use the docker image for [`m4b-tool`](https://github.com/sandreas/m4b-tool#installation). Trust me, you want the improved codec's that are included in the docker image. I tested both and noticed a huge drop in sound quality without them. When lossy formats like mp3 are transcoded they lose quality so it's important to use the docker image to retain the best quality.

Alternatively you could use ChatGPT to help you combine them. Something like this:
```
!for f in "/content/drive/MyDrive/name/成瀬は天下を取りに行く/"*.mp3; do echo "file '$f'" >> mylist.txt; done
!ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp3
```


# Thanks

Besides the other ones already mentioned & installed this project uses other open source projects subs2cia, & anki-csv-importer

https://github.com/gsingh93/anki-csv-importer

https://github.com/kanjieater/subs2cia

https://github.com/ym1234/audiobooktextsync

# Other Cool Projects

A cool tool to turn these audiobook subs into Visual Novels
- https://github.com/asayake-b5/audiobooksync2renpy
