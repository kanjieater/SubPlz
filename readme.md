# SubPlz🫴: Get Incredibly Accurate Subs for Anything


https://user-images.githubusercontent.com/32607317/219973521-5a5c2bf2-4df1-422b-874c-5731b4261736.mp4

Generate, sync, and manage subtitle files for any media type with special features for working with audiobooks and videos.

## Features
- **Sync Existing Subtitles**: Multiple options to automate synchronizing your subtitles with various techniques in bulk
- **Accurately Subtitle Narrated Media**: Leverage the original source text of an ebook to provide highly accurate subtitles
- **Create New Subtitles**: Generate new subtitles from the audio of a media file
- **File Management**: Automatically organize and rename your subtitles to match your media files
- **Provide Multiple Video Subtitle Options**: Combines other features to allow you to have multiple alignment & generation subs available to you, for easy auto-selecting of your preferred version  (dependent on video player support)


For a glimpse of some of the technologies & techniques we're using depending on the arguments, here's a short list: 
- **faster-whisper**: for using AI to generate subtitles fast
- **stable-ts**: for more accurate Whisper time stamps
- **Silero VAD**: for Voice Activity Detection
- **Alass**: for language agnostic subtitle alignment
- **Needleman–Wunsch algorithm**: for alignment to original source text

Currently I am only developing this tool for Japanese use, though rumor has it, the `lang` flag can be used for other languages too.

It requires a modern GPU with decent VRAM, CPU, and RAM. There's also a community built Google Colab notebook available on discord.

Current State of SubPlz alignments:
- The subtitle timings will be 98% accurate for most intended use cases
- The timings will be mostly accurate, but may come late or leave early
- Occasionally, non-spoken things like character names at the start of a line or sound effects in subtitles will be combined with other lines
- Theme songs might throw subs off time, but will auto-recover almost immediately after
- Known Issues: RAM usage. 5+ hr audiobooks can take more than 12 GB of RAM. I can't run a 19 hr one with 48GB of RAM. The current work around is to use an epub + chaptered m4b audiobook. Then we can automatically split the ebook text and audiobook chapters to sync in smaller chunks accurately. Alternatively you could use multiple text files and mp3 files to achieve a similar result.

How does this compare to Alass for video subtitles?
- Alass is usually either 100% right once it get's lined up - or way off and unusable. In contrast, SubPlz is probably right 98% but may have a few of the above issues. Ideally you'd have both types of subtitle available and could switch from an Alass version to a SubPlz version if need be. Alternatively, since SubPlz is consistent, you could just default to always using it, if you find it to be "good enough". [See Generating All Subtitle Algorithms in Batch](#Generating-All-Subtitle-Algorithms-in-Batch)

Current State of Alass alignments:
- Alass tends to struggle on large commercial gaps often found in Japanese TV subs like AT-X
- Once Alass get's thrown off it may stay misaligned for the rest of the episode
- SubPlz can extract the first subtitle embedded, but doesn't try to get the longest one. Sometimes you'll get Informational or Commentary subs which can't be used for alignments of the spoken dialog. We may be able to automate this in the future

Support for this tool can be found [on KanjiEater's thread](https://discord.com/channels/617136488840429598/1076966013268148314)  [on The Moe Way Discord](https://learnjapanese.moe/join/)

Support for any tool by KanjiEater can be found [on KanjiEater's Discord](https://discord.com/invite/agbwB4p)

# Support

The Deep Weeb Podcast - Sub Please 😉

<a href="https://youtube.com/c/kanjieater"><img src="https://github.com/gauravghongde/social-icons/blob/master/SVG/Color/Youtube.svg" height="50px" title="YouTube"></a>
<a href="https://tr.ee/-TOCGozNUI" title="Twitter"><img src="https://github.com/gauravghongde/social-icons/blob/master/SVG/Color/Twitter.svg" height="50px"></a>
<a href="https://tr.ee/FlmKJAest5" title="Discord"><img src="https://github.com/gauravghongde/social-icons/blob/master/SVG/Color/Discord.svg" height="50px"></a>

If you find my tools useful please consider supporting via Patreon. I have spent countless hours to make these useful for not only myself but other's as well and am now offering them completely 100% free.

<a href="https://www.patreon.com/kanjieater" rel="nofollow"><img src="https://i.imgur.com/VCTLqLj.png"></a>

If you can't contribute monetarily please consider following on a social platform, joining the discord & sharing a kind message or sharing this with a friend.


# How to Use

## Quick Guide
### Sync
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
3. Run `subplz sync -d "<full folder path>"` using something like `/mnt/d/sync/Harry Potter 1`
4. From there, use a [texthooker](https://github.com/Renji-XD/texthooker-ui) with something like [mpv_websocket](https://github.com/kuroahna/mpv_websocket) and enjoy Immersion Reading.

### Gen
1. Put an audio/video file and a text file in a folder.
   1. Audio / Video files: `m4b`, `mkv` or any other audio/video file
```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising EP00.mkv
   └── NeoOtaku Uprising EP01.avi
```
1. List the directories you want to run this on. The `-d` parameter can multiple files to process like: `subplz gen -d "/mnt/d/NeoOtaku Uprising The Anime" --model large-v3`
2. Run `subplz gen -d "<full folder path>" --model large-v3` using something like `/mnt/d/sync/NeoOtaku Uprising The Anime`. Large models are highly recommended for `gen` (unlike `sync`)
3. From there, use a video player like MPV or Plex. You can also use `--lang-ext az` to set a language you wouldn't otherwise need as a designated "AI subtitle", and use it as a fallback when sync doesn't work or you don't have existing subtitles already

### Alass
1. Put a video(s) with embdedded subs & sub file(s) that need alignment in a folder.
   1. Video & Sub files: `m4b`, `mkv` or any other audio/video file
   2. If you don't have embdedded subs, you'll need it to have a `*.en.srt` extension in the folder
   3. Consider using Rename to get your files ready for alass
```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.srt
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.en.srt
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.srt
```
1. List the directories you want to run this on. The `-d` parameter can multiple files to process like: `subplz sync -d "/mnt/d/NeoOtaku Uprising The Anime" --alass --lang-ext "ja" --lang-ext-original "en"`
   1. You could also add `--lang-ext-incorrect "ja"` if you had `NeoOtaku Uprising With No Embedded Eng Subs EP01.ja.srt` instead of `NeoOtaku Uprising With No Embedded Eng Subs EP01.srt`. This is the incorrect timed sub from Alass
2. From there, SubPlz will extract the first available subs from videos writing them with `--lang-ext-original` extension, make sure the subtitles are sanitized, convert subs to the same format for Alass if need be, and align the incorrect timings with the timed subs to give you correctly timed subs like below:
```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.en.srt (embedded)
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.ja.srt (timed)
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.srt (original/incorrect timings)
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.en.srt (no change)
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.ja.srt (timed)
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.srt (original/incorrect timings)
```

### Rename (When Sub Names Don't Match)
1. Put a video(s) & sub file(s) that need alignment in a folder.
   1. Video & Sub files: `m4b`, `mkv` or any other audio/video file
```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── 1.srt
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   └── 2.ass
```

1. Run `subplz rename -d "/mnt/v/Videos/J-Anime Shows/NeoOtaku Uprising The Anime/" --lang-ext "ab" --dry-run` to see what the rename would be
2. If the renames look right, run it again without the `--dry-run` flag: `subplz rename -d "/mnt/v/Videos/J-Anime Shows/NeoOtaku Uprising The Anime/" --lang-ext ab --dry-run`
```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.ab.srt
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.ab.ass
```

### Rename a Language Extension (When Sub Names Match)
1. Put a video(s) & sub file(s) that match names in a folder.
   1. Video & Sub files: `m4b`, `mkv` or any other audio/video file
   2. The names must be exactly the same besides language extension & hearing impaired code
```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.ab.cc.srt (notice the hearing impaired cc)
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.ab.srt
```

1. Run `subplz rename -d "/mnt/v/Videos/J-Anime Shows/NeoOtaku Uprising The Anime/" --lang-ext jp --lang-ext-original ab` to get:
```bash
/sync/
└── /NeoOtaku Uprising The Anime/
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.mkv
   ├── NeoOtaku Uprising With Embedded Eng Subs EP00.jp.srt (notice the removed cc)
   ├── NeoOtaku Uprising With No Embedded Eng Subs EP01.mkv
   └── NeoOtaku Uprising With No Embedded Eng Subs EP01.jp.srt
```


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
   -v /mnt/d/sync/変な家/:/sync \
   -v /mnt/d/SyncCache:/app/SyncCache \
   kanjieater/subplz:latest \
   sync -d "/sync/"
   ```
   a. Optional: `--gpus all` will allow you to run with GPU. If this doesn't work make sure you've enabled your GPU in docker (outside the scope of this project)

   b. `-v <your folder path>:/sync` ex: `-v /mnt/d/sync:/sync` This is where your files that you want to sync are at. The part to the left of the `:` if your machine, the part to the right is what the app will see as the folder name.

   c. The SyncCache part is the same thing as the folder syncing. This is just mapping where things are locally to your machine. As long as the app can find the SyncCache folder, it will be able to resync things much faster.

   d. `<command> <params>` ex: `sync -d /sync/`, this runs a `subplz <command> <params>` as you would outside of docker

### Running from Docker: Batch
1. `➜ docker run --entrypoint ./helpers/subplz.sh -it --rm --name subplz --gpus all -v "/mnt/v/Videos/J-Anime Shows/Under Ninja/Season 01":/sync -v /home/ke/code/subplz/SyncCache:/app/SyncCache kanjieater/subplz:latest /sync/`


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


# Generating All Subtitle Algorithms in Batch
Let's say you want to automate getting the best subs for every piece of media in your library. SubPlz takes advantage of how well video players integrate with language codes by overriding them to map them to algorithms, instead of different languages. This makes it so you can quickly switch between a sub on the fly while watching content, and easily update your preferred option for a series later on if your default doesn't work.

Just run `./helpers/subplz.sh` with a sub like `sub1.ja.srt` and `video1.mkv` and it will genearate the following:
| Algorithm    | Default Language Code | Mnemonic | Description |
| -------- | ------- | -------- | ------- |
| Bazarr  | ab    | B for Bazarr | Default potentially untimed subs in target language|
| Alass | as     | S for Ala_ss_ | Subs that have been aligned using `en` & `ab` subs via Alass|
| SubPlz    | ak    | K for KanjiEater | Generated alignment from AI with the `ab` subs text |
| FasterWhisper    | az    | Z for the last option | Generated purely based on audio. Surprisingly accurate but not perfect. |
| Original    | en    | Animes subs tend to be in EN | This would be the original timings used for Alass, and what would be extracted from you videos automatically|
| Preferred    | ja    | Your target language | This is a copy of one of the other options, named with your target language so it plays this by default |

# Anki Support

- Generates subs2srs style deck
- Imports the deck into Anki automatically

The Anki support currently takes your m4b file in `<full_folder_path>` named `<name>.m4b`, where `<name>` is the name of the media, and it outputs srs audio and a TSV file that can is sent via AnkiConnect to Anki. This is useful for searching across [GoldenDict](https://www.youtube.com/playlist?list=PLV9y64Yrq5i-1ztReLQQ2oyg43uoeyri-) to find sentences that use a word, or to merge automatically with custom scripts (more releases to support this coming hopefully).


1. Install ankiconnect add-on to Anki.
2. I recommend using `ANKICONNECT` as an environment variable. Set `export ANKICONNECT=localhost:8755` or `export ANKICONNECT="$(hostname).local:8765"` in your `~/.zshrc` or bashrc & activate it.
3. Just like the line above, Set `ANKI_MEDIA_DIR` to your anki profile's media path: `export ANKI_MEDIA_DIR="/mnt/f/Anki2/KanjiEater/collection.media/"`. You need to change this path.
4. Make sure you are in the project directory `cd ./AudiobookTextSync`
5. Install the main project `pip install .` (only needs to be done once)
6. Install `pip install .[anki]` (only needs to be done once)
7. Copy the file from the project in `./anki_importer/mapping.template.json` to `./anki_importer/mapping.json`. `mapping.json` is your personal configuration file that you can and should modify to set the mapping of fields that you want populated.
My actual config looks like this:
```json
{
  "deckName": "!優先::Y メディア::本",
  "modelName": "JapaneseNote",
  "fields": {
    "Audio": 3,
    "Expression": 1,
    "Vocab": ""
  },
  "options": {
    "allowDuplicate": true
  },
  "tags": [
    "mmi",
    "suspendMe"
  ]
}
```
The number next to the Expression and Audio maps to the fields like so
```
1: Text of subtitle: `パルスに援軍を求めてきたのである。`
2: Timestamps of sub: `90492-92868`
3: Sound file: `[sound:アルスラーン戦記9　旌旗流転_90492-92868.mp3]`
4: Image (not very really useful for audiobooks): <img src='アルスラーン戦記9　旌旗流転_90492-92868.jpg'>
5: Sub file name: アルスラーン戦記9　旌旗流転.m4b,アルスラーン戦記9　旌旗流転.srt
```
Notice you can also set fields and tags manually. You can set multiple tags. Or like in my example, you can set `Vocab` to be empty, even though it's my first field in Anki.
8. Run the command below


Command:
`./anki.sh "<full_folder_path>"`

Example:
`./anki.sh "/mnt/d/sync/kokoro/"`


# FAQ
## Can I run this with multiple Audio files and _One_ script?
It's not recommended. You will have a bad time.

If your audiobook is huge (eg 38 hours long & 31 audio files), then break up each section into an m4b or audio file with a text file for it: one text file per one audio file. This will work fine.

But it _can_ work in very specific circumstances. The exception to the Sort Order rule, is if we find one transcript and multiple audio files. We'll assume that's something like a bunch of `mp3`s or other audio files that you want to sync to a single transcript like an `epub`. This only works if the `epub` chapters and the `mp3` match. `Txt ` files don't work very well for this case currently. I still don't recommend it.

## How do I get a bunch of MP3's into one file then?
Please use m4b for audiobooks. I know you may have gotten them in mp3 and it's an extra step, but it's _the_ audiobook format.

I've heard of people using https://github.com/yermak/AudioBookConverter

Personally, I use the docker image for [`m4b-tool`](https://github.com/sandreas/m4b-tool#installation). If you go down this route, make sure you use the docker version of m4b-tool as the improved codecs are included in it. I tested m4b-tool without the docker image and noticed a huge drop in sound quality without them. When lossy formats like mp3 are transcoded they lose quality so it's important to use the docker image to retain the best quality. I use the `helpers/merge2.sh` to merge audiobooks together in batch with this method.

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

The GOAT delivers again; The best Japanese reading experience ttu-reader paired with SubPlz subs
- https://github.com/Renji-XD/ttu-whispersync
- Demo: https://x.com/kanjieater/status/1834309526129930433

A cool tool to turn these audiobook subs into Visual Novels
- https://github.com/asayake-b5/audiobooksync2renpy
