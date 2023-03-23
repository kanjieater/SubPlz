# AudiobookTextSync


https://user-images.githubusercontent.com/32607317/219973521-5a5c2bf2-4df1-422b-874c-5731b4261736.mp4





This tool allows you to use AI models to generate subtitles from only audio, then match the subtitles to an accurate text, like a book. It requires a modern GPU with decent VRAM, CPU, and RAM.

Current State: The transcript will be extremely accurate. The timings will be mostly accurate, but may come late or leave early. The currently used library for generating those offsets is the best I've found so far that works stably, but leaves much to be desired. See the video at the bottom for such an example.

 I'm looking forward to being able to run more accurate models to fix this in the future.

# Install

Currently supports unix based OS's like Ubuntu 20.04 on WSL2.

1. Install `ffmpeg` and make it available on the path

2. Use python `3.9.9`

3. `pip install stable-ts`

4. Be able to run the docker image for [`m4b-tool`](https://github.com/sandreas/m4b-tool#installation). Trust me, you want the improved codec's that are included in the docker image. I tested both and noticed a huge drop in sound quality without them.



# How to use


Primarily I'm using this for syncing audiobooks to their book script. So while you could use this for video files, I'm not doing that just yet. Note: I'm using the term "splitted" because that's what m4b refers to as the split files.

1. `git clone https://github.com/kanjieater/AudiobookTextSync.git`
2. Make sure you run any commands that start with `./` from the project root, eg after you clone you can run `cd ./AudiobookTextSync`
1. Setup the folder. Create a folder to hold a single media file (like an audiobook). Name it whatever you name your media file, eg `Arslan Senki 7`, this is what should go anywhere you see me write `<name>`
2. Get the book script as text from a digital copy. Put the script at: `./<name>/script.txt`. Everything in this file will show up in your subtitles. So it's important you trim out excess (table of contents, character bios that aren't in the audiobook etc)
3. If you have less than ~8GB of free RAM you may need _both_ the audiobook as a full m4b (technically other formats would work), AND the split parts. As long as you have one, you can easily get the other. You could technically only use the full single file, but you may run out of ram for longer works. See [Whisper ~13GB Memory Usage Issue for 19hr Audiobook](https://github.com/jianfch/stable-ts/issues/79). By using small splits, we can have more confidence the Speech To Text analysis won't get killed by an Out Of Memory error. As of this tool using stable-ts 2.0.0, this has been improved significantly. I no longer recommend using split files, as it's more likely to introduce a ~50ms delay in the subtitles. It's shouldn't be an issue if you use MPV and can correct the subtitle timings with a hotkey press though.
4. If you are not using split files you can skip this step. Split files should be `./<name>/<name>_splitted/`.If you have the full audiobook as a m4b, you can split it into chapters using `./split.sh "<full folder path>"`. eg `./split.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"`. If you don't want to use split files, make sure there isn't a folder with that name existing, or the program will automatically look for the split files.
5. Single media file should be in `./<name>/<name>.m4b`. If you have the split audiobook as m4b,mp3, or mp4's you can run `./merge.sh "<full folder path>"`,
 eg `./merge.sh "/mnt/d/Editing/Audiobooks/ｍｅｄｉｕｍ霊媒探偵城塚翡翠"`
6. If you have the `script.txt` and either `./<name>/<name>.m4b` or `./<name>/<name>_splitted/`, you can now run the GPU intense, time intense, and occasionally CPU intense script part. `./run.sh "<full folder path>"` eg `./run.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"`. This runs each split file individually to get a word level transcript. It then creates a sub format that can be matched to the `script.txt`. Each word level subtitle is merged into a phrase level, and your result should be a `<name>.srt` file that can be watched with `MPV`, showing audio in time with the full book as a subtitle. 
7. From there, use a [texthooker](https://github.com/Renji-XD/texthooker-ui) with something like [mpv_websocket](https://github.com/kuroahna/mpv_websocket) and enjoy Immersion Reading.

# Split m4b by chapter
`./split.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"`

# Get a subtitle with synced transcript from split files
`./run.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"`

# Single File

You can also run for a single file. Beware if it's over 1GB/19hr you need as much as 8GB of RAM available.
You need two copies of your file. One in "<full folder path>" and one in `<full folder path>/splitted_<name>`, as described in the How to Use section. The single file will only run if you don't have `<name>_splitted` folder, otherwise we'll assume you want to use the data from there in parts.

`./run.sh "<full folder path>"` eg `./run.sh "$(wslpath -a "D:\Editing\Audiobooks\かがみの孤城\\")"`

# Merge split files into a single m4b
`./merge.sh "/mnt/d/Editing/Audiobooks/ｍｅｄｉｕｍ霊媒探偵城塚翡翠"`


# What does "bad" look like using the stable-ts library?

At this point I would recommend reading from the texthooker instead of a sub. (CTRL+SHIFT+RIGHT in mpv to set offset as the next sub). Then you can see the next line coming in the texthooker, and not be distracted by subtitle jumps.

Update: The timing is much more accurate, but it still makes sense to show what going wrong could look like

https://user-images.githubusercontent.com/32607317/219973663-7fcac162-b162-4a02-839c-0be2385f6166.mp4





# Anki Support

- Generates subs2srs style deck
- Imports the deck into Anki automatically

The Anki support currently takes your m4b file in `<full_folder_path>` named `<name>.m4b`, where `<name>` is the name of the media, and it outputs srs audio and a TSV file that can is sent via AnkiConnect to Anki. This is useful for searching across [GoldenDict](https://www.youtube.com/playlist?list=PLV9y64Yrq5i-1ztReLQQ2oyg43uoeyri-) to find sentences that use a word, or to merge automatically with custom scripts (more releases to support this coming hopefully).


1. Install ankiconnect add-on to Anki.
2. I recommend using `ANKICONNECT` as an environment variable. Set `export ANKICONNECT=localhost:8755` or `export ANKICONNECT="$(hostname).local:8765"` in your `~/.zshrc` or bashrc & activate it.
3. Make sure you are in the project directory `cd ./AudiobookTextSync`
4. Install `pip install ./requirements.txt` (only needs to be done once)
5. Set `ANKI_MEDIA_DIR` to your anki profile's media path: `/mnt/f/Anki2/KanjiEater/collection.media/`
6. Run the command below



Command:
`./anki.sh "<full_folder_path>"`

Example:
`./anki.sh "/mnt/d/sync/kokoro/"`




# WSL2

If you're using WSL2 there a few networking quirks.

1. Enable WSL2 to talk to your Windows machine. https://github.com/microsoft/WSL/issues/4585#issuecomment-610061194
2. Set your `$ANKICONNECT` url to your windows machine url, `export ANKICONNECT="http://$(hostname).local:8765"`. https://github.com/microsoft/WSL/issues/5211
3. Make sure inside of Anki's addon config `"webBindAddress": "0.0.0.0", "webBindPort": "8765"`. `0.0.0.0` binds to all network interfaces, so WSL2 can connect.

# Testing connection to Anki from WSL2

```
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{ "action": "guiBrowse", "version": 6, "params": { "query": "flag:3 is:new -is:suspended -tag:重複 tag:重複3" } }' \
  http://172.18.224.1:8765
```
# Troubleshooting
You might see various issues while trying this out in the early state. Here are some of the pieces at work in sequence:
## Stages
1. Filter down audio to improve future results - slow & probably not heavy cpu or gpu usage. Heavier on cpu
2. split_run & stable-ts: Starts off heavy on CPU & RAM to identify the audio spectrum
3. stable-ts: GPU heavy & requires lots of vRAM depending on the model. This is the part with the long taskbar, where it tries to transcribe a text from the audio. Currently the default is [tiny](https://github.com/openai/whisper#available-models-and-languages). Ironically tiny, does a better job of keeping the phrases short, at the cost of accuracy of transcription, which since we are matching a script, doesn't matter. Also it runs 32x faster than large.
4. Merge vtt's for split subs
5. Split the script
6. match the script to the generated transcription to get good timestamps

# Getting Book Scripts 

This program supports `txt` files. You may need to use an external program like Calibre to convert your `epub` or kindle formats like `azw3` to a `txt` file. 

To convert in Calibre:
1. Right click on the book and convert the individual book (or use the batch option beneath it)
![image](https://user-images.githubusercontent.com/32607317/226463043-f2f89382-a75f-48ea-bb91-00efe0f05893.png)
2. At the top right for output format, select `txt`
![image](https://user-images.githubusercontent.com/32607317/226463797-1c19385d-c6e7-4564-a795-926e04716562.png)
3. Click Find & Replace. If your book has 《》for furigana as some aozora books do (戦場《せんじょう》), then add a regex. If they have rt for furigana use the rt one: 《(.+?)》 or <rt>(.*?)<\/rt>. When you copy the regex into the regex box, don't forget to click the Add button
![image](https://user-images.githubusercontent.com/32607317/226463912-48bcfd57-4935-48fb-af7e-13d2a024cdee.png)
4. You can add multiple regexes to strip any extra content or furigana as need be.
![image](https://user-images.githubusercontent.com/32607317/226464346-a752970e-0f1c-42db-b64d-a3bc6df6ebdd.png)
5. Click ok and convert it & you should now be able to find the file wherever Calibre is saving your books

# Thanks

Besides the other ones already mentioned & installed this project uses other open source projects subs2cia, & anki-csv-importer

https://github.com/gsingh93/anki-csv-importer

https://github.com/kanjieater/subs2cia
