# v1 Readme

# Split m4b by chapter
`./split.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"`

# Get a subtitle with synced transcript from split files
`subplz sync -d "/mnt/d/Editing/Audiobooks/かがみの孤城/"`

`subplz sync -d "<full folder path>"` eg `subplz sync -d "$(wslpath -a "D:\Editing\Audiobooks\かがみの孤城\\")"` or `subplz sync -d "/mnt/d/sync/Harry Potter 1/" "/mnt/d/sync/Harry Potter The Sequel/"`

# Generate subs for a folder of video or audio file
`python gen.py -d "/mnt/u/Videos/J-Shows/MAG Net/"`

# Merge split files into a single m4b
`./merge.sh "/mnt/d/Editing/Audiobooks/ｍｅｄｉｕｍ霊媒探偵城塚翡翠"`

# Merge split files into a single m4b for a library

This assumes you just have mp4's in a folder like `/mnt/d/Editing/Audiobooks/ｍｅｄｉｕｍ霊媒探偵城塚翡翠`. It will run all of the folder's with mp4's and do a check on them after to make sure the chapters line up. Requires `docker` command to be available.

`python ./helpers/merge.py "/mnt/d/Editing/Audiobooks/"`

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

UPDATE: Books now have furigana automatically escaped in txt and epub. You can use calibre though to export them in appropriate formats.

OLD:
This program supports `txt` files. You may need to use an external program like Calibre to convert your kindle formats like `azw3` to a `txt` of `epub` file.

To convert in Calibre:
1. Right click on the book and convert the individual book (or use the batch option beneath it)
![image](https://user-images.githubusercontent.com/32607317/226463043-f2f89382-a75f-48ea-bb91-00efe0f05893.png)
2. At the top right for output format, select `txt`
![image](https://user-images.githubusercontent.com/32607317/226463797-1c19385d-c6e7-4564-a795-926e04716562.png)
3. Click Find & Replace. If your book has 《》for furigana as some aozora books do (戦場《せんじょう》), then add a regex. If they have rt for furigana use the rt one: `《(.+?)》` or `<rt>(.*?)<\/rt>`. When you copy the regex into the regex box, don't forget to click the Add button
![image](https://user-images.githubusercontent.com/32607317/226463912-48bcfd57-4935-48fb-af7e-13d2a024cdee.png)
4. You can add multiple regexes to strip any extra content or furigana as need be.
![image](https://user-images.githubusercontent.com/32607317/226464346-a752970e-0f1c-42db-b64d-a3bc6df6ebdd.png)
5. Click ok and convert it & you should now be able to find the file wherever Calibre is saving your books