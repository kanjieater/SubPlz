# AudiobookTextSync


https://user-images.githubusercontent.com/32607317/219973521-5a5c2bf2-4df1-422b-874c-5731b4261736.mp4





This tool allows you to use AI models to generate subtitles from only audio, then match the subtitles to an accurate text, like a book. 

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

1. Setup the folder. Create a folder to hold a single media file (like an audiobook). Name it whatever you name your media file, eg `Arslan Senki 7`, this is what should go anywhere you see me write `<name>`
2. Put the script in place: `./<name>/script.txt`. Everything in this file will show up in your subtitles. So it's important you trim out excess (table of contents, character bios that aren't in the audiobook etc)
3. You need _both_ the audiobook as a full m4b (technically other formats would work), AND the split parts. As long as you have one, you can easily get the other. You could technically only use the full single file, but you will most-likely run out of ram for longer works. See [Whisper ~13GB Memory Usage Issue for 19hr Audiobook](https://github.com/jianfch/stable-ts/issues/79). By using small splits, we can have more confidence the Speech To Text analysis won't get killed by an Out Of Memory error.
4. Split files should be `./<name>/<name>_splitted/`.If you have the full audiobook as a m4b, you can split it into chapters using `./split.sh "<full folder path>"`. eg `./split.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"`
5. Single media file should be in `./<name>/<name>.m4b`. If you have the split audiobook as m4b,mp3, or mp4's you can run `./combine.sh "<full folder path>"`,
 eg `./combine.sh "/mnt/d/Editing/Audiobooks/ｍｅｄｉｕｍ霊媒探偵城塚翡翠"`
6. If you have the `script.txt` and `./<name>/<name>_splitted/`, you can now run the GPU intense, time intense, and occasionally CPU intense script part. `./split_run.sh "<full folder path>"` eg `./split_run.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"`. This runs each split file individually to get a word level transcript. It then creates a sub format that can be matched to the `script.txt`. Each word level subtitle is combined into a phrase level, and your result should be a `<name>.srt` file that can be watched with `mpv`, showing audio in time with the full book as a subtitle. From there use a texthooker and enjoy.


# Single File

You can also run for a single file. Beware if it's over 1GB/19hr you need as much as 23GB of RAM available.
`./run.sh "<full folder path>"` eg `./run.sh "$(wslpath -a "D:\Editing\Audiobooks\かがみの孤城\\")"`


# Get a single transcript from split files
`./split_run.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"`

# Split m4b by chapter
`./split.sh "/mnt/d/Editing/Audiobooks/かがみの孤城/"`

# Combine split files into a single m4b
`./combine.sh "/mnt/d/Editing/Audiobooks/ｍｅｄｉｕｍ霊媒探偵城塚翡翠"`


# What does "bad" look like using the stable-ts library?

At this point I would recommend reading from the texthooker instead of a sub. (CTRL+SHIFT+RIGHT in mpv to set offset as the next sub). Then you can see the next line coming in the texthooker, and not be distracted by subtitle jumps.

https://user-images.githubusercontent.com/32607317/219973663-7fcac162-b162-4a02-839c-0be2385f6166.mp4

