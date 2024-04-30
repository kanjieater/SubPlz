# AudiobookTextSync
https://user-images.githubusercontent.com/32607317/219973521-5a5c2bf2-4df1-422b-874c-5731b4261736.mp4


## Usage
```
usage: main.py [-h] --audio AUDIO [AUDIO ...] --text TEXT [TEXT ...] [--model MODEL] [--language LANGUAGE] [--progress | --no-progress]
               [--overwrite | --no-overwrite] [--use-cache | --no-use-cache] [--cache-dir CACHE_DIR] [--overwrite-cache | --no-overwrite-cache]
               [--threads THREADS] [--device DEVICE] [--dynamic-quantization | --no-dynamic-quantization | --dq | --no-dq] [--quantize | --no-quantize]
               [--faster-whisper | --no-faster-whisper] [--fast-decoder | --no-fast-decoder] [--fast-decoder-overlap FAST_DECODER_OVERLAP]
               [--fast-decoder-batches FAST_DECODER_BATCHES] [--ignore-tags IGNORE_TAGS [IGNORE_TAGS ...]]
               [--prefix-chapter-name | --no-prefix-chapter-name] [--follow-links | --no-follow-links] [--beam_size BEAM_SIZE] [--patience PATIENCE]
               [--length_penalty LENGTH_PENALTY] [--suppress_tokens SUPPRESS_TOKENS] [--initial_prompt INITIAL_PROMPT]
               [--condition_on_previous_text | --no-condition_on_previous_text] [--temperature TEMPERATURE]
               [--temperature_increment_on_fallback TEMPERATURE_INCREMENT_ON_FALLBACK] [--compression_ratio_threshold COMPRESSION_RATIO_THRESHOLD]
               [--logprob_threshold LOGPROB_THRESHOLD] [--no_speech_threshold NO_SPEECH_THRESHOLD] [--word_timestamps | --no-word_timestamps]
               [--prepend_punctuations PREPEND_PUNCTUATIONS] [--append_punctuations APPEND_PUNCTUATIONS] [--nopend_punctuations NOPEND_PUNCTUATIONS]
               [--highlight_words | --no-highlight_words] [--max_line_width MAX_LINE_WIDTH] [--max_line_count MAX_LINE_COUNT]
               [--max_words_per_line MAX_WORDS_PER_LINE] [--output-dir OUTPUT_DIR] [--output-format OUTPUT_FORMAT] [--local-only | --no-local-only]
```

## Install
AudiobookTextSync needs ffmpeg to be installed.

```
git clone https://github.com/ym1234/AudiobookTextSync.git
pip install -r requirements.txt
```

## Credits
- KanjiEater
