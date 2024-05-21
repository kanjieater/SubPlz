import os
import ffmpeg
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import mimetypes

@dataclass(eq=True, frozen=True)
class AudioStream:
    stream: ffmpeg.Stream
    duration: float
    title: str
    id: int

    def audio(self):
        data, _ = self.stream.output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k').run(quiet=True, input='')
        return np.frombuffer(data, np.int16).astype(np.float32) / 32768.0

@dataclass(eq=True, frozen=True)
class AudioFile:
    path: Path
    title: str
    duration: float
    chapters: list

    def audio(self):
        return np.concatenate([c.audio() for c in self.chapters])

    @classmethod
    def from_file(cls, path, whole=False):
        if not path.is_file(): raise FileNotFoundError(f"file {str(path)} is not a file")

        try:
            info = ffmpeg.probe(path, show_chapters=None)
        except ffmpeg.Error as e:
            raise Exception(e.stderr.decode('utf8'))

        ftitle, fduration = info.get('format', {}).get('tags', {}).get('title', path.name), float(info['streams'][0]['duration'])
        if whole or 'chapters' not in info or len(info['chapters']) < 1:
            chapters = chapters=[AudioStream(stream=ffmpeg.input(path), duration=fduration, title=ftitle, id=-1)]
        else:
            chapters = [AudioStream(stream=ffmpeg.input(path, ss=float(chapter['start_time']), to=float(chapter['end_time'])),
                                    duration=float(chapter['end_time']) - float(chapter['start_time']),
                                    title=chapter.get('tags', {}).get('title', str(i)),
                                    id=chapter['id'])
                        for i, chapter in enumerate(info['chapters'])]
        return cls(title=ftitle, path=path, duration=fduration, chapters=chapters)

    @classmethod
    def from_dir(cls, path, whole=False):
        if not path.exists(): raise FileNotFoundError(f"file {str(path)} does not exist")
        if path.is_file():
            yield cls.from_file(path, whole)
            return
        mt = {'video', 'audio'}
        for root, _, files in os.walk(str(path)): # TODO path.walk is python3.12
            for f in files:
                p = Path(root) / f
                t, enc = mimetypes.guess_type(p)
                if t.split('/', 1)[0] in mt:
                    yield cls.from_file(p, whole)

@dataclass(eq=True, frozen=True)
class TranscribedAudioStream:
    stream: AudioStream
    language: str
    segments: list

    @classmethod
    def from_map(cls, stream, transcript): return cls(stream=stream, language=transcript['language'], segments=transcript['segments'])

@dataclass(eq=True, frozen=True)
class TranscribedAudioFile:
    file: AudioFile
    chapters: list[TranscribedAudioStream]
