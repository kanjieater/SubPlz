import os
import ffmpeg
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import mimetypes
import pycountry

@dataclass(eq=True, frozen=True)
class AudioStream:
    stream: ffmpeg.Stream
    args: map
    duration: float
    title: str
    id: int

    def audio(self):
        try:
            data, _ = self.stream.output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k', **self.args).run(quiet=True, input='')
            return np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
        except ffmpeg.Error as e:
            raise Exception(e.stderr.decode('utf8'))

@dataclass(eq=True, frozen=True)
class AudioFile:
    path: Path
    title: str
    duration: float
    chapters: list

    def audio(self):
        return np.concatenate([c.audio() for c in self.chapters])

    @classmethod
    def from_file(cls, path, track=None, whole=False):
        if not path.is_file(): raise FileNotFoundError(f"file {str(path)} is not a file")

        if track is not None:
            langcode = pycountry.languages.get(alpha_2=track).alpha_3
            args = {'map': f'0:m:language:{langcode}?'}
        else:
            args = {}

        try:
            info = ffmpeg.probe(path, show_chapters=None)
        except ffmpeg.Error as e:
            raise Exception(e.stderr.decode('utf8'))

        ftitle = info.get('format', {}).get('tags', {}).get('title', path.name)
        fduration = info['duration'] if 'duration' in info else info['format']['duration'] if 'duration' in info['format'] else None
        if fduration is None:
            raise Exception(f"Couldn't determine duration {path.name}")

        if whole or 'chapters' not in info or len(info['chapters']) < 1:
            chapters = chapters=[AudioStream(stream=ffmpeg.input(path), duration=float(fduration), title=ftitle, id=-1, args=args)]
        else:
            chapters = [AudioStream(stream=ffmpeg.input(path, ss=float(chapter['start_time']), to=float(chapter['end_time'])),
                                    duration=float(chapter['end_time']) - float(chapter['start_time']),
                                    title=chapter.get('tags', {}).get('title', str(i)),
                                    id=chapter['id'],
                                    args=args)
                        for i, chapter in enumerate(info['chapters'])]
        return cls(title=ftitle, path=path, duration=float(fduration), chapters=chapters)

    @classmethod
    def from_dir(cls, path, track=None, whole=False):
        if not path.exists(): raise FileNotFoundError(f"file {str(path)} does not exist")
        if path.is_file():
            yield cls.from_file(path, track, whole)
            return
        mt = {'video', 'audio'}
        for root, _, files in os.walk(str(path)): # TODO path.walk is python3.12
            for f in files:
                p = Path(root) / f
                t, enc = mimetypes.guess_type(p)
                if t.split('/', 1)[0] in mt:
                    yield cls.from_file(p, track, whole)

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
