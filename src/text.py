# All of this kind of needs a better design
from pathlib import Path
from bs4 import element
from bs4 import BeautifulSoup
from dataclasses import dataclass
from ebooklib import epub
import os
import urllib

@dataclass(eq=True, frozen=True)
class TextParagraph:
    idx: int
    content: str
    references: list

    def text(self):
        return self.content

@dataclass(eq=True, frozen=True)
class Txt:
    path: Path
    @property
    def chapters(self): return [self]
    @property
    def title(self): return self.path.name

    def text(self, *args, **kwargs):
        return [TextParagraph(idx=i, content=o, references=[])
                for i, v in enumerate(self.path.read_text().split('\n'))
                if (o := v.strip())]

@dataclass(eq=True, frozen=True)
class SubFile(Txt):
    def text(self):
        ext = self.path.suffix
        content = self.path.read_text()
        if ext == 'srt': # Split multiline subtitles? leave them as is?
            return [TextParagraph(idx=i, content=o, references=[]) for i, n in enumerate(content.split('\n\n')) if (o := '\n'.join(n.split('\n')[2:]))]
        elif ext == 'vtt':
            return [TextParagraph(idx=i, content=o, references=[]) for i, n in enumerate(content.split('\n\n')[1:]) if (o := '\n'.join(n.split('\n')[1:]))]
        elif ext == 'ass':
            raise Exception(f'ASS is currently not supported: {self.path.name}')
        else:
            raise Exception(f'Unknown file format: {self.path.name}')

@dataclass(eq=True, frozen=True)
class EpubParagraph:
    chapter: int
    element: element.Tag
    references: list

    def text(self):
        return ''.join(self.element.stripped_strings)

@dataclass(eq=True, frozen=True)
class EpubChapter:
    content: BeautifulSoup
    title: str
    is_linear: bool
    idx: int

    def text(self):
        paragraphs = self.content.find("body").find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"])
        r = []
        for p in paragraphs:
            if 'id' in p.attrs: continue
            r.append(EpubParagraph(chapter=self.idx, element=p, references=[])) # For now
        return r

# TODO: append parent/child titles together?
def flatten(t):
    return [j for i in t for j in flatten(i)] if isinstance(t, (tuple, list)) else [t] if isinstance(t, epub.Link) else []

@dataclass(eq=True, frozen=True)
class Epub:
    epub: epub.EpubBook
    path: Path
    title: str
    chapters: list

    def text(self):
        return [p for c in self.chapters for p in c.text()]

    @classmethod
    def from_file(cls, path):
        file = epub.read_epub(path, {"ignore_ncx": True})

        flat_toc = flatten(file.toc)
        m = {it.id: i for i, e in enumerate(flat_toc) if (it := file.get_item_with_href(urllib.parse.unquote(e.href.split("#")[0])))}
        if len(m) != len(flat_toc):
            print("WARNING: Couldn't fully map toc to chapters, contact the dev, preferably with the epub")

        chapters = []
        prev_title = ''
        for i, v in enumerate(file.spine):
            item = file.get_item_with_id(v[0])
            title = flat_toc[m[v[0]]].title if v[0] in m else ''

            if item.media_type != 'application/xhtml+xml':
                if title: prev_title = title
                continue

            content = BeautifulSoup(item.get_content(), 'html.parser')

            r = content.find('body').find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"])
            # Most of the time chapter names are on images
            idx = 0
            while idx < len(r) and not r[idx].get_text().strip():
                idx += 1
            if idx >= len(r):
                if title: prev_title = title
                continue

            if not title:
                if (t := prev_title.strip()):
                    title = t
                    prev_title = ''
                elif len(t := r[idx].get_text().strip()) < 25:
                    title = t
                else:
                    title = item.get_name()

            chapter = EpubChapter(content=content, title=title, is_linear=v[1], idx=i)
            chapters.append(chapter)
        return cls(epub=file, path=path, title=file.title.strip() or path.name, chapters=chapters)

class TextFile:
    exts: list = set(['txt', 'epub', 'srt', 'vtt'])
    @classmethod
    def from_file(cls, path):
        ext = path.suffix
        if 'txt' == ext:
            return Txt(path)
        elif 'srt' == ext or 'vtt' == ext:
            return SubFile(path)
        elif 'epub' in ext:
            return Epub.from_file(path)
        elif ext in cls.exts:
            raise NotImplementationError(f"filetype {ext} not implemented yet")
        else:
            raise NotImplementationError(f"filetype {ext} not implemented")

    @classmethod
    def from_dir(cls, path):
        if path.is_file():
            yield cls.from_file(path)
            return

        extensions = ['txt' ]
        for root, _, files in os.walk(str(path)): # TODO path.walk is python3.12
            for f in files:
                p = Path(root)/f
                if p.suffix in cls.exts:
                    yield cls.from_file(p)
