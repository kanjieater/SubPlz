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
    path: str
    title: str

    @property
    def chapters(self): return [self]
    def name(self): return self.title
    def text(self, *args, **kwargs):
        return [TextParagraph(path=self.path, idx=i, content=o, references=[])
                for i, v in enumerate(self.path.read_text().split('\n'))
                if (o := v.strip())]

def flatten(t):
    return [j for i in t for j in flatten(i)] if isinstance(t, (tuple, list)) else [t] if isinstance(t, epub.Link) else []

@dataclass(eq=True, frozen=True)
class EpubParagraph:
    chapter: int
    element: element.Tag
    references: list

    def text(self):
        return ''.join(self.element.stripped_strings)

class EpubChapter:
    content: BeautifulSoup
    title: str
    is_linear: bool
    epub_id: str
    idx: int

    def text(self):
        paragraphs = self.content.find("body").find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"])
        r = []
        for p in paragraphs:
            if 'id' in p.attrs: continue
            r.append(Paragraph(chapter=self.idx, element=p, references=[])) # For now
        return r

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
        flattoc = flatten(file.toc)
        spine, toc = list(file.spine), [file.get_item_with_href(urllib.parse.unquote(x.href.split("#")[0])) for x in flattoc]
        if None in toc:
            print("Couldn't map toc to chapters, contact the dev, preferably with the epub")
            exit(1)
        for i in range(len(spine)):
            c = list(spine[i])
            spine[i] = c
            for i, j in enumerate(toc):
                if c[0] == j.id and flattoc[i].title.strip():
                    c.append(flattoc[i].title)
                    break

        r = []
        for i, v in enumerate(spine):
            content = BeautifulSoup(file.get_item_with_id(v[0]).get_content(), 'html.parser')
            # find+string=not_empty doesn't work for some reason??? wtf
            for t in content.find('body').find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]):
                if t.get_text().strip():
                    v.append(t.get_text())
                    break
            if len(v) == 2:
                v.append(v[0])
            r.append(cls(epub=file, titles=v[2:], idx=i, content=content, is_linear=v[1]))
        return r

@dataclass(eq=True, frozen=True)
class TextFile:
    @classmethod
    def from_file(cls, path):
        if 'txt' in path.name:
            return Txt(path)
        elif 'epub' in f:
            return Epub.from_file(path)

    @classmethod
    def from_dir(cls, path):
        if path.is_file():
            yield cls.from_file(path)
            return

        for root, _, files in os.walk(str(path)): # TODO path.walk is python3.12
            for f in files:
                p = Path(root)/f
                if 'txt' in f or 'epub' in f:
                    yield cls.from_file(p)
