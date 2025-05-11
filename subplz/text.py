from pathlib import Path
from bs4 import element
from bs4 import BeautifulSoup
from dataclasses import dataclass
from ebooklib import epub
import urllib

import pysbd
from tqdm import tqdm


def fix_end_of_quotes(lines):
    fixed = []
    for i, line in enumerate(lines):
        if i > 0 and line and line[0] in ["」", "’"]:
            fixed[-1] += line[0]
            line = line[1:]
        if line:
            fixed.append(line)
    return fixed


def merge_short_lines_with_quotes(lines):
    fixed = []
    for line in lines:
        if fixed and fixed[-1][0] in ["」", "’"] and len(fixed[-1] + line) <= 1:
            fixed[-1] += line
        else:
            fixed.append(line)
    return fixed


def split_sentences(input_file, output_path, lang):
    # for file_name in input_file:
    with open(input_file, "r", encoding="UTF-8") as file:
        input_lines = file.readlines()

    split_sentences_from_input(input_lines, output_path, lang)


def get_segments(input_lines, lang):
    seg = pysbd.Segmenter(language=lang, clean=False)
    lines = []
    print("✂️  Splitting transcript into sentences")
    for text in tqdm(input_lines):
        text = text.rstrip("\n")
        s = seg.segment(text)
        lines += s
    return lines


def split_sentences_from_input(input_lines, output_path, lang):
    lines = get_segments(input_lines, lang)
    lines = fix_end_of_quotes(lines)
    lines = merge_short_lines_with_quotes(lines)

    with open(output_path, "w", encoding="utf-8") as fo:
        for line in lines:
            if line:
                fo.write(line + "\n")


def flatten(t):
    return (
        [j for i in t for j in flatten(i)]
        if isinstance(t, (tuple, list))
        else [t]
        if isinstance(t, epub.Link)
        else []
    )


@dataclass(eq=True, frozen=True)
class EpubParagraph:
    chapter: int
    element: element.Tag
    references: list

    def text(self):
        return "".join(self.element.strings)


@dataclass(eq=True, frozen=True)
class EpubChapter:
    content: BeautifulSoup
    title: str
    is_linear: bool
    idx: int

    def text(self):
        paragraphs = self.content.find("body").find_all(
            ["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]
        )
        r = []
        for p in paragraphs:
            if "id" in p.attrs:
                continue
            r.append(EpubParagraph(chapter=self.idx, element=p, references=[]))
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

        flat_toc = flatten(file.toc)
        m = {
            it.id: i
            for i, e in enumerate(flat_toc)
            if (
                it := file.get_item_with_href(
                    urllib.parse.unquote(e.href.split("#")[0])
                )
            )
        }
        if len(m) != len(flat_toc):
            print(
                "WARNING: Couldn't fully map toc to chapters, contact the dev, preferably with the epub"
            )

        chapters = []
        prev_title = ""
        for i, v in enumerate(file.spine):
            item = file.get_item_with_id(v[0])
            title = flat_toc[m[v[0]]].title if v[0] in m else ""

            if item.media_type != "application/xhtml+xml":
                if title:
                    prev_title = title
                continue

            content = BeautifulSoup(item.get_content(), "html.parser")

            r = content.find("body").find_all(
                ["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]
            )
            # Most of the time chapter names are on images
            idx = 0
            while idx < len(r) and not r[idx].get_text().strip():
                idx += 1
            if idx >= len(r):
                if title:
                    prev_title = title
                continue

            if not title:
                if t := prev_title.strip():
                    title = t
                    prev_title = ""
                elif len(t := r[idx].get_text().strip()) < 25:
                    title = t
                else:
                    title = item.get_name()

            chapter = EpubChapter(content=content, title=title, is_linear=v[1], idx=i)
            chapters.append(chapter)
        return cls(
            epub=file,
            path=path,
            title=file.title.strip() or path.name,
            chapters=chapters,
        )
