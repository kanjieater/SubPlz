import pytest
from subplz.align import (
    shift_align,
    Segment,
)
from ats.main import Segment

# Mock data
sample_script = """
Hello, world!
This is a test script.
""".strip()

sample_subtitles = """
WEBVTT

00:00:01.000 --> 00:00:03.000
Hello, world!

00:00:04.000 --> 00:00:06.000
This is a test script.
""".strip()


@pytest.fixture
def mock_script_file():
    return sample_script


@pytest.fixture
def mock_subs_file():
    return sample_subtitles


@pytest.fixture
def mock_recursively_find_match():
    def _mock_recursively_find_match(*args, **kwargs):
        result = [(0, 1, 0, 1), (1, 1, 1, 1)]
        return result

    return _mock_recursively_find_match


# @pytest.mark.skip(reason="debug")
def test_shift_align_start_case():
    # Given segments that should remain unchanged
    segments = [
        Segment(text="真田さんのグループ", start=168.16, end=169.38),
        Segment(
            text="が、その子とどれだけ仲良くしたがっても。その子は、「私はこころちゃんといる」と、",
            start=170.44,
            end=178.72,
        ),
        Segment(text="真田さんのグルー", start=168.16, end=169.38),
        Segment(
            text="プが、その子とどれだけ仲良くしたがっても。その子は、「私はこころちゃんといる」と、",
            start=170.44,
            end=178.72,
        ),
        Segment(text="真田さんのグルー", start=168.16, end=169.38),
        Segment(
            text="ープが、その子とどれだけ仲良くしたがっても。その子は、「私はこころちゃんといる」と、",
            start=170.44,
            end=178.72,
        ),
    ]
    result = shift_align(segments)

    assert len(result) == len(segments)
    assert result[0].text == "真田さんのグループが、"
    assert (
        result[1].text
        == "その子とどれだけ仲良くしたがっても。その子は、「私はこころちゃんといる」と、"
    )
    assert result[2].text == "真田さんのグループが、"
    assert (
        result[3].text
        == "その子とどれだけ仲良くしたがっても。その子は、「私はこころちゃんといる」と、"
    )
    assert result[4].text == segments[4].text
    assert result[5].text == segments[5].text


# @pytest.mark.skip(reason="debug")
def test_shift_align_punctuation_overlap():
    # Given segments with punctuation overlap
    segments = [
        Segment(text="騒音問題になっている、", start=365.16, end=365.16),
        Segment(text="とも。騒音、", start=365.16, end=367.36),
        Segment(text="「ああ、", start=365.16, end=367.36),
    ]
    result = shift_align(segments)

    assert len(result) == len(segments)
    assert result[0].text == "騒音問題になっている、とも。"
    assert result[1].text == "騒音、"
    assert result[2].text == "「ああ、"


def test_shift_align_end_start_punctuation():
    # Given segments with punctuation at the end and start
    segments = [
        Segment(text="午前中もあとちょっとだ」と", start=373.82, end=374.36),
        Segment(text="思う。", start=374.36, end=375.00),
    ]
    result = shift_align(segments)

    assert len(result) == 2
    assert result[0].text == "午前中もあとちょっとだ」"
    assert result[1].text == "と思う。"


def test_shift_two_in_a_row():
    segments = [
        Segment(
            text="「そのとおりじゃ。王都に帰って、陛下や軍師どのにこの件をご報告申しあげねばならぬ。ひ",
            start=373.82,
            end=374.36,
        ),
        Segment(
            text="とつまちがえば、パルス国の存亡にかかわることゆえ」　フ",
            start=373.82,
            end=374.36,
        ),
        Segment(
            text="ァランギースは懐中から小さな翡翠の笛をとり出した。",
            start=374.36,
            end=375.00,
        ),
    ]
    result = shift_align(segments)
    assert result[-2].text == "ひとつまちがえば、パルス国の存亡にかかわることゆえ」"
    assert result[-1].text == "ファランギースは懐中から小さな翡翠の笛をとり出した。"


def test_shift_align_start_end_wrong():
    segments = [
        Segment(text="「あ、", start=375.00, end=376.00),
        Segment(text="いえ、忘れてました」「ばか、", start=376.00, end=378.00),
        Segment(text="さっさといれろ。", start=379.00, end=380.00),
    ]
    result = shift_align(segments)

    assert result[0].text == "「あ、"
    assert result[1].text == "いえ、忘れてました」"
    assert result[2].text == "「ばか、さっさといれろ。"


@pytest.mark.skip(reason="Not worth doing yet")
def test_shift_align_start_end_wrong_other_matches():
    # Additional test that fails because of question mark
    # 「いま現在、そこのようすは？」「い
    # ま、そこのようすは
    segments = [
        Segment(
            text="＜今度こそ死を覚悟しないとかもな＞＜う～ん", start=375.00, end=376.00
        ),
        Segment(text="先生が待ってるはずだがいないな＞", start=376.00, end=378.00),
        Segment(text="さっさといれろ。", start=379.00, end=380.00),
    ]
    result = shift_align(segments)

    assert result[1].text == "＜今度こそ死を覚悟しないとかもな＞"
    assert result[2].text == "＜う～ん先生が待ってるはずだがいないな＞"


def test_shift_align_complex_join():
    segments = [
        Segment(text="最初はそれで心地よかったものが、だ", start=375.00, end=376.00),
        Segment(
            text="んだんと、やっぱりいけないんだと思うように", start=376.00, end=378.00
        ),
    ]
    result = shift_align(segments)

    assert result[0].text == "最初はそれで心地よかったものが、"
    assert result[1].text == "だんだんと、やっぱりいけないんだと思うように"


def test_shift_align_wa_pattern():
    segments = [
        Segment(text="そんな奇跡が起きないこと", start=375.00, end=376.00),
        Segment(text="は、知っている。", start=376.00, end=378.00),
    ]
    result = shift_align(segments)

    assert result[0].text == "そんな奇跡が起きないことは、"
    assert result[1].text == "知っている。"


def test_shift_align_maintain_double_comma():
    segments = [
        Segment(text="一群の騎馬が、", start=375.00, end=376.00),
        Segment(
            text="東へ、ペシャワール城塞の方角へと疾駆している。",
            start=376.00,
            end=378.00,
        ),
        Segment(text="一群の騎馬が、東へ、", start=375.00, end=376.00),
        Segment(
            text="ペシャワール城塞の方角へと疾駆している。", start=376.00, end=378.00
        ),
    ]
    result = shift_align(segments)

    assert len(result) == len(segments)
    assert result[0] == segments[0]
    assert result[1] == segments[1]
    assert result[2] == segments[2]
    assert result[3] == segments[3]


def test_shift_align_no_change():
    segments = [
        Segment(text="あの、たしかヒルメス王子の腹", start=0.00, end=2.00),
        Segment(text="心の……」", start=0.00, end=2.00),
        Segment(
            text="身を硬くしている平日に見るものではなかった。", start=0.00, end=2.00
        ),
        Segment(text="去年、までは。", start=2.10, end=4.00),
        Segment(text="「ああ、", start=4.10, end=6.00),
    ]
    result = shift_align(segments)
    assert len(result) == len(segments)
    assert result[0] == segments[0]
    assert result[1] == segments[1]
    assert result[2] == segments[2]


def test_shift_align_no_change_opening_quote():
    segments = [
        Segment(text="妻たちでござる」", start=0.00, end=2.00),
        Segment(text="「ほう、トゥース卿はご結婚なさったのか」", start=2.10, end=4.00),
        Segment(text="トゥース卿はご結婚なさったのか」", start=2.10, end=4.00),
    ]
    result = shift_align(segments)
    assert len(result) == len(segments)
    assert result[0] == segments[0]
    assert result[1] == segments[1]
    assert result[2] == segments[2]


def test_shift_align_kuromaru():
    segments = [
        Segment(text="「ハーチ", start=0.00, end=2.00),
        Segment(text="ム・マイマイ」などと呼べば、", start=2.10, end=4.00),
    ]

    result = shift_align(segments)

    assert len(result) == len(segments)
    assert result[0].text == "「ハーチム・"
    assert result[1].text == "マイマイ」などと呼べば、"


def test_shift_open_quote():
    segments = [
        # Segment(text='学塾に行きたい子にも援助してやろ', start=0.00, end=2.00),
        # Segment(text='う」「けっこうな御意ですが、', start=2.10, end=4.00),
        Segment(text="学塾に行きたい子にも援助してやろう」「", start=0.00, end=2.00),
        Segment(text="けっこうな御意ですが、", start=2.10, end=4.00),
    ]

    result = shift_align(segments)

    assert len(result) == len(segments)
    assert result[0].text == "学塾に行きたい子にも援助してやろう」"
    assert result[1].text == "「けっこうな御意ですが、"


def test_shift_open_other():
    segments = [
        Segment(text="《えっ　私が払うの？", start=2.10, end=4.00),
        Segment(text="》《雲隠九郎君》", start=0.00, end=2.00),
        Segment(text="《講談高校で待ってるよ》", start=2.10, end=4.00),
    ]
    result = shift_align(segments)
    assert len(result) == len(segments)
    assert result[0].text == "《えっ　私が払うの？》"
    assert result[1].text == "《雲隠九郎君》"
    assert result[2].text == "《講談高校で待ってるよ》"


def test_shift_align_emdash():
    segments = [
        Segment(text="老国王カトリコスの孫娘――アル", start=0.00, end=2.00),
        Segment(text="ガシュの妹の娘を妻としたオスロエスが、", start=2.10, end=4.00),
    ]

    result = shift_align(segments)

    assert len(result) == len(segments)
    assert result[0].text == "老国王カトリコスの孫娘――"
    assert result[1].text == "アルガシュの妹の娘を妻としたオスロエスが、"


def test_shift_align_unicode_space():
    segments = [
        Segment(text="　陛下のご下問である。奉答せよ」　あた", start=0.00, end=2.00),
        Segment(
            text="らしく冷水の壺を用意しようとしていたエラムは、", start=2.10, end=4.00
        ),
    ]

    result = shift_align(segments)

    assert len(result) == len(segments)
    assert result[0].text == "陛下のご下問である。奉答せよ」"
    assert result[1].text == "あたらしく冷水の壺を用意しようとしていたエラムは、"


def test_shift_align_unicode_parenthesis():
    segments = [
        Segment(text="　王都エクバターナより東へ四十ファルサン", start=0.00, end=2.00),
        Segment(text="グ（約二百キロ）。", start=2.10, end=4.00),
    ]

    result = shift_align(segments)

    assert len(result) == len(segments)
    assert result[0].text == "王都エクバターナより東へ四十ファルサング"
    assert result[1].text == "（約二百キロ）。"
