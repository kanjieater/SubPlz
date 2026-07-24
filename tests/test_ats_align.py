import pytest
import numpy as np
from unittest.mock import MagicMock
from ats.align import align_sub, fix_punc, fix, align
from ats.lang import Japanese


@pytest.fixture
def mock_lang():
    """Provides a mocked Language object for testing."""
    lang = MagicMock()
    lang.clean.side_effect = lambda x: x
    lang.translate.side_effect = lambda x: x
    return lang


def test_align_sub_basic():
    """Verifies basic alignment of subtitles with equal lengths."""
    coords = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    )
    text = ["あいうえお", "かきくけこ"]
    subs = ["あいうえお", "かきくけこ"]
    result = align_sub(coords, text, subs)
    assert len(result) == 2


def test_fix_punc_no_op():
    """Ensures fix_punc does nothing when no punctuation rules are provided."""
    text = ["こんにちは世界。", "二行目。"]
    segments = [[[0, 8, 0]], [[0, 4, 1]]]
    fix_punc(text, segments, set(), set(), set())
    assert segments == [[[0, 8, 0]], [[0, 4, 1]]]


def test_fix_basic(mock_lang):
    """Checks basic index mapping logic in fix function."""
    original = ["こんにちは！", "元気ですか？"]
    edited = ["こんにちは", "元気ですか"]
    segments = [[[0, 5, 0]], [[0, 5, 1]]]
    mock_lang.translate.side_effect = lambda x: x
    fix(mock_lang, original, edited, segments)
    assert len(segments) == 2


def test_align_sub_multiple_segments():
    """Tests mapping of multiple transcript segments into multiple text lines."""
    coords = np.array([[0, 5, 5, 10], [0, 5, 5, 10]])
    text = ["あいうえお", "かきくけこ"]
    subs = ["あいうえお", "かきくけこ"]
    result = align_sub(coords, text, subs)
    assert len(result) == 2
    assert result[0] == [[0, 5, 0]]
    assert result[1] == [[0, 5, 1]]


def test_fix_punc_shifting():
    """Verifies punctuation shifting to include specific characters."""
    text = ["こんにちは。"]
    segments = [[[0, 4, 0]]]
    fix_punc(text, segments, set(), {"は"}, set())
    assert segments[0][0][1] == 5


def test_align_sub_c2_counter():
    """Tests inner while loop trigger in align_sub when target exceeds current line."""
    coords = np.array([[0, 2], [0, 5]])
    text = ["あ", "い"]
    subs = ["あいうえお"]
    result = align_sub(coords, text, subs)
    assert len(result) == 2
    assert len(result[0]) > 0


def test_fix_mapping_snaps(mock_lang):
    """Verifies index snapping in fix function for spaced text."""
    original = ["あ い う"]
    edited = ["あいう"]
    segments = [[[0, 3, 0]]]
    mock_lang.translate.side_effect = lambda x: "あ い う"
    fix(mock_lang, original, edited, segments)
    assert segments[0][0][1] == 5


def test_japanese_lang_clean():
    """Verifies Japanese cleaning logic in Japanese language class."""
    lang = Japanese(prepend="", append="", nopend="")
    result = lang.clean("ハロー、世界！")
    assert "はろ" in result
    assert "世界" in result


def test_align_sub_empty_inputs():
    """Ensures align_sub handles empty inputs gracefully."""
    coords = np.array([[], []])
    text = []
    subs = []
    result = align_sub(coords, text, subs)
    assert result == []


def test_align_sub_trailing_subs():
    """Tests align_sub behavior with trailing subtitles."""
    coords = np.array([[0, 5], [0, 5]])
    text = ["あいうえお"]
    subs = ["あいうえお", "かきくけこ", "おまけ"]
    result = align_sub(coords, text, subs)
    assert len(result) == 1
    assert result[0] == [[0, 5, 0]]


def test_fix_punc_empty_segments():
    """Ensures fix_punc handles empty segments gracefully."""
    text = ["こんにちは"]
    segments = [[]]
    fix_punc(text, segments, set(), set(), set())
    assert segments == [[]]


def test_fix_empty_m(mock_lang):
    """Verifies fix function with empty edited text."""
    original = ["あいう"]
    edited = [""]
    segments = [[[0, 0, 0]]]
    mock_lang.translate.side_effect = lambda x: "あいう"
    fix(mock_lang, original, edited, segments)
    assert segments[0][0] == [0, 0, 0]


def test_align_sub_no_subs():
    """Ensures align_sub handles cases with no subtitles."""
    coords = np.array([[0, 1], [0, 1]])
    text = ["あ"]
    subs = []
    result = align_sub(coords, text, subs)
    assert result == [[]]


def test_fix_punc_kanji_bail_out():
    """Verifies fix_punc bails out properly when hitting a Kanji boundary."""
    text = ["これは。漢字"]
    segments = [[[0, 4, 0]]]
    nopend = {"。"}
    fix_punc(text, segments, set(), set(), nopend)
    assert segments[0][0][1] == 4


def test_align_sub_text_index_out_of_bounds():
    """Tests align_sub early return when text index is out of bounds."""
    coords = np.array([[0, 10], [0, 10]])
    text = ["あ"]
    subs = ["あいうえおかきくけ"]
    result = align_sub(coords, text, subs)
    assert len(result) == 1


def test_align_sub_gaps():
    """Verifies isgap logic in align_sub."""
    coords = np.array([[0, 1, 1, 2], [0, 0, 1, 2]])
    text = ["あ", "い"]
    subs = ["あ", "い"]
    result = align_sub(coords, text, subs)
    assert len(result) == 2


def test_fix_punc_shifting_full():
    """Tests complex shifting logic in fix_punc."""
    text = ["あいうえお"]
    segments = [[[0, 2, 0]]]
    fix_punc(text, segments, set(), {"う"}, set())
    assert segments[0][0][1] == 3

    segments = [[[0, 3, 0]]]
    fix_punc(text, segments, {"う"}, set(), set())
    assert segments[0][0][1] == 2


def test_fix_complex_mapping(mock_lang):
    """Verifies fix function with complex mappings."""
    original = ["あいうえお"]
    edited = ["あえお"]
    segments = [[[0, 3, 0]]]
    mock_lang.translate.side_effect = lambda x: "あいうえお"
    fix(mock_lang, original, edited, segments)
    assert segments[0][0][1] <= 5


def test_align_sub_text_exhausted():
    """Tests align_sub when text is exhausted but subtitles remain."""
    coords = np.array([[0, 10], [0, 10]])
    text = ["あ"]
    subs = ["あ", "い"]
    result = align_sub(coords, text, subs)
    assert len(result) == 1


def test_align_sub_low_diff():
    """Tests align_sub behavior with low difference between subtitle lengths."""
    coords = np.array([[0, 1, 2], [0, 1, 2]])
    text = ["あいうえお"]
    subs = ["あ", ""]
    result = align_sub(coords, text, subs)
    assert len(result) == 1


def test_fix_punc_complex_branches():
    """Verifies various complex logic branches in fix_punc."""
    text = ["「あ。い」"]
    segments = [[[1, 2, 0]]]
    fix_punc(text, segments, set(), set(), {"。"})
    assert segments[0][0][1] == 2
    text = ["あ。漢"]
    segments = [[[0, 1, 0]]]
    fix_punc(text, segments, set(), set(), {"。"})
    assert segments[0][0][1] == 1


def test_fix_mapping_snaps_full(mock_lang):
    """Verifies complete index snapping logic in fix function."""
    original = ["a  b"]
    edited = ["ab"]
    segments = [[[0, 2, 0]]]
    mock_lang.translate.side_effect = lambda x: "a  b"
    fix(mock_lang, original, edited, segments)
    assert segments[0][0][1] == 4


def test_fix_punc_loop_limit():
    """Ensures fix_punc loop limit prevents infinite loops."""
    text = ["あ" * 30]
    segments = [[[0, 1, 0]]]
    fix_punc(text, segments, {"あ"}, set(), set())
    assert segments[0][0][1] == 0


def test_fix_punc_nopend_branches():
    """Verifies nopend related logic branches in fix_punc."""
    text = ["あ。い"]
    segments = [[[0, 1, 0]]]
    fix_punc(text, segments, {"あ"}, set(), {"。"})
    assert segments[0][0][1] == 0
    text = ["あ。い"]
    segments = [[[0, 1, 0]]]
    fix_punc(text, segments, set(), {"あ"}, {"。"})
    assert segments[0][0][1] == 1


def test_fix_punc_nopend_end_branches():
    """Verifies end-of-text nopend branches in fix_punc."""
    text = ["あ。い"]
    segments = [[[0, 1, 0]]]
    fix_punc(text, segments, {"い"}, set(), {"。"})
    assert segments[0][0][1] == 2
    segments = [[[0, 1, 0]]]
    fix_punc(text, segments, set(), {"い"}, {"。"})
    assert segments[0][0][1] == 3


def test_align_sub_target_exceeds_end():
    """Tests align_sub when target exceeds the end of text line."""
    coords = np.array([[0, 2], [0, 10]])
    text = ["あ"]
    subs = ["あいうえおかきくけこ"]
    result = align_sub(coords, text, subs, thing=2)
    assert len(result) == 1


def test_align_sub_gap_logic():
    """Verifies gap detection and tracking in align_sub."""
    coords = np.array([[0, 1, 5, 6], [0, 1, 5, 6]])
    text = ["あ", "い", "う"]
    subs = ["あ", "い", "う"]
    result = align_sub(coords, text, subs)
    assert len(result) == 3


def test_align_sub_low_diff_gap_check():
    """Verifies gap checking logic in low-diff scenarios in align_sub."""
    coords = np.array([[0, 1, 2], [0, 1, 2]])
    text = ["あいうえお"]
    subs = ["あ", ""]
    result = align_sub(coords, text, subs)
    assert result[0] == [[0, 1, 0]]
    coords = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    text = ["あいうえお"]
    subs = ["あ", "い", "う", ""]
    result = align_sub(coords, text, subs)
    assert len(result) == 1


def test_fix_punc_complex_nopend_2():
    """Tests additional complex nopend scenarios in fix_punc."""
    text = ["あ。い"]
    segments = [[[0, 1, 0]]]
    fix_punc(text, segments, {"あ"}, set(), {"い"})


def test_fix_mapping_loop_bounds(mock_lang):
    """Verifies index mapping bounds in fix function."""
    original = ["A"]
    edited = ["B"]
    segments = [[[0, 1, 0]]]
    mock_lang.translate.side_effect = lambda x: "A"
    fix(mock_lang, original, edited, segments)
    assert segments[0][0] == [0, 0, 0]


def test_align_early_return_joined(mock_lang):
    """Ensures align returns early for empty text or transcript."""
    model = MagicMock()
    transcript = [""]
    text = [""]
    mock_lang.clean.return_value = ""
    res = align(model, mock_lang, transcript, text, [], set(), set(), set())
    assert res == []


def test_align_entry_point_new(mock_lang):
    """Verifies main entry point of align function."""
    model = MagicMock()
    transcript = ["sub1", "sub2"]
    text = ["line1", "line2"]
    res = align(model, mock_lang, transcript, text, [], set(), set(), set())
    assert isinstance(res, list)


def test_align_sub_c2_count_increment():
    """Tests the increment logic for c2 and count in align_sub."""
    coords = np.array([[0, 2], [0, 5]])
    text = ["あ"]
    subs = ["あいうえお"]
    result = align_sub(coords, text, subs, thing=1)
    assert len(result) == 1


def test_fix_punc_extreme_nopend():
    """Tests extreme nopend cases in fix_punc at boundaries."""
    text = ["あ。い"]
    segments = [[[1, 1, 0]]]
    fix_punc(text, segments, set(), set(), {"い"})
    segments = [[[1, 1, 0]]]
    fix_punc(text, segments, set(), {"あ"}, {"。"})
    assert segments[0][0][1] == 1
