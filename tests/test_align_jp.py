from subplz.align import Segment, shift_align


def test_shift_align_japanese_brackets():
    """Test handling of Japanese brackets 「 」 and other punctuation."""
    segments = [
        Segment(text="彼は「こんにちは、", start=0.0, end=1.0),
        Segment(text="世界」と言った。", start=1.0, end=2.0),
    ]
    result = shift_align(segments)
    assert len(result) == 2


def test_shift_align_double_comma_prevention():
    """Verify that has_double_comma prevents moving a comma if it would create a double comma."""
    segments = [
        Segment(text="昨日、", start=0.0, end=1.0),
        Segment(text="、今日", start=1.0, end=2.0),
    ]
    result = shift_align(segments)
    assert result[0].text == "昨日、"
    assert result[1].text == "、今日"


def test_shift_align_specific_pattern_speaker_split():
    """
    Verify that handle_specific_pattern and handle_starting_punctuation work together
    to split at the speaker change boundary while keeping terminal punctuation with the dialogue.
    """
    segments = [
        Segment(text="と言った。」「ばか、", start=0.0, end=1.0),
        Segment(text="」と彼は笑った。", start=1.0, end=2.0),
    ]
    result = shift_align(segments)
    assert result[0].text == "と言った。」"
    assert result[1].text == "「ばか、」と彼は笑った。"


def test_shift_align_no_move_to_previous_if_full_sentence():
    """
    Test the first loop logic: do NOT move text to previous segment if
    previous segment is a full sentence (ends with terminal punctuation).
    """
    segments = [
        Segment(text="これはテストです。", start=0.0, end=1.0),
        Segment(text="ね、わかった？", start=1.0, end=2.0),
    ]
    # "ね、わかった？" -> punctuation at index 1 ('、').
    # text[:1] is "ね" (1 non-punc char). 1 <= 2.
    # HOWEVER, previous segment ends with "。", which is END_PUNC.
    # So it should NOT move.

    result = shift_align(segments)
    assert result[0].text == "これはテストです。"
    assert result[1].text == "ね、わかった？"


def test_shift_align_no_move_if_double_comma():
    """Test that short chunk move to previous is blocked by has_double_comma."""
    segments = [
        Segment(text="多分、", start=0.0, end=1.0),
        Segment(text="さ、行こう。", start=1.0, end=2.0),
    ]
    result = shift_align(segments)
    assert result[0].text == "多分、"
    assert result[1].text == "さ、行こう。"


def test_shift_align_non_punc_count_limit_next():
    """Test the second loop logic: move text to next segment if trailing chunk is short."""
    # Input with NO leading punctuation in next segment
    segments = [
        Segment(text="わかった、ね ", start=0.0, end=1.0),
        Segment(text="了解。", start=1.0, end=2.0),
    ]

    result = shift_align(segments)
    assert result[0].text == "わかった、"
    assert result[1].text == "ね 了解。"


def test_handle_starting_punctuation_direct():
    """Verify handle_starting_punctuation behavior directly."""
    segments = [
        Segment(text="前のセグメント", start=0.0, end=1.0),
        Segment(text="。次のセグメント", start=1.0, end=2.0),
    ]
    result = shift_align(segments)
    assert result[0].text == "前のセグメント。"
    assert result[1].text == "次のセグメント"


def test_handle_ending_punctuation_direct():
    """Verify handle_ending_punctuation behavior directly."""
    segments = [
        Segment(text="前のセグメント「", start=0.0, end=1.0),
        Segment(text="次のセグメント", start=1.0, end=2.0),
    ]
    result = shift_align(segments)
    assert result[0].text == "前のセグメント"
    assert result[1].text == "「次のセグメント"
