from subplz.align import Segment, shift_align


def test_shift_align_data_loss_bug():
    """
    Reproduces the bug where using 'segments[i+1]' instead of 'new_segments[i+1]'
    causes data loss if the first loop already moved text from segments[i+2] to new_segments[i+1].
    """
    segments = [
        Segment(text="Part 0, a", start=0.0, end=1.0),
        Segment(text="Part 1, rest", start=1.0, end=2.0),
        Segment(text="b, Part 2", start=2.0, end=3.0),
    ]

    # First loop in shift_align:
    # i=0: new_segments = [Segment("Part 0, a")]
    # i=1: new_segments = [..., Segment("Part 1, rest")] (New object because it has punctuation)
    # i=2: moves "b," from segments[2] to new_segments[1]
    #      new_segments[1].text becomes "Part 1, restb,"
    #      new_segments[2] becomes " Part 2"

    # Second loop in shift_align at i=0:
    # segment = "Part 0, a". indices = [6]. count_non_punctuation(text[6:]) = 1.
    # Triggers move to next segment.
    # BUG: next_segment = segments[1] -> "Part 1, rest" (Original object, doesn't have "b,")
    # next_segment.text = " a" + "Part 1, rest" = " aPart 1, rest"
    # new_segments[1] = next_segment
    # "b," which was in the previous new_segments[1] is now LOST!

    result = shift_align(segments)

    print("\nResults:")
    for j, seg in enumerate(result):
        print(f"[{j}]: '{seg.text}'")

    assert "b," in result[1].text
