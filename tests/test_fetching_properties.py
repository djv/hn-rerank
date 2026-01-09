
import pytest
from hypothesis import given, strategies as st
from api.fetching import _clean_text, _extract_comments_recursive

# Strategies for clean_text
@given(st.text())
def test_clean_text_properties(text):
    """
    Properties for _clean_text:
    1. If it returns a string, it must not contain excessive punctuation or certain block characters.
    2. If text is too short or has low alphanumeric ratio, it should return None.
    3. It should not crash on any input.
    """
    result = _clean_text(text)

    if result is not None:
        # Should be non-empty
        assert len(result) > 0
        # Should not have braille/box drawing chars (simplified check)
        assert not any('\u2800' <= c <= '\u28FF' for c in result)
        # Should have reasonable length
        assert len(result.strip()) > 20

# Define recursive strategy for comments
def comment_strategy():
    return st.recursive(
        st.fixed_dictionaries(
            {
                "type": st.one_of(st.just("comment"), st.just("story"), st.text()),
                "text": st.text(),
                "points": st.integers(),
            }
        ),
        lambda children: st.fixed_dictionaries(
            {
                "type": st.one_of(st.just("comment"), st.just("story"), st.text()),
                "text": st.text(),
                "points": st.integers(),
                "children": st.lists(children, max_size=3)
            }
        ),
        max_leaves=10
    )

@given(st.lists(comment_strategy(), max_size=5))
def test_extract_comments_properties(children):
    """
    Properties for _extract_comments_recursive:
    1. Should return a flat list of dicts.
    2. Each result should have 'text' and 'score'.
    3. Score should be calculated (but hard to verify exact value without replicating logic).
    4. Should handle arbitrary nesting without crashing.
    """
    results = _extract_comments_recursive(children)

    for item in results:
        assert "text" in item
        assert "score" in item
        assert isinstance(item["score"], (int, float))
        # Logic in fetching.py filters out short comments (<20)
        assert len(item["text"]) >= 20
