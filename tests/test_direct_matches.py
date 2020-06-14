"""
Test the various direct matches
"""

import nmfu
from hypothesis import given
import hypothesis.strategies as st
from collections import defaultdict

@given(st.text(st.characters(min_codepoint=0, max_codepoint=255), min_size=2))
def test_direct_match(text):
    match = nmfu.DirectMatch(text)

    dfa = match.convert(defaultdict(lambda: None))

    assert len(dfa.accepting_states) == 1
    assert dfa.simulate(text) in dfa.accepting_states
    assert dfa.simulate(text[:-1]) not in dfa.accepting_states
    assert dfa.simulate(text[:-1] + chr((ord(text[-1])+1) % 256)) is None

@given(st.text(st.characters(min_codepoint=ord('!'), max_codepoint=ord('z')), min_size=2))
def test_casei_match(text):
    match = nmfu.CaseDirectMatch(text)

    dfa = match.convert(defaultdict(lambda: None))

    assert len(dfa.accepting_states) == 1
    assert dfa.simulate(text) in dfa.accepting_states
    assert dfa.simulate(text.lower()) in dfa.accepting_states
    assert dfa.simulate(text.upper()) in dfa.accepting_states
    assert dfa.simulate(text[:-1]) not in dfa.accepting_states
    assert dfa.simulate(text[:-1] + chr((ord(text[-1])+1) % 256)) is None
