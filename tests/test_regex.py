import re
import nmfu
from hypothesis import given
import hypothesis.strategies as st
import pytest
from collections import defaultdict

def create_regex_match(x):
    return nmfu.RegexMatch(nmfu.parser.parse(x, start="regex"))

DISJOINT_EXAMPLES = {
        b"test": create_regex_match(r"/test/"),
        b"ter+t": create_regex_match(r"/ter+t/"),
        br't[^ter"]*a': create_regex_match(r'/t[^ter"]*a/'),
        br't"[^"]+"': create_regex_match(r'/t"[^"]+"/'),
        br"oof": create_regex_match(r"/oof/"),
        br"yar.le": create_regex_match(r"/yar.le/"),
        br"oor+t": create_regex_match(r"/oor+t/"),
        br"(te|r)y": create_regex_match(r"/(te)|ry/")
}

@given(st.one_of(st.from_regex(x, fullmatch=True) for x in DISJOINT_EXAMPLES))
def test_regexes_match(input_str):
    corresponding = None

    # figure out which one it matches
    for i in DISJOINT_EXAMPLES:
        if re.fullmatch(i, input_str):
            corresponding = DISJOINT_EXAMPLES[i]
            break

    if corresponding is None:
        return # error in hypothesis

    dfa = corresponding.convert(defaultdict(lambda: None))
    assert dfa.simulate([chr(x) for x in input_str]) in dfa.accepting_states

def test_regex_charclass_splits():
    c1, c2 = nmfu.RegexCharClass(frozenset("abc")), nmfu.RegexCharClass(frozenset("bcd"))

    overlap, c1, c2 = c1.split(c2)

    assert isinstance(overlap, nmfu.RegexCharClass)
    assert isinstance(c1, nmfu.RegexCharClass)
    assert isinstance(c2, nmfu.RegexCharClass)
    
    assert overlap.chars == frozenset({"b", "c"})
    assert c1.chars == frozenset({"a"})
    assert c2.chars == frozenset({"d"})

def test_regex_inverted_charclass_splits():
    c1, c2 = nmfu.InvertedRegexCharClass(frozenset("abc")), nmfu.InvertedRegexCharClass(frozenset("bcd"))

    overlap, c1, c2 = c1.split(c2)

    assert isinstance(overlap, nmfu.InvertedRegexCharClass)
    assert isinstance(c1, nmfu.RegexCharClass)
    assert isinstance(c2, nmfu.RegexCharClass)
    
    assert overlap.chars == frozenset({"a", "b", "c", "d"})
    assert c1.chars == frozenset({"d"})
    assert c2.chars == frozenset({"a"})

def test_regex_mixed_charclass_splits():
    c1, c2 = nmfu.InvertedRegexCharClass(frozenset("abc")), nmfu.RegexCharClass(frozenset("bcd"))

    overlap, c1, c2 = c1.split(c2)

    assert isinstance(overlap, nmfu.RegexCharClass)
    assert isinstance(c1, nmfu.InvertedRegexCharClass)
    assert isinstance(c2, nmfu.RegexCharClass)
    
    assert overlap.chars == frozenset({"d"})
    assert c1.chars == frozenset({"a", "b", "c", "d"})
    assert c2.chars == frozenset({"b", "c"})

    c2, c1 = nmfu.InvertedRegexCharClass(frozenset("abc")), nmfu.RegexCharClass(frozenset("bcd"))

    overlap, c2, c1 = c1.split(c2)

    assert isinstance(overlap, nmfu.RegexCharClass)
    assert isinstance(c1, nmfu.InvertedRegexCharClass)
    assert isinstance(c2, nmfu.RegexCharClass)
    
    assert overlap.chars == frozenset({"d"})
    assert c1.chars == frozenset({"a", "b", "c", "d"})
    assert c2.chars == frozenset({"b", "c"})
